import os
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.pipeline_stage import PipelineStageModel
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.distributed import set_tp_group


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.pp_size = config.pipeline_parallel_size
        self.pp_rank = config.pp_rank

        # Isolate Triton cache per rank to avoid JIT compilation races
        os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton_cache_rank{self.pp_rank}")

        # For PP-only (TP=1): each stage is its own process, world_size=pp_size
        if self.pp_size > 1 and self.world_size == 1:
            nccl_world_size = self.pp_size
            nccl_rank = self.pp_rank
        else:
            nccl_world_size = self.world_size
            nccl_rank = rank

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=nccl_world_size, rank=nccl_rank)
        torch.cuda.set_device(nccl_rank)

        # Create TP subgroups so TP collectives only run within a PP stage
        if self.pp_size > 1:
            for pp in range(self.pp_size):
                ranks = list(range(pp * self.world_size, (pp + 1) * self.world_size))
                group = dist.new_group(ranks)
                if pp == self.pp_rank:
                    self.tp_group = group
            set_tp_group(self.tp_group)
        else:
            self.tp_group = None

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")

        if self.pp_size > 1:
            self.model = PipelineStageModel(hf_config, self.pp_rank, self.pp_size)
            load_model(self.model, config.model, self.pp_rank, self.pp_size)
        else:
            self.model = Qwen3ForCausalLM(hf_config)
            load_model(self.model, config.model)

        self.sampler = Sampler() if self._is_last_stage() else None

        # Setup shared memory BEFORE warmup (warmup needs PP coordination)
        if self.pp_size > 1:
            if self.pp_rank == 0:
                try:
                    existing = SharedMemory(name="nanovllm_pp")
                    existing.close()
                    existing.unlink()
                except FileNotFoundError:
                    pass
                self.shm_pp = SharedMemory(name="nanovllm_pp", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm_pp = SharedMemory(name="nanovllm_pp")

        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.pp_size > 1 and self.pp_rank > 0:
            self.pp_loop()
        elif self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def _is_last_stage(self):
        return self.pp_rank == self.pp_size - 1

    def exit(self):
        if self.pp_size > 1:
            if self.pp_rank == 0:
                self.shm_pp.close()
                self.shm_pp.unlink()
            else:
                self.shm_pp.close()
        elif self.world_size > 1:
            self.shm.close()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    # ── TP worker loop ──────────────────────────────────────────────

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    # ── PP worker loop ──────────────────────────────────────────────

    def pp_loop(self):
        """Non-first pipeline stage loop: receive commands from stage 0."""
        while True:
            method_name, args = self._read_pp_shm()
            getattr(self, method_name)(*args)
            if method_name == "exit":
                break

    def _read_pp_shm(self):
        assert self.pp_rank > 0
        # Block until stage 0 signals via NCCL barrier that shm data is ready
        dist.barrier()
        n = int.from_bytes(self.shm_pp.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm_pp.buf[4:n+4])
        return method_name, args

    def _write_pp_shm(self, method_name, *args):
        assert self.pp_rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm_pp.buf[0:4] = n.to_bytes(4, "little")
        self.shm_pp.buf[4:n+4] = data
        dist.barrier()

    def _call_pp(self, method_name, *args):
        """Call method on all PP stages (stage 0 writes shm, then calls locally)."""
        if self.pp_rank == 0:
            self._write_pp_shm(method_name, *args)
        return getattr(self, method_name)(*args)

    # ── Common methods ──────────────────────────────────────────────

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        # For PP, use smaller warmup to avoid Triton kernel OOM on older GPUs
        if self.pp_size > 1:
            seq_len = min(seq_len, 1024)
            num_seqs = min(num_seqs, 2)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        # Use direct NCCL path for warmup (shm-based PP loop not running yet)
        self._warmup_pp(seqs, True) if self.pp_size > 1 else self.run(seqs, True)
        torch.cuda.empty_cache()

    def _warmup_pp(self, seqs, is_prefill):
        """Direct NCCL warmup without shm-based coordination."""
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        self._run_model_pp_direct(input_ids, positions, is_prefill)
        reset_context()
        # Synchronize so both stages finish warmup before proceeding
        if self.pp_size > 1:
            dist.barrier()

    @torch.inference_mode()
    def _run_model_pp_direct(self, input_ids, positions, is_prefill):
        """Direct NCCL send/recv for warmup (no shm coordination)."""
        hf_config = self.config.hf_config
        hidden_size = hf_config.hidden_size

        if self.pp_rank == 0:
            hidden_states, residual = self.model(input_ids, positions)
            dist.send(hidden_states, dst=1)
            if residual is not None:
                dist.send(residual, dst=1)
        else:
            num_tokens = input_ids.size(0)
            hidden_states = torch.empty(num_tokens, hidden_size, device=f"cuda:{self.pp_rank}", dtype=hf_config.dtype)
            dist.recv(hidden_states, src=0)
            residual = torch.empty(num_tokens, hidden_size, device=f"cuda:{self.pp_rank}", dtype=hf_config.dtype)
            dist.recv(residual, src=0)
            # Skip model forward on rank 1 during warmup to avoid Triton OOM
            # on heterogeneous GPUs; first real request will JIT-compile kernels
            # self.model(input_ids, positions, hidden_states, residual)

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        num_layers = self.model.num_stage_layers if self.pp_size > 1 else hf_config.num_hidden_layers
        block_bytes = 2 * num_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        available = int(total * config.gpu_memory_utilization - used - peak + current)
        # Reserve ~2GB for Triton JIT compilation and runtime overhead
        reserved = 2 * 1024 * 1024 * 1024
        config.num_kvcache_blocks = max(1, (available - reserved) // block_bytes)
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, num_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        top_ps = [seq.top_p for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        top_ps = torch.tensor(top_ps, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures, top_ps

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if self.pp_size > 1:
            return self._run_model_pp(input_ids, positions, is_prefill)

        # Original non-PP path
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["output_hidden"][:bs])

    @torch.inference_mode()
    def _run_model_pp(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """Pipeline-parallel forward: stage 0 runs layers then sends to stage 1."""
        hf_config = self.config.hf_config
        hidden_size = hf_config.hidden_size

        if self.pp_rank == 0:
            # Stage 0: run forward, send hidden_states + residual to stage 1
            if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
                hidden_states, residual = self.model(input_ids, positions)
            else:
                hidden_states, residual = self._run_graph_pp(input_ids, positions)

            dist.send(hidden_states, dst=1)
            if residual is not None:
                dist.send(residual, dst=1)
            return None  # Stage 0 doesn't produce logits

        else:
            # Stage 1: recv hidden_states + residual, run forward
            num_tokens = input_ids.size(0)
            hidden_states = torch.empty(num_tokens, hidden_size, device=f"cuda:{self.pp_rank}", dtype=hf_config.dtype)
            dist.recv(hidden_states, src=0)
            residual = torch.empty(num_tokens, hidden_size, device=f"cuda:{self.pp_rank}", dtype=hf_config.dtype)
            dist.recv(residual, src=0)

            # Run stage 1 layers + norm + lm_head
            hidden_states, _ = self.model(input_ids, positions, hidden_states, residual)
            return hidden_states  # This is logits from lm_head

    def _run_graph_pp(self, input_ids, positions):
        """CUDA graph replay for PP stage 0 decode."""
        bs = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return graph_vars["output_hidden"][:bs], graph_vars["output_residual"][:bs]

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        sample_params = self.prepare_sample(seqs) if self._is_last_stage() else None

        if self.pp_size > 1:
            return self._run_pp(seqs, input_ids, positions, is_prefill, sample_params)

        # Original non-PP path
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, *sample_params).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    def _run_pp(self, seqs, input_ids, positions, is_prefill, sample_params):
        """PP orchestration: stage 0 drives, stage 1 follows.

        Note: shm signaling is already done by _call_pp; this method only runs
        the local model forward and sampling.
        """
        logits = self.run_model(input_ids, positions, is_prefill)
        if self._is_last_stage():
            token_ids = self.sampler(logits, *sample_params).tolist()
            # Send token_ids back to stage 0
            token_tensor = torch.tensor(token_ids, dtype=torch.int64, device=f"cuda:{self.pp_rank}")
            dist.send(token_tensor, dst=0)
            reset_context()
            return token_ids
        else:
            # Receive token_ids from last stage
            num_seqs = len(seqs)
            token_tensor = torch.empty(num_seqs, dtype=torch.int64, device=f"cuda:{self.pp_rank}")
            dist.recv(token_tensor, src=self.pp_size - 1)
            reset_context()
            return token_tensor.tolist()

    @torch.inference_mode()
    def capture_cudagraph(self):
        # Non-first PP stages have no graph replay path in _run_model_pp,
        # so skip capture entirely.
        if self.pp_size > 1 and self.pp_rank != 0:
            self.graphs = {}
            self.graph_vars = {}
            self.graph_bs = []
            self.graph_pool = None
            return

        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        hidden_size = hf_config.hidden_size
        output_hidden = torch.zeros(max_bs, hidden_size)
        output_residual = torch.zeros(max_bs, hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            if self.pp_size > 1 and self.pp_rank == 0:
                # Stage 0: model returns (hidden_states, residual)
                output_hidden[:bs], output_residual[:bs] = self.model(input_ids[:bs], positions[:bs])
                with torch.cuda.graph(graph, self.graph_pool):
                    output_hidden[:bs], output_residual[:bs] = self.model(input_ids[:bs], positions[:bs])
            else:
                # Original path or last stage
                outputs = torch.zeros(max_bs, hidden_size)
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
                with torch.cuda.graph(graph, self.graph_pool):
                    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
                output_hidden = outputs
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            output_hidden=output_hidden,
            output_residual=output_residual,
        )
