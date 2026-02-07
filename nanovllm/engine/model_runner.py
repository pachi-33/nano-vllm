"""
模型运行器模块

该模块提供了 ModelRunner 类，负责实际的模型计算和推理执行。
它是整个推理系统的执行引擎，承担了以下核心职责：

1. **初始化和管理**：配置分布式环境、加载模型、分配 KV 缓存
2. **张量并行计算**：支持多 GPU 分布式推理，通过 NCCL 进行通信
3. **内存管理**：智能分配 KV 缓存，管理 GPU 内存使用
4. **性能优化**：使用 CUDA Graph 加速小批量推理，内存锁页（pin memory）优化
5. **分布式协调**：主从进程间通信，共享内存数据传输
6. **数据处理**：准备输入数据、处理预填充（prefill）和解码（decode）阶段

使用场景：
    该类的实例通常在 LLMEngine 中被创建和管理，每个 GPU 会有一个实例
    负责执行模型推理的实际计算。
"""

import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """
    模型运行器类 - 核心模型计算引擎

    负责管理模型的实际推理执行，包括：
    - 分布式环境初始化和管理（张量并行）
    - KV 缓存的分配和管理
    - 输入数据的预处理
    - 模型前向计算执行
    - 使用 CUDA Graph 加速推理
    - 采样和后处理

    在多 GPU 环境下，该类支持主从模式：
    - rank 0 为主进程，负责协调其他进程
    - rank > 0 为从进程，等待主进程的指令
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        初始化模型运行器

        初始化过程包括：
        1. 设置分布式环境（如果需要张量并行）
        2. 加载模型和权重
        3. 预热模型（warmup）
        4. 分配 KV 缓存
        5. 捕获 CUDA Graph（如果启用）
        6. 在主进程中创建共享内存，或在从进程中进入响应循环

        Args:
            config: 配置对象，包含模型和推理参数
            rank: GPU 设备号（0, 1, 2, ...）
            event: 事件对象（单个事件或多个事件列表）
                   - 单个事件: 从进程使用，等待主进程命令
                   - 事件列表: 主进程使用，控制所有从进程

        Note:
            如果启用了张量并行（world_size > 1），会将模型分布在多个 GPU 上。
            rank 为 0 的进程是主进程，负责协调计算；其他为从进程，等待指令。
        """
        # 保存配置和基本参数
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # 初始化 NCCL 分布式进程组，用于张量并行
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)  # 设置当前 GPU 设备

        # 临时更改为模型的默认 dtype 和设备
        default_dtype = torch.get_default_dtype()

        if default_dtype == torch.bfloat16:
            print('switch bf16 to fp16')
            default_dtype = torch.float16
        else:
            print('default_dtype: ', default_dtype)

        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")

        # 创建模型实例并加载权重
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)

        # 创建采样器
        self.sampler = Sampler()

        # 预热模型（测试最大批处理规模）
        self.warmup_model()

        # 分配 KV 缓存
        self.allocate_kv_cache()

        # 捕获 CUDA Graph 以加速小批量推理（除非强制使用 eager 模式）
        if not self.enforce_eager:
            self.capture_cudagraph()

        # 恢复默认的 dtype 和设备设置
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 如果启用张量并行，设置共享内存用于进程间通信
        if self.world_size > 1:
            if rank == 0:
                # 主进程：创建共享内存
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()  # 等待所有从进程准备好
            else:
                # 从进程：等待主进程创建共享内存，然后进入响应循环
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()  # 从进程进入循环，等待主进程指令

    def exit(self):
        """
        退出并清理资源

        该方法会进行以下清理操作：
        1. 关闭共享内存（如果是从进程）
        2. 等待所有进程同步
        3. 删除共享内存对象（如果是主进程）
        4. 释放 CUDA Graph 资源（如果建立了）
        5. 同步 CUDA 设备
        6. 销毁分布式进程组
        """
        if self.world_size > 1:
            self.shm.close()  # 关闭共享内存
            dist.barrier()  # 等待所有进程到达
            if self.rank == 0:
                self.shm.unlink()  # 主进程删除共享内存对象
        if not self.enforce_eager:
            del self.graphs, self.graph_pool  # 释放 CUDA Graph 资源
        torch.cuda.synchronize()  # 等待所有 CUDA 操作完成
        dist.destroy_process_group()  # 销毁 NCCL 进程组

    def loop(self):
        """
        从进程的主循环方法

        在从进程中运行的无限循环，持续执行以下操作：
        1. 从共享内存读取主进程的命令
        2. 调用对应的本地方法
        3. 如果是 exit 命令则退出循环

        Note:
            该方法只在 rank > 0 的从进程中运行，rank 0 的主进程直接调用方法。
        """
        while True:
            method_name, args = self.read_shm()  # 读取主进程命令
            self.call(method_name, *args)       # 执行命令
            if method_name == "exit":            # 退出循环
                break

    def read_shm(self):
        """
        从共享内存读取命令和数据

        从进程使用该方法读取主进程通过共享内存发送的命令：
        1. 等待事件信号（主进程已写入数据）
        2. 读取数据大小（前 4 字节）
        3. 读取并反序列化数据
        4. 清除事件信号，准备下一次读取

        Returns:
            tuple: (method_name, args) 命令名称和参数列表

        Note:
            该方法只在 rank > 0 的从进程中调用。
        """
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()                                                      # 等待主进程写入数据
        n = int.from_bytes(self.shm.buf[0:4], "little")                        # 读取数据大小
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])                 # 反序列化数据
        self.event.clear()                                                     # 清除事件，允许主进程再次写入
        return method_name, args

    def write_shm(self, method_name, *args):
        """
        向共享内存写入命令和数据

        主进程使用该方法向所有从进程发送命令：
        1. 序列化方法名和参数
        2. 写入数据大小（前 4 字节）
        3. 写入序列化数据
        4. 触发事件通知所有从进程读取

        Args:
            method_name: 要调用的方法名称
            *args: 方法的参数列表

        Note:
            该方法只在 rank == 0 的主进程中调用。
        """
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])                             # 序列化数据
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")                           # 写入数据大小
        self.shm.buf[4:n+4] = data                                            # 写入数据
        for event in self.event:
            event.set()                                                       # 通知所有从进程读取

    def call(self, method_name, *args):
        """
        通用方法调用接口

        该方法会根据当前设备和并行模式：
        - 如果是主进程且启用了多 GPU，先写入共享内存通知从进程
        - 调用指定的本地方法

        Args:
            method_name: 要调用的方法名称
            *args: 方法的参数列表

        Returns:
            被调用方法的返回值
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)  # 通知从进程
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """
        模型预热

        预热模型以确定最大批处理能力和分配 KV 缓存。
        方法创建一个包含最大可能序列数的伪请求批次，执行一次前向计算。

        Process:
            1. 清空 CUDA 缓存
            2. 重置内存统计
            3. 计算给定批处理 token 限制下能容纳的最大序列数
            4. 创建虚拟序列并执行推理
            5. 清空 CUDA 缓存
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)  # 执行预热推理
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        分配 KV 缓存

        根据 GPU 内存使用情况和配置文件，智能计算并分配 KV 缓存。
        这是模型高效运行的关键步骤，缓存了之前计算的 Key 和 Value 矩阵，
        避免在每一步生成时重新计算。

        计算逻辑：
        1. 获取当前 GPU 内存使用情况（已使用、峰值等）
        2. 计算 KV 缓存的维度（考虑张量并行分割的头部数和隐藏维度）
        3. 计算每个缓存块占用的字节数（2 × 层数 × 块大小 × 头数 × 头维度 × 数据类型大小）
        4. 根据配置的内存利用率，计算可用的缓存块数量
        5. 创建大的 KV 缓存张量（形状：[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]）
        6. 为模型的每一层分配指向 KV 缓存的引用

        Note:
            该方法会在预热后调用，以确保准确的内存使用量统计。
            返回的 KV 缓存块数会在调度器中使用，用于管理请求的内存分配。

        Raises:
            AssertionError: 如果计算出的缓存块数小于等于 0，表示内存不足
                           无法分配任何 KV 缓存块
        """
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        # 计算 KV 缓存的头部数量（考虑张量并行）和维度
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

        # 计算每个 KV 缓存块的字节大小
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize

        # 根据内存使用率和利用率配置计算可用的缓存块数量
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0

        # 创建大的 KV 缓存张量（形状：[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]）
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)

        # 为模型的每一层分配指向 KV 缓存的引用
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """
        准备块表（Block Tables）用于 KV 缓存寻址

        块表是一个二维张量，记录每个序列的 KV 缓存块索引。
        在分布式推理和变长序列批处理中，块表用于快速定位每个位置的 KV 值。

        Args:
            seqs: 序列列表，每个序列有一个块表记录分配的缓存块

        Returns:
            torch.Tensor: 填充后的块表张量，形状 [num_seqs, max_block_table_len]
                       其中 -1 表示该位置没有分配块（填充）
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        准备预填充（Prefill）阶段的输入数据

        预填充阶段用于处理新请求的初始计算，包括：
        - 收集所有新 token（从 num_cached_tokens 之后）
        - 构建查询（q）和键值（k/v）的长度序列
        - 创建 slot 映射用于 KV 缓存
        - 为 Flash Attention 准备 cu_seqlens

        Args:
            seqs: 序列列表，需要预填充计算的序列

        Returns:
            tuple: (input_ids, positions)
                - input_ids: 需要计算的输入 token ID
                - positions: 对应的 token 位置（用于位置编码）

        Note:
            该方法会设置上下文，包含 cu_seqlens_q、cu_seqlens_k、max_seqlen、
            slot_mapping 等信息，用于后续的 Flash Attention 计算。
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None

        # 遍历所有序列，收集输入和元数据
        for seq in seqs:
            seqlen = len(seq)
            # 只处理缓存之后的新 token
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))

            # 计算查询（q）和键值（k/v）的长度
            seqlen_q = seqlen - seq.num_cached_tokens  # 需要计算的 token 数（查询）
            seqlen_k = seqlen                          # 完整的序列长度（键值）

            # 累积序列长度（用于 Flash Attention 的变长序列处理）
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            # 更新最大序列长度
            # 前缀和
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            # 跳过预热的序列（没有块表）
            if not seq.block_table:
                continue

            # 为新分配的块创建 slot 映射
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size  # 块起始位置
                if i != seq.num_blocks - 1:
                    end = start + self.block_size          # 完整块
                else:
                    end = start + seq.last_block_num_tokens  # 最后一个可能不完整的块
                slot_mapping.extend(list(range(start, end)))

        # 检查是否需要块表（存在前缀缓存）
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)

        # 将数据转换为 CUDA 张量（使用锁页内存加速）
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # 设置上下文（包含 Flash Attention 需要的元数据）
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)

        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        准备解码（Decode）阶段的输入数据

        解码阶段每次只生成一个 token，相比预填充阶段更简单。
        该方法为每个正在解码的序列收集最后一个 token 及其元数据。

        Args:
            seqs: 序列列表，正在解码的序列

        Returns:
            tuple: (input_ids, positions)
                - input_ids: 当前要处理的 token ID（每个序列最后一个）
                - positions: token 的位置（序列长度减 1）

        Note:
            该方法还会设置上下文，包含 slot_mapping、context_lens 和 block_tables，
            这些信息用于在 KV 缓存中定位每个位置的键值对。
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        # 收集每个序列的最后一个 token 和元数据
        for seq in seqs:
            input_ids.append(seq.last_token)  # 序列的最后一个 token
            positions.append(len(seq) - 1)     # token 的位置
            context_lens.append(len(seq))      # 上下文长度（完整序列长度）

            # 计算最后一个 token 在 KV 缓存中的 slot 位置
            # 公式：块索引 × 块大小 + 块内位置
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)

        # 将数据转换为 CUDA 张量（使用锁页内存加速）
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # 准备块表（用于 KV 缓存寻址）
        block_tables = self.prepare_block_tables(seqs)

        # 设置上下文
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)

        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """
        准备采样参数

        从每个序列中提取采样温度参数，用于控制生成文本的随机性：
        - 温度越高，生成越随机
        - 温度越低，生成越确定

        Args:
            seqs: 序列列表，每个序列包含采样配置

        Returns:
            torch.Tensor: 温度张量，形状 [num_seqs]，已移动到 GPU
        """
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        执行模型推理计算，支持两种模式：标准执行和 CUDA Graph 加速

        该方法根据输入批大小和配置选择最优的执行方式：
        1. **预填充（Prefill）**：使用标准 eager 模式，因为计算量大且不规则
        2. **小批量解码**：使用 CUDA Graph 加速，避免 Python 开销
        3. **大批量解码**（>512）：使用 eager 模式，避免 CUDA Graph 内存限制

        工作流程：
        - 如果是预填充或强制 eager 模式或批大小超过 512：
          直接调用模型前向计算然后计算 logits
        - 否则：
          1. 选择与批大小匹配的预捕获 CUDA Graph
          2. 将输入数据填充到固定的 CUDA Graph 缓冲区
          3. 重放 CUDA Graph（执行预记录的计算图）
          4. 计算并返回 logits

        Args:
            input_ids: 输入 token ID 张量，形状 [batch_size, seq_len]
            positions: 位置 ID 张量，形状 [batch_size, seq_len]
            is_prefill: 是否处于预填充阶段（True=预填充，False=解码）

        Returns:
            torch.Tensor: 模型输出的 logits，形状 [batch_size, vocab_size]
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # 预填充或强制 eager 模式或大批量：使用标准方式执行
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # 小批量解码：使用 CUDA Graph 加速
            bs = input_ids.size(0)
            context = get_context()

            # 选择合适的 CUDA Graph（查找第一个满足 graph_bs >= bs 的图）
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars

            # 将动态输入数据填充到 CUDA Graph 的固定缓冲区
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions

            # 初始化 slot_mapping（无效位置用-1填充）
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping

            # 初始化 context_lens（序列长度）
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens

            # 设置块表（用于 KV 缓存寻址）
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

            # 重放 CUDA Graph（执行预录制的计算图）
            graph.replay()

            # 计算并返回 logits
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        运行模型推理的完整流程

        这是模型推理的主入口方法，协调整个推理流程：
        1. 根据阶段准备输入数据（预填充或解码）
        2. 准备采样参数（rank 0 负责）
        3. 执行模型前向计算
        4. 采样生成新的 token（rank 0 负责）
        5. 清理上下文信息

        在分布式环境中，只有 rank 0 的进程会进行采样，其他进程返回 None。

        Args:
            seqs: 要处理的序列列表
            is_prefill: 是否处于预填充阶段（True=预填充，False=解码）

        Returns:
            list[int] | None: 生成的 token ID 列表，仅在主进程（rank==0）返回
        """
        # 准备输入：根据阶段选择预填充或解码的数据准备
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)

        # 准备采样参数：只有主进程需要
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None

        # 执行模型前向计算
        logits = self.run_model(input_ids, positions, is_prefill)

        # 采样生成新 token：只有主进程执行
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

        # 重置上下文，准备下一次推理
        reset_context()

        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        捕获 CUDA Graph 以加速解码阶段推理

        CUDA Graph 通过预录制推理计算图，消除运行时 Python 开销和内核启动开销，
        特别适用于需要连续执行、数据结构变化小的短序列推理（如自回归生成）。

        捕获流程：
        1. 准备工作：确定支持的批大小范围，创建较大的张量缓冲区
        2. 对每个支持的批大小（从大到小）：
           a. 设置输入数据（填充到同一缓冲区）
           b. 预热模型（运行一次 eager 模式）
           c. 捕获 CUDA Graph（记录所有 CUDA 操作）
           d. 保存 Graph 到字典，key 为批大小
        3. 存储所有 Graph 使用的变量缓冲区

        支持的批大小：优化集合 [1, 2, 4, 8, 16, 32, 48, 64, ..., 512]
        实际使用时会选择 >= 实际批大小 的最小预设 Graph。

        Note:
            捕获时需要模拟真实推理场景，包括设置 slot_mapping、context_lens 等元数据。
            从大到小捕获可以利用 CUDA Graph 的内存池共享，减少内存碎片。
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)  # 最大支持的批大小
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size  # 最大块表长度

        # 创建大容量缓冲区（将会被所有 Graph 复用）
        input_ids = torch.zeros(max_bs, dtype=torch.int64)              # 输入 token IDs
        positions = torch.zeros(max_bs, dtype=torch.int64)              # 位置编码
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)           # KV 缓存槽位映射
        context_lens = torch.zeros(max_bs, dtype=torch.int32)            # 上下文长度
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)  # 块表
        outputs = torch.zeros(max_bs, hf_config.hidden_size)            # 模型输出

        # 优化的批大小集合：小批量用幂次方，大批量用 16 为步长
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}  # 存储捕获的 Graph：key=批大小，value=CUDAGraph
        self.graph_pool = None  # Graph 内存池，用于内存复用

        # 从大到小捕获（可以复用内存池）
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()  # 创建新 Graph

            # 设置上下文：模拟解码阶段的元数据
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])

            # 预热：确保所有数据类型和形状正确
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

            # 捕获 Graph：记录所有 CUDA 操作
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # 这次执行将被录制

            # 第一个 Graph 创建内存池，后续 Graph 复用
            if self.graph_pool is None:
                self.graph_pool = graph.pool()

            # 保存捕获的 Graph
            self.graphs[bs] = graph

            # 同步并清理上下文
            torch.cuda.synchronize()
            reset_context()

        # 存储所有 Graph 共享的变量缓冲区（运行时使用）
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
