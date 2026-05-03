# Pipeline Parallelism (PP) 实现方案

## Context

nano-vllm 当前仅支持 Tensor Parallelism（TP），但 TP 的通信开销大（每层 all_reduce）。Pipeline Parallelism 将模型的不同层放在不同 GPU 上，通信仅需在层切分点传递 hidden_states，对大模型推理更友好。本方案从零设计 PP，支持 2 GPU 场景。

## 设计概览

```
GPU 0 (rank 0, pp_stage=0):
  embed_tokens → Layer[0] → ... → Layer[N//2-1] → send(hidden_states, residual)

GPU 1 (rank 1, pp_stage=1):
  recv(hidden_states, residual) → Layer[N//2] → ... → Layer[N-1] → norm → lm_head → logits
```

- **进程模型**：每个 pipeline stage 一个 spawn 进程（与现有 TP 一致）
- **通信**：NCCL `dist.send()` / `dist.recv()` 传递中间张量
- **KV cache**：每个 GPU 只存自己负责的层的 KV cache
- **调度**：严格顺序执行，无训练时的 pipeline bubble

## Step 1: 修改 Config — 添加 `pipeline_parallel_size`

**文件**: `nanovllm/config.py`

```python
@dataclass(slots=True)
class Config:
    ...
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1    # 新增
    pp_rank: int = 0                   # 新增，当前进程的 pipeline stage 编号
    ...
```

添加属性方法：

```python
@property
def num_stages(self):
    return self.pipeline_parallel_size

@property
def is_first_stage(self):
    return self.pp_rank == 0

@property
def is_last_stage(self):
    return self.pp_rank == self.pipeline_parallel_size - 1
```

## Step 2: 创建分片模型 — `PipelineStageModel`

**新文件**: `nanovllm/models/pipeline_stage.py`

将完整的 Qwen3ForCausalLM 拆分为各 stage 只包含自己负责的层：

```python
class PipelineStageModel(nn.Module):
    """单个 pipeline stage 的模型，只持有部分 decoder layers"""

    def __init__(self, hf_config, pp_rank, pp_size):
        super().__init__()
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        num_layers = hf_config.num_hidden_layers
        # 均匀切分
        layers_per_stage = num_layers // pp_size
        self.start_layer = pp_rank * layers_per_stage
        self.end_layer = self.start_layer + layers_per_stage

        # 第一个 stage 持有 embed_tokens
        if pp_rank == 0:
            self.embed_tokens = VocabParallelEmbedding(hf_config.vocab_size, hf_config.hidden_size)

        # 只实例化自己负责的层
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(hf_config)
            for _ in range(self.start_layer, self.end_layer)
        ])

        # 最后一个 stage 持有 norm + lm_head
        if pp_rank == pp_size - 1:
            self.norm = RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
            self.lm_head = ParallelLMHead(hf_config.vocab_size, hf_config.hidden_size)

    def forward(self, input_ids, positions, hidden_states=None, residual=None):
        # Stage 0: 从 input_ids 开始
        if self.pp_rank == 0:
            hidden_states = self.embed_tokens(input_ids)
            residual = None

        # 执行本 stage 的层
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        # 最后一个 stage: norm + lm_head
        if self.pp_rank == self.pp_size - 1:
            hidden_states, _ = self.norm(hidden_states, residual)
            residual = None
        return hidden_states, residual

    def compute_logits(self, hidden_states):
        """仅最后一个 stage 调用"""
        return self.lm_head(hidden_states)
```

**权重加载**：需要处理层名映射。原始权重名为 `model.layers.3.self_attn...`，本 stage 的层从 0 开始编号，需要重映射。在 `load_model` 中添加层索引偏移逻辑。

## Step 3: 修改 ModelRunner — 支持 pipeline stage

**文件**: `nanovllm/engine/model_runner.py`

核心变更：

1. **模型实例化**：用 `PipelineStageModel` 替代 `Qwen3ForCausalLM`
2. **KV cache 分配**：只为本 stage 的层分配 KV cache
3. **进程间通信**：stage 间 send/recv hidden_states 和 residual
4. **CUDA graph**：每个 stage 独立 capture 自己的 graph

```python
class ModelRunner:
    def __init__(self, config, rank, event):
        self.config = config
        self.pp_rank = config.pp_rank
        self.pp_size = config.pipeline_parallel_size
        ...

        # NCCL init
        dist.init_process_group("nccl", "tcp://localhost:2333",
                                world_size=self.pp_size, rank=self.pp_rank)
        torch.cuda.set_device(self.pp_rank)

        # 实例化分片模型
        self.model = PipelineStageModel(hf_config, self.pp_rank, self.pp_size)
        load_model(self.model, config.model)   # 需适配层名映射

        self.sampler = Sampler() if self.pp_rank == self.pp_size - 1 else None
        ...

    def allocate_kv_cache(self):
        """只为本 stage 的层分配 KV cache"""
        num_stage_layers = self.model.end_layer - self.model.start_layer
        # 计算可用显存（每张卡独享全部显存）
        ...
        self.kv_cache = torch.empty(2, num_stage_layers, ...)
        # 只绑定本 stage 的层的 k_cache/v_cache
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, local_layer_id]
                module.v_cache = self.kv_cache[1, local_layer_id]
                local_layer_id += 1

    def run_model(self, input_ids, positions, is_prefill):
        """Pipeline stage 的前向传播"""
        hidden_states = None
        residual = None

        # 非第一个 stage: 从上一个 stage 接收中间结果
        if self.pp_rank > 0:
            # 预分配接收缓冲区
            hidden_states = torch.empty(batch_size, hidden_size, device=f"cuda:{self.pp_rank}")
            residual = torch.empty(batch_size, hidden_size, device=f"cuda:{self.pp_rank}")
            dist.recv(hidden_states, src=self.pp_rank - 1)
            dist.recv(residual, src=self.pp_rank - 1)

        # 运行本 stage
        hidden_states, residual = self.model(input_ids, positions, hidden_states, residual)

        # 非最后一个 stage: 发送中间结果给下一个 stage
        if self.pp_rank < self.pp_size - 1:
            dist.send(hidden_states, dst=self.pp_rank + 1)
            dist.send(residual, dst=self.pp_rank + 1)
            return None  # 中间 stage 不产生 logits
        else:
            logits = self.model.compute_logits(hidden_states)
            return logits

    def run(self, seqs, is_prefill):
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.pp_rank == self.pp_size - 1 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.pp_rank == self.pp_size - 1 else None
        reset_context()
        return token_ids
```

## Step 4: 修改 LLMEngine — 启动 pipeline stage 进程

**文件**: `nanovllm/engine/llm_engine.py`

```python
class LLMEngine:
    def __init__(self, model, **kwargs):
        config = Config(model, **kwargs)
        pp_size = config.pipeline_parallel_size
        Sequence.block_size = config.kvcache_block_size

        self.ps = []
        ctx = mp.get_context("spawn")
        for i in range(1, pp_size):
            config_copy = copy(config)
            config_copy.pp_rank = i
            process = ctx.Process(target=ModelRunner, args=(config_copy, i, None))
            process.start()
            self.ps.append(process)

        config.pp_rank = 0
        self.model_runner = ModelRunner(config, 0, None)
        ...
```

注意：PP 不需要 shared memory 事件同步机制（TP 的 `write_shm`/`read_shm`），因为各 stage 严格顺序执行，NCCL 的 send/recv 本身就是同步点。

## Step 5: 适配权重加载

**文件**: `nanovllm/utils/loader.py`

需要处理层名映射：
- 原始: `model.layers.3.self_attn.qkv_proj.weight`
- Stage 0 的第 0 层: `layers.0.self_attn.qkv_proj.weight`（本地编号）
- Stage 1 的第 0 层: `layers.0.self_attn.qkv_proj.weight`（但实际对应原始第 N//2 层）

方案：在 `PipelineStageModel` 中给每层设置原始层索引属性，`load_model` 时根据原始索引匹配权重名。

```python
for i, layer in enumerate(self.model.layers):
    layer.original_layer_idx = self.model.start_layer + i
```

`load_model` 逻辑：

```python
def load_pipeline_model(model, path):
    for weight_name, weight in iterate_safetensors(path):
        # 提取层索引，重映射到本地索引
        # model.layers.{orig_idx}.xxx → model.layers.{local_idx}.xxx
        ...
```

## Step 6: CUDA Graph 适配

每个 stage 独立 capture CUDA graph：
- `capture_cudagraph()` 只 capture 本 stage 的模型
- graph 输入包括本 stage 需要的中间张量（非第一个 stage 需要接收的 hidden_states/residual）
- graph replay 后，非最后 stage 需要发送结果

注意：send/recv 不能在 CUDA graph 内部，需要在 graph replay 之后执行。

## Step 7: 其他适配

- **`nanovllm/layers/embed_head.py`**: `ParallelLMHead` 中 `dist.gather` 仅在 TP>1 时使用，PP 不影响
- **`nanovllm/layers/linear.py`**: `RowParallelLinear` 的 `all_reduce` 仅在 TP>1 时触发，PP 不影响
- **`nanovllm/engine/scheduler.py`**: 无需修改，调度逻辑与 PP 无关
- **`nanovllm/engine/block_manager.py`**: 无需修改，block 管理按 stage 独立运行

## 文件变更总结

| 文件 | 变更 |
|------|------|
| `nanovllm/config.py` | 添加 `pipeline_parallel_size`, `pp_rank` |
| `nanovllm/models/pipeline_stage.py` | **新建** — PipelineStageModel |
| `nanovllm/engine/model_runner.py` | 大改 — 支持分 stage 运行、send/recv 通信 |
| `nanovllm/engine/llm_engine.py` | 改 — 启动 PP worker 进程 |
| `nanovllm/utils/loader.py` | 改 — 支持层名重映射 |
| `example.py` | 改 — 演示 `pipeline_parallel_size=2` |

## 实现顺序

1. `config.py` — 添加 PP 配置字段
2. `pipeline_stage.py` — 实现分片模型 + 权重加载适配
3. `model_runner.py` — PP 感知的 ModelRunner（先不考虑 CUDA graph）
4. `llm_engine.py` — 启动 PP worker
5. 端到端验证（eager 模式）
6. CUDA graph 适配
7. 性能测试

## 验证方案

1. **正确性**：`pipeline_parallel_size=1` 输出应与原实现完全一致
2. **2-GPU 正确性**：`pipeline_parallel_size=2` 输出应与单卡一致（数值误差 < 1e-3）
3. **显存**：每张卡的 KV cache 大小应约为单卡的一半
4. **功能**：prefill、decode、prefix caching、chunked prefill 均应正常工作
