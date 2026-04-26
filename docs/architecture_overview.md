# Nano-vLLM 架构概览

## 整体架构

Nano-vLLM 是一个轻量级的 LLM 推理引擎，代码约 1200 行 Python，实现了 vLLM 的核心功能。

```
LLM (入口)
 └── LLMEngine (推理引擎)
      ├── Scheduler (调度器)
      │    └── BlockManager (KV Cache 块管理)
      └── ModelRunner (模型运行器)
           ├── Qwen3ForCausalLM (模型)
           │    ├── Qwen3Model
           │    │    ├── VocabParallelEmbedding
           │    │    ├── Qwen3DecoderLayer × N
           │    │    │    ├── Qwen3Attention
           │    │    │    │    ├── QKVParallelLinear
           │    │    │    │    ├── RMSNorm (q_norm, k_norm)
           │    │    │    │    ├── RotaryEmbedding
           │    │    │    │    ├── Attention (flash-attn)
           │    │    │    │    └── RowParallelLinear (o_proj)
           │    │    │    └── Qwen3MLP
           │    │    │         ├── MergedColumnParallelLinear (gate_up_proj)
           │    │    │         ├── SiluAndMul
           │    │    │         └── RowParallelLinear (down_proj)
           │    │    └── RMSNorm
           │    └── ParallelLMHead
           └── Sampler (采样器)
```

## 推理流程

### 初始化阶段
1. 创建 `Config`，加载 HuggingFace 模型配置
2. 如果 TP > 1，spawn worker 进程，通过共享内存通信
3. 初始化 NCCL 进程组 (`tcp://localhost:2333`)
4. 创建模型，加载 safetensors 权重（支持 TP 分片）
5. Warmup 模型（确定 GPU 内存占用）
6. 根据剩余 GPU 内存分配 KV Cache（paged，block_size=256）
7. 如果 `enforce_eager=False`，捕获 CUDA Graph（decode 阶段专用）

### 推理步骤 (`step()`)
1. **调度** (`Scheduler.schedule()`)
   - Prefill 阶段：从等待队列取出 sequence，分配 KV cache 块，支持 chunked prefill
   - Decode 阶段：对运行中的 sequence 逐一解码，内存不足时抢占（preempt）
2. **模型前向** (`ModelRunner.run()`)
   - Prefill：拼接所有 sequence 的 input_ids，用 `flash_attn_varlen_func`
   - Decode：每个 sequence 一个 token，用 `flash_attn_with_kvcache`
   - CUDA Graph replay（decode 且 batch ≤ 512）
3. **后处理** (`Scheduler.postprocess()`)
   - 追加生成 token，更新 KV cache 哈希
   - 检查 EOS 和 max_tokens，完成时释放 KV cache 块

## 关键设计决策

### Paged KV Cache + Prefix Caching
- KV cache 按 block（256 tokens）管理，类似操作系统分页
- 使用 xxhash 对 block 内容做哈希，实现 prefix caching
- 共享 prefix 的 sequence 可以复用相同的 KV cache block（通过引用计数）

### 全局上下文 (Context)
- 使用全局 `_CONTEXT` 对象在 ModelRunner 和各层之间传递调度元数据
- 避免了将 slot_mapping、block_tables 等参数逐层传递
- 每次 forward 前设置，forward 后重置

### Tensor Parallelism
- 通过 spawn 多进程实现，每个 GPU 一个进程
- Rank 0 通过共享内存广播命令给 worker
- ColumnParallel: 权重按输出维度切分
- RowParallel: 权重按输入维度切分，forward 后 all_reduce
- LM Head: vocabulary 维度切分，gather 到 rank 0

### CUDA Graph
- 仅用于 decode 阶段（batch ≤ 512）
- 预捕获不同 batch size 的图: [1, 2, 4, 8, 16, 32, ..., 512]
- 使用共享 graph pool 节省内存
- 每步 replay 时填入实际数据

### torch.compile
- 用于 RMSNorm、RoPE、Sampler、SiluAndMul 等计算密集操作
- 第一次调用时编译，后续复用编译结果
