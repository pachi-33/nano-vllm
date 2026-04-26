# Tensor Parallelism 实现细节

## 进程模型

```
主进程 (rank 0)
  ├── LLMEngine
  │    ├── Scheduler
  │    └── ModelRunner (rank 0)
  └── 共享内存写入

Worker 进程 (rank 1..N-1)
  └── ModelRunner (rank i)
       └── 循环读取共享内存 → 执行命令
```

### 通信方式
- **控制面**: 共享内存 (`SharedMemory("nanovllm")`, 1MB) + `multiprocessing.Event`
  - Rank 0 将 (method_name, args) pickle 序列化后写入
  - Worker 通过 Event 被唤醒，读取并执行
- **数据面**: NCCL (`nccl` backend, `tcp://localhost:2333`)
  - `all_reduce`: RowParallelLinear 的输出同步
  - `gather`: ParallelLMHead 的 logits 汇聚到 rank 0
  - `barrier`: 同步点

## 并行策略

### Column Parallel (QKV, Gate-Up)
```
权重 [out_features, in_features] → 按 dim=0 切分
每个 rank 持有 [out_features/tp, in_features] 的分片
输出在 feature 维度上是完整输出的 1/tp
```

### Row Parallel (O_proj, Down_proj)
```
权重 [out_features, in_features] → 按 dim=1 切分
每个 rank 持有 [out_features, in_features/tp] 的分片
forward 后 all_reduce 聚合结果
```

### QKVParallelLinear
```
总输出 = [Q | K | V] = [(num_heads + 2*num_kv_heads) * head_dim]
每个 rank 持有: Q[num_heads/tp * head_dim], K[num_kv_heads/tp * head_dim], V[num_kv_heads/tp * head_dim]
```

### MergedColumnParallelLinear (Gate-Up)
```
总输出 = [gate | up] = [intermediate_size * 2]
每个 rank 持有: gate[intermediate_size/tp], up[intermediate_size/tp]
```

### VocabParallelEmbedding / ParallelLMHead
```
词表按 tp 切分，每个 rank 负责 vocab_size/tp 个 token
Embedding: mask → 局部查表 → all_reduce
LM Head: 局部计算 logits → gather 到 rank 0
```

## 权重加载

每个 `nn.Parameter` 附带 `weight_loader` 属性，在 `load_model()` 时根据 rank 自动切分：

```python
# ColumnParallelLinear.weight_loader
shard_size = param.size(tp_dim)
start_idx = tp_rank * shard_size
loaded_weight = loaded_weight.narrow(tp_dim, start_idx, shard_size)
param.copy_(loaded_weight)
```

`packed_modules_mapping` 处理融合参数的映射：
```python
# HuggingFace: model.layers.0.self_attn.q_proj.weight
# 映射到:     model.layers.0.self_attn.qkv_proj.weight 的 q 分片
```
