# KV Cache 管理与调度机制

## Block 结构

KV Cache 以 block 为单位管理，每个 block 固定存储 `block_size`（默认 256）个 token 的 Key 和 Value。

```
KV Cache Tensor Shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
                         ↑ K/V  ↑ 层数      ↑ block数    ↑ token数    ↑ head数   ↑ 维度
```

每个 GPU 上的 block 数量根据可用显存动态计算：
```python
num_blocks = (total * gpu_util - used - peak + current) // block_bytes
```

## Block Manager

### Block 生命周期

```
空闲 → 分配 (allocate) → 使用中 → 释放 (deallocate) → 空闲
```

### Block 属性
- `block_id`: 唯一标识
- `ref_count`: 引用计数（支持 prefix sharing）
- `hash`: 基于 token 内容的哈希值（用于 prefix caching）
- `token_ids`: block 中存储的 token ID 列表

### Prefix Caching 机制

1. **分配前检查** (`can_allocate`):
   - 逐 block 计算 xxhash（含前缀哈希链）
   - 对比 `hash_to_block_id` 查找缓存命中
   - 只需为新 block 分配空间

2. **分配** (`allocate`):
   - 命中的 block: 增加引用计数，复用
   - 未命中的 block: 从空闲列表分配

3. **写入哈希** (`hash_blocks`):
   - 前向传播完成后，对新生成的 block 计算哈希
   - 维护 `hash_to_block_id` 映射

4. **释放** (`deallocate`):
   - 减少引用计数，归零时回收

## Scheduler 调度策略

### Prefill 阶段

```
等待队列 (waiting) → 调度 → 运行队列 (running)
```

- 从 waiting 队列头部取 sequence
- 受 `max_num_batched_tokens` 和 `max_num_seqs` 约束
- 支持 chunked prefill：长 prompt 可分多步处理
- 只有第一个 sequence 允许超过单步 token 预算

### Decode 阶段

```
运行队列 (running) → 解码 → 运行队列 (running)
```

- 对每个 running sequence 分配一个新 block（如果需要）
- 内存不足时执行抢占（preempt）：
  - 从 running 队列尾部取出 sequence
  - 释放其所有 KV cache block
  - 放回 waiting 队列头部，下次重新 prefill

### Sequence 状态转换

```
WAITING → (prefill开始) → RUNNING → (生成完成/EOS) → FINISHED
                                    → (被抢占) → WAITING
```

## Slot Mapping

Slot mapping 将 token 位置映射到 KV cache 中的物理槽位：

```python
slot = block_table[block_index] * block_size + offset_in_block
```

- Prefill: 为所有 scheduled tokens 计算 slot mapping
- Decode: 只计算最后一个 token 的 slot（`last_block_num_tokens - 1`）
- 特殊值 `-1` 表示 warmup，不写入 cache
