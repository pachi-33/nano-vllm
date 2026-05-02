# 测试文档

## 目录结构

```
tests/
├── conftest.py              # 全局配置：GPU 设备选择、模型参数、参考实现
├── unit/                    # 单元测试（不需要模型，秒级完成）
│   ├── conftest.py          # 单元测试专用 fixtures（预留）
│   ├── test_store_kvcache.py       # KV 缓存写入 kernel 测试
│   ├── test_decode_attention.py    # Decode attention kernel 测试
│   └── test_prefill_attention.py   # Prefill attention kernel 测试
└── integration/             # 集成测试（需要加载模型，分钟级）
    ├── conftest.py                 # LLM Engine fixture、辅助工厂函数
    ├── test_single_prefill.py      # 单序列 prefill
    ├── test_batched_prefill.py     # 多序列 batched prefill
    ├── test_decode.py              # Decode 阶段（单步、多步、batched）
    ├── test_prefill_then_decode.py # 完整推理流水线（prefill → decode 循环）
    └── test_prefix_cache.py        # 前缀缓存（block 复用、正确性验证）
```

## 运行测试

```bash
# 跑全部单元测试（不需要模型，几秒钟）
pytest tests/unit/ -v

# 跑全部集成测试（需要模型，几分钟）
pytest tests/integration/ -v

# 在指定 GPU 上跑
pytest tests/unit/ -v --cuda-device cuda:1

# 跑特定测试文件
pytest tests/integration/test_decode.py -v

# 跑特定测试方法
pytest tests/unit/test_prefill_attention.py::TestPrefillNoPrefixCache::test_single_sequence -v

# 跑全部测试
pytest -v
```

## 添加新测试

### 新增单元测试

1. 在 `tests/unit/` 下创建 `test_xxx.py`
2. 使用 `tests/conftest.py` 中的 fixture：
   - `device`：CUDA 设备（通过 `--cuda-device` 指定，默认 `cuda:0`）
   - `small_config` / `default_config`：模型参数配置
   - `make_kv_cache` / `make_block_table` / `make_slot_mapping`：张量工厂函数
   - `naive_sdpa` / `naive_sdpa_decode` / `naive_sdpa_prefill_paged`：PyTorch 参考实现
3. 测试模式：构造输入 → 调用 Triton kernel → 对比 naive 参考实现的输出

### 新增集成测试

1. 在 `tests/integration/` 下创建 `test_xxx.py`
2. 使用 `tests/integration/conftest.py` 中的 fixture：
   - `engine`：LLMEngine 实例（session 级别，所有测试共享，只加载一次模型）
   - `scheduler`：已清理状态的 Scheduler（function 级别，每个测试前自动重置）
   - `tokenizer`：分词器
   - `model_runner`：模型运行器
   - `make_sequence`：创建 Sequence 对象的工厂函数
   - `do_prefill`：执行 prefill 步骤的辅助函数
   - `do_step`：执行一次完整 engine.step() 的辅助函数
3. 测试模式：构造 Sequence → do_prefill → do_step → 验证序列状态

## Fixture 说明

### 全局 fixture（tests/conftest.py）

| Fixture | 作用域 | 说明 |
|---------|--------|------|
| `device` | function | CUDA 设备，支持 `--cuda-device` 参数 |
| `small_config` | function | 小型模型配置（4 heads, 2 kv_heads, head_dim=64），用于快速测试 |
| `default_config` | function | Qwen3-0.6B 真实配置（16 heads, 8 kv_heads, head_dim=128） |
| `make_kv_cache` | function | 创建分页 KV 缓存张量的工厂 |
| `make_block_table` | function | 创建 block_table 张量的工厂 |
| `make_slot_mapping` | function | 创建 slot_mapping 张量的工厂 |
| `naive_sdpa` | — | 标准 SDPA 参考实现（函数，非 fixture） |
| `naive_sdpa_decode` | — | Decode attention 参考实现（函数，非 fixture） |
| `naive_sdpa_prefill_paged` | — | 分页 KV 的 Prefill attention 参考实现 |

### 集成测试 fixture（tests/integration/conftest.py）

| Fixture | 作用域 | 说明 |
|---------|--------|------|
| `engine` | session | LLMEngine 实例，所有集成测试共享 |
| `scheduler` | function | 每个测试前重置的 Scheduler（清空队列、重建 BlockManager） |
| `tokenizer` | function | 分词器 |
| `model_runner` | function | 模型运行器 |
| `make_sequence` | function | 创建 Sequence 对象的工厂 |
| `do_prefill` | function | 执行 prefill 的辅助函数 |
| `do_step` | function | 执行一次 engine.step() 的辅助函数 |

## 注意事项

- 集成测试的 `engine` fixture 是 session 级别的，只会加载一次模型。这意味着所有集成测试共享同一个 LLMEngine 实例
- 每个集成测试前 `scheduler` fixture 会清空等待/运行队列并重建 BlockManager，确保测试之间不互相干扰
- `TORCHDYNAMO_DISABLE=1` 由 `tests/conftest.py` 自动设置，无需手动 export
- 集成测试需要模型文件在 `~/huggingface/Qwen3-0.6B_fp16/`（在 `tests/integration/conftest.py` 中配置）
- 由于 `SamplingParams` 要求 `temperature > 0`（不支持贪心采样），输出是随机的。需要确定性输出的测试应使用非常低的 temperature
