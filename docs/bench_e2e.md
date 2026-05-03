# bench_e2e.py — 端侧推理 Benchmark

面向端侧/本地部署场景的推理 Benchmark 套件，覆盖延迟体验、显存占用、长时间稳定性、模型质量四个维度。

## 快速开始

```bash
# 跑全部模式 + 全部场景
python bench_e2e.py --model ~/huggingface/Qwen3-0.6B_fp16/

# 跑单个模式
python bench_e2e.py --mode latency
python bench_e2e.py --mode memory
python bench_e2e.py --mode stress --duration 900
python bench_e2e.py --mode quality

# 指定场景
python bench_e2e.py --mode latency --workload doc_qa

# 流水线并行
python bench_e2e.py --mode latency --pipeline-parallel-size 2

# 结果保存为 JSON
python bench_e2e.py --output result.json
```

## 四种测试模式

### 1. latency — 速度与延迟

测量 prefill/decode 吞吐量、首 Token 延迟 (TTFT)、端到端延迟。批量提交所有 prompt，逐步调度并记录每个 step 的耗时。

输出示例：

```
[Latency] Scenario: short_chat (50 prompts)
  Prompt len:    80 tokens
  Max output:    200 tokens
  TTFT:          avg=42.1ms  p50=42.1ms  p99=42.2ms
  Prefill:       1899 tok/s
  Decode:        avg=51 tok/s  p50=51 tok/s
  E2E latency:   avg=3.76s  p50=3.94s
```

### 2. memory — 显存占用

在生成前后记录 `torch.cuda.memory_stats()`，报告峰值分配、当前分配和总 VRAM。

输出示例：

```
[Memory] Scenario: doc_qa
  Peak allocated:    9.22 GB
  Current allocated: 9.07 GB
  Peak total VRAM:   9.22 GB
  Total VRAM:        15.77 GB
```

### 3. stress — 压力烤机

循环生成指定时长（默认 15 分钟），按分钟采样 decode speed，输出速度衰减百分比。用于检测设备发热降频和内存泄漏。

输出示例：

```
[Stress] Duration: 900s (312 iterations)
  Minute  1:  99 tok/s
  Minute  5:  98 tok/s
  Minute 15:  97 tok/s
  Throttle:   -2.0%
```

### 4. quality — 模型质量

两项检测：

- **Perplexity (PPL)**：在 Wikitext-2 test 集上用滑动窗口 (window=2048, stride=512) 计算困惑度。需要安装 `datasets` 库。PP 模式下跳过（因为直接用 HF 模型计算）。
- **Sanity Check**：30 道常识题，检查模型输出是否包含预期关键词。用于检测量化后的精度损失或输出退化。

输出示例：

```
[Quality]
  PPL (wikitext2):  12.34
  Sanity Check:     28/30 passed
```

## 三种测试场景 (Workloads)

| 场景 | prompt 长度 | 输出长度 | 请求数 | 测试重点 |
|------|-----------|---------|--------|---------|
| `short_chat` | 80 tokens | 200 tokens | 50 | 冷启动、绝对延迟 |
| `doc_qa` | 3000 tokens | 100 tokens | 10 | Prefill 速度、KV cache 内存 |
| `code_gen` | 200 tokens | 1024 tokens | 5 | Decode 稳定性、长时间内存 |

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `~/huggingface/Qwen3-0.6B_fp16/` | 模型路径 |
| `--mode` | `all` | 测试模式：`all` / `latency` / `memory` / `stress` / `quality` |
| `--workload` | `all` | 测试场景：`all` / `short_chat` / `doc_qa` / `code_gen` |
| `--duration` | `900` | stress 模式持续时间（秒） |
| `--pipeline-parallel-size` | `1` | 流水线并行度 |
| `--tensor-parallel-size` | `1` | 张量并行度 |
| `--enforce-eager` | `False` | 禁用 CUDA graph，使用 eager 模式 |
| `--output` | 无 | 将结果保存为 JSON 文件 |

## 依赖

- 必需：`torch`、`nanovllm`
- PPL 测试额外需要：`datasets`、`transformers`

```bash
pip install datasets
```
