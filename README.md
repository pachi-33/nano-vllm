<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

从零开始构建的 LLM 推理引擎，约 3,000 行 Python。不是对 vLLM 的封装或精简，而是一次完整的重新实现——用自己的代码讲清楚 LLM 推理的核心思想。

## 为什么做这个项目？

生产级推理引擎（vLLM、TensorRT-LLM、SGLang）功能强大，但动辄十几万行代码、子系统深度耦合，想要理解"它到底怎么工作的"并不容易。

Nano-vLLM 的出发点很简单：**用最少、最清晰的代码，把 LLM 推理引擎的每个关键模块从头实现一遍。** 不是为了替代生产引擎，而是为了让你能在一个下午读完整个项目，真正理解 PagedAttention、调度策略、CUDA Graph、分布式并行这些技术背后的原理。

而在这个过程中，我们也在持续加入生产级的特性——流水线并行、OpenAI 兼容 API、自定义 Triton kernel、完整的测试框架——让它不仅是教学工具，也可以作为推理引擎研究的实验平台。

## 性能

在同等硬件条件下，Nano-vLLM 的推理吞吐与 vLLM 处于同一水平：

| 推理引擎 | 输出 Token 数 | 耗时 (s) | 吞吐量 (tokens/s) |
|---------|-------------|---------|-------------------|
| vLLM | 133,966 | 98.37 | 1,361 |
| Nano-vLLM | 133,966 | 93.41 | 1,434 |

测试环境：RTX 4070 Laptop 8GB / Qwen3-0.6B / 256 条请求 / 输入输出长度 100-1024 tokens 随机采样。

## 功能

**推理引擎**
- 分页 KV Cache + 前缀缓存（xxhash 哈希匹配，相同前缀复用缓存块）
- CUDA Graph 捕获（Decode 阶段预捕获 batch size 1-512 的计算图，每步回放）
- Chunked Prefill（长 prompt 分步处理，不阻塞 decode）
- `torch.compile` 优化（融合 SiLU+gate、编译 RoPE、编译 RMSNorm 残差加法）

**分布式并行**
- 张量并行（TP）—— Column/Row 并行线性层，NCCL all-reduce 通信
- 流水线并行（PP）—— 模型按层切分到多卡，stage 间通过 NCCL P2P 传递激活

**推理服务**
- OpenAI 兼容 API 服务器—— `/v1/chat/completions`、`/v1/completions`、`/v1/models`
- 流式输出—— Chat 和 Completion 端点均支持 SSE 流式响应
- 命令行启动—— `python -m nanovllm.serve --model <path> --pp 2`

**自定义 Kernel**
- Flash Attention 集成—— Prefill 使用 `flash_attn_varlen_func`，Decode 使用 `flash_attn_with_kvcache`
- Triton KV Cache 写入—— 向量化分页写入，支持部分 block 的掩码处理
- Gumbel 采样—— 指数噪声 + argmax 实现可微采样

**测试框架**
- 三层测试体系—— 单元测试（kernel 正确性）、集成测试（引擎行为）、服务测试（HTTP API）
- 参考实现—— 使用纯 PyTorch SDPA 作为 kernel 正确性基线

## 架构

```
用户代码
   │
   ▼
  LLM ─────────────────────────────────────────────────
   │                                                    │
   ▼                                                    │
LLMEngine                                               │
   │                                                    │
   ├─► Scheduler ─► BlockManager (分页 KV Cache)       │
   │        │                                            │
   │        └─ Prefill/Decode 调度                      │
   │           抢占式调度 + Chunked Prefill               │
   │                                                    │
   └─► ModelRunner ─► Qwen3 Model                      │
           │              │                              │
           │              ├─ Attention (Flash Attention) │
           │              ├─ RoPE (torch.compile)        │
           │              ├─ RMSNorm (torch.compile)     │
           │              ├─ SwiGLU (融合 SiLU+mul)      │
           │              └─ LM Head (词表并行)          │
           │                                            │
           ├─► CUDA Graph (Decode 回放)                 │
           └─► KV Cache (单一连续张量)                   │
                                                        │
   全局 Context ◄── slot_mapping, block_table 等 ───────┘
```

引擎采用单步迭代模型：`schedule() → run() → postprocess()`。Scheduler 决定推进哪些序列、是 prefill 还是 decode。ModelRunner 执行前向计算并采样。BlockManager 管理分页 KV Cache 的分配与前缀缓存。

一个线程局部的 `Context` 对象将调度元数据（slot mapping、block table 等）从 ModelRunner 传递到 Attention 层，避免在每个层的 forward 签名中透传这些张量。

## 快速开始

### 安装

```bash
pip install -e .
# 或
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

需要 Python 3.10-3.13 和 CUDA GPU。

### 下载模型

```bash
huggingface-cli download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

### 离线推理

```python
from nanovllm import LLM, SamplingParams

llm = LLM("~/huggingface/Qwen3-0.6B/", enforce_eager=True)
params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["你好，Nano-vLLM。"], params)
print(outputs[0]["text"])
```

### 流水线并行

```python
llm = LLM("~/huggingface/Qwen3-0.6B/", enforce_eager=True, pipeline_parallel_size=2)
```

### 启动 API 服务器

```bash
# 单卡
python -m nanovllm.serve --model ~/huggingface/Qwen3-0.6B_fp16/

# 流水线并行（2 卡）
CUDA_VISIBLE_DEVICES=0,1 python -m nanovllm.serve \
  --model ~/huggingface/Qwen3-0.6B_fp16/ \
  --pipeline-parallel-size 2 --enforce-eager
```

然后使用任何兼容 OpenAI 的 SDK 调用：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="Qwen3-0.6B_fp16",
    messages=[{"role": "user", "content": "你好！"}],
    max_tokens=64,
)
```

### 运行测试

```bash
pytest tests/unit/ -v           # Kernel 测试，无需模型
pytest tests/integration/ -v    # 引擎测试，需要模型
pytest tests/server/ -v -s      # API 测试，需要模型 + httpx
```

## 项目特色

**与生产引擎的核心区别：**

生产引擎追求"能用"——覆盖百种模型、适配各种硬件、处理各种边界情况。这带来了巨大的代码复杂度。Nano-vLLM 追求"能懂"——每个模块只做一件事，每个设计决策都能在代码中看到原因。然后在"能懂"的基础上，逐步叠加生产级能力。

**具体来说：**

- **可控的复杂度**：3,000 行代码实现完整的推理引擎，包括分布式并行和自定义 kernel。整个项目的代码量大约相当于 vLLM 一个子模块
- **可读的 Kernel**：使用 Triton 而非 CUDA C++ 编写自定义算子，Python 原生的 kernel 代码配有详细注释
- **教学级测试**：每个 kernel 都有对应的纯 PyTorch 参考实现，测试即文档
- **持续进化的能力**：在保持代码简洁的前提下，逐步加入流水线并行、API 服务、完整的测试框架等生产级特性

## 开发计划

### 近期目标
- **Chunked Prefill 优化** —— 进一步优化长上下文场景下的 prefill 分块策略
- **KV Cache Offloading** —— 将冷 KV Cache 页面卸载到 CPU 内存，在有限显存下支持更长的上下文
- **模型量化** —— INT8/INT4 仅权重量化（AWQ/GPTQ 风格），降低显存占用、提升吞吐

### 中期目标
- **KV Cache 量化** —— FP8/INT8 KV Cache，在不损失精度的前提下将有效上下文长度翻倍
- **MoE 模型支持** —— 专家并行推理，支持 Mixtral、DeepSeek-V3 等 MoE 架构
- **多模型支持** —— 从 Qwen3 扩展到 Llama、Mistral 等主流模型架构

### 实验性方向
- **Ray 原生集成** —— 基于 Ray 框架实现分布式调度与资源管理，支持异构 GPU 集群的弹性扩缩容。这是最具挑战性的目标——将 Nano-vLLM 的架构简洁性带到分布式集群场景

## 参与贡献

欢迎贡献！代码库刻意保持精简，大部分功能只需要阅读少量文件即可理解和修改。架构细节请参考 `CLAUDE.md`。

## 许可证

MIT
