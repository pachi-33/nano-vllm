# TODO

nano-vllm 已知问题与改进计划，按模块和优先级分类。

---

## Pipeline Parallelism (PP)

### P0: PP 模式下 CUDA Graph 不可用

- **现状**: 所有 PP 测试均使用 `enforce_eager=True`。`capture_cudagraph` 中 stage 0 捕获图时调用 `self.model(input_ids, positions)`，但 `PipelineStageModel.forward()` 的输入方式与 `Qwen3Model` 不同。Stage 1 的 `hidden_states`/`residual` 来自 NCCL recv，CUDA graph 无法捕获动态通信。
- **影响**: Decode 每步都有 Python 开销，无法发挥 CUDA graph 的性能优势。
- **涉及文件**: `nanovllm/engine/model_runner.py` (capture_cudagraph), `nanovllm/models/pipeline_stage.py`
- **方向**: PP 模式下每个 stage 独立捕获 CUDA graph（只捕获自身 stage 的 forward），通信在 graph 外用 stream 做异步 send/recv。

### P0: Rank 1 首次请求 Triton JIT 编译延迟

- **现状**: `_warmup_pp()` 中 rank 1 跳过了模型 forward（异构 GPU 上 Triton 编译会 OOM），导致第一次真实请求时 rank 1 上所有 Triton kernel 为冷启动。
- **影响**: 首次 TTFT 从 ~42ms（单卡）飙到 ~448ms，约 10 倍退化。
- **涉及文件**: `nanovllm/engine/model_runner.py` (_warmup_pp)
- **方向**: 在 `_warmup_pp` 中为 rank 1 做一次真实的 model forward（减小 seq_len/batch_size），确保 Triton kernel 提前编译。

### P1: NCCL send/recv 硬编码 dst/src

- **现状**: `_run_model_pp` 和 `_run_pp` 中 `dist.send(hidden, dst=1)` / `dist.recv(buf, src=0)` 写死，仅支持 2 级 PP。
- **影响**: 无法扩展到 PP=3 或 PP=4。
- **涉及文件**: `nanovllm/engine/model_runner.py` (_run_model_pp, _run_pp)
- **方向**: 根据 `pp_rank` 和 `pp_size` 动态计算 dst/src，支持多级拓扑。

### P1: 共享内存泄漏和僵尸进程

- **现状**: 崩溃或中断后 `/dev/shm/nanovllm_pp` 和 GPU 进程经常残留。`atexit` handler 不能覆盖所有退出路径（SIGKILL、OOM 等），`torch.cuda.empty_cache()` 也回收不了僵尸进程占用的显存。
- **影响**: 多次运行后显存耗尽导致 OOM，需要手动 `kill` 和 `rm /dev/shm/nanovllm_pp`。
- **涉及文件**: `nanovllm/engine/model_runner.py`, `nanovllm/engine/llm_engine.py`
- **方向**: 增加 signal handler (SIGTERM/SIGINT)；启动时检测并清理残留 shm；在 `ModelRunner.__init__` 中加 `shm.unlink` 的 try/catch 兜底。

### P2: 无 TP+PP 混合模式

- **现状**: TP subgroup 创建逻辑假设简单的 `pp * tp_size` rank 映射，实际上 TP+PP 混合模式需要更复杂的进程组拓扑。
- **影响**: 无法在多卡环境下同时使用 TP 和 PP。
- **涉及文件**: `nanovllm/engine/model_runner.py`, `nanovllm/utils/distributed.py`

---

## Benchmark (bench_e2e.py)

### P1: 内存统计不准确

- **现状**: `model_weights_gb` 用 `peak * 0.3` 粗略估算，无法区分模型权重、KV cache、激活内存各自占比。
- **影响**: memory 模式的数据对调优没有参考价值。
- **涉及文件**: `bench_e2e.py` (run_memory)
- **方向**: 在 ModelRunner 的关键节点（load_model 后、allocate_kv_cache 后、warmup 后）分别记录 `torch.cuda.memory_stats()`，将分段数据返回给 bench 脚本。

### P2: PPL 测试在 PP 模式下跳过

- **现状**: `run_perplexity()` 用 HF 的 `AutoModelForCausalLM` 做 forward，PP 模式下直接跳过。
- **影响**: PP 模式下无法测量模型质量。
- **涉及文件**: `bench_e2e.py` (run_perplexity)
- **方向**: 可用 nano-vllm 引擎做近似 PPL（逐 token 计算交叉熵），或对每个 stage 单独加载计算。

---

## 核心引擎

### P1: 2GB 硬编码 Triton 预留内存

- **现状**: `allocate_kv_cache()` 中 `reserved = 2 * 1024**3` 是暴力 workaround。
- **影响**: 在显存较小的卡上（如 V100 16GB）浪费大量 KV cache 空间，降低并发能力。
- **涉及文件**: `nanovllm/engine/model_runner.py` (allocate_kv_cache)
- **方向**: 根据实际 GPU 显存动态计算预留量，或在第一次 Triton 编译后测量实际占用再分配 KV cache。

### P2: NCCL P2P 性能警告

- **现状**: 未批量化的 `dist.send`/`dist.recv` 触发 NCCL lazy initialization 警告。
- **影响**: 不影响正确性，但日志有噪音，且小包 P2P 通信效率低。
- **方向**: 考虑使用 `group` 操作或 batched send/recv 减少 NCCL 调用次数。
