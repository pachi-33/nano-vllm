# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nano-vLLM is a lightweight, from-scratch implementation of vLLM for LLM inference, written in ~1,200 lines of Python. It supports Qwen3 models with features including prefix caching, tensor parallelism, CUDA graph capture, chunked prefill, and `torch.compile` optimizations. The codebase targets readability and educational value over completeness.

## Build & Run

```bash
# Install (requires Python 3.10–3.13, CUDA GPU)
pip install -e .

# Run example (expects model at ~/huggingface/Qwen3-0.6B/)
python example.py

# Run benchmark
python bench.py

# Run OpenAI-compatible API server
python -m nanovllm.serve --model ~/huggingface/Qwen3-0.6B_fp16/
```

## Testing

The test suite is organized into three tiers under `tests/`:

- `tests/unit/` — Kernel-level tests (no model needed, seconds)
- `tests/integration/` — Engine-level tests with in-process LLMEngine (needs model, minutes)
- `tests/server/` — HTTP API tests with subprocess server (needs model + `httpx`)

```bash
pytest tests/unit/ -v                      # unit tests only
pytest tests/integration/ -v               # integration tests
pytest tests/server/ -v -s                 # server tests
pytest tests/server/ -v -s --pp 2          # server tests with pipeline parallelism
```

## Architecture

The system follows a pipeline: `LLM` → `LLMEngine` → `Scheduler` + `ModelRunner` → `Model` + `Layers`.

### Data flow (one inference step)

1. `LLMEngine.step()` calls `Scheduler.schedule()` which returns a list of `Sequence` objects and an `is_prefill` flag.
2. `ModelRunner.run()` prepares tensors from sequences (via `prepare_prefill` or `prepare_decode`), runs the forward pass, and samples output tokens.
3. `Scheduler.postprocess()` appends generated tokens to sequences, handles EOS/completion, and manages block deallocation.

### Key components

- **`nanovllm/engine/llm_engine.py`** — Entry point. Manages the inference loop: add requests → schedule → run model → postprocess. Handles tensor parallelism by spawning worker processes (rank > 0) that communicate via shared memory.

- **`nanovllm/engine/scheduler.py`** — Implements prefill/decode scheduling with chunked prefill support. Prefill processes prompt tokens (potentially across multiple steps); decode generates one token per sequence per step. Preemption evicts running sequences back to the waiting queue when KV cache blocks are scarce.

- **`nanovllm/engine/sequence.py`** — `Sequence` dataclass tracking token IDs, block table, scheduling state (`num_cached_tokens`, `num_scheduled_tokens`), and completion status. Uses `__getstate__`/`__setstate__` for efficient serialization across processes (only sends token IDs during prefill, just the last token during decode).

- **`nanovllm/engine/block_manager.py`** — Paged KV cache management with prefix caching via xxhash. Blocks are 256 tokens by default. `can_allocate` checks hash-based cache hits; `allocate` reuses cached blocks and allocates new ones; `hash_blocks` writes hashes after tokens are written to cache.

- **`nanovllm/engine/model_runner.py`** — Loads model, allocates KV cache based on available GPU memory, and manages CUDA graph capture for decode. KV cache is a single contiguous tensor `shape=[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`. CUDA graphs are captured for batch sizes [1, 2, 4, 8, 16, 32, ..., 512] and replayed during decode when batch size ≤ 512.

- **`nanovllm/models/qwen3.py`** — Qwen3 model implementation. Uses `packed_modules_mapping` to map HuggingFace weight names (e.g., `q_proj`) to fused parameters (e.g., `qkv_proj`).

- **`nanovllm/layers/`** — CUDA/Triton-optimized building blocks:
  - `attention.py` — Uses flash-attn (`flash_attn_varlen_func` for prefill, `flash_attn_with_kvcache` for decode). Triton kernel `store_kvcache_kernel` writes K/V to paged cache.
  - `sampler.py` — Gumbel-based sampling via `torch.compile` (exponential noise + argmax).
  - `linear.py` — Tensor-parallel linear layers (`ColumnParallelLinear`, `RowParallelLinear`, `QKVParallelLinear`, `MergedColumnParallelLinear`) with weight loader sharding.
  - `embed_head.py` — Vocabulary-parallel embedding and LM head. During decode, LM head only computes logits for rank 0 then gathers.
  - `rotary_embedding.py` — Precomputed RoPE cos/sin cache, applied via `torch.compile`.
  - `layernorm.py` — RMSNorm with fused residual add, `torch.compile`-optimized.
  - `activation.py` — Fused SiLU-and-multiply for SwiGLU MLP.

- **`nanovllm/utils/context.py`** — Thread-local-like global `Context` object that carries scheduling metadata (slot mappings, sequence lengths, block tables) from `ModelRunner` into attention layers. Set before model forward, reset after.

- **`nanovllm/utils/loader.py`** — Loads safetensors weights with support for packed module mapping and TP sharding via `weight_loader` attributes on parameters.

### Global context pattern

The `Context` object (`nanovllm/utils/context.py`) is set globally before each forward pass and read by attention/embed_head layers. This avoids threading these tensors through every layer's forward signature. Always `reset_context()` after use.

### Tensor parallelism

Worker processes (rank > 0) run a blocking `loop()` reading commands from shared memory. Rank 0 broadcasts method calls via `write_shm()`. All processes initialize NCCL process group on `tcp://localhost:2333`. Row-parallel layers use `all_reduce`; the LM head uses `gather` to rank 0.

### CUDA graph

Decode batches ≤ 512 tokens use pre-captured CUDA graphs. Graph variables (input_ids, positions, slot_mapping, context_lens, block_tables) are fixed-size tensors that get partially filled each step. Prefill always runs eagerly.
