# nanovllm/kernels — Custom Triton Operators

This directory contains Triton implementations of the attention operators that replace
the `flash-attn` library dependency. Our hardware does not support flash-attn, so these
kernels are written from scratch using OpenAI Triton.

## Operator Overview

| Operator | File | Replaces | Phase |
|----------|------|----------|-------|
| `decode_attention` | `decode_attention.py` | `flash_attn_with_kvcache` | Decode (1 query token/seq) |
| `prefill_attention` | `prefill_attention.py` | `flash_attn_varlen_func` | Prefill (variable-length, multi-seq) |
| `store_kvcache` | `layers/attention.py` | N/A (already Triton) | Both |

The `store_kvcache` kernel already exists as a Triton kernel in `nanovllm/layers/attention.py`
and is not duplicated here.

## Call Signature Mapping

### decode_attention

```python
# Original flash-attn call (in attention.py):
o = flash_attn_with_kvcache(
    q.unsqueeze(1),        # [batch, 1, num_heads, head_dim]
    k_cache,               # [num_blocks, page_size, num_kv_heads, head_dim]
    v_cache,               # [num_blocks, page_size, num_kv_heads, head_dim]
    cache_seqlens=...,     # [batch] int32
    block_table=...,       # [batch, max_blocks_per_seq] int32
    softmax_scale=...,     # float = 1/sqrt(head_dim)
    causal=True,
)

# Our replacement:
o = decode_attention(
    q,                     # [batch, 1, num_heads, head_dim]
    k_cache,               # [num_blocks, page_size, num_kv_heads, head_dim]
    v_cache,               # same shape
    cache_seqlens,         # [batch] int32
    block_table,           # [batch, max_blocks_per_seq] int32
    scale,                 # float
)                          # returns [batch, 1, num_heads, head_dim]
```

Note: `causal` parameter is dropped because decode always has seqlen_q=1, meaning all
KV positions are from earlier tokens and no masking is needed.

### prefill_attention

```python
# Original flash-attn call (in attention.py):
o = flash_attn_varlen_func(
    q, k, v,
    max_seqlen_q=..., cu_seqlens_q=...,
    max_seqlen_k=..., cu_seqlens_k=...,
    softmax_scale=...,
    causal=True,
    block_table=...,       # None when no prefix cache
)

# Our replacement:
o = prefill_attention(
    q,                     # [total_q, num_heads, head_dim]
    k,                     # [total_k, num_kv_heads, head_dim] or k_cache
    v,                     # same as k
    cu_seqlens_q,          # [num_seqs + 1] int32
    cu_seqlens_k,          # [num_seqs + 1] int32
    max_seqlen_q,          # int
    max_seqlen_k,          # int
    scale,                 # float
    causal=True,           # bool
    block_table=None,      # [num_seqs, max_blocks] int32 or None
)                          # returns [total_q, num_heads, head_dim]
```

## Tensor Shape and Stride Conventions

All tensors use the same conventions as the existing codebase:

| Tensor | Shape | Stride (row-major) | dtype |
|--------|-------|--------------------|-------|
| Q (prefill) | `[total_q, num_heads, head_dim]` | `(num_heads*head_dim, head_dim, 1)` | float16/bfloat16 |
| Q (decode) | `[batch, 1, num_heads, head_dim]` | `(num_heads*head_dim, num_heads*head_dim, head_dim, 1)` | float16/bfloat16 |
| K/V (direct) | `[total_k, num_kv_heads, head_dim]` | `(num_kv_heads*head_dim, head_dim, 1)` | float16/bfloat16 |
| K/V cache | `[num_blocks, page_size, num_kv_heads, head_dim]` | `(page_size*D, D, head_dim, 1)` where D=num_kv_heads*head_dim | float16/bfloat16 |
| block_table | `[batch, max_blocks_per_seq]` | `(max_blocks_per_seq, 1)` | int32 |
| cu_seqlens | `[num_seqs + 1]` | `(1,)` | int32 |
| cache_seqlens | `[batch]` | `(1,)` | int32 |

Key stride invariant (verified in `store_kvcache`): `k_cache.stride(1) == num_kv_heads * head_dim`.
This means within a single physical block, rows are contiguous in the `[num_kv_heads, head_dim]` dimensions.

## Integration with attention.py

To switch from flash-attn to our Triton kernels, change the import in `nanovllm/layers/attention.py`:

```python
# Before:
# from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

# After:
from nanovllm.kernels import prefill_attention, decode_attention
```

Then update the `Attention.forward()` method to call our functions instead. The call sites
are at lines 67-74 of `attention.py`.

## Paged KV Cache Architecture

The KV cache is a single GPU tensor allocated in `ModelRunner.allocate_kv_cache()`:

```python
kv_cache = torch.empty(2, num_layers, num_blocks, page_size, num_kv_heads, head_dim)
```

Each layer gets a view: `k_cache = kv_cache[0, layer_id]` with shape
`[num_blocks, page_size, num_kv_heads, head_dim]`.

**Page size** is 256 tokens (configurable via `Config.kvcache_block_size`).

**Block table** maps logical blocks to physical blocks. For sequence `b`, its KV tokens
at logical positions `[0, 1, ..., cache_seqlens[b]-1]` are stored in physical blocks
`block_table[b, 0], block_table[b, 1], ...`. Within each block, tokens occupy offsets
`0, 1, ..., min(page_size, remaining)`.

**Address resolution:**
```
logical_pos t → physical_block = block_table[b, t // page_size]
                offset = t % page_size
                K[t] = k_cache[physical_block, offset, kv_head, :]
```

## Flash Attention 2 Algorithm Summary

Both kernels implement the Flash Attention 2 algorithm (Tri Dao, 2023):

1. **Tiling**: Split Q into blocks of BLOCK_M rows, K/V into blocks of BLOCK_N rows
2. **Online softmax**: Accumulate softmax statistics (running max m, running sum l)
   incrementally to avoid materializing the full attention matrix
3. **SRAM-only intermediates**: Score matrix S and probability matrix P stay in
   on-chip memory (shared memory in CUDA, SRAM in Triton)
4. **Rescaling**: When a new max is found, rescale previous accumulations by
   `exp(m_old - m_new)` to maintain numerical equivalence with standard softmax

The core rescaling loop per KV block:
```
S = Q_block @ K_block^T * scale
S = apply_mask(S)           # causal or boundary
m_new = max(m_old, rowmax(S))
rescale = exp(m_old - m_new)
l = l * rescale + rowsum(exp(S - m_new))
O = O * rescale + exp(S - m_new) @ V_block
m = m_new
O_final = O / l             # after all KV blocks
```

This is mathematically identical to `softmax(QK^T / sqrt(d)) @ V` but uses O(N) memory
instead of O(N^2).

## References

- Flash Attention 2 paper: https://arxiv.org/abs/2307.08691
- Triton official fused attention: https://github.com/triton-lang/kernels/blob/main/kernels/flash_attention.py
- vLLM Triton attention backend: https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/triton_attn.py
- vLLM Triton deep dive blog: https://vllm.ai/blog/vllm-triton-backend-deep-dive
- Triton tutorial: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
