"""
Decode Attention Kernel — Triton replacement for ``flash_attn_with_kvcache``.

Used during the decode (autoregressive generation) phase where each sequence
contributes exactly **1 query token** and attends over its entire cached KV history.

This kernel reads K/V from a paged KV cache via ``block_table``, computes scaled
dot-product attention using the online softmax algorithm, and returns the attention
output — all without materializing the full attention matrix.

================================================================================
Algorithm Overview
================================================================================

Each program instance handles one ``(batch_idx, query_head)`` pair independently.

Since seqlen_q = 1 (only the latest token), there is **no causal masking** needed:
the single query token is at the most recent position, so all cached KV tokens are
valid (they all come from earlier positions). The ``causal=True`` flag from the
flash-attn API is effectively a no-op here.

The computation is a simple single-row online softmax over all KV tokens:

    S = q @ K_all^T * scale           # [num_kv_tokens] attention scores
    weights = softmax(S)              # [num_kv_tokens]
    o = weights @ V_all               # [head_dim]

But to avoid reading all KV tokens at once (which may exceed SRAM), we process
KV in tiles (blocks) and accumulate the softmax incrementally.

================================================================================
Paged KV Cache Access
================================================================================

The KV cache is a paged tensor with shape ``[num_blocks, page_size, num_kv_heads, head_dim]``.
Each sequence's KV tokens are scattered across physical blocks. The ``block_table``
maps logical block indices to physical block IDs.

**Address resolution** for logical position ``t`` of sequence ``b``::

    logical_block  = t // page_size              # which logical block
    offset_in_block = t % page_size              # position within the block
    physical_block = block_table[b, logical_block]
    K[t] = k_cache[physical_block, offset_in_block, kv_head, :]

In Triton, when processing a KV tile of ``BLOCK_N`` tokens starting at logical
position ``n_block * BLOCK_N``:

**Case 1: BLOCK_N == page_size == 256 (recommended)**
    Each tile maps to exactly one page. Only one ``block_table`` lookup needed::

        page_idx = n_block        # tile index == page index (they're aligned)
        physical_block = block_table[batch_idx, page_idx]
        k_tile = k_cache[physical_block, :, kv_head, :]     # [256, head_dim]
        v_tile = v_cache[physical_block, :, kv_head, :]     # [256, head_dim]

**Case 2: BLOCK_N < page_size (e.g., 64 or 128)**
    A single tile may span two pages. Must handle cross-page boundaries::

        for local_j in range(BLOCK_N):
            logical_pos = n_block * BLOCK_N + local_j
            page = logical_pos // page_size
            offset = logical_pos % page_size
            phys_block = block_table[batch_idx, page]
            # load k_cache[phys_block, offset, kv_head, :]

    In Triton this can be vectorized by computing offsets for two potential pages,
    checking if the tile crosses a boundary, and doing conditional loads.

**Recommendation:** Start with BLOCK_N = page_size = 256 for simplicity. If SRAM
is insufficient, fall back to BLOCK_N = 128 with cross-page handling.

================================================================================
Online Softmax Algorithm (Decode Variant — Single Row)
================================================================================

Since there is only 1 query row, the online softmax is simplified (no row dimension)::

    m = -inf                           # running max of attention scores
    l = 0.0                            # running sum of exp(score - m)
    acc = zeros([head_dim])            # running (unnormalized) output

    for n_block in range(num_kv_blocks):
        # Load K/V tile from paged cache
        K_tile = load_kv_block(...)    # [BLOCK_N, head_dim]
        V_tile = load_kv_block(...)    # [BLOCK_N, head_dim]

        # Compute attention scores for this tile
        S = q @ K_tile^T * scale       # [BLOCK_N]

        # Mask invalid positions (last block may be partial, block_table padding)
        S = mask_boundary(S, ...)

        # Online softmax update
        m_new = max(m, max(S))
        rescale = exp(m - m_new)
        l = l * rescale + sum(exp(S - m_new))
        acc = acc * rescale + exp(S - m_new) @ V_tile    # [head_dim]
        m = m_new

    O = acc / l                        # normalize

This is mathematically identical to computing the full softmax at once, but processes
KV in tiles to keep memory usage bounded.

================================================================================
GQA (Grouped Query Attention) Support
================================================================================

When ``num_kv_heads < num_heads``, multiple query heads share the same KV head.

**Mapping:**
    kv_head_idx = query_head_idx * num_kv_heads // num_heads

Example (Qwen3-0.6B): num_heads=16, num_kv_heads=8, so every 2 query heads share
1 KV head. Query head 0 and 1 both read from KV head 0, query heads 2 and 3 from
KV head 1, etc.

In the kernel, this is a simple integer division. The K/V tile load uses ``kv_head_idx``
instead of ``query_head_idx``.

================================================================================
Kernel Grid Configuration
================================================================================

::

    grid = (batch_size, num_heads)

Each program handles one ``(batch_idx, head_idx)`` pair and iterates over all KV
blocks for that sequence.

**Alternative — Split-KV for long sequences:**
For very long KV sequences (e.g., 4096+ tokens), a single program iterating over
all blocks may not provide enough parallelism. The split-KV variant partitions the
KV dimension across multiple programs and combines results:

    grid = (batch_size, num_heads, num_kv_splits)

Each split produces partial ``(O_partial, log_sum_exp_partial)``. A separate combine
kernel merges them::

    global_max = max(lse[split]) over all splits
    O_final = sum(exp(lse[split] - global_max) * O_partial[split]) / sum(exp(...))

Start without split-KV; add it later if decode throughput is bottlenecked.

================================================================================
Triton Implementation Notes
================================================================================

**Compile-time constants (tl.constexpr):**
    - ``HEAD_DIM``: Must be a compile-time constant for Triton to unroll inner loops.
      Typical values: 64, 96, 128. (Qwen3-0.6B uses head_dim=128.)
    - ``BLOCK_N``: KV tile size. Recommended: 256 (= page_size) for simplest logic.
      Can be 64 or 128 if SRAM budget is tight, but requires cross-page handling.

**SRAM budget estimation:**
    - Q vector:  ``HEAD_DIM * sizeof(dtype)``                       ≈ 256 B
    - K tile:    ``BLOCK_N * HEAD_DIM * sizeof(dtype)``              ≈ 64 KB
    - V tile:    ``BLOCK_N * HEAD_DIM * sizeof(dtype)``              ≈ 64 KB
    - Scores:    ``BLOCK_N * sizeof(dtype)``                         ≈ 512 B
    - Accumulator: ``HEAD_DIM * sizeof(float32)``                    ≈ 512 B
    - Total ≈ 130 KB for BLOCK_N=256, HEAD_DIM=128, float16

    Most modern GPUs have 48-164 KB shared memory per SM. 130 KB is tight.
    If it doesn't fit, reduce BLOCK_N to 128 (SRAM ≈ 65 KB + overhead).

**Memory access patterns:**
    - Q is loaded once per program (small, fits in registers).
    - K/V tiles are loaded sequentially from global memory via block_table.
    - block_table itself is read from global memory (one int32 per tile).
    - Output O is written once at the end.

**Autotuning:**
    Consider using ``triton.autotune`` to search over BLOCK_N values [64, 128, 256]
    and pick the fastest for the target hardware.

================================================================================
Boundary Conditions
================================================================================

1. **Last KV block may be partially filled:**
   ``valid_k = min(BLOCK_N, cache_seqlens[batch_idx] - n_block * BLOCK_N)``
   Load only ``valid_k`` rows; mask the rest in the score computation.

2. **block_table padding with -1:**
   Sequences shorter than ``max_blocks_per_seq`` have -1 in unused block_table entries.
   These should never be accessed because ``num_kv_blocks`` is bounded by
   ``ceil(cache_seqlens / page_size)``.

3. **Empty sequences (cache_seqlens == 0):**
   Should produce a zero output. The kernel can check this early and return.

4. **block_table dtype is int32:**
   Physical block IDs are non-negative. The -1 sentinel for padding must not
   be used as an index.

================================================================================
Numerical Precision
================================================================================

- Inputs (Q, K_cache, V_cache) are float16 or bfloat16 (matching model dtype).
- Attention scores and softmax should be computed in float32 for numerical stability,
  especially the ``exp()`` and ``max()`` operations.
- Output is stored in the same dtype as input (float16/bfloat16).

Reference: ``torch.nn.functional.scaled_dot_product_attention`` uses float32 internally
for the same reason.

================================================================================
Triton Implementation Gotchas (from Community Experience)
================================================================================

**1. Use ``tl.math.exp2`` instead of ``tl.exp`` for performance:**

Pre-multiply the softmax scale by ``log2(e) = 1.44269504``::

    qk_scale = sm_scale * 1.44269504   # instead of sm_scale
    # Then use tl.math.exp2(x) instead of tl.exp(x)

This allows the compiler to fuse into FMA instructions. Used by triton-lang/kernels
and Dao-AILab's flash-attn Triton backend.

**2. Head dimension padding:**

Always pad ``BLOCK_HEADDIM`` to a power of 2 (minimum 16) using
``triton.next_power_of_2(d)``. This enables optimized tensor core operations.
For head_dim=128: ``BLOCK_HEADDIM = 128`` (already a power of 2).
For head_dim=96: ``BLOCK_HEADDIM = 128`` (padded).

**3. Store-then-load workaround for Triton compiler bug:**

Some Triton versions produce wrong results when rescaling accumulators in-place.
The fix is to store to a scratch buffer and immediately load back::

    tl.store(scratch_ptr, acc_rescale)
    acc_rescale = tl.load(scratch_ptr)

Known to affect accumulator rescaling in the online softmax loop.

**4. num_warps and num_stages recommendations:**

    head_dim <= 64:  num_warps=4, num_stages=4
    head_dim <= 128: num_warps=8, num_stages=4

Higher ``num_stages`` enables software pipelining (overlapping loads and compute)
but uses more shared memory. Reduce if SRAM budget is tight.

**5. Memory coalescing hint:**

Use ``tl.multiple_of(ptr, alignment_value)`` to hint that a pointer is aligned,
which helps the compiler generate coalesced memory accesses.

**6. K/V cache layout for paged attention (from vLLM):**

vLLM uses a 5D layout for K cache and 4D for V cache to optimize vectorized loads:

    K cache: [num_blocks, num_kv_heads, head_size // x, block_size, x]
    V cache: [num_blocks, num_kv_heads, head_size, block_size]

where ``x`` is a vectorization factor (typically 4 or 8). This differs from
nano-vLLM's layout of ``[num_blocks, page_size, num_kv_heads, head_dim]``.
Our kernel must match nano-vLLM's existing layout.

================================================================================
References
================================================================================

- FlashAttention-2 paper: https://arxiv.org/abs/2307.08691
- Triton official kernels: https://github.com/triton-lang/kernels/blob/main/kernels/flash_attention.py
- Dao-AILab flash-attn Triton: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py
- vLLM paged decode kernel: https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/chunked_prefill_paged_decode.py
- vLLM Triton deep dive: https://vllm.ai/blog/vllm-triton-backend-deep-dive
- SageAttention varlen (clean cu_seqlens pattern): https://github.com/thu-ml/SageAttention/blob/main/sageattention/triton/attn_qk_int8_block_varlen.py
"""

import torch
from torch import Tensor
import triton
import triton.language as tl


@triton.jit
def _decode_attention_kernel(
    Q_ptr, q_stride_b, q_stride_s, q_stride_h, q_stride_d,
    K_cache_ptr, k_stride_block, k_stride_page, k_stride_head, k_stride_dim,
    V_cache_ptr, v_stride_block, v_stride_page, v_stride_head, v_stride_dim,
    O_ptr, o_stride_b, o_stride_s, o_stride_h, o_stride_d,
    Block_table_ptr, bt_stride_b, bt_stride_s,
    Cache_seqlens_ptr,
    scale,
    num_kv_heads,
    page_size,
    MAX_NUM_BLOCKS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    kv_head = head_idx * num_kv_heads // tl.num_programs(1)

    seqlen = tl.load(Cache_seqlens_ptr + batch_idx)
    if seqlen == 0:
        return

    # Load query: q[batch_idx, 0, head_idx, :] → shape [HEAD_DIM]
    q_offsets = head_idx * q_stride_h + tl.arange(0, HEAD_DIM)
    q = tl.load(Q_ptr + batch_idx * q_stride_b + q_offsets)

    # Online softmax accumulators (m_val/l_val are scalars, seqlen_q=1)
    m_val = float('-inf')
    l_val = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    num_kv_blocks = (seqlen + BLOCK_N - 1) // BLOCK_N

    for n_block in range(MAX_NUM_BLOCKS):
        if n_block < num_kv_blocks:
            # Physical block from block_table
            physical_block = tl.load(Block_table_ptr + batch_idx * bt_stride_b + n_block * bt_stride_s)

            # Number of valid tokens in this block
            valid_tokens = tl.minimum(BLOCK_N, seqlen - n_block * BLOCK_N)
            offsets_in_block = tl.arange(0, BLOCK_N)
            mask_k = offsets_in_block < valid_tokens

            # Load K tile: k_cache[physical_block, 0:BLOCK_N, kv_head, :]
            k_base = physical_block * k_stride_block + kv_head * k_stride_head
            k_offsets = offsets_in_block[:, None] * k_stride_page + tl.arange(0, HEAD_DIM)[None, :] * k_stride_dim
            k_tile = tl.load(K_cache_ptr + k_base + k_offsets, mask=mask_k[:, None], other=0.0)

            # Compute scores: q @ k_tile^T → [BLOCK_N]
            scores = tl.sum(q[None, :] * k_tile, axis=1) * scale  # [BLOCK_N]

            # Mask invalid positions to -inf
            scores = tl.where(mask_k, scores, float('-inf'))

            # Online softmax update (m_new = max(m_prev, max(S)))
            m_new = tl.max(scores)
            m_new = tl.maximum(m_val, m_new)
            rescale = tl.math.exp(m_val - m_new)
            l_val = l_val * rescale + tl.sum(tl.math.exp(scores - m_new))
            acc = acc * rescale

            # Load V tile: v_cache[physical_block, 0:BLOCK_N, kv_head, :]
            v_base = physical_block * v_stride_block + kv_head * v_stride_head
            v_offsets = offsets_in_block[:, None] * v_stride_page + tl.arange(0, HEAD_DIM)[None, :] * v_stride_dim
            v_tile = tl.load(V_cache_ptr + v_base + v_offsets, mask=mask_k[:, None], other=0.0)

            # Weighted sum: exp(score - m_new) @ V_tile → [HEAD_DIM]
            p = tl.math.exp(scores - m_new)  # [BLOCK_N]
            acc += tl.sum(p[:, None] * v_tile, axis=0)

            m_val = m_new

    # Normalize
    out = acc / l_val  # [HEAD_DIM]

    # Store output
    o_offsets = head_idx * o_stride_h + tl.arange(0, HEAD_DIM)
    tl.store(O_ptr + batch_idx * o_stride_b + o_offsets, out)


def decode_attention(
    q: Tensor,              # [batch, 1, num_heads, head_dim]
    k_cache: Tensor,        # [num_blocks, page_size, num_kv_heads, head_dim]
    v_cache: Tensor,        # [num_blocks, page_size, num_kv_heads, head_dim]
    cache_seqlens: Tensor,  # [batch] int32
    block_table: Tensor,    # [batch, max_blocks_per_seq] int32
    scale: float,           # 1/sqrt(head_dim)
) -> Tensor:                # [batch, 1, num_heads, head_dim]
    batch, _, num_heads, head_dim = q.shape
    num_blocks, page_size, num_kv_heads, _ = k_cache.shape
    max_num_blocks = block_table.shape[1]

    o = torch.empty_like(q)

    grid = (batch, num_heads)
    _decode_attention_kernel[grid](
        q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_cache, k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache, v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        o, o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        block_table, block_table.stride(0), block_table.stride(1),
        cache_seqlens,
        scale,
        num_kv_heads,
        page_size,
        MAX_NUM_BLOCKS=max_num_blocks,
        HEAD_DIM=triton.next_power_of_2(head_dim),
        BLOCK_N=page_size,
    )
    return o
