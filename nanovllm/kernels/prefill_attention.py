"""
Prefill Attention Kernel — Triton replacement for ``flash_attn_varlen_func``.

Used during the prefill phase where multiple variable-length sequences are packed
into contiguous tensors and processed together. Each sequence's query tokens attend
to their full key/value context (which may include prefix-cached tokens stored in
the paged KV cache).

This kernel implements the full Flash Attention 2 algorithm with:
- Variable-length sequence packing (via ``cu_seqlens``)
- Optional paged KV cache access (via ``block_table`` for prefix caching)
- Causal masking
- Grouped Query Attention (GQA)

================================================================================
Two Execution Paths
================================================================================

The kernel handles two distinct scenarios based on whether prefix caching is active:

**Path A: No prefix cache (``block_table is None``)**

In this case, no previously cached KV tokens exist. The K and V tensors are freshly
computed and packed contiguously in memory:

- ``cu_seqlens_q == cu_seqlens_k`` (Q and K have the same length per sequence)
- K/V shape: ``[total_tokens, num_kv_heads, head_dim]``
- K/V are accessed directly by offset (no block_table needed)
- Causal mask: ``K[j] valid for Q[i] iff j <= i`` (standard lower-triangular)

This is the simpler path. It applies when a sequence is being prefilled for the
first time (no cached prefix).

**Path B: Prefix cache active (``block_table is not None``)**

Some KV tokens from a previous step are already cached in the paged KV cache.
The caller replaces K/V with the full ``k_cache``/``v_cache`` tensors:

- ``cu_seqlens_k[-1] > cu_seqlens_q[-1]`` (K is longer: cached prefix + new tokens)
- K/V shape: ``[num_blocks, page_size, num_kv_heads, head_dim]`` (the paged cache)
- K/V are accessed via ``block_table`` (logical-to-physical block mapping)
- Causal mask: ``K[j] valid for Q[i] iff j <= i + (seqlen_k - seqlen_q)``
  The offset ``(seqlen_k - seqlen_q) = num_cached_tokens`` shifts the mask right.

The new K/V tokens have already been written to cache by ``store_kvcache()``
before this function is called (see ``attention.py:63``).

================================================================================
Variable-Length Sequence Packing (Varlen)
================================================================================

Multiple sequences of different lengths are packed into contiguous tensors.
Their boundaries are delimited by cumulative sequence length arrays:

- ``cu_seqlens_q[i]``: start offset of sequence ``i``'s Q tokens in the packed tensor
- ``cu_seqlens_q[i+1]``: end offset (exclusive)
- Similarly for ``cu_seqlens_k``

Example with 3 sequences having seqlen_q = [4, 2, 6]::

    cu_seqlens_q = [0, 4, 6, 12]
    Q tensor:     [s0_t0, s0_t1, s0_t2, s0_t3, s1_t0, s1_t1, s2_t0, ...]

In the kernel, given program index ``(m_block, bidb, bidh)``:
- ``bidb`` identifies the sequence
- ``m_block`` identifies which Q tile within that sequence
- ``bidh`` identifies the query head

Sequence metadata is read from ``cu_seqlens``::

    seqlen_q = cu_seqlens_q[bidb + 1] - cu_seqlens_q[bidb]
    seqlen_k = cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb]
    q_start  = cu_seqlens_q[bidb] + m_block * BLOCK_M

If ``m_block * BLOCK_M >= seqlen_q``, the program returns early (out-of-bounds tile
for a sequence shorter than ``max_seqlen_q``).

================================================================================
Causal Mask — Exact Semantics
================================================================================

The causal mask determines which key positions each query position can attend to.
This is the most subtle part of the prefill kernel because the mask boundary depends
on whether prefix caching is active.

**Fundamental rule:**
    Token at absolute position ``p`` can attend to tokens at positions ``0, 1, ..., p``.

**In terms of local indices within the packed tensors:**
    Q local index ``i`` corresponds to absolute position ``num_cached + i``
    K local index ``j`` corresponds to absolute position ``j``
    (where ``num_cached = seqlen_k - seqlen_q``)

**Causal condition:**
    ``j <= i + (seqlen_k - seqlen_q)``

**Examples:**

No prefix cache (``seqlen_k == seqlen_q``, ``num_cached = 0``):
    Q[i] attends to K[0..i]. Standard lower-triangular mask.
    The (i, j) attention matrix has the pattern::

        1 0 0 0
        1 1 0 0
        1 1 1 0
        1 1 1 1

With prefix cache (``num_cached = 4``, ``seqlen_q = 4``, ``seqlen_k = 8``):
    Q[i] attends to K[0..(i+4)]. The mask shifts right by ``num_cached``.
    The (i, j) attention matrix (4 Q rows × 8 K cols)::

        1 1 1 1 1 0 0 0
        1 1 1 1 1 1 0 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 1

**Block-level optimization:**
Instead of masking individual elements, the kernel limits which KV blocks each Q
block iterates over. For a Q block at ``m_block``, the maximum KV block index is::

    q_end = min((m_block + 1) * BLOCK_M, seqlen_q)
    max_k_valid = q_end + (seqlen_k - seqlen_q) - 1
    n_block_max = ceil((max_k_valid + 1) / BLOCK_N)

Only the last few KV blocks (near the mask boundary) need element-wise masking.
Early KV blocks are fully valid for all rows in the Q block.

**Triton vectorized implementation:**

::

    rows = tl.arange(0, BLOCK_M)[:, None]           # [BLOCK_M, 1]
    cols = tl.arange(0, BLOCK_N)[None, :]           # [1, BLOCK_N]
    q_local = m_block * BLOCK_M + rows              # [BLOCK_M, 1]
    k_local = n_block * BLOCK_N + cols              # [1, BLOCK_N]
    causal_mask = k_local <= q_local + (seqlen_k - seqlen_q)
    # Also mask boundary (last Q/K blocks may be partial):
    q_valid = q_local < seqlen_q
    k_valid = k_local < seqlen_k
    valid = q_valid & k_valid & causal_mask
    S = tl.where(valid, S, float('-inf'))

================================================================================
Flash Attention 2 — Tiled Algorithm
================================================================================

The kernel implements Flash Attention 2 (Tri Dao, ICLR 2024) which avoids
materializing the O(seq_len^2) attention matrix by processing Q and K/V in tiles
and using online softmax accumulation.

**Tiling:**
- Q is split into tiles of ``BLOCK_M`` rows (along the sequence dimension)
- K/V are split into tiles of ``BLOCK_N`` rows

**Per-tile computation:**
For each Q block (indexed by ``m_block``), iterate over all valid KV blocks
(indexed by ``n_block`` from 0 to ``n_block_max - 1``)::

    # Initialize accumulators for this Q block
    m = -inf * ones([BLOCK_M])                # running max per Q row
    l = zeros([BLOCK_M])                      # running sum exp(score - m)
    O = zeros([BLOCK_M, HEAD_DIM])            # running unnormalized output

    for n_block in range(n_block_max):
        # Load K/V tile (from paged cache or direct memory)
        K_tile = load_kv(m_block, n_block)    # [BLOCK_N, HEAD_DIM]
        V_tile = load_kv(m_block, n_block)    # [BLOCK_N, HEAD_DIM]

        # Compute attention scores
        S = Q_tile @ K_tile^T * scale         # [BLOCK_M, BLOCK_N]

        # Apply causal + boundary mask
        S = apply_mask(S, m_block, n_block)

        # Online softmax rescaling
        m_new = rowmax(S)                     # [BLOCK_M]
        rescale = exp(m - m_new)              # [BLOCK_M]
        l = l * rescale                       # [BLOCK_M]
        O = O * rescale[:, None]              # [BLOCK_M, HEAD_DIM]

        P = exp(S - m_new[:, None])           # [BLOCK_M, BLOCK_N]
        l += rowsum(P)                        # [BLOCK_M]
        O += P @ V_tile                       # [BLOCK_M, HEAD_DIM]
        m = m_new                             # [BLOCK_M]

    # Final normalization
    O = O / l[:, None]

**Correctness:** This is mathematically identical to
``softmax(QK^T / sqrt(d)) @ V`` computed over the full matrix. The rescaling
trick maintains exact numerical equivalence (not an approximation).

**Why this saves memory:** The full ``[seqlen_q, seqlen_k]`` attention matrix is
never stored in HBM. Only the ``[BLOCK_M, BLOCK_N]`` score tile exists at any time,
kept in on-chip SRAM. HBM reads/writes are limited to Q, K, V, and O.

================================================================================
Paged KV Cache Access (Path B — Prefix Cache)
================================================================================

When ``block_table is not None``, K/V are read from the paged cache using the same
address resolution as the decode kernel:

::

    logical_pos = n_block * BLOCK_N + local_j
    page = logical_pos // page_size
    offset = logical_pos % page_size
    physical_block = block_table[bidb, page]
    K_row = k_cache[physical_block, offset, kv_head, :]

**BLOCK_N == page_size == 256 (aligned):**
Each KV tile maps to exactly one page. One ``block_table`` lookup per tile.

**BLOCK_N < page_size (e.g., 128):**
A tile may span two pages. Handle by checking if ``(n_block * BLOCK_N) % page_size + BLOCK_N > page_size``.
If so, split the load into two parts from different pages.

**Recommendation:** Use BLOCK_N = page_size = 256 if SRAM allows (simplest).
Fall back to 128 or 64 with cross-page handling if needed.

================================================================================
Direct KV Access (Path A — No Prefix Cache)
================================================================================

When ``block_table is None``, K/V are contiguous in memory::

    k_start = cu_seqlens_k[bidb] + n_block * BLOCK_N
    valid_k = min(BLOCK_N, seqlen_k - n_block * BLOCK_N)
    K_tile = K[k_start : k_start + valid_k, kv_head, :]    # [valid_k, HEAD_DIM]

No block_table lookup needed. Memory access is sequential and coalesced.

================================================================================
GQA (Grouped Query Attention) Support
================================================================================

When ``num_kv_heads < num_heads``, multiple query heads share the same KV head:

::

    kv_head_idx = bidh * num_kv_heads // num_heads

The K/V tile load uses ``kv_head_idx``. The Q tile and output use ``bidh`` (the
full query head index). The matrix multiply ``Q @ K^T`` naturally handles the
dimension mismatch because K has fewer heads than Q — each Q head multiplies against
the same K tile as its group members.

================================================================================
Kernel Grid Configuration
================================================================================

**Standard (non-split-KV) grid::

    grid = (
        ceil(max_seqlen_q / BLOCK_M),     # Q tile dimension
        num_seqs,                          # sequence (batch) dimension
        num_heads,                         # head dimension
    )

Each program: ``(m_block, bidb, bidh)``.

Programs where ``m_block * BLOCK_M >= seqlen_q[bidb]`` return early. This wastes
some thread blocks for short sequences but simplifies the kernel (no binary search
in cu_seqlens needed).

**Split-KV grid (for parallelism on long sequences)::

    grid = (
        ceil(max_seqlen_q / BLOCK_M),
        num_kv_splits,                    # KV dimension split
        num_seqs * num_heads,
    )

Each split processes a subset of KV blocks and produces partial results. A combine
kernel merges them using the log-sum-exp trick. Recommended for prefill sequences
longer than ~1024 tokens.

================================================================================
Triton Implementation Notes
================================================================================

**Compile-time constants:**

- ``HEAD_DIM: tl.constexpr`` — Must be fixed at compile time. Values: 64, 96, 128.
  (Qwen3-0.6B: head_dim=128.)
- ``BLOCK_M: tl.constexpr`` — Q tile size. Recommended: 64 or 128.
- ``BLOCK_N: tl.constexpr`` — KV tile size. Recommended: 256 (= page_size) for
  simplicity, or 128/64 for SRAM savings.

**SRAM budget estimation (float16, HEAD_DIM=128)::

    Q tile:    BLOCK_M * HEAD_DIM * 2 = 64 * 128 * 2 = 16 KB
    K tile:    BLOCK_N * HEAD_DIM * 2 = 256 * 128 * 2 = 64 KB
    V tile:    BLOCK_N * HEAD_DIM * 2 = 256 * 128 * 2 = 64 KB
    Scores:    BLOCK_M * BLOCK_N * 2  = 64 * 256 * 2  = 32 KB
    Accum:     BLOCK_M * HEAD_DIM * 4 = 64 * 128 * 4  = 32 KB (float32)
    m, l:      BLOCK_M * 4 * 2        = 64 * 4 * 2    = 0.5 KB
    Total ≈ 209 KB

    This exceeds most GPUs' shared memory (48-164 KB/SM). Reduce BLOCK_N to 128:

    K+V tiles: 128 * 128 * 2 * 2 = 64 KB
    Scores:    64 * 128 * 2 = 16 KB
    Total ≈ 128 KB (still tight, but fits on A100/H100 with 164 KB)

    Or BLOCK_N=64: Total ≈ 80 KB (fits comfortably).

**Autotuning:**
Use ``triton.autotune`` or ``triton.Config`` to search over (BLOCK_M, BLOCK_N)
combinations: e.g., [(64, 64), (64, 128), (128, 64), (128, 128)].

**Memory access:**
- Q tile is loaded once per (m_block, bidb, bidh) and reused across all n_blocks.
- K/V tiles are loaded once per n_block.
- block_table reads: one int32 per KV tile (when paged).
- cu_seqlens reads: 4 int32 values per program (cached in registers after first load).

================================================================================
Boundary Conditions
================================================================================

1. **Partial Q blocks:**
   The last Q block may have fewer than BLOCK_M rows.
   ``valid_q = min(BLOCK_M, seqlen_q - m_block * BLOCK_M)``
   Only load/process ``valid_q`` rows. Mask the rest in scores.

2. **Partial KV blocks:**
   The last KV block may have fewer than BLOCK_N rows.
   ``valid_k = min(BLOCK_N, seqlen_k - n_block * BLOCK_N)``
   Only load ``valid_k`` rows. Mask invalid positions to -inf in scores.

3. **block_table padding (-1):**
   Shorter sequences have -1 in unused block_table entries. Not accessed because
   KV iteration is bounded by ``ceil(seqlen_k / page_size)``.

4. **Zero-length sequences:**
   If ``seqlen_q == 0``, the program returns immediately (no work to do).

5. **Single-token sequences:**
   seqlen_q == seqlen_k == 1 with no prefix cache. Should work correctly with
   BLOCK_M >= 1 and BLOCK_N >= 1.

================================================================================
Numerical Precision
================================================================================

- Inputs: float16 or bfloat16 (matching model dtype).
- Score computation and softmax: float32 for numerical stability.
  The ``exp()`` and ``max()`` operations are especially sensitive to precision.
- Output: stored in the input dtype (float16/bfloat16).
- Accumulator ``O`` must be float32 to prevent drift during the rescaling loop.

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

**3. Store-then-load workaround for Triton compiler bug:**

Some Triton versions produce wrong results when rescaling accumulators in-place.
Fix: store to a scratch buffer and immediately load back.

**4. num_warps and num_stages recommendations:**

    head_dim <= 64:  num_warps=4, num_stages=4
    head_dim <= 128: num_warps=8, num_stages=4

**5. ``tl.make_block_ptr`` vs manual pointer arithmetic:**

``tl.make_block_ptr`` handles boundary checks and memory coalescing automatically.
However, for complex addressing (paged KV cache, varlen cu_seqlens), manual pointer
arithmetic is still needed. Pattern from Dao-AILab::

    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + \
             (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)

**6. Varlen indexing pattern (from SageAttention):**

Clean pattern for varlen with cu_seqlens::

    grid = (cdiv(max_seqlen_q, BLOCK_M), num_heads, num_seqs)

    off_z = tl.program_id(2)   # sequence index
    off_h = tl.program_id(1)   # head index
    off_m = tl.program_id(0)   # Q block index

    cu_q_start = tl.load(cu_seqlens_q + off_z)
    cu_q_end = tl.load(cu_seqlens_q + off_z + 1)
    seqlen_q = cu_q_end - cu_q_start
    if off_m * BLOCK_M >= seqlen_q:
        return  # out of bounds

**7. Causal mask block-level optimization:**

Skip entire KV blocks that are fully masked. For Q block at ``m_block``:
if ``m_block * BLOCK_M + BLOCK_M <= n_block * BLOCK_N``, all positions in this
(Q_block, K_block) pair are above the diagonal → skip entirely. Only the boundary
block needs element-wise masking. This gives ~1.7-1.8x speedup.

**8. ``triton.heuristics`` for compile-time constants:**

Use ``triton.heuristics`` to set ``EVEN_M``, ``EVEN_N``, ``EVEN_HEADDIM`` flags
at kernel launch time. This allows the compiler to generate optimized paths for
the common case (no boundary handling) while still supporting edge cases.

================================================================================
References
================================================================================

- FlashAttention-2 paper: https://arxiv.org/abs/2307.08691
- Triton official kernels: https://github.com/triton-lang/kernels/blob/main/kernels/flash_attention.py
- Dao-AILab flash-attn Triton: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py
- vLLM Triton attention: https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/triton_attn.py
- vLLM Triton deep dive: https://vllm.ai/blog/vllm-triton-backend-deep-dive
- SageAttention varlen (clean cu_seqlens pattern): https://github.com/thu-ml/SageAttention/blob/main/sageattention/triton/attn_qk_int8_block_varlen.py
- Online softmax explained: https://www.isztld.com/posts/online-softmax.html

================================================================================
Integration with attention.py
================================================================================

Current call site in ``nanovllm/layers/attention.py``::

    if context.is_prefill:
        if context.block_tables is not None:    # prefix cache
            k, v = k_cache, v_cache
        o = flash_attn_varlen_func(q, k, v,
                                   max_seqlen_q=..., cu_seqlens_q=...,
                                   max_seqlen_k=..., cu_seqlens_k=...,
                                   softmax_scale=..., causal=True,
                                   block_table=...)

To switch to our kernel::

        o = prefill_attention(q, k, v,
                              cu_seqlens_q, cu_seqlens_k,
                              max_seqlen_q, max_seqlen_k,
                              scale=...,
                              causal=True,
                              block_table=...)

The signatures are intentionally similar to minimize the integration change.
"""

import torch
from torch import Tensor


def prefill_attention(
    q: Tensor,              # [total_q, num_heads, head_dim]
    k: Tensor,              # [total_k, num_kv_heads, head_dim] or k_cache
    v: Tensor,              # same as k
    cu_seqlens_q: Tensor,   # [num_seqs + 1] int32
    cu_seqlens_k: Tensor,   # [num_seqs + 1] int32
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float,
    causal: bool = True,
    block_table: Tensor | None = None,  # [num_seqs, max_blocks_per_seq] int32
) -> Tensor:                # [total_q, num_heads, head_dim]
    """
    Prefill-phase paged attention (replacement for flash_attn_varlen_func).

    Multiple variable-length sequences are packed into contiguous Q/K/V tensors.
    Supports optional paged KV cache access for prefix caching via ``block_table``.

    Args:
        q: Packed query tensor. Shape ``[total_q, num_heads, head_dim]``.
        k: Packed key tensor or paged key cache. Shape ``[total_k, num_kv_heads, head_dim]``
            (direct) or ``[num_blocks, page_size, num_kv_heads, head_dim]`` (paged).
        v: Same as ``k``.
        cu_seqlens_q: Cumulative sequence lengths for Q. Shape ``[num_seqs + 1]``, dtype int32.
            ``cu_seqlens_q[i]`` and ``cu_seqlens_q[i+1]`` delimit sequence ``i``'s Q tokens.
        cu_seqlens_k: Cumulative sequence lengths for K. Shape ``[num_seqs + 1]``, dtype int32.
            When prefix cache is active, ``cu_seqlens_k[-1] > cu_seqlens_q[-1]``.
        max_seqlen_q: Maximum Q sequence length (used for kernel launch configuration).
        max_seqlen_k: Maximum K sequence length.
        scale: Softmax scale factor, typically ``1.0 / sqrt(head_dim)``.
        causal: Whether to apply causal masking. Default True.
        block_table: Optional logical-to-physical block mapping for paged KV cache.
            Shape ``[num_seqs, max_blocks_per_seq]``, dtype int32.
            When provided, ``k`` and ``v`` are treated as paged cache tensors.
            When None, ``k`` and ``v`` are accessed directly as contiguous tensors.

    Returns:
        Attention output tensor. Shape ``[total_q, num_heads, head_dim]``.

    Note:
        This function currently raises NotImplementedError. The Triton kernel
        implementation will be added in a future commit.
    """
    raise NotImplementedError(
        "prefill_attention Triton kernel not yet implemented. "
        "See module docstring for full technical specification."
    )
