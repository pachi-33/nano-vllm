"""
Shared test fixtures and reference implementations for nano-vLLM kernel tests.

This module provides:
- Pytest fixtures for GPU detection, model configs, and tensor factories
- Naive PyTorch attention implementations used as correctness baselines
"""

import pytest
import torch
from torch import Tensor
from typing import Optional


# ---------------------------------------------------------------------------
# GPU availability
# ---------------------------------------------------------------------------

def cuda_available():
    return torch.cuda.is_available()


@pytest.fixture
def device():
    """Return 'cuda' if available, otherwise skip the test."""
    if not cuda_available():
        pytest.skip("CUDA not available")
    return "cuda"


# ---------------------------------------------------------------------------
# Model configuration fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config():
    """Typical model configuration parameters (Qwen3-0.6B-like)."""
    return {
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 128,
        "page_size": 256,
        "num_blocks": 1024,
        "dtype": torch.float16,
    }


@pytest.fixture
def small_config():
    """Small configuration for fast unit tests."""
    return {
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 64,
        "page_size": 256,
        "num_blocks": 128,
        "dtype": torch.float32,
    }


# ---------------------------------------------------------------------------
# KV cache factory
# ---------------------------------------------------------------------------

@pytest.fixture
def make_kv_cache(device):
    """Factory fixture that creates a paged KV cache tensor.

    Usage::

        k_cache, v_cache = make_kv_cache(num_blocks=512, page_size=256,
                                          num_kv_heads=8, head_dim=128,
                                          dtype=torch.float16)
    """
    def _make(num_blocks=1024, page_size=256, num_kv_heads=8, head_dim=128,
              dtype=torch.float16):
        k_cache = torch.zeros(num_blocks, page_size, num_kv_heads, head_dim,
                              dtype=dtype, device=device)
        v_cache = torch.zeros(num_blocks, page_size, num_kv_heads, head_dim,
                              dtype=dtype, device=device)
        return k_cache, v_cache
    return _make


# ---------------------------------------------------------------------------
# Block table factory
# ---------------------------------------------------------------------------

@pytest.fixture
def make_block_table(device):
    """Factory fixture that creates a block_table tensor.

    Given a list of block_tables (one per sequence, each a list of physical
    block IDs), produces a padded int32 tensor with -1 for unused entries.

    Usage::

        bt = make_block_table([[3, 7, 12], [5, 8]])
        # Shape: [2, 3], dtype int32
        # [[3, 7, 12], [5, 8, -1]]
    """
    def _make(tables: list[list[int]]):
        max_blocks = max(len(t) for t in tables)
        padded = [t + [-1] * (max_blocks - len(t)) for t in tables]
        return torch.tensor(padded, dtype=torch.int32, device=device)
    return _make


# ---------------------------------------------------------------------------
# Slot mapping factory
# ---------------------------------------------------------------------------

@pytest.fixture
def make_slot_mapping(device):
    """Factory fixture that creates a slot_mapping tensor from block_tables
    and token ranges.

    For each sequence with block_table [b0, b1, ...] and token range [start, end),
    computes slot = block_table[start // page_size] * page_size + start % page_size
    for each token position.
    """
    def _make(block_tables: list[list[int]], token_ranges: list[tuple[int, int]],
              page_size=256):
        slot_mapping = []
        for bt, (start, end) in zip(block_tables, token_ranges):
            for pos in range(start, end):
                logical_block = pos // page_size
                offset = pos % page_size
                slot_mapping.append(bt[logical_block] * page_size + offset)
        return torch.tensor(slot_mapping, dtype=torch.int32, device=device)
    return _make


# ---------------------------------------------------------------------------
# Naive reference attention implementations
# ---------------------------------------------------------------------------

def naive_sdpa(
    q: Tensor,       # [seq_q, num_heads, head_dim]
    k: Tensor,       # [seq_k, num_kv_heads, head_dim]
    v: Tensor,       # [seq_k, num_kv_heads, head_dim]
    scale: float,
    causal: bool = False,
) -> Tensor:         # [seq_q, num_heads, head_dim]
    """Standard scaled dot-product attention (reference implementation).

    Supports GQA via automatic broadcasting: when num_kv_heads < num_heads,
    K and V are expanded to match num_heads by repeating each KV head.

    This is the correctness baseline for Triton kernel tests.
    """
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]

    # GQA: expand K/V to match num_heads
    if num_kv_heads < num_heads:
        heads_per_kv = num_heads // num_kv_heads
        k = k.repeat_interleave(heads_per_kv, dim=1)  # [seq_k, num_heads, head_dim]
        v = v.repeat_interleave(heads_per_kv, dim=1)

    # Compute attention scores: [num_heads, seq_q, seq_k]
    # q: [seq_q, num_heads, head_dim] -> transpose to [num_heads, seq_q, head_dim]
    q_t = q.permute(1, 0, 2)  # [num_heads, seq_q, head_dim]
    k_t = k.permute(1, 0, 2)  # [num_heads, seq_k, head_dim]
    v_t = v.permute(1, 0, 2)  # [num_heads, seq_k, head_dim]

    scores = torch.bmm(q_t, k_t.transpose(1, 2)) * scale  # [num_heads, seq_q, seq_k]

    if causal:
        seq_q_len = q.shape[0]
        seq_k_len = k.shape[0]
        # causal mask: position i can attend to position j if j <= i + (seq_k - seq_q)
        # For standard (no prefix cache): seq_k == seq_q, so j <= i
        # For prefix cache: seq_k > seq_q, so j <= i + (seq_k - seq_q)
        i_idx = torch.arange(seq_q_len, device=q.device)[:, None]
        j_idx = torch.arange(seq_k_len, device=q.device)[None, :]
        mask = j_idx > i_idx + (seq_k_len - seq_q_len)
        scores.masked_fill_(mask[None, :, :], float('-inf'))

    weights = torch.softmax(scores, dim=-1)  # [num_heads, seq_q, seq_k]

    # Handle NaN from softmax(-inf) rows (all-masked rows)
    weights = weights.nan_to_num(0.0)

    out = torch.bmm(weights, v_t)  # [num_heads, seq_q, head_dim]
    return out.permute(1, 0, 2)    # [seq_q, num_heads, head_dim]


def naive_sdpa_decode(
    q: Tensor,       # [batch, 1, num_heads, head_dim]
    k_cache: Tensor, # [num_blocks, page_size, num_kv_heads, head_dim]
    v_cache: Tensor, # [num_blocks, page_size, num_kv_heads, head_dim]
    cache_seqlens: Tensor,  # [batch] int32
    block_table: Tensor,    # [batch, max_blocks_per_seq] int32
    scale: float,
) -> Tensor:        # [batch, 1, num_heads, head_dim]
    """Naive decode attention using paged KV cache (reference implementation).

    For each sequence, gathers all KV tokens from the paged cache, then computes
    standard SDPA. Used to verify the Triton decode_attention kernel.
    """
    batch = q.shape[0]
    num_heads = q.shape[2]
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]
    page_size = k_cache.shape[1]

    outputs = []
    for b in range(batch):
        seqlen = cache_seqlens[b].item()

        # Gather K/V from paged cache
        num_blocks = (seqlen + page_size - 1) // page_size
        k_rows = []
        v_rows = []
        for block_idx in range(num_blocks):
            physical_block = block_table[b, block_idx].item()
            valid_tokens = min(page_size, seqlen - block_idx * page_size)
            k_rows.append(k_cache[physical_block, :valid_tokens])
            v_rows.append(v_cache[physical_block, :valid_tokens])

        k_all = torch.cat(k_rows, dim=0)  # [seqlen, num_kv_heads, head_dim]
        v_all = torch.cat(v_rows, dim=0)  # [seqlen, num_kv_heads, head_dim]

        q_b = q[b, 0]  # [num_heads, head_dim]
        o = naive_sdpa(q_b.unsqueeze(0), k_all, v_all, scale, causal=False)
        outputs.append(o)

    return torch.stack(outputs, dim=0)  # [batch, 1, num_heads, head_dim]


def naive_sdpa_prefill_paged(
    q: Tensor,              # [total_q, num_heads, head_dim]
    k_cache: Tensor,        # [num_blocks, page_size, num_kv_heads, head_dim]
    v_cache: Tensor,        # same
    cu_seqlens_q: Tensor,   # [num_seqs + 1] int32
    cu_seqlens_k: Tensor,   # [num_seqs + 1] int32
    scale: float,
    block_table: Tensor,    # [num_seqs, max_blocks_per_seq] int32
) -> Tensor:                # [total_q, num_heads, head_dim]
    """Naive prefill attention with paged KV cache (reference implementation).

    For each sequence, gathers K/V from the paged cache, then computes naive SDPA
    with causal masking. Used to verify the Triton prefill_attention kernel with
    prefix cache.
    """
    num_seqs = cu_seqlens_q.shape[0] - 1
    page_size = k_cache.shape[1]
    outputs = []

    for b in range(num_seqs):
        q_start = cu_seqlens_q[b].item()
        q_end = cu_seqlens_q[b + 1].item()
        seqlen_q = q_end - q_start

        k_len = cu_seqlens_k[b + 1].item() - cu_seqlens_k[b].item()

        # Gather K/V from paged cache
        num_blocks = (k_len + page_size - 1) // page_size
        k_rows = []
        v_rows = []
        for block_idx in range(num_blocks):
            physical_block = block_table[b, block_idx].item()
            valid_tokens = min(page_size, k_len - block_idx * page_size)
            k_rows.append(k_cache[physical_block, :valid_tokens])
            v_rows.append(v_cache[physical_block, :valid_tokens])

        k_all = torch.cat(k_rows, dim=0)  # [seqlen_k, num_kv_heads, head_dim]
        v_all = torch.cat(v_rows, dim=0)

        q_b = q[q_start:q_end]  # [seqlen_q, num_heads, head_dim]

        o = naive_sdpa(q_b, k_all, v_all, scale, causal=True)
        outputs.append(o)

    return torch.cat(outputs, dim=0)  # [total_q, num_heads, head_dim]
