"""
Unit tests for the decode_attention Triton kernel.

Tests compare the Triton kernel output against the naive PyTorch reference
implementation (naive_sdpa_decode from conftest.py).

The kernel is not yet implemented, so all tests are marked with
pytest.skip until the kernel is available.
"""

import pytest
import torch

from tests.conftest import naive_sdpa_decode


# Skip all tests in this module until the kernel is implemented
pytestmark = pytest.mark.skip(reason="decode_attention kernel not yet implemented")


def _try_import_kernel():
    from nanovllm.kernels.decode_attention import decode_attention
    return decode_attention


class TestDecodeAttentionBasic:
    """Basic correctness tests for single-batch decode."""

    def test_single_sequence_minimal(self, device, small_config):
        """batch=1, seqlen=1 (smallest possible decode)."""
        cfg = small_config
        decode_attention = _try_import_kernel()

        q = torch.randn(1, 1, cfg["num_heads"], cfg["head_dim"],
                        dtype=cfg["dtype"], device=device)
        k_cache = torch.randn(cfg["num_blocks"], cfg["page_size"],
                               cfg["num_kv_heads"], cfg["head_dim"],
                               dtype=cfg["dtype"], device=device)
        v_cache = torch.randn_like(k_cache)
        cache_seqlens = torch.tensor([1], dtype=torch.int32, device=device)
        block_table = torch.tensor([[0]], dtype=torch.int32, device=device)
        scale = 1.0 / (cfg["head_dim"] ** 0.5)

        out = decode_attention(q, k_cache, v_cache, cache_seqlens, block_table, scale)
        ref = naive_sdpa_decode(q, k_cache, v_cache, cache_seqlens, block_table, scale)

        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

    def test_single_sequence_long(self, device, small_config):
        """batch=1, seqlen=512 (spans 2 pages with page_size=256)."""
        cfg = small_config
        decode_attention = _try_import_kernel()
        seqlen = 512

        q = torch.randn(1, 1, cfg["num_heads"], cfg["head_dim"],
                        dtype=cfg["dtype"], device=device)
        k_cache = torch.randn(cfg["num_blocks"], cfg["page_size"],
                               cfg["num_kv_heads"], cfg["head_dim"],
                               dtype=cfg["dtype"], device=device)
        v_cache = torch.randn_like(k_cache)
        cache_seqlens = torch.tensor([seqlen], dtype=torch.int32, device=device)
        block_table = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
        scale = 1.0 / (cfg["head_dim"] ** 0.5)

        out = decode_attention(q, k_cache, v_cache, cache_seqlens, block_table, scale)
        ref = naive_sdpa_decode(q, k_cache, v_cache, cache_seqlens, block_table, scale)

        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


class TestDecodeAttentionBatched:
    """Multi-sequence decode with varying lengths."""

    def test_batch_variable_lengths(self, device, small_config):
        """batch=3, different sequence lengths."""
        cfg = small_config
        decode_attention = _try_import_kernel()

        seqlens = [128, 300, 500]
        batch = len(seqlens)
        max_blocks = (max(seqlens) + cfg["page_size"] - 1) // cfg["page_size"]

        q = torch.randn(batch, 1, cfg["num_heads"], cfg["head_dim"],
                        dtype=cfg["dtype"], device=device)
        k_cache = torch.randn(cfg["num_blocks"], cfg["page_size"],
                               cfg["num_kv_heads"], cfg["head_dim"],
                               dtype=cfg["dtype"], device=device)
        v_cache = torch.randn_like(k_cache)
        cache_seqlens = torch.tensor(seqlens, dtype=torch.int32, device=device)

        block_table = torch.full((batch, max_blocks), -1, dtype=torch.int32, device=device)
        block_table[0, 0] = 0
        block_table[1, :2] = [1, 2]
        block_table[2, :2] = [3, 4]

        scale = 1.0 / (cfg["head_dim"] ** 0.5)

        out = decode_attention(q, k_cache, v_cache, cache_seqlens, block_table, scale)
        ref = naive_sdpa_decode(q, k_cache, v_cache, cache_seqlens, block_table, scale)

        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


class TestDecodeAttentionGQA:
    """Tests with Grouped Query Attention (num_kv_heads < num_heads)."""

    def test_gqa_ratio_2(self, device):
        """num_heads=4, num_kv_heads=2 (ratio 2:1)."""
        decode_attention = _try_import_kernel()

        num_heads, num_kv_heads, head_dim = 4, 2, 64
        page_size = 256
        seqlen = 100

        q = torch.randn(1, 1, num_heads, head_dim, dtype=torch.float16, device=device)
        k_cache = torch.randn(32, page_size, num_kv_heads, head_dim,
                               dtype=torch.float16, device=device)
        v_cache = torch.randn_like(k_cache)
        cache_seqlens = torch.tensor([seqlen], dtype=torch.int32, device=device)
        block_table = torch.tensor([[0]], dtype=torch.int32, device=device)
        scale = 1.0 / (head_dim ** 0.5)

        out = decode_attention(q, k_cache, v_cache, cache_seqlens, block_table, scale)
        ref = naive_sdpa_decode(q, k_cache, v_cache, cache_seqlens, block_table, scale)

        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
