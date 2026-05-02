"""
Unit tests for the store_kvcache Triton kernel.

This kernel is already implemented in nanovllm/layers/attention.py.
These tests verify its correctness against naive PyTorch writes.
"""

import pytest
import torch

# The store_kvcache kernel lives in nanovllm/layers/attention.py.
# Importing it requires the full nanovllm package to be importable
# (attention.py imports from nanovllm.utils.context, which triggers the
# package __init__). If the environment is broken (e.g., triton.backends
# or transformers missing), skip these tests entirely.
try:
    from nanovllm.layers.attention import store_kvcache
except ImportError:
    pytestmark = pytest.mark.skip(reason="nanovllm package not importable (environment issue)")
    store_kvcache = None


@pytest.fixture
def cache_tensors(device, small_config):
    """Create K, V, k_cache, v_cache tensors for testing."""
    cfg = small_config
    N = 8  # number of tokens
    k_cache, v_cache = torch.zeros(cfg["num_blocks"], cfg["page_size"],
                                    cfg["num_kv_heads"], cfg["head_dim"],
                                    dtype=cfg["dtype"], device=device), \
                       torch.zeros(cfg["num_blocks"], cfg["page_size"],
                                    cfg["num_kv_heads"], cfg["head_dim"],
                                    dtype=cfg["dtype"], device=device)
    k = torch.randn(N, cfg["num_kv_heads"], cfg["head_dim"],
                    dtype=cfg["dtype"], device=device)
    v = torch.randn(N, cfg["num_kv_heads"], cfg["head_dim"],
                    dtype=cfg["dtype"], device=device)
    return k, v, k_cache, v_cache


class TestStoreKvCacheBasic:
    """Basic correctness: write K/V into cache and verify values."""

    def test_write_contiguous_block(self, cache_tensors, small_config):
        """Write 8 tokens into a single block and verify."""
        k, v, k_cache, v_cache = cache_tensors
        cfg = small_config

        # All 8 tokens go to physical block 0, offsets 0-7
        slot_mapping = torch.arange(0, 8, dtype=torch.int32,
                                     device=k.device)

        store_kvcache(k, v, k_cache, v_cache, slot_mapping)

        # Verify each token was written correctly
        D = cfg["num_kv_heads"] * cfg["head_dim"]
        for i in range(8):
            expected_k = k[i].flatten()
            actual_k = k_cache[0, i].flatten()
            torch.testing.assert_close(actual_k, expected_k)

            expected_v = v[i].flatten()
            actual_v = v_cache[0, i].flatten()
            torch.testing.assert_close(actual_v, expected_v)

    def test_write_across_blocks(self, cache_tensors, small_config):
        """Write tokens spanning two physical blocks."""
        k, v, k_cache, v_cache = cache_tensors
        cfg = small_config

        # Tokens 0-3 → block 0 offsets 0-3
        # Tokens 4-7 → block 5 offsets 0-3
        slot_mapping = torch.tensor(
            [0 * cfg["page_size"] + 0,
             0 * cfg["page_size"] + 1,
             0 * cfg["page_size"] + 2,
             0 * cfg["page_size"] + 3,
             5 * cfg["page_size"] + 0,
             5 * cfg["page_size"] + 1,
             5 * cfg["page_size"] + 2,
             5 * cfg["page_size"] + 3],
            dtype=torch.int32, device=k.device
        )

        store_kvcache(k, v, k_cache, v_cache, slot_mapping)

        # Verify block 0
        for i in range(4):
            torch.testing.assert_close(k_cache[0, i].flatten(), k[i].flatten())
            torch.testing.assert_close(v_cache[0, i].flatten(), v[i].flatten())

        # Verify block 5
        for i in range(4):
            torch.testing.assert_close(k_cache[5, i].flatten(), k[i + 4].flatten())
            torch.testing.assert_close(v_cache[5, i].flatten(), v[i + 4].flatten())

    def test_slot_minus_one_skips(self, cache_tensors, small_config):
        """slot_mapping = -1 should skip writing to that position."""
        k, v, k_cache, v_cache = cache_tensors
        cfg = small_config

        # Pre-fill cache with known values
        k_cache.fill_(0.0)
        v_cache.fill_(0.0)

        # Only write tokens 0, 2, 4, 6 (skip odd ones)
        slot_mapping = torch.tensor(
            [0, -1, 1, -1, 2, -1, 3, -1],
            dtype=torch.int32, device=k.device
        )

        store_kvcache(k, v, k_cache, v_cache, slot_mapping)

        # Verify written positions
        torch.testing.assert_close(k_cache[0, 0].flatten(), k[0].flatten())
        torch.testing.assert_close(k_cache[0, 1].flatten(), k[2].flatten())
        torch.testing.assert_close(k_cache[0, 2].flatten(), k[4].flatten())
        torch.testing.assert_close(k_cache[0, 3].flatten(), k[6].flatten())

        # Verify skipped positions remain zero
        assert torch.all(k_cache[0, 4] == 0)
        assert torch.all(v_cache[0, 4] == 0)

    def test_does_not_overwrite_other_blocks(self, cache_tensors, small_config):
        """Writing to block 0 should not affect block 1."""
        k, v, k_cache, v_cache = cache_tensors
        cfg = small_config

        # Fill block 1 with sentinel values
        k_cache[1].fill_(42.0)
        v_cache[1].fill_(42.0)

        slot_mapping = torch.arange(0, 8, dtype=torch.int32, device=k.device)
        store_kvcache(k, v, k_cache, v_cache, slot_mapping)

        # Block 1 should be untouched
        assert torch.all(k_cache[1] == 42.0)
        assert torch.all(v_cache[1] == 42.0)

    def test_float16_precision(self, device):
        """Test with float16 dtype (production dtype)."""
        N, num_kv_heads, head_dim = 16, 4, 64
        k = torch.randn(N, num_kv_heads, head_dim, dtype=torch.float16, device=device)
        v = torch.randn(N, num_kv_heads, head_dim, dtype=torch.float16, device=device)
        k_cache = torch.zeros(64, 256, num_kv_heads, head_dim, dtype=torch.float16, device=device)
        v_cache = torch.zeros(64, 256, num_kv_heads, head_dim, dtype=torch.float16, device=device)

        slot_mapping = torch.arange(0, N, dtype=torch.int32, device=device)
        store_kvcache(k, v, k_cache, v_cache, slot_mapping)

        for i in range(N):
            torch.testing.assert_close(k_cache[0, i], k[i], atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(v_cache[0, i], v[i], atol=1e-3, rtol=1e-3)
