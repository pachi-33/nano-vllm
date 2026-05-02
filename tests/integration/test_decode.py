"""
Integration test: decode phase.

Tests the autoregressive decode loop where each step generates one token
per sequence from the paged KV cache.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Attention kernels not yet implemented")


class TestDecode:
    """Decode phase integration tests."""

    def test_single_step_decode(self):
        """After prefill, one decode step should produce exactly 1 token."""
        # TODO:
        # 1. Prefill a single prompt
        # 2. Run one decode step
        # 3. Verify seq.num_tokens increased by 1
        pass

    def test_multi_step_decode(self):
        """Multiple decode steps should produce one token each."""
        # TODO:
        # 1. Prefill a prompt
        # 2. Run 5 decode steps
        # 3. Verify seq has 5 more tokens than after prefill
        pass

    def test_batched_decode(self):
        """Multiple sequences decode in parallel, each getting 1 token."""
        # TODO:
        # 1. Prefill 3 prompts
        # 2. Run one decode step
        # 3. Verify each sequence got 1 new token
        pass

    def test_kv_cache_grows_correctly(self):
        """Block table should grow as decode adds tokens crossing page boundaries."""
        # TODO:
        # 1. Prefill until seq_len is just below a page boundary
        # 2. Decode until crossing the boundary
        # 3. Verify a new block was allocated and block_table grew
        pass
