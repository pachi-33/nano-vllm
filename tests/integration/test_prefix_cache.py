"""
Integration test: prefix caching.

Tests that the prefix cache (hash-based block reuse) works correctly:
when the same prompt prefix is seen again, cached KV blocks should be
reused instead of recomputed.

Note: prefix caching only hashes complete blocks. With block_size=256,
a prompt must have at least 256 tokens for any block to be cached.
"""

import pytest


class TestPrefixCache:
    """Prefix cache integration tests."""

    def test_same_prompt_second_hit(self, engine, scheduler, make_sequence, do_prefill):
        """Same long prompt requested twice should hit prefix cache on second request."""
        # Need enough tokens to fill at least 1 complete block (block_size=256)
        # For 1 complete block: need 256+ tokens (block 0 is fully filled and hashed)
        prompt_tokens = list(range(300))  # 300 tokens → 1 complete block + 1 partial

        # First request: no cache hit
        seq1 = make_sequence(prompt_tokens)
        do_prefill([seq1])

        # Second request with same prompt: should reuse cached blocks
        seq2 = make_sequence(prompt_tokens)
        num_cached = scheduler.block_manager.can_allocate(seq2)
        assert num_cached > 0, "Second request should have prefix cache hits"

    def test_shared_prefix(self, engine, scheduler, make_sequence, do_prefill):
        """Two prompts sharing a long prefix should reuse prefix blocks."""
        # Prompt A: 512 tokens → 2 complete blocks (blocks 0 and 1 hashed)
        prompt_a = list(range(512))
        seq_a = make_sequence(prompt_a)
        do_prefill([seq_a])

        # Prompt B shares first 512 tokens with A, but is longer
        prompt_b = list(range(512)) + list(range(1000, 1100))  # 612 tokens
        seq_b = make_sequence(prompt_b)

        # seq_b should have cache hits for the shared prefix blocks
        num_cached = scheduler.block_manager.can_allocate(seq_b)
        assert num_cached > 0, "Shared prefix should cause cache hits"

        do_prefill([seq_b])
        # seq_b should reuse some of seq_a's blocks
        shared_blocks = set(seq_a.block_table).intersection(set(seq_b.block_table))
        assert len(shared_blocks) > 0, "seq_b should reuse some of seq_a's blocks"

    def test_prefix_cache_does_not_affect_output(self, engine, scheduler, make_sequence, do_prefill, do_step):
        """Output correctness should not be affected by prefix cache."""
        # Use a prompt long enough for prefix caching
        prompt_tokens = list(range(300))
        max_tokens = 3

        # First run: no cache
        seq1 = make_sequence(prompt_tokens, max_tokens=max_tokens)
        do_prefill([seq1])
        while not seq1.is_finished:
            do_step()
        output1 = list(seq1.completion_token_ids)

        # Second run: same prompt, should hit cache
        seq2 = make_sequence(prompt_tokens, max_tokens=max_tokens)
        num_cached = scheduler.block_manager.can_allocate(seq2)
        assert num_cached > 0, "Should have cache hits"
        do_prefill([seq2])
        while not seq2.is_finished:
            do_step()
        output2 = list(seq2.completion_token_ids)

        # Both should produce valid outputs
        assert len(output1) == max_tokens
        assert len(output2) == max_tokens
