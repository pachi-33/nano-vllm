"""
Integration test: decode phase.

Tests the autoregressive decode loop where each step generates one token
per sequence from the paged KV cache.
"""

import pytest
from nanovllm.engine.sequence import SequenceStatus


class TestDecode:
    """Decode phase integration tests."""

    def test_single_step_decode(self, engine, scheduler, make_sequence, do_prefill, do_step):
        """After prefill, one decode step should produce exactly 1 additional token."""
        seq = make_sequence(list(range(10, 50)))
        do_prefill([seq])
        tokens_after_prefill = seq.num_completion_tokens  # 1

        outputs, _ = do_step()
        assert seq.num_completion_tokens == tokens_after_prefill + 1

    def test_multi_step_decode(self, engine, scheduler, make_sequence, do_prefill, do_step):
        """Multiple decode steps should produce one token each."""
        seq = make_sequence(list(range(10, 50)))
        do_prefill([seq])
        tokens_after_prefill = seq.num_tokens

        for _ in range(5):
            do_step()
        assert seq.num_tokens == tokens_after_prefill + 5

    def test_batched_decode(self, engine, scheduler, make_sequence, do_prefill, do_step):
        """Multiple sequences decode in parallel, each getting 1 token."""
        seq1 = make_sequence(list(range(20)))
        seq2 = make_sequence(list(range(30)))
        seq3 = make_sequence(list(range(40)))
        do_prefill([seq1, seq2, seq3])

        tokens_before = [s.num_tokens for s in [seq1, seq2, seq3]]
        do_step()
        tokens_after = [s.num_tokens for s in [seq1, seq2, seq3]]

        for before, after in zip(tokens_before, tokens_after):
            assert after == before + 1

    def test_decode_crosses_page_boundary(self, engine, scheduler, make_sequence, do_prefill, do_step):
        """Decode should correctly allocate new blocks when crossing page boundaries.

        may_append allocates a new block when len(seq) % block_size == 1.
        With prompt_len=254: after prefill → 255 tokens, 1 block.
        After 2 decode steps → 257 tokens. At the next schedule call (3rd decode),
        len=257 → 257 % 256 == 1 → new block allocated.
        """
        prompt_len = 254
        seq = make_sequence(list(range(prompt_len)), max_tokens=10)
        do_prefill([seq])

        blocks_before = len(seq.block_table)

        # Decode 3 steps to trigger new block allocation
        for _ in range(3):
            do_step()

        # A new block should have been allocated
        assert len(seq.block_table) > blocks_before
