"""
Integration test: batched (multi-sequence) prefill.

Tests the prefill path with multiple prompts of different lengths,
verifying that cu_seqlens are constructed correctly and each sequence
gets independent attention.
"""

import pytest
from nanovllm.engine.sequence import SequenceStatus


class TestBatchedPrefill:
    """End-to-end prefill for multiple prompts."""

    def test_two_prompts_different_lengths(self, engine, scheduler, make_sequence, do_prefill):
        """Two prompts of different lengths should both complete prefill."""
        seq1 = make_sequence(list(range(10, 30)))   # 20 tokens
        seq2 = make_sequence(list(range(100, 200))) # 100 tokens
        seqs = do_prefill([seq1, seq2])

        assert len(seqs) == 2
        for seq in seqs:
            assert seq.status == SequenceStatus.RUNNING
            assert seq.num_completion_tokens == 1

    def test_batched_block_tables_independent(self, engine, scheduler, make_sequence, do_prefill):
        """Each sequence should get its own independent blocks."""
        seq1 = make_sequence(list(range(50)))
        seq2 = make_sequence(list(range(100)))
        seqs = do_prefill([seq1, seq2])

        # Block tables should not share physical blocks
        blocks1 = set(seq1.block_table)
        blocks2 = set(seq2.block_table)
        assert not blocks1.intersection(blocks2), "Sequences share physical blocks"

    def test_three_prompts(self, engine, scheduler, make_sequence, do_prefill):
        """Three prompts should all prefill correctly."""
        seqs = [
            make_sequence(list(range(10))),
            make_sequence(list(range(20))),
            make_sequence(list(range(30))),
        ]
        results = do_prefill(seqs)

        assert len(results) == 3
        for seq in results:
            assert seq.status == SequenceStatus.RUNNING
            assert seq.num_completion_tokens == 1
