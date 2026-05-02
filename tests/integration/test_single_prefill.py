"""
Integration test: single prompt prefill.

Tests the full prefill path for a single sequence:
  Sequence → scheduler.add → schedule → model_runner.run → postprocess
"""

import pytest
from nanovllm.engine.sequence import SequenceStatus


class TestSinglePrefill:
    """End-to-end prefill for a single prompt."""

    def test_single_prompt_produces_tokens(self, engine, scheduler, make_sequence, do_prefill):
        """A single prompt should produce exactly 1 output token after prefill."""
        prompt_tokens = list(range(10, 50))  # 40 tokens
        seq = make_sequence(prompt_tokens)
        seqs = do_prefill([seq])

        assert len(seqs) == 1
        # After prefill, the sequence should have one completion token
        assert seq.num_completion_tokens == 1
        assert seq.status == SequenceStatus.RUNNING

    def test_scheduler_state_after_prefill(self, engine, scheduler, make_sequence, do_prefill):
        """After prefill, the sequence should be in RUNNING state with correct metadata."""
        prompt_tokens = list(range(10, 100))  # 90 tokens
        seq = make_sequence(prompt_tokens)
        seqs = do_prefill([seq])

        assert seq.status == SequenceStatus.RUNNING
        assert seq.num_cached_tokens == len(prompt_tokens)
        assert len(seq.block_table) > 0
        # Block table should have enough blocks for all prompt tokens
        expected_blocks = (len(prompt_tokens) + scheduler.block_size - 1) // scheduler.block_size
        assert len(seq.block_table) == expected_blocks

    def test_prefill_short_prompt(self, engine, scheduler, make_sequence, do_prefill):
        """Prefill with a very short prompt (1 token)."""
        seq = make_sequence([42])
        seqs = do_prefill([seq])

        assert seq.status == SequenceStatus.RUNNING
        assert seq.num_completion_tokens == 1

    def test_prefill_long_prompt(self, engine, scheduler, make_sequence, do_prefill):
        """Prefill with a prompt that spans multiple pages (block_size=256)."""
        prompt_tokens = list(range(300))  # > 1 page
        seq = make_sequence(prompt_tokens)
        seqs = do_prefill([seq])

        assert seq.status == SequenceStatus.RUNNING
        assert seq.num_cached_tokens == len(prompt_tokens)
        assert len(seq.block_table) == 2  # 300 tokens → 2 pages
