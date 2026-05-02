"""
Integration test: full prefill → decode loop.

Tests the complete inference pipeline from prompt input to output generation,
covering both the prefill and decode phases.
"""

import pytest


class TestPrefillThenDecode:
    """Full pipeline integration tests."""

    def test_generate_max_tokens(self, engine, scheduler, make_sequence, do_prefill, do_step):
        """Full pipeline should produce exactly max_tokens completion tokens."""
        max_tokens = 5
        seq = make_sequence(list(range(10, 30)), max_tokens=max_tokens)
        do_prefill([seq])

        while not seq.is_finished:
            do_step()

        assert seq.num_completion_tokens == max_tokens

    def test_eos_termination(self, engine, scheduler, tokenizer, make_sequence, do_prefill, do_step):
        """Generation should stop either by EOS or reaching max_tokens."""
        max_tokens = 64
        seq = make_sequence(
            tokenizer.encode("Hello, how are you?"),
            max_tokens=max_tokens,
        )
        do_prefill([seq])

        while not seq.is_finished:
            do_step()

        # Generation should have completed (either by EOS or max_tokens)
        assert seq.is_finished
        assert seq.num_completion_tokens <= max_tokens

    def test_output_text_is_coherent(self, engine, scheduler, tokenizer, make_sequence, do_prefill, do_step):
        """Generated text should be valid and non-empty."""
        prompt_text = "The capital of France is"
        prompt_tokens = tokenizer.encode(prompt_text)
        seq = make_sequence(prompt_tokens, max_tokens=10)
        do_prefill([seq])

        while not seq.is_finished:
            do_step()

        # Decode completion tokens
        completion_text = tokenizer.decode(seq.completion_token_ids)
        assert len(completion_text) > 0
        assert isinstance(completion_text, str)

    def test_multiple_prompts_independent(self, engine, scheduler, tokenizer, make_sequence, do_prefill, do_step):
        """Multiple prompts should generate independently to completion."""
        prompts = [
            tokenizer.encode("Hello"),
            tokenizer.encode("The weather today"),
        ]
        max_tokens = 5
        seqs = [make_sequence(p, max_tokens=max_tokens) for p in prompts]
        do_prefill(seqs)

        while not all(s.is_finished for s in seqs):
            do_step()

        for seq in seqs:
            assert seq.num_completion_tokens == max_tokens
