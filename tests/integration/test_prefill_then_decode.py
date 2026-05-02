"""
Integration test: full prefill → decode loop.

Tests the complete inference pipeline from prompt input to output generation,
covering both the prefill and decode phases.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Attention kernels not yet implemented")


class TestPrefillThenDecode:
    """Full pipeline integration tests."""

    def test_generate_max_tokens(self):
        """generate() should produce exactly max_tokens completion tokens."""
        # TODO:
        # 1. Create LLM with max_tokens=10
        # 2. Call generate(["test prompt"])
        # 3. Verify output has exactly 10 completion tokens
        pass

    def test_eos_termination(self):
        """Generation should stop when EOS token is produced."""
        # TODO:
        # 1. Create LLM with max_tokens=1000
        # 2. Generate (EOS should appear well before 1000 tokens)
        # 3. Verify output has fewer than max_tokens tokens
        pass

    def test_multiple_prompts_different_lengths(self):
        """Multiple prompts should generate independently."""
        # TODO:
        # 1. Create LLM
        # 2. Generate for 3 prompts with different lengths and max_tokens
        # 3. Verify each output has the correct number of tokens
        pass

    def test_output_text_is_coherent(self):
        """Generated text should be readable (not garbage)."""
        # TODO:
        # 1. Load Qwen3-0.6B
        # 2. Generate with temperature=0
        # 3. Verify output is valid UTF-8 and non-empty
        pass
