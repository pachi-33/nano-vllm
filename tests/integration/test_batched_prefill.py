"""
Integration test: batched (multi-sequence) prefill.

Tests the prefill path with multiple prompts of different lengths,
verifying that cu_seqlens are constructed correctly and each sequence
gets independent attention.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Attention kernels not yet implemented")


class TestBatchedPrefill:
    """End-to-end prefill for multiple prompts."""

    def test_two_prompts_different_lengths(self):
        """Two prompts of different lengths should both complete prefill."""
        # TODO: Implement after kernels are ready
        # 1. Create LLM
        # 2. Add two requests with different prompt lengths
        # 3. Run step() once
        # 4. Verify both sequences have is_prefill transitioned correctly
        # 5. Verify each sequence produced output tokens independently
        pass

    def test_cu_seqlens_correctness(self):
        """Verify cu_seqlens_q and cu_seqlens_k are correctly constructed."""
        # TODO: Verify the tensors built by prepare_prefill match expected values
        # for a set of sequences with known token counts
        pass

    def test_independent_outputs(self):
        """Each sequence's output should not be affected by other sequences."""
        # TODO: Run same prompt alone and in batch → verify output is close
        pass
