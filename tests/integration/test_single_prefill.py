"""
Integration test: single prompt prefill.

Tests the full prefill path for a single sequence:
  input_ids → model_runner.prepare_prefill → model forward → token output

Requires the attention kernels to be implemented.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Attention kernels not yet implemented")


class TestSinglePrefill:
    """End-to-end prefill for a single prompt."""

    def test_single_prompt_produces_tokens(self):
        """A single prompt should produce at least one output token after prefill."""
        # TODO: Implement after kernels are ready
        # 1. Create LLM with enforce_eager=True
        # 2. Add a single request
        # 3. Run scheduler.schedule() → expect is_prefill=True
        # 4. Run model_runner.run(seqs, is_prefill=True)
        # 5. Verify token_ids is a non-empty list
        pass

    def test_scheduler_state_after_prefill(self):
        """After prefill, the sequence should be in RUNNING state with correct metadata."""
        # TODO: Verify:
        # - seq.status == RUNNING
        # - seq.num_cached_tokens == num_prompt_tokens
        # - seq.block_table is populated
        # - block_manager has allocated the right number of blocks
        pass
