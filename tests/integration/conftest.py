"""
Integration test fixtures for nano-vLLM.

Provides:
- Session-scoped LLMEngine (loaded once, shared across all integration tests)
- Function-scoped scheduler cleanup (resets state between tests)
- Helper factories for creating sequences and prefilling them
"""

import os
import pytest

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B_fp16/")


# ---------------------------------------------------------------------------
# Session-scoped engine
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def engine():
    """LLMEngine instance, loaded once per test session and shared by all integration tests.

    Uses enforce_eager=True (no CUDA graph capture) for faster startup and
    easier debugging. tensor_parallel_size=1 for single-GPU testing.
    """
    engine = LLMEngine(MODEL_PATH, enforce_eager=True, tensor_parallel_size=1)
    yield engine
    if hasattr(engine, 'model_runner'):
        engine.exit()


# ---------------------------------------------------------------------------
# Function-scoped state reset
# ---------------------------------------------------------------------------

@pytest.fixture
def scheduler(engine):
    """Scheduler with clean state for each test.

    Clears waiting/running queues and recreates the block manager so that
    tests do not interfere with each other. The block manager is rebuilt
    with the same num_blocks and block_size from the original.
    """
    sch = engine.scheduler
    # Deallocate any leftover sequences
    for seq in list(sch.waiting):
        sch.block_manager.deallocate(seq)
    for seq in list(sch.running):
        sch.block_manager.deallocate(seq)
    # Clear queues
    sch.waiting.clear()
    sch.running.clear()
    # Recreate block manager to reset all block state (ref counts, hashes)
    num_blocks = len(sch.block_manager.blocks)
    block_size = sch.block_manager.block_size
    sch.block_manager = BlockManager(num_blocks, block_size)
    return sch


# ---------------------------------------------------------------------------
# Accessory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tokenizer(engine):
    """Tokenizer from the LLMEngine."""
    return engine.tokenizer


@pytest.fixture
def model_runner(engine):
    """ModelRunner from the LLMEngine."""
    return engine.model_runner


# ---------------------------------------------------------------------------
# Sequence factories
# ---------------------------------------------------------------------------

@pytest.fixture
def make_sequence():
    """Factory that creates Sequence objects with sensible defaults.

    Usage::

        seq = make_sequence([1, 2, 3, 4, 5])
        seq = make_sequence([1, 2, 3], temperature=0.8, max_tokens=128)
    """
    def _make(token_ids: list[int], temperature: float = 0.6, max_tokens: int = 64):
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        return Sequence(token_ids, sp)
    return _make


@pytest.fixture
def do_prefill(engine):
    """Helper that runs the prefill step for given sequences.

    Handles the full flow: scheduler.schedule() → model_runner.call("run") →
    scheduler.postprocess(). Returns the list of sequences after prefill.

    Usage::

        seqs = do_prefill([seq1, seq2])
        # seqs are now in RUNNING state, ready for decode
    """
    def _run(sequences):
        for seq in sequences:
            engine.scheduler.add(seq)
        seqs, is_prefill = engine.scheduler.schedule()
        assert is_prefill, "Expected prefill scheduling"
        token_ids = engine.model_runner.call("run", seqs, is_prefill)
        engine.scheduler.postprocess(seqs, token_ids, is_prefill)
        return seqs
    return _run


@pytest.fixture
def do_step(engine):
    """Helper that runs one engine step (schedule → run → postprocess).

    Returns (finished_outputs, num_tokens) like LLMEngine.step().

    Usage::

        outputs, num_tokens = do_step()
    """
    def _run():
        return engine.step()
    return _run
