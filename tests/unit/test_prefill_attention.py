"""
Unit tests for the prefill_attention Triton kernel.

Tests compare the Triton kernel output against naive PyTorch reference
implementations (naive_sdpa and naive_sdpa_prefill_paged from conftest.py).
"""

import pytest
import torch

from tests.conftest import naive_sdpa, naive_sdpa_prefill_paged

try:
    from nanovllm.kernels.prefill_attention import prefill_attention
except ImportError as e:
    pytestmark = pytest.mark.skip(reason=f"Cannot import prefill_attention: {e}")
    prefill_attention = None


class TestPrefillNoPrefixCache:
    """Tests without prefix cache (block_table=None, direct K/V access)."""

    def test_single_sequence(self, device, small_config):
        """Single sequence, seqlen_q == seqlen_k."""
        cfg = small_config

        seqlen = 32

        q = torch.randn(seqlen, cfg["num_heads"], cfg["head_dim"],
                        dtype=cfg["dtype"], device=device)
        k = torch.randn(seqlen, cfg["num_kv_heads"], cfg["head_dim"],
                        dtype=cfg["dtype"], device=device)
        v = torch.randn_like(k)
        cu_q = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
        cu_k = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
        scale = 1.0 / (cfg["head_dim"] ** 0.5)

        out = prefill_attention(q, k, v, cu_q, cu_k, seqlen, seqlen, scale,
                                causal=True, block_table=None)
        ref = naive_sdpa(q, k, v, scale, causal=True)

        torch.testing.assert_close(out, ref, atol=5e-3, rtol=1e-2)

    def test_multi_sequence_variable_lengths(self, device, small_config):
        """3 sequences with different lengths, packed together."""
        cfg = small_config


        seqlens = [16, 8, 24]
        total_q = sum(seqlens)

        q = torch.randn(total_q, cfg["num_heads"], cfg["head_dim"],
                        dtype=cfg["dtype"], device=device)
        k = torch.randn(total_q, cfg["num_kv_heads"], cfg["head_dim"],
                        dtype=cfg["dtype"], device=device)
        v = torch.randn_like(k)

        cu_q = torch.tensor([0] + [sum(seqlens[:i+1]) for i in range(len(seqlens))],
                            dtype=torch.int32, device=device)
        cu_k = cu_q.clone()  # no prefix cache: Q and K same length

        max_seqlen = max(seqlens)
        scale = 1.0 / (cfg["head_dim"] ** 0.5)

        out = prefill_attention(q, k, v, cu_q, cu_k, max_seqlen, max_seqlen, scale,
                                causal=True, block_table=None)

        # Reference: compute each sequence independently and concatenate
        ref_parts = []
        for i, slen in enumerate(seqlens):
            q_start = cu_q[i].item()
            q_end = cu_q[i + 1].item()
            ref_parts.append(naive_sdpa(q[q_start:q_end], k[q_start:q_end],
                                        v[q_start:q_end], scale, causal=True))
        ref = torch.cat(ref_parts, dim=0)

        torch.testing.assert_close(out, ref, atol=5e-3, rtol=1e-2)

    def test_causal_mask_correctness(self, device, small_config):
        """Verify that position i only attends to positions <= i."""
        cfg = small_config

        seqlen = 16

        q = torch.randn(seqlen, cfg["num_heads"], cfg["head_dim"],
                        dtype=cfg["dtype"], device=device)
        k = torch.randn(seqlen, cfg["num_kv_heads"], cfg["head_dim"],
                        dtype=cfg["dtype"], device=device)
        v = torch.randn_like(k)
        cu_q = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
        cu_k = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
        scale = 1.0 / (cfg["head_dim"] ** 0.5)

        out = prefill_attention(q, k, v, cu_q, cu_k, seqlen, seqlen, scale,
                                causal=True, block_table=None)

        # Verify output at position i is independent of k[j] for j > i
        # by perturbing K at position j and checking output at position i < j
        for j in range(1, seqlen):
            k_perturbed = k.clone()
            k_perturbed[j] = 999.0  # arbitrary large perturbation
            out_perturbed = prefill_attention(
                q, k_perturbed, v, cu_q, cu_k, seqlen, seqlen, scale,
                causal=True, block_table=None
            )
            # Output at positions 0..j-1 should be unchanged
            if j > 0:
                torch.testing.assert_close(
                    out[:j], out_perturbed[:j], atol=1e-5, rtol=1e-5
                )


class TestPrefillWithPrefixCache:
    """Tests with prefix cache (block_table!=None, paged KV access)."""

    def test_single_sequence_with_prefix(self, device, small_config):
        """Single sequence where K has cached prefix + new tokens."""
        cfg = small_config


        num_cached = 100  # prefix cache tokens
        num_new = 32      # new tokens to prefill
        seqlen_q = num_new
        seqlen_k = num_cached + num_new

        # Build paged KV cache with data
        k_cache = torch.randn(cfg["num_blocks"], cfg["page_size"],
                               cfg["num_kv_heads"], cfg["head_dim"],
                               dtype=cfg["dtype"], device=device)
        v_cache = torch.randn_like(k_cache)

        # Q tokens (the new tokens)
        q = torch.randn(seqlen_q, cfg["num_heads"], cfg["head_dim"],
                        dtype=cfg["dtype"], device=device)

        cu_q = torch.tensor([0, seqlen_q], dtype=torch.int32, device=device)
        cu_k = torch.tensor([0, seqlen_k], dtype=torch.int32, device=device)

        # Block table: sequence uses physical blocks [0, 1] (covers 512 tokens)
        block_table = torch.tensor([[0, 1]], dtype=torch.int32, device=device)

        scale = 1.0 / (cfg["head_dim"] ** 0.5)

        out = prefill_attention(q, k_cache, v_cache, cu_q, cu_k,
                                seqlen_q, seqlen_k, scale,
                                causal=True, block_table=block_table)

        ref = naive_sdpa_prefill_paged(q, k_cache, v_cache, cu_q, cu_k,
                                       scale, block_table)

        torch.testing.assert_close(out, ref, atol=5e-3, rtol=1e-2)


class TestPrefillGQA:
    """Tests with Grouped Query Attention."""

    def test_gqa_no_prefix_cache(self, device):
        """GQA with no prefix cache."""


        num_heads, num_kv_heads, head_dim = 4, 2, 64
        seqlen = 16

        q = torch.randn(seqlen, num_heads, head_dim, dtype=torch.float16, device=device)
        k = torch.randn(seqlen, num_kv_heads, head_dim, dtype=torch.float16, device=device)
        v = torch.randn_like(k)
        cu_q = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
        cu_k = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
        scale = 1.0 / (head_dim ** 0.5)

        out = prefill_attention(q, k, v, cu_q, cu_k, seqlen, seqlen, scale,
                                causal=True, block_table=None)
        ref = naive_sdpa(q, k, v, scale, causal=True)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
