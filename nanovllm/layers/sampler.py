import os
import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        self._use_compile = os.environ.get("TORCHDYNAMO_DISABLE", "0") != "1"

    def _forward_impl(self, logits: torch.Tensor, temperatures: torch.Tensor, top_ps: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # Top-p (nucleus) filtering
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # Zero out tokens with cumulative prob above the threshold (keep at least one)
        sorted_mask = cumulative_probs - sorted_probs > top_ps.unsqueeze(dim=1)
        sorted_logits[sorted_mask] = float('-inf')
        # Scatter back to original ordering
        logits.scatter_(1, sorted_indices, sorted_logits)
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, top_ps: torch.Tensor):
        if self._use_compile:
            return self._compiled_forward(logits, temperatures, top_ps)
        return self._forward_impl(logits, temperatures, top_ps)

    @torch.compile
    def _compiled_forward(self, logits: torch.Tensor, temperatures: torch.Tensor, top_ps: torch.Tensor):
        return self._forward_impl(logits, temperatures, top_ps)
