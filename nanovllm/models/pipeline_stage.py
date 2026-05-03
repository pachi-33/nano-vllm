import torch
from torch import nn
from transformers import Qwen3Config

from nanovllm.layers.layernorm import RMSNorm
from nanovllm.models.qwen3 import Qwen3DecoderLayer
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class PipelineStageModel(nn.Module):
    """Single pipeline stage: holds a contiguous slice of decoder layers."""

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, hf_config: Qwen3Config, pp_rank: int, pp_size: int):
        super().__init__()
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        num_layers = hf_config.num_hidden_layers
        layers_per_stage = num_layers // pp_size
        self.start_layer = pp_rank * layers_per_stage
        self.end_layer = self.start_layer + layers_per_stage
        self.num_stage_layers = self.end_layer - self.start_layer

        # Record original layer indices for weight loading
        self.layer_indices = list(range(self.start_layer, self.end_layer))

        # First stage holds embed_tokens
        if pp_rank == 0:
            self.embed_tokens = VocabParallelEmbedding(hf_config.vocab_size, hf_config.hidden_size)

        # Only instantiate this stage's layers
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(hf_config) for _ in range(self.num_stage_layers)
        ])

        # Last stage holds norm + lm_head
        if pp_rank == pp_size - 1:
            self.norm = RMSNorm(hf_config.hidden_size, eps=hf_config.rms_norm_eps)
            self.lm_head = ParallelLMHead(hf_config.vocab_size, hf_config.hidden_size)
            if hf_config.tie_word_embeddings and pp_rank == 0:
                self.lm_head.weight.data = self.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                hidden_states: torch.Tensor | None = None,
                residual: torch.Tensor | None = None):
        # Stage 0: embed input_ids
        if self.pp_rank == 0:
            hidden_states = self.embed_tokens(input_ids)
            residual = None

        # Run this stage's layers
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)

        # Last stage: norm + lm_head logits
        if self.pp_rank == self.pp_size - 1:
            hidden_states, _ = self.norm(hidden_states, residual)
            residual = None
            return self.lm_head(hidden_states), residual

        return hidden_states, residual
