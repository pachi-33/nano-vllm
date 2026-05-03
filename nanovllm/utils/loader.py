import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _remap_pp_weight(weight_name: str, model: nn.Module) -> str | None:
    """Remap weight names for PipelineStageModel layer indices.

    Safetensors key patterns:
      - "model.layers.N.xxx" → stage layers use local indices
      - "model.embed_tokens.weight" → only stage 0
      - "model.norm.weight" → only last stage
      - "lm_head.weight" → only last stage
    """
    if not hasattr(model, 'pp_rank'):
        return weight_name

    prefix = "model.layers."
    if weight_name.startswith(prefix):
        # Layer weight: check if this layer belongs to our stage
        rest = weight_name[len(prefix):]
        dot = rest.index(".")
        orig_idx = int(rest[:dot])
        suffix = rest[dot:]  # e.g. ".self_attn.qkv_proj.weight"
        if orig_idx not in model.layer_indices:
            return None
        local_idx = model.layer_indices.index(orig_idx)
        return f"layers.{local_idx}{suffix}"

    # Non-layer weights: embed_tokens, norm, lm_head
    # PipelineStageModel puts these at top level (no "model." prefix)
    # Try both with and without "model." prefix
    for candidate in [weight_name, weight_name.removeprefix("model.")]:
        try:
            model.get_parameter(candidate)
            return candidate
        except (AttributeError, RuntimeError):
            continue
    return None


def load_model(model: nn.Module, path: str, pp_rank: int = 0, pp_size: int = 1):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    is_pp = pp_size > 1
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Remap for pipeline parallelism
                if is_pp:
                    remapped = _remap_pp_weight(weight_name, model)
                    if remapped is None:
                        continue
                    weight_name_local = remapped
                else:
                    weight_name_local = weight_name

                for k in packed_modules_mapping:
                    if k in weight_name_local:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name_local.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name_local)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
