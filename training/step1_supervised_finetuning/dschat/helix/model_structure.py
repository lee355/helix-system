"""Model structure inference that respects an explicit Qwen3 head_dim."""

from __future__ import annotations

from .masking import ModelStructure, _config_value, _find_first, _layer_id


def infer_model_structure(state_dict, config=None) -> ModelStructure:
    _, q_weight = _find_first(state_dict, ".self_attn.q_proj.weight")
    _, k_weight = _find_first(state_dict, ".self_attn.k_proj.weight")
    _, gate_weight = _find_first(state_dict, ".mlp.gate_proj.weight")
    num_attention_heads = _config_value(config, "num_attention_heads")
    num_key_value_heads = _config_value(config, "num_key_value_heads")
    hidden_size = _config_value(config, "hidden_size") or int(q_weight.shape[1])
    if num_attention_heads is None:
        raise ValueError("config.num_attention_heads is required")
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads
    if num_attention_heads % num_key_value_heads:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    head_dim = _config_value(config, "head_dim")
    if head_dim is None:
        head_dim = hidden_size // num_attention_heads
    if q_weight.shape[0] != num_attention_heads * head_dim:
        raise ValueError(
            f"q_proj output {q_weight.shape[0]} does not match "
            f"{num_attention_heads} heads x {head_dim}"
        )
    if k_weight.shape[0] != num_key_value_heads * head_dim:
        raise ValueError(
            f"k_proj output {k_weight.shape[0]} does not match "
            f"{num_key_value_heads} KV heads x {head_dim}"
        )
    layer_ids = tuple(
        sorted(
            {
                layer
                for name in state_dict
                if (layer := _layer_id(name)) is not None
            }
        )
    )
    if not layer_ids:
        raise ValueError("cannot infer transformer layer ids")
    return ModelStructure(
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_size=int(gate_weight.shape[0]),
        layer_ids=layer_ids,
    )
