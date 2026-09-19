"""Unified paper model factory for official Llama and Qwen3 backport paths."""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from .qwen3_compat import Qwen3CompatConfig, install_qwen3_compat_attention


def shape_safe_rate(rate: float) -> float:
    if not 0.0 < rate <= 1.0:
        raise ValueError(f"pruning rate must be in (0, 1], got {rate}")
    return 1.0 if rate == 1.0 else math.nextafter(float(rate), math.inf)


def _raw_config(model_name_or_path: str):
    payload, _ = PretrainedConfig.get_config_dict(model_name_or_path)
    return payload


def _configure_llama_topology(config, attention_rate: float, ffn_rate: float) -> None:
    """Represent a dense structured submodel using official Llama dimensions."""

    original_query_heads = int(config.num_attention_heads)
    original_kv_heads = int(config.num_key_value_heads)
    original_intermediate = int(config.intermediate_size)
    original_head_dim = int(
        getattr(config, "head_dim", None)
        or config.hidden_size // original_query_heads
    )
    if original_query_heads % original_kv_heads:
        raise ValueError("Llama query heads must be divisible by KV heads")
    retained_kv_heads = max(1, int(round(original_kv_heads * attention_rate)))
    retained_query_heads = retained_kv_heads * (
        original_query_heads // original_kv_heads
    )
    retained_intermediate = max(1, int(round(original_intermediate * ffn_rate)))
    config.head_dim = original_head_dim
    config.num_key_value_heads = retained_kv_heads
    config.num_attention_heads = retained_query_heads
    config.intermediate_size = retained_intermediate


def create_paper_model(
    model_class,
    model_name_or_path: str,
    attention_rate: float = 1.0,
    ffn_rate: float = 1.0,
    submodel_ids: Optional[Mapping] = None,
    freeze_blocks: Optional[Sequence[int]] = None,
    num_hidden_layers: Optional[int] = None,
    dropout: Optional[float] = None,
):
    raw = _raw_config(model_name_or_path)
    model_type = raw.get("model_type")
    if model_type == "qwen3":
        install_qwen3_compat_attention()
        config = Qwen3CompatConfig(**raw)
        effective_model_class = Qwen2ForCausalLM
        config.architectures = ["Qwen3ForCausalLM"]
    else:
        config = AutoConfig.from_pretrained(model_name_or_path)
        effective_model_class = model_class

    if model_type == "llama":
        _configure_llama_topology(config, attention_rate, ffn_rate)
        # Official Llama consumes exact topology fields; no forked attention
        # implementation is required.
        config.head_prune_rate = 1.0
        config.prune_rate = 1.0
    else:
        config.head_prune_rate = shape_safe_rate(attention_rate)
        config.prune_rate = shape_safe_rate(ffn_rate)

    config.freeze_blocks = None if submodel_ids is None else list(freeze_blocks or [])
    config.submodel_ids = submodel_ids
    config._attn_implementation = "sdpa"
    if hasattr(config, "pretraining_tp"):
        config.pretraining_tp = 1
    if hasattr(config, "output_attentions"):
        config.output_attentions = False
    if num_hidden_layers is not None:
        config.num_hidden_layers = int(num_hidden_layers)
    if dropout is not None:
        for name in (
            "dropout",
            "attention_dropout",
            "hidden_dropout",
            "activation_dropout",
        ):
            if hasattr(config, name):
                setattr(config, name, float(dropout))

    return effective_model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        ignore_mismatched_sizes=(submodel_ids is not None),
    )
