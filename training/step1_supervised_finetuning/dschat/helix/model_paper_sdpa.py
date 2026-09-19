"""Audited model factory for the custom Llama/Qwen2 structured paths."""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

from transformers import AutoConfig


def shape_safe_rate(rate: float) -> float:
    if not 0.0 < rate <= 1.0:
        raise ValueError(f"pruning rate must be in (0, 1], got {rate}")
    return 1.0 if rate == 1.0 else math.nextafter(float(rate), math.inf)


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
    """Create a shape-exact submodel and force the audited SDPA backend.

    In ``ours_math``, SDPA is the only backend supporting retained head counts
    for both Llama and Qwen2 (with ``output_attentions=False``). FlashAttention2
    still reshapes with the original number of heads. ``pretraining_tp`` must
    be one because the custom TP branch also uses original dimensions.
    """

    config = AutoConfig.from_pretrained(model_name_or_path)
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
    return model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        ignore_mismatched_sizes=True,
    )
