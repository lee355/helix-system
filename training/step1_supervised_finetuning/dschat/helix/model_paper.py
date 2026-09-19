"""Model construction details needed by structured Helix submodels."""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

from transformers import AutoConfig


def shape_safe_rate(rate: float) -> float:
    """Nudge a rational pruning rate upward to survive ``int(N * rate)``."""

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
    """Create a model using the only audited custom-attention backend.

    The installed FlashAttention2 path still reshapes with the original head
    count.  Eager attention uses the retained GQA-group count throughout and
    is therefore the safe backend for the current custom Transformers build.
    """

    config = AutoConfig.from_pretrained(model_name_or_path)
    config.head_prune_rate = shape_safe_rate(attention_rate)
    config.prune_rate = shape_safe_rate(ffn_rate)
    config.freeze_blocks = None if submodel_ids is None else list(freeze_blocks or [])
    config.submodel_ids = submodel_ids
    config._attn_implementation = "eager"
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
