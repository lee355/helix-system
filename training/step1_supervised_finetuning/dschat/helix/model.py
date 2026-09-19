"""Model construction helpers for structurally pruned Helix ranks."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch
from transformers import AutoConfig


def create_helix_model(
    model_class,
    model_name_or_path: str,
    attention_rate: float = 1.0,
    ffn_rate: float = 1.0,
    submodel_ids: Optional[Mapping] = None,
    freeze_blocks: Optional[Sequence[int]] = None,
    num_hidden_layers: Optional[int] = None,
    dropout: Optional[float] = None,
):
    """Create a full model or one rank's dense structured submodel.

    This intentionally does not couple ``attention_rate`` to ``ffn_rate``.
    The old helper silently replaced a full attention rate with the FFN rate,
    which is invalid for planner-generated configurations.
    """

    if not 0.0 < attention_rate <= 1.0:
        raise ValueError(f"attention_rate must be in (0, 1], got {attention_rate}")
    if not 0.0 < ffn_rate <= 1.0:
        raise ValueError(f"ffn_rate must be in (0, 1], got {ffn_rate}")
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.head_prune_rate = float(attention_rate)
    config.prune_rate = float(ffn_rate)
    config.freeze_blocks = None if submodel_ids is None else list(freeze_blocks or [])
    config.submodel_ids = submodel_ids
    if num_hidden_layers is not None:
        config.num_hidden_layers = int(num_hidden_layers)
    if dropout is not None:
        for name in ("dropout", "attention_dropout", "hidden_dropout", "activation_dropout"):
            if hasattr(config, name):
                setattr(config, name, float(dropout))
    return model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        ignore_mismatched_sizes=True,
    )


def validate_state_shapes(model, expected_state: Mapping[str, torch.Tensor]) -> None:
    """Fail early when custom Transformers rounded a planner size incorrectly."""

    actual_state = model.state_dict()
    missing = sorted(set(expected_state).difference(actual_state))
    unexpected = sorted(set(actual_state).difference(expected_state))
    mismatched = [
        (name, tuple(actual_state[name].shape), tuple(expected_state[name].shape))
        for name in expected_state.keys() & actual_state.keys()
        if actual_state[name].shape != expected_state[name].shape
    ]
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing[:5]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:5]}")
        if mismatched:
            details.append(f"shape_mismatch={mismatched[:5]}")
        raise ValueError(
            "The installed Transformers submodel implementation does not match the Helix mask: "
            + "; ".join(details)
        )
