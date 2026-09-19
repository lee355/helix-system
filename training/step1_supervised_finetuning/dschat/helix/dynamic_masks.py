"""In-place-by-rank mask adjustment for Helix Section 3.4.2.

Unlike initial cumulative assignment, a live adjustment must not shift every
later rank.  Only the affected rank changes: shrink removes its least-important
redundant regions; growth adds the least-covered regions while preserving all
other ranks exactly.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Mapping, Sequence, Tuple

import torch

from .masking import (
    ModelStructure,
    SubmodelSpec,
    TensorMask,
    _expand_heads,
    _full_mask,
    _layer_id,
)
from .paper_semantics import paper_quantized_count


@dataclass(frozen=True)
class DynamicMaskUpdate:
    masks: Tuple[OrderedDict[str, TensorMask], ...]
    specs: Tuple[SubmodelSpec, ...]
    changed_rank: int
    removed_attention_groups: Tuple[int, ...]
    added_attention_groups: Tuple[int, ...]
    removed_ffn_columns: Tuple[int, ...]
    added_ffn_columns: Tuple[int, ...]


def _resize_regions(
    current: Sequence[int],
    target_count: int,
    coverage: Sequence[int],
    total_regions: int,
    label: str,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    retained = set(int(index) for index in current)
    removed: List[int] = []
    added: List[int] = []
    if target_count < len(retained):
        # Canonical indices are ordered by descending L1 importance. Remove
        # the least-important (largest index) only when another rank retains it.
        candidates = sorted(
            (index for index in retained if coverage[index] > 1),
            reverse=True,
        )
        required = len(retained) - target_count
        if len(candidates) < required:
            raise ValueError(
                f"cannot shrink {label} by {required}: only {len(candidates)} "
                "selected regions have another owner"
            )
        removed = candidates[:required]
        retained.difference_update(removed)
    elif target_count > len(retained):
        required = target_count - len(retained)
        candidates = sorted(
            (index for index in range(total_regions) if index not in retained),
            key=lambda index: (coverage[index], index),
        )
        if len(candidates) < required:
            raise ValueError(f"cannot grow {label} to {target_count} regions")
        added = candidates[:required]
        retained.update(added)
    return tuple(sorted(retained)), tuple(sorted(removed)), tuple(sorted(added))


def _rank_mask(
    state_dict: Mapping[str, torch.Tensor],
    structure: ModelStructure,
    attention_groups: Sequence[int],
    ffn_columns: Sequence[int],
) -> OrderedDict[str, TensorMask]:
    result: OrderedDict[str, TensorMask] = OrderedDict()
    q_heads: List[int] = []
    for group in attention_groups:
        start = group * structure.query_heads_per_kv
        q_heads.extend(range(start, start + structure.query_heads_per_kv))

    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor) or tensor.ndim not in (1, 2):
            continue
        if _layer_id(name) is None:
            result[name] = _full_mask(tensor)
            continue
        q_rows = _expand_heads(q_heads, structure.head_dim, tensor.device)
        kv_rows = _expand_heads(attention_groups, structure.head_dim, tensor.device)
        ffn = torch.tensor(ffn_columns, dtype=torch.long, device=tensor.device)
        full_rows = torch.arange(tensor.shape[0], device=tensor.device)
        full_columns = (
            torch.arange(tensor.shape[1], device=tensor.device)
            if tensor.ndim == 2
            else None
        )

        if ".self_attn.q_proj." in name:
            result[name] = (q_rows, full_columns) if tensor.ndim == 2 else q_rows
        elif ".self_attn.k_proj." in name or ".self_attn.v_proj." in name:
            result[name] = (kv_rows, full_columns) if tensor.ndim == 2 else kv_rows
        elif name.endswith(".self_attn.o_proj.weight"):
            result[name] = (full_rows, q_rows)
        elif ".mlp.gate_proj." in name or ".mlp.up_proj." in name:
            result[name] = (ffn, full_columns) if tensor.ndim == 2 else ffn
        elif name.endswith(".mlp.down_proj.weight"):
            result[name] = (full_rows, ffn)
        else:
            # Layer norms and Qwen3 q_norm/k_norm remain replicated.
            result[name] = _full_mask(tensor)
    return result


def resize_rank_masks(
    state_dict: Mapping[str, torch.Tensor],
    old_masks: Sequence[Mapping[str, TensorMask]],
    old_specs: Sequence[SubmodelSpec],
    *,
    changed_rank: int,
    new_size: float,
    structure: ModelStructure,
    ffn_alignment: int = 1,
) -> DynamicMaskUpdate:
    if len(old_masks) != len(old_specs) or not old_specs:
        raise ValueError("old masks/specs must have the same non-zero world size")
    if changed_rank < 0 or changed_rank >= len(old_specs):
        raise ValueError(f"changed rank {changed_rank} is outside the world")

    attention_coverage = [0] * structure.num_key_value_heads
    ffn_coverage = [0] * structure.intermediate_size
    for spec in old_specs:
        for index in spec.attention_groups:
            attention_coverage[index] += 1
        for index in spec.ffn_columns:
            ffn_coverage[index] += 1
    if min(attention_coverage) < 1 or min(ffn_coverage) < 1:
        raise ValueError("old masks do not cover every attention/FFN region")

    old = old_specs[changed_rank]
    attention_target = paper_quantized_count(
        new_size,
        structure.num_key_value_heads,
    )
    ffn_target = paper_quantized_count(
        new_size,
        structure.intermediate_size,
        alignment=ffn_alignment,
    )
    attention, removed_attention, added_attention = _resize_regions(
        old.attention_groups,
        attention_target,
        attention_coverage,
        structure.num_key_value_heads,
        "attention",
    )
    ffn, removed_ffn, added_ffn = _resize_regions(
        old.ffn_columns,
        ffn_target,
        ffn_coverage,
        structure.intermediate_size,
        "FFN",
    )

    masks = [OrderedDict(mask) for mask in old_masks]
    masks[changed_rank] = _rank_mask(state_dict, structure, attention, ffn)
    specs = list(old_specs)
    specs[changed_rank] = SubmodelSpec(
        requested_size=float(new_size),
        attention_groups=attention,
        ffn_columns=ffn,
        attention_rate=len(attention) / structure.num_key_value_heads,
        ffn_rate=len(ffn) / structure.intermediate_size,
    )
    return DynamicMaskUpdate(
        masks=tuple(masks),
        specs=tuple(specs),
        changed_rank=changed_rank,
        removed_attention_groups=removed_attention,
        added_attention_groups=added_attention,
        removed_ffn_columns=removed_ffn,
        added_ffn_columns=added_ffn,
    )
