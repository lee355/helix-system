"""Structured submodel masks used by Helix.

This implements Algorithm 1 from the paper.  Transformer regions are first
ordered by their L1 importance and then assigned to ranks with a cumulative
circular offset.  The state dictionary is physically canonicalized into that
importance order.  Consequently, the full model is functionally unchanged,
while every selected region remains a regular, structurally valid projection
that can be represented by the existing DeepSpeed rectangle interface.

For grouped-query attention (GQA), a region is one KV head together with all
of its associated query heads.  Splitting that unit would invalidate the
constant query-to-KV mapping used by Llama/Qwen attention kernels.
"""

from __future__ import annotations

import copy
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import torch


TensorMask = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class ModelStructure:
    """The structural dimensions needed to construct valid submodels."""

    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    layer_ids: Tuple[int, ...]

    @property
    def query_heads_per_kv(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


@dataclass(frozen=True)
class SubmodelSpec:
    """The effective, quantized size assigned to one rank."""

    requested_size: float
    attention_groups: Tuple[int, ...]
    ffn_columns: Tuple[int, ...]
    attention_rate: float
    ffn_rate: float

    @property
    def num_attention_heads(self) -> int:
        return len(self.attention_groups)

    @property
    def num_ffn_columns(self) -> int:
        return len(self.ffn_columns)


def _config_value(config, name: str) -> Optional[int]:
    if config is None:
        return None
    value = getattr(config, name, None)
    return int(value) if value is not None else None


def _find_first(state_dict: Mapping[str, torch.Tensor], suffix: str) -> Tuple[str, torch.Tensor]:
    for name, tensor in state_dict.items():
        if name.endswith(suffix):
            return name, tensor
    raise ValueError(f"Cannot infer model structure: no tensor ending in {suffix!r}")


def _layer_id(name: str) -> Optional[int]:
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def infer_model_structure(
    state_dict: Mapping[str, torch.Tensor],
    config=None,
) -> ModelStructure:
    """Infer attention/FFN dimensions without model-name-specific constants."""

    _, q_weight = _find_first(state_dict, ".self_attn.q_proj.weight")
    _, k_weight = _find_first(state_dict, ".self_attn.k_proj.weight")
    _, gate_weight = _find_first(state_dict, ".mlp.gate_proj.weight")

    num_attention_heads = _config_value(config, "num_attention_heads")
    num_key_value_heads = _config_value(config, "num_key_value_heads")
    hidden_size = _config_value(config, "hidden_size") or int(q_weight.shape[1])

    if num_attention_heads is None:
        raise ValueError("config.num_attention_heads is required to infer head_dim safely")
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError(
            "num_attention_heads must be divisible by num_key_value_heads: "
            f"{num_attention_heads} vs {num_key_value_heads}"
        )

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

    layer_ids = sorted(
        {
            layer_id
            for name in state_dict
            if (layer_id := _layer_id(name)) is not None
        }
    )
    if not layer_ids:
        raise ValueError("No transformer layers were found in the state dictionary")

    return ModelStructure(
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_size=int(gate_weight.shape[0]),
        layer_ids=tuple(layer_ids),
    )


def _sum_abs(tensor: torch.Tensor, dims: Union[int, Tuple[int, ...]]) -> torch.Tensor:
    # L1 scores do not participate in autograd.  float32 avoids fp16 reduction
    # overflow while keeping profiling-time memory bounded.
    return tensor.detach().to(dtype=torch.float32).abs().sum(dim=dims).cpu()


def _attention_group_scores(
    state_dict: Mapping[str, torch.Tensor],
    prefix: str,
    structure: ModelStructure,
) -> torch.Tensor:
    num_kv = structure.num_key_value_heads
    q_per_kv = structure.query_heads_per_kv
    head_dim = structure.head_dim

    q = state_dict[f"{prefix}.self_attn.q_proj.weight"]
    k = state_dict[f"{prefix}.self_attn.k_proj.weight"]
    v = state_dict[f"{prefix}.self_attn.v_proj.weight"]
    o = state_dict[f"{prefix}.self_attn.o_proj.weight"]

    scores = _sum_abs(q.reshape(num_kv, q_per_kv, head_dim, q.shape[1]), (1, 2, 3))
    scores += _sum_abs(k.reshape(num_kv, head_dim, k.shape[1]), (1, 2))
    scores += _sum_abs(v.reshape(num_kv, head_dim, v.shape[1]), (1, 2))
    scores += _sum_abs(o.reshape(o.shape[0], num_kv, q_per_kv, head_dim), (0, 2, 3))

    for projection, grouped_q in (("q_proj", True), ("k_proj", False), ("v_proj", False)):
        bias_name = f"{prefix}.self_attn.{projection}.bias"
        if bias_name not in state_dict:
            continue
        bias = state_dict[bias_name]
        if grouped_q:
            scores += _sum_abs(bias.reshape(num_kv, q_per_kv, head_dim), (1, 2))
        else:
            scores += _sum_abs(bias.reshape(num_kv, head_dim), 1)
    return scores


def _ffn_column_scores(state_dict: Mapping[str, torch.Tensor], prefix: str) -> torch.Tensor:
    gate = state_dict[f"{prefix}.mlp.gate_proj.weight"]
    up = state_dict[f"{prefix}.mlp.up_proj.weight"]
    down = state_dict[f"{prefix}.mlp.down_proj.weight"]
    scores = _sum_abs(gate, 1) + _sum_abs(up, 1) + _sum_abs(down, 0)
    for projection in ("gate_proj", "up_proj"):
        bias_name = f"{prefix}.mlp.{projection}.bias"
        if bias_name in state_dict:
            scores += state_dict[bias_name].detach().to(dtype=torch.float32).abs().cpu()
    return scores


def _head_rows(head_ids: torch.Tensor, head_dim: int, device: torch.device) -> torch.Tensor:
    head_ids = head_ids.to(device=device, dtype=torch.long)
    offsets = torch.arange(head_dim, device=device, dtype=torch.long)
    return (head_ids[:, None] * head_dim + offsets[None, :]).reshape(-1)


def _query_heads_for_groups(
    group_ids: torch.Tensor,
    q_per_kv: int,
    device: torch.device,
) -> torch.Tensor:
    group_ids = group_ids.to(device=device, dtype=torch.long)
    offsets = torch.arange(q_per_kv, device=device, dtype=torch.long)
    return (group_ids[:, None] * q_per_kv + offsets[None, :]).reshape(-1)


def _copy_reordered(tensor: torch.Tensor, axis: int, indices: torch.Tensor) -> None:
    indices = indices.to(device=tensor.device, dtype=torch.long)
    reordered = tensor.index_select(axis, indices)
    with torch.no_grad():
        tensor.copy_(reordered)


def canonicalize_state_dict_by_importance(
    state_dict: MutableMapping[str, torch.Tensor],
    config=None,
    structure: Optional[ModelStructure] = None,
) -> Dict[int, Dict[str, List[int]]]:
    """Order every prunable region by descending joint L1 importance.

    Paired rows/columns are permuted together, so this is a function-preserving
    reparameterization of the full model.  The returned permutations map the
    canonical coordinates back to the original checkpoint coordinates.
    """

    structure = structure or infer_model_structure(state_dict, config)
    permutations: Dict[int, Dict[str, List[int]]] = {}
    q_per_kv = structure.query_heads_per_kv

    for layer_id in structure.layer_ids:
        prefix = f"model.layers.{layer_id}"
        q_name = f"{prefix}.self_attn.q_proj.weight"
        if q_name not in state_dict:
            # Some wrappers prepend a module name. Resolve it from the suffix.
            suffix = f".layers.{layer_id}.self_attn.q_proj.weight"
            q_name, _ = _find_first(state_dict, suffix)
            prefix = q_name[: -len(".self_attn.q_proj.weight")]

        attention_order = torch.argsort(
            _attention_group_scores(state_dict, prefix, structure),
            descending=True,
            stable=True,
        )
        ffn_order = torch.argsort(
            _ffn_column_scores(state_dict, prefix),
            descending=True,
            stable=True,
        )

        q_heads = _query_heads_for_groups(attention_order, q_per_kv, torch.device("cpu"))
        q_rows = _head_rows(q_heads, structure.head_dim, torch.device("cpu"))
        kv_rows = _head_rows(attention_order, structure.head_dim, torch.device("cpu"))

        for projection in ("q_proj",):
            _copy_reordered(state_dict[f"{prefix}.self_attn.{projection}.weight"], 0, q_rows)
            bias_name = f"{prefix}.self_attn.{projection}.bias"
            if bias_name in state_dict:
                _copy_reordered(state_dict[bias_name], 0, q_rows)
        for projection in ("k_proj", "v_proj"):
            _copy_reordered(state_dict[f"{prefix}.self_attn.{projection}.weight"], 0, kv_rows)
            bias_name = f"{prefix}.self_attn.{projection}.bias"
            if bias_name in state_dict:
                _copy_reordered(state_dict[bias_name], 0, kv_rows)
        _copy_reordered(state_dict[f"{prefix}.self_attn.o_proj.weight"], 1, q_rows)

        for projection in ("gate_proj", "up_proj"):
            _copy_reordered(state_dict[f"{prefix}.mlp.{projection}.weight"], 0, ffn_order)
            bias_name = f"{prefix}.mlp.{projection}.bias"
            if bias_name in state_dict:
                _copy_reordered(state_dict[bias_name], 0, ffn_order)
        _copy_reordered(state_dict[f"{prefix}.mlp.down_proj.weight"], 1, ffn_order)

        permutations[layer_id] = {
            "attention_groups": attention_order.tolist(),
            "ffn_columns": ffn_order.tolist(),
        }
    return permutations


def _quantized_count(size: float, regions: int, alignment: int = 1) -> int:
    if not 0.0 < size <= 1.0:
        raise ValueError(f"submodel size must be in (0, 1], got {size}")
    if alignment < 1:
        raise ValueError(f"alignment must be >= 1, got {alignment}")
    count = int(math.ceil(size * regions - 1e-12))
    count = int(math.ceil(count / alignment) * alignment)
    return max(1, min(regions, count))


def _circular_assignments(
    sizes: Sequence[float],
    regions: int,
    alignment: int = 1,
) -> List[Tuple[int, ...]]:
    assignments: List[Tuple[int, ...]] = []
    current_index = 0
    for size in sizes:
        count = _quantized_count(float(size), regions, alignment)
        assignments.append(tuple((current_index + offset) % regions for offset in range(count)))
        current_index = (current_index + count) % regions
    return assignments


def _expand_heads(heads: Sequence[int], head_dim: int, device: torch.device) -> torch.Tensor:
    return _head_rows(torch.tensor(heads, dtype=torch.long), head_dim, device)


def _full_mask(tensor: torch.Tensor) -> TensorMask:
    if tensor.ndim == 1:
        return torch.arange(tensor.shape[0], device=tensor.device, dtype=torch.long)
    if tensor.ndim == 2:
        return (
            torch.arange(tensor.shape[0], device=tensor.device, dtype=torch.long),
            torch.arange(tensor.shape[1], device=tensor.device, dtype=torch.long),
        )
    raise ValueError(f"Only one- and two-dimensional parameters are supported, got {tensor.shape}")


def build_structured_masks(
    state_dict: Mapping[str, torch.Tensor],
    submodel_sizes: Sequence[float],
    config=None,
    structure: Optional[ModelStructure] = None,
    freeze_blocks_by_rank: Optional[Sequence[Sequence[int]]] = None,
    ffn_alignment: int = 1,
) -> Tuple[List[OrderedDict[str, TensorMask]], List[SubmodelSpec]]:
    """Build exact masks for all ranks using cumulative circular assignment."""

    if not submodel_sizes:
        raise ValueError("At least one submodel size is required")
    structure = structure or infer_model_structure(state_dict, config)
    world_size = len(submodel_sizes)
    if freeze_blocks_by_rank is None:
        freeze_blocks_by_rank = [()] * world_size
    if len(freeze_blocks_by_rank) != world_size:
        raise ValueError("freeze_blocks_by_rank must have one entry per rank")

    attention_assignments = _circular_assignments(
        submodel_sizes,
        structure.num_key_value_heads,
    )
    ffn_assignments = _circular_assignments(
        submodel_sizes,
        structure.intermediate_size,
        alignment=ffn_alignment,
    )

    masks: List[OrderedDict[str, TensorMask]] = [OrderedDict() for _ in submodel_sizes]
    specs: List[SubmodelSpec] = []
    for rank, size in enumerate(submodel_sizes):
        specs.append(
            SubmodelSpec(
                requested_size=float(size),
                attention_groups=attention_assignments[rank],
                ffn_columns=ffn_assignments[rank],
                attention_rate=len(attention_assignments[rank]) / structure.num_key_value_heads,
                ffn_rate=len(ffn_assignments[rank]) / structure.intermediate_size,
            )
        )

    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue
        if tensor.ndim not in (1, 2):
            # Non-parameter buffers are copied verbatim by slice_state_dict.
            continue
        layer_id = _layer_id(name)

        for rank in range(world_size):
            frozen = layer_id is not None and layer_id in set(freeze_blocks_by_rank[rank])
            if frozen or layer_id is None:
                masks[rank][name] = _full_mask(tensor)
                continue

            groups = attention_assignments[rank]
            q_heads: List[int] = []
            for group in groups:
                first = group * structure.query_heads_per_kv
                q_heads.extend(range(first, first + structure.query_heads_per_kv))
            q_rows = _expand_heads(q_heads, structure.head_dim, tensor.device)
            kv_rows = _expand_heads(groups, structure.head_dim, tensor.device)
            ffn_columns = torch.tensor(ffn_assignments[rank], device=tensor.device, dtype=torch.long)

            if ".self_attn.q_proj." in name:
                selected = q_rows
                masks[rank][name] = (
                    selected,
                    torch.arange(tensor.shape[1], device=tensor.device),
                ) if tensor.ndim == 2 else selected
            elif ".self_attn.k_proj." in name or ".self_attn.v_proj." in name:
                selected = kv_rows
                masks[rank][name] = (
                    selected,
                    torch.arange(tensor.shape[1], device=tensor.device),
                ) if tensor.ndim == 2 else selected
            elif name.endswith(".self_attn.o_proj.weight"):
                masks[rank][name] = (
                    torch.arange(tensor.shape[0], device=tensor.device),
                    q_rows,
                )
            elif ".mlp.gate_proj." in name or ".mlp.up_proj." in name:
                masks[rank][name] = (
                    ffn_columns,
                    torch.arange(tensor.shape[1], device=tensor.device),
                ) if tensor.ndim == 2 else ffn_columns
            elif name.endswith(".mlp.down_proj.weight"):
                masks[rank][name] = (
                    torch.arange(tensor.shape[0], device=tensor.device),
                    ffn_columns,
                )
            else:
                masks[rank][name] = _full_mask(tensor)

    return masks, specs


def slice_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    mask: Mapping[str, TensorMask],
) -> OrderedDict[str, torch.Tensor]:
    """Extract one rank's dense, structurally pruned state dictionary."""

    local_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, tensor in state_dict.items():
        if name not in mask or not torch.is_tensor(tensor) or tensor.ndim not in (1, 2):
            local_state[name] = copy.deepcopy(tensor)
            continue
        indices = mask[name]
        if tensor.ndim == 1:
            assert isinstance(indices, torch.Tensor)
            local_state[name] = tensor.index_select(0, indices.to(tensor.device)).detach().clone()
        else:
            assert isinstance(indices, tuple)
            rows, columns = indices
            local_state[name] = (
                tensor.index_select(0, rows.to(tensor.device))
                .index_select(1, columns.to(tensor.device))
                .detach()
                .clone()
            )
    return local_state


def serialize_mask(mask: Mapping[str, TensorMask]) -> Dict[str, Union[List[int], Tuple[List[int], List[int]]]]:
    """Convert a mask to the JSON/pickle-friendly form consumed by Transformers."""

    serialized: Dict[str, Union[List[int], Tuple[List[int], List[int]]]] = {}
    for name, indices in mask.items():
        if isinstance(indices, tuple):
            serialized[name] = (indices[0].cpu().tolist(), indices[1].cpu().tolist())
        else:
            serialized[name] = indices.cpu().tolist()
    return serialized
