"""CPU oracle and compact metadata checks for Helix full-model reconstruction.

The distributed gather intentionally transfers one owner for every global
parameter rectangle.  This module provides an independent implementation used
by tests and by the gather preflight:

* rectangle metadata must partition every required tensor exactly once;
* local/global rectangles must have identical extents;
* an omitted state-dict key is accepted only when it is an exact tied alias of
  a covered key (for example Llama's embedding and language-model head); and
* when rank masks and local states are available, every overlapping value must
  already be synchronized before one owner is selected for reconstruction.

Coverage checks operate on rectangle bounds rather than a per-element bitmap,
so auditing a multi-billion-parameter model does not allocate model-sized
temporary tensors.
"""

from __future__ import annotations

import copy
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from .masking import TensorMask


Rectangle = Tuple[List[int], List[int]]
LayerRectangles = Mapping[str, Sequence[Rectangle]]
RankRectangles = Mapping[int, LayerRectangles]
LocalStates = Union[
    Sequence[Mapping[str, torch.Tensor]],
    Mapping[int, Mapping[str, torch.Tensor]],
]


@dataclass(frozen=True)
class ReconstructionAudit:
    """Summary of an exact reconstruction-metadata audit."""

    covered_names: Tuple[str, ...]
    alias_sources: Mapping[str, str]
    owner_numel_by_rank: Mapping[int, int]


def _rank_mapping(states: LocalStates) -> Dict[int, Mapping[str, torch.Tensor]]:
    if isinstance(states, Mapping):
        return {int(rank): state for rank, state in states.items()}
    return {rank: state for rank, state in enumerate(states)}


def _exact_alias_key(tensor: torch.Tensor):
    if tensor.layout != torch.strided or tensor.device.type == "meta" or tensor.numel() == 0:
        return None
    storage = tensor.untyped_storage()
    return (
        tensor.device.type,
        tensor.device.index,
        tensor.dtype,
        storage.data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
    )


def exact_state_dict_alias_groups(
    state_dict: Mapping[str, torch.Tensor],
) -> Tuple[Tuple[str, ...], ...]:
    """Return state-dict keys that are exact views of the same tensor storage."""

    grouped: Dict[object, List[str]] = defaultdict(list)
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue
        key = _exact_alias_key(tensor)
        if key is not None:
            grouped[key].append(name)
    return tuple(tuple(names) for names in grouped.values() if len(names) > 1)


def _alias_lookup(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, Tuple[str, ...]]:
    aliases: Dict[str, Tuple[str, ...]] = {}
    for group in exact_state_dict_alias_groups(state_dict):
        for name in group:
            aliases[name] = group
    return aliases


def _rectangle_bounds(
    rectangle: Rectangle,
    shape: torch.Size,
    label: str,
) -> Tuple[int, int, int, int]:
    if not isinstance(rectangle, (tuple, list)) or len(rectangle) != 2:
        raise ValueError(f"{label}: rectangle must contain inclusive start/end points")
    start, end = rectangle
    if len(start) != 2 or len(end) != 2:
        raise ValueError(f"{label}: rectangle points must have two coordinates")
    row_start, col_start = int(start[0]), int(start[1])
    row_end, col_end = int(end[0]), int(end[1])
    if len(shape) == 1:
        if row_start != 0 or row_end != 0:
            raise ValueError(f"{label}: a vector rectangle must use row coordinate zero")
        if col_start < 0 or col_end < col_start or col_end >= int(shape[0]):
            raise ValueError(f"{label}: rectangle lies outside vector shape {tuple(shape)}")
        return 0, 0, col_start, col_end
    if len(shape) != 2:
        raise ValueError(f"{label}: only one- and two-dimensional tensors are reconstructible")
    if (
        row_start < 0
        or row_end < row_start
        or row_end >= int(shape[0])
        or col_start < 0
        or col_end < col_start
        or col_end >= int(shape[1])
    ):
        raise ValueError(f"{label}: rectangle lies outside matrix shape {tuple(shape)}")
    return row_start, row_end, col_start, col_end


def _rectangle_shape(rectangle: Rectangle, ndim: int) -> Tuple[int, ...]:
    if ndim == 1:
        return (int(rectangle[1][1]) - int(rectangle[0][1]) + 1,)
    return (
        int(rectangle[1][0]) - int(rectangle[0][0]) + 1,
        int(rectangle[1][1]) - int(rectangle[0][1]) + 1,
    )


def _rectangle_numel(bounds: Tuple[int, int, int, int], ndim: int) -> int:
    row_start, row_end, col_start, col_end = bounds
    columns = col_end - col_start + 1
    return columns if ndim == 1 else (row_end - row_start + 1) * columns


def _rectangles_overlap(
    left: Tuple[int, int, int, int],
    right: Tuple[int, int, int, int],
    ndim: int,
) -> bool:
    if left[3] < right[2] or right[3] < left[2]:
        return False
    if ndim == 1:
        return True
    return not (left[1] < right[0] or right[1] < left[0])


def _validate_partition(
    name: str,
    shape: torch.Size,
    rectangles: Sequence[Tuple[int, Rectangle]],
) -> Dict[int, int]:
    if not rectangles:
        raise ValueError(f"reconstruction metadata leaves {name} uncovered")
    parsed: List[Tuple[int, Tuple[int, int, int, int]]] = []
    owner_numel: Dict[int, int] = defaultdict(int)
    for index, (rank, rectangle) in enumerate(rectangles):
        bounds = _rectangle_bounds(rectangle, shape, f"{name}/rank{rank}/rectangle{index}")
        for previous_rank, previous in parsed:
            if _rectangles_overlap(bounds, previous, len(shape)):
                raise ValueError(
                    f"reconstruction metadata conflict for {name}: rank {rank} and "
                    f"rank {previous_rank} own overlapping rectangles"
                )
        size = _rectangle_numel(bounds, len(shape))
        owner_numel[rank] += size
        parsed.append((rank, bounds))

    covered = sum(owner_numel.values())
    required = int(torch.tensor(shape).prod().item())
    if covered != required:
        raise ValueError(
            f"reconstruction metadata leaves {name} uncovered: "
            f"covered {covered} of {required} elements"
        )
    return dict(owner_numel)


def _rectangle_view(tensor: torch.Tensor, rectangle: Rectangle) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor[int(rectangle[0][1]) : int(rectangle[1][1]) + 1]
    if tensor.ndim == 2:
        return tensor[
            int(rectangle[0][0]) : int(rectangle[1][0]) + 1,
            int(rectangle[0][1]) : int(rectangle[1][1]) + 1,
        ]
    raise ValueError(f"unsupported reconstruction tensor shape {tuple(tensor.shape)}")


def validate_reconstruction_metadata(
    reference_state: Mapping[str, torch.Tensor],
    reconstruction_global_by_rank: RankRectangles,
    reconstruction_local_by_rank: RankRectangles,
    *,
    required_names: Optional[Iterable[str]] = None,
    allowed_sources: Optional[Iterable[int]] = None,
    local_states_by_rank: Optional[LocalStates] = None,
) -> ReconstructionAudit:
    """Validate exact global coverage and paired local rectangle metadata.

    ``required_names`` defaults to every one- or two-dimensional tensor in the
    reference state.  Tied aliases may be represented by one member, but a
    missing ordinary tensor is always rejected.
    """

    global_sources = {int(rank) for rank in reconstruction_global_by_rank}
    local_sources = {int(rank) for rank in reconstruction_local_by_rank}
    if global_sources != local_sources:
        raise ValueError(
            "reconstruction global/local metadata has different source ranks: "
            f"global={sorted(global_sources)}, local={sorted(local_sources)}"
        )
    if allowed_sources is not None:
        allowed = {int(rank) for rank in allowed_sources}
        missing_sources = sorted(global_sources.difference(allowed))
        if missing_sources:
            raise ValueError(
                "reconstruction metadata references owner sources absent from gather: "
                + ", ".join(str(rank) for rank in missing_sources)
            )

    local_states = (
        {} if local_states_by_rank is None else _rank_mapping(local_states_by_rank)
    )
    rectangles_by_name: Dict[str, List[Tuple[int, Rectangle]]] = defaultdict(list)
    owner_numel: Dict[int, int] = defaultdict(int)

    for rank in sorted(global_sources):
        global_layers = reconstruction_global_by_rank[rank]
        local_layers = reconstruction_local_by_rank[rank]
        if set(global_layers) != set(local_layers):
            raise ValueError(
                f"reconstruction metadata names differ for rank {rank}: "
                f"global={sorted(global_layers)}, local={sorted(local_layers)}"
            )
        for name, global_rectangles in global_layers.items():
            if name not in reference_state or not torch.is_tensor(reference_state[name]):
                raise ValueError(f"reconstruction metadata references unknown tensor {name}")
            tensor = reference_state[name]
            if tensor.ndim not in (1, 2):
                raise ValueError(
                    f"reconstruction metadata references unsupported tensor {name}: {tuple(tensor.shape)}"
                )
            local_rectangles = local_layers[name]
            if len(local_rectangles) != len(global_rectangles):
                raise ValueError(f"reconstruction metadata count mismatch for {name}/rank{rank}")
            local_state = local_states.get(rank)
            if local_state is not None:
                if name not in local_state:
                    raise ValueError(f"rank {rank} local state has no tensor {name}")
                local_tensor = local_state[name]
                if not torch.is_tensor(local_tensor) or local_tensor.ndim != tensor.ndim:
                    raise ValueError(f"rank {rank} local tensor {name} has incompatible rank")
            else:
                local_tensor = None

            for index, (global_rectangle, local_rectangle) in enumerate(
                zip(global_rectangles, local_rectangles)
            ):
                global_bounds = _rectangle_bounds(
                    global_rectangle,
                    tensor.shape,
                    f"{name}/rank{rank}/global{index}",
                )
                if local_tensor is not None:
                    _rectangle_bounds(
                        local_rectangle,
                        local_tensor.shape,
                        f"{name}/rank{rank}/local{index}",
                    )
                elif any(int(value) < 0 for point in local_rectangle for value in point):
                    raise ValueError(f"{name}/rank{rank}/local{index}: negative local coordinate")
                if _rectangle_shape(global_rectangle, tensor.ndim) != _rectangle_shape(
                    local_rectangle, tensor.ndim
                ):
                    raise ValueError(
                        f"reconstruction local/global shape mismatch for {name}/rank{rank}/rectangle{index}"
                    )
                rectangles_by_name[name].append((rank, global_rectangle))
                owner_numel[rank] += _rectangle_numel(global_bounds, tensor.ndim)

    if required_names is None:
        required = [
            name
            for name, tensor in reference_state.items()
            if torch.is_tensor(tensor) and tensor.ndim in (1, 2)
        ]
    else:
        required = list(required_names)
    unknown_required = sorted(set(required).difference(reference_state))
    if unknown_required:
        raise ValueError(f"required reconstruction tensors are absent: {unknown_required[:5]}")

    validated: Dict[str, Dict[int, int]] = {}
    for name, rectangles in rectangles_by_name.items():
        validated[name] = _validate_partition(name, reference_state[name].shape, rectangles)

    alias_lookup = _alias_lookup(reference_state)
    alias_sources: Dict[str, str] = {}
    for name in required:
        tensor = reference_state[name]
        if not torch.is_tensor(tensor) or tensor.ndim not in (1, 2):
            continue
        if name in validated:
            continue
        source = next(
            (candidate for candidate in alias_lookup.get(name, ()) if candidate in validated),
            None,
        )
        if source is None:
            raise ValueError(f"reconstruction metadata leaves required tensor {name} uncovered")
        alias_sources[name] = source

    # Report all omitted aliases, even when only the canonical named parameter
    # was listed in ``required_names``.  This makes tied-head handling explicit.
    for name, group in alias_lookup.items():
        if name in validated:
            continue
        source = next((candidate for candidate in group if candidate in validated), None)
        if source is not None:
            alias_sources[name] = source

    return ReconstructionAudit(
        covered_names=tuple(name for name in reference_state if name in validated),
        alias_sources=alias_sources,
        owner_numel_by_rank=dict(sorted(owner_numel.items())),
    )


def _normalize_mask_axis(
    indices: torch.Tensor,
    dimension: int,
    label: str,
) -> torch.Tensor:
    result = indices.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    if result.numel() == 0:
        raise ValueError(f"{label}: empty mask axis")
    if int(result.min()) < 0 or int(result.max()) >= dimension:
        raise ValueError(f"{label}: mask index lies outside [0, {dimension})")
    if torch.unique(result).numel() != result.numel():
        raise ValueError(f"{label}: mask contains duplicate indices")
    return result


def _mask_axes(mask: TensorMask, shape: torch.Size, label: str):
    if len(shape) == 1:
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"{label}: expected vector mask")
        return _normalize_mask_axis(mask, int(shape[0]), label), None
    if len(shape) != 2 or not isinstance(mask, tuple) or len(mask) != 2:
        raise TypeError(f"{label}: expected matrix mask")
    return (
        _normalize_mask_axis(mask[0], int(shape[0]), label + ".rows"),
        _normalize_mask_axis(mask[1], int(shape[1]), label + ".columns"),
    )


def _common_positions(left: torch.Tensor, right: torch.Tensor):
    if torch.equal(left, right):
        positions = torch.arange(left.numel(), dtype=torch.long)
        return positions, positions
    right_positions = {int(value): index for index, value in enumerate(right.tolist())}
    left_result: List[int] = []
    right_result: List[int] = []
    for left_index, value in enumerate(left.tolist()):
        right_index = right_positions.get(int(value))
        if right_index is not None:
            left_result.append(left_index)
            right_result.append(right_index)
    return torch.tensor(left_result, dtype=torch.long), torch.tensor(right_result, dtype=torch.long)


def _selected_overlap(
    tensor: torch.Tensor,
    row_positions: torch.Tensor,
    column_positions: Optional[torch.Tensor],
) -> torch.Tensor:
    selected = tensor.index_select(0, row_positions.to(tensor.device))
    if column_positions is not None:
        selected = selected.index_select(1, column_positions.to(tensor.device))
    return selected


def _values_match(left: torch.Tensor, right: torch.Tensor, atol: float, rtol: float) -> bool:
    if left.dtype != right.dtype:
        right = right.to(dtype=left.dtype)
    if left.is_floating_point() or left.is_complex():
        return bool(torch.allclose(left, right, atol=atol, rtol=rtol, equal_nan=True))
    return bool(torch.equal(left, right))


def validate_synchronized_overlaps_cpu(
    reference_state: Mapping[str, torch.Tensor],
    local_states: LocalStates,
    masks: Sequence[Mapping[str, TensorMask]],
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    """Require every multiply-owned parameter coordinate to have one value."""

    states = _rank_mapping(local_states)
    if set(states) != set(range(len(masks))):
        raise ValueError("local_states and masks must contain the same contiguous ranks")
    for rank, state in states.items():
        for tensor in state.values():
            if torch.is_tensor(tensor) and tensor.device.type != "cpu":
                raise ValueError(f"rank {rank} local state is not entirely on CPU")

    aliases = exact_state_dict_alias_groups(reference_state)
    for rank, state in states.items():
        for group in aliases:
            present = [name for name in group if name in state]
            if len(present) < 2:
                continue
            first = state[present[0]]
            for name in present[1:]:
                if not _values_match(first, state[name], atol, rtol):
                    raise ValueError(
                        f"rank {rank} tied aliases {present[0]} and {name} have different values"
                    )

    for name, reference in reference_state.items():
        if not torch.is_tensor(reference) or reference.ndim not in (1, 2):
            continue
        axes = []
        for rank, mask in enumerate(masks):
            if name not in mask or name not in states[rank]:
                raise ValueError(f"rank {rank} has no reconstruction data for {name}")
            row_axis, column_axis = _mask_axes(
                mask[name], reference.shape, f"{name}/rank{rank}"
            )
            expected_shape = (
                (row_axis.numel(),)
                if column_axis is None
                else (row_axis.numel(), column_axis.numel())
            )
            if tuple(states[rank][name].shape) != tuple(expected_shape):
                raise ValueError(
                    f"rank {rank} local tensor {name} has shape {tuple(states[rank][name].shape)}, "
                    f"expected {tuple(expected_shape)}"
                )
            axes.append((row_axis, column_axis))

        for left_rank in range(len(masks)):
            left_rows, left_columns = axes[left_rank]
            for right_rank in range(left_rank + 1, len(masks)):
                right_rows, right_columns = axes[right_rank]
                left_row_positions, right_row_positions = _common_positions(
                    left_rows, right_rows
                )
                if left_row_positions.numel() == 0:
                    continue
                if left_columns is None:
                    left_column_positions = right_column_positions = None
                else:
                    assert right_columns is not None
                    left_column_positions, right_column_positions = _common_positions(
                        left_columns, right_columns
                    )
                    if left_column_positions.numel() == 0:
                        continue
                left_values = _selected_overlap(
                    states[left_rank][name], left_row_positions, left_column_positions
                )
                right_values = _selected_overlap(
                    states[right_rank][name], right_row_positions, right_column_positions
                )
                if not _values_match(left_values, right_values, atol, rtol):
                    raise ValueError(
                        f"unsynchronized overlap for {name} between ranks "
                        f"{left_rank} and {right_rank}"
                    )


def reconstruct_full_state_dict_cpu(
    reference_state: Mapping[str, torch.Tensor],
    local_states: LocalStates,
    reconstruction_global_by_rank: RankRectangles,
    reconstruction_local_by_rank: RankRectangles,
    *,
    masks: Optional[Sequence[Mapping[str, TensorMask]]] = None,
    required_names: Optional[Iterable[str]] = None,
    allowed_sources: Optional[Iterable[int]] = None,
    overlap_atol: float = 0.0,
    overlap_rtol: float = 0.0,
) -> OrderedDict:
    """Reconstruct a strict-loadable full state on CPU from owner rectangles."""

    states = _rank_mapping(local_states)
    for name, tensor in reference_state.items():
        if torch.is_tensor(tensor) and tensor.device.type != "cpu":
            raise ValueError(f"reference tensor {name} is not on CPU")
    for rank, state in states.items():
        for name, tensor in state.items():
            if torch.is_tensor(tensor) and tensor.device.type != "cpu":
                raise ValueError(f"rank {rank} local tensor {name} is not on CPU")

    audit = validate_reconstruction_metadata(
        reference_state,
        reconstruction_global_by_rank,
        reconstruction_local_by_rank,
        required_names=required_names,
        allowed_sources=allowed_sources,
        local_states_by_rank=states,
    )
    if masks is not None:
        validate_synchronized_overlaps_cpu(
            reference_state,
            states,
            masks,
            atol=overlap_atol,
            rtol=overlap_rtol,
        )

    reconstructed: Dict[str, torch.Tensor] = {}
    for name in audit.covered_names:
        destination = torch.empty_like(reference_state[name], device="cpu")
        for rank, global_layers in reconstruction_global_by_rank.items():
            if name not in global_layers:
                continue
            if int(rank) not in states:
                raise ValueError(f"missing local state for reconstruction owner rank {rank}")
            local_layers = reconstruction_local_by_rank[int(rank)]
            for global_rectangle, local_rectangle in zip(
                global_layers[name], local_layers[name]
            ):
                _rectangle_view(destination, global_rectangle).copy_(
                    _rectangle_view(states[int(rank)][name], local_rectangle).to(
                        dtype=destination.dtype
                    )
                )
        reconstructed[name] = destination

    alias_lookup = _alias_lookup(reference_state)
    for group in exact_state_dict_alias_groups(reference_state):
        covered = [name for name in group if name in reconstructed]
        if not covered:
            continue
        canonical = covered[0]
        for name in covered[1:]:
            if not _values_match(
                reconstructed[canonical], reconstructed[name], overlap_atol, overlap_rtol
            ):
                raise ValueError(
                    f"reconstructed tied aliases {canonical} and {name} have different values"
                )
        for name in group:
            reconstructed[name] = reconstructed[canonical]

    result = OrderedDict()
    for name, reference in reference_state.items():
        if name in reconstructed:
            result[name] = reconstructed[name]
        elif name in audit.alias_sources:
            result[name] = reconstructed[audit.alias_sources[name]]
        elif torch.is_tensor(reference):
            # Persistent non-parameter buffers are replicated in Helix.  Prefer
            # rank zero's current value when its shape matches the full schema.
            rank_zero = states.get(0, {}).get(name)
            if torch.is_tensor(rank_zero) and rank_zero.shape == reference.shape:
                result[name] = rank_zero.detach().clone()
            else:
                result[name] = reference.detach().clone()
        else:
            result[name] = copy.deepcopy(reference)

    # Preserve exact tied aliases in the returned mapping even when aliases
    # were both explicitly present in reconstruction metadata.
    for name, group in alias_lookup.items():
        canonical = next((candidate for candidate in group if candidate in result), None)
        if canonical is not None:
            result[name] = result[canonical]
    return result
