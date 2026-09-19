"""Exact Cluster-Reduce metadata and conflict-free scheduling.

The previous implementation projected masks onto a small 16x192 proxy grid
and manually selected a subset of rank groups. That can silently omit real
parameter regions. This module derives overlap rectangles directly from the
actual structured masks, validates complete coverage, and applies greedy
conflict-graph coloring as described in Section 3.2 of the paper.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch

from .masking import TensorMask


RankGroup = Tuple[int, ...]
Point = List[int]
Rectangle = Tuple[Point, Point]
LayerRectangles = OrderedDict[str, List[Rectangle]]
GroupRectangles = OrderedDict[RankGroup, LayerRectangles]


@dataclass
class CommunicationPlan:
    """All global and rank-local metadata needed by Cluster-Reduce."""

    groups_to_global_rectangles: GroupRectangles
    local_groups_to_rectangles: List[GroupRectangles]
    overlap_groups: List[RankGroup]
    clusters: List[List[RankGroup]]
    reconstruction_global_by_rank: Dict[int, LayerRectangles]
    reconstruction_local_by_rank: Dict[int, LayerRectangles]
    gather_ranks: List[int]

    def local_plan(self, rank: int) -> GroupRectangles:
        return self.local_groups_to_rectangles[rank]


def _normalize_indices(indices: torch.Tensor, dimension: int, name: str) -> torch.Tensor:
    indices = indices.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    if indices.numel() == 0:
        raise ValueError(f"{name} contains an empty selection")
    if int(indices.min()) < 0 or int(indices.max()) >= dimension:
        raise ValueError(f"{name} contains an index outside [0, {dimension})")
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError(f"{name} contains duplicate indices")
    return indices


def _parameter_axes(
    masks: Sequence[Mapping[str, TensorMask]],
    name: str,
    shape: torch.Size,
) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
    first_axis: List[torch.Tensor] = []
    second_axis: Optional[List[torch.Tensor]] = [] if len(shape) == 2 else None
    for rank, mask in enumerate(masks):
        if name not in mask:
            raise ValueError(f"rank {rank} has no mask for parameter {name}")
        indices = mask[name]
        if len(shape) == 1:
            if not isinstance(indices, torch.Tensor):
                raise TypeError(f"rank {rank} has a matrix mask for vector {name}")
            first_axis.append(_normalize_indices(indices, shape[0], f"{name}/rank{rank}"))
        elif len(shape) == 2:
            if not isinstance(indices, tuple) or len(indices) != 2:
                raise TypeError(f"rank {rank} has a vector mask for matrix {name}")
            first_axis.append(_normalize_indices(indices[0], shape[0], f"{name}.rows/rank{rank}"))
            assert second_axis is not None
            second_axis.append(_normalize_indices(indices[1], shape[1], f"{name}.cols/rank{rank}"))
        else:
            raise ValueError(f"Cluster-Reduce only supports 1D/2D parameters, got {name}: {shape}")
    return first_axis, second_axis


def _coverage_runs(selections: Sequence[torch.Tensor], dimension: int) -> List[Tuple[int, int, RankGroup]]:
    membership = torch.zeros((len(selections), dimension), dtype=torch.bool)
    for rank, indices in enumerate(selections):
        membership[rank, indices] = True
    powers = torch.tensor([1 << rank for rank in range(len(selections))], dtype=torch.long)
    signatures = (membership.to(torch.long) * powers[:, None]).sum(dim=0)
    if torch.any(signatures == 0):
        first_gap = int(torch.nonzero(signatures == 0, as_tuple=False)[0])
        raise ValueError(f"structured masks do not cover region index {first_gap}")

    boundaries = torch.nonzero(signatures[1:] != signatures[:-1], as_tuple=False).reshape(-1) + 1
    starts = [0] + boundaries.tolist()
    ends = (boundaries - 1).tolist() + [dimension - 1]
    runs: List[Tuple[int, int, RankGroup]] = []
    for start, end in zip(starts, ends):
        bitmask = int(signatures[start])
        ranks = tuple(rank for rank in range(len(selections)) if bitmask & (1 << rank))
        runs.append((int(start), int(end), ranks))
    return runs


def _rect_numel(rectangle: Rectangle, ndim: int) -> int:
    if ndim == 1:
        return rectangle[1][1] - rectangle[0][1] + 1
    return (
        (rectangle[1][0] - rectangle[0][0] + 1)
        * (rectangle[1][1] - rectangle[0][1] + 1)
    )


def _local_interval(indices: torch.Tensor, start: int, end: int, label: str) -> Tuple[int, int]:
    global_to_local = torch.full((int(indices.max()) + 1,), -1, dtype=torch.long)
    global_to_local[indices] = torch.arange(indices.numel(), dtype=torch.long)
    if end >= global_to_local.numel():
        raise ValueError(f"{label} is not retained locally")
    positions = global_to_local[start : end + 1]
    if torch.any(positions < 0):
        raise ValueError(f"{label} is only partially retained locally")
    expected = torch.arange(int(positions[0]), int(positions[0]) + positions.numel())
    if not torch.equal(positions, expected):
        raise ValueError(
            f"{label} is non-contiguous in local coordinates; canonical circular masks are required"
        )
    return int(positions[0]), int(positions[-1])


def _global_to_local_rectangle(
    rectangle: Rectangle,
    ndim: int,
    row_indices: torch.Tensor,
    column_indices: Optional[torch.Tensor],
    label: str,
) -> Rectangle:
    if ndim == 1:
        start, end = _local_interval(row_indices, rectangle[0][1], rectangle[1][1], label)
        return ([0, start], [0, end])
    row_start, row_end = _local_interval(row_indices, rectangle[0][0], rectangle[1][0], label + ".rows")
    assert column_indices is not None
    col_start, col_end = _local_interval(
        column_indices,
        rectangle[0][1],
        rectangle[1][1],
        label + ".cols",
    )
    return ([row_start, col_start], [row_end, col_end])


def greedy_color_groups(groups: Iterable[RankGroup]) -> List[List[RankGroup]]:
    """Color the subgroup conflict graph using stable largest-first ordering."""

    unique_groups = sorted(set(groups))
    neighbors: Dict[RankGroup, Set[RankGroup]] = {group: set() for group in unique_groups}
    for index, left in enumerate(unique_groups):
        left_ranks = set(left)
        for right in unique_groups[index + 1 :]:
            if left_ranks.intersection(right):
                neighbors[left].add(right)
                neighbors[right].add(left)

    order = sorted(unique_groups, key=lambda group: (-len(neighbors[group]), group))
    colors: Dict[RankGroup, int] = {}
    for group in order:
        unavailable = {colors[neighbor] for neighbor in neighbors[group] if neighbor in colors}
        color = 0
        while color in unavailable:
            color += 1
        colors[group] = color

    clusters: List[List[RankGroup]] = []
    for color in range(max(colors.values(), default=-1) + 1):
        clusters.append(sorted(group for group, assigned in colors.items() if assigned == color))
    return clusters


def _append_rectangle(
    destination: DefaultDict[RankGroup, DefaultDict[str, List[Rectangle]]],
    group: RankGroup,
    name: str,
    rectangle: Rectangle,
) -> None:
    destination[group][name].append(rectangle)


def _ordered_group_rectangles(
    raw: Mapping[RankGroup, Mapping[str, List[Rectangle]]],
    order: Sequence[RankGroup],
) -> GroupRectangles:
    result: GroupRectangles = OrderedDict()
    for group in order:
        if group not in raw:
            continue
        layers: LayerRectangles = OrderedDict()
        for name, rectangles in raw[group].items():
            layers[name] = list(rectangles)
        result[group] = layers
    return result


def build_communication_plan(
    state_dict: Mapping[str, torch.Tensor],
    masks: Sequence[Mapping[str, TensorMask]],
    parameter_names: Optional[Iterable[str]] = None,
) -> CommunicationPlan:
    """Derive exact overlap buckets, color clusters, and checkpoint owners."""

    if not masks:
        raise ValueError("At least one rank mask is required")
    world_size = len(masks)
    selected_names = set(parameter_names) if parameter_names is not None else set(masks[0])

    global_all: DefaultDict[RankGroup, DefaultDict[str, List[Rectangle]]] = defaultdict(
        lambda: defaultdict(list)
    )
    axes_by_name: Dict[str, Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]] = {}
    ndim_by_name: Dict[str, int] = {}

    for name, tensor in state_dict.items():
        if name not in selected_names or name not in masks[0] or not torch.is_tensor(tensor):
            continue
        if tensor.ndim not in (1, 2):
            continue
        row_indices, column_indices = _parameter_axes(masks, name, tensor.shape)
        axes_by_name[name] = (row_indices, column_indices)
        ndim_by_name[name] = tensor.ndim
        row_runs = _coverage_runs(row_indices, tensor.shape[0])

        if tensor.ndim == 1:
            for start, end, group in row_runs:
                _append_rectangle(global_all, group, name, ([0, start], [0, end]))
            continue

        assert column_indices is not None
        column_runs = _coverage_runs(column_indices, tensor.shape[1])
        for row_start, row_end, row_group in row_runs:
            row_ranks = set(row_group)
            for col_start, col_end, col_group in column_runs:
                group = tuple(sorted(row_ranks.intersection(col_group)))
                if not group:
                    raise ValueError(
                        f"structured masks leave {name}[{row_start}:{row_end + 1}, "
                        f"{col_start}:{col_end + 1}] uncovered"
                    )
                _append_rectangle(
                    global_all,
                    group,
                    name,
                    ([row_start, col_start], [row_end, col_end]),
                )

    overlap_groups = [group for group in global_all if len(group) > 1]
    clusters = greedy_color_groups(overlap_groups)
    communication_order = [group for cluster in clusters for group in cluster]
    groups_to_global = _ordered_group_rectangles(global_all, communication_order)

    local_by_rank: List[GroupRectangles] = []
    for rank in range(world_size):
        raw_local: DefaultDict[RankGroup, DefaultDict[str, List[Rectangle]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for group in communication_order:
            if rank not in group:
                continue
            for name, rectangles in global_all[group].items():
                rows, columns = axes_by_name[name]
                for rectangle in rectangles:
                    local = _global_to_local_rectangle(
                        rectangle,
                        ndim_by_name[name],
                        rows[rank],
                        columns[rank] if columns is not None else None,
                        f"{name}/rank{rank}",
                    )
                    _append_rectangle(raw_local, group, name, local)
        local_by_rank.append(_ordered_group_rectangles(raw_local, communication_order))

    # Assign every full-model region to exactly one source for checkpoint
    # reconstruction. Prefer rank 0 when it owns a region; otherwise balance
    # transfer volume across eligible ranks.
    owner_load = [0] * world_size
    reconstruction_global: Dict[int, LayerRectangles] = {
        rank: OrderedDict() for rank in range(world_size)
    }
    reconstruction_local: Dict[int, LayerRectangles] = {
        rank: OrderedDict() for rank in range(world_size)
    }
    for group in sorted(global_all):
        for name, rectangles in global_all[group].items():
            rows, columns = axes_by_name[name]
            for rectangle in rectangles:
                size = _rect_numel(rectangle, ndim_by_name[name])
                owner = 0 if 0 in group else min(group, key=lambda rank: (owner_load[rank], rank))
                owner_load[owner] += size
                reconstruction_global[owner].setdefault(name, []).append(rectangle)
                local = _global_to_local_rectangle(
                    rectangle,
                    ndim_by_name[name],
                    rows[owner],
                    columns[owner] if columns is not None else None,
                    f"{name}/owner{owner}",
                )
                reconstruction_local[owner].setdefault(name, []).append(local)

    reconstruction_global = {
        rank: layers for rank, layers in reconstruction_global.items() if layers
    }
    reconstruction_local = {
        rank: layers for rank, layers in reconstruction_local.items() if layers
    }
    gather_ranks = sorted(set(reconstruction_global).union({0}))

    return CommunicationPlan(
        groups_to_global_rectangles=groups_to_global,
        local_groups_to_rectangles=local_by_rank,
        overlap_groups=communication_order,
        clusters=clusters,
        reconstruction_global_by_rank=reconstruction_global,
        reconstruction_local_by_rank=reconstruction_local,
        gather_ranks=gather_ranks,
    )
