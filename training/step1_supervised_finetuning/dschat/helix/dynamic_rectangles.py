"""Compact old-mask to new-mask copy plans for live Helix resizing.

Metadata is expressed as axis-aligned rectangles.  It therefore scales with
mask membership runs rather than parameter ``numel``; a 394M-element embedding
does not allocate a multi-gigabyte int64 index tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import torch

from .communication import (
    Rectangle,
    _coverage_runs,
    _global_to_local_rectangle,
    _parameter_axes,
)
from .masking import TensorMask


@dataclass(frozen=True)
class StateCopyRectangle:
    name: str
    source_rank: int
    destination_rank: int
    global_rectangle: Rectangle
    source_local_rectangle: Rectangle
    destination_local_rectangle: Rectangle


@dataclass(frozen=True)
class DynamicCopyPlan:
    world_size: int
    copies: Tuple[StateCopyRectangle, ...]

    def copies_for_destination(self, rank: int) -> Tuple[StateCopyRectangle, ...]:
        return tuple(item for item in self.copies if item.destination_rank == rank)


def _joint_runs(
    old_selections: Sequence[torch.Tensor],
    new_selections: Sequence[torch.Tensor],
    dimension: int,
):
    # Validate each generation independently before making a combined
    # membership signature.  Combined coverage alone could hide an old/new gap.
    _coverage_runs(old_selections, dimension)
    _coverage_runs(new_selections, dimension)
    world_size = len(old_selections)
    result = []
    for start, end, members in _coverage_runs(
        list(old_selections) + list(new_selections),
        dimension,
    ):
        old_members = tuple(rank for rank in members if rank < world_size)
        new_members = tuple(rank - world_size for rank in members if rank >= world_size)
        if not old_members or not new_members:
            raise ValueError(
                f"dynamic masks leave [{start}, {end}] uncovered in one generation"
            )
        result.append((start, end, old_members, new_members))
    return result


def _append_copies(
    destination: List[StateCopyRectangle],
    *,
    name: str,
    rectangle: Rectangle,
    ndim: int,
    old_owners: Tuple[int, ...],
    new_owners: Tuple[int, ...],
    old_rows: Sequence[torch.Tensor],
    old_columns: Optional[Sequence[torch.Tensor]],
    new_rows: Sequence[torch.Tensor],
    new_columns: Optional[Sequence[torch.Tensor]],
) -> None:
    if not old_owners or not new_owners:
        raise ValueError(f"{name}: a dynamic rectangle has no old or new owner")
    for target in new_owners:
        # Prefer the destination's own old replica. Besides reducing network
        # traffic, this preserves locality even when its dense local offset
        # changed because an earlier rank resized.
        source = target if target in old_owners else min(old_owners)
        source_local = _global_to_local_rectangle(
            rectangle,
            ndim,
            old_rows[source],
            old_columns[source] if old_columns is not None else None,
            f"dynamic-old/{name}/rank{source}",
        )
        target_local = _global_to_local_rectangle(
            rectangle,
            ndim,
            new_rows[target],
            new_columns[target] if new_columns is not None else None,
            f"dynamic-new/{name}/rank{target}",
        )
        destination.append(
            StateCopyRectangle(
                name=name,
                source_rank=source,
                destination_rank=target,
                global_rectangle=rectangle,
                source_local_rectangle=source_local,
                destination_local_rectangle=target_local,
            )
        )


def build_dynamic_copy_plan(
    state_dict: Mapping[str, torch.Tensor],
    old_masks: Sequence[Mapping[str, TensorMask]],
    new_masks: Sequence[Mapping[str, TensorMask]],
    parameter_names: Sequence[str],
) -> DynamicCopyPlan:
    """Cover every new local coordinate from a deterministic old owner."""

    if not old_masks or len(old_masks) != len(new_masks):
        raise ValueError("dynamic resizing requires equal, non-zero old/new world sizes")
    world_size = len(old_masks)
    copies: List[StateCopyRectangle] = []
    selected = set(parameter_names)

    for name, tensor in state_dict.items():
        if name not in selected:
            continue
        if tensor.ndim not in (1, 2):
            raise ValueError(f"dynamic Adam migration only supports 1D/2D parameter {name}")
        old_rows, old_columns = _parameter_axes(old_masks, name, tensor.shape)
        new_rows, new_columns = _parameter_axes(new_masks, name, tensor.shape)
        row_runs = _joint_runs(old_rows, new_rows, tensor.shape[0])

        if tensor.ndim == 1:
            for start, end, old_owners, new_owners in row_runs:
                _append_copies(
                    copies,
                    name=name,
                    rectangle=([0, start], [0, end]),
                    ndim=1,
                    old_owners=old_owners,
                    new_owners=new_owners,
                    old_rows=old_rows,
                    old_columns=None,
                    new_rows=new_rows,
                    new_columns=None,
                )
            continue

        assert old_columns is not None and new_columns is not None
        column_runs = _joint_runs(old_columns, new_columns, tensor.shape[1])
        for row_start, row_end, old_row_owners, new_row_owners in row_runs:
            for col_start, col_end, old_col_owners, new_col_owners in column_runs:
                old_owners = tuple(sorted(set(old_row_owners).intersection(old_col_owners)))
                new_owners = tuple(sorted(set(new_row_owners).intersection(new_col_owners)))
                if not old_owners:
                    raise ValueError(
                        f"old masks leave {name}[{row_start}:{row_end + 1}, "
                        f"{col_start}:{col_end + 1}] uncovered"
                    )
                if not new_owners:
                    raise ValueError(
                        f"new masks leave {name}[{row_start}:{row_end + 1}, "
                        f"{col_start}:{col_end + 1}] uncovered"
                    )
                _append_copies(
                    copies,
                    name=name,
                    rectangle=([row_start, col_start], [row_end, col_end]),
                    ndim=2,
                    old_owners=old_owners,
                    new_owners=new_owners,
                    old_rows=old_rows,
                    old_columns=old_columns,
                    new_rows=new_rows,
                    new_columns=new_columns,
                )

    return DynamicCopyPlan(world_size=world_size, copies=tuple(copies))
