"""Complete old-local to new-local copy plans for Helix state migration.

Unlike an inter-rank transfer list, a complete copy plan also includes values
that remain on the same rank.  This distinction matters because changing one
submodel size advances the cumulative circular cursor and can move a retained
global coordinate to a different rank-local offset.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import torch

from .adam_migration import (
    assert_complete_mask_coverage,
    mask_linear_indices,
    mask_local_shape,
)
from .masking import TensorMask


@dataclass(frozen=True)
class StateCopy:
    """One vectorized copy from an old local tensor to a new local tensor."""

    name: str
    source_rank: int
    destination_rank: int
    global_linear_indices: torch.Tensor
    source_local_linear_indices: torch.Tensor
    destination_local_linear_indices: torch.Tensor


def _first_old_owners(
    total: int,
    old_indices_by_rank: Sequence[torch.Tensor],
) -> torch.Tensor:
    owners = torch.full((total,), -1, dtype=torch.long)
    for rank, indices in enumerate(old_indices_by_rank):
        unseen = owners.index_select(0, indices) < 0
        owners[indices[unseen]] = rank
    if bool(torch.any(owners < 0)):
        missing = int(torch.nonzero(owners < 0, as_tuple=False)[0])
        raise ValueError(f"old masks do not cover global linear index {missing}")
    return owners


def build_complete_state_copy_plan(
    reference_parameters: Mapping[str, torch.Tensor],
    old_masks: Sequence[Mapping[str, TensorMask]],
    new_masks: Sequence[Mapping[str, TensorMask]],
    *,
    source_rank_by_parameter: Optional[Mapping[str, torch.Tensor]] = None,
) -> Tuple[StateCopy, ...]:
    """Map every new local coordinate to exactly one old local coordinate.

    A destination rank is its own preferred source whenever it already holds
    the coordinate.  Otherwise, the canonical snapshot owner is used when
    supplied; without one, the lowest old rank containing the coordinate is
    selected.  Consequently, only records whose source and destination differ
    require communication, while self-copy records perform local reordering.
    """

    if len(old_masks) != len(new_masks):
        raise ValueError("online resizing currently requires a fixed world size")
    assert_complete_mask_coverage(reference_parameters, old_masks, label="old")
    assert_complete_mask_coverage(reference_parameters, new_masks, label="new")

    copies: List[StateCopy] = []
    for name, reference in reference_parameters.items():
        total = reference.numel()
        old_indices_by_rank = [
            mask_linear_indices(mask[name], reference.shape, f"old/{name}/rank{rank}")
            for rank, mask in enumerate(old_masks)
        ]
        old_inverse = []
        old_membership = []
        for indices in old_indices_by_rank:
            inverse = torch.full((total,), -1, dtype=torch.long)
            inverse[indices] = torch.arange(indices.numel(), dtype=torch.long)
            old_inverse.append(inverse)
            old_membership.append(inverse >= 0)

        owners = _first_old_owners(total, old_indices_by_rank)
        if source_rank_by_parameter is not None:
            if name not in source_rank_by_parameter:
                raise ValueError(f"source owner map is missing {name}")
            owners = source_rank_by_parameter[name].detach().cpu().reshape(-1).to(torch.long)
            if owners.numel() != total:
                raise ValueError(f"source owner map for {name} has the wrong shape")
            if int(owners.min()) < 0 or int(owners.max()) >= len(old_masks):
                raise ValueError(f"source owner map for {name} contains an invalid rank")
            coordinates = torch.arange(total, dtype=torch.long)
            for rank in torch.unique(owners, sorted=True).tolist():
                selected = owners == rank
                if bool(torch.any(~old_membership[rank].index_select(0, coordinates[selected]))):
                    raise ValueError(
                        f"source owner map for {name} assigns a coordinate absent from rank {rank}"
                    )

        for destination, new_mask in enumerate(new_masks):
            new_indices = mask_linear_indices(
                new_mask[name], reference.shape, f"new/{name}/rank{destination}"
            )
            destination_positions = torch.arange(new_indices.numel(), dtype=torch.long)
            chosen_sources = owners.index_select(0, new_indices).clone()
            retained = old_membership[destination].index_select(0, new_indices)
            chosen_sources[retained] = destination

            for source in torch.unique(chosen_sources, sorted=True).tolist():
                selected = chosen_sources == source
                global_indices = new_indices[selected]
                source_positions = old_inverse[source].index_select(0, global_indices)
                if bool(torch.any(source_positions < 0)):
                    raise RuntimeError(
                        f"copy-plan owner error for {name}: rank {source} lacks a selected coordinate"
                    )
                copies.append(
                    StateCopy(
                        name=name,
                        source_rank=int(source),
                        destination_rank=destination,
                        global_linear_indices=global_indices.clone(),
                        source_local_linear_indices=source_positions.clone(),
                        destination_local_linear_indices=destination_positions[selected].clone(),
                    )
                )
    return tuple(copies)


def execute_complete_state_copy_plan(
    reference_parameters: Mapping[str, torch.Tensor],
    new_masks: Sequence[Mapping[str, TensorMask]],
    old_local_tensors: Sequence[Mapping[str, torch.Tensor]],
    copies: Sequence[StateCopy],
) -> List[OrderedDict[str, torch.Tensor]]:
    """CPU oracle that applies one complete plan to a tensor-valued state slot."""

    if len(old_local_tensors) != len(new_masks):
        raise ValueError("old_local_tensors must have one entry per rank")

    result: List[OrderedDict[str, torch.Tensor]] = [OrderedDict() for _ in new_masks]
    filled = {}
    for destination, destination_mask in enumerate(new_masks):
        for name, reference in reference_parameters.items():
            prototype = old_local_tensors[0][name]
            shape = mask_local_shape(
                destination_mask[name], reference.shape, f"new/{name}/rank{destination}"
            )
            result[destination][name] = torch.empty(shape, dtype=prototype.dtype)
            filled[(destination, name)] = torch.zeros(result[destination][name].numel(), dtype=torch.bool)

    for copy_record in copies:
        if copy_record.name not in reference_parameters:
            raise ValueError(f"copy plan references unknown parameter {copy_record.name}")
        source = old_local_tensors[copy_record.source_rank][copy_record.name].detach().cpu().reshape(-1)
        destination = result[copy_record.destination_rank][copy_record.name].reshape(-1)
        source_indices = copy_record.source_local_linear_indices
        destination_indices = copy_record.destination_local_linear_indices
        if source_indices.numel() != destination_indices.numel():
            raise ValueError("copy plan has unequal source and destination lengths")
        destination_filled = filled[(copy_record.destination_rank, copy_record.name)]
        if bool(torch.any(destination_filled.index_select(0, destination_indices))):
            raise ValueError("copy plan writes a destination coordinate more than once")
        destination[destination_indices] = source.index_select(0, source_indices)
        destination_filled[destination_indices] = True

    for (destination, name), coverage in filled.items():
        if not bool(torch.all(coverage)):
            missing = int(torch.nonzero(~coverage, as_tuple=False)[0])
            raise ValueError(
                f"copy plan does not fill {name}/rank{destination} local linear index {missing}"
            )
    return result
