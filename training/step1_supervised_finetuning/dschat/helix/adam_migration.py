"""Reference implementation of lossless Helix Adam-state migration.

Dynamic submodel resizing changes the global coordinates represented by a
rank-local dense parameter.  Migrating tensors by their local offsets is
therefore incorrect: local offset zero can refer to a different attention
head or FFN column after resizing.  This module deliberately performs the
migration in two phases:

1. reconstruct every model/optimizer tensor in canonical global coordinates;
2. slice the reconstructed tensors with the new structured masks.

The implementation is CPU based.  It is useful both as an executable
correctness oracle and as the packing/unpacking layer for a distributed
implementation.  Communication and synchronization are intentionally left to
the caller.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from .masking import TensorMask


ScalarState = Union[int, float, torch.Tensor]


@dataclass
class NamedAdamParameterState:
    """The state that must survive replacement of one local parameter.

    ``master_parameter`` is the FP32 optimizer parameter used by DeepSpeed's
    FP16 optimizer.  Omitting it is safe only when the optimizer updates the
    model parameter directly (for example, ordinary FP32 AdamW).
    """

    parameter: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    step: ScalarState
    master_parameter: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class StateTransfer:
    """Coordinates newly acquired by a destination rank.

    The three index tensors have equal length.  They map a canonical global
    flattened coordinate to its old owner-local and new destination-local
    flattened coordinates, respectively.
    """

    name: str
    source_rank: int
    destination_rank: int
    global_linear_indices: torch.Tensor
    source_local_linear_indices: torch.Tensor
    destination_local_linear_indices: torch.Tensor


@dataclass
class AdamMigrationResult:
    """Migrated local states plus the canonical snapshot and transfer proof."""

    local_states: List[OrderedDict[str, NamedAdamParameterState]]
    global_states: OrderedDict[str, NamedAdamParameterState]
    source_rank_by_parameter: OrderedDict[str, torch.Tensor]
    transfers: Tuple[StateTransfer, ...]


def _normalized_axes(
    mask: TensorMask,
    shape: torch.Size,
    label: str,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if len(shape) not in (1, 2):
        raise ValueError(f"{label}: only 1D/2D tensors are supported, got {tuple(shape)}")

    if len(shape) == 1:
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"{label}: expected a vector mask")
        axes = (mask.detach().to(device="cpu", dtype=torch.long).reshape(-1), None)
    else:
        if not isinstance(mask, tuple) or len(mask) != 2:
            raise TypeError(f"{label}: expected a matrix mask")
        axes = (
            mask[0].detach().to(device="cpu", dtype=torch.long).reshape(-1),
            mask[1].detach().to(device="cpu", dtype=torch.long).reshape(-1),
        )

    dimensions = (int(shape[0]),) if len(shape) == 1 else (int(shape[0]), int(shape[1]))
    for axis, dimension in zip(axes, dimensions):
        assert axis is not None
        if axis.numel() == 0:
            raise ValueError(f"{label}: mask axes cannot be empty")
        if int(axis.min()) < 0 or int(axis.max()) >= dimension:
            raise ValueError(f"{label}: mask index lies outside [0, {dimension})")
        if torch.unique(axis).numel() != axis.numel():
            raise ValueError(f"{label}: mask contains duplicate indices")
    return axes


def mask_linear_indices(mask: TensorMask, shape: torch.Size, label: str = "mask") -> torch.Tensor:
    """Return canonical flattened indices in rank-local flattening order."""

    rows, columns = _normalized_axes(mask, shape, label)
    if len(shape) == 1:
        return rows.clone()
    assert columns is not None
    return (rows[:, None] * int(shape[1]) + columns[None, :]).reshape(-1)


def mask_local_shape(mask: TensorMask, shape: torch.Size, label: str = "mask") -> torch.Size:
    rows, columns = _normalized_axes(mask, shape, label)
    if len(shape) == 1:
        return torch.Size((rows.numel(),))
    assert columns is not None
    return torch.Size((rows.numel(), columns.numel()))


def assert_complete_mask_coverage(
    reference_parameters: Mapping[str, torch.Tensor],
    masks: Sequence[Mapping[str, TensorMask]],
    *,
    label: str,
) -> None:
    """Reject a transition whose collective masks omit a global coordinate."""

    if not masks:
        raise ValueError(f"{label}: at least one rank mask is required")
    for name, reference in reference_parameters.items():
        covered = torch.zeros(reference.numel(), dtype=torch.bool)
        for rank, rank_mask in enumerate(masks):
            if name not in rank_mask:
                raise ValueError(f"{label}: rank {rank} has no mask for {name}")
            indices = mask_linear_indices(
                rank_mask[name], reference.shape, f"{label}/{name}/rank{rank}"
            )
            covered[indices] = True
        if not bool(torch.all(covered)):
            missing = int(torch.nonzero(~covered, as_tuple=False)[0])
            raise ValueError(
                f"{label}: structured masks do not cover {name} global linear index {missing}"
            )


def _clone_step(step: ScalarState) -> ScalarState:
    if torch.is_tensor(step):
        if step.numel() != 1:
            raise ValueError(f"Adam step must be scalar, got shape {tuple(step.shape)}")
        return step.detach().cpu().clone()
    if not isinstance(step, (int, float)):
        raise TypeError(f"Adam step must be a number or scalar tensor, got {type(step)!r}")
    return copy.deepcopy(step)


def _step_value(step: ScalarState) -> Union[int, float]:
    cloned = _clone_step(step)
    return cloned.item() if torch.is_tensor(cloned) else cloned


def _slice_global_tensor(
    tensor: torch.Tensor,
    mask: TensorMask,
    label: str,
) -> torch.Tensor:
    indices = mask_linear_indices(mask, tensor.shape, label)
    local_shape = mask_local_shape(mask, tensor.shape, label)
    return tensor.reshape(-1).index_select(0, indices).reshape(local_shape).clone()


def _reconstruct_tensor_field(
    *,
    name: str,
    field: str,
    shape: torch.Size,
    old_masks: Sequence[Mapping[str, TensorMask]],
    old_states: Sequence[Mapping[str, NamedAdamParameterState]],
    source_rank: torch.Tensor,
    rtol: float,
    atol: float,
) -> torch.Tensor:
    first = getattr(old_states[0][name], field)
    if first is None:
        raise ValueError(f"{name}: field {field} is unexpectedly absent")
    assert torch.is_tensor(first)
    dtype = first.dtype
    reconstructed = torch.empty(int(torch.tensor(shape).prod()), dtype=dtype)
    covered = torch.zeros(reconstructed.numel(), dtype=torch.bool)

    for rank, (rank_mask, rank_states) in enumerate(zip(old_masks, old_states)):
        local = getattr(rank_states[name], field)
        if local is None or not torch.is_tensor(local):
            raise ValueError(f"{name}/rank{rank}: field {field} is absent")
        if local.dtype != dtype:
            raise ValueError(
                f"{name}/{field}: dtype differs across ranks ({dtype} vs {local.dtype})"
            )
        expected = mask_local_shape(
            rank_mask[name], shape, f"old/{name}/{field}/rank{rank}"
        )
        if local.shape != expected:
            raise ValueError(
                f"{name}/{field}/rank{rank}: local shape {tuple(local.shape)} "
                f"does not match mask shape {tuple(expected)}"
            )

        indices = mask_linear_indices(
            rank_mask[name], shape, f"old/{name}/{field}/rank{rank}"
        )
        local_flat = local.detach().cpu().reshape(-1)
        already_seen = covered.index_select(0, indices)
        if bool(torch.any(already_seen)):
            overlap_indices = indices[already_seen]
            expected_values = reconstructed.index_select(0, overlap_indices)
            actual_values = local_flat[already_seen]
            if not torch.allclose(
                expected_values,
                actual_values,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            ):
                raise ValueError(
                    f"{name}/{field}: rank {rank} disagrees with an old owner "
                    "on an overlapping global coordinate"
                )

        unseen = ~already_seen
        if bool(torch.any(unseen)):
            new_indices = indices[unseen]
            reconstructed[new_indices] = local_flat[unseen]
            covered[new_indices] = True
            if field == "parameter":
                source_rank[new_indices] = rank

    if not bool(torch.all(covered)):
        missing = int(torch.nonzero(~covered, as_tuple=False)[0])
        raise ValueError(f"old masks do not cover {name}/{field} global linear index {missing}")
    return reconstructed.reshape(shape)


def _build_transfers(
    reference_parameters: Mapping[str, torch.Tensor],
    old_masks: Sequence[Mapping[str, TensorMask]],
    new_masks: Sequence[Mapping[str, TensorMask]],
    source_ranks: Mapping[str, torch.Tensor],
) -> Tuple[StateTransfer, ...]:
    transfers: List[StateTransfer] = []
    for name, reference in reference_parameters.items():
        total = reference.numel()
        old_indices_by_rank = [
            mask_linear_indices(mask[name], reference.shape, f"old/{name}/rank{rank}")
            for rank, mask in enumerate(old_masks)
        ]
        old_membership = []
        old_inverse = []
        for indices in old_indices_by_rank:
            membership = torch.zeros(total, dtype=torch.bool)
            membership[indices] = True
            old_membership.append(membership)
            inverse = torch.full((total,), -1, dtype=torch.long)
            inverse[indices] = torch.arange(indices.numel(), dtype=torch.long)
            old_inverse.append(inverse)

        owners = source_ranks[name].reshape(-1)
        for destination, new_mask in enumerate(new_masks):
            new_indices = mask_linear_indices(
                new_mask[name], reference.shape, f"new/{name}/rank{destination}"
            )
            added_positions = torch.nonzero(
                ~old_membership[destination].index_select(0, new_indices),
                as_tuple=False,
            ).reshape(-1)
            if added_positions.numel() == 0:
                continue
            added_globals = new_indices.index_select(0, added_positions)
            added_owners = owners.index_select(0, added_globals)
            for source in torch.unique(added_owners, sorted=True).tolist():
                selected = added_owners == source
                global_indices = added_globals[selected]
                source_local = old_inverse[source].index_select(0, global_indices)
                if bool(torch.any(source_local < 0)):
                    raise RuntimeError(
                        f"internal owner error for {name}: rank {source} does not hold a selected coordinate"
                    )
                transfers.append(
                    StateTransfer(
                        name=name,
                        source_rank=int(source),
                        destination_rank=destination,
                        global_linear_indices=global_indices.clone(),
                        source_local_linear_indices=source_local.clone(),
                        destination_local_linear_indices=added_positions[selected].clone(),
                    )
                )
    return tuple(transfers)


def migrate_named_adam_states(
    reference_parameters: Mapping[str, torch.Tensor],
    old_masks: Sequence[Mapping[str, TensorMask]],
    new_masks: Sequence[Mapping[str, TensorMask]],
    old_states: Sequence[Mapping[str, NamedAdamParameterState]],
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> AdamMigrationResult:
    """Losslessly migrate named Adam states between structured masks.

    All rank-local states are reconstructed in canonical coordinates before
    any new local state is produced.  This guarantees that a region removed
    from one rank remains available to any new owner.  Replica disagreement,
    missing coverage, inconsistent steps, and missing FP32 master tensors are
    reported instead of being silently repaired.
    """

    if len(old_masks) != len(old_states):
        raise ValueError("old_masks and old_states must have one entry per old rank")
    if len(new_masks) != len(old_masks):
        raise ValueError("online resizing currently requires a fixed world size")

    assert_complete_mask_coverage(reference_parameters, old_masks, label="old")
    assert_complete_mask_coverage(reference_parameters, new_masks, label="new")

    for rank, rank_states in enumerate(old_states):
        missing = set(reference_parameters).difference(rank_states)
        if missing:
            raise ValueError(f"old state rank {rank} is missing parameters: {sorted(missing)}")

    global_states: OrderedDict[str, NamedAdamParameterState] = OrderedDict()
    source_ranks: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, reference in reference_parameters.items():
        master_presence = [old_states[rank][name].master_parameter is not None for rank in range(len(old_states))]
        if any(master_presence) and not all(master_presence):
            raise ValueError(f"{name}: FP32 master parameter is missing on some ranks")

        expected_step = _step_value(old_states[0][name].step)
        for rank in range(1, len(old_states)):
            actual_step = _step_value(old_states[rank][name].step)
            if actual_step != expected_step:
                raise ValueError(
                    f"{name}: Adam step differs across ranks ({expected_step} vs {actual_step})"
                )

        owners = torch.full((reference.numel(),), -1, dtype=torch.long)
        fields: Dict[str, torch.Tensor] = {}
        field_names = ["parameter", "exp_avg", "exp_avg_sq"]
        if all(master_presence):
            field_names.append("master_parameter")
        for field in field_names:
            fields[field] = _reconstruct_tensor_field(
                name=name,
                field=field,
                shape=reference.shape,
                old_masks=old_masks,
                old_states=old_states,
                source_rank=owners,
                rtol=rtol,
                atol=atol,
            )
        if bool(torch.any(owners < 0)):
            raise RuntimeError(f"internal owner construction failed for {name}")

        source_ranks[name] = owners.reshape(reference.shape)
        global_states[name] = NamedAdamParameterState(
            parameter=fields["parameter"],
            exp_avg=fields["exp_avg"],
            exp_avg_sq=fields["exp_avg_sq"],
            step=_clone_step(old_states[0][name].step),
            master_parameter=fields.get("master_parameter"),
        )

    local_states: List[OrderedDict[str, NamedAdamParameterState]] = []
    for rank, rank_mask in enumerate(new_masks):
        local: OrderedDict[str, NamedAdamParameterState] = OrderedDict()
        for name, global_state in global_states.items():
            local[name] = NamedAdamParameterState(
                parameter=_slice_global_tensor(
                    global_state.parameter, rank_mask[name], f"new/{name}/parameter/rank{rank}"
                ),
                exp_avg=_slice_global_tensor(
                    global_state.exp_avg, rank_mask[name], f"new/{name}/exp_avg/rank{rank}"
                ),
                exp_avg_sq=_slice_global_tensor(
                    global_state.exp_avg_sq, rank_mask[name], f"new/{name}/exp_avg_sq/rank{rank}"
                ),
                step=_clone_step(global_state.step),
                master_parameter=(
                    _slice_global_tensor(
                        global_state.master_parameter,
                        rank_mask[name],
                        f"new/{name}/master_parameter/rank{rank}",
                    )
                    if global_state.master_parameter is not None
                    else None
                ),
            )
        local_states.append(local)

    transfers = _build_transfers(
        reference_parameters,
        old_masks,
        new_masks,
        source_ranks,
    )
    return AdamMigrationResult(
        local_states=local_states,
        global_states=global_states,
        source_rank_by_parameter=source_ranks,
        transfers=transfers,
    )
