"""FP16_UnfusedOptimizer snapshot/restore for live Helix resizing."""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch

from .deepspeed_paper import _bounded_views
from .deepspeed_runtime import _rectangle_view
from .dynamic_rectangles import DynamicCopyPlan


@dataclass
class LocalFP16AdamParameter:
    master_parameter: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    step: torch.Tensor


@dataclass
class FP16AdamSnapshot:
    parameters: OrderedDict[str, LocalFP16AdamParameter]
    optimizer_groups: Tuple[Dict, ...]
    loss_scaler: Dict
    scheduler_state: Optional[Dict]
    engine_counters: Dict[str, int]
    cpu_rng_state: torch.Tensor
    cuda_rng_state: torch.Tensor


@dataclass
class LiveFP16AdamParameter:
    fp16_parameter: torch.Tensor
    master_parameter: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    step: torch.Tensor


_LOSS_SCALER_FIELDS = (
    "dynamic_loss_scale",
    "cur_scale",
    "cur_iter",
    "last_overflow_iter",
    "scale_factor",
    "scale_window",
    "min_loss_scale",
    "custom_loss_scaler",
    "external_loss_scale",
)

_ENGINE_COUNTER_FIELDS = (
    "global_steps",
    "global_samples",
    "micro_steps",
    "skipped_steps",
    "gas_boundary_ctr",
)


def _require_unfused_fp16(engine):
    optimizer = engine.optimizer
    if optimizer.__class__.__name__ != "FP16_UnfusedOptimizer":
        raise TypeError(
            "live Helix state migration currently requires "
            f"FP16_UnfusedOptimizer, got {optimizer.__class__.__name__}"
        )
    if not hasattr(optimizer, "fp16_groups") or not hasattr(optimizer, "fp32_groups"):
        raise TypeError("DeepSpeed FP16 optimizer lacks fp16_groups/fp32_groups")
    return optimizer


def _live_parameter_map(engine) -> OrderedDict[str, LiveFP16AdamParameter]:
    optimizer = _require_unfused_fp16(engine)
    name_by_identity = {
        id(parameter): name
        for name, parameter in engine.module.named_parameters()
        if parameter.requires_grad
    }
    result: OrderedDict[str, LiveFP16AdamParameter] = OrderedDict()
    for fp16_group, fp32_group in zip(optimizer.fp16_groups, optimizer.fp32_groups):
        if len(fp16_group) != len(fp32_group):
            raise RuntimeError("FP16/master optimizer groups have different lengths")
        for fp16_parameter, master_parameter in zip(fp16_group, fp32_group):
            name = name_by_identity.get(id(fp16_parameter))
            if name is None:
                raise RuntimeError("an FP16 optimizer parameter has no model name")
            state = optimizer.optimizer.state.get(master_parameter)
            if state is None:
                raise RuntimeError(f"Adam state is absent for {name}")
            missing = {"step", "exp_avg", "exp_avg_sq"}.difference(state)
            if missing:
                raise RuntimeError(f"Adam state for {name} is missing {sorted(missing)}")
            result[name] = LiveFP16AdamParameter(
                fp16_parameter=fp16_parameter,
                master_parameter=master_parameter,
                exp_avg=state["exp_avg"],
                exp_avg_sq=state["exp_avg_sq"],
                step=state["step"],
            )
    if set(result) != set(name_by_identity.values()):
        missing = sorted(set(name_by_identity.values()).difference(result))
        raise RuntimeError(f"optimizer mapping misses model parameters: {missing[:5]}")
    return result


def capture_fp16_adam_snapshot(engine, scheduler=None) -> FP16AdamSnapshot:
    """Copy the authoritative local master and both Adam moments to CPU."""

    optimizer = _require_unfused_fp16(engine)
    live = _live_parameter_map(engine)
    steps = {float(item.step.detach().cpu().item()) for item in live.values()}
    if len(steps) != 1:
        raise RuntimeError(f"local Adam parameters disagree on step: {sorted(steps)}")

    # All ranks must switch from the same optimizer step.
    local_step = next(iter(steps))
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, local_step)
    if len(set(gathered)) != 1:
        raise RuntimeError(f"ranks disagree on Adam step: {gathered}")

    parameters: OrderedDict[str, LocalFP16AdamParameter] = OrderedDict()
    for name, item in live.items():
        parameters[name] = LocalFP16AdamParameter(
            master_parameter=item.master_parameter.detach().to("cpu", copy=True),
            exp_avg=item.exp_avg.detach().to("cpu", copy=True),
            exp_avg_sq=item.exp_avg_sq.detach().to("cpu", copy=True),
            step=item.step.detach().to("cpu", copy=True),
        )

    optimizer_groups = tuple(
        {
            key: copy.deepcopy(value)
            for key, value in group.items()
            if key != "params"
        }
        for group in optimizer.optimizer.param_groups
    )
    loss_scaler = {
        field: copy.deepcopy(getattr(optimizer, field))
        for field in _LOSS_SCALER_FIELDS
        if hasattr(optimizer, field)
    }
    counters = {
        field: int(getattr(engine, field))
        for field in _ENGINE_COUNTER_FIELDS
        if hasattr(engine, field)
    }
    return FP16AdamSnapshot(
        parameters=parameters,
        optimizer_groups=optimizer_groups,
        loss_scaler=loss_scaler,
        scheduler_state=(copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None),
        engine_counters=counters,
        cpu_rng_state=torch.get_rng_state().clone(),
        cuda_rng_state=torch.cuda.get_rng_state(engine.device).clone(),
    )


def _copy_local_field(
    source: torch.Tensor,
    source_rectangle,
    destination: torch.Tensor,
    destination_rectangle,
    limit: int,
) -> None:
    source_view = _rectangle_view(source, source_rectangle)
    destination_view = _rectangle_view(destination, destination_rectangle)
    if source_view.shape != destination_view.shape:
        raise RuntimeError(
            f"dynamic state shape mismatch {tuple(source_view.shape)} vs "
            f"{tuple(destination_view.shape)}"
        )
    source_tiles = _bounded_views(source_view, limit)
    destination_tiles = _bounded_views(destination_view, limit)
    with torch.no_grad():
        for source_tile, destination_tile in zip(source_tiles, destination_tiles):
            destination_tile.copy_(
                source_tile.to(
                    device=destination_tile.device,
                    dtype=destination_tile.dtype,
                )
            )


def _transfer_remote_field(
    *,
    rank: int,
    source_rank: int,
    destination_rank: int,
    source: Optional[torch.Tensor],
    source_rectangle,
    destination: Optional[torch.Tensor],
    destination_rectangle,
    device: torch.device,
    limit: int,
) -> None:
    if rank == source_rank:
        assert source is not None
        source_view = _rectangle_view(source, source_rectangle)
        for source_tile in _bounded_views(source_view, limit):
            payload = source_tile.contiguous().reshape(-1).to(device=device)
            torch.distributed.send(payload, dst=destination_rank)
    elif rank == destination_rank:
        assert destination is not None
        destination_view = _rectangle_view(destination, destination_rectangle)
        for destination_tile in _bounded_views(destination_view, limit):
            payload = torch.empty(
                destination_tile.numel(),
                dtype=destination_tile.dtype,
                device=device,
            )
            torch.distributed.recv(payload, src=source_rank)
            with torch.no_grad():
                destination_tile.copy_(payload.reshape(destination_tile.shape))


def restore_fp16_adam_from_rectangles(
    new_engine,
    copy_plan: DynamicCopyPlan,
    old_snapshot: FP16AdamSnapshot,
    *,
    scheduler=None,
    elements_per_transfer: int = 6_291_456,
) -> None:
    """Fill a new local optimizer from distributed old-rank CPU snapshots."""

    if elements_per_transfer < 1:
        raise ValueError("elements_per_transfer must be positive")
    rank = torch.distributed.get_rank()
    live = _live_parameter_map(new_engine)
    if set(live) != set(old_snapshot.parameters):
        raise RuntimeError("old/new named parameter sets differ during dynamic resize")

    for item in copy_plan.copies:
        source_state = old_snapshot.parameters.get(item.name) if rank == item.source_rank else None
        destination_state = live.get(item.name) if rank == item.destination_rank else None
        for field in ("master_parameter", "exp_avg", "exp_avg_sq"):
            source_tensor = getattr(source_state, field) if source_state is not None else None
            destination_tensor = (
                getattr(destination_state, field) if destination_state is not None else None
            )
            if item.source_rank == item.destination_rank:
                if rank == item.source_rank:
                    assert source_tensor is not None and destination_tensor is not None
                    _copy_local_field(
                        source_tensor,
                        item.source_local_rectangle,
                        destination_tensor,
                        item.destination_local_rectangle,
                        elements_per_transfer,
                    )
            else:
                _transfer_remote_field(
                    rank=rank,
                    source_rank=item.source_rank,
                    destination_rank=item.destination_rank,
                    source=source_tensor,
                    source_rectangle=item.source_local_rectangle,
                    destination=destination_tensor,
                    destination_rectangle=item.destination_local_rectangle,
                    device=new_engine.device,
                    limit=elements_per_transfer,
                )

    # Adam step is global in this training path; each rank had every named
    # parameter before and after resizing, so its local snapshot is sufficient.
    with torch.no_grad():
        for name, item in live.items():
            saved = old_snapshot.parameters[name]
            item.step.copy_(saved.step.to(device=item.step.device, dtype=item.step.dtype))
            item.fp16_parameter.copy_(
                item.master_parameter.to(dtype=item.fp16_parameter.dtype)
            )

    optimizer = _require_unfused_fp16(new_engine)
    if len(optimizer.optimizer.param_groups) != len(old_snapshot.optimizer_groups):
        raise RuntimeError("old/new optimizer group counts differ")
    for group, saved in zip(optimizer.optimizer.param_groups, old_snapshot.optimizer_groups):
        for key, value in saved.items():
            group[key] = copy.deepcopy(value)
    for field, value in old_snapshot.loss_scaler.items():
        setattr(optimizer, field, copy.deepcopy(value))
    for field, value in old_snapshot.engine_counters.items():
        setattr(new_engine, field, int(value))
    if scheduler is not None and old_snapshot.scheduler_state is not None:
        scheduler.load_state_dict(copy.deepcopy(old_snapshot.scheduler_state))

    optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(old_snapshot.cpu_rng_state)
    torch.cuda.set_rng_state(old_snapshot.cuda_rng_state, device=new_engine.device)
    torch.distributed.barrier()
