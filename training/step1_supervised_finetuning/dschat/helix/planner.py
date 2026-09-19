"""Online planner implementing the search in Section 3.4 of the paper."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .profiling import DeviceProfile


@dataclass(frozen=True)
class RankAllocation:
    rank: int
    micro_batch_size: int
    submodel_size: float
    estimated_compute_seconds: float
    estimated_memory_bytes: float


@dataclass(frozen=True)
class TrainingPlan:
    allocations: Tuple[RankAllocation, ...]
    total_batch_size: int
    estimated_communication_seconds: float
    estimated_step_seconds: float
    estimated_total_cost: float
    dataset_size: int

    @property
    def batch_sizes(self) -> List[int]:
        return [allocation.micro_batch_size for allocation in self.allocations]

    @property
    def submodel_sizes(self) -> List[float]:
        return [allocation.submodel_size for allocation in self.allocations]

    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            "allocations": [asdict(allocation) for allocation in self.allocations],
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "TrainingPlan":
        return cls(
            allocations=tuple(RankAllocation(**item) for item in payload["allocations"]),
            total_batch_size=int(payload["total_batch_size"]),
            estimated_communication_seconds=float(payload["estimated_communication_seconds"]),
            estimated_step_seconds=float(payload["estimated_step_seconds"]),
            estimated_total_cost=float(payload["estimated_total_cost"]),
            dataset_size=int(payload["dataset_size"]),
        )


def save_plan(path: str, plan: TrainingPlan) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"schema_version": 1, "plan": plan.to_dict()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_plan(path: str) -> TrainingPlan:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported plan schema: {payload.get('schema_version')}")
    return TrainingPlan.from_dict(payload["plan"])


def _rank_candidates(
    profile: DeviceProfile,
    max_micro_batch_size: int,
    minimum_submodel_size: float,
    memory_slack_bytes: float,
) -> List[RankAllocation]:
    budget = profile.memory_budget_bytes - memory_slack_bytes
    if budget <= 0:
        raise ValueError(f"rank {profile.rank} has no usable memory after slack")
    candidates: List[RankAllocation] = []
    for batch_size in range(1, max_micro_batch_size + 1):
        try:
            size = min(1.0, profile.memory.size_at_budget(batch_size, budget))
        except ValueError as error:
            raise ValueError(f"invalid memory profile for rank {profile.rank}: {error}") from error
        if size < minimum_submodel_size:
            continue
        size = max(minimum_submodel_size, size)
        memory = profile.memory.predict(batch_size, size)
        if memory > budget + max(1.0, abs(budget) * 1e-9):
            continue
        compute = profile.compute.predict(batch_size, size)
        if compute <= 0.0:
            raise ValueError(f"rank {profile.rank} predicts non-positive compute time")
        candidates.append(
            RankAllocation(
                rank=profile.rank,
                micro_batch_size=batch_size,
                submodel_size=size,
                estimated_compute_seconds=compute,
                estimated_memory_bytes=memory,
            )
        )
    if not candidates:
        raise ValueError(f"rank {profile.rank} has no feasible (b_i, s_i) combination")
    return candidates


def _target_times(candidates: Sequence[Sequence[RankAllocation]], count: int = 33) -> List[float]:
    values = sorted(
        allocation.estimated_compute_seconds
        for rank_candidates in candidates
        for allocation in rank_candidates
    )
    if len(values) <= count:
        return values
    indices = {round(index * (len(values) - 1) / (count - 1)) for index in range(count)}
    return [values[index] for index in sorted(indices)]


def _balanced_allocations(
    candidates: Sequence[Sequence[RankAllocation]],
    maximum_total_batch: int,
) -> Dict[int, Tuple[RankAllocation, ...]]:
    """Approximate Eq.2 by integer DP around several common-time targets."""

    best: Dict[int, Tuple[Tuple[float, float, float], Tuple[RankAllocation, ...]]] = {}
    for target in _target_times(candidates):
        # total batch -> (squared error to common time, allocations)
        states: Dict[int, Tuple[float, Tuple[RankAllocation, ...]]] = {0: (0.0, ())}
        for rank_candidates in candidates:
            next_states: Dict[int, Tuple[float, Tuple[RankAllocation, ...]]] = {}
            for total, (error, chosen) in states.items():
                for allocation in rank_candidates:
                    new_total = total + allocation.micro_batch_size
                    if new_total > maximum_total_batch:
                        continue
                    new_error = error + (allocation.estimated_compute_seconds - target) ** 2
                    previous = next_states.get(new_total)
                    if previous is None or new_error < previous[0]:
                        next_states[new_total] = (new_error, chosen + (allocation,))
            states = next_states

        for total, (error, chosen) in states.items():
            times = [allocation.estimated_compute_seconds for allocation in chosen]
            score = (max(times) - min(times), error, max(times))
            if total not in best or score < best[total][0]:
                best[total] = (score, chosen)
    return {total: chosen for total, (_, chosen) in best.items()}


def _maximum_coverage_by_batch(
    candidates: Sequence[Sequence[RankAllocation]],
    maximum_total_batch: int,
) -> Dict[int, float]:
    states: Dict[int, float] = {0: 0.0}
    for rank_candidates in candidates:
        next_states: Dict[int, float] = {}
        for total, coverage in states.items():
            for allocation in rank_candidates:
                new_total = total + allocation.micro_batch_size
                if new_total > maximum_total_batch:
                    continue
                next_states[new_total] = max(
                    next_states.get(new_total, -math.inf),
                    coverage + allocation.submodel_size,
                )
        states = next_states
    return states


def fast_search(
    profiles: Sequence[DeviceProfile],
    dataset_size: int,
    communication_time: Optional[Callable[[Sequence[float]], float]] = None,
    max_micro_batch_size: int = 32,
    max_total_batch_size: Optional[int] = None,
    minimum_submodel_size: float = 1e-3,
    minimum_coverage: float = 1.0,
    memory_slack_bytes: float = 0.0,
) -> TrainingPlan:
    """Enumerate total batch size and return the minimum-cost feasible plan.

    Memory is saturated analytically for each integer ``b_i``.  A small
    dynamic program then finds allocations whose predicted compute times are
    as equal as possible, matching the two constraints in Eq.2.
    """

    if dataset_size < 1:
        raise ValueError("dataset_size must be positive")
    if not profiles:
        raise ValueError("At least one device profile is required")
    ordered = sorted(profiles, key=lambda profile: profile.rank)
    expected_ranks = list(range(len(ordered)))
    if [profile.rank for profile in ordered] != expected_ranks:
        raise ValueError(f"profiles must contain contiguous ranks {expected_ranks}")
    if max_micro_batch_size < 1:
        raise ValueError("max_micro_batch_size must be positive")

    candidates = [
        _rank_candidates(
            profile,
            max_micro_batch_size,
            minimum_submodel_size,
            memory_slack_bytes,
        )
        for profile in ordered
    ]
    maximum_total = max_total_batch_size or sum(
        max(allocation.micro_batch_size for allocation in rank_candidates)
        for rank_candidates in candidates
    )
    minimum_total = len(profiles)
    balanced = _balanced_allocations(candidates, maximum_total)
    max_coverage = _maximum_coverage_by_batch(candidates, maximum_total)
    communication_time = communication_time or (lambda _sizes: 0.0)

    best_plan: Optional[TrainingPlan] = None
    for total_batch in range(minimum_total, maximum_total + 1):
        if max_coverage.get(total_batch, -math.inf) + 1e-12 < minimum_coverage:
            continue
        allocations = balanced.get(total_batch)
        if allocations is None:
            continue
        total_size = sum(allocation.submodel_size for allocation in allocations)
        if total_size + 1e-12 < minimum_coverage:
            # Integer batch rounding may make the most balanced solution fail
            # coverage even if an unbalanced one exists. A neighboring total
            # will still be considered; never accept an uncovered model.
            continue
        sizes = [allocation.submodel_size for allocation in allocations]
        comm_seconds = float(communication_time(sizes))
        if comm_seconds < 0.0:
            raise ValueError("communication_time returned a negative value")
        compute_seconds = max(allocation.estimated_compute_seconds for allocation in allocations)
        step_seconds = comm_seconds + compute_seconds
        # Eq.1: iterations per epoch times step time, normalized by the total
        # trained-model coverage in the iteration.
        total_cost = (dataset_size / total_batch) * step_seconds / total_size
        plan = TrainingPlan(
            allocations=allocations,
            total_batch_size=total_batch,
            estimated_communication_seconds=comm_seconds,
            estimated_step_seconds=step_seconds,
            estimated_total_cost=total_cost,
            dataset_size=dataset_size,
        )
        if best_plan is None or plan.estimated_total_cost < best_plan.estimated_total_cost:
            best_plan = plan

    if best_plan is None:
        raise ValueError("No feasible plan covers the full model within the memory budgets")
    return best_plan


def adjust_rank_to_memory_change(
    profile: DeviceProfile,
    previous_compute_seconds: float,
    new_memory_budget_bytes: float,
    max_micro_batch_size: int = 32,
    minimum_submodel_size: float = 1e-3,
    memory_slack_bytes: float = 0.0,
) -> RankAllocation:
    """Solve the per-rank constant-compute adjustment from Section 3.4.2."""

    changed = DeviceProfile(
        rank=profile.rank,
        device_name=profile.device_name,
        memory_budget_bytes=new_memory_budget_bytes,
        compute=profile.compute,
        memory=profile.memory,
        samples=profile.samples,
        model_name=profile.model_name,
        sequence_length=profile.sequence_length,
        dtype=profile.dtype,
    )
    candidates = _rank_candidates(
        changed,
        max_micro_batch_size,
        minimum_submodel_size,
        memory_slack_bytes,
    )
    return min(
        candidates,
        key=lambda allocation: (
            abs(allocation.estimated_compute_seconds - previous_compute_seconds),
            -allocation.submodel_size,
        ),
    )
