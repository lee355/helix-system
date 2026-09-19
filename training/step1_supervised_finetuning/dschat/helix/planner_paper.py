"""Coverage-safe variant of the Helix online search.

The base implementation follows the paper's fast balanced search.  This
wrapper additionally keeps a maximum-coverage path for every total batch so
integer rounding cannot discard all feasible full-model plans merely because
the single most balanced path at that batch has coverage below one.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Tuple

from .planner import (
    RankAllocation,
    TrainingPlan,
    _balanced_allocations,
    _rank_candidates,
)
from .profiling import DeviceProfile


def _coverage_paths(
    candidates: Sequence[Sequence[RankAllocation]],
    maximum_total_batch: int,
) -> Dict[int, Tuple[RankAllocation, ...]]:
    states: Dict[int, Tuple[float, Tuple[RankAllocation, ...]]] = {0: (0.0, ())}
    for rank_candidates in candidates:
        next_states: Dict[int, Tuple[float, Tuple[RankAllocation, ...]]] = {}
        for total, (coverage, chosen) in states.items():
            for allocation in rank_candidates:
                new_total = total + allocation.micro_batch_size
                if new_total > maximum_total_batch:
                    continue
                candidate = (coverage + allocation.submodel_size, chosen + (allocation,))
                previous = next_states.get(new_total)
                if previous is None or candidate[0] > previous[0]:
                    next_states[new_total] = candidate
        states = next_states
    return {total: chosen for total, (_, chosen) in states.items()}


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
    if dataset_size < 1:
        raise ValueError("dataset_size must be positive")
    if not profiles:
        raise ValueError("at least one device profile is required")
    ordered = sorted(profiles, key=lambda profile: profile.rank)
    if [profile.rank for profile in ordered] != list(range(len(ordered))):
        raise ValueError("profiles must contain contiguous ranks starting at zero")
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
        max(item.micro_batch_size for item in rank_candidates)
        for rank_candidates in candidates
    )
    balanced = _balanced_allocations(candidates, maximum_total)
    coverage = _coverage_paths(candidates, maximum_total)
    communication_time = communication_time or (lambda _sizes: 0.0)

    best: Optional[TrainingPlan] = None
    for total_batch in range(len(ordered), maximum_total + 1):
        # The balanced solution normally implements Eq.2.  The coverage path
        # is a correctness fallback for discrete b_i rounding.  Deduplicate
        # when both dynamic programs selected the same tuple.
        alternatives = []
        for allocation_tuple in (balanced.get(total_batch), coverage.get(total_batch)):
            if allocation_tuple is not None and allocation_tuple not in alternatives:
                alternatives.append(allocation_tuple)
        for allocations in alternatives:
            sizes = [allocation.submodel_size for allocation in allocations]
            total_size = sum(sizes)
            if total_size + 1e-12 < minimum_coverage:
                continue
            comm_seconds = float(communication_time(sizes))
            if not math.isfinite(comm_seconds) or comm_seconds < 0.0:
                raise ValueError("communication_time must return a finite non-negative value")
            compute_seconds = max(item.estimated_compute_seconds for item in allocations)
            step_seconds = compute_seconds + comm_seconds
            total_cost = (dataset_size / total_batch) * step_seconds / total_size
            plan = TrainingPlan(
                allocations=tuple(allocations),
                total_batch_size=total_batch,
                estimated_communication_seconds=comm_seconds,
                estimated_step_seconds=step_seconds,
                estimated_total_cost=total_cost,
                dataset_size=dataset_size,
            )
            if best is None or plan.estimated_total_cost < best.estimated_total_cost:
                best = plan

    if best is None:
        raise ValueError("no feasible plan covers the full model within the memory budgets")
    return best
