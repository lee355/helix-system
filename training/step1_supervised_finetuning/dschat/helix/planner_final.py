"""Coverage- and profile-bounded Helix online search."""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

from .planner import TrainingPlan, _balanced_allocations
from .planner_bounded import _profile_bounded_candidates
from .planner_paper import _coverage_paths
from .profiling import DeviceProfile


def fast_search(
    profiles: Sequence[DeviceProfile],
    dataset_size: int,
    communication_time: Optional[Callable[[Sequence[float]], float]] = None,
    max_micro_batch_size: int = 32,
    max_total_batch_size: Optional[int] = None,
    minimum_submodel_size: float = 1e-3,
    minimum_coverage: float = 1.0,
    memory_slack_bytes: float = 0.0,
    coverage_predicate: Optional[Callable[[Sequence[float]], bool]] = None,
) -> TrainingPlan:
    if dataset_size < 1:
        raise ValueError("dataset_size must be positive")
    if not profiles:
        raise ValueError("at least one device profile is required")
    ordered = sorted(profiles, key=lambda profile: profile.rank)
    if [profile.rank for profile in ordered] != list(range(len(ordered))):
        raise ValueError("profiles must contain contiguous ranks starting at zero")

    candidates = [
        _profile_bounded_candidates(
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
    coverage_paths = _coverage_paths(candidates, maximum_total)
    communication_time = communication_time or (lambda _sizes: 0.0)

    best = None
    for total_batch in range(len(ordered), maximum_total + 1):
        alternatives = []
        for chosen in (balanced.get(total_batch), coverage_paths.get(total_batch)):
            if chosen is not None and chosen not in alternatives:
                alternatives.append(chosen)
        for allocations in alternatives:
            sizes = [item.submodel_size for item in allocations]
            total_size = sum(sizes)
            if total_size + 1e-12 < minimum_coverage:
                continue
            if coverage_predicate is not None and not coverage_predicate(sizes):
                continue
            communication_seconds = float(communication_time(sizes))
            if not math.isfinite(communication_seconds) or communication_seconds < 0.0:
                raise ValueError("communication_time must be finite and non-negative")
            compute_seconds = max(item.estimated_compute_seconds for item in allocations)
            step_seconds = compute_seconds + communication_seconds
            cost = (dataset_size / total_batch) * step_seconds / total_size
            plan = TrainingPlan(
                allocations=tuple(allocations),
                total_batch_size=total_batch,
                estimated_communication_seconds=communication_seconds,
                estimated_step_seconds=step_seconds,
                estimated_total_cost=cost,
                dataset_size=dataset_size,
            )
            if best is None or plan.estimated_total_cost < best.estimated_total_cost:
                best = plan
    if best is None:
        raise ValueError(
            "no feasible plan covers every quantized attention/FFN region "
            "inside the successful profile domain"
        )
    return best
