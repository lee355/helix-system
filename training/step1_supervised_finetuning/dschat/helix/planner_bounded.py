"""Planner entry that forbids extrapolation outside successful profile points."""

from __future__ import annotations

from typing import List

from . import planner_paper
from .planner import RankAllocation


def _profile_bounded_candidates(
    profile,
    max_micro_batch_size: int,
    minimum_submodel_size: float,
    memory_slack_bytes: float,
) -> List[RankAllocation]:
    budget = profile.memory_budget_bytes - memory_slack_bytes
    if budget <= 0.0:
        raise ValueError(f"rank {profile.rank} has no usable memory after slack")
    if not profile.samples:
        return planner_paper._base_rank_candidates(
            profile,
            max_micro_batch_size,
            minimum_submodel_size,
            memory_slack_bytes,
        )

    sampled_minimum = min(sample.submodel_size for sample in profile.samples)
    lower_bound = max(minimum_submodel_size, sampled_minimum)
    candidates: List[RankAllocation] = []
    for batch_size in range(1, max_micro_batch_size + 1):
        # A successful point with batch >= b certifies that its size is in the
        # interpolation domain for b.  Never infer a larger size from an OOM-
        # truncated profile surface.
        certified_sizes = [
            sample.submodel_size
            for sample in profile.samples
            if sample.micro_batch_size >= batch_size
        ]
        if not certified_sizes:
            continue
        certified_maximum = max(certified_sizes)
        try:
            predicted_size = profile.memory.size_at_budget(batch_size, budget)
        except ValueError as error:
            raise ValueError(
                f"invalid memory profile for rank {profile.rank}: {error}"
            ) from error
        size = min(1.0, certified_maximum, predicted_size)
        if size + 1e-12 < lower_bound:
            continue
        size = max(lower_bound, size)
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
        raise ValueError(
            f"rank {profile.rank} has no feasible profiled (b_i, s_i) combination"
        )
    return candidates


# Preserve the original helper under an explicit name before substituting it
# during the wrapper call.  This also supports old schema-v1 profiles without
# sample payloads.
if not hasattr(planner_paper, "_base_rank_candidates"):
    planner_paper._base_rank_candidates = planner_paper._rank_candidates


def fast_search(*args, **kwargs):
    original = planner_paper._rank_candidates
    planner_paper._rank_candidates = _profile_bounded_candidates
    try:
        return planner_paper.fast_search(*args, **kwargs)
    finally:
        planner_paper._rank_candidates = original
