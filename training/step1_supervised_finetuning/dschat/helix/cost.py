"""Communication-cost estimator for colored Cluster-Reduce groups."""

from __future__ import annotations

from typing import Mapping, Tuple

import torch

from .communication import CommunicationPlan, RankGroup, Rectangle


def _rectangle_numel(rectangle: Rectangle, ndim: int) -> int:
    if ndim == 1:
        return rectangle[1][1] - rectangle[0][1] + 1
    return (
        (rectangle[1][0] - rectangle[0][0] + 1)
        * (rectangle[1][1] - rectangle[0][1] + 1)
    )


def _slowest_bandwidth(
    group: RankGroup,
    bandwidth_bytes_per_second: Mapping[Tuple[int, int], float],
) -> float:
    links = []
    for left_index, left in enumerate(group):
        for right in group[left_index + 1 :]:
            link = (min(left, right), max(left, right))
            if link not in bandwidth_bytes_per_second:
                raise ValueError(f"Missing bandwidth measurement for rank link {link}")
            links.append(float(bandwidth_bytes_per_second[link]))
    if not links or min(links) <= 0.0:
        raise ValueError(f"Invalid bandwidth for communication group {group}")
    return min(links)


def estimate_cluster_reduce_seconds(
    plan: CommunicationPlan,
    state_dict: Mapping[str, torch.Tensor],
    bandwidth_bytes_per_second: Mapping[Tuple[int, int], float],
    communication_dtype_bytes: int = 2,
) -> float:
    """Estimate Eq.1 communication time with the ring model in Section 3.3.2.

    Same-colored rank-disjoint groups execute concurrently, so a cluster costs
    the maximum subgroup time; colored clusters execute in sequence.
    """

    if communication_dtype_bytes < 1:
        raise ValueError("communication_dtype_bytes must be positive")
    group_seconds = {}
    for group, layers in plan.groups_to_global_rectangles.items():
        elements = 0
        for name, rectangles in layers.items():
            if name not in state_dict:
                raise ValueError(f"Communication plan references unknown parameter {name}")
            ndim = state_dict[name].ndim
            elements += sum(_rectangle_numel(rectangle, ndim) for rectangle in rectangles)
        volume = elements * communication_dtype_bytes
        slowest = _slowest_bandwidth(group, bandwidth_bytes_per_second)
        # The paper denotes by N' the GPUs sharing the slowest link. In the
        # absence of topology-specific contention metadata, treating all
        # subgroup members as N' is the conservative ring estimate.
        participants = len(group)
        group_seconds[group] = 2.0 * (participants - 1) * volume / (participants * slowest)

    seconds = 0.0
    for cluster in plan.clusters:
        if cluster:
            seconds += max(group_seconds[group] for group in cluster)
    return seconds
