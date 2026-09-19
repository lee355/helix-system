#!/usr/bin/env python
"""Canonical paper-aligned Helix entry with live complete-Adam migration."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

import helix_train_impl as implementation
from dschat.helix.deepspeed_paper_v2 import (
    install_bounded_paper_deepspeed_runtime,
)
from dschat.helix.dynamic_controller import install_dynamic_hooks
from dschat.helix.masking import infer_model_structure
from dschat.helix.model_factory import create_paper_model
from dschat.helix.paper_semantics import (
    install_paper_mask_semantics,
    paper_quantized_count,
)
from dschat.helix.planner import save_plan
from dschat.helix.planner_final import fast_search as final_fast_search


install_paper_mask_semantics()
install_bounded_paper_deepspeed_runtime()
implementation.create_helix_model = create_paper_model


_base_parse_args = implementation.parse_args


def parse_args():
    dynamic_parser = argparse.ArgumentParser(add_help=False)
    dynamic_parser.add_argument(
        "--helix_dynamic_check_interval",
        type=int,
        default=0,
        help="Check per-rank available memory every N optimizer steps; 0 disables live adjustment.",
    )
    dynamic_parser.add_argument(
        "--helix_dynamic_memory_threshold_gib",
        type=float,
        default=1.0,
        help="Minimum available-memory change that queues an epoch-boundary adjustment.",
    )
    dynamic_parser.add_argument(
        "--helix_dynamic_max_adjustments",
        type=int,
        default=1,
        help="Maximum live engine rebuilds; -1 allows unlimited rebuilds.",
    )
    dynamic_parser.add_argument(
        "--helix_dynamic_profiles_path",
        default=None,
        help="Profiles used for live adjustment when the initial source is a saved/explicit plan.",
    )
    dynamic, remaining = dynamic_parser.parse_known_args(sys.argv[1:])
    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining]
        args = _base_parse_args()
    finally:
        sys.argv = original_argv
    for name, value in vars(dynamic).items():
        setattr(args, name, value)

    if args.gradient_accumulation_steps != 1:
        raise ValueError(
            "paper-aligned planning/profiling currently requires "
            "--gradient_accumulation_steps 1; GAS>1 changes Eq.1 and the "
            "communication frequency and must be profiled as a separate mode"
        )
    if args.helix_dynamic_check_interval < 0:
        raise ValueError("--helix_dynamic_check_interval must be non-negative")
    if args.helix_dynamic_memory_threshold_gib <= 0.0:
        raise ValueError("--helix_dynamic_memory_threshold_gib must be positive")
    if args.helix_dynamic_max_adjustments < -1:
        raise ValueError("--helix_dynamic_max_adjustments must be -1 or non-negative")
    if args.helix_dynamic_check_interval > 0 and args.helix_acceptance_dir:
        raise ValueError("Initial Math acceptance currently requires a static plan")
    if args.helix_dynamic_check_interval > 0 and args.dtype != "fp16":
        raise ValueError("complete live Adam migration currently supports FP16 only")
    return args


implementation.parse_args = parse_args


def _covers_quantized_regions(
    sizes: Sequence[float],
    structure,
    ffn_alignment: int,
) -> bool:
    attention_regions = sum(
        paper_quantized_count(size, structure.num_key_value_heads)
        for size in sizes
    )
    ffn_regions = sum(
        paper_quantized_count(
            size,
            structure.intermediate_size,
            alignment=ffn_alignment,
        )
        for size in sizes
    )
    return (
        attention_regions >= structure.num_key_value_heads
        and ffn_regions >= structure.intermediate_size
    )


_base_resolve_plan = implementation._resolve_plan_on_rank_zero


def _resolve_plan_on_rank_zero(
    args,
    dataset_size,
    full_model,
    full_state,
    world_size,
):
    structure = infer_model_structure(full_state, full_model.config)

    def coverage_predicate(sizes):
        return _covers_quantized_regions(
            sizes,
            structure,
            args.helix_ffn_alignment,
        )

    previous_fast_search = implementation.fast_search
    save_path = args.helix_save_plan_path

    def search_with_quantized_coverage(*search_args, **search_kwargs):
        search_kwargs["coverage_predicate"] = coverage_predicate
        return final_fast_search(*search_args, **search_kwargs)

    implementation.fast_search = search_with_quantized_coverage
    args.helix_save_plan_path = None
    try:
        plan = _base_resolve_plan(
            args,
            dataset_size,
            full_model,
            full_state,
            world_size,
        )
    finally:
        implementation.fast_search = previous_fast_search
        args.helix_save_plan_path = save_path

    if not coverage_predicate(plan.submodel_sizes):
        raise ValueError(
            "plan has sum(s_i)>=1 but nearest-integer discretization leaves "
            "at least one attention or FFN region uncovered; increase one or "
            "more s_i"
        )
    if save_path:
        save_plan(save_path, plan)
    return plan


implementation._resolve_plan_on_rank_zero = _resolve_plan_on_rank_zero
install_dynamic_hooks(implementation, create_paper_model)


def main():
    implementation.main()


if __name__ == "__main__":
    main()
