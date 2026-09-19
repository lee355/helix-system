#!/usr/bin/env python
"""Canonical launcher/body for the real-runtime Helix profiler."""

from __future__ import annotations

import gc
import json
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM

import deepspeed
from deepspeed import get_accelerator

import helix_profile_ds as profile_impl
from dschat.helix.masking import infer_model_structure
from dschat.helix.model_paper_sdpa import create_paper_model
from dschat.helix.profiling import ProfileSample, build_device_profile, save_profiles
from dschat.utils.utils import load_hf_tokenizer, set_random_seed


# Ensure both the full model and profile submodels use the same audited
# attention backend. The lower-level module intentionally stays reusable.
profile_impl.create_paper_model = create_paper_model


def main():
    args = profile_impl.parse_args()
    if args.local_rank >= 0:
        get_accelerator().set_device(args.local_rank)
    deepspeed.init_distributed()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = torch.device(get_accelerator().device_name(), args.local_rank)
    set_random_seed(args.seed)

    tokenizer = load_hf_tokenizer(
        args.model_name_or_path,
        fast_tokenizer=True,
        model_max_length=args.max_seq_len,
        padding_side="right",
    )
    full_model = create_paper_model(
        AutoModelForCausalLM,
        args.model_name_or_path,
        num_hidden_layers=args.num_hidden_layers,
    )
    if full_model.get_input_embeddings().num_embeddings != len(tokenizer):
        full_model.resize_token_embeddings(len(tokenizer))
    full_state = full_model.state_dict()
    structure = infer_model_structure(full_state, full_model.config)
    parameter_names = [name for name, _ in full_model.named_parameters()]
    total_memory = torch.cuda.get_device_properties(device).total_memory
    initialization_budget = (
        total_memory - args.initialization_reserve_gib * 1024**3
    )
    if initialization_budget <= 0:
        raise ValueError("initialization reserve leaves no device memory")

    if args.profile_points:
        points = profile_impl._parse_explicit_points(args.profile_points)
        maximum_size = max(size for _, size in points)
    else:
        maximum_size = profile_impl._safe_maximum_size(
            full_state,
            structure,
            parameter_names,
            args.minimum_submodel_size,
            args.maximum_submodel_size,
            args.ffn_alignment,
            initialization_budget,
            args.optimizer_peak_bytes_per_parameter,
        )
        points = profile_impl._default_points(
            args.max_micro_batch_size,
            args.minimum_submodel_size,
            maximum_size,
        )
    print(
        f"rank={rank} device={torch.cuda.get_device_name(device)} "
        f"profile_s_max={maximum_size:.6f} points={points}",
        flush=True,
    )

    points_by_size = defaultdict(list)
    for batch_size, size in points:
        points_by_size[size].append(batch_size)
    samples = []
    failures = []

    for size, batch_sizes in sorted(points_by_size.items()):
        engine = None
        local_state = None
        try:
            engine, local_state = profile_impl._create_profile_engine(
                args,
                full_model,
                full_state,
                structure,
                len(tokenizer),
                size,
                rank,
                world_size,
            )
            for batch_size in sorted(set(batch_sizes)):
                try:
                    compute, memory = profile_impl._profile_engine(
                        engine,
                        batch_size,
                        args.max_seq_len,
                        len(tokenizer),
                        args.warmup_steps,
                        args.measure_steps,
                    )
                    samples.append(
                        ProfileSample(
                            micro_batch_size=batch_size,
                            submodel_size=size,
                            compute_seconds=compute,
                            memory_bytes=memory,
                        )
                    )
                    print(
                        f"rank={rank} profile b={batch_size} s={size:.6f} "
                        f"compute={compute:.6f}s "
                        f"memory={memory / 1024**3:.3f}GiB",
                        flush=True,
                    )
                except torch.cuda.OutOfMemoryError as error:
                    failures.append(
                        {
                            "batch_size": batch_size,
                            "submodel_size": size,
                            "error": str(error),
                        }
                    )
                    torch.cuda.empty_cache()
                    break
        except torch.cuda.OutOfMemoryError as error:
            failures.extend(
                {
                    "batch_size": batch_size,
                    "submodel_size": size,
                    "error": str(error),
                }
                for batch_size in batch_sizes
            )
        finally:
            # Drop the caller's references before collection; deleting only a
            # variadic helper's local aliases does not release the DS engine.
            engine = None
            local_state = None
            gc.collect()
            torch.cuda.empty_cache()
        torch.distributed.barrier()

    if len(samples) < 4:
        local_result = {
            "error": (
                f"rank {rank} has only {len(samples)} successful samples; "
                "lower max batch/size or the minimum submodel size"
            ),
            "failures": failures,
        }
    else:
        try:
            local_result = {
                "profile": build_device_profile(
                    rank=rank,
                    device_name=torch.cuda.get_device_name(device),
                    memory_budget_bytes=(
                        total_memory - args.memory_reserve_gib * 1024**3
                    ),
                    samples=samples,
                    model_name=args.model_name_or_path,
                    sequence_length=args.max_seq_len,
                    dtype=args.dtype,
                ),
                "failures": failures,
            }
        except Exception as error:
            local_result = {
                "error": f"rank {rank} profile fit failed: {error}",
                "failures": failures,
            }

    gathered = [None] * world_size
    torch.distributed.all_gather_object(gathered, local_result)
    errors = [item["error"] for item in gathered if "error" in item]
    if errors:
        if rank == 0:
            print(json.dumps({"profile_errors": errors}, indent=2), flush=True)
        raise RuntimeError("; ".join(errors))
    if rank == 0:
        profiles = [item["profile"] for item in gathered]
        save_profiles(args.output_path, profiles)
        failure_summary = {
            index: item["failures"]
            for index, item in enumerate(gathered)
            if item["failures"]
        }
        print(
            json.dumps(
                {
                    "saved_profiles": len(profiles),
                    "path": args.output_path,
                    "oom_points": failure_summary,
                },
                indent=2,
            ),
            flush=True,
        )
    torch.distributed.barrier()


if __name__ == "__main__":
    main()
