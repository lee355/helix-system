#!/usr/bin/env python
"""Profile Helix submodels through the actual DeepSpeed optimizer path."""

from __future__ import annotations

import argparse
import gc
import json
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM

import deepspeed
from deepspeed import get_accelerator

from dschat.helix.masking import (
    TensorMask,
    build_structured_masks,
    infer_model_structure,
    serialize_mask,
    slice_state_dict,
)
from dschat.helix.model import validate_state_shapes
from dschat.helix.model_paper import create_paper_model
from dschat.helix.paper_semantics import (
    apply_paper_deepspeed_config,
    install_paper_mask_semantics,
)
from dschat.helix.precision import restore_rotary_fp32
from dschat.helix.profiling import ProfileSample, build_device_profile, save_profiles
from dschat.utils.ds_utils import get_train_ds_config
from dschat.utils.utils import load_hf_tokenizer, set_random_seed


install_paper_mask_semantics()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile Helix using the real DeepSpeed FP16/BF16 optimizer"
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_micro_batch_size", type=int, default=16)
    parser.add_argument("--minimum_submodel_size", type=float, default=0.125)
    parser.add_argument("--maximum_submodel_size", type=float, default=1.0)
    parser.add_argument(
        "--profile_points",
        default=None,
        help="Optional comma-separated b:s points; every rank uses the same points.",
    )
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--measure_steps", type=int, default=3)
    parser.add_argument("--memory_reserve_gib", type=float, default=1.0)
    parser.add_argument(
        "--initialization_reserve_gib",
        type=float,
        default=4.0,
        help="Conservative non-parameter allowance used only to choose a safe profile s ceiling.",
    )
    parser.add_argument(
        "--optimizer_peak_bytes_per_parameter",
        type=float,
        default=20.0,
        help="FP16 parameter/gradient/master/moments/transient estimate for safe initialization.",
    )
    parser.add_argument("--ffn_alignment", type=int, default=1)
    parser.add_argument("--num_hidden_layers", type=int, default=None)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    if args.max_micro_batch_size < 2:
        parser.error("--max_micro_batch_size must be at least two for bilinear fitting")
    if not 0.0 < args.minimum_submodel_size <= args.maximum_submodel_size <= 1.0:
        parser.error("submodel bounds must satisfy 0 < minimum <= maximum <= 1")
    if args.warmup_steps < 1 or args.measure_steps < 1:
        parser.error("warmup and measurement steps must be positive")
    if args.ffn_alignment < 1:
        parser.error("--ffn_alignment must be positive")
    if args.memory_reserve_gib < 0.0 or args.initialization_reserve_gib < 0.0:
        parser.error("memory reserves must be non-negative")
    if args.optimizer_peak_bytes_per_parameter <= 0.0:
        parser.error("--optimizer_peak_bytes_per_parameter must be positive")
    return args


def _parse_explicit_points(value: str) -> List[Tuple[int, float]]:
    points = []
    for item in value.split(","):
        try:
            batch, size = item.split(":", 1)
            point = int(batch), float(size)
        except ValueError as error:
            raise ValueError(f"invalid profile point {item!r}; expected b:s") from error
        if point[0] < 1 or not 0.0 < point[1] <= 1.0:
            raise ValueError(f"invalid profile point {item!r}")
        points.append(point)
    if len(points) < 4:
        raise ValueError("at least four profile points are required")
    return points


def _default_points(
    max_batch: int,
    minimum_size: float,
    maximum_size: float,
) -> List[Tuple[int, float]]:
    batches = sorted(
        {
            1,
            max_batch,
            1 + round((max_batch - 1) / 3),
            1 + round(2 * (max_batch - 1) / 3),
        }
    )
    if len(batches) < 2:
        raise ValueError("profile design requires at least two distinct batch sizes")
    if maximum_size <= minimum_size + 1e-6:
        raise ValueError(
            "this device has no two distinct safe submodel sizes; lower "
            "--minimum_submodel_size or increase initialization headroom"
        )
    # Four batch levels by two size levels gives the paper's eight-point
    # bilinear design for the normal max_batch>=4 case.
    return [
        (batch, size)
        for size in (minimum_size, maximum_size)
        for batch in batches
    ]


def _masked_numel(
    state: Mapping[str, torch.Tensor],
    mask: Mapping[str, TensorMask],
    parameter_names: Iterable[str],
) -> int:
    total = 0
    for name in parameter_names:
        tensor = state[name]
        indices = mask[name]
        if tensor.ndim == 1:
            total += int(indices.numel())
        elif tensor.ndim == 2:
            total += int(indices[0].numel()) * int(indices[1].numel())
    return total


def _safe_maximum_size(
    state,
    structure,
    parameter_names,
    minimum_size: float,
    requested_maximum: float,
    ffn_alignment: int,
    device_budget_bytes: float,
    bytes_per_parameter: float,
) -> float:
    def estimated_bytes(size: float) -> float:
        masks, _ = build_structured_masks(
            state,
            [size],
            structure=structure,
            ffn_alignment=ffn_alignment,
        )
        return _masked_numel(state, masks[0], parameter_names) * bytes_per_parameter

    minimum_bytes = estimated_bytes(minimum_size)
    if minimum_bytes > device_budget_bytes:
        raise RuntimeError(
            f"minimum submodel s={minimum_size} needs an estimated "
            f"{minimum_bytes / 1024**3:.2f} GiB before activations, but only "
            f"{device_budget_bytes / 1024**3:.2f} GiB was reserved"
        )
    if estimated_bytes(requested_maximum) <= device_budget_bytes:
        return requested_maximum
    low, high = minimum_size, requested_maximum
    for _ in range(32):
        middle = (low + high) / 2.0
        if estimated_bytes(middle) <= device_budget_bytes:
            low = middle
        else:
            high = middle
    # Stay strictly inside the estimated boundary.
    return max(minimum_size, math.nextafter(low, minimum_size))


def _deepspeed_config(args, world_size: int):
    config = get_train_ds_config(
        offload=False,
        dtype=args.dtype,
        stage=0,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    config["train_micro_batch_size_per_gpu"] = 1
    config["train_batch_size"] = world_size
    config["gradient_accumulation_steps"] = 1
    config.pop("activation_checkpointing", None)
    # Profile at the requested fixed AdamW learning rate, without historical WarmupLR.
    config.pop("scheduler", None)
    if "flops_profiler" in config:
        config["flops_profiler"]["enabled"] = False
    if args.dtype == "bf16":
        config.pop("fp16", None)
        config["bf16"] = {"enabled": True}
    config["zero_allow_untested_optimizer"] = True
    return apply_paper_deepspeed_config(config)


def _create_profile_engine(
    args,
    full_model,
    full_state,
    structure,
    tokenizer_size: int,
    size: float,
    rank: int,
    world_size: int,
):
    masks, specs = build_structured_masks(
        full_state,
        [size],
        structure=structure,
        ffn_alignment=args.ffn_alignment,
    )
    local_state = slice_state_dict(full_state, masks[0])
    model = create_paper_model(
        AutoModelForCausalLM,
        args.model_name_or_path,
        attention_rate=specs[0].attention_rate,
        ffn_rate=specs[0].ffn_rate,
        submodel_ids=serialize_mask(masks[0]),
        freeze_blocks=[],
        num_hidden_layers=args.num_hidden_layers,
    )
    if model.get_input_embeddings().num_embeddings != tokenizer_size:
        model.resize_token_embeddings(tokenizer_size)
    validate_state_shapes(model, local_state)
    model.load_state_dict(local_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        foreach=False,
    )
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        training_data=None,
        args=args,
        config=_deepspeed_config(args, world_size),
        dist_init_required=False,
        head_prune_rate=specs[0].attention_rate,
        prune_rate=specs[0].ffn_rate,
        roll=0.0,
        full_model=full_model,
        groups_to_offset_rectangles={},
        all_overlap_groups_ls=[],
        complementary_rank_ls=list(range(world_size)),
        complementary_groups_ranks_to_rectangles={},
        complementary_groups_to_offset_rectangles={},
    )
    rotary_fp32_buffers = restore_rotary_fp32(engine.module)
    engine.helix_rotary_fp32_buffers = rotary_fp32_buffers
    print(
        f"[rank {rank}] Helix RoPE precision: {rotary_fp32_buffers} official "
        "Llama inv_freq buffers are FP32 and persistent=False",
        flush=True,
    )
    engine.train()
    return engine, local_state


def _train_step(engine, input_ids, attention_mask, labels):
    engine.zero_grad(set_to_none=True)
    loss = engine(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    ).loss
    engine.backward(loss)
    engine.step()


def _profile_engine(
    engine,
    batch_size: int,
    sequence_length: int,
    vocab_size: int,
    warmup_steps: int,
    measure_steps: int,
) -> Tuple[float, float]:
    input_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length),
        device=engine.device,
    )
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    applied_warmup_steps = 0
    for _ in range(warmup_steps):
        _train_step(engine, input_ids, attention_mask, labels)
        applied_warmup_steps += int(engine.was_step_applied())
    # FP16 dynamic loss scaling can skip every requested warmup step. Adam's
    # moments are allocated lazily at the first real update, so measuring such
    # an engine would underestimate both memory and optimizer computation.
    extra_warmup_limit = 32
    extra_warmup_steps = 0
    while applied_warmup_steps == 0 and extra_warmup_steps < extra_warmup_limit:
        _train_step(engine, input_ids, attention_mask, labels)
        extra_warmup_steps += 1
        applied_warmup_steps += int(engine.was_step_applied())
    if applied_warmup_steps == 0:
        raise RuntimeError(
            "Helix profiling warmup produced no optimizer update after "
            f"{warmup_steps + extra_warmup_steps} attempts "
            f"(batch_size={batch_size}, sequence_length={sequence_length}); "
            "FP16 overflow/skipped steps prevent a valid Adam memory/time sample"
        )
    torch.cuda.synchronize(engine.device)
    torch.cuda.reset_peak_memory_stats(engine.device)
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    for measurement_step in range(measure_steps):
        _train_step(engine, input_ids, attention_mask, labels)
        if not engine.was_step_applied():
            raise RuntimeError(
                "Helix profiling measurement skipped an optimizer update "
                f"at step {measurement_step + 1}/{measure_steps} "
                f"(batch_size={batch_size}, sequence_length={sequence_length}); "
                "discarding the entire FP16 overflow/skipped-step sample"
            )
    finished.record()
    finished.synchronize()
    seconds = started.elapsed_time(finished) / 1000.0 / measure_steps
    memory = float(torch.cuda.max_memory_allocated(engine.device))
    return seconds, memory


def _release(*objects) -> None:
    for item in objects:
        del item
    gc.collect()
    torch.cuda.empty_cache()


def main():
    args = parse_args()
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
    initialization_budget = total_memory - args.initialization_reserve_gib * 1024**3
    if initialization_budget <= 0:
        raise ValueError("initialization reserve leaves no device memory")

    if args.profile_points:
        points = _parse_explicit_points(args.profile_points)
        maximum_size = max(size for _, size in points)
    else:
        maximum_size = _safe_maximum_size(
            full_state,
            structure,
            parameter_names,
            args.minimum_submodel_size,
            args.maximum_submodel_size,
            args.ffn_alignment,
            initialization_budget,
            args.optimizer_peak_bytes_per_parameter,
        )
        points = _default_points(
            args.max_micro_batch_size,
            args.minimum_submodel_size,
            maximum_size,
        )
    print(
        f"rank={rank} device={torch.cuda.get_device_name(device)} "
        f"profile_s_max={maximum_size:.6f} points={points}",
        flush=True,
    )

    points_by_size: Dict[float, List[int]] = defaultdict(list)
    for batch_size, size in points:
        points_by_size[size].append(batch_size)
    samples: List[ProfileSample] = []
    failures = []

    # Every rank has the same number of size levels, so custom DeepSpeed's
    # construction barriers/new_group calls remain ordered even though each
    # heterogeneous rank may use a different automatically selected s ceiling.
    for size, batch_sizes in sorted(points_by_size.items()):
        engine = local_state = None
        try:
            engine, local_state = _create_profile_engine(
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
                    compute, memory = _profile_engine(
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
                    # Larger batches at the same size cannot be certified.
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
            torch.cuda.empty_cache()
        finally:
            _release(engine, local_state)
        # Keep engine construction order synchronized. Training steps contain
        # no collectives because profile engines have no overlap groups.
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
        budget = total_memory - args.memory_reserve_gib * 1024**3
        try:
            local_result = {
                "profile": build_device_profile(
                    rank=rank,
                    device_name=torch.cuda.get_device_name(device),
                    memory_budget_bytes=budget,
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
