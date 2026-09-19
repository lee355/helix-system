#!/usr/bin/env python
"""Paper-aligned Helix SFT entry point.

Unlike the historical ``math_main.py``, this entry point has no model-name,
world-size, or batch-ID-specific plan.  It consumes an offline profile, a
saved plan, or explicit per-rank ``(b_i, s_i)`` values; generates exact masks;
and gives every rank a synchronized heterogeneous data stream.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import List, Sequence

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, SchedulerType, get_scheduler

import deepspeed
from deepspeed import get_accelerator

from dschat.helix.communication import build_communication_plan
from dschat.helix.cost import estimate_cluster_reduce_seconds
from dschat.helix.data import HeterogeneousDistributedBatchSampler
from dschat.helix.masking import (
    build_structured_masks,
    canonicalize_state_dict_by_importance,
    infer_model_structure,
    serialize_mask,
    slice_state_dict,
)
from dschat.helix.model import create_helix_model, validate_state_shapes
from dschat.helix.planner import (
    RankAllocation,
    TrainingPlan,
    fast_search,
    load_plan,
    save_plan,
)
from dschat.helix.profiling import load_profiles
from dschat.utils.data.math_data_utils import (
    DataCollatorForFinanceDataset,
    DataCollatorForSupervisedDataset,
    SupervisedDataset,
)
from dschat.utils.ds_utils import get_train_ds_config
from dschat.utils.utils import (
    get_optimizer_grouped_parameters,
    load_hf_tokenizer,
    print_rank_0,
    save_hf_format,
    set_random_seed,
    to_device,
)


def _csv_numbers(value: str, cast, label: str) -> List:
    try:
        result = [cast(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid {label}: {value}") from error
    if not result:
        raise argparse.ArgumentTypeError(f"{label} must not be empty")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Train heterogeneous Helix submodels")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--trainset", choices=["math", "finance"], default="math")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--zero_stage", type=int, default=0)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    parser.add_argument("--print_loss", action="store_true")
    parser.add_argument(
        "--helix_max_steps", type=int, default=0,
        help="Stop after N training microsteps across epochs; 0 runs all epochs.",
    )
    parser.add_argument(
        "--helix_metrics_dir", default=None,
        help="Write per-rank JSONL measurements; unset leaves instrumentation disabled.",
    )
    parser.add_argument(
        "--helix_metrics_warmup_steps", type=int, default=5,
        help="Mark the first N microsteps as warmup in metrics (still recorded).",
    )
    parser.add_argument(
        "--helix_skip_checkpoint", action="store_true",
        help="Skip interval and epoch checkpoint reconstruction/saving.",
    )

    acceptance = parser.add_argument_group("Full-model Math acceptance")
    acceptance.add_argument("--helix_token_cache_path", default=None)
    acceptance.add_argument("--helix_mmlu_path", default=None)
    acceptance.add_argument("--helix_acceptance_dir", default=None)
    acceptance.add_argument("--helix_target_accuracy", type=float, default=0.62)
    acceptance.add_argument("--helix_eval_interval", type=int, default=250)
    acceptance.add_argument("--helix_eval_fewshot", type=int, default=5)
    acceptance.add_argument("--helix_eval_max_length", type=int, default=4096)
    acceptance.add_argument("--helix_eval_batch_size", type=int, default=1)
    acceptance.add_argument("--helix_max_train_seconds", type=float, default=0)
    acceptance.add_argument("--helix_max_successful_steps", type=int, default=0)
    acceptance.add_argument("--helix_initial_scale_power", type=int, default=None)

    plan = parser.add_argument_group("Helix plan")
    source = plan.add_mutually_exclusive_group(required=True)
    source.add_argument("--helix_plan_path")
    source.add_argument("--helix_profiles_path")
    source.add_argument(
        "--helix_submodel_sizes",
        help="Comma-separated s_i values; requires --helix_micro_batch_sizes",
    )
    plan.add_argument("--helix_micro_batch_sizes", help="Comma-separated b_i values")
    plan.add_argument("--helix_save_plan_path")
    plan.add_argument("--helix_max_micro_batch_size", type=int, default=32)
    plan.add_argument("--helix_max_total_batch_size", type=int, default=None)
    plan.add_argument("--helix_memory_slack_gib", type=float, default=0.0)
    plan.add_argument("--helix_ffn_alignment", type=int, default=128)
    plan.add_argument("--helix_link_bandwidth_gbps", type=float, default=10.0)
    plan.add_argument("--helix_disable_communication_estimator", action="store_true")
    plan.add_argument("--helix_disable_importance_sort", action="store_true")
    plan.add_argument("--helix_dry_run", action="store_true")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    if args.zero_stage != 0:
        parser.error("Helix variable-shape submodels currently require --zero_stage 0")
    if args.helix_submodel_sizes and not args.helix_micro_batch_sizes:
        parser.error("--helix_submodel_sizes requires --helix_micro_batch_sizes")
    if args.helix_micro_batch_sizes and not args.helix_submodel_sizes:
        parser.error("--helix_micro_batch_sizes is only valid with --helix_submodel_sizes")
    if args.helix_ffn_alignment < 1:
        parser.error("--helix_ffn_alignment must be positive")
    if args.helix_max_steps < 0:
        parser.error("--helix_max_steps must be non-negative")
    if args.helix_metrics_warmup_steps < 0:
        parser.error("--helix_metrics_warmup_steps must be non-negative")
    if bool(args.helix_mmlu_path) != bool(args.helix_acceptance_dir):
        parser.error("--helix_mmlu_path and --helix_acceptance_dir must be provided together")
    if args.helix_max_train_seconds < 0 or args.helix_max_successful_steps < 0:
        parser.error("Acceptance budgets must be nonnegative")
    if args.helix_acceptance_dir:
        if args.gradient_accumulation_steps != 1:
            parser.error("Full-model acceptance requires gradient_accumulation_steps=1")
        if not args.helix_profiles_path:
            parser.error("Acceptance runs require --helix_profiles_path for automatic plan search")
        if not 0 < args.helix_target_accuracy <= 1:
            parser.error("Target accuracy must be a fraction in (0,1]")
        if min(args.helix_eval_interval, args.helix_eval_max_length, args.helix_eval_batch_size) < 1 or args.helix_eval_fewshot < 0:
            parser.error("Invalid evaluation interval, batch, fewshot or length")
        if not args.helix_metrics_dir:
            args.helix_metrics_dir = str(Path(args.helix_acceptance_dir) / "metrics")
        # This mode saves reconstructed target/final weights through its controller.
        args.helix_skip_checkpoint = True
    return args


def _initialize_distributed(args):
    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
    deepspeed.init_distributed()
    args.global_rank = torch.distributed.get_rank()
    return device


def _build_dataset(args, tokenizer):
    if getattr(args, "helix_token_cache_path", None):
        from dschat.helix.token_cache import MathTokenCache
        from transformers import default_data_collator
        return MathTokenCache(args.helix_token_cache_path, args.max_seq_len), default_data_collator
    if args.trainset == "math":
        dataset = SupervisedDataset(
            tokenizer=tokenizer,
            data_split=0.8,
            data_path=args.data_path,
            template_variation=True,
        )
        collator = DataCollatorForSupervisedDataset(tokenizer, pipeline_parallelism=False)
    else:
        from datasets import load_from_disk

        dataset = load_from_disk(args.data_path)
        collator = DataCollatorForFinanceDataset(tokenizer, pipeline_parallelism=False)
    return dataset, collator


def _explicit_plan(args, world_size: int, dataset_size: int) -> TrainingPlan:
    sizes = _csv_numbers(args.helix_submodel_sizes, float, "submodel sizes")
    batches = _csv_numbers(args.helix_micro_batch_sizes, int, "micro batch sizes")
    if len(sizes) != world_size or len(batches) != world_size:
        raise ValueError(
            f"explicit plan must contain {world_size} entries; got {len(sizes)} sizes and {len(batches)} batches"
        )
    if any(not 0.0 < size <= 1.0 for size in sizes):
        raise ValueError("every explicit s_i must be in (0, 1]")
    if any(batch < 1 for batch in batches):
        raise ValueError("every explicit b_i must be positive")
    if sum(sizes) < 1.0:
        raise ValueError(f"explicit plan does not cover the full model: sum(s_i)={sum(sizes):.6f}")
    allocations = tuple(
        RankAllocation(
            rank=rank,
            micro_batch_size=batches[rank],
            submodel_size=sizes[rank],
            estimated_compute_seconds=float("nan"),
            estimated_memory_bytes=float("nan"),
        )
        for rank in range(world_size)
    )
    return TrainingPlan(
        allocations=allocations,
        total_batch_size=sum(batches),
        estimated_communication_seconds=float("nan"),
        estimated_step_seconds=float("nan"),
        estimated_total_cost=float("nan"),
        dataset_size=dataset_size,
    )


def _resolve_plan_on_rank_zero(args, dataset_size, full_state, config, world_size):
    if args.helix_plan_path:
        plan = load_plan(args.helix_plan_path)
    elif args.helix_submodel_sizes:
        plan = _explicit_plan(args, world_size, dataset_size)
    else:
        profiles = load_profiles(args.helix_profiles_path)
        if len(profiles) != world_size:
            raise ValueError(f"profile has {len(profiles)} ranks, distributed job has {world_size}")
        for profile in profiles:
            if profile.sequence_length and profile.sequence_length != args.max_seq_len:
                raise ValueError(
                    f"rank {profile.rank} profile seq_len={profile.sequence_length}, requested {args.max_seq_len}"
                )
            if profile.dtype and profile.dtype != args.dtype:
                raise ValueError(f"rank {profile.rank} profile dtype={profile.dtype}, requested {args.dtype}")

        structure = infer_model_structure(full_state, config)
        trainable_names = [
            name
            for name in full_state
            if "embed_tokens" not in name and not name.startswith("lm_head")
        ]
        bandwidth = args.helix_link_bandwidth_gbps * 1e9 / 8.0
        links = {
            (left, right): bandwidth
            for left in range(world_size)
            for right in range(left + 1, world_size)
        }
        communication_cache = {}

        def communication_time(sizes: Sequence[float]) -> float:
            masks, specs = build_structured_masks(
                full_state,
                sizes,
                structure=structure,
                ffn_alignment=args.helix_ffn_alignment,
            )
            key = tuple(
                (len(spec.attention_groups), len(spec.ffn_columns))
                for spec in specs
            )
            if key not in communication_cache:
                communication_plan = build_communication_plan(full_state, masks, trainable_names)
                communication_cache[key] = estimate_cluster_reduce_seconds(
                    communication_plan,
                    full_state,
                    links,
                    communication_dtype_bytes=2 if args.dtype in {"fp16", "bf16"} else 4,
                )
            return communication_cache[key]

        plan = fast_search(
            profiles,
            dataset_size=dataset_size,
            communication_time=None if args.helix_disable_communication_estimator else communication_time,
            max_micro_batch_size=args.helix_max_micro_batch_size,
            max_total_batch_size=args.helix_max_total_batch_size,
            minimum_submodel_size=1.0 / structure.num_key_value_heads,
            memory_slack_bytes=args.helix_memory_slack_gib * 1024**3,
        )

    if len(plan.allocations) != world_size:
        raise ValueError(f"plan has {len(plan.allocations)} ranks, distributed job has {world_size}")
    if [allocation.rank for allocation in plan.allocations] != list(range(world_size)):
        raise ValueError("plan allocations must be ordered by contiguous global rank")
    if sum(plan.submodel_sizes) < 1.0:
        raise ValueError("plan violates full-model coverage: sum(s_i) < 1")
    if args.helix_save_plan_path:
        save_plan(args.helix_save_plan_path, plan)
    return plan


def _broadcast_plan(args, dataset_size, full_state, config) -> TrainingPlan:
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    payload = [None]
    if rank == 0:
        try:
            payload[0] = {
                "plan": _resolve_plan_on_rank_zero(args, dataset_size, full_state, config, world_size)
            }
        except Exception as error:  # broadcast failure instead of stranding peers
            payload[0] = {"error": f"{type(error).__name__}: {error}"}
    torch.distributed.broadcast_object_list(payload, src=0)
    if "error" in payload[0]:
        raise RuntimeError(f"Helix planner failed on rank 0: {payload[0]['error']}")
    return payload[0]["plan"]


def _configure_deepspeed(args, local_batch_size: int, world_size: int):
    config = get_train_ds_config(
        offload=args.offload,
        dtype=args.dtype,
        stage=args.zero_stage,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    config["train_micro_batch_size_per_gpu"] = local_batch_size
    # DeepSpeed validates this identity locally. The real heterogeneous global
    # batch is sum(b_i)*GAS and is reported separately.
    config["train_batch_size"] = local_batch_size * world_size * args.gradient_accumulation_steps
    config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    config.pop("activation_checkpointing", None)
    # DeepSpeed gives a JSON scheduler precedence over the client scheduler.
    # Training (including dynamic rebuilds) supplies the requested HF scheduler.
    config.pop("scheduler", None)
    initial_scale_power = getattr(args, "helix_initial_scale_power", None)
    if initial_scale_power is not None and args.dtype == "fp16":
        config["fp16"]["initial_scale_power"] = initial_scale_power
    if "flops_profiler" in config:
        config["flops_profiler"]["enabled"] = False
    if args.dtype == "bf16":
        config.pop("fp16", None)
        config["bf16"] = {"enabled": True}
    return config


def _save_checkpoint(engine, tokenizer, args, sub_folder):
    engine.gather_full_model()
    if args.global_rank == 0:
        save_hf_format(engine, tokenizer, args, sub_folder=sub_folder)
    torch.distributed.barrier()


def main():
    args = parse_args()
    device = _initialize_distributed(args)
    world_size = torch.distributed.get_world_size()
    set_random_seed(args.seed)

    tokenizer = load_hf_tokenizer(
        args.model_name_or_path,
        fast_tokenizer=True,
        model_max_length=args.max_seq_len,
        padding_side="right",
    )
    dataset, collator = _build_dataset(args, tokenizer)

    # Build the canonical full model before planning so exact Cluster-Reduce
    # volume can be included in Eq.1.
    full_model = create_helix_model(
        AutoModelForCausalLM,
        args.model_name_or_path,
        num_hidden_layers=args.num_hidden_layers,
        dropout=args.dropout,
    )
    if full_model.get_input_embeddings().num_embeddings != len(tokenizer):
        full_model.resize_token_embeddings(len(tokenizer))
    full_state = full_model.state_dict()
    structure = infer_model_structure(full_state, full_model.config)
    plan = _broadcast_plan(args, len(dataset), full_state, full_model.config)
    batch_sizes = plan.batch_sizes
    submodel_sizes = plan.submodel_sizes

    print_rank_0(
        "Helix plan: "
        + json.dumps(
            {
                "b_i": batch_sizes,
                "s_i": submodel_sizes,
                "B_total": sum(batch_sizes),
                "estimated_step_seconds": plan.estimated_step_seconds,
                "estimated_total_cost": plan.estimated_total_cost,
            },
            allow_nan=True,
        ),
        args.global_rank,
    )

    if not args.helix_disable_importance_sort:
        permutations = canonicalize_state_dict_by_importance(
            full_state,
            config=full_model.config,
            structure=structure,
        )
        if args.global_rank == 0 and args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            (Path(args.output_dir) / "importance_permutations.json").write_text(
                json.dumps(permutations, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    masks, specs = build_structured_masks(
        full_state,
        submodel_sizes,
        structure=structure,
        ffn_alignment=args.helix_ffn_alignment,
    )
    rank = args.global_rank
    local_state = slice_state_dict(full_state, masks[rank])
    local_model = create_helix_model(
        AutoModelForCausalLM,
        args.model_name_or_path,
        attention_rate=specs[rank].attention_rate,
        ffn_rate=specs[rank].ffn_rate,
        submodel_ids=serialize_mask(masks[rank]),
        freeze_blocks=[],
        num_hidden_layers=args.num_hidden_layers,
        dropout=args.dropout,
    )
    if local_model.get_input_embeddings().num_embeddings != len(tokenizer):
        local_model.resize_token_embeddings(len(tokenizer))
    validate_state_shapes(local_model, local_state)
    local_model.load_state_dict(local_state, strict=True)
    local_model.get_input_embeddings().requires_grad_(False)
    local_model.get_output_embeddings().requires_grad_(False)

    trainable_names = [name for name, parameter in local_model.named_parameters() if parameter.requires_grad]
    communication_plan = build_communication_plan(full_state, masks, trainable_names)
    local_rectangles = communication_plan.local_plan(rank)

    if args.helix_dry_run:
        print(
            f"rank={rank} dry-run passed: b_i={batch_sizes[rank]}, "
            f"attention_rate={specs[rank].attention_rate:.6f}, "
            f"ffn_rate={specs[rank].ffn_rate:.6f}, "
            f"groups={len(communication_plan.overlap_groups)}, "
            f"clusters={len(communication_plan.clusters)}"
        )
        torch.distributed.barrier()
        return

    sampler = HeterogeneousDistributedBatchSampler(
        dataset_size=len(dataset),
        batch_sizes=batch_sizes,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
        drop_last=False,
    )
    train_dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collator)
    if len(train_dataloader) == 0:
        raise ValueError("training dataset is empty")

    local_batch_size = batch_sizes[rank]
    args.per_device_train_batch_size = local_batch_size
    args.global_batch_size = sum(batch_sizes) * args.gradient_accumulation_steps
    ds_config = _configure_deepspeed(args, local_batch_size, world_size)
    optimizer_groups = get_optimizer_grouped_parameters(local_model, args.weight_decay, args.learning_rate)
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        foreach=False,
    )
    updates_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * updates_per_epoch,
    )

    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=local_model,
        optimizer=optimizer,
        training_data=None,
        args=args,
        config=ds_config,
        lr_scheduler=scheduler,
        dist_init_required=False,
        head_prune_rate=specs[rank].attention_rate,
        prune_rate=specs[rank].ffn_rate,
        roll=0.0,
        full_model=full_model,
        groups_to_offset_rectangles=local_rectangles,
        all_overlap_groups_ls=communication_plan.overlap_groups,
        complementary_rank_ls=communication_plan.gather_ranks,
        complementary_groups_ranks_to_rectangles=communication_plan.reconstruction_global_by_rank,
        complementary_groups_to_offset_rectangles=communication_plan.reconstruction_local_by_rank,
    )
    if args.gradient_checkpointing:
        engine.gradient_checkpointing_enable()

    print_rank_0(
        f"***** Helix training: {len(train_dataloader)} microsteps/epoch, "
        f"heterogeneous global batch={args.global_batch_size} *****",
        args.global_rank,
    )
    global_microstep = 0
    for epoch in range(args.num_train_epochs):
        sampler.set_epoch(epoch)
        engine.train()
        for step, batch in enumerate(train_dataloader):
            started = time.perf_counter()
            batch = to_device(batch, engine.device)
            outputs = engine(**batch, use_cache=False)
            loss = outputs.loss
            engine.backward(loss)
            engine.step()
            global_microstep += 1

            if args.print_loss and (step == 0 or global_microstep % 10 == 0):
                print_rank_0(
                    f"epoch={epoch} step={step} loss={loss.detach().float().item():.6f} "
                    f"step_seconds={time.perf_counter() - started:.4f}",
                    args.global_rank,
                )
            if (
                args.output_dir
                and args.checkpoint_interval > 0
                and global_microstep % args.checkpoint_interval == 0
            ):
                _save_checkpoint(engine, tokenizer, args, f"step_{global_microstep}")

        engine.tput_timer.update_epoch_count()
        if args.output_dir:
            _save_checkpoint(engine, tokenizer, args, f"epoch_{epoch + 1}")


if __name__ == "__main__":
    main()
