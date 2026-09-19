#!/usr/bin/env python
"""Canonical paper-aligned Helix full-parameter SFT entry point."""

from __future__ import annotations

import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import List, Sequence

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, get_scheduler

import deepspeed

import helix_train_base as base
from dschat.helix.communication import build_communication_plan
from dschat.helix.cost import estimate_cluster_reduce_seconds
from dschat.helix.data import HeterogeneousDistributedBatchSampler
from dschat.helix.deepspeed_paper import install_paper_deepspeed_runtime
from dschat.helix.masking import (
    build_structured_masks,
    canonicalize_state_dict_by_importance,
    infer_model_structure,
    serialize_mask,
    slice_state_dict,
)
from dschat.helix.model import create_helix_model, validate_state_shapes
from dschat.helix.metrics import HelixMetrics, batch_counts
from dschat.helix.paper_semantics import (
    apply_paper_deepspeed_config,
    install_paper_mask_semantics,
)
from dschat.helix.planner import load_plan, save_plan
from dschat.helix.planner_paper import fast_search
from dschat.helix.precision import restore_rotary_fp32
from dschat.helix.profiling import load_profiles
from dschat.utils.utils import (
    get_optimizer_grouped_parameters,
    load_hf_tokenizer,
    print_rank_0,
    save_hf_format,
    set_random_seed,
    to_device,
)


# Install at the real training module, rather than relying only on a wrapper,
# so direct ``deepspeed helix_train.py ...`` launches cannot bypass safety.
install_paper_mask_semantics()
install_paper_deepspeed_runtime()


def parse_args():
    args = base.parse_args()
    # The historical parser exposed 128-column hardware alignment.  The paper
    # allocation has no such rounding; preserve an explicit experimental
    # override, but make direct canonical launches default to one.
    if not any(
        argument == "--helix_ffn_alignment"
        or argument.startswith("--helix_ffn_alignment=")
        for argument in sys.argv[1:]
    ):
        args.helix_ffn_alignment = 1
    return args


def _profile_search_bounds(profiles, requested_max_batch: int, paper_minimum_size: float):
    sampled_profiles = [profile for profile in profiles if profile.samples]
    if len(sampled_profiles) != len(profiles):
        return requested_max_batch, paper_minimum_size
    sampled_max_batch = min(
        max(sample.micro_batch_size for sample in profile.samples)
        for profile in sampled_profiles
    )
    sampled_minimum_size = max(
        min(sample.submodel_size for sample in profile.samples)
        for profile in sampled_profiles
    )
    return (
        min(requested_max_batch, sampled_max_batch),
        max(paper_minimum_size, sampled_minimum_size),
    )


def _resolve_plan_on_rank_zero(
    args,
    dataset_size: int,
    full_model,
    full_state,
    world_size: int,
):
    if args.helix_plan_path:
        plan = load_plan(args.helix_plan_path)
    elif args.helix_submodel_sizes:
        plan = base._explicit_plan(args, world_size, dataset_size)
    else:
        profiles = load_profiles(args.helix_profiles_path)
        if len(profiles) != world_size:
            raise ValueError(
                f"profile has {len(profiles)} ranks, distributed job has {world_size}"
            )
        for profile in profiles:
            if profile.sequence_length and profile.sequence_length != args.max_seq_len:
                raise ValueError(
                    f"rank {profile.rank} profile seq_len={profile.sequence_length}, "
                    f"requested {args.max_seq_len}"
                )
            if profile.dtype and profile.dtype != args.dtype:
                raise ValueError(
                    f"rank {profile.rank} profile dtype={profile.dtype}, requested {args.dtype}"
                )

        structure = infer_model_structure(full_state, full_model.config)
        parameter_names = [name for name, _ in full_model.named_parameters()]
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
                communication_plan = build_communication_plan(
                    full_state,
                    masks,
                    parameter_names,
                )
                communication_cache[key] = estimate_cluster_reduce_seconds(
                    communication_plan,
                    full_state,
                    links,
                    communication_dtype_bytes=2,
                )
            return communication_cache[key]

        search_max_batch, search_minimum_size = _profile_search_bounds(
            profiles,
            args.helix_max_micro_batch_size,
            1.0 / structure.num_key_value_heads,
        )
        if search_max_batch < args.helix_max_micro_batch_size:
            print(
                f"Helix planner caps b_i at profiled maximum {search_max_batch} "
                f"(requested {args.helix_max_micro_batch_size})",
                flush=True,
            )
        plan = fast_search(
            profiles,
            dataset_size=dataset_size,
            communication_time=(
                None
                if args.helix_disable_communication_estimator
                else communication_time
            ),
            max_micro_batch_size=search_max_batch,
            max_total_batch_size=args.helix_max_total_batch_size,
            minimum_submodel_size=search_minimum_size,
            memory_slack_bytes=args.helix_memory_slack_gib * 1024**3,
        )

    if len(plan.allocations) != world_size:
        raise ValueError(
            f"plan has {len(plan.allocations)} ranks, distributed job has {world_size}"
        )
    if [allocation.rank for allocation in plan.allocations] != list(range(world_size)):
        raise ValueError("plan allocations must be ordered by contiguous global rank")
    if sum(plan.submodel_sizes) + 1e-12 < 1.0:
        raise ValueError("plan violates full-model coverage: sum(s_i) < 1")
    if args.helix_save_plan_path:
        save_plan(args.helix_save_plan_path, plan)
    return plan


def _broadcast_plan(args, dataset_size: int, full_model, full_state):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    payload: List[object] = [None]
    if rank == 0:
        try:
            payload[0] = {
                "plan": _resolve_plan_on_rank_zero(
                    args,
                    dataset_size,
                    full_model,
                    full_state,
                    world_size,
                )
            }
        except Exception as error:
            payload[0] = {"error": f"{type(error).__name__}: {error}"}
    torch.distributed.broadcast_object_list(payload, src=0)
    if "error" in payload[0]:
        raise RuntimeError(f"Helix planner failed on rank 0: {payload[0]['error']}")
    return payload[0]["plan"]


def _configure_deepspeed(args, local_batch_size: int, world_size: int):
    return apply_paper_deepspeed_config(
        base._configure_deepspeed(args, local_batch_size, world_size)
    )


def _save_checkpoint(engine, tokenizer, args, sub_folder: str) -> None:
    engine.gather_full_model()
    if args.global_rank == 0:
        save_hf_format(engine, tokenizer, args, sub_folder=sub_folder)
    torch.distributed.barrier()


def main():
    initialization_started = time.perf_counter()
    args = parse_args()
    base._initialize_distributed(args)
    world_size = torch.distributed.get_world_size()
    set_random_seed(args.seed)

    tokenizer = load_hf_tokenizer(
        args.model_name_or_path,
        fast_tokenizer=True,
        model_max_length=args.max_seq_len,
        padding_side="right",
    )
    dataset, collator = base._build_dataset(args, tokenizer)

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
    search_started = time.perf_counter()
    plan = _broadcast_plan(args, len(dataset), full_model, full_state)
    search_seconds = time.perf_counter() - search_started
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

    initial_digest = None
    if args.helix_acceptance_dir:
        from dschat.helix.acceptance import model_state_digest
        if args.global_rank == 0:
            initial_digest = model_state_digest(full_model)
            acceptance_path = Path(args.helix_acceptance_dir)
            acceptance_path.mkdir(parents=True, exist_ok=True)
            import hashlib
            (acceptance_path / "planning.json").write_text(json.dumps({
                "source": "profiles_search", "profiles_path": args.helix_profiles_path,
                "profiles_sha256": hashlib.sha256(Path(args.helix_profiles_path).read_bytes()).hexdigest(),
                "search_seconds": search_seconds, "batch_sizes": batch_sizes,
                "submodel_sizes": submodel_sizes, "initial_canonical_fp16_sha256": initial_digest,
            }, indent=2) + "\n")

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

    # Full-parameter FT: embeddings and lm_head stay trainable and are
    # represented by the all-rank overlap group.
    trainable_names = [
        name for name, parameter in local_model.named_parameters()
        if parameter.requires_grad
    ]
    communication_plan = build_communication_plan(
        full_state,
        masks,
        trainable_names,
    )
    local_rectangles = communication_plan.local_plan(rank)

    if args.helix_dry_run:
        print(
            f"rank={rank} dry-run passed: b_i={batch_sizes[rank]}, "
            f"attention_rate={specs[rank].attention_rate:.6f}, "
            f"ffn_rate={specs[rank].ffn_rate:.6f}, "
            f"groups={len(communication_plan.overlap_groups)}, "
            f"clusters={len(communication_plan.clusters)}",
            flush=True,
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
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
    )
    if len(train_dataloader) == 0:
        raise ValueError("training dataset is empty")

    local_batch_size = batch_sizes[rank]
    args.per_device_train_batch_size = local_batch_size
    args.global_batch_size = sum(batch_sizes) * args.gradient_accumulation_steps
    ds_config = _configure_deepspeed(args, local_batch_size, world_size)
    optimizer_groups = get_optimizer_grouped_parameters(
        local_model,
        args.weight_decay,
        args.learning_rate,
    )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        foreach=False,
    )
    updates_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
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
        complementary_groups_ranks_to_rectangles=(
            communication_plan.reconstruction_global_by_rank
        ),
        complementary_groups_to_offset_rectangles=(
            communication_plan.reconstruction_local_by_rank
        ),
    )
    rotary_fp32_buffers = restore_rotary_fp32(engine.module)
    engine.helix_rotary_fp32_buffers = rotary_fp32_buffers
    print(
        f"[rank {rank}] Helix RoPE precision: {rotary_fp32_buffers} official "
        "Llama inv_freq buffers are FP32 and persistent=False",
        flush=True,
    )
    engine.helix_overlap_group_clusters = communication_plan.clusters
    if args.gradient_checkpointing:
        engine.gradient_checkpointing_enable()

    print_rank_0(
        f"***** Helix full-parameter training: {len(train_dataloader)} "
        f"microsteps/epoch, heterogeneous global batch={args.global_batch_size} *****",
        args.global_rank,
    )
    metrics = (
        HelixMetrics(
            args.helix_metrics_dir,
            rank=rank,
            warmup_steps=args.helix_metrics_warmup_steps,
            args=args,
            plan=plan,
            ds_config=ds_config,
        )
        if args.helix_metrics_dir else None
    )
    acceptance = None
    if args.helix_acceptance_dir:
        from dschat.helix.acceptance import MathAcceptance
        acceptance = MathAcceptance(args, tokenizer, initial_digest, plan, initialization_started)
    global_microstep = 0
    stopped_reason = None
    try:
        if acceptance:
            metrics.attach_engine(engine)
            if acceptance.begin(engine):
                acceptance.finish(engine, "initial_target_reached")
                return
        for epoch in range(args.num_train_epochs):
            # The live controller can replace the engine in set_epoch().
            if metrics:
                metrics.detach_engine()
            sampler.set_epoch(epoch)
            engine.train()
            batches = iter(train_dataloader)
            for step in range(len(train_dataloader)):
                if metrics:
                    metrics.begin_step(
                        engine, microstep=global_microstep + 1,
                        epoch=epoch, epoch_step=step,
                    )
                batch = next(batches)
                started = time.perf_counter()
                counts = batch_counts(batch) if metrics else None
                batch = to_device(batch, engine.device)
                with metrics.phase("forward") if metrics else nullcontext():
                    outputs = engine(**batch, use_cache=False)
                    loss = outputs.loss
                with metrics.phase("backward") if metrics else nullcontext():
                    engine.backward(loss)
                with metrics.phase("optimizer") if metrics else nullcontext():
                    engine.step()
                global_microstep += 1
                measurement = metrics.end_step(loss=loss, counts=counts) if metrics else None

                if args.print_loss and (step == 0 or global_microstep % 10 == 0):
                    step_seconds = (
                        measurement["step_seconds"] if measurement
                        else time.perf_counter() - started
                    )
                    print_rank_0(
                        f"epoch={epoch} step={step} "
                        f"loss={loss.detach().float().item():.6f} "
                        f"step_seconds={step_seconds:.4f}",
                        args.global_rank,
                    )
                if acceptance:
                    applied = bool(engine.was_step_applied())
                    del outputs, loss, batch
                    stopped_reason = acceptance.after_step(
                        engine, applied=applied,
                        step_seconds=measurement["step_seconds"],
                        global_batch=args.global_batch_size,
                    )
                    if stopped_reason:
                        break
                if (
                    args.output_dir
                    and not args.helix_skip_checkpoint
                    and args.checkpoint_interval > 0
                    and global_microstep % args.checkpoint_interval == 0
                ):
                    _save_checkpoint(
                        engine, tokenizer, args, f"step_{global_microstep}",
                    )
                if args.helix_max_steps and global_microstep >= args.helix_max_steps:
                    break

            engine.tput_timer.update_epoch_count()
            if args.output_dir and not args.helix_skip_checkpoint:
                _save_checkpoint(
                    engine, tokenizer, args, f"epoch_{epoch + 1}",
                )
            if stopped_reason or (args.helix_max_steps and global_microstep >= args.helix_max_steps):
                break
        if acceptance:
            acceptance.finish(engine, stopped_reason or (
                "attempted_step_limit" if args.helix_max_steps and global_microstep >= args.helix_max_steps
                else "epoch_limit"
            ))
    finally:
        if metrics:
            metrics.close()
        if acceptance and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
