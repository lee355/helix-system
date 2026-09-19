"""Epoch-boundary live Helix reconfiguration with complete FP16 Adam state."""

from __future__ import annotations

import copy
import gc
import math
from dataclasses import replace
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, get_scheduler

from .communication import build_communication_plan
from .deepspeed_adam_state import (
    capture_fp16_adam_snapshot,
    restore_fp16_adam_from_rectangles,
)
from .dynamic_masks import resize_rank_masks
from .dynamic_rectangles import build_dynamic_copy_plan
from .masking import build_structured_masks, infer_model_structure, serialize_mask
from .planner import RankAllocation, TrainingPlan
from .planner_bounded import _profile_bounded_candidates
from .profiling import load_profiles


class DynamicRegistry:
    def __init__(self):
        self.plan = None
        self.sampler = None
        self.controller = None
        self.initial_local_state = None


REGISTRY = DynamicRegistry()


def _same_quantized_specs(left, right) -> bool:
    return all(
        a.attention_groups == b.attention_groups
        and a.ffn_columns == b.ffn_columns
        for a, b in zip(left, right)
    )


class DynamicEngineProxy:
    def __init__(self, engine, controller):
        object.__setattr__(self, "_engine", engine)
        object.__setattr__(self, "_controller", controller)

    def __getattr__(self, name):
        engine = object.__getattribute__(self, "_engine")
        if engine is None:
            raise RuntimeError("Helix engine is between atomic configurations")
        return getattr(engine, name)

    def __setattr__(self, name, value):
        if name in {"_engine", "_controller"}:
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_engine"), name, value)

    def __call__(self, *args, **kwargs):
        return object.__getattribute__(self, "_engine")(*args, **kwargs)

    def step(self, *args, **kwargs):
        result = object.__getattribute__(self, "_engine").step(*args, **kwargs)
        object.__getattribute__(self, "_controller").after_step()
        return result


class DynamicHelixController:
    """Monitor memory and commit complete-state transitions at epoch boundaries."""

    def __init__(
        self,
        *,
        engine,
        scheduler,
        original_initialize,
        implementation,
        args,
        plan: TrainingPlan,
        sampler,
        full_model,
        model_factory,
    ):
        if args.dtype != "fp16" or args.zero_stage != 0:
            raise ValueError("dynamic Helix migration currently requires FP16 and ZeRO stage 0")
        if args.gradient_accumulation_steps != 1:
            raise ValueError("dynamic Helix migration requires gradient_accumulation_steps=1")
        profiles_path = args.helix_dynamic_profiles_path or args.helix_profiles_path
        if not profiles_path:
            raise ValueError(
                "dynamic adjustment requires --helix_dynamic_profiles_path "
                "when the initial plan did not come from profiles"
            )

        self.original_initialize = original_initialize
        self.implementation = implementation
        self.args = args
        self.current_plan = plan
        self.sampler = sampler
        self.full_model = full_model
        self.model_factory = model_factory
        self.profiles = load_profiles(profiles_path)
        if len(self.profiles) != torch.distributed.get_world_size():
            raise ValueError("dynamic profiles do not match distributed world size")
        self.full_state = self.full_model.state_dict()
        self.structure = infer_model_structure(self.full_state, self.full_model.config)
        self.current_masks, self.current_specs = build_structured_masks(
            self.full_state,
            self.current_plan.submodel_sizes,
            structure=self.structure,
            ffn_alignment=self.args.helix_ffn_alignment,
        )
        self.scheduler = scheduler
        self.proxy = None
        self.pending_plan: Optional[TrainingPlan] = None
        self.completed_microsteps = 0
        self.adjustments = 0
        self.processed_global_samples = 0
        self.initial_total_steps = self.args.num_train_epochs * math.ceil(
            len(self.sampler) / self.args.gradient_accumulation_steps
        )
        self.baseline_budgets = self._collect_available_budgets(engine.device)

    def attach_proxy(self, proxy) -> None:
        self.proxy = proxy

    @staticmethod
    def _local_available_budget(device) -> float:
        free, total = torch.cuda.mem_get_info(device)
        reserved = torch.cuda.memory_reserved(device)
        return float(min(total, free + reserved))

    def _collect_available_budgets(self, device):
        local = torch.tensor(
            [self._local_available_budget(device)],
            dtype=torch.float64,
            device=device,
        )
        gathered = [torch.empty_like(local) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered, local)
        return [float(item.item()) for item in gathered]

    def _adjusted_plan(self, budgets) -> Optional[TrainingPlan]:
        threshold = self.args.helix_dynamic_memory_threshold_gib * 1024**3
        changed = [
            rank
            for rank, (old, new) in enumerate(zip(self.baseline_budgets, budgets))
            if abs(new - old) >= threshold
        ]
        if not changed:
            return None

        allocations = list(self.current_plan.allocations)
        for rank in changed:
            profile = replace(self.profiles[rank], memory_budget_bytes=budgets[rank])
            candidates = _profile_bounded_candidates(
                profile,
                self.args.helix_max_micro_batch_size,
                1.0 / self.structure.num_key_value_heads,
                self.args.helix_memory_slack_gib * 1024**3,
            )
            previous = allocations[rank]
            previous_compute = previous.estimated_compute_seconds
            if not math.isfinite(previous_compute):
                previous_compute = profile.compute.predict(
                    previous.micro_batch_size,
                    previous.submodel_size,
                )
            replacement = min(
                candidates,
                key=lambda item: (
                    abs(item.estimated_compute_seconds - previous_compute),
                    -item.submodel_size,
                    -item.micro_batch_size,
                ),
            )
            allocations[rank] = replacement

        if all(
            old.micro_batch_size == new.micro_batch_size
            and abs(old.submodel_size - new.submodel_size) < 1e-12
            for old, new in zip(self.current_plan.allocations, allocations)
        ):
            return None
        total_batch = sum(item.micro_batch_size for item in allocations)
        total_size = sum(item.submodel_size for item in allocations)
        compute = max(item.estimated_compute_seconds for item in allocations)
        return TrainingPlan(
            allocations=tuple(allocations),
            total_batch_size=total_batch,
            estimated_communication_seconds=float("nan"),
            estimated_step_seconds=compute,
            estimated_total_cost=(
                self.current_plan.dataset_size / total_batch * compute / total_size
            ),
            dataset_size=self.current_plan.dataset_size,
        )

    def after_step(self) -> None:
        self.completed_microsteps += 1
        self.processed_global_samples += sum(self.current_plan.batch_sizes)
        engine = self.proxy._engine
        # Correct the custom engine's homogeneous global-sample accounting.
        engine.global_samples = self.processed_global_samples
        interval = self.args.helix_dynamic_check_interval
        if interval <= 0 or self.pending_plan is not None:
            return
        if self.completed_microsteps % interval:
            return
        if 0 <= self.args.helix_dynamic_max_adjustments <= self.adjustments:
            return

        budgets = self._collect_available_budgets(engine.device)
        payload = [None]
        if torch.distributed.get_rank() == 0:
            try:
                payload[0] = {"plan": self._adjusted_plan(budgets)}
            except Exception as error:
                payload[0] = {"error": f"{type(error).__name__}: {error}"}
        torch.distributed.broadcast_object_list(payload, src=0)
        if "error" in payload[0]:
            raise RuntimeError(f"dynamic Helix planning failed: {payload[0]['error']}")
        self.pending_plan = payload[0]["plan"]
        self.baseline_budgets = budgets
        if self.pending_plan is not None and torch.distributed.get_rank() == 0:
            print(
                "Helix queued epoch-boundary adjustment: "
                f"b_i={self.pending_plan.batch_sizes}, "
                f"s_i={self.pending_plan.submodel_sizes}",
                flush=True,
            )

    def _incremental_masks(self, new_plan):
        masks = self.current_masks
        specs = self.current_specs
        changed = [
            rank
            for rank, (old, new) in enumerate(
                zip(self.current_plan.allocations, new_plan.allocations)
            )
            if abs(old.submodel_size - new.submodel_size) >= 1e-12
        ]
        # Grow first so a subsequent shrink has the largest possible coverage.
        ordered = sorted(
            changed,
            key=lambda rank: (
                new_plan.allocations[rank].submodel_size
                < self.current_plan.allocations[rank].submodel_size,
                rank,
            ),
        )
        for rank in ordered:
            update = resize_rank_masks(
                self.full_state,
                masks,
                specs,
                changed_rank=rank,
                new_size=new_plan.allocations[rank].submodel_size,
                structure=self.structure,
                ffn_alignment=self.args.helix_ffn_alignment,
            )
            masks, specs = list(update.masks), list(update.specs)
        return masks, specs

    def _build_new_engine(self, new_plan, new_masks, new_specs):
        rank = torch.distributed.get_rank()
        spec = new_specs[rank]
        local_model = self.model_factory(
            AutoModelForCausalLM,
            self.args.model_name_or_path,
            attention_rate=spec.attention_rate,
            ffn_rate=spec.ffn_rate,
            submodel_ids=serialize_mask(new_masks[rank]),
            freeze_blocks=[],
            num_hidden_layers=self.args.num_hidden_layers,
            dropout=self.args.dropout,
        )
        vocab_size = self.full_model.get_input_embeddings().num_embeddings
        if local_model.get_input_embeddings().num_embeddings != vocab_size:
            local_model.resize_token_embeddings(vocab_size)

        parameter_names = [
            name for name, parameter in local_model.named_parameters()
            if parameter.requires_grad
        ]
        communication = build_communication_plan(
            self.full_state,
            new_masks,
            parameter_names,
        )
        local_batch = new_plan.batch_sizes[rank]
        self.args.per_device_train_batch_size = local_batch
        self.args.global_batch_size = sum(new_plan.batch_sizes)
        config = self.implementation._configure_deepspeed(
            self.args,
            local_batch,
            torch.distributed.get_world_size(),
        )

        from dschat.utils.utils import get_optimizer_grouped_parameters

        optimizer = torch.optim.AdamW(
            get_optimizer_grouped_parameters(
                local_model,
                self.args.weight_decay,
                self.args.learning_rate,
            ),
            lr=self.args.learning_rate,
            betas=(0.9, 0.95),
            foreach=False,
        )
        scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.initial_total_steps,
        )
        engine, _, _, scheduler = self.original_initialize(
            model=local_model,
            optimizer=optimizer,
            training_data=None,
            args=self.args,
            config=config,
            lr_scheduler=scheduler,
            dist_init_required=False,
            head_prune_rate=spec.attention_rate,
            prune_rate=spec.ffn_rate,
            roll=0.0,
            full_model=self.full_model,
            groups_to_offset_rectangles=communication.local_plan(rank),
            all_overlap_groups_ls=communication.overlap_groups,
            complementary_rank_ls=communication.gather_ranks,
            complementary_groups_ranks_to_rectangles=(
                communication.reconstruction_global_by_rank
            ),
            complementary_groups_to_offset_rectangles=(
                communication.reconstruction_local_by_rank
            ),
        )
        engine.helix_overlap_group_clusters = communication.clusters
        if self.args.gradient_checkpointing:
            engine.gradient_checkpointing_enable()
        return engine, scheduler

    def apply_pending_at_epoch_boundary(self) -> None:
        new_plan = self.pending_plan
        if new_plan is None:
            return
        if self.proxy is None:
            raise RuntimeError("dynamic controller has no engine proxy")
        new_masks, new_specs = self._incremental_masks(new_plan)
        masks_changed = not _same_quantized_specs(self.current_specs, new_specs)
        if not masks_changed:
            self.sampler.reconfigure(new_plan.batch_sizes)
            self.current_plan = new_plan
            self.current_masks = new_masks
            self.current_specs = new_specs
            self.pending_plan = None
            self.adjustments += 1
            return

        old_engine = self.proxy._engine
        copy_plan = build_dynamic_copy_plan(
            self.full_state,
            self.current_masks,
            new_masks,
            list(capture_name for capture_name, _ in old_engine.module.named_parameters()),
        )
        torch.cuda.synchronize(old_engine.device)
        snapshot = capture_fp16_adam_snapshot(old_engine, self.scheduler)
        torch.distributed.barrier()

        # The trainer still has a reference to the initial local_model. Move
        # that object to meta so it cannot pin the old GPU storage after the
        # engine proxy switches to a newly constructed module.
        old_engine.module.to("meta")
        self.proxy._engine = None
        self.scheduler = None
        del old_engine
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.barrier()

        new_engine, new_scheduler = self._build_new_engine(
            new_plan,
            new_masks,
            new_specs,
        )
        restore_fp16_adam_from_rectangles(
            new_engine,
            copy_plan,
            snapshot,
            scheduler=new_scheduler,
        )
        self.proxy._engine = new_engine
        self.scheduler = new_scheduler
        self.sampler.reconfigure(new_plan.batch_sizes)
        self.current_plan = new_plan
        self.current_masks = new_masks
        self.current_specs = new_specs
        self.pending_plan = None
        self.adjustments += 1
        del snapshot
        gc.collect()


def install_dynamic_hooks(implementation, model_factory) -> None:
    """Patch the canonical trainer's plan/sampler/initialize seams once."""

    if getattr(implementation, "_helix_dynamic_hooks_installed", False):
        return
    original_sampler = implementation.HeterogeneousDistributedBatchSampler
    original_broadcast_plan = implementation._broadcast_plan
    original_slice_state = implementation.slice_state_dict
    original_initialize = implementation.deepspeed.initialize

    class RegisteredSampler(original_sampler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            REGISTRY.sampler = self

        def reconfigure(self, batch_sizes):
            batch_sizes = tuple(int(value) for value in batch_sizes)
            if len(batch_sizes) != self.world_size or any(value < 1 for value in batch_sizes):
                raise ValueError("dynamic batch sizes must be positive and match world size")
            self.batch_sizes = batch_sizes
            offsets = [0]
            for value in batch_sizes:
                offsets.append(offsets[-1] + value)
            self._rank_offsets = tuple(offsets)
            self.global_batch_size = offsets[-1]

        def set_epoch(self, epoch):
            if REGISTRY.controller is not None:
                REGISTRY.controller.apply_pending_at_epoch_boundary()
            return super().set_epoch(epoch)

    def broadcast_plan(*args, **kwargs):
        plan = original_broadcast_plan(*args, **kwargs)
        REGISTRY.plan = plan
        return plan

    def slice_state(*args, **kwargs):
        state = original_slice_state(*args, **kwargs)
        REGISTRY.initial_local_state = state
        return state

    def initialize(*args, **kwargs):
        result = original_initialize(*args, **kwargs)
        args_object = kwargs.get("args")
        if REGISTRY.initial_local_state is not None:
            REGISTRY.initial_local_state.clear()
            REGISTRY.initial_local_state = None
        if args_object is None or args_object.helix_dynamic_check_interval <= 0:
            return result
        if REGISTRY.plan is None or REGISTRY.sampler is None:
            raise RuntimeError("dynamic trainer did not capture its plan/sampler")
        controller = DynamicHelixController(
            engine=result[0],
            scheduler=result[3],
            original_initialize=original_initialize,
            implementation=implementation,
            args=args_object,
            plan=REGISTRY.plan,
            sampler=REGISTRY.sampler,
            full_model=kwargs["full_model"],
            model_factory=model_factory,
        )
        proxy = DynamicEngineProxy(result[0], controller)
        controller.attach_proxy(proxy)
        REGISTRY.controller = controller
        # optimizer/scheduler locals in the monolithic trainer are unused after
        # initialize. Returning None prevents them from pinning the old engine
        # across a live rebuild; the proxy owns the active runtime.
        return proxy, None, result[2], None

    implementation.HeterogeneousDistributedBatchSampler = RegisteredSampler
    implementation._broadcast_plan = broadcast_plan
    implementation.slice_state_dict = slice_state
    implementation.deepspeed.initialize = initialize
    implementation._helix_dynamic_hooks_installed = True
