"""Failure-consensus and process-group lifecycle for dynamic reconfiguration."""

from __future__ import annotations

import gc

import torch
from transformers import AutoModelForCausalLM, get_scheduler

from . import dynamic_controller
from .communication import build_communication_plan
from .deepspeed_adam_state import (
    _live_parameter_map,
    capture_fp16_adam_snapshot,
    restore_fp16_adam_from_rectangles,
)
from .dynamic_controller import _same_quantized_specs
from .dynamic_rectangles import build_dynamic_copy_plan
from .masking import serialize_mask


def _collective_error(stage: str, error) -> None:
    local = None if error is None else f"rank {torch.distributed.get_rank()}: {type(error).__name__}: {error}"
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, local)
    errors = [item for item in gathered if item is not None]
    if errors:
        raise RuntimeError(f"dynamic Helix {stage} failed collectively: " + " | ".join(errors))


def _old_custom_groups(engine):
    groups = []
    seen = set()
    for process_group in list(engine.all_overlap_groups.values()) + [engine.complementary_group]:
        identity = id(process_group)
        if identity not in seen:
            seen.add(identity)
            groups.append(process_group)
    return groups


def _destroy_custom_groups(groups) -> None:
    non_member = torch.distributed.GroupMember.NON_GROUP_MEMBER
    for process_group in reversed(groups):
        if process_group is None or process_group == non_member:
            continue
        torch.distributed.destroy_process_group(process_group)
    torch.distributed.barrier()


class SafeDynamicHelixController(dynamic_controller.DynamicHelixController):
    def _construct_local_model(self, new_masks, new_specs):
        rank = torch.distributed.get_rank()
        spec = new_specs[rank]
        model = self.model_factory(
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
        if model.get_input_embeddings().num_embeddings != vocab_size:
            model.resize_token_embeddings(vocab_size)
        return model

    def _initialize_prebuilt_model(self, local_model, new_plan, new_masks, new_specs):
        rank = torch.distributed.get_rank()
        spec = new_specs[rank]
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

    def _update_batch_metadata(self, plan):
        local_batch = plan.batch_sizes[torch.distributed.get_rank()]
        engine = self.proxy._engine
        engine.set_train_micro_batch_size(local_batch)
        self.args.per_device_train_batch_size = local_batch
        self.args.global_batch_size = sum(plan.batch_sizes)

    def apply_pending_at_epoch_boundary(self) -> None:
        new_plan = self.pending_plan
        if new_plan is None:
            return

        new_masks = new_specs = copy_plan = None
        metadata_error = None
        try:
            new_masks, new_specs = self._incremental_masks(new_plan)
            if not _same_quantized_specs(self.current_specs, new_specs):
                copy_plan = build_dynamic_copy_plan(
                    self.full_state,
                    self.current_masks,
                    new_masks,
                    [name for name, _ in self.proxy._engine.module.named_parameters()],
                )
        except Exception as error:
            metadata_error = error
        _collective_error("metadata preparation", metadata_error)

        if _same_quantized_specs(self.current_specs, new_specs):
            self.sampler.reconfigure(new_plan.batch_sizes)
            self._update_batch_metadata(new_plan)
            self.current_plan = new_plan
            self.current_masks = new_masks
            self.current_specs = new_specs
            self.pending_plan = None
            self.adjustments += 1
            return

        old_engine = self.proxy._engine
        validation_error = None
        try:
            _live_parameter_map(old_engine)
        except Exception as error:
            validation_error = error
        _collective_error("old optimizer validation", validation_error)

        snapshot = None
        snapshot_error = None
        try:
            torch.cuda.synchronize(old_engine.device)
            snapshot = capture_fp16_adam_snapshot(old_engine, self.scheduler)
        except Exception as error:
            snapshot_error = error
        _collective_error("CPU optimizer snapshot", snapshot_error)

        local_model = None
        model_error = None
        try:
            local_model = self._construct_local_model(new_masks, new_specs)
        except Exception as error:
            model_error = error
        _collective_error("new local model construction", model_error)

        old_groups = _old_custom_groups(old_engine)
        torch.distributed.barrier()
        old_engine.module.to("meta")
        self.proxy._engine = None
        self.scheduler = None
        del old_engine
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.barrier()

        # DeepSpeed initialize contains its own WORLD barriers/new_group calls;
        # all rank-local work that can fail safely has already reached consensus.
        new_engine, new_scheduler = self._initialize_prebuilt_model(
            local_model,
            new_plan,
            new_masks,
            new_specs,
        )
        new_validation_error = None
        try:
            _live_parameter_map(new_engine)
        except Exception as error:
            new_validation_error = error
        _collective_error("new optimizer validation", new_validation_error)

        # Keep old group handles until the replacement engine exists, then
        # release them in the same global order before state P2P begins.
        _destroy_custom_groups(old_groups)
        restore_error = None
        try:
            restore_fp16_adam_from_rectangles(
                new_engine,
                copy_plan,
                snapshot,
                scheduler=new_scheduler,
            )
        except Exception as error:
            restore_error = error
        _collective_error("Adam state restore", restore_error)

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


def install_dynamic_safety_controller() -> None:
    dynamic_controller.DynamicHelixController = SafeDynamicHelixController
