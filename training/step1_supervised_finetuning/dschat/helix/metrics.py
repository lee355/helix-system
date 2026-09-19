"""Opt-in local measurements without monitoring collectives or mid-phase syncs.

Gradient-sync CUDA duration includes packing, collective waits and unpacking;
it is not an isolated NCCL kernel duration or measured network wire bytes.
"""

from __future__ import annotations

import json
import math
import time
import weakref
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch


def _json_value(value):
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def describe_all_reduce(tensor, group_size):
    """Algorithm-independent payload and theoretical ring send volume."""
    payload = tensor.numel() * tensor.element_size()
    return {
        "dtype": str(tensor.dtype),
        "numel": tensor.numel(),
        "group_size": group_size,
        "payload_bytes": payload,
        "ring_estimated_send_bytes": (
            2.0 * (group_size - 1) / group_size * payload if group_size > 1 else 0.0
        ),
    }


def batch_counts(batch):
    """Count actual tokens and causal-LM targets, excluding pad/-100 labels."""
    inputs = batch["input_ids"]
    mask = batch.get("attention_mask")
    labels = batch.get("labels")
    return {
        "local_batch_size": int(inputs.shape[0]),
        "sequence_length": int(inputs.shape[-1]),
        "input_slots": int(inputs.numel()),
        "attention_tokens": int(mask.sum().item()) if mask is not None else int(inputs.numel()),
        "valid_shifted_labels": (
            int((labels[..., 1:] != -100).sum().item()) if labels is not None else None
        ),
    }


class HelixMetrics:
    """Per-rank JSONL writer whose lifetime spans optional engine rebuilds."""

    def __init__(self, directory, *, rank, warmup_steps, args, plan, ds_config):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.warmup_steps = warmup_steps
        self.args = args
        self.initial_plan = plan
        self.ds_config = ds_config
        self._file = (self.directory / f"rank_{rank}.jsonl").open("w", encoding="utf-8")
        self._engine = None
        self._original_reduce = None
        self._generation = -1
        self._last_engine_ref = lambda: None
        self._active = False

    def attach_engine(self, engine):
        actual = vars(engine).get("_engine", engine)
        if actual is self._engine:
            return
        self.detach_engine()
        self._engine = actual
        self.device = torch.device(actual.device)
        self.cuda = self.device.type == "cuda"
        new_generation = self._last_engine_ref() is not actual
        if new_generation:
            self._generation += 1
            self._last_engine_ref = weakref.ref(actual)
        original = actual._reduce_non_expert_gradients
        self._original_reduce = original
        self._reduce_had_instance_attribute = "_reduce_non_expert_gradients" in vars(actual)

        def measured_reduce(*args, **kwargs):
            if not self._active:
                return original(*args, **kwargs)
            from deepspeed.runtime import engine as engine_module

            communication = engine_module.dist
            original_all_reduce = communication.all_reduce

            def measured_all_reduce(tensor, *call_args, **call_kwargs):
                group = call_kwargs.get("group", call_args[1] if len(call_args) > 1 else None)
                group_size = communication.get_world_size(group)
                self._all_reduces.append(describe_all_reduce(tensor, group_size))
                return original_all_reduce(tensor, *call_args, **call_kwargs)

            with self.phase("gradient_sync"):
                communication.all_reduce = measured_all_reduce
                try:
                    return original(*args, **kwargs)
                finally:
                    communication.all_reduce = original_all_reduce

        actual._reduce_non_expert_gradients = measured_reduce
        controller = vars(engine).get("_controller")
        plan = getattr(controller, "current_plan", self.initial_plan)
        if new_generation:
            self._write_metadata(actual, plan)

    def _write_metadata(self, engine, plan):
        properties = torch.cuda.get_device_properties(self.device) if self.cuda else None
        metadata = {
            "schema_version": 1,
            "rank": self.rank,
            "world_size": torch.distributed.get_world_size(),
            "engine_generation": self._generation,
            "trainable_parameter_numel": sum(p.numel() for p in engine.module.parameters() if p.requires_grad),
            "total_parameter_numel": sum(p.numel() for p in engine.module.parameters()),
            "local_model_config": engine.module.config.to_dict(),
            "plan": plan,
            "initial_deepspeed_config": self.ds_config,
            "arguments": vars(self.args),
            "device": {
                "torch_device": str(self.device),
                "name": properties.name if properties else "cpu",
                "total_memory_bytes": properties.total_memory if properties else None,
                "multiprocessor_count": properties.multi_processor_count if properties else None,
                "compute_capability": [properties.major, properties.minor] if properties else None,
            },
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "metric_definitions": {
                "step": "One dataloader microstep; canonical paper entry requires gradient accumulation = 1.",
                "warmup": "First warmup_steps microsteps are recorded and flagged, never discarded here.",
                "step_seconds": "Wall time from before next(batch) through synchronized optimizer completion; includes CPU data collation, H2D, forward, backward and optimizer; excludes JSONL writing, checkpointing and prior-step draining.",
                "phase_cuda_seconds": "CUDA events on current stream at phase boundaries, resolved only after step-end device synchronize; backward includes gradient_sync. Do not add overlapping phase durations.",
                "gradient_sync_cuda_seconds": "Whole _reduce_non_expert_gradients phase including gradient pack/unpack and collective dependencies/waits; not isolated network time.",
                "gradient_all_reduce_payload_bytes": "Sum of actual tensor numel * element_size for deepspeed dist.all_reduce calls inside gradient reduction only; singleton calls may have zero wire traffic.",
                "gradient_all_reduce_ring_estimated_send_bytes": "Sum of 2*(group_size-1)/group_size*payload per local call; theoretical ring sends, not measured physical network traffic; excludes receives, protocol overhead and other collectives.",
                "loss": "Local model mean causal-LM loss; aggregate across ranks weighted by valid_shifted_labels if global token-mean loss is needed.",
                "learning_rates": "Optimizer group learning rates at microstep start, before optimizer.step and scheduler.step; learning_rates_after_step records the rates after scheduler advancement.",
                "attention_tokens": "Sum of attention mask; padding excluded.",
                "valid_shifted_labels": "Number of labels[..., 1:] != -100, matching causal-LM shift.",
                "cuda_peak_allocated_bytes": "Peak live CUDA allocator bytes during this microstep, including resident parameters/optimizer state.",
                "optimizer_steps_succeeded": "Increase in engine.global_steps minus increase in engine.skipped_steps; accumulation-only microsteps report zero.",
                "overflow_steps": "Increase in engine.skipped_steps (overflow-skipped optimizer updates).",
                "scope": "No cross-rank metric collectives; utilization and actual network traffic require external monitoring. Instrumentation adds CUDA event and per-step synchronization overhead.",
            },
        }
        suffix = "" if self._generation == 0 else f"_generation_{self._generation}"
        path = self.directory / f"rank_{self.rank}{suffix}_metadata.json"
        path.write_text(json.dumps(_json_value(metadata), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def begin_step(self, engine, *, microstep, epoch, epoch_step):
        self.attach_engine(engine)
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._events = {}
        self._all_reduces = []
        self._record = {
            "rank": self.rank, "microstep": microstep, "epoch": epoch,
            "epoch_step": epoch_step, "engine_generation": self._generation,
            "warmup": microstep <= self.warmup_steps,
            "learning_rates": [float(group["lr"]) for group in self._engine.optimizer.param_groups],
        }
        self._previous_global_steps = int(self._engine.global_steps)
        self._previous_skipped_steps = int(self._engine.skipped_steps)
        self._active = True
        self._started_unix = time.time()
        self._started = time.perf_counter()

    @contextmanager
    def phase(self, name):
        if not self.cuda:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(self.device))
        try:
            yield
        finally:
            end.record(torch.cuda.current_stream(self.device))
            self._events.setdefault(name, []).append((start, end))

    def end_step(self, *, loss, counts):
        if self.cuda:
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - self._started
        finished_unix = time.time()
        self._active = False
        attempted = int(self._engine.global_steps) - self._previous_global_steps
        skipped = int(self._engine.skipped_steps) - self._previous_skipped_steps
        record = {
            **self._record,
            **counts,
            "step_seconds": elapsed,
            "started_unix": self._started_unix,
            "finished_unix": finished_unix,
            "loss": float(loss.detach().float().item()),
            "learning_rates_after_step": [float(group["lr"]) for group in self._engine.optimizer.param_groups],
            "phase_cuda_seconds": {
                name: sum(start.elapsed_time(end) for start, end in events) / 1000.0
                for name, events in self._events.items()
            },
            "gradient_all_reduce_calls": self._all_reduces,
            "gradient_all_reduce_payload_bytes": sum(call["payload_bytes"] for call in self._all_reduces),
            "gradient_all_reduce_ring_estimated_send_bytes": sum(call["ring_estimated_send_bytes"] for call in self._all_reduces),
            "optimizer_steps_attempted": attempted,
            "optimizer_steps_succeeded": attempted - skipped,
            "overflow_steps": skipped,
            "optimizer_steps_attempted_total": int(self._engine.global_steps),
            "optimizer_steps_succeeded_total": int(self._engine.global_steps) - int(self._engine.skipped_steps),
            "overflow_steps_total": int(self._engine.skipped_steps),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.cuda else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.cuda else None,
        }
        record["gradient_sync_cuda_seconds"] = record["phase_cuda_seconds"].get("gradient_sync", 0.0) if self.cuda else None
        self._file.write(json.dumps(_json_value(record), allow_nan=False) + "\n")
        self._file.flush()
        return record

    def detach_engine(self):
        # Release bound methods before an epoch-boundary engine replacement.
        if self._engine is not None:
            if self._reduce_had_instance_attribute:
                self._engine._reduce_non_expert_gradients = self._original_reduce
            else:
                delattr(self._engine, "_reduce_non_expert_gradients")
        self._engine = None
        self._original_reduce = None
        self._active = False

    def close(self):
        self.detach_engine()
        self._file.close()
