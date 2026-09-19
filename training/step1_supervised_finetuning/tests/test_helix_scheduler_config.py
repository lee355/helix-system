"""The explicit training scheduler must survive DeepSpeed configuration selection."""

from types import MethodType, SimpleNamespace
import unittest

import torch
from transformers import get_scheduler
from deepspeed.runtime.engine import DeepSpeedEngine

import helix_profile_ds as profiler
import helix_train_base as training


def make_args(dtype="fp16"):
    return SimpleNamespace(
        offload=False,
        dtype=dtype,
        zero_stage=0,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        weight_decay=0.0,
    )


def select_engine_scheduler(config, optimizer, client_scheduler):
    """Run the real vendored scheduler-selection methods without a CUDA engine."""
    engine = SimpleNamespace(
        optimizer=optimizer,
        basic_optimizer=optimizer,
        scheduler_name=lambda: config.get("scheduler", {}).get("type"),
        scheduler_params=lambda: config.get("scheduler", {}).get("params", {}),
    )
    engine._scheduler_from_config = MethodType(DeepSpeedEngine._scheduler_from_config, engine)
    DeepSpeedEngine._configure_lr_scheduler(engine, client_scheduler)
    return engine.lr_scheduler


class HelixSchedulerConfigTest(unittest.TestCase):
    def test_training_and_profile_configs_leave_scheduler_to_caller(self):
        for dtype in ("fp16", "bf16"):
            with self.subTest(dtype=dtype):
                args = make_args(dtype)
                self.assertNotIn("scheduler", training._configure_deepspeed(args, 4, 3))
                self.assertNotIn("scheduler", profiler._deepspeed_config(args, 3))

    def test_training_uses_requested_constant_learning_rate(self):
        args = make_args()
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=args.learning_rate)
        client_scheduler = get_scheduler(
            "constant", optimizer=optimizer, num_warmup_steps=0, num_training_steps=8
        )
        scheduler = select_engine_scheduler(
            training._configure_deepspeed(args, 4, 3), optimizer, client_scheduler
        )
        self.assertIs(scheduler, client_scheduler)
        for _ in range(4):
            self.assertEqual(optimizer.param_groups[0]["lr"], args.learning_rate)
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            scheduler.step()
            self.assertEqual(optimizer.param_groups[0]["lr"], args.learning_rate)

    def test_profile_keeps_fixed_optimizer_lr_without_implicit_scheduler(self):
        args = make_args()
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=args.learning_rate)
        scheduler = select_engine_scheduler(profiler._deepspeed_config(args, 3), optimizer, None)
        self.assertIsNone(scheduler)
        for _ in range(4):
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            self.assertEqual(optimizer.param_groups[0]["lr"], args.learning_rate)


if __name__ == "__main__":
    unittest.main()
