"""Single-GPU smoke test for the real ours_math FP16 optimizer adapter."""

import argparse

import torch
from torch import nn

import deepspeed

from dschat.helix.deepspeed_adam_state import (
    _live_parameter_map,
    capture_fp16_adam_snapshot,
    restore_fp16_adam_from_rectangles,
)
from dschat.helix.deepspeed_paper_v2 import install_bounded_paper_deepspeed_runtime
from dschat.helix.dynamic_rectangles import build_dynamic_copy_plan


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10
        )

    @property
    def device(self):
        return self.weight.device

    def forward(self, inputs):
        return (inputs @ self.weight).square().mean()


def _engine(args, full_model):
    model = ToyModel()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        foreach=False,
    )
    config = {
        "train_micro_batch_size_per_gpu": 2,
        "train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 0.0,
        "zero_allow_untested_optimizer": True,
        "zero_optimization": {"stage": 0},
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 8,
            "loss_scale_window": 100,
        },
    }
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=config,
        dist_init_required=False,
        head_prune_rate=1.0,
        prune_rate=1.0,
        roll=0.0,
        full_model=full_model,
        groups_to_offset_rectangles={},
        all_overlap_groups_ls=[],
        complementary_rank_ls=[0],
        complementary_groups_ranks_to_rectangles={},
        complementary_groups_to_offset_rectangles={},
    )
    return engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    install_bounded_paper_deepspeed_runtime()
    deepspeed.init_distributed()
    try:
        full_model = ToyModel()
        old_engine = _engine(args, full_model)
        inputs = torch.randn(2, 4, device=old_engine.device, dtype=torch.float16)
        old_engine.zero_grad()
        old_engine.backward(old_engine(inputs))
        old_engine.step()
        snapshot = capture_fp16_adam_snapshot(old_engine)

        reference = {"weight": full_model.weight.detach().cpu()}
        full_mask = {"weight": (torch.arange(4), torch.arange(4))}
        copy_plan = build_dynamic_copy_plan(
            reference,
            [full_mask],
            [full_mask],
            ["weight"],
        )
        new_engine = _engine(args, full_model)
        restore_fp16_adam_from_rectangles(new_engine, copy_plan, snapshot)
        live = _live_parameter_map(new_engine)["weight"]
        saved = snapshot.parameters["weight"]
        torch.testing.assert_close(live.master_parameter.cpu(), saved.master_parameter)
        torch.testing.assert_close(live.exp_avg.cpu(), saved.exp_avg)
        torch.testing.assert_close(live.exp_avg_sq.cpu(), saved.exp_avg_sq)
        torch.testing.assert_close(live.step.cpu(), saved.step)
        torch.testing.assert_close(
            live.fp16_parameter.cpu(),
            saved.master_parameter.to(dtype=torch.float16),
        )
        assert new_engine.global_steps == old_engine.global_steps
        if torch.distributed.get_rank() == 0:
            print("ours_math FP16 complete-Adam restore smoke test passed", flush=True)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
