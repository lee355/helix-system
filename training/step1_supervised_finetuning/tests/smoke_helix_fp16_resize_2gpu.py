"""Two-GPU NCCL smoke test for shape-changing complete Adam migration."""

import argparse
from collections import OrderedDict

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


GLOBAL_WEIGHT = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10


class ToyModel(nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.weight = nn.Parameter(GLOBAL_WEIGHT.index_select(0, torch.tensor(rows)).clone())

    @property
    def device(self):
        return self.weight.device

    def forward(self, inputs):
        return (inputs @ self.weight.t()).square().mean()


def _engine(args, full_model, rows):
    model = ToyModel(rows)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        foreach=False,
    )
    config = {
        "train_micro_batch_size_per_gpu": 2,
        "train_batch_size": 4,
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
        complementary_rank_ls=[0, 1],
        complementary_groups_ranks_to_rectangles={},
        complementary_groups_to_offset_rectangles={},
    )
    return engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    install_bounded_paper_deepspeed_runtime()
    deepspeed.init_distributed()
    rank = torch.distributed.get_rank()
    try:
        old_rows = ([0, 1], [2, 3])
        new_rows = ([0, 2, 3], [1])
        full_model = ToyModel([0, 1, 2, 3])
        old_engine = _engine(args, full_model, old_rows[rank])
        inputs = torch.randn(2, 4, device=old_engine.device, dtype=torch.float16)
        old_engine.zero_grad()
        old_engine.backward(old_engine(inputs))
        old_engine.step()
        snapshot = capture_fp16_adam_snapshot(old_engine)

        gathered = [None, None]
        torch.distributed.all_gather_object(gathered, snapshot.parameters["weight"])
        expected = {}
        for field in ("master_parameter", "exp_avg", "exp_avg_sq"):
            tensor = torch.empty(4, 4)
            for source_rank in range(2):
                tensor[torch.tensor(old_rows[source_rank])] = getattr(
                    gathered[source_rank], field
                )
            expected[field] = tensor.index_select(0, torch.tensor(new_rows[rank]))

        old_masks = [
            OrderedDict(weight=(torch.tensor(rows), torch.arange(4)))
            for rows in old_rows
        ]
        new_masks = [
            OrderedDict(weight=(torch.tensor(rows), torch.arange(4)))
            for rows in new_rows
        ]
        copy_plan = build_dynamic_copy_plan(
            {"weight": GLOBAL_WEIGHT},
            old_masks,
            new_masks,
            ["weight"],
        )
        new_engine = _engine(args, full_model, new_rows[rank])
        restore_fp16_adam_from_rectangles(new_engine, copy_plan, snapshot)
        live = _live_parameter_map(new_engine)["weight"]
        torch.testing.assert_close(live.master_parameter.cpu(), expected["master_parameter"])
        torch.testing.assert_close(live.exp_avg.cpu(), expected["exp_avg"])
        torch.testing.assert_close(live.exp_avg_sq.cpu(), expected["exp_avg_sq"])
        torch.testing.assert_close(
            live.fp16_parameter.cpu(),
            expected["master_parameter"].half(),
        )
        assert float(live.step.item()) == float(snapshot.parameters["weight"].step.item())
        torch.distributed.barrier()
        if rank == 0:
            print("2-GPU NCCL shape-changing complete-Adam migration passed", flush=True)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
