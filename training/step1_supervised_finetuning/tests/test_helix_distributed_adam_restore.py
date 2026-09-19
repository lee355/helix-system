import os
import tempfile
import unittest
from collections import OrderedDict

import torch
import torch.multiprocessing as mp

from dschat.helix.deepspeed_adam_state import (
    FP16AdamSnapshot,
    LocalFP16AdamParameter,
    restore_fp16_adam_from_rectangles,
)
from dschat.helix.dynamic_rectangles import build_dynamic_copy_plan


class _InnerAdam:
    def __init__(self, master):
        self.param_groups = [{"params": [master], "lr": 0.1}]
        self.state = {
            master: {
                "step": torch.tensor(1.0),
                "exp_avg": torch.zeros_like(master),
                "exp_avg_sq": torch.zeros_like(master),
            }
        }


class FP16_UnfusedOptimizer:
    def __init__(self, fp16, master):
        self.fp16_groups = [[fp16]]
        self.fp32_groups = [[master]]
        self.optimizer = _InnerAdam(master)

    def zero_grad(self, set_to_none=True):
        for parameter in self.fp16_groups[0]:
            parameter.grad = None


class _Model(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(size, dtype=torch.float16))


class _Engine:
    def __init__(self, size):
        self.module = _Model(size)
        master = torch.nn.Parameter(torch.zeros(size, dtype=torch.float32))
        self.optimizer = FP16_UnfusedOptimizer(self.module.weight, master)
        self.device = torch.device("cpu")
        self.global_steps = 0
        self.global_samples = 0
        self.micro_steps = 0
        self.skipped_steps = 0
        self.gas_boundary_ctr = 0


def _distributed_restore_worker(rank, world_size, init_file):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    # The production adapter restores CUDA RNG. This test intentionally runs
    # the exact state-copy protocol on CPU/Gloo.
    original_set_rng = torch.cuda.set_rng_state
    torch.cuda.set_rng_state = lambda *_args, **_kwargs: None
    try:
        reference = OrderedDict(weight=torch.zeros(4))
        old_masks = [
            OrderedDict(weight=torch.tensor([0, 1, 2])),
            OrderedDict(weight=torch.tensor([2, 3])),
        ]
        new_masks = [
            OrderedDict(weight=torch.tensor([0, 3])),
            OrderedDict(weight=torch.tensor([1, 2])),
        ]
        copy_plan = build_dynamic_copy_plan(
            reference,
            old_masks,
            new_masks,
            ["weight"],
        )
        master_global = torch.tensor([10.0, 11.0, 12.0, 13.0])
        average_global = master_global + 100.0
        square_global = master_global + 200.0
        old_indices = old_masks[rank]["weight"]
        snapshot = FP16AdamSnapshot(
            parameters=OrderedDict(
                weight=LocalFP16AdamParameter(
                    master_parameter=master_global.index_select(0, old_indices),
                    exp_avg=average_global.index_select(0, old_indices),
                    exp_avg_sq=square_global.index_select(0, old_indices),
                    step=torch.tensor(9.0),
                )
            ),
            optimizer_groups=({"lr": 0.025},),
            loss_scaler={},
            scheduler_state=None,
            engine_counters={"global_steps": 8},
            cpu_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.empty(0, dtype=torch.uint8),
        )
        engine = _Engine(size=2)
        restore_fp16_adam_from_rectangles(
            engine,
            copy_plan,
            snapshot,
            elements_per_transfer=1,
        )
        new_indices = new_masks[rank]["weight"]
        master = engine.optimizer.fp32_groups[0][0]
        state = engine.optimizer.optimizer.state[master]
        torch.testing.assert_close(master, master_global.index_select(0, new_indices))
        torch.testing.assert_close(state["exp_avg"], average_global.index_select(0, new_indices))
        torch.testing.assert_close(state["exp_avg_sq"], square_global.index_select(0, new_indices))
        torch.testing.assert_close(
            engine.module.weight,
            master_global.index_select(0, new_indices).half(),
        )
        assert float(state["step"].item()) == 9.0
        assert engine.global_steps == 8
        assert engine.optimizer.optimizer.param_groups[0]["lr"] == 0.025
    finally:
        torch.cuda.set_rng_state = original_set_rng
        torch.distributed.destroy_process_group()


class HelixDistributedAdamRestoreTest(unittest.TestCase):
    def test_two_rank_gloo_restores_local_and_remote_rectangles(self):
        if os.name == "nt":
            self.skipTest("file:// Gloo spawn test is Linux-only")
        with tempfile.TemporaryDirectory() as directory:
            init_file = os.path.join(directory, "process_group")
            mp.spawn(
                _distributed_restore_worker,
                args=(2, init_file),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
