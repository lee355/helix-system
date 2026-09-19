"""CPU regression tests for unfused FP16 optimizer state initialization."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "deepspeed"))

from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer


def _make_wrapper(optimizer_type, *, weight_decay=0.37, amsgrad=False):
    fp16_parameter = torch.nn.Parameter(
        torch.tensor([1.5, -2.0, 0.25], dtype=torch.float16)
    )
    master_parameter = fp16_parameter.detach().float().clone().requires_grad_(True)
    optimizer = optimizer_type(
        [master_parameter],
        lr=0.025,
        betas=(0.8, 0.95),
        eps=1e-7,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
        foreach=False,
    )
    wrapper = FP16_UnfusedOptimizer.__new__(FP16_UnfusedOptimizer)
    wrapper.optimizer = optimizer
    wrapper.fp16_groups = [[fp16_parameter]]
    wrapper.fp32_groups = [[master_parameter]]
    return wrapper, fp16_parameter, master_parameter


class FP16OptimizerInitializationTest(unittest.TestCase):
    def test_adam_initialization_preserves_fresh_first_step(self):
        for optimizer_type in (torch.optim.Adam, torch.optim.AdamW):
            with self.subTest(optimizer=optimizer_type.__name__):
                wrapper, fp16_parameter, master_parameter = _make_wrapper(
                    optimizer_type
                )
                initial_master = master_parameter.detach().clone()
                initial_fp16 = fp16_parameter.detach().clone()
                reference_parameter = (
                    initial_master.detach().clone().requires_grad_(True)
                )
                reference_optimizer = optimizer_type(
                    [reference_parameter],
                    lr=0.025,
                    betas=(0.8, 0.95),
                    eps=1e-7,
                    weight_decay=0.37,
                    amsgrad=False,
                    foreach=False,
                )

                with patch.object(
                    wrapper.optimizer,
                    "step",
                    wraps=wrapper.optimizer.step,
                ) as optimizer_step:
                    wrapper.initialize_optimizer_states()
                    optimizer_step.assert_not_called()

                state = wrapper.optimizer.state[master_parameter]
                self.assertEqual(state["step"].item(), 0)
                torch.testing.assert_close(
                    master_parameter, initial_master, rtol=0.0, atol=0.0
                )
                torch.testing.assert_close(
                    fp16_parameter, initial_fp16, rtol=0.0, atol=0.0
                )
                self.assertEqual(torch.count_nonzero(state["exp_avg"]).item(), 0)
                self.assertEqual(
                    torch.count_nonzero(state["exp_avg_sq"]).item(), 0
                )

                gradient = torch.tensor([0.3, -0.7, 0.4])
                master_parameter.grad = gradient.clone()
                reference_parameter.grad = gradient.clone()
                wrapper.optimizer.step()
                reference_optimizer.step()

                torch.testing.assert_close(
                    master_parameter, reference_parameter, rtol=0.0, atol=0.0
                )
                reference_state = reference_optimizer.state[reference_parameter]
                self.assertEqual(state["step"].item(), 1)
                torch.testing.assert_close(
                    state["exp_avg"], reference_state["exp_avg"]
                )
                torch.testing.assert_close(
                    state["exp_avg_sq"], reference_state["exp_avg_sq"]
                )

    def test_allocated_states_support_snapshot_restore_contract(self):
        wrapper, _, master_parameter = _make_wrapper(
            torch.optim.AdamW, amsgrad=True
        )
        wrapper.initialize_optimizer_states()
        state = wrapper.optimizer.state[master_parameter]
        self.assertEqual(
            set(state), {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
        )
        state["step"].fill_(7)
        state["exp_avg"].copy_(torch.tensor([1.0, 2.0, 3.0]))
        state["exp_avg_sq"].copy_(torch.tensor([4.0, 5.0, 6.0]))
        state["max_exp_avg_sq"].copy_(torch.tensor([7.0, 8.0, 9.0]))
        snapshot = deepcopy(wrapper.optimizer.state_dict())

        restored, restored_fp16, restored_master = _make_wrapper(
            torch.optim.AdamW, amsgrad=True
        )
        restored.initialize_optimizer_states()
        restored.optimizer.load_state_dict(snapshot)
        restored_state = restored.optimizer.state[restored_master]
        for name in ("step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            torch.testing.assert_close(restored_state[name], state[name])
        self.assertIsNone(restored_fp16.grad)
        self.assertIsNone(restored_master.grad)

    def test_unknown_optimizer_keeps_eager_step_fallback(self):
        class CountingSGD(torch.optim.SGD):
            def __init__(self, params):
                super().__init__(params, lr=0.1, momentum=0.9)
                self.step_calls = 0

            def step(self, closure=None):
                self.step_calls += 1
                return super().step(closure)

        fp16_parameter = torch.nn.Parameter(
            torch.tensor([1.0, -1.0], dtype=torch.float16)
        )
        master_parameter = (
            fp16_parameter.detach().float().clone().requires_grad_(True)
        )
        optimizer = CountingSGD([master_parameter])
        wrapper = FP16_UnfusedOptimizer.__new__(FP16_UnfusedOptimizer)
        wrapper.optimizer = optimizer
        wrapper.fp16_groups = [[fp16_parameter]]
        wrapper.fp32_groups = [[master_parameter]]

        wrapper.initialize_optimizer_states()

        self.assertEqual(optimizer.step_calls, 1)
        self.assertIn("momentum_buffer", optimizer.state[master_parameter])
        self.assertIsNone(fp16_parameter.grad)
        self.assertIsNone(master_parameter.grad)


if __name__ == "__main__":
    unittest.main()
