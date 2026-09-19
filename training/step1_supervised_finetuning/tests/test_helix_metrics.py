"""CPU checks for metric accounting and transparent communication wrapping."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "dschat/helix/metrics.py"
SPEC = importlib.util.spec_from_file_location("helix_metrics", MODULE_PATH)
metrics_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics_module)


class FakeEngine:
    def __init__(self, communication):
        self.device = torch.device("cpu")
        self.module = torch.nn.Linear(3, 2)
        self.module.config = types.SimpleNamespace(to_dict=lambda: {"hidden_size": 3})
        self.global_steps = 0
        self.skipped_steps = 0
        self.optimizer = types.SimpleNamespace(param_groups=[{"lr": 0.001}])
        self.communication = communication

    def _reduce_non_expert_gradients(self, grads, elements_per_buffer):
        result = self.communication.all_reduce(grads, group="pair", async_op=True)
        self.communication.all_reduce(torch.ones(1), "SUM", "singleton")
        return result


class HelixMetricsTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.calls = []
        self.result = object()

        def all_reduce(tensor, *args, **kwargs):
            self.calls.append((tensor, args, kwargs))
            return self.result

        self.communication = types.SimpleNamespace(
            all_reduce=all_reduce,
            get_world_size=lambda group: 2 if group == "pair" else 1,
        )
        deepspeed_module = types.ModuleType("deepspeed")
        runtime = types.ModuleType("deepspeed.runtime")
        runtime.engine = types.SimpleNamespace(dist=self.communication)
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {"deepspeed": deepspeed_module, "deepspeed.runtime": runtime}).start()
        patch.object(torch.distributed, "get_world_size", return_value=2).start()
        self.engine = FakeEngine(self.communication)
        self.metrics = metrics_module.HelixMetrics(
            self.tempdir.name, rank=0, warmup_steps=1,
            args=types.SimpleNamespace(helix_max_steps=3),
            plan={"batch_sizes": [1, 1]}, ds_config={},
        )
        self.addCleanup(self.metrics.close)

    def test_causal_target_count_excludes_first_position_and_ignored_labels(self):
        counts = metrics_module.batch_counts({
            "input_ids": torch.ones(2, 4, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
            "labels": torch.tensor([[7, -100, 8, -100], [9, 10, -100, -100]]),
        })
        self.assertEqual(counts["local_batch_size"], 2)
        self.assertEqual(counts["attention_tokens"], 5)
        self.assertEqual(counts["valid_shifted_labels"], 2)
        self.assertEqual(counts["input_slots"], 8)

    def test_ring_estimate_uses_actual_dtype_and_group_size(self):
        description = metrics_module.describe_all_reduce(torch.zeros(12, dtype=torch.float16), 4)
        self.assertEqual(description["payload_bytes"], 24)
        self.assertEqual(description["ring_estimated_send_bytes"], 36)
        self.assertEqual(metrics_module.describe_all_reduce(torch.ones(12), 1)["ring_estimated_send_bytes"], 0)

    def test_wrapper_counts_only_model_reduction_and_preserves_return(self):
        original_all_reduce = self.communication.all_reduce
        original_reduce = self.engine._reduce_non_expert_gradients
        self.metrics.begin_step(self.engine, microstep=1, epoch=0, epoch_step=0)
        self.communication.all_reduce(torch.ones(100), group="pair")
        result = self.engine._reduce_non_expert_gradients(torch.zeros(12, dtype=torch.float16), 8)
        self.assertIs(result, self.result)
        self.assertIs(self.communication.all_reduce, original_all_reduce)
        self.assertEqual(self.calls[1][2], {"group": "pair", "async_op": True})
        self.engine.global_steps = 1
        record = self.metrics.end_step(loss=torch.tensor(1.5), counts={})
        self.assertEqual(len(record["gradient_all_reduce_calls"]), 2)
        self.assertEqual(record["gradient_all_reduce_payload_bytes"], 28)
        self.assertEqual(record["gradient_all_reduce_ring_estimated_send_bytes"], 24)
        self.assertEqual(record["optimizer_steps_succeeded"], 1)
        self.assertTrue(record["warmup"])
        self.assertEqual(record["learning_rates"], [0.001])
        self.assertEqual(record["learning_rates_after_step"], [0.001])
        self.assertGreaterEqual(record["finished_unix"], record["started_unix"])
        self.assertEqual(json.loads((Path(self.tempdir.name) / "rank_0.jsonl").read_text())["loss"], 1.5)
        self.metrics.detach_engine()
        self.assertEqual(self.engine._reduce_non_expert_gradients, original_reduce)

    def test_accumulation_and_overflow_are_not_successful_updates(self):
        self.metrics.begin_step(self.engine, microstep=1, epoch=0, epoch_step=0)
        record = self.metrics.end_step(loss=torch.tensor(1.0), counts={})
        self.assertEqual(record["optimizer_steps_attempted"], 0)
        self.assertEqual(record["optimizer_steps_succeeded"], 0)
        self.metrics.begin_step(self.engine, microstep=2, epoch=0, epoch_step=1)
        self.engine.global_steps += 1
        self.engine.skipped_steps += 1
        record = self.metrics.end_step(loss=torch.tensor(float("nan")), counts={})
        self.assertEqual(record["optimizer_steps_succeeded"], 0)
        self.assertEqual(record["overflow_steps"], 1)
        self.assertFalse(record["warmup"])
        self.assertIsNone(json.loads((Path(self.tempdir.name) / "rank_0.jsonl").read_text().splitlines()[-1])["loss"])

    def test_all_reduce_is_restored_when_reduction_raises(self):
        original_all_reduce = self.communication.all_reduce

        def fail(*args, **kwargs):
            self.communication.all_reduce(torch.ones(3), group="pair")
            raise RuntimeError("test collective failure")

        self.engine._reduce_non_expert_gradients = fail
        self.metrics.begin_step(self.engine, microstep=1, epoch=0, epoch_step=0)
        with self.assertRaisesRegex(RuntimeError, "collective failure"):
            self.engine._reduce_non_expert_gradients([], 3)
        self.assertIs(self.communication.all_reduce, original_all_reduce)

    def test_proxy_rebind_metadata_changes_only_when_engine_changes(self):
        proxy = types.SimpleNamespace(_engine=self.engine)
        self.metrics.attach_engine(proxy)
        self.metrics.detach_engine()
        self.metrics.attach_engine(proxy)
        self.assertEqual(self.metrics._generation, 0)
        proxy._engine = FakeEngine(self.communication)
        self.metrics.attach_engine(proxy)
        self.assertEqual(self.metrics._generation, 1)
        self.assertTrue((Path(self.tempdir.name) / "rank_0_generation_1_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
