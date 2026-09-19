from __future__ import annotations

import copy
import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from vendor_bootstrap import activate_local_dependencies


activate_local_dependencies()

import torch

from dschat.helix.acceptance import MathAcceptance, model_state_digest, stop_reason


class TinyFullModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(11, 4)
        self.projection = torch.nn.Linear(4, 11, bias=False)
        self.projection.weight = self.embed.weight
        self.norm = torch.nn.LayerNorm(4)
        self.config = types.SimpleNamespace()

    def save_pretrained(self, directory, **kwargs):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "pytorch_model.bin")
        (directory / "save_options.json").write_text(json.dumps(kwargs, sort_keys=True))


class FakeTokenizer:
    def save_pretrained(self, directory):
        path = Path(directory) / "tokenizer.json"
        path.write_text("{}\n")


class FakeEngine:
    def __init__(self):
        torch.manual_seed(17)
        self.full_model = TinyFullModel()
        self.device = torch.device("cpu")
        self.gather_calls = 0

    def gather_full_model(self):
        self.gather_calls += 1


def _result(accuracy, protocol="fixed-protocol-v1"):
    correct = int(round(accuracy * 100))
    return {
        "accuracy": accuracy,
        "num_questions": 100,
        "num_correct": correct,
        "protocol_hash": protocol,
        "predictions": [{"question": index} for index in range(100)],
    }


def _args(directory, **overrides):
    values = {
        "helix_acceptance_dir": str(directory),
        "helix_mmlu_path": "/unused/fake-mmlu",
        "helix_target_accuracy": 0.62,
        "helix_eval_batch_size": 1,
        "helix_eval_max_length": 128,
        "helix_eval_fewshot": 5,
        "helix_eval_interval": 10,
        "helix_max_successful_steps": 0,
        "helix_max_train_seconds": 0,
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


class MathAcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.directory = Path(self.tempdir.name) / "acceptance"
        self.data = types.SimpleNamespace(test=tuple(range(100)))
        self.tokenizer = FakeTokenizer()
        self.dist_patches = [
            mock.patch("dschat.helix.acceptance.dist.get_rank", return_value=0),
            mock.patch("dschat.helix.acceptance.dist.get_world_size", return_value=1),
            mock.patch("dschat.helix.acceptance.dist.all_gather_object"),
            mock.patch("dschat.helix.acceptance.dist.broadcast_object_list"),
            mock.patch("dschat.helix.acceptance.dist.barrier"),
            mock.patch(
                "dschat.helix.mmlu_math.load_mmlu_math", return_value=self.data
            ),
        ]
        for patcher in self.dist_patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _controller(self, engine, results, **arg_overrides):
        controller = MathAcceptance(
            _args(self.directory, **arg_overrides),
            self.tokenizer,
            model_state_digest(engine.full_model),
            plan={"fixed": True},
            initialization_started=time.perf_counter(),
        )
        remaining = list(results)

        def evaluate_on_root(this, current_engine):
            this.eval_model = copy.deepcopy(current_engine.full_model)
            return remaining.pop(0)

        controller._evaluate_on_root = types.MethodType(
            evaluate_on_root, controller
        )
        return controller

    def test_initial_exact_target_has_zero_training_time_and_saves_full_weights(self):
        engine = FakeEngine()
        controller = self._controller(engine, [_result(0.62)])

        self.assertTrue(controller.begin(engine))
        self.assertEqual(controller.target["accuracy"], 0.62)
        self.assertEqual(controller.target["time_to_target_seconds"], 0.0)
        self.assertTrue(controller.target["initial_model_already_met_target"])
        self.assertIsNone(controller.training_started)

        controller.finish(engine, "initial_target_reached")
        self.assertEqual(engine.gather_calls, 1)
        summary = json.loads(
            (self.directory / "acceptance_summary.json").read_text()
        )
        self.assertEqual(summary["status"], "target_reached")
        self.assertEqual(summary["successful_steps"], 0)
        self.assertEqual(summary["overflow_steps"], 0)
        self.assertEqual(summary["training_wall_seconds"], 0.0)
        self.assertEqual(summary["target"]["time_to_target_seconds"], 0.0)
        self.assertEqual(
            summary["checkpoint_kind"], "full_reconstructed_model_weights_only"
        )

        checkpoint = Path(summary["checkpoint"])
        saved_state = torch.load(
            checkpoint / "pytorch_model.bin", map_location="cpu", weights_only=True
        )
        self.assertEqual(saved_state.keys(), engine.full_model.state_dict().keys())
        restored = TinyFullModel()
        restored.load_state_dict(saved_state, strict=True)
        self.assertEqual(
            model_state_digest(restored),
            summary["checkpoint_state_sha256_fp16"],
        )
        self.assertTrue((checkpoint / "tokenizer.json").is_file())
        save_options = json.loads((checkpoint / "save_options.json").read_text())
        self.assertTrue(save_options["safe_serialization"])
        self.assertEqual(save_options["max_shard_size"], "2GB")

    def test_overflow_does_not_consume_success_budget_or_trigger_periodic_eval(self):
        engine = FakeEngine()
        controller = self._controller(
            engine,
            [_result(0.61), _result(0.61)],
            helix_max_successful_steps=1,
            helix_eval_interval=1,
        )
        self.assertFalse(controller.begin(engine))
        self.assertIsNotNone(controller.training_started)

        reason = controller.after_step(
            engine, applied=False, step_seconds=0.4, global_batch=12
        )
        self.assertIsNone(reason)
        self.assertEqual(engine.gather_calls, 1)
        self.assertEqual(controller.attempts, 1)
        self.assertEqual(controller.successful_steps, 0)
        self.assertEqual(controller.samples, 12)
        self.assertEqual(controller.successful_samples, 0)

        reason = controller.after_step(
            engine, applied=True, step_seconds=0.6, global_batch=12
        )
        self.assertEqual(reason, "successful_step_limit")
        self.assertEqual(engine.gather_calls, 2)
        self.assertEqual(controller.attempts, 2)
        self.assertEqual(controller.successful_steps, 1)
        self.assertEqual(controller.samples, 24)
        self.assertEqual(controller.successful_samples, 12)
        self.assertAlmostEqual(controller.training_seconds, 1.0)

        controller.finish(engine, reason)
        summary = json.loads(
            (self.directory / "acceptance_summary.json").read_text()
        )
        self.assertEqual(summary["stop_reason"], "successful_step_limit")
        self.assertEqual(summary["attempted_steps"], 2)
        self.assertEqual(summary["successful_steps"], 1)
        self.assertEqual(summary["overflow_steps"], 1)
        self.assertEqual(summary["status"], "target_not_reached")
        self.assertEqual(Path(summary["checkpoint"]).name, "final_model")
        saved_state = torch.load(
            Path(summary["checkpoint"]) / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )
        restored = TinyFullModel()
        restored.load_state_dict(saved_state, strict=True)
        self.assertEqual(
            model_state_digest(restored),
            summary["checkpoint_state_sha256_fp16"],
        )

    def test_protocol_hash_cannot_change_during_run(self):
        engine = FakeEngine()
        controller = self._controller(
            engine,
            [_result(0.61, "protocol-a"), _result(0.61, "protocol-b")],
        )
        self.assertFalse(controller.begin(engine))
        controller.attempts = 1
        with self.assertRaisesRegex(RuntimeError, "protocol changed"):
            controller.evaluate(engine, reason="periodic")
        self.assertEqual(controller.protocol_hash, "protocol-a")
        self.assertEqual(len(controller.evaluations), 1)

    def test_root_evaluation_failure_is_broadcast_as_acceptance_failure(self):
        engine = FakeEngine()
        controller = self._controller(engine, [])

        def fail(_engine):
            raise RuntimeError("fake CPU evaluator failure")

        controller._evaluate_on_root = fail
        with self.assertRaisesRegex(RuntimeError, "fake CPU evaluator failure"):
            controller.evaluate(engine, reason="initial_reconstruction")

    def test_preflight_collective_failure_is_reported(self):
        self.dist_patches[2].stop()

        def inject_remote_failure(errors, local_error):
            self.assertIsNone(local_error)
            errors[:] = [None, "rank 1: fake collective preflight failure"]

        with mock.patch(
            "dschat.helix.acceptance.dist.get_world_size", return_value=2
        ), mock.patch(
            "dschat.helix.acceptance.dist.all_gather_object",
            side_effect=inject_remote_failure,
        ):
            with self.assertRaisesRegex(RuntimeError, "fake collective preflight failure"):
                MathAcceptance(
                    _args(self.directory), self.tokenizer, "digest", {}, time.perf_counter()
                )

    def test_rank_local_mkdir_failure_enters_preflight_consensus(self):
        observed = {}

        def gather_local_failure(errors, local_error):
            observed["local_error"] = local_error
            errors[:] = [local_error, None]

        with mock.patch(
            "dschat.helix.acceptance.Path.mkdir",
            side_effect=PermissionError("fake rank-local mkdir failure"),
        ), mock.patch(
            "dschat.helix.acceptance.dist.get_world_size", return_value=2
        ), mock.patch(
            "dschat.helix.acceptance.dist.all_gather_object",
            side_effect=gather_local_failure,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "fake rank-local mkdir failure"
            ):
                MathAcceptance(
                    _args(self.directory),
                    self.tokenizer,
                    "digest",
                    {},
                    time.perf_counter(),
                )

        self.assertIn("PermissionError", observed["local_error"])
        self.assertIn("rank 0", observed["local_error"])

    def test_stop_reason_uses_exact_budget_boundaries(self):
        self.assertEqual(
            stop_reason(3, 9.0, max_steps=3, max_seconds=9.0),
            "successful_step_limit",
        )
        self.assertEqual(
            stop_reason(2, 9.0, max_steps=3, max_seconds=9.0),
            "wall_time_limit",
        )
        self.assertIsNone(stop_reason(2, 8.999, max_steps=3, max_seconds=9.0))


class AcceptanceArgumentIntegrationTest(unittest.TestCase):
    def test_default_target_is_exact_point_62_and_acceptance_owns_checkpointing(self):
        import helix_train_base

        arguments = [
            "helix_train_base.py",
            "--data_path", "/unused/train",
            "--model_name_or_path", "/unused/model",
            "--helix_profiles_path", "/unused/profiles.json",
            "--helix_mmlu_path", "/unused/mmlu",
            "--helix_acceptance_dir", str(Path(self.id()).name),
        ]
        with mock.patch.object(sys, "argv", arguments):
            args = helix_train_base.parse_args()
        self.assertEqual(args.helix_target_accuracy, 0.62)
        self.assertTrue(args.helix_skip_checkpoint)
        self.assertEqual(
            args.helix_metrics_dir,
            str(Path(args.helix_acceptance_dir) / "metrics"),
        )

    def test_acceptance_rejects_gradient_accumulation(self):
        import helix_train_base

        arguments = [
            "helix_train_base.py",
            "--data_path", "/unused/train",
            "--model_name_or_path", "/unused/model",
            "--helix_profiles_path", "/unused/profiles.json",
            "--helix_mmlu_path", "/unused/mmlu",
            "--helix_acceptance_dir", "unused-acceptance",
            "--gradient_accumulation_steps", "2",
        ]
        with mock.patch.object(sys, "argv", arguments), mock.patch(
            "sys.stderr"
        ):
            with self.assertRaises(SystemExit):
                helix_train_base.parse_args()


if __name__ == "__main__":
    unittest.main()
