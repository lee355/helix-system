"""CPU mock tests for the Helix acceptance summarizer."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.helix_acceptance import summarize


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def _metric(rank, microstep, *, seconds, warmup=False, overflow=False):
    return {
        "rank": rank,
        "microstep": microstep,
        "epoch": 0,
        "epoch_step": microstep - 1,
        "engine_generation": 0,
        "warmup": warmup,
        "local_batch_size": 2 + rank,
        "sequence_length": 8,
        "input_slots": (2 + rank) * 8,
        "attention_tokens": (2 + rank) * 7,
        "valid_shifted_labels": 10 + 20 * rank,
        "step_seconds": seconds,
        "started_unix": 1000.0 + microstep,
        "finished_unix": 1000.0 + microstep + seconds,
        "loss": 1.0 + 2.0 * rank,
        "learning_rates": [5e-6],
        "learning_rates_after_step": [5e-6],
        "phase_cuda_seconds": {
            "forward": seconds / 4,
            "backward": seconds / 2,
            "optimizer": seconds / 4,
            "gradient_sync": seconds / 5,
        },
        "gradient_all_reduce_calls": [],
        "gradient_all_reduce_payload_bytes": 100 + 100 * rank,
        "gradient_all_reduce_ring_estimated_send_bytes": 150 + 100 * rank,
        "optimizer_steps_attempted": 1,
        "optimizer_steps_succeeded": 0 if overflow else 1,
        "overflow_steps": 1 if overflow else 0,
        "optimizer_steps_attempted_total": microstep,
        "optimizer_steps_succeeded_total": microstep - int(overflow),
        "overflow_steps_total": int(overflow),
        "cuda_peak_allocated_bytes": 400 + 100 * rank,
        "cuda_peak_reserved_bytes": 500 + 100 * rank,
        "gradient_sync_cuda_seconds": seconds / 5,
    }


def _metadata(rank):
    return {
        "schema_version": 1,
        "rank": rank,
        "world_size": 2,
        "engine_generation": 0,
        "trainable_parameter_numel": 1000 + rank,
        "total_parameter_numel": 1000 + rank,
        "local_model_config": {},
        "plan": {},
        "initial_deepspeed_config": {
            "gradient_accumulation_steps": 1,
        },
        "arguments": {},
        "device": {
            "torch_device": f"cuda:{rank}",
            "name": "fixture GPU",
            "total_memory_bytes": 1000,
            "multiprocessor_count": 1,
            "compute_capability": [8, 0],
        },
        "torch_version": "fixture",
        "cuda_version": "fixture",
        "metric_definitions": {},
    }


def _fixture(root):
    run = root / "run"
    collection = run / "collections" / "attempt-1"
    node0 = collection / "node-0-host0"
    node1 = collection / "node-1-host1"
    metrics0 = node0 / "acceptance" / "metrics"
    metrics1 = node1 / "acceptance" / "metrics"

    rows0 = [
        _metric(0, 1, seconds=0.2, warmup=True),
        _metric(0, 2, seconds=0.5, overflow=True),
        _metric(0, 3, seconds=1.0),
    ]
    rows1 = [
        _metric(1, 1, seconds=0.4, warmup=True),
        _metric(1, 2, seconds=0.8, overflow=True),
        _metric(1, 3, seconds=2.0),
    ]
    _write_jsonl(metrics0 / "rank_0.jsonl", rows0)
    _write_jsonl(metrics1 / "rank_1.jsonl", rows1)
    _write_json(metrics0 / "rank_0_metadata.json", _metadata(0))
    _write_json(metrics1 / "rank_1_metadata.json", _metadata(1))

    acceptance = node0 / "acceptance"
    evaluations = [
        {
            "accuracy": 0.35,
            "num_questions": 100,
            "num_correct": 35,
            "protocol_hash": "protocol",
            "reason": "initial_reconstruction",
            "attempted_steps": 0,
            "successful_steps": 0,
        },
        {
            "accuracy": 0.63,
            "num_questions": 100,
            "num_correct": 63,
            "protocol_hash": "protocol",
            "reason": "successful_step_limit",
            "attempted_steps": 3,
            "successful_steps": 2,
        },
    ]
    _write_jsonl(acceptance / "evaluations.jsonl", evaluations)
    _write_json(
        acceptance / "acceptance_summary.json",
        {
            "status": "target_reached",
            "stop_reason": "target_reached",
            "target_accuracy": 0.62,
            "best_accuracy": 0.63,
            "target": {
                "accuracy": 0.63,
                "successful_steps": 2,
            },
            "final_accuracy": 0.63,
            "protocol_hash": "protocol",
            "attempted_steps": 3,
            "successful_steps": 2,
            "overflow_steps": 1,
            "measured_training_step_seconds": 3.9,
            "training_wall_seconds": 4.2,
            "since_initialization_seconds": 6.0,
        },
    )
    _write_json(
        acceptance / "planning.json",
        {
            "source": "profiles_search",
            "profiles_path": "/runtime/profiles/profiles.json",
            "profiles_sha256": "profile-hash",
            "search_seconds": 1.25,
            "batch_sizes": [2, 3],
            "submodel_sizes": [0.5, 0.25],
        },
    )
    _write_json(
        node0 / "plan.json",
        {
            "world_size": 2,
            "batch_sizes": [2, 3],
            "submodel_sizes": [0.5, 0.25],
        },
    )
    _write_json(
        collection / "collection.json",
        {
            "name": "fixture",
            "nodes": [
                {"host": "host0", "status": "collected"},
                {"host": "host1", "status": "collected"},
            ],
            "integrity_errors": [],
            "checkpoint": {"status": "collected"},
        },
    )
    _write_json(
        run / "launch.finished.json",
        {
            "name": "fixture",
            "world_size": 2,
            "nodes": [
                {"host": "host0", "gpus": [0]},
                {"host": "host1", "gpus": [1]},
            ],
            "exit_codes": [0, 0],
            "error": None,
            "integrity_errors": [],
        },
    )
    return collection


class SummarizeFixtureTest(unittest.TestCase):
    def test_overflow_is_excluded_and_slowest_rank_sets_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection = _fixture(root)
            output = root / "summary"

            result = summarize.summarize(collection, output)

            self.assertEqual(result["world_size"], 2)
            self.assertEqual(result["attempted_microsteps"], 3)
            self.assertEqual(
                result["included_post_warmup_successful_microsteps"],
                1,
            )
            self.assertEqual(
                result["excluded_microsteps"],
                {
                    "warmup": 1,
                    "overflow": 1,
                    "no_successful_optimizer_update": 0,
                },
            )
            self.assertEqual(
                result["measurement_classification"],
                "functional_only",
            )
            self.assertFalse(result["steady_state_claimed"])
            self.assertAlmostEqual(
                result["synchronized_step_seconds"]["total"],
                2.0,
            )
            self.assertAlmostEqual(
                result["throughput"]["samples_per_second"],
                2.5,
            )
            self.assertAlmostEqual(
                result["throughput"]["effective_tokens_per_second"],
                20.0,
            )
            self.assertAlmostEqual(
                result["training_loss"][
                    "token_weighted_local_submodel_loss"
                ],
                2.5,
            )
            self.assertEqual(
                result["communication"][
                    "actual_collective_payload_bytes_sum_across_ranks"
                ],
                300,
            )
            self.assertIsNone(
                result["nvml"]["gpu_busy_percent"]
            )
            self.assertEqual(
                result["acceptance"]["timing"]["search"]["seconds"],
                1.25,
            )
            self.assertIsNone(
                result["acceptance"]["timing"]["profile"]["seconds"]
            )

            with (output / "microsteps.csv").open(
                encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertEqual(
                rows[1]["exclusion_reason"],
                "overflow",
            )
            self.assertEqual(
                float(rows[2]["synchronized_step_seconds"]),
                2.0,
            )
            self.assertEqual(float(rows[2]["rank_0_loss"]), 1.0)
            self.assertEqual(float(rows[2]["rank_1_loss"]), 3.0)

    def test_missing_rank_metrics_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection = _fixture(root)
            next(
                collection.rglob("rank_1.jsonl")
            ).unlink()

            with self.assertRaisesRegex(
                RuntimeError,
                "Missing rank metrics",
            ):
                summarize.summarize(
                    collection,
                    root / "summary",
                )


if __name__ == "__main__":
    unittest.main()
