#!/usr/bin/env python
"""Summarize a collected multi-rank Helix acceptance run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


RANK_METRIC_RE = re.compile(r"^rank_(\d+)\.jsonl$")
RANK_METADATA_RE = re.compile(
    r"^rank_(\d+)(?:_generation_(\d+))?_metadata\.json$"
)


def _reject_nonfinite_json(value):
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _read_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_json,
    )


def _read_jsonl(path: Path):
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                parse_constant=_reject_nonfinite_json,
            )
        except Exception as error:
            raise ValueError(
                f"Invalid JSONL at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        records.append(value)
    return records


def _finite_number(value, label, *, positive=False, nonnegative=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive, got {value!r}")
    if nonnegative and value < 0:
        raise ValueError(f"{label} must be nonnegative, got {value!r}")
    return value


def _integer(value, label, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}, got {value}")
    return value


def _mean(values):
    return statistics.mean(values) if values else None


def _quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return (
        ordered[low]
        + (ordered[high] - ordered[low]) * (position - low)
    )


def _node_index(path: Path, collection: Path):
    relative = path.resolve().relative_to(collection.resolve())
    for part in relative.parts:
        match = re.fullmatch(r"node-(\d+)-.*", part)
        if match:
            return int(match.group(1))
    return None


def _find_launch(collection: Path):
    for directory in (collection, *collection.parents):
        finished = directory / "launch.finished.json"
        if finished.is_file():
            return finished
        started = directory / "launch.started.json"
        if started.is_file():
            candidate = started
        else:
            candidate = None
        if candidate is not None:
            return candidate
    raise FileNotFoundError(
        f"No launch.finished.json/launch.started.json above {collection}"
    )


def _unique_artifact(collection: Path, name: str, *, required=True):
    paths = sorted(path for path in collection.rglob(name) if path.is_file())
    if not paths and not required:
        return None
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected exactly one rank-0 {name}, found {len(paths)}: "
            f"{[str(path) for path in paths]}"
        )
    if _node_index(paths[0], collection) != 0:
        raise RuntimeError(f"{name} must come from node 0: {paths[0]}")
    return paths[0]


def _rank_to_node(nodes):
    result = {}
    rank = 0
    for node_index, node in enumerate(nodes):
        gpus = node.get("gpus")
        if not isinstance(gpus, list) or not gpus:
            raise ValueError(f"launch.nodes[{node_index}].gpus is invalid")
        for _ in gpus:
            result[rank] = node_index
            rank += 1
    return result


def _load_rank_evidence(collection: Path, launch):
    world_size = _integer(
        launch.get("world_size"),
        "launch.world_size",
        minimum=1,
    )
    nodes = launch.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("launch.nodes must be a nonempty list")
    rank_to_node = _rank_to_node(nodes)
    if len(rank_to_node) != world_size:
        raise RuntimeError(
            "launch node GPU counts disagree with launch.world_size"
        )

    metrics = {}
    metric_paths = {}
    for path in collection.rglob("rank_*.jsonl"):
        match = RANK_METRIC_RE.fullmatch(path.name)
        if match is None:
            continue
        rank = int(match.group(1))
        if rank in metrics:
            raise RuntimeError(
                f"Duplicate metrics for rank {rank}: "
                f"{metric_paths[rank]} and {path}"
            )
        if _node_index(path, collection) != rank_to_node.get(rank):
            raise RuntimeError(
                f"Rank {rank} metrics are on the wrong collected node: {path}"
            )
        metrics[rank] = _read_jsonl(path)
        metric_paths[rank] = path

    expected_ranks = set(range(world_size))
    if set(metrics) != expected_ranks:
        raise RuntimeError(
            f"Missing rank metrics: expected {sorted(expected_ranks)}, "
            f"found {sorted(metrics)}"
        )
    if any(not records for records in metrics.values()):
        raise RuntimeError("Every rank metrics file must contain records")

    metadata = {}
    metadata_paths = {}
    for path in collection.rglob("rank_*_metadata.json"):
        match = RANK_METADATA_RE.fullmatch(path.name)
        if match is None:
            continue
        rank = int(match.group(1))
        generation = int(match.group(2) or 0)
        key = (rank, generation)
        if key in metadata:
            raise RuntimeError(
                f"Duplicate metadata for rank/generation {key}"
            )
        if _node_index(path, collection) != rank_to_node.get(rank):
            raise RuntimeError(
                f"Rank {rank} metadata are on the wrong collected node: {path}"
            )
        value = _read_json(path)
        if value.get("rank") != rank:
            raise RuntimeError(f"Metadata rank field disagrees: {path}")
        if value.get("world_size") != world_size:
            raise RuntimeError(f"Metadata world_size disagrees: {path}")
        if value.get("engine_generation") != generation:
            raise RuntimeError(f"Metadata generation disagrees: {path}")
        metadata[key] = value
        metadata_paths[key] = path

    indexed = {}
    reference_steps = None
    for rank in range(world_size):
        rank_rows = {}
        for row_index, row in enumerate(metrics[rank]):
            microstep = _integer(
                row.get("microstep"),
                f"rank {rank} row {row_index} microstep",
                minimum=1,
            )
            if microstep in rank_rows:
                raise RuntimeError(
                    f"Rank {rank} repeats microstep {microstep}"
                )
            if row.get("rank") != rank:
                raise RuntimeError(
                    f"Rank field mismatch in rank {rank} microstep {microstep}"
                )
            generation = _integer(
                row.get("engine_generation", 0),
                f"rank {rank} microstep {microstep} engine_generation",
                minimum=0,
            )
            if (rank, generation) not in metadata:
                raise RuntimeError(
                    f"Missing metadata for rank {rank} generation {generation}"
                )
            _finite_number(
                row.get("loss"),
                f"rank {rank} microstep {microstep} loss",
            )
            _finite_number(
                row.get("step_seconds"),
                f"rank {rank} microstep {microstep} step_seconds",
                positive=True,
            )
            _integer(
                row.get("local_batch_size"),
                f"rank {rank} microstep {microstep} local_batch_size",
                minimum=1,
            )
            _integer(
                row.get("valid_shifted_labels"),
                f"rank {rank} microstep {microstep} valid_shifted_labels",
                minimum=0,
            )
            for field in (
                "optimizer_steps_succeeded",
                "overflow_steps",
            ):
                value = _integer(
                    row.get(field),
                    f"rank {rank} microstep {microstep} {field}",
                    minimum=0,
                )
                if value not in (0, 1):
                    raise RuntimeError(
                        f"rank {rank} microstep {microstep} {field} "
                        f"must be 0 or 1"
                    )
            if not isinstance(row.get("warmup"), bool):
                raise ValueError(
                    f"rank {rank} microstep {microstep} warmup must be bool"
                )
            for field in (
                "gradient_all_reduce_payload_bytes",
                "gradient_all_reduce_ring_estimated_send_bytes",
            ):
                _finite_number(
                    row.get(field),
                    f"rank {rank} microstep {microstep} {field}",
                    nonnegative=True,
                )
            for optional in (
                "input_slots",
                "attention_tokens",
                "cuda_peak_allocated_bytes",
                "cuda_peak_reserved_bytes",
                "gradient_sync_cuda_seconds",
            ):
                if row.get(optional) is not None:
                    _finite_number(
                        row[optional],
                        f"rank {rank} microstep {microstep} {optional}",
                        nonnegative=True,
                    )
            rank_rows[microstep] = row
        steps = set(rank_rows)
        if reference_steps is None:
            reference_steps = steps
        elif steps != reference_steps:
            raise RuntimeError(
                "Ranks have incomplete/mismatched microsteps: "
                f"rank 0={sorted(reference_steps)}, "
                f"rank {rank}={sorted(steps)}"
            )
        indexed[rank] = rank_rows

    return (
        world_size,
        nodes,
        sorted(reference_steps),
        indexed,
        metadata,
        metric_paths,
        metadata_paths,
    )


def _step_rows(world_size, steps, indexed):
    output = []
    for microstep in steps:
        rows = [indexed[rank][microstep] for rank in range(world_size)]
        warmups = {row["warmup"] for row in rows}
        succeeded = {row["optimizer_steps_succeeded"] for row in rows}
        overflows = {row["overflow_steps"] for row in rows}
        if len(warmups) != 1:
            raise RuntimeError(
                f"Ranks disagree on warmup at microstep {microstep}"
            )
        if len(succeeded) != 1 or len(overflows) != 1:
            raise RuntimeError(
                f"Ranks disagree on optimizer outcome at microstep {microstep}"
            )
        warmup = warmups.pop()
        success = succeeded.pop()
        overflow = overflows.pop()
        reasons = []
        if warmup:
            reasons.append("warmup")
        if overflow:
            reasons.append("overflow")
        if not success and not overflow:
            reasons.append("no_successful_optimizer_update")
        included = not reasons and success == 1

        duration = max(row["step_seconds"] for row in rows)
        samples = sum(row["local_batch_size"] for row in rows)
        targets = sum(row["valid_shifted_labels"] for row in rows)
        weighted_loss = (
            sum(
                row["loss"] * row["valid_shifted_labels"]
                for row in rows
            )
            / targets
            if targets
            else None
        )
        record = {
            "microstep": microstep,
            "included": included,
            "exclusion_reason": ",".join(reasons),
            "warmup": warmup,
            "overflow": overflow,
            "optimizer_step_succeeded": success,
            "synchronized_step_seconds": duration,
            "global_samples": samples,
            "effective_shifted_targets": targets,
            "samples_per_second": samples / duration,
            "effective_tokens_per_second": targets / duration,
            "token_weighted_local_submodel_loss": weighted_loss,
            "local_submodel_loss_mean": _mean(
                [row["loss"] for row in rows]
            ),
            "local_submodel_loss_min": min(row["loss"] for row in rows),
            "local_submodel_loss_max": max(row["loss"] for row in rows),
            "actual_collective_payload_bytes_sum_across_ranks": sum(
                row["gradient_all_reduce_payload_bytes"] for row in rows
            ),
            "ring_estimated_send_bytes_sum_across_ranks": sum(
                row[
                    "gradient_all_reduce_ring_estimated_send_bytes"
                ]
                for row in rows
            ),
        }
        for rank, row in enumerate(rows):
            record[f"rank_{rank}_loss"] = row["loss"]
            record[f"rank_{rank}_step_seconds"] = row["step_seconds"]
            record[f"rank_{rank}_local_batch_size"] = row[
                "local_batch_size"
            ]
            record[f"rank_{rank}_valid_shifted_labels"] = row[
                "valid_shifted_labels"
            ]
        output.append(record)
    return output


def _rank_summaries(world_size, included_steps, indexed, metadata):
    result = []
    for rank in range(world_size):
        rows = [indexed[rank][step] for step in included_steps]
        generations = sorted(
            generation
            for metadata_rank, generation in metadata
            if metadata_rank == rank
        )
        totals = {
            metadata[(rank, generation)]["device"].get(
                "total_memory_bytes"
            )
            for generation in generations
        }
        totals.discard(None)
        if len(totals) > 1:
            raise RuntimeError(
                f"Rank {rank} device memory changed across generations"
            )
        total_memory = next(iter(totals), None)
        peaks = [
            row["cuda_peak_allocated_bytes"]
            for row in rows
            if row.get("cuda_peak_allocated_bytes") is not None
        ]
        reserved = [
            row["cuda_peak_reserved_bytes"]
            for row in rows
            if row.get("cuda_peak_reserved_bytes") is not None
        ]
        peak = max(peaks, default=None)
        result.append(
            {
                "rank": rank,
                "engine_generations": generations,
                "device": [
                    metadata[(rank, generation)]["device"]
                    for generation in generations
                ],
                "trainable_parameter_numel": [
                    metadata[(rank, generation)].get(
                        "trainable_parameter_numel"
                    )
                    for generation in generations
                ],
                "local_step_seconds_mean": _mean(
                    [row["step_seconds"] for row in rows]
                ),
                "local_submodel_token_weighted_loss": (
                    sum(
                        row["loss"] * row["valid_shifted_labels"]
                        for row in rows
                    )
                    / sum(
                        row["valid_shifted_labels"] for row in rows
                    )
                    if sum(
                        row["valid_shifted_labels"] for row in rows
                    )
                    else None
                ),
                "actual_collective_payload_bytes_total": sum(
                    row["gradient_all_reduce_payload_bytes"]
                    for row in rows
                ),
                "ring_estimated_send_bytes_total": sum(
                    row[
                        "gradient_all_reduce_ring_estimated_send_bytes"
                    ]
                    for row in rows
                ),
                "gradient_sync_seconds_mean": _mean(
                    [
                        row["gradient_sync_cuda_seconds"]
                        for row in rows
                        if row.get("gradient_sync_cuda_seconds")
                        is not None
                    ]
                ),
                "cuda_peak_allocated_bytes": peak,
                "cuda_peak_reserved_bytes": max(
                    reserved,
                    default=None,
                ),
                "memory_utilization_peak_allocated_percent": (
                    100.0 * peak / total_memory
                    if peak is not None and total_memory
                    else None
                ),
            }
        )
    return result


def _load_acceptance(collection: Path):
    summary_path = _unique_artifact(
        collection,
        "acceptance_summary.json",
    )
    evaluations_path = _unique_artifact(
        collection,
        "evaluations.jsonl",
    )
    planning_path = _unique_artifact(
        collection,
        "planning.json",
    )
    plan_path = _unique_artifact(
        collection,
        "plan.json",
        required=False,
    )
    acceptance_summary = _read_json(summary_path)
    evaluations = _read_jsonl(evaluations_path)
    planning = _read_json(planning_path)
    plan = _read_json(plan_path) if plan_path is not None else None
    if not evaluations:
        raise RuntimeError("evaluations.jsonl is empty")
    for index, evaluation in enumerate(evaluations):
        accuracy = _finite_number(
            evaluation.get("accuracy"),
            f"evaluation {index} accuracy",
        )
        if not 0.0 <= accuracy <= 1.0:
            raise RuntimeError(
                f"evaluation {index} accuracy is outside [0, 1]"
            )
        _integer(
            evaluation.get("num_questions"),
            f"evaluation {index} num_questions",
            minimum=1,
        )
        _integer(
            evaluation.get("num_correct"),
            f"evaluation {index} num_correct",
            minimum=0,
        )
    final_accuracy = _finite_number(
        acceptance_summary.get("final_accuracy"),
        "acceptance_summary.final_accuracy",
    )
    if abs(final_accuracy - evaluations[-1]["accuracy"]) > 1e-12:
        raise RuntimeError(
            "acceptance_summary final accuracy disagrees with evaluations"
        )
    protocol_hashes = {
        evaluation.get("protocol_hash")
        for evaluation in evaluations
    }
    if (
        len(protocol_hashes) != 1
        or acceptance_summary.get("protocol_hash")
        != next(iter(protocol_hashes))
    ):
        raise RuntimeError(
            "Evaluation protocol hash is missing or inconsistent"
        )
    search_seconds = _finite_number(
        planning.get("search_seconds"),
        "planning.search_seconds",
        nonnegative=True,
    )
    profile_seconds = planning.get("profile_seconds")
    if profile_seconds is not None:
        profile_seconds = _finite_number(
            profile_seconds,
            "planning.profile_seconds",
            nonnegative=True,
        )
    return {
        "acceptance_summary": acceptance_summary,
        "evaluations": evaluations,
        "planning": planning,
        "execution_plan": plan,
        "sources": {
            "acceptance_summary": str(summary_path),
            "evaluations": str(evaluations_path),
            "planning": str(planning_path),
            "execution_plan": (
                str(plan_path) if plan_path is not None else None
            ),
        },
        "timing": {
            "profile": {
                "seconds": profile_seconds,
                "source": (
                    f"{planning_path}:profile_seconds"
                    if profile_seconds is not None
                    else None
                ),
                "profiles_path": planning.get("profiles_path"),
                "profiles_sha256": planning.get("profiles_sha256"),
                "note": (
                    None
                    if profile_seconds is not None
                    else "No measured profile duration is present in the "
                    "collected planning artifact."
                ),
            },
            "search": {
                "seconds": search_seconds,
                "source": f"{planning_path}:search_seconds",
            },
            "training": {
                "measured_step_seconds": acceptance_summary.get(
                    "measured_training_step_seconds"
                ),
                "wall_seconds": acceptance_summary.get(
                    "training_wall_seconds"
                ),
                "since_initialization_seconds": acceptance_summary.get(
                    "since_initialization_seconds"
                ),
                "source": str(summary_path),
            },
        },
    }


def summarize(collection: Path, output: Path):
    collection = collection.resolve()
    output = output.resolve()
    if not collection.is_dir():
        raise FileNotFoundError(collection)
    if output.exists():
        raise FileExistsError(
            f"Choose a new --output directory: {output}"
        )
    collection_manifest_path = collection / "collection.json"
    collection_manifest = _read_json(collection_manifest_path)
    node_results = collection_manifest.get("nodes")
    if (
        not isinstance(node_results, list)
        or not node_results
        or any(row.get("status") != "collected" for row in node_results)
    ):
        raise RuntimeError("Collection is incomplete across nodes")
    if collection_manifest.get("integrity_errors"):
        raise RuntimeError("Collection reports source/input integrity errors")

    launch_path = _find_launch(collection)
    if launch_path.name != "launch.finished.json":
        raise RuntimeError("Run has no launch.finished.json")
    launch = _read_json(launch_path)
    if launch.get("error") is not None:
        raise RuntimeError(f"Run failed: {launch['error']}")
    if launch.get("integrity_errors"):
        raise RuntimeError("Launch reports source/input integrity errors")
    exit_codes = launch.get("exit_codes")
    if (
        not isinstance(exit_codes, list)
        or not exit_codes
        or any(code != 0 for code in exit_codes)
    ):
        raise RuntimeError(f"Run exit codes are not all zero: {exit_codes}")

    (
        world_size,
        nodes,
        steps,
        indexed,
        metadata,
        metric_paths,
        metadata_paths,
    ) = _load_rank_evidence(collection, launch)
    per_step = _step_rows(world_size, steps, indexed)
    included = [row for row in per_step if row["included"]]
    included_steps = [row["microstep"] for row in included]
    elapsed = sum(
        row["synchronized_step_seconds"] for row in included
    )
    total_samples = sum(row["global_samples"] for row in included)
    total_targets = sum(
        row["effective_shifted_targets"] for row in included
    )
    weighted_loss = (
        sum(
            row["token_weighted_local_submodel_loss"]
            * row["effective_shifted_targets"]
            for row in included
            if row["token_weighted_local_submodel_loss"] is not None
        )
        / total_targets
        if total_targets
        else None
    )
    durations = [
        row["synchronized_step_seconds"] for row in included
    ]
    functional_only = len(included) < 10
    rank_summaries = _rank_summaries(
        world_size,
        included_steps,
        indexed,
        metadata,
    )
    acceptance = _load_acceptance(collection)
    acceptance_summary = acceptance["acceptance_summary"]
    successful_all = sum(
        row["optimizer_step_succeeded"] for row in per_step
    )
    overflow_all = sum(row["overflow"] for row in per_step)
    if acceptance_summary.get("attempted_steps") != len(per_step):
        raise RuntimeError(
            "Acceptance attempted_steps disagrees with rank metrics"
        )
    if acceptance_summary.get("successful_steps") != successful_all:
        raise RuntimeError(
            "Acceptance successful_steps disagrees with rank metrics"
        )
    if acceptance_summary.get("overflow_steps") != overflow_all:
        raise RuntimeError(
            "Acceptance overflow_steps disagrees with rank metrics"
        )

    nvml_paths = sorted(
        str(path) for path in collection.rglob("gpu.csv")
        if path.is_file()
    )
    summary = {
        "schema_version": 1,
        "collection": str(collection),
        "launch": str(launch_path),
        "world_size": world_size,
        "nodes": nodes,
        "attempted_microsteps": len(per_step),
        "included_post_warmup_successful_microsteps": len(included),
        "excluded_microsteps": {
            "warmup": sum(row["warmup"] for row in per_step),
            "overflow": sum(row["overflow"] for row in per_step),
            "no_successful_optimizer_update": sum(
                "no_successful_optimizer_update"
                in row["exclusion_reason"]
                for row in per_step
            ),
        },
        "measurement_classification": (
            "functional_only"
            if functional_only
            else "post_warmup_measurement"
        ),
        "steady_state_claimed": False,
        "measurement_note": (
            "Fewer than 10 included microsteps; functional-only evidence, "
            "with no steady-state performance claim."
            if functional_only
            else "Post-warmup successful microsteps are summarized; this "
            "tool does not independently prove steady state."
        ),
        "synchronized_step_seconds": {
            "total": elapsed,
            "mean": _mean(durations),
            "p50": _quantile(durations, 0.50),
            "p95": _quantile(durations, 0.95),
            "definition": (
                "For each microstep, max(step_seconds) across all ranks."
            ),
        },
        "throughput": {
            "global_samples": total_samples,
            "effective_shifted_targets": total_targets,
            "samples_per_second": (
                total_samples / elapsed if elapsed else None
            ),
            "effective_tokens_per_second": (
                total_targets / elapsed if elapsed else None
            ),
            "definition": (
                "Sums actual rank-local batch sizes or valid shifted "
                "targets, divided by sum of synchronized max-rank step "
                "seconds; warmup and any-rank overflow/no-update excluded."
            ),
        },
        "training_loss": {
            "token_weighted_local_submodel_loss": weighted_loss,
            "definition": (
                "Rank-local submodel causal-LM losses weighted by each "
                "rank's labels[..., 1:] != -100 count. This is not loss "
                "from the reconstructed complete model and is not MMLU."
            ),
        },
        "full_model_mmlu": {
            "status": acceptance_summary.get("status"),
            "target_accuracy": acceptance_summary.get(
                "target_accuracy"
            ),
            "best_accuracy": acceptance_summary.get("best_accuracy"),
            "final_accuracy": acceptance_summary.get("final_accuracy"),
            "target": acceptance_summary.get("target"),
            "definition": (
                "MMLU accuracy comes only from rank-0 evaluation of the "
                "reconstructed complete model under the recorded protocol."
            ),
        },
        "communication": {
            "actual_collective_payload_bytes_sum_across_ranks": sum(
                row[
                    "actual_collective_payload_bytes_sum_across_ranks"
                ]
                for row in included
            ),
            "ring_estimated_send_bytes_sum_across_ranks": sum(
                row["ring_estimated_send_bytes_sum_across_ranks"]
                for row in included
            ),
            "definition": (
                "Actual collective payload is measured tensor input bytes "
                "passed to gradient all-reduce. Ring send bytes use "
                "2*(group_size-1)/group_size*payload and are an estimate, "
                "not measured NIC/PCIe traffic."
            ),
        },
        "memory": {
            "ranks": rank_summaries,
            "definition": (
                "Memory utilization (MU) is peak torch CUDA allocated "
                "bytes divided by device total bytes. It is distinct from "
                "reserved memory, NVML used memory, MFU, and GPU busy."
            ),
        },
        "nvml": {
            "source_files": nvml_paths,
            "gpu_busy_reported": False,
            "gpu_busy_percent": None,
            "note": (
                "No NVML output was collected; no GPU-busy value is "
                "inferred."
                if not nvml_paths
                else "NVML files exist, but this summarizer does not "
                "combine them without a validated rank/time mapping."
            ),
        },
        "acceptance": acceptance,
        "evidence": {
            "collection_manifest": str(collection_manifest_path),
            "rank_metrics": {
                str(rank): str(path)
                for rank, path in metric_paths.items()
            },
            "rank_metadata": {
                f"{rank}:{generation}": str(path)
                for (rank, generation), path in metadata_paths.items()
            },
        },
    }

    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "microsteps.csv"
    rank_fields = [
        field
        for rank in range(world_size)
        for field in (
            f"rank_{rank}_loss",
            f"rank_{rank}_step_seconds",
            f"rank_{rank}_local_batch_size",
            f"rank_{rank}_valid_shifted_labels",
        )
    ]
    base_fields = [
        "microstep",
        "included",
        "exclusion_reason",
        "warmup",
        "overflow",
        "optimizer_step_succeeded",
        "synchronized_step_seconds",
        "global_samples",
        "effective_shifted_targets",
        "samples_per_second",
        "effective_tokens_per_second",
        "token_weighted_local_submodel_loss",
        "local_submodel_loss_mean",
        "local_submodel_loss_min",
        "local_submodel_loss_max",
        "actual_collective_payload_bytes_sum_across_ranks",
        "ring_estimated_send_bytes_sum_across_ranks",
    ]
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=base_fields + rank_fields,
        )
        writer.writeheader()
        writer.writerows(per_step)
    summary_path = output / "summary.json"
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(
            summary,
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.collection, args.output)
    print(
        json.dumps(
            {
                "summary": str(args.output.resolve() / "summary.json"),
                "microsteps": str(
                    args.output.resolve() / "microsteps.csv"
                ),
                "world_size": summary["world_size"],
                "attempted_microsteps": summary[
                    "attempted_microsteps"
                ],
                "included_microsteps": summary[
                    "included_post_warmup_successful_microsteps"
                ],
                "measurement_classification": summary[
                    "measurement_classification"
                ],
                "final_mmlu_accuracy": summary[
                    "full_model_mmlu"
                ]["final_accuracy"],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
