#!/usr/bin/env python
"""Aggregate rank-local evidence; never confuse ring estimates with wire counters."""
import argparse
import csv
import datetime
import json
import math
import statistics
from pathlib import Path


def mean(values):
    return statistics.mean(values) if values else None


def quantile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def token_loss(records):
    total = sum(row["valid_shifted_labels"] for row in records)
    return sum(row["loss"] * row["valid_shifted_labels"] for row in records) / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    paths = sorted((run / "metrics").glob("rank_*.jsonl"))
    if not paths:
        raise RuntimeError("No rank metrics found")
    paths.sort(key=lambda path: int(path.stem.split("_")[1]))
    records = {int(path.stem.split("_")[1]): [json.loads(line) for line in path.read_text().splitlines()] for path in paths}
    metadata = {rank: json.loads((run / "metrics" / f"rank_{rank}_metadata.json").read_text()) for rank in records}
    launch = json.loads((run / "launch.json").read_text())
    expected_ranks = set(range(len(launch["physical_gpus"])))
    if set(records) != expected_ranks or any(meta["world_size"] != len(expected_ranks) for meta in metadata.values()):
        raise RuntimeError("Missing rank evidence or inconsistent world size")
    if launch.get("exit_code") != 0:
        raise RuntimeError("Run did not finish successfully")
    indexed = {rank: {row["microstep"]: row for row in rows} for rank, rows in records.items()}
    common = sorted(set.intersection(*(set(rows) for rows in indexed.values())))
    if any(len(rows) != len(common) for rows in records.values()):
        raise RuntimeError("Ranks have incomplete/mismatched measurements")
    if any(row["loss"] is None or not math.isfinite(row["loss"]) for rows in records.values() for row in rows):
        raise RuntimeError("Non-finite loss")
    steady = [step for step in common if all(not rows[step]["warmup"] and rows[step]["optimizer_steps_succeeded"] == 1 for rows in indexed.values())]
    if not steady:
        raise RuntimeError("No successful steady-state steps")
    durations = [max(rows[step]["step_seconds"] for rows in indexed.values()) for step in steady]
    elapsed = sum(durations)
    all_steady = [rows[step] for rows in indexed.values() for step in steady]
    all_success = [step for step in common if all(rows[step]["optimizer_steps_succeeded"] == 1 for rows in indexed.values())]
    window = min(10, len(all_success))
    first = [rows[step] for rows in indexed.values() for step in all_success[:window]]
    last = [rows[step] for rows in indexed.values() for step in all_success[-window:]]
    left = min(rows[steady[0]]["started_unix"] for rows in indexed.values())
    right = max(rows[steady[-1]]["finished_unix"] for rows in indexed.values())
    nvml = {}
    with (run / "gpu.csv").open() as stream:
        for row in csv.DictReader(stream, skipinitialspace=True):
            stamp = datetime.datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc).timestamp()
            if left <= stamp <= right:
                nvml.setdefault(int(row["index"]), []).append(row)
    rank_stats = []
    for rank, rows in records.items():
        meta = metadata[rank]
        valid = [indexed[rank][step] for step in steady]
        phys = int(launch["physical_gpus"][rank]["index"])
        gpu = nvml.get(phys, [])
        total = meta["device"]["total_memory_bytes"]
        peak = max(row["cuda_peak_allocated_bytes"] for row in valid)
        sync = mean([row["gradient_sync_cuda_seconds"] for row in valid])
        rank_stats.append({
            "rank": rank, "physical_gpu": phys, "device": meta["device"]["name"],
            "parameters": meta["trainable_parameter_numel"],
            "query_heads": meta["local_model_config"]["num_attention_heads"],
            "kv_heads": meta["local_model_config"]["num_key_value_heads"],
            "ffn_size": meta["local_model_config"]["intermediate_size"],
            "micro_batch_size": valid[0]["local_batch_size"],
            "observed_learning_rates": sorted({lr for row in rows for lr in row.get("learning_rates", [])}),
            "successful_updates": sum(row["optimizer_steps_succeeded"] for row in rows),
            "overflow_steps": sum(row["overflow_steps"] for row in rows),
            "loss_first_10_successful": token_loss([indexed[rank][s] for s in all_success[:window]]),
            "loss_last_10_successful": token_loss([indexed[rank][s] for s in all_success[-window:]]),
            "step_seconds_mean": mean([row["step_seconds"] for row in valid]),
            "phase_seconds_mean": {key: mean([row["phase_cuda_seconds"].get(key, 0) for row in valid]) for key in ["forward", "backward", "optimizer"]},
            "gradient_sync_seconds_mean": sync,
            "gradient_sync_fraction_of_local_step": sync / mean([row["step_seconds"] for row in valid]),
            "payload_bytes_per_step": mean([row["gradient_all_reduce_payload_bytes"] for row in valid]),
            "ring_send_bytes_per_step_estimated": mean([row["gradient_all_reduce_ring_estimated_send_bytes"] for row in valid]),
            "collective_calls_per_step": mean([len(row["gradient_all_reduce_calls"]) for row in valid]),
            "cuda_peak_allocated_gib": peak / 1024**3,
            "cuda_peak_reserved_gib": max(row["cuda_peak_reserved_bytes"] for row in valid) / 1024**3,
            "mu_peak_allocated_percent": 100 * peak / total,
            "gpu_busy_mean_percent": mean([float(row["utilization.gpu [%]"]) for row in gpu]),
            "nvml_peak_used_gib": max([float(row["memory.used [MiB]"]) / 1024 for row in gpu], default=None),
            "nvml_sample_count": len(gpu),
        })
    summary = {
        "run": str(run), "attempted_steps": len(common), "steady_successful_steps": len(steady),
        "excluded_steps": len(common) - len(steady),
        "step_seconds_mean": mean(durations), "step_seconds_p50": quantile(durations, .5),
        "step_seconds_p95": quantile(durations, .95),
        "samples_per_second": sum(row["local_batch_size"] for row in all_steady) / elapsed,
        "padded_tokens_per_second": sum(row["input_slots"] for row in all_steady) / elapsed,
        "nonpadding_input_tokens_per_second": sum(row["attention_tokens"] for row in all_steady) / elapsed,
        "supervised_tokens_per_second": sum(row["valid_shifted_labels"] for row in all_steady) / elapsed,
        "token_weighted_loss_first_10_successful": token_loss(first),
        "token_weighted_loss_last_10_successful": token_loss(last),
        "cluster_ring_send_gib_per_step_estimated": sum(row["ring_send_bytes_per_step_estimated"] for row in rank_stats) / 1024**3,
        "steady_wall_window_seconds": right - left,
        "ranks": rank_stats,
        "definitions": {
            "throughput": "Sum actual rank batch/tokens divided by sum(max rank step time); warmup and any-rank overflow excluded. No startup/checkpoint cost.",
            "loss": "Observed local-submodel losses aggregated with valid shifted-label counts; not reconstructed full-model validation loss.",
            "MU": "Per-rank peak torch allocated bytes / CUDA device total memory; NVML used and reserved shown separately.",
            "communication": "Actual gradient all_reduce input payload; ring-send bytes are algorithm estimates, not NIC/PCIe hardware counters. Sync CUDA phase includes pack/unpack and waiting. Backward includes sync.",
            "nvml": "500ms samples between first and last steady successful step; UTC host clock; GPU busy is not MU or MFU.",
        },
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for rank, rows in records.items():
            rolling = [token_loss(rows[max(0, i - 9): i + 1]) for i in range(len(rows))]
            axes[0, 0].plot(common, rolling, label=f"Rank {rank} / GPU {rank_stats[rank]['physical_gpu']}")
        axes[0, 0].set(title="Local submodel loss (10-step token-weighted window)", xlabel="Attempted step", ylabel="Cross entropy")
        axes[0, 0].legend()
        axes[0, 1].plot(steady, durations)
        axes[0, 1].set(title="Synchronized slowest-rank step", xlabel="Successful step after warmup", ylabel="Seconds")
        labels = [f"GPU {row['physical_gpu']}" for row in rank_stats]
        axes[1, 0].bar(labels, [row["mu_peak_allocated_percent"] for row in rank_stats], label="MU: peak allocated / total")
        axes[1, 0].set(title="Peak memory utilization", ylabel="Percent", ylim=(0, 100))
        forward = [row["phase_seconds_mean"]["forward"] for row in rank_stats]
        backward = [row["phase_seconds_mean"]["backward"] - row["gradient_sync_seconds_mean"] for row in rank_stats]
        sync = [row["gradient_sync_seconds_mean"] for row in rank_stats]
        optim = [row["phase_seconds_mean"]["optimizer"] for row in rank_stats]
        bottom = [0] * len(labels)
        for name, values in [("Forward", forward), ("Backward excluding sync", backward), ("Gradient sync (incl. waits)", sync), ("Optimizer", optim)]:
            axes[1, 1].bar(labels, values, bottom=bottom, label=name)
            bottom = [x + y for x, y in zip(bottom, values)]
        axes[1, 1].set(title="Mean GPU phase durations", ylabel="Seconds")
        axes[1, 1].legend(fontsize=8)
        fig.suptitle("Helix / Llama-3.2-3B / 28 layers / FP16 / sequence 512")
        fig.savefig(run / "performance.png", dpi=160)
        plt.close(fig)
    except ImportError:
        pass
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
