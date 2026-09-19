#!/usr/bin/env python
"""Bounded local launcher with explicit GPU selection and NVML sampling."""
import argparse
import csv
import json
import os
import runpy
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENTRY_ROOT = ROOT / "training/step1_supervised_finetuning"


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        entry, rest = sys.argv[2], sys.argv[3:]
        sys.path.insert(0, str(ENTRY_ROOT))
        sys.argv = [str(ENTRY_ROOT / entry), *rest, "--local_rank", os.environ["LOCAL_RANK"]]
        runpy.run_path(sys.argv[0], run_name="__main__")
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True, help="Physical GPU indices, e.g. 2,3,4")
    parser.add_argument("--output", required=True)
    parser.add_argument("--entry", choices=["helix_run.py", "helix_profile.py"], default="helix_run.py")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    gpu_ids = args.gpus.split(",")
    if len(set(gpu_ids)) != len(gpu_ids) or not all(i.isdecimal() for i in gpu_ids):
        parser.error("GPU indices must be unique integers")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "launch.json").exists():
        parser.error("Choose a fresh output directory to preserve previous results")
    query = subprocess.check_output([
        "nvidia-smi", f"--id={args.gpus}",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,nounits"], text=True)
    gpu_rows = list(csv.DictReader(query.splitlines(), skipinitialspace=True))
    if any(int(row["memory.used [MiB]"]) > 512 or int(row["utilization.gpu [%]"]) > 5 for row in gpu_rows):
        raise RuntimeError("Selected GPUs are occupied; choose idle devices: " + query)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpus, CUDA_DEVICE_ORDER="PCI_BUS_ID",
               OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.setdefault("NCCL_DEBUG", "WARN")
    env["PYTHONPATH"] = os.pathsep.join(map(str, [
        ROOT / "third_party/transformers/src",
        ROOT / "third_party/deepspeed",
        ROOT / "third_party/thop", ENTRY_ROOT,
    ]))
    rest = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               f"--nproc_per_node={len(gpu_ids)}", str(Path(__file__).resolve()),
               "--worker", args.entry, *rest]
    started = time.time()
    recorded_env = ["CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
                    "OMP_NUM_THREADS", "NCCL_DEBUG", "PYTHONPATH"]
    manifest = {"command": command, "physical_gpus": gpu_rows, "started_unix": started,
                "timeout_seconds": args.timeout,
                "environment": {k: env[k] for k in recorded_env if k in env}}
    (output / "launch.json").write_text(json.dumps(manifest, indent=2))
    print(f"Launching {args.entry} on physical GPU {args.gpus}; logs: {output}", flush=True)
    with (output / "train.log").open("w") as log, (output / "gpu.csv").open("w") as gpu_log:
        monitor = subprocess.Popen(["nvidia-smi", f"--id={args.gpus}",
            "--query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
            "--format=csv,nounits", "--loop-ms=500"], stdout=gpu_log, stderr=subprocess.STDOUT)
        child = None
        try:
            child = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                code = 124
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=20)
            monitor.terminate()
            monitor.wait(timeout=10)
    manifest.update(finished_unix=time.time(), exit_code=code)
    (output / "launch.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"exit_code": code, "elapsed_seconds": time.time() - started, "output": str(output)}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
