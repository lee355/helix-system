#!/usr/bin/env python
"""Evaluate a complete checkpoint using the fixed acceptance protocol."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/step1_supervised_finetuning"))
from vendor_bootstrap import activate_local_dependencies
activate_local_dependencies()

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from dschat.helix.acceptance import model_state_digest
from dschat.helix.mmlu_math import load_mmlu_math, evaluate_mmlu_math


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mmlu-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--fewshot", type=int, default=5)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("Choose a fresh evaluation output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data = load_mmlu_math(args.mmlu_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, use_fast=True,
                                              model_max_length=args.max_length, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=False,
        attn_implementation="sdpa", local_files_only=True,
    ).eval().requires_grad_(False)
    digest = model_state_digest(model)
    model.to(args.device)
    eval_started = time.perf_counter()
    result = evaluate_mmlu_math(model, tokenizer, data, device=args.device,
                               batch_size=args.batch_size, max_length=args.max_length,
                               num_fewshot=args.fewshot, overflow_policy="error")
    result.update(model=str(Path(args.model).resolve()), model_state_sha256_fp16=digest,
                  model_config_sha256=hashlib.sha256((Path(args.model) / "config.json").read_bytes()).hexdigest(),
                  model_layers=model.config.num_hidden_layers,
                  unique_parameters=sum(p.numel() for p in model.parameters()),
                  torch_version=torch.__version__, transformers_version=transformers.__version__,
                  evaluation_seconds=time.perf_counter() - eval_started,
                  total_seconds=time.perf_counter() - started,
                  target_accuracy=0.62, target_reached=result["accuracy"] >= 0.62)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"predictions", "protocol"}}, indent=2))


if __name__ == "__main__":
    main()
