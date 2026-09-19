#!/usr/bin/env python
"""Compare the acceptance evaluator against an independent checkpoint evaluation."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from evaluate_checkpoint import (
    AutoModelForCausalLM, AutoTokenizer, load_mmlu_math, model_state_digest, torch,
)
from dschat.helix.acceptance import MathAcceptance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mmlu-path", required=True)
    parser.add_argument("--reference-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    reference = json.loads(Path(args.reference_result).read_text())
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, low_cpu_mem_usage=False,
        attn_implementation="sdpa", local_files_only=True,
    ).eval().requires_grad_(False)
    controller = MathAcceptance.__new__(MathAcceptance)
    controller.args = SimpleNamespace(
        helix_eval_batch_size=1, helix_eval_max_length=4096, helix_eval_fewshot=5,
    )
    controller.eval_model = None
    controller.tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, local_files_only=True, model_max_length=4096,
        padding_side="right",
    )
    controller.tokenizer.pad_token = controller.tokenizer.eos_token
    controller.data = load_mmlu_math(args.mmlu_path)
    result = controller._evaluate_on_root(SimpleNamespace(
        full_model=model, device=torch.device("cuda:0"),
    ))
    observed = {row["question_id"]: row for row in result["predictions"]}
    expected = {row["question_id"]: row for row in reference["predictions"]}
    if observed.keys() != expected.keys():
        raise ValueError("Different question coverage")
    comparison = {
        "weights_sha256_equal": model_state_digest(controller.eval_model) == reference["model_state_sha256_fp16"],
        "protocol_equal": result["protocol_hash"] == reference["protocol_hash"],
        "prediction_mismatches": [key for key in expected if expected[key]["prediction"] != observed[key]["prediction"]],
        "max_choice_loglikelihood_absolute_difference": max(
            abs(expected[key]["choice_loglikelihoods"][choice] - observed[key]["choice_loglikelihoods"][choice])
            for key in expected for choice in "ABCD"
        ),
        "nonpersistent_rope_buffer_dtypes": {
            name: str(value.dtype) for name, value in controller.eval_model.named_buffers()
            if name.endswith("inv_freq")
        },
    }
    comparison["passed"] = bool(comparison["weights_sha256_equal"] and comparison["protocol_equal"]
                                and not comparison["prediction_mismatches"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"comparison": comparison, "evaluation": result}, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))
    raise SystemExit(0 if comparison["passed"] else 1)


if __name__ == "__main__":
    main()
