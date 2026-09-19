#!/usr/bin/env python
"""Select a deterministic, length-bounded subset of the local MathInstruct data."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/step1_supervised_finetuning"))
from vendor_bootstrap import activate_local_dependencies
activate_local_dependencies()
from dschat.utils.utils import load_hf_tokenizer
from dschat.utils.data.math_utils import PROMPT_TEMPLATE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sequence-length", type=int, default=512)
    args = parser.parse_args()
    tokenizer = load_hf_tokenizer(args.model, fast_tokenizer=True,
                                  model_max_length=args.sequence_length, padding_side="right")
    source = Path(args.source)
    data = json.loads(source.read_text())
    candidates = [i for i, row in enumerate(data) if "PoT" in row.get("source", "")]
    random.Random(args.seed).shuffle(candidates)
    template = random.Random(args.seed).choice(PROMPT_TEMPLATE)
    selected, indices, lengths = [], [], []
    for index in candidates:
        row = data[index]
        key = "prompt_input" if row.get("input", "") else "prompt_no_input"
        prompt = template[key].format_map(row)
        target = row["output"] + tokenizer.eos_token
        # Exclude truncation and near-empty responses so every rank has valid supervision.
        p = len(tokenizer(prompt)["input_ids"])
        n = len(tokenizer(prompt + target)["input_ids"])
        if n <= args.sequence_length and n - p >= 16:
            selected.append(row)
            indices.append(index)
            lengths.append({"tokens": n, "target_tokens": n - p})
            if len(selected) == args.count:
                break
    if len(selected) != args.count:
        raise RuntimeError(f"Only {len(selected)} suitable examples")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, ensure_ascii=False))
    manifest = {
        "source": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "subset_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "count": len(selected), "seed": args.seed, "sequence_length": args.sequence_length,
        "source_indices": indices, "token_lengths": lengths,
        "prompt_template": template,
        "note": "Real PoT examples, deterministic shuffled subset; legacy dataset additionally drops its longest character-length example. No held-out quality evaluation.",
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"count": len(selected), "output": str(output),
                      "mean_tokens": sum(x["tokens"] for x in lengths) / len(lengths)}))


if __name__ == "__main__":
    main()
