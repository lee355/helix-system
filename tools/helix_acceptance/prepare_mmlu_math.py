#!/usr/bin/env python3
"""Prepare the fixed five-subject MMLU mathematics acceptance dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
STEP1_ROOT = REPO_ROOT / "training" / "step1_supervised_finetuning"
sys.path.insert(0, str(STEP1_ROOT))

from dschat.helix.mmlu_math import prepare_mmlu_math_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy only the five selected MMLU subjects' dev/test records into "
            "hash-verified JSONL files for Helix acceptance evaluation."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="MMLU root containing data/dev and data/test, or the data directory itself",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New directory for manifest.json, dev.jsonl, and test.jsonl",
    )
    parser.add_argument(
        "--source-revision",
        default=None,
        help=(
            "Upstream dataset revision/commit. If omitted, the manifest uses a "
            "combined SHA256 of the ten selected CSV files as its revision."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing prepared files in output-dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_mmlu_math_data(
        args.source,
        args.output_dir,
        source_revision=args.source_revision,
        overwrite=args.overwrite,
    )
    summary = {
        "manifest": str(Path(args.output_dir).expanduser() / "manifest.json"),
        "dataset_sha256": manifest["dataset_sha256"],
        "source_revision": manifest["source"]["revision"],
        "splits": {
            split: {
                "count": info["count"],
                "per_subject": info["per_subject"],
                "sha256": info["sha256"],
            }
            for split, info in manifest["splits"].items()
        },
        "training_separation": manifest["training_separation"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
