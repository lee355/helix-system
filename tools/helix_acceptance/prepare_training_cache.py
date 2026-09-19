#!/usr/bin/env python3
"""Remove exact normalized MMLU question matches from a MathInstruct cache.

The audit deliberately compares only the original MathInstruct ``instruction``
field with MMLU dev/test ``question`` stems.  It never reads MMLU answers when
deciding which training rows to exclude, and it makes no claim about semantic
or near-duplicate contamination.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
STEP1_ROOT = REPO_ROOT / "training" / "step1_supervised_finetuning"
PROTOCOL_VERSION = "helix-mathinstruct-mmlu-exact-question-filter-v1"
LIMITATION = (
    "This audit removes only exact question-string matches after the declared "
    "normalization; it cannot establish the absence of paraphrases or other "
    "near-duplicates."
)
REQUIRED_TENSORS = ("input_ids", "attention_mask", "labels")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def normalize_question(text: str) -> str:
    """Apply the complete, intentionally narrow exact-match normalization."""

    if not isinstance(text, str):
        raise TypeError(f"Question text must be a string, found {type(text).__name__}")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _safe_load_cache(path: Path) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This tool requires a PyTorch version supporting "
            "torch.load(..., weights_only=True); unsafe pickle loading is disabled"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Training cache must contain a dict, found {type(payload).__name__}")
    unexpected = sorted(set(payload).difference((*REQUIRED_TENSORS, "metadata")))
    if unexpected:
        raise ValueError(
            "Training cache has unsupported top-level fields that cannot be safely "
            f"row-filtered: {unexpected}"
        )
    missing = sorted(set((*REQUIRED_TENSORS, "metadata")).difference(payload))
    if missing:
        raise ValueError(f"Training cache is missing required fields: {missing}")
    return payload


def _validate_cache(
    payload: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    source_sha256: str,
) -> Tuple[Mapping[str, Any], List[int], Tuple[int, int]]:
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("Training cache metadata must be a dict")
    if metadata.get("dataset") != "MathInstruct-PoT":
        raise ValueError(
            "Training cache metadata.dataset must be 'MathInstruct-PoT', found "
            f"{metadata.get('dataset')!r}"
        )

    tensors: List[torch.Tensor] = []
    for name in REQUIRED_TENSORS:
        tensor = payload[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Training cache {name!r} must be a torch.Tensor")
        if tensor.device.type != "cpu":
            raise ValueError(f"Training cache {name!r} was not loaded on CPU")
        if tensor.ndim != 2:
            raise ValueError(
                f"Training cache {name!r} must be rank 2, found shape {tuple(tensor.shape)}"
            )
        tensors.append(tensor)
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) != 1:
        raise ValueError(
            "input_ids, attention_mask, and labels must have identical shapes; "
            f"found {[tuple(tensor.shape) for tensor in tensors]}"
        )
    shape = tuple(tensors[0].shape)
    if shape[0] <= 0 or shape[1] <= 1:
        raise ValueError(f"Training cache has unusable tensor shape {shape}")

    source_indices = metadata.get("source_indices_in_training_order")
    if not isinstance(source_indices, list):
        raise ValueError("metadata.source_indices_in_training_order must be a list")
    if len(source_indices) != shape[0]:
        raise ValueError(
            "source_indices_in_training_order length does not match cache rows: "
            f"{len(source_indices)} != {shape[0]}"
        )
    if metadata.get("kept_rows") != shape[0]:
        raise ValueError(
            f"metadata.kept_rows={metadata.get('kept_rows')!r} does not match "
            f"cache rows={shape[0]}"
        )
    if metadata.get("sequence_length") != shape[1]:
        raise ValueError(
            f"metadata.sequence_length={metadata.get('sequence_length')!r} does "
            f"not match tensor width={shape[1]}"
        )
    if metadata.get("source_total_rows") != len(source_rows):
        raise ValueError(
            f"metadata.source_total_rows={metadata.get('source_total_rows')!r} "
            f"does not match source JSON rows={len(source_rows)}"
        )
    if metadata.get("source_sha256") != source_sha256:
        raise ValueError(
            "Source JSON SHA256 does not match metadata.source_sha256: "
            f"{source_sha256} != {metadata.get('source_sha256')!r}"
        )

    checked_indices: List[int] = []
    for training_row, source_index in enumerate(source_indices):
        if (
            not isinstance(source_index, int)
            or isinstance(source_index, bool)
            or not 0 <= source_index < len(source_rows)
        ):
            raise ValueError(
                f"source_indices_in_training_order[{training_row}]={source_index!r} "
                f"is outside [0, {len(source_rows)})"
            )
        source_row = source_rows[source_index]
        if not isinstance(source_row, dict):
            raise ValueError(f"Source JSON row {source_index} must be an object")
        if not isinstance(source_row.get("instruction"), str):
            raise ValueError(f"Source JSON row {source_index} has no string instruction")
        if "PoT" not in str(source_row.get("source", "")):
            raise ValueError(
                f"Cache source index {source_index} does not identify a MathInstruct PoT row"
            )
        checked_indices.append(source_index)
    if len(set(checked_indices)) != len(checked_indices):
        raise ValueError("source_indices_in_training_order contains duplicate source indices")
    return metadata, checked_indices, shape  # type: ignore[return-value]


def _load_source_json(path: Path) -> Tuple[List[Mapping[str, Any]], str]:
    raw = path.read_bytes()
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid MathInstruct source JSON: {exc}") from exc
    if not isinstance(rows, list):
        raise ValueError("MathInstruct source JSON must contain a list")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"MathInstruct source row {index} must be an object")
    return rows, hashlib.sha256(raw).hexdigest()


def _load_mmlu_question_stems(
    path: Path,
) -> Tuple[Dict[str, Tuple[str, ...]], Dict[str, Any]]:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if manifest_path.name != "manifest.json" or not manifest_path.is_file():
        raise FileNotFoundError(
            f"--mmlu-path must name a prepared bundle or manifest.json: {path}"
        )
    if str(STEP1_ROOT) not in sys.path:
        sys.path.insert(0, str(STEP1_ROOT))
    from dschat.helix.mmlu_math import load_mmlu_math

    data = load_mmlu_math(manifest_path)
    normalized_to_ids: Dict[str, List[str]] = {}
    split_counts: Dict[str, int] = {}
    for split, questions in (("dev", data.dev), ("test", data.test)):
        split_counts[split] = len(questions)
        for question in questions:
            normalized = normalize_question(question.question)
            if not normalized:
                raise ValueError(f"MMLU question {question.question_id} normalizes to empty text")
            normalized_to_ids.setdefault(normalized, []).append(question.question_id)
    return (
        {key: tuple(value) for key, value in normalized_to_ids.items()},
        {
            "dataset_sha256": data.data_sha256,
            "manifest_sha256": _sha256_file(manifest_path),
            "split_counts": split_counts,
            "question_count": sum(split_counts.values()),
        },
    )


def _protocol_descriptor(
    *,
    input_cache_sha256: str,
    source_json_sha256: str,
    mmlu_info: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "inputs": {
            "training_cache_sha256": input_cache_sha256,
            "source_json_sha256": source_json_sha256,
            "mmlu_dataset_sha256": mmlu_info["dataset_sha256"],
            "mmlu_manifest_sha256": mmlu_info["manifest_sha256"],
        },
        "training_field": "instruction",
        "mmlu_field": "question",
        "mmlu_splits": ["dev", "test"],
        "normalization_in_order": [
            "Unicode NFKC",
            "Unicode casefold",
            "collapse and strip Unicode whitespace",
        ],
        "comparison": "exact equality of the complete normalized strings",
        "duplicate_policy": "exclude every cached training row with any exact match",
        "retained_order": "original cache row order with excluded rows removed",
        "mmlu_answer_or_choice_fields_used_for_selection": False,
        "limitation": LIMITATION,
    }


def prepare_training_cache(
    cache: Union[str, Path],
    source_json: Union[str, Path],
    mmlu_path: Union[str, Path],
    output: Union[str, Path],
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Filter one trusted tensor-only training cache and save a CPU cache."""

    cache_path = Path(cache).expanduser()
    source_path = Path(source_json).expanduser()
    mmlu_bundle_path = Path(mmlu_path).expanduser()
    output_path = Path(output).expanduser()
    if output_path.resolve() == cache_path.resolve():
        raise ValueError("Output cache must not overwrite the input cache")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    input_cache_sha256 = _sha256_file(cache_path)
    source_rows, source_json_sha256 = _load_source_json(source_path)
    payload = _safe_load_cache(cache_path)
    metadata, source_indices, original_shape = _validate_cache(
        payload, source_rows, source_json_sha256
    )
    normalized_mmlu, mmlu_info = _load_mmlu_question_stems(mmlu_bundle_path)

    keep_training_rows: List[int] = []
    exclusions: List[Dict[str, Any]] = []
    for training_row, source_index in enumerate(source_indices):
        instruction = source_rows[source_index]["instruction"]
        normalized = normalize_question(instruction)
        matched_question_ids = normalized_mmlu.get(normalized)
        if matched_question_ids:
            exclusions.append(
                {
                    "training_row": training_row,
                    "source_index": source_index,
                    "matched_question_ids": list(matched_question_ids),
                }
            )
        else:
            keep_training_rows.append(training_row)
    if not keep_training_rows:
        raise ValueError("Exact-question filtering would leave an empty training cache")

    filtered_source_indices = [source_indices[index] for index in keep_training_rows]
    row_index = torch.tensor(keep_training_rows, dtype=torch.long)
    filtered_payload: Dict[str, Any] = {
        name: payload[name].index_select(0, row_index).contiguous()
        for name in REQUIRED_TENSORS
    }

    protocol = _protocol_descriptor(
        input_cache_sha256=input_cache_sha256,
        source_json_sha256=source_json_sha256,
        mmlu_info=mmlu_info,
    )
    protocol_hash = hashlib.sha256(_canonical_json(protocol)).hexdigest()
    filtered_metadata = copy.deepcopy(metadata)
    filtered_metadata["pre_mmlu_filter_kept_rows"] = original_shape[0]
    filtered_metadata["kept_rows"] = len(keep_training_rows)
    filtered_metadata["source_indices_in_training_order"] = filtered_source_indices
    filtered_metadata["mmlu_exact_question_filter"] = {
        "protocol": protocol,
        "protocol_hash": protocol_hash,
        "mmlu_split_counts": mmlu_info["split_counts"],
        "mmlu_question_count": mmlu_info["question_count"],
        "excluded_rows": len(exclusions),
        "excluded_training_source_indices": [
            exclusion["source_index"] for exclusion in exclusions
        ],
        "exclusions": exclusions,
        "answer_guided_selection": False,
        "audit_scope": "exact normalized full-question equality only",
        "limitation": LIMITATION,
    }
    filtered_payload["metadata"] = filtered_metadata

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.tmp-{os.getpid()}"
    )
    try:
        torch.save(filtered_payload, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    output_sha256 = _sha256_file(output_path)

    matched_ids = sorted(
        {
            question_id
            for exclusion in exclusions
            for question_id in exclusion["matched_question_ids"]
        }
    )
    return {
        "output": str(output_path.resolve()),
        "output_sha256": output_sha256,
        "input_cache_sha256": input_cache_sha256,
        "source_json_sha256": source_json_sha256,
        "mmlu_dataset_sha256": mmlu_info["dataset_sha256"],
        "mmlu_manifest_sha256": mmlu_info["manifest_sha256"],
        "rows_before": original_shape[0],
        "rows_after": len(keep_training_rows),
        "excluded_rows": len(exclusions),
        "excluded_training_source_indices": [
            exclusion["source_index"] for exclusion in exclusions
        ],
        "matched_mmlu_question_ids": matched_ids,
        "exclusions": exclusions,
        "protocol_hash": protocol_hash,
        "answer_guided_selection": False,
        "limitation": LIMITATION,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove exact normalized MMLU dev/test question-stem matches from a "
            "historical MathInstruct-PoT tensor cache without re-tokenizing it."
        )
    )
    parser.add_argument("--cache", required=True, help="Historical MathInstruct-PoT .pt cache")
    parser.add_argument("--source-json", required=True, help="Original MathInstruct JSON")
    parser.add_argument("--mmlu-path", required=True, help="Prepared MMLU evaluation bundle")
    parser.add_argument("--output", required=True, help="Filtered output .pt path")
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output cache"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_training_cache(
        args.cache,
        args.source_json,
        args.mmlu_path,
        args.output,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
