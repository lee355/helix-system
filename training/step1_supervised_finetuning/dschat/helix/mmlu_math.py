"""Reproducible five-subject MMLU mathematics evaluation.

This module intentionally has no dependency on lm-evaluation-harness or the
training data pipeline.  It consumes only the MMLU ``dev`` examples used for
few-shot prompting and the ``test`` examples used for evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch


MMLU_MATH_SUBJECTS: Tuple[str, ...] = (
    "abstract_algebra",
    "college_mathematics",
    "elementary_mathematics",
    "high_school_mathematics",
    "high_school_statistics",
)
CHOICE_LABELS: Tuple[str, ...] = ("A", "B", "C", "D")
CHOICE_CONTINUATIONS: Tuple[str, ...] = (" A", " B", " C", " D")
PROTOCOL_VERSION = "helix-mmlu-math-v1"
MANIFEST_SCHEMA_VERSION = 1


class MMLUMathError(RuntimeError):
    """Base class for MMLU mathematics preparation/evaluation errors."""


class PromptTooLongError(MMLUMathError):
    """Raised instead of silently truncating an evaluation prompt."""


class TokenizationBoundaryError(MMLUMathError):
    """Raised when a tokenizer does not preserve the prompt/answer boundary."""


@dataclass(frozen=True)
class MMLUQuestion:
    subject: str
    split: str
    index: int
    question: str
    choices: Tuple[str, str, str, str]
    answer: int

    @property
    def question_id(self) -> str:
        return f"{self.subject}:{self.split}:{self.index}"


@dataclass(frozen=True)
class MMLUMathData:
    dev: Tuple[MMLUQuestion, ...]
    test: Tuple[MMLUQuestion, ...]
    data_sha256: str
    source: Mapping[str, Any]
    manifest: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class _CandidateScore:
    choice_index: int
    positions: Tuple[int, ...]
    target_ids: Tuple[int, ...]


@dataclass(frozen=True)
class _ModelRequest:
    question_index: int
    input_ids: Tuple[int, ...]
    candidates: Tuple[_CandidateScore, ...]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _question_record(question: MMLUQuestion) -> Dict[str, Any]:
    return {
        "subject": question.subject,
        "split": question.split,
        "index": question.index,
        "question": question.question,
        "choices": list(question.choices),
        "answer": question.answer,
        "answer_label": CHOICE_LABELS[question.answer],
    }


def _data_sha256(
    dev: Sequence[MMLUQuestion], test: Sequence[MMLUQuestion]
) -> str:
    digest = hashlib.sha256()
    for split, questions in (("dev", dev), ("test", test)):
        digest.update(split.encode("ascii") + b"\0")
        for question in questions:
            digest.update(_canonical_json(_question_record(question)) + b"\n")
    return digest.hexdigest()


def _resolve_csv_root(path: Path) -> Path:
    candidates = (path, path / "data")
    for candidate in candidates:
        if (candidate / "dev").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Expected MMLU CSV directories 'dev' and 'test' below {path} "
        f"or {path / 'data'}"
    )


def _read_subject_csv(path: Path, subject: str, split: str) -> Tuple[MMLUQuestion, ...]:
    questions: List[MMLUQuestion] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            if len(row) != 6:
                raise ValueError(
                    f"{path}:{row_number}: expected 6 CSV columns, found {len(row)}"
                )
            answer_label = row[5].strip()
            if answer_label not in CHOICE_LABELS:
                raise ValueError(
                    f"{path}:{row_number}: invalid answer label {answer_label!r}"
                )
            questions.append(
                MMLUQuestion(
                    subject=subject,
                    split=split,
                    index=len(questions),
                    question=row[0].strip(),
                    choices=tuple(row[1:5]),  # type: ignore[arg-type]
                    answer=CHOICE_LABELS.index(answer_label),
                )
            )
    if not questions:
        raise ValueError(f"{path}: MMLU split must not be empty")
    return tuple(questions)


def _load_csv_questions(csv_root: Path) -> Tuple[Tuple[MMLUQuestion, ...], Tuple[MMLUQuestion, ...], Dict[str, str]]:
    by_split: Dict[str, List[MMLUQuestion]] = {"dev": [], "test": []}
    source_hashes: Dict[str, str] = {}
    for split in ("dev", "test"):
        for subject in MMLU_MATH_SUBJECTS:
            relative = Path(split) / f"{subject}_{split}.csv"
            source_path = csv_root / relative
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing MMLU source file: {source_path}")
            by_split[split].extend(_read_subject_csv(source_path, subject, split))
            source_hashes[relative.as_posix()] = _sha256_file(source_path)
    return tuple(by_split["dev"]), tuple(by_split["test"]), source_hashes


def _validate_fixed_subjects(questions: Sequence[MMLUQuestion], split: str) -> None:
    counts = {subject: 0 for subject in MMLU_MATH_SUBJECTS}
    indices = {subject: [] for subject in MMLU_MATH_SUBJECTS}
    for question in questions:
        if question.split != split:
            raise ValueError(
                f"Question {question.question_id} is in {question.split!r}, expected {split!r}"
            )
        if question.subject not in counts:
            raise ValueError(f"Unexpected MMLU mathematics subject: {question.subject!r}")
        counts[question.subject] += 1
        indices[question.subject].append(question.index)
    missing = [subject for subject, count in counts.items() if count == 0]
    if missing:
        raise ValueError(f"MMLU {split} split is missing subjects: {missing}")
    for subject, subject_indices in indices.items():
        if subject_indices != list(range(len(subject_indices))):
            raise ValueError(
                f"MMLU {split}/{subject} indices must be contiguous and start at zero"
            )


def _question_from_record(record: Mapping[str, Any], expected_split: str) -> MMLUQuestion:
    required = {"subject", "split", "index", "question", "choices", "answer"}
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"JSONL question record is missing keys: {missing}")
    choices = record["choices"]
    if not isinstance(choices, list) or len(choices) != 4:
        raise ValueError("MMLU JSONL 'choices' must be a list of four strings")
    answer = record["answer"]
    if isinstance(answer, str):
        if answer not in CHOICE_LABELS:
            raise ValueError(f"Invalid MMLU answer label: {answer!r}")
        answer = CHOICE_LABELS.index(answer)
    if not isinstance(answer, int) or isinstance(answer, bool) or not 0 <= answer < 4:
        raise ValueError(f"Invalid MMLU answer index: {answer!r}")
    question = MMLUQuestion(
        subject=str(record["subject"]),
        split=str(record["split"]),
        index=int(record["index"]),
        question=str(record["question"]),
        choices=tuple(str(choice) for choice in choices),  # type: ignore[arg-type]
        answer=answer,
    )
    if question.split != expected_split:
        raise ValueError(
            f"Record {question.question_id} has split {question.split!r}; "
            f"expected {expected_split!r}"
        )
    return question


def _safe_manifest_child(manifest_path: Path, child: str) -> Path:
    candidate = (manifest_path.parent / child).resolve()
    root = manifest_path.parent.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Manifest path escapes its directory: {child!r}") from exc
    return candidate


def _load_manifest(path: Path) -> MMLUMathData:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported MMLU manifest schema: {manifest.get('schema_version')!r}"
        )
    if tuple(manifest.get("subjects", ())) != MMLU_MATH_SUBJECTS:
        raise ValueError(
            "MMLU manifest subjects do not match the fixed five-subject mathematics protocol"
        )

    loaded: Dict[str, Tuple[MMLUQuestion, ...]] = {}
    for split in ("dev", "test"):
        split_info = manifest.get("splits", {}).get(split)
        if not isinstance(split_info, dict) or "path" not in split_info:
            raise ValueError(f"MMLU manifest is missing splits.{split}.path")
        split_path = _safe_manifest_child(manifest_path, str(split_info["path"]))
        expected_hash = split_info.get("sha256")
        actual_hash = _sha256_file(split_path)
        if expected_hash != actual_hash:
            raise ValueError(
                f"MMLU {split} JSONL SHA256 mismatch: expected {expected_hash}, "
                f"found {actual_hash}"
            )
        questions: List[MMLUQuestion] = []
        with split_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"{split_path}:{line_number}: blank JSONL line")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{split_path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                questions.append(_question_from_record(record, split))
        if len(questions) != split_info.get("count"):
            raise ValueError(
                f"MMLU {split} count mismatch: manifest says {split_info.get('count')}, "
                f"JSONL has {len(questions)}"
            )
        actual_subject_counts = _split_counts(questions)
        if actual_subject_counts != split_info.get("per_subject"):
            raise ValueError(
                f"MMLU {split} per-subject counts do not match the manifest: "
                f"{actual_subject_counts}"
            )
        loaded[split] = tuple(questions)

    _validate_fixed_subjects(loaded["dev"], "dev")
    _validate_fixed_subjects(loaded["test"], "test")
    actual_data_hash = _data_sha256(loaded["dev"], loaded["test"])
    expected_data_hash = manifest.get("dataset_sha256")
    if expected_data_hash != actual_data_hash:
        raise ValueError(
            f"MMLU dataset SHA256 mismatch: expected {expected_data_hash}, "
            f"found {actual_data_hash}"
        )
    return MMLUMathData(
        dev=loaded["dev"],
        test=loaded["test"],
        data_sha256=actual_data_hash,
        source=dict(manifest.get("source", {})),
        manifest=manifest,
    )


def load_mmlu_math(path: Union[str, Path]) -> MMLUMathData:
    """Load and validate a prepared manifest/JSONL bundle or an MMLU CSV tree."""

    source_path = Path(path).expanduser()
    if source_path.is_file():
        if source_path.name != "manifest.json":
            raise ValueError(
                "A file input must be a prepared manifest.json; pass the CSV directory "
                "or prepared bundle directory otherwise"
            )
        return _load_manifest(source_path)
    if (source_path / "manifest.json").is_file():
        return _load_manifest(source_path)

    csv_root = _resolve_csv_root(source_path)
    dev, test, source_hashes = _load_csv_questions(csv_root)
    _validate_fixed_subjects(dev, "dev")
    _validate_fixed_subjects(test, "test")
    source_digest = hashlib.sha256(_canonical_json(source_hashes)).hexdigest()
    return MMLUMathData(
        dev=dev,
        test=test,
        data_sha256=_data_sha256(dev, test),
        source={
            "kind": "cais/mmlu CSV tree",
            "revision": f"selected-csv-sha256:{source_digest}",
            "selected_files_sha256": source_hashes,
        },
    )


def _split_counts(questions: Sequence[MMLUQuestion]) -> Dict[str, int]:
    counts = {subject: 0 for subject in MMLU_MATH_SUBJECTS}
    for question in questions:
        counts[question.subject] += 1
    return counts


def prepare_mmlu_math_data(
    source: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    source_revision: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Create a fixed, hash-verified dev/test JSONL bundle from MMLU CSVs."""

    source_path = Path(source).expanduser()
    csv_root = _resolve_csv_root(source_path)
    dev, test, source_hashes = _load_csv_questions(csv_root)
    _validate_fixed_subjects(dev, "dev")
    _validate_fixed_subjects(test, "test")

    selected_source_hash = hashlib.sha256(_canonical_json(source_hashes)).hexdigest()
    if source_revision is None:
        source_revision = f"selected-csv-sha256:{selected_source_hash}"

    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "dev": destination / "dev.jsonl",
        "test": destination / "test.jsonl",
        "manifest": destination / "manifest.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite prepared MMLU files: "
            + ", ".join(str(path) for path in existing)
        )

    split_questions = {"dev": dev, "test": test}
    split_manifest: Dict[str, Dict[str, Any]] = {}
    for split in ("dev", "test"):
        output_path = output_paths[split]
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            for question in split_questions[split]:
                handle.write(_canonical_json(_question_record(question)).decode("utf-8"))
                handle.write("\n")
        split_manifest[split] = {
            "path": output_path.name,
            "sha256": _sha256_file(output_path),
            "count": len(split_questions[split]),
            "per_subject": _split_counts(split_questions[split]),
        }

    source_info: Dict[str, Any] = {
        "dataset": "cais/mmlu",
        "format": "Hendrycks MMLU CSV",
        "revision": source_revision,
        "selected_files_sha256": source_hashes,
        "selected_files_combined_sha256": selected_source_hash,
    }
    archive_candidates = (csv_root.parent / "data.tar", source_path / "data.tar")
    for archive_path in archive_candidates:
        if archive_path.is_file():
            source_info["archive_sha256"] = _sha256_file(archive_path)
            break

    manifest: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": "mmlu_math_five_subject",
        "purpose": "evaluation_only",
        "subjects": list(MMLU_MATH_SUBJECTS),
        "source": source_info,
        "splits": split_manifest,
        "dataset_sha256": _data_sha256(dev, test),
        "training_separation": {
            "included_source_splits": ["dev", "test"],
            "excluded_source_splits": ["auxiliary_train", "validation", "val"],
            "fewshot_split": "dev",
            "evaluation_split": "test",
            "training_dataset": "MathInstruct (external to this bundle)",
        },
    }
    with output_paths["manifest"].open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def format_subject(subject: str) -> str:
    if subject not in MMLU_MATH_SUBJECTS:
        raise ValueError(f"Unsupported MMLU mathematics subject: {subject!r}")
    return subject.replace("_", " ")


def format_mmlu_question(question: MMLUQuestion, include_answer: bool) -> str:
    lines = [question.question.strip()]
    lines.extend(
        f"{label}. {choice}" for label, choice in zip(CHOICE_LABELS, question.choices)
    )
    text = "\n".join(lines) + "\nAnswer:"
    if include_answer:
        text += f" {CHOICE_LABELS[question.answer]}\n\n"
    return text


def build_mmlu_prompt(
    subject: str,
    dev_questions: Sequence[MMLUQuestion],
    test_question: MMLUQuestion,
    *,
    num_fewshot: int = 5,
) -> str:
    """Build the original MMLU dev-fewshot prompt for one test question."""

    if num_fewshot < 0:
        raise ValueError("num_fewshot must be non-negative")
    if test_question.subject != subject or test_question.split != "test":
        raise ValueError("Test question subject/split does not match the prompt")
    if len(dev_questions) < num_fewshot:
        raise ValueError(
            f"{subject} has {len(dev_questions)} dev examples, fewer than "
            f"the requested {num_fewshot}"
        )
    selected = dev_questions[:num_fewshot]
    for question in selected:
        if question.subject != subject or question.split != "dev":
            raise ValueError("Few-shot question subject/split does not match the prompt")
    header = (
        "The following are multiple choice questions (with answers) about "
        f"{format_subject(subject)}.\n\n"
    )
    return (
        header
        + "".join(format_mmlu_question(question, True) for question in selected)
        + format_mmlu_question(test_question, False)
    )


def protocol_descriptor(
    *,
    data_sha256: str,
    num_fewshot: int,
    max_length: int,
    overflow_policy: str,
) -> Dict[str, Any]:
    """Return a fresh JSON-compatible description of every semantic choice."""

    return {
        "version": PROTOCOL_VERSION,
        "dataset_sha256": data_sha256,
        "subjects": list(MMLU_MATH_SUBJECTS),
        "fewshot_split": "dev",
        "evaluation_split": "test",
        "fewshot_sampler": "first_n_in_source_order",
        "num_fewshot": num_fewshot,
        "prompt": {
            "header": "The following are multiple choice questions (with answers) about {subject words}.\\n\\n",
            "question": "{question}\\nA. {choice0}\\nB. {choice1}\\nC. {choice2}\\nD. {choice3}\\nAnswer:",
            "answered_suffix": " {gold}\\n\\n",
        },
        "choices": list(CHOICE_LABELS),
        "continuations": list(CHOICE_CONTINUATIONS),
        "tokenization": {
            "add_special_tokens": False,
            "boundary": "tokenize_prompt_and_prompt_plus_continuation_then_require_exact_prefix",
        },
        "scoring": "sum_causal_loglikelihood_over_all_continuation_tokens",
        "choice_selection": "first_argmax_in_A_B_C_D_order",
        "aggregation": "micro_average_over_test_questions",
        "max_length_including_continuation": max_length,
        "overflow_policy": overflow_policy,
    }


def protocol_hash(
    *,
    data_sha256: str,
    num_fewshot: int,
    max_length: int,
    overflow_policy: str,
) -> str:
    descriptor = protocol_descriptor(
        data_sha256=data_sha256,
        num_fewshot=num_fewshot,
        max_length=max_length,
        overflow_policy=overflow_policy,
    )
    return hashlib.sha256(_canonical_json(descriptor)).hexdigest()


def _tokenize(tokenizer: Any, text: str) -> Tuple[int, ...]:
    if hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
    else:
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.detach().cpu().reshape(-1).tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("Tokenizer returned an unexpected batched encoding")
        encoded = encoded[0]
    return tuple(int(token_id) for token_id in encoded)


def _encode_candidates(
    tokenizer: Any, prompt: str
) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]:
    context_ids = _tokenize(tokenizer, prompt)
    if not context_ids:
        raise TokenizationBoundaryError("The MMLU prompt tokenized to an empty sequence")
    continuations: List[Tuple[int, ...]] = []
    for continuation in CHOICE_CONTINUATIONS:
        whole_ids = _tokenize(tokenizer, prompt + continuation)
        if whole_ids[: len(context_ids)] != context_ids:
            raise TokenizationBoundaryError(
                "Tokenizer changed prompt tokens across the answer boundary for "
                f"continuation {continuation!r}; refusing an ambiguous score"
            )
        continuation_ids = whole_ids[len(context_ids) :]
        if not continuation_ids:
            raise TokenizationBoundaryError(
                f"Continuation {continuation!r} produced no answer tokens"
            )
        continuations.append(continuation_ids)
    return context_ids, tuple(continuations)


def _prepare_encoded_question(
    tokenizer: Any,
    dev_questions: Sequence[MMLUQuestion],
    question: MMLUQuestion,
    *,
    num_fewshot: int,
    max_length: int,
    overflow_policy: str,
) -> Tuple[str, Tuple[int, ...], Tuple[Tuple[int, ...], ...], int]:
    shots = num_fewshot
    while True:
        prompt = build_mmlu_prompt(
            question.subject,
            dev_questions,
            question,
            num_fewshot=shots,
        )
        context_ids, continuation_ids = _encode_candidates(tokenizer, prompt)
        longest = max(len(context_ids) + len(ids) for ids in continuation_ids)
        if longest <= max_length:
            return prompt, context_ids, continuation_ids, shots
        if overflow_policy == "error" or shots == 0:
            raise PromptTooLongError(
                f"MMLU prompt {question.question_id} requires {longest} tokens "
                f"including its answer continuation, exceeding max_length={max_length}; "
                f"num_fewshot={shots}, overflow_policy={overflow_policy!r}"
            )
        shots -= 1


def _supports_num_logits_to_keep(model: Any) -> bool:
    forward = getattr(model, "forward", None)
    if forward is None:
        return False
    try:
        return "num_logits_to_keep" in inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return False


def _resolve_device(model: Any, device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _resolve_max_length(model: Any, tokenizer: Any, max_length: Optional[int]) -> int:
    if max_length is not None:
        if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 1:
            raise ValueError("max_length must be an integer greater than one")
        return max_length
    candidates = [
        getattr(getattr(model, "config", None), "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, int) and 1 < candidate < 10**7:
            return candidate
    raise ValueError(
        "max_length was not supplied and could not be inferred from model/tokenizer"
    )


def _resolve_pad_token_id(tokenizer: Any) -> int:
    for attribute in ("pad_token_id", "eos_token_id"):
        token_id = getattr(tokenizer, attribute, None)
        if token_id is not None:
            return int(token_id)
    raise ValueError("Tokenizer must define pad_token_id or eos_token_id")


def _score_request_batch(
    model: Any,
    requests: Sequence[_ModelRequest],
    scores: List[List[Optional[float]]],
    *,
    device: torch.device,
    pad_token_id: int,
    padding_side: str,
    use_num_logits_to_keep: bool,
) -> None:
    padded_length = max(len(request.input_ids) for request in requests)
    input_ids = torch.full(
        (len(requests), padded_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    shifted_positions: List[Tuple[int, ...]] = []
    padding_offsets: List[int] = []
    for row, request in enumerate(requests):
        length = len(request.input_ids)
        if padding_side == "left":
            offset = padded_length - length
            input_ids[row, offset:] = torch.tensor(request.input_ids, device=device)
            attention_mask[row, offset:] = 1
        else:
            offset = 0
            input_ids[row, :length] = torch.tensor(request.input_ids, device=device)
            attention_mask[row, :length] = 1
        padding_offsets.append(offset)
        shifted_positions.append(
            tuple(position + offset for candidate in request.candidates for position in candidate.positions)
        )

    forward_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    requested_logits = None
    if use_num_logits_to_keep:
        earliest = min(min(positions) for positions in shifted_positions)
        requested_logits = padded_length - earliest
        forward_kwargs["num_logits_to_keep"] = requested_logits

    output = model(**forward_kwargs)
    logits = output.logits if hasattr(output, "logits") else output[0]
    if logits.ndim != 3 or logits.shape[0] != len(requests):
        raise ValueError(
            f"Model returned logits with shape {tuple(logits.shape)}, expected [batch, sequence, vocab]"
        )
    returned_length = logits.shape[1]
    if requested_logits is not None and returned_length not in (requested_logits, padded_length):
        raise ValueError(
            f"Model returned {returned_length} logits positions after "
            f"num_logits_to_keep={requested_logits}"
        )
    logits_offset = padded_length - returned_length

    for row, request in enumerate(requests):
        unique_positions = sorted(
            {position for candidate in request.candidates for position in candidate.positions}
        )
        adjusted_positions = [
            position + padding_offsets[row] - logits_offset
            for position in unique_positions
        ]
        if min(adjusted_positions) < 0 or max(adjusted_positions) >= returned_length:
            raise ValueError("Model omitted logits required to score an MMLU continuation")
        position_tensor = torch.tensor(adjusted_positions, dtype=torch.long, device=logits.device)
        selected_logits = logits[row].index_select(0, position_tensor)
        selected_logprobs = torch.log_softmax(selected_logits.float(), dim=-1)
        row_for_position = {position: index for index, position in enumerate(unique_positions)}
        for candidate in request.candidates:
            token_rows = torch.tensor(
                [row_for_position[position] for position in candidate.positions],
                dtype=torch.long,
                device=logits.device,
            )
            target_ids = torch.tensor(
                candidate.target_ids, dtype=torch.long, device=logits.device
            )
            candidate_score = selected_logprobs[token_rows, target_ids].sum().item()
            scores[request.question_index][candidate.choice_index] = candidate_score


def evaluate_mmlu_math(
    model: Any,
    tokenizer: Any,
    data: Union[MMLUMathData, str, Path],
    *,
    device: Optional[Union[str, torch.device]] = None,
    batch_size: int = 1,
    max_length: Optional[int] = None,
    num_fewshot: int = 5,
    overflow_policy: str = "error",
    padding_side: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a full causal HF model with strict MMLU answer loglikelihood.

    ``batch_size`` counts unique input prefixes/sequences.  The usual one-token
    A/B/C/D continuations share one prefix and therefore one forward request per
    question.  Multi-token continuations use a general sum-loglikelihood path.
    No labels are passed to the model.
    """

    if not isinstance(data, MMLUMathData):
        data = load_mmlu_math(data)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(num_fewshot, int) or isinstance(num_fewshot, bool) or num_fewshot < 0:
        raise ValueError("num_fewshot must be a non-negative integer")
    if overflow_policy not in ("error", "reduce_fewshot"):
        raise ValueError("overflow_policy must be 'error' or 'reduce_fewshot'")
    resolved_padding_side = padding_side or getattr(tokenizer, "padding_side", "right")
    if resolved_padding_side not in ("left", "right"):
        raise ValueError("padding_side must be 'left' or 'right'")
    resolved_max_length = _resolve_max_length(model, tokenizer, max_length)
    resolved_device = _resolve_device(model, device)
    pad_token_id = _resolve_pad_token_id(tokenizer)

    _validate_fixed_subjects(data.dev, "dev")
    _validate_fixed_subjects(data.test, "test")
    dev_by_subject: Dict[str, List[MMLUQuestion]] = {
        subject: [] for subject in MMLU_MATH_SUBJECTS
    }
    for question in data.dev:
        dev_by_subject[question.subject].append(question)
    for subject, dev_questions in dev_by_subject.items():
        if len(dev_questions) < num_fewshot:
            raise ValueError(
                f"{subject} has {len(dev_questions)} dev examples; "
                f"num_fewshot={num_fewshot} was requested"
            )

    predictions: List[Dict[str, Any]] = []
    score_slots: List[List[Optional[float]]] = []
    requests: List[_ModelRequest] = []
    for question_index, question in enumerate(data.test):
        _, context_ids, continuation_ids, actual_shots = _prepare_encoded_question(
            tokenizer,
            dev_by_subject[question.subject],
            question,
            num_fewshot=num_fewshot,
            max_length=resolved_max_length,
            overflow_policy=overflow_policy,
        )
        score_slots.append([None, None, None, None])
        predictions.append(
            {
                "question_id": question.question_id,
                "subject": question.subject,
                "index": question.index,
                "gold": CHOICE_LABELS[question.answer],
                "prompt_token_length": len(context_ids),
                "candidate_token_lengths": {
                    label: len(tokens)
                    for label, tokens in zip(CHOICE_LABELS, continuation_ids)
                },
                "max_sequence_token_length": max(
                    len(context_ids) + len(tokens) for tokens in continuation_ids
                ),
                "requested_num_fewshot": num_fewshot,
                "actual_num_fewshot": actual_shots,
                "fewshot_reduced": actual_shots != num_fewshot,
            }
        )

        if all(len(tokens) == 1 for tokens in continuation_ids):
            last_context_position = len(context_ids) - 1
            candidates = tuple(
                _CandidateScore(
                    choice_index=choice_index,
                    positions=(last_context_position,),
                    target_ids=tokens,
                )
                for choice_index, tokens in enumerate(continuation_ids)
            )
            requests.append(
                _ModelRequest(
                    question_index=question_index,
                    input_ids=context_ids,
                    candidates=candidates,
                )
            )
        else:
            for choice_index, tokens in enumerate(continuation_ids):
                full_ids = context_ids + tokens
                first_logit_position = len(context_ids) - 1
                requests.append(
                    _ModelRequest(
                        question_index=question_index,
                        input_ids=full_ids[:-1],
                        candidates=(
                            _CandidateScore(
                                choice_index=choice_index,
                                positions=tuple(
                                    range(
                                        first_logit_position,
                                        first_logit_position + len(tokens),
                                    )
                                ),
                                target_ids=tokens,
                            ),
                        ),
                    )
                )

    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    supports_keep = _supports_num_logits_to_keep(model)
    try:
        with torch.no_grad():
            for start in range(0, len(requests), batch_size):
                _score_request_batch(
                    model,
                    requests[start : start + batch_size],
                    score_slots,
                    device=resolved_device,
                    pad_token_id=pad_token_id,
                    padding_side=resolved_padding_side,
                    use_num_logits_to_keep=supports_keep,
                )
    finally:
        if hasattr(model, "train"):
            model.train(was_training)

    per_subject: Dict[str, Dict[str, Any]] = {
        subject: {"num_correct": 0, "total": 0, "accuracy": 0.0}
        for subject in MMLU_MATH_SUBJECTS
    }
    num_correct = 0
    for question, prediction, question_scores in zip(
        data.test, predictions, score_slots
    ):
        if any(score is None for score in question_scores):
            raise RuntimeError(f"Incomplete scores for {question.question_id}")
        numeric_scores = [float(score) for score in question_scores]  # type: ignore[arg-type]
        predicted_index = max(range(4), key=lambda index: numeric_scores[index])
        correct = predicted_index == question.answer
        prediction.update(
            {
                "prediction": CHOICE_LABELS[predicted_index],
                "correct": correct,
                "choice_loglikelihoods": {
                    label: score for label, score in zip(CHOICE_LABELS, numeric_scores)
                },
            }
        )
        subject_result = per_subject[question.subject]
        subject_result["total"] += 1
        subject_result["num_correct"] += int(correct)
        num_correct += int(correct)

    for subject_result in per_subject.values():
        subject_result["accuracy"] = (
            subject_result["num_correct"] / subject_result["total"]
        )
    total = len(predictions)
    prompt_lengths = [prediction["prompt_token_length"] for prediction in predictions]
    actual_shots = [prediction["actual_num_fewshot"] for prediction in predictions]
    shot_histogram = {
        str(shots): actual_shots.count(shots) for shots in sorted(set(actual_shots))
    }
    descriptor = protocol_descriptor(
        data_sha256=data.data_sha256,
        num_fewshot=num_fewshot,
        max_length=resolved_max_length,
        overflow_policy=overflow_policy,
    )
    digest = hashlib.sha256(_canonical_json(descriptor)).hexdigest()
    micro_accuracy = num_correct / total
    return {
        "num_correct": num_correct,
        "total": total,
        "num_questions": total,
        "micro_accuracy": micro_accuracy,
        "accuracy": micro_accuracy,
        "num_fewshot": num_fewshot,
        "max_length": resolved_max_length,
        "overflow_policy": overflow_policy,
        "padding_side": resolved_padding_side,
        "per_subject": per_subject,
        "predictions": predictions,
        "prompt_length_stats": {
            "min": min(prompt_lengths),
            "max": max(prompt_lengths),
            "mean": math.fsum(prompt_lengths) / total,
        },
        "fewshot_stats": {
            "requested": num_fewshot,
            "min_actual": min(actual_shots),
            "max_actual": max(actual_shots),
            "num_reduced_questions": sum(shots != num_fewshot for shots in actual_shots),
            "histogram": shot_histogram,
        },
        "protocol": descriptor,
        "protocol_hash": digest,
        "data_sha256": data.data_sha256,
    }


__all__ = [
    "CHOICE_CONTINUATIONS",
    "CHOICE_LABELS",
    "MMLU_MATH_SUBJECTS",
    "MMLUMathData",
    "MMLUMathError",
    "MMLUQuestion",
    "PromptTooLongError",
    "TokenizationBoundaryError",
    "build_mmlu_prompt",
    "evaluate_mmlu_math",
    "format_mmlu_question",
    "format_subject",
    "load_mmlu_math",
    "prepare_mmlu_math_data",
    "protocol_descriptor",
    "protocol_hash",
]
