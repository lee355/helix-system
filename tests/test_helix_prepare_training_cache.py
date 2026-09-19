import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "helix_acceptance" / "prepare_training_cache.py"
SPEC = importlib.util.spec_from_file_location("helix_prepare_training_cache", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load cache preparation tool from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

STEP1_ROOT = REPO_ROOT / "training" / "step1_supervised_finetuning"
sys.path.insert(0, str(STEP1_ROOT))
from dschat.helix.mmlu_math import MMLU_MATH_SUBJECTS, prepare_mmlu_math_data


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_mmlu_bundle(root):
    csv_root = root / "mmlu_csv"
    special_questions = {
        ("dev", "abstract_algebra"): "ＡＬＰＨＡ\u3000question",
        ("test", "abstract_algebra"): "alpha question",
        ("dev", "college_mathematics"): "beta question",
    }
    for split in ("dev", "test"):
        (csv_root / split).mkdir(parents=True, exist_ok=True)
        for subject in MMLU_MATH_SUBJECTS:
            question = special_questions.get(
                (split, subject), f"Unique {split} {subject} question"
            )
            with (
                csv_root / split / f"{subject}_{split}.csv"
            ).open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(
                    [question, "choice one", "choice two", "choice three", "choice four", "D"]
                )
    bundle = root / "mmlu_bundle"
    prepare_mmlu_math_data(csv_root, bundle, source_revision="mock-mmlu")
    return bundle


def make_source_json(root):
    source_rows = [
        {
            "source": "data/PoT/mock.json",
            "instruction": "  alpha   QUESTION\n",
            "output": "unused answer zero",
        },
        {
            "source": "data/PoT/mock.json",
            "instruction": "Alpha question!",
            "output": "near duplicate retained",
        },
        {
            "source": "data/PoT/mock.json",
            "instruction": "ＢＥＴＡ\tQuestion",
            "output": "unused answer two",
        },
        {
            "source": "data/PoT/mock.json",
            "instruction": "Keep this unrelated instruction",
            "output": "retained",
        },
    ]
    source_path = root / "MathInstruct.json"
    source_path.write_text(json.dumps(source_rows), encoding="utf-8")
    return source_path, source_rows


def valid_payload(source_path, source_rows):
    input_ids = torch.arange(12, dtype=torch.int32).reshape(4, 3)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
        "labels": input_ids.add(100),
        "metadata": {
            "dataset": "MathInstruct-PoT",
            "source": str(source_path),
            "source_sha256": sha256_file(source_path),
            "source_total_rows": len(source_rows),
            "pot_rows": len(source_rows),
            "kept_rows": 4,
            "sequence_length": 3,
            "seed": 1234,
            "model": "mock-model",
            "prompt_template": {"prompt_no_input": "mock"},
            "source_indices_in_training_order": [2, 1, 0, 3],
            "order": "mock shuffled order",
        },
    }


class PrepareTrainingCacheTest(unittest.TestCase):
    def test_exact_normalized_filter_preserves_order_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path, source_rows = make_source_json(root)
            mmlu_bundle = make_mmlu_bundle(root)
            payload = valid_payload(source_path, source_rows)
            cache_path = root / "input.pt"
            output_path = root / "output.pt"
            torch.save(payload, cache_path)

            report = MODULE.prepare_training_cache(
                cache_path, source_path, mmlu_bundle, output_path
            )

            self.assertEqual(report["rows_before"], 4)
            self.assertEqual(report["rows_after"], 2)
            self.assertEqual(report["excluded_rows"], 2)
            self.assertEqual(report["excluded_training_source_indices"], [2, 0])
            self.assertFalse(report["answer_guided_selection"])
            self.assertIn("near-duplicates", report["limitation"])
            self.assertEqual(report["output_sha256"], sha256_file(output_path))
            self.assertEqual(len(report["protocol_hash"]), 64)

            filtered = torch.load(
                output_path, map_location="cpu", weights_only=True
            )
            torch.testing.assert_close(
                filtered["input_ids"], payload["input_ids"].index_select(0, torch.tensor([1, 3]))
            )
            torch.testing.assert_close(
                filtered["attention_mask"],
                payload["attention_mask"].index_select(0, torch.tensor([1, 3])),
            )
            torch.testing.assert_close(
                filtered["labels"], payload["labels"].index_select(0, torch.tensor([1, 3]))
            )
            metadata = filtered["metadata"]
            self.assertEqual(metadata["source_indices_in_training_order"], [1, 3])
            self.assertEqual(metadata["pre_mmlu_filter_kept_rows"], 4)
            self.assertEqual(metadata["kept_rows"], 2)
            self.assertEqual(metadata["model"], "mock-model")
            audit = metadata["mmlu_exact_question_filter"]
            self.assertEqual(audit["excluded_training_source_indices"], [2, 0])
            self.assertFalse(audit["answer_guided_selection"])
            self.assertEqual(audit["protocol_hash"], report["protocol_hash"])
            source_zero = next(
                item for item in audit["exclusions"] if item["source_index"] == 0
            )
            self.assertEqual(
                source_zero["matched_question_ids"],
                ["abstract_algebra:dev:0", "abstract_algebra:test:0"],
            )
            self.assertNotIn(1, audit["excluded_training_source_indices"])

    def test_rejects_tensor_shape_index_count_and_index_range_mismatches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path, source_rows = make_source_json(root)
            mmlu_bundle = make_mmlu_bundle(root)

            cases = []
            shape_payload = valid_payload(source_path, source_rows)
            shape_payload["labels"] = shape_payload["labels"][:, :2]
            cases.append(("shape", shape_payload, "identical shapes"))

            count_payload = valid_payload(source_path, source_rows)
            count_payload["metadata"]["source_indices_in_training_order"] = [2, 1, 0]
            cases.append(("count", count_payload, "length does not match"))

            range_payload = valid_payload(source_path, source_rows)
            range_payload["metadata"]["source_indices_in_training_order"][-1] = 4
            cases.append(("range", range_payload, "outside \[0, 4\)"))

            for name, payload, error in cases:
                with self.subTest(name=name):
                    cache_path = root / f"{name}.pt"
                    torch.save(payload, cache_path)
                    with self.assertRaisesRegex(ValueError, error):
                        MODULE.prepare_training_cache(
                            cache_path,
                            source_path,
                            mmlu_bundle,
                            root / f"{name}-output.pt",
                        )

    def test_normalization_is_nfkc_casefold_and_whitespace_only(self):
        self.assertEqual(
            MODULE.normalize_question("  ＡlPhA\u3000\n Question  "),
            "alpha question",
        )
        self.assertNotEqual(
            MODULE.normalize_question("Alpha question!"),
            MODULE.normalize_question("Alpha question"),
        )


if __name__ == "__main__":
    unittest.main()
