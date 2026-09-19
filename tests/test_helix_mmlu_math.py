import csv
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "training"
    / "step1_supervised_finetuning"
    / "dschat"
    / "helix"
    / "mmlu_math.py"
)
SPEC = importlib.util.spec_from_file_location("helix_mmlu_math", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load MMLU evaluator from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


MARKERS = ("α", "β", "γ", "δ", "ε")
GOLD_LABELS = ("A", "B", "C", "D", "A")
PREDICTED_LABELS = ("A", "B", "C", "D", "B")


def make_unit_data():
    dev = []
    test = []
    for subject, marker, gold in zip(
        MODULE.MMLU_MATH_SUBJECTS, MARKERS, GOLD_LABELS
    ):
        dev.append(
            MODULE.MMLUQuestion(
                subject=subject,
                split="dev",
                index=0,
                question=f"Demonstration {marker}",
                choices=("one", "two", "three", "four"),
                answer=0,
            )
        )
        test.append(
            MODULE.MMLUQuestion(
                subject=subject,
                split="test",
                index=0,
                question=f"Evaluation marker {marker}",
                choices=("red", "green", "blue", "black"),
                answer=MODULE.CHOICE_LABELS.index(gold),
            )
        )
    return MODULE.MMLUMathData(
        dev=tuple(dev),
        test=tuple(test),
        data_sha256="unit-test-data",
        source={"revision": "unit-test"},
    )


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    model_max_length = 100000

    def __init__(self, padding_side="right"):
        self.padding_side = padding_side

    @staticmethod
    def token_id(character):
        return ord(character) + 1

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("MMLU protocol must disable special tokens")
        return [self.token_id(character) for character in text]


class SingleAnswerTokenTokenizer(CharacterTokenizer):
    ANSWER_IDS = {label: 1400 + index for index, label in enumerate("ABCD")}

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("MMLU protocol must disable special tokens")
        for label, token_id in self.ANSWER_IDS.items():
            suffix = f" {label}"
            if text.endswith(suffix):
                return super().encode(text[: -len(suffix)]) + [token_id]
        return super().encode(text)


class MarkerCausalLM(torch.nn.Module):
    def __init__(self, tokenizer, answer_token_ids):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.tokenizer = tokenizer
        self.answer_token_ids = answer_token_ids
        self.marker_to_answer = {
            tokenizer.token_id(marker): answer_token_ids[label]
            for marker, label in zip(MARKERS, PREDICTED_LABELS)
        }
        self.calls = []

    def forward(self, input_ids, attention_mask, num_logits_to_keep=0):
        self.calls.append(
            {
                "input_ids": input_ids.detach().cpu().clone(),
                "attention_mask": attention_mask.detach().cpu().clone(),
                "num_logits_to_keep": num_logits_to_keep,
            }
        )
        batch, sequence = input_ids.shape
        logits = torch.zeros(batch, sequence, 2048, device=input_ids.device)
        space_id = self.tokenizer.token_id(" ")
        logits[:, :, space_id] = 2.0
        for row in range(batch):
            desired = None
            row_ids = set(input_ids[row].detach().cpu().tolist())
            for marker_id, answer_id in self.marker_to_answer.items():
                if marker_id in row_ids:
                    desired = answer_id
            if desired is None:
                raise AssertionError("No unit-test marker found in model input")
            logits[row, :, desired] = 8.0
        if num_logits_to_keep:
            logits = logits[:, -num_logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


def assert_padding_masks(test_case, calls, padding_side):
    saw_padding = False
    for call in calls:
        for mask in call["attention_mask"].tolist():
            if 0 not in mask:
                continue
            saw_padding = True
            first_one = mask.index(1)
            last_one = len(mask) - 1 - list(reversed(mask)).index(1)
            if padding_side == "left":
                test_case.assertTrue(all(value == 0 for value in mask[:first_one]))
                test_case.assertTrue(all(value == 1 for value in mask[first_one:]))
            else:
                test_case.assertTrue(all(value == 1 for value in mask[: last_one + 1]))
                test_case.assertTrue(all(value == 0 for value in mask[last_one + 1 :]))
    test_case.assertTrue(saw_padding)


class MMLUMathScoringTest(unittest.TestCase):
    def test_multitoken_scoring_padding_masks_and_micro_aggregation(self):
        data = make_unit_data()
        for padding_side in ("left", "right"):
            with self.subTest(padding_side=padding_side):
                tokenizer = CharacterTokenizer(padding_side=padding_side)
                answer_ids = {
                    label: tokenizer.token_id(label) for label in MODULE.CHOICE_LABELS
                }
                model = MarkerCausalLM(tokenizer, answer_ids)
                model.train()
                result = MODULE.evaluate_mmlu_math(
                    model,
                    tokenizer,
                    data,
                    device="cpu",
                    batch_size=5,
                    max_length=10000,
                    num_fewshot=1,
                    padding_side=padding_side,
                )

                self.assertEqual(result["num_correct"], 4)
                self.assertEqual(result["num_questions"], 5)
                self.assertEqual(result["total"], 5)
                self.assertTrue(math.isclose(result["accuracy"], 0.8))
                self.assertEqual(result["micro_accuracy"], result["accuracy"])
                self.assertEqual(
                    [item["prediction"] for item in result["predictions"]],
                    list(PREDICTED_LABELS),
                )
                self.assertTrue(model.training, "evaluator must restore training mode")
                self.assertTrue(
                    all(
                        lengths == {"A": 2, "B": 2, "C": 2, "D": 2}
                        for lengths in (
                            item["candidate_token_lengths"]
                            for item in result["predictions"]
                        )
                    )
                )
                assert_padding_masks(self, model.calls, padding_side)
                if padding_side == "left":
                    self.assertTrue(
                        all(call["num_logits_to_keep"] == 2 for call in model.calls)
                    )
                else:
                    self.assertTrue(
                        all(call["num_logits_to_keep"] >= 2 for call in model.calls)
                    )

    def test_single_token_choices_share_one_prefix_forward(self):
        data = make_unit_data()
        tokenizer = SingleAnswerTokenTokenizer(padding_side="left")
        model = MarkerCausalLM(tokenizer, tokenizer.ANSWER_IDS)
        result = MODULE.evaluate_mmlu_math(
            model,
            tokenizer,
            data,
            device="cpu",
            batch_size=2,
            max_length=10000,
            num_fewshot=1,
        )

        self.assertEqual(result["num_correct"], 4)
        self.assertEqual(len(model.calls), 3, "five prompts at batch_size=2 need 3 calls")
        self.assertTrue(all(call["num_logits_to_keep"] == 1 for call in model.calls))
        self.assertTrue(
            all(
                lengths == {"A": 1, "B": 1, "C": 1, "D": 1}
                for lengths in (
                    item["candidate_token_lengths"] for item in result["predictions"]
                )
            )
        )

    def test_overflow_is_error_or_explicit_recorded_shot_reduction(self):
        data = make_unit_data()
        tokenizer = SingleAnswerTokenTokenizer(padding_side="left")
        zero_shot_lengths = []
        one_shot_lengths = []
        dev_by_subject = {question.subject: [question] for question in data.dev}
        for question in data.test:
            zero = MODULE.build_mmlu_prompt(
                question.subject, dev_by_subject[question.subject], question, num_fewshot=0
            )
            one = MODULE.build_mmlu_prompt(
                question.subject, dev_by_subject[question.subject], question, num_fewshot=1
            )
            zero_shot_lengths.append(len(tokenizer.encode(zero)) + 1)
            one_shot_lengths.append(len(tokenizer.encode(one)) + 1)
        max_length = max(zero_shot_lengths)
        self.assertLess(max_length, min(one_shot_lengths))

        model = MarkerCausalLM(tokenizer, tokenizer.ANSWER_IDS)
        with self.assertRaises(MODULE.PromptTooLongError):
            MODULE.evaluate_mmlu_math(
                model,
                tokenizer,
                data,
                device="cpu",
                max_length=max_length,
                num_fewshot=1,
                overflow_policy="error",
            )

        result = MODULE.evaluate_mmlu_math(
            model,
            tokenizer,
            data,
            device="cpu",
            batch_size=5,
            max_length=max_length,
            num_fewshot=1,
            overflow_policy="reduce_fewshot",
        )
        self.assertEqual(result["fewshot_stats"]["num_reduced_questions"], 5)
        self.assertEqual(result["fewshot_stats"]["histogram"], {"0": 5})
        self.assertTrue(
            all(item["actual_num_fewshot"] == 0 for item in result["predictions"])
        )


class MMLUMathProtocolAndDataTest(unittest.TestCase):
    def test_protocol_hash_is_deterministic_and_descriptor_is_fresh(self):
        arguments = {
            "data_sha256": "abc",
            "num_fewshot": 5,
            "max_length": 4096,
            "overflow_policy": "error",
        }
        expected = MODULE.protocol_hash(**arguments)
        descriptor = MODULE.protocol_descriptor(**arguments)
        descriptor["subjects"].append("mutation")
        descriptor["prompt"]["header"] = "mutation"
        self.assertEqual(MODULE.protocol_hash(**arguments), expected)
        self.assertNotIn(
            "mutation", MODULE.protocol_descriptor(**arguments)["subjects"]
        )
        self.assertNotEqual(
            expected,
            MODULE.protocol_hash(
                data_sha256="abc",
                num_fewshot=0,
                max_length=4096,
                overflow_policy="error",
            ),
        )

    def test_prepare_manifest_load_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            for split in ("dev", "test"):
                (source / split).mkdir(parents=True, exist_ok=True)
                for subject in MODULE.MMLU_MATH_SUBJECTS:
                    csv_path = source / split / f"{subject}_{split}.csv"
                    with csv_path.open("w", encoding="utf-8", newline="") as handle:
                        csv.writer(handle).writerow(
                            [f"Question {subject}", "a", "b", "c", "d", "A"]
                        )

            output = root / "prepared"
            manifest = MODULE.prepare_mmlu_math_data(
                source,
                output,
                source_revision="unit-revision",
            )
            self.assertEqual(manifest["purpose"], "evaluation_only")
            self.assertEqual(manifest["source"]["revision"], "unit-revision")
            self.assertEqual(manifest["splits"]["dev"]["count"], 5)
            self.assertEqual(manifest["splits"]["test"]["count"], 5)
            self.assertEqual(
                manifest["training_separation"]["included_source_splits"],
                ["dev", "test"],
            )
            self.assertIn(
                "auxiliary_train",
                manifest["training_separation"]["excluded_source_splits"],
            )

            loaded = MODULE.load_mmlu_math(output)
            self.assertEqual(len(loaded.dev), 5)
            self.assertEqual(len(loaded.test), 5)
            self.assertEqual(loaded.data_sha256, manifest["dataset_sha256"])

            with (output / "test.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                MODULE.load_mmlu_math(output)


if __name__ == "__main__":
    unittest.main()
