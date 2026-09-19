"""Regression checks for SFT with the shared Llama EOS/padding token."""

import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from dschat.utils.data.math_data_utils import (
    DataCollatorForSupervisedDataset,
    IGNORE_INDEX,
)


class HelixSFTLabelsTest(unittest.TestCase):
    def setUp(self):
        tokenizer = Tokenizer(
            WordLevel({"[UNK]": 0, "prompt": 1, "answer": 2, "<eos>": 3}, unk_token="[UNK]")
        )
        tokenizer.pre_tokenizer = WhitespaceSplit()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            eos_token="<eos>",
            pad_token="<eos>",
            model_max_length=6,
            padding_side="right",
        )
        self.collator = DataCollatorForSupervisedDataset(self.tokenizer, False)

    def test_real_eos_is_supervised_but_padding_is_ignored(self):
        batch = self.collator([{"input_ids": "prompt ", "labels": "answer<eos>"}])
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2, 3, 3, 3, 3]])
        self.assertEqual(batch["labels"].tolist(), [[IGNORE_INDEX, 2, 3, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]])
        self.assertEqual(batch["attention_mask"].tolist(), [[True, True, True, False, False, False]])

        # Arbitrarily bad predictions on padding must not affect the objective.
        logits = torch.zeros(1, 6, len(self.tokenizer))
        logits[0, 0, 2] = 3.0
        logits[0, 1, 3] = 2.0
        logits[0, 2:, 0] = 50.0
        actual = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, len(self.tokenizer)), batch["labels"][:, 1:].reshape(-1)
        )
        expected = torch.nn.functional.cross_entropy(logits[0, :2], torch.tensor([2, 3]))
        torch.testing.assert_close(actual, expected)

    def test_real_eos_inside_prompt_counts_toward_prompt_mask(self):
        batch = self.collator([{"input_ids": "prompt<eos> ", "labels": "answer<eos>"}])
        self.assertEqual(batch["labels"].tolist(), [[IGNORE_INDEX, IGNORE_INDEX, 2, 3, IGNORE_INDEX, IGNORE_INDEX]])
        self.assertEqual(batch["attention_mask"].tolist(), [[True, True, True, True, False, False]])

    def test_batch_without_shifted_targets_reports_truncated_prompt(self):
        with self.assertRaisesRegex(ValueError, "no supervised next-token targets"):
            self.collator([{"input_ids": "prompt " * 6, "labels": "answer<eos>"}])

    def test_pipeline_collator_preserves_attention_and_labels(self):
        pipeline_collator = DataCollatorForSupervisedDataset(self.tokenizer, True)
        (input_ids, attention_mask), labels = pipeline_collator(
            [{"input_ids": "prompt ", "labels": "answer<eos>"}]
        )
        self.assertEqual(input_ids.shape, labels.shape)
        self.assertTrue(attention_mask[0, 2].item())
        self.assertFalse(attention_mask[0, 3].item())
        self.assertEqual(labels[0, 3].item(), IGNORE_INDEX)


if __name__ == "__main__":
    unittest.main()
