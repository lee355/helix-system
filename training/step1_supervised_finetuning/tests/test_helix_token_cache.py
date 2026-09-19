from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

import torch

from dschat.helix.token_cache import MathTokenCache


def _payload():
    return {
        "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        "labels": torch.tensor([[-100, 2, 3, -100], [-100, 5, -100, -100]]),
        "metadata": {"tokenizer_sha256": "fixed"},
    }


class MathTokenCacheTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.path = Path(self.tempdir.name) / "tokens.pt"

    def _save(self, payload):
        torch.save(payload, self.path)

    def test_valid_cache_is_deterministic_and_returns_long_rows(self):
        self._save(_payload())
        cache = MathTokenCache(self.path, sequence_length=4)
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.path, str(self.path.resolve()))
        self.assertEqual(cache.metadata, {"tokenizer_sha256": "fixed"})
        first = cache[0]
        self.assertEqual(set(first), {"input_ids", "attention_mask", "labels"})
        self.assertTrue(all(value.dtype == torch.long for value in first.values()))
        self.assertEqual(first["input_ids"].tolist(), [1, 2, 3, 0])

        second = cache[0]
        self.assertEqual(second["input_ids"].tolist(), first["input_ids"].tolist())

    def test_shape_padding_and_shifted_target_validation(self):
        payload = _payload()
        payload["labels"] = payload["labels"][:, :3]
        self._save(payload)
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            MathTokenCache(self.path, sequence_length=4)

        payload = _payload()
        payload["labels"][0, 3] = 7
        self._save(payload)
        with self.assertRaisesRegex(ValueError, "padding positions"):
            MathTokenCache(self.path, sequence_length=4)

        payload = _payload()
        payload["labels"][1] = torch.tensor([8, -100, -100, -100])
        self._save(payload)
        with self.assertRaisesRegex(ValueError, "without causal targets"):
            MathTokenCache(self.path, sequence_length=4)

        self._save(_payload())
        with self.assertRaisesRegex(ValueError, "nonempty"):
            MathTokenCache(self.path, sequence_length=3)

    def test_nonbinary_attention_masks_are_rejected(self):
        payload = _payload()
        payload["attention_mask"][0, 1] = 2
        self._save(payload)
        with self.assertRaisesRegex(ValueError, "attention[_ ]mask|0/1"):
            MathTokenCache(self.path, sequence_length=4)

    def test_negative_token_ids_and_invalid_negative_labels_are_rejected(self):
        payload = _payload()
        payload["input_ids"][0, 0] = -1
        self._save(payload)
        with self.assertRaisesRegex(ValueError, "input_ids.*nonnegative"):
            MathTokenCache(self.path, sequence_length=4)

        payload = _payload()
        payload["labels"][0, 1] = -1
        self._save(payload)
        with self.assertRaisesRegex(ValueError, "labels.*-100"):
            MathTokenCache(self.path, sequence_length=4)

    def test_fractional_token_ids_are_rejected_instead_of_truncated(self):
        payload = _payload()
        payload["input_ids"] = payload["input_ids"].float()
        payload["input_ids"][0, 0] = 1.5
        self._save(payload)
        with self.assertRaisesRegex(ValueError, "integer|input_ids"):
            MathTokenCache(self.path, sequence_length=4)

    def test_helix_dataset_builder_routes_to_the_token_cache(self):
        import helix_train_base

        self._save(_payload())
        args = types.SimpleNamespace(
            helix_token_cache_path=str(self.path),
            max_seq_len=4,
            trainset="math",
            data_path="/must/not/be/read",
        )
        dataset, collator = helix_train_base._build_dataset(args, tokenizer=object())
        self.assertIsInstance(dataset, MathTokenCache)
        batch = collator([dataset[0], dataset[1]])
        self.assertEqual(batch["input_ids"].shape, (2, 4))
        self.assertEqual(batch["labels"].shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
