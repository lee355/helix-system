"""Validated, deterministic token cache shared by Helix acceptance runs."""
from pathlib import Path

import torch
from torch.utils.data import Dataset


class MathTokenCache(Dataset):
    def __init__(self, path, sequence_length):
        self.path = str(Path(path).resolve())
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.tensors = {key: payload[key] for key in ("input_ids", "attention_mask", "labels")}
        integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
        for key, value in self.tensors.items():
            allowed = integer_dtypes | ({torch.bool} if key == "attention_mask" else set())
            if not isinstance(value, torch.Tensor) or value.dtype not in allowed:
                raise ValueError(f"Token cache {key} must be an integer tensor")
        raw_mask = self.tensors["attention_mask"]
        if torch.any((raw_mask != 0) & (raw_mask != 1)):
            raise ValueError("Token cache attention_mask must contain only 0/1")
        if torch.any(self.tensors["input_ids"] < 0):
            raise ValueError("Token cache input_ids must be nonnegative")
        raw_labels = self.tensors["labels"]
        if torch.any((raw_labels < 0) & (raw_labels != -100)):
            raise ValueError("Token cache labels must be token ids or -100")
        shapes = {tuple(value.shape) for value in self.tensors.values()}
        if len(shapes) != 1:
            raise ValueError("Token cache input/mask/label shapes differ")
        shape = next(iter(shapes))
        if len(shape) != 2 or not shape[0] or shape[1] != sequence_length:
            raise ValueError("Token cache must be nonempty [N, max_seq_len]")
        labels = self.tensors["labels"]
        mask = self.tensors["attention_mask"].bool()
        if torch.any(labels[~mask] != -100):
            raise ValueError("Token cache supervises padding positions")
        if torch.any(labels[:, 1:].ne(-100).sum(dim=1) == 0):
            raise ValueError("Token cache contains rows without causal targets")
        self.metadata = payload.get("metadata", {})

    def __len__(self):
        return self.tensors["input_ids"].shape[0]

    def __getitem__(self, index):
        return {key: value[index].long() for key, value in self.tensors.items()}
