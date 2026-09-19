"""Data-loading primitives for heterogeneous Helix training.

The batch sampler below gives every rank the same number of iterations while
allowing each rank to use a different local microbatch size.  For a given
epoch, every rank independently constructs the same global permutation and
takes its rank-specific slice from each global batch.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence


class HeterogeneousDistributedBatchSampler:
    """Yield rank-local batches from a shared, heterogeneous global batch.

    Args:
        dataset_size: Number of examples in the dataset.
        batch_sizes: Local microbatch size for every rank.  Its length must
            equal ``world_size`` and every value must be positive.
        rank: Rank whose local batches this sampler should yield.
        world_size: Number of participating ranks.
        seed: Base seed used to create the shared epoch permutation.
        drop_last: If true, discard the final incomplete global batch.  If
            false, pad it by cycling from the beginning of the same epoch's
            permutation.

    This object implements the ``batch_sampler`` protocol expected by a
    PyTorch ``DataLoader`` without importing PyTorch itself.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_sizes: Sequence[int],
        rank: int,
        world_size: int,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset_size = self._validate_int(
            "dataset_size", dataset_size, minimum=0
        )
        self.world_size = self._validate_int(
            "world_size", world_size, minimum=1
        )
        self.rank = self._validate_int("rank", rank, minimum=0)
        if self.rank >= self.world_size:
            raise ValueError(
                f"rank must be smaller than world_size; got "
                f"rank={self.rank}, world_size={self.world_size}"
            )

        if isinstance(batch_sizes, (str, bytes)) or not isinstance(
            batch_sizes, Sequence
        ):
            raise TypeError("batch_sizes must be a sequence of positive integers")
        if len(batch_sizes) != self.world_size:
            raise ValueError(
                "batch_sizes length must equal world_size; got "
                f"{len(batch_sizes)} and {self.world_size}"
            )
        self.batch_sizes = tuple(
            self._validate_int(
                f"batch_sizes[{index}]", batch_size, minimum=1
            )
            for index, batch_size in enumerate(batch_sizes)
        )

        self.seed = self._validate_int("seed", seed)
        if not isinstance(drop_last, bool):
            raise TypeError("drop_last must be a bool")
        self.drop_last = drop_last
        self.epoch = 0

        offsets = [0]
        for batch_size in self.batch_sizes:
            offsets.append(offsets[-1] + batch_size)
        self._rank_offsets = tuple(offsets)
        self.global_batch_size = offsets[-1]

    @staticmethod
    def _validate_int(name: str, value: int, minimum: int | None = None) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if minimum is not None and value < minimum:
            raise ValueError(f"{name} must be at least {minimum}; got {value}")
        return value

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch used to seed the shared global permutation."""

        self.epoch = self._validate_int("epoch", epoch, minimum=0)

    def __len__(self) -> int:
        if self.dataset_size == 0:
            return 0
        if self.drop_last:
            return self.dataset_size // self.global_batch_size
        return math.ceil(self.dataset_size / self.global_batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        num_steps = len(self)
        if num_steps == 0:
            return

        indices = list(range(self.dataset_size))
        random.Random(self.seed + self.epoch).shuffle(indices)

        total_size = num_steps * self.global_batch_size
        if total_size > len(indices):
            repeats, remainder = divmod(total_size, len(indices))
            indices = indices * repeats + indices[:remainder]
        else:
            indices = indices[:total_size]

        local_start = self._rank_offsets[self.rank]
        local_end = self._rank_offsets[self.rank + 1]
        for step in range(num_steps):
            global_start = step * self.global_batch_size
            yield indices[
                global_start + local_start : global_start + local_end
            ]
