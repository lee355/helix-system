import importlib.util
import random
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "training"
    / "step1_supervised_finetuning"
    / "dschat"
    / "helix"
    / "data.py"
)
SPEC = importlib.util.spec_from_file_location("helix_data", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load sampler module from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
HeterogeneousDistributedBatchSampler = (
    MODULE.HeterogeneousDistributedBatchSampler
)


def make_samplers(dataset_size, batch_sizes, seed=0, drop_last=False):
    world_size = len(batch_sizes)
    return [
        HeterogeneousDistributedBatchSampler(
            dataset_size=dataset_size,
            batch_sizes=batch_sizes,
            rank=rank,
            world_size=world_size,
            seed=seed,
            drop_last=drop_last,
        )
        for rank in range(world_size)
    ]


def concatenate_global_steps(rank_batches):
    return [
        [index for batches in step_batches for index in batches]
        for step_batches in zip(*rank_batches)
    ]


class HeterogeneousDistributedBatchSamplerTest(unittest.TestCase):
    def test_drop_last_uses_shared_permutation_and_equal_step_counts(self):
        dataset_size = 23
        batch_sizes = [2, 3, 1]
        seed = 17
        samplers = make_samplers(
            dataset_size, batch_sizes, seed=seed, drop_last=True
        )
        rank_batches = [list(sampler) for sampler in samplers]

        self.assertEqual([len(batches) for batches in rank_batches], [3, 3, 3])
        for rank, batches in enumerate(rank_batches):
            self.assertTrue(
                all(len(batch) == batch_sizes[rank] for batch in batches)
            )

        permutation = list(range(dataset_size))
        random.Random(seed).shuffle(permutation)
        global_steps = concatenate_global_steps(rank_batches)
        self.assertEqual(
            [index for step in global_steps for index in step],
            permutation[:18],
        )

    def test_padding_cycles_from_same_epoch_permutation(self):
        dataset_size = 23
        batch_sizes = [2, 3, 1]
        seed = 29
        samplers = make_samplers(
            dataset_size, batch_sizes, seed=seed, drop_last=False
        )
        rank_batches = [list(sampler) for sampler in samplers]

        self.assertEqual([len(batches) for batches in rank_batches], [4, 4, 4])
        permutation = list(range(dataset_size))
        random.Random(seed).shuffle(permutation)
        expected = permutation + permutation[:1]
        global_steps = concatenate_global_steps(rank_batches)
        self.assertEqual(
            [index for step in global_steps for index in step], expected
        )

    def test_padding_can_repeat_dataset_more_than_once(self):
        dataset_size = 3
        batch_sizes = [4, 3]
        seed = 5
        samplers = make_samplers(
            dataset_size, batch_sizes, seed=seed, drop_last=False
        )
        rank_batches = [list(sampler) for sampler in samplers]

        permutation = list(range(dataset_size))
        random.Random(seed).shuffle(permutation)
        expected = (permutation * 3)[:7]
        self.assertEqual(concatenate_global_steps(rank_batches), [expected])

    def test_set_epoch_is_deterministic_and_changes_shared_order(self):
        samplers = make_samplers(30, [1, 2, 3], seed=11)
        for sampler in samplers:
            sampler.set_epoch(4)
        epoch_four = [list(sampler) for sampler in samplers]

        for sampler in samplers:
            sampler.set_epoch(5)
        epoch_five = [list(sampler) for sampler in samplers]
        self.assertNotEqual(epoch_four, epoch_five)

        recreated = make_samplers(30, [1, 2, 3], seed=11)
        for sampler in recreated:
            sampler.set_epoch(4)
        self.assertEqual(epoch_four, [list(sampler) for sampler in recreated])

    def test_empty_dataset_has_no_batches(self):
        for drop_last in (False, True):
            samplers = make_samplers(0, [1, 2], drop_last=drop_last)
            self.assertEqual([len(sampler) for sampler in samplers], [0, 0])
            self.assertEqual([list(sampler) for sampler in samplers], [[], []])

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            HeterogeneousDistributedBatchSampler(-1, [1], 0, 1)
        with self.assertRaises(ValueError):
            HeterogeneousDistributedBatchSampler(10, [1], 1, 1)
        with self.assertRaises(ValueError):
            HeterogeneousDistributedBatchSampler(10, [1, 2], 0, 1)
        with self.assertRaises(ValueError):
            HeterogeneousDistributedBatchSampler(10, [0], 0, 1)
        with self.assertRaises(TypeError):
            HeterogeneousDistributedBatchSampler(10, [1.5], 0, 1)
        with self.assertRaises(TypeError):
            HeterogeneousDistributedBatchSampler(10, [1], 0, 1, drop_last=1)


if __name__ == "__main__":
    unittest.main()
