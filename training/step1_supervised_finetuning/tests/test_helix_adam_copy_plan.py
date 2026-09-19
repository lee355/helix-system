import unittest
from collections import OrderedDict

import torch

from dschat.helix.adam_copy_plan import (
    build_complete_state_copy_plan,
    execute_complete_state_copy_plan,
)
from dschat.helix.adam_migration import mask_linear_indices, mask_local_shape


def _slice(tensor, mask):
    return tensor.reshape(-1).index_select(
        0, mask_linear_indices(mask, tensor.shape)
    ).reshape(mask_local_shape(mask, tensor.shape))


class HelixAdamCompleteCopyPlanTest(unittest.TestCase):
    def test_retained_coordinates_are_reordered_even_without_a_rank_transfer(self):
        reference = OrderedDict(weight=torch.zeros(3, 4))
        full_slot = OrderedDict(weight=torch.arange(12, dtype=torch.float32).reshape(3, 4))
        columns = torch.arange(4)
        old_masks = [
            OrderedDict(weight=(torch.tensor([2, 0]), columns)),
            OrderedDict(weight=(torch.tensor([1]), columns)),
        ]
        new_masks = [
            OrderedDict(weight=(torch.tensor([0, 2]), columns)),
            OrderedDict(weight=(torch.tensor([1]), columns)),
        ]
        old_local = [
            OrderedDict(weight=_slice(full_slot["weight"], mask["weight"]))
            for mask in old_masks
        ]

        plan = build_complete_state_copy_plan(reference, old_masks, new_masks)
        migrated = execute_complete_state_copy_plan(
            reference, new_masks, old_local, plan
        )

        torch.testing.assert_close(
            migrated[0]["weight"],
            torch.tensor([[0.0, 1.0, 2.0, 3.0], [8.0, 9.0, 10.0, 11.0]]),
            rtol=0,
            atol=0,
        )
        self.assertTrue(
            all(
                copy_record.source_rank == copy_record.destination_rank
                for copy_record in plan
            )
        )
        rank0_copy = next(
            copy_record
            for copy_record in plan
            if copy_record.source_rank == 0 and copy_record.destination_rank == 0
        )
        self.assertEqual(rank0_copy.source_local_linear_indices.tolist(), list(range(4, 8)) + list(range(4)))
        self.assertEqual(rank0_copy.destination_local_linear_indices.tolist(), list(range(8)))

    def test_complete_plan_covers_local_and_remote_coordinates_exactly_once(self):
        reference = OrderedDict(vector=torch.zeros(6))
        values = OrderedDict(vector=torch.arange(6, dtype=torch.float32) + 0.5)
        old_masks = [
            OrderedDict(vector=torch.tensor([0, 1, 2])),
            OrderedDict(vector=torch.tensor([2, 3, 4, 5])),
        ]
        new_masks = [
            OrderedDict(vector=torch.tensor([1, 2, 3, 4])),
            OrderedDict(vector=torch.tensor([5, 0])),
        ]
        old_local = [
            OrderedDict(vector=_slice(values["vector"], mask["vector"]))
            for mask in old_masks
        ]

        plan = build_complete_state_copy_plan(reference, old_masks, new_masks)
        migrated = execute_complete_state_copy_plan(
            reference, new_masks, old_local, plan
        )
        for rank, mask in enumerate(new_masks):
            torch.testing.assert_close(
                migrated[rank]["vector"], _slice(values["vector"], mask["vector"]), rtol=0, atol=0
            )
            destination_positions = torch.cat(
                [
                    record.destination_local_linear_indices
                    for record in plan
                    if record.destination_rank == rank and record.name == "vector"
                ]
            )
            self.assertEqual(
                sorted(destination_positions.tolist()),
                list(range(migrated[rank]["vector"].numel())),
            )
        self.assertTrue(any(record.source_rank != record.destination_rank for record in plan))


if __name__ == "__main__":
    unittest.main()
