import copy
import types
import unittest
from collections import OrderedDict

import torch

from dschat.helix.adam_migration import (
    NamedAdamParameterState,
    assert_complete_mask_coverage,
    mask_linear_indices,
    mask_local_shape,
    migrate_named_adam_states,
)
from dschat.helix.masking import build_structured_masks


def _toy_state():
    prefix = "model.layers.0"
    return OrderedDict(
        {
            f"{prefix}.self_attn.q_proj.weight": torch.zeros(4, 4),
            f"{prefix}.self_attn.q_proj.bias": torch.zeros(4),
            f"{prefix}.self_attn.k_proj.weight": torch.zeros(2, 4),
            f"{prefix}.self_attn.k_proj.bias": torch.zeros(2),
            f"{prefix}.self_attn.v_proj.weight": torch.zeros(2, 4),
            f"{prefix}.self_attn.v_proj.bias": torch.zeros(2),
            f"{prefix}.self_attn.o_proj.weight": torch.zeros(4, 4),
            f"{prefix}.mlp.gate_proj.weight": torch.zeros(4, 4),
            f"{prefix}.mlp.up_proj.weight": torch.zeros(4, 4),
            f"{prefix}.mlp.down_proj.weight": torch.zeros(4, 4),
            f"{prefix}.input_layernorm.weight": torch.zeros(4),
        }
    )


def _config():
    return types.SimpleNamespace(
        hidden_size=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=4,
    )


def _selected_reference(state):
    suffixes = (
        "self_attn.q_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.down_proj.weight",
    )
    return OrderedDict((name, tensor) for name, tensor in state.items() if name.endswith(suffixes))


def _select_masks(masks, names):
    return [OrderedDict((name, rank_mask[name]) for name in names) for rank_mask in masks]


def _slice(tensor, mask):
    indices = mask_linear_indices(mask, tensor.shape)
    return tensor.reshape(-1).index_select(0, indices).reshape(
        mask_local_shape(mask, tensor.shape)
    )


def _global_adam_states(reference):
    states = OrderedDict()
    for parameter_index, (name, tensor) in enumerate(reference.items()):
        base = (
            torch.arange(tensor.numel(), dtype=torch.float32).reshape(tensor.shape)
            + parameter_index * 1000
        )
        states[name] = NamedAdamParameterState(
            parameter=base + 0.125,
            master_parameter=base + 100.25,
            exp_avg=base + 200.5,
            exp_avg_sq=base + 300.75,
            step=torch.tensor(37.0),
        )
    return states


def _localize(global_states, masks):
    local_states = []
    for rank_mask in masks:
        rank_states = OrderedDict()
        for name, state in global_states.items():
            rank_states[name] = NamedAdamParameterState(
                parameter=_slice(state.parameter, rank_mask[name]),
                master_parameter=_slice(state.master_parameter, rank_mask[name]),
                exp_avg=_slice(state.exp_avg, rank_mask[name]),
                exp_avg_sq=_slice(state.exp_avg_sq, rank_mask[name]),
                step=state.step.clone(),
            )
        local_states.append(rank_states)
    return local_states


class HelixAdamMigrationTest(unittest.TestCase):
    def setUp(self):
        self.model_state = _toy_state()
        self.reference = _selected_reference(self.model_state)
        names = tuple(self.reference)

        # Increasing rank 0 from .5 to 1.0 changes not only its shape but also
        # the cumulative circular offsets of every later rank.
        old_all, _ = build_structured_masks(
            self.model_state, [0.5, 0.5, 0.5, 0.5], config=_config()
        )
        new_all, _ = build_structured_masks(
            self.model_state, [1.0, 0.5, 0.5, 0.5], config=_config()
        )
        self.old_masks = _select_masks(old_all, names)
        self.new_masks = _select_masks(new_all, names)
        self.global_states = _global_adam_states(self.reference)
        self.old_states = _localize(self.global_states, self.old_masks)

    def test_global_local_coordinate_mapping_preserves_mask_order(self):
        full = torch.arange(12).reshape(3, 4)
        mask = (torch.tensor([2, 0]), torch.tensor([3, 1]))

        indices = mask_linear_indices(mask, full.shape)
        self.assertEqual(indices.tolist(), [11, 9, 3, 1])
        self.assertEqual(tuple(mask_local_shape(mask, full.shape)), (2, 2))
        torch.testing.assert_close(
            full.reshape(-1).index_select(0, indices).reshape(2, 2),
            torch.tensor([[11, 9], [3, 1]]),
        )

    def test_structural_resize_losslessly_migrates_parameter_and_full_adam_state(self):
        result = migrate_named_adam_states(
            self.reference,
            self.old_masks,
            self.new_masks,
            self.old_states,
        )

        fields = ("parameter", "master_parameter", "exp_avg", "exp_avg_sq")
        for name, expected_global in self.global_states.items():
            actual_global = result.global_states[name]
            for field in fields:
                torch.testing.assert_close(
                    getattr(actual_global, field), getattr(expected_global, field), rtol=0, atol=0
                )
            torch.testing.assert_close(actual_global.step, expected_global.step, rtol=0, atol=0)

            for rank, new_mask in enumerate(self.new_masks):
                migrated = result.local_states[rank][name]
                for field in fields:
                    expected_local = _slice(getattr(expected_global, field), new_mask[name])
                    torch.testing.assert_close(
                        getattr(migrated, field), expected_local, rtol=0, atol=0
                    )
                torch.testing.assert_close(migrated.step, expected_global.step, rtol=0, atol=0)

        name = "model.layers.0.mlp.gate_proj.weight"
        old_rank0 = set(mask_linear_indices(self.old_masks[0][name], self.reference[name].shape).tolist())
        new_rank0_indices = mask_linear_indices(
            self.new_masks[0][name], self.reference[name].shape
        )
        added_to_rank0 = {index for index in new_rank0_indices.tolist() if index not in old_rank0}
        self.assertTrue(added_to_rank0)

        # Rows 2/3 are newly acquired by rank 0.  Their deterministic old
        # owner is rank 1, and every tensor slot follows the same transfer map.
        rank0_transfers = [
            transfer
            for transfer in result.transfers
            if transfer.name == name and transfer.destination_rank == 0
        ]
        transferred = set()
        for transfer in rank0_transfers:
            self.assertEqual(transfer.source_rank, 1)
            transferred.update(transfer.global_linear_indices.tolist())
            old_exp_avg = self.old_states[transfer.source_rank][name].exp_avg.reshape(-1)
            new_exp_avg = result.local_states[transfer.destination_rank][name].exp_avg.reshape(-1)
            torch.testing.assert_close(
                old_exp_avg.index_select(0, transfer.source_local_linear_indices),
                new_exp_avg.index_select(0, transfer.destination_local_linear_indices),
                rtol=0,
                atol=0,
            )
        self.assertEqual(transferred, added_to_rank0)

        # A region removed from rank 1 remains in the canonical snapshot and
        # is present on a new rank; no optimizer history is discarded.
        old_rank1 = set(mask_linear_indices(self.old_masks[1][name], self.reference[name].shape).tolist())
        new_rank1 = set(mask_linear_indices(self.new_masks[1][name], self.reference[name].shape).tolist())
        removed_from_rank1 = old_rank1.difference(new_rank1)
        self.assertTrue(removed_from_rank1)
        all_new = set().union(
            *(
                set(mask_linear_indices(mask[name], self.reference[name].shape).tolist())
                for mask in self.new_masks
            )
        )
        self.assertTrue(removed_from_rank1.issubset(all_new))
        for index in removed_from_rank1:
            self.assertEqual(
                result.global_states[name].exp_avg_sq.reshape(-1)[index],
                self.global_states[name].exp_avg_sq.reshape(-1)[index],
            )

    def test_replica_disagreement_is_rejected_instead_of_overwriting_history(self):
        corrupted = copy.deepcopy(self.old_states)
        name = "model.layers.0.mlp.gate_proj.weight"
        # Ranks 0 and 2 have identical old FFN masks, so this modifies a true
        # replica rather than a disjoint coordinate.
        corrupted[2][name].exp_avg.reshape(-1)[0] += 1
        with self.assertRaisesRegex(ValueError, "disagrees with an old owner"):
            migrate_named_adam_states(
                self.reference,
                self.old_masks,
                self.new_masks,
                corrupted,
            )

    def test_step_and_fp32_master_state_must_be_collectively_consistent(self):
        bad_step = copy.deepcopy(self.old_states)
        first_name = next(iter(self.reference))
        bad_step[3][first_name].step += 1
        with self.assertRaisesRegex(ValueError, "Adam step differs"):
            migrate_named_adam_states(
                self.reference, self.old_masks, self.new_masks, bad_step
            )

        missing_master = copy.deepcopy(self.old_states)
        missing_master[2][first_name].master_parameter = None
        with self.assertRaisesRegex(ValueError, "master parameter is missing"):
            migrate_named_adam_states(
                self.reference, self.old_masks, self.new_masks, missing_master
            )

    def test_old_and_new_collective_coverage_are_both_required(self):
        reference = OrderedDict(vector=torch.zeros(4))
        complete = [
            OrderedDict(vector=torch.tensor([0, 1])),
            OrderedDict(vector=torch.tensor([2, 3])),
        ]
        incomplete = [
            OrderedDict(vector=torch.tensor([0, 1])),
            OrderedDict(vector=torch.tensor([2])),
        ]
        assert_complete_mask_coverage(reference, complete, label="complete")
        with self.assertRaisesRegex(ValueError, "do not cover"):
            assert_complete_mask_coverage(reference, incomplete, label="incomplete")


if __name__ == "__main__":
    unittest.main()
