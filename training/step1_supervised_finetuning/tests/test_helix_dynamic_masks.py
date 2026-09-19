import types
import unittest
from collections import OrderedDict

import torch

from dschat.helix.dynamic_masks import resize_rank_masks
from dschat.helix.dynamic_rectangles import build_dynamic_copy_plan
from dschat.helix.masking import build_structured_masks, infer_model_structure
from dschat.helix.paper_semantics import install_paper_mask_semantics


install_paper_mask_semantics()


def _state():
    prefix = "model.layers.0"
    return OrderedDict(
        {
            "model.embed_tokens.weight": torch.zeros(7, 4),
            f"{prefix}.self_attn.q_proj.weight": torch.zeros(4, 4),
            f"{prefix}.self_attn.k_proj.weight": torch.zeros(2, 4),
            f"{prefix}.self_attn.v_proj.weight": torch.zeros(2, 4),
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


class HelixDynamicMaskTest(unittest.TestCase):
    def test_shrink_changes_only_affected_rank_and_removes_redundant_tail(self):
        state = _state()
        structure = infer_model_structure(state, _config())
        masks, specs = build_structured_masks(
            state,
            [1.0, 0.5, 0.5, 0.5],
            structure=structure,
        )
        update = resize_rank_masks(
            state,
            masks,
            specs,
            changed_rank=0,
            new_size=0.5,
            structure=structure,
        )
        self.assertEqual(update.specs[0].attention_groups, (0,))
        self.assertEqual(update.removed_attention_groups, (1,))
        for rank in range(1, 4):
            self.assertEqual(update.specs[rank], specs[rank])
            for name in masks[rank]:
                old = masks[rank][name]
                new = update.masks[rank][name]
                if isinstance(old, tuple):
                    self.assertTrue(torch.equal(old[0], new[0]))
                    self.assertTrue(torch.equal(old[1], new[1]))
                else:
                    self.assertTrue(torch.equal(old, new))

    def test_complete_rectangle_plan_copies_every_new_parameter_coordinate(self):
        state = _state()
        structure = infer_model_structure(state, _config())
        masks, specs = build_structured_masks(
            state,
            [1.0, 0.5, 0.5, 0.5],
            structure=structure,
        )
        update = resize_rank_masks(
            state,
            masks,
            specs,
            changed_rank=1,
            new_size=1.0,
            structure=structure,
        )
        parameter_names = list(state)
        plan = build_dynamic_copy_plan(
            state,
            masks,
            update.masks,
            parameter_names,
        )
        self.assertTrue(plan.copies)
        self.assertTrue(any(item.source_rank != item.destination_rank for item in plan.copies))
        self.assertTrue(any(item.source_rank == item.destination_rank for item in plan.copies))

        # Rectangle areas for each destination/name must equal its new local numel.
        for rank in range(4):
            for name, tensor in state.items():
                expected_mask = update.masks[rank][name]
                if tensor.ndim == 1:
                    expected = expected_mask.numel()
                else:
                    expected = expected_mask[0].numel() * expected_mask[1].numel()
                actual = 0
                for item in plan.copies_for_destination(rank):
                    if item.name != name:
                        continue
                    rectangle = item.destination_local_rectangle
                    if tensor.ndim == 1:
                        actual += rectangle[1][1] - rectangle[0][1] + 1
                    else:
                        actual += (
                            rectangle[1][0] - rectangle[0][0] + 1
                        ) * (
                            rectangle[1][1] - rectangle[0][1] + 1
                        )
                self.assertEqual(actual, expected, (rank, name))


if __name__ == "__main__":
    unittest.main()
