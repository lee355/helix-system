import math
import types
import unittest
from collections import OrderedDict

import torch

from dschat.helix.masking import (
    build_structured_masks,
    canonicalize_state_dict_by_importance,
    infer_model_structure,
    serialize_mask,
    slice_state_dict,
)


def _toy_state():
    generator = torch.Generator().manual_seed(7)

    def rand(*shape):
        return torch.randn(*shape, generator=generator)

    prefix = "model.layers.0"
    return OrderedDict(
        {
            "model.embed_tokens.weight": rand(7, 4),
            f"{prefix}.self_attn.q_proj.weight": rand(4, 4),
            f"{prefix}.self_attn.q_proj.bias": rand(4),
            f"{prefix}.self_attn.k_proj.weight": rand(2, 4),
            f"{prefix}.self_attn.k_proj.bias": rand(2),
            f"{prefix}.self_attn.v_proj.weight": rand(2, 4),
            f"{prefix}.self_attn.v_proj.bias": rand(2),
            f"{prefix}.self_attn.o_proj.weight": rand(4, 4),
            f"{prefix}.mlp.gate_proj.weight": rand(4, 4),
            f"{prefix}.mlp.up_proj.weight": rand(4, 4),
            f"{prefix}.mlp.down_proj.weight": rand(4, 4),
            f"{prefix}.input_layernorm.weight": rand(4),
            "model.norm.weight": rand(4),
            "lm_head.weight": rand(7, 4),
        }
    )


def _config():
    return types.SimpleNamespace(
        hidden_size=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=4,
    )


def _layer_output(state, inputs):
    prefix = "model.layers.0"
    q = inputs @ state[f"{prefix}.self_attn.q_proj.weight"].T
    q = q + state[f"{prefix}.self_attn.q_proj.bias"]
    k = inputs @ state[f"{prefix}.self_attn.k_proj.weight"].T
    k = k + state[f"{prefix}.self_attn.k_proj.bias"]
    v = inputs @ state[f"{prefix}.self_attn.v_proj.weight"].T
    v = v + state[f"{prefix}.self_attn.v_proj.bias"]
    # Each KV head serves two adjacent query heads in this toy GQA model.
    repeated_k = k.repeat_interleave(2, dim=-1)
    repeated_v = v.repeat_interleave(2, dim=-1)
    attention = torch.sigmoid(q * repeated_k) * repeated_v
    attention = attention @ state[f"{prefix}.self_attn.o_proj.weight"].T

    gate = inputs @ state[f"{prefix}.mlp.gate_proj.weight"].T
    up = inputs @ state[f"{prefix}.mlp.up_proj.weight"].T
    mlp = (torch.nn.functional.silu(gate) * up) @ state[f"{prefix}.mlp.down_proj.weight"].T
    return attention + mlp


class HelixMaskingTest(unittest.TestCase):
    def test_importance_canonicalization_preserves_full_model_function(self):
        state = _toy_state()
        inputs = torch.randn(3, 4, generator=torch.Generator().manual_seed(11))
        expected = _layer_output(state, inputs)

        permutations = canonicalize_state_dict_by_importance(state, _config())

        self.assertEqual(set(permutations[0]), {"attention_groups", "ffn_columns"})
        torch.testing.assert_close(_layer_output(state, inputs), expected)

    def test_cumulative_circular_masks_are_balanced_and_shape_safe(self):
        state = _toy_state()
        structure = infer_model_structure(state, _config())
        masks, specs = build_structured_masks(
            state,
            submodel_sizes=[0.5, 0.5, 0.5],
            structure=structure,
        )

        self.assertEqual(specs[0].attention_groups, (0,))
        self.assertEqual(specs[1].attention_groups, (1,))
        self.assertEqual(specs[2].attention_groups, (0,))
        self.assertEqual(specs[0].ffn_columns, (0, 1))
        self.assertEqual(specs[1].ffn_columns, (2, 3))
        self.assertEqual(specs[2].ffn_columns, (0, 1))

        local = slice_state_dict(state, masks[1])
        self.assertEqual(local["model.layers.0.self_attn.q_proj.weight"].shape, (2, 4))
        self.assertEqual(local["model.layers.0.self_attn.k_proj.weight"].shape, (1, 4))
        self.assertEqual(local["model.layers.0.self_attn.o_proj.weight"].shape, (4, 2))
        self.assertEqual(local["model.layers.0.mlp.gate_proj.weight"].shape, (2, 4))
        self.assertEqual(local["model.layers.0.mlp.down_proj.weight"].shape, (4, 2))

    def test_one_dimensional_mask_of_length_two_serializes_as_vector(self):
        state = _toy_state()
        masks, _ = build_structured_masks(state, [0.5], config=_config())
        serialized = serialize_mask(masks[0])
        q_bias = serialized["model.layers.0.self_attn.q_proj.bias"]
        self.assertIsInstance(q_bias, list)
        self.assertEqual(len(q_bias), 2)

    def test_invalid_size_fails_before_distributed_launch(self):
        with self.assertRaisesRegex(ValueError, "submodel size"):
            build_structured_masks(_toy_state(), [0.0], config=_config())


if __name__ == "__main__":
    unittest.main()
