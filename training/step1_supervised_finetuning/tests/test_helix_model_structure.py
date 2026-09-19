import types
import unittest
from collections import OrderedDict

import torch

from dschat.helix.model_structure import infer_model_structure


class HelixModelStructureTest(unittest.TestCase):
    def test_explicit_qwen3_head_dim_overrides_hidden_div_heads(self):
        state = OrderedDict(
            {
                "model.layers.0.self_attn.q_proj.weight": torch.zeros(32, 40),
                "model.layers.0.self_attn.k_proj.weight": torch.zeros(16, 40),
                "model.layers.0.mlp.gate_proj.weight": torch.zeros(24, 40),
            }
        )
        config = types.SimpleNamespace(
            hidden_size=40,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
        structure = infer_model_structure(state, config)
        self.assertEqual(structure.head_dim, 8)
        self.assertEqual(structure.num_attention_heads, 4)
        self.assertEqual(structure.num_key_value_heads, 2)


if __name__ == "__main__":
    unittest.main()
