import unittest

import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from dschat.helix.qwen3_compat import (
    Qwen3CompatConfig,
    install_qwen3_compat_attention,
)


class HelixQwen3ExplicitHeadDimTest(unittest.TestCase):
    def test_head_dim_can_differ_from_hidden_over_attention_heads(self):
        install_qwen3_compat_attention()
        config = Qwen3CompatConfig(
            vocab_size=48,
            hidden_size=40,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
            attention_bias=False,
        )
        config.head_prune_rate = 0.5
        config.prune_rate = 0.5
        config.freeze_blocks = []
        config.submodel_ids = {
            "model.layers.0.self_attn.o_proj.weight": None,
            "model.layers.0.mlp.down_proj.weight": None,
        }
        config._attn_implementation = "sdpa"
        model = Qwen2ForCausalLM(config)
        attention = model.model.layers[0].self_attn
        self.assertEqual(tuple(attention.q_proj.weight.shape), (16, 40))
        self.assertEqual(tuple(attention.o_proj.weight.shape), (40, 16))
        input_ids = torch.randint(0, 48, (2, 4))
        loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


if __name__ == "__main__":
    unittest.main()
