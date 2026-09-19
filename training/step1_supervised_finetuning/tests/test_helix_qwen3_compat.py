import unittest

import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from dschat.helix.qwen3_compat import (
    HelixQwenSdpaAttention,
    Qwen3CompatConfig,
    install_qwen3_compat_attention,
)


class HelixQwen3CompatTest(unittest.TestCase):
    def _model(self):
        install_qwen3_compat_attention()
        config = Qwen3CompatConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
            attention_bias=False,
            tie_word_embeddings=False,
        )
        config.head_prune_rate = 0.5
        config.prune_rate = 0.5
        config.freeze_blocks = []
        config.submodel_ids = {
            "model.layers.0.self_attn.o_proj.weight": None,
            "model.layers.0.mlp.down_proj.weight": None,
        }
        config._attn_implementation = "sdpa"
        config.output_attentions = False
        return Qwen2ForCausalLM(config)

    def test_qwen3_attention_has_qk_norm_and_no_qkv_bias(self):
        model = self._model()
        attention = model.model.layers[0].self_attn
        self.assertIsInstance(attention, HelixQwenSdpaAttention)
        self.assertIsNone(attention.q_proj.bias)
        self.assertIsNone(attention.k_proj.bias)
        self.assertIsNone(attention.v_proj.bias)
        self.assertEqual(tuple(attention.q_norm.weight.shape), (8,))
        self.assertEqual(tuple(attention.k_norm.weight.shape), (8,))
        self.assertEqual(tuple(attention.q_proj.weight.shape), (16, 32))
        self.assertEqual(tuple(attention.k_proj.weight.shape), (8, 32))
        self.assertEqual(tuple(model.model.layers[0].mlp.gate_proj.weight.shape), (8, 32))

    def test_pruned_sdpa_forward_and_backward_use_qk_norm(self):
        model = self._model()
        input_ids = torch.randint(0, 64, (2, 5))
        output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        attention = model.model.layers[0].self_attn
        self.assertIsNotNone(attention.q_norm.weight.grad)
        self.assertIsNotNone(attention.k_norm.weight.grad)


if __name__ == "__main__":
    unittest.main()
