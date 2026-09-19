from __future__ import annotations

import unittest
import sys
from pathlib import Path

STEP1_ROOT = Path(__file__).resolve().parents[1]
if str(STEP1_ROOT) not in sys.path:
    sys.path.insert(0, str(STEP1_ROOT))
from vendor_bootstrap import activate_local_dependencies


activate_local_dependencies()

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from dschat.helix.precision import RotaryPrecisionError, restore_rotary_fp32


def _gqa_submodel_config():
    return LlamaConfig(
        vocab_size=31,
        hidden_size=64,
        intermediate_size=80,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=4096,
        rope_theta=500000.0,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 256,
        },
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )


def _rotaries(model):
    return {
        name: module
        for name, module in model.named_modules()
        if type(module) is LlamaRotaryEmbedding
    }


class RestoreLlamaRotaryPrecisionTest(unittest.TestCase):
    def test_half_model_restores_canonical_bits_without_changing_parameters_or_gqa(self):
        model = LlamaForCausalLM(_gqa_submodel_config())
        canonical = {
            name: module.inv_freq.detach().clone()
            for name, module in _rotaries(model).items()
        }
        self.assertEqual(len(canonical), 3)
        attention = model.model.layers[0].self_attn
        topology = (
            attention.num_heads,
            attention.num_key_value_heads,
            attention.head_dim,
            attention.q_proj.out_features,
            attention.k_proj.out_features,
        )

        model.half()
        self.assertTrue(all(parameter.dtype == torch.float16 for parameter in model.parameters()))
        self.assertTrue(
            all(module.inv_freq.dtype == torch.float16 for module in _rotaries(model).values())
        )
        self.assertTrue(
            all(module.original_inv_freq.dtype == torch.float32 for module in _rotaries(model).values())
        )

        restored_count = restore_rotary_fp32(model)

        self.assertEqual(restored_count, len(canonical))
        for name, module in _rotaries(model).items():
            self.assertEqual(module.inv_freq.dtype, torch.float32)
            self.assertTrue(torch.equal(module.inv_freq, canonical[name]))
            self.assertIn("inv_freq", module._non_persistent_buffers_set)
        self.assertFalse(any(key.endswith("inv_freq") for key in model.state_dict()))
        self.assertTrue(all(parameter.dtype == torch.float16 for parameter in model.parameters()))
        restored_attention = model.model.layers[0].self_attn
        self.assertEqual(
            (
                restored_attention.num_heads,
                restored_attention.num_key_value_heads,
                restored_attention.head_dim,
                restored_attention.q_proj.out_features,
                restored_attention.k_proj.out_features,
            ),
            topology,
        )
        self.assertEqual(topology, (4, 1, 8, 32, 8))

    def test_invalid_original_rebuilds_instead_of_promoting_quantized_frequency(self):
        model = LlamaForCausalLM(_gqa_submodel_config())
        canonical = {
            name: module.inv_freq.detach().clone()
            for name, module in _rotaries(model).items()
        }
        model.half()
        quantized_promotions = {}
        for name, module in _rotaries(model).items():
            module.original_inv_freq = module.inv_freq
            quantized_promotions[name] = module.inv_freq.float().clone()
        self.assertTrue(
            any(
                not torch.equal(quantized_promotions[name], canonical[name])
                for name in canonical
            )
        )

        restore_rotary_fp32(model)

        for name, module in _rotaries(model).items():
            self.assertTrue(torch.equal(module.inv_freq, canonical[name]))
            self.assertFalse(torch.equal(module.inv_freq, quantized_promotions[name]))

    def test_unreliable_original_and_rebuilder_fail_explicitly(self):
        rotary = LlamaRotaryEmbedding(config=_gqa_submodel_config()).half()
        rotary.original_inv_freq = None
        rotary.rope_init_fn = lambda config, device, **kwargs: (
            torch.ones_like(rotary.inv_freq),
            1.0,
        )

        with self.assertRaisesRegex(RotaryPrecisionError, "unreliable inv_freq"):
            restore_rotary_fp32(rotary)

    def test_nonofficial_lookalike_is_untouched(self):
        class RotaryLookalike(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "inv_freq", torch.tensor([0.25], dtype=torch.float16), persistent=False
                )
                self.original_inv_freq = torch.tensor([0.5], dtype=torch.float32)

        lookalike = RotaryLookalike()
        pointer = lookalike.inv_freq.data_ptr()
        self.assertEqual(restore_rotary_fp32(lookalike), 0)
        self.assertEqual(lookalike.inv_freq.dtype, torch.float16)
        self.assertEqual(lookalike.inv_freq.data_ptr(), pointer)


if __name__ == "__main__":
    unittest.main()
