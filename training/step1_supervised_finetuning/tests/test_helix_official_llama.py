import tempfile
import unittest

import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from dschat.helix.model_factory import create_paper_model


class HelixOfficialLlamaTopologyTest(unittest.TestCase):
    def test_local_topology_keeps_official_llama3_rope_and_sdpa(self):
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            rope_scaling={
                "rope_type": "llama3",
                "factor": 4.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 2.0,
                "original_max_position_embeddings": 16,
            },
            tie_word_embeddings=False,
        )
        source = LlamaForCausalLM(config)
        with tempfile.TemporaryDirectory() as directory:
            source.save_pretrained(directory, safe_serialization=False)
            local = create_paper_model(
                AutoModelForCausalLM,
                directory,
                attention_rate=0.5,
                ffn_rate=0.5,
                submodel_ids={"shape_only": []},
                num_hidden_layers=1,
            )
        attention = local.model.layers[0].self_attn
        self.assertEqual(attention.__class__.__name__, "LlamaSdpaAttention")
        self.assertEqual(tuple(attention.q_proj.weight.shape), (16, 32))
        self.assertEqual(tuple(attention.k_proj.weight.shape), (8, 32))
        self.assertEqual(tuple(attention.o_proj.weight.shape), (32, 16))
        self.assertEqual(tuple(local.model.layers[0].mlp.gate_proj.weight.shape), (8, 32))
        self.assertEqual(attention.rotary_emb.rope_type, "llama3")
        input_ids = torch.randint(0, 64, (2, 5))
        loss = local(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


if __name__ == "__main__":
    unittest.main()
