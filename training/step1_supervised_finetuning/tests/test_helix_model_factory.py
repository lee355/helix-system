import tempfile
import unittest

from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from dschat.helix.model_factory import create_paper_model
from dschat.helix.qwen3_compat import (
    Qwen3CompatConfig,
    install_qwen3_compat_attention,
)


class HelixModelFactoryTest(unittest.TestCase):
    def test_qwen3_checkpoint_is_loaded_with_qknorm_not_plain_qwen2(self):
        install_qwen3_compat_attention()
        config = Qwen3CompatConfig(
            vocab_size=32,
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
        config.head_prune_rate = 1.0
        config.prune_rate = 1.0
        config.freeze_blocks = None
        config.submodel_ids = None
        config._attn_implementation = "sdpa"
        source = Qwen2ForCausalLM(config)
        with tempfile.TemporaryDirectory() as directory:
            source.save_pretrained(directory, safe_serialization=False)
            loaded = create_paper_model(
                AutoModelForCausalLM,
                directory,
            )
        attention = loaded.model.layers[0].self_attn
        self.assertTrue(hasattr(attention, "q_norm"))
        self.assertTrue(hasattr(attention, "k_norm"))
        self.assertIsNone(attention.q_proj.bias)


if __name__ == "__main__":
    unittest.main()
