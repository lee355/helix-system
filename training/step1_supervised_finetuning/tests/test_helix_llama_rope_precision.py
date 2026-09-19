from __future__ import annotations

import copy
import tempfile
import types
import unittest
from contextlib import nullcontext
from unittest import mock

from vendor_bootstrap import activate_local_dependencies


activate_local_dependencies()

import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from transformers.modeling_utils import no_init_weights
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from dschat.helix.acceptance import MathAcceptance, model_state_digest


def _config():
    return LlamaConfig(
        vocab_size=101,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
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
        tie_word_embeddings=True,
    )


def _half_parameter_model(source):
    config = copy.deepcopy(source.config)
    config.use_cache = False
    with no_init_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )
    model.tie_weights()
    model.load_state_dict(source.state_dict(), strict=True)
    return model.eval()


def _rotary(model):
    rotary = model.model.rotary_emb
    if not isinstance(rotary, LlamaRotaryEmbedding):
        raise AssertionError(f"unexpected shared rotary module: {type(rotary)!r}")
    return rotary


def _logits(model, input_ids):
    with torch.inference_mode():
        return model(input_ids=input_ids, use_cache=False).logits.float()


class LlamaRopePrecisionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.source = LlamaForCausalLM(_config()).eval()
        self.input_ids = torch.arange(512).remainder(self.source.config.vocab_size).unsqueeze(0)

    def test_dtype_cast_quantizes_nonpersistent_rope_and_reload_changes_logits(self):
        device_only = _half_parameter_model(self.source)
        rotary = _rotary(device_only)
        self.assertEqual(next(device_only.parameters()).dtype, torch.float16)
        self.assertIs(
            device_only.get_input_embeddings().weight,
            device_only.get_output_embeddings().weight,
        )
        self.assertEqual(rotary.inv_freq.dtype, torch.float32)
        self.assertEqual(rotary.original_inv_freq.dtype, torch.float32)
        expected_logits = _logits(device_only, self.input_ids)

        dtype_cast = copy.deepcopy(device_only).to(
            device=torch.device("cpu"), dtype=torch.float16
        ).eval()
        cast_rotary = _rotary(dtype_cast)
        self.assertEqual(cast_rotary.inv_freq.dtype, torch.float16)
        self.assertEqual(cast_rotary.original_inv_freq.dtype, torch.float32)
        self.assertNotEqual(
            cast_rotary.inv_freq.data_ptr(), cast_rotary.original_inv_freq.data_ptr()
        )
        cast_logits = _logits(dtype_cast, self.input_ids)
        self.assertGreater(float((cast_logits - expected_logits).abs().max()), 0.0)

        # inv_freq is persistent=False, so the state digest cannot reveal this
        # runtime precision difference.
        self.assertEqual(model_state_digest(dtype_cast), model_state_digest(device_only))
        with tempfile.TemporaryDirectory() as directory:
            dtype_cast.save_pretrained(directory, safe_serialization=False)
            reloaded = LlamaForCausalLM.from_pretrained(
                directory,
                torch_dtype=torch.float16,
                local_files_only=True,
                attn_implementation="sdpa",
            ).eval()
            self.assertEqual(_rotary(reloaded).inv_freq.dtype, torch.float32)
            self.assertIs(
                reloaded.get_input_embeddings().weight,
                reloaded.get_output_embeddings().weight,
            )
            self.assertEqual(model_state_digest(reloaded), model_state_digest(dtype_cast))
            reloaded_logits = _logits(reloaded, self.input_ids)

        torch.testing.assert_close(
            reloaded_logits, expected_logits, atol=0.0, rtol=0.0
        )
        self.assertGreater(float((reloaded_logits - cast_logits).abs().max()), 0.0)

    def test_original_fp32_frequency_restores_logits_after_module_half(self):
        expected = _half_parameter_model(self.source)
        expected_logits = _logits(expected, self.input_ids)
        deepspeed_style = copy.deepcopy(expected).half().eval()
        rotary = _rotary(deepspeed_style)
        self.assertEqual(rotary.inv_freq.dtype, torch.float16)
        self.assertEqual(rotary.original_inv_freq.dtype, torch.float32)
        degraded_logits = _logits(deepspeed_style, self.input_ids)
        self.assertGreater(float((degraded_logits - expected_logits).abs().max()), 0.0)

        restored = rotary.original_inv_freq.to(
            device=rotary.inv_freq.device, dtype=torch.float32
        )
        rotary.register_buffer("inv_freq", restored, persistent=False)
        self.assertEqual(rotary.inv_freq.dtype, torch.float32)
        torch.testing.assert_close(
            _logits(deepspeed_style, self.input_ids),
            expected_logits,
            atol=0.0,
            rtol=0.0,
        )

    def test_acceptance_evaluator_moves_fp16_model_without_casting_rope(self):
        controller = MathAcceptance.__new__(MathAcceptance)
        controller.args = types.SimpleNamespace(
            helix_eval_batch_size=1,
            helix_eval_max_length=512,
            helix_eval_fewshot=5,
        )
        controller.tokenizer = object()
        controller.data = object()
        controller.eval_model = None
        engine = types.SimpleNamespace(
            device=torch.device("cpu"),
            full_model=self.source,
        )
        observed = {}

        def fake_evaluate(model, tokenizer, data, **kwargs):
            observed["parameter_dtype"] = next(model.parameters()).dtype
            observed["inv_freq_dtype"] = _rotary(model).inv_freq.dtype
            observed["tied"] = (
                model.get_input_embeddings().weight
                is model.get_output_embeddings().weight
            )
            observed["overflow_policy"] = kwargs["overflow_policy"]
            return {"sentinel": True}

        with mock.patch(
            "dschat.helix.mmlu_math.evaluate_mmlu_math",
            side_effect=fake_evaluate,
        ), mock.patch(
            "dschat.helix.acceptance.torch.random.fork_rng",
            return_value=nullcontext(),
        ), mock.patch(
            "dschat.helix.acceptance.torch.cuda.empty_cache"
        ):
            result = controller._evaluate_on_root(engine)

        self.assertEqual(result, {"sentinel": True})
        self.assertEqual(observed["parameter_dtype"], torch.float16)
        self.assertEqual(observed["inv_freq_dtype"], torch.float32)
        self.assertTrue(observed["tied"])
        self.assertEqual(observed["overflow_policy"], "error")
        self.assertEqual(_rotary(controller.eval_model).inv_freq.dtype, torch.float32)
        self.assertIs(
            controller.eval_model.get_input_embeddings().weight,
            controller.eval_model.get_output_embeddings().weight,
        )

        expected_logits = _logits(controller.eval_model, self.input_ids)
        with tempfile.TemporaryDirectory() as directory:
            controller.eval_model.save_pretrained(directory, safe_serialization=False)
            reloaded = LlamaForCausalLM.from_pretrained(
                directory,
                torch_dtype=torch.float16,
                local_files_only=True,
                attn_implementation="sdpa",
            ).eval()
            self.assertEqual(next(reloaded.parameters()).dtype, torch.float16)
            self.assertEqual(_rotary(reloaded).inv_freq.dtype, torch.float32)
            self.assertIs(
                reloaded.get_input_embeddings().weight,
                reloaded.get_output_embeddings().weight,
            )
            torch.testing.assert_close(
                _logits(reloaded, self.input_ids),
                expected_logits,
                atol=0.0,
                rtol=0.0,
            )


if __name__ == "__main__":
    unittest.main()
