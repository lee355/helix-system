from __future__ import annotations

import copy
import tempfile
import types
import unittest
from unittest import mock

from vendor_bootstrap import activate_local_dependencies


activate_local_dependencies()

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from dschat.helix.communication import build_communication_plan
from dschat.helix.deepspeed_paper import _collective_validate_reconstruction
from dschat.helix.masking import (
    build_structured_masks,
    canonicalize_state_dict_by_importance,
    slice_state_dict,
)
from dschat.helix.reconstruction import (
    exact_state_dict_alias_groups,
    reconstruct_full_state_dict_cpu,
    validate_reconstruction_metadata,
)


def _tiny_llama_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 32,
        },
        attention_dropout=0.0,
        tie_word_embeddings=True,
    )


def _local_model(full_config, spec) -> LlamaForCausalLM:
    config = copy.deepcopy(full_config)
    query_heads_per_kv = (
        full_config.num_attention_heads // full_config.num_key_value_heads
    )
    config.num_key_value_heads = len(spec.attention_groups)
    config.num_attention_heads = len(spec.attention_groups) * query_heads_per_kv
    config.intermediate_size = len(spec.ffn_columns)
    return LlamaForCausalLM(config)


class HelixReconstructionEquivalenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.config = _tiny_llama_config()
        self.model = LlamaForCausalLM(self.config).eval()
        self.input_ids = torch.tensor([[1, 7, 13, 2, 19, 3]], dtype=torch.long)

    def _logits(self, model):
        model.eval()
        with torch.no_grad():
            return model(input_ids=self.input_ids, use_cache=False).logits

    def test_llama3_gqa_importance_reorder_and_full_slice_are_equivalent(self):
        self.assertEqual(self.model.model.rotary_emb.rope_type, "llama3")
        layer = self.model.model.layers[0].self_attn
        self.assertEqual(layer.num_heads, 4)
        self.assertEqual(layer.num_key_value_heads, 2)
        self.assertIs(
            self.model.model.embed_tokens.weight,
            self.model.lm_head.weight,
        )

        original_logits = self._logits(self.model)
        canonical_state = self.model.state_dict()
        permutations = canonicalize_state_dict_by_importance(
            canonical_state, self.model.config
        )
        self.assertEqual(
            set(permutations[0]), {"attention_groups", "ffn_columns"}
        )
        torch.testing.assert_close(
            self._logits(self.model), original_logits, atol=2e-6, rtol=2e-6
        )

        masks, specs = build_structured_masks(
            canonical_state, [1.0], config=self.model.config
        )
        sliced = slice_state_dict(canonical_state, masks[0])
        full_slice_model = _local_model(self.model.config, specs[0]).eval()
        incompatible = full_slice_model.load_state_dict(sliced, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertIs(
            full_slice_model.model.embed_tokens.weight,
            full_slice_model.lm_head.weight,
        )
        torch.testing.assert_close(
            self._logits(full_slice_model),
            self._logits(self.model),
            atol=0.0,
            rtol=0.0,
        )

    def test_three_rank_reconstruction_strict_load_forward_and_save_reload(self):
        canonicalize_state_dict_by_importance(
            self.model.state_dict(), self.model.config
        )
        reference_state = self.model.state_dict()
        masks, specs = build_structured_masks(
            reference_state, [0.5, 0.5, 0.18], config=self.model.config
        )
        parameter_names = [name for name, _ in self.model.named_parameters()]
        self.assertIn("model.embed_tokens.weight", parameter_names)
        self.assertNotIn("lm_head.weight", parameter_names)
        plan = build_communication_plan(reference_state, masks, parameter_names)

        metadata_names = {
            name
            for layers in plan.reconstruction_global_by_rank.values()
            for name in layers
        }
        self.assertNotIn("lm_head.weight", metadata_names)
        audit = validate_reconstruction_metadata(
            reference_state,
            plan.reconstruction_global_by_rank,
            plan.reconstruction_local_by_rank,
            allowed_sources=plan.gather_ranks,
        )
        self.assertEqual(
            audit.alias_sources["lm_head.weight"], "model.embed_tokens.weight"
        )
        self.assertIn(
            ("model.embed_tokens.weight", "lm_head.weight"),
            exact_state_dict_alias_groups(reference_state),
        )
        self.assertGreater(len(plan.gather_ranks), 1)
        self.assertTrue(
            any(
                rank != 0 and layers
                for rank, layers in plan.reconstruction_global_by_rank.items()
            ),
            "the test must include full-model regions not owned by rank 0",
        )

        # Simulate a completed optimizer step.  Every local slice is made from
        # the same updated canonical state, exactly as Cluster-Reduce requires
        # before gather chooses one owner for each overlap.
        expected_model = copy.deepcopy(self.model).eval()
        with torch.no_grad():
            for index, parameter in enumerate(expected_model.parameters()):
                parameter.add_((index + 1) * 1e-4)
        expected_state = expected_model.state_dict()
        local_states = [slice_state_dict(expected_state, mask) for mask in masks]

        for rank, (local_state, spec) in enumerate(zip(local_states, specs)):
            local_model = _local_model(self.model.config, spec)
            incompatible = local_model.load_state_dict(local_state, strict=True)
            self.assertEqual(incompatible.missing_keys, [], msg=f"rank {rank}")
            self.assertEqual(incompatible.unexpected_keys, [], msg=f"rank {rank}")

        reconstructed = reconstruct_full_state_dict_cpu(
            reference_state,
            local_states,
            plan.reconstruction_global_by_rank,
            plan.reconstruction_local_by_rank,
            masks=masks,
            allowed_sources=plan.gather_ranks,
        )
        for name in expected_state:
            torch.testing.assert_close(
                reconstructed[name], expected_state[name], atol=0.0, rtol=0.0
            )
        self.assertEqual(
            reconstructed["model.embed_tokens.weight"].data_ptr(),
            reconstructed["lm_head.weight"].data_ptr(),
        )

        reconstructed_model = LlamaForCausalLM(copy.deepcopy(self.config)).eval()
        incompatible = reconstructed_model.load_state_dict(reconstructed, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertIs(
            reconstructed_model.model.embed_tokens.weight,
            reconstructed_model.lm_head.weight,
        )
        torch.testing.assert_close(
            self._logits(reconstructed_model),
            self._logits(expected_model),
            atol=0.0,
            rtol=0.0,
        )

        with tempfile.TemporaryDirectory() as directory:
            reconstructed_model.save_pretrained(directory, safe_serialization=False)
            reloaded = LlamaForCausalLM.from_pretrained(
                directory, local_files_only=True
            ).eval()
            self.assertIs(
                reloaded.model.embed_tokens.weight,
                reloaded.lm_head.weight,
            )
            torch.testing.assert_close(
                self._logits(reloaded),
                self._logits(expected_model),
                atol=0.0,
                rtol=0.0,
            )

    def test_overlap_conflicts_and_incomplete_owner_metadata_are_rejected(self):
        canonicalize_state_dict_by_importance(
            self.model.state_dict(), self.model.config
        )
        reference_state = self.model.state_dict()
        masks, _ = build_structured_masks(
            reference_state, [0.5, 0.5, 0.18], config=self.model.config
        )
        parameter_names = [name for name, _ in self.model.named_parameters()]
        plan = build_communication_plan(reference_state, masks, parameter_names)
        local_states = [slice_state_dict(reference_state, mask) for mask in masks]

        unsynchronized = copy.deepcopy(local_states)
        unsynchronized[1]["model.norm.weight"][0] += 1.0
        with self.assertRaisesRegex(ValueError, "unsynchronized overlap"):
            reconstruct_full_state_dict_cpu(
                reference_state,
                unsynchronized,
                plan.reconstruction_global_by_rank,
                plan.reconstruction_local_by_rank,
                masks=masks,
                allowed_sources=plan.gather_ranks,
            )

        global_conflict = copy.deepcopy(plan.reconstruction_global_by_rank)
        local_conflict = copy.deepcopy(plan.reconstruction_local_by_rank)
        conflict_name = "model.norm.weight"
        global_conflict[0][conflict_name].append(
            copy.deepcopy(global_conflict[0][conflict_name][0])
        )
        local_conflict[0][conflict_name].append(
            copy.deepcopy(local_conflict[0][conflict_name][0])
        )
        with self.assertRaisesRegex(ValueError, "conflict"):
            validate_reconstruction_metadata(
                reference_state, global_conflict, local_conflict
            )

        global_gap = copy.deepcopy(plan.reconstruction_global_by_rank)
        local_gap = copy.deepcopy(plan.reconstruction_local_by_rank)
        del global_gap[0][conflict_name]
        del local_gap[0][conflict_name]
        with self.assertRaisesRegex(ValueError, "uncovered"):
            validate_reconstruction_metadata(reference_state, global_gap, local_gap)

        with self.assertRaisesRegex(ValueError, "owner sources absent"):
            validate_reconstruction_metadata(
                reference_state,
                plan.reconstruction_global_by_rank,
                plan.reconstruction_local_by_rank,
                allowed_sources=[0],
            )


    def test_owner_local_shape_error_is_reported_by_collective_consensus(self):
        canonicalize_state_dict_by_importance(
            self.model.state_dict(), self.model.config
        )
        reference_state = self.model.state_dict()
        masks, _ = build_structured_masks(
            reference_state, [0.5, 0.5, 0.18], config=self.model.config
        )
        plan = build_communication_plan(
            reference_state,
            masks,
            [name for name, _ in self.model.named_parameters()],
        )
        local_states = [slice_state_dict(reference_state, mask) for mask in masks]
        owner = next(
            rank
            for rank, layers in plan.reconstruction_global_by_rank.items()
            if rank != 0 and layers
        )
        broken_local = copy.deepcopy(local_states[owner])
        broken_name = next(iter(plan.reconstruction_global_by_rank[owner]))
        broken_local[broken_name] = torch.empty(0)
        engine = types.SimpleNamespace(
            full_model=self.model,
            module=types.SimpleNamespace(state_dict=lambda: broken_local),
            complementary_rank_ls=plan.gather_ranks,
            complementary_groups_ranks_to_rectangles=(
                plan.reconstruction_global_by_rank
            ),
            complementary_groups_to_offset_rectangles=(
                plan.reconstruction_local_by_rank
            ),
        )

        def gather_errors(errors, local_error):
            self.assertIsNotNone(local_error)
            self.assertIn(broken_name, local_error)
            for rank in range(len(errors)):
                errors[rank] = None
            errors[owner] = local_error

        with mock.patch.object(torch.distributed, "get_rank", return_value=owner), mock.patch.object(
            torch.distributed, "get_world_size", return_value=len(masks)
        ), mock.patch.object(
            torch.distributed, "all_gather_object", side_effect=gather_errors
        ):
            with self.assertRaisesRegex(RuntimeError, "failed collectively"):
                _collective_validate_reconstruction(engine, plan.gather_ranks)


if __name__ == "__main__":
    unittest.main()
