"""One-layer real-checkpoint validation for Llama-3.1-8B Helix topology."""

from __future__ import annotations

import argparse
import gc

from vendor_bootstrap import activate_local_dependencies

activate_local_dependencies()

import torch
from transformers import AutoModelForCausalLM

from dschat.helix.masking import (
    build_structured_masks,
    canonicalize_state_dict_by_importance,
    serialize_mask,
    slice_state_dict,
)
from dschat.helix.model import validate_state_shapes
from dschat.helix.model_factory import create_paper_model
from dschat.helix.model_structure import infer_model_structure
from dschat.helix.paper_semantics import install_paper_mask_semantics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Path or model identifier for a Llama-3.1-8B checkpoint",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    install_paper_mask_semantics()

    full_model = create_paper_model(
        AutoModelForCausalLM,
        args.model,
        num_hidden_layers=1,
    )
    full_state = full_model.state_dict()
    structure = infer_model_structure(full_state, full_model.config)
    assert structure.num_attention_heads == 32
    assert structure.num_key_value_heads == 8
    assert structure.head_dim == 128
    assert structure.intermediate_size == 14336
    canonicalize_state_dict_by_importance(
        full_state,
        full_model.config,
        structure,
    )
    masks, specs = build_structured_masks(
        full_state,
        [0.45, 0.55],
        structure=structure,
    )
    assert len(specs[0].attention_groups) == 4
    assert len(specs[0].attention_groups) * structure.query_heads_per_kv == 16
    assert len(specs[0].ffn_columns) == 6451
    local_state = slice_state_dict(full_state, masks[0])
    del full_state
    del full_model
    gc.collect()

    local_model = create_paper_model(
        AutoModelForCausalLM,
        args.model,
        attention_rate=specs[0].attention_rate,
        ffn_rate=specs[0].ffn_rate,
        submodel_ids=serialize_mask(masks[0]),
        freeze_blocks=[],
        num_hidden_layers=1,
    )
    validate_state_shapes(local_model, local_state)
    local_model.load_state_dict(local_state, strict=True)
    del local_state
    gc.collect()

    device = torch.device(args.device)
    local_model.to(device=device, dtype=torch.bfloat16)
    local_model.train()
    input_ids = torch.randint(
        0,
        local_model.config.vocab_size,
        (1, 8),
        device=device,
    )
    output = local_model(input_ids=input_ids, labels=input_ids, use_cache=False)
    if not torch.isfinite(output.loss):
        raise RuntimeError(f"non-finite loss: {output.loss}")
    output.loss.backward()
    layer = local_model.model.layers[0]
    if layer.self_attn.q_proj.weight.grad is None:
        raise RuntimeError("q_proj did not receive a gradient")
    print(
        "Llama-3.1-8B one-layer Helix validation passed: "
        f"loss={output.loss.detach().float().item():.6f}, "
        f"Q={tuple(layer.self_attn.q_proj.weight.shape)}, "
        f"K={tuple(layer.self_attn.k_proj.weight.shape)}, "
        f"FFN={tuple(layer.mlp.gate_proj.weight.shape)}, "
        f"rope={layer.self_attn.rotary_emb.rope_type}",
        flush=True,
    )


if __name__ == "__main__":
    main()
