"""Narrow runtime precision repairs for official Hugging Face Llama modules."""

from __future__ import annotations

from typing import Any, Tuple

import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


class RotaryPrecisionError(RuntimeError):
    """Raised when an official Llama RoPE buffer has no trustworthy FP32 source."""


def _valid_fp32_frequency(value: Any, expected_shape: torch.Size) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and not value.is_meta
        and value.dtype == torch.float32
        and value.shape == expected_shape
        and not value.requires_grad
        and bool(torch.isfinite(value).all().item())
    )


def _rebuild_frequency(
    module: LlamaRotaryEmbedding,
    device: torch.device,
    expected_shape: torch.Size,
    module_name: str,
) -> Tuple[torch.Tensor, Any]:
    rope_init_fn = getattr(module, "rope_init_fn", None)
    rope_kwargs = getattr(module, "rope_kwargs", None)
    if not callable(rope_init_fn) or not isinstance(rope_kwargs, dict):
        raise RotaryPrecisionError(
            f"Official Llama rotary module {module_name!r} has no reliable "
            "original_inv_freq and no callable rope_init_fn/rope_kwargs"
        )
    try:
        rebuilt, attention_scaling = rope_init_fn(
            getattr(module, "config", None),
            device,
            **rope_kwargs,
        )
    except Exception as exc:
        raise RotaryPrecisionError(
            f"Failed to rebuild FP32 inv_freq for official Llama rotary "
            f"module {module_name!r}"
        ) from exc
    if not _valid_fp32_frequency(rebuilt, expected_shape):
        dtype = getattr(rebuilt, "dtype", None)
        shape = getattr(rebuilt, "shape", None)
        raise RotaryPrecisionError(
            f"rope_init_fn returned an unreliable inv_freq for official Llama "
            f"rotary module {module_name!r}: dtype={dtype}, shape={shape}, "
            f"expected dtype=torch.float32, shape={tuple(expected_shape)}"
        )
    return rebuilt.to(device=device), attention_scaling


@torch.no_grad()
def restore_rotary_fp32(model: torch.nn.Module) -> int:
    """Restore official Llama RoPE frequencies after a module-wide dtype cast.

    ``nn.Module.half()`` replaces registered buffers, but the official
    Transformers 4.46.2 ``LlamaRotaryEmbedding.original_inv_freq`` attribute
    continues to reference the pre-cast FP32 tensor.  That tensor is the first
    choice.  If it is not structurally reliable, the official ``rope_init_fn``
    is used to rebuild the frequency instead.  The potentially quantized
    current ``inv_freq`` is never promoted and reused as a recovery source.
    """

    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"model must be torch.nn.Module, found {type(model).__name__}")

    restored_count = 0
    for module_name, module in model.named_modules():
        # Exact type is intentional: subclasses may implement different RoPE
        # state semantics despite inheriting the same attribute names.
        if type(module) is not LlamaRotaryEmbedding:
            continue
        inv_freq = getattr(module, "inv_freq", None)
        if (
            not isinstance(inv_freq, torch.Tensor)
            or module._buffers.get("inv_freq") is not inv_freq
            or inv_freq.is_meta
            or not inv_freq.is_floating_point()
        ):
            raise RotaryPrecisionError(
                f"Official Llama rotary module {module_name!r} has an invalid "
                "registered inv_freq buffer"
            )

        original = getattr(module, "original_inv_freq", None)
        if _valid_fp32_frequency(original, inv_freq.shape):
            restored = original.to(device=inv_freq.device, dtype=torch.float32)
        else:
            restored, attention_scaling = _rebuild_frequency(
                module,
                inv_freq.device,
                inv_freq.shape,
                module_name,
            )
            module.attention_scaling = attention_scaling

        module.register_buffer("inv_freq", restored, persistent=False)
        if (
            module.inv_freq.dtype != torch.float32
            or module.inv_freq.device != inv_freq.device
            or "inv_freq" not in module._non_persistent_buffers_set
        ):
            raise RotaryPrecisionError(
                f"Failed to register a nonpersistent FP32 inv_freq for official "
                f"Llama rotary module {module_name!r}"
            )
        restored_count += 1
    return restored_count


__all__ = ["RotaryPrecisionError", "restore_rotary_fp32"]
