"""Final bounded installer for the paper-safe DeepSpeed runtime."""

from __future__ import annotations

from .deepspeed_paper import (
    _paper_reduce_non_expert_gradients,
    install_paper_deepspeed_runtime,
)


PAPER_COMMUNICATION_BUCKET_ELEMENTS = 6_291_456


def _bounded_reduce(self, grads, elements_per_buffer):
    return _paper_reduce_non_expert_gradients(
        self,
        grads,
        min(int(elements_per_buffer), PAPER_COMMUNICATION_BUCKET_ELEMENTS),
    )


def install_bounded_paper_deepspeed_runtime() -> None:
    install_paper_deepspeed_runtime()
    from deepspeed.runtime.engine import DeepSpeedEngine

    if getattr(DeepSpeedEngine, "_helix_bounded_runtime_installed", False):
        return
    DeepSpeedEngine._reduce_non_expert_gradients = _bounded_reduce
    DeepSpeedEngine._helix_bounded_runtime_installed = True
