"""Explicit discrete semantics used by the Helix paper experiments."""

from __future__ import annotations

import math
from typing import MutableMapping


def paper_quantized_count(size: float, regions: int, alignment: int = 1) -> int:
    """Quantize ``s * regions`` to the nearest integer (half upward).

    Section 3.5's exact Llama allocation is the disambiguating source:
    ``s=.77/.45/.61`` retain ``18/11/15`` of 24 heads and
    ``6308/3686/4997`` of 8192 FFN columns.  These values consistently use
    nearest-integer quantization.  The earlier Figure 3 caption's ``.61 ->
    19/32`` conflicts with that exact allocation and is treated as a caption
    error.  Canonical paper runs use alignment one.
    """

    if not 0.0 < size <= 1.0:
        raise ValueError(f"submodel size must be in (0, 1], got {size}")
    if regions < 1:
        raise ValueError(f"regions must be positive, got {regions}")
    if alignment < 1:
        raise ValueError(f"alignment must be positive, got {alignment}")
    count = int(math.floor(float(size) * int(regions) + 0.5 + 1e-12))
    if alignment > 1:
        count = (count // alignment) * alignment
    return max(1, min(int(regions), count))


def apply_paper_deepspeed_config(config: MutableMapping) -> MutableMapping:
    # The current DS config defaults to no clipping. Keep that invariant
    # explicit: heterogeneous local whole-model norms cannot safely choose
    # independent clipping factors for shared regions.
    config["gradient_clipping"] = 0.0
    return config


def install_paper_mask_semantics() -> None:
    from . import masking

    masking._quantized_count = paper_quantized_count
