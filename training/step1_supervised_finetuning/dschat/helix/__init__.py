"""Helix submodel-parallel training primitives.

The modules in this package are deliberately independent from DeepSpeed so
that mask generation, planning, and communication metadata can be validated
on CPU before a distributed job is launched.
"""

from .masking import (
    ModelStructure,
    SubmodelSpec,
    build_structured_masks,
    canonicalize_state_dict_by_importance,
    infer_model_structure,
    serialize_mask,
    slice_state_dict,
)

__all__ = [
    "ModelStructure",
    "SubmodelSpec",
    "build_structured_masks",
    "canonicalize_state_dict_by_importance",
    "infer_model_structure",
    "serialize_mask",
    "slice_state_dict",
]
