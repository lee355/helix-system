#!/usr/bin/env python
"""Stable launcher using repository-local Helix dependencies."""

from vendor_bootstrap import activate_local_dependencies

activate_local_dependencies()

from dschat.helix.deepspeed_paper_v2 import (
    install_bounded_paper_deepspeed_runtime,
)
from dschat.helix.dynamic_safety import install_dynamic_safety_controller
from dschat.helix.model_structure import infer_model_structure

install_bounded_paper_deepspeed_runtime()
install_dynamic_safety_controller()

import helix_train
from dschat.helix import dynamic_controller

helix_train.infer_model_structure = infer_model_structure
helix_train.implementation.infer_model_structure = infer_model_structure
dynamic_controller.infer_model_structure = infer_model_structure
main = helix_train.main


if __name__ == "__main__":
    main()
