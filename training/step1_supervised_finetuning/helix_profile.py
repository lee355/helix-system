#!/usr/bin/env python
"""Real-runtime profiler using repository-local Helix dependencies."""

from vendor_bootstrap import activate_local_dependencies

activate_local_dependencies()

import helix_profile_ds as profile_runtime
import helix_profile_paper as profile_paper
from dschat.helix.model_factory import create_paper_model
from dschat.helix.model_structure import infer_model_structure


def _custom_engine_train_step(engine, input_ids, attention_mask, labels):
    engine.zero_grad()
    loss = engine(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    ).loss
    engine.backward(loss)
    engine.step()


profile_runtime._train_step = _custom_engine_train_step
profile_runtime.create_paper_model = create_paper_model
profile_runtime.infer_model_structure = infer_model_structure
profile_paper.create_paper_model = create_paper_model
profile_paper.infer_model_structure = infer_model_structure
main = profile_paper.main


if __name__ == "__main__":
    main()
