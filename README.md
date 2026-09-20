# Helix system implementation

This repository is the public source snapshot of Helix, a heterogeneous
submodel-parallel system for full-parameter fine-tuning. It focuses on the
Helix implementation itself; unrelated comparison baselines, experiment
reports, generated artifacts, and private development configuration are not
part of this repository.

## Repository layout

- `training/step1_supervised_finetuning/dschat/helix/` contains the core
  masking, planning, communication, model construction, optimizer-state
  migration, dynamic resizing, profiling, reconstruction, and evaluation
  logic.
- `training/step1_supervised_finetuning/helix_run.py` is the canonical training
  entry point.
- `training/step1_supervised_finetuning/helix_profile.py` is the canonical
  profiling entry point.
- `training/step1_supervised_finetuning/training_scripts/helix_*.sh` contains
  portable launch examples configured through environment variables.
- `training/step1_supervised_finetuning/tests/` and `tests/` contain Helix unit,
  integration, and smoke tests.
- `tools/helix_benchmark/` and `tools/helix_acceptance/` contain bounded local
  benchmarking and acceptance-data utilities.

The implementation design and invariants are documented in
[`training/step1_supervised_finetuning/HELIX_IMPLEMENTATION.md`](training/step1_supervised_finetuning/HELIX_IMPLEMENTATION.md).

## Runtime compatibility

Helix targets a customized runtime integration based on DeepSpeed 0.14.4,
Transformers 4.46.2, and THOP 0.1.1.post2209072238+helix. The runtime includes
the custom DeepSpeed engine API, model integration, and optimizer behavior used
by Helix. The entry points use `vendor_bootstrap.py` to activate the matching
dependency sources under `third_party/`.

Use the matching dependency stack for profiling, distributed training,
optimizer migration, checkpoint reconstruction, and runtime integration tests.

## Launch examples

Activate a compatible Python environment first. The Llama examples read model
and data locations from environment variables:

```bash
export HELIX_MODEL_PATH=/path/to/model
export HELIX_DATA_PATH=/path/to/MathInstruct.json

training/step1_supervised_finetuning/training_scripts/helix_profile.sh \
  profiles/llama.json 16
training/step1_supervised_finetuning/training_scripts/helix_math.sh \
  profiles/llama.json outputs/helix-math
```

For a multi-node run, additionally set `HELIX_HOSTFILE` and optionally
`HELIX_INCLUDE`. Network-interface and NCCL tuning remain environment-specific
and are deliberately not hard-coded in the public scripts.

## Source-level validation

With compatible PyTorch and Transformers packages installed:

```bash
export PYTHONPATH="$PWD/training/step1_supervised_finetuning${PYTHONPATH:+:$PYTHONPATH}"
python -m compileall -q training tools tests
python -m unittest discover \
  -s training/step1_supervised_finetuning/tests \
  -p 'test_helix_*.py'
```

Tests that exercise the customized DeepSpeed engine should run against the
matching runtime stack described above.
