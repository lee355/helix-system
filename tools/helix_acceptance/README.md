# Helix acceptance utilities

These utilities prepare deterministic training/evaluation inputs, evaluate a
reconstructed full checkpoint, verify equivalent evaluation behavior, and
summarize acceptance artifacts.

The fixed MMLU-math protocol covers `abstract_algebra`,
`college_mathematics`, `elementary_mathematics`,
`high_school_mathematics`, and `high_school_statistics`. Each subject's dev
examples are used as demonstrations and its test split is scored by candidate
conditional log likelihood.

From the repository root:

```bash
python tools/helix_acceptance/prepare_mmlu_math.py \
  --help
python tools/helix_acceptance/prepare_training_cache.py \
  --help
python tools/helix_acceptance/evaluate_checkpoint.py \
  --help
python tools/helix_acceptance/verify_evaluation_equivalence.py \
  --help
python tools/helix_acceptance/summarize.py \
  --help
```

Full checkpoint evaluation and reconstruction require the customized
third-party runtime that is not included in this initial public snapshot.
Generated datasets, model weights, metrics, and reports are intentionally
excluded from version control.
