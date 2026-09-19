# Helix implementation guide

The canonical implementation lives beside the historical `math_main.py`.
New jobs use `helix_run.py`; the historical entry remains untouched.

## Components

| Paper component | Canonical implementation |
| --- | --- |
| L1 canonical ordering and initial circular masks | `dschat/helix/masking.py`, `paper_semantics.py` |
| Exact overlaps, graph coloring and ring cost | `communication.py`, `cost.py` |
| Real DeepSpeed eight-point profiling | `helix_profile.py`, `helix_profile_ds.py`, `profiling.py` |
| Profile-bounded, quantized-coverage-safe search | `planner_final.py` |
| Heterogeneous shared data stream | `data.py` |
| Bounded Cluster-Reduce/gather and global FP16 overflow | `deepspeed_paper.py`, `deepspeed_paper_v2.py` |
| Complete Adam live migration | `dynamic_masks.py`, `dynamic_rectangles.py`, `deepspeed_adam_state.py`, `dynamic_controller.py` |
| Qwen3-Dense backport | `qwen3_compat.py`, `model_factory.py` |

Training is full-parameter: embeddings and output heads are trainable and
replicated parameters use an all-rank overlap group. Each parameter region is
averaged equally over retaining ranks, matching Section 4 rather than weighting
by local `b_i`.

## Discrete submodels

Section 3.5's exact allocation disambiguates integer conversion:
`s=.77/.45/.61` gives `18/11/15` of 24 heads and
`6308/3686/4997` of 8192 FFN columns. Canonical initial masks therefore use
nearest-integer quantization and validate coverage after quantization.

For Llama-3.2-3B, the installed attention kernel requires its 3Q:1KV mapping.
The user-selected safe rule is one KV head plus its three Q heads. Thus
`s=.45` retains four KV groups and 12 Q heads, not the paper's arbitrary
11-Q-head slice. Qwen3-8B uses its own 32Q/8KV layout, so each safe unit there
is 4Q:1KV.

## Qwen3

Qwen3 and Qwen2 share GQA/RoPE/SwiGLU/pre-RMSNorm skeletons, but are not the
same model. Qwen3 adds `q_norm/k_norm`, removes QKV biases, and has its own
`model_type`. The `ours_math` Transformers 4.46.2 release predates Qwen3.

The project backport uses the existing Qwen2 decoder/cache integration but:

- creates bias-free Q/K/V/O projections;
- restores QK-Norm immediately after projection reshape and before RoPE;
- respects explicit Qwen3 `head_dim`;
- forces the audited pruned SDPA path;
- keeps `q_norm/k_norm` replicated and synchronized;
- saves `model_type=qwen3`, so checkpoints remain compatible with an official
  newer Transformers installation.

Do not change a Qwen3 config to `qwen2`: that drops 72 QK-Norm tensors and adds
108 nonexistent QKV biases for a 36-layer model.

## Dynamic adjustment with complete Adam state

Live adjustment is optional and currently restricted to FP16, ZeRO-0,
`gradient_accumulation_steps=1`, and a fixed world size.

At each check, ranks report usable memory. Rank 0 applies the profile model to
the affected rank while keeping predicted compute close to its previous value.
The plan is queued and committed atomically at the next epoch boundary:

1. Only affected rank masks change. Growth retains old regions and adds the
   least-covered regions; shrink removes least-important regions only when
   another rank still owns them.
2. A compact rectangle plan covers every new local coordinate, including
   retained coordinates whose local offsets changed.
3. Each rank snapshots its local FP32 master, `exp_avg`, and `exp_avg_sq` to
   CPU. No full-model/root optimizer staging is used.
4. All ranks release and rebuild the custom DeepSpeed engine in the same
   group-creation order.
5. Self-owned and remote rectangles are restored in bounded P2P tiles. FP16
   parameters are regenerated from the authoritative FP32 master.
6. Adam `step`, optimizer groups/LR, loss scaler, scheduler, engine counters,
   RNG and true heterogeneous global-sample count are restored.

Enable it with:

```text
--helix_dynamic_check_interval 100
--helix_dynamic_memory_threshold_gib 1.0
--helix_dynamic_max_adjustments 1
--helix_dynamic_profiles_path /shared/path/profiles.json
```

The conservative first version switches only at an epoch boundary. If engine
reconstruction fails after the old engine is released, the job fails rather
than continuing with mixed masks; in-memory rollback would require the double
GPU capacity unavailable in a memory-pressure event.

## Running

Llama profiling and static training:

```bash
training/step1_supervised_finetuning/training_scripts/helix_profile.sh \
  /shared/path/llama_profiles.json 16
training/step1_supervised_finetuning/training_scripts/helix_math.sh \
  /shared/path/llama_profiles.json /shared/output/helix_math
```

Llama training with live adjustment:

```bash
training/step1_supervised_finetuning/training_scripts/helix_math_dynamic.sh \
  /shared/path/llama_profiles.json /shared/output/helix_math_dynamic
```

Qwen3 launchers require an explicit checkpoint path:

```bash
training/step1_supervised_finetuning/training_scripts/helix_profile_qwen3.sh \
  /shared/path/qwen3_profiles.json /path/to/Qwen3-8B 16
training/step1_supervised_finetuning/training_scripts/helix_math_qwen3.sh \
  /shared/path/qwen3_profiles.json /shared/output/helix_qwen3 /path/to/Qwen3-8B
```

## Validation

```bash
export PYTHONPATH=training/step1_supervised_finetuning:$PYTHONPATH
python -m unittest discover \
  -s training/step1_supervised_finetuning/tests \
  -p 'test_helix_*.py' -v
python -m unittest tests.test_helix_data -v
```

The suite includes an independent global Adam migration oracle, compact
rectangle coverage checks, and a two-process Gloo P2P restore test. A real
multi-node NCCL transition still needs to be validated on the target cluster
without active workloads.
