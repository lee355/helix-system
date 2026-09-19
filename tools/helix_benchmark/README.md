# Local Helix benchmarking

`run_local.py` launches the canonical `helix_run.py` or `helix_profile.py`
entry point on explicitly selected idle local GPUs. It records the exact
command, GPU inventory, logs, periodic NVML samples, exit status, and elapsed
time in a fresh output directory.

Prepare a deterministic MathInstruct subset:

```bash
python tools/helix_benchmark/prepare_math_subset.py \
  --source /path/to/MathInstruct.json \
  --model /path/to/model \
  --output results/helix/math_subset.json \
  --count 1024
```

Profile selected GPUs:

```bash
python tools/helix_benchmark/run_local.py \
  --gpus 0,1 --output results/helix/profile \
  --entry helix_profile.py --timeout 900 -- \
  --model_name_or_path /path/to/model \
  --output_path results/helix/profiles.json \
  --max_seq_len 512 --max_micro_batch_size 4 \
  --minimum_submodel_size 0.125 --maximum_submodel_size 0.5 \
  --warmup_steps 4 --measure_steps 3 --dtype fp16
```

Run a bounded training job:

```bash
python tools/helix_benchmark/run_local.py \
  --gpus 0,1 --output results/helix/train --timeout 1200 -- \
  --data_path results/helix/math_subset.json \
  --model_name_or_path /path/to/model --trainset math \
  --max_seq_len 512 --learning_rate 5e-6 --weight_decay 0 \
  --num_train_epochs 1 --gradient_accumulation_steps 1 \
  --lr_scheduler_type constant --num_warmup_steps 0 \
  --seed 1234 --dtype fp16 --zero_stage 0 --print_loss \
  --helix_submodel_sizes 0.5,0.5 \
  --helix_micro_batch_sizes 4,4 \
  --helix_max_steps 100 --helix_skip_checkpoint

python tools/helix_benchmark/summarize.py results/helix/train
```

The launcher refuses GPUs that are already busy and only terminates the process
group it started. Full profiling and training require the customized
third-party runtime that is not included in this initial public snapshot.
Choose GPU IDs, memory bounds, communication settings, and timeouts for the
target machine rather than treating these examples as universal defaults.
