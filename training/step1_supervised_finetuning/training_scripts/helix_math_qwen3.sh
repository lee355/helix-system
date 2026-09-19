#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 PROFILE_JSON OUTPUT_DIR QWEN3_MODEL_PATH" >&2
  exit 2
fi

PROFILE_JSON=$1
OUTPUT_DIR=$2
MODEL_PATH=$3
: "${HELIX_DATA_PATH:?set HELIX_DATA_PATH to the training dataset}"
DATA_PATH=$HELIX_DATA_PATH
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENTRYPOINT="$SCRIPT_DIR/../helix_run.py"
MASTER_PORT=${HELIX_MASTER_PORT:-29545}
LAUNCH_ARGS=(--master_port "$MASTER_PORT")
if [[ -n "${HELIX_HOSTFILE:-}" ]]; then
  LAUNCH_ARGS+=(--hostfile "$HELIX_HOSTFILE")
fi
if [[ -n "${HELIX_INCLUDE:-}" ]]; then
  LAUNCH_ARGS+=(--include "$HELIX_INCLUDE")
fi

mkdir -p "$OUTPUT_DIR"
export PDSH_RCMD_TYPE="${PDSH_RCMD_TYPE:-ssh}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

deepspeed "${LAUNCH_ARGS[@]}" \
  "$ENTRYPOINT" \
  --data_path "$DATA_PATH" \
  --model_name_or_path "$MODEL_PATH" \
  --trainset math \
  --max_seq_len 512 \
  --learning_rate 5e-6 \
  --weight_decay 0.0 \
  --num_train_epochs 5 \
  --gradient_accumulation_steps 1 \
  --lr_scheduler_type cosine \
  --num_warmup_steps 0 \
  --seed 1234 \
  --dtype fp16 \
  --zero_stage 0 \
  --print_loss \
  --output_dir "$OUTPUT_DIR" \
  --checkpoint_interval 500 \
  --helix_profiles_path "$PROFILE_JSON" \
  --helix_save_plan_path "$OUTPUT_DIR/helix_plan.json" \
  --helix_max_micro_batch_size 16 \
  --helix_memory_slack_gib 0.25 \
  --helix_ffn_alignment 1 \
  --helix_link_bandwidth_gbps 10.0
