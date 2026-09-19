#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 PROFILE_JSON QWEN3_MODEL_PATH [MAX_MICRO_BATCH]" >&2
  exit 2
fi

PROFILE_JSON=$1
MODEL_PATH=$2
MAX_MICRO_BATCH=${3:-16}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENTRYPOINT="$SCRIPT_DIR/../helix_profile.py"
MASTER_PORT=${HELIX_MASTER_PORT:-29545}
LAUNCH_ARGS=(--master_port "$MASTER_PORT")
if [[ -n "${HELIX_HOSTFILE:-}" ]]; then
  LAUNCH_ARGS+=(--hostfile "$HELIX_HOSTFILE")
fi
if [[ -n "${HELIX_INCLUDE:-}" ]]; then
  LAUNCH_ARGS+=(--include "$HELIX_INCLUDE")
fi

export PDSH_RCMD_TYPE="${PDSH_RCMD_TYPE:-ssh}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

deepspeed "${LAUNCH_ARGS[@]}" \
  "$ENTRYPOINT" \
  --model_name_or_path "$MODEL_PATH" \
  --output_path "$PROFILE_JSON" \
  --max_seq_len 512 \
  --max_micro_batch_size "$MAX_MICRO_BATCH" \
  --minimum_submodel_size 0.125 \
  --maximum_submodel_size 1.0 \
  --warmup_steps 2 \
  --measure_steps 3 \
  --memory_reserve_gib 1.0 \
  --initialization_reserve_gib 4.0 \
  --optimizer_peak_bytes_per_parameter 20.0 \
  --ffn_alignment 1 \
  --dtype fp16
