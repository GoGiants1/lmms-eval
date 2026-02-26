#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${ROOT_DIR}/llava15_official_eval.sh"

BASE_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR:-/mnt/tmp/mllm-data-selection/projects/LLaVA/checkpoints/llava-v1.5-7b-lora-v2-vision-flan-selected-t050-range0}"
MODEL_BASE="${MODEL_BASE:-lmsys/vicuna-7b-v1.5}"
BENCHMARKS="${BENCHMARKS:-all}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "[ERROR] Eval script not found: ${EVAL_SCRIPT}" >&2
  exit 1
fi

if [[ ! -d "${BASE_CHECKPOINT_DIR}" ]]; then
  echo "[ERROR] Checkpoint root not found: ${BASE_CHECKPOINT_DIR}" >&2
  exit 1
fi

MODEL_PATHS="$(
  find "${BASE_CHECKPOINT_DIR}" -type f -name adapter_config.json -printf '%h\n' \
  | grep -Ei '/[^/]*lora[^/]*/?$' \
  | sort -u \
  | paste -sd, -
)"

if [[ -z "${MODEL_PATHS}" ]]; then
  echo "[ERROR] No LoRA checkpoints found under: ${BASE_CHECKPOINT_DIR} (path must contain 'lora')" >&2
  exit 1
fi

echo "[INFO] Running benchmarks: ${BENCHMARKS}"
echo "[INFO] Using GPUs: ${GPUS}"
echo "[INFO] Model base: ${MODEL_BASE}"
echo "[INFO] Checkpoint count: $(tr ',' '\n' <<< "${MODEL_PATHS}" | wc -l | tr -d ' ')"

bash "${EVAL_SCRIPT}" run \
  --benchmarks "${BENCHMARKS}" \
  --model-paths "${MODEL_PATHS}" \
  --model-base "${MODEL_BASE}" \
  --gpus "${GPUS}"
