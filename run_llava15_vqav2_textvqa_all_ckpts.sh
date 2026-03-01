#!/usr/bin/env bash
set -euo pipefail


ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${ROOT_DIR}/llava15_official_eval.sh"
ENV_FILE="${ROOT_DIR}/.env"
SCORE_SCRIPT="${ROOT_DIR}/score_llava15_mmbench_textvqa_scienceqa.py"

# llava in the wild
uv pip install openai==0.28.0

BASE_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR:-/mnt/tmp/mllm-data-selection/projects/LLaVA/checkpoints}"

MODEL_BASE="${MODEL_BASE:-lmsys/vicuna-7b-v1.5}"
BENCHMARKS="${BENCHMARKS:-vqav2,textvqa,llava_wild,scienceqa,gqa,mme}" # all => vqav2,textvqa,gqa,mmbench,mmbench_cn,scienceqa,llava_wild,mme
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DOWNLOAD_FIRST="${DOWNLOAD_FIRST:-1}" # 1 => run download action before evaluation
LLAVA_REVIEW="${LLAVA_REVIEW:-1}" # 1 => run llava_wild GPT review (needs OPENAI_API_KEY)
AGGREGATE_LOCAL="${AGGREGATE_LOCAL:-1}" # 1 => run local aggregate scorer after eval
AGGREGATE_BENCHMARKS="${AGGREGATE_BENCHMARKS:-textvqa,scienceqa,llava_wild,gqa,mme}" # benchmarks to aggregate locally (must be subset of BENCHMARKS)
AGGREGATE_OUTPUT_DIR="${AGGREGATE_OUTPUT_DIR:-${ROOT_DIR}/LLaVA/playground/data/eval/aggregates}"

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "[ERROR] Eval script not found: ${EVAL_SCRIPT}" >&2
  exit 1
fi

if [[ ! -d "${BASE_CHECKPOINT_DIR}" ]]; then
  echo "[ERROR] Checkpoint root not found: ${BASE_CHECKPOINT_DIR}" >&2
  exit 1
fi

if [[ -f "${ENV_FILE}" ]]; then
  # Load environment variables (e.g., OPENAI_API_KEY) from project .env
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

# Allow MODEL_PATHS to be set externally (for merged models or manual specification)
if [[ -z "${MODEL_PATHS:-}" ]]; then
  MODEL_PATHS="$(
    find "${BASE_CHECKPOINT_DIR}" -type f -name adapter_config.json -printf '%h\n' \
    | grep -Ei '/[^/]*lora[^/]*/?$' \
    | sort -u \
    | paste -sd, -
  )"

  if [[ -z "${MODEL_PATHS}" ]]; then
    echo "[ERROR] No LoRA checkpoints found under: ${BASE_CHECKPOINT_DIR} (path must contain 'lora')" >&2
    echo "[ERROR] Set MODEL_PATHS environment variable manually for merged models." >&2
    exit 1
  fi
else
  echo "[INFO] Using externally provided MODEL_PATHS" >&2
fi

echo "[INFO] Running benchmarks: ${BENCHMARKS}"
echo "[INFO] Using GPUs: ${GPUS}"
echo "[INFO] Model base: ${MODEL_BASE}"
echo "[INFO] Skip existing results: ${SKIP_EXISTING}"
echo "[INFO] Download first: ${DOWNLOAD_FIRST}"
echo "[INFO] LLaVA review: ${LLAVA_REVIEW}"
echo "[INFO] Local aggregation: ${AGGREGATE_LOCAL}"
if [[ "${LLAVA_REVIEW}" == "1" ]]; then
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    echo "[INFO] OPENAI_API_KEY is set (from environment or .env)."
  else
    echo "[WARN] OPENAI_API_KEY is not set. llava_wild review will fail."
  fi
fi
echo "[INFO] Checkpoint count: $(tr ',' '\n' <<< "${MODEL_PATHS}" | wc -l | tr -d ' ')"

SKIP_ARG=()
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  SKIP_ARG=(--skip-existing)
fi

REVIEW_ARG=()
if [[ "${LLAVA_REVIEW}" == "1" ]]; then
  REVIEW_ARG=(--llava-review)
fi

if [[ "${DOWNLOAD_FIRST}" == "1" ]]; then
  bash "${EVAL_SCRIPT}" download \
    --benchmarks "${BENCHMARKS}"
fi

bash "${EVAL_SCRIPT}" run \
  --benchmarks "${BENCHMARKS}" \
  --model-paths "${MODEL_PATHS}" \
  --model-base "${MODEL_BASE}" \
  --gpus "${GPUS}" \
  "${REVIEW_ARG[@]}" \
  "${SKIP_ARG[@]}"

if [[ "${AGGREGATE_LOCAL}" == "1" ]]; then
  if [[ ! -f "${SCORE_SCRIPT}" ]]; then
    echo "[WARN] Local scorer not found: ${SCORE_SCRIPT}. Skip aggregation."
    exit 0
  fi

  SCORE_PY=()
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    SCORE_PY=("${ROOT_DIR}/.venv/bin/python")
  elif command -v python3 >/dev/null 2>&1; then
    SCORE_PY=("python3")
  elif command -v python >/dev/null 2>&1; then
    SCORE_PY=("python")
  elif command -v uv >/dev/null 2>&1; then
    SCORE_PY=("uv" "run" "python")
  else
    echo "[WARN] Python runtime not found. Skip aggregation."
    exit 0
  fi

  mkdir -p "${AGGREGATE_OUTPUT_DIR}"
  RUN_TS="$(date +%Y%m%d_%H%M%S)"
  AGG_JSON="${AGGREGATE_OUTPUT_DIR}/scores_${RUN_TS}.json"
  AGG_LOG="${AGGREGATE_OUTPUT_DIR}/scores_${RUN_TS}.log"

  echo "[INFO] Running local aggregation: ${AGGREGATE_BENCHMARKS}"
  echo "[INFO] Aggregate JSON: ${AGG_JSON}"
  echo "[INFO] Aggregate LOG:  ${AGG_LOG}"

  "${SCORE_PY[@]}" "${SCORE_SCRIPT}" \
    --benchmarks "${AGGREGATE_BENCHMARKS}" \
    --output-json "${AGG_JSON}" \
    | tee "${AGG_LOG}"
fi
