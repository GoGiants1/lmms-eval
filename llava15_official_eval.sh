#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAVA_DIR="${ROOT_DIR}/LLaVA"
EVAL_DIR="${LLAVA_DIR}/playground/data/eval"
CACHE_DIR="${ROOT_DIR}/.cache/llava_eval"

ACTION=""
BENCHMARKS="all"
DEFAULT_MODEL_PATH="liuhaotian/llava-v1.5-13b"
DEFAULT_MODEL_SCRIPTS="llava_7b.sh,llava_7b_2.sh,llava_7b_3.sh"
MODEL_PATH="${DEFAULT_MODEL_PATH}"
MODEL_PATH_SET=0
MODEL_PATHS=""
MODEL_SCRIPTS=""
MODEL_SCRIPTS_SET=0
MODEL_SCRIPT_TREE_MODE=0
CHECKPOINT_ROOTS_OVERRIDE=""
CHECKPOINT_ROOT_OVERRIDE=""
CHECKPOINT_GLOB_OVERRIDE=""
CHECKPOINT_REQUIRED_FILE_OVERRIDE=""
CHECKPOINT_RANGE_OVERRIDE=""
INCLUDE_PARENT_MODEL_OVERRIDE=""
PARENT_MODEL_SUBDIR_OVERRIDE=""
SORT_MODE_OVERRIDE=""
PRIORITY_FIRST_OVERRIDE=""
MODEL_BASE=""
MODEL_ID_OVERRIDE=""
CONV_MODE="vicuna_v1"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
LLAVA_REVIEW=0
FORCE=0
DOWNLOAD_WORKERS=3
HF_DOWNLOAD_WORKERS=16
FILE_DOWNLOAD_WORKERS=16
CURRENT_MODEL_PATH=""
CURRENT_MODEL_ID=""

declare -a PYTHON_CMD=()
declare -a MODEL_PATH_ARGS=()
declare -a SELECTED_BENCHMARKS=()
declare -a MODEL_LIST=()
declare -a MODEL_ID_LIST=()
declare -A MODEL_PATH_SEEN=()

log() {
  echo "[llava15-eval] $*"
}

die() {
  echo "[llava15-eval][ERROR] $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  ./llava15_official_eval.sh download [options]
  ./llava15_official_eval.sh run [options]
  ./llava15_official_eval.sh all [options]

Actions:
  download   Download datasets/assets needed by selected benchmarks.
  run        Run official LLaVA-1.5 eval code for selected benchmarks.
  all        Run download + run in sequence.

Options:
  --benchmarks LIST   Comma-separated benchmark list.
                      Supported: vqav2,textvqa,mmbench,mmbench_cn,llava_wild,mme
                      Default: all
  --model-path PATH   Single model path or HF id to evaluate.
  --model-paths LIST  Comma-separated model paths/HF ids to evaluate.
  --model-scripts LIST
                      Comma-separated script paths to resolve model list from
                      (parses the "Order:" section with DRY_RUN=1).
                      Default (when --model-path/--model-paths unset):
                      llava_7b.sh,llava_7b_2.sh,llava_7b_3.sh
  --model-script-tree Resolve model list with EVAL_CHECKPOINT_TREE=1.
  --checkpoint-roots LIST
                      Comma-separated CHECKPOINT_ROOTS override for tree mode.
  --checkpoint-root PATH
                      CHECKPOINT_ROOT override for tree mode.
  --checkpoint-glob GLOB
                      CHECKPOINT_GLOB override for tree mode.
  --checkpoint-required-file FILE
                      CHECKPOINT_REQUIRED_FILE override for tree mode.
  --checkpoint-range RANGE
                      CHECKPOINT_RANGE override for tree mode (e.g. 100:5200:200).
  --include-parent-model 0|1
                      INCLUDE_PARENT_MODEL override for tree mode.
  --parent-model-subdir DIR
                      PARENT_MODEL_SUBDIR override for tree mode.
  --sort-mode MODE    SORT_MODE override for tree mode (version|path).
  --priority-first LIST
                      PRIORITY_FIRST override for tree mode.
  --model-base PATH   Optional model base (for LoRA-style loading).
  --model-id NAME     Output name for answer/result folders (single model only).
  --conv-mode NAME    Conversation template name. Default: vicuna_v1
  --gpus LIST         GPU ids for VQAv2 multi-chunk inference. Default: CUDA_VISIBLE_DEVICES or 0
  --llava-review      Also run GPT review step for llava_wild (needs OPENAI_API_KEY).
                      If omitted, llava_wild runs inference-only.
  --download-workers N
                      Number of parallel benchmark download workers. Default: 3
  --hf-download-workers N
                      Max workers for HuggingFace snapshot download. Default: 16
  --file-download-workers N
                      Per-file parallel connections (used when aria2c exists). Default: 16
  --force             Re-download/re-extract files even if target exists.
  -h, --help          Show this help.

Examples:
  ./llava15_official_eval.sh download --benchmarks vqav2,textvqa,mmbench,mmbench_cn,llava_wild,mme
  ./llava15_official_eval.sh run --benchmarks vqav2,textvqa --model-path liuhaotian/llava-v1.5-7b --gpus 0,1
  ./llava15_official_eval.sh run --model-scripts llava_7b.sh,llava_7b_2.sh,llava_7b_3.sh
  ./llava15_official_eval.sh run --model-script-tree --checkpoint-roots /path/runA,/path/runB
  ./llava15_official_eval.sh download --download-workers 4 --hf-download-workers 32
  ./llava15_official_eval.sh all --model-paths /path/a,/path/b
EOF
}

ensure_repo_layout() {
  [[ -d "${LLAVA_DIR}" ]] || die "LLaVA directory not found: ${LLAVA_DIR}"
}

ensure_python() {
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_CMD=("${ROOT_DIR}/.venv/bin/python")
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=("python3")
  elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD=("python")
  elif command -v uv >/dev/null 2>&1; then
    PYTHON_CMD=("uv" "run" "python")
  else
    die "Python runtime not found. Prepare environment with 'uv sync'."
  fi
}

run_from_llava() {
  (
    cd "${LLAVA_DIR}"
    "${PYTHON_CMD[@]}" "$@"
  )
}

extract_zip() {
  local zip_path="$1"
  local dest_dir="$2"
  mkdir -p "${dest_dir}"

  "${PYTHON_CMD[@]}" - "${zip_path}" "${dest_dir}" <<'PY'
import pathlib
import sys
import zipfile

zip_path = pathlib.Path(sys.argv[1])
dest_dir = pathlib.Path(sys.argv[2])

with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(dest_dir)

print(f"Extracted {zip_path} -> {dest_dir}")
PY
}

download_file() {
  local url="$1"
  local output="$2"
  mkdir -p "$(dirname "${output}")"

  if [[ -f "${output}" && "${FORCE}" -eq 0 ]]; then
    log "Skip existing file: ${output}"
    return
  fi

  log "Downloading: ${url}"
  if command -v aria2c >/dev/null 2>&1 && [[ "${FILE_DOWNLOAD_WORKERS}" -gt 1 ]]; then
    aria2c \
      --continue=true \
      --allow-overwrite=true \
      --auto-file-renaming=false \
      --file-allocation=none \
      --max-tries=10 \
      --retry-wait=2 \
      --summary-interval=0 \
      --max-connection-per-server="${FILE_DOWNLOAD_WORKERS}" \
      --split="${FILE_DOWNLOAD_WORKERS}" \
      --min-split-size=1M \
      --dir="$(dirname "${output}")" \
      --out="$(basename "${output}")" \
      "${url}"
  else
    curl -L --fail --retry 5 --retry-delay 2 --continue-at - -o "${output}.part" "${url}"
    mv "${output}.part" "${output}"
  fi
}

normalize_benchmark() {
  local raw="$1"
  case "${raw}" in
    vqav2|vqa_v2|vqa2)
      echo "vqav2"
      ;;
    textvqa|text-vqa)
      echo "textvqa"
      ;;
    mmbench|mmbench_en|mmbench-en)
      echo "mmbench"
      ;;
    mmbench_cn|mmbench-cn|mmbenchcn)
      echo "mmbench_cn"
      ;;
    llava_wild|llava-wild|llava_bench|llava-bench|llava_in_the_wild|llava-in-the-wild|llavabench)
      echo "llava_wild"
      ;;
    mme)
      echo "mme"
      ;;
    *)
      return 1
      ;;
  esac
}

resolve_benchmarks() {
  SELECTED_BENCHMARKS=()
  if [[ "${BENCHMARKS}" == "all" ]]; then
    SELECTED_BENCHMARKS=("vqav2" "textvqa" "mmbench" "mmbench_cn" "llava_wild" "mme")
    return
  fi

  declare -A seen=()
  local token=""
  IFS=',' read -r -a raw_items <<< "${BENCHMARKS}"
  for token in "${raw_items[@]}"; do
    token="${token// /}"
    [[ -z "${token}" ]] && continue
    local normalized=""
    normalized="$(normalize_benchmark "${token}")" || die "Unsupported benchmark: ${token}"
    if [[ -z "${seen[${normalized}]:-}" ]]; then
      SELECTED_BENCHMARKS+=("${normalized}")
      seen["${normalized}"]=1
    fi
  done

  [[ "${#SELECTED_BENCHMARKS[@]}" -gt 0 ]] || die "No valid benchmark selected."
}

trim_whitespace() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

sanitize_model_id() {
  local raw="$1"
  raw="${raw%/}"
  raw="${raw##*/}"
  raw="${raw// /_}"
  raw="${raw//[^A-Za-z0-9._-]/_}"
  [[ -n "${raw}" ]] || raw="model"
  printf '%s' "${raw}"
}

append_model_if_new() {
  local model_path="$1"
  if [[ -z "${MODEL_PATH_SEEN[${model_path}]:-}" ]]; then
    MODEL_LIST+=("${model_path}")
    MODEL_PATH_SEEN["${model_path}"]=1
  fi
}

extract_models_from_script() {
  local script_rel="$1"
  local script_path="${ROOT_DIR}/${script_rel}"
  [[ -f "${script_path}" ]] || die "Model script not found: ${script_rel}"

  local -a env_args=(
    "DRY_RUN=1"
    "USE_WANDB=0"
    "TASK_SET=custom"
    "TASKS=vqav2_test"
  )

  if [[ "${MODEL_SCRIPT_TREE_MODE}" -eq 1 ]]; then
    env_args+=("EVAL_CHECKPOINT_TREE=1" "EVAL_MERGED=0")
    [[ -n "${CHECKPOINT_ROOTS_OVERRIDE}" ]] && env_args+=("CHECKPOINT_ROOTS=${CHECKPOINT_ROOTS_OVERRIDE}")
    [[ -n "${CHECKPOINT_ROOT_OVERRIDE}" ]] && env_args+=("CHECKPOINT_ROOT=${CHECKPOINT_ROOT_OVERRIDE}")
    [[ -n "${CHECKPOINT_GLOB_OVERRIDE}" ]] && env_args+=("CHECKPOINT_GLOB=${CHECKPOINT_GLOB_OVERRIDE}")
    [[ -n "${CHECKPOINT_REQUIRED_FILE_OVERRIDE}" ]] && env_args+=("CHECKPOINT_REQUIRED_FILE=${CHECKPOINT_REQUIRED_FILE_OVERRIDE}")
    [[ -n "${CHECKPOINT_RANGE_OVERRIDE}" ]] && env_args+=("CHECKPOINT_RANGE=${CHECKPOINT_RANGE_OVERRIDE}")
    [[ -n "${INCLUDE_PARENT_MODEL_OVERRIDE}" ]] && env_args+=("INCLUDE_PARENT_MODEL=${INCLUDE_PARENT_MODEL_OVERRIDE}")
    [[ -n "${PARENT_MODEL_SUBDIR_OVERRIDE}" ]] && env_args+=("PARENT_MODEL_SUBDIR=${PARENT_MODEL_SUBDIR_OVERRIDE}")
    [[ -n "${SORT_MODE_OVERRIDE}" ]] && env_args+=("SORT_MODE=${SORT_MODE_OVERRIDE}")
    [[ -n "${PRIORITY_FIRST_OVERRIDE}" ]] && env_args+=("PRIORITY_FIRST=${PRIORITY_FIRST_OVERRIDE}")
  else
    env_args+=("EVAL_CHECKPOINT_TREE=0" "EVAL_MERGED=0")
  fi

  local output=""
  output="$(
    cd "${ROOT_DIR}"
    env "${env_args[@]}" bash "${script_path}" 2>&1
  )" || die "Failed to resolve models from ${script_rel}"

  local line=""
  local parsed_count=0
  while IFS= read -r line; do
    if [[ "${line}" =~ ^[[:space:]]*[0-9]+/[[:space:]]*[0-9]+[[:space:]]+(.+)$ ]]; then
      local model_path=""
      model_path="$(trim_whitespace "${BASH_REMATCH[1]}")"
      [[ -n "${model_path}" ]] || continue
      append_model_if_new "${model_path}"
      parsed_count=$((parsed_count + 1))
    fi
  done <<< "${output}"

  (( parsed_count > 0 )) || die "Could not parse model list from ${script_rel}"
}

resolve_models() {
  MODEL_LIST=()
  MODEL_ID_LIST=()
  MODEL_PATH_SEEN=()

  if [[ -n "${MODEL_PATHS}" ]]; then
    local item=""
    local -a items=()
    IFS=',' read -r -a items <<< "${MODEL_PATHS}"
    for item in "${items[@]}"; do
      item="$(trim_whitespace "${item}")"
      [[ -n "${item}" ]] || continue
      append_model_if_new "${item}"
    done
  elif [[ "${MODEL_PATH_SET}" -eq 1 ]]; then
    append_model_if_new "${MODEL_PATH}"
  else
    local scripts_csv="${MODEL_SCRIPTS}"
    if [[ -z "${scripts_csv}" ]]; then
      scripts_csv="${DEFAULT_MODEL_SCRIPTS}"
    fi

    local -a scripts=()
    local script_item=""
    IFS=',' read -r -a scripts <<< "${scripts_csv}"
    for script_item in "${scripts[@]}"; do
      script_item="$(trim_whitespace "${script_item}")"
      [[ -z "${script_item}" ]] && continue
      if [[ ! -f "${ROOT_DIR}/${script_item}" ]]; then
        if [[ "${MODEL_SCRIPTS_SET}" -eq 1 ]]; then
          die "Requested model script does not exist: ${script_item}"
        fi
        continue
      fi
      log "Resolving model list from ${script_item}"
      extract_models_from_script "${script_item}"
    done

    if [[ "${#MODEL_LIST[@]}" -eq 0 ]]; then
      if [[ "${MODEL_SCRIPTS_SET}" -eq 1 ]]; then
        die "No model could be resolved from --model-scripts=${MODEL_SCRIPTS}"
      fi
      append_model_if_new "${DEFAULT_MODEL_PATH}"
    fi
  fi

  [[ "${#MODEL_LIST[@]}" -gt 0 ]] || die "No model resolved."

  if [[ -n "${MODEL_ID_OVERRIDE}" && "${#MODEL_LIST[@]}" -gt 1 ]]; then
    die "--model-id can only be used with one model. Use --model-path for single-model run."
  fi

  declare -A id_seen=()
  local model_path=""
  for model_path in "${MODEL_LIST[@]}"; do
    local model_id=""
    if [[ -n "${MODEL_ID_OVERRIDE}" ]]; then
      model_id="${MODEL_ID_OVERRIDE}"
    else
      model_id="$(sanitize_model_id "${model_path}")"
    fi

    local unique_id="${model_id}"
    local suffix_idx=1
    while [[ -n "${id_seen[${unique_id}]:-}" ]]; do
      suffix_idx=$((suffix_idx + 1))
      unique_id="${model_id}_${suffix_idx}"
    done
    id_seen["${unique_id}"]=1
    MODEL_ID_LIST+=("${unique_id}")
  done

  log "Resolved ${#MODEL_LIST[@]} model(s)."
  local idx=0
  for idx in "${!MODEL_LIST[@]}"; do
    log "  - ${MODEL_LIST[${idx}]} (id=${MODEL_ID_LIST[${idx}]})"
  done
}

prepare_model_args_for_current() {
  MODEL_PATH_ARGS=(--model-path "${CURRENT_MODEL_PATH}")
  if [[ -n "${MODEL_BASE}" ]]; then
    MODEL_PATH_ARGS+=(--model-base "${MODEL_BASE}")
  fi
}

require_file() {
  local file_path="$1"
  [[ -f "${file_path}" ]] || die "Missing required file: ${file_path}. Run download action first."
}

require_dir() {
  local dir_path="$1"
  [[ -d "${dir_path}" ]] || die "Missing required directory: ${dir_path}. Run download action first."
}

require_positive_int() {
  local opt_name="$1"
  local opt_value="$2"
  if ! [[ "${opt_value}" =~ ^[0-9]+$ ]] || [[ "${opt_value}" -lt 1 ]]; then
    die "${opt_name} must be a positive integer (got: ${opt_value})"
  fi
}

download_eval_base() {
  local eval_zip="${CACHE_DIR}/eval.zip"
  local sentinel="${EVAL_DIR}/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl"

  if [[ -f "${sentinel}" && "${FORCE}" -eq 0 ]]; then
    log "Skip eval.zip extraction (already prepared): ${sentinel}"
    return
  fi

  download_file "https://drive.google.com/uc?export=download&id=1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy" "${eval_zip}"
  mkdir -p "${EVAL_DIR}"
  extract_zip "${eval_zip}" "${EVAL_DIR}"
}

download_vqav2() {
  local test2015_zip="${CACHE_DIR}/vqav2/test2015.zip"
  local image_dir="${EVAL_DIR}/vqav2/test2015"
  if [[ -d "${image_dir}" && "${FORCE}" -eq 0 ]]; then
    log "Skip VQAv2 images (already exists): ${image_dir}"
    return
  fi

  download_file "http://images.cocodataset.org/zips/test2015.zip" "${test2015_zip}"
  mkdir -p "${EVAL_DIR}/vqav2"
  extract_zip "${test2015_zip}" "${EVAL_DIR}/vqav2"
}

download_textvqa() {
  local ann_path="${EVAL_DIR}/textvqa/TextVQA_0.5.1_val.json"
  local image_zip="${CACHE_DIR}/textvqa/train_val_images.zip"
  local image_dir="${EVAL_DIR}/textvqa/train_images"

  download_file "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json" "${ann_path}"

  if [[ -d "${image_dir}" && "${FORCE}" -eq 0 ]]; then
    log "Skip TextVQA images (already exists): ${image_dir}"
    return
  fi

  download_file "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip" "${image_zip}"
  mkdir -p "${EVAL_DIR}/textvqa"
  extract_zip "${image_zip}" "${EVAL_DIR}/textvqa"
}

download_mmbench() {
  download_file \
    "https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv" \
    "${EVAL_DIR}/mmbench/mmbench_dev_20230712.tsv"
}

download_mmbench_cn() {
  download_file \
    "https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv" \
    "${EVAL_DIR}/mmbench_cn/mmbench_dev_cn_20231003.tsv"
}

download_llava_wild() {
  local target_dir="${EVAL_DIR}/llava-bench-in-the-wild"

  if [[ -f "${target_dir}/questions.jsonl" && -d "${target_dir}/images" && "${FORCE}" -eq 0 ]]; then
    log "Skip llava-bench-in-the-wild (already prepared): ${target_dir}"
    return
  fi

  mkdir -p "${target_dir}"
  "${PYTHON_CMD[@]}" - "${target_dir}" "${HF_DOWNLOAD_WORKERS}" <<'PY'
import sys
from huggingface_hub import snapshot_download

target_dir = sys.argv[1]
max_workers = int(sys.argv[2])
snapshot_download(
    repo_id="liuhaotian/llava-bench-in-the-wild",
    repo_type="dataset",
    local_dir=target_dir,
    allow_patterns=["questions.jsonl", "context.jsonl", "answers_gpt4.jsonl", "images/*"],
    max_workers=max_workers,
)
print(f"Downloaded llava-bench-in-the-wild to {target_dir}")
PY
}

download_mme() {
  local mme_zip="${CACHE_DIR}/mme/MME_Benchmark_release_version.zip"
  local mme_eval_zip="${CACHE_DIR}/mme/eval_tool.zip"
  local mme_dir="${EVAL_DIR}/MME"

  if [[ ! -d "${mme_dir}/MME_Benchmark_release_version" || "${FORCE}" -eq 1 ]]; then
    download_file \
      "https://huggingface.co/datasets/darkyarding/MME/resolve/main/MME_Benchmark_release_version.zip" \
      "${mme_zip}"
    mkdir -p "${mme_dir}"
    extract_zip "${mme_zip}" "${mme_dir}"
  else
    log "Skip MME benchmark images (already exists): ${mme_dir}/MME_Benchmark_release_version"
  fi

  if [[ ! -d "${mme_dir}/eval_tool" || "${FORCE}" -eq 1 ]]; then
    download_file \
      "https://raw.githubusercontent.com/BradyFU/Awesome-Multimodal-Large-Language-Models/Evaluation/tools/eval_tool.zip" \
      "${mme_eval_zip}"
    mkdir -p "${mme_dir}"
    extract_zip "${mme_eval_zip}" "${mme_dir}"
  else
    log "Skip MME eval_tool (already exists): ${mme_dir}/eval_tool"
  fi
}

run_vqav2() {
  local split="llava_vqav2_mscoco_test-dev2015"
  require_file "${EVAL_DIR}/vqav2/${split}.jsonl"
  require_file "${EVAL_DIR}/vqav2/llava_vqav2_mscoco_test2015.jsonl"
  require_dir "${EVAL_DIR}/vqav2/test2015"

  local gpu_list="${GPUS// /}"
  [[ -n "${gpu_list}" ]] || die "No GPU specified for VQAv2. Set --gpus or CUDA_VISIBLE_DEVICES."

  local raw_gpu=""
  local -a raw_gpu_arr=()
  local -a gpu_arr=()
  IFS=',' read -r -a raw_gpu_arr <<< "${gpu_list}"
  for raw_gpu in "${raw_gpu_arr[@]}"; do
    [[ -n "${raw_gpu}" ]] && gpu_arr+=("${raw_gpu}")
  done

  local chunks="${#gpu_arr[@]}"
  (( chunks > 0 )) || die "Failed to parse --gpus: ${GPUS}"

  log "Running VQAv2 with ${chunks} chunk(s) on GPU list: ${gpu_list}"
  local -a pids=()
  local idx=0
  for idx in "${!gpu_arr[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpu_arr[${idx}]}" run_from_llava -m llava.eval.model_vqa_loader \
      "${MODEL_PATH_ARGS[@]}" \
      --question-file "./playground/data/eval/vqav2/${split}.jsonl" \
      --image-folder "./playground/data/eval/vqav2/test2015" \
      --answers-file "./playground/data/eval/vqav2/answers/${split}/${CURRENT_MODEL_ID}/${chunks}_${idx}.jsonl" \
      --num-chunks "${chunks}" \
      --chunk-idx "${idx}" \
      --temperature 0 \
      --conv-mode "${CONV_MODE}" &
    pids+=("$!")
  done

  local failed=0
  local pid=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  (( failed == 0 )) || die "VQAv2 inference failed on one or more chunks."

  local merged_file="${EVAL_DIR}/vqav2/answers/${split}/${CURRENT_MODEL_ID}/merge.jsonl"
  : > "${merged_file}"
  for idx in "${!gpu_arr[@]}"; do
    local chunk_file="${EVAL_DIR}/vqav2/answers/${split}/${CURRENT_MODEL_ID}/${chunks}_${idx}.jsonl"
    require_file "${chunk_file}"
    cat "${chunk_file}" >> "${merged_file}"
  done

  run_from_llava scripts/convert_vqav2_for_submission.py --split "${split}" --ckpt "${CURRENT_MODEL_ID}"
  log "VQAv2 done. Submission json: ${EVAL_DIR}/vqav2/answers_upload/${split}/${CURRENT_MODEL_ID}.json"
}

run_textvqa() {
  require_file "${EVAL_DIR}/textvqa/llava_textvqa_val_v051_ocr.jsonl"
  require_file "${EVAL_DIR}/textvqa/TextVQA_0.5.1_val.json"
  require_dir "${EVAL_DIR}/textvqa/train_images"

  run_from_llava -m llava.eval.model_vqa_loader \
    "${MODEL_PATH_ARGS[@]}" \
    --question-file "./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl" \
    --image-folder "./playground/data/eval/textvqa/train_images" \
    --answers-file "./playground/data/eval/textvqa/answers/${CURRENT_MODEL_ID}.jsonl" \
    --temperature 0 \
    --conv-mode "${CONV_MODE}"

  run_from_llava -m llava.eval.eval_textvqa \
    --annotation-file "./playground/data/eval/textvqa/TextVQA_0.5.1_val.json" \
    --result-file "./playground/data/eval/textvqa/answers/${CURRENT_MODEL_ID}.jsonl"

  log "TextVQA done. Answer file: ${EVAL_DIR}/textvqa/answers/${CURRENT_MODEL_ID}.jsonl"
}

run_mmbench() {
  local split="mmbench_dev_20230712"
  require_file "${EVAL_DIR}/mmbench/${split}.tsv"

  run_from_llava -m llava.eval.model_vqa_mmbench \
    "${MODEL_PATH_ARGS[@]}" \
    --question-file "./playground/data/eval/mmbench/${split}.tsv" \
    --answers-file "./playground/data/eval/mmbench/answers/${split}/${CURRENT_MODEL_ID}.jsonl" \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode "${CONV_MODE}"

  run_from_llava scripts/convert_mmbench_for_submission.py \
    --annotation-file "./playground/data/eval/mmbench/${split}.tsv" \
    --result-dir "./playground/data/eval/mmbench/answers/${split}" \
    --upload-dir "./playground/data/eval/mmbench/answers_upload/${split}" \
    --experiment "${CURRENT_MODEL_ID}"

  log "MMBench(en) done. Upload dir: ${EVAL_DIR}/mmbench/answers_upload/${split}"
}

run_mmbench_cn() {
  local split="mmbench_dev_cn_20231003"
  local question_file="./playground/data/eval/mmbench_cn/${split}.tsv"
  local base_dir="./playground/data/eval/mmbench_cn"

  if [[ ! -f "${LLAVA_DIR}/playground/data/eval/mmbench_cn/${split}.tsv" && -f "${LLAVA_DIR}/playground/data/eval/mmbench/${split}.tsv" ]]; then
    question_file="./playground/data/eval/mmbench/${split}.tsv"
    base_dir="./playground/data/eval/mmbench"
  fi
  require_file "${LLAVA_DIR}/${question_file#./}"

  run_from_llava -m llava.eval.model_vqa_mmbench \
    "${MODEL_PATH_ARGS[@]}" \
    --question-file "${question_file}" \
    --answers-file "${base_dir}/answers/${split}/${CURRENT_MODEL_ID}.jsonl" \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode "${CONV_MODE}"

  run_from_llava scripts/convert_mmbench_for_submission.py \
    --annotation-file "${question_file}" \
    --result-dir "${base_dir}/answers/${split}" \
    --upload-dir "${base_dir}/answers_upload/${split}" \
    --experiment "${CURRENT_MODEL_ID}"

  log "MMBench(cn) done. Upload dir: ${LLAVA_DIR}/${base_dir#./}/answers_upload/${split}"
}

run_llava_wild() {
  require_file "${EVAL_DIR}/llava-bench-in-the-wild/questions.jsonl"
  require_file "${EVAL_DIR}/llava-bench-in-the-wild/context.jsonl"
  require_file "${EVAL_DIR}/llava-bench-in-the-wild/answers_gpt4.jsonl"
  require_dir "${EVAL_DIR}/llava-bench-in-the-wild/images"

  run_from_llava -m llava.eval.model_vqa \
    "${MODEL_PATH_ARGS[@]}" \
    --question-file "./playground/data/eval/llava-bench-in-the-wild/questions.jsonl" \
    --image-folder "./playground/data/eval/llava-bench-in-the-wild/images" \
    --answers-file "./playground/data/eval/llava-bench-in-the-wild/answers/${CURRENT_MODEL_ID}.jsonl" \
    --temperature 0 \
    --conv-mode "${CONV_MODE}"

  if [[ "${LLAVA_REVIEW}" -eq 1 ]]; then
    [[ -n "${OPENAI_API_KEY:-}" ]] || die "--llava-review requires OPENAI_API_KEY."

    run_from_llava llava/eval/eval_gpt_review_bench.py \
      --question "./playground/data/eval/llava-bench-in-the-wild/questions.jsonl" \
      --context "./playground/data/eval/llava-bench-in-the-wild/context.jsonl" \
      --rule "./llava/eval/table/rule.json" \
      --answer-list \
        "./playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl" \
        "./playground/data/eval/llava-bench-in-the-wild/answers/${CURRENT_MODEL_ID}.jsonl" \
      --output "./playground/data/eval/llava-bench-in-the-wild/reviews/${CURRENT_MODEL_ID}.jsonl"

    run_from_llava llava/eval/summarize_gpt_review.py \
      -f "./playground/data/eval/llava-bench-in-the-wild/reviews/${CURRENT_MODEL_ID}.jsonl"
    log "LLaVA-in-the-Wild done with GPT review."
  else
    log "LLaVA-in-the-Wild inference done. Re-run with --llava-review to include GPT review."
  fi
}

run_mme() {
  require_file "${EVAL_DIR}/MME/llava_mme.jsonl"
  require_dir "${EVAL_DIR}/MME/MME_Benchmark_release_version"
  require_file "${EVAL_DIR}/MME/convert_answer_to_mme.py"
  require_file "${EVAL_DIR}/MME/eval_tool/calculation.py"

  run_from_llava -m llava.eval.model_vqa_loader \
    "${MODEL_PATH_ARGS[@]}" \
    --question-file "./playground/data/eval/MME/llava_mme.jsonl" \
    --image-folder "./playground/data/eval/MME/MME_Benchmark_release_version" \
    --answers-file "./playground/data/eval/MME/answers/${CURRENT_MODEL_ID}.jsonl" \
    --temperature 0 \
    --conv-mode "${CONV_MODE}"

  (
    cd "${EVAL_DIR}/MME"
    "${PYTHON_CMD[@]}" convert_answer_to_mme.py --experiment "${CURRENT_MODEL_ID}"
    cd eval_tool
    "${PYTHON_CMD[@]}" calculation.py --results_dir "answers/${CURRENT_MODEL_ID}"
  )

  log "MME done. Result dir: ${EVAL_DIR}/MME/eval_tool/answers/${CURRENT_MODEL_ID}"
}

download_one_benchmark() {
  local benchmark=""
  benchmark="$1"
  case "${benchmark}" in
    vqav2)
      download_vqav2
      ;;
    textvqa)
      download_textvqa
      ;;
    mmbench)
      download_mmbench
      ;;
    mmbench_cn)
      download_mmbench_cn
      ;;
    llava_wild)
      download_llava_wild
      ;;
    mme)
      download_mme
      ;;
    *)
      die "Internal error: unknown benchmark ${benchmark}"
      ;;
  esac
}

download_selected() {
  log "Preparing shared eval assets (eval.zip)"
  download_eval_base

  if [[ "${#SELECTED_BENCHMARKS[@]}" -le 1 || "${DOWNLOAD_WORKERS}" -eq 1 ]]; then
    local benchmark=""
    for benchmark in "${SELECTED_BENCHMARKS[@]}"; do
      download_one_benchmark "${benchmark}"
    done
    return
  fi

  log "Downloading benchmark assets in parallel (workers=${DOWNLOAD_WORKERS}, hf_workers=${HF_DOWNLOAD_WORKERS})"
  local running=0
  local failed=0
  local benchmark=""
  for benchmark in "${SELECTED_BENCHMARKS[@]}"; do
    (
      log "Download start: ${benchmark}"
      download_one_benchmark "${benchmark}"
      log "Download done: ${benchmark}"
    ) &
    running=$((running + 1))
    if [[ "${running}" -ge "${DOWNLOAD_WORKERS}" ]]; then
      if ! wait -n; then
        failed=1
      fi
      running=$((running - 1))
    fi
  done

  while [[ "${running}" -gt 0 ]]; do
    if ! wait -n; then
      failed=1
    fi
    running=$((running - 1))
  done

  [[ "${failed}" -eq 0 ]] || die "One or more parallel download jobs failed."
}

run_selected() {
  local benchmark=""
  for benchmark in "${SELECTED_BENCHMARKS[@]}"; do
    log "Running benchmark: ${benchmark}"
    case "${benchmark}" in
      vqav2)
        run_vqav2
        ;;
      textvqa)
        run_textvqa
        ;;
      mmbench)
        run_mmbench
        ;;
      mmbench_cn)
        run_mmbench_cn
        ;;
      llava_wild)
        run_llava_wild
        ;;
      mme)
        run_mme
        ;;
      *)
        die "Internal error: unknown benchmark ${benchmark}"
        ;;
    esac
  done
}

run_all_models() {
  local idx=0
  for idx in "${!MODEL_LIST[@]}"; do
    CURRENT_MODEL_PATH="${MODEL_LIST[${idx}]}"
    CURRENT_MODEL_ID="${MODEL_ID_LIST[${idx}]}"
    prepare_model_args_for_current
    log "Running model [$((idx + 1))/${#MODEL_LIST[@]}]: ${CURRENT_MODEL_PATH} (id=${CURRENT_MODEL_ID})"
    run_selected
  done
}

parse_args() {
  [[ $# -gt 0 ]] || {
    usage
    exit 1
  }

  ACTION="$1"
  shift

  case "${ACTION}" in
    download|run|all)
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown action: ${ACTION}"
      ;;
  esac

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --benchmarks)
        BENCHMARKS="$2"
        shift 2
        ;;
      --model-path)
        MODEL_PATH="$2"
        MODEL_PATH_SET=1
        shift 2
        ;;
      --model-paths)
        MODEL_PATHS="$2"
        shift 2
        ;;
      --model-scripts)
        MODEL_SCRIPTS="$2"
        MODEL_SCRIPTS_SET=1
        shift 2
        ;;
      --model-script-tree)
        MODEL_SCRIPT_TREE_MODE=1
        shift
        ;;
      --checkpoint-roots)
        CHECKPOINT_ROOTS_OVERRIDE="$2"
        shift 2
        ;;
      --checkpoint-root)
        CHECKPOINT_ROOT_OVERRIDE="$2"
        shift 2
        ;;
      --checkpoint-glob)
        CHECKPOINT_GLOB_OVERRIDE="$2"
        shift 2
        ;;
      --checkpoint-required-file)
        CHECKPOINT_REQUIRED_FILE_OVERRIDE="$2"
        shift 2
        ;;
      --checkpoint-range)
        CHECKPOINT_RANGE_OVERRIDE="$2"
        shift 2
        ;;
      --include-parent-model)
        INCLUDE_PARENT_MODEL_OVERRIDE="$2"
        shift 2
        ;;
      --parent-model-subdir)
        PARENT_MODEL_SUBDIR_OVERRIDE="$2"
        shift 2
        ;;
      --sort-mode)
        SORT_MODE_OVERRIDE="$2"
        shift 2
        ;;
      --priority-first)
        PRIORITY_FIRST_OVERRIDE="$2"
        shift 2
        ;;
      --model-base)
        MODEL_BASE="$2"
        shift 2
        ;;
      --model-id)
        MODEL_ID_OVERRIDE="$2"
        shift 2
        ;;
      --conv-mode)
        CONV_MODE="$2"
        shift 2
        ;;
      --gpus)
        GPUS="$2"
        shift 2
        ;;
      --llava-review)
        LLAVA_REVIEW=1
        shift
        ;;
      --download-workers)
        DOWNLOAD_WORKERS="$2"
        shift 2
        ;;
      --hf-download-workers)
        HF_DOWNLOAD_WORKERS="$2"
        shift 2
        ;;
      --file-download-workers)
        FILE_DOWNLOAD_WORKERS="$2"
        shift 2
        ;;
      --force)
        FORCE=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "Unknown option: $1"
        ;;
    esac
  done
}

main() {
  parse_args "$@"
  ensure_repo_layout
  ensure_python
  require_positive_int "--download-workers" "${DOWNLOAD_WORKERS}"
  require_positive_int "--hf-download-workers" "${HF_DOWNLOAD_WORKERS}"
  require_positive_int "--file-download-workers" "${FILE_DOWNLOAD_WORKERS}"
  resolve_benchmarks
  mkdir -p "${CACHE_DIR}"

  if [[ "${ACTION}" == "download" || "${ACTION}" == "all" ]]; then
    download_selected
  fi

  if [[ "${ACTION}" == "run" || "${ACTION}" == "all" ]]; then
    resolve_models
    run_all_models
  fi

  log "Completed action: ${ACTION}"
}

main "$@"
