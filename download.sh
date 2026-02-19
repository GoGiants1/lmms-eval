#!/bin/bash
set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
MAX_WORKERS=32
# 기본값은 시스템/환경 설정을 우선 사용하고, 미설정 시 OS 기본 경로를 사용한다.
HF_DEFAULT_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
if [ -z "${HF_HOME-}" ]; then
    export HF_HOME="$HF_DEFAULT_HOME"
fi
if [ -z "${HF_DATASETS_CACHE-}" ]; then
    export HF_DATASETS_CACHE="$HF_HOME/datasets"
fi
if [ -z "${HF_HUB_CACHE-}" ]; then
    export HF_HUB_CACHE="$HF_HOME/hub"
fi

mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

# 모델 리스트
models=(

)

datasets=(
    # Table 1 tasks
    lmms-lab/MME
    lmms-lab/ScienceQA
    lmms-lab/POPE
    lmms-lab/VQAv2
    lmms-lab/textvqa
    lmms-lab/MMBench
    lmms-lab/GQA
    lmms-lab/VizWiz-VQA
    lmms-lab/llava-bench-in-the-wild

    # Table 7 tasks
    lmms-lab/ai2d
    lmms-lab/ChartQA
    lmms-lab/DocVQA
    lmms-lab/CMMMU
    lmms-lab/MMVet
    BaiqiL/NaturalBench-lmms-eval
    lmms-lab/RealWorldQA

    # Extra tasks
    lmms-lab/COCO-Caption
    lmms-lab/MMMU
    BLINK-Benchmark/BLINK
    MathLLMs/MathVision
    lmms-lab/SEED-Bench
    Lin-Chen/MMStar
    AI4Math/MathVista
    Kyunnilee/amber_g
    lmms-lab/vstar-bench
    echo840/OCRBench
)

# 로그인 체크
if ! huggingface-cli whoami &>/dev/null; then
    echo "🔐 You need to log in to Hugging Face CLI first:"
    echo "Run: huggingface-cli login"
    exit 1
fi


# 다운로드 (캐시 경로에 저장)
for model in "${models[@]}"; do
    echo "📦 Downloading $model into HF cache..."
    hf download "$model" --max-workers "$MAX_WORKERS"
done


# 데이터셋 다운로드 (HF datasets 캐시에 저장)
FAILED_DATASETS=()

for dataset in "${datasets[@]}"; do
    echo "📦 Downloading dataset $dataset into Hugging Face datasets cache..."
    if HF_DATASET_NAME="$dataset" python - <<'PY'
import ast
import os
import re
import requests
import sys
from datasets import load_dataset

dataset_name = os.environ["HF_DATASET_NAME"]
cache_dir = os.getenv("HF_DATASETS_CACHE")


def _load_dataset(config_name=None):
    kwargs = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if config_name is not None:
        kwargs["name"] = config_name
    return load_dataset(dataset_name, **kwargs)


def _config_list_from_error(error_msg: str):
    m = re.search(r"available configs:\s*(\[[^\]]+\])", error_msg)
    if not m:
        return []
    try:
        return ast.literal_eval(m.group(1))
    except Exception:
        return []


def _config_list_from_api(dataset_name: str):
    resp = requests.get(f"https://huggingface.co/api/datasets/{dataset_name}", timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    configs = payload.get("cardData", {}).get("configs", [])
    return [cfg.get("config_name") for cfg in configs if cfg.get("config_name")]


try:
    _load_dataset()
    print(f"✅ Loaded dataset: {dataset_name}")
    sys.exit(0)
except Exception as first_err:
    first_err_msg = str(first_err)
    config_names = _config_list_from_error(first_err_msg)

    if not config_names:
        try:
            config_names = _config_list_from_api(dataset_name)
        except Exception:
            config_names = []

    if not config_names:
        print(f"❌ Failed to download dataset: {dataset_name}", file=sys.stderr)
        print(first_err_msg, file=sys.stderr)
        raise

    failed_configs = []
    for cfg in config_names:
        try:
            _load_dataset(config_name=cfg)
            print(f"✅ Loaded dataset: {dataset_name} ({cfg})")
        except Exception as err:
            failed_configs.append((cfg, str(err)))

    if failed_configs:
        print(f"❌ Failed to download dataset: {dataset_name}", file=sys.stderr)
        for cfg, msg in failed_configs:
            print(f"  - {cfg}: {msg}", file=sys.stderr)
        raise RuntimeError(f"All configs for '{dataset_name}' were not downloaded successfully.")
    print(f"✅ Loaded dataset: {dataset_name} (all resolved configs)")
PY
    then
        : 
    else
        FAILED_DATASETS+=("$dataset")
    fi
done

if ((${#FAILED_DATASETS[@]} > 0)); then
    echo "❌ Failed datasets:"
    for failed_dataset in "${FAILED_DATASETS[@]}"; do
        echo "  - $failed_dataset"
    done
    echo "⚠️  Some datasets failed to download. Check the error logs above."
    exit 1
fi

echo "✅ Done."
