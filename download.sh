#!/bin/bash
export HF_HUB_ENABLE_HF_TRANSFER=1
MAX_WORKERS=32
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


# 데이터 셋 다운로드
for dataset in "${datasets[@]}"; do
    echo "📦 Downloading dataset $dataset into HF cache..."
    hf download "$dataset" --repo-type dataset --max-workers "$MAX_WORKERS"
done

echo "✅ All models cached in: ~/.cache/huggingface/hub"
