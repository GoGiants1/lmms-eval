# LLaVA 1.5 Official Eval Wrapper Usage

`llava15_official_eval.sh`는 `LLaVA/docs/Evaluation.md` 기준으로 아래 6개 벤치마크의
1) 데이터 다운로드
2) 원본 LLaVA 평가 코드 실행
을 자동화합니다.

- `vqav2`
- `textvqa`
- `mmbench` (en)
- `mmbench_cn` (cn)
- `llava_wild` (LLaVA-Bench-in-the-Wild)
- `mme`

---

## 1. 기본 사용법

```bash
bash llava15_official_eval.sh download [options]
bash llava15_official_eval.sh run [options]
bash llava15_official_eval.sh all [options]
```

- `download`: 선택 벤치 데이터 다운로드/압축 해제
- `run`: 선택 벤치 평가 실행
- `all`: `download + run`

도움말:

```bash
bash llava15_official_eval.sh --help
```

---

## 2. 다운로드 항목 (eval.zip 외 포함)

스크립트는 `eval.zip` 외에도 벤치별 필수 자산을 다운로드합니다.

- 공통
  - `eval.zip` (Google Drive)
- `vqav2`
  - `test2015.zip` (COCO test2015 이미지)
- `textvqa`
  - `TextVQA_0.5.1_val.json`
  - `train_val_images.zip`
- `mmbench`
  - `mmbench_dev_20230712.tsv`
- `mmbench_cn`
  - `mmbench_dev_cn_20231003.tsv`
- `llava_wild`
  - Hugging Face dataset `liuhaotian/llava-bench-in-the-wild`에서
    `questions.jsonl`, `context.jsonl`, `answers_gpt4.jsonl`, `images/*`
- `mme`
  - `MME_Benchmark_release_version.zip`
  - `eval_tool.zip`

---

## 3. 빠른 시작

### 3.1 데이터만 먼저 다운로드

```bash
bash llava15_official_eval.sh download \
  --benchmarks vqav2,textvqa,mmbench,mmbench_cn,llava_wild,mme
```

대용량 다운로드 가속(병렬 worker):

```bash
bash llava15_official_eval.sh download \
  --benchmarks vqav2,textvqa,mmbench,mmbench_cn,llava_wild,mme \
  --download-workers 4 \
  --hf-download-workers 32 \
  --file-download-workers 16
```

### 3.2 평가 실행 (기본: `llava_7b*.sh` 모델 목록 자동 사용)

```bash
bash llava15_official_eval.sh run \
  --benchmarks vqav2,textvqa,mmbench,mmbench_cn,llava_wild,mme
```

### 3.3 다운로드+평가 한 번에

```bash
bash llava15_official_eval.sh all \
  --benchmarks vqav2,textvqa,mmbench,mmbench_cn,llava_wild,mme
```

---

## 4. 모델 선택 방식

모델 입력 우선순위는 아래와 같습니다.

1. `--model-paths` (쉼표 구분 다중 모델)
2. `--model-path` (단일 모델)
3. 둘 다 없으면 `--model-scripts`로 모델 목록 파싱  
   기본값: `llava_7b.sh,llava_7b_2.sh,llava_7b_3.sh`

### 4.1 기본 모델 목록 (`llava_7b*.sh` 기준)

기본 실행(`--model-path`, `--model-paths` 미지정) 시, 아래 모델들을 대상으로 평가합니다.

- `llava_7b.sh` (5개)
  - `liuhaotian/llava-v1.5-7b`
  - `/mnt/tmp/llava/llava_v1.5_7b_sel_static_r20_s42_merged`
  - `/mnt/tmp/llava/llava_v1.5_7b_sel_static_r40_s42_merged`
  - `/mnt/tmp/llava/llava_v1.5_7b_r20_merged`
  - `/mnt/tmp/llava/llava_v1.5_7b_r40_merged`
- `llava_7b_2.sh` (2개)
  - `/mnt/tmp/llava/llava_v1.5_7b_sel_static_range100_r20_s42_merged`
  - `/mnt/tmp/llava/llava_v1.5_7b_sel_static_range100_r40_s42_merged`
- `llava_7b_3.sh` (2개)
  - `/mnt/tmp/llava/llava_v1.5_7b_sel_static_range200_r20_s42_merged`
  - `/mnt/tmp/llava/llava_v1.5_7b_sel_static_range200_r40_s42_merged`
- joon0822/llava_v1.5_7b_vf191_r100_merged
- joon0822/llava_v1.5_7b_vf191_r40_merged
- joon0822/llava_v1.5_7b_vf191_r20_merged
- joon0822/llava_v1.5_7b_r60_merged
- joon0822/llava_v1.5_7b_r80_merged
- joon0822/llava_v1.5_7b_r100_merged
합계: 기본 9개 모델

참고:
- 위 목록은 `DRY_RUN=1 EVAL_CHECKPOINT_TREE=0 EVAL_MERGED=0` 기준 `Order` 출력에서 확인했습니다.
- 로컬 경로는 각 스크립트의 `MODEL_BASE_PATHS` 기본값(`/mnt/tmp/llava`)에 따라 달라질 수 있습니다.

예시:

```bash
# 단일 모델
bash llava15_official_eval.sh run --benchmarks mme --model-path liuhaotian/llava-v1.5-7b

# 다중 모델
bash llava15_official_eval.sh run --benchmarks mme --model-paths /path/a,/path/b

# 특정 스크립트에서만 모델 목록 파싱
bash llava15_official_eval.sh run --benchmarks mme --model-scripts llava_7b.sh
```

`--model-id`는 단일 모델 실행에서만 사용 가능합니다.

---

## 5. 트리 모드 (checkpoint tree)

`llava_7b*.sh`의 `EVAL_CHECKPOINT_TREE=1` 로직을 사용해 체크포인트 목록을 가져오려면:

```bash
bash llava15_official_eval.sh run \
  --benchmarks mme \
  --model-script-tree \
  --model-scripts llava_7b.sh,llava_7b_2.sh,llava_7b_3.sh \
  --checkpoint-roots /path/runA,/path/runB \
  --checkpoint-range 100:5200:200
```

트리 모드 관련 옵션:

- `--checkpoint-roots`
- `--checkpoint-root`
- `--checkpoint-glob`
- `--checkpoint-required-file`
- `--checkpoint-range`
- `--include-parent-model` (`0|1`)
- `--parent-model-subdir`
- `--sort-mode` (`version|path`)
- `--priority-first`

---

## 6. 주요 실행 옵션

- `--benchmarks vqav2,textvqa,mmbench,mmbench_cn,llava_wild,mme`
- `--model-base` (LoRA 등에서 base 필요 시)
- `--conv-mode` (기본: `vicuna_v1`)
- `--gpus` (벤치마크 추론 청크 분할 GPU 목록)
- `--llava-review` (`llava_wild` GPT 리뷰까지 수행, `OPENAI_API_KEY` 필요)
- `--download-workers` (벤치마크 단위 병렬 다운로드 worker 수, 기본 `3`)
- `--hf-download-workers` (`llava_wild`의 HuggingFace snapshot worker 수, 기본 `16`)
- `--file-download-workers` (파일 단위 병렬 연결 수, `aria2c` 사용 시 적용, 기본 `16`)
- `--force` (기존 파일 있어도 재다운로드/재압축해제)

참고:
- `aria2c`가 설치되어 있으면 `--file-download-workers` 값으로 멀티 커넥션 다운로드를 사용합니다.
- `aria2c`가 없으면 `curl` fallback으로 동작합니다.

---

## 7. 출력 위치

평가 산출물은 주로 아래 경로에 생성됩니다.

- `LLaVA/playground/data/eval/vqav2/answers/...`
- `LLaVA/playground/data/eval/textvqa/answers/...`
- `LLaVA/playground/data/eval/mmbench/answers/...`
- `LLaVA/playground/data/eval/mmbench_cn/answers/...`
- `LLaVA/playground/data/eval/llava-bench-in-the-wild/answers/...`
- `LLaVA/playground/data/eval/MME/answers/...`
- `LLaVA/playground/data/eval/MME/eval_tool/answers/...`

---

## 8. 자주 나는 오류

- `Missing required file/dir ... Run download action first.`
  - `download`를 먼저 수행하세요.
- `--llava-review requires OPENAI_API_KEY`
  - `llava_wild` 리뷰 모드 사용 시 환경변수 설정이 필요합니다.

```
bash llava15_official_eval.sh run \
    --benchmarks vqav2,textvqa \
    --model-paths joon0822/llava_v1.5_7b_vf191_r100_merged,joon0822/llava_v1.5_7b_vf191_r40_merged,joon0822/llava_v1.5_7b_vf191_r20_merged \
    --gpus 0,1,2,3,4,5,6,7
```

```
bash llava15_official_eval.sh run \
    --benchmarks vqav2,textvqa \
    --model-paths joon0822/llava_v1.5_7b_r60_merged,joon0822/llava_v1.5_7b_r80_merged,joon0822/llava_v1.5_7b_r100_merged \
    --gpus 0,1,2,3,4,5,6,7
```
