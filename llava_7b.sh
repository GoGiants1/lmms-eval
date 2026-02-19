#!/usr/bin/env bash
set -euo pipefail

# FIXME: only for cluster with module system (e.g., SLURM)
# ml load cuda/11.8

source .venv/bin/activate

ENV_FILE="${ENV_FILE:-.env}"

if [ -f "$ENV_FILE" ]; then
  set -o allexport
  source "$ENV_FILE"
  set +o allexport
fi

# Set this to a local Hugging Face checkpoint directory (must contain config.json, etc.).
# Example: MODEL_PATH=/data/checkpoints/llava-v1.5-7b
MODEL_PATH=${MODEL_PATH:-liuhaotian/llava-v1.5-7b}
# Optional: comma-separated list of checkpoints to evaluate.
# Examples:
#   MODEL_PATHS=/data/ckpts/llava-v1.5-7b,/data/ckpts/llava-v1.5-7b-lora
#   MODEL_PATHS=liuhaotian/llava-v1.5-7b,liuhaotian/llava-v1.5-13b

MODEL_BASE_PATHS=${MODEL_BASE_PATHS:-/mnt/tmp/llava}

DEFAULT_MODEL_PATHS=(
	liuhaotian/llava-v1.5-7b
	"$MODEL_BASE_PATHS/llava_v1.5_7b_sel_static_r20_s42_merged"
	"$MODEL_BASE_PATHS/llava_v1.5_7b_sel_static_r40_s42_merged"
	"$MODEL_BASE_PATHS/llava_v1.5_7b_r20_merged"
	"$MODEL_BASE_PATHS/llava_v1.5_7b_r40_merged"
)
MODEL_PATHS=${MODEL_PATHS:-$(IFS=,; echo "${DEFAULT_MODEL_PATHS[*]}")}

# Bulk-eval mode: evaluate every checkpoint directory containing "_merged" under MERGED_ROOT.
# Example:
#   EVAL_MERGED=1 MERGED_ROOT=/home/joonki/data/projects/joonki/mllm/llava/checkpoints/finetune ./llava_7b.sh
EVAL_MERGED=${EVAL_MERGED:-0}
MERGED_ROOT=${MERGED_ROOT:-/mnt/tmp/llava}

# Checkpoint-tree eval mode: evaluate checkpoint-* directories under one or more roots.
# Example:
#   EVAL_CHECKPOINT_TREE=1 CHECKPOINT_ROOTS=/path/runA,/path/runB ./llava_7b.sh
EVAL_CHECKPOINT_TREE=${EVAL_CHECKPOINT_TREE:-1}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-}
DEFAULT_CHECKPOINT_ROOTS=(
	"/mnt/tmp/mllm-data-selection/projects/LLaVA/checkpoints/checkpoints/llava-v1.5-7b-lora-v2"
	# "/mnt/tmp/mllm-data-selection/projects/LLaVA/checkpoints/checkpoints/llava-v1.5-7b-lora-v2-vision-flan"
)
CHECKPOINT_ROOTS=${CHECKPOINT_ROOTS:-$(IFS=,; echo "${DEFAULT_CHECKPOINT_ROOTS[*]}")}
if [[ -n "$CHECKPOINT_ROOT" ]]; then
	CHECKPOINT_ROOTS="$CHECKPOINT_ROOT"
fi
CHECKPOINT_GLOB=${CHECKPOINT_GLOB:-checkpoint-*}
CHECKPOINT_REQUIRED_FILE=${CHECKPOINT_REQUIRED_FILE:-adapter_model.safetensors}

# If set, only prints which checkpoints would be evaluated (and in what order), then exits.
DRY_RUN=${DRY_RUN:-0}

# Ordering for bulk eval list:
# - SORT_MODE=version  (default; natural sort, good for r20,r40,r100)
# - SORT_MODE=path     (pure lexicographic path sort)
SORT_MODE=${SORT_MODE:-version}

# Optional: move matching checkpoints to the front (comma-separated glob patterns).
# Matching is attempted against both the full path and the basename.
# Example:
#   PRIORITY_FIRST='*sel_static_r20_s42_merged,*sel_static_r40_s42_merged'
PRIORITY_FIRST=${PRIORITY_FIRST:-llava_v1.5_7b_sel_static_r20_s42_merged}

# Select task bundle:
# - TASK_SET=table1   (paper Table 1)
# - TASK_SET=table7   (paper Table 7)
# - TASK_SET=all      (table1 + table7 )
# - TASK_SET=extra    (additional benchmarks; mostly val-only where available)
# - TASK_SET=custom   (use TASKS=...)
TASK_SET=${TASK_SET:-all}
TASKS=${TASKS:-}

TABLE1_TASKS="vqav2_test,mme,scienceqa_img,pope,textvqa,mmbench_en,gqa,vizwiz_vqa,mmbench_cn,llava_in_the_wild"
TABLE7_TASKS="ai2d,chartqa,docvqa,infovqa,naturalbench,realworldqa,cmmmu,mmvet,mmmu_val,mathvision_testmini,mmstar,mathvista_testmini"
SINGLE_PROCESS_TASKS="naturalbench"
# Extra set (mapped from internal benchmark nicknames):
# - MMBench_DEV_EN        -> mmbench_en_dev
# - COCO_VAL              -> coco2014_cap_val (captioning)
# - MMMU_DEV_VAL          -> mmmu_val
# - BLINK                 -> blink
# - InfoVQA_VAL           -> infovqa_val
# - MathVision            -> mathvision_testmini
# - SEEDBench_IMG         -> seedbench
# - MMStar                -> mmstar
# - MathVista_MINI        -> mathvista_testmini
# - AMBER                 -> amber_g
# - VStarBench            -> vstar_bench
# - TextVQA_VAL           -> textvqa_val
# - DocVQA_VAL            -> docvqa_val
# - OCRBench              -> ocrbench
EXTRA_TASKS="coco2014_cap_val,blink,infovqa_val,seedbench,amber_g,vstar_bench,ocrbench"


case "$TASK_SET" in
	table1)
		TASKS="$TABLE1_TASKS"
		;;
	table7)
		TASKS="$TABLE7_TASKS"
		;;
	all)
		TASKS="$TABLE1_TASKS,$TABLE7_TASKS,$EXTRA_TASKS"
		;;
	extra)
		TASKS="$EXTRA_TASKS"
		;;
	custom)
		if [[ -z "$TASKS" ]]; then
			echo "TASK_SET=custom requires TASKS=..." >&2
			exit 2
		fi
		;;
	*)
		echo "Unknown TASK_SET: $TASK_SET (expected: table1|table7|all|extra|custom)" >&2
		exit 2
		;;
esac

# Short stable identifier for the exact task list, useful for W&B grouping.
REPO_ROOT=$(dirname "$(realpath "$0")")

NUM_PROCESSES=${NUM_PROCESSES:-}
BATCH_SIZE=${BATCH_SIZE:-1}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$REPO_ROOT/outputs/llava_7b_evals/${TASK_SET}"}
LOG_SUFFIX=${LOG_SUFFIX:-llava7b} # FIXME: customize as needed

# Optional Weights & Biases logging.
# Enable by setting USE_WANDB=1 and having wandb configured (e.g., WANDB_API_KEY).
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-lmms-eval}
WANDB_NAME_PREFIX=${WANDB_NAME_PREFIX:-$LOG_SUFFIX}
WANDB_GROUP=${WANDB_GROUP:-}
WANDB_JOB_TYPE=${WANDB_JOB_TYPE:-eval}
WANDB_NOTES=${WANDB_NOTES:-}

# A stable tag for this script invocation, used to avoid name collisions.
WANDB_RUN_TAG=${WANDB_RUN_TAG:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}

# Per-dataset GPU selection: default to CUDA_VISIBLE_DEVICES, else GPU 0.
GPU_LIST=${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-}}
GPU_LIST="${GPU_LIST// /}"
if [[ -z "$GPU_LIST" ]]; then
	if command -v nvidia-smi >/dev/null 2>&1; then
		gpu_detected_count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
		if [[ "$gpu_detected_count" =~ ^[0-9]+$ ]] && [[ "$gpu_detected_count" -gt 0 ]]; then
			GPU_LIST="$(seq 0 $((gpu_detected_count - 1)) | paste -sd, -)"
		else
			GPU_LIST="0"
		fi
	else
		GPU_LIST="0"
	fi
fi
IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
GPU_COUNT=${#GPU_IDS[@]}

if [[ -z "$NUM_PROCESSES" ]]; then
	NUM_PROCESSES="$GPU_COUNT"
fi
if ! [[ "$NUM_PROCESSES" =~ ^[0-9]+$ ]]; then
	echo "NUM_PROCESSES must be an integer (got: '$NUM_PROCESSES')" >&2
	exit 2
fi
if [[ "$NUM_PROCESSES" -lt 1 ]]; then
	echo "NUM_PROCESSES must be >= 1 (got: '$NUM_PROCESSES')" >&2
	exit 2
fi
if [[ "$NUM_PROCESSES" -gt "$GPU_COUNT" ]]; then
	echo "NUM_PROCESSES=$NUM_PROCESSES > GPU_COUNT=$GPU_COUNT; clamping to $GPU_COUNT." >&2
	NUM_PROCESSES="$GPU_COUNT"
fi

declare -a ACTIVE_GPU_IDS=()
for i in "${!GPU_IDS[@]}"; do
	if [[ "$i" -ge "$NUM_PROCESSES" ]]; then
		break
	fi
	ACTIVE_GPU_IDS+=("${GPU_IDS[$i]}")
done
ACTIVE_GPU_LIST="$(IFS=,; echo "${ACTIVE_GPU_IDS[*]}")"

trim_whitespace() {
	local s="$1"
	s="${s#"${s%%[![:space:]]*}"}"
	s="${s%"${s##*[![:space:]]}"}"
	printf '%s' "$s"
}

model_tag_for_path() {
	local path="${1%/}"
	local tag="${path//\//_}"
	tag="${tag// /_}"
	tag="${tag##_}"
	printf '%s' "$tag"
}

MODEL_PATHS="${MODEL_PATHS//$'\n'/,}"
IFS=',' read -r -a _model_path_list <<< "$MODEL_PATHS"
declare -a MODEL_LIST=()
for raw_path in "${_model_path_list[@]}"; do
	model_path="$(trim_whitespace "$raw_path")"
	[[ -z "$model_path" ]] && continue
	MODEL_LIST+=("$model_path")
done
unset _model_path_list raw_path model_path

if [[ ${#MODEL_LIST[@]} -eq 0 ]]; then
	echo "No model checkpoints specified. Set MODEL_PATH or MODEL_PATHS." >&2
	exit 2
fi

run_eval() {
	local model_path="$1"
	local output_path="$2"
	local log_suffix="$3"
	local task="$4"
	local task_safe
	task_safe="${task//,/|}"
	local wandb_args=()

	if [[ "$USE_WANDB" == "1" ]]; then
		# NOTE: lmms_eval's --wandb_args parser splits on commas, so values must not contain raw commas.
		# When running multiple tasks at once, "$task" is comma-separated; replace commas in identifiers.
		local wandb_name="${WANDB_NAME_PREFIX}-${log_suffix}-${task_safe}-${WANDB_RUN_TAG}"
		local wandb_group="${WANDB_GROUP:-${WANDB_NAME_PREFIX}-${TASK_SET}-${WANDB_RUN_TAG}}"
		# NOTE: lmms_eval's --wandb_args parser splits on commas, so values must not contain raw commas.
		# TASKS is comma-separated by design; convert commas to '|' for logging.
		local tasks_for_notes
		tasks_for_notes="${TASKS//,/|}"
		local auto_notes="task_set=${TASK_SET};task=${task_safe};model_path=${model_path};output_path=${output_path}"
		local wandb_notes="${auto_notes}"
		if [[ -n "$WANDB_NOTES" ]]; then
			local user_notes_sanitized
			user_notes_sanitized="${WANDB_NOTES//,/;}"
			wandb_notes+=";user_notes=${user_notes_sanitized}"
		fi
		# Absolute safety: ensure the final notes string has no raw commas.
		wandb_notes="${wandb_notes//,/;}"
		local wandb_kv="project=${WANDB_PROJECT},name=${wandb_name},group=${wandb_group},job_type=${WANDB_JOB_TYPE}"
		wandb_kv+=",notes=${wandb_notes}"
		wandb_args=(--wandb_args "$wandb_kv")
	fi
	local effective_num_processes="$NUM_PROCESSES"
	local effective_gpu_list="$ACTIVE_GPU_LIST"
	# Run evaluation for a single process(GPU) task
	if [[ " $SINGLE_PROCESS_TASKS " == *" $task "* ]]; then
		effective_num_processes=1
		effective_gpu_list="${ACTIVE_GPU_IDS[0]}"
		echo "    -> Single-process task detected; using NUM_PROCESSES=1 and ACTIVE_GPU_LIST=${effective_gpu_list} for this task" >&2
	fi
	CUDA_VISIBLE_DEVICES="$effective_gpu_list" accelerate launch --config_file miscs/llava_acc_default_config.yaml --num_processes="$effective_num_processes" \
		-m lmms_eval \
		--model llava \
		--model_args pretrained="$model_path",device_map=auto \
		--tasks "$task" \
		--batch_size "$BATCH_SIZE" \
		"${wandb_args[@]}" \
		--log_samples \
		--log_samples_suffix "${log_suffix}-${task_safe}" \
		--output_path "$output_path" \
		--verbosity=DEBUG
}

run_tasks() {
	local model_path="$1"
	local output_path="$2"
	local log_suffix="$3"
	local tasks_csv="$4"
	local -a TASK_LIST
	local task

	IFS=',' read -r -a TASK_LIST <<< "$tasks_csv"
	for j in "${!TASK_LIST[@]}"; do
		task="${TASK_LIST[$j]}"
		echo "  --- Task [$((j + 1))/${#TASK_LIST[@]}]: $task (DP=$NUM_PROCESSES, GPUs=$ACTIVE_GPU_LIST)" >&2
		run_eval "$model_path" "$output_path" "$log_suffix" "$task"
	done
}

if [[ "$EVAL_CHECKPOINT_TREE" == "1" ]]; then
	if [[ "$SORT_MODE" != "version" && "$SORT_MODE" != "path" ]]; then
		echo "Unknown SORT_MODE: $SORT_MODE (expected: version|path)" >&2
		exit 2
	fi

	CHECKPOINT_ROOTS="${CHECKPOINT_ROOTS//$'\n'/,}"
	IFS=',' read -r -a _checkpoint_root_list <<< "$CHECKPOINT_ROOTS"
	declare -a CHECKPOINT_ROOT_LIST=()
	for raw_root in "${_checkpoint_root_list[@]}"; do
		checkpoint_root="$(trim_whitespace "$raw_root")"
		[[ -z "$checkpoint_root" ]] && continue
		CHECKPOINT_ROOT_LIST+=("$checkpoint_root")
	done
	unset _checkpoint_root_list raw_root checkpoint_root

	if [[ ${#CHECKPOINT_ROOT_LIST[@]} -eq 0 ]]; then
		echo "No checkpoint roots specified. Set CHECKPOINT_ROOT or CHECKPOINT_ROOTS." >&2
		exit 2
	fi

	declare -a TREE_CKPTS=()
	declare -a TREE_ROOT_NAMES=()

	echo "Checkpoint-tree mode enabled." >&2
	echo "Tasks: $TASKS" >&2
	echo "GPUs (visible): ${GPU_IDS[*]}" >&2
	echo "GPUs (active):  ${ACTIVE_GPU_IDS[*]} (NUM_PROCESSES=$NUM_PROCESSES)" >&2
	echo "Output root: $OUTPUT_ROOT" >&2
	echo "CHECKPOINT_GLOB: $CHECKPOINT_GLOB" >&2
	if [[ -n "$CHECKPOINT_REQUIRED_FILE" ]]; then
		echo "CHECKPOINT_REQUIRED_FILE: $CHECKPOINT_REQUIRED_FILE" >&2
	fi

	for root_idx in "${!CHECKPOINT_ROOT_LIST[@]}"; do
		checkpoint_root="${CHECKPOINT_ROOT_LIST[$root_idx]}"
		root_name="$(basename "$checkpoint_root")"

		if [[ ! -d "$checkpoint_root" ]]; then
			echo "Checkpoint root is not a directory: $checkpoint_root" >&2
			exit 2
		fi

		if [[ "$SORT_MODE" == "version" ]]; then
			mapfile -d '' ROOT_CKPTS < <(find "$checkpoint_root" -type d -name "$CHECKPOINT_GLOB" -print0 | sort -z -V)
		else
			mapfile -d '' ROOT_CKPTS < <(find "$checkpoint_root" -type d -name "$CHECKPOINT_GLOB" -print0 | sort -z)
		fi

		if [[ -n "$CHECKPOINT_REQUIRED_FILE" ]]; then
			declare -a _filtered_ckpts=()
			for ckpt_dir in "${ROOT_CKPTS[@]}"; do
				if [[ -f "$ckpt_dir/$CHECKPOINT_REQUIRED_FILE" ]]; then
					_filtered_ckpts+=("$ckpt_dir")
				fi
			done
			ROOT_CKPTS=("${_filtered_ckpts[@]}")
			unset _filtered_ckpts
		fi

		if [[ -n "$PRIORITY_FIRST" ]]; then
			IFS=',' read -r -a _priority_patterns <<< "$PRIORITY_FIRST"
			declare -a _prioritized=()
			declare -a _remaining=()

			for ckpt_dir in "${ROOT_CKPTS[@]}"; do
				_remaining+=("$ckpt_dir")
			done

			for pattern in "${_priority_patterns[@]}"; do
				[[ -z "$pattern" ]] && continue
				declare -a _next_remaining=()
				for ckpt_dir in "${_remaining[@]}"; do
					ckpt_base="$(basename "$ckpt_dir")"
					if [[ "$ckpt_dir" == $pattern || "$ckpt_base" == $pattern ]]; then
						_prioritized+=("$ckpt_dir")
					else
						_next_remaining+=("$ckpt_dir")
					fi
				done
				_remaining=("${_next_remaining[@]}")
			done

			ROOT_CKPTS=("${_prioritized[@]}" "${_remaining[@]}")
			unset _priority_patterns _prioritized _remaining _next_remaining
		fi

		echo "Root [$((root_idx + 1))/${#CHECKPOINT_ROOT_LIST[@]}]: $checkpoint_root" >&2
		echo "Order (SORT_MODE=$SORT_MODE):" >&2
		if [[ ${#ROOT_CKPTS[@]} -eq 0 ]]; then
			echo "  (no checkpoints matched)" >&2
			continue
		fi

		for i in "${!ROOT_CKPTS[@]}"; do
			ckpt_dir="${ROOT_CKPTS[$i]}"
			ckpt_name="$(basename "$ckpt_dir")"
			printf '  %3d/%3d  %s\n' "$((i + 1))" "${#ROOT_CKPTS[@]}" "$ckpt_dir" >&2
			echo "         -> output: $OUTPUT_ROOT/$root_name/$ckpt_name/" >&2
			TREE_CKPTS+=("$ckpt_dir")
			TREE_ROOT_NAMES+=("$root_name")
		done
	done

	if [[ ${#TREE_CKPTS[@]} -eq 0 ]]; then
		echo "No checkpoint directories matched CHECKPOINT_GLOB='$CHECKPOINT_GLOB' under CHECKPOINT_ROOTS." >&2
		exit 3
	fi

	echo "Found ${#TREE_CKPTS[@]} total checkpoints across ${#CHECKPOINT_ROOT_LIST[@]} root(s)." >&2

	if [[ "$DRY_RUN" == "1" ]]; then
		echo "DRY_RUN=1 set; exiting without running eval." >&2
		exit 0
	fi

	for i in "${!TREE_CKPTS[@]}"; do
		ckpt_dir="${TREE_CKPTS[$i]}"
		root_name="${TREE_ROOT_NAMES[$i]}"
		ckpt_name="$(basename "$ckpt_dir")"
		out_dir="$OUTPUT_ROOT/$root_name/$ckpt_name/"
		suffix="$LOG_SUFFIX-$root_name-$ckpt_name"
		echo "=== [$((i + 1))/${#TREE_CKPTS[@]}] Evaluating: $ckpt_dir" >&2

		run_tasks "$ckpt_dir" "$out_dir" "$suffix" "$TASKS"
	done
	exit 0
fi

if [[ "$EVAL_MERGED" == "1" ]]; then
	if [[ -z "$MERGED_ROOT" ]]; then
		echo "EVAL_MERGED=1 requires MERGED_ROOT=..." >&2
		exit 2
	fi
	if [[ ! -d "$MERGED_ROOT" ]]; then
		echo "MERGED_ROOT is not a directory: $MERGED_ROOT" >&2
		exit 2
	fi

	if [[ "$SORT_MODE" == "version" ]]; then
		mapfile -d '' MERGED_CKPTS < <(find "$MERGED_ROOT" -type d -name '*_merged*' -print0 | sort -z -V)
	elif [[ "$SORT_MODE" == "path" ]]; then
		mapfile -d '' MERGED_CKPTS < <(find "$MERGED_ROOT" -type d -name '*_merged*' -print0 | sort -z)
	else
		echo "Unknown SORT_MODE: $SORT_MODE (expected: version|path)" >&2
		exit 2
	fi

	# Auto-prioritize checkpoints whose basename contains the pattern r{int}_s{int}.
	# Example: llava_v1.5_7b_sel_static_r20_s42_merged
	# This makes those runs get evaluated first without requiring PRIORITY_FIRST.
	declare -a _rs_first=()
	declare -a _rs_rest=()
	for ckpt_dir in "${MERGED_CKPTS[@]}"; do
		ckpt_base="$(basename "$ckpt_dir")"
		if [[ "$ckpt_base" =~ r[0-9]+_s[0-9]+ ]]; then
			_rs_first+=("$ckpt_dir")
		else
			_rs_rest+=("$ckpt_dir")
		fi
	done
	MERGED_CKPTS=("${_rs_first[@]}" "${_rs_rest[@]}")
	unset _rs_first _rs_rest

	if [[ -n "$PRIORITY_FIRST" ]]; then
		IFS=',' read -r -a _priority_patterns <<< "$PRIORITY_FIRST"
		declare -a _prioritized=()
		declare -a _remaining=()

		for ckpt_dir in "${MERGED_CKPTS[@]}"; do
			_remaining+=("$ckpt_dir")
		done

		for pattern in "${_priority_patterns[@]}"; do
			[[ -z "$pattern" ]] && continue
			declare -a _next_remaining=()
			for ckpt_dir in "${_remaining[@]}"; do
				ckpt_base="$(basename "$ckpt_dir")"
				if [[ "$ckpt_dir" == $pattern || "$ckpt_base" == $pattern ]]; then
					_prioritized+=("$ckpt_dir")
				else
					_next_remaining+=("$ckpt_dir")
				fi
			done
			_remaining=("${_next_remaining[@]}")
		done

		MERGED_CKPTS=("${_prioritized[@]}" "${_remaining[@]}")
		unset _priority_patterns _prioritized _remaining _next_remaining
	fi
	if [[ ${#MERGED_CKPTS[@]} -eq 0 ]]; then
		echo "No '*_merged*' checkpoint directories found under: $MERGED_ROOT" >&2
		exit 3
	fi

	echo "Found ${#MERGED_CKPTS[@]} merged checkpoints under: $MERGED_ROOT" >&2
	echo "Tasks: $TASKS" >&2
	echo "GPUs (visible): ${GPU_IDS[*]}" >&2
	echo "GPUs (active):  ${ACTIVE_GPU_IDS[*]} (NUM_PROCESSES=$NUM_PROCESSES)" >&2
	echo "Output root: $OUTPUT_ROOT" >&2
	echo "Order (SORT_MODE=$SORT_MODE):" >&2
	for i in "${!MERGED_CKPTS[@]}"; do
		ckpt_dir="${MERGED_CKPTS[$i]}"
		ckpt_name="$(basename "$ckpt_dir")"
		printf '  %3d/%3d  %s\n' "$((i + 1))" "${#MERGED_CKPTS[@]}" "$ckpt_dir" >&2
		dir_summary="$OUTPUT_ROOT/$ckpt_name/"
		echo "         -> output: $dir_summary" >&2
	done

	if [[ "$DRY_RUN" == "1" ]]; then
		echo "DRY_RUN=1 set; exiting without running eval." >&2
		exit 0
	fi

	for i in "${!MERGED_CKPTS[@]}"; do
		ckpt_dir="${MERGED_CKPTS[$i]}"
		ckpt_name="$(basename "$ckpt_dir")"
		out_dir="$OUTPUT_ROOT/$ckpt_name/"
		suffix="$LOG_SUFFIX-$ckpt_name"
		echo "=== [$((i + 1))/${#MERGED_CKPTS[@]}] Evaluating: $ckpt_dir" >&2

		run_tasks "$ckpt_dir" "$out_dir" "$suffix" "$TASKS"
	done
	exit 0
fi

echo "Found ${#MODEL_LIST[@]} model checkpoints in MODEL_PATHS." >&2
echo "Tasks: $TASKS" >&2
echo "GPUs (visible): ${GPU_IDS[*]}" >&2
echo "GPUs (active):  ${ACTIVE_GPU_IDS[*]} (NUM_PROCESSES=$NUM_PROCESSES)" >&2
echo "Output root: $OUTPUT_ROOT" >&2
echo "Order:" >&2
for i in "${!MODEL_LIST[@]}"; do
	model_path="${MODEL_LIST[$i]}"
	model_tag="$(model_tag_for_path "$model_path")"
	printf '  %3d/%3d  %s\n' "$((i + 1))" "${#MODEL_LIST[@]}" "$model_path" >&2
	echo "         -> output: $OUTPUT_ROOT/$model_tag/" >&2
done

if [[ "$DRY_RUN" == "1" ]]; then
	echo "DRY_RUN=1 set; exiting without running eval." >&2
	exit 0
fi

for i in "${!MODEL_LIST[@]}"; do
	model_path="${MODEL_LIST[$i]}"
	model_tag="$(model_tag_for_path "$model_path")"
	out_dir="$OUTPUT_ROOT/$model_tag/"
	suffix="$LOG_SUFFIX-$model_tag"
	echo "=== [$((i + 1))/${#MODEL_LIST[@]}] Evaluating: $model_path" >&2

	run_tasks "$model_path" "$out_dir" "$suffix" "$TASKS"
done
