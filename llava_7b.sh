#!/usr/bin/env bash
set -euo pipefail

ml load cuda/11.8

source .venv/bin/activate

# Set this to a local Hugging Face checkpoint directory (must contain config.json, etc.).
# Example: MODEL_PATH=/data/checkpoints/llava-v1.5-7b
MODEL_PATH=${MODEL_PATH:-liuhaotian/llava-v1.5-7b}

# Bulk-eval mode: evaluate every checkpoint directory containing "_merged" under MERGED_ROOT.
# Example:
#   EVAL_MERGED=1 MERGED_ROOT=/home/joonki/data/projects/joonki/mllm/llava/checkpoints/finetune ./llava_7b.sh
EVAL_MERGED=${EVAL_MERGED:-1}
MERGED_ROOT=${MERGED_ROOT:-/home/joonki/data/projects/joonki/mllm/llava/checkpoints/finetune}

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
# - TASK_SET=all      (table1 + table7)
# - TASK_SET=custom   (use TASKS=...)
TASK_SET=${TASK_SET:-table1}
TASKS=${TASKS:-}

TABLE1_TASKS="mme,scienceqa_img,pope,vqav2,textvqa,mmbench_en,gqa,vizwiz_vqa,mmbench_cn" # llava_in_the_wild => need api key
TABLE7_TASKS="ai2d,chartqa,docvqa,infovqa,naturalbench,realworldqa,cmmmu" # mmvet => need api key


export LMMS_EVAL_USE_CACHE=True
export LMMS_EVAL_HOME="./.lmms_eval_cache"

case "$TASK_SET" in
	table1)
		TASKS="$TABLE1_TASKS"
		;;
	table7)
		TASKS="$TABLE7_TASKS"
		;;
	all)
		TASKS="$TABLE1_TASKS,$TABLE7_TASKS"
		;;
	custom)
		if [[ -z "$TASKS" ]]; then
			echo "TASK_SET=custom requires TASKS=..." >&2
			exit 2
		fi
		;;
	*)
		echo "Unknown TASK_SET: $TASK_SET (expected: table1|table7|all|custom)" >&2
		exit 2
		;;
esac

NUM_PROCESSES=${NUM_PROCESSES:-8}
BATCH_SIZE=${BATCH_SIZE:-1}
OUTPUT_ROOT=${OUTPUT_ROOT:-./outputs/}
LOG_SUFFIX=${LOG_SUFFIX:-llava7b}

run_eval() {
	local model_path="$1"
	local output_path="$2"
	local log_suffix="$3"

	accelerate launch --config_file miscs/llava_acc_default_config.yaml --num_processes="$NUM_PROCESSES" \
		-m lmms_eval \
		--model llava \
		--model_args pretrained="$model_path",device_map=auto \
		--tasks "$TASKS" \
		--batch_size "$BATCH_SIZE" \
		--log_samples \
		--log_samples_suffix "$log_suffix" \
		--output_path "$output_path" \
		--verbosity=DEBUG
}

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
		run_eval "$ckpt_dir" "$out_dir" "$suffix"
	done
	exit 0
fi

# run_eval "$MODEL_PATH" "$OUTPUT_ROOT" "$LOG_SUFFIX"