#!/usr/bin/env python3
# -*- coding: ascii -*-
import argparse
import csv
import json
import os
import re
import sys
from typing import Any, Dict, Tuple

DATE_RE = re.compile(r"^(\\d{8}_\\d{6})_results\\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate *_results.json into a single CSV."
    )
    parser.add_argument(
        "--root", default="outputs", help="Root directory to search for *_results.json"
    )
    parser.add_argument(
        "--out", default="outputs/llava_7b_evals/summary.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--metrics-out", default=None, help="Output JSON path for metric selection"
    )
    return parser.parse_args()


def parse_date(data: Dict[str, Any], file_path: str) -> int:
    date_val = data.get("date")
    if isinstance(date_val, str) and date_val.strip():
        return int(date_val.replace("_", ""))
    base = os.path.basename(file_path)
    match = DATE_RE.match(base)
    return int(match.group(1).replace("_", "")) if match else 0


def base_checkpoint_name(data: Dict[str, Any], file_path: str) -> str:
    model_name = data.get("model_name")
    if isinstance(model_name, str) and model_name.strip():
        return os.path.basename(model_name)
    model_name_sanitized = data.get("model_name_sanitized")
    if isinstance(model_name_sanitized, str) and model_name_sanitized.strip():
        return model_name_sanitized
    return os.path.basename(os.path.dirname(file_path))


def numeric_keys(result_obj: Dict[str, Any]) -> list:
    keys = []
    for key, val in result_obj.items():
        if key == "alias":
            continue
        if "stderr" in key:
            continue
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            keys.append(key)
    return keys


def pick_metric(
    task: str, result_obj: Dict[str, Any], config_obj: Dict[str, Any]
) -> Tuple[float, str]:
    if result_obj is None:
        return None, None

    # MME total score (perception + cognition)
    if task == "mme":
        perception = result_obj.get("mme_perception_score,none")
        cognition = result_obj.get("mme_cognition_score,none")
        if isinstance(perception, (int, float)) and isinstance(cognition, (int, float)):
            return perception + cognition, "mme_total_score"

    metric_list = (
        config_obj.get("metric_list") if isinstance(config_obj, dict) else None
    )
    if isinstance(metric_list, list):
        for metric in metric_list:
            name = metric.get("metric") if isinstance(metric, dict) else None
            if not name:
                continue
            for key in (f"{name},none", name):
                val = result_obj.get(key)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    return val, key

    for key in ("score,none", "score", "acc,none", "accuracy,none", "exact_match,none"):
        val = result_obj.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return val, key

    nums = numeric_keys(result_obj)
    if len(nums) == 1:
        key = nums[0]
        return result_obj.get(key), key
    if len(nums) > 1:
        key = sorted(nums)[0]
        return result_obj.get(key), key

    return None, None


def main() -> int:
    args = parse_args()
    root = args.root
    out_path = args.out or os.path.join(root, "summary.csv")
    metrics_out = args.metrics_out or os.path.join(
        os.path.dirname(out_path), "summary_metrics.json"
    )

    result_files = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith("_results.json"):
                result_files.append(os.path.join(dirpath, filename))

    if not result_files:
        print(f"No *_results.json files found under {root}", file=sys.stderr)
        return 1

    rows: Dict[str, Dict[str, Dict[str, Any]]] = {}
    metrics_used: Dict[str, str] = {}

    for file_path in result_files:
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            print(f"Skipping unreadable JSON: {file_path}", file=sys.stderr)
            continue

        date_num = parse_date(data, file_path)
        model = base_checkpoint_name(data, file_path)
        results = data.get("results") or {}
        configs = data.get("configs") or {}

        for task, task_results in results.items():
            value, metric_key = pick_metric(task, task_results, configs.get(task, {}))
            if value is None:
                continue
            if task not in metrics_used:
                metrics_used[task] = metric_key

            model_entry = rows.setdefault(model, {"values": {}, "dates": {}})
            prev_date = model_entry["dates"].get(task, 0)
            if date_num >= prev_date:
                model_entry["values"][task] = value
                model_entry["dates"][task] = date_num

    tasks = sorted({task for entry in rows.values() for task in entry["values"].keys()})
    models = sorted(rows.keys())

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["checkpoint"] + tasks + ["total"])
        for model in models:
            entry = rows[model]["values"]
            row_vals = [entry.get(task, "") for task in tasks]
            total = 0.0
            for val in row_vals:
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    total += float(val)
            writer.writerow([model] + row_vals + [total])

    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as handle:
        json.dump({k: metrics_used[k] for k in sorted(metrics_used)}, handle, indent=2)
        handle.write("\n")

    print(f"Wrote CSV: {out_path}")
    print(f"Wrote metrics map: {metrics_out}")
    print(
        f"Models: {len(models)}, Benchmarks: {len(tasks)}, Files: {len(result_files)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
