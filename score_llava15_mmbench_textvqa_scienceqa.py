#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from LLaVA.llava.eval.eval_textvqa import prompt_processor
from LLaVA.llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator

MME_EVAL_TYPE_DICT = {
    "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
    "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"],
}
MME_MAX_SCORE = 200 * sum(len(tasks) for tasks in MME_EVAL_TYPE_DICT.values())


def _is_missing(value: Optional[str]) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return text == "" or text == "nan" or text == "none"


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _iter_model_files(answer_dir: Path, model_ids: Optional[set], exclude_suffixes: Sequence[str] = ()) -> List[Tuple[str, Path]]:
    if not answer_dir.exists():
        return []
    out: List[Tuple[str, Path]] = []
    for path in sorted(answer_dir.glob("*.jsonl")):
        model_id = path.stem
        if any(model_id.endswith(suffix) for suffix in exclude_suffixes):
            continue
        if model_ids is not None and model_id not in model_ids:
            continue
        out.append((model_id, path))
    return out


def _iter_log_files(log_dir: Path, model_ids: Optional[set]) -> List[Tuple[str, Path]]:
    if not log_dir.exists():
        return []
    out: List[Tuple[str, Path]] = []
    for path in sorted(log_dir.glob("*.log")):
        model_id = path.stem
        if model_ids is not None and model_id not in model_ids:
            continue
        out.append((model_id, path))
    return out


@dataclass
class EvalRow:
    benchmark: str
    model_id: str
    score: float
    total: int
    parsed: int
    correct: int
    note: str = ""


def eval_textvqa(annotation_file: Path, answer_dir: Path, model_ids: Optional[set]) -> List[EvalRow]:
    files = _iter_model_files(answer_dir, model_ids)
    if not files:
        return []
    if not annotation_file.exists():
        raise FileNotFoundError(f"TextVQA annotation missing: {annotation_file}")

    annotations = json.loads(annotation_file.read_text(encoding="utf-8"))["data"]
    ann_map: Dict[Tuple[str, str], dict] = {}
    for ann in annotations:
        ann_map[(str(ann["image_id"]), ann["question"].lower())] = ann

    evaluator = TextVQAAccuracyEvaluator()
    rows: List[EvalRow] = []

    for model_id, result_file in files:
        preds = _read_jsonl(result_file)
        pred_list = []
        matched = 0

        for pred in preds:
            qid = str(pred.get("question_id", ""))
            prompt = pred.get("prompt", "")
            text = pred.get("text", "")
            try:
                question_key = prompt_processor(prompt)
            except Exception:
                continue
            ann = ann_map.get((qid, question_key))
            if ann is None:
                continue
            matched += 1
            pred_list.append({"pred_answer": text, "gt_answers": ann["answers"]})

        if not pred_list:
            rows.append(EvalRow("textvqa", model_id, 0.0, len(preds), 0, 0, note="no matched items"))
            continue

        accuracy = 100.0 * evaluator.eval_pred_list(pred_list)
        correct = round((accuracy / 100.0) * len(pred_list))
        note = ""
        if matched != len(preds):
            note = f"matched {matched}/{len(preds)}"
        rows.append(EvalRow("textvqa", model_id, accuracy, len(preds), matched, correct, note=note))
    return rows


def _scienceqa_parse_choice(text: str, options: Sequence[str]) -> Optional[str]:
    text = (text or "").strip()
    if not text:
        return None
    if text in options:
        return text
    if len(text) >= 2 and text[0] in options and text[1] in [".", ")", ":"]:
        return text[0]
    match = re.search(r"\b(?:the answer is|answer is)\s*\(?([A-Z])\)?\b", text, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).upper()
        if candidate in options:
            return candidate
    match = re.search(r"\b([A-Z])\b", text)
    if match:
        candidate = match.group(1).upper()
        if candidate in options:
            return candidate
    return None


def eval_scienceqa(base_dir: Path, answer_dir: Path, model_ids: Optional[set], split: str) -> List[EvalRow]:
    files = _iter_model_files(answer_dir, model_ids, exclude_suffixes=("_output",))
    if not files:
        return []
    pid_splits_file = base_dir / "pid_splits.json"
    problems_file = base_dir / "problems.json"
    if not pid_splits_file.exists() or not problems_file.exists():
        raise FileNotFoundError(f"ScienceQA metadata missing under: {base_dir}")

    pid_splits = json.loads(pid_splits_file.read_text(encoding="utf-8"))
    problems = json.loads(problems_file.read_text(encoding="utf-8"))
    split_ids = {str(x) for x in pid_splits[split]}
    options = ["A", "B", "C", "D", "E"]
    rows: List[EvalRow] = []

    for model_id, result_file in files:
        preds = _read_jsonl(result_file)
        pred_map = {str(x.get("question_id", "")): x for x in preds}

        total = 0
        parsed = 0
        correct = 0
        for qid in split_ids:
            problem = problems.get(qid)
            if problem is None:
                continue
            total += 1
            pred = pred_map.get(qid, {})
            parsed_choice = _scienceqa_parse_choice(pred.get("text", ""), options)
            if parsed_choice is None:
                continue
            pred_idx = options.index(parsed_choice)
            parsed += 1
            if int(problem["answer"]) == pred_idx:
                correct += 1

        score = (100.0 * correct / total) if total else 0.0
        note = ""
        if parsed != total:
            note = f"parsed {parsed}/{total}"
        rows.append(EvalRow("scienceqa", model_id, score, total, parsed, correct, note=note))
    return rows


def _available_options(row: dict) -> List[str]:
    opts: List[str] = []
    for letter in ["A", "B", "C", "D", "E"]:
        if _is_missing(row.get(letter)):
            break
        opts.append(letter)
    return opts


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _parse_mmbench_choice(pred_text: str, row: dict) -> Optional[str]:
    options = _available_options(row)
    if not options:
        return None

    text = (pred_text or "").strip()
    upper = text.upper()

    if upper in options:
        return upper
    if len(upper) >= 2 and upper[0] in options and upper[1] in [".", ")", ":"]:
        return upper[0]

    match = re.search(r"\b(?:the answer is|answer is)\s*\(?([A-E])\)?\b", text, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).upper()
        if candidate in options:
            return candidate

    normalized = _normalize_text(text)
    for letter in options:
        if normalized == _normalize_text(str(row.get(letter, ""))):
            return letter
    return None


def eval_mmbench(annotation_tsv: Path, answer_dir: Path, model_ids: Optional[set], benchmark_name: str = "mmbench") -> List[EvalRow]:
    files = _iter_model_files(answer_dir, model_ids)
    if not files:
        return []
    if not annotation_tsv.exists():
        raise FileNotFoundError(f"MMBench annotation missing: {annotation_tsv}")

    ann_by_qid: Dict[str, dict] = {}
    with annotation_tsv.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            ann_by_qid[str(row["index"])] = row

    rows: List[EvalRow] = []
    for model_id, result_file in files:
        preds = _read_jsonl(result_file)
        pred_map: Dict[str, dict] = {}
        for item in preds:
            qid = str(item.get("question_id", ""))
            if qid and qid not in pred_map:
                pred_map[qid] = item

        total = 0
        parsed = 0
        correct = 0
        for qid, ann in ann_by_qid.items():
            gt = str(ann.get("answer", "")).strip().upper()
            if not gt:
                continue
            total += 1
            pred = pred_map.get(qid)
            if pred is None:
                continue
            choice = _parse_mmbench_choice(pred.get("text", ""), ann)
            if choice is None:
                continue
            parsed += 1
            if choice == gt:
                correct += 1

        score = (100.0 * correct / total) if total else 0.0
        note = "local dev accuracy (not official server score)"
        if parsed != total:
            note += f"; parsed {parsed}/{total}"
        rows.append(EvalRow(benchmark_name, model_id, score, total, parsed, correct, note=note))
    return rows


def _normalize_llava_wild_model_id(model_id: str) -> str:
    return re.sub(r"-eval\d+$", "", model_id)


def _parse_llava_wild_pair(review: dict) -> Optional[Tuple[float, float]]:
    values = review.get("tuple")
    if isinstance(values, (list, tuple)) and len(values) >= 2:
        try:
            return float(values[0]), float(values[1])
        except (TypeError, ValueError):
            return None

    score = review.get("score")
    if isinstance(score, (list, tuple)) and len(score) >= 2:
        try:
            return float(score[0]), float(score[1])
        except (TypeError, ValueError):
            return None

    if isinstance(score, str):
        numbers = re.findall(r"-?\d+(?:\.\d+)?", score)
        if len(numbers) >= 2:
            return float(numbers[0]), float(numbers[1])
    return None


def eval_llava_wild(review_dir: Path, model_ids: Optional[set]) -> List[EvalRow]:
    if not review_dir.exists():
        return []

    files_by_model: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(review_dir.glob("*.jsonl")):
        raw_model_id = path.stem
        normalized_model_id = _normalize_llava_wild_model_id(raw_model_id)
        if model_ids is not None and raw_model_id not in model_ids and normalized_model_id not in model_ids:
            continue
        files_by_model[normalized_model_id].append(path)

    rows: List[EvalRow] = []
    for model_id, review_files in sorted(files_by_model.items()):
        total_rows = 0
        parsed_rows = 0
        ref_sum = 0.0
        model_sum = 0.0

        for review_file in review_files:
            for review in _read_jsonl(review_file):
                total_rows += 1
                pair = _parse_llava_wild_pair(review)
                if pair is None:
                    continue
                ref_score, model_score = pair
                if ref_score <= 0:
                    continue
                parsed_rows += 1
                ref_sum += ref_score
                model_sum += model_score

        if parsed_rows == 0:
            note = "no valid review tuples"
            if total_rows > 0:
                note += f"; parsed 0/{total_rows}"
            rows.append(EvalRow("llava_wild", model_id, 0.0, total_rows, 0, 0, note=note))
            continue

        ref_mean = ref_sum / parsed_rows
        model_mean = model_sum / parsed_rows
        ratio = (100.0 * model_mean / ref_mean) if ref_mean > 0 else 0.0
        note = f"ratio vs GPT-4 review baseline; mean_ref={ref_mean:.3f}, mean_model={model_mean:.3f}"
        if parsed_rows != total_rows:
            note += f"; parsed {parsed_rows}/{total_rows}"
        if len(review_files) > 1:
            note += f"; merged_files={len(review_files)}"
        rows.append(EvalRow("llava_wild", model_id, ratio, total_rows, parsed_rows, 0, note=note))
    return rows


def _extract_percent_like_score(log_text: str, keywords: Sequence[str]) -> Optional[Tuple[float, str]]:
    best: Optional[Tuple[int, int, float, str]] = None  # (priority, order, score, line)
    order = 0

    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if not any(keyword in lower for keyword in keywords):
            continue

        for match in re.finditer(r"(-?\d+(?:\.\d+)?)\s*(%)?", line):
            order += 1
            value = float(match.group(1))
            if value < 0:
                continue
            score = value if value > 1.0 or match.group(2) == "%" else value * 100.0
            if not (0.0 <= score <= 100.0):
                continue

            priority = 2 if "acc" in lower or "accuracy" in lower else 1
            candidate = (priority, order, score, line)
            if best is None or (candidate[0], candidate[1]) > (best[0], best[1]):
                best = candidate

    if best is None:
        return None
    return best[2], best[3]


def _count_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                count += 1
    return count


def _resolve_gqa_answer_file(answer_root: Path, split: str, model_id: str) -> Optional[Path]:
    candidates = [
        answer_root / split / model_id / "merge.jsonl",
        answer_root / split / f"{model_id}.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def eval_gqa(log_dir: Path, answer_root: Path, split: str, model_ids: Optional[set]) -> List[EvalRow]:
    log_files = _iter_log_files(log_dir, model_ids)
    if not log_files:
        return []

    rows: List[EvalRow] = []
    for model_id, log_file in log_files:
        log_text = log_file.read_text(encoding="utf-8", errors="ignore")
        parsed_score = _extract_percent_like_score(log_text, keywords=("acc", "accuracy", "score"))
        answer_file = _resolve_gqa_answer_file(answer_root, split, model_id)
        total = _count_lines(answer_file) if answer_file is not None else 0

        if parsed_score is None:
            note = f"could not parse score from official GQA eval log: {log_file.name}"
            if answer_file is None:
                note += "; answer file missing"
            rows.append(EvalRow("gqa", model_id, 0.0, total, 0, 0, note=note))
            continue

        score, source_line = parsed_score
        parsed = total if total > 0 else 0
        correct = round(total * score / 100.0) if total > 0 else 0
        note = f"official eval log; parsed from: {source_line}"
        if answer_file is not None:
            note += f"; answer_file={answer_file.name}"
        rows.append(EvalRow("gqa", model_id, score, total, parsed, correct, note=note))
    return rows


def _parse_mme_log(log_text: str) -> Optional[Tuple[float, float, float]]:
    perception: Optional[float] = None
    cognition: Optional[float] = None
    current_section: Optional[str] = None

    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if "perception" in lower and "===" in line:
            current_section = "perception"
            continue
        if "cognition" in lower and "===" in line:
            current_section = "cognition"
            continue
        score_match = re.search(r"total score:\s*([0-9]+(?:\.[0-9]+)?)", lower)
        if score_match is None or current_section is None:
            continue
        score_value = float(score_match.group(1))
        if current_section == "perception":
            perception = score_value
        elif current_section == "cognition":
            cognition = score_value

    if perception is None or cognition is None:
        return None
    return perception, cognition, perception + cognition


def eval_mme(log_dir: Path, model_ids: Optional[set]) -> List[EvalRow]:
    log_files = _iter_log_files(log_dir, model_ids)
    if not log_files:
        return []

    rows: List[EvalRow] = []
    for model_id, log_file in log_files:
        parsed = _parse_mme_log(log_file.read_text(encoding="utf-8", errors="ignore"))
        if parsed is None:
            rows.append(EvalRow("mme", model_id, 0.0, MME_MAX_SCORE, 0, 0, note=f"could not parse MME total score from {log_file.name}"))
            continue

        perception, cognition, total_score = parsed
        note = f"official MME score (max {MME_MAX_SCORE}); perception={perception:.1f}, cognition={cognition:.1f}"
        rows.append(EvalRow("mme", model_id, total_score, MME_MAX_SCORE, MME_MAX_SCORE, round(total_score), note=note))
    return rows


def _build_model_id_set(model_id: Optional[str], model_ids_csv: Optional[str]) -> Optional[set]:
    if model_id and model_ids_csv:
        raise ValueError("Use one of --model-id or --model-ids, not both.")
    if model_id:
        return {model_id}
    if model_ids_csv:
        items = [x.strip() for x in model_ids_csv.split(",") if x.strip()]
        return set(items) if items else set()
    return None


def _print_table(rows: Iterable[EvalRow]) -> None:
    rows = list(rows)
    if not rows:
        print("No evaluation rows.")
        return

    print("benchmark\tmodel_id\tscore\tcorrect/total\tparsed\tnote")
    for row in sorted(rows, key=lambda x: (x.benchmark, x.model_id)):
        print(f"{row.benchmark}\t{row.model_id}\t{row.score:.2f}\t{row.correct}/{row.total}\t{row.parsed}/{row.total}\t{row.note}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    eval_root = repo_root / "LLaVA" / "playground" / "data" / "eval"
    parser = argparse.ArgumentParser(description="Local scorer for MMBench/MMBench-CN/TextVQA/ScienceQA/GQA/MME/LLaVA-Wild (VQAv2 excluded).")
    parser.add_argument("--benchmarks", default="mmbench,mmbench_cn,textvqa,scienceqa,llava_wild,gqa,mme", help="Comma-separated benchmarks to score.")
    parser.add_argument("--model-id", default=None, help="Single model id (filename stem).")
    parser.add_argument("--model-ids", default=None, help="Comma-separated model ids.")
    parser.add_argument("--output-json", default=None, help="Optional output JSON file.")
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero status if any benchmark scoring fails.")

    parser.add_argument("--textvqa-annotation", default=str(eval_root / "textvqa" / "TextVQA_0.5.1_val.json"))
    parser.add_argument("--textvqa-answer-dir", default=str(eval_root / "textvqa" / "answers"))

    parser.add_argument("--mmbench-annotation", default=str(eval_root / "mmbench" / "mmbench_dev_20230712.tsv"))
    parser.add_argument("--mmbench-answer-dir", default=str(eval_root / "mmbench" / "answers" / "mmbench_dev_20230712"))
    parser.add_argument("--mmbench-cn-annotation", default=str(eval_root / "mmbench_cn" / "mmbench_dev_cn_20231003.tsv"))
    parser.add_argument("--mmbench-cn-answer-dir", default=str(eval_root / "mmbench_cn" / "answers" / "mmbench_dev_cn_20231003"))

    parser.add_argument("--scienceqa-base-dir", default=str(eval_root / "scienceqa"))
    parser.add_argument("--scienceqa-answer-dir", default=str(eval_root / "scienceqa" / "answers"))
    parser.add_argument("--scienceqa-split", default="test")

    parser.add_argument("--llava-wild-review-dir", default=str(eval_root / "llava-bench-in-the-wild" / "reviews"))
    parser.add_argument("--gqa-log-dir", default=str(eval_root / "gqa" / "eval_logs"))
    parser.add_argument("--gqa-answer-dir", default=str(eval_root / "gqa" / "answers"))
    parser.add_argument("--gqa-split", default="llava_gqa_testdev_balanced")
    parser.add_argument("--mme-log-dir", default=str(eval_root / "MME" / "eval_logs"))
    return parser.parse_args()


def _resolve_mmbench_cn_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    ann = Path(args.mmbench_cn_annotation)
    ans = Path(args.mmbench_cn_answer_dir)

    fallback_ann = Path(args.mmbench_annotation).with_name("mmbench_dev_cn_20231003.tsv")
    fallback_ans = Path(args.mmbench_answer_dir).parent / "mmbench_dev_cn_20231003"

    if not ann.exists() and fallback_ann.exists():
        ann = fallback_ann
    if not ans.exists() and fallback_ans.exists():
        ans = fallback_ans
    return ann, ans


def _normalize_benchmark_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    alias_map = {
        "vqa_v2": "vqav2",
        "vqa2": "vqav2",
        "text_vqa": "textvqa",
        "mmbench_en": "mmbench",
        "science_qa": "scienceqa",
        "sqa": "scienceqa",
        "llava_in_the_wild": "llava_wild",
        "llava_bench": "llava_wild",
        "llava_bench_in_the_wild": "llava_wild",
        "llava_wild": "llava_wild",
        "mme_benchmark": "mme",
    }
    return alias_map.get(normalized, normalized)


def main() -> None:
    args = parse_args()
    model_ids = _build_model_id_set(args.model_id, args.model_ids)
    benchmarks = {_normalize_benchmark_name(x) for x in args.benchmarks.split(",") if x.strip()}

    supported = {"mmbench", "mmbench_cn", "textvqa", "scienceqa", "llava_wild", "gqa", "mme"}
    unknown = benchmarks - supported
    if unknown:
        raise ValueError(f"Unsupported benchmarks: {sorted(unknown)}")

    all_rows: List[EvalRow] = []
    errors: List[str] = []

    if "textvqa" in benchmarks:
        try:
            all_rows.extend(eval_textvqa(Path(args.textvqa_annotation), Path(args.textvqa_answer_dir), model_ids))
        except Exception as exc:
            errors.append(f"textvqa: {exc}")

    if "mmbench" in benchmarks:
        try:
            all_rows.extend(eval_mmbench(Path(args.mmbench_annotation), Path(args.mmbench_answer_dir), model_ids))
        except Exception as exc:
            errors.append(f"mmbench: {exc}")

    if "mmbench_cn" in benchmarks:
        try:
            mmbench_cn_annotation, mmbench_cn_answer_dir = _resolve_mmbench_cn_paths(args)
            all_rows.extend(eval_mmbench(mmbench_cn_annotation, mmbench_cn_answer_dir, model_ids, benchmark_name="mmbench_cn"))
        except Exception as exc:
            errors.append(f"mmbench_cn: {exc}")

    if "scienceqa" in benchmarks:
        try:
            all_rows.extend(
                eval_scienceqa(
                    Path(args.scienceqa_base_dir),
                    Path(args.scienceqa_answer_dir),
                    model_ids,
                    split=args.scienceqa_split,
                )
            )
        except Exception as exc:
            errors.append(f"scienceqa: {exc}")

    if "llava_wild" in benchmarks:
        try:
            all_rows.extend(eval_llava_wild(Path(args.llava_wild_review_dir), model_ids))
        except Exception as exc:
            errors.append(f"llava_wild: {exc}")

    if "gqa" in benchmarks:
        try:
            all_rows.extend(eval_gqa(Path(args.gqa_log_dir), Path(args.gqa_answer_dir), args.gqa_split, model_ids))
        except Exception as exc:
            errors.append(f"gqa: {exc}")

    if "mme" in benchmarks:
        try:
            all_rows.extend(eval_mme(Path(args.mme_log_dir), model_ids))
        except Exception as exc:
            errors.append(f"mme: {exc}")

    _print_table(all_rows)

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([asdict(row) for row in all_rows], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved: {out}")

    if errors:
        print("\nWarnings:")
        for err in errors:
            print(f"- {err}")
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
