#!/usr/bin/env python3
"""Run end-to-end evaluation for DataAmbigBench.

This script intentionally reads API keys only from environment variables. It
does not contain, load, or expect any hard-coded credentials.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from prompts import SYSTEM_PROMPT, build_e2e_prompt


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

DEFAULT_MODELS = ["gpt-4o-mini", "qwen3.6-flash", "deepseek-v4-flash"]


@dataclass(frozen=True)
class EndpointConfig:
    api_key_env: str
    base_url_env: str
    default_base_url: str
    extra_payload: dict[str, Any]


def dataset_path(name_or_path: str) -> Path:
    aliases = {
        "sec": DATA_DIR / "sec_eval_standard.jsonl",
        "nasa": DATA_DIR / "nasa_eval_standard_v2.jsonl",
        "tatqa": DATA_DIR / "tatqa_eval_standard_v2.jsonl",
        "online_retail": DATA_DIR / "online_retail_eval_standard.jsonl",
        "retail": DATA_DIR / "online_retail_eval_standard.jsonl",
    }
    return aliases.get(name_or_path, Path(name_or_path)).resolve()


def model_endpoint(model: str) -> EndpointConfig:
    lower = model.lower()
    if lower.startswith("qwen"):
        return EndpointConfig(
            api_key_env="DASHSCOPE_API_KEY",
            base_url_env="QWEN_BASE_URL",
            default_base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
            extra_payload={"enable_thinking": False},
        )
    if lower.startswith("deepseek"):
        return EndpointConfig(
            api_key_env="DEEPSEEK_API_KEY",
            base_url_env="DEEPSEEK_BASE_URL",
            default_base_url="https://api.deepseek.com",
            extra_payload={"thinking": {"type": "disabled"}},
        )
    return EndpointConfig(
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        default_base_url="https://api.openai.com/v1",
        extra_payload={},
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def response_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(content)


def parse_json_output(text: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError as first_error:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1]), None
            except json.JSONDecodeError:
                pass
        return None, str(first_error)


def normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("\u2212", "-")
    text = text.replace("%", " percent")
    text = text.replace(",", "")
    text = re.sub(r"\bpercentage\b", "percent", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .;:")


def token_set(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9.\-]+", normalize_text(value)))


def f1_score(predicted: Iterable[str], gold: Iterable[str]) -> dict[str, float]:
    pred_set = {normalize_text(item) for item in predicted if normalize_text(item)}
    gold_set = {normalize_text(item) for item in gold if normalize_text(item)}
    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_set or not gold_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    hits = len(pred_set & gold_set)
    precision = hits / len(pred_set)
    recall = hits / len(gold_set)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def extract_numbers(value: Any) -> list[float]:
    numbers = []
    for raw in re.findall(r"-?\d+(?:\.\d+)?", normalize_text(value)):
        try:
            numbers.append(float(raw))
        except ValueError:
            continue
    return numbers


def numeric_match(predicted: Any, gold: Any, tolerance: float = 1e-3) -> bool:
    pred_numbers = extract_numbers(predicted)
    gold_numbers = extract_numbers(gold)
    if not pred_numbers or not gold_numbers or len(pred_numbers) != len(gold_numbers):
        return False
    return all(math.isclose(p, g, rel_tol=tolerance, abs_tol=tolerance) for p, g in zip(pred_numbers, gold_numbers))


def relaxed_answer_match(predicted: Any, gold: Any) -> bool:
    pred_norm = normalize_text(predicted)
    gold_norm = normalize_text(gold)
    if not pred_norm or not gold_norm:
        return False
    if pred_norm == gold_norm or numeric_match(pred_norm, gold_norm):
        return True
    if len(pred_norm.split()) <= 5 and pred_norm in gold_norm:
        return True
    if len(gold_norm.split()) <= 5 and gold_norm in pred_norm:
        return True
    pred_tokens = token_set(pred_norm)
    gold_tokens = token_set(gold_norm)
    return bool(pred_tokens) and pred_tokens.issubset(gold_tokens)


def gold_dimensions(sample: dict[str, Any]) -> list[str]:
    return [
        item.get("dimension", "")
        for item in sample.get("gold", {}).get("ambiguous_conditions", [])
        if item.get("dimension")
    ]


def predicted_dimensions(parsed: dict[str, Any] | None) -> list[str]:
    if not parsed:
        return []
    return [
        item.get("dimension", "")
        for item in parsed.get("ambiguous_conditions", [])
        if isinstance(item, dict) and item.get("dimension")
    ]


def gold_possible_conditions(sample: dict[str, Any]) -> list[str]:
    values = []
    for item in sample.get("gold", {}).get("ambiguous_conditions", []):
        values.extend(item.get("conditions", []))
    return values


def predicted_possible_conditions(parsed: dict[str, Any] | None) -> list[str]:
    if not parsed:
        return []
    values = []
    for item in parsed.get("ambiguous_conditions", []):
        if isinstance(item, dict):
            values.extend(item.get("possible_conditions", []))
    return values


def condition_values(condition_obj: dict[str, Any]) -> set[str]:
    values = set()
    for key, value in condition_obj.items():
        values.add(normalize_text(key))
        values.add(normalize_text(value))
    return {value for value in values if value}


def condition_matches(gold_condition: dict[str, Any], predicted_condition: dict[str, Any]) -> bool:
    gold_items = {normalize_text(value) for value in gold_condition.values() if normalize_text(value)}
    pred_values = condition_values(predicted_condition)
    return bool(gold_items) and gold_items.issubset(pred_values)


def evaluate_prediction(sample: dict[str, Any], parsed: dict[str, Any] | None, parse_error: str | None) -> dict[str, Any]:
    gold_pairs = sample.get("gold", {}).get("condition_answer_pairs", [])
    predicted_answers = parsed.get("answers", []) if parsed else []
    if not isinstance(predicted_answers, list):
        predicted_answers = []

    dimension = f1_score(predicted_dimensions(parsed), gold_dimensions(sample))
    conditions = f1_score(predicted_possible_conditions(parsed), gold_possible_conditions(sample))

    matched_condition_count = 0
    exact_answer_hits = 0
    relaxed_answer_hits = 0
    matched_pairs = []
    for gold_pair in gold_pairs:
        gold_condition = gold_pair.get("conditions", {})
        gold_answer = gold_pair.get("answer", "")
        match = None
        for predicted in predicted_answers:
            if not isinstance(predicted, dict):
                continue
            predicted_condition = predicted.get("conditions", {})
            if isinstance(predicted_condition, dict) and condition_matches(gold_condition, predicted_condition):
                match = predicted
                break
        if match is not None:
            matched_condition_count += 1
            predicted_answer = match.get("answer", "")
            exact_hit = normalize_text(predicted_answer) == normalize_text(gold_answer)
            relaxed_hit = relaxed_answer_match(predicted_answer, gold_answer)
            exact_answer_hits += int(exact_hit)
            relaxed_answer_hits += int(relaxed_hit)
            matched_pairs.append(
                {
                    "gold_conditions": gold_condition,
                    "predicted_conditions": match.get("conditions", {}),
                    "gold_answer": gold_answer,
                    "predicted_answer": predicted_answer,
                    "exact_match": exact_hit,
                    "relaxed_match": relaxed_hit,
                }
            )
        else:
            matched_pairs.append(
                {
                    "gold_conditions": gold_condition,
                    "predicted_conditions": None,
                    "gold_answer": gold_answer,
                    "predicted_answer": None,
                    "exact_match": False,
                    "relaxed_match": False,
                }
            )

    pair_total = len(gold_pairs)
    safe_total = pair_total or 1
    return {
        "scores": {
            "json_valid": 1.0 if parsed is not None and parse_error is None else 0.0,
            "dimension_precision": dimension["precision"],
            "dimension_recall": dimension["recall"],
            "dimension_f1": dimension["f1"],
            "condition_precision": conditions["precision"],
            "condition_recall": conditions["recall"],
            "condition_f1": conditions["f1"],
            "condition_coverage": matched_condition_count / safe_total,
            "answer_exact_accuracy": exact_answer_hits / safe_total,
            "answer_relaxed_accuracy": relaxed_answer_hits / safe_total,
        },
        "counts": {
            "gold_ambiguous_conditions": len(sample.get("gold", {}).get("ambiguous_conditions", [])),
            "predicted_ambiguous_conditions": len(parsed.get("ambiguous_conditions", [])) if parsed else 0,
            "gold_condition_answer_pairs": pair_total,
            "predicted_answers": len(predicted_answers),
            "matched_condition_pairs": matched_condition_count,
            "exact_answer_hits": exact_answer_hits,
            "relaxed_answer_hits": relaxed_answer_hits,
        },
        "matched_pairs": matched_pairs,
    }


def chat_completion(model: str, system_prompt: str, user_prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    endpoint = model_endpoint(model)
    api_key = os.getenv(endpoint.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key: set {endpoint.api_key_env}.")

    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    body.update(endpoint.extra_payload)

    base_url = os.getenv(endpoint.base_url_env, endpoint.default_base_url).rstrip("/")
    request = urllib.request.Request(
        url=f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    payload["_elapsed_ms"] = round((time.perf_counter() - started) * 1000)
    return payload


def run_with_retries(model: str, sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    prompt = build_e2e_prompt(sample)
    last_error = None
    for attempt in range(args.max_retries + 1):
        try:
            payload = chat_completion(model, SYSTEM_PROMPT, prompt, args)
            text = response_text(payload)
            parsed, parse_error = parse_json_output(text)
            return {
                **sample,
                "model": model,
                "prompt_task": "e2e",
                "status": "ok",
                "elapsed_ms": payload.get("_elapsed_ms"),
                "usage": payload.get("usage"),
                "prediction": {"raw_text": text, "parsed_output": parsed, "parse_error": parse_error},
                "evaluation": evaluate_prediction(sample, parsed, parse_error),
            }
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {details}"
        except urllib.error.URLError as exc:
            last_error = f"network error: {exc.reason}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        if attempt < args.max_retries:
            time.sleep(args.retry_backoff * (attempt + 1))

    return {
        **sample,
        "model": model,
        "prompt_task": "e2e",
        "status": "failed",
        "error": last_error,
        "prediction": {"raw_text": "", "parsed_output": None, "parse_error": last_error},
        "evaluation": evaluate_prediction(sample, None, last_error),
    }


def load_existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "ok":
                keys.add((row.get("model", ""), row.get("id", "")))
    return keys


def summarize_results(path: Path, summary_path: Path) -> dict[str, Any]:
    best_rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row.get("model", ""), row.get("id", ""))
            current = best_rows.get(key)
            row_ok = row.get("status") == "ok"
            current_ok = current and current.get("status") == "ok"
            if current is None or (row_ok and not current_ok) or row_ok == current_ok:
                best_rows[key] = row

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in best_rows.values():
        by_model[row.get("model", "")].append(row)

    score_names = [
        "json_valid",
        "dimension_f1",
        "condition_f1",
        "condition_coverage",
        "answer_exact_accuracy",
        "answer_relaxed_accuracy",
    ]
    summary = {"result_file": str(path), "models": {}}
    for model, rows in sorted(by_model.items()):
        model_summary = {
            "samples": len(rows),
            "ok": sum(1 for row in rows if row.get("status") == "ok"),
            "failed": sum(1 for row in rows if row.get("status") != "ok"),
            "macro_scores": {},
        }
        for name in score_names:
            values = [row.get("evaluation", {}).get("scores", {}).get(name, 0.0) for row in rows]
            model_summary["macro_scores"][name] = sum(values) / len(values) if values else 0.0
        summary["models"][model] = model_summary

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def choose_samples(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.sample_ids:
        wanted = set(args.sample_ids)
        rows = [row for row in rows if row.get("id") in wanted]
    if args.offset:
        rows = rows[args.offset :]
    if args.limit:
        rows = rows[: args.limit]
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="tatqa", help="Dataset alias: sec, nasa, tatqa, online_retail, or a JSONL path.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--sample-ids", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--output-jsonl")
    parser.add_argument("--summary-json")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    path = dataset_path(args.dataset)
    rows = choose_samples(read_jsonl(path), args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = path.stem.replace("_eval_standard_v2", "")
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else output_dir / f"{dataset_name}_e2e_results.jsonl"
    summary_json = Path(args.summary_json) if args.summary_json else output_dir / f"{dataset_name}_e2e_summary.json"

    existing = set() if args.no_resume else load_existing_keys(output_jsonl)
    jobs = [
        (model, sample)
        for sample in rows
        for model in args.models
        if (model, sample.get("id", "")) not in existing
    ]

    print(f"dataset: {path}")
    print(f"samples: {len(rows)}")
    print(f"models: {', '.join(args.models)}")
    print(f"already complete: {len(existing)}")
    print(f"new jobs: {len(jobs)}")
    print(f"output: {output_jsonl}")

    completed = 0
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_with_retries, model, sample, args) for model, sample in jobs]
        with output_jsonl.open("a", encoding="utf-8") as handle:
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                completed += 1
                failed += int(row.get("status") != "ok")
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                scores = row.get("evaluation", {}).get("scores", {})
                print(
                    f"[{completed}/{len(jobs)}] {row.get('model')} {row.get('id')} "
                    f"{row.get('status')} dim_f1={scores.get('dimension_f1', 0):.3f} "
                    f"cond_f1={scores.get('condition_f1', 0):.3f} "
                    f"ans_exact={scores.get('answer_exact_accuracy', 0):.3f}"
                )

    summary = summarize_results(output_jsonl, summary_json)
    print(f"summary: {summary_json}")
    print(json.dumps(summary["models"], ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
