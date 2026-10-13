#!/usr/bin/env python3
"""Shared utilities: API calls, JSON parsing, text normalization and paper answer matching."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import config as key_config  # noqa: E402


DEFAULT_MODELS = ["qwen3.6-flash", "deepseek-v4-flash", "gpt-4o-mini"]
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


@dataclass(frozen=True)
class EndpointConfig:
    api_key_env: str
    api_key_fallback: str
    base_url_env: str
    default_base_url: str
    extra_payload: dict[str, Any]


def safe_get_table(sample: dict[str, Any]) -> str:
    table = sample.get("table")
    if isinstance(table, dict):
        return str(table.get("content", table))
    return str(table)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def model_endpoint(model: str) -> EndpointConfig:
    lower = model.lower()
    if lower.startswith("qwen"):
        return EndpointConfig(
            api_key_env="DASHSCOPE_API_KEY",
            api_key_fallback=key_config.QWEN_API_KEY,
            base_url_env="QWEN_BASE_URL",
            default_base_url=key_config.DEFAULT_QWEN_BASE_URL,
            extra_payload={"enable_thinking": False},
        )
    if lower.startswith("deepseek"):
        return EndpointConfig(
            api_key_env="DEEPSEEK_API_KEY",
            api_key_fallback=key_config.DEEPSEEK_API_KEY,
            base_url_env="DEEPSEEK_BASE_URL",
            default_base_url=key_config.DEFAULT_DEEPSEEK_BASE_URL,
            extra_payload={"thinking": {"type": "disabled"}},
        )
    return EndpointConfig(
        api_key_env="OPENAI_API_KEY",
        api_key_fallback=getattr(key_config, "OPENAI_API_KEY", ""),
        base_url_env="OPENAI_BASE_URL",
        default_base_url=DEFAULT_OPENAI_BASE_URL,
        extra_payload={},
    )


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
    if pred_norm == gold_norm:
        return True
    if numeric_match(pred_norm, gold_norm):
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
    possible_conditions = f1_score(predicted_possible_conditions(parsed), gold_possible_conditions(sample))

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
    scores = {
        "json_valid": 1.0 if parsed is not None and parse_error is None else 0.0,
        "dimension_precision": dimension["precision"],
        "dimension_recall": dimension["recall"],
        "dimension_f1": dimension["f1"],
        "condition_precision": possible_conditions["precision"],
        "condition_recall": possible_conditions["recall"],
        "condition_f1": possible_conditions["f1"],
        "condition_coverage": matched_condition_count / safe_total,
        "answer_exact_accuracy": exact_answer_hits / safe_total,
        "answer_relaxed_accuracy": relaxed_answer_hits / safe_total,
    }
    return {
        "scores": scores,
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


def chat_completion(
    model: str,
    system_prompt: str,
    user_prompt: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    endpoint = model_endpoint(model)
    api_key = os.getenv(endpoint.api_key_env) or endpoint.api_key_fallback
    if not api_key:
        raise RuntimeError(f"missing API key: set {endpoint.api_key_env}")

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
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    payload["_elapsed_ms"] = round((time.perf_counter() - started) * 1000)
    return payload
