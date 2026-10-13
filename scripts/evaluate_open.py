#!/usr/bin/env python3
"""E2E-open: end-to-end evaluation WITHOUT telling the model that the query is ambiguous.

The prompt only asks the model to answer the question from the data context. The output schema
allows a list of answers, each with the interpretation used, but neither the system prompt nor the
task mentions ambiguity, dimensions, or multiple interpretations.

Metrics (per sample, x100 in summaries):
- multi_answer_rate: model returned >= 2 distinct answers, or its answer text covers >= 2 different gold
  answers (it surfaced more than one reading)
- gold_answer_recall: share of gold condition-answer pairs whose answer appears among the model's
  answers (paper answer matching = normalization + numeric equivalence + aliases;
  Retail: canonical period per gold unit)
- single_answer_correct: model gave exactly one answer and it matches one gold answer
  (i.e., silently picked one valid reading)

Resumable like run_new_eval.py.
"""
from __future__ import annotations

import argparse, concurrent.futures, json, sys, threading, time
from collections import defaultdict
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate as R  # noqa: E402
import prompts as PROMPTS  # noqa: E402
E = R.E

SYSTEM_PROMPT = PROMPTS.OPEN_SYSTEM_PROMPT
build_open_prompt = PROMPTS.build_open_prompt


def score_open(sample, parsed, ds):
    parsed = parsed if isinstance(parsed, dict) else None
    preds = [a for a in ((parsed or {}).get("answers") or []) if isinstance(a, dict)] if parsed else []
    texts = [str(a.get("answer", "")).strip() for a in preds]
    texts = [t for t in texts if t]
    distinct = {E.normalize_text(t) for t in texts}
    pairs = sample["gold"]["condition_answer_pairs"]
    n = len(pairs) or 1
    if ds == "retail":
        # one-to-one: a month answer like "2011-09" would otherwise also count for the year unit
        used, covered = set(), []
        for gi, gp in enumerate(pairs):
            for k, t in enumerate(texts):
                if k not in used and R.ans_relaxed(t, gp, ds):
                    used.add(k); covered.append(gi); break
    else:
        # one answer text may state several readings ("under § 3301 ...; under § 6901 ...")
        covered = [gi for gi, gp in enumerate(pairs) if any(R.ans_relaxed(t, gp, ds) for t in texts)]
    covered_answers = {E.normalize_text(pairs[gi]["answer"]) for gi in covered}
    multi = len(distinct) >= 2 or len(covered_answers) >= 2
    single_ok = (not multi) and len(covered) >= 1
    return {"json_valid": float(parsed is not None),
            "n_answers": float(len(distinct)),
            "multi_answer_rate": float(multi),
            "gold_answer_recall": len(covered) / n,
            "single_answer_correct": float(single_ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dirs", nargs="+", default=[str(HERE.parent / "data")])
    ap.add_argument("--out-dir", default=str(HERE.parent / "results" / "open"))
    ap.add_argument("--models", nargs="+", default=E.DEFAULT_MODELS)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers-per-model", type=int, default=16)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--backoff", type=float, default=3.0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()
    E.key_config.load_env_file(HERE.parent / ".env")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    samples = {}
    for d in args.data_dirs:
        for f in sorted(Path(d).glob("*.jsonl")):
            if args.datasets and f.stem not in args.datasets:
                continue
            rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
            samples[f.stem] = rows[: args.limit] if args.limit else rows

    if args.rescore:
        by = {(ds, r["id"]): r for ds, rows in samples.items() for r in rows}
        for f in out.glob("*_open_results.jsonl"):
            rows = [json.loads(l) for l in open(f) if l.strip()]
            for x in rows:
                r = by.get((x["dataset"], x["id"]))
                if r is not None:
                    x["scores"] = score_open(r, x.get("parsed"), x["dataset"])
            f.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))

    done = set()
    for f in out.glob("*_open_results.jsonl"):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                x = json.loads(line)
                if x.get("status") == "ok":
                    done.add((x["dataset"], x["model"], x["id"]))
    jobs = defaultdict(list)
    for ds, rows in samples.items():
        for r in rows:
            for m in args.models:
                if (ds, m, r["id"]) not in done:
                    jobs[m].append((ds, m, r))
    total = sum(len(v) for v in jobs.values())
    print(f"already done: {len(done)}  to run: {total}", flush=True)

    locks = defaultdict(threading.Lock); clock = threading.Lock(); cnt = {"n": 0, "f": 0}

    def run(job):
        ds, m, r = job
        prompt = build_open_prompt(r)
        res = R.call(m, SYSTEM_PROMPT, prompt, args, args.max_tokens)
        row = {"dataset": ds, "task": "e2e_open", "model": m, "id": r["id"], "query": r["query"],
               "gold": r["gold"], **res, "scores": score_open(r, res.get("parsed"), ds)}
        path = out / f"{ds}_open_results.jsonl"
        with locks[path]:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with clock:
            cnt["n"] += 1; cnt["f"] += res["status"] != "ok"
            if cnt["n"] % 100 == 0 or cnt["n"] == total:
                print(f"  {cnt['n']}/{total} done, failed={cnt['f']}  {time.strftime('%H:%M:%S')}", flush=True)

    if not args.summary_only and not args.rescore and total:
        pools = [concurrent.futures.ThreadPoolExecutor(max_workers=args.workers_per_model) for _ in jobs]
        futs = [p.submit(run, j) for p, (m, js) in zip(pools, jobs.items()) for j in js]
        for f in concurrent.futures.as_completed(futs):
            f.result()

    best = {}
    for f in sorted(out.glob("*_open_results.jsonl")):
        for line in open(f):
            if line.strip():
                x = json.loads(line); k = (x["dataset"], x["model"], x["id"])
                if k not in best or x["status"] == "ok":
                    best[k] = x
    agg = defaultdict(lambda: defaultdict(list)); meta = defaultdict(lambda: defaultdict(int))
    for k, x in best.items():
        for s, v in x["scores"].items():
            agg[k[:2]][s].append(v)
        meta[k[:2]]["n"] += 1; meta[k[:2]]["failed"] += x["status"] != "ok"
        meta[k[:2]]["truncated"] += x.get("finish_reason") == "length"
    rows = [{"dataset": g[0], "model": g[1], **meta[g],
             **{s: round(mean(v), 3) if s == "n_answers" else round(100 * mean(v), 2) for s, v in agg[g].items()}}
            for g in sorted(agg)]
    (out / "summary_open.json").write_text(json.dumps(rows, indent=2))
    cols = list(rows[0].keys()) if rows else []
    with open(out / "summary_open.csv", "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    for r in rows:
        print(" ".join(f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
