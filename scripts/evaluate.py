#!/usr/bin/env python3
"""DataAmbigBench evaluation: Detection, Condition-Given, and (hinted) End-to-End.

Prompts: scripts/prompts.py. Data: data/<subset>.jsonl.
Scoring follows the paper: text normalization + numeric equivalence (exact / relaxed),
extended only to accept the per-pair `acceptable_answers` aliases provided in the data.
A token-overlap "soft" score is also stored as a diagnostic (not a paper metric).

Resumable: rerunning skips jobs already finished with status ok.
"""
from __future__ import annotations

import argparse, concurrent.futures, json, re, sys, threading, time, urllib.error
from collections import defaultdict
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as E  # noqa: E402
import prompts as PROMPTS  # noqa: E402

DATASETS = ["sec", "nasa", "tatqa", "retail", "ambigdocs", "opinionqa", "usc", "sf311"]
TASKS = ["detection", "condition_given", "e2e"]


# ---------------------------------------------------------------- data / prompts
render_table = PROMPTS.safe_get_table


def load_prompts():
    return (PROMPTS.SYSTEM_PROMPT, PROMPTS.build_detection_prompt,
            PROMPTS.build_condition_answer_prompt, PROMPTS.build_e2e_prompt)


# ---------------------------------------------------------------- rule-based normalization (deterministic, no model)
import datetime as _dt, math as _math
COND_STOP = set("by the a an of in on for to and or under per apply applies applying definition defined provision "
                "section sec group respondents respondent only use using interpretation".split())
SYN = {"daily": "day", "weekly": "week", "monthly": "month", "quarterly": "quarter", "yearly": "year", "annual": "year",
       "annually": "year", "calendar": "", "iso": ""}
OPINION_AXES = ("AGE", "EDUCATION", "POLPARTY", "INCOME", "SEX")

def cond_tokens(text, ds, term=""):
    t = str(text)
    if ds == "opinionqa":
        t = re.sub(r"^\s*(%s)\s*=\s*" % "|".join(OPINION_AXES), "", t, flags=re.I)
    t = t.replace("–", "-").replace("—", "-").replace("!=", " not ").replace("≠", " not ")
    t = re.sub(r"\bsection\b", "§", t, flags=re.I)
    t = t.lower()
    t = re.sub(r"(?<=[a-z])\.(?=[a-z])", "", t).replace("u.s.c.", "usc").replace("usc.", "usc")
    toks = re.findall(r"[a-z0-9§\-]+", t)
    out = []
    for w in toks:
        w = SYN.get(w, w)
        if not w or w in COND_STOP:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    term_t = set(re.findall(r"[a-z0-9\-]+", str(term).lower()))
    core = [w for w in out if w not in term_t]
    return set(core or out)

def cond_match_norm(gold, pred, ds, term=""):
    g, p = cond_tokens(gold, ds, term), cond_tokens(pred, ds, term)
    if not g or not p:
        return 0.0
    if ds == "sf311":  # status fields/sets must match exactly (STATUS vs WORK STATUS; {a,b} vs {b})
        g2, p2 = g - {"status", "statu"}, p - {"status", "statu"}
        return 1.0 if g2 and g2 == p2 else 0.0
    if ds == "usc":  # the section identifier must appear in the prediction
        sec = sorted((w for w in g if re.search(r"\d", w)), key=len)
        if not sec or sec[-1] not in p:
            return 0.0
    if g <= p or (p <= g and len(p) >= _math.ceil(len(g) / 2)):
        return len(g & p) / len(g | p) + 1e-9
    return 0.0

def assign(golds, preds, ds, term):
    """greedy one-to-one assignment; returns {gold_index: pred_index}"""
    cand = sorted(((cond_match_norm(g, p, ds, term), gi, pi) for gi, g in enumerate(golds) for pi, p in enumerate(preds)), reverse=True)
    used_g, used_p, out = set(), set(), {}
    for sc, gi, pi in cand:
        if sc > 0 and gi not in used_g and pi not in used_p:
            out[gi] = pi; used_g.add(gi); used_p.add(pi)
    return out

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}

def canon_period(text, unit):
    t = str(text).lower().replace("–", "-")
    dates = re.findall(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    try:
        if unit == "by day":
            return "%s-%02d-%02d" % (dates[0][0], int(dates[0][1]), int(dates[0][2])) if len(set(dates)) == 1 else None
        if unit == "by week":
            m = re.search(r"(\d{4})\s*-?\s*w(?:eek)?\s*(\d{1,2})\b", t) or re.search(r"\bw(?:eek)?\s*(\d{1,2})\D{1,8}(\d{4})", t)
            if m:
                a, b = m.groups()
                y, w = (a, b) if len(a) == 4 else (b, a)
                return "%s-W%02d" % (y, int(w))
            if dates:
                y, w, _ = _dt.date(int(dates[0][0]), int(dates[0][1]), int(dates[0][2])).isocalendar()
                return "%d-W%02d" % (y, w)
            return None
        if unit == "by month":
            m = re.search(r"\b(\d{4})-(\d{1,2})\b(?!-)", t)
            if m:
                return "%s-%02d" % (m.group(1), int(m.group(2)))
            m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*,?\s*(\d{4})", t)
            return "%s-%02d" % (m.group(2), MONTHS[m.group(1)]) if m else None
        if unit == "by quarter":
            m = re.search(r"(\d{4})\s*-?\s*q([1-4])", t) or re.search(r"q([1-4])\s*,?\s*(?:of\s*)?(\d{4})", t)
            if not m:
                return None
            a, b = m.groups()
            return ("%sQ%s" % (a, b)) if len(a) == 4 else ("%sQ%s" % (b, a))
        if unit == "by year":
            ys = set(re.findall(r"\b(?:19|20)\d{2}\b", t))
            return ys.pop() if len(ys) == 1 else None
    except Exception:  # noqa: BLE001
        return None
    return None

# ---------------------------------------------------------------- scoring
def golds_for(pair):
    out = [pair.get("answer", "")]
    for a in pair.get("acceptable_answers", []) or []:
        if a not in out:
            out.append(a)
    return out


def _unit(pair):
    return pair.get("condition") or next(iter(pair.get("conditions", {}).values()), "")


def _sci_equal(a, b):
    """numeric equivalence for plain/scientific notation, e.g. 7.6762e+07 == 76762000.0"""
    def f(x):
        t = re.sub(r"[,$\s]", "", str(x))
        return float(t) if re.fullmatch(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", t) else None
    x, y = f(a), f(b)
    return x is not None and y is not None and _math.isclose(x, y, rel_tol=1e-3, abs_tol=1e-3)


def ans_exact(pred, pair, ds=""):
    if ds == "retail":
        g = canon_period(pair.get("answer", ""), _unit(pair))
        return g is not None and canon_period(pred, _unit(pair)) == g
    return any(E.normalize_text(pred) == E.normalize_text(g) for g in golds_for(pair))


def ans_relaxed(pred, pair, ds=""):
    if ds == "retail":  # periods: numeric tolerance would equate 2010 and 2011, so use canonical period match
        return ans_exact(pred, pair, ds)
    return any(E.relaxed_answer_match(pred, g) or _sci_equal(pred, g) for g in golds_for(pair))


STOP = set("the a an of in to for and or is are be by on with as at from that this".split())

def toks(x):
    return {t for t in re.findall(r"[a-z0-9.§\-]+", E.normalize_text(x)) if t not in STOP}

def soft_match(pred, gold, thr=0.6):
    if E.relaxed_answer_match(pred, gold):
        return True
    p, g = toks(pred), toks(gold)
    return bool(p and g) and len(p & g) / min(len(p), len(g)) >= thr

def ans_soft(pred, pair, ds=""):
    if ds == "retail":
        return ans_exact(pred, pair, ds)
    return any(soft_match(pred, g) for g in golds_for(pair))


def _term(sample):
    return " ".join(a.get("term", "") for a in sample["gold"]["ambiguous_conditions"])


def norm_f1(pc, gc, ds, term):
    pc = [str(x) for x in pc if str(x).strip()]
    if not pc or not gc:
        return 0.0
    hit = len(assign(gc, pc, ds, term))
    pr, rc = hit / len(pc), hit / len(gc)
    return 0.0 if hit == 0 else 2 * pr * rc / (pr + rc)


def score_detection(sample, parsed, ds=""):
    parsed = parsed if isinstance(parsed, dict) else None
    pd, gd = E.predicted_dimensions(parsed), E.gold_dimensions(sample)
    pc, gc = E.predicted_possible_conditions(parsed), E.gold_possible_conditions(sample)
    return {"json_valid": float(parsed is not None),
            "dimension_f1": E.f1_score(pd, gd)["f1"],
            "condition_f1_strict": E.f1_score(pc, gc)["f1"],
            "condition_f1_norm": norm_f1(pc, gc, ds, _term(sample))}


def score_cg(pair, parsed, ds=""):
    pa = (parsed or {}).get("answer", "") if isinstance(parsed, dict) else ""
    return {"json_valid": float(isinstance(parsed, dict)),
            "answer_exact": float(ans_exact(pa, pair, ds)),
            "answer_relaxed": float(ans_relaxed(pa, pair, ds)),
            "answer_soft_diag": float(ans_soft(pa, pair, ds))}


def score_e2e(sample, parsed, ds=""):
    parsed = parsed if isinstance(parsed, dict) else None
    pairs = sample["gold"]["condition_answer_pairs"]
    preds = [a for a in ((parsed or {}).get("answers") or []) if isinstance(a, dict) and isinstance(a.get("conditions"), dict)] if parsed else []
    pc, gc = E.predicted_possible_conditions(parsed), E.gold_possible_conditions(sample)
    term = _term(sample)
    n = len(pairs) or 1
    # strict (paper): exact normalized condition string match
    cov = ex = rel = 0
    for gp in pairs:
        m = next((a for a in preds if E.condition_matches(gp["conditions"], a["conditions"])), None)
        if m is not None:
            cov += 1; ex += ans_exact(m.get("answer", ""), gp, ds); rel += ans_relaxed(m.get("answer", ""), gp, ds)
    # normalized: rule-based condition normalization + one-to-one assignment
    gtxt = [" ".join(str(v) for v in gp["conditions"].values()) for gp in pairs]
    ptxt = [" ".join(str(v) for v in a["conditions"].values()) for a in preds]
    mp = assign(gtxt, ptxt, ds, term)
    ncov = len(mp)
    nex = sum(ans_exact(preds[pi].get("answer", ""), pairs[gi], ds) for gi, pi in mp.items())
    nrel = sum(ans_relaxed(preds[pi].get("answer", ""), pairs[gi], ds) for gi, pi in mp.items())
    return {"json_valid": float(parsed is not None),
            "dimension_f1": E.f1_score(E.predicted_dimensions(parsed), E.gold_dimensions(sample))["f1"],
            "condition_f1_strict": E.f1_score(pc, gc)["f1"], "coverage_strict": cov / n,
            "answer_exact_strict": ex / n, "answer_relaxed_strict": rel / n,
            "condition_f1_norm": norm_f1(pc, gc, ds, term), "coverage_norm": ncov / n,
            "answer_exact_norm": nex / n, "answer_relaxed_norm": nrel / n}


# ---------------------------------------------------------------- API
def call(model, system, prompt, args, max_tokens):
    a = argparse.Namespace(**vars(args)); a.max_tokens = max_tokens
    last = None
    for attempt in range(args.retries + 1):
        try:
            payload = E.chat_completion(model, system, prompt, a)
            text = E.response_text(payload)
            parsed, err = E.parse_json_output(text)
            if parsed is not None and not isinstance(parsed, dict):
                parsed, err = None, "top-level JSON is not an object"
            ch = (payload.get("choices") or [{}])[0]
            return {"status": "ok", "raw": text, "parsed": parsed, "parse_error": err,
                    "finish_reason": ch.get("finish_reason"), "usage": payload.get("usage"),
                    "elapsed_ms": payload.get("_elapsed_ms")}
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}"
            if e.code in (400, 401, 403, 404) and "rate" not in last.lower():
                break
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        time.sleep(min(60, args.backoff * 2 ** attempt))
    return {"status": "failed", "error": last, "raw": "", "parsed": None}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(HERE.parent / "data"))
    ap.add_argument("--out-dir", default=str(HERE.parent / "results" / "hinted"))
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--models", nargs="+", default=E.DEFAULT_MODELS)
    ap.add_argument("--limit", type=int, help="first N samples per dataset (pilot)")
    ap.add_argument("--workers-per-model", type=int, default=8)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--backoff", type=float, default=3.0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--rescore", action="store_true", help="recompute scores of saved outputs, then summarize")
    args = ap.parse_args()
    E.key_config.load_env_file(HERE.parent / ".env")

    system, b_det, b_cg, b_e2e = load_prompts()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    samples = {}
    for ds in args.datasets:
        rows = [json.loads(l) for l in open(Path(args.data_dir) / f"{ds}.jsonl", encoding="utf-8") if l.strip()]
        samples[ds] = rows[: args.limit] if args.limit else rows

    def key(ds, task, m, sid, i):
        return f"{ds}|{task}|{m}|{sid}|{i}"

    done = set()
    for f in out.glob("*_results.jsonl"):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                x = json.loads(line)
                if x.get("status") == "ok":
                    done.add(key(x["dataset"], x["task"], x["model"], x["id"], x["pair_index"]))

    jobs = defaultdict(list)  # per model
    for ds, rows in samples.items():
        for r in rows:
            for m in args.models:
                if "detection" in args.tasks:
                    jobs[m].append((ds, "detection", m, r, None))
                if "condition_given" in args.tasks:
                    for i in range(len(r["gold"]["condition_answer_pairs"])):
                        jobs[m].append((ds, "condition_given", m, r, i))
                if "e2e" in args.tasks:
                    jobs[m].append((ds, "e2e", m, r, None))
    for m in jobs:
        jobs[m] = [j for j in jobs[m] if key(j[0], j[1], j[2], j[3]["id"], j[4]) not in done]
    total = sum(len(v) for v in jobs.values())
    print(f"already done: {len(done)}  to run: {total}  ({ {m: len(v) for m, v in jobs.items()} })", flush=True)

    locks = defaultdict(threading.Lock)
    counter = {"n": 0, "fail": 0}
    clock = threading.Lock()

    def run(job):
        ds, task, m, r, i = job
        if task == "detection":
            prompt, mt = b_det(r), 2000
        elif task == "condition_given":
            prompt, mt = b_cg(r, r["gold"]["condition_answer_pairs"][i]), 3000
        else:
            prompt, mt = b_e2e(r), 4000
        res = call(m, system, prompt, args, mt)
        p = res.get("parsed")
        if task == "detection":
            sc = score_detection(r, p, ds)
            gold = r["gold"]["ambiguous_conditions"]
        elif task == "condition_given":
            sc = score_cg(r["gold"]["condition_answer_pairs"][i], p, ds)
            gold = r["gold"]["condition_answer_pairs"][i]
        else:
            sc = score_e2e(r, p, ds)
            gold = r["gold"]
        row = {"dataset": ds, "task": task, "model": m, "id": r["id"], "pair_index": i, "query": r["query"],
               "gold": gold, "prompt_file": "prompts.py", "prompt_chars": len(prompt), **res, "scores": sc}
        path = out / f"{ds}_{task}_results.jsonl"
        with locks[path]:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with clock:
            counter["n"] += 1; counter["fail"] += res["status"] != "ok"
            if counter["n"] % 50 == 0 or counter["n"] == total:
                print(f"  {counter['n']}/{total} done, failed={counter['fail']}  {time.strftime('%H:%M:%S')}", flush=True)

    if args.rescore:
        rescore(out, samples)
    if not args.summary_only and not args.rescore and total:
        pools = [concurrent.futures.ThreadPoolExecutor(max_workers=args.workers_per_model) for _ in jobs]
        futs = [pool.submit(run, j) for pool, (m, js) in zip(pools, jobs.items()) for j in js]
        for f in concurrent.futures.as_completed(futs):
            f.result()
        for p in pools:
            p.shutdown()

    summarize(out)


def rescore(out: Path, samples):
    by = {(ds, r["id"]): r for ds, rows in samples.items() for r in rows}
    for f in sorted(out.glob("*_results.jsonl")):
        rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        for x in rows:
            r = by.get((x["dataset"], x["id"]))
            if r is None:
                continue
            if x["task"] == "detection":
                x["scores"] = score_detection(r, x.get("parsed"), x["dataset"])
            elif x["task"] == "condition_given":
                x["scores"] = score_cg(r["gold"]["condition_answer_pairs"][x["pair_index"]], x.get("parsed"), x["dataset"])
            else:
                x["scores"] = score_e2e(r, x.get("parsed"), x["dataset"])
        f.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")


def summarize(out: Path):
    best = {}
    for f in sorted(out.glob("*_results.jsonl")):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            x = json.loads(line)
            k = (x["dataset"], x["task"], x["model"], x["id"], x["pair_index"])
            if k not in best or x["status"] == "ok":
                best[k] = x
    agg = defaultdict(lambda: defaultdict(list)); meta = defaultdict(lambda: defaultdict(int))
    for k, x in best.items():
        g = k[:3]
        for s, v in x["scores"].items():
            agg[g][s].append(v)
        meta[g]["n"] += 1
        meta[g]["failed"] += x["status"] != "ok"
        meta[g]["truncated"] += x.get("finish_reason") == "length"
    rows = []
    for g in sorted(agg):
        rows.append({"dataset": g[0], "task": g[1], "model": g[2], **meta[g],
                     **{s: round(100 * mean(v), 2) for s, v in agg[g].items()}})
    (out / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    cols = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    with open(out / "summary.csv", "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    for r in rows:
        print(" ".join(f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
