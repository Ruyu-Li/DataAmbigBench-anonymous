# DataAmbigBench

This anonymous repository contains the benchmark data and evaluation code for
**DataAmbigBench**, a benchmark for ambiguity-aware reasoning in data analytics.
Each instance pairs an underspecified analytical query with a data context
(a table, documents, or statute text), a gold ambiguity dimension, the valid
resolving conditions, and one answer per condition.

The repository is anonymized for double-blind review. It contains no author
names, institutional paths, private credentials, or API keys.

## Contents

```text
data/                      one JSONL file per subset (gold included; scoring only)
  sec.jsonl  nasa.jsonl  tatqa.jsonl  retail.jsonl
  ambigdocs.jsonl  opinionqa.jsonl  usc.jsonl  sf311.jsonl
scripts/
  prompts.py               all prompts used in the paper (hinted settings and E2E-open)
  evaluate.py              Detection, Condition-Given, and hinted End-to-End
  evaluate_open.py         E2E-open (no ambiguity hint)
  common.py                API calls, JSON parsing, answer matching
  config.py                endpoint defaults; keys are read from the environment
  bootstrap_ci.py          95% bootstrap confidence intervals
results/
  hinted/summary.{csv,json}    scores of the three hinted settings (paper Tables 2, 7)
  open/summary_open.{csv,json} E2E-open scores (paper Table 2, Figure 3)
  hinted_vs_open.csv           multi-interpretation rate and answer recall, hinted vs. open
  bootstrap_ci.json            confidence intervals (paper Table 8)
```

## Subsets

| Subset | Dimension | Source | Queries | Interpretation-answer pairs |
|---|---|---|---:|---:|
| `sec` | Definition-Metric | SEC Financial Statement Data Sets | 648 | 1,296 |
| `nasa` | Definition-Metric | NASA POWER solar irradiance | 2,240 | 8,960 |
| `tatqa` | Temporal | TAT-QA financial tables | 610 | 1,503 |
| `retail` | Unit | Online Retail II transactions | 200 | 1,000 |
| `ambigdocs` | Name Ambiguity | AmbigDocs (Wikipedia documents) | 200 | 479 |
| `opinionqa` | Perspective | OpinionQA (Pew American Trends Panel) | 200 | 576 |
| `usc` | Legal-Scope | United States Code | 200 | 400 |
| `sf311` | Event Status | San Francisco 311 cases | 200 | 400 |
| **Total** | 7 dimensions | 8 subsets | **4,498** | **14,614** |

## Data Format

Each JSONL row is one instance:

- `id`: unique identifier.
- `query`: the underspecified analytical query.
- `table`: the data context given to the model. Usually a string (markdown table,
  documents, or statute text); `nasa`/`sec`/`tatqa` may use `{"type", "content"}`,
  and `retail` uses `{"title", "header", "rows"}` (rendered as a markdown table by
  `scripts/prompts.py`).
- `gold.ambiguous_conditions`: the ambiguous term, its dimension, and the list of
  resolving conditions.
- `gold.condition_answer_pairs`: one `{"conditions": {dimension: condition},
  "answer": ...}` per interpretation; `acceptable_answers` lists aliases where needed.
- `metadata`: provenance and construction details (not shown to models).

Models only ever see `query` and `table` (plus one gold condition in Condition-Given).

## Evaluation Settings

| Task | Setting | Model input | Metric |
|---|---|---|---|
| 1 | E2E-open | query + data, no ambiguity hint | Surfacing Rate, Answer Recall |
| 2, 3 | Detection | query + data | Type F1, Interpretation F1 |
| 4 | Condition-Given | query + data + one gold condition | Given-Condition Accuracy |
| 5 | E2E (hinted) | query + data | E2E Accuracy |

Scoring is deterministic and uses no model. Answers are compared after text
normalization and numeric equivalence (aliases accepted; Online Retail II periods
compared in canonical form, e.g. `2011Q3` = `Q3 2011`). Conditions are matched after
rule-based normalization with one-to-one assignment; exact-match ("strict") scores
are reported alongside. Scores are averaged over queries within a subset, and
averages across subsets weight subsets equally.

## Running the Evaluation

The evaluation uses only the Python standard library (`numpy` is needed only for
`bootstrap_ci.py`). Set API keys as environment variables (or in a local `.env`,
see `.env.example`):

```bash
export OPENAI_API_KEY=...
export DASHSCOPE_API_KEY=...
export DEEPSEEK_API_KEY=...
```

Smoke test (one instance per subset):

```bash
python scripts/evaluate.py --limit 1
python scripts/evaluate_open.py --limit 1
```

Full runs (resumable; finished calls are skipped):

```bash
python scripts/evaluate.py                       # all subsets, Detection + Condition-Given + E2E
python scripts/evaluate.py --datasets usc sf311 --tasks e2e --models gpt-4o-mini
python scripts/evaluate_open.py                  # E2E-open
python scripts/bootstrap_ci.py                   # confidence intervals
```

Outputs go to `results/hinted/` and `results/open/` (one JSONL row per model call,
with the raw response, parsed JSON, and scores) and are summarized in
`summary.csv` / `summary_open.csv`. To recompute scores from saved outputs without
calling any model, add `--rescore`.

Models in the paper: `gpt-4o-mini`, `qwen3.6-flash`, `deepseek-v4-flash`, all with
temperature 0. Default OpenAI-compatible endpoints (override with
`OPENAI_BASE_URL`, `QWEN_BASE_URL`, `DEEPSEEK_BASE_URL`):

```text
https://api.openai.com/v1
https://dashscope-us.aliyuncs.com/compatible-mode/v1
https://api.deepseek.com
```

## Data Sources and Licensing

The subsets are derived from public data: SEC Financial Statement Data Sets,
NASA POWER, TAT-QA, Online Retail II (UCI), AmbigDocs, OpinionQA / Pew American
Trends Panel, the United States Code (public domain), and DataSF 311 Cases. Use of
each subset is subject to the terms of its original source. The benchmark is
intended for research on ambiguity-aware analytical reasoning and does not
constitute financial or legal advice.
