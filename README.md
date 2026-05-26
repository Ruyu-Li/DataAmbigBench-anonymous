# DataAmbigBench

This anonymous supplementary repository contains the benchmark data and
evaluation code for **DataAmbigBench**, a benchmark for ambiguity-aware
reasoning in data analytics.

DataAmbigBench evaluates whether language models can recognize
underspecified analytical queries over structured and semi-structured data,
enumerate valid resolving interpretations, and produce condition-specific
answers grounded in the provided data context.

## Contents

```text
data/
  sec_eval_standard.jsonl
  nasa_eval_standard_v2.jsonl
  tatqa_eval_standard_v2.jsonl
  online_retail_eval_standard.jsonl
scripts/
  prompts.py
  evaluate_e2e.py
requirements.txt
.env.example
```

The repository is anonymized for double-blind review. It intentionally does
not contain author names, institutional paths, private credentials, or API
keys.

## Benchmark Format

Each JSONL row is one benchmark instance. The core fields are:

- `id`: unique instance identifier.
- `domain`: source subset.
- `table`: data context supplied to the model.
- `query`: underspecified analytical query.
- `gold.ambiguous_conditions`: gold ambiguity dimension and resolving
  conditions.
- `gold.condition_answer_pairs`: interpretation-specific answers.
- `metadata`: non-essential source metadata.

Each instance contains at least two valid interpretations in the same data
context. The benchmark covers metric-definition ambiguity, temporal ambiguity,
and granularity ambiguity.

## Data Sources

The released benchmark instances are derived from four public or widely used
data sources:

- SEC financial statement data.
- NASA POWER solar irradiance data.
- TAT-QA financial tables.
- Online Retail II transaction data.

The benchmark uses these sources to construct controlled analytical
underspecification. It is not intended to reproduce the original datasets in
full.

## Evaluation

The main evaluation setting is end-to-end ambiguity resolution. Given a data
context and an underspecified query, a model must produce a JSON object with:

1. predicted ambiguous conditions,
2. possible resolving conditions, and
3. condition-specific answers.

The script computes:

- JSON validity,
- ambiguity dimension F1,
- resolving condition F1,
- condition coverage,
- exact answer accuracy, and
- relaxed answer accuracy.

## Running Evaluation

The scripts use only the Python standard library.

Set API keys with environment variables:

```bash
export OPENAI_API_KEY=...
export DASHSCOPE_API_KEY=...
export DEEPSEEK_API_KEY=...
```

Run a small smoke evaluation:

```bash
python scripts/evaluate_e2e.py \
  --dataset nasa \
  --models gpt-4o-mini qwen3.6-flash deepseek-v4-flash \
  --limit 5
```

Run a full subset:

```bash
python scripts/evaluate_e2e.py --dataset sec
python scripts/evaluate_e2e.py --dataset nasa
python scripts/evaluate_e2e.py --dataset tatqa
python scripts/evaluate_e2e.py --dataset online_retail
```

By default, outputs are written to `results/`. The script supports resuming:
completed `(model, instance)` pairs are skipped unless `--no-resume` is set.

## Endpoint Configuration

Default OpenAI-compatible endpoints are:

```text
OPENAI_BASE_URL=https://api.openai.com/v1
QWEN_BASE_URL=https://dashscope-us.aliyuncs.com/compatible-mode/v1
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

These can be overridden with environment variables.

