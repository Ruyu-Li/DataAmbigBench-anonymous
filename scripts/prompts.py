"""Prompt templates for end-to-end ambiguity-aware data analytics evaluation."""

from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = """You are an evaluator for ambiguous tabular question answering.
You must use only the provided table.
You must output valid JSON only.
Do not include markdown, explanations outside JSON, or extra text."""


ALLOWED_DIMENSIONS = "Definition-Metric, Temporal, Granularity, Geography, Other"


DIMENSION_DEFINITIONS = """
Dimension definitions:

1. Definition-Metric:
The query uses an abstract, underspecified, or overloaded metric term, and the table contains multiple possible concrete metrics.
The ambiguity comes from not knowing which metric definition should be used.

2. Temporal:
The query contains a time expression or asks for trends, but the exact time period, comparison window, or temporal reference is unclear.
The ambiguity comes from not knowing which time condition should be used.

3. Granularity:
The query asks for analysis over time or another dimension, but does not specify the grouping level.
The ambiguity comes from not knowing how finely or coarsely to group the data.

4. Geography:
The query contains or implies a location-related scope, but the location or geographic level is unclear.
The ambiguity comes from not knowing which country, region, or geographic subset should be used.

5. Other:
Use this only if the ambiguity does not fit the above dimensions.
""".strip()


def safe_get_table(sample: dict[str, Any]) -> str:
    table = sample.get("table")
    if isinstance(table, dict):
        return str(table.get("content", table))
    return str(table)


def build_e2e_prompt(sample: dict[str, Any]) -> str:
    table = safe_get_table(sample)
    query = sample["query"]

    return f"""
Task: End-to-End Ambiguous Tabular QA

Given a table and an ambiguous query:
1. Identify ambiguous conditions in the query.
2. Generate possible specific interpretations.
3. Answer the query under each interpretation.

Use only the provided table.

Allowed dimensions:
{ALLOWED_DIMENSIONS}

{DIMENSION_DEFINITIONS}

Output format requirement:
- In each answer, the "conditions" object must use actual dimension labels as keys.
- The condition key must be one of: Definition-Metric, Temporal, Granularity, Geography, Other.
- Do not use placeholder keys such as "DimensionName" or "AnotherDimension".
- The "answer" field should contain only the final answer value, not a full sentence.

Return JSON exactly in this schema:
{{
  "ambiguous_conditions": [
    {{
      "term": "...",
      "dimension": "Definition-Metric or Temporal or Granularity or Geography or Other",
      "possible_conditions": ["...", "..."]
    }}
  ],
  "answers": [
    {{
      "conditions": {{
        "Definition-Metric": "specific condition"
      }},
      "filtered_rows": ["row ids or short row descriptions used"],
      "group_values": {{
        "metric name or group name": "computed value"
      }},
      "answer": "...",
      "calculation": "brief calculation from the table"
    }}
  ]
}}

Now evaluate the following case.

Table:
{table}

Query:
{query}
""".strip()
