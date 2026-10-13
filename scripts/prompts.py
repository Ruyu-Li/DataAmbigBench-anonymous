"""Prompts used in the paper (v3 taxonomy: Unit, Name Ambiguity, Perspective, Legal-Scope, Event Status)."""
from typing import Any, Dict


def safe_get_table(sample: Dict[str, Any]) -> str:
    """Render the data context: strings as-is, {"content": ...} dicts as their content,
    and {"title", "header", "rows"} transaction tables (Online Retail II) as a markdown table."""
    t = sample.get("table")
    if isinstance(t, dict) and "rows" in t and "header" in t:
        hdr = t["header"]
        esc = lambda v: str(v).replace("|", "/").replace("\n", " ")
        lines = [str(t.get("title", "")), "", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        lines += ["| " + " | ".join(esc(r.get(h, "")) for h in hdr) + " |" for r in t["rows"]]
        return "\n".join(lines)
    if isinstance(t, dict):
        return str(t.get("content", t))
    return str(t)


SYSTEM_PROMPT = """You are an evaluator for ambiguous tabular question answering.
You must use only the provided table.
You must output valid JSON only.
Do not include markdown, explanations outside JSON, or extra text."""


ALLOWED_DIMENSIONS = "Definition-Metric, Temporal, Unit, Geography, Name Ambiguity, Perspective, Legal-Scope, Event Status, Other"


DIMENSION_DEFINITIONS = """
Dimension definitions:

1. Definition-Metric:
The query uses an abstract, underspecified, or overloaded metric term, and the data contains multiple possible concrete metrics or calculation rules.
The ambiguity comes from not knowing which metric definition should be used.

2. Temporal:
The query contains a time expression or asks for trends, but the exact time period, comparison window, or temporal reference is unclear.
The ambiguity comes from not knowing which time condition should be used.

3. Unit:
The query asks for analysis over time but does not specify the temporal aggregation unit (e.g., day, week, month, quarter, year).
The ambiguity comes from not knowing which time unit the records should be aggregated by.

4. Geography:
The query contains or implies a location-related scope, but the location or geographic level is unclear.
The ambiguity comes from not knowing which country, region, or geographic subset should be used.

5. Name Ambiguity:
The query uses a name shared by several distinct persons, places, organizations, or works in the provided documents.
The ambiguity comes from not knowing which same-named entity is meant; resolving it does not depend on choosing an applicable statutory definition.

6. Perspective:
The query asks about opinions, preferences, or judgments but leaves the respondent group unspecified, and different respondent groups yield different leading answers.
The ambiguity comes from not knowing whose opinions (which respondent group) the answer should reflect.

7. Legal-Scope:
The meaning of a legal term, or a fact derived from its definition, changes across the supplied legal provisions.
The ambiguity comes from not knowing which legal provision applies; when statutory definitions name different entities, use Legal-Scope.

8. Event Status:
The query uses a status word (e.g., pending, active, underway) that can map to different recorded status values or processing stages.
The ambiguity comes from not knowing which status definition should be used.

9. Other:
Use this only if the ambiguity does not fit the above dimensions.
""".strip()


def _few_shot_block(prompt_mode: str) -> str:
    """
    Minimum-prompt version:
    We intentionally do not include few-shot examples.
    This keeps zero_shot and few_shot behavior effectively the same unless examples are added later.
    """
    return ""


def build_detection_prompt(sample: Dict[str, Any], prompt_mode: str = "zero_shot") -> str:
    table = safe_get_table(sample)
    query = sample["query"]

    return f"""
Task: Ambiguity Detection for Tabular QA

Given a table and a user query, identify ambiguous terms or phrases in the query.

You need to identify:
- the ambiguous term or phrase
- the ambiguity dimension
- the possible concrete conditions or interpretations

Allowed dimensions:
{ALLOWED_DIMENSIONS}

{DIMENSION_DEFINITIONS}

Return JSON exactly in this schema:
{{
  "ambiguous_conditions": [
    {{
      "term": "...",
      "dimension": "Definition-Metric or Temporal or Unit or Geography or Name Ambiguity or Perspective or Legal-Scope or Event Status or Other",
      "possible_conditions": ["...", "..."]
    }}
  ]
}}

Now evaluate the following case.

Table:
{table}

Query:
{query}
""".strip()


def build_condition_answer_prompt(
    sample: Dict[str, Any],
    condition_obj: Dict[str, Any],
    prompt_mode: str = "zero_shot"
) -> str:
    table = safe_get_table(sample)
    query = sample["query"]
    conditions = condition_obj.get("conditions", {})

    return f"""
Task: Condition-Given Tabular QA

Given a table, a query, and gold conditions, answer the query using only the specified gold conditions.

Allowed dimensions:
{ALLOWED_DIMENSIONS}

{DIMENSION_DEFINITIONS}

Rules:
1. The gold condition overrides the original ambiguous query wording.
2. Use only the condition specified in the gold conditions.
3. Do not use another condition that is not specified.
4. The final answer must be the value, label, group, or trend requested by the gold condition.
5. Output only the final answer in the "answer" field.

Return JSON exactly in this schema:
{{
  "filtered_rows": ["short description of the rows or filter used (do not list every row)"],
  "group_values": {{
    "metric name or group name": "computed value"
  }},
  "answer": "...",
  "calculation": "brief calculation from the table"
}}

Now evaluate the following case.

Table:
{table}

Query:
{query}

Gold Conditions:
{conditions}
""".strip()


def build_e2e_prompt(sample: Dict[str, Any], prompt_mode: str = "zero_shot") -> str:
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
- The condition key must be one of: Definition-Metric, Temporal, Unit, Geography, Name Ambiguity, Perspective, Legal-Scope, Event Status, Other.
- Do not use placeholder keys such as "DimensionName" or "AnotherDimension".
- The "answer" field should contain only the final answer value, not a full sentence.

Return JSON exactly in this schema:
{{
  "ambiguous_conditions": [
    {{
      "term": "...",
      "dimension": "Definition-Metric or Temporal or Unit or Geography or Name Ambiguity or Perspective or Legal-Scope or Event Status or Other",
      "possible_conditions": ["...", "..."]
    }}
  ],
  "answers": [
    {{
      "conditions": {{
        "Definition-Metric": "specific condition"
      }},
      "filtered_rows": ["short description of the rows or filter used (do not list every row)"],
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


# ---------------------------------------------------------------- E2E-open (no ambiguity hint)
OPEN_SYSTEM_PROMPT = """You are a data analysis assistant.
You must use only the provided data context.
You must output valid JSON only.
Do not include markdown, explanations outside JSON, or extra text."""


def build_open_prompt(sample: Dict[str, Any]) -> str:
    table = safe_get_table(sample)
    return f"""
Task: Data Question Answering

Answer the user's question using only the provided data context.

Return JSON exactly in this schema:
{{
  "answers": [
    {{
      "interpretation": "how you understood the question",
      "answer": "final answer value only",
      "calculation": "brief calculation or evidence from the data"
    }}
  ]
}}

Data context:
{table}

Question:
{sample["query"]}
""".strip()
