You are generating training data for a Text-to-SQL model. The target SQL
dialect is **Spark SQL**. You are given a schema card describing the only
tables and columns available -- do not invent tables or columns that aren't
listed.

Schema card:
---
{schema_card}
---

Generate {n} distinct natural-language questions about this data, each paired
with a single Spark SQL query that answers it, all tagged with the difficulty
tag: **{tag}**.

Difficulty tag definitions:
- single_table: answerable from award_transactions alone, no joins, no aggregation.
- aggregation: uses SUM/COUNT/AVG/MIN/MAX or GROUP BY.
- join: requires joining award_transactions to at least one dimension table.
- filter_date_fy: filters on fiscal_year (not the calendar year of action_date).
- top_n: uses ORDER BY ... LIMIT to find top/bottom N results.
- nested: uses a subquery or CTE.
- window: uses a window function (e.g. RANK, ROW_NUMBER, SUM OVER).

Rules:
- Every SQL query must be valid Spark SQL and must only reference tables and
  columns from the schema card above.
- Prefer federal_action_obligation for obligation questions unless the
  question explicitly asks about outlays or total award value.
- Vary phrasing and the specific filter values (agencies, fiscal years,
  recipients, states, NAICS/PSC codes) across questions.
- Only write a question expecting an empty result if the question explicitly
  implies no matching rows should exist (set expects_empty_result: true).

Respond with **only** a JSON array, no markdown fences, no commentary. Each
element must have exactly these keys:
[{{"question": "...", "sql": "...", "tags": ["{tag}"], "expects_empty_result": false}}]
