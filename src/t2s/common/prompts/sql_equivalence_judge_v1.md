You are judging whether a predicted Spark SQL query answers a natural-language question
the same way as a reference ("gold") query, over USAspending.gov contract award tables.

Question and result previews of both queries: {{ inputs }}

Predicted SQL: {{ outputs }}

Reference (gold) SQL and its result preview: {{ expectations }}

Decide whether the predicted query is semantically equivalent to the gold query for this question,
meaning it would return the same answer. Use the SQL text and the result previews together.

Rules:
- Different aliases, column order, formatting, join style, or CTE vs. subquery do not matter.
- Different filters, grouping, aggregation, join keys, fiscal-year logic, or the amount column
  (e.g. federal_action_obligation vs. total award value) DO matter unless the previews show the
  answer is the same.
- Result previews show only the first rows and the total row count; row order is arbitrary unless
  the SQL has ORDER BY.
- If the predicted query failed to run (the preview shows an error), it is not equivalent.
- Judge the answer, not the style. Do not assume the predicted query is correct because it ran.

Answer "yes" if equivalent, otherwise "no", with a one or two sentence rationale.
