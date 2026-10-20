"""Post-process raw model completions into bare SQL text.

Instruction-tuned chat models often restate structural labels from the
prompt (e.g. echoing "SQL:" even though the prompt already ends with that
label) or wrap output in markdown code fences despite being told not to.
Both would otherwise make valid SQL fail to parse, so every predictor
(`t2s.eval.predictors`) runs its raw completion through this before
returning it.
"""

from __future__ import annotations

import re

_FENCE_RE = re.compile(r"^```(?:sql)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
_LABEL_PREFIX_RE = re.compile(r"^(?:sql|answer|query)\s*:\s*", re.IGNORECASE)


def clean_sql_output(text: str) -> str:
    """Strip markdown code fences and a leading "SQL:"/"Answer:"/"Query:"
    label from a raw model completion, returning the bare SQL text."""
    cleaned = text.strip()
    fenced = _FENCE_RE.match(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    cleaned = _LABEL_PREFIX_RE.sub("", cleaned).strip()
    return cleaned
