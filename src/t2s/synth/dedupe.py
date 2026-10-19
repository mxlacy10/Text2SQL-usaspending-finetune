"""Drop near-duplicate questions and semantically duplicated SQL.

AGENTS.md calls for "normalized text plus embedding similarity" question
dedup. Exact normalized-text dedup and canonical-SQL dedup are implemented
here; embedding-similarity near-dup detection needs an embeddings
model/API and is deferred.

TODO(max): add embedding-similarity question dedup once an embeddings
provider is wired up for this project; exact-normalized-text dedup alone
will miss paraphrases of the same question.
"""

from __future__ import annotations

import re

import sqlglot

from t2s.synth.models import ExecutedExample


def normalize_question(question: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = question.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def normalize_sql(sql: str) -> str:
    """Canonicalize SQL via sqlglot so cosmetically different but semantically
    identical queries compare equal."""
    try:
        return sqlglot.transpile(sql, read="spark", write="spark", pretty=False)[0]
    except sqlglot.errors.ParseError:
        return sql.strip()


def dedupe_candidates(examples: list[ExecutedExample]) -> list[ExecutedExample]:
    """Keep the first occurrence of each (normalized question, normalized SQL)."""
    seen_questions: set[str] = set()
    seen_sql: set[str] = set()
    kept: list[ExecutedExample] = []
    for example in examples:
        q_key = normalize_question(example.question)
        sql_key = normalize_sql(example.sql)
        if q_key in seen_questions or sql_key in seen_sql:
            continue
        seen_questions.add(q_key)
        seen_sql.add(sql_key)
        kept.append(example)
    return kept
