from t2s.synth.dedupe import dedupe_candidates, normalize_question, normalize_sql
from t2s.synth.models import ExecutedExample


def make_example(question: str, sql: str, tag: str = "single_table") -> ExecutedExample:
    return ExecutedExample(question=question, sql=sql, tags=[tag], valid=True, row_count=1)


def test_normalize_question_strips_punctuation_and_case() -> None:
    assert normalize_question("How many awards, total?") == "how many awards total"
    assert normalize_question("  Extra   spaces  ") == "extra spaces"


def test_normalize_sql_canonicalizes_whitespace() -> None:
    a = normalize_sql("SELECT   a,b FROM t")
    b = normalize_sql("SELECT a, b FROM t")
    assert a == b


def test_dedupe_drops_exact_duplicate_questions() -> None:
    examples = [
        make_example("How many awards?", "SELECT COUNT(*) FROM award_transactions"),
        make_example("how many awards??", "SELECT COUNT(*) FROM award_transactions WHERE 1=1"),
    ]
    kept = dedupe_candidates(examples)
    assert len(kept) == 1


def test_dedupe_drops_duplicate_sql() -> None:
    examples = [
        make_example("Question A", "SELECT COUNT(*) FROM award_transactions"),
        make_example("Question B", "SELECT   COUNT(*)   FROM award_transactions"),
    ]
    kept = dedupe_candidates(examples)
    assert len(kept) == 1


def test_dedupe_keeps_distinct_examples() -> None:
    examples = [
        make_example("Question A", "SELECT COUNT(*) FROM award_transactions"),
        make_example("Question B", "SELECT SUM(federal_action_obligation) FROM award_transactions"),
    ]
    kept = dedupe_candidates(examples)
    assert len(kept) == 2
