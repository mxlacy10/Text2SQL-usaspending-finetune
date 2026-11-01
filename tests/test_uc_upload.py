from t2s.ingest.uc_upload import parity_table, parse_column_comments

CARD = """# Schema Card

## agencies

Dimension. (6 rows)

| column | type | description | sample values |
|---|---|---|---|
| agency_role | string | Whether 'awarding' or 'funding'. | 'awarding', 'funding' |
| agency_name | string | Toptier agency name. | 'A', 'B' |

## locations

| column | type | description | sample values |
|---|---|---|---|
| state_code | string | State code. | 'VA' |
"""


def test_parse_column_comments() -> None:
    assert parse_column_comments(CARD) == {
        "agencies": {
            "agency_role": "Whether 'awarding' or 'funding'.",
            "agency_name": "Toptier agency name.",
        },
        "locations": {"state_code": "State code."},
    }


def test_parity_table_flags_mismatch() -> None:
    local = {"award_transactions": 1, "agencies": 2, "recipients": 3, "locations": 4}
    text, ok = parity_table(local, local)
    assert ok and "MISMATCH" not in text
    text, ok = parity_table(local, {**local, "agencies": 9})
    assert not ok and "agencies | 2 | 9 | MISMATCH" in text
