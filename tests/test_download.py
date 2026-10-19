from t2s.ingest.download import build_filters, fiscal_year_date_range


def test_fiscal_year_date_range() -> None:
    assert fiscal_year_date_range(2024) == ("2023-10-01", "2024-09-30")
    assert fiscal_year_date_range(2022) == ("2021-10-01", "2022-09-30")


def test_build_filters_contracts() -> None:
    filters = build_filters(2024, ["contracts"])
    assert filters["prime_award_types"] == ["A", "B", "C", "D"]
    assert filters["date_type"] == "action_date"
    assert filters["date_range"] == {"start_date": "2023-10-01", "end_date": "2024-09-30"}
