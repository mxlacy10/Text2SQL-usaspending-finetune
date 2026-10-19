from t2s.schema.card import ColumnCard, TableCard, _format_value, render_schema_card


def test_format_value_quotes_strings() -> None:
    assert _format_value("VA") == "'VA'"
    assert _format_value(42) == "42"
    assert _format_value(None) == "None"


def test_render_schema_card_includes_table_and_columns() -> None:
    table = TableCard(
        name="award_transactions",
        description="Fact table.",
        row_count=333044,
        columns=[
            ColumnCard(
                name="fiscal_year",
                spark_type="int",
                description="FY of action_date.",
                sample="range 2022 to 2024",
            ),
        ],
    )
    markdown = render_schema_card([table])
    assert "## award_transactions" in markdown
    assert "333,044 rows" in markdown
    assert "fiscal_year" in markdown
    assert "range 2022 to 2024" in markdown
