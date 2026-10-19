"""Schema card generator.

Builds `data/schema_card.md` from the silver Delta tables: table purposes,
documented column descriptions, types, and real sample/range values. Per
AGENTS.md, this file is the *only* schema context given to any model (base,
few-shot/RAG, frontier, or fine-tuned) -- never hand-edit it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, NumericType, TimestampType

from t2s.common.config import BaseConfig
from t2s.ingest.silver import (
    SILVER_AGENCIES_TABLE,
    SILVER_FACT_TABLE,
    SILVER_LOCATIONS_TABLE,
    SILVER_RECIPIENTS_TABLE,
)

DEFAULT_OUTPUT_PATH = Path("data/schema_card.md")

MAX_SAMPLE_VALUES = 8
CATEGORICAL_DISTINCT_THRESHOLD = 50

TABLE_DOCS: dict[str, str] = {
    SILVER_FACT_TABLE: (
        "Fact table. One row per prime contract award transaction (an award action "
        "or modification). Place-of-performance, agency, and recipient attributes "
        "are denormalized onto this table, so most questions can be answered "
        "without joining to the dimension tables below."
    ),
    SILVER_AGENCIES_TABLE: (
        "Dimension. Distinct (role, agency, sub-agency) combinations seen in "
        "award_transactions. Join on "
        "award_transactions.awarding_agency_name = agencies.agency_name AND "
        "award_transactions.awarding_sub_agency_name = agencies.sub_agency_name "
        "AND agencies.agency_role = 'awarding' (swap 'awarding' for 'funding' to "
        "match the funding_* columns instead)."
    ),
    SILVER_RECIPIENTS_TABLE: (
        "Dimension. Distinct recipients seen in award_transactions, keyed by "
        "recipient_uei. Join on "
        "award_transactions.recipient_uei = recipients.recipient_uei."
    ),
    SILVER_LOCATIONS_TABLE: (
        "Dimension. Distinct primary places of performance seen in "
        "award_transactions, keyed by location_id (a hash of city/state/country). "
        "Join on award_transactions.location_id = locations.location_id."
    ),
}

COLUMN_DOCS: dict[str, str] = {
    "transaction_id": "Unique key for one contract transaction (an award action or modification).",
    "award_id": "Unique key for the overall award; constant across all of its modifications.",
    "piid": "Procurement Instrument Identifier -- the contract number.",
    "parent_piid": "PIID of the parent award, if this is a task/delivery order under an IDV.",
    "action_date": "Calendar date this transaction/modification was signed.",
    "fiscal_year": (
        "Federal fiscal year of action_date. FY runs Oct-Sep, e.g. FY2024 = "
        "Oct 2023-Sep 2024. Use this column, not the calendar year of action_date, "
        "whenever a question asks about a fiscal year."
    ),
    "period_of_performance_start_date": "Start date of the award's period of performance.",
    "period_of_performance_end_date": (
        "Current (not potential/option) end date of the award's period of performance."
    ),
    "federal_action_obligation": (
        "Dollar amount obligated by this specific transaction. Use this column for "
        "obligation questions unless outlays or total award value is explicitly asked for."
    ),
    "current_total_value_of_award": (
        "Total obligated value of the award as of this transaction (base + exercised options)."
    ),
    "potential_total_value_of_award": "Total potential value if all options are exercised.",
    "awarding_agency_name": "Toptier federal agency that awarded the contract.",
    "awarding_sub_agency_name": "Sub-agency within the awarding agency.",
    "funding_agency_name": "Toptier agency that funded the contract (may differ from the awarder).",
    "funding_sub_agency_name": "Sub-agency within the funding agency.",
    "recipient_uei": "Unique Entity Identifier (UEI) of the award recipient.",
    "recipient_name": "Legal name of the award recipient.",
    "recipient_state_code": "US state code of the recipient's address.",
    "recipient_country_code": "Country code of the recipient's address.",
    "pop_city_name": "City of the primary place of performance.",
    "pop_state_code": "State code of the primary place of performance.",
    "pop_country_code": "Country code of the primary place of performance.",
    "location_id": "Hash key (sha256 of city|state|country) for the primary place of performance.",
    "award_type": (
        "Contract award type, e.g. DEFINITIVE CONTRACT, PURCHASE ORDER, BPA CALL, DELIVERY ORDER."
    ),
    "naics_code": "NAICS industry classification code for the contracted work.",
    "naics_description": "Human-readable description of the NAICS code.",
    "psc_code": "Product/Service Code (PSC) for the contracted item or service.",
    "psc_description": "Human-readable description of the PSC code.",
    "extent_competed": "How competed the action was, e.g. Full and Open Competition, Not Competed.",
    "number_of_offers_received": "Number of offers/bids received for this contract action.",
    "agency_role": "Whether this row represents the 'awarding' or 'funding' agency relationship.",
    "agency_name": "Toptier agency name.",
    "sub_agency_name": "Sub-agency name.",
}


@dataclass(frozen=True)
class ColumnCard:
    name: str
    spark_type: str
    description: str
    sample: str


@dataclass(frozen=True)
class TableCard:
    name: str
    description: str
    row_count: int
    columns: list[ColumnCard]


def _format_value(value: object) -> str:
    if isinstance(value, str):
        return f"'{value}'"
    return str(value)


def _profile_column(df: DataFrame, name: str) -> ColumnCard:
    field = df.schema[name]
    description = COLUMN_DOCS.get(name, "TODO(max): document this column.")

    if isinstance(field.dataType, (NumericType, DateType, TimestampType)):
        row = df.select(F.min(name).alias("min"), F.max(name).alias("max")).first()
        min_value = row["min"] if row is not None else None
        max_value = row["max"] if row is not None else None
        sample = f"range {_format_value(min_value)} to {_format_value(max_value)}"
    else:
        approx_row = df.select(F.approx_count_distinct(name).alias("n")).first()
        approx_distinct = int(approx_row["n"]) if approx_row is not None else 0
        values_df = df.select(name).where(F.col(name).isNotNull()).distinct()
        if approx_distinct <= CATEGORICAL_DISTINCT_THRESHOLD:
            values_df = values_df.orderBy(name)
        rows = values_df.limit(MAX_SAMPLE_VALUES).collect()
        values = [_format_value(r[name]) for r in rows]
        suffix = (
            "" if approx_distinct <= MAX_SAMPLE_VALUES else f", ... (~{approx_distinct} distinct)"
        )
        sample = ", ".join(values) + suffix

    return ColumnCard(
        name=name, spark_type=field.dataType.simpleString(), description=description, sample=sample
    )


def build_table_card(df: DataFrame, table_name: str) -> TableCard:
    """Profile one silver table: row count plus per-column type/description/sample."""
    columns = [_profile_column(df, name) for name in df.columns]
    return TableCard(
        name=table_name,
        description=TABLE_DOCS.get(table_name, "TODO(max): document this table."),
        row_count=df.count(),
        columns=columns,
    )


def render_schema_card(tables: list[TableCard]) -> str:
    """Render table cards into the markdown schema card given to models."""
    lines = [
        "# Schema Card",
        "",
        "Generated by `scripts/schema_card.py` from the silver Delta tables. "
        "Do not edit by hand; this is the only schema context given to models.",
        "",
    ]
    for table in tables:
        lines.append(f"## {table.name}")
        lines.append("")
        lines.append(f"{table.description} ({table.row_count:,} rows)")
        lines.append("")
        lines.append("| column | type | description | sample values |")
        lines.append("|---|---|---|---|")
        for col in table.columns:
            lines.append(f"| {col.name} | {col.spark_type} | {col.description} | {col.sample} |")
        lines.append("")
    return "\n".join(lines)


def build_schema_card(
    spark: SparkSession,
    config: BaseConfig,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Profile all silver tables and write the rendered schema card to output_path."""
    warehouse_dir = Path(config.spark.warehouse_dir)
    table_names = [
        SILVER_FACT_TABLE,
        SILVER_AGENCIES_TABLE,
        SILVER_RECIPIENTS_TABLE,
        SILVER_LOCATIONS_TABLE,
    ]
    tables = []
    for table_name in table_names:
        df = spark.read.format("delta").load(str(warehouse_dir / table_name))
        tables.append(build_table_card(df, table_name))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_schema_card(tables))
    return output_path
