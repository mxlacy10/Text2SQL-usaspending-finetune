"""Copy the local silver Delta tables into Unity Catalog.

Local Spark and Databricks Connect can't share a process (databricks-connect replaces
pyspark), so this is a three-phase flow, each phase runnable on its own:

1. `local_row_counts`  (local env)       -> JSON of per-table row counts
2. `upload_to_volume`  (Databricks SDK)  -> Delta folders copied to a UC Volume
3. `create_tables`, `uc_row_counts`      (Connect session) -> CTAS from the Volume, comments, parity
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from t2s.common.config import BaseConfig
from t2s.synth.execute import SILVER_TABLES

VOLUME_NAME = "raw"
VOLUME_SUBDIR = "silver"


def _quote(identifier: str) -> str:
    return "`" + identifier.replace("`", "``") + "`"


def volume_root(config: BaseConfig) -> str:
    return f"/Volumes/{config.catalog}/{config.schema_name}/{VOLUME_NAME}/{VOLUME_SUBDIR}"


def parse_column_comments(schema_card: str) -> dict[str, dict[str, str]]:
    """Extract {table: {column: description}} from the generated schema card markdown."""
    comments: dict[str, dict[str, str]] = {}
    table: str | None = None
    for line in schema_card.splitlines():
        heading = re.match(r"^## (\w+)\s*$", line)
        if heading:
            table = heading.group(1)
            comments[table] = {}
            continue
        if table is None or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cells) < 3 or cells[0] in ("column", "---") or set(cells[0]) <= {"-"}:
            continue
        comments[table][cells[0]] = cells[2]
    return comments


def local_row_counts(spark: Any, config: BaseConfig) -> dict[str, int]:
    warehouse = Path(config.spark.warehouse_dir)
    return {t: spark.read.format("delta").load(str(warehouse / t)).count() for t in SILVER_TABLES}


def upload_to_volume(config: BaseConfig) -> dict[str, int]:
    """Create the Volume if needed, copy each local Delta folder into it; files per table."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.catalog import VolumeType

    assert config.catalog
    w = WorkspaceClient()
    try:
        w.volumes.read(f"{config.catalog}.{config.schema_name}.{VOLUME_NAME}")
    except Exception:  # noqa: BLE001 - not found; create it
        w.volumes.create(
            catalog_name=config.catalog,
            schema_name=config.schema_name,
            name=VOLUME_NAME,
            volume_type=VolumeType.MANAGED,
        )

    warehouse = Path(config.spark.warehouse_dir)
    uploaded: dict[str, int] = {}
    for table in SILVER_TABLES:
        files = [p for p in (warehouse / table).rglob("*") if p.is_file()]
        for path in files:
            rel = path.relative_to(warehouse).as_posix()
            with path.open("rb") as f:
                w.files.upload(f"{volume_root(config)}/{rel}", f, overwrite=True)
        uploaded[table] = len(files)
    return uploaded


def create_tables(spark: Any, config: BaseConfig, schema_card: str) -> None:
    comments = parse_column_comments(schema_card)
    for table in SILVER_TABLES:
        path = f"{volume_root(config)}/{table}"
        spark.sql(f"CREATE OR REPLACE TABLE {_quote(table)} AS SELECT * FROM delta.`{path}`")
        existing = {f.name for f in spark.table(table).schema.fields}
        for column, description in comments.get(table, {}).items():
            if column in existing:
                escaped = description.replace("\\", "\\\\").replace("'", "\\'")
                spark.sql(
                    f"ALTER TABLE {_quote(table)} ALTER COLUMN {_quote(column)} COMMENT '{escaped}'"
                )


def uc_row_counts(spark: Any) -> dict[str, int]:
    return {t: spark.table(t).count() for t in SILVER_TABLES}


def parity_table(local: dict[str, int], uc: dict[str, int]) -> tuple[str, bool]:
    lines = ["table | local_rows | uc_rows | match"]
    ok = True
    for t in SILVER_TABLES:
        match = local.get(t) == uc.get(t)
        ok &= match
        lines.append(f"{t} | {local.get(t)} | {uc.get(t)} | {'OK' if match else 'MISMATCH'}")
    return "\n".join(lines), ok


def write_counts(counts: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(counts, indent=2))


def read_counts(path: Path) -> dict[str, int]:
    data: dict[str, int] = json.loads(path.read_text())
    return data
