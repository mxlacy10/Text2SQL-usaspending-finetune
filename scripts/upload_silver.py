"""Copy local silver Delta tables into Unity Catalog (Step 2 of docs/databricks_plan.md).

Phases (run in this order; phase 1 in the main env, 2-4 in .venv-dbx):
  local-counts  local Spark row counts -> data/uc_parity_local.json
  upload        copy Delta folders to a UC Volume
  create        CTAS into UC tables + column comments from data/schema_card.md
  parity        compare local vs UC row counts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from t2s.common.config import load_base_config
from t2s.ingest import uc_upload

LOCAL_COUNTS = Path("data/uc_parity_local.json")
SCHEMA_CARD = Path("data/schema_card.md")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["local-counts", "upload", "create", "parity"])
    parser.add_argument("--config", default="config/base.yaml")
    args = parser.parse_args()
    config = load_base_config(args.config)

    if args.phase == "local-counts":
        from t2s.common.spark import get_spark_session

        local = config.model_copy(deep=True)
        local.spark.mode = "local"
        counts = uc_upload.local_row_counts(get_spark_session(local), local)
        uc_upload.write_counts(counts, LOCAL_COUNTS)
        print(counts)
    elif args.phase == "upload":
        print(uc_upload.upload_to_volume(config))
    else:
        from t2s.common.spark import get_spark_session

        dbx = config.model_copy(deep=True)
        dbx.spark.mode = "databricks"
        spark = get_spark_session(dbx)
        if args.phase == "create":
            uc_upload.create_tables(spark, dbx, SCHEMA_CARD.read_text())
        table, ok = uc_upload.parity_table(
            uc_upload.read_counts(LOCAL_COUNTS), uc_upload.uc_row_counts(spark)
        )
        print(table)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
