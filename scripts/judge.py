"""Compute the secondary LLM-judge metric `judge_equivalent` from existing results.

Reads the Databricks-UC run files (plus Genie) in results/, never regenerates predictions.
For every record it executes gold and predicted SQL on Unity Catalog to build result previews,
asks the judge (mlflow.genai make_judge) whether the predicted query is equivalent, and writes
results/judge_calibration.json: judge vs EX on every record plus the disagreements for a human
to hand-label. Run in the Databricks Connect env (.venv-dbx):

  .venv-dbx/bin/python scripts/judge.py --config config/eval.yaml
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any

import mlflow
from mlflow.genai.judges import make_judge

from t2s.common.config import load_base_config
from t2s.common.mlflow_utils import init_mlflow
from t2s.common.prompt_loader import load_prompt
from t2s.common.spark import get_spark_session
from t2s.eval.config import load_eval_config
from t2s.eval.execute import execute_sql
from t2s.eval.judge import (
    JudgeRecord,
    agreement_stats,
    find_disagreements,
    format_preview,
    parse_verdict,
)

logger = logging.getLogger(__name__)

RUNS = ["base_zero_shot", "base_few_shot", "frontier", "finetuned", "genie"]
OUT_PATH = Path("results/judge_calibration.json")


def _result_file(results_dir: Path, run: str) -> Path:
    return results_dir / (f"{run}.json" if run == "genie" else f"{run}_databricks_uc.json")


def _judge_once(judge: Any, **kwargs: Any) -> Any:
    for attempt in range(4):
        try:
            return judge(**kwargs)
        except Exception as e:  # noqa: BLE001 - transient endpoint errors; give up after retries
            if attempt == 3:
                logger.warning("judge failed: %s", e)
                return None
            time.sleep(2 * 2**attempt)
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/eval.yaml")
    parser.add_argument("--base-config", default="config/base.yaml")
    args = parser.parse_args()

    eval_config = load_eval_config(args.config)
    base_config = load_base_config(args.base_config)
    base_config.spark.mode = "databricks"
    prompt = load_prompt(eval_config.judge_prompt)
    judge = make_judge(
        name="sql_equivalence", instructions=prompt.text, model=eval_config.judge_model
    )
    spark = get_spark_session(base_config)
    limit = eval_config.judge_preview_rows
    gold_cache: dict[str, str] = {}

    def preview(sql: str) -> str:
        res = execute_sql(spark, sql)
        return format_preview(res.rows, res.error, limit)

    results_dir = Path(eval_config.results_dir)
    runs_out: dict[str, Any] = {}
    all_disagreements = []
    try:
        for run in RUNS:
            data = json.loads(_result_file(results_dir, run).read_text())
            records: list[JudgeRecord] = []
            for r in data["records"]:
                if r["gold_sql"] not in gold_cache:
                    gold_cache[r["gold_sql"]] = preview(r["gold_sql"])
                pred_preview = (
                    preview(r["predicted_sql"]) if r["predicted_sql"].strip() else "ERROR: no SQL"
                )
                feedback = _judge_once(
                    judge,
                    inputs={
                        "question": r["question"],
                        "predicted_result_preview": pred_preview,
                    },
                    outputs=r["predicted_sql"] or "(no SQL produced)",
                    expectations={
                        "gold_sql": r["gold_sql"],
                        "gold_result_preview": gold_cache[r["gold_sql"]],
                    },
                )
                verdict = parse_verdict(feedback.value) if feedback is not None else None
                records.append(
                    JudgeRecord(
                        question=r["question"],
                        tags=list(r["tags"]),
                        execution_match=r["execution_match"],
                        judge_equivalent=verdict,
                        judge_rationale=feedback.rationale if feedback is not None else None,
                        predicted_sql=r["predicted_sql"],
                        gold_sql=r["gold_sql"],
                    )
                )
            stats = agreement_stats(records)
            logger.info("%s: %s", run, stats)
            runs_out[run] = {
                "source_result_file": str(_result_file(results_dir, run)),
                "source_git_commit": data.get("git_commit"),
                "stats": stats,
                "records": [rec.model_dump() for rec in records],
            }
            all_disagreements += find_disagreements(run, records)
    finally:
        spark.stop()

    out = {
        "judge_model": eval_config.judge_model,
        "judge_prompt": eval_config.judge_prompt,
        "judge_prompt_sha256": prompt.sha256,
        "judge_date": datetime.date.today().isoformat(),
        "primary_metric": "execution_accuracy (the judge never replaces it)",
        "human_signoff": None,
        "runs": runs_out,
        "disagreements": [d.model_dump() for d in all_disagreements],
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s (%d disagreements)", OUT_PATH, len(all_disagreements))

    init_mlflow(base_config.mlflow)
    with mlflow.start_run(run_name="judge-sql-equivalence"):
        mlflow.log_params(
            {
                "judge_model": eval_config.judge_model,
                "judge_prompt_sha256": prompt.sha256,
                "judge_date": out["judge_date"],
            }
        )
        for run, info in runs_out.items():
            mlflow.log_metric(f"{run}_judge_ex_agreement", info["stats"]["agreement"])
        mlflow.log_artifact(str(OUT_PATH))


if __name__ == "__main__":
    main()
