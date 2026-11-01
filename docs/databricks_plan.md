# Databricks plan: data + eval on Unity Catalog, plus a Genie baseline

Task file for an AI coding agent running autonomously. Read `AGENTS.md` first, especially
the "Databricks work: autopilot rules" section. All of those rules apply here.

## Goal (8-hour budget)

Move **data and evaluation** to Databricks Unity Catalog and add **Databricks AI/BI Genie**
as a fifth baseline, so the README can answer: *when does a fine-tuned 3B model beat the
platform's built-in Text2SQL?*

**Training and local inference stay on the Mac (mlx-lm).** The eval harness runs model
inference locally and executes SQL on Databricks via **Databricks Connect**. A HF/CUDA
training backend, Model Serving, and Asset Bundles are out of scope (see "Next phase").

## How to work

- Do the steps in order. Each step has a **Done when** check. Don't start the next step
  until it passes.
- After each step, run `uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest`,
  then commit with a conventional-commit message. One step, one or more commits; never batch steps.
- **`STOP (human)`** means: commit, write a short status note at the bottom of this file
  under "Run log", and stop. Don't work around it.
- If the same error happens twice (auth, permissions, Connect version mismatch, cluster or
  warehouse unavailable), stop and log it. Don't invent workarounds such as falling back to
  local Spark and reporting it as Databricks.
- Never push. The human reviews and pushes.

## Prerequisites — STOP (human) if any are missing

The human provides these through environment variables or `.env` (gitignored), never in git:

- `DATABRICKS_HOST`, plus auth through a `DATABRICKS_CONFIG_PROFILE` or OAuth (`databricks auth login`)
- `DATABRICKS_CLUSTER_ID` or serverless compute enabled for Databricks Connect
- `DATABRICKS_WAREHOUSE_ID` (SQL warehouse, needed for Genie)
- A catalog the human can write to. Its name goes in `config/base.yaml: catalog`
- The local silver Delta tables exist under `spark.warehouse_dir` (`./spark-warehouse`)

---

## Step 1 — Spark session that chooses local or Databricks (≈1 hr)

- Add `spark.mode: local | databricks` to `config/base.yaml` (default `local`) and the pydantic config.
- In `src/t2s/common/spark.py`, keep the local Delta builder unchanged. For `databricks`, build the session
  with `databricks.connect.DatabricksSession.builder.getOrCreate()`, then
  `USE CATALOG <catalog>` and `USE SCHEMA <schema>`.
  - Setting the current catalog/schema **lets the existing gold SQL (unqualified table names)
    run unchanged.** Don't rewrite SQL in `data/splits/*.jsonl`.
- Make `register_silver_tables` (in `synth/execute.py`, and wherever the eval harness registers views) a no-op in
  `databricks` mode, because the tables already exist in UC under the same names.
- Add `databricks-connect` as an optional dependency group (`[project.optional-dependencies] databricks`).
  Pin it to the workspace's runtime version. Write the reason in the commit message.
- Unit tests: mode selection and config validation, with the Databricks session mocked. No network in tests.

**Done when:** tests pass, and `uv run python -c "...get_spark_session(cfg).sql('select current_catalog(), current_schema()').show()"`
prints the configured catalog and schema. If Connect can't reach the workspace: **STOP (human).**

## Step 2 — Put silver tables in Unity Catalog (≈1 hr)

- Add `scripts/upload_silver.py` (logic in `src/t2s/ingest/uc_upload.py`):
  1. For each name in `SILVER_TABLES`, read the local Delta table with **local** Spark and
     write it to UC. Prefer `df.write.saveAsTable(...)` through the Connect session if the data
     is small enough. Otherwise copy the Delta folder to a Volume
     (`/Volumes/<catalog>/<schema>/raw/silver/<table>`) with the Databricks SDK and run
     `CREATE OR REPLACE TABLE <table> AS SELECT * FROM delta.\`<volume path>\``.
  2. Apply column comments from `data/schema_card.md` (`ALTER TABLE ... ALTER COLUMN ... COMMENT`).
     These help Genie and serve as UC documentation.
- **Don't re-download from USAspending.** The data has to be the same rows the test set was built on.

**Done when:** a parity check prints a table of `table | local_rows | uc_rows` with every row
equal, and the agent saves that output in the Run log.

## Step 3 — Re-run all four baselines against UC (≈1 hr)

- Run `scripts/eval.py --model {base_zero_shot,base_few_shot,frontier,finetuned}` with `spark.mode: databricks`.
- Record the execution backend (`local_spark` or `databricks_uc`) in every results JSON and in MLflow params.
  Add the field to the results model, with a test.
- The old `base_zero_shot` result (commit `1b6e4f9`) failed on markdown code fences that the
  current `postprocess.py` strips. Its 0% score is a parsing artifact and is expected to change.
- If the frontier baseline's credentials aren't present, mark it "not yet run" in the README. Never copy the old number.

**Done when:** four new `results/*.json` files exist with `execution_backend: databricks_uc`, and
fine-tuned/few-shot EX is within ±1 question of the local run. If it differs by more than that:
**STOP (human)**. A difference means the data doesn't match, and that has to be explained before going on.

**STOP (human): review the raw predictions** for zero-shot and the 3 `join` questions before Step 4.

## Step 4 — Genie baseline (≈2 hrs)

Part A — **human:** create the Genie space in the UI over the silver tables, using the SQL warehouse.
Write the space's instructions from `data/schema_card.md` **without looking at
`data/splits/test.jsonl`** (rule: no tuning against the test set). The human puts the space ID in `.env` as `GENIE_SPACE_ID`.
→ **STOP (human)** until `GENIE_SPACE_ID` is set.

Part B — agent:
- Add `GeniePredictor` in `src/t2s/eval/predictors.py` behind the existing `Predictor` protocol.
  It uses the Databricks SDK Genie Conversation API (`w.genie.start_conversation_and_wait`), starts
  a **fresh conversation per question**, and pulls the generated SQL from the query attachment. Latency is measured from the
  start call to the completed message.
- Add `genie` to the `--model` choices in `scripts/eval.py`. Results go through the existing harness and EX metric.
  Genie's SQL is executed by our harness, not taken from Genie's own result rows.
- Handle rate limits with backoff. A question where Genie returns no SQL counts as invalid, not as skipped.
- Record the space ID, warehouse ID and date in the results JSON. Prompt hash = the hash of the exported space instructions, if available. Otherwise null and the reason is noted.
- Tests: SQL extraction from a mocked Genie response (text-only answer, query attachment, error).

**Done when:** `results/genie.json` exists for all 27 questions, and `report.py` shows Genie as a row.

## Step 5 — LLM judge for SQL equivalence (≈1.5 hrs)

- Add a secondary metric `judge_equivalent` using `mlflow.genai` (`make_judge`), with a versioned prompt file
  `src/t2s/common/prompts/sql_equivalence_judge_v1.md`. **Execution accuracy stays the primary metric.**
  The judge is reported next to it and never replaces it.
- Judge model is configured in `config/eval.yaml`, and its exact ID and date are recorded.
- Compute it from existing `results/*.json` records (gold SQL + predicted SQL + both result previews).
  Don't regenerate predictions.
- Calibration: write `results/judge_calibration.json` comparing judge vs EX on every record, and list the
  disagreements. **STOP (human)**: the human hand-labels the disagreements (expected to be a handful, including the 3 joins) before the
  judge column goes into the README.

**Done when:** calibration file exists and the human has signed off in the Run log.

## Step 6 — Report and README (≈1 hr)

- Extend `scripts/report.py`: Genie row, `judge_equivalent` column, `execution_backend` note.
- Add a README section "Fine-tune vs Genie vs frontier" generated by `report.py` from the results only. It shows correct-answer counts
  (e.g. 13/27) next to percentages, and a sample-size caveat: with n=27, a 2-question difference is not significant.
- Update "Limitations" and add "Next phase": HF/PEFT CUDA training backend (`backend: mlx | hf`),
  restore the 7B model, Model Serving + AI Gateway, Asset Bundle pipeline, larger verified test set.

**Done when:** README regenerated, all checks green, and **STOP (human)** for diff review and push.

---

## Next phase (not in this run)

- `backend: mlx | hf` training/inference switch (transformers + PEFT + TRL, bitsandbytes 4-bit) on a GPU cluster
- Qwen2.5-7B
- Model Serving endpoint with AI Gateway inference tables; `ai_query()` batch eval
- Lakeflow Jobs via Databricks Asset Bundle: ingest → synth → train → eval
- Larger and hand-verified (`test_verified`) test set, possibly with the full agency scope

## Run log

<!-- Agent: append one dated entry per step: what changed, check output, commit SHA, anything that needs the human. -->
