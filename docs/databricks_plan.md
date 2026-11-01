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

### 2026-11-01 — Step 1 (code done; live check BLOCKED) — STOP (human)

- Added `spark.mode: local | databricks` (+ validator: databricks requires `catalog`), lazy
  Databricks Connect session that runs `USE CATALOG/SCHEMA`, and a no-op `register_silver_tables`
  in databricks mode. Tests in `tests/test_spark_mode.py` (mocked, no network).
- Checks: ruff, ruff format, mypy --strict, pytest (80 tests) all pass in the local env.
- Setup facts: workspace is Free Edition (serverless only, no clusters/warehouses); catalog
  `workspace`; schema `usaspending` created via CLI. Auth via CLI OAuth profile, set in `.env`
  (`DATABRICKS_CONFIG_PROFILE`, `DATABRICKS_SERVERLESS_COMPUTE_ID=auto`).
- **Env split:** `databricks-connect` ships its own `pyspark` and clobbers the real one, and it needs
  Python 3.12. It lives in a separate venv (`.venv-dbx`, gitignored): `uv venv .venv-dbx --python 3.12`,
  `uv pip install --python .venv-dbx/bin/python -e . "databricks-connect>=17,<20"`, then
  `uv pip uninstall pyspark delta-spark` and `uv pip install --reinstall --no-deps databricks-connect==19.3.0`.
  Consequence for Step 2: local Spark (read) and Connect (write) can't share a process; use the
  plan's Volume-upload path or a two-process flow.
- **Blocker (hit twice):** Connect authenticates and reaches the workspace, but every session fails with
  `SERVERLESS_COMPUTE_EXHAUSTED` (Free Edition serverless quota). Not retrying or falling back to local.
  Human: check the usage/quota in the workspace, wait for the quota to reset, or provide other compute,
  then re-run the Step 1 live check with `.venv-dbx` and `spark.mode: databricks`.

### 2026-11-01 (later) — Step 1 live check PASSED — STOP (human)

- Moved to a new Free Edition workspace (the old one stayed `SERVERLESS_COMPUTE_EXHAUSTED`). Host and
  CLI profile (`t2s-new`) are set in `.env` only. Created schema `workspace.usaspending`.
- Check (`.venv-dbx`, `spark.mode: databricks`): `current_catalog() = workspace`, `current_schema() = usaspending`.
- Human: review and commit this note, then Step 2 (silver tables to UC) can begin.

### 2026-11-01 — Step 2 done (uncommitted)

- Added `src/t2s/ingest/uc_upload.py`, `scripts/upload_silver.py`, `tests/test_uc_upload.py`. Flow: local Spark
  counts (main env) -> SDK copies Delta folders to Volume `workspace.usaspending.raw` (`/silver/<table>`) ->
  `CREATE OR REPLACE TABLE ... AS SELECT * FROM delta.<path>` through Connect (`.venv-dbx`) -> column comments
  from `data/schema_card.md`. No re-download; same local rows.
- Parity check output:

```
table | local_rows | uc_rows | match
award_transactions | 333044 | 333044 | OK
agencies | 6 | 6 | OK
recipients | 18916 | 18916 | OK
locations | 4165 | 4165 | OK
```

- Checks: ruff, ruff format, mypy --strict, pytest (82) pass. Spot-checked UC column comments on `agencies`.
- Next: Step 3 (re-run four baselines against UC) is the following step; not started.

### 2026-11-01 — Step 3 done (uncommitted) — STOP (human): review raw predictions

- Added `execution_backend` (`local_spark | databricks_uc`) to `EvalSummary`, `run_eval`, results JSON and MLflow
  params; `scripts/eval.py --spark-mode databricks` (explicit, no implicit fallback). UC runs write
  `results/<model>_databricks_uc.json` so the local results are kept. `report.py` keys/labels UC runs separately
  (`[databricks_uc]`). Tests added in `tests/test_eval_harness.py` (84 pass; ruff, mypy clean).
- Env: installed `mlx-lm`, `openai`, `azure-identity` into `.venv-dbx` so eval runs in the Connect process.
- Execution-match counts out of 27 (local vs UC):

| run | local | databricks_uc |
|---|---|---|
| finetuned | 13 | 13 |
| base_few_shot | 11 | 11 |
| base_zero_shot | 0 | 2 (old 0 was the code-fence parsing artifact) |
| frontier | 18 | 19 (LLM re-sampled; within ±1) |

- Fine-tuned and few-shot are identical to local, so the UC data matches. README table regenerated by `report.py`.
- **STOP (human):** review raw predictions for zero-shot and the 3 `join` questions before Step 4.

### 2026-11-01 — Step 4 done

- Part A (human): Genie space `01f1d63466be1114918bfc5c3e40f708`, warehouse `3de5b3a375d6030e`, `GENIE_SPACE_ID` in `.env`.
  Agent did not write any space content and did not read the test set to configure it.
- Part B: `GeniePredictor` (fresh conversation per question, SQL from query attachment, backoff on
  `TooManyRequests`/`ResourceExhausted`, no SQL = invalid), `--model genie` (requires `--spark-mode databricks`),
  `run_metadata` in the results JSON (space ID, warehouse ID, date, tables). Tests: `tests/test_genie_predictor.py`.
  92 tests, ruff, mypy pass.
- Result: `results/genie.json`, 27/27 questions, EX 55.6% (15/27), valid SQL 100%, exact match 0%, p50 14.0s, p95 19.0s.
  `report.py` shows Genie as a row.
- Notes for the human: the space exposes only `agencies`, `award_transactions`, `recipients`
  (no `locations`) and has no instructions, so `prompt_sha256` is null (reason recorded in the results JSON).
  Genie's SQL is executed by our harness against UC, same EX metric as the other baselines.

### 2026-11-01 — Step 5 code + calibration file done — STOP (human): hand-label disagreements

- Added `src/t2s/eval/judge.py`, `scripts/judge.py`, prompt `src/t2s/common/prompts/sql_equivalence_judge_v1.md`
  (hash recorded), judge config in `config/eval.yaml`, tests (`tests/test_eval_judge.py`; 96 pass, ruff, mypy clean).
  `report.py` now skips non-run JSON files.
- Judge: `databricks:/databricks-gpt-oss-120b` via `mlflow.genai` `make_judge`, run 2026-11-01. It is not one of the
  evaluated models. Inputs are the question, gold and predicted SQL, and result previews (first 5 rows + row count)
  from executing both on UC. Predictions were not regenerated; it reads the `*_databricks_uc.json` files plus `genie.json`.
  The judge never sees the EX verdict. EX stays the primary metric.
- Output `results/judge_calibration.json` (every record, judge vs EX, 16 disagreements with empty `human_label`).

| run | EX correct | judge equivalent | agreement |
|---|---|---|---|
| base_zero_shot | 2/27 | 4/27 | 92.6% |
| base_few_shot | 11/27 | 13/27 | 92.6% |
| frontier | 19/27 | 25/27 | 77.8% |
| finetuned | 13/27 | 13/27 | 100% |
| genie | 15/27 | 21/27 | 77.8% |

- All 16 disagreements are judge=equivalent / EX=not-equivalent (none the other way); the 3 `join` questions are
  not among them. Frontier and Genie account for 12 of them, so the judge may favour those runs if wrong.
  Agent did not label anything.
- **STOP (human):** fill `human_label` (true = predicted answer is really equivalent) and `human_note` for each entry in
  `disagreements`, set `human_signoff`, and add a sign-off line here. The judge column stays out of the README until then.

### 2026-11-01 — Step 5 human sign-off

- Max Lacy reviewed all 16 disagreements in `results/judge_calibration.json` and labelled every one
  `human_label: true` (predicted answer equivalent; the judge was right, EX was a false negative). `human_signoff` set.
- Pattern in the 16: extra/missing columns, aliases, DISTINCT, redundant NOT NULL filters, RANK vs DENSE_RANK. EX
  (which compares full result sets) scores these as wrong; they stay wrong under EX, which remains the primary metric.
  Per this calibration, judge_equivalent is trustworthy on these cases (16/16), so it can go into the README in Step 6
  as a secondary column with that caveat. The blanket label means no per-case notes were written.

### 2026-11-01 — Step 6 done — STOP (human): diff review and push

- Report logic moved to `src/t2s/eval/report_tables.py` (tested in `tests/test_report_tables.py`); `scripts/report.py`
  is now a thin entry point. The README results block is generated from `results/*.json` only: Databricks UC table with
  counts next to percentages, a `judge_equivalent` column (shown only because the human signed off), per-tag EX,
  a "Fine-tune vs Genie vs frontier" section with the n=27 caveat, and the earlier local-Spark runs labelled by
  `execution_backend`.
- README prose updated by hand (not the generated block): status, Limitations (judge checked only on disagreements,
  Genie space missing `locations`/instructions, frontier re-sampling, Free Edition latency), and "Next phase".
- Checks: ruff, ruff format, mypy --strict, pytest (100) all pass. Committed locally; not pushed.

### Post-plan: execute-and-repair experiment (2026-11-01)
- Added `--self-correct N` to `scripts/eval.py` (`src/t2s/eval/self_correct.py`, prompt `t2s_sql_repair_v1.md`).
  The loop sees only the model's SQL and the engine error, never gold SQL or results.
- Fine-tuned model on UC, up to 2 retries: EX 13/27 (unchanged), valid SQL 77.8% -> 81.5% (+1 query),
  p95 latency 4.6s -> 9.1s. The repair loop did not move the headline metric at n=27.
- Frontier clients now record token usage; cost per 1K stays "not yet measured" until published prices are
  set in `config/eval.yaml`.
