# Running this in a federal environment

Design notes, not a certified architecture. Nothing here has been deployed or audited.

## Data and governance
- USAspending data is public, but the same pattern applies to controlled data. Tables live in Unity Catalog
  (`workspace.usaspending`), with column comments from `data/schema_card.md`.
- Access control is table-level grants in Unity Catalog; Genie and the eval harness only see tables the
  running identity can read. Every query is attributable to that identity.
- Prompts contain the schema card and sample questions, not row data. Result previews are only sent to the
  LLM judge, so for sensitive data the judge must be a model hosted inside the boundary.

## Model choice
| Option | Data leaves the boundary? | Measured here (n=27, EX) |
|---|---|---|
| Frontier API (gpt-5-chat on Azure AI Foundry) | Only to the tenant's Azure endpoint | 19/27 |
| Genie on a Databricks SQL warehouse | No (inside the workspace) | 15/27 |
| Self-hosted fine-tuned 3B model | No | 13/27 |

A self-hosted open model removes the external-API dependency at a cost of about 6 questions of accuracy on this
test set. With n=27 the differences are indicative only.

## Platform and compliance questions to settle with the customer
- Cloud and region: Azure Government / AWS GovCloud, and which Databricks offering is authorized there
  (FedRAMP / IL level). Not verified in this repo; confirm with Databricks and the customer's ATO team.
- Which model endpoints are available in that environment (Foundation Model APIs, Model Serving, external models).
- Network: private link, no public egress from compute, audit-log delivery.
- Clearance and on-site constraints affect who can see data, prompts and logs.

## Evaluation in production
- Keep the execution-accuracy harness as a regression gate on every prompt/model change.
- Log model ID, prompt hash, dataset version and git commit per run (already done in `results/*.json`).
- Use a human-labeled sample to calibrate any LLM judge before reporting it.
