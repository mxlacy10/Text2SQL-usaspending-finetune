# Customer readout: Text2SQL over federal spending data

Audience: a program or data team deciding how to give analysts natural-language access to award data.
All numbers are from committed `results/*.json` (27-question frozen test set, Unity Catalog execution).

## Question
Which approach should we use to turn questions into SQL: a managed product (Genie), a frontier model with
prompting, or a small fine-tuned model?

## Findings
| Approach | Execution accuracy | Valid SQL | p50 latency |
|---|---|---|---|
| Frontier model, few-shot | 19/27 (70.4%) | 100% | 1.5s |
| Genie space (no instructions) | 15/27 (55.6%) | 100% | 14.0s |
| Fine-tuned 3B (QLoRA) | 13/27 (48.1%) | 77.8% | 2.3s (laptop) |
| Fine-tuned + execute-and-repair | 13/27 (48.1%) | 81.5% | 2.7s (laptop) |
| Base 3B, few-shot | 11/27 (40.7%) | 77.8% | 20.3s (laptop, long prompt) |

- Strict result matching undercounts: an LLM judge (calibrated against 16 human-reviewed disagreements) rates
  the frontier at 25/27 and Genie at 21/27 as equivalent. Differences are mostly extra columns or aliases.
- Fine-tuning beat the base model's few-shot prompting by 2 questions, which is not statistically meaningful.
- An execute-and-repair loop fixed one invalid query and left accuracy unchanged, at roughly double p95 latency.

## Tradeoffs
- **Frontier**: best accuracy and no invalid SQL; data goes to an external endpoint; per-query token cost
  (not yet measured here).
- **Genie**: no ML work, governed inside the workspace, decent accuracy before any tuning. Its accuracy should
  improve with instructions and benchmarks written by a domain expert (not done here, deliberately).
- **Fine-tuned small model**: runs in-boundary and cheaply, but needs more data and evaluation before it
  can match the others.

## Recommendation
Start with Genie for governed self-service, add curated instructions, and use the eval harness as the gate.
Use a frontier model where policy allows. Pursue fine-tuning only after growing the training set and a
larger, human-verified test set.

## Limits of this evidence
Synthetic frontier-generated questions, one schema, FY2022-2024 for one agency, n=27 (one question is 3.7
points). Cost per 1K queries is not yet measured.
