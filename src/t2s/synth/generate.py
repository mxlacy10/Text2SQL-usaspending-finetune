"""Generate candidate question/SQL pairs from the schema card via a frontier model."""

from __future__ import annotations

import json
import re

from t2s.common.prompt_loader import load_prompt
from t2s.synth.llm import FrontierClient
from t2s.synth.models import CandidateExample, DifficultyTag

PROMPT_NAME = "synth_question_sql_v1.md"


def build_generation_prompt(schema_card: str, tag: DifficultyTag, n: int) -> str:
    """Fill the versioned prompt template for one difficulty tag's batch."""
    template = load_prompt(PROMPT_NAME).text
    return template.format(schema_card=schema_card, tag=tag, n=n)


def parse_candidates(raw_response: str) -> list[CandidateExample]:
    """Parse the model's JSON array response into CandidateExample records.

    Tolerant of markdown code fences around the JSON array, since models
    sometimes wrap output that way despite being asked not to.
    """
    text = raw_response.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    items = json.loads(text)
    return [CandidateExample.model_validate(item) for item in items]


def generate_candidates(
    client: FrontierClient,
    schema_card: str,
    tags: list[DifficultyTag],
    n_per_tag: int,
) -> list[CandidateExample]:
    """Generate n_per_tag candidates for each difficulty tag."""
    candidates: list[CandidateExample] = []
    for tag in tags:
        prompt = build_generation_prompt(schema_card, tag, n_per_tag)
        response = client.generate(prompt)
        candidates.extend(parse_candidates(response))
    return candidates
