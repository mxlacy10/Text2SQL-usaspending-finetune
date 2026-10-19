"""Load versioned prompt template files and hash them for run logging.

See AGENTS.md: "Prompt templates are versioned files in
src/t2s/common/prompts/. Log their hash with every run."
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass(frozen=True)
class LoadedPrompt:
    name: str
    text: str
    sha256: str


def load_prompt(name: str) -> LoadedPrompt:
    """Load a prompt template by file name (e.g. "synth_question_sql_v1.md")."""
    text = (PROMPTS_DIR / name).read_text()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return LoadedPrompt(name=name, text=text, sha256=digest)
