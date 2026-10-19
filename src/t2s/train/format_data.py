"""Convert data/splits/*.jsonl (our Example schema) into the mlx-lm `chat`
JSONL format expected by `mlx_lm.lora` (see mlx_lm LORA.md: "chat" dataset,
final message is the completion, train with --mask-prompt).

The fine-tuned model is evaluated zero-shot (AGENTS.md eval baselines): no
schema card is included in the prompt at train or inference time. The model
learns the schema implicitly from many (question, SQL) pairs during training.
"""

from __future__ import annotations

import json
from pathlib import Path

from t2s.common.prompt_loader import LoadedPrompt, load_prompt

SYSTEM_PROMPT_NAME = "t2s_sql_system_v1.md"


def build_chat_record(system_prompt: str, question: str, sql: str) -> dict[str, object]:
    """Build one mlx-lm `chat` format JSONL record for a single example."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": sql},
        ]
    }


def convert_split_file(src_path: Path, dst_path: Path, system_prompt: str) -> int:
    """Convert one data/splits/<split>.jsonl file to mlx-lm chat format.

    Returns the number of records written.
    """
    count = 0
    with src_path.open() as src, dst_path.open("w") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            record = build_chat_record(system_prompt, example["question"], example["sql"])
            dst.write(json.dumps(record) + "\n")
            count += 1
    return count


def convert_splits_for_mlx(
    splits_dir: Path, out_dir: Path, prompt_name: str = SYSTEM_PROMPT_NAME
) -> tuple[LoadedPrompt, dict[str, int]]:
    """Convert data/splits/{train,dev,test}.jsonl into out_dir in mlx-lm's
    expected layout: train.jsonl, valid.jsonl (from dev.jsonl), test.jsonl.

    Returns the loaded system prompt (for run logging: name + sha256) and a
    dict of split name -> record count written.
    """
    prompt = load_prompt(prompt_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # mlx-lm expects "valid.jsonl", we call it "dev" elsewhere in the project.
    name_map = {"train": "train.jsonl", "dev": "valid.jsonl", "test": "test.jsonl"}
    counts: dict[str, int] = {}
    for split_name, out_name in name_map.items():
        src_path = splits_dir / f"{split_name}.jsonl"
        if not src_path.exists():
            continue
        counts[split_name] = convert_split_file(src_path, out_dir / out_name, prompt.text)
    return prompt, counts
