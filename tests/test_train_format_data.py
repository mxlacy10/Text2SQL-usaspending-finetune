import json
from pathlib import Path

from t2s.train.format_data import build_chat_record, convert_split_file, convert_splits_for_mlx


def test_build_chat_record_shape() -> None:
    record = build_chat_record("system text", "How many awards?", "SELECT COUNT(*) FROM t")
    assert record["messages"] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "How many awards?"},
        {"role": "assistant", "content": "SELECT COUNT(*) FROM t"},
    ]


def test_convert_split_file_writes_one_record_per_line(tmp_path: Path) -> None:
    src = tmp_path / "train.jsonl"
    src.write_text(
        json.dumps({"question": "q1", "sql": "s1", "tags": ["single_table"], "split": "train"})
        + "\n"
        + json.dumps({"question": "q2", "sql": "s2", "tags": ["join"], "split": "train"})
        + "\n"
    )
    dst = tmp_path / "out.jsonl"
    count = convert_split_file(src, dst, "sys")
    assert count == 2
    lines = dst.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["messages"][1]["content"] == "q1"
    assert first["messages"][2]["content"] == "s1"


def test_convert_splits_for_mlx_renames_dev_to_valid(tmp_path: Path) -> None:
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    example = {"question": "q", "sql": "s", "tags": ["single_table"], "split": "train"}
    for name in ("train", "dev", "test"):
        (splits_dir / f"{name}.jsonl").write_text(json.dumps(example) + "\n")

    out_dir = tmp_path / "mlx_train"
    prompt, counts = convert_splits_for_mlx(splits_dir, out_dir)

    assert prompt.name == "t2s_sql_system_v1.md"
    assert counts == {"train": 1, "dev": 1, "test": 1}
    assert (out_dir / "train.jsonl").exists()
    assert (out_dir / "valid.jsonl").exists()
    assert (out_dir / "test.jsonl").exists()
