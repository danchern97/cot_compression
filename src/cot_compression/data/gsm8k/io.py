"""Read source examples and save workflow records."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

Record = dict[str, Any]
RecordsById = dict[str, Record]


def read_jsonl(path: Path) -> list[Record]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def index_by_source_id(rows: list[Record]) -> RecordsById:
    """Index rows after checking that every source ID is unique."""
    result = {row["source_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate source_id in input")
    return result


def write_jsonl(path: Path, rows: Iterable[Record]) -> None:
    """Replace a JSONL file only after writing every row successfully."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    tmp.replace(path)


def load_source(path: Path) -> RecordsById:
    """Read GSM8K and use its row indices for IDs when none are supplied."""
    return index_by_source_id(
        [
            {
                "source_id": row.get("source_id", f"gsm8k/train/{i}"),
                "question": row["question"],
                "answer": row["answer"],
            }
            for i, row in enumerate(read_jsonl(path))
        ]
    )
