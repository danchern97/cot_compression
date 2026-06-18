from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from research_project_template.data.text import ensure_text_file, load_text, split_text
from research_project_template.data.tokenizer import CharTokenizer


def prepare_data(cfg: DictConfig) -> Path:
    """Write simple prepared artifacts that students can inspect or replace."""

    source_path = Path(cfg.data.source_path)
    ensure_text_file(
        path=source_path,
        download_url=cfg.data.get("download_url"),
        allow_download=bool(cfg.data.allow_download),
    )

    text = load_text(source_path)
    tokenizer = CharTokenizer.from_text(text)
    train_text, val_text = split_text(
        text=text, train_split=float(cfg.data.train_split)
    )

    prepared_dir = Path(cfg.data.prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    np.save(prepared_dir / "train.npy", np.array(tokenizer.encode(train_text)))
    np.save(prepared_dir / "val.npy", np.array(tokenizer.encode(val_text)))
    (prepared_dir / "tokenizer.json").write_text(
        json.dumps(tokenizer.to_dict(), indent=2),
        encoding="utf-8",
    )
    return prepared_dir
