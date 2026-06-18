from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig

from research_project_template.data.text import ensure_text_file, load_text, split_text
from research_project_template.data.tokenizer import CharTokenizer


@dataclass
class CharDataModule:
    train_ids: torch.Tensor
    val_ids: torch.Tensor
    tokenizer: CharTokenizer
    block_size: int

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def get_batch(
        self,
        split: str,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_ids if split == "train" else self.val_ids
        max_start = len(data) - self.block_size - 1
        if max_start < 1:
            raise ValueError("Dataset is too short for the configured block_size.")

        starts = torch.randint(max_start, (batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in starts])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in starts])
        return x.to(device), y.to(device)


def build_data_module(cfg: DictConfig) -> CharDataModule:
    path = Path(cfg.data.source_path)
    ensure_text_file(
        path=path,
        download_url=cfg.data.get("download_url"),
        allow_download=bool(cfg.data.allow_download),
    )

    text = load_text(path)
    tokenizer = CharTokenizer.from_text(text)
    train_text, val_text = split_text(
        text=text, train_split=float(cfg.data.train_split)
    )

    train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_ids = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    return CharDataModule(
        train_ids=train_ids,
        val_ids=val_ids,
        tokenizer=tokenizer,
        block_size=int(cfg.data.block_size),
    )
