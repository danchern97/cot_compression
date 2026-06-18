from __future__ import annotations

import torch
from omegaconf import OmegaConf

from research_project_template.data import CharTokenizer, build_data_module


def test_char_tokenizer_round_trip() -> None:
    tokenizer = CharTokenizer.from_text("hello")
    ids = tokenizer.encode("hello")
    assert tokenizer.decode(ids) == "hello"


def test_text_batches_are_shifted(tmp_path) -> None:
    corpus = tmp_path / "input.txt"
    corpus.write_text("abcdefghijklmnopqrstuvwxyz" * 4, encoding="utf-8")
    cfg = OmegaConf.create(
        {
            "data": {
                "source_path": str(corpus),
                "download_url": None,
                "allow_download": False,
                "train_split": 0.8,
                "block_size": 8,
                "batch_size": 4,
                "eval_batch_size": 4,
            }
        }
    )

    data_module = build_data_module(cfg)
    x, y = data_module.get_batch("train", batch_size=4, device=torch.device("cpu"))

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert (x[:, 1:] == y[:, :-1]).all()
