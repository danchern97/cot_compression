from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from research_project_template.data.tokenizer import CharTokenizer
from research_project_template.methods import build_method
from research_project_template.methods.gpt import TinyGPT


def save_checkpoint(
    path: Path,
    model: TinyGPT,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    best_val_loss: float,
    cfg: DictConfig,
    tokenizer: CharTokenizer,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "best_val_loss": best_val_loss,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "tokenizer": tokenizer.to_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    return torch.load(path, map_location=device)


def load_model_from_checkpoint(
    path: Path, device: torch.device
) -> tuple[TinyGPT, dict]:
    checkpoint = load_checkpoint(path=path, device=device)
    cfg = OmegaConf.create(checkpoint["cfg"])
    tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])
    model = build_method(cfg, vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint
