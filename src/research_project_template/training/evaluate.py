from __future__ import annotations

import math
from pathlib import Path

import torch
from omegaconf import DictConfig

from research_project_template.data import build_data_module
from research_project_template.methods import build_method
from research_project_template.training.checkpointing import load_checkpoint
from research_project_template.training.utils import resolve_device


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    data_module,
    batch_size: int,
    eval_iters: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}

    for split in ("train", "val"):
        split_losses = []
        for _ in range(eval_iters):
            x, y = data_module.get_batch(split, batch_size=batch_size, device=device)
            _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("Model did not return a loss.")
            split_losses.append(float(loss.item()))
        losses[split] = sum(split_losses) / len(split_losses)

    model.train()
    return losses


def evaluate(cfg: DictConfig) -> dict[str, float]:
    if cfg.checkpoint.path is None:
        raise ValueError("Set checkpoint.path to evaluate a model.")

    device = resolve_device(cfg.training.device)
    data_module = build_data_module(cfg)
    model = build_method(cfg, vocab_size=data_module.vocab_size).to(device)

    checkpoint = load_checkpoint(Path(cfg.checkpoint.path), device=device)
    model.load_state_dict(checkpoint["model"])

    losses = estimate_loss(
        model=model,
        data_module=data_module,
        batch_size=int(cfg.data.eval_batch_size),
        eval_iters=int(cfg.eval_iters),
        device=device,
    )
    losses["perplexity"] = math.exp(losses["val"])
    print(losses)
    return losses
