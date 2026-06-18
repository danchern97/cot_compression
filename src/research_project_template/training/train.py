from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from omegaconf import DictConfig

from research_project_template.data import build_data_module
from research_project_template.methods import build_method
from research_project_template.training.checkpointing import save_checkpoint
from research_project_template.training.evaluate import estimate_loss
from research_project_template.training.logging import RunLogger
from research_project_template.training.utils import (
    get_run_dir,
    resolve_device,
    save_resolved_config,
    set_seed,
)


def build_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.optim.lr),
        betas=(float(cfg.optim.beta1), float(cfg.optim.beta2)),
        eps=float(cfg.optim.eps),
        weight_decay=float(cfg.optim.weight_decay),
    )


def train(cfg: DictConfig) -> Path:
    run_dir = get_run_dir(cfg)
    save_resolved_config(cfg, run_dir)
    logger = RunLogger(cfg=cfg, run_dir=run_dir)

    set_seed(
        seed=int(cfg.training.seed), deterministic=bool(cfg.training.deterministic)
    )
    device = resolve_device(cfg.training.device)
    logger.info(f"Using device: {device}")

    data_module = build_data_module(cfg)
    model = build_method(cfg, vocab_size=data_module.vocab_size).to(device)
    optimizer = build_optimizer(cfg, model)
    active_model: torch.nn.Module = model

    if cfg.training.compile:
        active_model = cast(torch.nn.Module, torch.compile(model))

    best_val_loss = float("inf")
    last_checkpoint = run_dir / "checkpoints" / "last.pt"

    try:
        for step in range(1, int(cfg.training.max_steps) + 1):
            # 1. Fetch a random language-modeling batch.
            x, y = data_module.get_batch(
                "train",
                batch_size=int(cfg.data.batch_size),
                device=device,
            )

            # 2. Run the model and compute next-character loss.
            _, loss = active_model(x, y)
            if loss is None:
                raise RuntimeError("Model did not return a loss.")

            # 3. Backpropagate and update parameters.
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(cfg.training.gradient_clip) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(cfg.training.gradient_clip),
                )
            optimizer.step()

            # 4. Print and wandb-log lightweight training metrics.
            if step % int(cfg.training.log_interval) == 0:
                logger.log_metrics({"train/loss": float(loss.item())}, step=step)

            # 5. Run validation with the same explicit evaluation function.
            if step % int(cfg.training.eval_interval) == 0:
                losses = estimate_loss(
                    model=active_model,
                    data_module=data_module,
                    batch_size=int(cfg.data.eval_batch_size),
                    eval_iters=int(cfg.training.eval_iters),
                    device=device,
                )
                logger.log_metrics(
                    {
                        "eval/train_loss": losses["train"],
                        "eval/val_loss": losses["val"],
                    },
                    step=step,
                )

                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    save_checkpoint(
                        run_dir / "checkpoints" / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        best_val_loss=best_val_loss,
                        cfg=cfg,
                        tokenizer=data_module.tokenizer,
                    )

            # 6. Save periodic artifacts in the run directory.
            if step % int(cfg.training.checkpoint_interval) == 0:
                save_checkpoint(
                    last_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_val_loss=best_val_loss,
                    cfg=cfg,
                    tokenizer=data_module.tokenizer,
                )

        save_checkpoint(
            last_checkpoint,
            model=model,
            optimizer=optimizer,
            step=int(cfg.training.max_steps),
            best_val_loss=best_val_loss,
            cfg=cfg,
            tokenizer=data_module.tokenizer,
        )
        logger.info(f"Saved final checkpoint to {last_checkpoint}")
        return last_checkpoint
    finally:
        logger.finish()
