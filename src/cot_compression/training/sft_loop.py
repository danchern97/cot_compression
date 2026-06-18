from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from cot_compression.training.logging import RunLogger


@dataclass
class LoopState:
    model: Any
    active_model: torch.nn.Module
    tokenizer: Any
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    logger: RunLogger
    cfg: DictConfig
    run_dir: Path
    device: torch.device
    total_steps: int
    global_step: int = 0
    best_eval_loss: float = float("inf")
    accumulated_batches: int = 0

    @property
    def last_checkpoint(self) -> Path:
        return self.run_dir / "checkpoints" / "last"


@dataclass(frozen=True)
class StepResult:
    should_log: bool
    should_eval: bool
    should_checkpoint: bool
    should_stop: bool
    loss: float


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def has_trainable_tokens(batch: dict[str, torch.Tensor]) -> bool:
    return bool((batch["labels"] != -100).any().item())


def save_checkpoint(
    path: Path,
    state: LoopState,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    state.model.save_pretrained(path)
    state.tokenizer.save_pretrained(path)
    torch.save(
        {
            "optimizer": state.optimizer.state_dict(),
            "scheduler": state.scheduler.state_dict(),
            "step": state.global_step,
            "best_eval_loss": state.best_eval_loss,
            "cfg": OmegaConf.to_container(state.cfg, resolve=True),
        },
        path / "training_state.pt",
    )


def load_training_state(
    checkpoint_dir: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[int, float]:
    state_path = checkpoint_dir / "training_state.pt"
    if not state_path.exists():
        return 0, float("inf")

    state = torch.load(state_path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return int(state["step"]), float(state["best_eval_loss"])


def run_batch(
    state: LoopState,
    batch: dict[str, torch.Tensor],
    end_of_epoch: bool,
) -> StepResult | None:
    batch = move_batch(batch, state.device)
    if not has_trainable_tokens(batch):
        state.logger.info("Skipping batch with no trainable assistant tokens.")
        return None

    outputs = state.active_model(**batch)
    loss = outputs.loss
    if loss is None:
        raise RuntimeError("Model did not return a loss.")

    scaled_loss = loss / int(state.cfg.training.gradient_accumulation_steps)
    scaled_loss.backward()
    state.accumulated_batches += 1

    if (
        state.accumulated_batches < int(state.cfg.training.gradient_accumulation_steps)
        and not end_of_epoch
    ):
        return None

    if float(state.cfg.training.gradient_clip) > 0:
        torch.nn.utils.clip_grad_norm_(
            state.model.parameters(),
            max_norm=float(state.cfg.training.gradient_clip),
        )
    state.optimizer.step()
    state.scheduler.step()
    state.optimizer.zero_grad(set_to_none=True)
    state.accumulated_batches = 0
    state.global_step += 1

    return StepResult(
        should_log=state.global_step % int(state.cfg.training.log_interval) == 0,
        should_eval=state.global_step % int(state.cfg.training.eval_interval) == 0,
        should_checkpoint=state.global_step
        % int(state.cfg.training.checkpoint_interval)
        == 0,
        should_stop=state.global_step >= state.total_steps,
        loss=float(loss.item()),
    )


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    eval_iters: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for index, batch in enumerate(loader):
        if index >= eval_iters:
            break
        batch = move_batch(cast(dict[str, torch.Tensor], batch), device)
        if not has_trainable_tokens(batch):
            continue
        outputs = model(**batch)
        losses.append(float(outputs.loss.item()))

    model.train()
    if not losses:
        raise RuntimeError("Evaluation produced no trainable assistant tokens.")
    return sum(losses) / len(losses)


def evaluate_and_checkpoint_if_best(
    state: LoopState,
    eval_loader: DataLoader,
) -> None:
    eval_loss = estimate_loss(
        model=state.active_model,
        loader=eval_loader,
        eval_iters=int(state.cfg.training.eval_iters),
        device=state.device,
    )
    state.logger.log_metrics({"eval/loss": eval_loss}, step=state.global_step)
    if eval_loss < state.best_eval_loss:
        state.best_eval_loss = eval_loss
        save_checkpoint(state.run_dir / "checkpoints" / "best", state)


def train_one_epoch(
    state: LoopState,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    epoch: int,
) -> bool:
    for batch_index, batch in enumerate(train_loader, start=1):
        result = run_batch(
            state=state,
            batch=cast(dict[str, torch.Tensor], batch),
            end_of_epoch=batch_index == len(train_loader),
        )
        if result is None:
            continue

        if result.should_log:
            state.logger.log_metrics(
                {
                    "train/loss": result.loss,
                    "train/lr": state.scheduler.get_last_lr()[0],
                    "train/epoch": float(epoch + 1),
                },
                step=state.global_step,
            )

        if result.should_eval:
            evaluate_and_checkpoint_if_best(state, eval_loader)

        if result.should_checkpoint:
            save_checkpoint(state.last_checkpoint, state)

        if result.should_stop:
            return True

    return False
