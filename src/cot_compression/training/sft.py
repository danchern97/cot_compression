from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from cot_compression.data.chat import QwenChatSFTCollator
from cot_compression.data.dolci import load_dolci_sft_data
from cot_compression.training.logging import RunLogger
from cot_compression.training.sft_loop import (
    LoopState,
    load_training_state,
    save_checkpoint,
    train_one_epoch,
)
from cot_compression.training.utils import (
    get_run_dir,
    optional_int,
    resolve_device,
    save_resolved_config,
    set_seed,
)


def parse_torch_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unknown torch dtype: {name}")


def count_optimizer_steps(
    examples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: int,
    max_steps: int | None,
) -> int:
    if max_steps is not None:
        return max_steps
    batches_per_epoch = math.ceil(examples / batch_size)
    return math.ceil(batches_per_epoch / gradient_accumulation_steps) * num_train_epochs


def build_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.optim.lr),
        betas=(float(cfg.optim.beta1), float(cfg.optim.beta2)),
        eps=float(cfg.optim.eps),
        weight_decay=float(cfg.optim.weight_decay),
    )


def build_sft_dataloaders(
    cfg: DictConfig,
    tokenizer: Any,
) -> tuple[DataLoader, DataLoader, int]:
    data = load_dolci_sft_data(cfg)
    collator = QwenChatSFTCollator(
        tokenizer=tokenizer,
        max_length=int(cfg.training.max_length),
    )
    generator = torch.Generator()
    generator.manual_seed(int(cfg.training.seed))
    train_loader = DataLoader(
        cast(TorchDataset[Any], data.train),
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    eval_loader = DataLoader(
        cast(TorchDataset[Any], data.eval),
        batch_size=int(cfg.training.eval_batch_size),
        shuffle=False,
        collate_fn=collator,
    )
    return train_loader, eval_loader, data.train.num_rows


def build_sft_model_and_tokenizer(
    cfg: DictConfig, device: torch.device
) -> tuple[Any, Any]:
    checkpoint = cfg.training.resume_from_checkpoint
    model_source = (
        str(checkpoint) if checkpoint is not None else str(cfg.method.model_name)
    )
    tokenizer_source = str(cfg.method.model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=bool(cfg.method.use_fast_tokenizer),
        trust_remote_code=bool(cfg.method.trust_remote_code),
    )
    model = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=parse_torch_dtype(str(cfg.training.torch_dtype)),
            trust_remote_code=bool(cfg.method.trust_remote_code),
        ),
    )
    if bool(cfg.training.gradient_checkpointing):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    return model.to(device), tokenizer


def train_sft(cfg: DictConfig) -> Path:
    run_dir = get_run_dir(cfg)
    save_resolved_config(cfg, run_dir)
    logger = RunLogger(cfg=cfg, run_dir=run_dir)

    set_seed(
        seed=int(cfg.training.seed),
        deterministic=bool(cfg.training.deterministic),
    )
    device = resolve_device(cfg.training.device)
    logger.info(f"Using device: {device}")

    model, tokenizer = build_sft_model_and_tokenizer(cfg, device)
    train_loader, eval_loader, train_examples = build_sft_dataloaders(cfg, tokenizer)
    optimizer = build_optimizer(cfg, model)

    total_steps = count_optimizer_steps(
        examples=train_examples,
        batch_size=int(cfg.training.batch_size),
        gradient_accumulation_steps=int(cfg.training.gradient_accumulation_steps),
        num_train_epochs=int(cfg.training.num_train_epochs),
        max_steps=optional_int(cfg.training.max_steps),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * float(cfg.training.warmup_ratio)),
        num_training_steps=total_steps,
    )

    global_step = 0
    best_eval_loss = float("inf")
    if cfg.training.resume_from_checkpoint is not None:
        global_step, best_eval_loss = load_training_state(
            Path(cfg.training.resume_from_checkpoint),
            optimizer=optimizer,
            scheduler=scheduler,
        )
        logger.info(f"Resumed from step {global_step}")

    active_model: torch.nn.Module = model
    if cfg.training.compile:
        active_model = cast(torch.nn.Module, torch.compile(model))

    state = LoopState(
        model=model,
        active_model=active_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        cfg=cfg,
        run_dir=run_dir,
        device=device,
        total_steps=total_steps,
        global_step=global_step,
        best_eval_loss=best_eval_loss,
    )
    optimizer.zero_grad(set_to_none=True)

    try:
        for epoch in range(int(cfg.training.num_train_epochs)):
            should_stop = train_one_epoch(
                state=state,
                train_loader=train_loader,
                eval_loader=eval_loader,
                epoch=epoch,
            )
            if should_stop:
                break

        save_checkpoint(state.last_checkpoint, state)
        logger.info(f"Saved final checkpoint to {state.last_checkpoint}")
        return state.last_checkpoint
    finally:
        logger.finish()
