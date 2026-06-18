from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig

from research_project_template.data.tokenizer import CharTokenizer
from research_project_template.methods.generation import generate_tokens
from research_project_template.training.checkpointing import load_model_from_checkpoint
from research_project_template.training.utils import get_run_dir, resolve_device


def generate(cfg: DictConfig) -> str:
    if cfg.checkpoint.path is None:
        raise ValueError("Set checkpoint.path to generate text.")

    device = resolve_device(cfg.device)
    model, checkpoint = load_model_from_checkpoint(Path(cfg.checkpoint.path), device)
    tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])

    prompt_ids = tokenizer.encode(str(cfg.prompt))
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = generate_tokens(
        model=model,
        idx=idx,
        max_new_tokens=int(cfg.max_new_tokens),
        temperature=float(cfg.temperature),
        top_k=cfg.top_k,
    )
    text = tokenizer.decode(generated[0].tolist())

    output_path = get_run_dir(cfg) / str(cfg.output_file)
    output_path.write_text(text, encoding="utf-8")
    print(text)
    return text
