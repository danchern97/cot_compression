from __future__ import annotations

from omegaconf import DictConfig

from research_project_template.methods.gpt import GPTConfig, TinyGPT


def build_method(cfg: DictConfig, vocab_size: int) -> TinyGPT:
    if cfg.method.family != "char_gpt":
        raise ValueError(f"Unknown method family: {cfg.method.family}")

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=int(cfg.method.block_size),
        n_layer=int(cfg.method.n_layer),
        n_head=int(cfg.method.n_head),
        n_embd=int(cfg.method.n_embd),
        dropout=float(cfg.method.dropout),
        bias=bool(cfg.method.bias),
    )
    return TinyGPT(config)
