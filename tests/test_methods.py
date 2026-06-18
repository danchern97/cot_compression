from __future__ import annotations

import pytest
import torch

from research_project_template.methods.generation import generate_tokens
from research_project_template.methods.gpt import GPTConfig, TinyGPT


def tiny_model() -> TinyGPT:
    return TinyGPT(
        GPTConfig(
            vocab_size=12,
            block_size=8,
            n_layer=1,
            n_head=1,
            n_embd=16,
            dropout=0.0,
            bias=True,
        )
    )


def test_tiny_gpt_forward_shape_and_loss() -> None:
    model = tiny_model()
    x = torch.randint(0, 12, (2, 8))
    y = torch.randint(0, 12, (2, 8))

    logits, loss = model(x, y)

    assert logits.shape == (2, 8, 12)
    assert loss is not None
    assert torch.isfinite(loss)


def test_tiny_gpt_uses_rope_not_absolute_position_embedding() -> None:
    model = tiny_model()

    assert not hasattr(model, "position_embedding")
    assert hasattr(model.blocks[0].attn, "rope_cos")
    assert hasattr(model.blocks[0].attn, "rope_sin")


def test_rope_requires_even_head_dimension() -> None:
    with pytest.raises(ValueError, match="even attention head dimension"):
        TinyGPT(
            GPTConfig(
                vocab_size=12,
                block_size=8,
                n_layer=1,
                n_head=2,
                n_embd=18,
                dropout=0.0,
                bias=True,
            )
        )


def test_generation_returns_more_tokens() -> None:
    model = tiny_model()
    idx = torch.zeros((1, 3), dtype=torch.long)

    out = generate_tokens(model, idx, max_new_tokens=5)

    assert out.shape == (1, 8)
    assert int(out.max()) < model.config.vocab_size
