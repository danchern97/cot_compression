from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig

from cot_compression.data.answers import AnswerTrace, replace_trace
from cot_compression.data.dolci import Message


@dataclass(frozen=True)
class CompressionMethod:
    name: str
    abstract_tokens: list[str]

    def transform(
        self,
        trace: AnswerTrace,
        sample_index: int,
        seed: int,
    ) -> list[Message]:
        raise NotImplementedError


@dataclass(frozen=True)
class BaseCompressionMethod(CompressionMethod):
    def __init__(self) -> None:
        super().__init__(name="base", abstract_tokens=[])

    def transform(
        self,
        trace: AnswerTrace,
        sample_index: int,
        seed: int,
    ) -> list[Message]:
        del sample_index, seed
        return trace.messages


@dataclass(frozen=True)
class RandomCompressionMethod(CompressionMethod):
    abstract_length: int

    def __init__(self, abstract_vocab_size: int, abstract_length: int) -> None:
        if abstract_vocab_size <= 0:
            raise ValueError("abstract_vocab_size must be positive.")
        if abstract_length <= 0:
            raise ValueError("abstract_length must be positive.")
        tokens = [f"<abs_{index:05d}>" for index in range(abstract_vocab_size)]
        super().__init__(name="random", abstract_tokens=tokens)
        object.__setattr__(self, "abstract_length", abstract_length)

    def transform(
        self,
        trace: AnswerTrace,
        sample_index: int,
        seed: int,
    ) -> list[Message]:
        rng = random.Random(seed + sample_index)
        replacement = " ".join(
            rng.choice(self.abstract_tokens) for _ in range(self.abstract_length)
        )
        return replace_trace(trace, replacement)


def build_compression_methods(cfg: DictConfig) -> list[CompressionMethod]:
    methods = []
    for name in cfg.evaluation.methods.enabled:
        if name == "base":
            methods.append(BaseCompressionMethod())
        elif name == "random":
            random_cfg = cfg.evaluation.methods.random
            methods.append(
                RandomCompressionMethod(
                    abstract_vocab_size=int(random_cfg.abstract_vocab_size),
                    abstract_length=int(random_cfg.abstract_length),
                )
            )
        else:
            raise ValueError(f"Unknown evaluation method: {name}")
    return methods


def extend_tokenizer_and_model(
    tokenizer: Any,
    model: Any,
    methods: list[CompressionMethod],
) -> None:
    tokens = []
    for method in methods:
        tokens.extend(method.abstract_tokens)
    if not tokens:
        return

    tokenizer.add_tokens(tokens)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
