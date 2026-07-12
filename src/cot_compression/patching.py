from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PatchingMethod:
    """Partitions a CoT token sequence into contiguous spans for compression.

    Concrete strategies decide where patch boundaries fall; each returned
    span is a half-open (start, end) range over token positions, and spans
    partition [0, num_tokens) in order.
    """

    name: str

    def requires_entropies(self) -> bool:
        """Whether split() needs per-token entropies to decide boundaries."""
        return False

    def split(
        self,
        num_tokens: int,
        sample_index: int,
        seed: int,
        entropies: torch.Tensor | None,
    ) -> list[tuple[int, int]]:
        raise NotImplementedError


@dataclass(frozen=True)
class UniformPatchingMethod(PatchingMethod):
    """Fixed-size contiguous chunks; the last chunk may be shorter."""

    patch_size: int

    def __init__(self, patch_size: int = 8) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        super().__init__(name="uniform")
        object.__setattr__(self, "patch_size", patch_size)

    def split(
        self,
        num_tokens: int,
        sample_index: int,
        seed: int,
        entropies: torch.Tensor | None,
    ) -> list[tuple[int, int]]:
        del sample_index, seed, entropies
        return [
            (start, min(start + self.patch_size, num_tokens))
            for start in range(0, num_tokens, self.patch_size)
        ]


@dataclass(frozen=True)
class RandomPatchingMethod(PatchingMethod):
    """Chunk lengths of 2**i, i sampled uniformly from [0, max_exponent]."""

    max_exponent: int

    def __init__(self, max_exponent: int = 6) -> None:
        if max_exponent < 0:
            raise ValueError("max_exponent must be non-negative.")
        super().__init__(name="exponential")
        object.__setattr__(self, "max_exponent", max_exponent)

    def split(
        self,
        num_tokens: int,
        sample_index: int,
        seed: int,
        entropies: torch.Tensor | None,
    ) -> list[tuple[int, int]]:
        del entropies
        rng = random.Random(seed + sample_index)
        spans = []
        start = 0
        while start < num_tokens:
            length = 2 ** rng.randint(0, self.max_exponent)
            end = min(start + length, num_tokens)
            spans.append((start, end))
            start = end
        return spans


@dataclass(frozen=True)
class EntropyPatchingMethod(PatchingMethod):
    """Starts a new patch immediately before every token whose entropy is
    at or above the Nth percentile of this trace's token entropies.

    E.g. entropies [high, low, low, high, low] with percentile chosen so
    only positions 0 and 3 clear the threshold split into
    [high, low, low] and [high, low]: a high-entropy token always begins a
    new patch, it's never grouped with what preceded it.
    """

    percentile: float

    def __init__(self, percentile: float = 80.0) -> None:
        if not 0.0 <= percentile <= 100.0:
            raise ValueError("percentile must be within [0, 100].")
        super().__init__(name="entropy")
        object.__setattr__(self, "percentile", percentile)

    def requires_entropies(self) -> bool:
        return True

    def split(
        self,
        num_tokens: int,
        sample_index: int,
        seed: int,
        entropies: torch.Tensor | None,
    ) -> list[tuple[int, int]]:
        del sample_index, seed
        assert entropies is not None
        # torch.quantile does not support bfloat16 (the model's usual dtype).
        threshold = torch.quantile(entropies.float(), self.percentile / 100.0)
        boundaries = (
            [0]
            + [i for i in range(1, num_tokens) if entropies[i] >= threshold]
            + [num_tokens]
        )
        return list(zip(boundaries[:-1], boundaries[1:], strict=True))
