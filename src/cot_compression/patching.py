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

    @property
    def param_tag(self) -> str:
        """Short provenance tag for the strategy's parameter (e.g. 'ps8')."""
        return self.name

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
    """Fixed-size contiguous chunks; the last chunk may be shorter.

    Parameterized by ``compression_ratio`` (= original_len / compressed_len =
    average patch length), the same universal knob the entropy strategies use.
    For uniform chunks the average patch length *is* the chunk size, so the
    patch size is simply ``round(compression_ratio)``. Consequence: uniform can
    only realize integer ratios (1.5 rounds to 2).
    """

    compression_ratio: float
    patch_size: int

    def __init__(self, compression_ratio: float = 8.0) -> None:
        if compression_ratio < 1.0:
            raise ValueError("compression_ratio must be >= 1.")
        super().__init__(name="uniform")
        object.__setattr__(self, "compression_ratio", float(compression_ratio))
        object.__setattr__(self, "patch_size", max(1, round(compression_ratio)))

    @property
    def param_tag(self) -> str:
        return f"cr{self.compression_ratio:g}"

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

    @property
    def param_tag(self) -> str:
        return f"exp{self.max_exponent}"

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
    """Abstract base for entropy-driven patching strategies.

    All entropy strategies are parameterized by a single universal knob,
    ``compression_ratio`` (= original_len / compressed_len = target average
    patch length, >= 1), and derive their per-trace splitting constraint
    (percentile threshold, monotonic-difference threshold, or information
    budget) so that the *realized* average patch length matches the target in
    expectation. Subclasses implement ``split``.
    """

    compression_ratio: float

    def __init__(self, compression_ratio: float, name: str) -> None:
        if compression_ratio < 1.0:
            raise ValueError("compression_ratio must be >= 1.")
        super().__init__(name=name)
        object.__setattr__(self, "compression_ratio", float(compression_ratio))

    @property
    def param_tag(self) -> str:
        return f"cr{self.compression_ratio:g}"

    def requires_entropies(self) -> bool:
        return True


@dataclass(frozen=True)
class EntropyThresholdPatchingMethod(EntropyPatchingMethod):
    """Global-constraint patching: start a new patch immediately before every
    token whose entropy is at or above the ``1 - 1/compression_ratio`` quantile
    of this trace's token entropies.

    A high-entropy token always begins a new patch, never grouped with what
    preceded it. Roughly a ``1/compression_ratio`` fraction of tokens clear the
    threshold, so the trace splits into ~L/compression_ratio patches, i.e. an
    average patch length ~= compression_ratio.
    """

    def __init__(self, compression_ratio: float = 2.0) -> None:
        super().__init__(compression_ratio, name="entropy_threshold")

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
        quantile = 1.0 - 1.0 / self.compression_ratio
        threshold = torch.quantile(entropies[:num_tokens].float(), quantile)
        # Compare on device and move the boundary indices across in a single
        # transfer. Testing `entropies[i] >= threshold` inside a Python loop
        # instead costs one host-device sync per CoT token (~10k per sample),
        # which dominated evaluation runtime for entropy patching.
        starts = (entropies[1:num_tokens] >= threshold).nonzero(as_tuple=True)[0]
        boundaries = [0] + (starts + 1).tolist() + [num_tokens]
        return list(zip(boundaries[:-1], boundaries[1:], strict=True))


@dataclass(frozen=True)
class EntropyDiffPatchingMethod(EntropyPatchingMethod):
    """Approximate monotonic-constraint patching (BLT, arXiv:2412.09871).

    Starts a new patch before token t when the entropy rises sharply from the
    previous token, H(x_t) - H(x_{t-1}) > theta_r, with theta_r set to the
    ``1 - 1/compression_ratio`` quantile of the consecutive entropy differences.
    A ~1/compression_ratio fraction of positions clear theta_r, giving an
    average patch length ~= compression_ratio.
    """

    def __init__(self, compression_ratio: float = 2.0) -> None:
        super().__init__(compression_ratio, name="entropy_diff")

    def split(
        self,
        num_tokens: int,
        sample_index: int,
        seed: int,
        entropies: torch.Tensor | None,
    ) -> list[tuple[int, int]]:
        del sample_index, seed
        assert entropies is not None
        if num_tokens <= 1:
            return [(0, num_tokens)]
        entropy = entropies[:num_tokens].float()
        diffs = entropy[1:] - entropy[:-1]  # diffs[i] = H[i+1] - H[i]
        quantile = 1.0 - 1.0 / self.compression_ratio
        theta_r = torch.quantile(diffs, quantile)
        # A boundary before token i+1 when its entropy jumps past theta_r.
        starts = (diffs > theta_r).nonzero(as_tuple=True)[0]
        boundaries = [0] + (starts + 1).tolist() + [num_tokens]
        return list(zip(boundaries[:-1], boundaries[1:], strict=True))


@dataclass(frozen=True)
class EntropySumPatchingMethod(EntropyPatchingMethod):
    """Equal-information (B-budget) patching: cut whenever the cumulative token
    entropy since the last cut passes an information budget B.

    B is chosen from the target ratio as B = total_entropy * compression_ratio
    / num_tokens, so the trace splits into ~round(num_tokens / compression_ratio)
    patches of roughly equal summed entropy, giving an average patch length
    ~= compression_ratio. Implemented via a vectorized cumulative-sum +
    searchsorted (equivalent to the sequential "sum exceeds B" greedy) to avoid
    a Python loop over tokens.
    """

    def __init__(self, compression_ratio: float = 2.0) -> None:
        super().__init__(compression_ratio, name="entropy_sum")

    def split(
        self,
        num_tokens: int,
        sample_index: int,
        seed: int,
        entropies: torch.Tensor | None,
    ) -> list[tuple[int, int]]:
        del sample_index, seed
        assert entropies is not None
        if num_tokens <= 1:
            return [(0, num_tokens)]
        cum = torch.cumsum(entropies[:num_tokens].float(), dim=0)
        total = cum[-1]
        # Degenerate all-zero-entropy trace: fall back to uniform chunks so we
        # still hit the target ratio instead of returning one giant patch.
        if float(total) <= 0.0:
            size = max(1, round(self.compression_ratio))
            return [
                (start, min(start + size, num_tokens))
                for start in range(0, num_tokens, size)
            ]
        budget = total * self.compression_ratio / num_tokens
        num_patches = max(1, int(round(float(total / budget))))
        if num_patches <= 1:
            return [(0, num_tokens)]
        # Interior cut budgets k*B; the first token whose running sum exceeds
        # each budget starts a new patch. right=True => first index with cum > t.
        targets = budget * torch.arange(1, num_patches, device=cum.device)
        raw = torch.searchsorted(cum, targets, right=True)
        starts = torch.unique(raw.clamp(1, num_tokens - 1))
        boundaries = [0] + starts.tolist() + [num_tokens]
        return list(zip(boundaries[:-1], boundaries[1:], strict=True))
