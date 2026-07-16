from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig

from cot_compression.data.answers import AnswerTrace, cot_token_ids, replace_trace
from cot_compression.data.dolci import Message
from cot_compression.patching import (
    EntropyPatchingMethod,
    PatchingMethod,
    RandomPatchingMethod,
    UniformPatchingMethod,
)

# Reserved Qwen3 token used as the per-patch placeholder. Its embedding is
# overwritten by the spliced slot vector at scoring time, so the token id is
# irrelevant to the forward pass (only its position matters) — no vocab growth.
# Chosen because it never appears in this text dataset and tokenizes to a single
# id; placeholders are joined WITHOUT spaces so K patches -> exactly K adjacent
# slot tokens (space-joining would inject spurious separator tokens between them).
PLACEHOLDER_TOKEN = "<|vision_pad|>"


@dataclass(frozen=True)
class CompressionResult:
    messages: list[Message]
    # [num_patches, hidden] to splice via inputs_embeds; None for text methods.
    slot_embeddings: torch.Tensor | None
    original_cot_tokens: int | None
    compressed_cot_tokens: int | None


@dataclass(frozen=True)
class CompressionMethod:
    name: str
    method_family: str
    patching: PatchingMethod | None

    @property
    def patching_param(self) -> str:
        return self.patching.param_tag if self.patching is not None else "none"

    @property
    def patching_name(self) -> str:
        return self.patching.name if self.patching is not None else "none"

    def needs_entropies(self) -> bool:
        return self.patching is not None and self.patching.requires_entropies()

    def compress(
        self,
        trace: AnswerTrace,
        sample_index: int,
        seed: int,
        tokenizer: Any,
        model: Any,
        device: torch.device,
        cot_entropy: torch.Tensor | None,
    ) -> CompressionResult:
        raise NotImplementedError


def _compose_name(family: str, patching: PatchingMethod | None) -> str:
    if patching is None:
        return family
    return f"{family}_{patching.name}_{patching.param_tag}"


def _resolve_spans(
    method: CompressionMethod,
    num_cot_tokens: int,
    cot_entropy: torch.Tensor | None,
    device: torch.device,
    sample_index: int,
    seed: int,
) -> tuple[list[tuple[int, int]], torch.Tensor | None]:
    """Materialize per-patch spans (and the on-device entropies, if needed)."""
    entropies = None
    if method.needs_entropies():
        if cot_entropy is None:
            raise ValueError(
                f"Method {method.name} needs CoT entropies but none were provided."
            )
        entropies = cot_entropy.to(device)
    spans = (
        method.patching.split(num_cot_tokens, sample_index, seed, entropies)
        if method.patching is not None
        else [(0, num_cot_tokens)]
    )
    return spans, entropies


def _splice_result(
    trace: AnswerTrace,
    spans: list[tuple[int, int]],
    original_cot_tokens: int,
    slot_embeddings: torch.Tensor,
) -> CompressionResult:
    placeholder = PLACEHOLDER_TOKEN * len(spans)
    return CompressionResult(
        messages=replace_trace(trace, placeholder),
        slot_embeddings=slot_embeddings,
        original_cot_tokens=original_cot_tokens,
        compressed_cot_tokens=len(spans),
    )


@dataclass(frozen=True)
class BaseCompressionMethod(CompressionMethod):
    def __init__(self) -> None:
        super().__init__(name="base", method_family="base", patching=None)

    def compress(
        self,
        trace: AnswerTrace,
        sample_index: int,
        seed: int,
        tokenizer: Any,
        model: Any,
        device: torch.device,
        cot_entropy: torch.Tensor | None,
    ) -> CompressionResult:
        del sample_index, seed, model, device, cot_entropy
        n = len(cot_token_ids(trace, tokenizer))
        return CompressionResult(
            messages=trace.messages,
            slot_embeddings=None,
            original_cot_tokens=n,
            compressed_cot_tokens=n,
        )


@dataclass(frozen=True)
class EmbeddingCompressionMethod(CompressionMethod):
    """Compresses each patch of CoT token embeddings into one spliced vector.

    Subclasses implement ``reduce_patch``; the trace text is replaced by one
    placeholder token per patch, whose embedding is overwritten with the
    computed vector directly in inputs_embeds at scoring time.
    """

    def reduce_patch(
        self,
        embeds: torch.Tensor,
        entropies: torch.Tensor | None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def compress(
        self,
        trace: AnswerTrace,
        sample_index: int,
        seed: int,
        tokenizer: Any,
        model: Any,
        device: torch.device,
        cot_entropy: torch.Tensor | None,
    ) -> CompressionResult:
        cot_ids = cot_token_ids(trace, tokenizer)
        spans, entropies = _resolve_spans(
            self, len(cot_ids), cot_entropy, device, sample_index, seed
        )
        with torch.no_grad():
            embeds = model.get_input_embeddings()(torch.tensor(cot_ids, device=device))
            slot_embeddings = torch.stack(
                [
                    self.reduce_patch(
                        embeds[start:end],
                        None if entropies is None else entropies[start:end],
                    )
                    for start, end in spans
                ]
            )
        return _splice_result(trace, spans, len(cot_ids), slot_embeddings)


@dataclass(frozen=True)
class SimpleMeanCompressionMethod(EmbeddingCompressionMethod):
    def __init__(self, patching: PatchingMethod | None = None) -> None:
        super().__init__(
            name=_compose_name("simple_mean", patching),
            method_family="simple_mean",
            patching=patching,
        )

    def reduce_patch(
        self,
        embeds: torch.Tensor,
        entropies: torch.Tensor | None,
    ) -> torch.Tensor:
        del entropies
        # Scale the sum by 1/sqrt(c) rather than 1/c (plain mean), so that for
        # zero-centered, uncorrelated, equal-variance token embeddings the
        # pooled vector keeps the original per-dim variance instead of shrinking
        # it by a factor of c. See Embedding Compress in arXiv:2505.16552.
        c = embeds.shape[0]
        return embeds.sum(dim=0) / (c**0.5)


@dataclass(frozen=True)
class EntropyWeightedMeanCompressionMethod(EmbeddingCompressionMethod):
    def __init__(self, patching: PatchingMethod | None = None) -> None:
        super().__init__(
            name=_compose_name("entropy_weighted_mean", patching),
            method_family="entropy_weighted_mean",
            patching=patching,
        )

    def needs_entropies(self) -> bool:
        return True

    def reduce_patch(
        self,
        embeds: torch.Tensor,
        entropies: torch.Tensor | None,
    ) -> torch.Tensor:
        assert entropies is not None
        weights = entropies / entropies.sum()
        pooled = (weights.unsqueeze(-1) * embeds).sum(dim=0)
        # Variance-preserving rescale (arXiv:2505.16552, generalized to
        # non-uniform weights): a weighted sum with sum(w)=1 has variance
        # sigma^2 * sum(w^2), so divide by sqrt(sum(w^2)) to restore sigma^2.
        return pooled / weights.pow(2).sum().sqrt()


def _regular_vocab_bound(tokenizer: Any, fallback: int) -> int:
    """Exclusive upper bound of regular token ids (first special/added id).

    Special/added tokens (highest ids) have outlier embeddings; a random
    baseline should draw from the regular vocabulary only.
    """
    try:
        added = tokenizer.get_added_vocab()
    except AttributeError:
        return fallback
    return min(added.values()) if added else fallback


@dataclass(frozen=True)
class RandomCompressionMethod(CompressionMethod):
    """Baseline: replace each patch with a random real-vocab token's embedding.

    In-distribution (real trained embeddings) and directly comparable to the
    mean methods (same patching, one slot per patch), but carrying no CoT
    information. Each patch draws an independent token from the regular vocab
    (special/added tokens excluded — outlier embeddings), so the slots vary.
    Uses the same placeholder-splice mechanism, so it grows no vocabulary.
    """

    def __init__(self, patching: PatchingMethod | None = None) -> None:
        super().__init__(
            name=_compose_name("random", patching),
            method_family="random",
            patching=patching,
        )

    def compress(
        self,
        trace: AnswerTrace,
        sample_index: int,
        seed: int,
        tokenizer: Any,
        model: Any,
        device: torch.device,
        cot_entropy: torch.Tensor | None,
    ) -> CompressionResult:
        cot_ids = cot_token_ids(trace, tokenizer)
        spans, _ = _resolve_spans(
            self, len(cot_ids), cot_entropy, device, sample_index, seed
        )
        weight = model.get_input_embeddings().weight
        high = _regular_vocab_bound(tokenizer, int(weight.shape[0]))
        rng = random.Random(seed + sample_index)
        ids = [rng.randrange(high) for _ in spans]
        with torch.no_grad():
            slot_embeddings = weight[ids].detach().clone()
        return _splice_result(trace, spans, len(cot_ids), slot_embeddings)


def build_patching_method(
    strategy: str | None,
    patching_cfg: DictConfig,
) -> PatchingMethod | None:
    if strategy is None:
        return None
    if strategy == "uniform":
        return UniformPatchingMethod(patch_size=int(patching_cfg.uniform.patch_size))
    if strategy == "random":
        return RandomPatchingMethod(max_exponent=int(patching_cfg.random.max_exponent))
    if strategy == "entropy":
        return EntropyPatchingMethod(percentile=float(patching_cfg.entropy.percentile))
    raise ValueError(f"Unknown patching strategy: {strategy}")


_METHOD_CLASSES = {
    "random": RandomCompressionMethod,
    "simple_mean": SimpleMeanCompressionMethod,
    "entropy_weighted_mean": EntropyWeightedMeanCompressionMethod,
}


def build_compression_methods(cfg: DictConfig) -> list[CompressionMethod]:
    methods: list[CompressionMethod] = []
    patching_cfg = cfg.evaluation.methods.patching
    for name in cfg.evaluation.methods.enabled:
        if name == "base":
            methods.append(BaseCompressionMethod())
        elif name in _METHOD_CLASSES:
            method_cfg = cfg.evaluation.methods[name]
            strategy = method_cfg.get("patching")
            methods.append(
                _METHOD_CLASSES[name](
                    patching=build_patching_method(strategy, patching_cfg)
                )
            )
        else:
            raise ValueError(f"Unknown evaluation method: {name}")
    return methods
