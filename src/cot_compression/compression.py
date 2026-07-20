from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig

from cot_compression.data.answers import AnswerTrace, cot_token_ids, replace_trace
from cot_compression.data.dolci import Message
from cot_compression.patching import (
    EntropyDiffPatchingMethod,
    EntropySumPatchingMethod,
    EntropyThresholdPatchingMethod,
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
    # Short provenance tag for the compression method's own parameter (e.g.
    # "t0.5" for entropy-weighted-mean temperature); "none" when it has none.
    compression_param: str = "none"

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
        cot_ids: list[int] | None = None,
    ) -> CompressionResult:
        """Compress ``trace``'s CoT.

        ``cot_ids`` is the tokenized original CoT; it depends only on the trace,
        so callers evaluating several methods over the same sample may pass a
        previously computed one instead of paying for re-tokenization.
        """
        raise NotImplementedError


def _compose_name(
    family: str, patching: PatchingMethod | None, compression_param: str = "none"
) -> str:
    tagged = family if compression_param == "none" else f"{family}_{compression_param}"
    if patching is None:
        return tagged
    return f"{tagged}_{patching.name}_{patching.param_tag}"


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


def _reduce_spans_grouped(
    method: EmbeddingCompressionMethod,
    embeds: torch.Tensor,
    entropies: torch.Tensor | None,
    spans: list[tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    """Pool every span into one vector, batching spans of equal length together.

    Spans sharing a length are gathered into a single ``[n, length, hidden]``
    tensor and reduced in one call rather than one call per span. Reducing along
    dim 1 of that tensor visits the same elements in the same order as reducing
    the ``[length, hidden]`` slice of a single span, so the result is bitwise
    identical to a per-span loop while cutting kernel launches by orders of
    magnitude: a trace has ~1.7k-2.6k spans but only a handful of distinct
    lengths (6 for exponential patching, ~12 for entropy).
    """
    rows_by_length: dict[int, list[int]] = {}
    for row, (start, end) in enumerate(spans):
        rows_by_length.setdefault(end - start, []).append(row)

    starts = torch.tensor([start for start, _ in spans], device=device)
    # Allocated from the first group's result: the entropy-weighted reduction
    # promotes bfloat16 embeddings to float32 via the float32 weights, and
    # preallocating at the embedding dtype would silently downcast it.
    out: torch.Tensor | None = None
    for length, rows in rows_by_length.items():
        row_index = torch.tensor(rows, device=device)
        gather = starts[row_index].unsqueeze(1) + torch.arange(length, device=device)
        reduced = method.reduce_patches(
            embeds[gather],
            None if entropies is None else entropies[gather],
        )
        if out is None:
            out = torch.empty(
                len(spans), reduced.shape[-1], device=device, dtype=reduced.dtype
            )
        out[row_index] = reduced

    assert out is not None, "spans must be non-empty"
    return out


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
        cot_ids: list[int] | None = None,
    ) -> CompressionResult:
        del sample_index, seed, model, device, cot_entropy
        n = len(cot_token_ids(trace, tokenizer) if cot_ids is None else cot_ids)
        return CompressionResult(
            messages=trace.messages,
            slot_embeddings=None,
            original_cot_tokens=n,
            compressed_cot_tokens=n,
        )


@dataclass(frozen=True)
class EmbeddingCompressionMethod(CompressionMethod):
    """Compresses each patch of CoT token embeddings into one spliced vector.

    Subclasses implement ``reduce_patches``; the trace text is replaced by one
    placeholder token per patch, whose embedding is overwritten with the
    computed vector directly in inputs_embeds at scoring time.
    """

    def reduce_patches(
        self,
        embeds: torch.Tensor,
        entropies: torch.Tensor | None,
    ) -> torch.Tensor:
        """Reduce a batch of equal-length patches.

        ``embeds`` is ``[num_patches, patch_length, hidden]`` and ``entropies``
        (when required) is ``[num_patches, patch_length]``; returns one pooled
        vector per patch, ``[num_patches, hidden]``.
        """
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
        cot_ids: list[int] | None = None,
    ) -> CompressionResult:
        if cot_ids is None:
            cot_ids = cot_token_ids(trace, tokenizer)
        spans, entropies = _resolve_spans(
            self, len(cot_ids), cot_entropy, device, sample_index, seed
        )
        with torch.no_grad():
            embeds = model.get_input_embeddings()(torch.tensor(cot_ids, device=device))
            slot_embeddings = _reduce_spans_grouped(
                self, embeds, entropies, spans, device
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

    def reduce_patches(
        self,
        embeds: torch.Tensor,
        entropies: torch.Tensor | None,
    ) -> torch.Tensor:
        del entropies
        # Scale the sum by 1/sqrt(c) rather than 1/c (plain mean), so that for
        # zero-centered, uncorrelated, equal-variance token embeddings the
        # pooled vector keeps the original per-dim variance instead of shrinking
        # it by a factor of c. See Embedding Compress in arXiv:2505.16552.
        c = embeds.shape[1]
        return embeds.sum(dim=1) / (c**0.5)


@dataclass(frozen=True)
class EntropyWeightedMeanCompressionMethod(EmbeddingCompressionMethod):
    """Pools each patch with entropy-derived weights w_i(T) = softmax(ln H_i / T).

    ``temperature`` T controls the sharpness of the weighting:
      * T = 1  -> w_i = H_i / sum_j H_j, the plain entropy-weighted mean.
      * T -> 0 -> one-hot on argmax H, i.e. keep only the highest-entropy token.
    Lower T concentrates weight on the highest-entropy tokens; higher T spreads
    it out (T -> inf approaches a uniform mean).
    """

    temperature: float = 1.0

    def __init__(
        self, patching: PatchingMethod | None = None, temperature: float = 1.0
    ) -> None:
        if temperature < 0.0:
            raise ValueError("temperature must be >= 0.")
        compression_param = f"t{temperature:g}"
        super().__init__(
            name=_compose_name("entropy_weighted_mean", patching, compression_param),
            method_family="entropy_weighted_mean",
            patching=patching,
            compression_param=compression_param,
        )
        object.__setattr__(self, "temperature", float(temperature))

    def needs_entropies(self) -> bool:
        return True

    def reduce_patches(
        self,
        embeds: torch.Tensor,
        entropies: torch.Tensor | None,
    ) -> torch.Tensor:
        assert entropies is not None
        if self.temperature == 0.0:
            # One-hot on the highest-entropy token of each patch.
            idx = entropies.argmax(dim=1, keepdim=True)
            weights = torch.zeros_like(entropies).scatter_(1, idx, 1.0)
        else:
            # softmax(ln H / T); clamp guards log(0) for zero-entropy tokens,
            # softmax is internally max-stable. At T=1 this equals H / sum(H).
            logits = entropies.clamp_min(1e-12).log() / self.temperature
            weights = torch.softmax(logits, dim=1)
        pooled = (weights.unsqueeze(-1) * embeds).sum(dim=1)
        # Variance-preserving rescale (arXiv:2505.16552, generalized to
        # non-uniform weights): a weighted sum with sum(w)=1 has variance
        # sigma^2 * sum(w^2), so divide by sqrt(sum(w^2)) to restore sigma^2.
        return pooled / weights.pow(2).sum(dim=1, keepdim=True).sqrt()


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
        cot_ids: list[int] | None = None,
    ) -> CompressionResult:
        if cot_ids is None:
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
    ratio = float(patching_cfg.compression_ratio)
    if strategy == "uniform":
        return UniformPatchingMethod(compression_ratio=ratio)
    if strategy == "random":
        return RandomPatchingMethod(max_exponent=int(patching_cfg.random.max_exponent))
    if strategy == "entropy_threshold":
        return EntropyThresholdPatchingMethod(compression_ratio=ratio)
    if strategy == "entropy_diff":
        return EntropyDiffPatchingMethod(compression_ratio=ratio)
    if strategy == "entropy_sum":
        return EntropySumPatchingMethod(compression_ratio=ratio)
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
            patching = build_patching_method(strategy, patching_cfg)
            if name == "entropy_weighted_mean":
                methods.append(
                    EntropyWeightedMeanCompressionMethod(
                        patching=patching,
                        temperature=float(method_cfg.get("temperature", 1.0)),
                    )
                )
            else:
                methods.append(_METHOD_CLASSES[name](patching=patching))
        else:
            raise ValueError(f"Unknown evaluation method: {name}")
    return methods
