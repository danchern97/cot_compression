from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from omegaconf import DictConfig

from cot_compression.data.answers import AnswerTrace, cot_token_ids, replace_trace
from cot_compression.data.dolci import Message
from cot_compression.patching import (
    PatchingMethod,
    RandomPatchingMethod,
    SignalDiffPatchingMethod,
    SignalSumPatchingMethod,
    SignalThresholdPatchingMethod,
    UniformPatchingMethod,
)
from cot_compression.signals import SIGNALS

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
class CompressionPlan:
    """How a trace's text is rewritten, decided without touching the model.

    Separated from the slot vectors so the text rewrite + tokenization can run in
    a DataLoader worker while the embedding lookup and pooling stay in the main
    process on the GPU. ``num_slots`` is None for text methods, which leave the
    trace untouched; otherwise it is the number of placeholder tokens to splice.
    """

    num_slots: int | None
    original_cot_tokens: int
    compressed_cot_tokens: int


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

    def weight_signal(self) -> str | None:
        """Which per-token signal the *pooling* consumes, or None.

        Distinct from the patching signal: a method can weight by one signal
        (e.g. surprisal) while patching by another (e.g. entropy). Overridden by
        the signal-weighted-mean family; None for everything else.
        """
        return None

    def required_signals(self) -> frozenset[str]:
        """The per-token signals this method needs — patching's plus pooling's.

        The eval loop loads exactly this union (see ``cot_compression.signals``)
        and hands each method only the signals it declares here.
        """
        signals: set[str] = set()
        if self.patching is not None:
            patch_signal = self.patching.required_signal()
            if patch_signal is not None:
                signals.add(patch_signal)
        weight = self.weight_signal()
        if weight is not None:
            signals.add(weight)
        return frozenset(signals)

    def plan(
        self,
        num_cot_tokens: int,
        sample_index: int,
        seed: int,
        cot_signals: dict[str, torch.Tensor | None] | None,
        device: torch.device,
    ) -> CompressionPlan:
        """Decide the slot count and token accounting, without the model.

        Patch boundaries are resolved on ``device``: for the signal strategies
        the quantile and cumsum that place them are floating-point reductions
        whose last bits depend on where they run, so the caller keeps this on the
        same device as the rest of evaluation rather than letting a CPU worker
        recompute them.
        """
        # Only the *patching* signal is consulted here. The weighted-mean family
        # also needs a signal, but for pooling, which happens in materialize --
        # demanding it here would stop a worker planning a uniform-patched sample.
        values = _patching_values(self, cot_signals, device)
        spans = _split_spans(self, num_cot_tokens, values, sample_index, seed)
        return CompressionPlan(len(spans), num_cot_tokens, len(spans))

    def materialize(
        self,
        cot_ids: list[int],
        sample_index: int,
        seed: int,
        tokenizer: Any,
        model: Any,
        device: torch.device,
        cot_signals: dict[str, torch.Tensor | None] | None,
    ) -> torch.Tensor | None:
        """Build the [num_slots, hidden] slot matrix, or None for text methods.

        Re-resolves the same spans ``plan`` did; both are deterministic in
        (num_cot_tokens, signal values, sample_index, seed), so the two agree.
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
        cot_signals: dict[str, torch.Tensor | None] | None,
        cot_ids: list[int] | None = None,
    ) -> CompressionResult:
        """Compress ``trace``'s CoT: ``plan`` then ``materialize``, in one call.

        ``cot_ids`` is the tokenized original CoT; it depends only on the trace,
        so callers evaluating several methods over the same sample may pass a
        previously computed one instead of paying for re-tokenization.
        """
        if cot_ids is None:
            cot_ids = cot_token_ids(trace, tokenizer)
        plan = self.plan(len(cot_ids), sample_index, seed, cot_signals, device)
        slot_embeddings = self.materialize(
            cot_ids, sample_index, seed, tokenizer, model, device, cot_signals
        )
        return CompressionResult(
            messages=compressed_messages(trace, plan.num_slots),
            slot_embeddings=slot_embeddings,
            original_cot_tokens=plan.original_cot_tokens,
            compressed_cot_tokens=plan.compressed_cot_tokens,
        )


def _compose_name(
    family: str, patching: PatchingMethod | None, compression_param: str = "none"
) -> str:
    tagged = family if compression_param == "none" else f"{family}_{compression_param}"
    if patching is None:
        return tagged
    return f"{tagged}_{patching.name}_{patching.param_tag}"


def _signal_on_device(
    method_name: str,
    signal: str,
    cot_signals: dict[str, torch.Tensor | None] | None,
    device: torch.device,
) -> torch.Tensor:
    """Fetch one required signal's values on ``device``, or raise to skip."""
    tensor = None if cot_signals is None else cot_signals.get(signal)
    if tensor is None:
        raise ValueError(
            f"Method {method_name} needs the {signal!r} CoT signal but none was provided."
        )
    return tensor.to(device)


def _patching_values(
    method: CompressionMethod,
    cot_signals: dict[str, torch.Tensor | None] | None,
    device: torch.device,
) -> torch.Tensor | None:
    """On-device values of the signal this method's patching needs, or None."""
    if method.patching is None:
        return None
    signal = method.patching.required_signal()
    if signal is None:
        return None
    return _signal_on_device(method.name, signal, cot_signals, device)


def _split_spans(
    method: CompressionMethod,
    num_cot_tokens: int,
    values: torch.Tensor | None,
    sample_index: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Partition the CoT, or return one span covering it when unpatched."""
    if method.patching is None:
        return [(0, num_cot_tokens)]
    return method.patching.split(num_cot_tokens, sample_index, seed, values)


def _reduce_spans_grouped(
    method: EmbeddingCompressionMethod,
    embeds: torch.Tensor,
    weights: torch.Tensor | None,
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
    lengths (6 for exponential patching, ~12 for signal patching).

    ``weights`` is the per-token weighting signal (None for unweighted pooling).
    """
    rows_by_length: dict[int, list[int]] = {}
    for row, (start, end) in enumerate(spans):
        rows_by_length.setdefault(end - start, []).append(row)

    starts = torch.tensor([start for start, _ in spans], device=device)
    # Allocated from the first group's result: the signal-weighted reduction
    # promotes bfloat16 embeddings to float32 via the float32 weights, and
    # preallocating at the embedding dtype would silently downcast it.
    out: torch.Tensor | None = None
    for length, rows in rows_by_length.items():
        row_index = torch.tensor(rows, device=device)
        gather = starts[row_index].unsqueeze(1) + torch.arange(length, device=device)
        reduced = method.reduce_patches(
            embeds[gather],
            None if weights is None else weights[gather],
        )
        if out is None:
            out = torch.empty(
                len(spans), reduced.shape[-1], device=device, dtype=reduced.dtype
            )
        out[row_index] = reduced

    assert out is not None, "spans must be non-empty"
    return out


def compressed_messages(trace: AnswerTrace, num_slots: int | None) -> list[Message]:
    """Rewrite the think block as ``num_slots`` placeholders (None => unchanged).

    Pure text work, so a DataLoader worker can call it before tokenizing without
    holding the model or touching a GPU.
    """
    if num_slots is None:
        return trace.messages
    return replace_trace(trace, PLACEHOLDER_TOKEN * num_slots)


@dataclass(frozen=True)
class BaseCompressionMethod(CompressionMethod):
    def __init__(self) -> None:
        super().__init__(name="base", method_family="base", patching=None)

    def plan(
        self,
        num_cot_tokens: int,
        sample_index: int,
        seed: int,
        cot_signals: dict[str, torch.Tensor | None] | None,
        device: torch.device,
    ) -> CompressionPlan:
        del sample_index, seed, cot_signals, device
        return CompressionPlan(None, num_cot_tokens, num_cot_tokens)

    def materialize(
        self,
        cot_ids: list[int],
        sample_index: int,
        seed: int,
        tokenizer: Any,
        model: Any,
        device: torch.device,
        cot_signals: dict[str, torch.Tensor | None] | None,
    ) -> torch.Tensor | None:
        del cot_ids, sample_index, seed, tokenizer, model, device, cot_signals
        return None


@dataclass(frozen=True)
class NoCotCompressionMethod(CompressionMethod):
    """Baseline: drop the reasoning, keeping an empty ``<think></think>`` block.

    Rewrites the think block's interior to the empty string, so the assistant
    turn is exactly what the slot methods produce in the K -> 0 limit (the
    ``<think>``/``</think>`` markers stay; only the interior tokens go). A text
    method — no slot embeddings — so its gap against any compressed method
    isolates the value of the CoT content alone. ``compressed_cot_tokens`` is 0
    and ``compression_ratio`` lands at 0.0 through the guarded division in eval.
    """

    def __init__(self) -> None:
        super().__init__(name="no_cot", method_family="no_cot", patching=None)

    def plan(
        self,
        num_cot_tokens: int,
        sample_index: int,
        seed: int,
        cot_signals: dict[str, torch.Tensor | None] | None,
        device: torch.device,
    ) -> CompressionPlan:
        del sample_index, seed, cot_signals, device
        # num_slots=0 => compressed_messages rewrites the interior to "" (K
        # placeholders with K=0), and no slots are spliced.
        return CompressionPlan(0, num_cot_tokens, 0)

    def materialize(
        self,
        cot_ids: list[int],
        sample_index: int,
        seed: int,
        tokenizer: Any,
        model: Any,
        device: torch.device,
        cot_signals: dict[str, torch.Tensor | None] | None,
    ) -> torch.Tensor | None:
        del cot_ids, sample_index, seed, tokenizer, model, device, cot_signals
        return None


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
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        """Reduce a batch of equal-length patches.

        ``embeds`` is ``[num_patches, patch_length, hidden]`` and ``weights``
        (the per-token weighting signal, when required) is
        ``[num_patches, patch_length]``; returns one pooled vector per patch,
        ``[num_patches, hidden]``.
        """
        raise NotImplementedError

    def materialize(
        self,
        cot_ids: list[int],
        sample_index: int,
        seed: int,
        tokenizer: Any,
        model: Any,
        device: torch.device,
        cot_signals: dict[str, torch.Tensor | None] | None,
    ) -> torch.Tensor | None:
        del tokenizer
        values = _patching_values(self, cot_signals, device)
        spans = _split_spans(self, len(cot_ids), values, sample_index, seed)
        weight_signal = self.weight_signal()
        weights = (
            None
            if weight_signal is None
            else _signal_on_device(self.name, weight_signal, cot_signals, device)
        )
        with torch.no_grad():
            embeds = model.get_input_embeddings()(torch.tensor(cot_ids, device=device))
            return _reduce_spans_grouped(self, embeds, weights, spans, device)


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
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        del weights
        # Scale the sum by 1/sqrt(c) rather than 1/c (plain mean), so that for
        # zero-centered, uncorrelated, equal-variance token embeddings the
        # pooled vector keeps the original per-dim variance instead of shrinking
        # it by a factor of c. See Embedding Compress in arXiv:2505.16552.
        c = embeds.shape[1]
        return embeds.sum(dim=1) / (c**0.5)


@dataclass(frozen=True)
class SignalWeightedMeanCompressionMethod(EmbeddingCompressionMethod):
    """Pools each patch with signal-derived weights w_i(T) = softmax(ln s_i / T).

    ``signal`` (``entropy`` or ``surprisal``) supplies the non-negative per-token
    value s_i; ``temperature`` T controls the sharpness of the weighting:
      * T = 1  -> w_i = s_i / sum_j s_j, the plain signal-weighted mean.
      * T -> 0 -> one-hot on argmax s, i.e. keep only the highest-signal token.
    Lower T concentrates weight on the highest-signal tokens; higher T spreads
    it out (T -> inf approaches a uniform mean). With ``signal="entropy"`` this
    is exactly the historical entropy-weighted mean (same family name, same tag).
    """

    temperature: float = 1.0
    signal: str = "entropy"

    def __init__(
        self,
        patching: PatchingMethod | None = None,
        temperature: float = 1.0,
        signal: str = "entropy",
    ) -> None:
        if temperature < 0.0:
            raise ValueError("temperature must be >= 0.")
        if signal not in SIGNALS:
            raise ValueError(f"Unknown signal {signal!r}. Expected one of {SIGNALS}.")
        family = f"{signal}_weighted_mean"
        compression_param = f"t{temperature:g}"
        super().__init__(
            name=_compose_name(family, patching, compression_param),
            method_family=family,
            patching=patching,
            compression_param=compression_param,
        )
        object.__setattr__(self, "temperature", float(temperature))
        object.__setattr__(self, "signal", signal)

    def weight_signal(self) -> str | None:
        return self.signal

    def reduce_patches(
        self,
        embeds: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        assert weights is not None
        if self.temperature == 0.0:
            # One-hot on the highest-signal token of each patch.
            idx = weights.argmax(dim=1, keepdim=True)
            w = torch.zeros_like(weights).scatter_(1, idx, 1.0)
        else:
            # softmax(ln s / T); clamp guards log(0) for zero-signal tokens,
            # softmax is internally max-stable. At T=1 this equals s / sum(s).
            logits = weights.clamp_min(1e-12).log() / self.temperature
            w = torch.softmax(logits, dim=1)
        pooled = (w.unsqueeze(-1) * embeds).sum(dim=1)
        # Variance-preserving rescale (arXiv:2505.16552, generalized to
        # non-uniform weights): a weighted sum with sum(w)=1 has variance
        # sigma^2 * sum(w^2), so divide by sqrt(sum(w^2)) to restore sigma^2.
        return pooled / w.pow(2).sum(dim=1, keepdim=True).sqrt()


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

    def materialize(
        self,
        cot_ids: list[int],
        sample_index: int,
        seed: int,
        tokenizer: Any,
        model: Any,
        device: torch.device,
        cot_signals: dict[str, torch.Tensor | None] | None,
    ) -> torch.Tensor | None:
        values = _patching_values(self, cot_signals, device)
        spans = _split_spans(self, len(cot_ids), values, sample_index, seed)
        weight = model.get_input_embeddings().weight
        high = _regular_vocab_bound(tokenizer, int(weight.shape[0]))
        rng = random.Random(seed + sample_index)
        ids = [rng.randrange(high) for _ in spans]
        with torch.no_grad():
            return weight[ids].detach().clone()


# Signal-patching rule -> class. A config strategy is "<signal>_<rule>", e.g.
# "entropy_sum" or "surprisal_threshold"; entropy names are unchanged from before
# the signal generalization, so existing artifacts keep their join keys.
_PATCHING_RULES = {
    "threshold": SignalThresholdPatchingMethod,
    "diff": SignalDiffPatchingMethod,
    "sum": SignalSumPatchingMethod,
}


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
    for signal in SIGNALS:
        prefix = f"{signal}_"
        if strategy.startswith(prefix):
            rule_cls = _PATCHING_RULES.get(strategy[len(prefix) :])
            if rule_cls is not None:
                return rule_cls(compression_ratio=ratio, signal=signal)
    raise ValueError(f"Unknown patching strategy: {strategy}")


# Config-free baselines: constructed by name with no sub-config.
_SIMPLE_METHODS = {
    "base": BaseCompressionMethod,
    "no_cot": NoCotCompressionMethod,
}

# Patched compression families that take a `patching` sub-key (and nothing else).
_PATCHED_METHODS = {
    "random": RandomCompressionMethod,
    "simple_mean": SimpleMeanCompressionMethod,
}

# Signal-weighted-mean families: `patching` + `temperature`, and the pooling
# signal fixed by the family name.
_WEIGHTED_MEAN_SIGNALS = {
    "entropy_weighted_mean": "entropy",
    "surprisal_weighted_mean": "surprisal",
}


def build_compression_methods(cfg: DictConfig) -> list[CompressionMethod]:
    methods: list[CompressionMethod] = []
    patching_cfg = cfg.evaluation.methods.patching
    for name in cfg.evaluation.methods.enabled:
        if name in _SIMPLE_METHODS:
            methods.append(_SIMPLE_METHODS[name]())
        elif name in _WEIGHTED_MEAN_SIGNALS:
            method_cfg = cfg.evaluation.methods[name]
            patching = build_patching_method(method_cfg.get("patching"), patching_cfg)
            methods.append(
                SignalWeightedMeanCompressionMethod(
                    patching=patching,
                    temperature=float(method_cfg.get("temperature", 1.0)),
                    signal=_WEIGHTED_MEAN_SIGNALS[name],
                )
            )
        elif name in _PATCHED_METHODS:
            method_cfg = cfg.evaluation.methods[name]
            patching = build_patching_method(method_cfg.get("patching"), patching_cfg)
            methods.append(_PATCHED_METHODS[name](patching=patching))
        else:
            raise ValueError(f"Unknown evaluation method: {name}")
    return methods
