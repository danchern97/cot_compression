"""Where patch boundaries fall: token-level rules, and paragraph steps.

Every strategy returns the *latent sub-spans* -- the partition each code
summarizes -- so pooling, initialization and the aux targets are correct whatever
chose the boundaries. Strategies that also carry a coarser grouping (today only
`paragraph`) expose it through `layout`, whose default wraps `split` as one step
per span; see `cot_compression.steps`.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from cot_compression.signals import SIGNALS
from cot_compression.steps import (
    StepLayout,
    segment_paragraph_steps,
    trivial_step_layout,
)


@dataclass(frozen=True)
class SplitContext:
    """Per-sample material a text-aware strategy needs beyond the token count.

    `split` otherwise sees only `num_tokens` and an optional per-token signal
    tensor, and neither can carry text. Widening `SIGNALS` instead would mint an
    npz cache stem and a GPU precompute pass for something that needs no forward
    pass at all, so this is a third channel rather than a fourth signal.

    Two callers hold different things. Training prep has the raw trace and can
    tokenize it with offsets. Evaluation plans K in a DataLoader worker but calls
    `materialize` later from the main process, which no longer has the text -- so
    the worker resolves the layout once and ships it, and `layout` wins over
    `text`/`offsets` when both are present. That makes a worker/main disagreement
    on K impossible rather than merely unlikely, which matters because
    `evaluate_method` silently drops a sample whose counts disagree.
    """

    text: str | None = None
    offsets: Sequence[tuple[int, int]] | None = None
    layout: StepLayout | None = None


@dataclass(frozen=True)
class PatchingMethod:
    """Partitions a CoT token sequence into contiguous spans for compression.

    Concrete strategies decide where patch boundaries fall; each returned
    span is a half-open (start, end) range over token positions, and spans
    partition [0, num_tokens) in order.
    """

    name: str

    def requires_context(self) -> bool:
        """Whether `split`/`layout` need a `SplitContext`.

        Mirrors `required_signal()`: the caller asks before doing the extra work,
        because a context needs the CoT tokenized with an offsets mapping, which is
        ~24% slower than the plain tokenization and only paragraph segmentation
        uses.
        """
        return False

    def required_signal(self) -> str | None:
        """Which per-token signal split() needs, or None if it needs none.

        A signal is one of the names in ``cot_compression.signals.SIGNALS``
        (``entropy`` / ``surprisal``); the caller supplies that signal's values
        as ``split``'s ``values`` argument.
        """
        return None

    @property
    def param_tag(self) -> str:
        """Short provenance tag for the strategy's parameter (e.g. 'ps8')."""
        return self.name

    def split(
        self,
        num_tokens: int,
        sample_index: int,
        seed: int,
        values: torch.Tensor | None,
        *,
        context: SplitContext | None = None,
    ) -> list[tuple[int, int]]:
        raise NotImplementedError

    def layout(
        self,
        num_tokens: int,
        sample_index: int,
        seed: int,
        values: torch.Tensor | None,
        *,
        context: SplitContext | None = None,
    ) -> StepLayout:
        """The two-level segmentation: steps, and the latents inside them.

        Defaults to one step per span, which is what token patching means in step
        vocabulary. That degeneracy is load-bearing: it lets the encoder run a
        single branch-free code path whose `cross_limit` and `cond_slot` reduce to
        their pre-existing values under `uniform`.
        """
        spans = self.split(num_tokens, sample_index, seed, values, context=context)
        return trivial_step_layout(spans)


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
        values: torch.Tensor | None,
        *,
        context: SplitContext | None = None,
    ) -> list[tuple[int, int]]:
        del sample_index, seed, values, context
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
        values: torch.Tensor | None,
        *,
        context: SplitContext | None = None,
    ) -> list[tuple[int, int]]:
        del values, context
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
class SignalPatchingMethod(PatchingMethod):
    """Abstract base for signal-driven patching strategies.

    A ``signal`` (``entropy`` or ``surprisal``) supplies one non-negative value
    per CoT token; the strategy places boundaries from that signal. All are
    parameterized by a single universal knob, ``compression_ratio`` (=
    original_len / compressed_len = target average patch length, >= 1), and
    derive their per-trace splitting constraint (percentile threshold,
    monotonic-difference threshold, or information budget) so that the *realized*
    average patch length matches the target in expectation. Subclasses implement
    ``split``; the same code works for any signal, only ``name`` differs.
    """

    compression_ratio: float
    signal: str

    def __init__(self, compression_ratio: float, signal: str, rule: str) -> None:
        if compression_ratio < 1.0:
            raise ValueError("compression_ratio must be >= 1.")
        if signal not in SIGNALS:
            raise ValueError(f"Unknown signal {signal!r}. Expected one of {SIGNALS}.")
        super().__init__(name=f"{signal}_{rule}")
        object.__setattr__(self, "compression_ratio", float(compression_ratio))
        object.__setattr__(self, "signal", signal)

    @property
    def param_tag(self) -> str:
        return f"cr{self.compression_ratio:g}"

    def required_signal(self) -> str | None:
        return self.signal


@dataclass(frozen=True)
class SignalThresholdPatchingMethod(SignalPatchingMethod):
    """Global-constraint patching: start a new patch immediately before every
    token whose signal is at or above the ``1 - 1/compression_ratio`` quantile
    of this trace's token signal values.

    A high-signal token always begins a new patch, never grouped with what
    preceded it. Roughly a ``1/compression_ratio`` fraction of tokens clear the
    threshold, so the trace splits into ~L/compression_ratio patches, i.e. an
    average patch length ~= compression_ratio.
    """

    def __init__(self, compression_ratio: float = 2.0, signal: str = "entropy") -> None:
        super().__init__(compression_ratio, signal, rule="threshold")

    def split(
        self,
        num_tokens: int,
        sample_index: int,
        seed: int,
        values: torch.Tensor | None,
        *,
        context: SplitContext | None = None,
    ) -> list[tuple[int, int]]:
        del sample_index, seed, context
        assert values is not None
        # torch.quantile does not support bfloat16 (the model's usual dtype).
        quantile = 1.0 - 1.0 / self.compression_ratio
        threshold = torch.quantile(values[:num_tokens].float(), quantile)
        # Compare on device and move the boundary indices across in a single
        # transfer. Testing `values[i] >= threshold` inside a Python loop
        # instead costs one host-device sync per CoT token (~10k per sample),
        # which dominated evaluation runtime for signal patching.
        starts = (values[1:num_tokens] >= threshold).nonzero(as_tuple=True)[0]
        boundaries = [0] + (starts + 1).tolist() + [num_tokens]
        return list(zip(boundaries[:-1], boundaries[1:], strict=True))


@dataclass(frozen=True)
class SignalDiffPatchingMethod(SignalPatchingMethod):
    """Approximate monotonic-constraint patching (BLT, arXiv:2412.09871).

    Starts a new patch before token t when the signal rises sharply from the
    previous token, s(x_t) - s(x_{t-1}) > theta_r, with theta_r set to the
    ``1 - 1/compression_ratio`` quantile of the consecutive signal differences.
    A ~1/compression_ratio fraction of positions clear theta_r, giving an
    average patch length ~= compression_ratio.
    """

    def __init__(self, compression_ratio: float = 2.0, signal: str = "entropy") -> None:
        super().__init__(compression_ratio, signal, rule="diff")

    def split(
        self,
        num_tokens: int,
        sample_index: int,
        seed: int,
        values: torch.Tensor | None,
        *,
        context: SplitContext | None = None,
    ) -> list[tuple[int, int]]:
        del sample_index, seed, context
        assert values is not None
        if num_tokens <= 1:
            return [(0, num_tokens)]
        signal = values[:num_tokens].float()
        diffs = signal[1:] - signal[:-1]  # diffs[i] = s[i+1] - s[i]
        quantile = 1.0 - 1.0 / self.compression_ratio
        theta_r = torch.quantile(diffs, quantile)
        # A boundary before token i+1 when its signal jumps past theta_r.
        starts = (diffs > theta_r).nonzero(as_tuple=True)[0]
        boundaries = [0] + (starts + 1).tolist() + [num_tokens]
        return list(zip(boundaries[:-1], boundaries[1:], strict=True))


@dataclass(frozen=True)
class SignalSumPatchingMethod(SignalPatchingMethod):
    """Equal-information (B-budget) patching: cut whenever the cumulative token
    signal since the last cut passes an information budget B.

    B is chosen from the target ratio as B = total_signal * compression_ratio
    / num_tokens, so the trace splits into ~round(num_tokens / compression_ratio)
    patches of roughly equal summed signal, giving an average patch length
    ~= compression_ratio. Implemented via a vectorized cumulative-sum +
    searchsorted (equivalent to the sequential "sum exceeds B" greedy) to avoid
    a Python loop over tokens.
    """

    def __init__(self, compression_ratio: float = 2.0, signal: str = "entropy") -> None:
        super().__init__(compression_ratio, signal, rule="sum")

    def split(
        self,
        num_tokens: int,
        sample_index: int,
        seed: int,
        values: torch.Tensor | None,
        *,
        context: SplitContext | None = None,
    ) -> list[tuple[int, int]]:
        del sample_index, seed, context
        assert values is not None
        if num_tokens <= 1:
            return [(0, num_tokens)]
        cum = torch.cumsum(values[:num_tokens].float(), dim=0)
        total = cum[-1]
        # Degenerate all-zero-signal trace: fall back to uniform chunks so we
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


@dataclass(frozen=True)
class ParagraphStepPatchingMethod(PatchingMethod):
    """Steps are `\\n\\n` paragraphs; each gets `max(1, round(len / ratio))` latents.

    The segmentation of arXiv:2508.03346, which splits the thinking content on
    `\\n\\n`. Unlike every other strategy here it is content-defined without being
    *signal*-defined: it needs no forward pass, no cache, and no GPU, only the
    trace text and its token offsets. Boundaries are therefore exactly
    reproducible between a DataLoader worker and the main process, which is why
    `required_signal()` stays None and `signal_span_counts` is correctly bypassed.

    `split` returns the latent sub-spans so that pooling and per-slot
    initialization behave as they do for any other strategy; `layout` adds the
    step grouping the encoder needs for its cross-attention bound and its
    next-step targets.
    """

    compression_ratio: float

    def __init__(self, compression_ratio: float = 4.0) -> None:
        if compression_ratio < 1.0:
            raise ValueError("compression_ratio must be >= 1.")
        super().__init__(name="paragraph")
        object.__setattr__(self, "compression_ratio", float(compression_ratio))

    @property
    def param_tag(self) -> str:
        return f"cr{self.compression_ratio:g}"

    def requires_context(self) -> bool:
        return True

    def layout(
        self,
        num_tokens: int,
        sample_index: int,
        seed: int,
        values: torch.Tensor | None,
        *,
        context: SplitContext | None = None,
    ) -> StepLayout:
        del sample_index, seed, values
        # RuntimeError, not ValueError, for both wiring failures below: the eval loop
        # reads a ValueError out of `materialize` as "this sample lacks data" and
        # counts a skip, so a caller that forgot the context would silently score
        # nothing instead of failing. A context is always constructible from the
        # trace, so its absence is a bug in the caller, never a property of the data.
        if context is None:
            raise RuntimeError(
                "paragraph patching needs a SplitContext carrying either the trace "
                "text and its token offsets, or a pre-resolved StepLayout. Callers "
                "learn this from `requires_context()`."
            )
        if context.layout is not None:
            resolved = context.layout
        elif context.text is not None and context.offsets is not None:
            resolved = segment_paragraph_steps(
                context.text, context.offsets, self.compression_ratio
            )
        else:
            raise RuntimeError(
                "SplitContext must carry `layout`, or both `text` and `offsets`."
            )
        # Cheap, and it is the only thing standing between a malformed layout and a
        # silent misalignment on the device, where `cross_limit`/`cond_slot` are
        # consumed by `gather` with no bounds check.
        resolved.validate(num_tokens)
        return resolved

    def split(
        self,
        num_tokens: int,
        sample_index: int,
        seed: int,
        values: torch.Tensor | None,
        *,
        context: SplitContext | None = None,
    ) -> list[tuple[int, int]]:
        return list(
            self.layout(num_tokens, sample_index, seed, values, context=context).latents
        )
