from __future__ import annotations

import pytest
import torch

from cot_compression.patching import (
    RandomPatchingMethod,
    SignalDiffPatchingMethod,
    SignalSumPatchingMethod,
    SignalThresholdPatchingMethod,
    UniformPatchingMethod,
)

# Every signal-patching rule is exercised under both signals: the code is
# signal-agnostic (only the name differs), so identical input arrays must yield
# identical spans regardless of which signal produced them.
SIGNAL_RULES = [
    SignalThresholdPatchingMethod,
    SignalDiffPatchingMethod,
    SignalSumPatchingMethod,
]


def _assert_partition(spans: list[tuple[int, int]], num_tokens: int) -> None:
    """Spans must tile [0, num_tokens) contiguously, in order, with no gaps."""
    assert spans[0][0] == 0
    assert spans[-1][1] == num_tokens
    for (_, end), (next_start, _) in zip(spans, spans[1:], strict=False):
        assert end == next_start
    for start, end in spans:
        assert start < end


def test_uniform_patching_covers_range_in_fixed_chunks() -> None:
    # compression_ratio is the average patch length; for uniform it is the size.
    method = UniformPatchingMethod(compression_ratio=8)
    spans = method.split(20, sample_index=0, seed=0, values=None)
    assert spans == [(0, 8), (8, 16), (16, 20)]
    assert method.param_tag == "cr8"


def test_uniform_patching_rounds_non_integer_ratio() -> None:
    # Uniform can only realize integer sizes, so 1.5 rounds to 2.
    method = UniformPatchingMethod(compression_ratio=1.5)
    assert method.patch_size == 2
    spans = method.split(6, sample_index=0, seed=0, values=None)
    assert spans == [(0, 2), (2, 4), (4, 6)]


def test_random_patching_covers_range_contiguously() -> None:
    method = RandomPatchingMethod(max_exponent=6)
    spans = method.split(100, sample_index=3, seed=42, values=None)
    _assert_partition(spans, 100)
    for start, end in spans[:-1]:
        assert (end - start) in {2**i for i in range(7)}


def test_random_patching_is_deterministic_per_seed_and_sample() -> None:
    method = RandomPatchingMethod(max_exponent=6)
    first = method.split(100, sample_index=3, seed=42, values=None)
    second = method.split(100, sample_index=3, seed=42, values=None)
    third = method.split(100, sample_index=4, seed=42, values=None)
    assert first == second
    assert first != third


@pytest.mark.parametrize(
    "cls, rule",
    [
        (SignalThresholdPatchingMethod, "threshold"),
        (SignalDiffPatchingMethod, "diff"),
        (SignalSumPatchingMethod, "sum"),
    ],
)
@pytest.mark.parametrize("signal", ["entropy", "surprisal"])
def test_signal_patching_name_composes_signal_and_rule(cls, rule, signal) -> None:
    method = cls(compression_ratio=4.0, signal=signal)
    assert method.name == f"{signal}_{rule}"
    assert method.param_tag == "cr4"
    assert method.required_signal() == signal


@pytest.mark.parametrize("signal", ["entropy", "surprisal"])
def test_threshold_starts_new_patch_before_high_signal_tokens(signal) -> None:
    # [high, low, low, high, low] -> [high,low,low] | [high,low]: a high-signal
    # token always starts a new patch. At ratio 5 the quantile is 0.8, whose
    # threshold (9.0) only the two 9.0 positions clear; index 0 is always a start.
    values = torch.tensor([9.0, 1.0, 1.0, 9.0, 1.0])
    method = SignalThresholdPatchingMethod(compression_ratio=5.0, signal=signal)
    spans = method.split(5, sample_index=0, seed=0, values=values)
    assert spans == [(0, 3), (3, 5)]


def test_threshold_handles_bfloat16_values() -> None:
    values = torch.tensor([9.0, 1.0, 1.0, 9.0, 1.0], dtype=torch.bfloat16)
    method = SignalThresholdPatchingMethod(compression_ratio=5.0)
    spans = method.split(5, sample_index=0, seed=0, values=values)
    assert spans == [(0, 3), (3, 5)]


@pytest.mark.parametrize("cls", SIGNAL_RULES)
def test_signal_methods_single_token(cls) -> None:
    method = cls(compression_ratio=2.0)
    spans = method.split(1, sample_index=0, seed=0, values=torch.tensor([5.0]))
    assert spans == [(0, 1)]


@pytest.mark.parametrize("cls", SIGNAL_RULES)
def test_signal_methods_require_ratio_at_least_one(cls) -> None:
    with pytest.raises(ValueError):
        cls(compression_ratio=0.5)


def test_signal_weighted_mean_rejects_unknown_signal() -> None:
    with pytest.raises(ValueError):
        SignalThresholdPatchingMethod(compression_ratio=2.0, signal="perplexity")


@pytest.mark.parametrize("cls", SIGNAL_RULES)
def test_signal_split_is_identical_across_signals(cls) -> None:
    # The split code is signal-agnostic: the same value array must produce the
    # same spans whether the class was tagged entropy or surprisal.
    torch.manual_seed(3)
    values = torch.rand(1500) * 5.0
    entropy_spans = cls(compression_ratio=4.0, signal="entropy").split(
        1500, sample_index=0, seed=0, values=values
    )
    surprisal_spans = cls(compression_ratio=4.0, signal="surprisal").split(
        1500, sample_index=0, seed=0, values=values
    )
    assert entropy_spans == surprisal_spans


@pytest.mark.parametrize("cls", SIGNAL_RULES)
@pytest.mark.parametrize("ratio", [2.0, 4.0, 8.0])
def test_signal_methods_realize_target_ratio(cls, ratio) -> None:
    # On a long random-signal trace every strategy should split into roughly
    # L / ratio patches (avg patch length ~= ratio). Tolerance is loose because
    # the constraint is derived per-trace from quantiles.
    torch.manual_seed(0)
    num_tokens = 4000
    values = torch.rand(num_tokens) * 5.0
    method = cls(compression_ratio=ratio)
    spans = method.split(num_tokens, sample_index=0, seed=0, values=values)
    _assert_partition(spans, num_tokens)
    avg_patch_len = num_tokens / len(spans)
    assert ratio * 0.75 <= avg_patch_len <= ratio * 1.35


def test_signal_sum_patches_carry_roughly_equal_information() -> None:
    torch.manual_seed(1)
    num_tokens = 2000
    values = torch.rand(num_tokens) * 5.0
    method = SignalSumPatchingMethod(compression_ratio=4.0)
    spans = method.split(num_tokens, sample_index=0, seed=0, values=values)
    _assert_partition(spans, num_tokens)
    budget = float(values.sum()) * 4.0 / num_tokens
    sums = [float(values[s:e].sum()) for s, e in spans]
    # Every interior patch accumulates about one budget of the signal.
    for total in sums[:-1]:
        assert 0.5 * budget <= total <= 1.6 * budget


def test_signal_sum_all_zero_falls_back_to_uniform() -> None:
    values = torch.zeros(12)
    method = SignalSumPatchingMethod(compression_ratio=4.0)
    spans = method.split(12, sample_index=0, seed=0, values=values)
    _assert_partition(spans, 12)
    assert spans == [(0, 4), (4, 8), (8, 12)]


def test_signal_diff_splits_before_jumps() -> None:
    # The signal rises sharply into positions 2 and 4; those should start patches.
    values = torch.tensor([1.0, 1.0, 9.0, 1.0, 9.0, 1.0])
    method = SignalDiffPatchingMethod(compression_ratio=3.0)
    spans = method.split(6, sample_index=0, seed=0, values=values)
    _assert_partition(spans, 6)
    starts = {s for s, _ in spans}
    assert {2, 4}.issubset(starts)


# --------------------------------------------------------------------------- #
# Paragraph steps
# --------------------------------------------------------------------------- #


def _paragraph_context(pieces: list[str]):
    """One token per piece; returns the SplitContext a worker would build."""
    from cot_compression.patching import SplitContext

    text = ""
    offsets = []
    for piece in pieces:
        offsets.append((len(text), len(text) + len(piece)))
        text += piece
    return SplitContext(text=text, offsets=offsets), len(offsets)


def test_paragraph_patching_needs_a_context() -> None:
    """The channel `split`'s four arguments cannot carry, so it must be demanded.

    A RuntimeError, deliberately not a ValueError: the eval loop reads a ValueError
    out of `materialize` as "this sample lacks data" and skips it, so a caller that
    forgot the context would silently score nothing instead of failing.
    """
    from cot_compression.patching import ParagraphStepPatchingMethod

    method = ParagraphStepPatchingMethod(compression_ratio=4.0)
    assert method.requires_context() is True
    assert method.required_signal() is None
    assert method.name == "paragraph" and method.param_tag == "cr4"
    with pytest.raises(RuntimeError, match="context"):
        method.split(10, sample_index=0, seed=0, values=None)


def test_paragraph_patching_partitions_and_groups() -> None:
    from cot_compression.patching import ParagraphStepPatchingMethod

    method = ParagraphStepPatchingMethod(compression_ratio=2.0)
    context, num_tokens = _paragraph_context(
        ["Alpha", " one", ".\n\n", "Beta", " two", " three", " four", ".\n\n", "End"]
    )
    spans = method.split(num_tokens, 0, 0, None, context=context)
    _assert_partition(spans, num_tokens)

    layout = method.layout(num_tokens, 0, 0, None, context=context)
    layout.validate(num_tokens)
    assert layout.steps == ((0, 3), (3, 8), (8, 9))
    # 3 tokens / 2 -> 2 latents; 5 / 2 -> 2 (banker's rounding of 2.5); 1 -> 1.
    assert layout.step_of_latent == (0, 0, 1, 1, 2)
    assert layout.last_latent_of_step == (1, 3, 4)
    assert list(layout.latents) == spans


def test_paragraph_patching_accepts_a_preresolved_layout() -> None:
    """How evaluation avoids a worker/main disagreement on K."""
    from cot_compression.patching import ParagraphStepPatchingMethod, SplitContext

    method = ParagraphStepPatchingMethod(compression_ratio=2.0)
    context, num_tokens = _paragraph_context(["a", ".\n\n", "b", " c", ".\n\n", "d"])
    resolved = method.layout(num_tokens, 0, 0, None, context=context)

    # Shipping the layout must give the same answer as re-deriving from the text,
    # and must not need the text at all.
    from_layout = method.layout(
        num_tokens, 0, 0, None, context=SplitContext(layout=resolved)
    )
    assert from_layout == resolved
    with pytest.raises(RuntimeError, match="layout"):
        method.layout(num_tokens, 0, 0, None, context=SplitContext())


def test_paragraph_layout_is_rejected_when_it_does_not_match_the_trace() -> None:
    """A shipped layout is validated against the token count it is used with."""
    from cot_compression.patching import ParagraphStepPatchingMethod, SplitContext

    method = ParagraphStepPatchingMethod(compression_ratio=2.0)
    context, num_tokens = _paragraph_context(["a", ".\n\n", "b", " c"])
    resolved = method.layout(num_tokens, 0, 0, None, context=context)
    with pytest.raises(ValueError, match="ends at"):
        method.layout(num_tokens + 3, 0, 0, None, context=SplitContext(layout=resolved))


def test_default_layout_is_one_step_per_span() -> None:
    """The degeneracy every existing strategy inherits, and the encoder relies on."""
    method = UniformPatchingMethod(compression_ratio=4.0)
    spans = method.split(11, 0, 0, None)
    layout = method.layout(11, 0, 0, None)
    layout.validate(11)
    assert list(layout.steps) == list(layout.latents) == spans
    assert layout.step_of_latent == tuple(range(len(spans)))
    assert layout.last_latent_of_step == tuple(range(len(spans)))


def test_paragraph_name_composes_into_a_new_join_key() -> None:
    """New method names, so no existing artifact is invalidated."""
    from cot_compression.compression import SimpleMeanCompressionMethod
    from cot_compression.patching import ParagraphStepPatchingMethod

    method = SimpleMeanCompressionMethod(
        patching=ParagraphStepPatchingMethod(compression_ratio=4.0)
    )
    assert method.name == "simple_mean_paragraph_cr4"


def test_build_patching_method_registers_paragraph() -> None:
    from omegaconf import OmegaConf

    from cot_compression.compression import build_patching_method

    cfg = OmegaConf.create({"compression_ratio": 8.0, "random": {"max_exponent": 6}})
    method = build_patching_method("paragraph", cfg)
    assert method is not None
    assert method.name == "paragraph" and method.param_tag == "cr8"
