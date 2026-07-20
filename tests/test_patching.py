from __future__ import annotations

import pytest
import torch

from cot_compression.patching import (
    EntropyDiffPatchingMethod,
    EntropySumPatchingMethod,
    EntropyThresholdPatchingMethod,
    RandomPatchingMethod,
    UniformPatchingMethod,
)


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
    spans = method.split(20, sample_index=0, seed=0, entropies=None)
    assert spans == [(0, 8), (8, 16), (16, 20)]
    assert method.param_tag == "cr8"


def test_uniform_patching_rounds_non_integer_ratio() -> None:
    # Uniform can only realize integer sizes, so 1.5 rounds to 2.
    method = UniformPatchingMethod(compression_ratio=1.5)
    assert method.patch_size == 2
    spans = method.split(6, sample_index=0, seed=0, entropies=None)
    assert spans == [(0, 2), (2, 4), (4, 6)]


def test_random_patching_covers_range_contiguously() -> None:
    method = RandomPatchingMethod(max_exponent=6)
    spans = method.split(100, sample_index=3, seed=42, entropies=None)
    _assert_partition(spans, 100)
    for start, end in spans[:-1]:
        assert (end - start) in {2**i for i in range(7)}


def test_random_patching_is_deterministic_per_seed_and_sample() -> None:
    method = RandomPatchingMethod(max_exponent=6)
    first = method.split(100, sample_index=3, seed=42, entropies=None)
    second = method.split(100, sample_index=3, seed=42, entropies=None)
    third = method.split(100, sample_index=4, seed=42, entropies=None)
    assert first == second
    assert first != third


def test_entropy_threshold_starts_new_patch_before_high_entropy_tokens() -> None:
    # [high, low, low, high, low] -> [high,low,low] | [high,low]: a high-entropy
    # token always starts a new patch. At ratio 5 the quantile is 0.8, whose
    # threshold (9.0) only the two 9.0 positions clear; index 0 is always a start.
    entropies = torch.tensor([9.0, 1.0, 1.0, 9.0, 1.0])
    method = EntropyThresholdPatchingMethod(compression_ratio=5.0)
    spans = method.split(5, sample_index=0, seed=0, entropies=entropies)
    assert spans == [(0, 3), (3, 5)]


def test_entropy_threshold_handles_bfloat16_entropies() -> None:
    entropies = torch.tensor([9.0, 1.0, 1.0, 9.0, 1.0], dtype=torch.bfloat16)
    method = EntropyThresholdPatchingMethod(compression_ratio=5.0)
    spans = method.split(5, sample_index=0, seed=0, entropies=entropies)
    assert spans == [(0, 3), (3, 5)]


@pytest.mark.parametrize(
    "cls",
    [
        EntropyThresholdPatchingMethod,
        EntropyDiffPatchingMethod,
        EntropySumPatchingMethod,
    ],
)
def test_entropy_methods_single_token(cls) -> None:
    method = cls(compression_ratio=2.0)
    spans = method.split(1, sample_index=0, seed=0, entropies=torch.tensor([5.0]))
    assert spans == [(0, 1)]


@pytest.mark.parametrize(
    "cls",
    [
        EntropyThresholdPatchingMethod,
        EntropyDiffPatchingMethod,
        EntropySumPatchingMethod,
    ],
)
def test_entropy_methods_require_ratio_at_least_one(cls) -> None:
    with pytest.raises(ValueError):
        cls(compression_ratio=0.5)


@pytest.mark.parametrize(
    "cls",
    [
        EntropyThresholdPatchingMethod,
        EntropyDiffPatchingMethod,
        EntropySumPatchingMethod,
    ],
)
@pytest.mark.parametrize("ratio", [2.0, 4.0, 8.0])
def test_entropy_methods_realize_target_ratio(cls, ratio) -> None:
    # On a long random-entropy trace every strategy should split into roughly
    # L / ratio patches (avg patch length ~= ratio). Tolerance is loose because
    # the constraint is derived per-trace from quantiles.
    torch.manual_seed(0)
    num_tokens = 4000
    entropies = torch.rand(num_tokens) * 5.0
    method = cls(compression_ratio=ratio)
    spans = method.split(num_tokens, sample_index=0, seed=0, entropies=entropies)
    _assert_partition(spans, num_tokens)
    avg_patch_len = num_tokens / len(spans)
    assert ratio * 0.75 <= avg_patch_len <= ratio * 1.35


def test_entropy_sum_patches_carry_roughly_equal_information() -> None:
    torch.manual_seed(1)
    num_tokens = 2000
    entropies = torch.rand(num_tokens) * 5.0
    method = EntropySumPatchingMethod(compression_ratio=4.0)
    spans = method.split(num_tokens, sample_index=0, seed=0, entropies=entropies)
    _assert_partition(spans, num_tokens)
    budget = float(entropies.sum()) * 4.0 / num_tokens
    sums = [float(entropies[s:e].sum()) for s, e in spans]
    # Every interior patch accumulates about one budget of entropy.
    for total in sums[:-1]:
        assert 0.5 * budget <= total <= 1.6 * budget


def test_entropy_sum_all_zero_entropy_falls_back_to_uniform() -> None:
    entropies = torch.zeros(12)
    method = EntropySumPatchingMethod(compression_ratio=4.0)
    spans = method.split(12, sample_index=0, seed=0, entropies=entropies)
    _assert_partition(spans, 12)
    assert spans == [(0, 4), (4, 8), (8, 12)]


def test_entropy_diff_splits_before_entropy_jumps() -> None:
    # Entropy rises sharply into positions 2 and 4; those should start patches.
    entropies = torch.tensor([1.0, 1.0, 9.0, 1.0, 9.0, 1.0])
    method = EntropyDiffPatchingMethod(compression_ratio=3.0)
    spans = method.split(6, sample_index=0, seed=0, entropies=entropies)
    _assert_partition(spans, 6)
    starts = {s for s, _ in spans}
    assert {2, 4}.issubset(starts)
