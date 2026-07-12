from __future__ import annotations

import torch

from cot_compression.patching import (
    EntropyPatchingMethod,
    RandomPatchingMethod,
    UniformPatchingMethod,
)


def test_uniform_patching_covers_range_in_fixed_chunks() -> None:
    method = UniformPatchingMethod(patch_size=8)
    spans = method.split(20, sample_index=0, seed=0, entropies=None)
    assert spans == [(0, 8), (8, 16), (16, 20)]


def test_uniform_patching_exact_multiple() -> None:
    method = UniformPatchingMethod(patch_size=4)
    spans = method.split(12, sample_index=0, seed=0, entropies=None)
    assert spans == [(0, 4), (4, 8), (8, 12)]


def test_random_patching_covers_range_contiguously() -> None:
    method = RandomPatchingMethod(max_exponent=6)
    spans = method.split(100, sample_index=3, seed=42, entropies=None)

    assert spans[0][0] == 0
    assert spans[-1][1] == 100
    for (_, end), (next_start, _) in zip(spans, spans[1:], strict=False):
        assert end == next_start

    for start, end in spans[:-1]:
        length = end - start
        assert length in {2**i for i in range(7)}


def test_random_patching_is_deterministic_per_seed_and_sample() -> None:
    method = RandomPatchingMethod(max_exponent=6)
    first = method.split(100, sample_index=3, seed=42, entropies=None)
    second = method.split(100, sample_index=3, seed=42, entropies=None)
    third = method.split(100, sample_index=4, seed=42, entropies=None)

    assert first == second
    assert first != third


def test_entropy_patching_starts_new_patch_before_high_entropy_tokens() -> None:
    # [high, low, low, high, low] -> [high,low,low] | [high,low]: a
    # high-entropy token always starts a new patch, never gets grouped
    # with what preceded it.
    entropies = torch.tensor([9.0, 1.0, 1.0, 9.0, 1.0])
    method = EntropyPatchingMethod(percentile=80.0)
    spans = method.split(5, sample_index=0, seed=0, entropies=entropies)
    assert spans == [(0, 3), (3, 5)]


def test_entropy_patching_handles_bfloat16_entropies() -> None:
    entropies = torch.tensor([9.0, 1.0, 1.0, 9.0, 1.0], dtype=torch.bfloat16)
    method = EntropyPatchingMethod(percentile=80.0)
    spans = method.split(5, sample_index=0, seed=0, entropies=entropies)
    assert spans == [(0, 3), (3, 5)]


def test_entropy_patching_single_token() -> None:
    method = EntropyPatchingMethod(percentile=80.0)
    spans = method.split(1, sample_index=0, seed=0, entropies=torch.tensor([5.0]))
    assert spans == [(0, 1)]


def test_entropy_patching_all_tied_entropies_splits_every_token() -> None:
    # With every entropy tied, the percentile threshold equals that same
    # value, so every position satisfies entropies[i] >= threshold and
    # becomes its own single-token patch -- a degenerate but faithful
    # reading of the rule when there's no real signal to split on.
    entropies = torch.tensor([1.0, 1.0, 1.0, 1.0])
    method = EntropyPatchingMethod(percentile=80.0)
    spans = method.split(4, sample_index=0, seed=0, entropies=entropies)
    assert spans == [(0, 1), (1, 2), (2, 3), (3, 4)]
