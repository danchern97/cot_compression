"""Invariants of the grouped, domain-stratified trace split.

The split is the one place where a silent bug is unrecoverable: because the
training target is the answer, a prompt shared between train and val leaks the
exact quantity the eval measures, and nothing downstream would flag it. These
tests assert the properties that make the split safe rather than checking golden
values, which would only pin the current shuffle.
"""

from __future__ import annotations

import numpy as np
import pytest

from cot_compression.data.dolci_traces import (
    SPLITS,
    FilterReport,
    assert_no_prompt_overlap,
    split_trace_groups,
)


def _corpus(
    prompts_per_domain: dict[str, int],
    traces_per_prompt: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic corpus shaped like the real one: every prompt repeated."""
    prompt_ids, domains, rollouts = [], [], []
    for domain, count in prompts_per_domain.items():
        for index in range(count):
            for rollout in range(traces_per_prompt):
                prompt_ids.append(f"{domain}-{index:05d}")
                domains.append(domain)
                rollouts.append(rollout)
    return (
        np.asarray(prompt_ids, dtype=object),
        np.asarray(domains, dtype=object),
        np.asarray(rollouts, dtype=np.int64),
    )


DOMAINS = {"math": 400, "code": 260, "ifeval": 700, "general_quality": 120}
FRACTIONS = (0.90, 0.05, 0.05)


def _split(**overrides):
    prompt_ids, domains, rollouts = overrides.pop("corpus", _corpus(DOMAINS))
    indices = split_trace_groups(
        prompt_ids,
        domains,
        rollouts,
        fractions=overrides.pop("fractions", FRACTIONS),
        seed=overrides.pop("seed", 1337),
    )
    return indices, prompt_ids, domains, rollouts


def test_no_prompt_appears_in_two_splits():
    indices, prompt_ids, _, _ = _split()
    assert_no_prompt_overlap(indices, prompt_ids)  # raises on contamination
    seen: set[str] = set()
    for name in SPLITS:
        prompts = set(prompt_ids[indices[name]].tolist())
        assert not (prompts & seen), f"{name} overlaps an earlier split"
        seen |= prompts


def test_every_prompt_lands_in_exactly_one_split():
    """Rows may be dropped by step 7, but no prompt may be lost entirely."""
    indices, prompt_ids, _, _ = _split()
    assigned = [set(prompt_ids[indices[name]].tolist()) for name in SPLITS]
    union = set().union(*assigned)
    assert union == set(prompt_ids.tolist())
    assert sum(len(part) for part in assigned) == len(union)


def test_domain_proportions_are_preserved_at_prompt_level():
    indices, prompt_ids, domains, _ = _split()
    total = sum(DOMAINS.values())
    for domain, count in DOMAINS.items():
        expected = count / total
        for name in SPLITS:
            rows = indices[name]
            here = np.unique(prompt_ids[rows][domains[rows] == domain]).size
            denom = np.unique(prompt_ids[rows]).size
            # Floor boundaries per stratum, so proportions are exact to within
            # one prompt per domain, not bitwise.
            assert abs(here / denom - expected) < 0.02, (name, domain)


def test_val_and_test_hold_one_trace_per_prompt():
    """What makes `sem_logprob` correct: eval samples must be independent."""
    indices, prompt_ids, _, _ = _split()
    for name in ("val", "test"):
        rows = indices[name]
        assert np.unique(prompt_ids[rows]).size == rows.size
    train_rows = indices["train"]
    assert np.unique(prompt_ids[train_rows]).size < train_rows.size


def test_val_selection_takes_the_lowest_rollout_index():
    corpus = _corpus(DOMAINS, traces_per_prompt=4)
    indices, prompt_ids, _, rollouts = _split(corpus=corpus)
    for name in ("val", "test"):
        rows = indices[name]
        assert set(rollouts[rows].tolist()) == {0}


def test_split_is_deterministic_for_a_fixed_seed():
    first, _, _, _ = _split()
    second, _, _, _ = _split()
    for name in SPLITS:
        assert np.array_equal(first[name], second[name])


def test_split_is_invariant_to_input_row_order():
    """Group ids come from sorted unique prompt_ids, so shuffling rows cannot
    move a prompt between splits -- only reorder rows within one."""
    prompt_ids, domains, rollouts = _corpus(DOMAINS)
    baseline = split_trace_groups(
        prompt_ids, domains, rollouts, fractions=FRACTIONS, seed=1337
    )
    perm = np.random.default_rng(0).permutation(prompt_ids.size)
    shuffled = split_trace_groups(
        prompt_ids[perm], domains[perm], rollouts[perm], fractions=FRACTIONS, seed=1337
    )
    for name in SPLITS:
        assert set(prompt_ids[baseline[name]].tolist()) == set(
            prompt_ids[perm][shuffled[name]].tolist()
        )


def test_a_prompt_spanning_two_domains_raises():
    prompt_ids = np.asarray(["p0", "p0", "p1", "p1"], dtype=object)
    domains = np.asarray(["math", "code", "math", "math"], dtype=object)
    rollouts = np.asarray([0, 1, 0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="spans more than one domain"):
        split_trace_groups(
            prompt_ids, domains, rollouts, fractions=FRACTIONS, seed=1337
        )


def test_a_domain_too_small_to_split_raises():
    with pytest.raises(ValueError, match="too few to split"):
        split_trace_groups(
            *_corpus({"math": 400, "tiny": 3}),
            fractions=FRACTIONS,
            seed=1337,
        )


@pytest.mark.parametrize("fractions", [(0.9, 0.05, 0.1), (0.9, 0.1), (1.0, 0.0, 0.0)])
def test_invalid_fractions_raise(fractions):
    with pytest.raises(ValueError):
        split_trace_groups(*_corpus(DOMAINS), fractions=fractions, seed=1337)


def test_filter_report_surfaces_flag_disagreement():
    """`complete` is authoritative; the other flags are reported to expose drift.

    On the real v1 snapshot 81 rows satisfy `ended_by_eos & ~truncated` yet are
    not `complete`, so a conjunction would keep rows the dataset calls unusable.
    """
    report = FilterReport(
        total=100,
        kept=90,
        dropped_incomplete=10,
        not_ended_by_eos=8,
        truncated=8,
        complete_disagrees_with_flags=2,
    )
    text = report.format()
    assert "kept 90/100" in text
    assert "complete!=(eos & ~truncated) on 2 rows" in text
