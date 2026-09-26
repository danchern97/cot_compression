"""Paragraph-step segmentation: boundaries, latent allocation, layout invariants.

Most of these need no tokenizer -- a `(text, offsets)` pair is all
`step_token_spans` consumes, and hand-building it keeps the tests fast and makes
the adversarial cases (an 18-newline run, a token carrying an interior break)
expressible at all. One test does go through the real Qwen3 tokenizer, because the
whole design rests on how that tokenizer merges `\\n\\n` into the preceding token
and a synthetic offsets array cannot check that claim.
"""

from __future__ import annotations

import pytest

from cot_compression.steps import (
    PARAGRAPH_BREAK,
    build_step_layout,
    even_subspans,
    latents_per_step,
    segment_paragraph_steps,
    step_token_spans,
    trivial_step_layout,
)


def _offsets(pieces: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Treat each piece as one token; return the joined text and its offsets."""
    text = ""
    offsets = []
    for piece in pieces:
        offsets.append((len(text), len(text) + len(piece)))
        text += piece
    return text, offsets


def _assert_tiles(spans: list[tuple[int, int]], num_tokens: int) -> None:
    assert spans[0][0] == 0
    assert spans[-1][1] == num_tokens
    for (_, end), (next_start, _) in zip(spans, spans[1:], strict=False):
        assert end == next_start
    for start, end in spans:
        assert start < end


# --------------------------------------------------------------------------
# step_token_spans
# --------------------------------------------------------------------------


def test_breaks_after_the_token_holding_the_delimiter() -> None:
    # The delimiter stays with the step it terminates, which is what the tokenizer
    # wants: the boundary token is `.\n\n` and carries the previous sentence's period.
    text, offsets = _offsets(["Alpha", ".\n\n", "Beta", ".\n\n", "Gamma"])
    assert step_token_spans(text, offsets) == [(0, 2), (2, 4), (4, 5)]


def test_a_maximal_newline_run_is_exactly_one_break() -> None:
    # `\n{2,}` is maximal, so three or four newlines never produce an empty step.
    for filler in ("\n\n", "\n\n\n", "\n\n\n\n"):
        text, offsets = _offsets(["Alpha", filler, "Beta"])
        assert step_token_spans(text, offsets) == [(0, 2), (2, 3)], filler


def test_a_run_straddling_two_tokens_breaks_once() -> None:
    # Qwen3 tokenizes 18 newlines as two tokens. Keying on the run's LAST character
    # yields one break; an "id ends with >= 2 newlines" rule would fire twice and
    # emit a degenerate step between them.
    text, offsets = _offsets(["Alpha", "\n" * 8, "\n" * 10, "Beta"])
    assert step_token_spans(text, offsets) == [(0, 3), (3, 4)]


def test_a_break_inside_a_token_attaches_to_the_preceding_step() -> None:
    # Token 89253 is `'\n\n    \n'` -- the one Qwen3 token whose double newline is
    # not at the end. The break lands after the whole token.
    text, offsets = _offsets(["def f():", "\n\n    \n", "    return 1"])
    assert step_token_spans(text, offsets) == [(0, 2), (2, 3)]


def test_no_delimiter_is_a_single_step() -> None:
    text, offsets = _offsets(["<think>", "\n", "One line only.", "\n", "</think>"])
    assert step_token_spans(text, offsets) == [(0, 5)]


def test_a_trailing_delimiter_does_not_make_an_empty_step() -> None:
    # `re.split` would yield a final `''` here; a step of zero tokens cannot tile,
    # so the break at the sequence end is dropped instead. The one deliberate
    # deviation from a literal text split.
    text, offsets = _offsets(["Alpha", ".\n\n", "Beta", ".\n\n"])
    spans = step_token_spans(text, offsets)
    assert spans == [(0, 2), (2, 4)]
    _assert_tiles(spans, len(offsets))


def test_a_leading_delimiter_does_not_make_an_empty_step() -> None:
    text, offsets = _offsets(["\n\n", "Alpha", ".\n\n", "Beta"])
    spans = step_token_spans(text, offsets)
    _assert_tiles(spans, len(offsets))
    assert spans == [(0, 1), (1, 3), (3, 4)]


def test_empty_token_sequence_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty token sequence"):
        step_token_spans("", [])


def test_matches_a_naive_text_split_on_random_inputs() -> None:
    """The vectorized searchsorted against the obvious O(S*L) oracle.

    Random word/newline-run texts, one token per word and one per run, which is how
    the real pre-tokenizer behaves (`\\s*[\\r\\n]+` isolates a whitespace run).
    """
    import random

    rng = random.Random(1337)
    for _ in range(200):
        pieces = []
        for index in range(rng.randint(1, 40)):
            pieces.append(f"w{index}")
            if rng.random() < 0.4:
                pieces.append("\n" * rng.randint(1, 5))
        text, offsets = _offsets(pieces)
        spans = step_token_spans(text, offsets)
        _assert_tiles(spans, len(offsets))

        # Oracle: for each break, scan for the token containing its last character.
        naive = []
        for match in PARAGRAPH_BREAK.finditer(text):
            target = match.end() - 1
            for index, (start, end) in enumerate(offsets):
                if start <= target < end:
                    naive.append(index + 1)
                    break
        kept = sorted({b for b in naive if 0 < b < len(offsets)})
        bounds = [0, *kept, len(offsets)]
        assert spans == list(zip(bounds[:-1], bounds[1:], strict=True))

        # And the text of each step must equal its paragraph, delimiter aside.
        pieces_text = [p for p in PARAGRAPH_BREAK.split(text) if p]
        recovered = [text[offsets[s][0] : offsets[e - 1][1]].strip() for s, e in spans]
        assert [r for r in recovered if r] == [
            p.strip() for p in pieces_text if p.strip()
        ]


# --------------------------------------------------------------------------
# latents_per_step / even_subspans
# --------------------------------------------------------------------------


def test_latents_per_step_matches_the_specified_example() -> None:
    # "if a step is of original length of 6, and compression ratio is 2, then we
    # will have 3 latent tokens".
    assert latents_per_step(6, 2.0) == 3
    assert latents_per_step(24, 4.0) == 6


def test_latents_per_step_is_at_least_one() -> None:
    # A step shorter than the ratio still needs a code; this floor is why the
    # realized ratio (3.99 at r=4) sits just under the nominal one.
    for length in (1, 2, 3):
        assert latents_per_step(length, 4.0) == 1


def test_latents_per_step_rounds_rather_than_ceils() -> None:
    # `round`, measured to track the nominal ratio; `ceil` would give 2 here and
    # under-compress. Banker's rounding, hence 2.5 -> 2.
    assert latents_per_step(5, 4.0) == 1
    assert latents_per_step(10, 4.0) == 2
    assert latents_per_step(6, 4.0) == 2


def test_latents_per_step_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="at least one token"):
        latents_per_step(0, 4.0)
    with pytest.raises(ValueError, match="compression_ratio"):
        latents_per_step(4, 0.5)


def test_even_subspans_tile_and_differ_by_at_most_one() -> None:
    for span, count in ((24, 6), (25, 6), (7, 3), (5, 5), (100, 7)):
        spans = even_subspans(10, 10 + span, count)
        assert len(spans) == count
        assert spans[0][0] == 10 and spans[-1][1] == 10 + span
        for (_, end), (next_start, _) in zip(spans, spans[1:], strict=False):
            assert end == next_start
        sizes = {end - start for start, end in spans}
        assert max(sizes) - min(sizes) <= 1


def test_even_subspans_collapses_to_the_whole_span_at_one() -> None:
    # This is what makes `query_anchor=substep` degenerate into the default when a
    # step gets a single latent.
    assert even_subspans(10, 34, 1) == [(10, 34)]


def test_even_subspans_rejects_more_spans_than_tokens() -> None:
    with pytest.raises(ValueError, match="Cannot split"):
        even_subspans(0, 3, 4)


# --------------------------------------------------------------------------
# StepLayout
# --------------------------------------------------------------------------


def test_build_step_layout_allocates_and_validates() -> None:
    layout = build_step_layout([(0, 24), (24, 27), (27, 33)], compression_ratio=4.0)
    layout.validate(33)
    assert layout.num_steps == 3
    # 24/4 = 6 latents, 3/4 -> 1, 6/4 -> 2 (banker's rounding of 1.5).
    assert layout.num_latents == 9
    assert layout.step_of_latent == (0, 0, 0, 0, 0, 0, 1, 2, 2)
    assert layout.last_latent_of_step == (5, 6, 8)
    assert layout.latents[:6] == ((0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24))
    assert layout.latents[6] == (24, 27)


def test_trivial_layout_is_one_step_per_span() -> None:
    """The degeneracy guarantee the whole design rests on.

    With this, `cross_limit` under `uniform` resolves to `prompt_len + span_end`
    and `cond_slot[m]` to `m - 1` -- byte-identical to the pre-step behaviour.
    """
    spans = [(0, 4), (4, 8), (8, 11)]
    layout = trivial_step_layout(spans)
    layout.validate(11)
    assert layout.steps == layout.latents == tuple(spans)
    assert layout.step_of_latent == (0, 1, 2)
    assert layout.last_latent_of_step == (0, 1, 2)


def test_validate_rejects_a_latent_outside_its_step() -> None:
    from cot_compression.steps import StepLayout

    broken = StepLayout(
        steps=((0, 4), (4, 8)),
        latents=((0, 6), (6, 8)),
        step_of_latent=(0, 1),
        last_latent_of_step=(0, 1),
    )
    with pytest.raises(ValueError, match="outside its step"):
        broken.validate(8)


def test_validate_rejects_a_non_tiling_layout() -> None:
    from cot_compression.steps import StepLayout

    broken = StepLayout(
        steps=((0, 4), (5, 8)),
        latents=((0, 4), (5, 8)),
        step_of_latent=(0, 1),
        last_latent_of_step=(0, 1),
    )
    with pytest.raises(ValueError, match="does not tile"):
        broken.validate(8)


def test_segment_paragraph_steps_end_to_end() -> None:
    text, offsets = _offsets(
        ["Alpha", ".\n\n", "Beta", " two", " three", ".\n\n", "End"]
    )
    layout = segment_paragraph_steps(text, offsets, compression_ratio=2.0)
    layout.validate(len(offsets))
    assert layout.steps == ((0, 2), (2, 6), (6, 7))
    assert layout.num_latents == 1 + 2 + 1
    assert layout.step_of_latent == (0, 1, 1, 2)
    assert layout.last_latent_of_step == (0, 2, 3)


def test_single_step_trace_falls_back_to_uniform_chunks() -> None:
    # The 1.9% of traces with no `\n\n`: one step, so the layout is exactly what
    # uniform patching would have produced.
    text, offsets = _offsets([f"w{i}" for i in range(12)])
    layout = segment_paragraph_steps(text, offsets, compression_ratio=4.0)
    layout.validate(12)
    assert layout.steps == ((0, 12),)
    assert layout.latents == ((0, 4), (4, 8), (8, 12))


# --------------------------------------------------------------------------
# The real tokenizer
# --------------------------------------------------------------------------


def test_segmentation_on_a_real_qwen3_tokenization() -> None:
    """The claim a synthetic offsets array cannot check: `\\n\\n` merges left."""
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception as error:  # pragma: no cover - offline box without the cache
        pytest.skip(f"Qwen3 tokenizer unavailable: {error}")

    text = "<think>\nFirst, sum them.\n\nNext, subtract:\n5000 - 25\n\nSo N = 25.\n</think>"
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    spans = step_token_spans(text, offsets)

    _assert_tiles(spans, len(offsets))
    assert len(spans) == len(PARAGRAPH_BREAK.split(text))
    recovered = [text[offsets[s][0] : offsets[e - 1][1]] for s, e in spans]
    assert "".join(recovered) == text
    # The delimiter belongs to the step it ends, and `<think>` is inside step 0.
    assert recovered[0].startswith("<think>")
    assert recovered[0].endswith("\n\n")
    assert recovered[-1].endswith("</think>")
    assert [piece.strip() for piece in recovered] == [
        piece.strip() for piece in PARAGRAPH_BREAK.split(text)
    ]


def test_the_boundary_token_is_usually_not_a_bare_newline_pair() -> None:
    """Guards the reason offsets are used instead of a token-id set."""
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception as error:  # pragma: no cover
        pytest.skip(f"Qwen3 tokenizer unavailable: {error}")

    ids = tokenizer("a.\n\nNext", add_special_tokens=False)["input_ids"]
    decoded = [tokenizer.decode([i]) for i in ids]
    assert ".\n\n" in decoded, decoded
    assert "\n\n" not in decoded, "a bare \\n\\n token would make the id rule look safe"

    # And no token holds two separate runs, so breaks can never collapse.
    doubles = [
        token
        for token in tokenizer.get_vocab()
        if len(PARAGRAPH_BREAK.findall(tokenizer.convert_tokens_to_string([token]))) > 1
    ]
    assert doubles == []
