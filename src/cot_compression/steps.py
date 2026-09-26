"""Paragraph steps: `\\n\\n`-delimited reasoning units, and their latent sub-spans.

A *step* is a paragraph of the CoT, the unit arXiv:2508.03346 uses ("the reasoning
steps S_1..S_N, delimited by \\n\\n, are extracted from the thinking content C").
A step gets `max(1, round(len / compression_ratio))` latents, so a long step buys
more capacity while the realized ratio still tracks the knob.

Two levels, because the encoder needs both and they are not the same partition:
the *step* bounds what a latent may cross-attend to, while the *latent sub-span*
is what `PatchingMethod.split` has always returned and what every pooling and
initialization path consumes. `StepLayout` carries them together;
`trivial_step_layout` makes token patching the degenerate one-step-per-span case,
so nothing downstream needs a branch.

Boundaries are found from **character offsets**, not token ids. `\\n\\n` merges into
the preceding token in Qwen3's BPE (565 vocab entries contain a double newline;
only 8.9% of real breaks are the bare `\\n\\n` token), and the "id ends with >= 2
newlines" shortcut is wrong twice over: token 89253 (`'\\n\\n    \\n'`) carries an
interior break, and a run of 18 newlines splits across two tokens, firing the rule
twice and emitting a degenerate step. Keying on the *last character* of each
maximal run does neither.

Verified against a text-level oracle on 3,000 real val traces: 428,152 paragraphs
-> 428,152 steps, zero count mismatches, zero text mismatches, and the steps tile
the trace exactly. No vocabulary token contains two separate `\\n{2,}` runs, so two
breaks can never collapse onto one token -- that is a property of the vocabulary,
not of the sample.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

Span = tuple[int, int]

# Maximal runs, so `\n\n\n` is ONE break rather than an empty step between two.
# Deliberately not `(?:\r?\n[ \t]*){2,}` (blank lines holding spaces or tabs):
# that finds +389 of 714,164 breaks, +0.054%, in 56 of 5,000 traces, and deviates
# from the paper's literal delimiter. There is no `\r` at all in this corpus.
PARAGRAPH_BREAK = re.compile(r"\n{2,}")


@dataclass(frozen=True)
class StepLayout:
    """A trace's two-level segmentation. Both levels tile ``[0, num_tokens)``."""

    steps: tuple[Span, ...]
    """Token span of each reasoning step."""
    latents: tuple[Span, ...]
    """Token sub-span of each latent. This is what ``split`` returns."""
    step_of_latent: tuple[int, ...]
    """Which step each latent belongs to; non-decreasing."""
    last_latent_of_step: tuple[int, ...]
    """The final latent index of each step -- what the next step conditions on."""

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def num_latents(self) -> int:
        return len(self.latents)

    def validate(self, num_tokens: int) -> None:
        """Assert every invariant the GPU path relies on, on the CPU.

        Called once per sample in the DataLoader worker. Checking here means the
        device code can index with `gather` and compare with `<` without a syncing
        assert, and a malformed layout fails on the row that produced it rather
        than as a shape error three functions away.
        """
        _assert_tiles(self.steps, num_tokens, "steps")
        _assert_tiles(self.latents, num_tokens, "latents")
        if len(self.step_of_latent) != len(self.latents):
            raise ValueError(
                f"step_of_latent has {len(self.step_of_latent)} entries for "
                f"{len(self.latents)} latents."
            )
        if len(self.last_latent_of_step) != len(self.steps):
            raise ValueError(
                f"last_latent_of_step has {len(self.last_latent_of_step)} entries "
                f"for {len(self.steps)} steps."
            )
        previous = -1
        for latent, step in enumerate(self.step_of_latent):
            if step < previous:
                raise ValueError(f"step_of_latent is not non-decreasing at {latent}.")
            previous = step
            start, end = self.latents[latent]
            step_start, step_end = self.steps[step]
            # The `end <= step_end` half is the `query_limit <= cross_limit`
            # guarantee: a latent's RoPE anchor must lie inside what it may attend.
            if start < step_start or end > step_end:
                raise ValueError(
                    f"latent {latent} spans {(start, end)}, outside its step "
                    f"{(step_start, step_end)}."
                )
        for step, latent in enumerate(self.last_latent_of_step):
            if self.step_of_latent[latent] != step:
                raise ValueError(
                    f"last_latent_of_step[{step}] = {latent} belongs to step "
                    f"{self.step_of_latent[latent]}."
                )
            if (
                latent + 1 < len(self.latents)
                and self.step_of_latent[latent + 1] == step
            ):
                raise ValueError(f"last_latent_of_step[{step}] = {latent} is not last.")


def _assert_tiles(spans: tuple[Span, ...], num_tokens: int, what: str) -> None:
    if not spans:
        raise ValueError(f"{what} is empty.")
    cursor = 0
    for start, end in spans:
        if start != cursor or end <= start:
            raise ValueError(f"{what} does not tile [0, {num_tokens}): {spans[:8]}...")
        cursor = end
    if cursor != num_tokens:
        raise ValueError(f"{what} ends at {cursor}, not {num_tokens}.")


def step_token_spans(text: str, offsets: Sequence[Span]) -> list[Span]:
    """Token spans of ``text``'s paragraphs, given its per-token char offsets.

    A break is placed *after* the token containing the last character of each
    maximal newline run, so the delimiter stays with the step it terminates --
    which is also what the tokenizer wants, since the boundary token is usually
    `'.\\n\\n'` and carries the preceding sentence's period.

    Vectorized on purpose: `searchsorted` over the token start offsets is
    O(S log L), where the obvious nested loop is O(S*L) and S reaches 1,072.

    Breaks at token 0 or at the end are dropped rather than emitted, so a leading
    or trailing `\\n\\n` never produces an empty step. That is the one place this
    deviates from a literal `text.split("\\n\\n")`, which would yield `''` there.
    """
    length = len(offsets)
    if length == 0:
        raise ValueError("Cannot segment an empty token sequence.")
    last_chars = [match.end() - 1 for match in PARAGRAPH_BREAK.finditer(text)]
    if not last_chars:
        return [(0, length)]
    starts = np.fromiter((start for start, _ in offsets), np.int64, length)
    containing = (
        np.searchsorted(starts, np.array(last_chars, np.int64), side="right") - 1
    )
    # `unique` also sorts. It can only ever collapse entries if one token held two
    # separate runs, which no Qwen3 token does -- kept because it costs nothing and
    # makes the tiling invariant hold for any tokenizer.
    breaks = np.unique(containing + 1)
    breaks = breaks[(breaks > 0) & (breaks < length)]
    bounds = np.concatenate(([0], breaks, [length]))
    return list(zip(bounds[:-1].tolist(), bounds[1:].tolist(), strict=True))


def latents_per_step(length: int, compression_ratio: float) -> int:
    """How many latents a step of ``length`` tokens gets. At least one.

    `round`, not `ceil`, and matching `UniformPatchingMethod`'s own
    `max(1, round(compression_ratio))`. Measured over 1,500 val traces, the
    realized ratio `sum(L) / sum(latents)` is 3.99 at r=4 and 15.08 at r=16 with
    `round`, against 3.85 and 13.37 with `ceil` -- which also inflates K, and with
    it the `[B, 1, K, M]` cross mask, by ~20% at r=16.

    This is Python's banker's rounding, so `round(1.5) == 2` but `round(2.5) == 2`.
    Stated because it is load-bearing and looks like a bug to a future reader.
    """
    if length <= 0:
        raise ValueError(f"A step must have at least one token, got {length}.")
    if compression_ratio < 1.0:
        raise ValueError(f"compression_ratio must be >= 1, got {compression_ratio}.")
    return max(1, round(length / compression_ratio))


def even_subspans(start: int, end: int, count: int) -> list[Span]:
    """Split ``[start, end)`` into ``count`` spans whose sizes differ by at most 1.

    Arbitrary but fixed, and pinned by a test so it is not silently "improved":
    it decides the `substep` RoPE anchors and the `simple_mean` init partition.
    Collapses to the whole span at ``count == 1``.
    """
    span = end - start
    if count < 1 or count > span:
        raise ValueError(f"Cannot split {span} tokens into {count} spans.")
    bounds = [start + (index * span) // count for index in range(count + 1)]
    return list(zip(bounds[:-1], bounds[1:], strict=True))


def build_step_layout(steps: Sequence[Span], compression_ratio: float) -> StepLayout:
    """Allocate latents across already-segmented steps."""
    latents: list[Span] = []
    step_of_latent: list[int] = []
    last_latent_of_step: list[int] = []
    for index, (start, end) in enumerate(steps):
        count = latents_per_step(end - start, compression_ratio)
        for span in even_subspans(start, end, count):
            latents.append(span)
            step_of_latent.append(index)
        last_latent_of_step.append(len(latents) - 1)
    return StepLayout(
        steps=tuple(steps),
        latents=tuple(latents),
        step_of_latent=tuple(step_of_latent),
        last_latent_of_step=tuple(last_latent_of_step),
    )


def trivial_step_layout(spans: Sequence[Span]) -> StepLayout:
    """One step per span: what token patching means in step vocabulary.

    This is the degeneracy that keeps a single code path in the encoder. Under
    `uniform`, `cross_limit` then resolves to `prompt_len + span_end` and
    `cond_slot[m]` to `m - 1` -- exactly the pre-existing values.
    """
    count = len(spans)
    return StepLayout(
        steps=tuple(spans),
        latents=tuple(spans),
        step_of_latent=tuple(range(count)),
        last_latent_of_step=tuple(range(count)),
    )


def segment_paragraph_steps(
    text: str, offsets: Sequence[Span], compression_ratio: float
) -> StepLayout:
    """`step_token_spans` then `build_step_layout`: the whole paragraph strategy."""
    return build_step_layout(step_token_spans(text, offsets), compression_ratio)
