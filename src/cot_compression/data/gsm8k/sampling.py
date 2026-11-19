"""Render legal numeric probes and check the text-diversity target."""

import random
from fractions import Fraction
from itertools import product

from cot_compression.data.gsm8k.io import Record
from cot_compression.data.gsm8k.templates import evaluate, number_text, render


def numeric_cases(t: Record) -> list[dict[str, Fraction]]:
    """Evaluate the original numbers and a legal setting with a different answer."""
    slots = t["numbers"]
    grids = [
        tuple(Fraction(n[k]) for k in ("source", "min", "max", "step")) for n in slots
    ]
    source = {n["id"]: g[0] for n, g in zip(slots, grids, strict=True)}
    ordinary = evaluate(t, source)
    # Try one-slot changes first so the probes expose boundary behavior. Random
    # draws cover cases where constraints require several slots to change together.
    edges = [
        {**source, slot["id"]: end}
        for slot, (original, low, high, step) in zip(slots, grids, strict=True)
        for end in (low, low + (high - low) // step * step)
        if end != original
    ]
    rng = random.Random(42)
    edges.extend(
        {
            n["id"]: low + step * rng.randrange(1 + (high - low) // step)
            for n, (_, low, high, step) in zip(slots, grids, strict=True)
        }
        for _ in range(512)
    )
    for givens in edges:
        try:
            boundary = evaluate(t, givens)
        except (ValueError, ZeroDivisionError):
            continue
        if ordinary[t["answer"]] != boundary[t["answer"]]:
            return [ordinary, boundary]
    raise ValueError(
        "Could not find two legal numeric assignments with different answers"
    )


def examples(t: Record) -> list[dict[str, str]]:
    """Render two numeric settings and a one-unit probe when available."""
    cases = numeric_cases(t)
    probes = [
        render(t, case, {c["id"]: c["values"][choice] for c in t["choices"]})
        for case, choice in zip(cases, (0, -1), strict=True)
    ]
    # A draw with one item can expose agreement issues that the first two miss.
    source = {n["id"]: Fraction(n["source"]) for n in t["numbers"]}
    for slot in t["numbers"]:
        sid = slot["id"]
        low, high, step = (Fraction(slot[name]) for name in ("min", "max", "step"))
        edges = (low, high) if source[sid] == 1 else (Fraction(1),)
        for edge in edges:
            if edge == source[sid] or not low <= edge <= high or (edge - low) % step:
                continue
            try:
                values = evaluate(t, {**source, sid: edge})
            except (ValueError, ZeroDivisionError):
                continue
            probe = render(t, values, {c["id"]: c["values"][0] for c in t["choices"]})
            if probe["question"] not in {r["question"] for r in probes}:
                return [*probes, probe]
    return probes


def expected_answer(row: Record) -> Fraction:
    return Fraction(row["answer"].rsplit("####", 1)[1].strip().replace(",", ""))


def legal_text_target(t: Record) -> None:
    """Require five text assignments that give ten distinct rendered questions."""
    choices = {c["id"]: c["values"] for c in t["choices"]}
    cases = numeric_cases(t)
    seen = set()
    # Each selected text assignment must work at both numeric settings.
    for assignment in product(*choices.values()):
        text = dict(zip(choices, assignment, strict=True))
        questions = {render(t, values, text)["question"] for values in cases}
        if len(questions) == 2 and not (questions & seen):
            seen.update(questions)
            if len(seen) == 10:
                return
    raise ValueError(
        "Could not render five distinct text settings at two numeric assignments"
    )


def legal_examples(t: Record) -> list[dict[str, str]]:
    return [
        {"question": r["question"], "intended_answer": number_text(expected_answer(r))}
        for r in examples(t)
    ]
