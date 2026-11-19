"""Sample shared GSM8K numeric setups and align their solution steps."""

import json
import random
import re
from collections import Counter
from fractions import Fraction
from math import lcm, prod

from cot_compression.data.gsm8k.templates import evaluate, render

SEED = 20261119
TEXT_SETTINGS = range(1, 6)
ALL_TEXT = (1 << len(TEXT_SETTINGS)) - 1
SINGULAR_AGREEMENT = re.compile(
    r"(?<![\w./-])1(?:\.0+)? "
    r"(?:people|children|men|women|dollars|cents|kits|rounds|bags|points|items|"
    r"pieces|tickets|files|books|plates|objects|boxes)\b",
    re.IGNORECASE,
)


def grid_values(index, grid):
    """Decode one point in a Cartesian product without materializing it."""
    values = []
    for low, step, size in grid:
        index, offset = divmod(index, size)
        values.append(low + offset * step)
    return tuple(values)


def common_axis(slots):
    """Intersect bounded fractional grids using integer congruences."""
    grids = [tuple(Fraction(s[k]) for k in ("min", "max", "step")) for s in slots]
    scale = lcm(*(p.denominator for grid in grids for p in grid))
    bounds = [
        (int(low * scale), int(high * scale), int(step * scale))
        for low, high, step in grids
    ]
    residue, period = 0, 1
    for low, _, step in bounds:
        combined = lcm(period, step)
        gcd = period * step // combined
        delta = low - residue
        if delta % gcd:
            return None
        modulus = step // gcd
        residue += period * (delta // gcd * pow(period // gcd, -1, modulus) % modulus)
        period = combined
    lower = max(low for low, _, _ in bounds)
    upper = min(high for _, high, _ in bounds)
    start = lower + (residue - lower) % period
    if start > upper:
        return None
    return (
        Fraction(start, scale),
        Fraction(period, scale),
        1 + (upper - start) // period,
    )


def find_cycle(candidates):
    """A setup appears at its two adjacent text settings in the cycle."""
    pools = [
        [i for i, (_, _, mask) in enumerate(candidates) if mask & required == required]
        for required in [(16 | 1), (1 | 2), (2 | 4), (4 | 8), (8 | 16)]
    ]
    if any(not pool for pool in pools):
        return None

    def extend(order):
        if len(order) == 5:
            return order
        previous = candidates[order[-1]][1] if order else None
        first = candidates[order[0]][1] if len(order) == 4 else None
        for i in pools[len(order)]:
            answer = candidates[i][1]
            if i in order or answer in (previous, first):
                continue
            result = extend([*order, i])
            if result is not None:
                return result
        return None

    cycle = extend([])
    return [candidates[i][0] for i in cycle] if cycle is not None else None


def clean(rendered):
    return all(
        "(s)" not in text and "(es)" not in text and not SINGULAR_AGREEMENT.search(text)
        for text in rendered.values()
    )


def numeric_cycle(base, stories, texts, previous):
    """Find five shared legal setups that render cleanly at their assigned texts."""
    tpl = base["template"]
    slot_ids = [s["id"] for s in tpl["numbers"]]
    story_slots = [{s["id"]: s for s in row["template"]["numbers"]} for row in stories]
    axes = [
        common_axis(
            [
                slot,
                *(
                    lookup[row["numeric_map"][slot["id"]]]
                    for lookup, row in zip(story_slots, stories, strict=True)
                ),
            ]
        )
        for slot in tpl["numbers"]
    ]
    if None in axes or prod(axis[2] for axis in axes) < 5:
        return None
    size = prod(axis[2] for axis in axes)
    rng = random.Random(f"{SEED}/{base['base_id']}")
    anchors = sorted(
        {
            tuple(Fraction(row["numeric_inputs"][slot]) for slot in slot_ids)
            for row in previous
        }
    )
    rng.shuffle(anchors)
    visited = set()
    good = []
    partial = []
    mask_counts = Counter()

    def collect(points):
        for point in points:
            if point in visited or any(
                v < low or v > low + (count - 1) * step or (v - low) % step
                for v, (low, step, count) in zip(point, axes, strict=True)
            ):
                continue
            visited.add(point)
            inputs = dict(zip(slot_ids, point, strict=True))
            try:
                values = evaluate(tpl, inputs)
            except (ValueError, ZeroDivisionError):
                continue
            mask = ALL_TEXT
            for story, settings in zip(stories, texts, strict=True):
                story_values = evaluate(
                    story["template"],
                    {
                        alias: inputs[original]
                        for original, alias in story["numeric_map"].items()
                    },
                )
                for t, text in settings.items():
                    if mask & (1 << (t - 1)) and not clean(
                        render(story["template"], story_values, text)
                    ):
                        mask &= ~(1 << (t - 1))
                if not mask:
                    break
            case = (inputs, values[tpl["answer"]], mask)
            key = case[1], mask
            if mask_counts[key] == 2:
                continue
            mask_counts[key] += 1
            if mask == ALL_TEXT:
                good.append(case)
                if len(good) >= 5 and len({answer for _, answer, _ in good}) >= 3:
                    cycle = find_cycle(good)
                    if cycle:
                        return cycle
            elif mask:
                partial.append(case)
        if (
            len(good) + len(partial) >= 5
            and len({answer for _, answer, _ in good + partial}) >= 3
        ):
            # A small pool keeps the fallback search bounded for large grids.
            return find_cycle(good[:25] + partial[:50])
        return None

    cycle = collect(anchors)
    if cycle:
        return cycle
    indices = range(size) if size <= 20_000 else rng.sample(range(size), 8_192)
    return collect(grid_values(i, axes) for i in indices)


def step_signatures(base):
    """Describe calculation dependencies independently of story and input names."""
    steps = {step["id"]: step for step in base["template"]["steps"]}
    slots = {slot["id"] for slot in base["template"]["numbers"]}

    def signature(step_id):
        input_roles = {}

        def visit(arg):
            if arg in slots:
                return ["input", input_roles.setdefault(arg, len(input_roles))]
            if arg not in steps:
                return ["literal", str(Fraction(arg))]
            step = steps[arg]
            return [step["op"], *map(visit, step["args"])]

        return json.dumps(visit(step_id), separators=(",", ":"))

    return {step_id: signature(step_id) for step_id in base["solution_step_ids"]}


def render_family(base, stories, texts, cycle):
    base_id = base["base_id"]
    steps = base["template"]["steps"]
    signatures = step_signatures(base)
    for story, settings in zip(stories, texts, strict=True):
        for t in TEXT_SETTINGS:
            text = settings[t]
            pair = (t - 1, t % 5)
            for n in pair:
                inputs = cycle[n]
                values = evaluate(base["template"], inputs)
                mapped = {
                    alias: inputs[key] for key, alias in story["numeric_map"].items()
                }
                story_values = evaluate(story["template"], mapped)
                assert all(
                    story_values[step["id"]] == values[step["id"]] for step in steps
                )
                rendered = render(story["template"], story_values, text)
                assert clean(rendered)
                step_values = [
                    {
                        "step_id": step["id"],
                        "operands": [
                            str(values[arg] if arg in values else Fraction(arg))
                            for arg in step["args"]
                        ],
                        "result": str(values[step["id"]]),
                        "structural_signature": signatures[step["id"]],
                    }
                    for step in steps
                    if step["id"] in signatures
                ]
                assert len(rendered["answer"].splitlines()) == len(step_values) + 1
                yield {
                    "source_id": base_id,
                    "base_id": base_id,
                    "variant_id": story["variant_id"],
                    "template_id": story["template_id"],
                    "view_id": f"{story['template_id']}/t{t}/n{n + 1}",
                    "numeric_group_id": f"{base_id}/n{n + 1}",
                    "numeric_setting": n + 1,
                    "text_setting": t,
                    "numeric_inputs": {
                        key: str(value) for key, value in inputs.items()
                    },
                    "text_inputs": text,
                    "solution_steps": step_values,
                    **rendered,
                }
