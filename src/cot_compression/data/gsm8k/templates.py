"""Evaluate extracted GSM8K templates with exact arithmetic and render samples."""

import operator
import random
import re
from collections import Counter
from collections.abc import Callable
from fractions import Fraction
from string import Formatter
from typing import Any

BAD_TEMPLATE = (ValueError, ArithmeticError, LookupError, TypeError)

OPERATIONS: dict[str, Callable[[Fraction, Fraction], Fraction]] = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
}
COMPARISONS = {
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
    "ne": operator.ne,
}
_SINGULAR_UNITS = {
    "hours": "hour",
    "minutes": "minute",
    "seconds": "second",
    "days": "day",
    "weeks": "week",
    "months": "month",
    "years": "year",
    "times": "time",
    "gallons": "gallon",
    "liters": "liter",
    "milliliters": "milliliter",
    "pounds": "pound",
    "kilograms": "kilogram",
    "grams": "gram",
    "miles": "mile",
    "feet": "foot",
    "inches": "inch",
    "ounces": "ounce",
    "pink hard hats": "pink hard hat",
    "cat collars": "cat collar",
    "flowering plants": "flowering plant",
    "fruiting plants": "fruiting plant",
}
_SINGULAR = re.compile(
    r"(?<![\w,./-])(1(?:\.0+)?) ("
    + "|".join(r"times(?! \d)" if unit == "times" else unit for unit in _SINGULAR_UNITS)
    + r")\b"
)
_PARENTHETICAL_PLURAL = re.compile(r"(?<![\w,./-])(\d+(?:\.\d+)?) ([a-z]+)\(s\)")


def fields(text: str) -> set[str]:
    names = set()
    for _, field, spec, conversion in Formatter().parse(text):
        if field is not None:
            if (
                not re.fullmatch(r"[a-z][a-z0-9_]*", field)
                or spec not in {"", "percent"}
                or conversion
            ):
                raise ValueError(f"Invalid text placeholder {field!r}")
            names.add(field)
    return names


def number_text(value: Fraction) -> str:
    denominator = value.denominator
    for factor in (2, 5):
        while denominator % factor == 0:
            denominator //= factor
    if denominator != 1:
        return str(value)
    whole, remainder = divmod(abs(value.numerator), value.denominator)
    sign = "-" if value < 0 else ""
    digits = []
    while remainder:
        digit, remainder = divmod(remainder * 10, value.denominator)
        digits.append(str(digit))
    decimal = "".join(digits)
    return f"{sign}{whole}.{decimal}" if digits else f"{sign}{whole}"


def evaluate(
    template: dict[str, Any], inputs: dict[str, Fraction]
) -> dict[str, Fraction]:
    """Evaluate an assignment exactly and reject it if a constraint fails.

    Args:
        template: An ordered calculation graph with named inputs and constraints.
        inputs: Values for the template's numeric slots.

    Returns:
        The input values and every computed result, keyed by slot or step ID.

    Raises:
        ValueError: If the assignment violates a constraint.
    """
    values = inputs.copy()

    def operand(text: str) -> Fraction:
        return values[text] if text in values else Fraction(text)

    for step in template["steps"]:
        left, right = map(operand, step["args"])
        values[step["id"]] = OPERATIONS[step["op"]](left, right)
    for condition in template["constraints"]:
        args = list(map(operand, condition["args"]))
        if condition["op"] == "is_integer":
            (value,) = args
            valid = value.denominator == 1
        else:
            left, right = args
            if condition["op"] == "multiple_of":
                valid = right > 0 and left % right == 0
            else:
                valid = COMPARISONS[condition["op"]](left, right)
        if not valid:
            raise ValueError("Sample violates a template constraint")
    return values


def step_dependencies(steps: list[dict[str, Any]], roots: set[str]) -> set[str]:
    needed = roots.copy()
    for step in reversed(steps):
        if step["id"] in needed:
            needed.update(step["args"])
    return needed


def solution_steps(
    steps: list[dict[str, Any]], answer: str, guards: set[str]
) -> set[str]:
    """Find answer dependencies and guard values cited in explanations."""
    needed = {answer}
    for step in reversed(steps):
        if step["id"] in needed:
            needed.update(step["args"])
            needed.update(fields(step["explanation"]) & guards)
    return needed


def format_quantities(text: str) -> str:
    text = _PARENTHETICAL_PLURAL.sub(
        lambda match: f"{match[1]} {match[2]}{'s' if Fraction(match[1]) != 1 else ''}",
        text,
    )
    return _SINGULAR.sub(lambda match: f"{match[1]} {_SINGULAR_UNITS[match[2]]}", text)


def render(
    template: dict[str, Any], values: dict[str, Fraction], choices: dict[str, str]
) -> dict[str, str]:
    """Render a question and worked solution for one evaluated assignment."""
    bindings = {
        **{name: number_text(value) for name, value in values.items()},
        **choices,
    }

    def render_text(text: str) -> str:
        parts = []
        for literal, field, spec, _ in Formatter().parse(text):
            parts.append(literal)
            if field is not None:
                parts.append(
                    f"{number_text(values[field] * 100)}%"
                    if spec == "percent"
                    else bindings[field]
                )
        return format_quantities("".join(parts))

    steps = template["steps"]
    guards = step_dependencies(
        steps, {arg for rule in template["constraints"] for arg in rule["args"]}
    )
    needed = solution_steps(steps, template["answer"], guards)
    explanation = [
        render_text(step["explanation"]) for step in steps if step["id"] in needed
    ]
    answer = bindings[template["answer"]]
    return {
        "question": render_text(template["question"]),
        "answer": "\n".join([*explanation, f"#### {answer}"]),
    }


def validate_template(template: dict[str, Any], seed_answer: str) -> dict[str, Any]:
    """Check template structure, replay the source answer, and probe its grid."""
    numbers = template["numbers"]
    choices = template["choices"]
    steps = template["steps"]
    identifiers = [item["id"] for item in [*numbers, *choices, *steps]]
    if len(set(identifiers)) != len(identifiers) or any(
        not re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in identifiers
    ):
        raise ValueError("IDs must be unique lowercase identifiers")
    if template["answer"] not in {step["id"] for step in steps}:
        raise ValueError("Answer must reference a calculation")
    if any(
        not item["values"] or not all(value.strip() for value in item["values"])
        for item in choices
    ):
        raise ValueError("Text choices must contain nonempty strings")
    given = {item["id"] for item in [*numbers, *choices]}
    if fields(template["question"]) != given:
        raise ValueError(
            "Question placeholders must match the given numbers and text choices"
        )

    available = given.copy()
    guards = step_dependencies(
        steps, {arg for rule in template["constraints"] for arg in rule["args"]}
    )
    solution = solution_steps(steps, template["answer"], guards)
    for step in steps:
        if step["id"] not in solution | guards:
            raise ValueError(f"Unused calculation {step['id']}")
        if step["id"] not in solution:
            continue
        available.add(step["id"])
        explanation = step["explanation"]
        if (
            not explanation.strip()
            or "####" in explanation
            or not fields(explanation) <= available
        ):
            raise ValueError(f"Invalid explanation for {step['id']}")

    grids = {}
    source = {}
    for slot in numbers:
        original, low, high, step = (
            Fraction(slot[key]) for key in ("source", "min", "max", "step")
        )
        if not (step > 0 and low <= original <= high and (original - low) % step == 0):
            raise ValueError(
                f"Source value is outside range or off-grid for {slot['id']}"
            )
        grids[slot["id"]] = (low, step, (high - low) // step + 1)
        source[slot["id"]] = original

    source_values = evaluate(template, source)
    expected = Fraction(seed_answer.rsplit("####", 1)[1].strip().replace(",", ""))
    answer_id = template["answer"]
    if source_values[answer_id] != expected:
        raise ValueError(
            f"Source replay got {source_values[answer_id]}, expected {expected}"
        )

    # Replaying the source is necessary. Sampling also checks that the template
    # generates varied answers within the proposed numeric ranges.
    randomizer = random.Random(42)
    answers: Counter[str] = Counter()
    samples = []
    valid = 0
    for _ in range(128):
        inputs = {
            name: low + randomizer.randrange(size) * step
            for name, (low, step, size) in grids.items()
        }
        try:
            values = evaluate(template, inputs)
        except (ValueError, ZeroDivisionError):
            continue
        valid += 1
        key = number_text(values[answer_id])
        answers[key] += 1
        if answers[key] == 1 and len(samples) < 3:
            text = {item["id"]: randomizer.choice(item["values"]) for item in choices}
            samples.append(render(template, values, text))
    if valid < 10 or len(answers) < 2:
        raise ValueError(
            f"Only {valid}/128 valid draws and {len(answers)} distinct answers"
        )

    # This combines branches that differ only in input labels. It is a coarse
    # diversity metric because it also groups some different input dependencies.
    shapes = dict.fromkeys(source, "input")
    for step in steps:
        left, right = (
            shapes[arg] if arg in shapes else str(Fraction(arg)) for arg in step["args"]
        )
        if step["op"] in {"add", "multiply"}:
            left, right = sorted((left, right))
        shapes[step["id"]] = f"{step['op']}({left},{right})"
    return {
        "valid_draws": valid,
        "distinct_answers": len(answers),
        "shape": shapes[answer_id],
        "samples": samples,
    }
