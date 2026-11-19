import json
import re
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import pytest

from cot_compression.data.gsm8k.templates import (
    evaluate,
    format_quantities,
    validate_template,
)

PROMPT = Path(__file__).resolve().parents[2] / "prompts/extract_gsm8k_template.txt"
EXAMPLES = [
    json.loads(block)
    for block in re.findall(
        r"```json\n(.*?)\n```", PROMPT.read_text(encoding="utf-8"), re.S
    )
]


@pytest.mark.parametrize(
    "source, response", list(zip(EXAMPLES[::2], EXAMPLES[1::2], strict=True))
)
def test_few_shot_templates_replay_and_sample(source, response):
    result = validate_template(response["template"], source["answer"])

    assert result["valid_draws"] >= 10
    assert result["distinct_answers"] >= 2
    assert all(sample["answer"].count("#### ") == 1 for sample in result["samples"])


def test_template_rejects_a_computed_value_in_the_question():
    seed, response = EXAMPLES[:2]
    template = response["template"].copy()
    template["question"] += " In May, they sold {may_sales}."

    with pytest.raises(ValueError, match="Question placeholders"):
        validate_template(template, seed["answer"])


def test_guard_only_calculation_can_follow_the_answer():
    seed, response = EXAMPLES[:2]
    template = deepcopy(response["template"])
    template["steps"].append(
        {
            "id": "guard_value",
            "op": "divide",
            "args": ["april_sales", "4"],
            "explanation": "{guard_value} used by a sampling constraint.",
        }
    )
    template["constraints"].append({"op": "is_integer", "args": ["guard_value"]})

    result = validate_template(template, seed["answer"])

    for sample in result["samples"]:
        assert "sampling constraint" not in sample["answer"]
        assert Fraction(sample["answer"].rsplit("#### ", 1)[1]) % 6 == 0


def test_explanation_can_show_a_constraint_calculation_it_uses():
    seed, response = EXAMPLES[:2]
    template = deepcopy(response["template"])
    template["steps"].insert(
        0,
        {
            "id": "groups",
            "op": "divide",
            "args": ["april_sales", "4"],
            "explanation": "April sales form {groups} groups of four.",
        },
    )
    final_step = template["steps"][-1]
    final_step["explanation"] += " The {groups} groups mean the batch size fits."
    template["constraints"].append({"op": "is_integer", "args": ["groups"]})

    result = validate_template(template, seed["answer"])

    assert all(
        sample["answer"].startswith("April sales form ") for sample in result["samples"]
    )


def test_currency_constraint_uses_exact_multiples():
    template = {
        "steps": [],
        "constraints": [{"op": "multiple_of", "args": ["cost", "0.01"]}],
    }

    assert evaluate(template, {"cost": Fraction("3.42")})["cost"] == Fraction("3.42")
    with pytest.raises(ValueError, match="template constraint"):
        evaluate(template, {"cost": Fraction("3.421")})


def test_quantity_inflection_preserves_multiplication_and_other_counts():
    assert format_quantities("1 hours, 21 hours, 1 time(s), 3 time(s), 1 times 5") == (
        "1 hour, 21 hours, 1 time, 3 times, 1 times 5"
    )


def test_percent_placeholder_preserves_exact_arithmetic_value():
    _, response = EXAMPLES[:2]
    template = deepcopy(response["template"])
    template["numbers"].append(
        {
            "id": "discount",
            "source": "0.125",
            "min": "0.125",
            "max": "0.25",
            "step": "0.125",
        }
    )
    template["question"] += " The discount is {discount:percent}."

    result = validate_template(template, "#### 72")

    for sample in result["samples"]:
        assert sample["question"].endswith(("12.5%.", "25%."))
