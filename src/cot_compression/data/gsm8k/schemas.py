"""Structured outputs for the prompts used in the GSM8K workflow."""

from typing import Any

STRING = {"type": "string"}
OPTIONAL_STRING = {"type": ["string", "null"]}
BAD_TEMPLATE = (ValueError, ArithmeticError, LookupError, TypeError)


def obj(**properties: Any) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def array(items: dict) -> dict:
    return {"type": "array", "items": items}


def enum(*values: str) -> dict:
    return {"type": "string", "enum": list(values)}


CHOICE = obj(id=STRING, values=array(STRING))
EXPLANATION = obj(id=STRING, explanation=STRING)
TEMPLATE = obj(
    numbers=array(obj(id=STRING, source=STRING, min=STRING, max=STRING, step=STRING)),
    choices=array(CHOICE),
    steps=array(
        obj(
            id=STRING,
            op=enum("add", "subtract", "multiply", "divide"),
            args=array(STRING),
            explanation=STRING,
        )
    ),
    constraints=array(
        obj(
            op=enum("is_integer", "multiple_of", "gt", "gte", "lt", "lte", "ne"),
            args=array(STRING),
        )
    ),
    question=STRING,
    answer=STRING,
)
EXTRACT = obj(
    source_id=STRING,
    status=enum("ok", "unsupported"),
    reason=OPTIONAL_STRING,
    template={"anyOf": [TEMPLATE, {"type": "null"}]},
)
SOLVE = obj(answer=OPTIONAL_STRING, issue=OPTIONAL_STRING, calculation=STRING)
EXPAND = obj(
    source_id=STRING,
    question=OPTIONAL_STRING,
    choices=array(CHOICE),
    explanation_updates=array(EXPLANATION),
)
TRIAGE = obj(
    decisions=array(
        obj(
            source_id=STRING,
            decision=enum("keep", "wording_edit", "review_ranges_or_logic", "both"),
            issues=array(
                obj(
                    kind=enum("grammar", "ambiguity", "numeric_plausibility", "logic"),
                    origin=enum("inherited", "introduced", "uncertain"),
                    evidence=STRING,
                    note=STRING,
                )
            ),
        )
    )
)
WORDING = obj(
    source_id=STRING,
    decision=enum("edit", "keep", "needs_math_review"),
    reason=STRING,
    question=OPTIONAL_STRING,
    choice_updates=array(CHOICE),
    remove_choice_ids=array(STRING),
    explanation_updates=array(EXPLANATION),
)
EDIT_AUDIT = obj(
    decisions=array(
        obj(
            source_id=STRING,
            decision=enum("accept", "revise", "needs_math_review"),
            reason=STRING,
        )
    )
)
STAGES = {
    "generate": ("extract_gsm8k_template", EXTRACT, 1),
    "repair": ("repair_gsm8k_template", EXTRACT, 1),
    "assess": ("audit_gsm8k_rendered", obj(ordinary=SOLVE, boundary=SOLVE), 1),
    "expand-text": ("expand_gsm8k_text_choices", EXPAND, 1),
    "classify": ("classify_gsm8k_template_quality", TRIAGE, 5),
    "repair-wording": ("repair_gsm8k_wording", WORDING, 1),
    "audit-wording": ("audit_gsm8k_wording", EDIT_AUDIT, 4),
}
COMMANDS = tuple(action for action in STAGES if action != "audit-wording")
