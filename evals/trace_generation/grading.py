from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace

from math_verify import parse, verify

_ANSWER_LINE = re.compile(r"\s*Answer\s*:\s*(.+?)\s*", re.IGNORECASE)
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


@dataclass(frozen=True)
class GradeResult:
    thinking: str | None
    final_response: str
    extracted_answer: str | None
    extraction_status: str
    score: float | None
    correct: bool | None
    qa_token_f1: float | None


def split_thinking(completion: str) -> tuple[str | None, str, str]:
    """Split a complete ``<think>...</think>`` response without inventing tags."""
    end = completion.rfind("</think>")
    if end < 0:
        return None, "", "missing_think_end"
    start = completion.rfind("<think>", 0, end)
    if start < 0:
        return None, "", "missing_think_start"
    thinking = completion[start + len("<think>") : end].strip()
    final = completion[end + len("</think>") :].strip()
    return thinking, final, "ok"


def extract_last_boxed(text: str) -> str | None:
    """Return the content of the last complete, balanced ``\\boxed{...}``."""
    results: list[str] = []
    cursor = 0
    marker = "\\boxed"
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            break
        brace = start + len(marker)
        while brace < len(text) and text[brace].isspace():
            brace += 1
        if brace >= len(text) or text[brace] != "{":
            cursor = start + len(marker)
            continue
        depth = 1
        pos = brace + 1
        while pos < len(text) and depth:
            escaped = pos > 0 and text[pos - 1] == "\\"
            if text[pos] == "{" and not escaped:
                depth += 1
            elif text[pos] == "}" and not escaped:
                depth -= 1
            pos += 1
        if depth == 0:
            results.append(text[brace + 1 : pos - 1].strip())
            cursor = pos
        else:
            cursor = start + len(marker)
    return results[-1] if results else None


def _math_parse(text: str):
    """Parse a standalone answer without losing its LaTeX structure."""
    expression = text.strip()
    boxed = extract_last_boxed(expression)
    if boxed is not None:
        expression = boxed
    if (
        (expression.startswith("$") and expression.endswith("$"))
        or (expression.startswith(r"\(") and expression.endswith(r"\)"))
        or (expression.startswith(r"\[") and expression.endswith(r"\]"))
    ):
        rendered = expression
    else:
        rendered = f"${expression}$"
    return parse(rendered)


def math_answer_is_verifiable(answer: str) -> bool:
    try:
        parsed = _math_parse(answer)
        return bool(parsed) and bool(verify(parsed, parsed))
    except Exception:
        return False


def math_answers_equivalent(prediction: str, ground_truth: str) -> bool:
    try:
        parsed_prediction = _math_parse(prediction)
        parsed_gold = _math_parse(ground_truth)
        if not parsed_prediction or not parsed_gold:
            return False
        prediction_symbols = {
            str(symbol) for symbol in getattr(parsed_prediction[0], "free_symbols", ())
        }
        gold_symbols = {
            str(symbol) for symbol in getattr(parsed_gold[0], "free_symbols", ())
        }
        return prediction_symbols == gold_symbols and bool(
            verify(parsed_gold, parsed_prediction)
        )
    except Exception:
        return False


def extract_qa_answer(text: str) -> str | None:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    match = _ANSWER_LINE.fullmatch(lines[-1])
    return match.group(1).strip() if match else None


def normalize_qa_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    return " ".join(_ARTICLES.sub(" ", text).split())


def qa_token_f1(prediction: str, ground_truth: str) -> float:
    pred = normalize_qa_answer(prediction).split()
    gold = normalize_qa_answer(ground_truth).split()
    if not pred or not gold:
        return float(pred == gold)
    overlap = sum((Counter(pred) & Counter(gold)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def grade_final_response(
    domain: str, final_response: str, ground_truth: str
) -> GradeResult:
    """Grade visible response text; useful for auditing upstream plain traces."""
    extracted: str | None = None
    f1: float | None = None
    correct = False
    status = "ok"
    if domain == "math":
        extracted = extract_last_boxed(final_response)
        if extracted is None:
            status = "missing_boxed_answer"
        else:
            correct = math_answers_equivalent(extracted, ground_truth)
    elif domain == "qa":
        extracted = extract_qa_answer(final_response)
        if extracted is None:
            status = "missing_answer_line"
        else:
            f1 = qa_token_f1(extracted, ground_truth)
            correct = normalize_qa_answer(extracted) == normalize_qa_answer(
                ground_truth
            )
    else:
        raise ValueError(f"Unknown domain: {domain}")
    if extracted is not None and not correct:
        status = "incorrect_answer"

    return GradeResult(
        thinking=None,
        final_response=final_response,
        extracted_answer=extracted,
        extraction_status=status,
        score=float(correct),
        correct=correct,
        qa_token_f1=f1,
    )


def grade_completion(
    domain: str,
    completion: str,
    ground_truth: str | None,
    *,
    generation_error: str | None = None,
    truncated: bool = False,
) -> GradeResult:
    thinking, final, status = split_thinking(completion)
    status = "generation_error" if generation_error is not None else status
    status = "truncated" if truncated else status
    if status != "ok":
        if domain in {"math", "qa"} and ground_truth is not None:
            return GradeResult(thinking, final, None, status, 0.0, False, None)
        return GradeResult(thinking, final, None, status, None, None, None)
    if domain not in {"math", "qa"} or ground_truth is None:
        return GradeResult(thinking, final, None, "ungraded", None, None, None)
    return replace(grade_final_response(domain, final, ground_truth), thinking=thinking)
