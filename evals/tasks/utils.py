"""Prompts and graders for the four benchmarks, matching the published protocols.

The YAMLs reach these through `!function utils.<name>`, which loads *this file by
path*, so everything a task needs has to live here. See evals/PROTOCOL.md for the
full audit and the sources.

Nothing upstream in lm-eval is reused: its MATH grader wants Minerva's
"Final Answer: ... I hope it is correct." format, its AIME grader compares
normalised strings (so `\\boxed{070}` != `70`), its GPQA task asks for no answer
format yet greps for one, and it has no HotpotQA task at all. Each grader below
is the one the benchmark's own reference implementation uses.
"""

import random
import re
import string
from collections import Counter

from math_verify import parse, verify

__all__ = [
    "aime_process_results",
    "gpqa_doc_to_text",
    "gpqa_process_docs",
    "gpqa_process_results",
    "hotpotqa_cot_doc_to_text",
    "hotpotqa_doc_to_text",
    "hotpotqa_process_results",
    "hotpotqa_sample_500",
    "math500_process_results",
]


def _samples(results: list) -> list[str]:
    """The `repeats` samples for one question.

    `generate_until` builds exactly one Instance per doc and appends every repeat
    to it, so `results` is a 1-element list whose entry is what the filter
    produced: a single string under `take_first`, or the list of `repeats`
    strings under `take_first_k`. Every task here declares `take_first_k`; the
    scalar case is handled so a missing filter_list degrades to k=1 rather than
    crashing.
    """
    first = results[0]
    return list(first) if isinstance(first, (list, tuple)) else [first]


# --------------------------------------------------------------------------
# MATH-500 and AIME'25 -- last \boxed{...}, symbolic equivalence, averaged
# --------------------------------------------------------------------------


def _boxed_equiv(gold: str, results: list) -> float:
    # Replies are already stripped of their <think> block by lm-eval's
    # think_end_token handling, so parse() only ever sees the visible answer.
    target = parse(gold)
    samples = _samples(results)
    return sum(float(bool(verify(target, parse(s)))) for s in samples) / len(samples)


def math500_process_results(doc: dict, results: list) -> dict[str, float]:
    # Gold is the full reference solution; its own \boxed{} is what parse picks up.
    return {"math_verify": _boxed_equiv(doc["solution"], results)}


def aime_process_results(doc: dict, results: list) -> dict[str, float]:
    return {"math_verify": _boxed_equiv(str(doc["answer"]), results)}


# --------------------------------------------------------------------------
# GPQA-Diamond -- port of openai/simple-evals gpqa_eval.py + common.py
# --------------------------------------------------------------------------

GPQA_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

_GPQA_ANSWER_RE = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?")


def gpqa_process_docs(dataset):
    """Permute the four choices, simple-evals style.

    simple-evals draws each permutation from one `random.Random(0)` stream over
    examples x n_repeats, so its choice order varies across repeats. lm-eval
    resamples a *fixed* prompt, so the order is instead pinned per question by row
    index: reproducible without a global RNG, at the cost of not marginalising
    position bias over the repeats.
    """

    def _process(doc, idx):
        correct = doc["Correct Answer"].strip()
        choices = [
            correct,
            doc["Incorrect Answer 1"].strip(),
            doc["Incorrect Answer 2"].strip(),
            doc["Incorrect Answer 3"].strip(),
        ]
        choices = [choices[i] for i in random.Random(idx).sample(range(4), 4)]
        return dict(
            zip("ABCD", choices, strict=True),
            answer="ABCD"[choices.index(correct)],
        )

    return dataset.map(_process, with_indices=True)


def gpqa_doc_to_text(doc: dict) -> str:
    return GPQA_TEMPLATE.format(
        Question=doc["Question"], A=doc["A"], B=doc["B"], C=doc["C"], D=doc["D"]
    )


def gpqa_process_results(doc: dict, results: list) -> dict[str, float]:
    samples = _samples(results)
    # simple-evals uses re.search (first match) and scores an unmatched reply 0.
    hits = sum(
        float(bool(m) and m.group(1).upper() == doc["answer"])
        for m in (_GPQA_ANSWER_RE.search(s) for s in samples)
    )
    return {"exact_match": hits / len(samples)}


# --------------------------------------------------------------------------
# HotpotQA (distractor) -- port of hotpot_evaluate_v1.py
# --------------------------------------------------------------------------

_HOTPOT_ANSWER_RE = re.compile(
    r"^\s*Answer\s*:\s*(.+?)\s*$", re.MULTILINE | re.IGNORECASE
)
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = frozenset(string.punctuation)
_CATEGORICAL = frozenset({"yes", "no", "noanswer"})


def _normalize(s: str) -> str:
    s = "".join(ch for ch in s.lower() if ch not in _PUNCTUATION)
    return " ".join(_ARTICLES_RE.sub(" ", s).split())


def _f1(pred: str, gold: str) -> float:
    # Official behaviour: a categorical answer gets no partial credit, in either
    # direction -- "yes" vs "no" must score 0, not token overlap.
    if (pred in _CATEGORICAL or gold in _CATEGORICAL) and pred != gold:
        return 0.0
    pred_tokens, gold_tokens = pred.split(), gold.split()
    num_same = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if num_same == 0:
        return 0.0
    precision, recall = num_same / len(pred_tokens), num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def hotpotqa_doc_to_text(doc: dict) -> str:
    paragraphs = "\n\n".join(
        f"{title}: {''.join(sentences)}"
        for title, sentences in zip(
            doc["context"]["title"], doc["context"]["sentences"], strict=True
        )
    )
    return (
        f"{paragraphs}\n\n"
        "Using only the passages above, answer the question. The answer is a short "
        "span copied from the passages, or yes/no. End your reply with a line of "
        "the form 'Answer: <answer>'.\n\n"
        f"Question: {doc['question']}"
    )


def hotpotqa_cot_doc_to_text(doc: dict) -> str:
    """The HotpotQA prompt plus an explicit zero-shot-CoT trigger.

    The other three benchmarks already instruct step-by-step reasoning in their
    own prompts (MATH-500/AIME "Please reason step by step, ...", GPQA's
    simple-evals template "Think step by step before answering."), so only this
    one needs the trigger added to match "standard CoT prompting". The format
    instruction stays ahead of it: hotpotqa_process_results reads the last
    "Answer:" line, so losing that line loses the answer.
    """
    return f"{hotpotqa_doc_to_text(doc)}\n\nLet's think step by step."


def hotpotqa_sample_500(dataset):
    """The 500-question HotpotQA subset the abstract-CoT paper evaluates on.

    Sampled rather than head-sliced: the distractor dev split carries a `level`
    field ("easy"/"medium"/"hard") and is not shuffled, so the first 500 rows are
    not a fair draw from the 7,405. Seeded, so the subset is identical across
    models and reruns, and sorted so the rows keep their dataset order.
    """
    n = min(500, len(dataset))
    return dataset.select(sorted(random.Random(1234).sample(range(len(dataset)), n)))


def hotpotqa_process_results(doc: dict, results: list) -> dict[str, float]:
    samples = _samples(results)
    gold = _normalize(doc["answer"])
    preds = []
    for s in samples:
        matches = _HOTPOT_ANSWER_RE.findall(s)
        preds.append(_normalize(matches[-1] if matches else s.strip().split("\n")[-1]))
    return {
        "exact_match": sum(float(p == gold) for p in preds) / len(preds),
        "f1": sum(_f1(p, gold) for p in preds) / len(preds),
    }
