"""Expand text choices, classify wording issues, and apply reviewed edits."""

import argparse
from string import Formatter

from cot_compression.data.gsm8k.api import run_requests
from cot_compression.data.gsm8k.io import (
    Record,
    RecordsById,
    index_by_source_id,
    read_jsonl,
    write_jsonl,
)
from cot_compression.data.gsm8k.sampling import (
    examples,
    expected_answer,
    legal_examples,
    legal_text_target,
)
from cot_compression.data.gsm8k.templates import (
    BAD_TEMPLATE,
    fields,
    number_text,
    validate_template,
)


def patch_template(row: Record, patch: Record, gold: str, *, expand: bool) -> Record:
    """Apply a text patch while keeping the calculation and the diversity target.

    Args:
        row: The original template record.
        patch: Changes to the question, choice pools, or explanation text.
        gold: Original GSM8K answer with its final answer marker.
        expand: Whether the edit extends the available text choices.

    Returns:
        The validated record with its revised template.

    Raises:
        ValueError: If the patch changes protected fields or fails validation.
    """
    base = row["template"]
    original = {c["id"]: c["values"] for c in base["choices"]}
    key = "choices" if expand else "choice_updates"
    changed = {c["id"]: c["values"] for c in patch[key]}
    removals = set(patch.get("remove_choice_ids", []))
    if len(changed) != len(patch[key]) or removals & changed.keys():
        raise ValueError("Duplicate or conflicting choice slots")
    if expand:
        if any(changed.get(k, [])[: len(v)] != v for k, v in original.items()):
            raise ValueError("Expansion must keep existing choices in order")
        if changed.keys() == original.keys() and (
            patch["question"] is not None or patch["explanation_updates"]
        ):
            raise ValueError("Pool-only expansion must keep the wording")
    elif not (changed.keys() | removals) <= original.keys() or any(
        len(values) < len(original[key]) for key, values in changed.items()
    ):
        raise ValueError("Wording edits must preserve retained choice pools")
    pools = original | changed
    for values in pools.values():
        if len({s.strip().casefold() for s in values}) != len(values) or any(
            not s.strip() or any(ch in s for ch in "{}\r\n") for s in values
        ):
            raise ValueError("Choice options must be distinct plain text")
    explanations = {s["id"]: s["explanation"] for s in patch["explanation_updates"]}
    if len(explanations) != len(
        patch["explanation_updates"]
    ) or not explanations.keys() <= {s["id"] for s in base["steps"]}:
        raise ValueError("Explanation updates must use existing step IDs")
    question = base["question"] if patch["question"] is None else patch["question"]
    number_ids = {n["id"] for n in base["numbers"]}
    base_units = {
        (name, spec)
        for _, name, spec, _ in Formatter().parse(base["question"])
        if name in number_ids
    }
    new_units = {
        (name, spec)
        for _, name, spec, _ in Formatter().parse(question)
        if name in number_ids
    }
    if new_units != base_units or "####" in question:
        raise ValueError("Keep every numeric placeholder and its format")
    # Copy arithmetic fields from the base. The patch can supply prose only.
    t = {
        **base,
        "question": question,
        "choices": [
            {"id": k, "values": v} for k, v in pools.items() if k not in removals
        ],
        "steps": [
            {**s, "explanation": explanations.get(s["id"], s["explanation"])}
            for s in base["steps"]
        ],
    }
    if t == base or any(removals & fields(s["explanation"]) for s in t["steps"]):
        raise ValueError("Patch makes no change or leaves references to removed slots")
    try:
        validate_template(t, gold)
    except ValueError as error:
        # This validator also samples numbers. A sparse random draw can fail
        # when the graph is valid, so check the required render matrix below.
        if not str(error).startswith("Only "):
            raise
    legal_text_target(t)
    renders = examples(t)
    if any(mark in r["question"] for mark in ("(s)", "(es)") for r in renders):
        raise ValueError("A plural marker survives rendering")
    return {**row, "template": t}


def classify(rows: RecordsById, source: RecordsById, args: argparse.Namespace) -> None:
    """Classify current wording using the source story and legal renders."""
    prior = (
        index_by_source_id(read_jsonl(args.prior_templates))
        if args.prior_templates
        else rows
    )
    payload = {}
    for sid, row in rows.items():
        t, before = row["template"], prior[sid]["template"]
        old_choices = {c["id"]: c["values"] for c in before["choices"]}
        payload[sid] = {
            "source_id": sid,
            "question": t["question"],
            "prior_template_question": before["question"]
            if before["question"] != t["question"]
            else None,
            "source_question": source[sid]["question"],
            "choices": [
                {**c, "prior_values": old_choices.get(c["id"], [])}
                for c in t["choices"]
            ],
            "numeric_ranges": [
                [n[k] for k in ("id", "min", "max", "step")] for n in t["numbers"]
            ],
            "legal_examples": legal_examples(t),
        }
    results = run_requests("classify", payload, args)
    if not args.dry_run:
        write_jsonl(
            args.output / "outcomes.jsonl",
            (
                results.get(
                    sid, {"source_id": sid, "decision": "missing", "issues": []}
                )
                for sid in rows
            ),
        )


def wording_inputs(row: Record, source: Record, finding: Record) -> Record:
    return {
        "source_id": row["source_id"],
        "template": row["template"],
        "source_question": source["question"],
        "source_gold_answer": number_text(expected_answer(source)),
        "legal_examples": legal_examples(row["template"]),
        "review_scope": finding.get("review_scope", finding["decision"]),
        "classifier_findings": finding["issues"],
        "prior_manual_findings": finding.get("manual_findings", []),
    }


def edit_audit(seed: Record, edited: Record, patch: Record) -> Record:
    base, revised = seed["template"], edited["template"]
    return {
        "source_id": seed["source_id"],
        "source_question": seed["source_question"],
        "base_question": base["question"],
        "base_choices": base["choices"],
        "base_legal_examples": seed["legal_examples"],
        "revised_question": revised["question"],
        "revised_choices": revised["choices"],
        "revised_legal_examples": legal_examples(revised),
        "proposal_decision": patch["decision"],
        "proposal_reason": patch["reason"],
        "explanation_updates": patch["explanation_updates"],
        **{
            k: seed[k]
            for k in ("review_scope", "classifier_findings", "prior_manual_findings")
        },
        **{k: base[k] for k in ("steps", "constraints", "answer")},
    }


def patch_stage(
    rows: RecordsById, source: RecordsById, args: argparse.Namespace
) -> None:
    """Apply passing proposals and keep review findings for the other templates."""
    expand = args.stage == "expand-text"
    jobs, outcomes = {}, {}
    findings = index_by_source_id(read_jsonl(args.findings)) if args.findings else {}
    feedback = index_by_source_id(read_jsonl(args.feedback)) if args.feedback else {}
    for sid, row in rows.items():
        if args.feedback and feedback.get(sid, {}).get("result") != "revise":
            continue
        if expand:
            try:
                legal_text_target(row["template"])
            except ValueError:
                jobs[sid] = {"source_id": sid, "template": row["template"]}
        elif findings.get(sid, {}).get(
            "review_scope", findings.get(sid, {}).get("decision")
        ) in {"wording_edit", "both"}:
            jobs[sid] = wording_inputs(row, source[sid], findings[sid])
        if sid in jobs and sid in feedback:
            jobs[sid]["retry_feedback"] = feedback[sid]
    proposals = run_requests(args.stage, jobs, args)
    if args.dry_run:
        return
    candidates, audits = {}, {}
    for sid in jobs:
        patch = proposals.get(sid)
        try:
            if patch is None:
                raise ValueError("Missing structured proposal")
            revised = rows[sid]
            if expand or patch["decision"] == "edit":
                revised = patch_template(
                    revised, patch, source[sid]["answer"], expand=expand
                )
            elif patch["question"] is not None or any(
                patch[k]
                for k in ("choice_updates", "remove_choice_ids", "explanation_updates")
            ):
                raise ValueError("Only edit decisions may change a template")
            if not expand:
                audits[sid] = edit_audit(jobs[sid], revised, patch)
            candidates[sid] = revised
        except BAD_TEMPLATE as error:
            outcomes[sid] = {
                "source_id": sid,
                "result": "revise",
                "reason": str(error),
                "patch": patch,
            }
    # Keep a structurally valid wording patch only after an independent review.
    judgments = run_requests("audit-wording", audits, args) if audits else {}
    for sid, revised in candidates.items():
        audit = judgments.get(sid, {})
        decision = "accept" if expand else audit.get("decision", "revise")
        if decision == "needs_math_review" and proposals[sid]["decision"] == "edit":
            decision = "revise"
        if decision == "accept":
            rows[sid] = revised
            decision = (
                "edited"
                if expand or proposals[sid]["decision"] == "edit"
                else (
                    "kept"
                    if proposals[sid]["decision"] == "keep"
                    else "needs_math_review"
                )
            )
        outcomes[sid] = {
            "source_id": sid,
            "result": decision,
            "reason": audit.get("reason", "No audit decision" if not expand else None),
            "patch": proposals[sid],
        }
    write_jsonl(args.output / "templates.jsonl", rows.values())
    write_jsonl(args.output / "outcomes.jsonl", outcomes.values())
