"""Extract templates, assess their arithmetic, and repair flagged templates."""

import argparse
import random
import re
from fractions import Fraction

from cot_compression.data.gsm8k.api import prompt_parts, run_requests
from cot_compression.data.gsm8k.io import (
    RecordsById,
    index_by_source_id,
    read_jsonl,
    write_jsonl,
)
from cot_compression.data.gsm8k.sampling import examples, expected_answer
from cot_compression.data.gsm8k.templates import BAD_TEMPLATE, validate_template


def generate(source: RecordsById, args: argparse.Namespace) -> None:
    """Extract templates from a seeded sample of GSM8K problems."""
    rng = random.Random(args.seed)
    prefix, _ = prompt_parts("generate", args.prompts)
    excluded = set(re.findall(r'"source_id"\s*:\s*"(gsm8k/train/\d+)"', prefix))
    if args.exclude:
        excluded.update(index_by_source_id(read_jsonl(args.exclude)))
    # Sample across source calculation counts. The first bucket covers 0–1
    # annotations and the last covers 6 or more.
    buckets = [[] for _ in range(6)]
    for sid, row in source.items():
        if sid not in excluded:
            count = len(re.findall(r"<<[^<>]+>>", row["answer"]))
            buckets[max(0, min(count, 6) - 1)].append(sid)
    eligible = [sid for bucket in buckets for sid in bucket]
    if args.count < 1 or args.count > len(eligible):
        raise ValueError(f"Choose a count from 1 to {len(eligible)}")
    initial = [
        sid
        for bucket, percent in zip(buckets, (10, 20, 20, 20, 15, 15), strict=True)
        for sid in rng.sample(bucket, min(len(bucket), percent * args.count // 100))
    ]
    chosen = set(initial)
    initial += rng.sample(
        [sid for sid in eligible if sid not in chosen], args.count - len(initial)
    )
    rng.shuffle(initial)
    results = run_requests("generate", {sid: source[sid] for sid in initial}, args)
    if not args.dry_run:
        write_jsonl(
            args.output / "templates.jsonl",
            (r for r in results.values() if r["status"] == "ok"),
        )
        write_jsonl(
            args.output / "outcomes.jsonl",
            (
                results.get(sid, {"source_id": sid, "status": "missing"})
                for sid in initial
            ),
        )


def audit_math(
    rows: RecordsById, source: RecordsById, args: argparse.Namespace
) -> RecordsById:
    """Replay the original answer, then compare two blind solves per template."""
    selected, notes = {}, {}
    for sid, row in rows.items():
        try:
            validate_template(row["template"], source[sid]["answer"])
            pair = examples(row["template"])
        except BAD_TEMPLATE as error:
            notes[sid] = {"source_id": sid, "findings": [str(error)]}
            continue
        selected[sid] = dict(zip(("ordinary", "boundary"), pair[:2], strict=True))
    judgments = run_requests(
        "assess",
        # Hide the template answer and solution from the independent solver.
        {
            sid: {key: example["question"] for key, example in pair.items()}
            for sid, pair in selected.items()
        },
        args,
    )
    if args.dry_run:
        return {}
    for sid, pair in selected.items():
        issues = []
        for key, example in pair.items():
            judged = judgments.get(sid, {}).get(key, {})
            if judged.get("issue"):
                issues.append(f"{key}: {judged['issue']}")
            try:
                if Fraction(judged["answer"]) != expected_answer(example):
                    issues.append(
                        f"{key}: independent answer {judged['answer']} disagrees with the template"
                    )
            except BAD_TEMPLATE:
                issues.append(f"{key}: no independent numeric answer")
        notes[sid] = {"source_id": sid, "findings": issues, **pair}
    return notes


def assess_or_repair(
    rows: RecordsById, source: RecordsById, args: argparse.Namespace
) -> None:
    """Save templates that pass assessment and record findings for every input."""
    if args.stage == "repair":
        findings = index_by_source_id(read_jsonl(args.findings))
        jobs = {
            sid: {**source[sid], "proposed_template": row["template"], **findings[sid]}
            for sid, row in rows.items()
            if findings[sid]["findings"]
        }
        replies = run_requests("repair", jobs, args)
        if args.dry_run:
            return
        repairs = {sid: r for sid, r in replies.items() if r["status"] == "ok"}
        checked = audit_math(repairs, source, args)
        outcomes = findings | checked
        for sid in jobs.keys() - repairs.keys():
            outcomes[sid] = {
                **findings[sid],
                "repair_status": replies.get(sid, {}).get("status", "missing"),
            }
        rows = rows | repairs
    else:
        outcomes = audit_math(rows, source, args)
    if not args.dry_run:
        write_jsonl(args.output / "outcomes.jsonl", outcomes.values())
        write_jsonl(
            args.output / "templates.jsonl",
            (row for sid, row in rows.items() if not outcomes[sid]["findings"]),
        )
