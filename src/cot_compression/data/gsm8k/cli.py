"""Run the GSM8K extraction and review workflow."""

import argparse
import fcntl
import logging
from pathlib import Path

from cot_compression.data.gsm8k.generation import assess_or_repair, generate
from cot_compression.data.gsm8k.io import index_by_source_id, load_source, read_jsonl
from cot_compression.data.gsm8k.revisions import classify, patch_stage
from cot_compression.data.gsm8k.schemas import COMMANDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=COMMANDS)
    parser.add_argument(
        "source", type=Path, help="GSM8K training JSONL in its original order"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--templates", type=Path, help="Input templates.jsonl from a prior stage"
    )
    parser.add_argument(
        "--findings", type=Path, help="Assessment, classifier, or manual review JSONL"
    )
    parser.add_argument(
        "--prior-templates",
        type=Path,
        help="Templates before text expansion, for classify",
    )
    parser.add_argument(
        "--feedback", type=Path, help="Earlier text-patch outcomes for a retry"
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        help="Exclude source IDs from this JSONL when generating",
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--prompts", type=Path, default=Path("prompts"))
    parser.add_argument("--model", help="Override the extraction or edit model")
    parser.add_argument(
        "--batch", action="store_true", help="Submit the primary requests through Batch"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare JSONL requests without calling the API",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.stage != "generate" and not args.templates:
        parser.error("--templates is required for this stage")
    if args.stage in {"repair", "repair-wording"} and not args.findings:
        parser.error("--findings is required for this stage")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    source = load_source(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == "generate":
            generate(source, args)
            return
        rows = index_by_source_id(read_jsonl(args.templates))
        if args.stage in {"assess", "repair"}:
            assess_or_repair(rows, source, args)
        elif args.stage == "classify":
            classify(rows, source, args)
        else:
            patch_stage(rows, source, args)


if __name__ == "__main__":
    main()
