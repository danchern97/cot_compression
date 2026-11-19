"""Run and resume cached OpenAI requests in Flex or Batch."""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import APIConnectionError, APIStatusError, OpenAI

from cot_compression.data.gsm8k.io import Record, RecordsById, read_jsonl, write_jsonl
from cot_compression.data.gsm8k.schemas import STAGES

LUNA = "gpt-6-luna"
SOL = "gpt-6.2-sol"
ENDPOINT = "/v1/responses"

logger = logging.getLogger(__name__)


def openai_client() -> OpenAI:
    """Load the API key from the environment or the private local config."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    path = Path.home() / ".config/cot-compression/openai.env"
    if not key and path.is_file():
        if path.stat().st_mode & 0o077:
            raise ValueError(f"Set permissions on {path} to 600")
        for line in path.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip().removeprefix("export ") == "OPENAI_API_KEY":
                key = value.strip().strip("\"'")
                break
    if not key:
        raise ValueError(
            "Set OPENAI_API_KEY or save it in ~/.config/cot-compression/openai.env"
        )
    return OpenAI(
        api_key=key, base_url="https://api.openai.com/v1", max_retries=0, timeout=180
    )


def prompt_parts(stage: str, path: Path) -> tuple[str, str]:
    """Read the shared instructions and the suffix that follows each input."""
    name = STAGES[stage][0]
    prompt = (path / f"{name}.txt").read_text(encoding="utf-8")
    if stage == "repair":
        # A computational repair uses the same template rules as extraction.
        prefix, _ = prompt_parts("generate", path)
        return prefix + "\n" + prompt, ""
    prefix, tail = re.split(r"\{\{(?:seed|template|items)_json\}\}", prompt, maxsplit=1)
    return prefix, tail


def build_requests(
    stage: str, inputs: RecordsById, args: argparse.Namespace
) -> list[Record]:
    """Build cacheable requests and keep their source IDs for reconciliation.

    Args:
        stage: Name of the prompt and response schema to use.
        inputs: Prompt inputs keyed by GSM8K source ID.
        args: Model, prompt directory, and workflow options from the CLI.

    Returns:
        Responses API requests with stable IDs and the source IDs they cover.
    """
    prompt, tail = prompt_parts(stage, args.prompts)
    name, schema, group_size = STAGES[stage]
    model = (args.model if stage == args.stage else None) or (
        SOL if stage == "repair" else LUNA
    )
    items = list(inputs.items())
    rows = []
    for start in range(0, len(items), group_size):
        group = items[start : start + group_size]
        source_ids, data = zip(*group, strict=True)
        payload = list(data) if group_size > 1 else data[0]
        body = {
            "model": model,
            "input": [
                {
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                            # Only the instructions are shared across templates.
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False) + tail,
                },
            ],
            "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
            "reasoning": {"effort": "low"},
            "max_output_tokens": getattr(args, "max_output_tokens", None)
            or (2048 if stage == "assess" else 4096),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": name,
                    "schema": schema,
                    "strict": True,
                }
            },
        }
        # A different prompt, model, or payload must have a different journal key.
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:24]
        rows.append(
            {
                "custom_id": key,
                "source_ids": list(source_ids),
                "method": "POST",
                "url": ENDPOINT,
                "body": body,
            }
        )
    return rows


def decode_response(body: Record) -> Record:
    if body.get("status") != "completed":
        raise ValueError(f"Response status {body.get('status')}")
    content = "".join(
        part["text"]
        for item in body.get("output", [])
        if item["type"] == "message"
        for part in item["content"]
        if part["type"] == "output_text"
    )
    return json.loads(content)


def credit_balance_exhausted(error: APIStatusError) -> bool:
    return error.status_code == 429 and (
        error.code in {"insufficient_quota", "credit_balance_exhausted"}
        or error.type == "insufficient_quota"
    )


def fetch_flex(client: OpenAI, request: Record) -> Record:
    for attempt in range(6):
        try:
            response = client.responses.create(**request["body"], service_tier="flex")
            if response.service_tier != "flex":
                raise ValueError(
                    f"API returned {response.service_tier!r} instead of Flex"
                )
            return response.model_dump(mode="json")
        except (APIStatusError, APIConnectionError) as error:
            if isinstance(error, APIStatusError) and (
                error.status_code not in {429, 500, 502, 503, 504}
                or credit_balance_exhausted(error)
            ):
                raise
            if attempt == 5:
                raise
            time.sleep(2**attempt + random.random())
    raise AssertionError("Unreachable retry state")


def collect_batch(
    client: OpenAI, pending: list[Record], directory: Path
) -> list[tuple[str, Record]]:
    """Resume an existing Batch job or submit one and collect its responses."""
    manifest = directory / "batch.json"
    if manifest.exists():
        job_id = json.loads(manifest.read_text())["id"]
    else:
        path = directory / "batch-input.jsonl"
        write_jsonl(
            path, ({k: v for k, v in r.items() if k != "source_ids"} for r in pending)
        )
        with path.open("rb") as handle:
            upload = client.files.create(file=handle, purpose="batch")
        job = client.batches.create(
            input_file_id=upload.id, endpoint=ENDPOINT, completion_window="24h"
        )
        job_id = job.id
        write_jsonl(manifest, [{"id": job_id}])
        logger.info("Submitted Batch %s", job_id)
    job = client.batches.retrieve(job_id)
    while job.status not in {"completed", "failed", "expired", "cancelled"}:
        time.sleep(30)
        job = client.batches.retrieve(job_id)
    if job.status != "completed" or not job.output_file_id:
        raise ValueError(f"Batch {job_id} ended with status {job.status}")
    lines = client.files.content(job.output_file_id).text.splitlines()
    result = []
    for line in lines:
        row = json.loads(line)
        response = row.get("response") or {}
        if response.get("status_code") == 200:
            result.append((row["custom_id"], response["body"]))
    return result


def run_requests(
    stage: str, inputs: RecordsById, args: argparse.Namespace
) -> RecordsById:
    """Run one stage and map completed structured responses to source IDs.

    Args:
        stage: Name of the prompt to run.
        inputs: Prompt inputs keyed by GSM8K source ID.
        args: Output directory, execution mode, and worker count from the CLI.

    Returns:
        Parsed responses indexed by source ID. Missing responses have no entry.
    """
    directory = args.output / stage
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "requests.jsonl"
    journal = directory / "responses.jsonl"
    work = build_requests(stage, inputs, args)
    if manifest.exists() and read_jsonl(manifest) != work:
        raise ValueError(
            f"Requests changed in {directory}. Use a new output directory."
        )
    write_jsonl(manifest, work)
    if args.dry_run or not work:
        return {}
    saved = (
        {row["custom_id"]: row["body"] for row in read_jsonl(journal)}
        if journal.exists()
        else {}
    )
    pending = [row for row in work if row["custom_id"] not in saved]
    if pending:
        client = openai_client()
        with journal.open("a", encoding="utf-8") as handle:

            def record(key, body):
                saved[key] = body
                handle.write(json.dumps({"custom_id": key, "body": body}) + "\n")
                handle.flush()
                if len(saved) % 100 == 0 or len(saved) == len(work):
                    logger.info("%s: %d/%d responses", stage, len(saved), len(work))

            if args.batch and stage == args.stage:
                for key, body in collect_batch(client, pending, directory):
                    if key not in saved:
                        record(key, body)
            else:
                # Warm the shared prompt cache before starting concurrent calls.
                first, *rest = pending
                record(first["custom_id"], fetch_flex(client, first))
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = {
                        pool.submit(fetch_flex, client, row): row["custom_id"]
                        for row in rest
                    }
                    for future in as_completed(futures):
                        try:
                            record(futures[future], future.result())
                        except (APIConnectionError, APIStatusError) as error:
                            if isinstance(
                                error, APIStatusError
                            ) and credit_balance_exhausted(error):
                                pool.shutdown(wait=True, cancel_futures=True)
                                # Preserve successes that were in flight when quota ran out.
                                for finished, key in futures.items():
                                    if (
                                        key not in saved
                                        and not finished.cancelled()
                                        and finished.exception() is None
                                    ):
                                        record(key, finished.result())
                                raise
                            logger.warning(
                                "%s: %s failed with %s",
                                stage,
                                futures[future],
                                type(error).__name__,
                            )
    result = {}
    for row in work:
        ids = row["source_ids"]
        try:
            answer = decode_response(saved[row["custom_id"]])
        except (KeyError, TypeError, ValueError):
            continue
        if "decisions" in answer:
            entries = answer["decisions"]
            seen = Counter(item["source_id"] for item in entries)
            result.update(
                (item["source_id"], item)
                for item in entries
                if item["source_id"] in ids and seen[item["source_id"]] == 1
            )
        elif stage == "assess" or answer.get("source_id") == ids[0]:
            result[ids[0]] = answer
    return result
