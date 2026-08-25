from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .artifacts import PREPARED_SCHEMA, atomic_write_json, atomic_write_text
from .config import DOMAIN_NAMES, GenerationConfig
from .preparation import PreparedPrompt

FULFILLMENT_PIPELINE = (
    "load and validate completed source-run manifests and prepared prompts",
    "select prompts for which no source rollout has complete=true",
    "retain only requested domains and apply the optional deterministic limit",
    "render with the fulfillment model chat template and thinking enabled",
    "reject prompts exceeding the domain-specific context budget",
    "generate new non-colliding rollout indices without modifying source traces",
)


def _manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing source manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def source_rollout_indices(source_runs: list[Path]) -> set[int]:
    indices: set[int] = set()
    for run_dir in source_runs:
        config = _manifest(run_dir)["config"]
        start = int(config.get("rollout_index_start", 0))
        count = int(config["num_rollouts"])
        indices.update(range(start, start + count))
    return indices


def next_rollout_index(source_runs: list[Path]) -> int:
    indices = source_rollout_indices(source_runs)
    return max(indices, default=-1) + 1


def _prepared_prompts(run_dir: Path, manifest: dict[str, Any]) -> list[PreparedPrompt]:
    prompts: list[PreparedPrompt] = []
    for shard_index, expected in enumerate(manifest["shard_prompt_counts"]):
        path = run_dir / "prepared" / f"shard-{shard_index:05d}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing source prepared shard: {path}")
        rows = pq.read_table(path, schema=PREPARED_SCHEMA).to_pylist()
        if len(rows) != int(expected):
            raise ValueError(
                f"Source prepared shard {path} has {len(rows)} prompts, "
                f"expected {expected}"
            )
        for row in rows:
            row["ground_truths"] = tuple(row["ground_truths"] or ())
            prompts.append(PreparedPrompt(**row))
    if len(prompts) != int(manifest["eligible_prompts"]):
        raise ValueError(f"Prepared prompt total is inconsistent in {run_dir}")
    return prompts


def _completion_state(
    run_dir: Path, manifest: dict[str, Any]
) -> tuple[set[str], set[str]]:
    path = run_dir / "traces.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Source run must be summarized before fulfillment: {path}"
        )
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != int(manifest["expected_rollouts"]):
        raise ValueError(
            f"Source traces {path} contain {parquet.metadata.num_rows} rows, "
            f"expected {manifest['expected_rollouts']}"
        )
    prompt_ids: set[str] = set()
    complete_ids: set[str] = set()
    for batch in parquet.iter_batches(columns=["prompt_id", "complete"]):
        for row in batch.to_pylist():
            prompt_id = str(row["prompt_id"])
            prompt_ids.add(prompt_id)
            if bool(row["complete"]):
                complete_ids.add(prompt_id)
    return prompt_ids, complete_ids


def _source_data(
    source_runs: list[Path],
) -> tuple[list[PreparedPrompt], set[str], list[dict[str, Any]]]:
    prompts_by_id: dict[str, PreparedPrompt] = {}
    complete_ids: set[str] = set()
    descriptors: list[dict[str, Any]] = []
    identity: tuple[str, str, str] | None = None

    for raw_run_dir in source_runs:
        run_dir = raw_run_dir.resolve()
        manifest = _manifest(run_dir)
        source_config = manifest["config"]
        source_identity = (
            str(source_config["dataset"]),
            str(source_config["dataset_revision"]),
            str(source_config["split"]),
        )
        if identity is None:
            identity = source_identity
        elif source_identity != identity:
            raise ValueError("Source runs use different dataset identities")

        source_prompts = _prepared_prompts(run_dir, manifest)
        trace_prompt_ids, source_complete_ids = _completion_state(run_dir, manifest)
        prepared_ids = {prompt.prompt_id for prompt in source_prompts}
        if trace_prompt_ids != prepared_ids:
            raise ValueError(f"Prepared and trace prompt IDs differ in {run_dir}")
        for prompt in source_prompts:
            existing = prompts_by_id.get(prompt.prompt_id)
            if existing is not None:
                stable_existing = (
                    existing.source_row_index,
                    existing.domain,
                    existing.prepared_prompt,
                    existing.ground_truths,
                )
                stable_prompt = (
                    prompt.source_row_index,
                    prompt.domain,
                    prompt.prepared_prompt,
                    prompt.ground_truths,
                )
                if stable_existing != stable_prompt:
                    raise ValueError(
                        f"Conflicting prompt metadata across source runs for "
                        f"{prompt.prompt_id}"
                    )
            else:
                prompts_by_id[prompt.prompt_id] = prompt
        complete_ids.update(source_complete_ids)
        manifest_bytes = (run_dir / "manifest.json").read_bytes()
        descriptors.append(
            {
                "path": str(run_dir),
                "config_hash": str(manifest["config_hash"]),
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "prompts": len(source_prompts),
                "rollouts": int(manifest["expected_rollouts"]),
            }
        )
    prompts = sorted(prompts_by_id.values(), key=lambda prompt: prompt.source_row_index)
    return prompts, complete_ids, descriptors


def _rerender(
    prompt: PreparedPrompt,
    tokenizer: Any,
    config: GenerationConfig,
) -> tuple[PreparedPrompt | None, str]:
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt.prepared_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        encoded = tokenizer(str(rendered), add_special_tokens=False)["input_ids"]
    except Exception:
        return None, "chat_template_error"
    if len(encoded) + config.max_tokens_for(prompt.domain) > config.max_model_len:
        return None, "prompt_over_context_budget"
    return (
        replace(
            prompt,
            rendered_prompt=str(rendered),
            prompt_tokens=len(encoded),
        ),
        "eligible",
    )


def prepare_fulfillment_prompts(
    source_runs: list[Path],
    tokenizer: Any,
    config: GenerationConfig,
) -> tuple[list[PreparedPrompt], dict[str, int], dict[str, Any]]:
    if not source_runs:
        raise ValueError("At least one --source-run is required")
    source_prompts, complete_ids, descriptors = _source_data(source_runs)
    expected_identity = (config.dataset, config.dataset_revision, config.split)
    for descriptor, run_dir in zip(descriptors, source_runs, strict=True):
        source_config = _manifest(run_dir)["config"]
        actual_identity = (
            source_config["dataset"],
            source_config["dataset_revision"],
            source_config["split"],
        )
        if actual_identity != expected_identity:
            raise ValueError(
                f"Fulfillment dataset identity differs from {descriptor['path']}"
            )

    counts: Counter[str] = Counter()
    selected: list[PreparedPrompt] = []
    for prompt in source_prompts:
        if prompt.domain not in config.domains:
            counts["domain_not_requested"] += 1
            continue
        counts["source_prompts"] += 1
        if prompt.prompt_id in complete_ids:
            counts["source_prompts_with_complete"] += 1
            continue
        counts["source_prompts_without_complete"] += 1
        if config.max_examples is not None and len(selected) >= config.max_examples:
            counts["not_selected_at_max_examples"] += 1
            continue
        rerendered, reason = _rerender(prompt, tokenizer, config)
        counts[reason] += 1
        if rerendered is not None:
            selected.append(rerendered)

    counts["eligible_total"] = len(selected)
    for domain in DOMAIN_NAMES:
        counts[f"eligible_{domain}"] = sum(
            prompt.domain == domain for prompt in selected
        )
    digest = hashlib.sha256()
    for prompt in selected:
        digest.update(f"{prompt.source_row_index}:{prompt.prompt_id}\n".encode())
    provenance = {
        "selection": "no source rollout has complete=true",
        "source_runs": descriptors,
        "source_prompt_count": int(counts["source_prompts"]),
        "source_prompts_with_complete": int(counts["source_prompts_with_complete"]),
        "source_prompts_without_complete": int(
            counts["source_prompts_without_complete"]
        ),
        "selected_prompt_hash": digest.hexdigest(),
    }
    return selected, dict(sorted(counts.items())), provenance


def _prompt_completion_by_domain(
    path: Path,
) -> tuple[dict[str, set[str]], dict[str, set[str]], int, int]:
    prompts: dict[str, set[str]] = defaultdict(set)
    complete: dict[str, set[str]] = defaultdict(set)
    rollouts = 0
    complete_rollouts = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["domain", "prompt_id", "complete"]):
        for row in batch.to_pylist():
            domain = str(row["domain"])
            prompt_id = str(row["prompt_id"])
            prompts[domain].add(prompt_id)
            rollouts += 1
            if bool(row["complete"]):
                complete[domain].add(prompt_id)
                complete_rollouts += 1
    return prompts, complete, rollouts, complete_rollouts


def summarize_fulfillment_coverage(run_dir: Path) -> dict[str, Any]:
    manifest = _manifest(run_dir)
    fulfillment = manifest.get("fulfillment")
    if not isinstance(fulfillment, dict):
        raise ValueError(f"Run at {run_dir} is not a fulfillment run")

    source_prompts: dict[str, set[str]] = defaultdict(set)
    source_complete: dict[str, set[str]] = defaultdict(set)
    allowed_domains = set(manifest["config"]["domains"])
    for source in fulfillment["source_runs"]:
        prompts, complete, _, _ = _prompt_completion_by_domain(
            Path(source["path"]) / "traces.parquet"
        )
        for domain, values in prompts.items():
            if domain not in allowed_domains:
                continue
            source_prompts[domain].update(values)
        for domain, values in complete.items():
            if domain not in allowed_domains:
                continue
            source_complete[domain].update(values)

    added_prompts, added_complete, added_rollouts, added_complete_rollouts = (
        _prompt_completion_by_domain(run_dir / "traces.parquet")
    )
    domains: dict[str, dict[str, int | float]] = {}
    for domain in sorted(set(source_prompts) | set(added_prompts)):
        baseline = source_prompts[domain]
        baseline_complete = source_complete[domain]
        targets = baseline - baseline_complete
        attempted = added_prompts[domain]
        if not attempted <= targets:
            raise ValueError(
                f"Fulfillment contains non-missing prompts in domain {domain}"
            )
        recovered = targets & added_complete[domain]
        after = baseline_complete | recovered
        domains[domain] = {
            "source_prompts": len(baseline),
            "source_prompts_with_complete": len(baseline_complete),
            "source_prompt_completion_rate": (
                len(baseline_complete) / len(baseline) if baseline else 0.0
            ),
            "target_prompts": len(targets),
            "attempted_prompts": len(attempted),
            "recovered_prompts": len(recovered),
            "fulfillment_success_rate": (
                len(recovered) / len(attempted) if attempted else 0.0
            ),
            "prompts_still_without_complete": len(baseline - after),
            "combined_prompts_with_complete": len(after),
            "combined_prompt_completion_rate": (
                len(after) / len(baseline) if baseline else 0.0
            ),
        }

    totals: Counter[str] = Counter()
    for values in domains.values():
        for key, value in values.items():
            if isinstance(value, int):
                totals[key] += value
    global_metrics: dict[str, int | float] = dict(totals)
    source_total = totals["source_prompts"]
    attempted_total = totals["attempted_prompts"]
    global_metrics.update(
        {
            "source_prompt_completion_rate": (
                totals["source_prompts_with_complete"] / source_total
                if source_total
                else 0.0
            ),
            "fulfillment_success_rate": (
                totals["recovered_prompts"] / attempted_total
                if attempted_total
                else 0.0
            ),
            "combined_prompt_completion_rate": (
                totals["combined_prompts_with_complete"] / source_total
                if source_total
                else 0.0
            ),
            "fulfillment_rollouts": added_rollouts,
            "fulfillment_complete_rollouts": added_complete_rollouts,
            "fulfillment_rollout_completion_rate": (
                added_complete_rollouts / added_rollouts if added_rollouts else 0.0
            ),
        }
    )
    payload = {
        "config_hash": manifest["config_hash"],
        "global": global_metrics,
        "domains": domains,
    }
    atomic_write_json(run_dir / "coverage_summary.json", payload)
    buffer = io.StringIO()
    rows = [dict(domain=domain, **values) for domain, values in domains.items()]
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_text(run_dir / "coverage_by_domain.csv", buffer.getvalue())
    return payload


def format_fulfillment_coverage(payload: dict[str, Any]) -> str:
    header = (
        f"{'domain':<20} {'targets':>9} {'attempted':>10} {'recovered':>10} "
        f"{'still miss':>11} {'before':>9} {'after':>9}"
    )
    lines = [header, "-" * len(header)]
    groups = [("global", payload["global"]), *payload["domains"].items()]
    for domain, values in groups:
        lines.append(
            f"{domain:<20} {values['target_prompts']:>9d} "
            f"{values['attempted_prompts']:>10d} "
            f"{values['recovered_prompts']:>10d} "
            f"{values['prompts_still_without_complete']:>11d} "
            f"{values['source_prompt_completion_rate']:>8.2%} "
            f"{values['combined_prompt_completion_rate']:>8.2%}"
        )
    return "\n".join(lines)
