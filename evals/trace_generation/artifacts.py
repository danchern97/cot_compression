from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import uuid
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .config import GenerationConfig, rollout_seed
from .preparation import FILTER_PIPELINE, PreparedPrompt

MESSAGE_TYPE = pa.struct([("role", pa.string()), ("content", pa.string())])
TRACE_SCHEMA = pa.schema(
    [
        ("config_hash", pa.string()),
        ("source_row_index", pa.int64()),
        ("prompt_id", pa.string()),
        ("rollout_id", pa.string()),
        ("rollout_index", pa.int16()),
        ("seed", pa.int64()),
        ("domain", pa.string()),
        ("dataset_label", pa.string()),
        ("dataset_source", pa.string()),
        ("original_dataset", pa.string()),
        ("source_id", pa.string()),
        ("source_key", pa.string()),
        ("source_metadata_json", pa.string()),
        ("original_prompt", pa.string()),
        ("normalized_prompt", pa.string()),
        ("prepared_prompt", pa.string()),
        ("rendered_prompt", pa.string()),
        ("ground_truths", pa.list_(pa.string())),
        ("messages", pa.list_(MESSAGE_TYPE)),
        ("completion_raw", pa.string()),
        ("thinking", pa.string()),
        ("final_response", pa.string()),
        ("extracted_answer", pa.string()),
        ("score", pa.float64()),
        ("correct", pa.bool_()),
        ("qa_token_f1", pa.float64()),
        ("prompt_tokens", pa.int32()),
        ("completion_tokens", pa.int32()),
        ("requested_max_tokens", pa.int32()),
        ("finish_reason", pa.string()),
        ("stop_reason", pa.string()),
        ("ended_by_eos", pa.bool_()),
        ("complete", pa.bool_()),
        ("truncated", pa.bool_()),
        ("extraction_status", pa.string()),
        ("generation_error", pa.string()),
    ]
)

PREPARED_SCHEMA = pa.schema(
    [
        ("source_row_index", pa.int64()),
        ("prompt_id", pa.string()),
        ("source_id", pa.string()),
        ("source_key", pa.string()),
        ("source_metadata_json", pa.string()),
        ("domain", pa.string()),
        ("dataset_label", pa.string()),
        ("dataset_source", pa.string()),
        ("original_dataset", pa.string()),
        ("original_prompt", pa.string()),
        ("normalized_prompt", pa.string()),
        ("prepared_prompt", pa.string()),
        ("rendered_prompt", pa.string()),
        ("ground_truths", pa.list_(pa.string())),
        ("prompt_tokens", pa.int32()),
    ]
)


@contextmanager
def _atomic_path(path: Path) -> Iterator[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    with _atomic_path(path) as temporary:
        temporary.write_text(text, encoding="utf-8")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    atomic_write_text(path, text)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    return rows


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def environment_metadata() -> dict[str, Any]:
    commit, dirty = _git_state()
    gpu_names: list[str] = []
    gpu_compute_capabilities: list[list[int]] = []
    cuda_runtime: str | None = None
    cudnn_version: int | None = None
    try:
        import torch

        gpu_names = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        gpu_compute_capabilities = [
            list(torch.cuda.get_device_capability(index))
            for index in range(torch.cuda.device_count())
        ]
        cuda_runtime = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
    except (ImportError, RuntimeError):
        pass
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpus": gpu_names,
        "gpu_compute_capabilities": gpu_compute_capabilities,
        "cuda_runtime": cuda_runtime,
        "cudnn_version": cudnn_version,
        "vllm_runtime": {
            "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT") == "1",
            "v1_multiprocessing": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
            == "1",
        },
        "packages": {
            name: _package_version(name)
            for name in (
                "datasets",
                "huggingface-hub",
                "math-verify",
                "pyarrow",
                "torch",
                "transformers",
                "vllm",
            )
        },
    }


def shard_prompt_counts(total: int, num_shards: int) -> list[int]:
    quotient, remainder = divmod(total, num_shards)
    return [quotient + int(index < remainder) for index in range(num_shards)]


def build_manifest(
    config: GenerationConfig,
    filter_counts: dict[str, int],
    shard_counts: list[int] | None = None,
    *,
    filter_pipeline: Iterable[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    eligible = int(filter_counts["eligible_total"])
    counts = shard_counts or shard_prompt_counts(eligible, config.num_shards)
    if len(counts) != config.num_shards or sum(counts) != eligible:
        raise ValueError("shard counts do not match the configured prompt total")
    manifest = {
        "schema_version": 2,
        "config_hash": config.config_hash,
        "config": config.payload(),
        "filter_counts": filter_counts,
        "filter_pipeline": list(filter_pipeline or FILTER_PIPELINE),
        "resolved_revisions": {
            "dataset": config.dataset_revision,
            "model": config.model_revision,
        },
        "eligible_prompts": eligible,
        "expected_rollouts": eligible * config.num_rollouts,
        "shard_prompt_counts": counts,
        "shard_chunk_counts": [
            (count + config.chunk_size - 1) // config.chunk_size for count in counts
        ],
        "environment": environment_metadata(),
        "reproducibility_note": (
            "Identical outputs require the pinned dataset/model and software "
            "revisions plus a compatible GPU/runtime stack."
        ),
    }
    if extra:
        overlap = set(manifest) & set(extra)
        if overlap:
            raise ValueError(f"extra manifest fields overlap built-ins: {overlap}")
        manifest.update(extra)
    return manifest


def ensure_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    path = run_dir / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("config_hash") != manifest["config_hash"]:
            raise ValueError(
                f"Existing manifest at {path} has config hash "
                f"{existing.get('config_hash')}, expected {manifest['config_hash']}"
            )
        if existing.get("filter_counts") != manifest["filter_counts"]:
            raise ValueError("Existing manifest has different deterministic filters")
        if existing.get("config") != manifest["config"]:
            raise ValueError("Existing manifest configuration is incompatible")
        if existing.get("filter_pipeline") != manifest["filter_pipeline"]:
            raise ValueError("Existing manifest filter pipeline is incompatible")
        if existing.get("fulfillment") != manifest.get("fulfillment"):
            raise ValueError("Existing manifest fulfillment sources are incompatible")
        return path
    atomic_write_json(path, manifest)
    return path


def write_prepared_shard(
    run_dir: Path, shard_index: int, prompts: list[PreparedPrompt]
) -> Path:
    path = run_dir / "prepared" / f"shard-{shard_index:05d}.parquet"
    table = pa.Table.from_pylist(
        [prompt.to_dict() for prompt in prompts], PREPARED_SCHEMA
    )
    with _atomic_path(path) as temporary:
        pq.write_table(table, temporary, compression="zstd")
    return path


def chunk_path(run_dir: Path, shard_index: int, chunk_index: int) -> Path:
    return (
        run_dir
        / "chunks"
        / f"shard-{shard_index:05d}"
        / f"chunk-{chunk_index:06d}.jsonl"
    )


def _prompt_value(prompt: PreparedPrompt | Mapping[str, Any], field: str) -> Any:
    return prompt[field] if isinstance(prompt, Mapping) else getattr(prompt, field)


def _validate_rows(
    rows: list[dict[str, Any]],
    prompts: Iterable[PreparedPrompt | Mapping[str, Any]],
    config: GenerationConfig,
    path: Path,
) -> None:
    requests = [
        (prompt, rollout_index)
        for prompt in prompts
        for rollout_index in config.rollout_indices
    ]
    if len(rows) != len(requests):
        raise ValueError(f"Chunk {path} has {len(rows)} rows, expected {len(requests)}")
    for row, (prompt, rollout_index) in zip(rows, requests, strict=True):
        source_index = int(_prompt_value(prompt, "source_row_index"))
        expected = {
            "config_hash": config.config_hash,
            "rollout_id": f"{_prompt_value(prompt, 'prompt_id')}:r{rollout_index}",
            "rollout_index": rollout_index,
            "seed": rollout_seed(config.base_seed, source_index, rollout_index),
            "source_row_index": source_index,
            "domain": _prompt_value(prompt, "domain"),
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise ValueError(f"{field} mismatch in {path}")


def validate_chunk(
    path: Path,
    prompts: list[PreparedPrompt],
    config: GenerationConfig,
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    _validate_rows(rows, prompts, config, path)
    return rows


class _Metrics:
    def __init__(self) -> None:
        self.rollouts = 0
        self.graded_rollouts = 0
        self.correct = 0
        self.prompt_ids: set[str] = set()
        self.graded_prompt_ids: set[str] = set()
        self.correct_prompt_ids: set[str] = set()
        self.complete = 0
        self.complete_prompt_ids: set[str] = set()
        self.lengths: list[int] = []
        self.extraction_failures = 0
        self.truncated = 0
        self.generation_errors = 0
        self.qa_f1: list[float] = []

    def add(self, row: dict[str, Any]) -> None:
        self.rollouts += 1
        prompt_id = str(row["prompt_id"])
        self.prompt_ids.add(prompt_id)
        if row.get("correct") is not None:
            self.graded_rollouts += 1
            self.graded_prompt_ids.add(prompt_id)
            if bool(row["correct"]):
                self.correct += 1
                self.correct_prompt_ids.add(prompt_id)
        if bool(row.get("complete")):
            self.complete += 1
            self.complete_prompt_ids.add(prompt_id)
        self.lengths.append(int(row["completion_tokens"]))
        if row["extraction_status"] not in {"ok", "incorrect_answer"}:
            self.extraction_failures += 1
        self.truncated += int(bool(row["truncated"]))
        self.generation_errors += int(row.get("generation_error") is not None)
        if row.get("qa_token_f1") is not None:
            self.qa_f1.append(float(row["qa_token_f1"]))

    def summary(self) -> dict[str, int | float | None]:
        prompts = len(self.prompt_ids)
        graded_prompts = len(self.graded_prompt_ids)
        return {
            "rollouts": self.rollouts,
            "graded_rollouts": self.graded_rollouts,
            "correct_rollouts": self.correct,
            "rollout_pass_rate": (
                self.correct / self.graded_rollouts if self.graded_rollouts else None
            ),
            "prompts": prompts,
            "graded_prompts": graded_prompts,
            "prompts_passed": len(self.correct_prompt_ids),
            "prompt_pass_at_k": (
                len(self.correct_prompt_ids) / graded_prompts
                if graded_prompts
                else None
            ),
            "complete_rollouts": self.complete,
            "completion_rate": self.complete / self.rollouts if self.rollouts else 0.0,
            "prompts_with_complete": len(self.complete_prompt_ids),
            "prompt_complete_at_k": (
                len(self.complete_prompt_ids) / prompts if prompts else 0.0
            ),
            "extraction_failures": self.extraction_failures,
            "extraction_failure_rate": (
                self.extraction_failures / self.rollouts if self.rollouts else 0.0
            ),
            "truncated_rollouts": self.truncated,
            "truncation_rate": self.truncated / self.rollouts if self.rollouts else 0.0,
            "generation_errors": self.generation_errors,
            "completion_tokens_mean": (
                statistics.fmean(self.lengths) if self.lengths else 0.0
            ),
            "completion_tokens_median": (
                statistics.median(self.lengths) if self.lengths else 0.0
            ),
            "qa_token_f1_mean": (statistics.fmean(self.qa_f1) if self.qa_f1 else None),
        }


def _prepared_shard_rows(
    run_dir: Path, manifest: dict[str, Any]
) -> list[list[dict[str, Any]]]:
    shards: list[list[dict[str, Any]]] = []
    all_indices: list[int] = []
    all_prompt_ids: list[str] = []
    for shard_index, expected_count in enumerate(manifest["shard_prompt_counts"]):
        path = run_dir / "prepared" / f"shard-{shard_index:05d}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Run is incomplete; missing prepared shard {path}")
        rows = pq.read_table(
            path, columns=["source_row_index", "prompt_id", "domain"]
        ).to_pylist()
        if len(rows) != int(expected_count):
            raise ValueError(
                f"Prepared shard {path} has {len(rows)} prompts, expected "
                f"{expected_count}"
            )
        shards.append(rows)
        all_indices.extend(int(row["source_row_index"]) for row in rows)
        all_prompt_ids.extend(str(row["prompt_id"]) for row in rows)

    if len(all_indices) != len(set(all_indices)):
        raise ValueError("Prepared shards are not a non-overlapping partition")
    for shard_index, rows in enumerate(shards):
        indices = [int(row["source_row_index"]) for row in rows]
        if indices != sorted(indices) or any(
            index % len(shards) != shard_index for index in indices
        ):
            raise ValueError("Prepared shards are not the stable modulo partition")
    if len(all_prompt_ids) != len(set(all_prompt_ids)):
        raise ValueError("Prepared shards contain duplicate prompt IDs")
    if len(all_indices) != int(manifest["eligible_prompts"]):
        raise ValueError("Prepared shard prompt total does not match the manifest")
    return shards


def summarize_run(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = GenerationConfig(**manifest["config"])
    prepared_shards = _prepared_shard_rows(run_dir, manifest)

    metrics: dict[str, _Metrics] = defaultdict(_Metrics)
    seen_rollouts: set[str] = set()
    with (
        _atomic_path(run_dir / "traces.parquet") as temporary,
        pq.ParquetWriter(temporary, TRACE_SCHEMA, compression="zstd") as writer,
    ):
        for shard_index, prepared in enumerate(prepared_shards):
            for chunk_index, start in enumerate(
                range(0, len(prepared), config.chunk_size)
            ):
                path = chunk_path(run_dir, shard_index, chunk_index)
                if not path.exists():
                    raise FileNotFoundError(f"Run is incomplete; missing chunk {path}")
                rows = read_jsonl(path)
                _validate_rows(
                    rows, prepared[start : start + config.chunk_size], config, path
                )
                for row in rows:
                    rollout_id = str(row["rollout_id"])
                    if rollout_id in seen_rollouts:
                        raise ValueError(f"Duplicate rollout ID: {rollout_id}")
                    seen_rollouts.add(rollout_id)
                    metrics["global"].add(row)
                    metrics[str(row["domain"])].add(row)
                writer.write_table(pa.Table.from_pylist(rows, TRACE_SCHEMA))

        if len(seen_rollouts) != int(manifest["expected_rollouts"]):
            raise ValueError(
                f"Expected {manifest['expected_rollouts']} rollouts, found "
                f"{len(seen_rollouts)} unique IDs"
            )

    payload = {
        "config_hash": manifest["config_hash"],
        "filter_counts": manifest["filter_counts"],
        "global": metrics["global"].summary(),
        "domains": {
            domain: metrics[domain].summary()
            for domain in sorted(name for name in metrics if name != "global")
        },
    }
    atomic_write_json(run_dir / "summary.json", payload)
    csv_path = run_dir / "summary_by_domain.csv"
    rows = [
        dict(domain=domain, **values) for domain, values in payload["domains"].items()
    ]
    with (
        _atomic_path(csv_path) as temporary,
        temporary.open("w", encoding="utf-8", newline="") as handle,
    ):
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    return payload


def format_summary(payload: dict[str, Any]) -> str:
    header = (
        f"{'domain':<20} {'rollouts':>10} {'complete':>9} {'prompt≥1':>9} "
        f"{'graded':>8} {'pass':>8} {'trunc':>8} {'errors':>7} {'mean tok':>9}"
    )
    lines = [header, "-" * len(header)]
    groups = [("global", payload["global"])] + list(payload["domains"].items())
    for name, values in groups:
        lines.append(
            f"{name:<20} {values['rollouts']:>10d} "
            f"{values['completion_rate']:>8.2%} "
            f"{values['prompt_complete_at_k']:>8.2%} "
            f"{values['graded_rollouts']:>8d} "
            f"{_format_rate(values['rollout_pass_rate']):>8} "
            f"{values['truncation_rate']:>7.2%} "
            f"{values['generation_errors']:>7d} "
            f"{values['completion_tokens_mean']:>9.1f}"
        )
    lines.append("")
    lines.append(
        "filters: "
        + ", ".join(
            f"{name}={count}"
            for name, count in sorted(payload["filter_counts"].items())
        )
    )
    return "\n".join(lines)


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"
