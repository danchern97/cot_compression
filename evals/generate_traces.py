#!/usr/bin/env python3
"""Prepare, generate, score, and summarize reproducible reasoning traces."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from trace_generation.artifacts import (
    atomic_write_jsonl,
    build_manifest,
    chunk_path,
    ensure_manifest,
    format_summary,
    summarize_run,
    validate_chunk,
    write_prepared_shard,
)
from trace_generation.config import (
    BASE_SEED,
    CODE_MAX_OUTPUT_TOKENS,
    DATASET_NAME,
    DATASET_REVISION,
    DATASET_SPLIT,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DOMAIN_NAMES,
    FULFILLMENT_NUM_ROLLOUTS,
    MAX_OUTPUT_TOKENS,
    MODEL_NAME,
    MODEL_REVISION,
    NUM_ROLLOUTS,
    GenerationConfig,
)
from trace_generation.fulfillment import (
    FULFILLMENT_PIPELINE,
    format_fulfillment_coverage,
    next_rollout_index,
    prepare_fulfillment_prompts,
    source_rollout_indices,
    summarize_fulfillment_coverage,
)
from trace_generation.generation import build_engine, generate_chunk
from trace_generation.preparation import partition_prompts, prepare_prompts


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _resolve_revision(repo_type: str, repo_id: str, revision: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    if repo_type == "dataset":
        info = api.dataset_info(repo_id, revision=revision)
    else:
        info = api.model_info(repo_id, revision=revision)
    if not info.sha:
        raise RuntimeError(f"Could not resolve {repo_type} revision for {repo_id}")
    return info.sha


def _model_context_length(model: str, revision: str) -> int:
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(
        model,
        revision=revision,
        trust_remote_code=False,
    )
    for key in ("max_position_embeddings", "n_positions", "seq_length"):
        value = getattr(model_config, key, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError(
        f"Model config for {model}@{revision} does not expose a context length; "
        "pass --max-model-len explicitly"
    )


def _make_config(
    args: argparse.Namespace,
    *,
    rollout_index_start: int | None = None,
) -> GenerationConfig:
    requested_dataset_revision = args.dataset_revision or (
        DATASET_REVISION if args.dataset == DATASET_NAME else "main"
    )
    requested_model_revision = args.model_revision or (
        MODEL_REVISION if args.model == MODEL_NAME else "main"
    )
    dataset_revision = _resolve_revision(
        "dataset", args.dataset, requested_dataset_revision
    )
    model_revision = _resolve_revision("model", args.model, requested_model_revision)
    max_model_len = args.max_model_len or _model_context_length(
        args.model, model_revision
    )
    return GenerationConfig(
        dataset=args.dataset,
        dataset_revision=dataset_revision,
        split=args.split,
        model=args.model,
        model_revision=model_revision,
        domains=tuple(args.domains),
        num_rollouts=args.num_rollouts,
        rollout_index_start=(
            args.rollout_index_start
            if rollout_index_start is None
            else rollout_index_start
        ),
        base_seed=args.base_seed,
        max_output_tokens=args.max_output_tokens,
        code_max_output_tokens=args.code_max_output_tokens,
        default_max_output_tokens=args.default_max_output_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        batch_invariant=args.batch_invariant,
        v1_multiprocessing=args.v1_multiprocessing,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=args.enable_prefix_caching,
        max_model_len=max_model_len,
        max_examples=args.max_examples,
        num_shards=args.num_shards,
        chunk_size=args.chunk_size,
    )


def _load_tokenizer(config: GenerationConfig) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        revision=config.model_revision,
        trust_remote_code=False,
        use_fast=True,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"{config.model} does not expose a Hugging Face chat template")
    return tokenizer


def _load_inputs(config: GenerationConfig) -> tuple[Any, Any]:
    from datasets import load_dataset

    tokenizer = _load_tokenizer(config)
    rows = load_dataset(
        config.dataset,
        revision=config.dataset_revision,
        split=config.split,
    )
    return tokenizer, rows


def _configure_runtime(config: GenerationConfig) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    requested_batch_invariance = "1" if config.batch_invariant else "0"
    existing_batch_invariance = os.environ.get("VLLM_BATCH_INVARIANT")
    if existing_batch_invariance not in {None, requested_batch_invariance}:
        raise RuntimeError(
            "VLLM_BATCH_INVARIANT conflicts with --batch-invariant setting"
        )
    os.environ["VLLM_BATCH_INVARIANT"] = requested_batch_invariance
    requested_multiprocessing = "1" if config.v1_multiprocessing else "0"
    existing_multiprocessing = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
    if existing_multiprocessing not in {None, requested_multiprocessing}:
        raise RuntimeError(
            "VLLM_ENABLE_V1_MULTIPROCESSING conflicts with --v1-multiprocessing setting"
        )
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = requested_multiprocessing


def _generate_prepared(
    args: argparse.Namespace,
    config: GenerationConfig,
    prompts: list[Any],
    filter_counts: dict[str, int],
    *,
    filter_pipeline: tuple[str, ...] | None = None,
    extra_manifest: dict[str, Any] | None = None,
) -> int:
    run_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else config.run_dir(Path(args.output_root))
    )
    shard_counts = [
        len(partition_prompts(prompts, config.num_shards, index))
        for index in range(config.num_shards)
    ]
    manifest = build_manifest(
        config,
        filter_counts,
        shard_counts,
        filter_pipeline=filter_pipeline,
        extra=extra_manifest,
    )
    ensure_manifest(run_dir, manifest)

    shard_prompts = partition_prompts(prompts, config.num_shards, args.shard_index)
    write_prepared_shard(run_dir, args.shard_index, shard_prompts)
    print(
        f"Prepared {len(prompts)} eligible prompts; shard {args.shard_index}/"
        f"{config.num_shards - 1} contains {len(shard_prompts)}."
    )
    print(f"Run directory: {run_dir}")

    pending: list[tuple[int, list[Any], Path]] = []
    for chunk_index, start in enumerate(
        range(0, len(shard_prompts), config.chunk_size)
    ):
        chunk_prompts = shard_prompts[start : start + config.chunk_size]
        path = chunk_path(run_dir, args.shard_index, chunk_index)
        if path.exists():
            validate_chunk(path, chunk_prompts, config)
            print(f"Validated completed {path.relative_to(run_dir)}")
        else:
            pending.append((chunk_index, chunk_prompts, path))

    if pending:
        engine = build_engine(config)
        for chunk_index, chunk_prompts, path in pending:
            requested = len(chunk_prompts) * config.num_rollouts
            print(
                f"Starting chunk {chunk_index + 1}: {len(chunk_prompts)} prompts, "
                f"{requested} rollouts"
            )
            started = time.monotonic()
            records = generate_chunk(
                engine,
                chunk_prompts,
                config,
                use_tqdm=not args.no_progress,
            )
            atomic_write_jsonl(path, records)
            validate_chunk(path, chunk_prompts, config)
            elapsed = time.monotonic() - started
            generated_tokens = sum(int(row["completion_tokens"]) for row in records)
            complete = sum(bool(row["complete"]) for row in records)
            print(
                f"Persisted chunk {chunk_index + 1}: {len(records)} rollouts, "
                f"{complete} complete, {generated_tokens} tokens in {elapsed:.1f}s "
                f"({generated_tokens / elapsed:.1f} tok/s)"
            )
    else:
        print("No generation work remains for this shard.")

    if config.num_shards == 1:
        summary = summarize_run(run_dir)
        print(format_summary(summary))
        if extra_manifest and "fulfillment" in extra_manifest:
            coverage = summarize_fulfillment_coverage(run_dir)
            print()
            print(format_fulfillment_coverage(coverage))
    else:
        print(
            "After every shard completes, run: "
            f"uv run --project evals python evals/generate_traces.py summarize "
            f"--output-dir {run_dir}"
        )
    return 0


def command_generate(args: argparse.Namespace) -> int:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")

    config = _make_config(args)
    _configure_runtime(config)
    tokenizer, rows = _load_inputs(config)
    prompts, filter_counts = prepare_prompts(
        rows,
        tokenizer,
        max_prompt_tokens=config.max_model_len - config.max_output_tokens,
        max_examples=config.max_examples,
        dataset_revision=config.dataset_revision,
        split=config.split,
        allowed_domains=set(config.domains),
    )
    return _generate_prepared(args, config, prompts, filter_counts)


def command_fulfill(args: argparse.Namespace) -> int:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    source_runs = [Path(path) for path in args.source_run]
    start = (
        next_rollout_index(source_runs)
        if args.rollout_index_start is None
        else args.rollout_index_start
    )
    config = _make_config(args, rollout_index_start=start)
    overlap = set(config.rollout_indices) & source_rollout_indices(source_runs)
    if overlap:
        raise ValueError(
            "Fulfillment rollout indices collide with source runs: "
            + ", ".join(map(str, sorted(overlap)))
        )
    _configure_runtime(config)
    tokenizer = _load_tokenizer(config)
    prompts, filter_counts, provenance = prepare_fulfillment_prompts(
        source_runs, tokenizer, config
    )
    return _generate_prepared(
        args,
        config,
        prompts,
        filter_counts,
        filter_pipeline=FULFILLMENT_PIPELINE,
        extra_manifest={"fulfillment": provenance},
    )


def command_summarize(args: argparse.Namespace) -> int:
    run_dir = Path(args.output_dir)
    summary = summarize_run(run_dir)
    print(format_summary(summary))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if "fulfillment" in manifest:
        coverage = summarize_fulfillment_coverage(run_dir)
        print()
        print(format_fulfillment_coverage(coverage))
    return 0


def _add_generation_arguments(
    command: argparse.ArgumentParser,
    *,
    num_rollouts: int,
    rollout_index_start: int | None,
    output_root: str,
) -> None:
    command.add_argument("--dataset", default=DATASET_NAME)
    command.add_argument(
        "--dataset-revision",
        help="revision to resolve (the pinned default, or main for dataset overrides)",
    )
    command.add_argument("--split", default=DATASET_SPLIT)
    command.add_argument("--model", default=MODEL_NAME)
    command.add_argument(
        "--domains",
        nargs="+",
        choices=DOMAIN_NAMES,
        default=list(DOMAIN_NAMES),
        help="dataset domains to generate in this run",
    )
    command.add_argument(
        "--model-revision",
        help="revision to resolve (the pinned default, or main for model overrides)",
    )
    command.add_argument("--num-rollouts", type=_positive_int, default=num_rollouts)
    command.add_argument(
        "--rollout-index-start",
        type=int,
        default=rollout_index_start,
        help="first rollout index (fulfillment derives it from source runs by default)",
    )
    command.add_argument("--max-examples", type=_positive_int)
    command.add_argument("--num-shards", type=_positive_int, default=1)
    command.add_argument("--shard-index", type=int, default=0)
    command.add_argument("--tensor-parallel-size", type=_positive_int, default=1)
    command.add_argument("--max-model-len", type=_positive_int)
    command.add_argument(
        "--max-output-tokens", type=_positive_int, default=MAX_OUTPUT_TOKENS
    )
    command.add_argument(
        "--code-max-output-tokens",
        type=_positive_int,
        default=CODE_MAX_OUTPUT_TOKENS,
    )
    command.add_argument(
        "--default-max-output-tokens",
        type=_positive_int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
    )
    command.add_argument("--base-seed", type=int, default=BASE_SEED)
    command.add_argument("--chunk-size", type=_positive_int, default=128)
    command.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    command.add_argument("--max-num-seqs", type=_positive_int, default=256)
    command.add_argument("--max-num-batched-tokens", type=_positive_int, default=32_768)
    command.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    command.add_argument(
        "--batch-invariant",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "use vLLM batch-invariant kernels; disable for higher throughput "
            "with reproducibility limited to an identical execution layout"
        ),
    )
    command.add_argument(
        "--v1-multiprocessing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "run the vLLM V1 engine in a child process; faster scheduling may "
            "reduce reproducibility across executions"
        ),
    )
    command.add_argument("--output-root", default=output_root)
    command.add_argument(
        "--output-dir", help="exact run directory (overrides --output-root)"
    )
    command.add_argument("--no-progress", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="prepare, shard, generate, score, and persist traces"
    )
    _add_generation_arguments(
        generate,
        num_rollouts=NUM_ROLLOUTS,
        rollout_index_start=0,
        output_root="outputs/trace_generation",
    )
    generate.set_defaults(handler=command_generate)

    fulfill = subparsers.add_parser(
        "fulfill",
        help="generate additional attempts only for prompts with no complete rollout",
    )
    _add_generation_arguments(
        fulfill,
        num_rollouts=FULFILLMENT_NUM_ROLLOUTS,
        rollout_index_start=None,
        output_root="outputs/trace_fulfillment",
    )
    fulfill.add_argument(
        "--source-run",
        action="append",
        required=True,
        help="completed base run directory; repeat for disjoint source phases",
    )
    fulfill.set_defaults(handler=command_fulfill)

    summarize = subparsers.add_parser(
        "summarize", help="validate all chunks, merge Parquet, and report metrics"
    )
    summarize.add_argument("--output-dir", required=True)
    summarize.set_defaults(handler=command_summarize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
