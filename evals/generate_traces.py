#!/usr/bin/env python3
"""Prepare, generate, score, and summarize reproducible reasoning traces."""

from __future__ import annotations

import argparse
import os
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
    DATASET_NAME,
    DATASET_REVISION,
    DATASET_SPLIT,
    MAX_OUTPUT_TOKENS,
    MODEL_NAME,
    MODEL_REVISION,
    GenerationConfig,
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


def _make_config(args: argparse.Namespace) -> GenerationConfig:
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
        base_seed=args.base_seed,
        max_output_tokens=args.max_output_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        max_examples=args.max_examples,
        num_shards=args.num_shards,
        chunk_size=args.chunk_size,
    )


def _load_inputs(config: GenerationConfig) -> tuple[Any, Any]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        revision=config.model_revision,
        trust_remote_code=False,
        use_fast=True,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"{config.model} does not expose a Hugging Face chat template")
    rows = load_dataset(
        config.dataset,
        revision=config.dataset_revision,
        split=config.split,
    )
    return tokenizer, rows


def command_generate(args: argparse.Namespace) -> int:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    existing_batch_invariance = os.environ.get("VLLM_BATCH_INVARIANT")
    if existing_batch_invariance not in {None, "1"}:
        raise RuntimeError(
            "VLLM_BATCH_INVARIANT must be unset or 1 for reproducible generation"
        )
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    existing_multiprocessing = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
    if existing_multiprocessing not in {None, "0"}:
        raise RuntimeError(
            "VLLM_ENABLE_V1_MULTIPROCESSING must be unset or 0 for "
            "reproducible offline generation"
        )
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    config = _make_config(args)
    tokenizer, rows = _load_inputs(config)
    prompts, filter_counts = prepare_prompts(
        rows,
        tokenizer,
        max_prompt_tokens=config.max_model_len - config.max_output_tokens,
        max_examples=config.max_examples,
        dataset_revision=config.dataset_revision,
        split=config.split,
    )
    run_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else config.run_dir(Path(args.output_root))
    )
    manifest = build_manifest(config, filter_counts)
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
            records = generate_chunk(
                engine,
                chunk_prompts,
                config,
                use_tqdm=not args.no_progress,
            )
            atomic_write_jsonl(path, records)
            validate_chunk(path, chunk_prompts, config)
            print(f"Persisted chunk {chunk_index + 1} ({len(records)} rollouts)")
    else:
        print("No generation work remains for this shard.")

    if config.num_shards == 1:
        summary = summarize_run(run_dir)
        print(format_summary(summary))
    else:
        print(
            "After every shard completes, run: "
            f"uv run --project evals python evals/generate_traces.py summarize "
            f"--output-dir {run_dir}"
        )
    return 0


def command_summarize(args: argparse.Namespace) -> int:
    summary = summarize_run(Path(args.output_dir))
    print(format_summary(summary))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="prepare, shard, generate, score, and persist traces"
    )
    generate.add_argument("--dataset", default=DATASET_NAME)
    generate.add_argument(
        "--dataset-revision",
        help="revision to resolve (the pinned default, or main for dataset overrides)",
    )
    generate.add_argument("--split", default=DATASET_SPLIT)
    generate.add_argument("--model", default=MODEL_NAME)
    generate.add_argument(
        "--model-revision",
        help="revision to resolve (the pinned default, or main for model overrides)",
    )
    generate.add_argument("--max-examples", type=_positive_int)
    generate.add_argument("--num-shards", type=_positive_int, default=1)
    generate.add_argument("--shard-index", type=int, default=0)
    generate.add_argument("--tensor-parallel-size", type=_positive_int, default=1)
    generate.add_argument("--max-model-len", type=_positive_int)
    generate.add_argument(
        "--max-output-tokens", type=_positive_int, default=MAX_OUTPUT_TOKENS
    )
    generate.add_argument("--base-seed", type=int, default=BASE_SEED)
    generate.add_argument("--chunk-size", type=_positive_int, default=64)
    generate.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    generate.add_argument("--output-root", default="outputs/trace_generation")
    generate.add_argument(
        "--output-dir", help="exact run directory (overrides --output-root)"
    )
    generate.add_argument("--no-progress", action="store_true")
    generate.set_defaults(handler=command_generate)

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
