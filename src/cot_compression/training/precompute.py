from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from cot_compression.signals import (
    SIGNALS,
    compute_cot_signals,
    load_signal_cache,
    merge_signal_shards,
    save_signal_cache,
    signal_cache_path,
)
from cot_compression.training.evaluate import build_eval_model_and_tokenizer
from cot_compression.training.logging import RunLogger
from cot_compression.training.utils import (
    disable_cudnn_sdpa,
    get_run_dir,
    optional_int,
    resolve_device,
    save_resolved_config,
    set_seed,
)


def _load(cfg):
    # Imported lazily: `evaluate` imports this module's siblings, and a top-level
    # import here closes the cycle.
    from cot_compression.training.evaluate import load_eval_dataset

    return load_eval_dataset(cfg)


def precompute_entropies(cfg: DictConfig) -> Path:
    """Compute CoT-token signals once and cache each for reuse.

    Runs on the plain (unextended) model, so the cached values are canonical and
    shared by every compression/patching config that needs them. Both signals
    (entropy and surprisal) come out of one forward pass and one softmax, so
    ``precompute_signals`` defaults to computing both in a single sweep; each is
    written to its own persistent npz keyed by model. Returns the path of the
    first requested signal's cache.
    """
    run_dir = get_run_dir(cfg)
    save_resolved_config(cfg, run_dir)
    logger = RunLogger(cfg=cfg, run_dir=run_dir)

    try:
        set_seed(seed=int(cfg.evaluation.seed), deterministic=False)
        device = resolve_device(cfg.evaluation.device)
        logger.info(f"Using device: {device}")

        disable_cudnn_sdpa(device)

        signals = tuple(cfg.evaluation.get("precompute_signals", SIGNALS))
        unknown = [name for name in signals if name not in SIGNALS]
        if unknown:
            raise ValueError(
                f"Unknown precompute_signals {unknown}. Expected a subset of {SIGNALS}."
            )

        model, tokenizer = build_eval_model_and_tokenizer(cfg, device)
        dataset = _load(cfg)

        max_examples = cfg.evaluation.max_examples
        limit = (
            len(dataset)
            if max_examples is None
            else min(len(dataset), int(max_examples))
        )
        # Sharding, so a 176k-row corpus is not one 14-hour serial job. Rows are
        # split into contiguous, disjoint, exhaustive ranges; `sample_index` stays
        # the row's index in the FULL split, so shards merge by dict update with no
        # renumbering and no possibility of two shards claiming one row.
        shard, num_shards = int(cfg.evaluation.shard), int(cfg.evaluation.num_shards)
        if not 0 <= shard < num_shards:
            raise ValueError(f"shard {shard} out of range for num_shards {num_shards}.")
        edges = np.linspace(0, limit, num_shards + 1).round().astype(int)
        rows = range(int(edges[shard]), int(edges[shard + 1]))
        logger.info(
            f"Computing CoT signals {list(signals)} for shard {shard + 1}/{num_shards}"
            f": rows [{rows.start}, {rows.stop}) of {limit}"
        )

        values_by_signal = compute_cot_signals(
            model=model,
            tokenizer=tokenizer,
            examples=dataset,
            sample_indices=rows,
            batch_size=int(cfg.evaluation.batch_size),
            max_batch_tokens=optional_int(cfg.evaluation.max_batch_tokens),
            device=device,
            signals=signals,
        )

        cache_dir = cfg.evaluation.entropy_cache_dir
        if cache_dir is None:
            raise ValueError("evaluation.entropy_cache_dir must be set to precompute.")
        cache_paths = []
        for name in signals:
            # Shards land in a subdirectory so a partial set can never be mistaken
            # for a finished cache: `merge_signal_shards` is the only thing that
            # writes the real path, and it refuses an incomplete set.
            base = Path(str(cache_dir))
            cache_path = (
                signal_cache_path(base, str(cfg.method.model_name), name)
                if num_shards == 1
                else signal_cache_path(
                    base / "shards", str(cfg.method.model_name), name
                ).with_suffix(f".shard{shard:02d}of{num_shards:02d}.npz")
            )
            save_signal_cache(cache_path, values_by_signal[name])
            logger.info(
                f"Saved {name} cache: {cache_path} "
                f"({len(values_by_signal[name])} samples)"
            )
            cache_paths.append(cache_path)
    except Exception:
        logger.finish(exit_code=1)
        raise
    else:
        logger.finish()
        return cache_paths[0]


def merge_precomputed_shards(cfg: DictConfig) -> Path:
    """Assemble sharded signal caches into the real ones. CPU-only."""
    cache_dir = cfg.evaluation.entropy_cache_dir
    if cache_dir is None:
        raise ValueError("evaluation.entropy_cache_dir must be set to merge.")
    signals = tuple(cfg.evaluation.get("precompute_signals", SIGNALS))
    num_shards = int(cfg.evaluation.num_shards)
    out = None
    for name in signals:
        out = merge_signal_shards(
            Path(str(cache_dir)), str(cfg.method.model_name), name, num_shards
        )
        loaded = load_signal_cache(out)
        print(f"merged {name}: {len(loaded)} rows -> {out}", flush=True)
    assert out is not None
    return out
