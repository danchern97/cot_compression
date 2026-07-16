from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig

from cot_compression.data.dolci import load_dolci_sft_data
from cot_compression.entropy import (
    compute_cot_entropies,
    entropy_cache_path,
    save_entropy_cache,
)
from cot_compression.training.evaluate import build_eval_model_and_tokenizer
from cot_compression.training.logging import RunLogger
from cot_compression.training.utils import (
    get_run_dir,
    optional_int,
    resolve_device,
    save_resolved_config,
    set_seed,
)


def precompute_entropies(cfg: DictConfig) -> Path:
    """Compute CoT-token entropies once and cache them for reuse.

    Runs on the plain (unextended) model, so the cached entropies are canonical
    and shared by every compression/patching config that needs them. Writes a
    persistent npz keyed by model.
    """
    run_dir = get_run_dir(cfg)
    save_resolved_config(cfg, run_dir)
    logger = RunLogger(cfg=cfg, run_dir=run_dir)

    try:
        set_seed(seed=int(cfg.evaluation.seed), deterministic=False)
        device = resolve_device(cfg.evaluation.device)
        logger.info(f"Using device: {device}")

        model, tokenizer = build_eval_model_and_tokenizer(cfg, device)
        dataset = load_dolci_sft_data(cfg).eval

        max_examples = cfg.evaluation.max_examples
        limit = (
            len(dataset)
            if max_examples is None
            else min(len(dataset), int(max_examples))
        )
        logger.info(f"Computing CoT entropies for {limit} examples")

        entropies = compute_cot_entropies(
            model=model,
            tokenizer=tokenizer,
            examples=dataset,
            sample_indices=range(limit),
            batch_size=int(cfg.evaluation.batch_size),
            max_batch_tokens=optional_int(cfg.evaluation.max_batch_tokens),
            device=device,
        )

        cache_dir = cfg.evaluation.entropy_cache_dir
        if cache_dir is None:
            raise ValueError("evaluation.entropy_cache_dir must be set to precompute.")
        cache_path = entropy_cache_path(
            Path(str(cache_dir)), str(cfg.method.model_name)
        )
        save_entropy_cache(cache_path, entropies)
        logger.info(f"Saved entropy cache: {cache_path} ({len(entropies)} samples)")
    except Exception:
        logger.finish(exit_code=1)
        raise
    else:
        logger.finish()
        return cache_path
