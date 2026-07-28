from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from cot_compression.data.dolci import load_dolci_sft_data
from cot_compression.signals import (
    SIGNALS,
    compute_cot_signals,
    save_signal_cache,
    signal_cache_path,
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

        signals = tuple(cfg.evaluation.get("precompute_signals", SIGNALS))
        unknown = [name for name in signals if name not in SIGNALS]
        if unknown:
            raise ValueError(
                f"Unknown precompute_signals {unknown}. Expected a subset of {SIGNALS}."
            )

        model, tokenizer = build_eval_model_and_tokenizer(cfg, device)
        dataset = load_dolci_sft_data(cfg).eval

        max_examples = cfg.evaluation.max_examples
        limit = (
            len(dataset)
            if max_examples is None
            else min(len(dataset), int(max_examples))
        )
        logger.info(f"Computing CoT signals {list(signals)} for {limit} examples")

        values_by_signal = compute_cot_signals(
            model=model,
            tokenizer=tokenizer,
            examples=dataset,
            sample_indices=range(limit),
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
            cache_path = signal_cache_path(
                Path(str(cache_dir)), str(cfg.method.model_name), name
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
