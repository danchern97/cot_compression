from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


def test_hydra_configs_compose() -> None:
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        sft_cfg = compose(
            config_name="run",
            overrides=["logging.enabled=false"],
        )
        eval_cfg = compose(
            config_name="run",
            overrides=["workflow=evaluate_methods", "logging.enabled=false"],
        )
        eval_32b_cfg = compose(
            config_name="run",
            overrides=[
                "workflow=evaluate_methods",
                "data=dolci_think_sft_32b_600k",
                "logging.enabled=false",
            ],
        )

    assert sft_cfg.mode == "sft_train"
    assert sft_cfg.method.model_name == "Qwen/Qwen3-4B"
    assert sft_cfg.data.source_name == "allenai/Dolci-Think-SFT-7B"
    assert sft_cfg.data.name == "dolci_think_sft_7b_600k"
    assert sft_cfg.data.eval_size == 20000
    assert sft_cfg.data.test_size == 100000
    # fp32 master weights: bf16 params would put the AdamW moments in bf16 too and
    # updates at lr 1e-5 would round away.
    assert sft_cfg.training.torch_dtype == "float32"
    assert sft_cfg.training.autocast_dtype == "bfloat16"
    assert sft_cfg.training.attn_implementation == "sdpa"
    # Rows are dropped, never truncated, at the full model context.
    assert sft_cfg.training.max_length == 32768
    assert sft_cfg.training.max_batch_tokens >= sft_cfg.training.max_length
    assert sft_cfg.training.target_global_batch == 32
    assert sft_cfg.training.resume_from_checkpoint == "auto"
    assert sft_cfg.training.max_steps is None
    # Heavy artifacts on scratch, light ones on HOME (200 GiB quota).
    assert sft_cfg.paths.output_dir.startswith("/scratch-shared/")
    assert sft_cfg.paths.tokenized_dir.startswith("/scratch-shared/")
    assert sft_cfg.paths.data_dir.startswith("/scratch-shared/")
    # W&B: one project, and artifact uploads off by default (tokens.jsonl is GBs).
    assert sft_cfg.logging.entity == "cot-compression"
    assert sft_cfg.logging.project == "cot-compression-qwen3"
    assert sft_cfg.logging.log_artifacts is False
    assert eval_cfg.mode == "evaluate_methods"
    assert eval_cfg.method.model_name == "Qwen/Qwen3-4B"
    assert eval_cfg.evaluation.max_length is None
    assert eval_cfg.evaluation.batch_size == 64
    assert eval_cfg.evaluation.num_workers == 14
    assert eval_cfg.evaluation.prefetch_factor == 4
    assert eval_cfg.evaluation.methods.patching.compression_ratio == 2.0
    assert eval_cfg.evaluation.methods.entropy_weighted_mean.temperature == 1.0
    assert eval_cfg.evaluation.methods.surprisal_weighted_mean.temperature == 1.0
    assert list(eval_cfg.evaluation.precompute_signals) == ["entropy", "surprisal"]
    assert eval_cfg.evaluation.entropy_cache_dir is None
    assert eval_cfg.evaluation.scratch_dir is None
    assert eval_32b_cfg.data.source_name == "allenai/Dolci-Think-SFT-32B"
    assert eval_32b_cfg.data.name == "dolci_think_sft_32b_600k"
