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
    assert sft_cfg.training.torch_dtype == "bfloat16"
    assert eval_cfg.mode == "evaluate_methods"
    assert eval_cfg.method.model_name == "Qwen/Qwen3-4B"
    assert eval_cfg.evaluation.max_length is None
    assert eval_cfg.evaluation.batch_size == 4
    assert eval_cfg.evaluation.methods.patching.compression_ratio == 2.0
    assert eval_cfg.evaluation.methods.entropy_weighted_mean.temperature == 1.0
    assert eval_cfg.evaluation.entropy_cache_dir is None
    assert eval_32b_cfg.data.source_name == "allenai/Dolci-Think-SFT-32B"
    assert eval_32b_cfg.data.name == "dolci_think_sft_32b_600k"
