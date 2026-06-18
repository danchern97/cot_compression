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

    assert sft_cfg.mode == "sft_train"
    assert sft_cfg.method.model_name == "Qwen/Qwen3-4B"
    assert sft_cfg.training.torch_dtype == "bfloat16"
