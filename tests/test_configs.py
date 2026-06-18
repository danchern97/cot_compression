from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


def test_hydra_configs_compose() -> None:
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        train_cfg = compose(config_name="train", overrides=["logging.enabled=false"])
        eval_cfg = compose(config_name="evaluate")
        generate_cfg = compose(config_name="generate")

    assert train_cfg.data.name == "tiny_shakespeare_char"
    assert eval_cfg.mode == "evaluate"
    assert generate_cfg.prompt == "First Citizen:"
