from __future__ import annotations

import hydra
from omegaconf import DictConfig

from research_project_template.training import train


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
