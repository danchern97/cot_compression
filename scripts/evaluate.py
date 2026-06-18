from __future__ import annotations

import hydra
from omegaconf import DictConfig

from research_project_template.training import evaluate


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
