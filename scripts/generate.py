from __future__ import annotations

import hydra
from omegaconf import DictConfig

from research_project_template.training import generate


@hydra.main(version_base=None, config_path="../configs", config_name="generate")
def main(cfg: DictConfig) -> None:
    generate(cfg)


if __name__ == "__main__":
    main()
