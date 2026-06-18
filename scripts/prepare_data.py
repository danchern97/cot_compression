from __future__ import annotations

import hydra
from omegaconf import DictConfig

from research_project_template.data import prepare_data


@hydra.main(version_base=None, config_path="../configs", config_name="prepare_data")
def main(cfg: DictConfig) -> None:
    prepared_dir = prepare_data(cfg)
    print(f"Prepared data in {prepared_dir}")


if __name__ == "__main__":
    main()
