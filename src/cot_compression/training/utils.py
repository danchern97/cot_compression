from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def get_run_dir(cfg: DictConfig) -> Path:
    run_dir = cfg.paths.get("run_dir")
    if run_dir is not None:
        path = Path(run_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return Path.cwd()


def save_resolved_config(cfg: DictConfig, run_dir: Path) -> None:
    resolved = OmegaConf.to_yaml(cfg, resolve=True)
    (run_dir / "resolved_config.yaml").write_text(resolved, encoding="utf-8")
