from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import wandb
from omegaconf import DictConfig, OmegaConf


class RunLogger:
    def __init__(self, cfg: DictConfig, run_dir: Path) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.logger = self._build_text_logger(run_dir / cfg.logging.log_file_name)
        self.wandb_run = None

        if cfg.logging.enabled:
            config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
            self.wandb_run = wandb.init(
                project=cfg.logging.project,
                entity=cfg.logging.entity,
                group=cfg.logging.group,
                name=cfg.logging.name,
                tags=list(cfg.logging.tags),
                dir=str(run_dir),
                mode=cfg.logging.mode,
                config=config,
            )

    def _build_text_logger(self, log_path: Path) -> logging.Logger:
        logger = logging.getLogger("cot_compression")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.propagate = False
        formatter = logging.Formatter(
            "%(levelname)s | %(asctime)s | %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        return logger

    def info(self, message: str) -> None:
        self.logger.info(message)

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        text = ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        self.info(f"step={step} | {text}")
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)

    def log_artifact(self, path: Path) -> None:
        self.info(f"Saved artifact: {path}")
        if self.wandb_run is not None:
            artifact = wandb.Artifact(path.stem, type="evaluation")
            artifact.add_file(str(path))
            self.wandb_run.log_artifact(artifact)

    def finish(self) -> None:
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers.clear()
        if self.wandb_run is not None:
            self.wandb_run.finish()
