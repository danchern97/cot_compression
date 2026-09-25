from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, cast

import wandb
from omegaconf import DictConfig, OmegaConf


def _wandb_id(cfg: DictConfig) -> str:
    """Stable W&B run id derived from the run name.

    Must be deterministic across requeues (that is the whole point) and unique
    per project. run_name already encodes tag + hyperparameters and is what
    paths.run_dir is keyed on, so two runs sharing an id would also share a
    checkpoint directory -- i.e. they are the same run by construction.
    """
    name = str(cfg.get("run_name", None) or cfg.logging.get("name", None) or "run")
    safe = re.sub(r"[^A-Za-z0-9_.-]", "-", name)
    if len(safe) <= 63:
        return safe
    # W&B ids cap at 64 characters. Truncating alone collapses every run name that
    # differs only past character 63 onto ONE W&B run -- with the encoder's
    # run_name that is the masks, the position encoding and the lr -- and
    # resume="allow" then appends one arm's history to another's. A readable
    # prefix plus a hash of the WHOLE name keeps ids unique; names that already
    # fit are unchanged, so existing runs keep their ids.
    return f"{safe[:54]}-{hashlib.sha1(safe.encode()).hexdigest()[:8]}"


class RunLogger:
    """Text + W&B logging for one run.

    Under DDP every rank executes the same Hydra job in the same run dir, so only
    rank 0 opens the log file or a W&B run; other ranks keep a stderr-only logger
    so their warnings are still visible without four processes interleaving into
    one file or opening four W&B runs per job.
    """

    def __init__(self, cfg: DictConfig, run_dir: Path, is_main: bool = True) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.is_main = is_main
        log_path = run_dir / cfg.logging.log_file_name if is_main else None
        self.logger = self._build_text_logger(log_path)
        self.wandb_run = None

        if cfg.logging.enabled and is_main:
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
                # A deterministic id plus resume="allow" is what makes a requeued
                # job continue the SAME W&B run. The full SFT needs ~61 h inside
                # 48 h windows, so it requeues at least once; with an
                # auto-generated id each window would appear as a separate run and
                # the loss curve would arrive in disconnected pieces. Steps stay
                # monotonic across the seam because global_step is restored from
                # the checkpoint before any logging happens.
                id=_wandb_id(cfg),
                resume="allow",
            )

    def _build_text_logger(self, log_path: Path | None) -> logging.Logger:
        logger = logging.getLogger("cot_compression")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.propagate = False
        formatter = logging.Formatter(
            "%(levelname)s | %(asctime)s | %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )

        if log_path is not None:
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
        """Record an artifact locally; upload to W&B only if explicitly allowed.

        Uploads are opt-in and size-capped because the callers here include
        tokens.jsonl, which runs 0.6-4.1 GB per eval cell -- a single 27-cell
        sweep would push ~80 GB into a 200 GB W&B quota. The cap is enforced even
        when uploads are enabled, so turning the flag on cannot silently ship a
        multi-GB file.
        """
        self.info(f"Saved artifact: {path}")
        if self.wandb_run is None or not self.cfg.logging.get("log_artifacts", False):
            return

        size_mb = path.stat().st_size / (1 << 20)
        limit_mb = float(self.cfg.logging.get("max_artifact_mb", 32))
        if size_mb > limit_mb:
            self.info(
                f"Skipping W&B upload of {path.name}: "
                f"{size_mb:.1f} MB exceeds logging.max_artifact_mb={limit_mb:g}"
            )
            return

        artifact = wandb.Artifact(path.stem, type="evaluation")
        artifact.add_file(str(path))
        self.wandb_run.log_artifact(artifact)

    def finish(self, exit_code: int = 0) -> None:
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers.clear()
        if self.wandb_run is not None:
            self.wandb_run.finish(exit_code=exit_code)
