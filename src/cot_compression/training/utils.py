from __future__ import annotations

import logging
import os
import random
import sys
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


def disable_cudnn_sdpa(device: torch.device) -> bool:
    """Take cuDNN's fused attention out of SDPA's backend choice. Returns whether.

    cuDNN's fused MHA has produced two distinct, reproducible failures on this
    workload's masked attention:

    * a hard abort in the FORWARD -- "Expected mha_graph.execute(...).is_good() to
      be true" -- which killed precompute shard 1 of job 26232131 at 23 minutes
      while three identical shards ran to completion;
    * silent NaN out of the BACKWARD --
      "Function 'ScaledDotProductCudnnAttentionBackward0' returned nan values" --
      confirmed by anomaly detection (job 26376052) on the encoder's
      cross-attention, with gradient norms flat at ~0.05 for the ten steps before
      it. That NaN destroyed both arms of the encoder campaign at steps ~780 and
      ~995, and clipping cannot contain it: clip_coef = max_norm/(NaN + 1e-6) is
      NaN, so one NaN gradient poisons every parameter.

    Neither is our bug and neither is recoverable in-run. Flash, mem-efficient and
    math backends are all available on H100, so SDPA falls through to flash -- the
    backend PyTorch used before cuDNN attention became a default -- at a cost of a
    few percent, not a fallback to the math kernel.
    """
    if device.type != "cuda":
        return False
    torch.backends.cuda.enable_cudnn_sdp(False)
    return True


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


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


def exit_failed_rank(rank: int, code: int = 1) -> None:
    """End a rank that raised in a multi-process run NOW, without a graceful teardown.

    A graceful exit cannot finish once one rank has left the step: the others are
    blocked inside a collective, so `destroy_process_group()` -- and NCCL's destructors
    at interpreter shutdown -- wait on them, and the raising rank never exits. srun's
    `SLURM_KILL_BAD_EXIT` only acts on an exit, so the job then sits until the
    30-minute NCCL watchdog. Measured: an injected fault on rank 3 (job 26697398) left
    the whole node idle after the traceback was already logged.

    `os._exit` skips unwinding, atexit handlers and destructors. Call it only after the
    traceback is logged; checkpoint writes are crash-safe by construction (`LATEST` is
    moved last), so nothing is left half-written that resume would read.
    """
    logging.getLogger(__name__).error("rank %d exiting after failure", rank)
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
