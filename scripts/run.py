from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import hydra
from omegaconf import DictConfig

from cot_compression.training import evaluate_methods, train_sft

Workflow = Callable[[DictConfig], object]

WORKFLOWS: dict[str, Workflow] = {
    "evaluate_methods": evaluate_methods,
    "sft_train": train_sft,
}

ALIASES = {
    "eval": "evaluate_methods",
    "sft": "sft_train",
}


def normalize_workflow_args(argv: list[str]) -> None:
    for index, arg in enumerate(argv[1:], start=1):
        if arg in WORKFLOWS or arg in ALIASES:
            argv[index] = f"workflow={ALIASES.get(arg, arg)}"
        elif arg.startswith("mode="):
            mode = arg.removeprefix("mode=")
            argv[index] = f"workflow={ALIASES.get(mode, mode)}"


@hydra.main(version_base=None, config_path="../configs", config_name="run")
def main(cfg: DictConfig) -> None:
    workflow = WORKFLOWS.get(str(cfg.mode))
    if workflow is None:
        choices = ", ".join(sorted(WORKFLOWS))
        raise ValueError(f"Unknown mode {cfg.mode!r}. Expected one of: {choices}.")

    result = workflow(cfg)
    if isinstance(result, Path):
        print(result)


if __name__ == "__main__":
    normalize_workflow_args(sys.argv)
    main()
