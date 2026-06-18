# Research Project Template

A small, pure-PyTorch research template for student projects. It uses Hydra for
configuration, uv for dependency management, ruff for formatting/linting, and ty
for type checking.

The runnable example is a tiny character-level GPT language model inspired by
nanoGPT. The project layout follows the same separation you should keep in your
own work:

- `src/research_project_template/data`: loading and preprocessing data.
- `src/research_project_template/methods`: models, methods, and generation.
- `src/research_project_template/training`: training, evaluation, logging, and
  checkpointing.
- `scripts`: thin Hydra entrypoints that call the package code.
- `configs`: hierarchical YAML configuration.
- `tests`: unit and smoke tests.

## Code Style and Philosophy

This template is meant to be read, copied, and changed. Prefer short files with
one clear job over large modules that hide many ideas at once. A student should
be able to open a file, read it top to bottom, and understand what part of the
experiment it owns.

Keep research logic explicit. The training loop is intentionally written as a
plain function with visible steps: fetch a batch, run the model, compute loss,
update parameters, log metrics, evaluate, and checkpoint. Avoid abstractions
that make those steps hard to find unless they remove real repetition.

Use configuration for experiment choices, not for core program logic. Hydra
YAML files should describe which dataset, method, optimizer, paths, and training
settings are used. Python modules should still contain the actual behavior.

Keep scripts thin. Files in `scripts/` should compose a config and call package
code. Put reusable implementation in `src/research_project_template/`, where it
can be tested and imported by other scripts.

Write tests for behavior, not implementation trivia. Good tests check that data
batches are shaped correctly, model losses are finite, checkpoints can be
loaded, and small end-to-end runs work. They should stay fast enough that
students run them often.

## Setup

```bash
uv sync
```

## Common Commands

Prepare the example corpus:

```bash
uv run python scripts/prepare_data.py
```

Train a tiny character GPT:

```bash
uv run python scripts/train.py
```

Evaluate a checkpoint:

```bash
uv run python scripts/evaluate.py checkpoint.path=/path/to/checkpoint.pt
```

Generate text from a checkpoint:

```bash
uv run python scripts/generate.py checkpoint.path=/path/to/checkpoint.pt
```

Run checks:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

## Configuration

Hydra configs live under `configs/`. Override values from the command line:

```bash
uv run python scripts/train.py training.max_steps=50 data.block_size=64
```

Outputs are written under `outputs/runs/...` by default. Each run contains the
resolved config, printed logs, checkpoints, wandb files, and generated artifacts.
The output and report folders are ignored by Git.
