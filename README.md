# CoT Compression

A small, pure-PyTorch research project for chain-of-thought compression
experiments. It uses Hydra for configuration, uv for dependency management,
ruff for formatting/linting, and ty for type checking.

The project includes a plain PyTorch SFT path for Qwen3-4B on Dolci reasoning
traces. The layout follows this separation:

- `src/cot_compression/data`: loading and preprocessing data.
- `src/cot_compression/training`: training, logging, and checkpointing.
- `scripts`: thin Hydra entrypoints that call the package code.
- `configs`: hierarchical YAML configuration.
- `tests`: unit and smoke tests.

## Code Style and Philosophy

This project is meant to be read and changed. Prefer short files with one clear
job over large modules that hide many ideas at once. A researcher should be able
to open a file, read it top to bottom, and understand what part of the
experiment it owns.

Keep research logic explicit. The training loop is intentionally written as a
plain function with visible steps: fetch a batch, run the model, compute loss,
update parameters, log metrics, evaluate, and checkpoint. Avoid abstractions
that make those steps hard to find unless they remove real repetition.

Use configuration for experiment choices, not for core program logic. Hydra
YAML files should describe which dataset, method, optimizer, paths, and training
settings are used. Python modules should still contain the actual behavior.

Keep scripts thin. Files in `scripts/` should compose a config and call package
code. Put reusable implementation in `src/cot_compression/`, where it
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

Fine-tune Qwen3-4B on the Dolci reasoning SFT subset with a plain PyTorch loop:

```bash
uv run python scripts/run.py workflow=sft_train
```

For convenience, the default workflow and `sft` alias do the same thing:

```bash
uv run python scripts/run.py
uv run python scripts/run.py sft
```

For a shorter real-model smoke run, override the dataset and training sizes:

```bash
uv run python scripts/run.py sft data.train_size=1000 data.eval_size=100 training.max_steps=10 logging.enabled=false
```

The SFT path loads `Qwen/Qwen3-4B` in full bf16 by default, so real runs need a
GPU with enough memory for full-parameter training. It materializes a
deterministic 600k train plus 2k eval subset from
`allenai/Dolci-Think-SFT-7B` under `data/dolci_think_sft_600k`, preserves the
dataset's existing `<think>...</think>` assistant traces, and trains only on
assistant-message tokens.

Evaluate pretrained Qwen methods on answer-only Dolci dev loss:

```bash
uv run python scripts/run.py eval
```

The default evaluation model is `Qwen/Qwen3-0.6B`. The `base` method leaves the
full trace unchanged, while `random` replaces the contents inside
`<think>...</think>` with deterministic abstract tokens before scoring only the
final answer tokens.
Artifacts are written under the run directory as `summary.json`, `samples.jsonl`,
and optionally `tokens.jsonl`.

Useful overrides:

```bash
uv run python scripts/run.py eval method.model_name=Qwen/Qwen3-4B
uv run python scripts/run.py eval evaluation.max_examples=100
uv run python scripts/run.py eval evaluation.methods.random.abstract_vocab_size=2048 evaluation.methods.random.abstract_length=128
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
uv run python scripts/run.py training.max_steps=50 data.train_size=1000
```

Outputs are written under `outputs/runs/...` by default. Each run contains the
resolved config, printed logs, checkpoints, wandb files, and generated artifacts.
The output and report folders are ignored by Git.
