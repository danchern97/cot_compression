# CoT Compression

This project develops embedding initialization schemes for compressing long
chain-of-thought traces into short sequences of learned abstract tokens. The
core question is whether a model can condition on compact abstract-token
representations of reasoning traces while preserving low final-answer loss.

We evaluate methods by replacing or preserving the reasoning trace, masking
scores to answer tokens only, and comparing final answer log probability over a
Dolci dev split. Higher answer log probability is better. The main baselines
are:

- `base`: the original textual `<think>...</think>` chain of thought.
- `random`: fixed-length abstract tokens with randomly initialized embeddings.

The research goal is to learn or design better abstract-token embedding
initializations that maximize answer log probability relative to these baselines.

The repo uses Hydra for configuration, uv for dependency management, ruff for
formatting/linting, and ty for type checking. The main layout is:

- `src/cot_compression/data`: loading and preprocessing data.
- `src/cot_compression/training`: training, logging, and checkpointing.
- `scripts`: thin Hydra entrypoints that call the package code.
- `configs`: hierarchical YAML configuration.
- `tests`: unit and smoke tests.

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
deterministic 600k train, 20k dev/eval, and 100k test subset from
`allenai/Dolci-Think-SFT-7B` under `data/dolci_think_sft_7b_600k`, preserves the
dataset's existing `<think>...</think>` assistant traces, and trains only on
assistant-message tokens.

The 7B Dolci source is the default because its traces fit Qwen3 context windows
without RoPE scaling. The 32B source is available as an explicit override:

```bash
uv run python scripts/run.py eval data=dolci_think_sft_32b_600k
```

Evaluate pretrained Qwen methods on answer-only Dolci dev log probability:

```bash
uv run python scripts/run.py eval
```

The default evaluation model is `Qwen/Qwen3-0.6B`. The `base` method leaves the
full trace unchanged, while `random` replaces the contents inside
`<think>...</think>` with deterministic abstract tokens before scoring only the
final answer tokens. Evaluation uses bf16 by default, fast tokenizers when
available, no tokenizer truncation, and dynamically padded batches.
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

## Agent Notes

Keep reusable implementation in `src/cot_compression/` and keep `scripts/` thin.
Prefer explicit research code over clever abstractions, especially around
training/evaluation loops. Put disposable local experiments in `one-off/`, which
is ignored by Git.
