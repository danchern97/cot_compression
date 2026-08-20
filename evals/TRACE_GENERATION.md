# Reproducible RL trace generation

This pipeline filters `allenai/Dolci-Think-RL-7B` to verifiable math and compact
short-answer QA prompts, generates four independent thinking rollouts per
prompt, grades them, and saves every successful or failed rollout. It is
isolated from the training environment and does not change the existing SFT or
compression-evaluation datasets.

## Setup

```bash
uv sync --project evals --group dev
export HF_HOME=/scratch-shared/$USER/huggingface  # recommended on Snellius
```

The defaults pin the dataset and `Qwen/Qwen3-0.6B` to immutable revisions. They
also use bf16, four rollouts, seed 1337, Qwen's recommended thinking sampler,
and a 32,768-token output allowance. A full run is:

```bash
uv run --project evals python evals/generate_traces.py generate
```

The CLI enables vLLM batch-invariant kernels and its deterministic offline
scheduler (`VLLM_ENABLE_V1_MULTIPROCESSING=0`). Per-request seeds then make a
rollout independent of batch composition, shard order, and fresh process
scheduling on the same hardware and pinned GPU/software stack. These settings
can reduce peak throughput compared with vLLM's defaults.

For a smoke run:

```bash
uv run --project evals python evals/generate_traces.py generate \
  --max-examples 2 \
  --max-output-tokens 256 \
  --output-dir outputs/trace-generation-smoke
```

For multi-GPU tensor parallelism, pass `--tensor-parallel-size N`. To distribute
the prompt set across independent jobs, every job must use identical arguments
apart from `--shard-index`:

```bash
for shard in 0 1 2 3; do
  uv run --project evals python evals/generate_traces.py generate \
    --num-shards 4 --shard-index "$shard" \
    --output-dir outputs/trace-generation-qwen &
done
wait

uv run --project evals python evals/generate_traces.py summarize \
  --output-dir outputs/trace-generation-qwen
```

Each chunk is written to a temporary file and atomically renamed. Re-running a
command validates and skips complete chunks; it fails rather than mixing data
when the manifest, filters, rollout IDs, or configuration differ.

## Artifacts

The run directory contains:

- `manifest.json`: resolved revisions, complete sampler and filter config,
  dependency/hardware/Git metadata, filter counts, and shard layout.
- `prepared/shard-*.parquet`: the exact selected prompts for each shard.
- `chunks/shard-*/chunk-*.jsonl`: atomic resumable generation chunks.
- `traces.parquet`: merged long-form rollouts, including errors and truncations.
- `summary.json` and `summary_by_domain.csv`: global, math, and QA rollout pass
  rates and prompt pass@4, plus extraction, truncation, and length diagnostics.

The manifest records the limitations of reproducibility: bitwise-identical
generation requires the pinned software and model revisions and a compatible
GPU/runtime stack. Model overrides must provide a Hugging Face chat template
that supports `enable_thinking=True` and emits `<think>...</think>`.
