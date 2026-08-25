# Reproducible compression-trace generation

Prompt preparation selects all six families in `allenai/Dolci-Think-RL-7B` for
compression learning: math, reference QA, IFEval, code, code-stdio, and general
quality. Selection does not depend on upstream pass rate, source, reference
length, line count, or current grader availability. After family-specific prompt
adaptation, only prompts exceeding the model's input-token budget are rejected.

The generation path saves structural completion status for every family. Math
is graded when a reference is present; other families remain explicitly
ungraded until their task-specific execution or judge policy is selected.

## Setup

```bash
uv sync --project evals --group dev
export HF_HOME=/scratch-shared/$USER/huggingface  # recommended on Snellius
```

The defaults pin the dataset and `Qwen/Qwen3-0.6B` to immutable revisions. They
also use bf16, two rollouts, seed 1337, and Qwen's recommended thinking sampler
(`temperature=0.6`, `top_p=0.95`, `top_k=20`). Output caps are 32,768 tokens for
math, 16,384 for code/code-stdio, and 8,192 for all other families. Only a
natural EOS marks a rollout complete; the pipeline never inserts a closing
think tag or fabricates a final answer.

```bash
uv run --project evals python evals/generate_traces.py generate
```

The throughput-oriented defaults use vLLM's process-separated engine and do not
enable batch-invariant kernels. Seeds, prompt selection, stable modulo shard
membership, and request order are deterministic, but outputs are reproducible
only with an identical execution layout and compatible hardware/software stack.
Pass `--batch-invariant --no-v1-multiprocessing` when independence from batching
is more important than peak throughput.

For a smoke run:

```bash
uv run --project evals python evals/generate_traces.py generate \
  --max-examples 2 \
  --max-output-tokens 256 \
  --code-max-output-tokens 256 \
  --default-max-output-tokens 256 \
  --max-model-len 4096 \
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

## Fulfill prompts without a complete rollout

The `fulfill` subcommand reads a completed source run and selects only prompts
for which none of the existing rollouts has `complete=true`. It does not modify
or copy the source traces. By default it generates four additional attempts and
derives the first safe rollout index from the source manifest; a two-rollout
base run therefore receives `r2` through `r5` with distinct stable seeds.

For code and code-stdio, a 24,576-token retry cap gives the long-tail programs
more room than the 16,384-token base run:

```bash
uv run --project evals python evals/generate_traces.py fulfill \
  --source-run outputs/base/code \
  --domains code code_stdio \
  --num-rollouts 4 \
  --max-output-tokens 24576 \
  --code-max-output-tokens 24576 \
  --default-max-output-tokens 8192 \
  --max-model-len 32768 \
  --output-dir outputs/base/fulfillment_v1/code
```

Fulfillment uses the same `--num-shards`, `--shard-index`, atomic chunks, and
resume validation as normal generation. Once all shards finish, summarize it
normally:

```bash
uv run --project evals python evals/generate_traces.py summarize \
  --output-dir outputs/base/fulfillment_v1/code
```

In addition to the ordinary trace summary, this writes
`coverage_summary.json` and `coverage_by_domain.csv`. They report baseline
prompt coverage, attempted missing prompts, prompts recovered by the extra
rollouts, prompts still requiring another pass, and combined coverage. The
base and fulfillment Parquets remain separate to avoid duplicating large trace
payloads.

For another round, pass both the base and previous fulfillment directories as
repeated `--source-run` arguments. Completion is unioned across their overlapping
prompt histories, and the next non-colliding rollout index is derived
automatically.

## Artifacts

The run directory contains:

- `manifest.json`: resolved revisions, complete sampler and filter config,
  dependency/hardware/Git metadata, filter counts, and shard layout.
- `prepared/shard-*.parquet`: the exact selected prompts for each shard.
- `chunks/shard-*/chunk-*.jsonl`: atomic resumable generation chunks.
- `traces.parquet`: merged long-form rollouts, including errors and truncations.
- `summary.json` and `summary_by_domain.csv`: natural-EOS completion rates for
  every family and pass rates for graded families, plus truncation and length
  diagnostics.
- `coverage_summary.json` and `coverage_by_domain.csv` (fulfillment only):
  before/after prompt-level complete-rollout coverage without copying the base
  traces.

Prepared records retain every non-empty reference/verifier payload once in the
list-valued `ground_truths` field; graders derive the primary answer when needed.
IFEval and general-quality prompts are not given a boxed-answer or `Answer:`
suffix because such text can conflict with their requested output format. Code
and code-stdio prompts receive a canonical single-Python-code-block instruction.

The manifest records the limitations of reproducibility: bitwise-identical
generation requires the pinned software and model revisions and a compatible
GPU/runtime stack. Model overrides must provide a Hugging Face chat template
that supports `enable_thinking=True` and emits `<think>...</think>`.
