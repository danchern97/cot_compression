#!/bin/bash
# Downstream benchmarks (MATH-500, AIME'25, GPQA-Diamond, HotpotQA) for one model.
# Run from the repo root; lives in evals/ because scripts/ is gitignored.
#
#   sbatch evals/run_benchmarks.sh                            # raw Qwen/Qwen3-4B
#   sbatch evals/run_benchmarks.sh <model-or-path> <name>
#   sbatch evals/run_benchmarks.sh <model> <name> <limit>     # smoke test
#   sbatch evals/run_benchmarks.sh Qwen/Qwen3-0.6B qwen3-0.6b-raw "" 1 cotc_aime25_16
#   sbatch evals/run_benchmarks.sh <ckpt> <name> "" 1 "" nothink   # thinking off
#
# Args: <model> <name> <limit> <TP> <aime-task> <mode>
#
# One array task per benchmark, one GPU each, data_parallel_size=1.
# NOT a single 4-GPU data_parallel job: lm-eval builds every ray actor from the
# same model_args, so all four vLLM engines share one seed (vllm_causallms.py
# :135) while requests are handed out by uniform interleave -- the 64 repeats of
# an AIME question land at matching batch positions in identically-seeded
# engines, which can silently collapse avg@64 into far fewer distinct samples.
# One engine per benchmark makes the repeats independent by construction, costs
# the same GPU-hours, and isolates a failure (e.g. GPQA auth) to one benchmark.
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
# Generous because AIME'25 (1,920 generations of up to 38,912 tokens) is the long
# pole: measured ~1.5k output tok/s on one H100 at low concurrency, so ~2-5 h.
# Unused walltime is not billed, and each array task frees its GPU when it ends.
#SBATCH --time=10:00:00
#SBATCH --array=0-3
#SBATCH --job-name=bench
#SBATCH --output=results/benchmarks/%x-%A_%a.out
set -euo pipefail

MODEL=${1:-Qwen/Qwen3-4B}
NAME=${2:-qwen3-4b-raw}
# Positional, not an env var: `LIMIT=2 sbatch ...` did not reach lm_eval and the
# "smoke test" silently ran the full benchmark.
LIMIT=${3:-}
# Tensor parallelism. AIME'25 needs it: its traces run 15-25k tokens, so at
# TP=1 only ~21 of the 546k-token KV cache's worth fit concurrently and decode
# throughput collapses to ~1.5k tok/s (measured 4.4 completions/min -> ~7.3 h,
# uncomfortably close to the walltime). TP=4 quadruples KV capacity and compute.
# Still ONE engine with one sampler RNG, so the 64 repeats stay independent --
# which data_parallel_size=4 would not guarantee.
#   sbatch --array=1 --gpus-per-node=4 --cpus-per-task=64 \
#     evals/run_benchmarks.sh <model> <name> "" 4
TP=${4:-1}
# Which AIME variant array index 1 runs. cotc_aime25 is avg@64 (the Qwen3 report
# protocol, and what the 4B baseline used); cotc_aime25_16 is the avg@16 variant.
# Positional rather than an env var for the same reason as LIMIT above.
AIME_TASK=${5:-cotc_aime25}
# think   -> enable_thinking=True,  the cotc_* tasks, Qwen3 thinking sampler.
# nothink -> enable_thinking=False, the cotc_nothink_* tasks, which carry Qwen3's
# non-thinking sampler (t=0.7/top-p=0.8) and the 500-question HotpotQA subset.
# This is the "SFT (CoT)" setting of arXiv 2604.22709: "We evaluate models without
# their 'thinking mode' with standard CoT prompting to ensure controlled comparison."
MODE=${6:-think}

case $MODE in
  think)
    THINKING=True
    TASKS=(cotc_math500 "$AIME_TASK" cotc_gpqa_diamond cotc_hotpotqa)
    ;;
  nothink)
    THINKING=False
    # AIME_TASK is ignored here: the repeat count is baked into the task file, and
    # silently accepting a cotc_* name would run the thinking-mode sampler.
    TASKS=(cotc_nothink_math500 cotc_nothink_aime25_16 cotc_nothink_gpqa_diamond cotc_nothink_hotpotqa500)
    ;;
  *)
    echo "MODE must be 'think' or 'nothink', got '$MODE'" >&2
    exit 1
    ;;
esac
TASK=${TASKS[${SLURM_ARRAY_TASK_ID:-0}]}

[ -d evals/tasks ] || { echo "run from the repo root" >&2; exit 1; }

export HF_HOME=/scratch-shared/$USER/huggingface
export TOKENIZERS_PARALLELISM=false
# The compute nodes have no CUDA toolkit (no nvcc, no /usr/local/cuda), so every
# runtime JIT path has to be off. top_k=20 would otherwise route sampling through
# FlashInfer, which compiles its kernel with nvcc on first use and dies. vLLM's
# native apply_top_k_top_p_pytorch is exact, so the sampling distribution is
# unchanged; only the sampling step is slower, which is negligible next to the
# forward pass. DeepGEMM is likewise nvcc-dependent and unused for bf16.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_DEEP_GEMM=0

OUT=results/benchmarks/$NAME
mkdir -p "$OUT"

# think_end_token makes lm-eval strip the <think> block before answer extraction
# and send only EOS to vLLM as a stop string (task stops would otherwise fire
# inside the reasoning trace). It is kept in nothink mode too: lm-eval only
# *requires* it when enable_thinking=True (vllm_causallms.py:182) but applies the
# split whenever it is set (:615), and this checkpoint was SFT'd exclusively on
# think-traces, so it can reopen a <think> block even with thinking off. Stripping
# it keeps grading robust; token_stats.py reports how often it happens.
# --seed also seeds the vLLM engine, so a rerun reproduces the same samples.
# dtype is explicit because the SFT checkpoint's config.json reports float32
# while its safetensors are bf16.
# max_model_len is the model's max_position_embeddings; AIME generates up to
# 38,912 tokens and must still fit its prompt.
# --batch_size auto is NOT optional. lm-eval's CLI default is 1, which makes it
# hand vLLM a single request at a time -- continuous batching off, the GPU ~idle.
# Measured at batch_size=1: AIME 236 s/request, i.e. ~126 h for the task. "auto"
# submits every request at once and lets vLLM schedule them.
uv run --project evals lm_eval \
  --model vllm \
  --model_args "pretrained=$MODEL,dtype=bfloat16,data_parallel_size=1,tensor_parallel_size=$TP,gpu_memory_utilization=0.90,max_model_len=40960,enable_thinking=$THINKING,think_end_token=</think>" \
  --tasks "$TASK" \
  --include_path evals/tasks \
  --apply_chat_template \
  --seed 1234 \
  --batch_size auto \
  --log_samples \
  --output_path "$OUT" \
  ${LIMIT:+--limit "$LIMIT"} \
  2>&1 | tee "$OUT/$TASK.log"
