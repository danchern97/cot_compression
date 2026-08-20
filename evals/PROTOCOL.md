# Benchmark protocol

What `evals/` runs, why each choice was made, and where every deviation from a
published protocol lies. This is the reference for the evaluation section of the
paper.

Harness: `lm-evaluation-harness` 0.4.12 + vLLM 0.26.0, in `evals/` as a separate
uv project (vLLM pins `torch==2.11.0`; the training env is on 2.12.1).
Task definitions are in `evals/tasks/`; nothing upstream in lm-eval is reused
(see *Why not the stock lm-eval tasks* below).

## Results summary (all runs, 2026-08-15)

Five runs, one checkpoint under two inference modes plus controls. `no-think` =
`enable_thinking=False` + standard CoT prompting, Qwen3's non-thinking sampler
(t=0.7/top-p=0.8/top-k=20); `think` = the original protocol (t=0.6/top-p=0.95).
SFT = Qwen3-4B finetuned on Dolci-Think-SFT-7B (600k, 1 epoch, eval loss 0.8442).

### 1. Accuracy

| Model / mode | MATH-500 | AIME'25 | GPQA-D | HotpotQA EM | HotpotQA F1 |
|---|---|---|---|---|---|
| *Qwen3 report* — 4B thinking | 97.0 | 65.6 | 55.9 | — | — |
| **raw 4B, think** | 96.6 ± 0.7 | 65.7 ± 7.0 ᵃ | 54.1 ± 2.9 | 59.9 ± 0.6 | 76.2 ± 0.4 |
| **SFT 4B, think** | 95.0 ± 0.8 | 56.5 ± 7.1 | 40.7 ± 2.6 | 58.2 ± 0.6 | 73.6 ± 0.4 |
| *Qwen3 report* — 4B non-thinking | 84.8 | 19.1 | 41.7 | — | — |
| **raw 4B, no-think** | 84.4 ± 1.4 | 20.8 ± 6.6 | 45.1 ± 2.6 | 54.6 ± 2.2 ᵇ | 69.2 ± 1.8 ᵇ |
| **SFT 4B, no-think** | 83.6 ± 1.4 | 25.2 ± 6.4 | 33.6 ± 2.2 | 50.6 ± 2.2 ᵇ | 64.7 ± 1.9 ᵇ |
| *paper* Baseline (raw 4B) | 83.2 | — | — | — | 46.7 ᶜ |
| *paper* **SFT (CoT)** (4B) | **88.4** | — | — | — | 50.3 ᶜ |
| *paper* SFT + RL (4B) | 91.2 | — | — | — | 53.4 ᶜ |
| *paper* Abstract-CoT (4B) | 89.8 | — | — | — | 53.8 ᶜ |
| *Qwen3 report* — 0.6B thinking | 77.6 | 15.1 | 27.9 | — | — |
| **raw 0.6B, think** | 74.6 ± 1.6 | 18.1 ± 5.0 | 25.4 ± 1.8 | 17.3 ± 0.4 | 22.6 ± 0.5 |

ᵃ avg@64; every other AIME cell is avg@16.
ᵇ 500-question subset (±2 pp) vs the full 7,405-row split (±0.6 pp) — not comparable to the `think` rows above.
ᶜ paper's HotpotQA prompt is unknown and ours is homegrown; see *Not comparable* below.

The paper's AIME'25/GPQA-Diamond table is **Qwen3-8B only** — no 4B reference exists:
Baseline 23.3 / 44.9, SFT (CoT) 23.3 / 45.5.

### 2. Generated tokens and truncation

Only the `no-think` rows are comparable to the paper. In `think` mode lm-eval splits the
reply on `</think>` before logging, so the counts measure the *visible answer only* and omit
the reasoning trace that the paper's "reasoning and response tokens" counts. `<think>` appears
in **zero** of the 9,900 non-thinking generations, so those counts are exact.

| Run | MATH-500 | AIME'25 | GPQA-D | HotpotQA |
|---|---|---|---|---|
| *paper* Baseline (4B) | 1,087 | — ᵈ | — ᵈ | 482 |
| *paper* SFT (CoT) (4B) | 1,396 | — ᵈ | — ᵈ | 612 |
| *paper* SFT + RL (4B) | 1,523 | — ᵈ | — ᵈ | 683 |
| *paper* Abstract-CoT (4B) | 141 | — ᵈ | — ᵈ | 169 |
| **raw 4B, no-think** — mean | 1,075 | 6,187 | 1,292 | 66 |
| **SFT 4B, no-think** — mean | 5,125 | 27,883 | 7,604 | 296 |
| raw — median | 560 | 2,368 | 711 | 57 |
| SFT — median | 558 | **38,912** | 824 | 86 |
| raw — terminated-only mean | 900 | 2,884 | 871 | 66 |
| SFT — terminated-only mean | 866 | 2,653 | 868 | 100 |

ᵈ paper reports AIME/GPQA tokens for Qwen3-8B only: Baseline 3,981 / 767, SFT (CoT) 4,627 / 1,106.

**Truncation rate** (generation hit `max_gen_toks`: 38,912 on AIME, 32,768 elsewhere):

| Run | MATH-500 | AIME'25 | GPQA-D | HotpotQA |
|---|---|---|---|---|
| raw 4B, no-think | 0.55% | **9.17%** | 1.32% | 0.00% |
| SFT 4B, no-think | 13.35% | **69.58%** | 21.12% | 0.60% |
| amplification | 24x | 7.6x | 16x | — |

The SFT median on AIME *is* the cap: more than half of its generations never finished.
Raw's own 9.17% shows non-termination is partly intrinsic to hard math in non-thinking
mode, not purely an SFT artifact.

**The finetune did not make reasoning longer.** On generations that terminate, its traces are
the same length as raw's or shorter — 866 vs 900 on MATH-500, 2,653 vs 2,884 on AIME'25, 868 vs
871 on GPQA-D (only HotpotQA rises, 100 vs 66, which is the appended CoT trigger doing its job).
The entire token blow-up is a minority of generations that never stop, not a model that reasons
at greater length. Medians say the same thing: 558 vs 560 on MATH-500.

### 3. Accuracy with truncation removed

Every repeat regraded individually, split on whether it terminated. A truncated generation
scores ~0 by construction (it never emits a final `\boxed{}` or `Answer:` line).

| Benchmark | raw term. | SFT term. | Δ terminated | Δ headline | SFT non-trunc share |
|---|---|---|---|---|---|
| MATH-500 | 84.9 | **94.5** | +9.6 | −0.8 | 86.7% |
| AIME'25 | 22.9 | 70.5 | +47.6 ⚠ | +4.4 | **30.4%** |
| GPQA-Diamond | 45.5 | 42.4 | −3.1 | −11.5 | 78.9% |
| HotpotQA (F1) | 69.2 | 65.1 | −4.1 | −4.5 | 99.4% |

Conditioning on termination is **not** a neutral filter — hard questions truncate most — so
these are biased upward by an amount that grows as the non-truncated share falls:

- **HotpotQA (99.4%) is clean**: the −4.1 F1 is a real regression.
- **MATH-500 (86.7%) is near-clean**: the finetune adds ~+9.6 on generations that finish, and
  truncation eats all of it, turning +9.6 into a −0.8 headline.
- **GPQA-D**: ~8 of the 11.5-point drop is truncation; ~3 points look like real capability loss.
- **AIME'25's +47.6 is not interpretable** — 30.4% of SFT generations survive against 90.8% of
  raw, so the conditionals cover very different question subsets. The defensible AIME claim is
  the headline: **+4.4 despite 69.58% of generations being pre-scored zero**.

### 4. Protocol validation

The non-thinking protocol was checked against two independent published sources before any
conclusion was drawn from it:

| Check | ours | published | Δ |
|---|---|---|---|
| MATH-500 acc vs Qwen3 report | 84.4 | 84.8 | −0.4 |
| AIME'25 acc vs Qwen3 report | 20.8 | 19.1 | +1.7 |
| GPQA-D acc vs Qwen3 report | 45.1 | 41.7 | +3.4 |
| MATH-500 acc vs paper Baseline | 84.4 | 83.2 | +1.2 |
| MATH-500 **tokens** vs paper Baseline | 1,075 | 1,087 | −12 |

Reproducing the paper's Baseline on *both* accuracy and token count is what rules out the
sampler (top-p 0.8) and the prompt/template/grader chain as causes of the SFT's collapse.

### 5. Not comparable — do not put these side by side

- **`think` vs `no-think` token counts.** The former omit the stripped reasoning trace.
- **HotpotQA against the paper.** No standard LLM prompt exists for HotpotQA; ours is
  homegrown, and raw averages 66 tokens against the paper's Baseline 482 — a 7x gap that
  reflects prompt design, not model behaviour. The column is comparable across our own
  checkpoints only.
- **HotpotQA `think` (7,405 rows) vs `no-think` (500 rows).** Different subsets, ±0.6 vs ±2 pp.
- **AIME avg@64 (raw 4B, think) vs avg@16** everywhere else.
- **Our SFT vs the paper's `SFT (CoT)`.** Same base model and same 600k Dolci-Think-SFT
  source, but our training hyperparameters are our own, and the paper does not state how it
  formatted CoT for supervision — the most likely origin of the termination difference.

### 6. What the paper leaves unspecified

Confirmed absent from the paper and all three appendices; each was our choice:
decoding hyperparameters, generation cap, samples-per-question, exact CoT prompt wording,
HotpotQA split/passage setup and how its 500 were drawn.

### 7. Cost

**2,961 SBU total** (~15.4 GPU-h on H100, billing 192 SBU/h per GPU): 2,668 non-thinking SFT
array + 6 smoke + 46 raw MATH-500 control + 248 raw AIME/GPQA/HotpotQA control. The SFT's
repetition loops dominate: its four benchmarks took 13.9 GPU-h against 1.4 for raw.


## Shared settings

| | |
|---|---|
| Decoding | temperature 0.6, top-p 0.95, top-k 20, presence penalty 0 |
| Chat template | applied (`--apply_chat_template`), thinking enabled, no system prompt |
| Max output | 32,768 tokens (38,912 for AIME'25) |
| Context | `max_model_len=40960` (Qwen3-4B `max_position_embeddings`) |
| Precision | bf16 |
| Seed | 1234 |

These are the Qwen3 thinking-mode settings from the [Qwen3 Technical
Report](https://arxiv.org/abs/2505.09388) §5.3: *"For all Qwen3 models in the
thinking mode, we utilize a sampling temperature of 0.6, a top-p value of 0.95,
and a top-k value of 20"* and *"we set the max output length to 32,768 tokens,
except AIME'24 and AIME'25 where we extend this length to 38,912 tokens to
provide sufficient thinking space"*. Presence penalty 1.5 applies only to Creative
Writing v3 / WritingBench, so it is 0 here. Greedy decoding is explicitly
discouraged for Qwen3 (repetition collapse).

**Thinking blocks are stripped before grading.** `think_end_token=</think>` makes
lm-eval (a) send only EOS to vLLM as a stop string and (b) score
`reply.split("</think>")[-1]`. Both matter: task-level stop strings would
otherwise fire inside the reasoning trace, and an answer regex would match text
the model was only considering.

## Per benchmark

| | MATH-500 | AIME'25 | GPQA-Diamond | HotpotQA |
|---|---|---|---|---|
| Dataset | `HuggingFaceH4/MATH-500` test | `math-ai/aime25` test | `Idavidrein/gpqa` `gpqa_diamond` | `hotpotqa/hotpot_qa` `distractor` validation |
| n | 500 | 30 | 198 | 7,405 |
| Samples | 4 | **64** | **10** | 1 |
| Prompt | `{problem}` + *"Please reason step by step, and put your final answer within \boxed{}."* | same | simple-evals `QUERY_TEMPLATE_MULTICHOICE`, verbatim | see below |
| Grader | `math_verify` on last `\boxed{}` | `math_verify` | `(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?`, first match | official `hotpot_evaluate_v1.py` EM + F1 |
| Reported | `math_verify` | `math_verify` | `exact_match` | `exact_match`, `f1` |
| Qwen3-4B (thinking) reference | 97.0 | 65.6 | 55.9 | not reported |
| Qwen3-0.6B (thinking) reference | 77.6 | 15.1 | 27.9 | not reported |

Qwen3-4B references are Table 17 of the Qwen3 report, Qwen3-0.6B references Table 19
(*"Comparison among Qwen3-1.7B / Qwen3-0.6B (Thinking) and other reasoning baselines"*).
HotpotQA appears nowhere in the report at any model size.

`cotc_aime25_16` is a 16-repeat variant of `cotc_aime25`, identical in every other
respect, selected by the 5th positional argument of `run_benchmarks.sh`. It exists
because the 0.6B run was commissioned at avg@16; `cotc_aime25` stays at 64 so the 4B
baseline remains reproducible byte for byte. **The two AIME columns are therefore not
the same protocol** — `summarize.py` gives them separate rows rather than stacking them.

Sample counts follow the Qwen3 report: *"For GPQA-Diamond, we sample 10 times for
each query and report the averaged accuracy"* and, for AIME, *"For each question,
we sample 64 times and take the average accuracy as the final score"*. It gives no
repeat count for MATH-500; [DeepSeek-R1](https://arxiv.org/abs/2501.12948) uses 64
samples on every benchmark including MATH-500. 4 sits inside both conventions.

The counts are driven by statistics, not convention alone — but **two different
error bars matter, and they behave differently.**

*Generation noise* is what repeats shrink: `SE_gen ≈ √(p(1−p)/(N·k))`.

| | n | k | SE_gen at k=1 | SE_gen at chosen k |
|---|---:|---:|---:|---:|
| AIME'25 | 30 | 64 | ±8.7 pp | ±1.1 pp |
| GPQA-Diamond | 198 | 10 | ±3.5 pp | ±1.1 pp |
| MATH-500 | 500 | 4 | ±0.8 pp (at p=.97) | ±0.4 pp |
| HotpotQA | 7,405 | 1 | ±0.6 pp | — |

*Question-sampling noise* is the `±` **lm-eval actually reports**: the standard
error of the mean over the n per-question scores. Repeats do **not** shrink it —
it reflects which questions are in the benchmark. It is therefore always the
larger of the two: GPQA-Diamond reports ±2.9 pp, not the ±1.1 pp above.

Which to quote depends on the claim. For a single model's absolute score against
published numbers, the reported (question-sampling) `±` is correct. For
**raw vs SFT on the identical question set**, the comparison is paired and the
question-sampling component largely cancels, so the uncertainty on the
*difference* is nearer `SE_gen` — which is exactly what the repeat counts buy.

HotpotQA gets one sample because 7,405 questions already average generation noise
below ±0.6 pp; repeats there would be the most expensive and least informative
spend in the suite.

### GPQA-Diamond

Prompt and answer regex are [openai/simple-evals](https://github.com/openai/simple-evals)
`gpqa_eval.py` + `common.py`, verbatim — the implementation behind the GPQA
numbers in most model cards. Choices are built as `[correct, wrong1, wrong2,
wrong3]` and permuted; an unmatched reply scores 0.

The dataset is gated (`gated: auto`): it needs an accepted licence on the Hub,
not merely a token — an unauthorised account gets repo *metadata* fine and 403s
only on download.

2 of the 198 rows carry a duplicate among their three *incorrect* answers, so
those questions present three distinct options rather than four. **0 rows**
duplicate the correct answer as an incorrect one, so the gold letter is never
ambiguous. simple-evals resolves this identically (`choices.index(correct)`),
so no correction is applied.

### HotpotQA

EM and F1 are ports of the official
[`hotpot_evaluate_v1.py`](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py),
cross-checked against a verbatim copy of the original on 5,000 random
prediction/gold pairs (0 mismatches). The critical detail is that the official F1
returns 0 whenever prediction *or* gold is `yes`/`no`/`noanswer` and the two
differ — SQuAD F1, which lm-eval ships with its `qasper` task, would award token
overlap there instead.

Prompt: the ten distractor paragraphs as `title: sentences`, then

> Using only the passages above, answer the question. The answer is a short span
> copied from the passages, or yes/no. End your reply with a line of the form
> 'Answer: <answer>'.

then `Question: {question}`. The prediction is the last `Answer:` line, falling
back to the last line of the reply.

## Deviations, and what they cost

1. **HotpotQA has no standard LLM prompt.** Published numbers are largely from
   supervised readers, so this column is comparable *across your checkpoints*,
   not to the literature. The metric is the official one; the prompt is ours.
2. **GPQA choice permutation is fixed per question, not per repeat.**
   simple-evals draws a fresh permutation for every one of its `n_repeats`
   copies, marginalising position bias over repeats. lm-eval resamples a fixed
   prompt, so the order is pinned by row index instead — reproducible, but
   position bias is not averaged out.
3. **MATH-500 at 4 samples, not 64.** Inside both published conventions, and the
   residual noise (±0.4 pp) is far below any effect worth reporting.
4. **`\boxed{}` prompting on MATH-500/AIME rather than each benchmark's original
   framing.** MATH-500's original protocol is a 4-shot completion format; the
   boxed instruction is the Qwen3/DeepSeek-R1 convention for reasoning models and
   is what the report's own numbers use.

## Sampling independence

`evals/run_benchmarks.sh` runs **one benchmark per GPU with
`data_parallel_size=1`**, as a 4-way SLURM array — not one 4-GPU data-parallel
job. lm-eval constructs every ray actor from the same `model_args`, so all four
vLLM engines share one seed
(`lm_eval/models/vllm_causallms.py:135`), while requests are dealt out by uniform
interleave. The 64 repeats of an AIME question therefore land at matching batch
positions in identically-seeded engines, which can silently collapse avg@64 into
far fewer distinct samples. One engine per benchmark makes repeats independent by
construction, costs the same GPU-hours, and isolates a failure to one benchmark.

lm-eval also discards repeats by default: with no `filter_list` it installs
`take_first` (`lm_eval/api/task.py:771-776`) and every stock grader reads
`results[0]`, so `repeats: 64` would cost 64x and change nothing. Each task here
declares a `take_first_k` filter, which is what hands all samples to the grader.

## Why not the stock lm-eval tasks

| stock task | why not |
|---|---|
| `minerva_math500` | 4-shot completion prompt; `exact_match` requires Minerva's *"Final Answer: … I hope it is correct."*, which a 0-shot chat model never emits — it scores ~0 regardless of correctness |
| `aime25` | grader compares normalised strings, so `\boxed{070}` ≠ `70`; `math_verify` accepts it |
| `gpqa_diamond_cot_zeroshot` | prompt requests no answer format, yet its `strict-match` filter greps for `"The answer is "` — reports a meaningless 0 |
| HotpotQA | does not exist; `longbench/hotpotqa` and `ruler/qa_hotpot` are long-context variants, not the distractor benchmark |

## Raw-model baseline (`Qwen/Qwen3-4B`, 2026-08-05)

`results/benchmarks/qwen3-4b-raw/` on HOME. The `±` is question-sampling error,
as reported by lm-eval.

| Benchmark | measured | Qwen3 report | Δ |
|---|---:|---:|---:|
| MATH-500 (avg@4) | 96.6 ± 0.7 | 97.0 | −0.4 |
| AIME'25 (avg@64) | 65.7 ± 7.0 | 65.6 | +0.1 |
| GPQA-Diamond (avg@10) | 54.1 ± 2.9 | 55.9 | −1.8 |
| HotpotQA EM | 59.9 ± 0.6 | — | — |
| HotpotQA F1 | 76.2 ± 0.4 | — | — |

**All three reproducible benchmarks land inside their error bars of Qwen's own
published numbers.** That agreement is the harness's validation: prompts,
graders, sampling and thinking-mode handling together reproduce the reference
model's reported behaviour, so a later divergence on the SFT checkpoint can be
attributed to the checkpoint rather than to the measurement.

Cost: 2,238 SBU (5.1 GPU-h). AIME'25 alone is 1,691 of that — 1,920 generations
of ~18k tokens each, at 768 SBU/h on a TP=4 node.

## Raw-model baseline (`Qwen/Qwen3-0.6B`, 2026-08-11)

`results/benchmarks/qwen3-0.6b-raw/`. AIME'25 here is **avg@16**, not the avg@64 of
the 4B run above (`cotc_aime25_16`); every other setting is identical, and all four
benchmarks ran at TP=1 on one H100 each.

| Benchmark | measured | Qwen3 report | Δ | Δ / SE |
|---|---:|---:|---:|---:|
| MATH-500 (avg@4) | 74.6 ± 1.6 | 77.6 | −3.0 | 1.9 |
| AIME'25 (avg@16) | 18.1 ± 5.0 | 15.1 | +3.0 | 0.6 |
| GPQA-Diamond (avg@10) | 25.4 ± 1.8 | 27.9 | −2.5 | 1.4 |
| HotpotQA EM | 17.3 ± 0.4 | — | — | — |
| HotpotQA F1 | 22.6 ± 0.5 | — | — | — |

All three reproducible benchmarks again land inside their error bars, so the harness
reproduces the reference model at a second, very different model scale. Two caveats
make these numbers weaker evidence than the 4B run's, both about the model rather
than the measurement:

1. **GPQA-Diamond is at chance.** 25.4 on a 4-way multiple choice, with a near-uniform
   predicted-letter distribution (A 514 / B 541 / C 530 / D 368 over 1,980 generations).
   The report's own 27.9 is barely above the 25% floor, so agreement here confirms
   little beyond the harness not being broken. GPQA has no usable headroom at 0.6B and
   will not discriminate between compression methods.
2. **HotpotQA is dominated by a degenerate yes-bias.** 65% of replies are `Answer: yes`
   or `Answer: no` while only 6% of gold answers are yes/no questions — the prompt's
   *"a short span copied from the passages, or yes/no"* offers a weak model an easy
   branch to collapse into. Prompts are byte-identical to the 4B run (matching
   `prompt_hash`), so the 59.9 → 17.3 EM drop is real capability, but it is measuring
   instruction-following collapse more than multi-hop retrieval.

Output truncation is not a confound. Replies that never emit `</think>` return the
whole reasoning trace and grade ~0, but they are rare: 0.8% of MATH-500 generations,
2.5% of AIME'25, 1.8% of GPQA, 0.1% of HotpotQA. On the two math benchmarks the set of
replies containing no `\boxed{}` is exactly the set of unterminated ones (15/15 and
12/12), so untruncated generations essentially always produce a gradable answer. The
implied ceiling correction to MATH-500 is under +0.8 pp, well short of the −3.0 gap.

Cost: 602 SBU (3.1 GPU-h) — MATH-500 232, AIME'25 226, GPQA 111, HotpotQA 33. Wall
clock 1 h 13 m, set by the two math benchmarks running in parallel.

## Finetuned Qwen3-4B (SFT on Dolci-Think-SFT-7B, 2026-08-13)

`results/benchmarks/qwen3-4b-sft/`. The checkpoint is the completed one-epoch full-parameter
SFT run (4,518 steps, 5.964 B supervised tokens, eval loss 0.8442 — see `results/HANDOFF.md`):

```
/scratch-shared/$USER/cot_compression/outputs/runs/sft/dolci_think_sft_7b_600k/full_lr1e-05_gb128/checkpoints/best
```

AIME'25 here is **avg@16** (`cotc_aime25_16`), not the avg@64 of the raw 4B row above, so the
two AIME cells are not the same protocol — `summarize.py` gives them separate rows. avg@k is
unbiased for any k, so the values still compare, with a wider generation error bar. Everything
else is identical to the raw 4B run, and all four benchmarks ran at TP=1 on one H100 each.

| Benchmark | raw 4B | SFT 4B | Δ | SFT terminated-only | Δ vs raw terminated-only |
|---|---:|---:|---:|---:|---:|
| MATH-500 (avg@4) | 96.6 ± 0.7 | 95.0 ± 0.8 | −1.6 | 96.0 | −0.8 |
| AIME'25 | 65.7 ± 7.0 (@64) | 56.5 ± 7.1 (@16) | −9.2 | 58.7 | −7.8 |
| GPQA-Diamond (avg@10) | 54.1 ± 2.9 | 40.7 ± 2.6 | −13.5 | 50.9 | −3.5 |
| HotpotQA EM | 59.9 ± 0.6 | 58.2 ± 0.6 | −1.7 | 59.1 | −0.8 |
| HotpotQA F1 | 76.2 ± 0.4 | 73.6 ± 0.4 | −2.6 | 74.8 | −1.5 |

**Every one of these drops is statistically real.** The two runs score the identical question
sets, so the comparison is paired; the mean per-question difference is 2.9–7.8 paired standard
errors from zero, including AIME'25 despite its wide unpaired `±`:

| | MATH-500 | AIME'25 | GPQA-D | HotpotQA EM | HotpotQA F1 |
|---|---:|---:|---:|---:|---:|
| paired Δ | −1.6 | −9.3 | −13.4 | −1.7 | −2.6 |
| paired SE | 0.53 | 2.20 | 1.78 | 0.43 | 0.33 |
| Δ / SE | 2.9 | 4.2 | 7.5 | 3.9 | 7.8 |

**Truncation is a confound here, unlike in both raw baselines, and it dominates the GPQA
number.** Unterminated replies grade 0 under every grader, so a quarter of GPQA's generations
were scored wrong by construction:

| | MATH-500 | AIME'25 | GPQA-D | HotpotQA |
|---|---:|---:|---:|---:|
| raw 4B unterminated | 0.20% | 4.01% | 0.40% | 0.14% |
| SFT 4B unterminated | 1.90% | 10.42% | **26.67%** | 1.61% |

### Trace-length inflation is task-dependent

It is tempting to say the SFT model "reasons longer" across the board. It does not. lm-eval's
progress line reports input and output token rates, and elapsed × rate recovers the token
totals — the input totals agree between the two runs to within 2%, which independently confirms
the prompts are identical:

| | MATH-500 | AIME'25 | GPQA-D | HotpotQA |
|---|---:|---:|---:|---:|
| raw 4B mean output tokens/generation | 5,306 | 18,519 | 6,602 | 530 |
| SFT 4B mean output tokens/generation | 5,508 | 19,340 | 14,341 | 1,074 |
| ratio | ×1.04 | ×1.04 | **×2.17** | **×2.03** |

On the two math benchmarks trace length is unchanged; on GPQA-Diamond and HotpotQA it doubles.
Both of those means are **censored at the generation cap**, so where truncation is common the
true underlying length is higher than the table shows — which is also why MATH-500 and AIME'25
can show a flat mean while their truncation rates still rise.

The mechanism is visible in the training data. The supervised assistant turn in the 600 k Dolci
subset averages **10,087 tokens** (median 6,820, p90 24,795), so one epoch teaches a strong
length prior that is close to task-independent. Where the raw model already reasoned at that
scale (MATH-500 5.3 k, AIME'25 18.5 k) nothing changes; where it was terse because the task is
easy — HotpotQA is single-hop-ish retrieval the raw model answers in 530 tokens — the finetuned
model spends twice as long, and on GPQA it lands near the training mean.

That also explains the wall clock, which grew faster than the token count. Longer sequences fit
less concurrently in the fixed 545,920-token KV cache (≈38 concurrent GPQA sequences for SFT
against ≈83 for raw), so decode throughput fell from 2,637 to 1,042 tok/s at the same time as
the token count rose 2.17×. Multiplied, that is the 1 h 22 m → 7 h 34 m blow-up. Budget for it.

### Is the regression an artifact?

No — every measurement-side explanation was checked and eliminated:

- **Prompts are byte-identical.** `prompt_hash` and `target_hash` match doc-for-doc, and the
  independently-derived input-token totals agree.
- **Thinking mode was on in both runs.** The rendered prompt ends `<|im_start|>assistant\n`
  with no `<think>` prefill, so the model opens the block itself; `enable_thinking=True` and
  `think_end_token=</think>` were passed identically.
- **Sampling is identical.** Temperature/top-p/top-k come from the task YAML, and the
  checkpoint's `generation_config.json` matches the base model's exactly.
- **The weights load correctly.** bf16, the same 545,920-token KV cache, and MATH-500 at 95.0 —
  a broken load does not score 95 on MATH-500.
- **The checkpoint is the converged model.** Eval loss is flat from step 4,100 (0.8443) to the
  final 4,518 (0.8442), so `best/` and the last step are the same model in all but name.

The regression is what one epoch of full-parameter SFT does to a model that was *already*
post-trained. Qwen3-4B ships with its own SFT+RL reasoning behaviour (65.6 AIME'25); training
it on 5.96 B tokens of a different think distribution replaces that behaviour with the new one,
and the eval loss of 0.8442 says it fit the target distribution well. The damage is smallest on
math, which the Dolci think mix covers heavily, and largest on GPQA-Diamond and HotpotQA. Read
this table as *what the finetune did*, not as a bug to hunt.

The test for "unterminated" is exact rather than heuristic. lm-eval stores replies already
split on `</think>` (`lm_eval/models/utils.py:911`), so a stored reply that *still* begins with
`<think>` is one that never closed its reasoning block:

```python
unterminated = gen.lstrip().startswith("<think>")
```

It agrees with the length test the 0.6B section used — the char-length distribution is sharply
bimodal (median ~1.2–1.7 k, max ~165 k) — and on the math benchmarks the unterminated set again
almost exactly equals the no-`\boxed{}` set (38 vs 36, and 50 vs 49).

The **terminated-only** column recomputes each metric over terminated generations only, using
the graders in `evals/tasks/utils.py`; recomputing the *unrestricted* metric the same way
reproduces every harness headline number for both models exactly, which is what validates the
column. Read it as an optimistic bound, not a correction: conditioning on termination also
conditions on the questions the model found short enough to finish, which are the easier ones.
The honest reading is that the true gap lies between the two Δ columns — small on MATH-500 and
HotpotQA, real on AIME'25, and on GPQA-Diamond mostly an artifact of the 32,768-token cap
rather than a loss of knowledge.

Two further notes:

1. **The finetune did not break output formatting.** The checkpoint's `tokenizer.json` is
   re-serialised by transformers 5.x and differs from the hub copy in
   `pre_tokenizer.ByteLevel.trim_offsets` and the `decoder` flags — fields that affect
   character-offset reporting, not token ids; `vocab`, `merges`, `added_tokens`, `normalizer`
   and `post_processor` are identical, and `chat_template.jinja` is byte-identical to the hub
   template. The matching `prompt_hash` confirms this empirically. 97.2% of HotpotQA replies
   still carry a parseable `Answer:` line versus 99.9% for raw, and the residual is the
   truncated set.
2. **No yes/no collapse.** 4.7% of HotpotQA replies answer yes/no against a 6.2% gold rate — the
   SFT model is *better* calibrated on that branch than the raw model (0.9%), and shows none of
   the degenerate yes-bias that dominates the 0.6B baseline.

`config.json` reports `dtype: float32` while the safetensors are bf16; vLLM logs
`Downcasting torch.float32 to torch.bfloat16` and allocates the same 545,920-token KV cache as
the raw run, so the explicit `dtype=bfloat16` in `run_benchmarks.sh` is doing its job.

Cost: 2,587 SBU (13.5 GPU-h) — GPQA-Diamond 1,453, AIME'25 497, MATH-500 285, HotpotQA 341,
plus 11 for a `--limit 2` smoke run. Wall clock 7 h 34 m, set entirely by GPQA-Diamond; the
other three finished within 2 h 35 m. Note this is **more than the raw 4B run's 2,238 SBU
despite four times fewer AIME samples and TP=1 throughout**, purely because of trace length —
budget SFT checkpoints accordingly.

## Non-thinking mode (`cotc_nothink_*`, the abstract-CoT paper's `SFT (CoT)` setting)

Everything above evaluates in **thinking mode**. That is the wrong comparison for the
abstract-CoT line of work: *Thinking Without Words: Efficient Latent Reasoning with Abstract
Chain-of-Thought* (arXiv 2604.22709), whose `SFT (CoT)` baseline is what this repo's Qwen3-4B
SFT reproduces, states

> "We evaluate models without their 'thinking mode' with standard CoT prompting to ensure
> controlled comparison."

So there is a parallel task family, selected by the `nothink` mode argument of
`run_benchmarks.sh`. It sets `enable_thinking=False`, which makes the Qwen3 chat template emit
`<think>\n\n</think>\n\n` as part of the *prompt* — the reasoning then happens in the content
region, which is what "standard CoT prompting" means here.

| | thinking (`cotc_*`) | non-thinking (`cotc_nothink_*`) |
|---|---|---|
| `enable_thinking` | `True` | `False` |
| Decoding | t=0.6, top-p=0.95, top-k=20 | **t=0.7, top-p=0.8, top-k=20** |
| MATH-500 | 500 x avg@4 | same |
| AIME'25 | 30 x avg@64 (or @16) | 30 x avg@16 |
| GPQA-Diamond | 198 x avg@10 | same |
| HotpotQA | 7,405 x 1 | **500 x 1**, CoT trigger appended |

Four things are worth stating plainly, because each is a judgement call the paper does not make
for us:

1. **The sampler changes with the mode.** `generation_config.json` ships t=0.6/top-p=0.95, but
   those are Qwen3's *thinking-mode* values; the model card gives t=0.7/top-p=0.8/top-k=20 for
   `enable_thinking=False`. The paper specifies no decoding parameters at all, so Qwen's own
   per-mode guidance decides. This does mean the thinking/non-thinking delta is not a clean
   single-variable comparison — mode and sampler move together.
2. **The CoT trigger is added to HotpotQA only.** The other three prompts already instruct
   step-by-step reasoning (MATH-500/AIME: *"Please reason step by step, and put your final answer
   within \boxed{}."*; GPQA, simple-evals verbatim: *"Think step by step before answering."*).
   Adding a second instruction to those would have changed prompts that already comply.
3. **HotpotQA is the paper's 500-question subset**, drawn with `random.Random(1234)` rather than
   head-sliced — the distractor dev split carries a `level` field and is not shuffled. Generation
   noise is then ~±2 pp against ~±0.6 pp on the full split, so this cell is *not* comparable to the
   thinking-mode HotpotQA row. `summarize.py` keeps them as separate rows for that reason.
4. **`repeats` are not specified by the paper either** (nor is the prompt wording); avg@4 / @16 /
   @10 / @1 carry over from the thinking-mode protocol.

`think_end_token=</think>` is kept in both modes. lm-eval only *requires* it when
`enable_thinking=True` (`vllm_causallms.py:182`) but applies the split whenever it is set (`:615`),
and a checkpoint SFT'd exclusively on think-traces can reopen a `<think>` block even with thinking
off; stripping keeps the graders robust to that. The cost is that a *completed* stray block is
absent from the logged reply and its tokens go uncounted — see the caveat in `token_stats.py`.

### Token cost is part of the result

Table 1 of the paper pairs every accuracy with "the average number of generated tokens per prompt
during evaluation, combining reasoning and response tokens", so accuracy alone cannot be placed
against it. `evals/token_stats.py` reports mean/median/p95 generated tokens, truncation rate against
`max_gen_toks`, and the stray-`<think>` rate, from the `--log_samples` JSONL.

Its truncation column is also the honest way to read a thinking-mode run: on the 2026-08-13 SFT
benchmarks the means are dominated by runaway traces, not typical ones.

| Task | gens | mean tok | median | p95 | trunc % |
|---|---|---|---|---|---|
| `cotc_math500` | 2,000 | 1,102 | 468 | 901 | 1.85 |
| `cotc_aime25_16` | 480 | 4,633 | 663 | 38,912 | 10.42 |
| `cotc_gpqa_diamond` | 1,970 | 9,030 | 402 | 32,768 | 25.89 |
| `cotc_hotpotqa` | 7,405 | 572 | 35 | 108 | 1.59 |

A GPQA mean of 9,030 against a median of 402 is one in four generations hitting the 32,768 cap, not
a model that reasons at length.

**These thinking-mode token counts are not comparable to the paper's**, and must never be quoted
against its Table 1. With `enable_thinking=True` the reply is split on `</think>` before it is
logged, so what is counted is the *visible answer only* — the entire reasoning trace, which is
most of the generation and exactly what the paper's "reasoning and response tokens" includes, is
absent. Only the terminated-mid-think generations keep their full length, which is why the means
above are truncation artifacts on top of an undercount. The non-thinking runs have no such problem:
`<think>` and `</think>` appear in zero of their 4,950 generations, nothing was ever split, and
their counts are exact.

Target to compare against (Qwen3-4B, `SFT (CoT)` row of the paper's Table 1): MATH-500 **88.4** at
~1,396 generated tokens, HotpotQA **F1 50.3** at ~612. Those values were read off the arXiv HTML,
not the PDF — verify before quoting. The paper's AIME'25/GPQA-Diamond table is Qwen3-8B only, so
there is no published 4B row for those two; they stay useful as internal thinking-vs-not deltas.

## Finetuned Qwen3-4B, non-thinking (2026-08-15)

Same checkpoint as the 2026-08-13 run, `enable_thinking=False`, Qwen3's non-thinking sampler.
Job 25685046, four H100s in parallel: MATH 3 h 14 m, AIME 5 h 34 m, GPQA 4 h 56 m, HotpotQA 6 m.

| Benchmark | raw 4B (think) | SFT (think) | SFT (no-think) |
|---|---|---|---|
| MATH-500 (avg@4) | 96.6 ± 0.7 | 95.0 ± 0.8 | 83.6 ± 1.4 |
| AIME'25 (avg@16) | 65.7 ± 7.0 (@64) | 56.5 ± 7.1 | 25.2 ± 6.4 |
| GPQA-Diamond (avg@10) | 54.1 ± 2.9 | 40.7 ± 2.6 | 33.6 ± 2.2 |
| HotpotQA EM | 59.9 ± 0.6 | 58.2 ± 0.6 | 50.6 ± 2.2 (500) |
| HotpotQA F1 | 76.2 ± 0.4 | 73.6 ± 0.4 | 64.7 ± 1.9 (500) |

**These are termination numbers, not reasoning numbers.** With thinking off the model very often
never stops:

| Task | mean tok | median | p95 | trunc % | stray `<think>` % |
|---|---|---|---|---|---|
| `cotc_nothink_aime25_16` | 27,883 | **38,912** | 38,912 | **69.58** | 0.00 |
| `cotc_nothink_gpqa_diamond` | 7,604 | 824 | 32,768 | 21.12 | 0.00 |
| `cotc_nothink_math500` | 5,125 | 558 | 32,768 | 13.35 | 0.00 |
| `cotc_nothink_hotpotqa500` | 296 | 86 | 210 | 0.60 | 0.00 |

AIME's *median* generation hits the 38,912 cap. Regrading each repeat individually and splitting on
whether it terminated (the overall column reproduces lm-eval's aggregate exactly, which is the check
that the regrade is faithful):

| Task | terminated n | acc | truncated n | acc | overall | non-truncated share |
|---|---|---|---|---|---|---|
| `cotc_nothink_math500` | 1,733 | **94.5** | 267 | 12.7 | 83.6 | 86.7 |
| `cotc_nothink_aime25_16` | 146 | **70.5** | 334 | 5.4 | 25.2 | 30.4 |
| `cotc_nothink_gpqa_diamond` | 1,554 | **42.4** | 416 | 0.0 | 33.5 | 78.9 |
| `cotc_nothink_hotpotqa500` (F1) | 497 | **65.1** | 3 | 0.0 | 64.7 | 99.4 |

A truncated generation scores ~0 by construction — it never emits a final `\boxed{}` or `Answer:`
line (MATH's 12.7% is an intermediate `\boxed{}` that happened to be the right one; GPQA's is 0.0%
exactly). So the deficit is almost entirely non-termination: on the generations that *do* finish,
non-thinking CoT matches thinking mode on MATH-500 (94.5 vs 95.0) and GPQA-D (42.4 vs 40.7), and is
nominally above it on AIME (70.5 vs 56.5).

**Read those conditional numbers with care — they are selection-biased upward.** Hard questions are
exactly the ones that reason longest and therefore truncate, so conditioning on termination
preferentially keeps easy questions. This is sharpest on AIME, where only 30.4% of generations
survive; 70.5 is not an estimate of what AIME accuracy would be with an unlimited budget. The safe
claim is directional: the gap is dominated by a termination failure, not by degraded reasoning.

### The truncation is repetition collapse, not a harness bug

Diagnosed on MATH-500, where 267 of 2,000 generations hit the cap:

- **EOS works.** 1,733 of 2,000 generations terminated normally. A misconfigured stop string or
  `until: []` swallowing EOS would break all of them, not 13%.
- **94.8% of truncated generations (253/267) literally loop** — a 60-character window from the tail
  recurs earlier in the same reply. Sampled output shows the model enumerating
  `911. 2·(3·(4·(5+1))) = 144 / 912. ... / 913. ...` indefinitely. This is the failure Qwen3's model
  card warns about: near-greedy decoding "can lead to performance degradation and endless
  repetitions".
- **It concentrates on hard problems**, and touches only 119 of 500 docs:

  | MATH level | 1 | 2 | 3 | 4 | 5 |
  |---|---|---|---|---|---|
  | truncated / gens | 7/172 | 2/360 | 23/420 | 72/512 | **163/536** |

- `stray <think>` is 0.00% everywhere — in fact `<think>` and `</think>` appear in **zero** of the
  4,950 generations, so `think_end_token` never split anything and the token counts above are exact
  rather than lower bounds.

So the 5,125 mean decomposes into ~866 tokens of ordinary reasoning plus a repetition tail.

### Raw-model control, all four benchmarks (jobs 25724962, 25747002)

Raw `Qwen/Qwen3-4B` through this exact protocol — same sampler, prompts, chat template, graders,
full datasets. Wall: 14 m / 51 m / 25 m / 1.5 m, against 3 h 14 m / 5 h 34 m / 4 h 56 m / 6.7 m for
the SFT checkpoint.

| Benchmark | raw | SFT | Δ | Qwen3 report (raw, non-think) |
|---|---|---|---|---|
| MATH-500 | 84.4 ± 1.4 | 83.6 ± 1.4 | −0.8 | 84.8 |
| AIME'25 | 20.8 ± 6.6 | 25.2 ± 6.4 | **+4.4** | 19.1 |
| GPQA-Diamond | 45.1 ± 2.6 | 33.6 ± 2.2 | **−11.5** | 41.7 |
| HotpotQA EM | 54.6 ± 2.2 | 50.6 ± 2.2 | −4.0 | — |
| HotpotQA F1 | 69.2 ± 1.8 | 64.7 ± 1.9 | −4.5 | — |

| | raw trunc | SFT trunc | raw mean tok | SFT mean tok |
|---|---|---|---|---|
| MATH-500 | 0.55% | 13.35% | 1,075 | 5,125 |
| AIME'25 | **9.17%** | 69.58% | 6,187 | 27,883 |
| GPQA-Diamond | 1.32% | 21.12% | 1,292 | 7,604 |
| HotpotQA | 0.00% | 0.60% | 66 | 296 |

**The protocol is validated three times over.** Raw reproduces Qwen's published non-thinking numbers
on all three benchmarks that have them (84.4/84.8, 20.8/19.1, 45.1/41.7) and the paper's `Baseline`
row on both of its axes (84.4 vs 83.2 accuracy, 1,075 vs 1,087 tokens). top-p 0.8 is not the cause
of anything; the prompt/template/grader chain behaves as the published ones did.

**Non-termination is not purely an SFT artifact.** Raw truncates 9.17% on AIME'25 by itself — hard
math in non-thinking mode has an intrinsic termination problem. What the finetune does is *amplify*
it, by 7.6x on AIME and 16-24x on GPQA and MATH-500. An earlier draft of this section said the
finetune "is the cause"; the single-benchmark control it rested on could not see the 9.17%.

### What survives once truncation is removed

Regrading every repeat individually and conditioning on termination:

| Benchmark | raw term. | SFT term. | Δ (terminated) | Δ (headline) | SFT non-trunc share |
|---|---|---|---|---|---|
| MATH-500 | 84.9 | **94.5** | +9.6 | −0.8 | 86.7% |
| AIME'25 | 22.9 | 70.5 | +47.6 ⚠ | +4.4 | **30.4%** |
| GPQA-Diamond | 45.5 | 42.4 | −3.1 | −11.5 | 78.9% |
| HotpotQA (F1) | 69.2 | 65.1 | −4.1 | −4.5 | 99.4% |

Conditioning on termination is not a neutral filter — hard questions truncate most — so these are
biased upward by an amount that grows as the non-truncated share falls. Read them accordingly:

- **HotpotQA is clean** (99.4% / 100% terminate): the −4.1 F1 is a real regression, not truncation.
- **MATH-500 is close to clean** at 86.7%: the finetune genuinely adds ~+9.6 on generations that
  finish, and truncation eats all of it, turning +9.6 into a −0.8 headline.
- **GPQA-Diamond**: truncation accounts for roughly 8 of the 11.5-point drop; ~3 points look like
  real capability loss.
- **AIME'25's +47.6 is not interpretable.** Only 30.4% of SFT generations survive against 90.8% of
  raw, so the two conditional accuracies are computed over very different question subsets. The
  defensible AIME statement is the headline one: +4.4 *despite* 69.58% of its generations being
  pre-scored zero.

Net: the finetune is strongly math-favoring (consistent with Dolci-Think-SFT's composition) and
mildly harmful on science QA and multi-hop QA, but its dominant effect in non-thinking mode is a
termination failure that masks the gains.

Raising `max_gen_toks` is not available as a fix: AIME is already at 38,912 against a 40,960
context. The remedies are `presence_penalty` in (0, 2], which Qwen sanctions for exactly this mode,
or accepting the termination failure as the result.

One bookkeeping note: the regrade sees 1,970 GPQA generations (197 docs x 10) where lm-eval reports
198 docs, so its GPQA overall column reads ~0.2 pp low against `summarize.py`. It reproduces every
other task's aggregate exactly.

## Reproducing

```bash
export HF_HOME=/scratch-shared/$USER/huggingface
export HF_TOKEN=hf_...                       # Idavidrein/gpqa is gated
sbatch evals/run_benchmarks.sh Qwen/Qwen3-4B qwen3-4b-raw
uv run --project evals python evals/summarize.py results/benchmarks/qwen3-4b-raw
```

The 0.6B baseline, at avg@16 on AIME'25 and TP=1 throughout (positional args are
`<model> <name> <limit> <TP> <aime-task> <mode>`):

```bash
sbatch evals/run_benchmarks.sh Qwen/Qwen3-0.6B qwen3-0.6b-raw "" 1 cotc_aime25_16
uv run --project evals python evals/summarize.py \
  results/benchmarks/qwen3-4b-raw results/benchmarks/qwen3-0.6b-raw
```

Cost: ~60 M generated tokens, ~3.3 GPU-h, ~640 SBU; AIME'25 is the critical path
at ~1.6 h wall.

The SFT checkpoint, same shape as the 0.6B invocation — a local path works anywhere a hub id
does, since it is passed straight to vLLM's `pretrained=`:

```bash
CKPT=/scratch-shared/$USER/cot_compression/outputs/runs/sft/dolci_think_sft_7b_600k/full_lr1e-05_gb128/checkpoints/best
sbatch evals/run_benchmarks.sh "$CKPT" qwen3-4b-sft "" 1 cotc_aime25_16
uv run --project evals python evals/summarize.py \
  results/benchmarks/qwen3-4b-raw results/benchmarks/qwen3-4b-sft results/benchmarks/qwen3-0.6b-raw
```

Smoke-test a new checkpoint first — it costs ~11 SBU and catches a bad load or a broken chat
template before the array commits ~2,600:

```bash
sbatch --array=0 --time=00:40:00 evals/run_benchmarks.sh "$CKPT" qwen3-4b-sft-smoke 2 1 cotc_aime25_16
```

Budget the full run at **10 h walltime**, not less: GPQA-Diamond took 7 h 34 m on the SFT
checkpoint against 1 h 22 m on the raw model, and a finetune that reasons longer can move that
number again.

The same checkpoint in non-thinking mode — the sixth positional argument. `<aime-task>` is passed
empty because `nothink` pins AIME to `cotc_nothink_aime25_16` rather than accepting a `cotc_*` name
that would carry the thinking-mode sampler:

```bash
sbatch --array=3 --time=00:30:00 evals/run_benchmarks.sh "$CKPT" smoke-nothink 8 1 "" nothink
sbatch --array=0-3 evals/run_benchmarks.sh "$CKPT" qwen3-4b-sft-nothink "" 1 "" nothink

uv run --project evals python evals/summarize.py \
  results/benchmarks/qwen3-4b-sft results/benchmarks/qwen3-4b-sft-nothink
uv run --project evals python evals/token_stats.py results/benchmarks/qwen3-4b-sft-nothink
```

Non-thinking traces are far shorter and HotpotQA drops 7,405 → 500, so this should cost well under
the thinking-mode 2,587 SBU. The default 10 h walltime is still the right request: unused walltime
is not billed, and the only thing a tighter `--time` buys is timeout risk on GPQA, which is the task
that ran 7 h 34 m in thinking mode. Tighten it only if the budget is nearly spent — Snellius rejects
jobs whose walltime × billing rate exceeds what remains.

The smoke test above is worth the 2 minutes: it confirms the chat template emits
`<|im_start|>assistant\n<think>\n\n</think>\n\n` (thinking genuinely off), that the reasoning lands
in the content region, and that the `Answer:` grader still fires.
