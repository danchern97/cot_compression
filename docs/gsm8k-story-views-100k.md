# GSM8K Story Views 100K generation

The release contains 100,000 rendered questions and solutions from 2,000 GSM8K
training problems. Each base has five story variants with the same calculation
graph and five shared numeric setups. The public release is pinned to revision
`3210202882745c9a1b9ee54d5e2cc9003612fc5d`.
[Dataset card and viewer](https://huggingface.co/datasets/danchern/gsm8k-story-views-100k/tree/3210202882745c9a1b9ee54d5e2cc9003612fc5d)

## Template generation

1. **Extract and repair.** Extract templates from the pinned GSM8K training
   data with GPT-6 Luna, verify source answers with exact arithmetic, and audit
   rendered questions. Use GPT-6.2 Sol on Batch for arithmetic repairs, then
   expand text choices and review wording. The extraction commands and prompts
   are in the [project README](../README.md#reproduce-base-template-extraction).
2. **Add text slots.** Expand templates toward at least two varying text slots
   while preserving their calculations and numeric ranges. Review the rendered
   edits and keep five distinct legal text settings per accepted template.
   [Text-slot prompt](../prompts/add_gsm8k_text_slots_v4.txt) ·
   [Review prompt](../prompts/audit_gsm8k_text_slots_full.txt)
3. **Generate and review five stories.** Use GPT-6 Luna on Flex with low
   reasoning effort and explicit prompt caching. Generate distinct premises,
   rename numeric inputs where appropriate, and adapt grids to the stories.
   Compile structured replies against the base graph and check novelty,
   wording, and numeric validity. Blind-solve a sample of the renders.
   The run produced 13,016 approved stories and 2,192 complete families
   eligible for sampling after excluding two flagged bases.
   [Story prompt](../prompts/vary_gsm8k_story_family_v3.txt) ·
   [Single-family audit](../prompts/audit_gsm8k_story_family.txt) ·
   [Grouped audit](../prompts/audit_gsm8k_story_family_grouped.txt)
4. **Select and render.** Start from the prior 2,000 families, stratified by
   visible solution-step count with seed `42`. Intersect each base grid with
   its five story grids using the canonical numeric input mapping. With seed
   `20261119`, find five exact assignments that satisfy the calculation
   constraints and render cleanly at their assigned text settings. The five
   text settings receive A/B, B/C, C/D, D/E, and E/A. Each pair has two
   different answers. The final sample keeps 1,978 families and substitutes
   22 reviewed reserves, 21 of which match the original visible step count.
   One two-step family replaces a one-step family.

## Replay and package

The deterministic replay reads the following archived JSONL inputs and checks
their SHA-256 hashes. These local checkpoints contain the reviewed model
outputs and are kept outside Git. A fresh API run may yield different stories.

| Checkpoint | Files |
|---|---|
| `data/gsm8k-luna-100k-views-per-text/` | `profile.json`, `templates.jsonl`, `views.jsonl` |
| `data/gsm8k-luna-6000-expanded-text-slots/` | `templates.jsonl`, `text-slot-review.jsonl` |
| `data/gsm8k-luna-6000-five-stories-partial/` | `templates.jsonl`, `views.jsonl` |

Run from the repository root with the archived inputs in place. The release
used Python 3.12.13 and PyArrow 24.0.0. These steps are local and require no
OpenAI API calls.

```sh
uv sync --locked --python 3.12.13
uv run --locked python scripts/regenerate_gsm8k_shared_views.py
uv run --locked --with pyarrow==24.0.0 python scripts/package_gsm8k_huggingface.py
```

The sampler writes `views.jsonl`, `templates.jsonl`, and `profile.json` under
`data/gsm8k-luna-100k-shared-views/`. The packager writes the two Parquet subsets
under `outputs/gsm8k-story-views-100k-shared-huggingface/data/`. Both scripts
accept `--output` to write a separate replay. The checked-in profile records
the archive hashes, output hashes, and the exact family replacements.
[Release profile](gsm8k-story-views-100k-profile.json)

## Validation and keys

- Exact arithmetic replay covered all 100,000 views. The export contains
  50,000 text settings with two different answers each and zero duplicate
  questions after case and whitespace normalization.
- Each of the 10,000 `numeric_group_id` values spans ten views with identical
  numeric inputs and intermediate values. Pair it with `step_id` to align
  computations across wording and stories.
- Match `base_id` and `step_id`, then compare `solution_steps.operands` to pair
  steps with different inputs. Cross-template `structural_signature` matches
  are candidates for semantic review.
