# GSM8K templates for latent reasoning

We build a dataset for contrastive learning of LLM reasoning steps. A shared
calculation graph aligns steps across different views of a problem so their
representations can track the computation and logic while the wording can change.
[SimCLR](https://proceedings.mlr.press/v119/chen20j.html)

## Dataset

We extract runnable templates from GSM8K training problems, replay the original
answers with exact arithmetic, and check rendered questions independently. We
expand text choices and repair unclear wording while preserving the calculation
graph. [GSM8K source](https://github.com/openai/grade-school-math) ·
[Prompts](prompts/)

The local snapshot contains **6,000 templates**. Each supports five text
versions at two legal numeric settings with different answers. Four templates
are flagged for math review. The snapshot and profile are in
`data/gsm8k-luna-6000-pre-story-variation/`.

**GSM8K Story Views 100K** contains 100,000 rendered problems from 2,000 base
templates and 10,000 story variants. Each base has five numeric setups shared
across its five stories. Each story has five text settings with two numeric
instances each, so every numeric setup has ten matching views.
[Hugging Face dataset](https://huggingface.co/datasets/danchern/gsm8k-story-views-100k) ·
[Generation and validation record](docs/gsm8k-story-views-100k.md)

## Models and cost

The OpenAI Responses API uses **GPT-6 Luna on Flex** with prompt caching and
**GPT-6.2 Sol on Batch** for targeted arithmetic repairs. The base dataset cost
about **$4.22** through wording review, estimated from recorded token usage.

## Reproduce base template extraction

Install the locked dependencies and download the original GSM8K training file.

```sh
uv sync --locked
mkdir -p data/gsm8k
curl -fsSL \
  https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl \
  -o data/gsm8k/train.jsonl
```

Set `OPENAI_API_KEY` in the environment or in
`~/.config/cot-compression/openai.env` with permissions `600`. Start with 100
source problems.

```sh
source_file=data/gsm8k/train.jsonl
run_dir=outputs/gsm8k-run

uv run --locked gsm8k generate "$source_file" "$run_dir/raw" --count 100 --seed 42
uv run --locked gsm8k assess "$source_file" "$run_dir/assessed" \
  --templates "$run_dir/raw/templates.jsonl"
uv run --locked gsm8k repair "$source_file" "$run_dir/repaired" --batch \
  --templates "$run_dir/raw/templates.jsonl" \
  --findings "$run_dir/assessed/outcomes.jsonl"
uv run --locked gsm8k expand-text "$source_file" "$run_dir/expanded" \
  --templates "$run_dir/repaired/templates.jsonl"
uv run --locked gsm8k classify "$source_file" "$run_dir/classified" \
  --templates "$run_dir/expanded/templates.jsonl" \
  --prior-templates "$run_dir/repaired/templates.jsonl"
uv run --locked gsm8k repair-wording "$source_file" "$run_dir/reviewed" \
  --templates "$run_dir/expanded/templates.jsonl" \
  --findings "$run_dir/classified/outcomes.jsonl"
```

Use `--dry-run` to prepare requests without API calls. Review
`outcomes.jsonl` after each stage. The saved snapshot includes manual review,
so a fresh run may yield a different accepted set.
