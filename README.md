# GSM8K template generation

One CLI extracts runnable templates and runs the assessments and repairs used
for the GSM8K dataset. The prompts are in `prompts/`. The template evaluator
uses exact arithmetic and renders questions with worked solutions.

## Setup

```sh
uv sync --locked
mkdir -p data/gsm8k
curl -fsSL https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl -o data/gsm8k/train.jsonl
```

The CLI uses zero-based row indices from the original training file as source
IDs. [Official GSM8K training file](https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/train.jsonl)

Set `OPENAI_API_KEY` in the environment, or put `OPENAI_API_KEY=...` in
`~/.config/cot-compression/openai.env` and run
`chmod 600 ~/.config/cot-compression/openai.env`.

## Run the workflow

```sh
source_file=data/gsm8k/train.jsonl
run_dir=outputs/gsm8k-run

uv run --locked gsm8k generate "$source_file" "$run_dir/raw" --count 100
uv run --locked gsm8k assess "$source_file" "$run_dir/assessed" \
  --templates "$run_dir/raw/templates.jsonl"
uv run --locked gsm8k repair "$source_file" "$run_dir/repaired" --batch \
  --templates "$run_dir/raw/templates.jsonl" --findings "$run_dir/assessed/outcomes.jsonl"
uv run --locked gsm8k expand-text "$source_file" "$run_dir/expanded" \
  --templates "$run_dir/repaired/templates.jsonl"
uv run --locked gsm8k classify "$source_file" "$run_dir/classified" \
  --templates "$run_dir/expanded/templates.jsonl" --prior-templates "$run_dir/repaired/templates.jsonl"
uv run --locked gsm8k repair-wording "$source_file" "$run_dir/reviewed" \
  --templates "$run_dir/expanded/templates.jsonl" --findings "$run_dir/classified/outcomes.jsonl"
```

Extraction uses Luna Flex. Assessment solves two rendered questions
independently and checks their answers against the template. The computational
repair uses Sol with the original question and assessment findings, then
reassesses its changes with Luna. Text expansion and wording repair preserve
the calculation graph and verify five text versions at two legal numeric
settings. Wording changes also get a separate Luna review.

Use `--dry-run` to write requests without an API call. Rerunning the same stage
with the same inputs resumes from its response journal. Read
`outcomes.jsonl` for failures and review decisions. For text or wording
retries, use a new output directory, the previous `templates.jsonl`, and
`--feedback` with the previous `outcomes.jsonl`.

The local 6,000-template snapshot and its profile are in
`data/gsm8k-luna-6000-pre-story-variation/`. API responses, generated datasets,
and credentials stay local.

## Code layout

The `gsm8k` command starts in `cli.py`. Under `src/cot_compression/data/gsm8k/`,
`generation.py` runs extraction and arithmetic review, and `revisions.py` applies
text changes. `templates.py` evaluates the calculations, and `sampling.py` renders
examples. `api.py` and `schemas.py` handle requests and structured responses.
