"""Generated-token statistics for one or more benchmark runs.

    uv run --project evals python evals/token_stats.py results/benchmarks/qwen3-4b-sft-nothink

Table 1 of arXiv 2604.22709 pairs every accuracy with "the average number of
generated tokens per prompt during evaluation, combining reasoning and response
tokens". summarize.py reports only the accuracies, so without this the run cannot
be placed against the paper's `SFT (CoT)` row -- the whole point of that row is
accuracy *at* a token cost.

Also reports the two failure modes that made the thinking-mode run hard to read:
truncation against max_gen_toks, and replies that reopened a <think> block despite
thinking being off.

CAVEAT on the token counts. lm-eval stores replies already split on think_end_token
(lm_eval/models/utils.py:911), so a *completed* stray <think>...</think> block is
gone from the logged text and its tokens are not counted here; the counts are then
a lower bound. A stray block that was truncated before its </think> never got split
and is still present, which is what `stray_think` measures. In non-thinking mode the
prompt already closes the think block, so both cases should be rare -- `stray_think`
is the check that this assumption held.
"""

import argparse
import json
import pathlib
import re
import statistics

from transformers import AutoTokenizer

# samples_<task>_<timestamp>.jsonl pairs with the results_<timestamp>.json written
# by the same array task; the timestamp is the join key.
_SAMPLES_RE = re.compile(r"^samples_(?P<task>.+)_(?P<ts>\d{4}-\d{2}-\d{2}T[\d\-.]+)$")


def _load_configs(root: pathlib.Path) -> dict[str, dict]:
    """timestamp -> {model, tasks: {task: max_gen_toks}} for every run under root."""
    configs: dict[str, dict] = {}
    for path in sorted(root.rglob("results_*.json")):
        blob = json.loads(path.read_text())
        configs[path.stem.removeprefix("results_")] = {
            "model": blob.get("config", {}).get("model_args", {}).get("pretrained"),
            "tasks": {
                task: cfg.get("generation_kwargs", {}).get("max_gen_toks")
                for task, cfg in blob.get("configs", {}).items()
            },
        }
    return configs


def _replies(path: pathlib.Path) -> list[str]:
    """Every generated reply in a samples file, flattened across docs and repeats.

    `resps` is a 1-element list (generate_until builds one Instance per doc) whose
    entry holds all `repeats` strings. A killed job can leave a half-written final
    line, which is skipped rather than fatal.
    """
    out: list[str] = []
    for line in path.read_text().splitlines():
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue
        for group in doc.get("resps", []):
            out.extend(group if isinstance(group, (list, tuple)) else [group])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=pathlib.Path)
    args = ap.parse_args()

    tokenizers: dict[str, object] = {}
    rows = []
    for root in args.roots:
        configs = _load_configs(root)
        for path in sorted(root.rglob("samples_*.jsonl")):
            match = _SAMPLES_RE.match(path.stem)
            if not match or match["ts"] not in configs:
                continue
            run = configs[match["ts"]]
            task = match["task"]
            model = run["model"]
            if model not in tokenizers:
                tokenizers[model] = AutoTokenizer.from_pretrained(model)

            replies = _replies(path)
            if not replies:
                continue
            # One batched call, not one per reply: ~5k replies of a few thousand
            # tokens each is seconds batched and minutes one at a time.
            lengths = [
                len(ids)
                for ids in tokenizers[model](replies, add_special_tokens=False)[
                    "input_ids"
                ]
            ]
            cap = run["tasks"].get(task)
            rows.append(
                {
                    "model": root.name,
                    "task": task,
                    "gens": len(lengths),
                    "mean": statistics.mean(lengths),
                    "median": statistics.median(lengths),
                    "p95": sorted(lengths)[int(0.95 * (len(lengths) - 1))],
                    "trunc": (
                        sum(n >= cap for n in lengths) / len(lengths) if cap else None
                    ),
                    "stray": sum(r.lstrip().startswith("<think>") for r in replies)
                    / len(replies),
                }
            )

    header = [
        "Model",
        "Task",
        "gens",
        "mean tok",
        "median",
        "p95",
        "trunc %",
        "stray <think> %",
    ]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        trunc = "-" if r["trunc"] is None else f"{100 * r['trunc']:.2f}"
        print(
            f"| {r['model']} | {r['task']} | {r['gens']} | {r['mean']:.0f} | "
            f"{r['median']:.0f} | {r['p95']} | {trunc} | {100 * r['stray']:.2f} |"
        )


if __name__ == "__main__":
    main()
