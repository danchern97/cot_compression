"""Merge the per-benchmark lm-eval result JSONs for one or more models.

    uv run --project evals python evals/summarize.py results/benchmarks/qwen3-4b-raw
    uv run --project evals python evals/summarize.py results/benchmarks/*

Each array task of run_benchmarks.sh writes its own results_<timestamp>.json, so
one model's numbers arrive as four files; this prints them as a single markdown
table. With several roots the models become columns, which is the raw-vs-SFT view.
"""

import argparse
import json
import pathlib

# Display order and labels; anything unlisted still prints, at the end.
LABELS = {
    ("cotc_math500", "math_verify"): "MATH-500 (avg@4)",
    ("cotc_aime25", "math_verify"): "AIME'25 (avg@64)",
    ("cotc_aime25_16", "math_verify"): "AIME'25 (avg@16)",
    ("cotc_gpqa_diamond", "exact_match"): "GPQA-Diamond (avg@10)",
    ("cotc_hotpotqa", "exact_match"): "HotpotQA EM",
    ("cotc_hotpotqa", "f1"): "HotpotQA F1",
    # Non-thinking mode (enable_thinking=False) with standard CoT prompting, the
    # setting of arXiv 2604.22709. Separate rows, not another column on the rows
    # above: the sampler differs (Qwen3's non-thinking t=0.7/top-p=0.8), and
    # HotpotQA is the paper's 500-question subset with a CoT trigger appended, so
    # neither is the same protocol as its thinking-mode namesake.
    ("cotc_nothink_math500", "math_verify"): "MATH-500 (avg@4, no-think)",
    ("cotc_nothink_aime25_16", "math_verify"): "AIME'25 (avg@16, no-think)",
    ("cotc_nothink_gpqa_diamond", "exact_match"): "GPQA-Diamond (avg@10, no-think)",
    ("cotc_nothink_hotpotqa500", "exact_match"): "HotpotQA EM (500, no-think)",
    ("cotc_nothink_hotpotqa500", "f1"): "HotpotQA F1 (500, no-think)",
}


def collect(
    root: pathlib.Path,
) -> dict[tuple[str, str], tuple[float, float | None, int]]:
    """(task, metric) -> (value, stderr, n) for the newest run of each task."""
    out: dict[tuple[str, str], tuple[float, float | None, int]] = {}
    for path in sorted(root.rglob("results_*.json")):
        blob = json.loads(path.read_text())
        for task, res in blob.get("results", {}).items():
            n = blob.get("n-samples", {}).get(task, {}).get("effective", 0)
            for key, value in res.items():
                # Real metrics are always "<metric>,<filter>"; the bare keys
                # alongside them ("name", "alias", "sample_len") are not metrics
                # and would otherwise crash or render as a junk row.
                if "," not in key or not isinstance(value, (int, float)):
                    continue
                metric, _, flt = key.partition(",")
                if metric.endswith("_stderr"):
                    continue
                stderr = res.get(f"{metric}_stderr,{flt}")
                out[task, metric] = (
                    value,
                    stderr if isinstance(stderr, (int, float)) else None,
                    n,
                )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", type=pathlib.Path)
    args = ap.parse_args()

    models = {root.name: collect(root) for root in args.roots}
    keys = list(LABELS) + sorted({k for m in models.values() for k in m} - set(LABELS))

    header = ["Benchmark", "n", *models]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for key in keys:
        cells = []
        n = 0
        for scores in models.values():
            if key not in scores:
                cells.append("-")
                continue
            value, stderr, n = scores[key]
            cells.append(
                f"{100 * value:.1f}" + (f" ± {100 * stderr:.1f}" if stderr else "")
            )
        if set(cells) == {"-"}:
            continue
        print(
            f"| {LABELS.get(key, ' '.join(key))} | {n or '-'} | "
            + " | ".join(cells)
            + " |"
        )


if __name__ == "__main__":
    main()
