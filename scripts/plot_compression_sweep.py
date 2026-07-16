"""Build distribution (box + density) and scatter plots for the compression
sweep, routed into per-method folders under reports/.

Reads per-sample data (samples.jsonl) from every run under the sweep root,
selects the relevant (patch, compression) method series for each target
folder, and writes:
  - <folder>/<prefix>_logprob_box.{png,pdf}      per-sample mean answer logprob
  - <folder>/<prefix>_logprob_density.{png,pdf}
  - <folder>/<prefix>_probability_box.{png,pdf}   exp(logprob_mean)
  - <folder>/<prefix>_probability_density.{png,pdf}
  - <folder>/<prefix>_scatter.{png,pdf}           compression ratio vs logprob

Baselines (base, random_uniform_ps1) have a constant compression ratio, so
their scatter is omitted (a vertical strip carries no information); they still
get box + density.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cot_compression.reporting import (
    load_sample_logprobs,
    load_sample_xy,
    plot_density,
    plot_mean_std_box,
    plot_sample_scatter,
    to_probabilities,
)

# Which method series belong in each report folder, and how to label them.
# method name -> short label used in plots.
BASE = {"base": "base (natural CoT)"}
RANDOM_BASELINE = {"random_uniform_ps1": "random baseline (per-token)"}
SIMPLE_MEAN = {
    "simple_mean_uniform_ps8": "uniform ps8",
    "simple_mean_exponential_exp6": "exponential exp6",
    "simple_mean_entropy_p80": "entropy p80",
}
RANDOM = {
    "random_uniform_ps8": "uniform ps8",
    "random_exponential_exp6": "exponential exp6",
    "random_entropy_p80": "entropy p80",
}

# folder (relative to --out) -> (selection, filename prefix, human title, is_baseline)
TARGETS = [
    ("baselines/natural_cot", BASE, "base", "Base (natural-language CoT)", True),
    ("baselines/random", RANDOM_BASELINE, "random_baseline",
     "Random baseline (each CoT token replaced)", True),
    ("simple_mean_compression", SIMPLE_MEAN, "simple_mean",
     "Simple-mean compression", False),
    ("random_compression", RANDOM, "random", "Random compression", False),
]


def _select(by_method, selection):
    """Keep only the requested methods (that actually have data), relabelled."""
    out = {}
    for name, label in selection.items():
        series = by_method.get(name)
        if series:
            out[label] = series
    return out


def _select_xy(by_method_xy, selection):
    out = {}
    for name, label in selection.items():
        xy = by_method_xy.get(name)
        if xy and len(xy[0]) > 0:
            out[label] = xy
    return out


def run(sweep_dir: Path, out: Path) -> None:
    sample_paths = sorted(sweep_dir.glob("**/artifacts/samples.jsonl"))
    if not sample_paths:
        raise SystemExit(f"No samples.jsonl found under {sweep_dir}")

    # Per-sample mean logprob per method, merged across all runs.
    logprobs: dict[str, list[float]] = {}
    for p in sample_paths:
        for method, values in load_sample_logprobs(p).items():
            logprobs.setdefault(method, []).extend(values)
    xy = load_sample_xy(sweep_dir, x_field="compression_ratio", y_field="logprob_mean")

    print("Per-method sample counts (from samples.jsonl):")
    for method in sorted(logprobs):
        print(f"  {method:40s} n={len(logprobs[method])}")
    print()

    for folder, selection, prefix, title, is_baseline in TARGETS:
        dest = out / folder
        dest.mkdir(parents=True, exist_ok=True)

        lp = _select(logprobs, selection)
        if not lp:
            print(f"[skip] {folder}: no data for any of {list(selection)}")
            continue
        pr = to_probabilities(lp)

        plot_mean_std_box(
            lp, title=f"{title}: per-sample answer logprob",
            ylabel="mean answer logprob", out_path=dest / f"{prefix}_logprob_box",
        )
        plot_density(
            lp, title=f"{title}: per-sample answer logprob",
            xlabel="mean answer logprob", out_path=dest / f"{prefix}_logprob_density",
        )
        plot_mean_std_box(
            pr, title=f"{title}: per-sample answer probability",
            ylabel="answer probability (geom. mean/token)",
            out_path=dest / f"{prefix}_probability_box",
        )
        plot_density(
            pr, title=f"{title}: per-sample answer probability",
            xlabel="answer probability (geom. mean/token)",
            out_path=dest / f"{prefix}_probability_density",
        )

        labels = ", ".join(f"{k} (n={len(v)})" for k, v in lp.items())
        print(f"[ok] {folder}: box+density for {labels}")

        if is_baseline:
            # Constant compression ratio -> scatter is a vertical strip; skip.
            continue
        sxy = _select_xy(xy, selection)
        if sxy:
            plot_sample_scatter(
                sxy, dest / f"{prefix}_scatter",
                xlabel="compression ratio (compressed / original CoT tokens)",
                ylabel="answer logprob (per-token mean)",
                title=f"{title}: per-sample answer logprob vs compression ratio",
            )
            print(f"      scatter for {list(sxy)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sweep_dir", type=Path, nargs="?",
        default=Path("multirun/compression_sweep"),
        help="Sweep root (globs **/artifacts/samples.jsonl).",
    )
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    run(args.sweep_dir, args.out)


if __name__ == "__main__":
    main()
