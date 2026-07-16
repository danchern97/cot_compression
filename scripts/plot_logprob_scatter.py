from __future__ import annotations

import argparse
from pathlib import Path

from cot_compression.reporting import load_sample_xy, plot_sample_scatter


def run(sweep_dir: Path, out: Path) -> None:
    by_method = load_sample_xy(
        sweep_dir, x_field="compression_ratio", y_field="logprob_mean"
    )
    if not by_method:
        raise SystemExit(f"No usable samples.jsonl found under {sweep_dir}")
    out.mkdir(parents=True, exist_ok=True)
    plot_sample_scatter(
        by_method,
        out / "logprob_vs_compression_scatter",
        xlabel="compression ratio (compressed / original CoT tokens)",
        ylabel="answer logprob (per-token mean)",
        title="Per-sample answer logprob vs CoT compression ratio",
    )
    total = sum(len(xs) for xs, _ in by_method.values())
    print(
        f"Wrote scatter for {len(by_method)} (patch, compression) pairs, "
        f"{total} points, to {out}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-sample scatter of answer logprob vs CoT compression ratio, "
        "one color per (patch, compression) pair."
    )
    parser.add_argument(
        "sweep_dir",
        type=Path,
        help="Root containing runs (globs **/artifacts/samples.jsonl).",
    )
    parser.add_argument("--out", type=Path, default=Path("reports/logprob_scatter"))
    args = parser.parse_args()
    run(args.sweep_dir, args.out)


if __name__ == "__main__":
    main()
