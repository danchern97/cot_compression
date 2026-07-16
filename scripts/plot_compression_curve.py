from __future__ import annotations

import argparse
import csv
from pathlib import Path

from cot_compression.reporting import load_summaries, plot_logprob_vs_compression

CSV_FIELDS = [
    "method",
    "method_family",
    "patching",
    "patching_param",
    "mean_compression_ratio",
    "std_compression_ratio",
    "mean_compressed_cot_tokens",
    "mean_logprob",
    "sem_logprob",
    "samples",
]


def write_csv(path: Path, summaries: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for summary in sorted(
            summaries,
            key=lambda s: (s.get("method_family", ""), s.get("mean_compression_ratio", 0)),
        ):
            writer.writerow(summary)


def run(sweep_dir: Path, out: Path) -> None:
    summaries = load_summaries(sweep_dir)
    if not summaries:
        raise SystemExit(f"No summary.json found under {sweep_dir}")
    out.mkdir(parents=True, exist_ok=True)
    plot_logprob_vs_compression(summaries, out / "logprob_vs_compression")
    write_csv(out / "compression_curve.csv", summaries)
    print(f"Wrote curve + CSV for {len(summaries)} runs to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot answer-logprob vs compression-ratio across a sweep."
    )
    parser.add_argument(
        "sweep_dir",
        type=Path,
        help="Root containing one run per point (globs **/artifacts/summary.json).",
    )
    parser.add_argument("--out", type=Path, default=Path("reports/compression_curve"))
    args = parser.parse_args()
    run(args.sweep_dir, args.out)


if __name__ == "__main__":
    main()
