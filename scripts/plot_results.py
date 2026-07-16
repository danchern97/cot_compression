from __future__ import annotations

import argparse
from pathlib import Path

import submitit

from cot_compression.reporting import (
    load_sample_logprobs,
    load_token_logprobs,
    plot_density,
    plot_mean_std_box,
    to_probabilities,
)


def merge(*by_method: dict[str, list[float]]) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for mapping in by_method:
        merged.update(mapping)
    return merged


def run(multirun_dir: Path, out: Path) -> None:
    # Recursive glob so multiple multirun sweeps can be nested under one
    # root directory (e.g. multirun_dir/<compression_mode>/<job_num>/artifacts)
    # without colliding on Hydra's per-invocation job numbering.
    artifacts_dirs = sorted(multirun_dir.glob("**/artifacts"))
    sample_paths = [
        d / "samples.jsonl" for d in artifacts_dirs if (d / "samples.jsonl").exists()
    ]
    token_paths = [
        d / "tokens.jsonl" for d in artifacts_dirs if (d / "tokens.jsonl").exists()
    ]
    if not sample_paths:
        raise SystemExit(f"No samples.jsonl found under {multirun_dir}")

    sample_logprobs = merge(*(load_sample_logprobs(p) for p in sample_paths))
    sample_probabilities = to_probabilities(sample_logprobs)

    plot_mean_std_box(
        sample_probabilities,
        title="Distribution of sample probability by method",
        ylabel="Probability",
        out_path=out / "sample_probability_box",
    )
    plot_density(
        sample_probabilities,
        title="Distribution of sample probability by method",
        xlabel="Probability",
        out_path=out / "sample_probability_density",
    )
    plot_mean_std_box(
        sample_logprobs,
        title="Distribution of logprob_mean by method",
        ylabel="logprob_mean",
        out_path=out / "sample_logprob_box",
    )
    plot_density(
        sample_logprobs,
        title="Distribution of logprob_mean by method",
        xlabel="logprob_mean",
        out_path=out / "sample_logprob_density",
    )

    # Token-level plots require tokens.jsonl (written only when
    # save_token_logprobs is set); skip them cleanly when it's absent.
    if not token_paths:
        print(f"Wrote sample-level plots to {out} (no tokens.jsonl -> skipped token plots)")
        return

    token_logprobs = merge(*(load_token_logprobs(p) for p in token_paths))
    token_probabilities = to_probabilities(token_logprobs)
    plot_mean_std_box(
        token_probabilities,
        title="Distribution of token probability by method",
        ylabel="Probability",
        out_path=out / "token_probability_box",
    )
    plot_density(
        token_probabilities,
        title="Distribution of token probability by method",
        xlabel="Probability",
        out_path=out / "token_probability_density",
    )
    plot_mean_std_box(
        token_logprobs,
        title="Distribution of logprob by method",
        ylabel="logprob",
        out_path=out / "token_logprob_box",
    )
    plot_density(
        token_logprobs,
        title="Distribution of logprob by method",
        xlabel="logprob",
        out_path=out / "token_logprob_density",
    )
    print(f"Wrote plots to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot answer-probability results from a Hydra multirun's eval jobs."
    )
    parser.add_argument(
        "multirun_dir",
        type=Path,
        help="Multirun directory containing one artifacts/ folder per job.",
    )
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit as a SLURM job via submitit instead of running locally.",
    )
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--partition", default="gpu_a100")
    parser.add_argument("--timeout-min", type=int, default=15)
    args = parser.parse_args()

    if args.submit:
        executor = submitit.AutoExecutor(folder="outputs/submitit_logs/plot_results")
        executor.update_parameters(
            slurm_partition=args.partition,
            gpus_per_node=1,
            cpus_per_task=4,
            mem_gb=16,
            timeout_min=args.timeout_min,
        )
        job = executor.submit(run, args.multirun_dir, args.out)
        print(f"Submitted job {job.job_id}")
        return

    run(args.multirun_dir, args.out)


if __name__ == "__main__":
    main()
