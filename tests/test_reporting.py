from __future__ import annotations

import json
import math

import matplotlib
import pytest

matplotlib.use("Agg")

from cot_compression.reporting import (  # noqa: E402
    load_sample_rows,
    load_sample_xy,
    load_summaries,
    plot_logprob_vs_compression,
    plot_sample_scatter,
    subset_summary,
)


def _summary(method, family, patching, param, ratio, logprob, sem=0.02, samples=100):
    return {
        "method": method,
        "method_family": family,
        "patching": patching,
        "patching_param": param,
        "mean_compression_ratio": ratio,
        "std_compression_ratio": 0.0,
        "mean_compressed_cot_tokens": ratio * 100,
        "mean_logprob": logprob,
        "sem_logprob": sem,
        "samples": samples,
    }


def test_plot_logprob_vs_compression(tmp_path) -> None:
    summaries = [
        _summary("base", "base", "none", "none", 1.0, -1.0),
        _summary("random_uniform_ps8", "random", "uniform", "ps8", 0.12, -2.0),
        _summary("random_uniform_ps4", "random", "uniform", "ps4", 0.25, -1.8),
        _summary(
            "simple_mean_uniform_ps8", "simple_mean", "uniform", "ps8", 0.12, -1.5
        ),
        _summary(
            "simple_mean_uniform_ps4", "simple_mean", "uniform", "ps4", 0.25, -1.3
        ),
    ]
    out = tmp_path / "curve"
    plot_logprob_vs_compression(summaries, out)
    assert out.with_suffix(".png").exists()
    assert out.with_suffix(".pdf").exists()


def test_sample_scatter_and_loader(tmp_path) -> None:
    import json

    art = tmp_path / "run" / "artifacts"
    art.mkdir(parents=True)
    with (art / "samples.jsonl").open("w") as fh:
        for i in range(5):
            fh.write(
                json.dumps(
                    {
                        "method": "entropy_weighted_mean_uniform_ps8",
                        "compression_ratio": 0.125,
                        "logprob_mean": -1.0 - i * 0.1,
                    }
                )
                + "\n"
            )
        # a row missing compression_ratio must be skipped
        fh.write(
            json.dumps(
                {"method": "base", "compression_ratio": None, "logprob_mean": -1.0}
            )
            + "\n"
        )

    by_method = load_sample_xy(tmp_path)
    assert set(by_method) == {"entropy_weighted_mean_uniform_ps8"}
    xs, ys = by_method["entropy_weighted_mean_uniform_ps8"]
    assert len(xs) == 5 and len(ys) == 5

    out = tmp_path / "scatter"
    plot_sample_scatter(by_method, out, xlabel="x", ylabel="y", title="t")
    assert out.with_suffix(".png").exists()


def test_load_summaries(tmp_path) -> None:
    for name, summary in [
        ("run1", _summary("base", "base", "none", "none", 1.0, -1.0)),
        (
            "run2",
            _summary(
                "simple_mean_uniform_ps8", "simple_mean", "uniform", "ps8", 0.1, -1.5
            ),
        ),
    ]:
        artifacts = tmp_path / name / "artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "summary.json").write_text(json.dumps({"methods": [summary]}))

    summaries = load_summaries(tmp_path)
    assert len(summaries) == 2
    assert {s["method_family"] for s in summaries} == {"base", "simple_mean"}


def _sample_row(index, logprob_mean, tokens=10, ratio=0.5):
    return {
        "method": "simple_mean_uniform_cr2",
        "sample_index": index,
        "sample_id": f"id{index}",
        "dataset_source": "src",
        "answer_tokens": tokens,
        "logprob_sum": logprob_mean * tokens,
        "logprob_mean": logprob_mean,
        "logprob_sumsq": logprob_mean**2 * tokens,
        "compressed_cot_tokens": int(ratio * 100),
        "compression_ratio": ratio,
    }


def test_subset_summary_matches_summarize_method(tmp_path) -> None:
    """subset_summary must reproduce the eval loop's own summary, field for field.

    This is the contract that lets a 20k run be compared against a 10k run without
    re-scoring: restricting to the 10k prefix has to give exactly what a 10k run
    would have written.
    """
    from dataclasses import asdict

    from cot_compression.compression import SimpleMeanCompressionMethod
    from cot_compression.patching import UniformPatchingMethod
    from cot_compression.training.evaluate import SampleScore, summarize_method

    rows = [_sample_row(i, -1.0 - 0.01 * i) for i in range(20)]
    method = SimpleMeanCompressionMethod(
        patching=UniformPatchingMethod(compression_ratio=2.0)
    )
    prefix = [row for row in rows if row["sample_index"] < 10]
    expected = asdict(
        summarize_method(
            method,
            [SampleScore(**row) for row in prefix],
            skipped=3,
        )
    )

    got = subset_summary(rows, sample_indices=set(range(10)), population=13)

    assert got["method"] == expected["method"]
    for field in (
        "samples",
        "skipped",
        "mean_logprob",
        "median_logprob",
        "std_logprob",
        "sem_logprob",
        "mean_logprob_sum",
        "median_logprob_sum",
        "mean_answer_tokens",
        "median_answer_tokens",
        "total_answer_tokens",
        "mean_token_logprob",
        "std_token_logprob",
        "mean_compression_ratio",
        "median_compression_ratio",
        "std_compression_ratio",
        "mean_compressed_cot_tokens",
        "median_compressed_cot_tokens",
    ):
        assert got[field] == pytest.approx(expected[field]), field


def test_subset_summary_tolerates_legacy_rows_without_sumsq() -> None:
    """Pre-2026-07-20 samples.jsonl has no logprob_sumsq; only that stat is lost."""
    rows = [_sample_row(i, -1.0) for i in range(4)]
    for row in rows:
        del row["logprob_sumsq"]

    got = subset_summary(rows)

    assert got["samples"] == 4
    assert got["mean_logprob"] == pytest.approx(-1.0)
    assert got["mean_token_logprob"] == pytest.approx(-1.0)
    assert math.isnan(got["std_token_logprob"])
    assert got["skipped"] is None


def test_load_sample_rows_globs_a_sweep(tmp_path) -> None:
    for name, index in [("run1", 0), ("run2", 1)]:
        artifacts = tmp_path / name / "artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "samples.jsonl").write_text(
            json.dumps(_sample_row(index, -1.0)) + "\n"
        )

    by_method = load_sample_rows(tmp_path)

    assert set(by_method) == {"simple_mean_uniform_cr2"}
    assert len(by_method["simple_mean_uniform_cr2"]) == 2
