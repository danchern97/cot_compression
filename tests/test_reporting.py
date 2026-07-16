from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

from cot_compression.reporting import (  # noqa: E402
    load_sample_xy,
    load_summaries,
    plot_logprob_vs_compression,
    plot_sample_scatter,
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
        _summary("simple_mean_uniform_ps8", "simple_mean", "uniform", "ps8", 0.12, -1.5),
        _summary("simple_mean_uniform_ps4", "simple_mean", "uniform", "ps4", 0.25, -1.3),
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
            fh.write(json.dumps({
                "method": "entropy_weighted_mean_uniform_ps8",
                "compression_ratio": 0.125,
                "logprob_mean": -1.0 - i * 0.1,
            }) + "\n")
        # a row missing compression_ratio must be skipped
        fh.write(json.dumps({"method": "base", "compression_ratio": None, "logprob_mean": -1.0}) + "\n")

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
        ("run2", _summary("simple_mean_uniform_ps8", "simple_mean", "uniform", "ps8", 0.1, -1.5)),
    ]:
        artifacts = tmp_path / name / "artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "summary.json").write_text(json.dumps({"methods": [summary]}))

    summaries = load_summaries(tmp_path)
    assert len(summaries) == 2
    assert {s["method_family"] for s in summaries} == {"base", "simple_mean"}
