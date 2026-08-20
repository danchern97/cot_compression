from __future__ import annotations

import json
import math
import statistics
from collections.abc import Container, Iterable, Sequence
from pathlib import Path
from typing import Any


def load_summaries(root: Path) -> list[dict[str, Any]]:
    """Flatten every run's per-method summary under ``root``.

    Globs ``**/artifacts/summary.json`` so a whole compression-rate sweep (one
    run per point) can be aggregated into one list of method-summary dicts.
    """
    summaries: list[dict[str, Any]] = []
    for path in sorted(Path(root).glob("**/artifacts/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        summaries.extend(payload.get("methods", []))
    return summaries


def load_sample_xy(
    root: Path,
    x_field: str = "compression_ratio",
    y_field: str = "logprob_mean",
) -> dict[str, tuple[list[float], list[float]]]:
    """Per-sample (x, y) pairs per method, from every samples.jsonl under root.

    Globs ``**/artifacts/samples.jsonl`` so a whole sweep aggregates; each
    method name (patch+compression pair) becomes one series. Rows missing
    either field (e.g. compression_ratio None for a text method) are skipped.
    """
    by_method: dict[str, tuple[list[float], list[float]]] = {}
    for path in sorted(Path(root).glob("**/artifacts/samples.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # Tolerate a truncated trailing line (e.g. a run killed
                    # mid-write); skip it rather than aborting the aggregation.
                    continue
                x, y = row.get(x_field), row.get(y_field)
                if x is None or y is None:
                    continue
                xs, ys = by_method.setdefault(row["method"], ([], []))
                xs.append(x)
                ys.append(y)
    return by_method


def load_sample_rows(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Whole per-sample rows keyed by method, from every samples.jsonl under root.

    ``load_sample_xy`` and ``load_sample_logprobs`` project out one or two fields;
    this keeps the full row, which is what re-summarizing a *subset* of samples
    needs (``subset_summary``). Accepts a run dir, a sweep root, or a single
    samples.jsonl path.
    """
    path = Path(root)
    paths = (
        [path] if path.is_file() else sorted(path.glob("**/artifacts/samples.jsonl"))
    )
    by_method: dict[str, list[dict[str, Any]]] = {}
    for samples_path in paths:
        with samples_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # Tolerate a truncated trailing line from an interrupted write.
                    continue
                by_method.setdefault(row["method"], []).append(row)
    return by_method


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else float("nan")


def _stdev(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def subset_summary(
    rows: Iterable[dict[str, Any]],
    sample_indices: Container[int] | None = None,
    population: int | None = None,
) -> dict[str, Any]:
    """Re-summarize per-sample rows over a subset, shaped like MethodSummary.

    Runs at different ``evaluation.max_examples`` cover different populations, and
    a mean is only comparable within one population. Because ``max_examples`` is a
    prefix cap over an unshuffled dataset, a longer run's rows contain a shorter
    run's population exactly, so restricting here recovers the shorter run's
    numbers offline rather than re-scoring on a GPU.

    Reproduces ``training.evaluate.summarize_method``'s formulas field for field,
    minus what per-sample rows cannot carry: ``median_token_logprob`` (streamed
    from a histogram) and the method's provenance tags (those live in
    summary.json). ``std_token_logprob`` needs ``logprob_sumsq``, absent from
    pre-2026-07-20 artifacts, and is nan there. ``skipped`` is reported only when
    ``population`` (the number of dataset indices the subset was drawn from) is
    given, since rows alone cannot say what is missing.
    """
    selected = [
        row
        for row in rows
        if sample_indices is None or row["sample_index"] in sample_indices
    ]
    logprobs = [row["logprob_mean"] for row in selected]
    summed = [row["logprob_sum"] for row in selected]
    token_counts = [float(row["answer_tokens"]) for row in selected]
    ratios = [
        row["compression_ratio"]
        for row in selected
        if row.get("compression_ratio") is not None
    ]
    compressed = [
        float(row["compressed_cot_tokens"])
        for row in selected
        if row.get("compressed_cot_tokens") is not None
    ]

    total_tokens = int(sum(token_counts))
    sumsq = [row["logprob_sumsq"] for row in selected if "logprob_sumsq" in row]
    if total_tokens:
        mean_token_logprob = sum(summed) / total_tokens
        if len(sumsq) == len(selected):
            variance = max(0.0, sum(sumsq) / total_tokens - mean_token_logprob**2)
            std_token_logprob = math.sqrt(variance)
        else:
            std_token_logprob = float("nan")
    else:
        mean_token_logprob = float("nan")
        std_token_logprob = float("nan")

    std = _stdev(logprobs)
    return {
        "method": selected[0]["method"] if selected else None,
        "samples": len(selected),
        "skipped": None if population is None else population - len(selected),
        "mean_logprob": _mean(logprobs),
        "median_logprob": _median(logprobs),
        "std_logprob": std,
        "sem_logprob": std / math.sqrt(len(logprobs)) if logprobs else float("nan"),
        "mean_logprob_sum": _mean(summed),
        "median_logprob_sum": _median(summed),
        "mean_answer_tokens": _mean(token_counts),
        "median_answer_tokens": _median(token_counts),
        "total_answer_tokens": total_tokens,
        "mean_token_logprob": mean_token_logprob,
        "std_token_logprob": std_token_logprob,
        "mean_compression_ratio": _mean(ratios),
        "median_compression_ratio": _median(ratios),
        "std_compression_ratio": _stdev(ratios),
        "mean_compressed_cot_tokens": _mean(compressed),
        "median_compressed_cot_tokens": _median(compressed),
    }


def _load_field(path: Path, field: str) -> dict[str, list[float]]:
    by_method: dict[str, list[float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Tolerate a truncated trailing line from an interrupted write.
                continue
            by_method.setdefault(row["method"], []).append(row[field])
    return by_method


def to_probabilities(by_method: dict[str, list[float]]) -> dict[str, list[float]]:
    return {
        method: [math.exp(value) for value in values]
        for method, values in by_method.items()
    }


def load_sample_logprobs(samples_path: Path) -> dict[str, list[float]]:
    """Per-sample mean answer log-probability, read from samples.jsonl."""
    return _load_field(samples_path, "logprob_mean")


def load_sample_probabilities(samples_path: Path) -> dict[str, list[float]]:
    """Per-sample answer probability, converted from samples.jsonl's logprob_mean.

    logprob_mean is the per-token average log-probability over a sample's answer
    span, so exp() of it is the geometric-mean per-token probability for that
    sample, not a raw single-token probability.
    """
    return to_probabilities(load_sample_logprobs(samples_path))


def load_token_logprobs(tokens_path: Path) -> dict[str, list[float]]:
    """Per-token answer log-probability, read from tokens.jsonl."""
    return _load_field(tokens_path, "logprob")


def load_token_probabilities(tokens_path: Path) -> dict[str, list[float]]:
    """Per-token answer probability, converted from tokens.jsonl's logprob."""
    return to_probabilities(load_token_logprobs(tokens_path))
