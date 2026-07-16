from __future__ import annotations

import json
import math
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
