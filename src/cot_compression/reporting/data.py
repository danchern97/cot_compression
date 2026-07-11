from __future__ import annotations

import json
import math
from pathlib import Path


def _load_field(path: Path, field: str) -> dict[str, list[float]]:
    by_method: dict[str, list[float]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
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
