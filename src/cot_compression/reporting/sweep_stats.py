from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

# Log-prob histogram grid. Answer log-probs are <= 0 and concentrate near 0;
# -30 covers all but a negligible tail, which is folded into an underflow count
# so quantiles below P0.01 stay correct. 0.01-wide bins make every quantile we
# report accurate to +/-0.005 nats, far below any effect size of interest.
HIST_LO = -30.0
HIST_HI = 0.0
HIST_BINS = 3000

# Per-answer-position curves are tracked for the first POS_MAX tokens; the mean
# answer is ~523 tokens, so this covers essentially the whole answer for most
# samples without an unbounded accumulator.
POS_MAX = 512


@dataclass
class MethodAccumulator:
    """Streaming accumulator for one method's token-level log-probs."""

    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    under: int = 0
    hist: np.ndarray = field(
        default_factory=lambda: np.zeros(HIST_BINS, dtype=np.int64)
    )
    pos_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(POS_MAX, dtype=np.float64)
    )
    pos_cnt: np.ndarray = field(
        default_factory=lambda: np.zeros(POS_MAX, dtype=np.int64)
    )

    def merge(self, other: MethodAccumulator) -> None:
        self.n += other.n
        self.total += other.total
        self.total_sq += other.total_sq
        self.under += other.under
        self.hist += other.hist
        self.pos_sum += other.pos_sum
        self.pos_cnt += other.pos_cnt

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "total": self.total,
            "total_sq": self.total_sq,
            "under": self.under,
            "hist": self.hist.tolist(),
            "pos_sum": self.pos_sum.tolist(),
            "pos_cnt": self.pos_cnt.tolist(),
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> MethodAccumulator:
        acc = MethodAccumulator(
            n=payload["n"],
            total=payload["total"],
            total_sq=payload["total_sq"],
            under=payload["under"],
        )
        acc.hist = np.asarray(payload["hist"], dtype=np.int64)
        acc.pos_sum = np.asarray(payload["pos_sum"], dtype=np.float64)
        acc.pos_cnt = np.asarray(payload["pos_cnt"], dtype=np.int64)
        return acc


_SCALE = HIST_BINS / (HIST_HI - HIST_LO)


def _parse_tokens_file(path: str) -> dict[str, dict[str, Any]]:
    """Stream one tokens.jsonl, accumulating per-method token stats.

    Uses byte-level field slicing rather than json.loads: the writer emits a
    fixed key order (method, sample_index, token_index, token_id, logprob), and
    at ~363M rows across the sweep a full JSON parse is the difference between
    minutes and hours. Falls back to json.loads for any line that doesn't match
    the expected shape, so a format change degrades to slow-but-correct.

    Answer-relative position is the running row count within each
    (method, sample_index) group: evaluate.py emits token rows in ascending
    position order, so the k-th row for a sample is its k-th answer token.
    Absolute token_index cannot be used for this — it includes the prompt and
    the (differently sized) compressed CoT, so it is not comparable across
    methods.
    """
    accs: dict[str, MethodAccumulator] = {}
    malformed = 0
    prev_key: bytes | None = None
    pos = 0
    with open(path, "rb") as handle:
        for line in handle:
            try:
                i = line.index(b'"method": "', 0) + 11
                j = line.index(b'"', i)
                method = line[i:j]
                i2 = line.index(b'"sample_index": ', j) + 16
                j2 = line.index(b",", i2)
                sample = line[i2:j2]
                i3 = line.rindex(b'"logprob": ') + 11
                logprob = float(line[i3 : line.rindex(b"}")])
            except (ValueError, IndexError):
                # A run killed mid-write (e.g. disk quota) leaves a truncated
                # final line. Skip it rather than aborting the aggregation, the
                # same tolerance reporting/data.py applies.
                malformed += 1
                continue

            key = method + b"\x00" + sample
            if key != prev_key:
                prev_key = key
                pos = 0

            name = method.decode()
            acc = accs.get(name)
            if acc is None:
                acc = accs[name] = MethodAccumulator()

            acc.n += 1
            acc.total += logprob
            acc.total_sq += logprob * logprob
            index = int((logprob - HIST_LO) * _SCALE)
            if index < 0:
                acc.under += 1
            else:
                acc.hist[min(index, HIST_BINS - 1)] += 1
            if pos < POS_MAX:
                acc.pos_sum[pos] += logprob
                acc.pos_cnt[pos] += 1
            pos += 1
    result = {name: acc.to_dict() for name, acc in accs.items()}
    result["__malformed__"] = {"path": path, "count": malformed}
    return result


def aggregate_token_stats(
    root: Path, workers: int = 8
) -> tuple[dict[str, MethodAccumulator], list[dict[str, Any]]]:
    """Aggregate token-level stats across every tokens.jsonl under ``root``.

    Returns the per-method accumulators and a list of per-file malformed-line
    reports, so a truncated artifact surfaces in the report instead of silently
    biasing a distribution.
    """
    paths = sorted(str(p) for p in Path(root).glob("**/artifacts/tokens.jsonl"))
    merged: dict[str, MethodAccumulator] = {}
    damage: list[dict[str, Any]] = []
    with Pool(processes=workers) as pool:
        for per_file in pool.imap_unordered(_parse_tokens_file, paths):
            report = per_file.pop("__malformed__")
            if report["count"]:
                damage.append(report)
            for name, payload in per_file.items():
                acc = MethodAccumulator.from_dict(payload)
                if name in merged:
                    merged[name].merge(acc)
                else:
                    merged[name] = acc
    return merged, damage


def hist_quantiles(
    acc: MethodAccumulator, quantiles: tuple[float, ...]
) -> dict[str, float]:
    """Quantiles from the binned log-prob histogram, linearly interpolated.

    The underflow count is treated as mass at HIST_LO, so a quantile falling in
    the far tail is reported at the grid edge rather than silently shifted.
    """
    edges = np.linspace(HIST_LO, HIST_HI, HIST_BINS + 1)
    counts = acc.hist
    total = acc.n
    cum = np.cumsum(counts) + acc.under
    out: dict[str, float] = {}
    for q in quantiles:
        target = q * total
        if target <= acc.under:
            out[f"p{q * 100:g}"] = HIST_LO
            continue
        index = int(np.searchsorted(cum, target, side="left"))
        index = min(index, HIST_BINS - 1)
        before = cum[index - 1] if index > 0 else acc.under
        width = counts[index]
        frac = (target - before) / width if width > 0 else 0.0
        out[f"p{q * 100:g}"] = float(
            edges[index] + frac * (edges[index + 1] - edges[index])
        )
    return out


DENSITY_LO = -12.0
DENSITY_BINS = 240


def _density(acc: MethodAccumulator) -> dict[str, Any]:
    """Coarse density over [DENSITY_LO, 0] for the distribution plots.

    Rebins the fine histogram by an integer factor so the curve stays cheap to
    embed. Mass below DENSITY_LO is returned separately rather than dropped, so
    a reader can see how much tail the axis excludes.
    """
    edges = np.linspace(HIST_LO, HIST_HI, HIST_BINS + 1)
    start = int(np.searchsorted(edges, DENSITY_LO))
    tail = int(acc.hist[:start].sum() + acc.under)
    body = acc.hist[start:]
    factor = len(body) // DENSITY_BINS
    trimmed = body[: factor * DENSITY_BINS].reshape(DENSITY_BINS, factor).sum(axis=1)
    return {
        "lo": DENSITY_LO,
        "hi": HIST_HI,
        "counts": trimmed.astype(int).tolist(),
        "below_lo": tail,
    }


def token_summary(acc: MethodAccumulator) -> dict[str, Any]:
    mean = acc.total / acc.n if acc.n else float("nan")
    var = (acc.total_sq / acc.n - mean * mean) if acc.n else float("nan")
    quantiles = hist_quantiles(acc, (0.05, 0.25, 0.5, 0.75, 0.95))
    valid = acc.pos_cnt > 0
    positions = np.where(valid)[0]
    pos_mean = np.divide(
        acc.pos_sum, acc.pos_cnt, out=np.zeros_like(acc.pos_sum), where=valid
    )
    # Mean *probability* is E[exp(logprob)], which is not exp(E[logprob]) --
    # Jensen's inequality makes the latter strictly smaller. Estimate it from
    # the histogram's bin centers so the probability box plots can carry a true
    # arithmetic mean rather than a geometric one.
    centers = np.linspace(HIST_LO, HIST_HI, HIST_BINS + 1)[:-1] + (
        (HIST_HI - HIST_LO) / HIST_BINS / 2
    )
    mean_prob = (
        float((np.exp(centers) * acc.hist).sum() / acc.n) if acc.n else float("nan")
    )
    return {
        "n": acc.n,
        "mean": mean,
        "std": float(np.sqrt(max(var, 0.0))),
        "mean_prob": mean_prob,
        **quantiles,
        "density": _density(acc),
        "pos_index": positions.tolist(),
        "pos_mean": [float(pos_mean[i]) for i in positions],
        "pos_count": [int(acc.pos_cnt[i]) for i in positions],
    }
