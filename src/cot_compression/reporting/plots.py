from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

METHOD_COLORS = {"base": "#4c72b0", "random": "#dd8452"}
METHOD_ORDER = ["base", "random"]
BASELINE_FAMILIES = ("base", "random")

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def _ordered_methods(by_method: dict[str, list[float]]) -> list[str]:
    ordered = [method for method in METHOD_ORDER if method in by_method]
    ordered += [method for method in by_method if method not in ordered]
    return ordered


def _method_colors(methods: list[str]) -> dict[str, str]:
    """Assign each method a distinct color: known methods keep their fixed
    color, anything else cycles through a qualitative palette so an
    arbitrary number of methods (e.g. per-patching-strategy variants) never
    collide on the same fallback gray.
    """
    palette = plt.get_cmap("tab10").colors
    colors = {}
    next_index = 0
    for method in methods:
        if method in METHOD_COLORS:
            colors[method] = METHOD_COLORS[method]
            continue
        colors[method] = palette[next_index % len(palette)]
        next_index += 1
    return colors


def _save(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_mean_std_box(
    by_method: dict[str, list[float]], title: str, ylabel: str, out_path: Path
) -> None:
    methods = _ordered_methods(by_method)
    data = [by_method[method] for method in methods]

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(methods)), 6))
    ax.boxplot(
        data,
        tick_labels=methods,
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops={"facecolor": "#a8d0e6", "alpha": 0.7},
        medianprops={"color": "firebrick", "linewidth": 2},
        meanprops={"color": "black", "linewidth": 2, "linestyle": "--"},
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    ax.set_ylim(ymin, ymax + 0.18 * yrange)
    for i, values in enumerate(data, start=1):
        mean = float(np.mean(values))
        std = float(np.std(values))
        ax.text(
            i,
            ymax + 0.03 * yrange,
            f"mean={mean:.3f}\nstd={std:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out_path)


def plot_density(
    by_method: dict[str, list[float]],
    title: str,
    xlabel: str,
    out_path: Path,
    bins: int = 80,
    zoom_percentiles: tuple[float, float] = (0.5, 99.5),
) -> None:
    methods = _ordered_methods(by_method)
    colors = _method_colors(methods)
    pooled = np.concatenate([np.asarray(by_method[method]) for method in methods])

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(13, 5))

    full_bins = np.linspace(pooled.min(), pooled.max(), bins)
    for method in methods:
        values = np.asarray(by_method[method])
        color = colors[method]
        mean, std = float(values.mean()), float(values.std())
        ax_full.hist(
            values,
            bins=full_bins,
            alpha=0.55,
            density=True,
            color=color,
            label=f"{method} (mean={mean:.3f}, std={std:.3f})",
        )
        ax_full.axvline(mean, color=color, linestyle="--", linewidth=2)
    ax_full.set_xlabel(xlabel)
    ax_full.set_ylabel("density")
    ax_full.set_title("Full range")
    ax_full.legend(fontsize=9)
    ax_full.spines[["top", "right"]].set_visible(False)

    lo, hi = np.percentile(pooled, zoom_percentiles)
    zoom_bins = np.linspace(lo, hi, bins)
    for method in methods:
        values = np.asarray(by_method[method])
        color = colors[method]
        ax_zoom.hist(
            values, bins=zoom_bins, alpha=0.55, density=True, color=color, label=method
        )
        ax_zoom.axvline(float(values.mean()), color=color, linestyle="--", linewidth=2)
    ax_zoom.set_xlabel(xlabel)
    ax_zoom.set_ylabel("density")
    ax_zoom.set_title(
        f"Zoomed ({zoom_percentiles[0]}-{zoom_percentiles[1]} percentile)"
    )
    ax_zoom.legend(fontsize=9)
    ax_zoom.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    _save(fig, out_path)


def plot_sample_scatter(
    by_method: dict[str, tuple[list[float], list[float]]],
    out_path: Path,
    xlabel: str,
    ylabel: str,
    title: str,
    alpha: float = 0.15,
    marker_size: float = 6.0,
) -> None:
    """Per-sample scatter, one translucent color per method (patch+compression).

    Points are rasterized so the PDF stays small despite tens of thousands of
    markers; legend markers are drawn opaque so colors stay readable.
    """
    methods = _ordered_methods({method: [] for method in by_method})
    colors = _method_colors(methods)

    fig, ax = plt.subplots(figsize=(9, 6))
    for method in methods:
        xs, ys = by_method[method]
        ax.scatter(
            xs,
            ys,
            s=marker_size,
            alpha=alpha,
            color=colors[method],
            edgecolors="none",
            rasterized=True,
            label=f"{method} (n={len(xs)})",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    legend = ax.legend(fontsize=8, markerscale=2.5)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out_path)


def plot_logprob_vs_compression(
    summaries: list[dict[str, Any]],
    out_path: Path,
    title: str = "Answer logprob vs CoT compression ratio",
) -> None:
    """Sweep curve: X=mean compression ratio, Y=mean answer logprob (sem bars).

    One line per method family, ordered by compression ratio. ``base`` (full
    CoT) is a horizontal dashed ceiling; ``random`` a dashed swept floor; other
    families solid. Expects the summary dicts written to summary.json.
    """
    by_family: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        by_family.setdefault(summary["method_family"], []).append(summary)
    families = _ordered_methods({family: [] for family in by_family})
    colors = _method_colors(families)

    fig, ax = plt.subplots(figsize=(8, 6))
    for family in families:
        points = sorted(
            by_family[family],
            key=lambda s: s.get("mean_compression_ratio", math.nan),
        )
        xs = [s.get("mean_compression_ratio", math.nan) for s in points]
        ys = [s["mean_logprob"] for s in points]
        errs = [s.get("sem_logprob", 0.0) for s in points]
        color = colors[family]
        if family == "base":
            ax.axhline(
                ys[0], color=color, linestyle="--", linewidth=2, label="base (full CoT)"
            )
        else:
            ax.errorbar(
                xs,
                ys,
                yerr=errs,
                color=color,
                marker="o",
                capsize=3,
                linestyle="--" if family in BASELINE_FAMILIES else "-",
                label=family,
            )

    ax.set_xlabel("compression ratio (compressed / original CoT tokens)")
    ax.set_ylabel("mean answer logprob")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out_path)
