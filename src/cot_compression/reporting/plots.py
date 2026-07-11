from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHOD_COLORS = {"base": "#4c72b0", "random": "#dd8452"}
METHOD_ORDER = ["base", "random"]

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


def _color(method: str) -> str:
    return METHOD_COLORS.get(method, "#999999")


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

    fig, ax = plt.subplots(figsize=(8, 6))
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
    pooled = np.concatenate([np.asarray(by_method[method]) for method in methods])

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(13, 5))

    full_bins = np.linspace(pooled.min(), pooled.max(), bins)
    for method in methods:
        values = np.asarray(by_method[method])
        color = _color(method)
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
        color = _color(method)
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
