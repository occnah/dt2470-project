"""
Figures for results/: detection curves and per-drummer heatmaps.

Colours follow a fixed categorical order (validated for colour-vision
deficiency); every series also has its own marker, so no line is
identified by colour alone.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "groove-matplotlib"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.common.config import RELIABLE_F1  # noqa: E402
from src.data.conditions import BY_NAME, CONDITIONS, JITTER_KINDS  # noqa: E402

SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MARKERS = ("o", "s", "^", "D")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})


def _style(ax, ylabel: str | None = "F1 for edited clips") -> None:
    ax.set_ylim(-0.03, 1.05)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(RELIABLE_F1, color=MUTED, linewidth=1, linestyle="--")
    if ylabel:
        ax.set_ylabel(ylabel)


def _rows(summary: pd.DataFrame, model: str, names: list[str]) -> pd.DataFrame:
    return summary[summary.model == model].set_index("condition").reindex(names)


def plot_quantization(summary: pd.DataFrame, models: list[tuple[str, str]], path: Path) -> None:
    """F1 against 1/16 quantization strength and velocity normalization strength."""
    panels = [
        ("quantize_16", "Timing: movement toward the 1/16 grid (%)"),
        ("velocity", "Velocity: movement toward a constant (%)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, layout="constrained")
    for ax, (series, xlabel) in zip(axes, panels):
        names = [c.name for c in CONDITIONS if c.series == series]
        x = [BY_NAME[n].strength * 100 for n in names]
        for i, (model, label) in enumerate(models):
            rows = _rows(summary, model, names)
            ax.errorbar(
                x, rows.drummer_mean_f1, yerr=rows.drummer_sd_f1, color=SERIES_COLORS[i],
                marker=MARKERS[i], markersize=6, linewidth=2, capsize=3, label=label,
            )
        ax.set_xlabel(xlabel)
        ax.set_xticks(x)
        ax.tick_params(axis="x", labelsize=8)
        _style(ax, "F1 for edited clips" if ax is axes[0] else None)
    axes[0].legend(loc="lower right")
    fig.suptitle("RQ1: detection of quantization (mean ± SD over held-out drummers)", color=INK)
    fig.text(0.5, -0.04, f"Dashed line: reliability criterion F1 = {RELIABLE_F1} "
             "(must hold for every held-out drummer, not only on average).", ha="center", color=MUTED, fontsize=9)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_humanization(summary: pd.DataFrame, models: list[tuple[str, str]], path: Path) -> None:
    """F1 against jitter size for each jitter kind, one panel per model."""
    levels = ["5ms", "10ms", "20ms", "matched"]
    ticks = ["5 ms", "10 ms", "20 ms", "matched", "shuffled"]
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 4), sharey=True, layout="constrained")
    for ax, (model, label) in zip(np.atleast_1d(axes), models):
        for i, kind in enumerate(JITTER_KINDS):
            rows = _rows(summary, model, [f"humanize_{kind}_{level}" for level in levels])
            ax.errorbar(
                range(len(levels)), rows.drummer_mean_f1, yerr=rows.drummer_sd_f1,
                color=SERIES_COLORS[i], marker=MARKERS[i], markersize=6, linewidth=2,
                capsize=3, label=f"{kind} jitter",
            )
        shuffled = _rows(summary, model, ["deviations_shuffled"])
        ax.errorbar(
            [len(levels)], shuffled.drummer_mean_f1, yerr=shuffled.drummer_sd_f1,
            color=MUTED, marker="X", markersize=8, capsize=3, linestyle="none", label="shuffled human",
        )
        ax.axvline(len(levels) - 0.5, color=GRID, linewidth=1.5)  # shuffled is a control, not a jitter size
        ax.set_xticks(range(len(ticks)), ticks, fontsize=8)
        ax.set_title(label, color=INK, fontsize=10)
        ax.set_xlabel("Jitter SD  |  control")
        _style(ax, "F1 for edited clips" if ax is np.atleast_1d(axes)[0] else None)
    np.atleast_1d(axes)[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("RQ2/RQ3: detection of humanization after full 1/16 quantization", color=INK)
    fig.text(0.5, -0.05, "matched: jitter scaled to each clip's human timing mean and SD. "
             "shuffled: the clip's own human deviations in random order.", ha="center", color=MUTED, fontsize=9)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_f1_by_drummer(folds: pd.DataFrame, models: list[tuple[str, str]], path: Path) -> None:
    """Heatmap of F1 per condition (rows) and held-out drummer (columns), one panel per model."""
    names = [c.name for c in CONDITIONS if c.name in set(folds.condition)]
    drummers = sorted(folds.test_drummer.unique(), key=lambda d: int("".join(filter(str.isdigit, d)) or 0))
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 0.3 * len(names) + 1.8),
                             sharey=True, layout="constrained")
    for ax, (model, label) in zip(np.atleast_1d(axes), models):
        values = folds[folds.model == model].pivot(index="condition", columns="test_drummer", values="f1")
        values = values.reindex(index=names, columns=drummers)
        image = ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for (i, j), value in np.ndenumerate(values.to_numpy()):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if value > 0.6 else INK)
        ax.set_xticks(range(len(drummers)), ["".join(filter(str.isdigit, d)) for d in drummers], fontsize=8)
        ax.set_yticks(range(len(names)), [BY_NAME[n].label for n in names], fontsize=7)
        ax.set_xlabel("Held-out drummer")
        ax.set_title(label, color=INK, fontsize=10)
        ax.spines[:].set_visible(False)
    fig.colorbar(image, ax=axes, shrink=0.6, label="F1 for edited clips")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(summary: pd.DataFrame, feature_sets: list[str], conditions: list[str],
                  model: str, path: Path) -> None:
    """Grouped bars: F1 per condition (groups) and feature set (bars) for one model."""
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(conditions)), 4.2), layout="constrained")
    width = 0.8 / len(feature_sets)
    for i, feature_set in enumerate(feature_sets):
        rows = summary[(summary.model == model) & (summary.feature_set == feature_set)]
        rows = rows.set_index("condition").reindex(conditions)
        ax.bar(np.arange(len(conditions)) + (i - (len(feature_sets) - 1) / 2) * width,
               rows.drummer_mean_f1, width * 0.92, color=SERIES_COLORS[i % len(SERIES_COLORS)],
               hatch=["", "//", "..", "xx"][i % 4], edgecolor="white", label=feature_set)
    ax.set_xticks(range(len(conditions)), [BY_NAME[c].label for c in conditions],
                  rotation=35, ha="right", fontsize=8)
    _style(ax)
    ax.legend(ncols=len(feature_sets), loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=8)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
