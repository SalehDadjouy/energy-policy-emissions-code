#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "panel_lag2.csv"
FIGURE_DIR = ROOT / "outputs" / "figures"

TITLE_SIZE = 20
PANEL_TITLE_SIZE = 18
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 14
STATE_LABEL_SIZE = 13
ANNOTATION_LABEL_SIZE = 8

LINE_COLOR = "#b8b8b8"
POINT_COLOR = "#7a7a7a"
HIGHLIGHT_COLOR = "#111111"
GRID_COLOR = "#d4d4d4"
ZERO_COLOR = "#b0b0b0"


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": TICK_LABEL_SIZE,
        "axes.titlesize": PANEL_TITLE_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "figure.titlesize": TITLE_SIZE,
        "axes.edgecolor": "#222222",
        "axes.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def ordered_exposure(data_path: Path = DATA) -> pd.DataFrame:
    """Estimate the full-panel state exposure coefficients plotted in the paper."""
    panel = pd.read_csv(data_path).dropna(subset=["W_it_lag2", "Z_t_lag2"])
    panel = panel[panel["year"].between(2008, 2022)].copy()
    years = np.arange(2008, 2023, dtype=int)
    z = (
        panel.drop_duplicates("year")
        .set_index("year")
        .reindex(years)["Z_t_lag2"]
        .to_numpy(dtype=float)
    )
    if z.size != years.size or not np.all(np.isfinite(z)):
        raise RuntimeError("The aggregate instrument path is incomplete for 2008-2022.")
    zc = z - float(z.mean())
    denominator = float(zc @ zc)
    rows: list[dict[str, float | str]] = []
    for state, group in panel.groupby("state", sort=True):
        w = (
            group.set_index("year")
            .reindex(years)["W_it_lag2"]
            .to_numpy(dtype=float)
        )
        if w.size != years.size or not np.all(np.isfinite(w)):
            raise RuntimeError(f"Treatment path is incomplete for {state}.")
        rows.append(
            {
                "state": state,
                "D_full": float(((w - float(w.mean())) @ zc) / denominator),
            }
        )
    result = pd.DataFrame(rows).sort_values("D_full").reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", length=5, width=1.2)


def draw_vertical_lollipop(ax: plt.Axes, df: pd.DataFrame, label_focus: bool) -> None:
    ax.vlines(df["rank"], 0, df["D_full"], color=LINE_COLOR, linewidth=1.25, zorder=1)
    ax.scatter(
        df["rank"],
        df["D_full"],
        s=30,
        facecolors="white",
        edgecolors=POINT_COLOR,
        linewidths=1.5,
        zorder=3,
    )
    focus = df[df["state"].isin(["CA", "VT"])]
    ax.scatter(
        focus["rank"],
        focus["D_full"],
        s=58,
        facecolors="white",
        edgecolors=HIGHLIGHT_COLOR,
        linewidths=2.1,
        zorder=4,
    )
    ax.axhline(0, color=ZERO_COLOR, linestyle=":", linewidth=1.6, zorder=0)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=1.0)
    ax.set_xlim(0.5, len(df) + 0.5)
    ax.set_ylim(-75, 205)
    ax.set_xticks([10, 20, 30, 40, 50])
    ax.set_yticks([-50, 0, 50, 100, 150, 200])
    ax.set_xlabel("State Rank", labelpad=8)
    ax.set_ylabel("Estimated Exposure Coefficient", labelpad=8)
    clean_axis(ax)

    if label_focus:
        offsets = {"CA": (14, 20), "VT": (16, 28)}
        for _, row in focus.iterrows():
            ax.annotate(
                row["state"],
                xy=(row["rank"], row["D_full"]),
                xytext=offsets[row["state"]],
                textcoords="offset points",
                fontsize=STATE_LABEL_SIZE,
                ha="left",
                va="center",
                color=HIGHLIGHT_COLOR,
                bbox=dict(
                    boxstyle="round,pad=0.12",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.92,
                ),
                arrowprops=dict(
                    arrowstyle="-",
                    color=HIGHLIGHT_COLOR,
                    linewidth=0.7,
                    shrinkA=1.5,
                    shrinkB=4,
                ),
                zorder=6,
            )


def draw_horizontal_panel(
    ax: plt.Axes,
    panel: pd.DataFrame,
    *,
    title: str,
    xlim: tuple[float, float],
    xticks: list[int],
    highlight_states: set[str],
) -> None:
    y = np.arange(len(panel))
    ax.hlines(y, 0, panel["D_full"], color=LINE_COLOR, linewidth=1.35, zorder=1)
    ax.scatter(
        panel["D_full"],
        y,
        s=38,
        facecolors="white",
        edgecolors=POINT_COLOR,
        linewidths=1.5,
        zorder=3,
    )
    highlight = panel[panel["state"].isin(highlight_states)]
    ax.scatter(
        highlight["D_full"],
        [panel.index.get_loc(idx) for idx in highlight.index],
        s=60,
        facecolors="white",
        edgecolors=HIGHLIGHT_COLOR,
        linewidths=2.1,
        zorder=4,
    )
    ax.axvline(0, color=ZERO_COLOR, linestyle=":", linewidth=1.6, zorder=0)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(panel["state"])
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=13)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xlabel("Estimated Exposure Coefficient", labelpad=8)
    ax.set_title(title, fontsize=PANEL_TITLE_SIZE, pad=10)
    ax.invert_yaxis()
    clean_axis(ax)


def save_main_figure(df: pd.DataFrame, figure_dir: Path = FIGURE_DIR) -> None:
    fig = plt.figure(figsize=(9.45, 7.02))
    grid = GridSpec(2, 2, figure=fig, height_ratios=[2.25, 1.55])
    fig.subplots_adjust(left=0.105, right=0.985, top=0.925, bottom=0.105, hspace=0.72, wspace=0.34)

    ax_top = fig.add_subplot(grid[0, :])
    draw_vertical_lollipop(ax_top, df, label_focus=True)
    ax_top.set_title("Exposure Coefficients by State", fontsize=TITLE_SIZE, pad=10)

    neg = df.nsmallest(12, "D_full").sort_values("D_full")
    pos = df.nlargest(12, "D_full").sort_values("D_full", ascending=False)

    ax_neg = fig.add_subplot(grid[1, 0])
    draw_horizontal_panel(
        ax_neg,
        neg,
        title="Most Negative States",
        xlim=(-66, 3),
        xticks=[-60, -50, -40, -30, -20, -10, 0],
        highlight_states={"CA", "VT"},
    )

    ax_pos = fig.add_subplot(grid[1, 1])
    draw_horizontal_panel(
        ax_pos,
        pos,
        title="Most Positive States",
        xlim=(-10, 195),
        xticks=[0, 50, 100, 150],
        highlight_states={"SD", "WY"},
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "paper_exposure_distribution.pdf")
    fig.savefig(figure_dir / "paper_exposure_distribution.png", dpi=300)
    plt.close(fig)


def save_all_labels_figure(df: pd.DataFrame, figure_dir: Path = FIGURE_DIR) -> None:
    fig, ax = plt.subplots(figsize=(11.23, 5.63))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.90, bottom=0.14)
    draw_vertical_lollipop(ax, df, label_focus=False)
    ax.set_title("Exposure Coefficients by State", fontsize=TITLE_SIZE, pad=10)
    for _, row in df.iterrows():
        va = "bottom" if row["D_full"] >= 0 else "top"
        dy = 4 if row["D_full"] >= 0 else -4
        ax.text(
            row["rank"],
            row["D_full"] + dy,
            row["state"],
            rotation=90,
            ha="center",
            va=va,
            fontsize=ANNOTATION_LABEL_SIZE,
            color=HIGHLIGHT_COLOR,
        )
    fig.savefig(figure_dir / "paper_exposure_distribution_all_labels.pdf")
    fig.savefig(
        figure_dir / "paper_exposure_distribution_all_labels.png",
        dpi=300,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the paper's state-exposure figure.")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=FIGURE_DIR)
    args = parser.parse_args()
    df = ordered_exposure(args.data)
    save_main_figure(df, args.out)
    save_all_labels_figure(df, args.out)
    df[["state", "D_full", "rank"]].to_csv(args.out / "exposure_coefficients.csv", index=False)


if __name__ == "__main__":
    main()
