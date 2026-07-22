from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PANELS = (
    (
        "Representative forecasting",
        "results/forecasting_group1_dj30_full253",
        ("arima_lstm", "arima_xgboost", "xgboost_lstm"),
    ),
    (
        "Modern deep forecasting",
        "results/forecasting_group2_dj30_full253",
        ("patchtst_itransformer",),
    ),
    (
        "Balanced DRL main test",
        "results/rl_group3_ppo_tqc_dj30_full253",
        ("ppo_tqc",),
    ),
    (
        "DRL imbalance stress test",
        "results/rl_group3_td3_tqc_dj30_full253",
        ("td3_tqc",),
    ),
)

PAIR_LABELS = {
    "arima_lstm": "ARIMA + LSTM",
    "arima_xgboost": "ARIMA + XGBoost",
    "xgboost_lstm": "XGBoost + LSTM",
    "patchtst_itransformer": "PatchTST + iTransformer",
    "ppo_tqc": "PPO + TQC",
    "td3_tqc": "TD3 + TQC",
}

PAIR_COLORS = {
    "arima_lstm": "#167D5A",
    "arima_xgboost": "#4177A8",
    "xgboost_lstm": "#D4872C",
    "patchtst_itransformer": "#328A9D",
    "ppo_tqc": "#C9514B",
    "td3_tqc": "#666A73",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", default=".")
    parser.add_argument(
        "--output-dir",
        default=(
            "../paper/arxiv/arxiv_final/data/experiments/model_family_extensions"
        ),
    )
    return parser.parse_args()


def plot_pair(ax, frame: pd.DataFrame, pair: str, positions: np.ndarray, width: float):
    pair_frame = frame.loc[frame["pair"] == pair].sort_values("classifier_group")
    means = pair_frame["delta_sharpe_mean"].to_numpy(dtype=float)
    low = pair_frame["delta_sharpe_ci_low"].to_numpy(dtype=float)
    high = pair_frame["delta_sharpe_ci_high"].to_numpy(dtype=float)
    errors = np.vstack([means - low, high - means])
    bars = ax.bar(
        positions,
        means,
        width=width,
        color=PAIR_COLORS[pair],
        edgecolor="white",
        linewidth=0.5,
        label=PAIR_LABELS[pair],
        zorder=3,
    )
    ax.errorbar(
        positions,
        means,
        yerr=errors,
        fmt="none",
        ecolor="#202124",
        elinewidth=0.8,
        capsize=2,
        capthick=0.8,
        zorder=4,
    )
    for bar, tau, mean in zip(
        bars,
        pair_frame["selected_global_tau"].to_numpy(dtype=float),
        means,
    ):
        offset = 0.018 if mean >= 0 else -0.025
        va = "bottom" if mean >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + offset,
            f"{tau:.2f}",
            ha="center",
            va=va,
            fontsize=6.2,
            color="#303238",
            rotation=90 if len(frame["pair"].unique()) > 1 else 0,
        )


def main() -> None:
    args = parse_args()
    code_root = Path(args.code_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 6.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.35), sharey=True)
    group_positions = np.arange(1, 6, dtype=float)
    panel_letters = "abcd"

    for panel_index, (ax, panel) in enumerate(zip(axes.flat, PANELS)):
        title, relative_root, pairs = panel
        frame = pd.read_csv(code_root / relative_root / "selected_tau_summary.csv")
        if len(pairs) == 1:
            plot_pair(ax, frame, pairs[0], group_positions, 0.58)
        else:
            width = 0.23
            offsets = np.linspace(-width, width, len(pairs))
            for pair, offset in zip(pairs, offsets):
                plot_pair(ax, frame, pair, group_positions + offset, width)

        ax.axhline(0.0, color="#202124", linewidth=0.8, zorder=2)
        ax.grid(axis="y", color="#D9DDE2", linewidth=0.55, zorder=1)
        ax.set_xlim(0.45, 5.55)
        ax.set_ylim(-0.35, 0.50)
        ax.set_xticks(group_positions, [f"G{i}" for i in range(1, 6)])
        ax.set_title(f"({panel_letters[panel_index]}) {title}", loc="left", pad=5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if len(pairs) > 1:
            ax.legend(frameon=False, loc="upper right", ncol=1)
        else:
            ax.text(
                0.98,
                0.95,
                PAIR_LABELS[pairs[0]],
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color=PAIR_COLORS[pairs[0]],
                fontweight="bold",
            )

    axes[0, 0].set_ylabel("Sharpe difference vs. stronger component")
    axes[1, 0].set_ylabel("Sharpe difference vs. stronger component")
    axes[1, 0].set_xlabel("Classifier group")
    axes[1, 1].set_xlabel("Classifier group")
    figure.text(
        0.5,
        0.012,
        "Numbers on bars are selected fixed global thresholds.",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#45484F",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1), h_pad=1.15, w_pad=1.05)

    stem = output_dir / "figure_model_family_sharpe_delta"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
