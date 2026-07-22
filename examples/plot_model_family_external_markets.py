from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = (
    (
        "SSE50",
        {
            "group1": "results/forecasting_group1_sse50_full208",
            "group2": "results/forecasting_group2_sse50_full208",
            "main": "results/rl_group3_ppo_tqc_sse50_full208",
            "stress": "results/rl_group3_td3_tqc_sse50_full208",
        },
    ),
    (
        "HSTech10",
        {
            "group1": "results/forecasting_group1_hstech10_full206",
            "group2": "results/forecasting_group2_hstech10_full206",
            "main": "results/rl_group3_ppo_tqc_hstech10_full206",
            "stress": "results/rl_group3_td3_tqc_hstech10_full206",
        },
    ),
)

ROWS = (
    (
        "group1",
        "Representative models",
        ("arima_lstm", "arima_xgboost", "xgboost_lstm"),
    ),
    (
        "group2",
        "PatchTST+iTransformer",
        ("patchtst_itransformer",),
    ),
    ("main", "PPO+TQC main pair", ("ppo_tqc",)),
    ("stress", "TD3+TQC stress pair", ("td3_tqc",)),
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


def load_frames(code_root: Path) -> dict[tuple[str, str], pd.DataFrame]:
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for dataset, roots in DATASETS:
        for family, relative_root in roots.items():
            source = code_root / relative_root / "selected_tau_summary.csv"
            frames[(dataset, family)] = pd.read_csv(source)
    return frames


def row_limits(
    frames: dict[tuple[str, str], pd.DataFrame], family: str
) -> tuple[float, float]:
    low = min(
        float(frames[(dataset, family)]["delta_sharpe_ci_low"].min())
        for dataset, _roots in DATASETS
    )
    high = max(
        float(frames[(dataset, family)]["delta_sharpe_ci_high"].max())
        for dataset, _roots in DATASETS
    )
    span = max(high - low, 0.10)
    padding = max(0.04, 0.16 * span)
    return min(low - padding, -0.04), max(high + padding, 0.04)


def plot_pair(
    ax: plt.Axes,
    frame: pd.DataFrame,
    pair: str,
    positions: np.ndarray,
    width: float,
) -> list[object]:
    pair_frame = frame.loc[frame["pair"] == pair].sort_values("classifier_group")
    if pair_frame["classifier_group"].tolist() != [1, 2, 3, 4, 5]:
        raise ValueError(f"incomplete classifier groups for {pair}")
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
        elinewidth=0.75,
        capsize=2,
        capthick=0.75,
        zorder=4,
    )
    y_low, y_high = ax.get_ylim()
    label_offset = 0.018 * (y_high - y_low)
    for bar, tau, mean in zip(
        bars,
        pair_frame["selected_global_tau"].to_numpy(dtype=float),
        means,
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + (label_offset if mean >= 0 else -label_offset),
            f"{tau:.2f}",
            ha="center",
            va="bottom" if mean >= 0 else "top",
            fontsize=5.8,
            color="#303238",
            rotation=90 if len(frame["pair"].unique()) > 1 else 0,
            clip_on=True,
        )
    return list(bars)


def main() -> None:
    args = parse_args()
    code_root = Path(args.code_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_frames(code_root)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 8.4,
            "axes.labelsize": 8,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(4, 2, figsize=(7.2, 8.2), sharex=True)
    group_positions = np.arange(1, 6, dtype=float)
    letters = "abcdefgh"
    legend_handles = None

    for row_index, (family, title, pairs) in enumerate(ROWS):
        limits = row_limits(frames, family)
        for column_index, (dataset, _roots) in enumerate(DATASETS):
            ax = axes[row_index, column_index]
            ax.set_ylim(*limits)
            frame = frames[(dataset, family)]
            if len(pairs) == 1:
                plot_pair(ax, frame, pairs[0], group_positions, 0.58)
            else:
                width = 0.23
                offsets = np.linspace(-width, width, len(pairs))
                for pair, offset in zip(pairs, offsets):
                    plot_pair(ax, frame, pair, group_positions + offset, width)
                if legend_handles is None:
                    legend_handles, _labels = ax.get_legend_handles_labels()

            panel_index = row_index * 2 + column_index
            ax.set_title(
                f"({letters[panel_index]}) {dataset}: {title}", loc="left", pad=4
            )
            ax.axhline(0.0, color="#202124", linewidth=0.8, zorder=2)
            ax.grid(axis="y", color="#D9DDE2", linewidth=0.5, zorder=1)
            ax.set_xlim(0.45, 5.55)
            ax.set_xticks(group_positions, [f"G{i}" for i in range(1, 6)])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if column_index == 0:
                ax.set_ylabel("Sharpe difference vs. stronger")
            if row_index == len(ROWS) - 1:
                ax.set_xlabel("Classifier group")

    if legend_handles:
        figure.legend(
            legend_handles,
            [PAIR_LABELS[pair] for pair in ROWS[0][2]],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.002),
            ncol=3,
            frameon=False,
        )
    figure.text(
        0.5,
        0.006,
        "Difference is relative to the stronger component; bar labels are selected fixed global thresholds.",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#45484F",
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.978), h_pad=1.0, w_pad=0.9)

    stem = output_dir / "figure_model_family_external_sharpe_delta"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
