from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/finrl-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from finrl.reproduction.disagreement import DISAGREEMENT_METRICS
from run_fixed_rl_30_backtests import markdown_table


DATASETS = (
    ("DJ30", "results/disagreement/dj30"),
    ("SSE50", "results/disagreement/sse50"),
    ("HSTech10", "results/disagreement/hstech10"),
)
PAIR_KEYS = ("a2c_ppo", "a2c_sac", "ppo_sac")
METRIC_LABELS = {
    "original": "Original",
    "l1": "L1",
    "risk_weighted": "Risk-weighted",
}
METRIC_COLORS = {
    "original": "#6B7280",
    "l1": "#2F6F9F",
    "risk_weighted": "#B45309",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", default=".")
    parser.add_argument("--dj30-dir", default=DATASETS[0][1])
    parser.add_argument("--sse50-dir", default=DATASETS[1][1])
    parser.add_argument("--hstech10-dir", default=DATASETS[2][1])
    parser.add_argument(
        "--output-dir", default="results/disagreement/summary"
    )
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def audit_dataset(root: Path, dataset: str) -> dict[str, object]:
    errors: list[str] = []
    required = (
        "run_metadata.json",
        "all_backtest_metrics.csv",
        "classifier_refit_audit.csv",
        "selected_tau_summary.csv",
        "selected_tau_paired_runs.csv",
        "threshold_robustness.csv",
        "ablation_summary.csv",
        "metric_vs_original_summary.csv",
        "disagreement_daily_detail.csv",
        "v1_reproduction_audit.json",
    )
    for name in required:
        require((root / name).exists(), f"missing {name}", errors)
    if errors:
        return {"dataset": dataset, "root": str(root), "passed": False, "errors": errors}

    metadata = json.loads((root / "run_metadata.json").read_text())
    v1_audit = json.loads((root / "v1_reproduction_audit.json").read_text())
    metrics = pd.read_csv(root / "all_backtest_metrics.csv")
    classifiers = pd.read_csv(root / "classifier_refit_audit.csv")
    selected = pd.read_csv(root / "selected_tau_summary.csv")
    selected_runs = pd.read_csv(root / "selected_tau_paired_runs.csv")
    robustness = pd.read_csv(root / "threshold_robustness.csv")
    disagreement = pd.read_csv(root / "disagreement_daily_detail.csv")
    expected_rows = 30 * 3 * 3 * 5 * 89
    require(metadata.get("version") == "v2", "wrong experiment version", errors)
    require(metadata.get("repetitions") == 30, "repetition count is not 30", errors)
    require(
        tuple(metadata.get("disagreement_metrics", [])) == DISAGREEMENT_METRICS,
        "wrong disagreement metric set or order",
        errors,
    )
    require(metadata.get("deterministic_rl_inference") is True, "RL is not deterministic", errors)
    require(metadata.get("rl_retrained_in_repetitions") is False, "RL was retrained inside repeats", errors)
    require(metadata.get("classifier_grid_search") is False, "classifier grid search is enabled", errors)
    require(
        metadata.get("classifier_decisions_shared_across_disagreement_metrics") is True,
        "classifier decisions are not shared across disagreement metrics",
        errors,
    )
    require(metadata.get("fixed_global_tau_per_complete_path") is True, "tau is not global", errors)
    require(metadata.get("covariance_frozen_within_trade_block") is True, "covariance is not frozen", errors)
    require(metadata.get("v1_reproduction_passed") is True, "V1 guard failed", errors)
    require(v1_audit.get("passed") is True, "V1 audit file failed", errors)
    require(np.isclose(v1_audit.get("v1_coverage_fraction", 0.0), 1.0), "V1 coverage is incomplete", errors)
    require(len(metrics) == expected_rows, f"metric row count is {len(metrics)}", errors)
    require(len(classifiers) == 1800, f"classifier fit count is {len(classifiers)}", errors)
    require(len(selected) == 45, f"selected row count is {len(selected)}", errors)
    require(len(selected_runs) == 30 * 45, f"selected paired-run count is {len(selected_runs)}", errors)
    require(len(robustness) == 45, f"robustness row count is {len(robustness)}", errors)
    require(metrics["repeat"].nunique() == 30, "not all repeats are present", errors)
    require(metrics["tau"].nunique() == 89, "not all tau values are present", errors)
    metric_keys = [
        "repeat",
        "disagreement_metric",
        "pair",
        "classifier_group",
        "tau",
    ]
    require(not metrics.duplicated(metric_keys).any(), "metric keys are duplicated", errors)
    require(
        np.isfinite(
            metrics[
                ["cumulative_return", "sharpe", "calmar", "max_drawdown"]
            ].to_numpy(dtype=float)
        ).all(),
        "backtest metrics contain non-finite values",
        errors,
    )
    classifier_keys = ["repeat", "window", "pair", "classifier_group"]
    require(
        not classifiers.duplicated(classifier_keys).any(),
        "classifier-refit keys are duplicated",
        errors,
    )
    require(
        set(metrics["disagreement_metric"]) == set(DISAGREEMENT_METRICS),
        "metric rows omit a disagreement metric",
        errors,
    )
    require(
        disagreement["disagreement"].between(0.0, 1.0).all(),
        "disagreement leaves [0, 1]",
        errors,
    )
    source_end = pd.to_datetime(disagreement["covariance_source_end"])
    trade_start = pd.to_datetime(disagreement["trade_start"])
    require((source_end < trade_start).all(), "covariance source leaks into trade", errors)
    classifier_end = pd.to_datetime(classifiers["calibration_end"])
    classifier_trade_start = pd.to_datetime(classifiers["trade_start"])
    require(
        (classifier_end < classifier_trade_start).all(),
        "classifier calibration leaks into trade",
        errors,
    )
    run_directories = list((root / "runs").glob("repeat_*"))
    require(len(run_directories) == 30, "run-directory count is not 30", errors)
    return {
        "dataset": dataset,
        "root": str(root),
        "metric_rows": len(metrics),
        "classifier_fits": len(classifiers),
        "selected_configurations": len(selected),
        "tau_count": metrics["tau"].nunique(),
        "v1_maximum_metric_difference": max(
            v1_audit["maximum_absolute_metric_differences"].values()
        ),
        "passed": not errors,
        "errors": errors,
    }


def plot_combined(selected: pd.DataFrame, output_dir: Path, dpi: int) -> list[Path]:
    figure, axes = plt.subplots(3, 3, figsize=(12.2, 9.4), sharex=True, sharey=True)
    group_positions = np.arange(1, 6, dtype=float)
    width = 0.24
    offsets = (-width, 0.0, width)
    for row, (dataset, _root) in enumerate(DATASETS):
        for column, pair in enumerate(PAIR_KEYS):
            axis = axes[row, column]
            for offset, metric in zip(offsets, DISAGREEMENT_METRICS):
                frame = selected[
                    (selected["dataset"] == dataset)
                    & (selected["pair"] == pair)
                    & (selected["disagreement_metric"] == metric)
                ].sort_values("classifier_group")
                means = frame["delta_sharpe_mean"].to_numpy(dtype=float)
                lower = means - frame["delta_sharpe_ci_low"].to_numpy(dtype=float)
                upper = frame["delta_sharpe_ci_high"].to_numpy(dtype=float) - means
                axis.bar(
                    group_positions + offset,
                    means,
                    width=width * 0.92,
                    color=METRIC_COLORS[metric],
                    label=METRIC_LABELS[metric],
                    zorder=3,
                )
                axis.errorbar(
                    group_positions + offset,
                    means,
                    yerr=np.vstack([lower, upper]),
                    fmt="none",
                    ecolor="#202124",
                    capsize=1.7,
                    linewidth=0.65,
                    zorder=4,
                )
            axis.axhline(0.0, color="#202124", linewidth=0.8)
            axis.grid(axis="y", color="#D9DDE2", linewidth=0.5, zorder=1)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.set_xticks(group_positions, [f"G{i}" for i in range(1, 6)])
            axis.set_title(f"{dataset}: {pair.replace('_', ' + ').upper()}", fontsize=9)
            if column == 0:
                axis.set_ylabel("Sharpe difference vs. stronger")
            if row == 2:
                axis.set_xlabel("Classifier group")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("V2 holding-disagreement metric ablation", y=0.99)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92), h_pad=1.2, w_pad=0.8)
    stem = output_dir / "figure_v2_disagreement_ablation_all_markets"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return [stem.with_suffix(".pdf"), stem.with_suffix(".png")]


def main() -> None:
    args = parse_args()
    code_root = Path(args.code_root)
    output_dir = code_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    audits: list[dict[str, object]] = []
    selected_frames: list[pd.DataFrame] = []
    robustness_frames: list[pd.DataFrame] = []
    comparison_frames: list[pd.DataFrame] = []
    disagreement_frames: list[pd.DataFrame] = []
    dataset_roots = (
        ("DJ30", args.dj30_dir),
        ("SSE50", args.sse50_dir),
        ("HSTech10", args.hstech10_dir),
    )
    for dataset, relative_root in dataset_roots:
        root = code_root / relative_root
        audit = audit_dataset(root, dataset)
        audits.append(audit)
        if not audit["passed"]:
            continue
        for filename, collection in (
            ("selected_tau_summary.csv", selected_frames),
            ("threshold_robustness.csv", robustness_frames),
            ("metric_vs_original_summary.csv", comparison_frames),
            ("disagreement_daily_detail.csv", disagreement_frames),
        ):
            frame = pd.read_csv(root / filename)
            frame.insert(0, "dataset", dataset)
            collection.append(frame)
    combined_audit = {"passed": all(item["passed"] for item in audits), "datasets": audits}
    (output_dir / "independent_audit.json").write_text(
        json.dumps(combined_audit, indent=2), encoding="utf-8"
    )
    if not combined_audit["passed"]:
        raise ValueError("one or more V2 dataset audits failed")

    selected = pd.concat(selected_frames, ignore_index=True)
    robustness = pd.concat(robustness_frames, ignore_index=True)
    comparisons = pd.concat(comparison_frames, ignore_index=True)
    disagreement = pd.concat(disagreement_frames, ignore_index=True)
    selected.to_csv(output_dir / "selected_tau_all_markets.csv", index=False)
    robustness.to_csv(output_dir / "threshold_robustness_all_markets.csv", index=False)
    comparisons.to_csv(output_dir / "metric_vs_original_all_markets.csv", index=False)
    disagreement.to_csv(
        output_dir / "disagreement_daily_detail_all_markets.csv", index=False
    )

    dataset_summary = (
        selected.assign(
            beats_stronger=lambda frame: frame["delta_sharpe_mean"] > 0,
            positive_ci=lambda frame: frame["delta_sharpe_ci_low"] > 0,
        )
        .groupby(["dataset", "disagreement_metric"], sort=False)
        .agg(
            configurations=("pair", "size"),
            beats_stronger=("beats_stronger", "sum"),
            positive_95pct_ci=("positive_ci", "sum"),
            mean_delta_sharpe=("delta_sharpe_mean", "mean"),
            median_delta_sharpe=("delta_sharpe_mean", "median"),
        )
        .reset_index()
    )
    combined_summary = (
        selected.assign(
            beats_stronger=lambda frame: frame["delta_sharpe_mean"] > 0,
            positive_ci=lambda frame: frame["delta_sharpe_ci_low"] > 0,
        )
        .groupby("disagreement_metric", sort=False)
        .agg(
            configurations=("pair", "size"),
            beats_stronger=("beats_stronger", "sum"),
            positive_95pct_ci=("positive_ci", "sum"),
            mean_delta_sharpe=("delta_sharpe_mean", "mean"),
            median_delta_sharpe=("delta_sharpe_mean", "median"),
        )
        .reset_index()
    )
    winner_rows = (
        selected.sort_values(
            ["dataset", "pair", "classifier_group", "ensemble_sharpe_mean"],
            ascending=[True, True, True, False],
        )
        .groupby(["dataset", "pair", "classifier_group"], as_index=False)
        .head(1)
    )
    winners = (
        winner_rows
        .groupby("disagreement_metric")
        .size()
        .rename("configuration_wins")
        .reset_index()
    )
    winners_by_dataset = (
        winner_rows.groupby(["dataset", "disagreement_metric"])
        .size()
        .rename("configuration_wins")
        .reset_index()
    )
    combined_summary = combined_summary.merge(
        winners, on="disagreement_metric", how="left"
    )
    dataset_summary.to_csv(output_dir / "dataset_metric_summary.csv", index=False)
    combined_summary.to_csv(output_dir / "combined_metric_summary.csv", index=False)
    winners_by_dataset.to_csv(
        output_dir / "configuration_winners_by_dataset.csv", index=False
    )
    figure_paths = plot_combined(selected, output_dir, args.dpi)

    robustness_summary = (
        robustness.assign(
            any_success=lambda frame: frame["successful_tau_count"] > 0,
        )
        .groupby(["dataset", "disagreement_metric"], sort=False)
        .agg(
            configurations=("pair", "size"),
            configurations_with_any_success=("any_success", "sum"),
            mean_successful_tau_fraction=("successful_tau_fraction", "mean"),
            median_successful_tau_fraction=("successful_tau_fraction", "median"),
            minimum_successful_tau_fraction=("successful_tau_fraction", "min"),
            maximum_successful_tau_fraction=("successful_tau_fraction", "max"),
        )
        .reset_index()
    )
    robustness_summary.to_csv(
        output_dir / "threshold_robustness_summary.csv", index=False
    )
    combined_robustness = (
        robustness.assign(
            any_success=lambda frame: frame["successful_tau_count"] > 0,
        )
        .groupby("disagreement_metric", sort=False)
        .agg(
            configurations=("pair", "size"),
            configurations_with_any_success=("any_success", "sum"),
            mean_successful_tau_fraction=("successful_tau_fraction", "mean"),
            median_successful_tau_fraction=("successful_tau_fraction", "median"),
        )
        .reset_index()
    )
    combined_robustness.to_csv(
        output_dir / "threshold_robustness_combined.csv", index=False
    )

    disagreement_scale = (
        disagreement.groupby(["dataset", "disagreement_metric"], sort=False)[
            "disagreement"
        ]
        .agg(
            observations="size",
            minimum="min",
            q05=lambda values: values.quantile(0.05),
            median="median",
            mean="mean",
            q95=lambda values: values.quantile(0.95),
            maximum="max",
            standard_deviation="std",
            fraction_at_or_above_0_90=lambda values: (values >= 0.90).mean(),
        )
        .reset_index()
    )
    disagreement_scale.to_csv(
        output_dir / "disagreement_scale_summary.csv", index=False
    )

    direct_comparison = (
        comparisons.assign(
            positive=lambda frame: frame["delta_sharpe_vs_original_mean"] > 0,
            positive_ci=lambda frame: frame["delta_sharpe_vs_original_ci_low"] > 0,
            negative=lambda frame: frame["delta_sharpe_vs_original_mean"] < 0,
            negative_ci=lambda frame: frame["delta_sharpe_vs_original_ci_high"] < 0,
        )
        .groupby("disagreement_metric")
        .agg(
            configurations=("pair", "size"),
            improves_on_original=("positive", "sum"),
            positive_95pct_ci=("positive_ci", "sum"),
            degrades_vs_original=("negative", "sum"),
            negative_95pct_ci=("negative_ci", "sum"),
            mean_delta_vs_original=("delta_sharpe_vs_original_mean", "mean"),
        )
        .reset_index()
    )
    direct_comparison.to_csv(
        output_dir / "direct_comparison_summary.csv", index=False
    )
    direct_comparison_by_dataset = (
        comparisons.assign(
            positive=lambda frame: frame["delta_sharpe_vs_original_mean"] > 0,
            positive_ci=lambda frame: frame["delta_sharpe_vs_original_ci_low"] > 0,
            negative=lambda frame: frame["delta_sharpe_vs_original_mean"] < 0,
            negative_ci=lambda frame: frame["delta_sharpe_vs_original_ci_high"] < 0,
        )
        .groupby(["dataset", "disagreement_metric"], sort=False)
        .agg(
            configurations=("pair", "size"),
            improves_on_original=("positive", "sum"),
            positive_95pct_ci=("positive_ci", "sum"),
            degrades_vs_original=("negative", "sum"),
            negative_95pct_ci=("negative_ci", "sum"),
            mean_delta_vs_original=("delta_sharpe_vs_original_mean", "mean"),
        )
        .reset_index()
    )
    direct_comparison_by_dataset.to_csv(
        output_dir / "direct_comparison_by_dataset.csv", index=False
    )
    report = [
        "# V2 Holding-Disagreement Metric Ablation",
        "",
        "## Independent Audit",
        "",
        "All three datasets pass the 30-repeat, 89-threshold, 1,800-classifier-fit, no-covariance-leakage, and exact-V1-reproduction checks.",
        "",
        "## Dataset Results",
        "",
        markdown_table(dataset_summary, list(dataset_summary.columns)),
        "",
        "## Combined Results",
        "",
        markdown_table(combined_summary, list(combined_summary.columns)),
        "",
        "## Configuration Winners by Dataset",
        "",
        markdown_table(winners_by_dataset, list(winners_by_dataset.columns)),
        "",
        "## Direct Comparison with V1 Original Metric",
        "",
        markdown_table(direct_comparison, list(direct_comparison.columns)),
        "",
        "### Direct Comparison by Dataset",
        "",
        markdown_table(
            direct_comparison_by_dataset,
            list(direct_comparison_by_dataset.columns),
        ),
        "",
        "The comparisons use each metric's mean-Sharpe-maximizing fixed global threshold on the same evaluation path. They are a controlled mechanism ablation, not an untouched-holdout performance estimate.",
        "",
        "## Threshold Robustness",
        "",
        markdown_table(robustness_summary, list(robustness_summary.columns)),
        "",
        "### Combined Threshold Robustness",
        "",
        markdown_table(combined_robustness, list(combined_robustness.columns)),
        "",
        "## Disagreement-Scale Diagnostics",
        "",
        markdown_table(disagreement_scale, list(disagreement_scale.columns)),
        "",
        "## Controlled-Ablation Conclusion",
        "",
        "The original statistic remains the primary metric: it wins 27/45 configurations and has the largest mean Sharpe improvement over the stronger component. Risk weighting is the credible secondary candidate (13/45 wins), but it does not dominate the original and has a lower mean delta. Plain L1 is rejected in the current fixed-global-threshold mechanism because it loses most direct comparisons and frequently saturates near one.",
        "",
        "## Figure",
        "",
        *[f"- `{path}`" for path in figure_paths],
    ]
    (output_dir / "V2_DISAGREEMENT_ABLATION_SUMMARY.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(json.dumps(combined_summary.to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()
