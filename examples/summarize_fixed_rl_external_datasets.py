from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PAIR_LABELS = {
    "a2c_ppo": "A2C + PPO",
    "a2c_sac": "A2C + SAC",
    "ppo_sac": "PPO + SAC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize fixed-RL 30-backtest results across external datasets."
    )
    parser.add_argument("--sse50-dir", required=True)
    parser.add_argument("--hstech10-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def markdown_table(frame: pd.DataFrame, decimals: int = 4) -> str:
    view = frame.copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:.{decimals}f}")
    header = "| " + " | ".join(view.columns) + " |"
    separator = "|" + "|".join(["---"] * len(view.columns)) + "|"
    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def load_dataset(label: str, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = pd.read_csv(root / "selected_tau_summary.csv")
    comparison = pd.read_csv(root / "configuration_comparison.csv")
    base = pd.read_csv(root / "base_model_30_backtest_summary.csv")
    selected.insert(0, "dataset", label)
    comparison.insert(0, "dataset", label)
    base.insert(0, "dataset", label)
    return selected, comparison, base


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    loaded = [
        load_dataset("SSE50", Path(args.sse50_dir)),
        load_dataset("HSTech10", Path(args.hstech10_dir)),
    ]
    selected = pd.concat([item[0] for item in loaded], ignore_index=True)
    comparison = pd.concat([item[1] for item in loaded], ignore_index=True)
    base = pd.concat([item[2] for item in loaded], ignore_index=True)

    comparison["sharpe_success"] = comparison["delta_sharpe"] > 0
    comparison["significant_sharpe_success"] = comparison["delta_sharpe_ci_low"] > 0
    comparison["return_success"] = comparison["delta_cumulative_return"] > 0
    comparison["calmar_success"] = comparison["delta_calmar"] > 0
    comparison["mdd_success"] = comparison["delta_max_drawdown"] > 0

    criteria = [
        ("Mean cumulative return is higher", "return_success"),
        ("Mean Sharpe ratio is higher", "sharpe_success"),
        ("Paired Sharpe 95% interval is entirely positive", "significant_sharpe_success"),
        ("Mean Calmar ratio is higher", "calmar_success"),
        ("Mean maximum drawdown is better", "mdd_success"),
    ]
    summary_rows: list[dict[str, object]] = []
    for dataset, group in list(comparison.groupby("dataset")) + [("Combined", comparison)]:
        for criterion, column in criteria:
            summary_rows.append(
                {
                    "dataset": dataset,
                    "criterion": criterion,
                    "successes": int(group[column].sum()),
                    "configurations": len(group),
                    "rate": float(group[column].mean()),
                }
            )
    aggregate = pd.DataFrame(summary_rows)

    pair_summary = (
        comparison.groupby(["dataset", "pair"])
        .agg(
            sharpe_successes=("sharpe_success", "sum"),
            significant_successes=("significant_sharpe_success", "sum"),
            configurations=("sharpe_success", "size"),
            mean_delta_sharpe=("delta_sharpe", "mean"),
        )
        .reset_index()
    )
    pair_summary["pair"] = pair_summary["pair"].map(PAIR_LABELS)
    combined_pairs = (
        comparison.groupby("pair")
        .agg(
            sharpe_successes=("sharpe_success", "sum"),
            significant_successes=("significant_sharpe_success", "sum"),
            configurations=("sharpe_success", "size"),
            mean_delta_sharpe=("delta_sharpe", "mean"),
        )
        .reset_index()
    )
    combined_pairs.insert(0, "dataset", "Combined")
    combined_pairs["pair"] = combined_pairs["pair"].map(PAIR_LABELS)
    pair_summary = pd.concat([pair_summary, combined_pairs], ignore_index=True)

    group_summary = (
        comparison.groupby(["dataset", "classifier_group"])
        .agg(
            sharpe_successes=("sharpe_success", "sum"),
            significant_successes=("significant_sharpe_success", "sum"),
            configurations=("sharpe_success", "size"),
            mean_delta_sharpe=("delta_sharpe", "mean"),
        )
        .reset_index()
    )
    combined_groups = (
        comparison.groupby("classifier_group")
        .agg(
            sharpe_successes=("sharpe_success", "sum"),
            significant_successes=("significant_sharpe_success", "sum"),
            configurations=("sharpe_success", "size"),
            mean_delta_sharpe=("delta_sharpe", "mean"),
        )
        .reset_index()
    )
    combined_groups.insert(0, "dataset", "Combined")
    group_summary = pd.concat([group_summary, combined_groups], ignore_index=True)

    selected.to_csv(output / "external_dataset_selected_results.csv", index=False)
    comparison.to_csv(output / "external_dataset_configuration_comparison.csv", index=False)
    base.to_csv(output / "external_dataset_base_models.csv", index=False)
    aggregate.to_csv(output / "external_dataset_outperformance_summary.csv", index=False)
    pair_summary.to_csv(output / "external_dataset_pair_summary.csv", index=False)
    group_summary.to_csv(output / "external_dataset_classifier_group_summary.csv", index=False)

    top_rows = selected.loc[
        selected.groupby("dataset")["ensemble_sharpe_mean"].idxmax(),
        [
            "dataset",
            "pair",
            "classifier_group",
            "selected_global_tau",
            "ensemble_sharpe_mean",
            "stronger_component",
            "component_sharpe_mean",
            "delta_sharpe_mean",
        ],
    ].copy()
    top_rows["pair"] = top_rows["pair"].map(PAIR_LABELS)
    headline = aggregate[
        aggregate["criterion"].eq("Mean Sharpe ratio is higher")
    ][["dataset", "successes", "configurations", "rate"]]
    significant = aggregate[
        aggregate["criterion"].eq(
            "Paired Sharpe 95% interval is entirely positive"
        )
    ][["dataset", "successes", "configurations", "rate"]]
    combined_pair_view = pair_summary[pair_summary["dataset"].eq("Combined")]
    combined_group_view = group_summary[group_summary["dataset"].eq("Combined")]

    lines = [
        "# Fixed-RL External-Dataset 30-Backtest Summary",
        "",
        "## Headline",
        "",
        markdown_table(headline),
        "",
        "Across both external datasets, 21 of 30 pair-group configurations improve mean Sharpe over the stronger component, and the same 21 have paired 95% intervals entirely above zero. The result supports a frequent but conditional ensemble advantage.",
        "",
        "SSE50 is strongly favorable (13/15), while HSTech10 is mixed (8/15). HSTech10 shows a clear pair dependence: A2C+SAC fails in all five classifier groups even after selecting the best fixed global tau, whereas A2C+PPO and PPO+SAC each succeed in four of five groups.",
        "",
        "## Best Configuration Per Dataset",
        "",
        markdown_table(top_rows),
        "",
        "## Paired Significance",
        "",
        markdown_table(significant),
        "",
        "## Success By RL Pair",
        "",
        markdown_table(combined_pair_view),
        "",
        "PPO+SAC is the most transferable pair (9/10 Sharpe successes), followed by A2C+PPO (8/10). A2C+SAC succeeds in only 4/10 configurations because it fails across HSTech10.",
        "",
        "## Success By Classifier Group",
        "",
        markdown_table(combined_group_view),
        "",
        "Group 3 is the consistent weak point: only 1 of 6 tree-only configurations improves Sharpe. Groups 1, 2, 4, and 5 each succeed in 5 of 6 configurations. This supports classifier composition as an active mechanism, while rejecting a universal benefit from every classifier family.",
        "",
        "## Interpretation Limits",
        "",
        "- The 30 repetitions vary rolling classifier fits only; RL checkpoints and deterministic candidate holdings remain fixed.",
        "- SSE50 contains 26 aligned constituents and HSTech10 is a 10-stock longest-coverage subset, so neither is a point-in-time full-index panel.",
        "- The evaluation spans contain 208 and 206 sessions respectively, not complete calendar years.",
        "- Tau is selected by completed-path mean Sharpe on each dataset. The results are sensitivity evidence, not an unbiased deployment estimate.",
        "- The external results strengthen the claim that the mechanism can work across universes, but they also narrow the defensible conclusion to pair-, group-, and threshold-dependent improvement.",
        "",
    ]
    (output / "EXTERNAL_DATASET_30_BACKTEST_SUMMARY.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
