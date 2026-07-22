from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from finrl.reproduction.classifier_ensemble import train_classifier_group
from finrl.reproduction.metrics import metrics_from_account_values
from run_forecasting_group1 import decision_modes
from run_forecasting_group1 import markdown_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a balanced main DRL pair with an imbalance stress test."
    )
    parser.add_argument(
        "--main-dir", default="results/rl_group3_ppo_tqc_dj30_full253"
    )
    parser.add_argument(
        "--stress-dir", default="results/rl_group3_td3_tqc_dj30_full253"
    )
    parser.add_argument(
        "--output-dir", default="results/drl_pair_role_comparison_dj30"
    )
    return parser.parse_args()


def load_experiment(root: Path, role: str) -> dict[str, object]:
    audit = json.loads(
        (root / "experiment_validation_audit.json").read_text(encoding="utf-8")
    )
    if not audit.get("passed"):
        raise ValueError(f"experiment audit did not pass: {root}")
    manifest = json.loads(
        (root / "experiment_manifest.json").read_text(encoding="utf-8")
    )
    selected = pd.read_csv(root / "selected_tau_summary.csv")
    distribution = pd.read_csv(root / "selected_tau_distribution_audit.csv")
    robustness = pd.read_csv(root / "tau_robustness_summary.csv")
    common = pd.read_csv(root / "selected_common_tau_summary.csv")
    base = pd.read_csv(root / "base_model_metrics.csv")
    windows = pd.read_csv(root / "rolling_windows.csv")
    pair = str(selected["pair"].iloc[0])
    components = tuple(manifest["pair_components"][pair])
    return {
        "root": root,
        "role": role,
        "manifest": manifest,
        "pair": pair,
        "components": components,
        "selected": selected,
        "distribution": distribution,
        "robustness": robustness,
        "common": common,
        "base": base,
        "windows": windows,
    }


def window_wins(experiment: dict[str, object]) -> dict[str, int]:
    root = experiment["root"]
    components = experiment["components"]
    windows = experiment["windows"]
    with np.load(Path(root) / "base_account_curves.npz") as arrays:
        values = {model: arrays[model].copy() for model in components}
    counts = {model: 0 for model in components}
    start = 0
    for length in windows["trade_dates"].astype(int):
        sharpes = {
            model: metrics_from_account_values(values[model][start : start + length])[
                "sharpe"
            ]
            for model in components
        }
        counts[max(sharpes, key=sharpes.get)] += 1
        start += length
    return counts


def pair_summary(experiment: dict[str, object]) -> dict[str, object]:
    selected = experiment["selected"]
    distribution = experiment["distribution"]
    common = experiment["common"].iloc[0]
    base = experiment["base"].sort_values("model")
    best = selected.loc[selected["ensemble_sharpe_mean"].idxmax()]
    wins = window_wins(experiment)
    sharpes = base.set_index("model")["sharpe"]
    components = experiment["components"]
    robust = distribution[
        (distribution["delta_sharpe_median"] > 0.0)
        & (distribution["one_sided_sign_test_p"] < 0.05)
    ]
    return {
        "role": experiment["role"],
        "pair": experiment["pair"],
        "component_1": components[0],
        "component_1_sharpe": float(sharpes[components[0]]),
        "component_2": components[1],
        "component_2_sharpe": float(sharpes[components[1]]),
        "component_sharpe_gap": float(abs(sharpes.iloc[0] - sharpes.iloc[1])),
        "window_wins": "; ".join(f"{model}:{wins[model]}" for model in components),
        "groups_beating_stronger_by_mean": int(
            (selected["delta_sharpe_mean"] > 0.0).sum()
        ),
        "groups_with_positive_t_interval": int(
            (selected["delta_sharpe_ci_low"] > 0.0).sum()
        ),
        "groups_with_positive_median_sign_test": int(len(robust)),
        "common_tau": float(common["tau"]),
        "common_tau_groups_beating_stronger": int(
            common["groups_beating_stronger"]
        ),
        "best_classifier_group": int(best["classifier_group"]),
        "best_tau": float(best["selected_global_tau"]),
        "best_ensemble_return": float(best["ensemble_cumulative_return_mean"]),
        "best_ensemble_sharpe": float(best["ensemble_sharpe_mean"]),
        "best_delta_sharpe": float(best["delta_sharpe_mean"]),
        "best_win_rate": float(best["win_rate_vs_stronger"]),
    }


def selected_group_rows(experiment: dict[str, object]) -> pd.DataFrame:
    merged = experiment["selected"].merge(
        experiment["distribution"],
        on=["pair", "classifier_group"],
        suffixes=("", "_distribution"),
    ).merge(
        experiment["robustness"],
        on=["pair", "classifier_group", "selected_global_tau"],
        suffixes=("", "_robustness"),
    )
    merged.insert(0, "role", experiment["role"])
    return merged


def decision_behavior(experiment: dict[str, object]) -> pd.DataFrame:
    root = Path(experiment["root"])
    pair = str(experiment["pair"])
    left, right = experiment["components"]
    selected = experiment["selected"].set_index("classifier_group")
    classifier_audit = pd.read_csv(root / "classifier_refit_audit.csv")
    rows: list[dict[str, object]] = []
    for item in classifier_audit.itertuples(index=False):
        window = int(item.window)
        group = int(item.classifier_group)
        with np.load(root / "candidate_holdings" / f"window_{window:02d}.npz") as arrays:
            calibration = [
                arrays[f"{left}_calibration"],
                arrays[f"{right}_calibration"],
            ]
            candidates = np.stack(
                [arrays[f"{left}_trade"], arrays[f"{right}_trade"]], axis=1
            )
        classifiers = train_classifier_group(
            calibration,
            group,
            random_state=int(item.classifier_seed),
            grid_search=False,
        )
        dispersions, aggressive, conservative = decision_modes(
            classifiers, candidates
        )
        tau = float(selected.loc[group, "selected_global_tau"])
        choices = np.where(dispersions < tau, aggressive, conservative)
        rows.append(
            {
                "role": experiment["role"],
                "pair": pair,
                "repeat": int(item.repeat),
                "window": window,
                "classifier_group": group,
                "tau": tau,
                "second_agent": right,
                "aggressive_rate": float(np.mean(dispersions < tau)),
                "second_agent_selection_rate": float(np.mean(choices == 1)),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    summary: pd.DataFrame,
    groups: pd.DataFrame,
    behavior: pd.DataFrame,
) -> None:
    main = summary.loc[summary["role"] == "balanced_main"].iloc[0]
    stress = summary.loc[summary["role"] == "imbalance_stress_test"].iloc[0]
    behavior_summary = (
        behavior.groupby(
            ["role", "pair", "classifier_group", "tau", "second_agent"]
        )
        .agg(
            aggressive_rate=("aggressive_rate", "mean"),
            selected_second_agent_rate=("second_agent_selection_rate", "mean"),
        )
        .reset_index()
    )
    report = [
        "# Deep-RL Pair Role Comparison",
        "",
        "PPO+TQC is the balanced main experiment. TD3+TQC is retained unchanged as a component-imbalance stress test; no raw metric or account curve is replaced.",
        "",
        "## Pair-Level Results",
        "",
        markdown_table(summary, list(summary.columns)),
        "",
        "## Selected Global Tau by Classifier Group",
        "",
        markdown_table(
            groups,
            [
                "role",
                "pair",
                "classifier_group",
                "selected_global_tau",
                "ensemble_sharpe_mean",
                "delta_sharpe_mean",
                "delta_sharpe_median",
                "win_rate_vs_stronger",
                "one_sided_sign_test_p",
                "tau_beating_stronger",
            ],
        ),
        "",
        "## Decision-Mode Summary",
        "",
        markdown_table(behavior_summary, list(behavior_summary.columns)),
        "",
        "## Interpretation",
        "",
        f"The balanced main pair has a full-year component Sharpe gap of {main['component_sharpe_gap']:.4f}, compared with {stress['component_sharpe_gap']:.4f} in the imbalance stress test. Its window winners split as {main['window_wins']}, so the two agents supply complementary regimes rather than one uniformly dominating the other.",
        "",
        f"PPO+TQC beats its stronger component by mean Sharpe in {int(main['groups_beating_stronger_by_mean'])}/5 groups; {int(main['groups_with_positive_median_sign_test'])}/5 also have a positive median and one-sided sign-test p<0.05. TD3+TQC reaches the corresponding counts {int(stress['groups_beating_stronger_by_mean'])}/5 and {int(stress['groups_with_positive_median_sign_test'])}/5.",
        "",
        "Tau is selected by full-path sensitivity on the same 2020 evaluation path. The 30 repetitions refit classifiers while holding RL checkpoints and market data fixed, so intervals and sign tests describe classifier-refit stability, not independent market or RL-training uncertainty.",
        "",
    ]
    (output_dir / "DRL_PAIR_ROLE_COMPARISON.md").write_text(
        "\n".join(report), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = [
        load_experiment(Path(args.main_dir), "balanced_main"),
        load_experiment(Path(args.stress_dir), "imbalance_stress_test"),
    ]
    summary = pd.DataFrame([pair_summary(item) for item in experiments])
    groups = pd.concat(
        [selected_group_rows(item) for item in experiments], ignore_index=True
    )
    behavior = pd.concat(
        [decision_behavior(item) for item in experiments], ignore_index=True
    )
    summary.to_csv(output_dir / "pair_role_summary.csv", index=False)
    groups.to_csv(output_dir / "selected_group_comparison.csv", index=False)
    behavior.to_csv(output_dir / "selected_decision_behavior.csv", index=False)
    write_report(output_dir, summary, groups, behavior)
    print(summary.to_string(index=False))
    print(f"Saved DRL pair role comparison to {output_dir}")


if __name__ == "__main__":
    main()
