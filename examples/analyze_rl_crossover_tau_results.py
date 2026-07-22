from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EVALUATION_PERIODS = ("2020",)
BLOCK_KEYS = ["period", "pair", "repeat", "classifier_group", "window"]
TOLERANCE = 1e-12
COMPARATORS = {
    "Retrospectively stronger fixed single RL": "stronger",
    "Causal rolling-validation single RL": "causal_single",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the guarded and mechanism-only 2020 causal "
            "crossover-tau experiments."
        )
    )
    parser.add_argument("--guarded-results", default="work/causal_tau_guarded")
    parser.add_argument(
        "--mechanism-only-results", default="work/causal_tau_mechanism_only"
    )
    parser.add_argument("--output-dir", default="work/causal_tau_comparison")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_result_root(path: Path, *, require_positive_lcb: bool) -> None:
    required = (
        "all_crossover_tau_metrics.csv",
        "all_block_tau_audit.csv",
        "all_daily_crossover_tau_decisions.csv",
        "all_daily_crossover_tau_feedback.npz",
        "crossover_tau_summary.csv",
        "experiment_manifest.json",
        "independent_validation_audit.json",
    )
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"{path} is missing: {', '.join(missing)}")
    manifest = read_json(path / "experiment_manifest.json")
    audit = read_json(path / "independent_validation_audit.json")
    if manifest.get("require_positive_lcb") is not require_positive_lcb:
        raise ValueError(f"unexpected LCB setting in {path}")
    if manifest.get("evaluation_periods") != list(EVALUATION_PERIODS):
        raise ValueError(f"only the 2020 paper experiment is accepted: {path}")
    if manifest.get("repetitions") != 30:
        raise ValueError(f"expected 30 classifier refits in {path}")
    if audit.get("passed") is not True:
        raise ValueError(f"independent validation failed for {path}")


def restore_lossless_feedback(root: Path, decisions: pd.DataFrame) -> pd.DataFrame:
    restored = decisions.copy()
    with np.load(root / "all_daily_crossover_tau_feedback.npz") as payload:
        for column in payload.files:
            if len(payload[column]) != len(restored):
                raise ValueError(f"lossless feedback does not align in {root}")
            restored[column] = payload[column]
    return restored


def outcome_counts(values: pd.Series) -> tuple[int, int, int]:
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("non-finite comparison values found")
    return (
        int(np.sum(array > TOLERANCE)),
        int(np.sum(np.abs(array) <= TOLERANCE)),
        int(np.sum(array < -TOLERANCE)),
    )


def comparator_summary(metrics: pd.DataFrame, *, method: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for comparator, suffix in COMPARATORS.items():
        delta = metrics[f"delta_sharpe_vs_{suffix}"]
        wins, ties, losses = outcome_counts(delta)
        rows.append(
            {
                "method": method,
                "period": "2020",
                "comparator": comparator,
                "paths": len(metrics),
                "mean_delta_sharpe": float(delta.mean()),
                "median_delta_sharpe": float(delta.median()),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "win_rate": wins / len(metrics),
            }
        )
    return pd.DataFrame(rows)


def pair_performance_summary(metrics: pd.DataFrame, *, method: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pair, group in metrics.groupby("pair", sort=True):
        causal = group["delta_sharpe_vs_causal_single"]
        stronger = group["delta_sharpe_vs_stronger"]
        causal_counts = outcome_counts(causal)
        stronger_counts = outcome_counts(stronger)
        rows.append(
            {
                "method": method,
                "period": "2020",
                "pair": pair,
                "paths": len(group),
                "mean_delta_sharpe_vs_causal_single": float(causal.mean()),
                "wins_vs_causal_single": causal_counts[0],
                "ties_vs_causal_single": causal_counts[1],
                "losses_vs_causal_single": causal_counts[2],
                "mean_delta_sharpe_vs_stronger_single": float(stronger.mean()),
                "wins_vs_stronger_single": stronger_counts[0],
                "ties_vs_stronger_single": stronger_counts[1],
                "losses_vs_stronger_single": stronger_counts[2],
            }
        )
    return pd.DataFrame(rows)


def configuration_summary(root: Path, *, method: str) -> pd.DataFrame:
    summary = pd.read_csv(root / "crossover_tau_summary.csv", dtype={"period": str})
    summary = summary.loc[summary["period"].eq("2020")].copy()
    columns = [
        "period",
        "pair",
        "classifier_group",
        "repetitions",
        "sharpe_mean",
        "sharpe_sd",
        "stronger_single_model",
        "stronger_single_sharpe",
        "causal_single_sharpe",
        "delta_sharpe_vs_stronger_mean",
        "delta_sharpe_vs_stronger_ci_low",
        "delta_sharpe_vs_stronger_ci_high",
        "wins_vs_stronger_single",
        "delta_sharpe_vs_causal_single_mean",
        "delta_sharpe_vs_causal_single_ci_low",
        "delta_sharpe_vs_causal_single_ci_high",
        "wins_vs_causal_single",
        "mean_fallback_day_rate",
        "mean_selected_blocks",
        "mean_selected_tau",
        "mean_threshold_quantile",
    ]
    result = summary.loc[:, columns]
    result.insert(0, "method", method)
    return result


def status_summary(root: Path, *, method: str) -> pd.DataFrame:
    blocks = pd.read_csv(root / "all_block_tau_audit.csv", dtype={"period": str})
    blocks = blocks.loc[blocks["period"].eq("2020")]
    result = blocks.groupby("status", sort=True).size().rename("blocks").reset_index()
    result["block_rate"] = result["blocks"] / result["blocks"].sum()
    result.insert(0, "period", "2020")
    result.insert(0, "method", method)
    return result


def block_mechanism_validation(root: Path) -> pd.DataFrame:
    blocks = pd.read_csv(root / "all_block_tau_audit.csv", dtype={"period": str})
    blocks = blocks.loc[blocks["period"].eq("2020")].copy()
    decisions = pd.read_csv(
        root / "all_daily_crossover_tau_decisions.csv", dtype={"period": str}
    )
    decisions = restore_lossless_feedback(root, decisions)
    decisions = decisions.loc[decisions["period"].eq("2020")].copy()
    lookup = blocks.set_index(BLOCK_KEYS, verify_integrity=True)
    rows: list[dict[str, object]] = []
    for key, days in decisions.groupby(BLOCK_KEYS, sort=True):
        block = lookup.loc[key]
        selected = str(block["status"]) == "selected"
        tau = float(block["selected_tau"]) if selected else np.nan
        dispersion = days["holding_dispersion"].to_numpy(dtype=float)
        informative = (
            days["branch_diverged"].astype(str).str.lower().isin(("true", "1"))
        ).to_numpy()
        mode_gap = (
            days["aggressive_return"].to_numpy(dtype=float)
            - days["conservative_return"].to_numpy(dtype=float)
        )
        low = informative & (dispersion < tau) if selected else np.zeros(len(days), bool)
        high = informative & ~low if selected else np.zeros(len(days), bool)
        if selected:
            realized = float(
                (
                    np.log1p(days["actual_master_return"].to_numpy(dtype=float))
                    - np.log1p(days["fallback_return"].to_numpy(dtype=float))
                ).mean()
            )
        else:
            realized = np.nan
        low_gap = float(mode_gap[low].mean()) if low.any() else np.nan
        high_gap = float(mode_gap[high].mean()) if high.any() else np.nan
        supported = bool(low.sum() >= 5 and high.sum() >= 5)
        rows.append(
            {
                **dict(zip(BLOCK_KEYS, key)),
                "status": block["status"],
                "selected": selected,
                "selected_tau": tau,
                "threshold_quantile": block["threshold_quantile"],
                "historical_policy_advantage_mean": block["policy_advantage_mean"],
                "historical_policy_advantage_lcb": block["policy_advantage_lcb"],
                "realized_daily_log_advantage_vs_fallback": realized,
                "realized_advantage_outcome": (
                    "win"
                    if realized > TOLERANCE
                    else "loss"
                    if realized < -TOLERANCE
                    else "tie"
                    if selected
                    else "not_selected"
                ),
                "informative_low_days": int(low.sum()),
                "informative_high_days": int(high.sum()),
                "realized_low_mode_gap": low_gap,
                "realized_high_mode_gap": high_gap,
                "both_sides_at_least_five_days": supported,
                "crossover_persisted_with_support": bool(
                    supported and low_gap > 0.0 and high_gap <= 0.0
                ),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != len(blocks):
        raise ValueError("daily decisions do not cover every 2020 block")
    return result


def mechanism_summary(blocks: pd.DataFrame) -> pd.DataFrame:
    selected = blocks.loc[blocks["selected"]]
    rows: list[dict[str, object]] = []
    groups = [("All", selected), *list(selected.groupby("pair", sort=True))]
    for pair, group in groups:
        outcomes = group["realized_advantage_outcome"].value_counts()
        supported = group.loc[group["both_sides_at_least_five_days"]]
        predicted = group["historical_policy_advantage_mean"]
        realized = group["realized_daily_log_advantage_vs_fallback"]
        rows.append(
            {
                "period": "2020",
                "pair": pair,
                "selected_blocks": len(group),
                "realized_wins": int(outcomes.get("win", 0)),
                "realized_ties": int(outcomes.get("tie", 0)),
                "realized_losses": int(outcomes.get("loss", 0)),
                "realized_win_rate": float((realized > TOLERANCE).mean()),
                "mean_daily_log_advantage_vs_fallback": float(realized.mean()),
                "predicted_realized_pearson": float(predicted.corr(realized)),
                "predicted_realized_spearman": float(
                    predicted.corr(realized, method="spearman")
                ),
                "blocks_with_both_sides_supported": len(supported),
                "supported_crossover_persistence_rate": (
                    float(supported["crossover_persisted_with_support"].mean())
                    if len(supported)
                    else np.nan
                ),
                "mean_selected_tau": float(group["selected_tau"].mean()),
                "sd_selected_tau": float(group["selected_tau"].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def tau_stability(blocks: pd.DataFrame) -> dict[str, float | int]:
    changes: list[float] = []
    for _key, sequence in blocks.sort_values(BLOCK_KEYS).groupby(
        ["pair", "repeat", "classifier_group"], sort=True
    ):
        selected = sequence["selected"].to_numpy(dtype=bool)
        tau = sequence["selected_tau"].to_numpy(dtype=float)
        for index in range(1, len(sequence)):
            if selected[index - 1] and selected[index]:
                changes.append(abs(tau[index] - tau[index - 1]))
    array = np.asarray(changes, dtype=float)
    return {
        "consecutive_selected_pairs": int(len(array)),
        "mean_absolute_tau_change": float(array.mean()) if len(array) else np.nan,
        "median_absolute_tau_change": float(np.median(array)) if len(array) else np.nan,
        "p90_absolute_tau_change": float(np.quantile(array, 0.9)) if len(array) else np.nan,
    }


def build_report(
    comparator: pd.DataFrame,
    configurations: pd.DataFrame,
    statuses: pd.DataFrame,
    mechanism: pd.DataFrame,
) -> str:
    def comparison(method: str, comparator_name: str) -> pd.Series:
        return comparator.loc[
            comparator["method"].eq(method)
            & comparator["comparator"].eq(comparator_name)
        ].iloc[0]

    guarded = comparison("Guarded", "Causal rolling-validation single RL")
    causal = comparison("Mechanism-only", "Causal rolling-validation single RL")
    stronger = comparison(
        "Mechanism-only", "Retrospectively stronger fixed single RL"
    )
    mechanism_row = mechanism.loc[mechanism["pair"].eq("All")].iloc[0]
    mechanism_configs = configurations.loc[configurations["method"].eq("Mechanism-only")]
    causal_config_wins = int(
        (mechanism_configs["delta_sharpe_vs_causal_single_mean"] > TOLERANCE).sum()
    )
    stronger_config_wins = int(
        (mechanism_configs["delta_sharpe_vs_stronger_mean"] > TOLERANCE).sum()
    )
    selected_blocks = int(
        statuses.loc[
            statuses["method"].eq("Mechanism-only")
            & statuses["status"].eq("selected"),
            "blocks",
        ].sum()
    )
    total_blocks = int(
        statuses.loc[statuses["method"].eq("Mechanism-only"), "blocks"].sum()
    )
    return f"""# DJ30 2020 causal crossover-tau comparison

The 2019 path is collection-only. Every 2020 threshold uses completed history
and is frozen before the next 63-session block. The experiment contains three
RL pairs, five fixed classifier groups, and 30 classifier refits (450 paths).

The guarded controller tied the causal single-RL baseline on
{int(guarded['ties'])}/450 paths. The mechanism-only controller selected
{selected_blocks}/{total_blocks} blocks, beat the causal baseline on
{int(causal['wins'])} paths, tied on {int(causal['ties'])}, and lost on
{int(causal['losses'])}; its mean Sharpe difference was
{causal['mean_delta_sharpe']:+.4f}. It beat the retrospectively stronger fixed
component on {int(stronger['wins'])}/450 paths with a mean Sharpe difference of
{stronger['mean_delta_sharpe']:+.4f}.

At configuration level, mechanism-only beat the causal baseline in
{causal_config_wins}/15 configurations and the retrospectively stronger fixed
component in {stronger_config_wins}/15. Among selected blocks, realized
same-state log-return advantage over fallback was positive in
{int(mechanism_row['realized_wins'])}, tied in
{int(mechanism_row['realized_ties'])}, and negative in
{int(mechanism_row['realized_losses'])}.

These are conditional results on the 2020 market path and fixed RL candidates.
They support a credible causal mechanism in complementary configurations, not
a universal deployment guarantee.
"""


def main() -> None:
    args = parse_args()
    guarded_root = Path(args.guarded_results)
    mechanism_root = Path(args.mechanism_only_results)
    output = Path(args.output_dir)
    validate_result_root(guarded_root, require_positive_lcb=True)
    validate_result_root(mechanism_root, require_positive_lcb=False)
    output.mkdir(parents=True, exist_ok=True)

    comparator_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    configuration_frames: list[pd.DataFrame] = []
    status_frames: list[pd.DataFrame] = []
    for method, root in (("Guarded", guarded_root), ("Mechanism-only", mechanism_root)):
        metrics = pd.read_csv(root / "all_crossover_tau_metrics.csv", dtype={"period": str})
        metrics = metrics.loc[metrics["period"].eq("2020")]
        comparator_frames.append(comparator_summary(metrics, method=method))
        pair_frames.append(pair_performance_summary(metrics, method=method))
        configuration_frames.append(configuration_summary(root, method=method))
        status_frames.append(status_summary(root, method=method))

    comparator = pd.concat(comparator_frames, ignore_index=True)
    pairs = pd.concat(pair_frames, ignore_index=True)
    configurations = pd.concat(configuration_frames, ignore_index=True)
    statuses = pd.concat(status_frames, ignore_index=True)
    block_validation = block_mechanism_validation(mechanism_root)
    mechanism = mechanism_summary(block_validation)
    stability = tau_stability(block_validation)

    comparator.to_csv(output / "sharpe_comparator_summary.csv", index=False)
    pairs.to_csv(output / "pair_performance_summary.csv", index=False)
    configurations.to_csv(output / "configuration_summary.csv", index=False)
    statuses.to_csv(output / "selection_status_summary.csv", index=False)
    block_validation.to_csv(output / "next_block_mechanism_validation.csv", index=False)
    mechanism.to_csv(output / "next_block_mechanism_summary.csv", index=False)
    (output / "tau_stability.json").write_text(
        json.dumps(stability, indent=2), encoding="utf-8"
    )
    report = build_report(comparator, configurations, statuses, mechanism)
    (output / "COMPARISON_REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
