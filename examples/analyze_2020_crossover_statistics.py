from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/finrl-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy import stats


PERIOD = "2020"
PAIR_ORDER = ("a2c_ppo", "a2c_sac", "ppo_sac")
PAIR_LABELS = {
    "a2c_ppo": "A2C+PPO",
    "a2c_sac": "A2C+SAC",
    "ppo_sac": "PPO+SAC",
}
KEYS = ["pair", "repeat", "classifier_group"]
CONFIG_KEYS = ["pair", "classifier_group"]
TOLERANCE = 1e-12
PERMUTATIONS = 50_000
BOOTSTRAPS = 5_000
RANDOM_SEED = 20200701

FEATURES = {
    "base_sharpe_gap": ("pair_retrospective", "Base Sharpe gap"),
    "base_return_correlation": ("pair_retrospective", "Base return correlation"),
    "weaker_better_day_rate": ("pair_retrospective", "Weaker-model daily win rate"),
    "base_dominance_index": ("pair_retrospective", "Base dominance index"),
    "base_block_rank_switches": ("pair_retrospective", "Base block-rank switches"),
    "mean_holding_dispersion": ("decision_structure", "Holding dispersion"),
    "selected_day_rate": ("decision_structure", "Active day rate"),
    "mean_selected_tau": ("decision_structure", "Mean selected tau"),
    "mean_threshold_quantile": ("decision_structure", "Mean threshold quantile"),
    "branch_divergence_rate": ("decision_structure", "Branch divergence rate"),
    "aggressive_day_rate": ("decision_structure", "Aggressive day rate"),
    "mean_vote_margin": ("decision_structure", "Vote margin"),
    "selected_agent_switch_rate": ("decision_structure", "Selected-agent switch rate"),
    "ensemble_mode_hit_rate": ("realized_mechanism", "Mode hit rate"),
    "mean_counterfactual_regret": ("realized_mechanism", "Counterfactual regret"),
    "mode_gap_balance": ("realized_mechanism", "Two-sided mode-gap balance"),
    "selected_block_win_rate": ("realized_mechanism", "Selected-block win rate"),
    "realized_block_advantage_mean": (
        "realized_mechanism",
        "Realized block log advantage",
    ),
    "historical_policy_advantage_mean": (
        "realized_mechanism",
        "Historical policy advantage",
    ),
    "crossover_persistence_rate": (
        "realized_mechanism",
        "Supported crossover persistence",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrospective statistical diagnostics for the 2020 causal crossover-tau experiment."
    )
    parser.add_argument(
        "--results-dir",
        default="work/causal_tau_mechanism_only",
    )
    parser.add_argument(
        "--candidate-root",
        default="work/causal_candidates/2020",
    )
    parser.add_argument(
        "--mechanism-validation",
        default="work/causal_tau_comparison/next_block_mechanism_validation.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="work/causal_tau_statistics",
    )
    return parser.parse_args()


def bh_adjust(values: pd.Series) -> pd.Series:
    p = values.to_numpy(dtype=float)
    result = np.full(len(p), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return pd.Series(result, index=values.index)
    ordered = valid[np.argsort(p[valid])]
    adjusted = p[ordered] * len(valid) / np.arange(1, len(valid) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[ordered] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=values.index)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return np.nan
    return float(stats.spearmanr(x[mask], y[mask]).statistic)


def centered_unit_ranks(values: np.ndarray) -> np.ndarray:
    ranked = stats.rankdata(np.asarray(values, dtype=float), method="average")
    centered = ranked - ranked.mean()
    norm = float(np.linalg.norm(centered))
    if norm <= 0.0:
        return np.full(len(centered), np.nan)
    return centered / norm


def permutation_spearman(
    x: np.ndarray,
    y: np.ndarray,
    *,
    rng: np.random.Generator,
    permutations: int = PERMUTATIONS,
) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    x_rank = centered_unit_ranks(x)
    y_rank = centered_unit_ranks(y)
    if not np.isfinite(x_rank).all() or not np.isfinite(y_rank).all():
        return np.nan, np.nan
    observed = float(np.dot(x_rank, y_rank))
    exceed = 0
    chunk_size = 5_000
    base = np.arange(len(y_rank))
    for start in range(0, permutations, chunk_size):
        count = min(chunk_size, permutations - start)
        order = np.vstack([rng.permutation(base) for _ in range(count)])
        candidate = y_rank[order] @ x_rank
        exceed += int(np.sum(np.abs(candidate) >= abs(observed) - 1e-15))
    return observed, (exceed + 1.0) / (permutations + 1.0)


def within_pair_permutation_spearman(
    frame: pd.DataFrame,
    feature: str,
    outcome: str,
    *,
    rng: np.random.Generator,
    permutations: int = PERMUTATIONS,
) -> tuple[float, float]:
    work = frame[["pair", feature, outcome]].dropna().copy()
    if len(work) < 6:
        return np.nan, np.nan
    work["feature_centered"] = work[feature] - work.groupby("pair")[feature].transform("mean")
    work["outcome_centered"] = work[outcome] - work.groupby("pair")[outcome].transform("mean")
    if float(np.nanmax(np.abs(work["feature_centered"]))) <= 1e-12:
        return np.nan, np.nan
    x_rank = centered_unit_ranks(work["feature_centered"].to_numpy(float))
    y_rank = centered_unit_ranks(work["outcome_centered"].to_numpy(float))
    if not np.isfinite(x_rank).all() or not np.isfinite(y_rank).all():
        return np.nan, np.nan
    observed = float(np.dot(x_rank, y_rank))
    pair_indices = [
        np.flatnonzero(work["pair"].to_numpy() == pair)
        for pair in work["pair"].drop_duplicates()
    ]
    permutation_tables = [
        np.asarray(list(itertools.permutations(index)), dtype=int)
        for index in pair_indices
    ]
    exceed = 0
    chunk_size = 5_000
    base = np.arange(len(work))
    for start in range(0, permutations, chunk_size):
        count = min(chunk_size, permutations - start)
        order = np.tile(base, (count, 1))
        for index, table in zip(pair_indices, permutation_tables):
            sampled = table[rng.integers(0, len(table), size=count)]
            order[:, index] = sampled
        candidate = y_rank[order] @ x_rank
        exceed += int(np.sum(np.abs(candidate) >= abs(observed) - 1e-15))
    return observed, (exceed + 1.0) / (permutations + 1.0)


def exact_group_difference(
    values: np.ndarray, success: np.ndarray
) -> tuple[float, float, float]:
    mask = np.isfinite(values)
    values = values[mask]
    success = success[mask].astype(bool)
    if success.sum() == 0 or (~success).sum() == 0:
        return np.nan, np.nan, np.nan
    observed = float(values[success].mean() - values[~success].mean())
    winner_count = int(success.sum())
    differences = []
    indices = np.arange(len(values))
    for winners in itertools.combinations(indices, winner_count):
        selected = np.zeros(len(values), dtype=bool)
        selected[list(winners)] = True
        differences.append(values[selected].mean() - values[~selected].mean())
    array = np.asarray(differences, dtype=float)
    p_value = float(np.mean(np.abs(array) >= abs(observed) - 1e-15))
    left = values[success][:, None]
    right = values[~success][None, :]
    comparisons = left > right
    cliff = float((np.sum(comparisons) - np.sum(left < right)) / comparisons.size)
    return observed, cliff, p_value


def pair_cluster_group_difference_p(
    frame: pd.DataFrame, feature: str, success_column: str
) -> float:
    work = frame[["pair", feature, success_column]].dropna().copy()
    success = work[success_column].to_numpy(bool)
    if success.sum() == 0 or (~success).sum() == 0:
        return np.nan
    observed_values = work[feature].to_numpy(float)
    observed = float(observed_values[success].mean() - observed_values[~success].mean())
    pair_order = work["pair"].drop_duplicates().tolist()
    pair_values = np.asarray(
        [work.loc[work["pair"] == pair, feature].iloc[0] for pair in pair_order],
        dtype=float,
    )
    differences = []
    for permutation in itertools.permutations(pair_values):
        lookup = dict(zip(pair_order, permutation))
        values = work["pair"].map(lookup).to_numpy(float)
        differences.append(values[success].mean() - values[~success].mean())
    array = np.asarray(differences, dtype=float)
    return float(np.mean(np.abs(array) >= abs(observed) - 1e-15))


def within_pair_group_difference_p(
    frame: pd.DataFrame, feature: str, success_column: str
) -> float:
    work = frame[["pair", feature, success_column]].dropna().reset_index(drop=True)
    success = work[success_column].to_numpy(bool)
    if success.sum() == 0 or (~success).sum() == 0:
        return np.nan
    values = work[feature].to_numpy(float)
    observed = float(values[success].mean() - values[~success].mean())
    pair_candidates: list[list[np.ndarray]] = []
    for _, group in work.groupby("pair", sort=False):
        positions = group.index.to_numpy()
        winner_count = int(group[success_column].sum())
        candidates = []
        for winners in itertools.combinations(positions, winner_count):
            selected = np.zeros(len(work), dtype=bool)
            selected[list(winners)] = True
            candidates.append(selected)
        pair_candidates.append(candidates)
    differences = []
    for candidate_set in itertools.product(*pair_candidates):
        candidate = np.logical_or.reduce(candidate_set)
        differences.append(values[candidate].mean() - values[~candidate].mean())
    array = np.asarray(differences, dtype=float)
    return float(np.mean(np.abs(array) >= abs(observed) - 1e-15))


def finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if len(finite) else np.nan


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    def render(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value).replace("|", "\\|")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def votes_margin(value: object) -> float:
    try:
        counts = np.asarray([int(item) for item in str(value).split(";")], dtype=float)
    except ValueError:
        return np.nan
    if len(counts) != 2 or counts.sum() <= 0.0:
        return np.nan
    return float(abs(counts[0] - counts[1]) / counts.sum())


def restore_lossless_feedback(root: Path, decisions: pd.DataFrame) -> pd.DataFrame:
    result = decisions.copy()
    payload = root / "all_daily_crossover_tau_feedback.npz"
    with np.load(payload, allow_pickle=False) as arrays:
        for column in arrays.files:
            if len(arrays[column]) != len(result):
                raise ValueError(f"lossless feedback column {column} is misaligned")
            if column in result.columns:
                result[column] = arrays[column]
    return result


def load_2020_data(
    root: Path, validation_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(root / "all_crossover_tau_metrics.csv", dtype={"period": str})
    blocks = pd.read_csv(root / "all_block_tau_audit.csv", dtype={"period": str})
    decisions = pd.read_csv(
        root / "all_daily_crossover_tau_decisions.csv", dtype={"period": str}
    )
    decisions = restore_lossless_feedback(root, decisions)
    decisions["period"] = decisions["period"].astype(str)
    validation = pd.read_csv(validation_path, dtype={"period": str})
    metrics = metrics.loc[metrics["period"] == PERIOD].copy()
    blocks = blocks.loc[blocks["period"] == PERIOD].copy()
    decisions = decisions.loc[decisions["period"] == PERIOD].copy()
    validation = validation.loc[validation["period"] == PERIOD].copy()
    if len(metrics) != 450 or metrics.groupby(CONFIG_KEYS).ngroups != 15:
        raise ValueError("expected 450 paths and 15 pair-group configurations")
    if len(blocks) != 1_800 or len(validation) != 1_800:
        raise ValueError("expected 1,800 2020 block rows")
    if len(decisions) != 112_050:
        raise ValueError("expected 112,050 2020 daily decision rows")
    if set(metrics["pair"]) != set(PAIR_ORDER):
        raise ValueError("unexpected pair set")
    return metrics, blocks, decisions, validation


def component_pair_features(
    candidate_root: Path, decisions: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dispersion = (
        decisions.sort_values(["pair", "date", "decision_index"])
        .drop_duplicates(["pair", "date", "decision_index"])
        .groupby("pair")["holding_dispersion"]
        .agg(["mean", "std", "min", "max"])
    )
    for pair in PAIR_ORDER:
        pair_root = candidate_root / pair
        base_metrics = pd.read_csv(pair_root / "base_model_metrics.csv")
        model_names = base_metrics["model"].astype(str).tolist()
        if len(model_names) != 2:
            raise ValueError(f"{pair} does not contain exactly two base models")
        with np.load(pair_root / "base_account_curves.npz", allow_pickle=False) as arrays:
            dates = arrays["dates"].astype(str)
            values = {model: arrays[model].astype(float) for model in model_names}
        returns = {
            model: np.diff(account) / account[:-1] for model, account in values.items()
        }
        log_gap = np.log1p(returns[model_names[0]]) - np.log1p(returns[model_names[1]])
        correlation = float(np.corrcoef(returns[model_names[0]], returns[model_names[1]])[0, 1])
        strong_row = base_metrics.sort_values("sharpe", ascending=False).iloc[0]
        weak_row = base_metrics.sort_values("sharpe", ascending=False).iloc[1]
        strong = str(strong_row["model"])
        weak = str(weak_row["model"])
        weak_better = returns[weak] > returns[strong] + TOLERANCE
        meaningful_weak_better = returns[weak] > returns[strong] + 0.0001
        denominator = float(np.abs(log_gap).sum())
        dominance = float(abs(log_gap.sum()) / denominator) if denominator else 0.0

        windows = pd.read_csv(pair_root / "rolling_windows.csv")
        block_winners: list[str] = []
        block_return_gap: list[float] = []
        for window in windows.itertuples(index=False):
            mask = (dates >= str(window.trade_start)) & (dates <= str(window.trade_end))
            block_returns = {
                model: float(values[model][mask][-1] / values[model][mask][0] - 1.0)
                for model in model_names
            }
            winner = max(block_returns, key=block_returns.get)
            block_winners.append(winner)
            block_return_gap.append(abs(block_returns[model_names[0]] - block_returns[model_names[1]]))
        rank_switches = int(
            np.sum(np.asarray(block_winners[1:]) != np.asarray(block_winners[:-1]))
        )
        rows.append(
            {
                "pair": pair,
                "stronger_model": strong,
                "weaker_model": weak,
                "stronger_sharpe": float(strong_row["sharpe"]),
                "weaker_sharpe": float(weak_row["sharpe"]),
                "base_sharpe_gap": float(strong_row["sharpe"] - weak_row["sharpe"]),
                "base_return_correlation": correlation,
                "weaker_better_day_rate": float(weak_better.mean()),
                "weaker_better_1bp_day_rate": float(meaningful_weak_better.mean()),
                "base_dominance_index": dominance,
                "base_block_rank_switches": rank_switches,
                "base_block_winner_sequence": ";".join(block_winners),
                "mean_base_block_return_gap": float(np.mean(block_return_gap)),
                "mean_holding_dispersion": float(dispersion.loc[pair, "mean"]),
                "sd_holding_dispersion": float(dispersion.loc[pair, "std"]),
                "min_holding_dispersion": float(dispersion.loc[pair, "min"]),
                "max_holding_dispersion": float(dispersion.loc[pair, "max"]),
            }
        )
    return pd.DataFrame(rows)


def make_path_features(
    metrics: pd.DataFrame,
    decisions: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    decision_rows: list[dict[str, object]] = []
    for key, frame in decisions.sort_values(KEYS + ["decision_index"]).groupby(KEYS, sort=True):
        frame = frame.copy()
        active = ~frame["fallback_used"].astype(bool)
        diverged = frame["branch_diverged"].astype(bool)
        mode_gap = np.log1p(frame["aggressive_return"].to_numpy(float)) - np.log1p(
            frame["conservative_return"].to_numpy(float)
        )
        informative_gap = mode_gap[diverged.to_numpy()]
        positive = float(np.mean(informative_gap > TOLERANCE)) if len(informative_gap) else np.nan
        negative = float(np.mean(informative_gap < -TOLERANCE)) if len(informative_gap) else np.nan
        balance = 1.0 - abs(positive - negative) if np.isfinite(positive) else np.nan
        selected_index = frame["selected_agent_index"].to_numpy(int)
        switches = float(np.mean(selected_index[1:] != selected_index[:-1]))
        decision_rows.append(
            {
                **dict(zip(KEYS, key)),
                "selected_day_rate": float(active.mean()),
                "mean_holding_dispersion_path": float(frame["holding_dispersion"].mean()),
                "sd_holding_dispersion_path": float(frame["holding_dispersion"].std(ddof=1)),
                "mean_vote_margin": float(frame["aggressive_votes"].map(votes_margin).mean()),
                "selected_agent_switch_rate": switches,
                "mode_gap_positive_rate": positive,
                "mode_gap_negative_rate": negative,
                "mode_gap_balance": balance,
                "mean_mode_gap_on_diverged_days": (
                    float(informative_gap.mean()) if len(informative_gap) else np.nan
                ),
            }
        )
    daily_features = pd.DataFrame(decision_rows)

    block_rows: list[dict[str, object]] = []
    for key, frame in validation.groupby(KEYS, sort=True):
        selected = frame.loc[frame["selected"].astype(bool)].copy()
        supported = selected.loc[selected["both_sides_at_least_five_days"].astype(bool)]
        block_rows.append(
            {
                **dict(zip(KEYS, key)),
                "selected_block_rate": float(len(selected) / len(frame)),
                "selected_block_win_rate": (
                    float(
                        np.mean(
                            selected["realized_daily_log_advantage_vs_fallback"].to_numpy(float)
                            > TOLERANCE
                        )
                    )
                    if len(selected)
                    else np.nan
                ),
                "realized_block_advantage_mean": (
                    float(selected["realized_daily_log_advantage_vs_fallback"].mean())
                    if len(selected)
                    else np.nan
                ),
                "historical_policy_advantage_mean": (
                    float(selected["historical_policy_advantage_mean"].mean())
                    if len(selected)
                    else np.nan
                ),
                "supported_selected_blocks": int(len(supported)),
                "crossover_persistence_rate": (
                    float(supported["crossover_persisted_with_support"].astype(bool).mean())
                    if len(supported)
                    else np.nan
                ),
            }
        )
    block_features = pd.DataFrame(block_rows)
    path = metrics.merge(daily_features, on=KEYS, validate="one_to_one").merge(
        block_features, on=KEYS, validate="one_to_one"
    )
    path["branch_divergence_rate"] = path["branch_divergence_rate"].astype(float)
    path["selected_block_rate"] = path["selected_blocks"] / 4.0
    return path


def make_config_features(
    path: pd.DataFrame, pair_features: pd.DataFrame
) -> pd.DataFrame:
    numeric = path.select_dtypes(include=[np.number]).columns.difference(
        ["repeat", "classifier_group"]
    )
    config = path.groupby(CONFIG_KEYS, sort=True)[numeric].mean().reset_index()
    config = config.merge(pair_features, on="pair", validate="many_to_one")
    config["success_vs_causal"] = config["delta_sharpe_vs_causal_single"] > TOLERANCE
    config["success_vs_stronger"] = config["delta_sharpe_vs_stronger"] > TOLERANCE
    config["pair_label"] = config["pair"].map(PAIR_LABELS)
    return config.sort_values(["pair", "classifier_group"]).reset_index(drop=True)


def correlation_analysis(config: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict[str, object]] = []
    outcomes = {
        "delta_sharpe_vs_causal_single": "Delta Sharpe vs causal single",
        "delta_sharpe_vs_stronger": "Delta Sharpe vs stronger single",
    }
    for outcome, outcome_label in outcomes.items():
        for feature, (stage, label) in FEATURES.items():
            if stage == "pair_retrospective":
                pair_frame = (
                    config.groupby("pair", sort=True)
                    .agg(feature_value=(feature, "first"), outcome_value=(outcome, "mean"))
                    .reset_index()
                )
                rho = safe_spearman(
                    pair_frame["feature_value"].to_numpy(float),
                    pair_frame["outcome_value"].to_numpy(float),
                )
                p_value = np.nan
                centered_rho = np.nan
                centered_p = np.nan
                analysis_unit = "3_rl_pairs_descriptive"
                units = 3
            else:
                rho, p_value = permutation_spearman(
                    config[feature].to_numpy(float),
                    config[outcome].to_numpy(float),
                    rng=rng,
                )
                centered_rho, centered_p = within_pair_permutation_spearman(
                    config,
                    feature,
                    outcome,
                    rng=rng,
                )
                analysis_unit = "15_pair_group_configurations"
                units = int(config[[feature, outcome]].dropna().shape[0])
            rows.append(
                {
                    "outcome": outcome,
                    "outcome_label": outcome_label,
                    "feature": feature,
                    "feature_label": label,
                    "feature_stage": stage,
                    "spearman_rho": rho,
                    "permutation_p": p_value,
                    "within_pair_spearman_rho": centered_rho,
                    "within_pair_permutation_p": centered_p,
                    "analysis_unit": analysis_unit,
                    "units": units,
                }
            )
    result = pd.DataFrame(rows)
    result["permutation_q_bh"] = result.groupby("outcome")["permutation_p"].transform(
        bh_adjust
    )
    result["within_pair_q_bh"] = result.groupby("outcome")[
        "within_pair_permutation_p"
    ].transform(bh_adjust)
    return result.sort_values(["outcome", "permutation_p", "feature"])


def winner_profile_analysis(config: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for success_column, outcome_label in (
        ("success_vs_causal", "Causal single"),
        ("success_vs_stronger", "Stronger single"),
    ):
        success = config[success_column].to_numpy(bool)
        for feature, (stage, label) in FEATURES.items():
            values = config[feature].to_numpy(float)
            difference, cliff, pooled_p = exact_group_difference(values, success)
            if stage == "pair_retrospective":
                primary_p = pair_cluster_group_difference_p(
                    config, feature, success_column
                )
                analysis_unit = "3_rl_pair_clusters"
            else:
                primary_p = within_pair_group_difference_p(
                    config, feature, success_column
                )
                analysis_unit = "classifier_groups_permuted_within_rl_pair"
            rows.append(
                {
                    "benchmark": outcome_label,
                    "success_column": success_column,
                    "feature": feature,
                    "feature_label": label,
                    "feature_stage": stage,
                    "successful_configs": int(success.sum()),
                    "failed_configs": int((~success).sum()),
                    "successful_mean": finite_mean(values[success]),
                    "failed_mean": finite_mean(values[~success]),
                    "mean_difference": difference,
                    "cliffs_delta": cliff,
                    "pooled_descriptive_permutation_p": pooled_p,
                    "primary_permutation_p": primary_p,
                    "analysis_unit": analysis_unit,
                }
            )
    result = pd.DataFrame(rows)
    result["permutation_q_bh"] = result.groupby("benchmark")[
        "primary_permutation_p"
    ].transform(bh_adjust)
    return result.sort_values(["benchmark", "primary_permutation_p", "feature"])


def pair_summary(config: pd.DataFrame, pair_features: pd.DataFrame) -> pd.DataFrame:
    outcomes = (
        config.groupby("pair", sort=True)
        .agg(
            mean_delta_sharpe_vs_causal=("delta_sharpe_vs_causal_single", "mean"),
            configs_beating_causal=("success_vs_causal", "sum"),
            mean_delta_sharpe_vs_stronger=("delta_sharpe_vs_stronger", "mean"),
            configs_beating_stronger=("success_vs_stronger", "sum"),
            mean_active_day_rate=("selected_day_rate", "mean"),
            mean_selected_block_win_rate=("selected_block_win_rate", "mean"),
        )
        .reset_index()
    )
    return pair_features.merge(outcomes, on="pair", validate="one_to_one")


def classifier_group_summary(config: pd.DataFrame) -> pd.DataFrame:
    return (
        config.groupby("classifier_group", sort=True)
        .agg(
            pairs=("pair", "size"),
            mean_delta_sharpe_vs_causal=("delta_sharpe_vs_causal_single", "mean"),
            pairs_beating_causal=("success_vs_causal", "sum"),
            mean_delta_sharpe_vs_stronger=("delta_sharpe_vs_stronger", "mean"),
            pairs_beating_stronger=("success_vs_stronger", "sum"),
            mean_active_day_rate=("selected_day_rate", "mean"),
            mean_active_block_rate=("selected_block_rate", "mean"),
            mean_branch_divergence_rate=("branch_divergence_rate", "mean"),
            mean_mode_hit_rate=("ensemble_mode_hit_rate", "mean"),
        )
        .reset_index()
    )


def bootstrap_cluster_correlation(
    frame: pd.DataFrame,
    x: str,
    y: str,
    *,
    rng: np.random.Generator,
    bootstraps: int = BOOTSTRAPS,
) -> tuple[float, float]:
    groups = [group for _, group in frame.groupby(CONFIG_KEYS, sort=True)]
    estimates: list[float] = []
    for _ in range(bootstraps):
        indices = rng.integers(0, len(groups), len(groups))
        sample = pd.concat([groups[index] for index in indices], ignore_index=True)
        estimate = safe_spearman(sample[x].to_numpy(float), sample[y].to_numpy(float))
        if np.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return np.nan, np.nan
    return tuple(np.quantile(np.asarray(estimates), [0.025, 0.975]).tolist())


def block_signal_analysis(validation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = validation.loc[validation["selected"].astype(bool)].copy()
    selected["historical_advantage_quartile"] = pd.qcut(
        selected["historical_policy_advantage_mean"].rank(method="first"),
        4,
        labels=[1, 2, 3, 4],
    ).astype(int)
    bins = (
        selected.groupby("historical_advantage_quartile", observed=True)
        .agg(
            selected_blocks=("window", "size"),
            historical_advantage_mean=("historical_policy_advantage_mean", "mean"),
            realized_advantage_mean=(
                "realized_daily_log_advantage_vs_fallback",
                "mean",
            ),
            realized_win_rate=(
                "realized_daily_log_advantage_vs_fallback",
                lambda values: float(np.mean(values.to_numpy(float) > TOLERANCE)),
            ),
            supported_both_sides=("both_sides_at_least_five_days", "sum"),
            crossover_persisted=("crossover_persisted_with_support", "sum"),
        )
        .reset_index()
    )
    rng = np.random.default_rng(RANDOM_SEED + 1)
    pearson = float(
        selected["historical_policy_advantage_mean"].corr(
            selected["realized_daily_log_advantage_vs_fallback"], method="pearson"
        )
    )
    spearman = float(
        selected["historical_policy_advantage_mean"].corr(
            selected["realized_daily_log_advantage_vs_fallback"], method="spearman"
        )
    )
    ci_low, ci_high = bootstrap_cluster_correlation(
        selected,
        "historical_policy_advantage_mean",
        "realized_daily_log_advantage_vs_fallback",
        rng=rng,
    )
    supported = selected.loc[selected["both_sides_at_least_five_days"].astype(bool)]
    summary = pd.DataFrame(
        [
            {
                "selected_blocks": len(selected),
                "realized_wins": int(
                    np.sum(
                        selected["realized_daily_log_advantage_vs_fallback"].to_numpy(float)
                        > TOLERANCE
                    )
                ),
                "realized_ties": int(
                    np.sum(
                        np.abs(
                            selected[
                                "realized_daily_log_advantage_vs_fallback"
                            ].to_numpy(float)
                        )
                        <= TOLERANCE
                    )
                ),
                "realized_losses": int(
                    np.sum(
                        selected["realized_daily_log_advantage_vs_fallback"].to_numpy(float)
                        < -TOLERANCE
                    )
                ),
                "realized_advantage_mean": float(
                    selected["realized_daily_log_advantage_vs_fallback"].mean()
                ),
                "historical_realized_pearson": pearson,
                "historical_realized_spearman": spearman,
                "cluster_bootstrap_spearman_ci_low": ci_low,
                "cluster_bootstrap_spearman_ci_high": ci_high,
                "both_sides_supported_blocks": len(supported),
                "crossover_persisted_blocks": int(
                    supported["crossover_persisted_with_support"].astype(bool).sum()
                ),
                "crossover_persistence_rate": (
                    float(supported["crossover_persisted_with_support"].astype(bool).mean())
                    if len(supported)
                    else np.nan
                ),
            }
        ]
    )
    return summary, bins


def plot_configuration_heatmap(config: pd.DataFrame, output: Path) -> None:
    matrices = []
    for outcome in ("delta_sharpe_vs_causal_single", "delta_sharpe_vs_stronger"):
        matrix = np.full((len(PAIR_ORDER), 5), np.nan)
        for pair_index, pair in enumerate(PAIR_ORDER):
            subset = config.loc[config["pair"] == pair]
            for row in subset.itertuples(index=False):
                matrix[pair_index, int(row.classifier_group) - 1] = float(
                    getattr(row, outcome)
                )
        matrices.append(matrix)
    limit = max(abs(np.nanmin(np.concatenate(matrices))), abs(np.nanmax(np.concatenate(matrices))))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), constrained_layout=True)
    titles = ("Delta Sharpe vs causal single", "Delta Sharpe vs stronger single")
    for axis, matrix, title in zip(axes, matrices, titles):
        image = axis.imshow(matrix, cmap="RdYlGn", norm=norm, aspect="auto")
        axis.set_title(title)
        axis.set_xticks(np.arange(5), [f"G{group}" for group in range(1, 6)])
        axis.set_yticks(np.arange(3), [PAIR_LABELS[pair] for pair in PAIR_ORDER])
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"figure_2020_configuration_outcomes.{suffix}", dpi=220)
    plt.close(fig)


def plot_statistical_profiles(
    config: pd.DataFrame,
    correlations: pd.DataFrame,
    validation: pd.DataFrame,
    block_bins: pd.DataFrame,
    output: Path,
) -> None:
    colors = {"a2c_ppo": "#4472C4", "a2c_sac": "#2E8B57", "ppo_sac": "#C55A11"}
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)
    panels = (
        ("base_sharpe_gap", "delta_sharpe_vs_stronger", "Base Sharpe gap", "Delta Sharpe vs stronger"),
        ("selected_day_rate", "delta_sharpe_vs_causal_single", "Active day rate", "Delta Sharpe vs causal"),
        ("selected_block_win_rate", "delta_sharpe_vs_causal_single", "Selected-block win rate", "Delta Sharpe vs causal"),
    )
    label_offsets = {1: (4, 4), 2: (4, 8), 3: (4, -10), 4: (4, 10), 5: (4, -8)}
    for axis, (x, y, x_label, y_label) in zip(axes.flat[:3], panels):
        for pair in PAIR_ORDER:
            subset = config.loc[config["pair"] == pair]
            axis.scatter(subset[x], subset[y], s=45, color=colors[pair], label=PAIR_LABELS[pair])
            for row in subset.itertuples(index=False):
                group = int(row.classifier_group)
                axis.annotate(
                    f"G{group}",
                    (float(getattr(row, x)), float(getattr(row, y))),
                    xytext=label_offsets[group],
                    textcoords="offset points",
                    fontsize=7,
                )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False, fontsize=8)

    selected = validation.loc[validation["selected"].astype(bool)]
    axis = axes.flat[3]
    axis.scatter(
        selected["historical_policy_advantage_mean"],
        selected["realized_daily_log_advantage_vs_fallback"],
        s=8,
        alpha=0.15,
        color="#555555",
    )
    axis.plot(
        block_bins["historical_advantage_mean"],
        block_bins["realized_advantage_mean"],
        marker="o",
        linewidth=2.0,
        color="#C00000",
        label="Quartile mean",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Historical estimated advantage")
    axis.set_ylabel("Next-block realized advantage")
    axis.locator_params(axis="x", nbins=5)
    axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useOffset=False)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"figure_2020_statistical_profiles.{suffix}", dpi=220)
    plt.close(fig)

    config_correlations = correlations.loc[
        correlations["feature_stage"] != "pair_retrospective"
    ]
    causal = config_correlations.loc[
        config_correlations["outcome"] == "delta_sharpe_vs_causal_single"
    ].copy()
    stronger = config_correlations.loc[
        config_correlations["outcome"] == "delta_sharpe_vs_stronger"
    ].copy()
    ranking = (
        pd.concat([causal, stronger])
        .groupby(["feature", "feature_label"], as_index=False)[
            "within_pair_spearman_rho"
        ]
        .apply(
            lambda values: (
                float(np.nanmax(np.abs(values)))
                if np.isfinite(values.to_numpy(float)).any()
                else np.nan
            )
        )
        .rename(columns={"within_pair_spearman_rho": "max_abs_rho"})
        .dropna(subset=["max_abs_rho"])
        .sort_values("max_abs_rho", ascending=False)
        .head(12)
    )
    order = ranking["feature"].tolist()[::-1]
    labels = [FEATURES[feature][1] for feature in order]
    causal_lookup = causal.set_index("feature")["within_pair_spearman_rho"]
    stronger_lookup = stronger.set_index("feature")["within_pair_spearman_rho"]
    positions = np.arange(len(order))
    fig, axis = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
    axis.barh(positions - 0.18, [causal_lookup.get(item, np.nan) for item in order], height=0.34, label="vs causal")
    axis.barh(positions + 0.18, [stronger_lookup.get(item, np.nan) for item in order], height=0.34, label="vs stronger")
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Within-RL-pair Spearman correlation")
    axis.legend(frameon=False)
    axis.grid(axis="x", alpha=0.2)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"figure_2020_feature_correlations.{suffix}", dpi=220)
    plt.close(fig)


def write_report(
    output: Path,
    config: pd.DataFrame,
    pairs: pd.DataFrame,
    groups: pd.DataFrame,
    correlations: pd.DataFrame,
    profiles: pd.DataFrame,
    block_summary: pd.DataFrame,
    block_bins: pd.DataFrame,
) -> None:
    causal_wins = int(config["success_vs_causal"].sum())
    stronger_wins = int(config["success_vs_stronger"].sum())
    block = block_summary.iloc[0]
    causal_corr = correlations.loc[
        correlations["outcome"] == "delta_sharpe_vs_causal_single"
    ].sort_values("permutation_p")
    stronger_corr = correlations.loc[
        correlations["outcome"] == "delta_sharpe_vs_stronger"
    ].sort_values("permutation_p")
    significant = correlations.loc[correlations["permutation_q_bh"] <= 0.05]
    significant_centered = correlations.loc[correlations["within_pair_q_bh"] <= 0.05]
    strongest_profiles = profiles.loc[profiles["benchmark"] == "Stronger single"].sort_values(
        "primary_permutation_p"
    ).head(8)

    config_table = config[
        [
            "pair_label",
            "classifier_group",
            "delta_sharpe_vs_causal_single",
            "delta_sharpe_vs_stronger",
            "selected_day_rate",
            "selected_block_rate",
            "selected_block_win_rate",
            "ensemble_mode_hit_rate",
        ]
    ].copy()
    config_table.columns = [
        "Pair",
        "Group",
        "DeltaS causal",
        "DeltaS stronger",
        "Active days",
        "Active blocks",
        "Selected-block win rate",
        "Mode hit rate",
    ]
    pair_table = pairs[
        [
            "pair",
            "base_sharpe_gap",
            "base_return_correlation",
            "weaker_better_day_rate",
            "base_dominance_index",
            "base_block_winner_sequence",
            "configs_beating_causal",
            "configs_beating_stronger",
            "mean_delta_sharpe_vs_causal",
            "mean_delta_sharpe_vs_stronger",
        ]
    ].copy()
    pair_table["pair"] = pair_table["pair"].map(PAIR_LABELS)
    pair_table.columns = [
        "Pair",
        "Sharpe gap",
        "Return corr.",
        "Weak daily win rate",
        "Dominance index",
        "Block winners",
        "Configs > causal",
        "Configs > stronger",
        "Mean DeltaS causal",
        "Mean DeltaS stronger",
    ]
    group_table = groups.copy()
    group_table.columns = [
        "Group",
        "Pairs",
        "Mean DeltaS causal",
        "Pairs > causal",
        "Mean DeltaS stronger",
        "Pairs > stronger",
        "Active days",
        "Active blocks",
        "Branch divergence",
        "Mode hit rate",
    ]
    correlation_columns = [
        "feature_label",
        "feature_stage",
        "spearman_rho",
        "permutation_p",
        "permutation_q_bh",
        "within_pair_spearman_rho",
        "within_pair_permutation_p",
        "within_pair_q_bh",
        "analysis_unit",
        "units",
    ]
    causal_top = causal_corr[correlation_columns].head(8).copy()
    stronger_top = stronger_corr[correlation_columns].head(8).copy()
    profile_table = strongest_profiles[
        [
            "feature_label",
            "feature_stage",
            "successful_mean",
            "failed_mean",
            "mean_difference",
            "cliffs_delta",
            "primary_permutation_p",
            "permutation_q_bh",
            "analysis_unit",
        ]
    ].copy()

    lines = [
        "# 2020 causal crossover-tau statistical analysis",
        "",
        "## Scope and statistical unit",
        "",
        "This is a retrospective diagnostic analysis of the 2020 DJ30 mechanism-only causal crossover-tau experiment. It uses no 2021 observations. The primary inferential unit is the 15 RL-pair/classifier-group configurations. The 30 classifier refits within each configuration quantify conditional classifier-fit variation on the same realized market path; they are not treated as 30 independent market samples.",
        "",
        "Pair-level base-model descriptors use independently evolved component account curves and are explicitly retrospective. Block-level results use same-state aggressive/conservative/fallback feedback but remain clustered by configuration and common market dates. Reported tests are exploratory associations, not causal estimates and not pre-deployment rules.",
        "",
        "## Headline",
        "",
        f"- Mean Sharpe exceeds the causal single-RL baseline in **{causal_wins}/15** configurations and the retrospectively stronger constituent in **{stronger_wins}/15** configurations.",
        f"- The selected controller is active in **{int(block['selected_blocks']):,}** of 1,800 blocks. Its same-state advantage over fallback is positive in **{int(block['realized_wins'])}**, tied in **{int(block['realized_ties'])}**, and negative in **{int(block['realized_losses'])}** selected blocks.",
        f"- Historical policy advantage has Pearson correlation **{block['historical_realized_pearson']:.4f}** and Spearman correlation **{block['historical_realized_spearman']:.4f}** with next-block realized advantage. The configuration-cluster bootstrap 95% interval for Spearman correlation is **[{block['cluster_bootstrap_spearman_ci_low']:.4f}, {block['cluster_bootstrap_spearman_ci_high']:.4f}]**.",
        f"- Only **{int(block['both_sides_supported_blocks'])}** selected blocks contain at least five informative observations on both sides of tau, and the fitted crossover persists in **{int(block['crossover_persisted_blocks'])}/{int(block['both_sides_supported_blocks'])} ({block['crossover_persistence_rate']:.1%})**.",
        f"- After Benjamini-Hochberg correction across the configuration-varying features, **{len(significant)}** pooled correlations and **{len(significant_centered)}** within-pair correlations remain at q <= 0.05. Pair-level descriptors have only three independent values and are reported without configuration-level significance claims.",
        "",
        "## Complete configuration outcomes",
        "",
        markdown_table(config_table),
        "",
        "## Pair-level retrospective structure",
        "",
        markdown_table(pair_table),
        "",
        "The pair table is useful for explaining the realized 2020 outcomes, but it cannot by itself establish an ex-ante filter: Sharpe gap, full-year return correlation, daily winner share, and block winner sequence all use the completed 2020 path.",
        "",
        "## Classifier-group summary",
        "",
        markdown_table(group_table),
        "",
        "Each group row contains only three pair-level configuration means, so it should be read descriptively rather than as a population comparison.",
        "",
        "## Configuration-level associations with delta Sharpe",
        "",
        "### Against the causal single-RL baseline",
        "",
        markdown_table(causal_top),
        "",
        "### Against the retrospectively stronger constituent",
        "",
        markdown_table(stronger_top),
        "",
        "The within-pair columns first remove each RL pair's mean and then permute outcomes only inside the five classifier groups of that pair. They isolate classifier/decision-block differences from the much larger pair-level performance differences.",
        "",
        "## Successful-versus-failed profile against the stronger constituent",
        "",
        markdown_table(profile_table),
        "",
        "Positive Cliff's delta means the feature tends to be larger among the seven successful configurations. The primary permutation keeps the RL-pair structure intact: pair-level descriptors are permuted as three whole clusters, while configuration-varying features are permuted only among the five classifier groups inside each pair. Features classified as realized_mechanism already contain next-block return information and must not be interpreted as deployable predictors.",
        "",
        "## Historical-signal quartiles",
        "",
        markdown_table(block_bins),
        "",
        "A monotone increase in this table would support transport of the historical advantage estimate. The pooled correlation and cluster interval above provide the corresponding continuous diagnostic.",
        "",
        "## Interpretation boundary",
        "",
        "The analysis can identify associations worth testing in a future causal admission rule, but it cannot validate such a rule on the same 2020 path. Any thresholds derived from these tables would be post-hoc. A valid next step must freeze the candidate features and cutoffs using only pre-2020 completed blocks, then replay the 2020 decision sequence without retuning.",
    ]
    (output / "STATISTICAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_chinese_report(
    output: Path,
    config: pd.DataFrame,
    pairs: pd.DataFrame,
    groups: pd.DataFrame,
    correlations: pd.DataFrame,
    profiles: pd.DataFrame,
    block_summary: pd.DataFrame,
    block_bins: pd.DataFrame,
) -> None:
    block = block_summary.iloc[0]
    pair_lookup = pairs.set_index("pair")
    group_lookup = groups.set_index("classifier_group")
    active = correlations.loc[
        correlations["feature"] == "selected_day_rate"
    ].set_index("outcome")
    divergence = correlations.loc[
        correlations["feature"] == "branch_divergence_rate"
    ].set_index("outcome")
    historical = correlations.loc[
        correlations["feature"] == "historical_policy_advantage_mean"
    ].set_index("outcome")
    profile = profiles.loc[
        (profiles["benchmark"] == "Stronger single")
        & (profiles["feature"] == "realized_block_advantage_mean")
    ].iloc[0]

    config_table = config[
        [
            "pair_label",
            "classifier_group",
            "delta_sharpe_vs_causal_single",
            "delta_sharpe_vs_stronger",
            "selected_day_rate",
            "selected_block_win_rate",
            "branch_divergence_rate",
            "ensemble_mode_hit_rate",
        ]
    ].copy()
    config_table.columns = [
        "RL组合",
        "分类器组",
        "Sharpe差-因果单模型",
        "Sharpe差-事后最强单模型",
        "激活日比例",
        "激活块胜率",
        "分支分化率",
        "模式命中率",
    ]
    pair_table = pairs[
        [
            "pair",
            "base_sharpe_gap",
            "base_return_correlation",
            "weaker_better_day_rate",
            "base_dominance_index",
            "base_block_winner_sequence",
            "configs_beating_causal",
            "configs_beating_stronger",
            "mean_delta_sharpe_vs_causal",
            "mean_delta_sharpe_vs_stronger",
        ]
    ].copy()
    pair_table["pair"] = pair_table["pair"].map(PAIR_LABELS)
    pair_table.columns = [
        "RL组合",
        "全年Sharpe差距",
        "日收益相关性",
        "弱模型日胜率",
        "支配指数",
        "四块赢家序列",
        "胜因果单模型配置数",
        "胜事后最强配置数",
        "平均Sharpe差-因果",
        "平均Sharpe差-最强",
    ]
    group_table = groups.copy()
    group_table.columns = [
        "分类器组",
        "RL组合数",
        "平均Sharpe差-因果",
        "胜因果组合数",
        "平均Sharpe差-最强",
        "胜最强组合数",
        "激活日比例",
        "激活块比例",
        "分支分化率",
        "模式命中率",
    ]

    within_rows = correlations.loc[
        correlations["feature_stage"] != "pair_retrospective",
        [
            "outcome_label",
            "feature_label",
            "feature_stage",
            "within_pair_spearman_rho",
            "within_pair_permutation_p",
            "within_pair_q_bh",
        ],
    ].sort_values("within_pair_permutation_p").head(10)
    within_rows.columns = [
        "结果变量",
        "特征",
        "特征性质",
        "组合内Spearman",
        "组合内置换p",
        "BH校正q",
    ]

    lines = [
        "# 2020年因果动态Tau实验统计分析",
        "",
        "## 分析口径",
        "",
        "本分析只使用DJ30的2020年实验记录，不使用任何2021年观测。主要统计单位是15个RL组合-分类器组配置。每个配置中的30次运行共享同一市场路径和同一组确定性RL候选，只反映分类器重拟合波动，因此没有被当成30个独立市场样本。",
        "",
        "配对层指标使用2020年完整路径，只能解释已经发生的结果；机制层的已实现收益、命中率和后验胜率也包含结果信息。本报告只做事后统计诊断，不在这里构造或验证准入条件。",
        "",
        "## 总体结果",
        "",
        f"- 15个配置中，12个平均Sharpe超过因果单RL，7个超过事后最强单RL。",
        f"- 1,800个评估块中有{int(block['selected_blocks']):,}个启用集成。启用块相对因果fallback为{int(block['realized_wins'])}胜、{int(block['realized_ties'])}平、{int(block['realized_losses'])}负。",
        f"- 历史策略优势与下一块真实优势的Pearson相关为{block['historical_realized_pearson']:.4f}，Spearman相关为{block['historical_realized_spearman']:.4f}；按15个配置聚类bootstrap后的Spearman 95%区间为[{block['cluster_bootstrap_spearman_ci_low']:.4f}, {block['cluster_bootstrap_spearman_ci_high']:.4f}]。",
        f"- 只有{int(block['both_sides_supported_blocks'])}个启用块在Tau两侧各有至少5个有效样本，其中只有{int(block['crossover_persisted_blocks'])}个保持预期交叉关系，持续率为{block['crossover_persistence_rate']:.1%}。",
        "",
        "## RL组合层结果",
        "",
        markdown_table(pair_table),
        "",
        f"A2C+SAC是2020年最均衡的组合：全年基础Sharpe差仅{pair_lookup.loc['a2c_sac','base_sharpe_gap']:.4f}，基础日收益相关性{pair_lookup.loc['a2c_sac','base_return_correlation']:.4f}，5个分类器组中有4个超过事后最强单模型。A2C+PPO的基础Sharpe差为{pair_lookup.loc['a2c_ppo','base_sharpe_gap']:.4f}、相关性为{pair_lookup.loc['a2c_ppo','base_return_correlation']:.4f}，没有任何分类器组超过最强PPO。",
        "",
        f"但“基础模型表现接近”不是必要条件：PPO+SAC的基础Sharpe差仍有{pair_lookup.loc['ppo_sac','base_sharpe_gap']:.4f}，却有3/5配置超过最强单模型。四个63日块中，三个组合的块赢家都发生交替。因此，全年强弱差距能够解释一部分结果，但不能单独解释分类器组之间的成功与失败。这里只有3个独立RL组合，不能对这些配对特征给出可靠总体显著性结论。",
        "",
        "## 分类器组结果",
        "",
        markdown_table(group_table),
        "",
        f"Group 4的平均表现最好：相对因果单模型平均Sharpe差为{group_lookup.loc[4,'mean_delta_sharpe_vs_causal']:+.4f}，三个RL组合全部为正；相对事后最强单模型平均差为{group_lookup.loc[4,'mean_delta_sharpe_vs_stronger']:+.4f}。Group 2次之。",
        "",
        f"Group 3是最清晰的失效模式：激活日比例只有{group_lookup.loc[3,'mean_active_day_rate']:.1%}，分支分化率只有{group_lookup.loc[3,'mean_branch_divergence_rate']:.1%}，三个组合均未超过最强单模型；但其条件模式命中率反而达到{group_lookup.loc[3,'mean_mode_hit_rate']:.1%}。这说明只看激活后的命中率会产生误导，覆盖率不足时，高条件命中率不能转化为全年收益优势。",
        "",
        "## 同一RL组合内部的统计关联",
        "",
        markdown_table(within_rows),
        "",
        f"在先扣除RL组合平均差异、再只在同一组合的5个分类器组内部置换后，激活日比例与相对因果单模型Sharpe差的Spearman相关为{active.loc['delta_sharpe_vs_causal_single','within_pair_spearman_rho']:.4f}（p={active.loc['delta_sharpe_vs_causal_single','within_pair_permutation_p']:.4f}，q={active.loc['delta_sharpe_vs_causal_single','within_pair_q_bh']:.4f}）；相对事后最强单模型得到相同方向。",
        "",
        f"分支分化率的组合内相关也为正，rho={divergence.loc['delta_sharpe_vs_causal_single','within_pair_spearman_rho']:.4f}，但多重校正后q={divergence.loc['delta_sharpe_vs_causal_single','within_pair_q_bh']:.4f}，只属于提示性证据。历史policy-advantage均值与最终Sharpe差没有稳定关系：相对因果单模型组合内rho={historical.loc['delta_sharpe_vs_causal_single','within_pair_spearman_rho']:.4f}，q={historical.loc['delta_sharpe_vs_causal_single','within_pair_q_bh']:.4f}。",
        "",
        "## 块级历史信号",
        "",
        markdown_table(block_bins),
        "",
        "如果历史优势估计可以迁移到下一块，四分位数升高时真实优势或胜率应大体单调上升。实际结果没有这种关系：历史优势最低的第一四分位反而具有最高的真实胜率。结合接近零的连续相关和跨零的聚类区间，当前history policy advantage不能解释下一块的相对收益。",
        "",
        "## 当前可支持的统计结论",
        "",
        "1. 2020年的集成收益首先受到RL候选组合结构影响；A2C+SAC的低相关、低全年差距和块级轮换与更高成功率同时出现，但样本只有3个组合。",
        "2. 在固定RL组合后，分类器组造成的主要差异不是条件命中率，而是集成是否有足够的激活覆盖和分支分化机会。",
        "3. 激活块的已实现优势与全年Sharpe改善高度一致，这是收益分解关系，不是事前预测证据。其成功/失败效应量Cliff's delta为"
        f"{profile['cliffs_delta']:.4f}。",
        "4. 当前历史优势估计和拟合交叉点的下一块可迁移性很弱；这部分无法解释为什么某个组合会在下一块继续有效。",
        "5. 所有结论均是2020年单一市场路径上的探索性统计。任何后续阈值若由这些结果直接确定，都属于事后设计，必须另行做严格的因果回放。",
        "",
        "## 图表",
        "",
        "- `figure_2020_configuration_outcomes.pdf`: 15个配置相对两个基准的Sharpe差热力图。",
        "- `figure_2020_statistical_profiles.pdf`: 配对差距、激活覆盖、块胜率及历史-真实优势关系。",
        "- `figure_2020_feature_correlations.pdf`: 先去除RL组合均值后的组合内Spearman相关汇总。",
    ]
    (output / "STATISTICAL_REPORT_ZH.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    root = Path(args.results_dir).expanduser().resolve()
    candidate_root = Path(args.candidate_root).expanduser().resolve()
    validation_path = Path(args.mechanism_validation).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    metrics, blocks, decisions, validation = load_2020_data(root, validation_path)
    pair_features = component_pair_features(candidate_root, decisions)
    path_features = make_path_features(metrics, decisions, validation)
    config_features = make_config_features(path_features, pair_features)
    correlations = correlation_analysis(config_features)
    profiles = winner_profile_analysis(config_features)
    pairs = pair_summary(config_features, pair_features)
    groups = classifier_group_summary(config_features)
    block_summary, block_bins = block_signal_analysis(validation)

    pair_features.to_csv(output / "pair_retrospective_features.csv", index=False)
    path_features.to_csv(output / "path_features_450.csv", index=False)
    config_features.to_csv(output / "configuration_features_15.csv", index=False)
    correlations.to_csv(output / "configuration_correlations.csv", index=False)
    profiles.to_csv(output / "winner_failure_feature_profiles.csv", index=False)
    pairs.to_csv(output / "pair_summary.csv", index=False)
    groups.to_csv(output / "classifier_group_summary.csv", index=False)
    block_summary.to_csv(output / "block_signal_summary.csv", index=False)
    block_bins.to_csv(output / "block_signal_quartiles.csv", index=False)
    (
        blocks.groupby(["pair", "classifier_group", "status"], sort=True)
        .size()
        .rename("blocks")
        .reset_index()
        .to_csv(output / "block_status_by_configuration.csv", index=False)
    )

    plot_configuration_heatmap(config_features, output)
    plot_statistical_profiles(
        config_features, correlations, validation, block_bins, output
    )
    write_report(
        output,
        config_features,
        pairs,
        groups,
        correlations,
        profiles,
        block_summary,
        block_bins,
    )
    write_chinese_report(
        output,
        config_features,
        pairs,
        groups,
        correlations,
        profiles,
        block_summary,
        block_bins,
    )
    manifest = {
        "period": PERIOD,
        "analysis_type": "retrospective_diagnostic",
        "primary_inferential_unit": "pair_classifier_group_configuration",
        "configurations": 15,
        "classifier_refit_paths": 450,
        "evaluation_blocks": 1800,
        "daily_decision_rows": 112050,
        "permutation_draws": PERMUTATIONS,
        "cluster_bootstrap_draws": BOOTSTRAPS,
        "random_seed": RANDOM_SEED,
        "sources": {
            "crossover_results": str(root),
            "candidate_root": str(candidate_root),
            "mechanism_validation": str(validation_path),
        },
        "excludes_2021": True,
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print((output / "STATISTICAL_REPORT_ZH.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
