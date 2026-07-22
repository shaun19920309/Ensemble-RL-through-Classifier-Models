from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/finrl-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from finrl.reproduction.classifier_ensemble import confidence_matrix
from finrl.reproduction.classifier_ensemble import select_holding_from_confidence
from finrl.reproduction.classifier_ensemble import tau_grid
from finrl.reproduction.classifier_ensemble import train_classifier_group
from finrl.reproduction.disagreement import DISAGREEMENT_METRICS
from finrl.reproduction.disagreement import estimate_shrinkage_covariance
from finrl.reproduction.disagreement import holding_disagreement
from finrl.reproduction.metrics import metrics_from_account_values
from reproduce_classifier_ensemble import append_account_curve
from reproduce_classifier_ensemble import build_env
from reproduce_classifier_ensemble import build_rolling_windows
from reproduce_classifier_ensemble import env_kwargs
from reproduce_classifier_ensemble import frame_for_dates
from reproduce_classifier_ensemble import load_trademaster_rolling_data
from reproduce_classifier_ensemble import shares_from_state
from run_fixed_rl_30_backtests import METRICS
from run_fixed_rl_30_backtests import MODEL_NAMES
from run_fixed_rl_30_backtests import PAIR_COMPONENTS
from run_fixed_rl_30_backtests import PAIR_KEYS
from run_fixed_rl_30_backtests import aggregate_metric_frame
from run_fixed_rl_30_backtests import align_account_curve
from run_fixed_rl_30_backtests import load_fixed_models
from run_fixed_rl_30_backtests import markdown_table
from run_fixed_rl_30_backtests import mean_sd_ci
from run_fixed_rl_30_backtests import rolling_window_summary
from run_fixed_rl_30_backtests import simulate_selected_targets
from run_fixed_rl_30_backtests import stable_seed


METRIC_LABELS = {
    "original": "Original min-max dispersion",
    "l1": "L1 portfolio distance",
    "risk_weighted": "Risk-weighted distance",
}
METRIC_COLORS = {
    "original": "#6B7280",
    "l1": "#2F6F9F",
    "risk_weighted": "#B45309",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the V1 holding-dispersion statistic with L1 and prior-block "
            "risk-weighted portfolio distances under the fixed-RL protocol."
        )
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--fixed-run-dir", required=True)
    parser.add_argument("--v1-result-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--trade-split", choices=["valid", "test"], required=True)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--repeat-start", type=int, default=0)
    parser.add_argument("--repeat-stop", type=int, default=None)
    parser.add_argument(
        "--worker-only",
        action="store_true",
        help="Write only the assigned repeat directories; defer aggregation to a final resume run.",
    )
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--master-seed", type=int, default=250217518)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--rebalance-window", type=int, default=63)
    parser.add_argument("--validation-window", type=int, default=63)
    parser.add_argument("--tau-start", type=float, default=0.01)
    parser.add_argument("--tau-stop", type=float, default=0.89)
    parser.add_argument("--tau-step", type=float, default=0.01)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def collect_target_path(
    model,
    data: pd.DataFrame,
    kwargs: dict[str, object],
    *,
    initial: bool = True,
    previous_state: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[float]]:
    """Collect exact V1 target shares plus V2 cash and execution-price context."""
    environment = build_env(
        data,
        kwargs,
        initial=initial,
        previous_state=[] if previous_state is None else previous_state,
    )
    vector_environment, observation = environment.get_sb_env()
    stock_dim = int(kwargs["stock_dim"])
    holdings: list[np.ndarray] = []
    cash_balances: list[float] = []
    prices: list[np.ndarray] = []
    decision_dates: list[str] = []
    for _step in range(len(data.index.unique()) - 1):
        state_before = np.asarray(environment.render(), dtype=float)
        prices.append(state_before[1 : 1 + stock_dim].copy())
        decision_dates.append(str(environment.date_memory[-1]))
        action, _states = model.predict(observation, deterministic=True)
        observation, _rewards, dones, _info = vector_environment.step(action)
        state_after = np.asarray(environment.render(), dtype=float)
        cash_balances.append(float(state_after[0]))
        holdings.append(shares_from_state(state_after, stock_dim))
        if dones[0]:
            break
    return (
        np.asarray(holdings, dtype=float),
        np.asarray(cash_balances, dtype=float),
        np.asarray(prices, dtype=float),
        np.asarray(decision_dates, dtype=str),
        environment.save_asset_memory(),
        list(environment.render()),
    )


def price_matrix(frame: pd.DataFrame, asset_order: list[object]) -> np.ndarray:
    dates = np.asarray(sorted(frame["date"].astype(str).unique()), dtype=str)
    matrix = (
        frame.pivot(index="date", columns="tic", values="close")
        .reindex(index=dates, columns=asset_order)
        .to_numpy(dtype=float)
    )
    if matrix.shape != (len(dates), len(asset_order)) or not np.isfinite(matrix).all():
        raise ValueError("price matrix is incomplete or misaligned")
    return matrix


def decision_choices(
    classifiers: list[tuple[str, object]], candidates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    aggressive: list[int] = []
    conservative: list[int] = []
    for day_candidates in candidates:
        q_matrix = confidence_matrix(classifiers, day_candidates, [0, 1])
        aggressive_decision = select_holding_from_confidence(
            day_candidates, q_matrix, tau=1.0, dispersion=0.0
        )
        conservative_decision = select_holding_from_confidence(
            day_candidates, q_matrix, tau=0.0, dispersion=1.0
        )
        aggressive.append(aggressive_decision.selected_index)
        conservative.append(conservative_decision.selected_index)
    return np.asarray(aggressive, dtype=np.int8), np.asarray(conservative, dtype=np.int8)


def prepare_fixed_candidates(
    *,
    models: dict[int, dict[str, object]],
    full_data: pd.DataFrame,
    windows: list[dict[str, object]],
    kwargs: dict[str, object],
    expected_dates: np.ndarray,
) -> tuple[list[dict[str, object]], pd.DataFrame, np.ndarray, pd.DataFrame]:
    first_date = str(sorted(full_data["date"].astype(str).unique())[0])
    asset_order = full_data.loc[full_data["date"] == first_date, "tic"].tolist()
    if len(asset_order) != int(kwargs["stock_dim"]):
        raise ValueError("asset order does not match the environment stock dimension")

    base_curves = {name: pd.DataFrame() for name in MODEL_NAMES}
    base_last_states: dict[str, list[float] | None] = {name: None for name in MODEL_NAMES}
    prepared_windows: list[dict[str, object]] = []
    disagreement_rows: list[dict[str, object]] = []

    for window_info in windows:
        window = int(window_info["window"])
        calibration = frame_for_dates(full_data, window_info["calibration_dates"])
        trade = frame_for_dates(full_data, window_info["trade_dates"])
        if str(calibration["date"].max()) >= str(trade["date"].min()):
            raise ValueError("risk covariance calibration overlaps the trade block")
        covariance = estimate_shrinkage_covariance(
            price_matrix(calibration, asset_order)
        )
        calibration_holdings: dict[str, np.ndarray] = {}
        trade_paths: dict[str, dict[str, np.ndarray]] = {}

        for model_name in MODEL_NAMES:
            model = models[window][model_name]
            calibration_path = collect_target_path(model, calibration, kwargs)
            calibration_holdings[model_name] = calibration_path[0]
            trade_path = collect_target_path(
                model,
                trade,
                kwargs,
                initial=base_last_states[model_name] is None,
                previous_state=base_last_states[model_name],
            )
            shares, cash, prices, decision_dates, account, last_state = trade_path
            base_last_states[model_name] = last_state
            base_curves[model_name] = append_account_curve(base_curves[model_name], account)
            trade_paths[model_name] = {
                "shares": shares,
                "cash": cash,
                "prices": prices,
                "decision_dates": decision_dates,
            }

        pair_paths: dict[str, dict[str, object]] = {}
        for pair in PAIR_KEYS:
            left, right = PAIR_COMPONENTS[pair]
            left_path = trade_paths[left]
            right_path = trade_paths[right]
            for field in ("prices", "decision_dates"):
                if not np.array_equal(left_path[field], right_path[field]):
                    raise ValueError(f"candidate {field} are not aligned for {pair}, window {window}")
            candidates = np.stack(
                [left_path["shares"], right_path["shares"]], axis=1
            )
            cash = np.stack([left_path["cash"], right_path["cash"]], axis=1)
            prices = np.asarray(left_path["prices"], dtype=float)
            dates = np.asarray(left_path["decision_dates"], dtype=str)
            disagreements = np.empty(
                (len(DISAGREEMENT_METRICS), len(candidates)), dtype=float
            )
            for metric_index, metric in enumerate(DISAGREEMENT_METRICS):
                for day in range(len(candidates)):
                    disagreements[metric_index, day] = holding_disagreement(
                        metric,
                        candidates[day],
                        prices=prices[day],
                        cash_balances=cash[day],
                        covariance=covariance,
                    )
                    disagreement_rows.append(
                        {
                            "window": window,
                            "date": dates[day],
                            "pair": pair,
                            "disagreement_metric": metric,
                            "disagreement": disagreements[metric_index, day],
                            "covariance_source_start": window_info["calibration_start"],
                            "covariance_source_end": window_info["calibration_end"],
                            "trade_start": window_info["trade_start"],
                        }
                    )
            if (disagreements < -1e-12).any() or (disagreements > 1.0 + 1e-12).any():
                raise ValueError("a disagreement metric left the unit interval")
            pair_paths[pair] = {
                "candidates": candidates,
                "disagreements": disagreements,
            }

        prepared_windows.append(
            {
                "window": window,
                "trade": trade,
                "calibration_holdings": calibration_holdings,
                "pairs": pair_paths,
                "calibration_source": window_info["calibration_source"],
                "calibration_start": window_info["calibration_start"],
                "calibration_end": window_info["calibration_end"],
                "trade_start": window_info["trade_start"],
                "trade_end": window_info["trade_end"],
            }
        )

    base_rows: list[dict[str, object]] = []
    base_curve_array = np.empty((len(MODEL_NAMES), len(expected_dates)), dtype=float)
    for model_index, model_name in enumerate(MODEL_NAMES):
        curve = align_account_curve(base_curves[model_name], expected_dates)
        base_curve_array[model_index] = curve
        base_rows.append({"model": model_name, **metrics_from_account_values(curve)})
    return (
        prepared_windows,
        pd.DataFrame(base_rows).sort_values("model"),
        base_curve_array,
        pd.DataFrame(disagreement_rows),
    )


def simulate_signature(
    pair: str,
    selected_windows: list[np.ndarray],
    prepared_windows: list[dict[str, object]],
    kwargs: dict[str, object],
    expected_dates: np.ndarray,
) -> np.ndarray:
    curve = pd.DataFrame()
    last_state: list[float] | None = None
    for selected, window_data in zip(selected_windows, prepared_windows):
        candidates = np.asarray(window_data["pairs"][pair]["candidates"])
        targets = candidates[np.arange(len(selected)), selected]
        account, last_state = simulate_selected_targets(
            targets,
            window_data["trade"],
            kwargs,
            initial=last_state is None,
            previous_state=last_state,
        )
        curve = append_account_curve(curve, account)
    return align_account_curve(curve, expected_dates)


def run_one_repeat(
    repeat: int,
    *,
    prepared_windows: list[dict[str, object]],
    base_metrics: pd.DataFrame,
    base_curves: np.ndarray,
    kwargs: dict[str, object],
    tau_values: np.ndarray,
    expected_dates: np.ndarray,
    master_seed: int,
    output_dir: Path,
    curve_cache: dict[tuple[str, bytes], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    config_keys = [(pair, group) for pair in PAIR_KEYS for group in range(1, 6)]
    decisions: dict[tuple[str, int], list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        key: [] for key in config_keys
    }
    classifier_rows: list[dict[str, object]] = []

    for window_data in prepared_windows:
        window = int(window_data["window"])
        for pair in PAIR_KEYS:
            left, right = PAIR_COMPONENTS[pair]
            candidates = np.asarray(window_data["pairs"][pair]["candidates"])
            disagreements = np.asarray(window_data["pairs"][pair]["disagreements"])
            for group in range(1, 6):
                classifier_seed = stable_seed(
                    master_seed, "classifier", repeat, window, pair, group
                )
                classifiers = train_classifier_group(
                    [
                        window_data["calibration_holdings"][left],
                        window_data["calibration_holdings"][right],
                    ],
                    group,
                    random_state=classifier_seed,
                    grid_search=False,
                )
                aggressive, conservative = decision_choices(classifiers, candidates)
                decisions[(pair, group)].append(
                    (disagreements, aggressive, conservative)
                )
                classifier_rows.append(
                    {
                        "repeat": repeat,
                        "window": window,
                        "pair": pair,
                        "classifier_group": group,
                        "classifier_seed": classifier_seed,
                        "calibration_source": window_data["calibration_source"],
                        "calibration_start": window_data["calibration_start"],
                        "calibration_end": window_data["calibration_end"],
                        "trade_start": window_data["trade_start"],
                        "trade_end": window_data["trade_end"],
                    }
                )

    curve_array = np.empty(
        (
            len(DISAGREEMENT_METRICS),
            len(config_keys),
            len(tau_values),
            len(expected_dates),
        ),
        dtype=float,
    )
    metric_rows: list[dict[str, object]] = []
    for config_index, (pair, group) in enumerate(config_keys):
        inputs = decisions[(pair, group)]
        signature_assignments: dict[bytes, list[tuple[int, int]]] = {}
        signature_selections: dict[bytes, list[np.ndarray]] = {}
        for metric_index, _metric in enumerate(DISAGREEMENT_METRICS):
            for tau_index, tau in enumerate(tau_values):
                selected_windows: list[np.ndarray] = []
                signature_parts: list[bytes] = []
                for disagreements, aggressive, conservative in inputs:
                    selected = np.where(
                        disagreements[metric_index] < float(tau),
                        aggressive,
                        conservative,
                    ).astype(np.int8)
                    selected_windows.append(selected)
                    signature_parts.append(selected.tobytes())
                signature = b"|".join(signature_parts)
                signature_assignments.setdefault(signature, []).append(
                    (metric_index, tau_index)
                )
                signature_selections.setdefault(signature, selected_windows)

        for signature, assignments in signature_assignments.items():
            cache_key = (pair, signature)
            if cache_key not in curve_cache:
                curve_cache[cache_key] = simulate_signature(
                    pair,
                    signature_selections[signature],
                    prepared_windows,
                    kwargs,
                    expected_dates,
                )
            values = curve_cache[cache_key]
            metric_values = metrics_from_account_values(values)
            for metric_index, tau_index in assignments:
                curve_array[metric_index, config_index, tau_index] = values
                metric_rows.append(
                    {
                        "repeat": repeat,
                        "disagreement_metric": DISAGREEMENT_METRICS[metric_index],
                        "pair": pair,
                        "classifier_group": group,
                        "tau": float(tau_values[tau_index]),
                        "equivalent_metric_tau_paths": len(assignments),
                        **metric_values,
                    }
                )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["disagreement_metric", "pair", "classifier_group", "tau"]
    )
    repeat_base = base_metrics.copy()
    repeat_base.insert(0, "repeat", repeat)
    classifier_audit = pd.DataFrame(classifier_rows)
    run_dir = output_dir / "runs" / f"repeat_{repeat:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(run_dir / "ensemble_metrics.csv", index=False)
    repeat_base.to_csv(run_dir / "base_metrics.csv", index=False)
    classifier_audit.to_csv(run_dir / "classifier_refit_audit.csv", index=False)
    np.savez_compressed(
        run_dir / "account_curves.npz",
        ensemble=curve_array,
        base=base_curves,
        dates=expected_dates,
        disagreement_metrics=np.asarray(DISAGREEMENT_METRICS),
    )
    return metrics, repeat_base, curve_array, base_curves, classifier_audit


def completed_repeat_exists(repeat: int, output_dir: Path) -> bool:
    run_dir = output_dir / "runs" / f"repeat_{repeat:02d}"
    return all(
        (run_dir / name).exists()
        for name in (
            "ensemble_metrics.csv",
            "base_metrics.csv",
            "classifier_refit_audit.csv",
            "account_curves.npz",
        )
    )


def load_completed_repeat(
    repeat: int, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    run_dir = output_dir / "runs" / f"repeat_{repeat:02d}"
    with np.load(run_dir / "account_curves.npz") as arrays:
        ensemble = arrays["ensemble"]
        base = arrays["base"]
        metrics = tuple(arrays["disagreement_metrics"].astype(str))
    if metrics != DISAGREEMENT_METRICS:
        raise ValueError("completed repeat uses a different disagreement metric order")
    return (
        pd.read_csv(run_dir / "ensemble_metrics.csv"),
        pd.read_csv(run_dir / "base_metrics.csv"),
        ensemble,
        base,
        pd.read_csv(run_dir / "classifier_refit_audit.csv"),
    )


def build_selected_summary(
    mean_metrics: pd.DataFrame,
    all_metrics: pd.DataFrame,
    all_base: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    best = (
        mean_metrics.sort_values(
            [
                "disagreement_metric",
                "pair",
                "classifier_group",
                "sharpe_mean",
                "tau",
            ],
            ascending=[True, True, True, False, True],
        )
        .groupby(["disagreement_metric", "pair", "classifier_group"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    base_means = all_base.groupby("model")[METRICS].mean()
    rows: list[dict[str, object]] = []
    paired_frames: list[pd.DataFrame] = []
    for best_row in best.itertuples(index=False):
        selected = all_metrics[
            (all_metrics["disagreement_metric"] == best_row.disagreement_metric)
            & (all_metrics["pair"] == best_row.pair)
            & (all_metrics["classifier_group"] == best_row.classifier_group)
            & np.isclose(all_metrics["tau"], best_row.tau)
        ].copy()
        components = PAIR_COMPONENTS[best_row.pair]
        stronger = max(
            components, key=lambda model: float(base_means.loc[model, "sharpe"])
        )
        component = all_base[all_base["model"] == stronger][
            ["repeat", *METRICS]
        ].rename(columns={metric: f"component_{metric}" for metric in METRICS})
        selected = selected.merge(component, on="repeat", validate="one_to_one")
        selected["stronger_component"] = stronger
        selected["delta_sharpe"] = selected["sharpe"] - selected["component_sharpe"]
        selected["beats_stronger_component"] = selected["delta_sharpe"] > 0
        paired_frames.append(selected)
        delta_mean, delta_sd, delta_low, delta_high = mean_sd_ci(
            selected["delta_sharpe"]
        )
        rows.append(
            {
                "disagreement_metric": best_row.disagreement_metric,
                "pair": best_row.pair,
                "classifier_group": int(best_row.classifier_group),
                "selected_global_tau": float(best_row.tau),
                "ensemble_return_mean": float(best_row.cumulative_return_mean),
                "ensemble_return_sd": float(best_row.cumulative_return_sd),
                "ensemble_sharpe_mean": float(best_row.sharpe_mean),
                "ensemble_sharpe_sd": float(best_row.sharpe_sd),
                "ensemble_calmar_mean": float(best_row.calmar_mean),
                "ensemble_calmar_sd": float(best_row.calmar_sd),
                "ensemble_mdd_mean": float(best_row.max_drawdown_mean),
                "ensemble_mdd_sd": float(best_row.max_drawdown_sd),
                "stronger_component": stronger,
                "component_sharpe_mean": float(base_means.loc[stronger, "sharpe"]),
                "delta_sharpe_mean": delta_mean,
                "delta_sharpe_sd": delta_sd,
                "delta_sharpe_ci_low": delta_low,
                "delta_sharpe_ci_high": delta_high,
                "wins_vs_stronger": int(selected["beats_stronger_component"].sum()),
                "n_backtests": len(selected),
            }
        )
    return pd.DataFrame(rows), pd.concat(paired_frames, ignore_index=True)


def build_threshold_robustness(
    mean_metrics: pd.DataFrame, all_base: pd.DataFrame
) -> pd.DataFrame:
    base_sharpe = all_base.groupby("model")["sharpe"].mean()
    rows: list[dict[str, object]] = []
    for keys, frame in mean_metrics.groupby(
        ["disagreement_metric", "pair", "classifier_group"], sort=True
    ):
        metric, pair, group = keys
        stronger = max(PAIR_COMPONENTS[pair], key=lambda model: base_sharpe.loc[model])
        benchmark = float(base_sharpe.loc[stronger])
        successful = frame[frame["sharpe_mean"] > benchmark]
        rows.append(
            {
                "disagreement_metric": metric,
                "pair": pair,
                "classifier_group": int(group),
                "stronger_component": stronger,
                "stronger_sharpe": benchmark,
                "successful_tau_count": len(successful),
                "tau_count": len(frame),
                "successful_tau_fraction": len(successful) / len(frame),
                "first_successful_tau": (
                    float(successful["tau"].min()) if len(successful) else np.nan
                ),
                "last_successful_tau": (
                    float(successful["tau"].max()) if len(successful) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def build_ablation_summary(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, frame in selected.groupby("disagreement_metric", sort=False):
        rows.append(
            {
                "disagreement_metric": metric,
                "configurations": len(frame),
                "beats_stronger": int((frame["delta_sharpe_mean"] > 0).sum()),
                "positive_95pct_ci": int((frame["delta_sharpe_ci_low"] > 0).sum()),
                "mean_delta_sharpe": float(frame["delta_sharpe_mean"].mean()),
                "median_delta_sharpe": float(frame["delta_sharpe_mean"].median()),
            }
        )
    return pd.DataFrame(rows)


def build_metric_vs_original(
    selected_runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = selected_runs[selected_runs["disagreement_metric"] == "original"]
    original = original[
        ["repeat", "pair", "classifier_group", "tau", "sharpe"]
    ].rename(columns={"tau": "original_tau", "sharpe": "original_sharpe"})
    details: list[pd.DataFrame] = []
    for metric in ("l1", "risk_weighted"):
        candidate = selected_runs[
            selected_runs["disagreement_metric"] == metric
        ].copy()
        candidate = candidate.merge(
            original,
            on=["repeat", "pair", "classifier_group"],
            validate="one_to_one",
        )
        candidate["delta_sharpe_vs_original"] = (
            candidate["sharpe"] - candidate["original_sharpe"]
        )
        details.append(candidate)
    detail = pd.concat(details, ignore_index=True)
    rows: list[dict[str, object]] = []
    for keys, frame in detail.groupby(
        ["disagreement_metric", "pair", "classifier_group"], sort=True
    ):
        mean, sd, low, high = mean_sd_ci(frame["delta_sharpe_vs_original"])
        rows.append(
            {
                "disagreement_metric": keys[0],
                "pair": keys[1],
                "classifier_group": int(keys[2]),
                "delta_sharpe_vs_original_mean": mean,
                "delta_sharpe_vs_original_sd": sd,
                "delta_sharpe_vs_original_ci_low": low,
                "delta_sharpe_vs_original_ci_high": high,
                "wins_vs_original": int(
                    (frame["delta_sharpe_vs_original"] > 0).sum()
                ),
                "n_backtests": len(frame),
            }
        )
    return detail, pd.DataFrame(rows)


def audit_v1_reproduction(
    metrics: pd.DataFrame,
    base: pd.DataFrame,
    v1_result_dir: Path,
) -> dict[str, object]:
    v1_metrics = pd.read_csv(v1_result_dir / "all_backtest_metrics.csv")
    v2_original = metrics[metrics["disagreement_metric"] == "original"].drop(
        columns=["disagreement_metric", "equivalent_metric_tau_paths"]
    )
    keys = ["repeat", "pair", "classifier_group", "tau"]
    merged = v2_original.merge(
        v1_metrics,
        on=keys,
        suffixes=("_v2", "_v1"),
        validate="one_to_one",
    )
    v1_base = pd.read_csv(v1_result_dir / "all_base_metrics.csv")
    merged_base = base.merge(
        v1_base,
        on=["repeat", "model"],
        suffixes=("_v2", "_v1"),
        validate="one_to_one",
    )
    differences = {
        metric: float(
            np.max(np.abs(merged[f"{metric}_v2"] - merged[f"{metric}_v1"]))
        )
        for metric in METRICS
    }
    base_differences = {
        metric: float(
            np.max(
                np.abs(
                    merged_base[f"{metric}_v2"] - merged_base[f"{metric}_v1"]
                )
            )
        )
        for metric in METRICS
    }
    passed = (
        len(merged) == len(v2_original)
        and len(merged_base) == len(base)
        and max([*differences.values(), *base_differences.values()]) <= 1e-10
    )
    return {
        "passed": passed,
        "v1_result_dir": str(v1_result_dir.resolve()),
        "v1_metric_rows": len(v1_metrics),
        "matched_metric_rows": len(merged),
        "v2_original_metric_rows": len(v2_original),
        "v1_coverage_fraction": len(merged) / len(v1_metrics),
        "maximum_absolute_metric_differences": differences,
        "maximum_absolute_base_differences": base_differences,
    }


def plot_ablation(
    selected: pd.DataFrame,
    output_dir: Path,
    dataset_label: str,
    dpi: int,
) -> list[Path]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), sharey=True)
    groups = np.arange(1, 6, dtype=float)
    width = 0.24
    offsets = (-width, 0.0, width)
    for axis, pair in zip(axes, PAIR_KEYS):
        for offset, metric in zip(offsets, DISAGREEMENT_METRICS):
            frame = selected[
                (selected["pair"] == pair)
                & (selected["disagreement_metric"] == metric)
            ].sort_values("classifier_group")
            means = frame["delta_sharpe_mean"].to_numpy(dtype=float)
            lower = means - frame["delta_sharpe_ci_low"].to_numpy(dtype=float)
            upper = frame["delta_sharpe_ci_high"].to_numpy(dtype=float) - means
            axis.bar(
                groups + offset,
                means,
                width=width * 0.92,
                color=METRIC_COLORS[metric],
                label=METRIC_LABELS[metric],
                zorder=3,
            )
            axis.errorbar(
                groups + offset,
                means,
                yerr=np.vstack([lower, upper]),
                fmt="none",
                ecolor="#202124",
                capsize=2,
                linewidth=0.7,
                zorder=4,
            )
        axis.axhline(0.0, color="#202124", linewidth=0.8)
        axis.grid(axis="y", color="#D9DDE2", linewidth=0.5, zorder=1)
        axis.set_title(pair.replace("_", " + ").upper())
        axis.set_xticks(groups, [f"G{i}" for i in range(1, 6)])
        axis.set_xlabel("Classifier group")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Sharpe difference vs. stronger component")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle(f"{dataset_label}: disagreement-metric ablation", y=0.98)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.82))
    stem = figures_dir / "figure_v2_disagreement_metric_ablation"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    return [stem.with_suffix(".pdf"), stem.with_suffix(".png")]


def write_report(
    output_dir: Path,
    *,
    dataset_label: str,
    summary: pd.DataFrame,
    metric_vs_original: pd.DataFrame,
    v1_audit: dict[str, object],
) -> None:
    best_metric = summary.sort_values(
        ["beats_stronger", "positive_95pct_ci", "mean_delta_sharpe"],
        ascending=False,
    ).iloc[0]
    comparison_counts = (
        metric_vs_original.assign(
            positive=lambda frame: frame["delta_sharpe_vs_original_mean"] > 0,
            positive_ci=lambda frame: frame["delta_sharpe_vs_original_ci_low"] > 0,
        )
        .groupby("disagreement_metric")[["positive", "positive_ci"]]
        .sum()
    )
    lines = [
        f"# V2 Disagreement-Metric Ablation: {dataset_label}",
        "",
        "## Protocol",
        "",
        "- Candidate RL checkpoints, deterministic holdings, rolling classifier seeds, classifier groups, transaction costs, and the 89 fixed global thresholds are shared across metrics.",
        "- `original` is the unchanged V1 min-max-normalized cross-agent holding standard deviation.",
        "- `l1` is total-variation distance between market-value portfolio weights, including cash.",
        "- `risk_weighted` is covariance-norm distance normalized by the two candidate risk norms.",
        "- Risk covariance uses Ledoit-Wolf shrinkage estimated only from the immediately preceding calibration block and is frozen in the next trade block.",
        "- Each candidate threshold remains fixed across the complete evaluation path.",
        "",
        "## V1 Reproduction Guard",
        "",
        f"- Passed: `{v1_audit['passed']}`.",
        f"- Matched V1 run-level rows: {v1_audit['matched_metric_rows']}/{v1_audit['v1_metric_rows']}.",
        f"- Maximum metric deviation: {max(v1_audit['maximum_absolute_metric_differences'].values()):.3e}.",
        "",
        "## Selected-Threshold Results",
        "",
        markdown_table(summary, list(summary.columns)),
        "",
        f"The strongest descriptive result is `{best_metric['disagreement_metric']}` with {int(best_metric['beats_stronger'])}/15 configurations above the stronger component and {int(best_metric['positive_95pct_ci'])}/15 positive conditional intervals.",
        "",
        "## Direct Comparison with V1 Original Metric",
        "",
        markdown_table(
            comparison_counts.reset_index(),
            list(comparison_counts.reset_index().columns),
        ),
        "",
        "All intervals remain conditional on fixed RL paths and post-hoc selection among 89 thresholds; this ablation isolates the gate statistic but is not an untouched-holdout estimate.",
    ]
    (output_dir / "V2_DISAGREEMENT_ABLATION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repeat_stop = args.repetitions if args.repeat_stop is None else args.repeat_stop
    if not (0 <= args.repeat_start < repeat_stop <= args.repetitions):
        raise ValueError("repeat range must satisfy 0 <= start < stop <= repetitions")
    full_data, indicators, trade_start, data_metadata = load_trademaster_rolling_data(
        args.data_dir, trade_split=args.trade_split
    )
    kwargs = env_kwargs(full_data, indicators)
    compatibility_environment = build_env(full_data, kwargs)
    windows = build_rolling_windows(
        full_data,
        trade_start_date=trade_start,
        rebalance_window=args.rebalance_window,
        validation_window=args.validation_window,
        max_windows=None,
    )
    if len(windows) != 4:
        raise ValueError(f"the V2 ablation requires four rolling windows, found {len(windows)}")
    expected_dates = np.concatenate(
        [np.asarray(window["trade_dates"], dtype=str) for window in windows]
    )
    if len(expected_dates) != len(np.unique(expected_dates)):
        raise ValueError("trade windows overlap")
    tau_values = tau_grid(args.tau_start, args.tau_stop, args.tau_step)
    expected_tau = np.round(np.arange(0.01, 0.90, 0.01), 2)
    if not np.array_equal(tau_values, expected_tau):
        raise ValueError("the ablation requires the V1 tau grid 0.01-0.89")

    windows_frame = rolling_window_summary(windows)
    if not args.worker_only:
        windows_frame.to_csv(output_dir / "rolling_windows.csv", index=False)
    models, checkpoint_manifest = load_fixed_models(
        Path(args.fixed_run_dir),
        windows,
        timesteps=args.timesteps,
        model_seed=args.model_seed,
        observation_space=compatibility_environment.observation_space,
        action_space=compatibility_environment.action_space,
    )
    if not args.worker_only:
        checkpoint_manifest.to_csv(
            output_dir / "fixed_rl_checkpoint_manifest.csv", index=False
        )
    prepared, fixed_base_metrics, fixed_base_curves, disagreement_detail = (
        prepare_fixed_candidates(
            models=models,
            full_data=full_data,
            windows=windows,
            kwargs=kwargs,
            expected_dates=expected_dates,
        )
    )
    if not args.worker_only:
        disagreement_detail.to_csv(
            output_dir / "disagreement_daily_detail.csv", index=False
        )
    disagreement_distribution = (
        disagreement_detail.groupby(["pair", "disagreement_metric"])["disagreement"]
        .agg(
            observations="size",
            minimum="min",
            q05=lambda values: values.quantile(0.05),
            q25=lambda values: values.quantile(0.25),
            median="median",
            mean="mean",
            q75=lambda values: values.quantile(0.75),
            q95=lambda values: values.quantile(0.95),
            maximum="max",
        )
        .reset_index()
    )
    if not args.worker_only:
        disagreement_distribution.to_csv(
            output_dir / "disagreement_distribution.csv", index=False
        )

    all_metrics: list[pd.DataFrame] = []
    all_base: list[pd.DataFrame] = []
    all_ensemble_curves: list[np.ndarray] = []
    all_base_curves: list[np.ndarray] = []
    classifier_audits: list[pd.DataFrame] = []
    curve_cache: dict[tuple[str, bytes], np.ndarray] = {}
    for repeat in range(args.repeat_start, repeat_stop):
        if args.resume and completed_repeat_exists(repeat, output_dir):
            result = load_completed_repeat(repeat, output_dir)
            print(f"REPEAT {repeat + 1}/{args.repetitions}: loaded")
        else:
            repeat_started = time.time()
            result = run_one_repeat(
                repeat,
                prepared_windows=prepared,
                base_metrics=fixed_base_metrics,
                base_curves=fixed_base_curves,
                kwargs=kwargs,
                tau_values=tau_values,
                expected_dates=expected_dates,
                master_seed=args.master_seed,
                output_dir=output_dir,
                curve_cache=curve_cache,
            )
            print(
                f"REPEAT {repeat + 1}/{args.repetitions}: completed in "
                f"{time.time() - repeat_started:.1f}s; cached paths={len(curve_cache)}"
            )
        metrics, base, ensemble_curves, base_curves, classifier_audit = result
        all_metrics.append(metrics)
        all_base.append(base)
        all_ensemble_curves.append(ensemble_curves)
        all_base_curves.append(base_curves)
        classifier_audits.append(classifier_audit)

    if args.worker_only:
        print(
            f"WORKER COMPLETE: repeats {args.repeat_start}-{repeat_stop - 1}; "
            f"elapsed={time.time() - started:.1f}s"
        )
        return

    metrics_frame = pd.concat(all_metrics, ignore_index=True)
    base_frame = pd.concat(all_base, ignore_index=True)
    classifier_audit_frame = pd.concat(classifier_audits, ignore_index=True)
    expected_rows = (
        args.repetitions
        * len(DISAGREEMENT_METRICS)
        * len(PAIR_KEYS)
        * 5
        * len(tau_values)
    )
    expected_refits = args.repetitions * len(windows) * len(PAIR_KEYS) * 5
    if len(metrics_frame) != expected_rows:
        raise ValueError(f"expected {expected_rows} metric rows, found {len(metrics_frame)}")
    if len(classifier_audit_frame) != expected_refits:
        raise ValueError(
            f"expected {expected_refits} classifier refits, found {len(classifier_audit_frame)}"
        )
    metrics_frame.to_csv(output_dir / "all_backtest_metrics.csv", index=False)
    base_frame.to_csv(output_dir / "all_base_metrics.csv", index=False)
    classifier_audit_frame.to_csv(
        output_dir / "classifier_refit_audit.csv", index=False
    )

    mean_metrics = aggregate_metric_frame(
        metrics_frame,
        ["disagreement_metric", "pair", "classifier_group", "tau"],
        METRICS,
    )
    mean_metrics.to_csv(output_dir / "mean_metrics_by_fixed_tau.csv", index=False)
    base_summary = aggregate_metric_frame(base_frame, ["model"], METRICS)
    base_summary.to_csv(output_dir / "base_model_summary.csv", index=False)
    selected, selected_runs = build_selected_summary(
        mean_metrics, metrics_frame, base_frame
    )
    selected.to_csv(output_dir / "selected_tau_summary.csv", index=False)
    selected_runs.to_csv(output_dir / "selected_tau_paired_runs.csv", index=False)
    robustness = build_threshold_robustness(mean_metrics, base_frame)
    robustness.to_csv(output_dir / "threshold_robustness.csv", index=False)
    ablation_summary = build_ablation_summary(selected)
    ablation_summary.to_csv(output_dir / "ablation_summary.csv", index=False)
    metric_detail, metric_summary = build_metric_vs_original(selected_runs)
    metric_detail.to_csv(output_dir / "metric_vs_original_paired_runs.csv", index=False)
    metric_summary.to_csv(output_dir / "metric_vs_original_summary.csv", index=False)

    v1_audit = audit_v1_reproduction(
        metrics_frame, base_frame, Path(args.v1_result_dir)
    )
    (output_dir / "v1_reproduction_audit.json").write_text(
        json.dumps(v1_audit, indent=2), encoding="utf-8"
    )
    if not v1_audit["passed"]:
        raise ValueError("V2 original metric failed to reproduce the frozen V1 result")

    figure_paths = plot_ablation(
        selected, output_dir, args.dataset_label, args.dpi
    )
    write_report(
        output_dir,
        dataset_label=args.dataset_label,
        summary=ablation_summary,
        metric_vs_original=metric_summary,
        v1_audit=v1_audit,
    )
    metadata = {
        **data_metadata,
        "dataset_label": args.dataset_label,
        "version": "v2",
        "experiment": "holding_disagreement_metric_ablation",
        "invocation": f"PYTHONPATH=. {shlex.join([sys.executable, *sys.argv])}",
        "disagreement_metrics": list(DISAGREEMENT_METRICS),
        "l1_definition": "total variation of market-value risky-asset plus cash weights",
        "risk_weighted_definition": "covariance norm of risky-weight difference divided by the sum of candidate covariance norms",
        "covariance_estimator": "LedoitWolf",
        "covariance_information_set": "immediately preceding calibration block only",
        "covariance_frozen_within_trade_block": True,
        "repetitions": args.repetitions,
        "model_training_seed": args.model_seed,
        "master_backtest_seed": args.master_seed,
        "rl_retrained_in_repetitions": False,
        "deterministic_rl_inference": True,
        "classifier_refits": len(classifier_audit_frame),
        "classifier_grid_search": False,
        "classifier_training_information_set": "immediately preceding calibration block only",
        "classifier_decisions_shared_across_disagreement_metrics": True,
        "tau_values": tau_values.tolist(),
        "fixed_global_tau_per_complete_path": True,
        "tau_selection_rule": "maximum mean Sharpe across 30 classifier repeats",
        "tau_tie_break": "lowest tau",
        "v1_result_dir": str(Path(args.v1_result_dir).resolve()),
        "v1_reproduction_passed": v1_audit["passed"],
        "elapsed_seconds": time.time() - started,
        "figures": [str(path) for path in figure_paths],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(ablation_summary.to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()
