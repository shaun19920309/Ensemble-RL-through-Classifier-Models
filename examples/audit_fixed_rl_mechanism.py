from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/finrl-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from finrl.reproduction.classifier_ensemble import confidence_matrix
from finrl.reproduction.classifier_ensemble import holding_dispersion
from finrl.reproduction.classifier_ensemble import select_holding_from_confidence
from finrl.reproduction.classifier_ensemble import train_classifier_group
from finrl.reproduction.metrics import metrics_from_account_values
from reproduce_classifier_ensemble import append_account_curve
from reproduce_classifier_ensemble import build_env
from reproduce_classifier_ensemble import build_rolling_windows
from reproduce_classifier_ensemble import collect_holdings_and_account
from reproduce_classifier_ensemble import env_kwargs
from reproduce_classifier_ensemble import frame_for_dates
from reproduce_classifier_ensemble import load_trademaster_rolling_data
from reproduce_classifier_ensemble import shares_from_state
from run_fixed_rl_30_backtests import MODEL_NAMES
from run_fixed_rl_30_backtests import PAIR_COMPONENTS
from run_fixed_rl_30_backtests import PAIR_KEYS
from run_fixed_rl_30_backtests import load_fixed_models
from run_fixed_rl_30_backtests import stable_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit policy sampling and classifier-boundary alignment."
    )
    parser.add_argument("--data-dir", default="external_data/trademaster_dj30")
    parser.add_argument(
        "--fixed-run-dir",
        default="work/core_dj30_candidates",
    )
    parser.add_argument(
        "--output-dir", default="work/fixed_rl_mechanism_audit_dj30"
    )
    parser.add_argument("--policy-repetitions", type=int, default=30)
    parser.add_argument("--classifier-repetitions", type=int, default=5)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--master-seed", type=int, default=250217518)
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--rebalance-window", type=int, default=63)
    parser.add_argument("--validation-window", type=int, default=63)
    parser.add_argument(
        "--selected-tau-file",
        default="results/main_dj30/selected_tau_summary.csv",
    )
    return parser.parse_args()


def own_label_confidence(estimator: object, x: np.ndarray, y: np.ndarray) -> float:
    probabilities = estimator.predict_proba(x)
    classes = np.asarray(getattr(estimator, "classes_", []))
    if classes.size == 0 and hasattr(estimator, "best_estimator_"):
        classes = np.asarray(estimator.best_estimator_.classes_)
    class_columns = {int(label): index for index, label in enumerate(classes)}
    values = [
        probabilities[row, class_columns[int(label)]]
        for row, label in enumerate(y)
        if int(label) in class_columns
    ]
    return float(np.mean(values)) if values else float("nan")


def policy_mode_audit(
    models: dict[int, dict[str, object]],
    full_data: pd.DataFrame,
    windows: list[dict[str, object]],
    kwargs: dict[str, object],
    *,
    repetitions: int,
    master_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for window_info in windows:
        window = int(window_info["window"])
        calibration = frame_for_dates(full_data, window_info["calibration_dates"])
        for model_name in MODEL_NAMES:
            model = models[window][model_name]
            _holdings, deterministic_account, _state = collect_holdings_and_account(
                model, calibration, kwargs, deterministic=True
            )
            deterministic_metrics = metrics_from_account_values(
                deterministic_account["account_value"]
            )
            rows.append(
                {
                    "window": window,
                    "model": model_name,
                    "mode": "deterministic",
                    "repeat": -1,
                    **deterministic_metrics,
                }
            )
            for repeat in range(repetitions):
                prediction_seed = stable_seed(
                    master_seed,
                    "checkpoint_validation",
                    repeat,
                    window,
                    model_name,
                )
                _holdings, account, _state = collect_holdings_and_account(
                    model,
                    calibration,
                    kwargs,
                    deterministic=False,
                    prediction_seed=prediction_seed,
                )
                rows.append(
                    {
                        "window": window,
                        "model": model_name,
                        "mode": "stochastic",
                        "repeat": repeat,
                        **metrics_from_account_values(account["account_value"]),
                    }
                )
    detail = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for (window, model_name), subset in detail.groupby(["window", "model"]):
        deterministic = subset[subset["mode"] == "deterministic"].iloc[0]
        stochastic = subset[subset["mode"] == "stochastic"]
        summary_rows.append(
            {
                "window": int(window),
                "model": model_name,
                "deterministic_return": float(deterministic["cumulative_return"]),
                "stochastic_return_mean": float(stochastic["cumulative_return"].mean()),
                "stochastic_return_sd": float(stochastic["cumulative_return"].std(ddof=1)),
                "deterministic_sharpe": float(deterministic["sharpe"]),
                "stochastic_sharpe_mean": float(stochastic["sharpe"].mean()),
                "stochastic_sharpe_sd": float(stochastic["sharpe"].std(ddof=1)),
                "sharpe_mode_gap": float(
                    deterministic["sharpe"] - stochastic["sharpe"].mean()
                ),
                "stochastic_sharpe_q05": float(stochastic["sharpe"].quantile(0.05)),
                "stochastic_sharpe_q95": float(stochastic["sharpe"].quantile(0.95)),
                "deterministic_mdd": float(deterministic["max_drawdown"]),
                "stochastic_mdd_mean": float(stochastic["max_drawdown"].mean()),
            }
        )
    return detail, pd.DataFrame(summary_rows).sort_values(["window", "model"])


def make_xy(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.vstack([left, right])
    y = np.concatenate(
        [np.zeros(len(left), dtype=int), np.ones(len(right), dtype=int)]
    )
    return x, y


def simulate_targets_with_audit(
    selected_holdings: np.ndarray,
    selected_agents: np.ndarray,
    selected_modes: list[str],
    trade_data: pd.DataFrame,
    kwargs: dict[str, object],
    *,
    initial: bool,
    previous_state: list[float] | None,
) -> tuple[list[float], list[dict[str, object]]]:
    environment = build_env(
        trade_data,
        kwargs,
        initial=initial,
        previous_state=[] if previous_state is None else previous_state,
    )
    vec_env, _obs = environment.get_sb_env()
    rows: list[dict[str, object]] = []
    hmax = float(kwargs["hmax"])
    stock_dim = int(kwargs["stock_dim"])
    for day, target in enumerate(selected_holdings):
        state_before = np.asarray(environment.render(), dtype=float)
        current = shares_from_state(state_before, stock_dim)
        raw_action = (target - current) / hmax
        action = np.clip(raw_action, -1.0, 1.0)
        _obs, _rewards, dones, _info = vec_env.step(np.asarray([action]))
        actual = shares_from_state(environment.render(), stock_dim)
        absolute_gap = float(np.abs(actual - target).sum())
        target_scale = float(np.abs(target).sum())
        rows.append(
            {
                "day": day,
                "date": str(environment.date_memory[-1]),
                "selected_agent": int(selected_agents[day]),
                "mode": selected_modes[day],
                "action_clipped": bool(np.any(np.abs(raw_action) > 1.0 + 1e-12)),
                "clipped_stock_fraction": float((np.abs(raw_action) > 1.0 + 1e-12).mean()),
                "target_exact": bool(np.allclose(actual, target, atol=1e-8)),
                "absolute_share_gap": absolute_gap,
                "relative_share_gap": absolute_gap / max(target_scale, 1.0),
            }
        )
        if dones[0]:
            break
    return list(environment.render()), rows


def classifier_boundary_audit(
    models: dict[int, dict[str, object]],
    full_data: pd.DataFrame,
    windows: list[dict[str, object]],
    kwargs: dict[str, object],
    *,
    repetitions: int,
    master_seed: int,
    selected_taus: dict[tuple[str, int], float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    classifier_rows: list[dict[str, object]] = []
    vote_rows: list[dict[str, object]] = []
    execution_rows: list[dict[str, object]] = []
    for repeat in range(repetitions):
        base_last_states: dict[str, list[float] | None] = {
            name: None for name in MODEL_NAMES
        }
        ensemble_last_states: dict[tuple[str, int], list[float] | None] = {
            (pair, group): None for pair in PAIR_KEYS for group in range(1, 6)
        }
        for window_info in windows:
            window = int(window_info["window"])
            calibration = frame_for_dates(full_data, window_info["calibration_dates"])
            trade = frame_for_dates(full_data, window_info["trade_dates"])
            calibration_holdings: dict[str, np.ndarray] = {}
            trade_holdings: dict[str, np.ndarray] = {}
            for model_name in MODEL_NAMES:
                model = models[window][model_name]
                calibration_seed = stable_seed(
                    master_seed, "calibration", repeat, window, model_name
                )
                trade_seed = stable_seed(
                    master_seed, "trade", repeat, window, model_name
                )
                calibration_holdings[model_name], _account, _state = (
                    collect_holdings_and_account(
                        model,
                        calibration,
                        kwargs,
                        deterministic=False,
                        prediction_seed=calibration_seed,
                    )
                )
                trade_holdings[model_name], _account, last_state = (
                    collect_holdings_and_account(
                        model,
                        trade,
                        kwargs,
                        initial=base_last_states[model_name] is None,
                        previous_state=base_last_states[model_name],
                        deterministic=False,
                        prediction_seed=trade_seed,
                    )
                )
                base_last_states[model_name] = last_state

            for pair in PAIR_KEYS:
                left, right = PAIR_COMPONENTS[pair]
                train_x, train_y = make_xy(
                    calibration_holdings[left], calibration_holdings[right]
                )
                trade_x, trade_y = make_xy(
                    trade_holdings[left], trade_holdings[right]
                )
                candidates = np.stack(
                    [trade_holdings[left], trade_holdings[right]], axis=1
                )
                for group in range(1, 6):
                    classifier_seed = stable_seed(
                        master_seed, "classifier", repeat, window, pair, group
                    )
                    classifiers = train_classifier_group(
                        [calibration_holdings[left], calibration_holdings[right]],
                        group,
                        random_state=classifier_seed,
                        grid_search=False,
                    )
                    selected_agents: list[int] = []
                    selected_modes: list[str] = []
                    selected_targets: list[np.ndarray] = []
                    tau = selected_taus[(pair, group)]
                    for classifier_name, estimator in classifiers:
                        classifier_rows.append(
                            {
                                "repeat": repeat,
                                "window": window,
                                "pair": pair,
                                "classifier_group": group,
                                "classifier": classifier_name,
                                "train_accuracy": float(estimator.score(train_x, train_y)),
                                "trade_accuracy": float(estimator.score(trade_x, trade_y)),
                                "train_own_confidence": own_label_confidence(
                                    estimator, train_x, train_y
                                ),
                                "trade_own_confidence": own_label_confidence(
                                    estimator, trade_x, trade_y
                                ),
                            }
                        )

                    for day, day_candidates in enumerate(candidates):
                        q = confidence_matrix(classifiers, day_candidates, [0, 1])
                        dispersion = holding_dispersion(day_candidates)
                        aggressive_votes = np.bincount(
                            np.argmax(q, axis=1), minlength=2
                        )
                        conservative_votes = np.bincount(
                            np.argmin(q, axis=1), minlength=2
                        )
                        aggressive = select_holding_from_confidence(
                            day_candidates, q, tau=1.0, dispersion=0.0
                        )
                        conservative = select_holding_from_confidence(
                            day_candidates, q, tau=0.0, dispersion=1.0
                        )
                        selected = select_holding_from_confidence(
                            day_candidates, q, tau=tau, dispersion=dispersion
                        )
                        selected_agents.append(selected.selected_index)
                        selected_modes.append(selected.mode)
                        selected_targets.append(selected.selected_holding)
                        vote_rows.append(
                            {
                                "repeat": repeat,
                                "window": window,
                                "day": day,
                                "pair": pair,
                                "classifier_group": group,
                                "dispersion": dispersion,
                                "mean_abs_q_gap": float(np.abs(q[:, 0] - q[:, 1]).mean()),
                                "aggressive_selected_agent": aggressive.selected_index,
                                "conservative_selected_agent": conservative.selected_index,
                                "aggressive_tie": bool(
                                    aggressive_votes[0] == aggressive_votes[1]
                                ),
                                "conservative_tie": bool(
                                    conservative_votes[0] == conservative_votes[1]
                                ),
                            }
                        )
                    key = (pair, group)
                    last_state, window_execution_rows = simulate_targets_with_audit(
                        np.asarray(selected_targets, dtype=float),
                        np.asarray(selected_agents, dtype=int),
                        selected_modes,
                        trade,
                        kwargs,
                        initial=ensemble_last_states[key] is None,
                        previous_state=ensemble_last_states[key],
                    )
                    ensemble_last_states[key] = last_state
                    for row in window_execution_rows:
                        execution_rows.append(
                            {
                                "repeat": repeat,
                                "window": window,
                                "pair": pair,
                                "classifier_group": group,
                                "tau": tau,
                                **row,
                            }
                        )
    return (
        pd.DataFrame(classifier_rows),
        pd.DataFrame(vote_rows),
        pd.DataFrame(execution_rows),
    )


def summarize_classifier_audit(
    classifier_detail: pd.DataFrame, vote_detail: pd.DataFrame
) -> pd.DataFrame:
    classifier_summary = (
        classifier_detail.groupby(["pair", "classifier_group"], as_index=False)
        .agg(
            train_accuracy=("train_accuracy", "mean"),
            trade_accuracy=("trade_accuracy", "mean"),
            train_own_confidence=("train_own_confidence", "mean"),
            trade_own_confidence=("trade_own_confidence", "mean"),
        )
    )
    vote_summary = (
        vote_detail.groupby(["pair", "classifier_group"], as_index=False)
        .agg(
            dispersion_mean=("dispersion", "mean"),
            dispersion_sd=("dispersion", "std"),
            mean_abs_q_gap=("mean_abs_q_gap", "mean"),
            aggressive_left_rate=(
                "aggressive_selected_agent", lambda values: float((values == 0).mean())
            ),
            conservative_left_rate=(
                "conservative_selected_agent", lambda values: float((values == 0).mean())
            ),
            aggressive_tie_rate=("aggressive_tie", "mean"),
            conservative_tie_rate=("conservative_tie", "mean"),
        )
    )
    return classifier_summary.merge(
        vote_summary, on=["pair", "classifier_group"], validate="one_to_one"
    )


def summarize_execution_audit(execution_detail: pd.DataFrame) -> pd.DataFrame:
    ordered = execution_detail.sort_values(
        ["repeat", "pair", "classifier_group", "window", "day"]
    ).copy()
    ordered["switched_agent"] = (
        ordered.groupby(["repeat", "pair", "classifier_group"])["selected_agent"]
        .diff()
        .fillna(0)
        .ne(0)
    )
    return (
        ordered.groupby(["pair", "classifier_group", "tau"], as_index=False)
        .agg(
            target_exact_rate=("target_exact", "mean"),
            action_clipped_rate=("action_clipped", "mean"),
            clipped_stock_fraction=("clipped_stock_fraction", "mean"),
            relative_share_gap=("relative_share_gap", "mean"),
            selected_left_rate=(
                "selected_agent", lambda values: float((values == 0).mean())
            ),
            aggressive_mode_rate=("mode", lambda values: float((values == "aggressive").mean())),
            switch_rate=("switched_agent", "mean"),
        )
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_data, indicators, trade_start, _metadata = load_trademaster_rolling_data(
        args.data_dir, trade_split="valid"
    )
    kwargs = env_kwargs(full_data, indicators)
    compatibility_env = build_env(full_data, kwargs)
    windows = build_rolling_windows(
        full_data,
        trade_start_date=trade_start,
        rebalance_window=args.rebalance_window,
        validation_window=args.validation_window,
        max_windows=None,
    )
    models, manifest = load_fixed_models(
        Path(args.fixed_run_dir),
        windows,
        timesteps=args.timesteps,
        model_seed=args.model_seed,
        observation_space=compatibility_env.observation_space,
        action_space=compatibility_env.action_space,
    )
    manifest.to_csv(output_dir / "fixed_rl_checkpoint_manifest.csv", index=False)

    selected_tau_frame = pd.read_csv(args.selected_tau_file)
    selected_taus = {
        (str(row.pair), int(row.classifier_group)): float(row.selected_global_tau)
        for row in selected_tau_frame.itertuples(index=False)
    }
    expected_tau_keys = {
        (pair, group) for pair in PAIR_KEYS for group in range(1, 6)
    }
    if set(selected_taus) != expected_tau_keys:
        raise ValueError("selected tau file does not contain all 15 configurations")

    policy_detail, policy_summary = policy_mode_audit(
        models,
        full_data,
        windows,
        kwargs,
        repetitions=args.policy_repetitions,
        master_seed=args.master_seed,
    )
    policy_detail.to_csv(output_dir / "checkpoint_policy_mode_detail.csv", index=False)
    policy_summary.to_csv(output_dir / "checkpoint_policy_mode_summary.csv", index=False)

    classifier_detail, vote_detail, execution_detail = classifier_boundary_audit(
        models,
        full_data,
        windows,
        kwargs,
        repetitions=args.classifier_repetitions,
        master_seed=args.master_seed,
        selected_taus=selected_taus,
    )
    classifier_detail.to_csv(output_dir / "classifier_boundary_detail.csv", index=False)
    vote_detail.to_csv(output_dir / "classifier_vote_detail.csv", index=False)
    classifier_summary = summarize_classifier_audit(classifier_detail, vote_detail)
    classifier_summary.to_csv(output_dir / "classifier_boundary_summary.csv", index=False)
    execution_detail.to_csv(output_dir / "target_execution_detail.csv", index=False)
    execution_summary = summarize_execution_audit(execution_detail)
    execution_summary.to_csv(output_dir / "target_execution_summary.csv", index=False)

    print("Checkpoint policy-mode audit")
    print(policy_summary.to_string(index=False))
    print("\nClassifier-boundary audit")
    print(classifier_summary.to_string(index=False))
    print("\nTarget-execution audit")
    print(execution_summary.to_string(index=False))
    print(f"\nSaved mechanism audit to {output_dir}")


if __name__ == "__main__":
    main()
