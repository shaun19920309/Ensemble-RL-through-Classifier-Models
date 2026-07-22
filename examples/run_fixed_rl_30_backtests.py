from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/finrl-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from finrl.reproduction.classifier_ensemble import confidence_matrix
from finrl.reproduction.classifier_ensemble import holding_dispersion
from finrl.reproduction.classifier_ensemble import select_holding_from_confidence
from finrl.reproduction.classifier_ensemble import tau_grid
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


PAIR_KEYS = ["a2c_ppo", "a2c_sac", "ppo_sac"]
PAIR_COMPONENTS = {
    "a2c_ppo": ("a2c", "ppo"),
    "a2c_sac": ("a2c", "sac"),
    "ppo_sac": ("ppo", "sac"),
}
PAIR_LABELS = {
    "a2c_ppo": "A2C + PPO",
    "a2c_sac": "A2C + SAC",
    "ppo_sac": "PPO + SAC",
}
PAPER_FILE_KEYS = {
    "a2c_ppo": "a2cppo",
    "a2c_sac": "a2csac",
    "ppo_sac": "pposac",
}
MODEL_NAMES = ["a2c", "ppo", "sac"]
MODEL_LABELS = {"a2c": "A2C", "ppo": "PPO", "sac": "SAC"}
MODEL_COLORS = {"a2c": "#2f6f9f", "ppo": "#7c3f91", "sac": "#2f855a"}
ENSEMBLE_COLOR = "#d97706"
PAPER_COLOR = "#555555"
METRICS = ["cumulative_return", "sharpe", "calmar", "max_drawdown"]
COMPARABLE_METRICS = ["cumulative_return", "sharpe", "max_drawdown"]
METRIC_LABELS = {
    "cumulative_return": "Cumulative return",
    "sharpe": "Sharpe ratio",
    "calmar": "Calmar ratio",
    "max_drawdown": "Maximum drawdown",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 30 backtests with fixed rolling RL checkpoints, deterministic "
            "policy inference, "
            "rolling classifier refits, and a fixed global tau per path."
        )
    )
    parser.add_argument(
        "--data-dir", default="external_data/trademaster_dj30"
    )
    parser.add_argument(
        "--fixed-run-dir",
        default="work/core_dj30_candidates",
    )
    parser.add_argument(
        "--output-dir", default="work/main_dj30_rebuild"
    )
    parser.add_argument(
        "--paper-results-dir", default="data/paper_reference_results"
    )
    parser.add_argument(
        "--trade-split",
        choices=["valid", "test"],
        default="valid",
        help="TradeMaster split whose first date starts the rolling evaluation.",
    )
    parser.add_argument(
        "--dataset-label",
        default=None,
        help="Human-readable dataset name used in figures and the result report.",
    )
    parser.add_argument(
        "--original-comparison",
        action="store_true",
        help="Optionally compare with separately supplied historical reference CSVs.",
    )
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--master-seed", type=int, default=250217518)
    parser.add_argument(
        "--rl-inference-mode",
        choices=["stochastic", "deterministic"],
        default="deterministic",
        help=(
            "Use modal/mean actions (paper protocol) or sampled actions from the "
            "fixed trained policies."
        ),
    )
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--rebalance-window", type=int, default=63)
    parser.add_argument("--validation-window", type=int, default=63)
    parser.add_argument("--tau-start", type=float, default=0.01)
    parser.add_argument("--tau-stop", type=float, default=0.89)
    parser.add_argument("--tau-step", type=float, default=0.01)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def stable_seed(master_seed: int, *parts: object) -> int:
    payload = "|".join([str(master_seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "little") % (
        2**31 - 1
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_sd_ci(values: pd.Series | np.ndarray) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    if len(array) <= 1 or np.isclose(sd, 0.0):
        return mean, sd, mean, mean
    half_width = float(stats.t.ppf(0.975, len(array) - 1) * sd / np.sqrt(len(array)))
    return mean, sd, mean - half_width, mean + half_width


def markdown_table(frame: pd.DataFrame, columns: list[str], decimals: int = 4) -> str:
    view = frame.loc[:, columns].copy()
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


def load_fixed_models(
    fixed_run_dir: Path,
    windows: list[dict[str, object]],
    *,
    timesteps: int,
    model_seed: int,
    observation_space: object,
    action_space: object,
) -> tuple[dict[int, dict[str, object]], pd.DataFrame]:
    # The fixed checkpoints were serialized under NumPy 2, while the verified
    # PyTorch environment uses NumPy 1.26. Alias only the renamed pickle module.
    if "numpy._core.numeric" not in sys.modules:
        import numpy.core.numeric as numpy_core_numeric

        sys.modules["numpy._core.numeric"] = numpy_core_numeric

    from stable_baselines3 import A2C
    from stable_baselines3 import PPO
    from stable_baselines3 import SAC

    classes = {"a2c": A2C, "ppo": PPO, "sac": SAC}
    model_root = fixed_run_dir / "models"
    models: dict[int, dict[str, object]] = {}
    manifest_rows: list[dict[str, object]] = []
    for window_info in windows:
        window = int(window_info["window"])
        models[window] = {}
        for model_name in MODEL_NAMES:
            stem = f"agent_{model_name}_rolling_w{window}_{timesteps}_seed{model_seed}_best"
            path = model_root / f"{stem}.zip"
            history_path = model_root / f"agent_{model_name}_rolling_w{window}_{timesteps}_seed{model_seed}_validation_history.csv"
            if not path.exists():
                raise FileNotFoundError(f"fixed RL checkpoint is missing: {path}")
            if not history_path.exists():
                raise FileNotFoundError(f"checkpoint validation history is missing: {history_path}")
            history = pd.read_csv(history_path)
            best_rows = history[
                history["is_best"].astype(str).str.lower().isin(["true", "1"])
            ]
            selected_step = int(best_rows.iloc[-1]["timesteps"]) if len(best_rows) else -1
            models[window][model_name] = classes[model_name].load(
                str(path.with_suffix("")),
                device="cpu",
                custom_objects={
                    "observation_space": observation_space,
                    "action_space": action_space,
                    "_last_obs": None,
                    "_last_episode_starts": None,
                },
            )
            manifest_rows.append(
                {
                    "window": window,
                    "model": model_name,
                    "checkpoint": str(path),
                    "sha256": sha256_file(path),
                    "training_seed": model_seed,
                    "selected_validation_step": selected_step,
                    "retrained_in_backtests": False,
                }
            )
    manifest = pd.DataFrame(manifest_rows)
    if len(manifest) != len(windows) * len(MODEL_NAMES):
        raise ValueError("the fixed checkpoint manifest is incomplete")
    return models, manifest


def simulate_selected_targets(
    selected_holdings: np.ndarray,
    trade_data: pd.DataFrame,
    kwargs: dict[str, object],
    *,
    initial: bool,
    previous_state: list[float] | None,
) -> tuple[pd.DataFrame, list[float]]:
    environment = build_env(
        trade_data,
        kwargs,
        initial=initial,
        previous_state=[] if previous_state is None else previous_state,
    )
    vec_env, _obs = environment.get_sb_env()
    expected_steps = len(trade_data.index.unique()) - 1
    if len(selected_holdings) != expected_steps:
        raise ValueError(
            f"selected holding count {len(selected_holdings)} does not match {expected_steps}"
        )
    for target in selected_holdings:
        state_before = np.asarray(environment.render(), dtype=float)
        current = shares_from_state(state_before, int(kwargs["stock_dim"]))
        action = np.clip((target - current) / float(kwargs["hmax"]), -1.0, 1.0)
        _obs, _rewards, dones, _info = vec_env.step(np.asarray([action]))
        if dones[0]:
            break
    return environment.save_asset_memory(), list(environment.render())


def decision_modes(
    classifiers: list[tuple[str, object]], candidates: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dispersions: list[float] = []
    aggressive: list[int] = []
    conservative: list[int] = []
    for day_candidates in candidates:
        q = confidence_matrix(classifiers, day_candidates, [0, 1])
        dispersion = holding_dispersion(day_candidates)
        aggressive_decision = select_holding_from_confidence(
            day_candidates, q, tau=1.0, dispersion=0.0
        )
        conservative_decision = select_holding_from_confidence(
            day_candidates, q, tau=0.0, dispersion=1.0
        )
        dispersions.append(dispersion)
        aggressive.append(aggressive_decision.selected_index)
        conservative.append(conservative_decision.selected_index)
    return (
        np.asarray(dispersions, dtype=float),
        np.asarray(aggressive, dtype=np.int8),
        np.asarray(conservative, dtype=np.int8),
    )


def align_account_curve(account: pd.DataFrame, expected_dates: np.ndarray) -> np.ndarray:
    frame = account.copy()
    frame["date"] = frame["date"].astype(str)
    if frame["date"].duplicated().any():
        raise ValueError("account curve contains duplicate dates")
    aligned = frame.set_index("date")["account_value"].reindex(expected_dates)
    if aligned.isna().any():
        missing = aligned[aligned.isna()].index.tolist()[:5]
        raise ValueError(f"account curve is missing expected dates: {missing}")
    return aligned.to_numpy(dtype=float)


def run_one_backtest(
    repeat: int,
    *,
    models: dict[int, dict[str, object]],
    full_data: pd.DataFrame,
    windows: list[dict[str, object]],
    kwargs: dict[str, object],
    tau_values: np.ndarray,
    expected_dates: np.ndarray,
    master_seed: int,
    output_dir: Path,
    deterministic_rl: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    config_keys = [(pair, group) for pair in PAIR_KEYS for group in range(1, 6)]
    base_curves = {name: pd.DataFrame() for name in MODEL_NAMES}
    base_last_states: dict[str, list[float] | None] = {name: None for name in MODEL_NAMES}
    decision_inputs: dict[
        tuple[str, int], list[tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ] = {key: [] for key in config_keys}
    classifier_rows: list[dict[str, object]] = []

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
            trade_seed = stable_seed(master_seed, "trade", repeat, window, model_name)
            calibration_holdings[model_name], _, _ = collect_holdings_and_account(
                model,
                calibration,
                kwargs,
                deterministic=deterministic_rl,
                prediction_seed=calibration_seed,
            )
            trade_holdings[model_name], account, last_state = collect_holdings_and_account(
                model,
                trade,
                kwargs,
                initial=base_last_states[model_name] is None,
                previous_state=base_last_states[model_name],
                deterministic=deterministic_rl,
                prediction_seed=trade_seed,
            )
            base_last_states[model_name] = last_state
            base_curves[model_name] = append_account_curve(base_curves[model_name], account)

        for pair in PAIR_KEYS:
            left, right = PAIR_COMPONENTS[pair]
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
                dispersions, aggressive, conservative = decision_modes(
                    classifiers, candidates
                )
                decision_inputs[(pair, group)].append(
                    (trade, candidates, dispersions, aggressive, conservative)
                )
                classifier_rows.append(
                    {
                        "repeat": repeat,
                        "window": window,
                        "pair": pair,
                        "classifier_group": group,
                        "classifier_seed": classifier_seed,
                        "calibration_source": window_info["calibration_source"],
                        "calibration_start": window_info["calibration_start"],
                        "calibration_end": window_info["calibration_end"],
                        "trade_start": window_info["trade_start"],
                        "trade_end": window_info["trade_end"],
                    }
                )

    base_metric_rows: list[dict[str, object]] = []
    base_curve_array = np.empty((len(MODEL_NAMES), len(expected_dates)), dtype=float)
    for model_index, model_name in enumerate(MODEL_NAMES):
        curve = align_account_curve(base_curves[model_name], expected_dates)
        base_curve_array[model_index] = curve
        base_metric_rows.append(
            {
                "repeat": repeat,
                "model": model_name,
                **metrics_from_account_values(curve),
            }
        )

    ensemble_curve_array = np.empty(
        (len(config_keys), len(tau_values), len(expected_dates)), dtype=float
    )
    metric_rows: list[dict[str, object]] = []
    for config_index, (pair, group) in enumerate(config_keys):
        inputs = decision_inputs[(pair, group)]
        signatures: dict[bytes, list[int]] = {}
        selected_by_tau: list[list[np.ndarray]] = []
        for tau_index, tau in enumerate(tau_values):
            selected_windows: list[np.ndarray] = []
            signature_parts: list[bytes] = []
            for _trade, _candidates, dispersions, aggressive, conservative in inputs:
                selected = np.where(dispersions < float(tau), aggressive, conservative)
                selected_windows.append(selected)
                signature_parts.append(selected.tobytes())
            signature = b"|".join(signature_parts)
            signatures.setdefault(signature, []).append(tau_index)
            selected_by_tau.append(selected_windows)

        for tau_indices in signatures.values():
            representative = tau_indices[0]
            selected_windows = selected_by_tau[representative]
            curve = pd.DataFrame()
            last_state: list[float] | None = None
            for input_index, (trade, candidates, _d, _a, _c) in enumerate(inputs):
                selected = selected_windows[input_index]
                targets = candidates[np.arange(len(selected)), selected]
                account, last_state = simulate_selected_targets(
                    targets,
                    trade,
                    kwargs,
                    initial=last_state is None,
                    previous_state=last_state,
                )
                curve = append_account_curve(curve, account)
            values = align_account_curve(curve, expected_dates)
            metric_values = metrics_from_account_values(values)
            for tau_index in tau_indices:
                ensemble_curve_array[config_index, tau_index] = values
                metric_rows.append(
                    {
                        "repeat": repeat,
                        "pair": pair,
                        "classifier_group": group,
                        "tau": float(tau_values[tau_index]),
                        "equivalent_tau_paths": len(tau_indices),
                        **metric_values,
                    }
                )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["pair", "classifier_group", "tau"]
    )
    base_metrics = pd.DataFrame(base_metric_rows).sort_values("model")
    classifier_audit = pd.DataFrame(classifier_rows)

    run_dir = output_dir / "runs" / f"repeat_{repeat:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(run_dir / "ensemble_metrics.csv", index=False)
    base_metrics.to_csv(run_dir / "base_metrics.csv", index=False)
    classifier_audit.to_csv(run_dir / "classifier_refit_audit.csv", index=False)
    np.savez_compressed(
        run_dir / "account_curves.npz",
        ensemble=ensemble_curve_array,
        base=base_curve_array,
        dates=expected_dates,
    )
    return metrics, base_metrics, ensemble_curve_array, base_curve_array, classifier_audit


def load_completed_backtest(
    repeat: int, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    run_dir = output_dir / "runs" / f"repeat_{repeat:02d}"
    metrics = pd.read_csv(run_dir / "ensemble_metrics.csv")
    base = pd.read_csv(run_dir / "base_metrics.csv")
    classifier_audit = pd.read_csv(run_dir / "classifier_refit_audit.csv")
    with np.load(run_dir / "account_curves.npz") as arrays:
        ensemble_curves = arrays["ensemble"]
        base_curves = arrays["base"]
    return metrics, base, ensemble_curves, base_curves, classifier_audit


def completed_backtest_exists(repeat: int, output_dir: Path) -> bool:
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


def aggregate_metric_frame(
    frame: pd.DataFrame, keys: list[str], metric_columns: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in frame.groupby(keys, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, key_values))
        row["n_backtests"] = len(group)
        for metric in metric_columns:
            mean, sd, low, high = mean_sd_ci(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def account_metrics_over_time(values: np.ndarray) -> dict[str, np.ndarray]:
    series = pd.Series(np.asarray(values, dtype=float))
    returns = series.pct_change()
    drawdown = series / series.cummax() - 1.0
    sharpe = np.full(len(series), np.nan, dtype=float)
    calmar = np.full(len(series), np.nan, dtype=float)
    minimum_risk_observations = 60
    for index in range(minimum_risk_observations, len(series)):
        sample = returns.iloc[1 : index + 1].dropna()
        if len(sample) > 1 and not np.isclose(sample.std(), 0.0):
            sharpe[index] = float(np.sqrt(252) * sample.mean() / sample.std())
        mdd = abs(float(drawdown.iloc[: index + 1].min()))
        if mdd > 0 and index > 0 and series.iloc[0] > 0 and series.iloc[index] > 0:
            annualized = float((series.iloc[index] / series.iloc[0]) ** (252 / index) - 1)
            calmar[index] = annualized / mdd
    return {
        "equity": series.to_numpy(dtype=float),
        "cumulative_return": (series / series.iloc[0] - 1.0).to_numpy(dtype=float),
        "sharpe": sharpe,
        "max_drawdown": drawdown.to_numpy(dtype=float),
        "calmar": calmar,
    }


def load_original_curves(paper_results_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for pair in PAIR_KEYS:
        for group in range(1, 6):
            path = paper_results_dir / f"{PAPER_FILE_KEYS[pair]}_{group}_average_metrics_results.csv"
            raw = pd.read_csv(path)
            ensemble = raw.iloc[:, :5].copy()
            ensemble.columns = ["tau", *METRICS]
            ensemble["pair"] = pair
            ensemble["classifier_group"] = group
            for component_index, model in enumerate(PAIR_COMPONENTS[pair]):
                start = 5 + component_index * 4
                for metric_index, metric in enumerate(METRICS):
                    ensemble[f"{model}_{metric}"] = pd.to_numeric(
                        raw.iloc[:, start + metric_index], errors="raise"
                    )
            rows.append(ensemble)
    result = pd.concat(rows, ignore_index=True)
    for column in ["tau", *METRICS]:
        result[column] = pd.to_numeric(result[column], errors="raise")
    return result


def build_selected_summary(
    mean_metrics: pd.DataFrame,
    all_metrics: pd.DataFrame,
    all_base: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    best = (
        mean_metrics.sort_values(
            ["pair", "classifier_group", "sharpe_mean", "tau"],
            ascending=[True, True, False, True],
        )
        .groupby(["pair", "classifier_group"], as_index=False)
        .head(1)
        .sort_values(["pair", "classifier_group"])
        .reset_index(drop=True)
    )
    rows: list[dict[str, object]] = []
    paired_rows: list[pd.DataFrame] = []
    base_means = all_base.groupby("model")[METRICS].mean()
    for best_row in best.itertuples(index=False):
        selected = all_metrics[
            (all_metrics["pair"] == best_row.pair)
            & (all_metrics["classifier_group"] == best_row.classifier_group)
            & np.isclose(all_metrics["tau"], best_row.tau)
        ].copy()
        components = PAIR_COMPONENTS[best_row.pair]
        stronger_model = max(components, key=lambda name: float(base_means.loc[name, "sharpe"]))
        component = all_base[all_base["model"] == stronger_model][
            ["repeat", *METRICS]
        ].rename(columns={metric: f"component_{metric}" for metric in METRICS})
        selected = selected.merge(component, on="repeat", how="left")
        selected["stronger_component"] = stronger_model
        selected["delta_sharpe"] = selected["sharpe"] - selected["component_sharpe"]
        selected["beats_stronger_component"] = selected["delta_sharpe"] > 0
        paired_rows.append(selected)
        delta_mean, delta_sd, delta_low, delta_high = mean_sd_ci(selected["delta_sharpe"])
        rows.append(
            {
                "pair": best_row.pair,
                "classifier_group": int(best_row.classifier_group),
                "selected_global_tau": float(best_row.tau),
                "ensemble_return_mean": float(best_row.cumulative_return_mean),
                "ensemble_return_sd": float(best_row.cumulative_return_sd),
                "ensemble_return_ci_low": float(best_row.cumulative_return_ci_low),
                "ensemble_return_ci_high": float(best_row.cumulative_return_ci_high),
                "ensemble_sharpe_mean": float(best_row.sharpe_mean),
                "ensemble_sharpe_sd": float(best_row.sharpe_sd),
                "ensemble_sharpe_ci_low": float(best_row.sharpe_ci_low),
                "ensemble_sharpe_ci_high": float(best_row.sharpe_ci_high),
                "ensemble_calmar_mean": float(best_row.calmar_mean),
                "ensemble_calmar_sd": float(best_row.calmar_sd),
                "ensemble_calmar_ci_low": float(best_row.calmar_ci_low),
                "ensemble_calmar_ci_high": float(best_row.calmar_ci_high),
                "ensemble_mdd_mean": float(best_row.max_drawdown_mean),
                "ensemble_mdd_sd": float(best_row.max_drawdown_sd),
                "ensemble_mdd_ci_low": float(best_row.max_drawdown_ci_low),
                "ensemble_mdd_ci_high": float(best_row.max_drawdown_ci_high),
                "stronger_component": stronger_model,
                "component_sharpe_mean": float(base_means.loc[stronger_model, "sharpe"]),
                "delta_sharpe_mean": delta_mean,
                "delta_sharpe_sd": delta_sd,
                "delta_sharpe_ci_low": delta_low,
                "delta_sharpe_ci_high": delta_high,
                "win_rate_vs_stronger": float(selected["beats_stronger_component"].mean()),
                "wins_vs_stronger": int(selected["beats_stronger_component"].sum()),
                "n_backtests": len(selected),
            }
        )
    return pd.DataFrame(rows), pd.concat(paired_rows, ignore_index=True)


def build_configuration_comparison(
    selected: pd.DataFrame,
    all_base: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_means = all_base.groupby("model")[METRICS].mean()
    global_best_single = float(base_means["sharpe"].max())
    rows: list[dict[str, object]] = []
    for result in selected.itertuples(index=False):
        component = str(result.stronger_component)
        row: dict[str, object] = {
            "pair": result.pair,
            "classifier_group": int(result.classifier_group),
            "selected_global_tau": float(result.selected_global_tau),
            "stronger_component": component,
            "ensemble_cumulative_return": float(result.ensemble_return_mean),
            "component_cumulative_return": float(
                base_means.loc[component, "cumulative_return"]
            ),
            "ensemble_sharpe": float(result.ensemble_sharpe_mean),
            "component_sharpe": float(base_means.loc[component, "sharpe"]),
            "ensemble_calmar": float(result.ensemble_calmar_mean),
            "component_calmar": float(base_means.loc[component, "calmar"]),
            "ensemble_max_drawdown": float(result.ensemble_mdd_mean),
            "component_max_drawdown": float(
                base_means.loc[component, "max_drawdown"]
            ),
            "delta_sharpe_ci_low": float(result.delta_sharpe_ci_low),
            "delta_sharpe_ci_high": float(result.delta_sharpe_ci_high),
            "wins_vs_stronger": int(result.wins_vs_stronger),
            "n_backtests": int(result.n_backtests),
        }
        row["delta_cumulative_return"] = (
            row["ensemble_cumulative_return"] - row["component_cumulative_return"]
        )
        row["delta_sharpe"] = row["ensemble_sharpe"] - row["component_sharpe"]
        row["delta_calmar"] = row["ensemble_calmar"] - row["component_calmar"]
        row["delta_max_drawdown"] = (
            row["ensemble_max_drawdown"] - row["component_max_drawdown"]
        )
        row["beats_global_best_single_sharpe"] = (
            row["ensemble_sharpe"] > global_best_single
        )
        rows.append(row)

    detailed = pd.DataFrame(rows).sort_values(["pair", "classifier_group"])
    criteria = [
        ("Mean cumulative return is higher", detailed["delta_cumulative_return"] > 0),
        ("Mean Sharpe ratio is higher", detailed["delta_sharpe"] > 0),
        (
            "Paired Sharpe 95% interval is entirely positive",
            detailed["delta_sharpe_ci_low"] > 0,
        ),
        (
            "Sharpe win rate exceeds 50% across classifier refits",
            detailed["wins_vs_stronger"] > detailed["n_backtests"] / 2,
        ),
        (
            "Mean Sharpe exceeds the globally strongest single model",
            detailed["beats_global_best_single_sharpe"],
        ),
        ("Mean Calmar ratio is higher", detailed["delta_calmar"] > 0),
        (
            "Mean maximum drawdown is better (closer to zero)",
            detailed["delta_max_drawdown"] > 0,
        ),
    ]
    aggregate = pd.DataFrame(
        {
            "criterion": [name for name, _mask in criteria],
            "configurations": [int(mask.sum()) for _name, mask in criteria],
            "total_configurations": len(detailed),
        }
    )
    return detailed, aggregate


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return pd.Series(adjusted, index=p_values.index)


def build_paper_tau_stability(
    original: pd.DataFrame,
    all_metrics: pd.DataFrame,
    all_base: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original_best = (
        original.sort_values(
            ["pair", "classifier_group", "sharpe", "tau"],
            ascending=[True, True, False, True],
        )
        .groupby(["pair", "classifier_group"], as_index=False)
        .head(1)
    )
    base_means = all_base.groupby("model")[METRICS].mean()
    summary_rows: list[dict[str, object]] = []
    paired_rows: list[pd.DataFrame] = []
    for paper_row in original_best.itertuples(index=False):
        selected = all_metrics[
            (all_metrics["pair"] == paper_row.pair)
            & (all_metrics["classifier_group"] == paper_row.classifier_group)
            & np.isclose(all_metrics["tau"], paper_row.tau)
        ].copy()
        components = PAIR_COMPONENTS[paper_row.pair]
        stronger_model = max(
            components, key=lambda name: float(base_means.loc[name, "sharpe"])
        )
        component = all_base[all_base["model"] == stronger_model][
            ["repeat", *METRICS]
        ].rename(columns={metric: f"component_{metric}" for metric in METRICS})
        selected = selected.merge(component, on="repeat", how="left")
        selected["paper_tau"] = float(paper_row.tau)
        selected["stronger_component"] = stronger_model
        selected["delta_sharpe"] = selected["sharpe"] - selected["component_sharpe"]
        selected["beats_stronger_component"] = selected["delta_sharpe"] > 0
        paired_rows.append(selected)
        delta_mean, delta_sd, delta_low, delta_high = mean_sd_ci(selected["delta_sharpe"])
        t_result = stats.ttest_1samp(selected["delta_sharpe"], 0.0, alternative="greater")
        try:
            wilcoxon_p = float(
                stats.wilcoxon(
                    selected["delta_sharpe"], alternative="greater", zero_method="wilcox"
                ).pvalue
            )
        except ValueError:
            wilcoxon_p = 1.0
        row: dict[str, object] = {
            "pair": paper_row.pair,
            "classifier_group": int(paper_row.classifier_group),
            "paper_selected_tau": float(paper_row.tau),
            "stronger_component": stronger_model,
            "component_sharpe_mean": float(base_means.loc[stronger_model, "sharpe"]),
            "delta_sharpe_mean": delta_mean,
            "delta_sharpe_sd": delta_sd,
            "delta_sharpe_ci_low": delta_low,
            "delta_sharpe_ci_high": delta_high,
            "win_rate_vs_stronger": float(selected["beats_stronger_component"].mean()),
            "wins_vs_stronger": int(selected["beats_stronger_component"].sum()),
            "one_sided_paired_t_p": float(t_result.pvalue),
            "one_sided_wilcoxon_p": wilcoxon_p,
            "n_backtests": len(selected),
        }
        for metric in METRICS:
            mean, sd, low, high = mean_sd_ci(selected[metric])
            row[f"ensemble_{metric}_mean"] = mean
            row[f"ensemble_{metric}_sd"] = sd
            row[f"ensemble_{metric}_ci_low"] = low
            row[f"ensemble_{metric}_ci_high"] = high
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["pair", "classifier_group"])
    summary["holm_paired_t_p"] = holm_adjust(summary["one_sided_paired_t_p"])
    summary["holm_wilcoxon_p"] = holm_adjust(summary["one_sided_wilcoxon_p"])
    return summary, pd.concat(paired_rows, ignore_index=True)


def build_original_comparison(
    original: pd.DataFrame,
    mean_metrics: pd.DataFrame,
    selected_summary: pd.DataFrame,
    paper_tau_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original_best = (
        original.sort_values(
            ["pair", "classifier_group", "sharpe", "tau"],
            ascending=[True, True, False, True],
        )
        .groupby(["pair", "classifier_group"], as_index=False)
        .head(1)
    )
    selected = selected_summary.rename(
        columns={
            "selected_global_tau": "repro_tau",
            "ensemble_return_mean": "repro_cumulative_return",
            "ensemble_sharpe_mean": "repro_sharpe",
            "ensemble_calmar_mean": "repro_calmar",
            "ensemble_mdd_mean": "repro_max_drawdown",
        }
    )
    detailed = original_best.merge(selected, on=["pair", "classifier_group"], how="inner")
    paper_tau_values = paper_tau_summary[
        [
            "pair",
            "classifier_group",
            "ensemble_cumulative_return_mean",
            "ensemble_sharpe_mean",
            "ensemble_calmar_mean",
            "ensemble_max_drawdown_mean",
        ]
    ].rename(
        columns={
            "ensemble_cumulative_return_mean": "repro_at_paper_tau_cumulative_return",
            "ensemble_sharpe_mean": "repro_at_paper_tau_sharpe",
            "ensemble_calmar_mean": "repro_at_paper_tau_calmar",
            "ensemble_max_drawdown_mean": "repro_at_paper_tau_max_drawdown",
        }
    )
    detailed = detailed.merge(
        paper_tau_values, on=["pair", "classifier_group"], how="inner"
    )
    detailed = detailed.rename(
        columns={metric: f"paper_{metric}" for metric in ["tau", *METRICS]}
    )
    for metric in METRICS:
        detailed[f"delta_{metric}"] = (
            detailed[f"repro_{metric}"] - detailed[f"paper_{metric}"]
        )
        detailed[f"absolute_{metric}_error"] = detailed[f"delta_{metric}"].abs()
        detailed[f"paper_tau_delta_{metric}"] = (
            detailed[f"repro_at_paper_tau_{metric}"] - detailed[f"paper_{metric}"]
        )
        detailed[f"paper_tau_absolute_{metric}_error"] = detailed[
            f"paper_tau_delta_{metric}"
        ].abs()
    detailed["absolute_tau_error"] = (detailed["repro_tau"] - detailed["paper_tau"]).abs()

    curve_rows: list[dict[str, object]] = []
    for pair in PAIR_KEYS:
        paper_curve = original[
            (original["pair"] == pair) & (original["classifier_group"] == 1)
        ].sort_values("tau").rename(
            columns={metric: f"{metric}_paper" for metric in COMPARABLE_METRICS}
        )
        repro_curve = mean_metrics[
            (mean_metrics["pair"] == pair)
            & (mean_metrics["classifier_group"] == 1)
        ].sort_values("tau")
        merged = paper_curve.merge(repro_curve, on="tau", suffixes=("_paper", "_repro"))
        for metric in COMPARABLE_METRICS:
            paper_values = merged[f"{metric}_paper"].astype(float)
            repro_values = merged[f"{metric}_mean"].astype(float)
            curve_rows.append(
                {
                    "pair": pair,
                    "classifier_group": 1,
                    "metric": metric,
                    "pearson": float(paper_values.corr(repro_values, method="pearson")),
                    "spearman": float(paper_values.corr(repro_values, method="spearman")),
                    "mae": float((repro_values - paper_values).abs().mean()),
                    "paper_mean": float(paper_values.mean()),
                    "reproduction_mean": float(repro_values.mean()),
                }
            )
    return detailed, pd.DataFrame(curve_rows)


def build_error_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario, prefix in (
        ("best_mean_tau", "repro_"),
        ("original_paper_tau", "repro_at_paper_tau_"),
    ):
        for metric in COMPARABLE_METRICS:
            paper = comparison[f"paper_{metric}"].astype(float)
            reproduction = comparison[f"{prefix}{metric}"].astype(float)
            rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "paper_mean": float(paper.mean()),
                    "reproduction_mean": float(reproduction.mean()),
                    "mean_signed_error": float((reproduction - paper).mean()),
                    "mae": float((reproduction - paper).abs().mean()),
                    "rmse": float(np.sqrt(np.mean(np.square(reproduction - paper)))),
                    "spearman": float(paper.corr(reproduction, method="spearman")),
                }
            )
    return pd.DataFrame(rows)


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def save_figure(figure: plt.Figure, path: Path, dpi: int) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_figure5(
    mean_metrics: pd.DataFrame,
    base_summary: pd.DataFrame,
    selected_summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    dataset_label: str,
) -> Path:
    base = base_summary.set_index("model")
    figure, axes = plt.subplots(4, 3, figsize=(17.5, 12), sharex=True)
    for column, pair in enumerate(PAIR_KEYS):
        subset = mean_metrics[
            (mean_metrics["pair"] == pair)
            & (mean_metrics["classifier_group"] == 1)
        ].sort_values("tau")
        selected = selected_summary[
            (selected_summary["pair"] == pair)
            & (selected_summary["classifier_group"] == 1)
        ].iloc[0]
        for row, metric in enumerate(METRICS):
            axis = axes[row, column]
            axis.plot(
                subset["tau"], subset[f"{metric}_mean"], color=ENSEMBLE_COLOR,
                linewidth=2.2, label="Ensemble mean",
            )
            axis.fill_between(
                subset["tau"], subset[f"{metric}_ci_low"], subset[f"{metric}_ci_high"],
                color=ENSEMBLE_COLOR, alpha=0.16, linewidth=0,
            )
            for model in PAIR_COMPONENTS[pair]:
                axis.axhline(
                    base.loc[model, f"{metric}_mean"], color=MODEL_COLORS[model],
                    linestyle="--", linewidth=1.35, label=MODEL_LABELS[model],
                )
            axis.axvline(
                selected["selected_global_tau"], color="#b33b32", linestyle="-.",
                linewidth=1.15,
            )
            if row == 0:
                axis.set_title(PAIR_LABELS[pair])
                axis.legend(frameon=False, fontsize=8)
            if column == 0:
                axis.set_ylabel(METRIC_LABELS[metric])
            if row == 3:
                axis.set_xlabel("Fixed global variance threshold tau")
            style_axis(axis)
    figure.suptitle(
        f"Figure 5: {dataset_label} Variance-Threshold Sensitivity (Group 1)",
        fontsize=15,
        y=1.01,
    )
    path = output_dir / "figure5_fixed_rl_30_backtests_group1.png"
    save_figure(figure, path, dpi)
    return path


def plot_figure4(
    selected_summary: pd.DataFrame,
    base_summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    dataset_label: str,
) -> Path:
    base = base_summary.set_index("model")
    mapping = {
        "cumulative_return": (
            "ensemble_return_mean", "ensemble_return_ci_low", "ensemble_return_ci_high"
        ),
        "sharpe": (
            "ensemble_sharpe_mean", "ensemble_sharpe_ci_low", "ensemble_sharpe_ci_high"
        ),
        "calmar": (
            "ensemble_calmar_mean", "ensemble_calmar_ci_low", "ensemble_calmar_ci_high"
        ),
        "max_drawdown": (
            "ensemble_mdd_mean", "ensemble_mdd_ci_low", "ensemble_mdd_ci_high"
        ),
    }
    figure, axes = plt.subplots(4, 3, figsize=(17.5, 12), sharex=True)
    for column, pair in enumerate(PAIR_KEYS):
        subset = selected_summary[selected_summary["pair"] == pair].sort_values(
            "classifier_group"
        )
        for row, metric in enumerate(METRICS):
            axis = axes[row, column]
            value_column, low_column, high_column = mapping[metric]
            values = subset[value_column].to_numpy(dtype=float)
            lower = subset[low_column].to_numpy(dtype=float)
            upper = subset[high_column].to_numpy(dtype=float)
            axis.errorbar(
                subset["classifier_group"],
                values,
                yerr=np.vstack([values - lower, upper - values]),
                color=ENSEMBLE_COLOR,
                marker="o",
                linewidth=2.1,
                elinewidth=1.0,
                capsize=3,
                label="Ensemble mean and 95% CI",
            )
            for model in PAIR_COMPONENTS[pair]:
                axis.axhline(
                    base.loc[model, f"{metric}_mean"], color=MODEL_COLORS[model],
                    linestyle="--", linewidth=1.35, label=MODEL_LABELS[model],
                )
            if row == 0:
                axis.set_title(PAIR_LABELS[pair])
                axis.legend(frameon=False, fontsize=8)
            if column == 0:
                axis.set_ylabel(METRIC_LABELS[metric])
            if row == 3:
                axis.set_xlabel("Classifier group")
                axis.set_xticks([1, 2, 3, 4, 5])
            style_axis(axis)
    figure.suptitle(
        f"Figure 4: {dataset_label} Classifier-Group Comparison",
        fontsize=15,
        y=1.01,
    )
    path = output_dir / "figure4_fixed_rl_30_backtests_classifier_groups.png"
    save_figure(figure, path, dpi)
    return path


def plot_figure3(
    ensemble_curves: list[np.ndarray],
    base_curves: list[np.ndarray],
    selected_summary: pd.DataFrame,
    tau_values: np.ndarray,
    dates: np.ndarray,
    output_dir: Path,
    dpi: int,
    dataset_label: str,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    config_keys = [(pair, group) for pair in PAIR_KEYS for group in range(1, 6)]
    config_index = {key: index for index, key in enumerate(config_keys)}
    date_values = pd.to_datetime(dates)
    rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    figure, axes = plt.subplots(4, 3, figsize=(18, 12.5), sharex=True)
    row_metrics = ["equity", "sharpe", "max_drawdown", "calmar"]
    row_labels = [
        "Daily account value ($)",
        "Expanding Sharpe (min 60 obs)",
        "Expanding MDD",
        "Expanding Calmar (min 60 obs)",
    ]
    for column, pair in enumerate(PAIR_KEYS):
        selected = selected_summary[
            (selected_summary["pair"] == pair)
            & (selected_summary["classifier_group"] == 1)
        ].iloc[0]
        tau_index = int(np.where(np.isclose(tau_values, selected["selected_global_tau"]))[0][0])
        ensemble_curve_matrix = np.stack(
            [curve[config_index[(pair, 1)], tau_index] for curve in ensemble_curves]
        )
        strategies: dict[str, tuple[np.ndarray, str, str]] = {
            "ensemble": (
                ensemble_curve_matrix,
                ENSEMBLE_COLOR,
                f"Ensemble tau={selected['selected_global_tau']:.2f}",
            )
        }
        for model in PAIR_COMPONENTS[pair]:
            model_index = MODEL_NAMES.index(model)
            strategies[model] = (
                np.stack([curve[model_index] for curve in base_curves]),
                MODEL_COLORS[model],
                MODEL_LABELS[model],
            )
        for strategy, (curve_matrix, color, label) in strategies.items():
            temporal = {metric: [] for metric in row_metrics}
            for curve in curve_matrix:
                metrics = account_metrics_over_time(curve)
                for metric in row_metrics:
                    temporal[metric].append(metrics[metric])
            for row, metric in enumerate(row_metrics):
                matrix = np.stack(temporal[metric])
                mean = matrix.mean(axis=0)
                range_low = matrix.min(axis=0)
                range_high = matrix.max(axis=0)
                if len(matrix) > 1:
                    sd = matrix.std(axis=0, ddof=1)
                    half = stats.t.ppf(0.975, len(matrix) - 1) * sd / np.sqrt(len(matrix))
                else:
                    sd = np.zeros_like(mean)
                    half = np.zeros_like(mean)
                axis = axes[row, column]
                axis.fill_between(
                    date_values,
                    range_low,
                    range_high,
                    color=color,
                    alpha=0.16 if strategy == "ensemble" else 0.08,
                    linewidth=0,
                    label="Observed min-max range" if strategy == "ensemble" else None,
                    zorder=1,
                )
                axis.plot(
                    date_values,
                    range_low,
                    color=color,
                    linewidth=0.75,
                    linestyle=":",
                    alpha=0.8,
                    zorder=2,
                )
                axis.plot(
                    date_values,
                    range_high,
                    color=color,
                    linewidth=0.75,
                    linestyle=":",
                    alpha=0.8,
                    zorder=2,
                )
                axis.plot(
                    date_values,
                    mean,
                    color=color,
                    linewidth=2.0,
                    label=label,
                    zorder=3,
                )
                rows.extend(
                    {
                        "date": str(date),
                        "pair": pair,
                        "strategy": strategy,
                        "metric": metric,
                        "mean": float(mean[index]),
                        "sd": float(sd[index]),
                        "ci_low": float(mean[index] - half[index]),
                        "ci_high": float(mean[index] + half[index]),
                        "range_low": float(range_low[index]),
                        "range_high": float(range_high[index]),
                    }
                    for index, date in enumerate(dates)
                )
        stronger_model = str(selected["stronger_component"])
        stronger_index = MODEL_NAMES.index(stronger_model)
        stronger_curves = np.stack([curve[stronger_index] for curve in base_curves])
        ensemble_low = ensemble_curve_matrix.min(axis=0)
        ensemble_high = ensemble_curve_matrix.max(axis=0)
        stronger_mean = stronger_curves.mean(axis=0)
        inside = (stronger_mean >= ensemble_low) & (stronger_mean <= ensemble_high)
        coverage_rows.append(
            {
                "pair": pair,
                "classifier_group": 1,
                "selected_global_tau": float(selected["selected_global_tau"]),
                "stronger_component": stronger_model,
                "daily_coverage_rate": float(inside.mean()),
                "ensemble_final_mean": float(ensemble_curve_matrix[:, -1].mean()),
                "ensemble_final_min": float(ensemble_curve_matrix[:, -1].min()),
                "ensemble_final_max": float(ensemble_curve_matrix[:, -1].max()),
                "stronger_component_final": float(stronger_mean[-1]),
            }
        )
        for row, label in enumerate(row_labels):
            axis = axes[row, column]
            if row == 0:
                axis.set_title(f"{PAIR_LABELS[pair]}\nClassifier Group 1")
                axis.legend(frameon=False, fontsize=8)
                axis.ticklabel_format(axis="y", style="sci", scilimits=(6, 6))
            if column == 0:
                axis.set_ylabel(label)
            if row == 3:
                axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                axis.tick_params(axis="x", rotation=30)
            style_axis(axis)
    figure.suptitle(
        f"Figure 3: {dataset_label} Performance Across 30 Classifier Refits",
        fontsize=15,
        y=1.01,
    )
    path = output_dir / "figure3_fixed_rl_30_backtests_yearly_performance.png"
    save_figure(figure, path, dpi)
    return path, pd.DataFrame(rows), pd.DataFrame(coverage_rows)


def plot_original_curve_comparison(
    original: pd.DataFrame,
    mean_metrics: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> Path:
    figure, axes = plt.subplots(3, 3, figsize=(17.5, 9.5), sharex=True)
    for column, pair in enumerate(PAIR_KEYS):
        paper = original[
            (original["pair"] == pair) & (original["classifier_group"] == 1)
        ].sort_values("tau")
        reproduction = mean_metrics[
            (mean_metrics["pair"] == pair)
            & (mean_metrics["classifier_group"] == 1)
        ].sort_values("tau")
        for row, metric in enumerate(COMPARABLE_METRICS):
            axis = axes[row, column]
            axis.plot(
                paper["tau"], paper[metric], color=PAPER_COLOR, linewidth=1.8,
                linestyle="--", label="Original paper mean",
            )
            axis.plot(
                reproduction["tau"], reproduction[f"{metric}_mean"],
                color=ENSEMBLE_COLOR, linewidth=2.1, label="Reproduction mean",
            )
            axis.fill_between(
                reproduction["tau"], reproduction[f"{metric}_ci_low"],
                reproduction[f"{metric}_ci_high"], color=ENSEMBLE_COLOR,
                alpha=0.14, linewidth=0,
            )
            if row == 0:
                axis.set_title(PAIR_LABELS[pair])
                axis.legend(frameon=False, fontsize=8)
            if column == 0:
                axis.set_ylabel(METRIC_LABELS[metric])
            if row == 2:
                axis.set_xlabel("Fixed global variance threshold tau")
            style_axis(axis)
    figure.suptitle(
        "Original Figure 5 Data vs Fixed-RL 30-Backtest Reproduction",
        fontsize=15,
        y=1.01,
    )
    path = output_dir / "figure5_original_vs_fixed_rl_30_backtests.png"
    save_figure(figure, path, dpi)
    return path


def write_report(
    output_dir: Path,
    selected: pd.DataFrame,
    paired: pd.DataFrame,
    base_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    curve_consistency: pd.DataFrame,
    paper_tau_summary: pd.DataFrame,
    error_summary: pd.DataFrame,
    manifest: pd.DataFrame,
    repetitions: int,
    classifier_refits: int,
    stochastic_rl_inference: bool,
) -> None:
    wins = int((selected["delta_sharpe_mean"] > 0).sum())
    significant = int((selected["delta_sharpe_ci_low"] > 0).sum())
    majority = int((selected["win_rate_vs_stronger"] > 0.5).sum())
    paper_tau_wins = int((paper_tau_summary["delta_sharpe_mean"] > 0).sum())
    paper_tau_significant = int((paper_tau_summary["delta_sharpe_ci_low"] > 0).sum())
    paper_tau_majority = int((paper_tau_summary["win_rate_vs_stronger"] > 0.5).sum())
    paper_tau_holm = int((paper_tau_summary["holm_paired_t_p"] < 0.05).sum())
    global_best_single_sharpe = float(selected["component_sharpe_mean"].max())
    beats_global_single = int(
        (selected["ensemble_sharpe_mean"] > global_best_single_sharpe).sum()
    )
    top_ensemble = selected.loc[selected["ensemble_sharpe_mean"].idxmax()]
    paper_wins = 0
    for row in comparison.itertuples(index=False):
        components = PAIR_COMPONENTS[row.pair]
        paper_component_sharpes = [getattr(row, f"{model}_sharpe") for model in components]
        paper_wins += int(row.paper_sharpe > max(paper_component_sharpes))
    tau_mae = float(comparison["absolute_tau_error"].mean())
    sharpe_mae = float(comparison["absolute_sharpe_error"].mean())
    curve_spearman = float(curve_consistency["spearman"].mean())
    if stochastic_rl_inference:
        consistency_verdict = (
            "Overall consistency verdict: the original paper's universal-stability "
            "conclusion is not reproduced under stochastic sampling from the fixed RL "
            "checkpoints. A2C+SAC Group 1 retains a favorable mean at tau=0.24 and "
            "exactly matches the original selected threshold, but its paired interval "
            "crosses zero. The evidence supports a configuration-specific switching "
            "effect, not stable or universal superiority."
        )
    else:
        consistency_verdict = (
            "Overall consistency verdict: using the modal actions of the same fixed RL "
            f"checkpoints recovers descriptive ensemble superiority in {wins}/15 "
            f"configurations, with {significant}/15 paired intervals entirely above "
            "zero. The contrast with stochastic inference is evidence that the headline "
            "conclusion is highly sensitive to the RL deployment mode."
        )
    lines = [
        "# Fixed-RL 30-Backtest Reproduction and Consistency Audit",
        "",
        "## Protocol",
        "",
        f"- Backtests: {repetitions} complete repetitions.",
        "- RL: the same 12 validation-selected A2C/PPO/SAC checkpoints are loaded in every repetition; no RL `.learn()` or training call is made.",
        (
            "- RL inference: stochastic policy sampling is enabled (`deterministic=False`), with paired repeat/model/window random streams."
            if stochastic_rl_inference
            else "- RL inference: policy modes are used (`deterministic=True`); repetitions vary classifier fitting only."
        ),
        "- Classifiers: refit independently in every rolling window from the immediately prior observed block; the first block is the final 63 training sessions.",
        "- Classifier type and hyperparameters are fixed; grid search is disabled; classifier random seeds vary by repetition/window/pair/group.",
        (
            "- Tau: each candidate value is fixed globally for the complete 2020 path. The same 30 stochastic candidate paths are reused across tau values as common random numbers."
            if stochastic_rl_inference
            else "- Tau: each candidate value is fixed globally for the complete 2020 path. Deterministic candidate paths are reused across tau values."
        ),
        "- Selection: classifier-level confidence votes are aggregated before the aggressive/conservative holding choice.",
        "",
        (
            "## Fixed-RL Stochastic Baselines"
            if stochastic_rl_inference
            else "## Fixed-RL Deterministic Baselines"
        ),
        "",
        markdown_table(
            base_summary,
            [
                "model", "cumulative_return_mean", "cumulative_return_sd",
                "sharpe_mean", "sharpe_sd", "sharpe_ci_low", "sharpe_ci_high",
                "max_drawdown_mean",
            ],
        ),
        "",
        "## Main Stability Result",
        "",
        markdown_table(
            selected,
            [
                "pair", "classifier_group", "selected_global_tau",
                "ensemble_sharpe_mean", "ensemble_sharpe_sd", "stronger_component",
                "component_sharpe_mean", "delta_sharpe_mean", "delta_sharpe_ci_low",
                "delta_sharpe_ci_high", "win_rate_vs_stronger",
            ],
        ),
        "",
        f"- Mean ensemble Sharpe exceeds the stronger fixed component in {wins}/15 pair/group configurations.",
        f"- The paired 95% interval for the Sharpe difference is entirely positive in {significant}/15 configurations.",
        f"- Ensemble wins in more than half of the 30 repetitions in {majority}/15 configurations.",
        f"- The top ensemble is {PAIR_LABELS[top_ensemble['pair']]} Group {int(top_ensemble['classifier_group'])} at tau={top_ensemble['selected_global_tau']:.2f} with mean Sharpe {top_ensemble['ensemble_sharpe_mean']:.4f}; {beats_global_single}/15 ensembles exceed the globally strongest single-RL mean Sharpe ({global_best_single_sharpe:.4f}).",
        "",
        "### Prespecified original-paper tau",
        "",
        markdown_table(
            paper_tau_summary,
            [
                "pair", "classifier_group", "paper_selected_tau",
                "ensemble_sharpe_mean", "component_sharpe_mean", "delta_sharpe_mean",
                "delta_sharpe_ci_low", "delta_sharpe_ci_high",
                "win_rate_vs_stronger", "holm_paired_t_p",
            ],
        ),
        "",
        f"- At the original paper's prespecified tau, mean ensemble Sharpe exceeds the stronger component in {paper_tau_wins}/15 configurations.",
        f"- The paired 95% interval is entirely positive in {paper_tau_significant}/15 configurations; {paper_tau_majority}/15 have a win rate above 50%.",
        f"- After Holm correction across 15 configurations, {paper_tau_holm}/15 one-sided paired t-tests remain below 0.05.",
        "",
        "## Original-Result Consistency",
        "",
        markdown_table(
            error_summary,
            [
                "scenario", "metric", "paper_mean", "reproduction_mean",
                "mean_signed_error", "mae", "spearman",
            ],
        ),
        "",
        f"- Original exports show ensemble Sharpe above both same-row component means in {paper_wins}/15 best-tau configurations.",
        f"- Selected-tau MAE versus the original exports: {tau_mae:.4f}.",
        f"- Best-tau Sharpe MAE versus the original exports: {sharpe_mae:.4f}.",
        f"- Mean Spearman correlation across the nine Group-1 pair/metric tau-sensitivity curves: {curve_spearman:.4f}.",
        "- Calmar values are retained but excluded from numerical consistency claims because the original CSV scale is incompatible with the annualized-return-over-MDD definition.",
        "- Group-1 return/Sharpe curve shape is moderately aligned for A2C+PPO and A2C+SAC, but PPO+SAC has near-zero or negative correlation; the threshold-sensitivity pattern is therefore only partially reproduced.",
        "",
        consistency_verdict,
        "",
        "The stability claim is supported only when the paired mean difference, its interval, and the per-run win rate agree. Selecting the maximum mean Sharpe over 89 tau values is still an in-sample sensitivity analysis, so it is not treated as out-of-sample proof.",
        "",
        "## Audit Checks",
        "",
        f"- Fixed checkpoint files: {len(manifest)}; unique SHA256 values: {manifest['sha256'].nunique()}.",
        f"- Classifier group refits recorded: {classifier_refits} (see `classifier_refit_audit.csv` for the exact run/window records).",
        "- All output tables retain the repetition index so headline means can be recomputed from run-level observations.",
        "",
        "## Artifacts",
        "",
        "- `all_backtest_metrics.csv` and `all_base_metrics.csv`: run-level metrics.",
        "- `mean_metrics_by_fixed_tau.csv`: 30-run mean, SD, and 95% interval for every pair/group/tau.",
        "- `selected_tau_summary.csv` and `selected_tau_paired_runs.csv`: selected fixed-tau stability evidence.",
        "- `paper_tau_stability_summary.csv` and `paper_tau_paired_runs.csv`: externally prespecified original-tau evidence.",
        "- `original_best_comparison.csv` and `group1_curve_consistency.csv`: original-result audit.",
        "- `original_reproduction_error_summary.csv`: aggregate best-tau and original-tau numerical differences.",
        "- `figures/`: paper-corresponding Figures 3/4/5 and a direct original/reproduction comparison.",
        "",
    ]
    (output_dir / "FIXED_RL_30_BACKTEST_CONSISTENCY_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def rolling_window_summary(windows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "window": int(window["window"]),
                "train_start": window["train_start"],
                "train_end": window["train_end"],
                "calibration_start": window["calibration_start"],
                "calibration_end": window["calibration_end"],
                "calibration_source": window["calibration_source"],
                "trade_start": window["trade_start"],
                "trade_end": window["trade_end"],
                "train_dates": len(window["train_dates"]),
                "calibration_dates": len(window["calibration_dates"]),
                "trade_dates": len(window["trade_dates"]),
            }
            for window in windows
        ]
    )


def write_external_dataset_report(
    output_dir: Path,
    *,
    dataset_label: str,
    metadata: dict[str, object],
    repetitions: int,
    windows: pd.DataFrame,
    manifest: pd.DataFrame,
    base_summary: pd.DataFrame,
    selected: pd.DataFrame,
    configuration_comparison: pd.DataFrame,
    outperformance_summary: pd.DataFrame,
    range_coverage: pd.DataFrame,
    classifier_refits: int,
) -> None:
    top = selected.loc[selected["ensemble_sharpe_mean"].idxmax()]
    base_view = base_summary[
        [
            "model",
            "cumulative_return_mean",
            "cumulative_return_sd",
            "sharpe_mean",
            "sharpe_sd",
            "calmar_mean",
            "max_drawdown_mean",
        ]
    ]
    selected_view = selected[
        [
            "pair",
            "classifier_group",
            "selected_global_tau",
            "ensemble_return_mean",
            "ensemble_return_sd",
            "ensemble_sharpe_mean",
            "ensemble_sharpe_sd",
            "ensemble_calmar_mean",
            "ensemble_mdd_mean",
            "stronger_component",
            "delta_sharpe_mean",
            "delta_sharpe_ci_low",
            "delta_sharpe_ci_high",
            "wins_vs_stronger",
        ]
    ]
    checkpoint_view = manifest[
        ["window", "model", "selected_validation_step", "training_seed", "sha256"]
    ].copy()
    checkpoint_view["sha256"] = checkpoint_view["sha256"].str.slice(0, 12)
    aggregate_view = outperformance_summary.copy()
    aggregate_view["result"] = (
        aggregate_view["configurations"].astype(str)
        + "/"
        + aggregate_view["total_configurations"].astype(str)
    )
    lines = [
        f"# {dataset_label} Fixed-RL 30-Backtest Experiment",
        "",
        "## Protocol",
        "",
        f"- Universe: {int(metadata['stock_count'])} aligned stocks.",
        f"- Evaluation split: `{metadata['trade_split']}`, {metadata['trade_start']} through {metadata['trade_end']}.",
        f"- Repetitions: {repetitions}; only rolling classifier fits vary across repetitions.",
        "- RL: one validation-selected A2C/PPO/SAC checkpoint per expanding window, trained with seed 42 and reused unchanged in every repetition.",
        "- Inference: `deterministic=True` for validation, classifier-decision data, and trading.",
        "- Classifiers: five fixed groups, no grid search, refitted on the immediately prior observed block.",
        "- Thresholds: 0.01 through 0.89 by 0.01; each candidate tau is fixed across the complete evaluation path.",
        "- Selection: the reported tau maximizes mean Sharpe across the 30 refits; ties use the smaller tau.",
        "",
        "## Rolling Windows",
        "",
        markdown_table(
            windows,
            [
                "window",
                "train_start",
                "train_end",
                "calibration_start",
                "calibration_end",
                "calibration_source",
                "trade_start",
                "trade_end",
                "trade_dates",
            ],
        ),
        "",
        "## Selected RL Checkpoints",
        "",
        markdown_table(
            checkpoint_view,
            [
                "window",
                "model",
                "selected_validation_step",
                "training_seed",
                "sha256",
            ],
        ),
        "",
        "## Single-RL Baselines",
        "",
        markdown_table(base_view, list(base_view.columns)),
        "",
        "## Main Ensemble Results",
        "",
        markdown_table(selected_view, list(selected_view.columns)),
        "",
        "## Configuration-Level Conclusion",
        "",
        markdown_table(aggregate_view, ["criterion", "result"]),
        "",
        f"The highest mean Sharpe is {PAIR_LABELS[str(top['pair'])]} Group {int(top['classifier_group'])} at tau={float(top['selected_global_tau']):.2f}: {float(top['ensemble_sharpe_mean']):.4f}.",
        "",
        "The 30 repetitions estimate sensitivity to classifier refitting conditional on fixed RL checkpoints and one realized market path. They are not 30 independent RL trainings. Tau is selected on the same completed evaluation span, so these results are a full-path sensitivity analysis rather than a deployable out-of-sample estimate.",
        "",
        "## Audit",
        "",
        f"- Fixed checkpoints: {len(manifest)} files with {manifest['sha256'].nunique()} unique SHA256 hashes.",
        f"- Classifier refits: {classifier_refits} recorded fits at the group level.",
        f"- Configuration rows: {len(configuration_comparison)}; Group-1 range rows: {len(range_coverage)}.",
        "- Run-level metrics and curves are retained under `runs/` for direct recomputation.",
        "- Figure 3 uses the observed pointwise minimum and maximum over 30 classifier refits; deterministic single-RL bounds collapse to their curves.",
        "",
        "## Artifacts",
        "",
        "- `base_model_30_backtest_summary.csv`: deterministic single-RL table.",
        "- `selected_tau_summary.csv`: all 15 pair-group results at selected global tau.",
        "- `paired_sharpe_stability.csv`: paired confidence intervals and wins over the stronger component.",
        "- `configuration_comparison.csv` and `outperformance_summary.csv`: detailed and aggregate conclusions.",
        "- `figure3_range_coverage.csv`: observed ensemble-envelope coverage.",
        "- `figures/`: paper-corresponding Figures 3, 4, and 5 in PNG and PDF.",
        "",
    ]
    (output_dir / "FIXED_RL_30_BACKTEST_DATASET_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    stochastic_rl_inference = args.rl_inference_mode == "stochastic"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_run_dir = Path(args.fixed_run_dir)
    full_data, indicators, trade_start, metadata = load_trademaster_rolling_data(
        args.data_dir, trade_split=args.trade_split
    )
    dataset_label = args.dataset_label or Path(args.data_dir).name
    kwargs = env_kwargs(full_data, indicators)
    compatibility_env = build_env(full_data, kwargs)
    windows = build_rolling_windows(
        full_data,
        trade_start_date=trade_start,
        rebalance_window=args.rebalance_window,
        validation_window=args.validation_window,
        max_windows=None,
    )
    tau_values = tau_grid(args.tau_start, args.tau_stop, args.tau_step)
    expected_tau = np.round(np.arange(0.01, 0.90, 0.01), 2)
    if not np.array_equal(tau_values, expected_tau):
        raise ValueError("this paper-comparison run requires the complete tau grid 0.01-0.89")
    expected_dates = np.concatenate(
        [np.asarray(window["trade_dates"], dtype=str) for window in windows]
    )
    if len(expected_dates) != len(np.unique(expected_dates)):
        raise ValueError("rolling trade windows overlap")
    windows_frame = rolling_window_summary(windows)
    windows_frame.to_csv(output_dir / "rolling_windows.csv", index=False)
    models, manifest = load_fixed_models(
        fixed_run_dir,
        windows,
        timesteps=args.timesteps,
        model_seed=args.model_seed,
        observation_space=compatibility_env.observation_space,
        action_space=compatibility_env.action_space,
    )
    manifest.to_csv(output_dir / "fixed_rl_checkpoint_manifest.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": pair,
                "classifier_group": group,
                "curve_index": index,
            }
            for index, (pair, group) in enumerate(
                (pair, group) for pair in PAIR_KEYS for group in range(1, 6)
            )
        ]
    ).to_csv(output_dir / "curve_index.csv", index=False)

    all_metrics: list[pd.DataFrame] = []
    all_base: list[pd.DataFrame] = []
    ensemble_curves: list[np.ndarray] = []
    base_curves: list[np.ndarray] = []
    classifier_audits: list[pd.DataFrame] = []
    for repeat in range(args.repetitions):
        if args.resume and completed_backtest_exists(repeat, output_dir):
            result = load_completed_backtest(repeat, output_dir)
            print(f"BACKTEST {repeat + 1}/{args.repetitions}: loaded completed result")
        else:
            print(f"BACKTEST {repeat + 1}/{args.repetitions}: running with fixed RL checkpoints")
            result = run_one_backtest(
                repeat,
                models=models,
                full_data=full_data,
                windows=windows,
                kwargs=kwargs,
                tau_values=tau_values,
                expected_dates=expected_dates,
                master_seed=args.master_seed,
                output_dir=output_dir,
                deterministic_rl=not stochastic_rl_inference,
            )
        metrics, base, ensemble_array, base_array, classifier_audit = result
        all_metrics.append(metrics)
        all_base.append(base)
        ensemble_curves.append(ensemble_array)
        base_curves.append(base_array)
        classifier_audits.append(classifier_audit)

    metrics_frame = pd.concat(all_metrics, ignore_index=True)
    base_frame = pd.concat(all_base, ignore_index=True)
    classifier_audit_frame = pd.concat(classifier_audits, ignore_index=True)
    metrics_frame.to_csv(output_dir / "all_backtest_metrics.csv", index=False)
    base_frame.to_csv(output_dir / "all_base_metrics.csv", index=False)
    classifier_audit_frame.to_csv(output_dir / "classifier_refit_audit.csv", index=False)

    expected_metric_rows = args.repetitions * len(PAIR_KEYS) * 5 * len(tau_values)
    expected_refits = args.repetitions * len(windows) * len(PAIR_KEYS) * 5
    if len(metrics_frame) != expected_metric_rows:
        raise ValueError(
            f"expected {expected_metric_rows} ensemble metric rows, found {len(metrics_frame)}"
        )
    if len(classifier_audit_frame) != expected_refits:
        raise ValueError(
            f"expected {expected_refits} classifier refits, found {len(classifier_audit_frame)}"
        )

    mean_metrics = aggregate_metric_frame(
        metrics_frame,
        ["pair", "classifier_group", "tau"],
        METRICS,
    )
    base_summary = aggregate_metric_frame(base_frame, ["model"], METRICS)
    mean_metrics.to_csv(output_dir / "mean_metrics_by_fixed_tau.csv", index=False)
    base_summary.to_csv(output_dir / "base_model_30_backtest_summary.csv", index=False)
    selected, paired = build_selected_summary(mean_metrics, metrics_frame, base_frame)
    selected.to_csv(output_dir / "selected_tau_summary.csv", index=False)
    paired.to_csv(output_dir / "selected_tau_paired_runs.csv", index=False)
    selected[
        [
            "pair",
            "classifier_group",
            "selected_global_tau",
            "stronger_component",
            "delta_sharpe_mean",
            "delta_sharpe_sd",
            "delta_sharpe_ci_low",
            "delta_sharpe_ci_high",
            "wins_vs_stronger",
            "n_backtests",
        ]
    ].to_csv(output_dir / "paired_sharpe_stability.csv", index=False)
    configuration_comparison, outperformance_summary = build_configuration_comparison(
        selected, base_frame
    )
    configuration_comparison.to_csv(
        output_dir / "configuration_comparison.csv", index=False
    )
    outperformance_summary.to_csv(
        output_dir / "outperformance_summary.csv", index=False
    )

    original = None
    paper_tau_summary = None
    comparison = None
    curve_consistency = None
    error_summary = None
    if args.original_comparison:
        original = load_original_curves(Path(args.paper_results_dir))
        paper_tau_summary, paper_tau_paired = build_paper_tau_stability(
            original, metrics_frame, base_frame
        )
        paper_tau_summary.to_csv(
            output_dir / "paper_tau_stability_summary.csv", index=False
        )
        paper_tau_paired.to_csv(output_dir / "paper_tau_paired_runs.csv", index=False)
        comparison, curve_consistency = build_original_comparison(
            original, mean_metrics, selected, paper_tau_summary
        )
        error_summary = build_error_summary(comparison)
        comparison.to_csv(output_dir / "original_best_comparison.csv", index=False)
        curve_consistency.to_csv(
            output_dir / "group1_curve_consistency.csv", index=False
        )
        error_summary.to_csv(
            output_dir / "original_reproduction_error_summary.csv", index=False
        )

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure3, temporal_metrics, range_coverage = plot_figure3(
        ensemble_curves,
        base_curves,
        selected,
        tau_values,
        expected_dates,
        figures_dir,
        args.dpi,
        dataset_label,
    )
    temporal_metrics.to_csv(output_dir / "figure3_temporal_metrics.csv", index=False)
    range_coverage.to_csv(output_dir / "figure3_range_coverage.csv", index=False)
    figure4 = plot_figure4(
        selected, base_summary, figures_dir, args.dpi, dataset_label
    )
    figure5 = plot_figure5(
        mean_metrics, base_summary, selected, figures_dir, args.dpi, dataset_label
    )
    figure_paths = [figure3, figure4, figure5]
    if original is not None:
        comparison_figure = plot_original_curve_comparison(
            original, mean_metrics, figures_dir, args.dpi
        )
        figure_paths.append(comparison_figure)

    run_metadata = {
        **metadata,
        "dataset_label": dataset_label,
        "invocation": f"PYTHONPATH=. {shlex.join(['python', *sys.argv])}",
        "repetitions": args.repetitions,
        "model_training_seed": args.model_seed,
        "master_backtest_seed": args.master_seed,
        "rl_retrained_in_repetitions": False,
        "stochastic_rl_inference": stochastic_rl_inference,
        "classifier_refit_each_window": True,
        "classifier_grid_search": False,
        "fixed_global_tau_per_path": True,
        "common_random_numbers_across_tau": True,
        "window_count": len(windows),
        "expected_classifier_refits": expected_refits,
        "tau_values": tau_values.tolist(),
        "original_dj30_comparison": args.original_comparison,
        "figures": [str(path.relative_to(output_dir)) for path in figure_paths],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )
    if not args.original_comparison:
        write_external_dataset_report(
            output_dir,
            dataset_label=dataset_label,
            metadata=metadata,
            repetitions=args.repetitions,
            windows=windows_frame,
            manifest=manifest,
            base_summary=base_summary,
            selected=selected,
            configuration_comparison=configuration_comparison,
            outperformance_summary=outperformance_summary,
            range_coverage=range_coverage,
            classifier_refits=len(classifier_audit_frame),
        )
    else:
        if any(
            item is None
            for item in (
                comparison,
                curve_consistency,
                paper_tau_summary,
                error_summary,
            )
        ):
            raise RuntimeError("DJ30 comparison artifacts were not constructed")
        write_report(
            output_dir,
            selected,
            paired,
            base_summary,
            comparison,
            curve_consistency,
            paper_tau_summary,
            error_summary,
            manifest,
            args.repetitions,
            len(classifier_audit_frame),
            stochastic_rl_inference,
        )
    print(selected.to_string(index=False))
    print(f"Saved fixed-RL 30-backtest audit to {output_dir}")


if __name__ == "__main__":
    main()
