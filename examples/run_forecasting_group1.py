from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/finrl-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import brier_score_loss

from finrl.reproduction.classifier_ensemble import confidence_matrix
from finrl.reproduction.classifier_ensemble import holding_dispersion
from finrl.reproduction.classifier_ensemble import select_holding_from_confidence
from finrl.reproduction.classifier_ensemble import tau_grid
from finrl.reproduction.classifier_ensemble import train_classifier_group
from finrl.reproduction.forecasting_ensemble import ForecastConfig
from finrl.reproduction.forecasting_ensemble import fit_predict
from finrl.reproduction.forecasting_ensemble import forecasts_to_weights
from finrl.reproduction.forecasting_ensemble import log_return_matrix
from finrl.reproduction.forecasting_ensemble import ordered_asset_labels
from finrl.reproduction.forecasting_ensemble import price_matrix
from finrl.reproduction.forecasting_ensemble import simulate_selected_holdings
from finrl.reproduction.forecasting_ensemble import simulate_weight_strategy
from finrl.reproduction.metrics import metrics_from_account_values
from reproduce_classifier_ensemble import append_account_curve
from reproduce_classifier_ensemble import build_rolling_windows
from reproduce_classifier_ensemble import env_kwargs
from reproduce_classifier_ensemble import frame_for_dates
from reproduce_classifier_ensemble import load_trademaster_rolling_data


METRICS = ("cumulative_return", "sharpe", "calmar", "max_drawdown")


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    report_title: str
    report_filename: str
    model_description: str
    model_names: tuple[str, ...]
    pairs: tuple[tuple[str, str, str], ...]

    @property
    def pair_components(self) -> dict[str, tuple[str, str]]:
        return {name: (left, right) for name, left, right in self.pairs}


EXPERIMENT_SPECS = {
    "group1": ExperimentSpec(
        key="group1",
        report_title="Representative Forecasting Model Ensemble Experiment",
        report_filename="FORECASTING_GROUP1_REPORT.md",
        model_description=(
            "ARIMA(1,0,1) with innovations MLE, XGBRegressor, and a two-layer LSTM"
        ),
        model_names=("arima", "xgboost", "lstm"),
        pairs=(
            ("arima_xgboost", "arima", "xgboost"),
            ("arima_lstm", "arima", "lstm"),
            ("xgboost_lstm", "xgboost", "lstm"),
        ),
    ),
    "group2": ExperimentSpec(
        key="group2",
        report_title="Modern Deep Forecasting Model Ensemble Experiment",
        report_filename="FORECASTING_GROUP2_REPORT.md",
        model_description=(
            "one-step supervised adaptations of PatchTST with channel-independent "
            "patch tokens and iTransformer with variate tokens"
        ),
        model_names=("patchtst", "itransformer"),
        pairs=(("patchtst_itransformer", "patchtst", "itransformer"),),
    ),
}


def parse_args(default_experiment_group: str = "group1") -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a forecasting-model ensemble experiment with expanding model fits, "
            "rolling classifier refits, and fixed global tau."
        )
    )
    parser.add_argument(
        "--experiment-group",
        choices=sorted(EXPERIMENT_SPECS),
        default=default_experiment_group,
    )
    parser.add_argument("--data-dir", default="external_data/trademaster_dj30")
    parser.add_argument("--trade-split", choices=["valid", "test"], default="valid")
    parser.add_argument("--dataset-label", default="DJ30")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=250217518)
    parser.add_argument("--rebalance-window", type=int, default=63)
    parser.add_argument("--validation-window", type=int, default=63)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--tau-start", type=float, default=0.01)
    parser.add_argument("--tau-stop", type=float, default=0.89)
    parser.add_argument("--tau-step", type=float, default=0.01)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--risk-window", type=int, default=20)
    parser.add_argument("--max-weight", type=float, default=0.20)
    parser.add_argument("--gross-exposure", type=float, default=0.95)
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument("--xgb-estimators", type=int, default=300)
    parser.add_argument("--lstm-epochs", type=int, default=50)
    parser.add_argument("--transformer-epochs", type=int, default=50)
    parser.add_argument("--force-forecasts", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one window, two refits, a short tau grid, and lightweight learners.",
    )
    return parser.parse_args()


def stable_seed(master_seed: int, *parts: object) -> int:
    payload = "|".join([str(master_seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "little") % (
        2**31 - 1
    )


def config_from_args(args: argparse.Namespace) -> ForecastConfig:
    return ForecastConfig(
        lookback=args.lookback,
        xgb_estimators=args.xgb_estimators,
        lstm_epochs=args.lstm_epochs,
        transformer_epochs=args.transformer_epochs,
        risk_window=args.risk_window,
        max_weight=args.max_weight,
        gross_exposure=args.gross_exposure,
        softmax_temperature=args.softmax_temperature,
    )


def manifest_forecast_config(
    config: ForecastConfig,
    spec: ExperimentSpec,
) -> dict[str, object]:
    values = asdict(config)
    common = {
        "lookback",
        "validation_fraction",
        "risk_window",
        "softmax_temperature",
        "max_weight",
        "gross_exposure",
    }
    group_fields = {
        "group1": {
            "arima_order",
            "arima_method",
            "xgb_estimators",
            "xgb_max_depth",
            "xgb_learning_rate",
            "lstm_hidden_size",
            "lstm_layers",
            "lstm_dropout",
            "lstm_epochs",
            "lstm_batch_size",
            "lstm_learning_rate",
            "lstm_patience",
        },
        "group2": {
            "transformer_d_model",
            "transformer_heads",
            "transformer_layers",
            "transformer_ffn",
            "transformer_dropout",
            "transformer_epochs",
            "transformer_batch_size",
            "transformer_learning_rate",
            "transformer_weight_decay",
            "transformer_patience",
            "patch_length",
            "patch_stride",
        },
    }
    selected = common | group_fields[spec.key]
    return {name: values[name] for name in values if name in selected}


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.max_windows = 1
    args.repetitions = min(args.repetitions, 2)
    args.tau_start = 0.20
    args.tau_stop = 0.60
    args.tau_step = 0.20
    args.xgb_estimators = min(args.xgb_estimators, 20)
    args.lstm_epochs = min(args.lstm_epochs, 3)
    args.transformer_epochs = min(args.transformer_epochs, 3)


def mean_sd_ci(values: pd.Series | np.ndarray) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    if len(array) <= 1 or np.isclose(sd, 0.0):
        return mean, sd, mean, mean
    half_width = float(stats.t.ppf(0.975, len(array) - 1) * sd / np.sqrt(len(array)))
    return mean, sd, mean - half_width, mean + half_width


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


def rolling_window_summary(windows: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for window in windows:
        rows.append(
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
        )
    return pd.DataFrame(rows)


def decision_modes(
    classifiers: list[tuple[str, object]],
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dispersions = []
    aggressive = []
    conservative = []
    for day_candidates in candidates:
        q_matrix = confidence_matrix(classifiers, day_candidates, [0, 1])
        dispersion = holding_dispersion(day_candidates)
        high_confidence = select_holding_from_confidence(
            day_candidates, q_matrix, tau=1.0, dispersion=0.0
        )
        low_confidence = select_holding_from_confidence(
            day_candidates, q_matrix, tau=0.0, dispersion=1.0
        )
        dispersions.append(dispersion)
        aggressive.append(high_confidence.selected_index)
        conservative.append(low_confidence.selected_index)
    return (
        np.asarray(dispersions, dtype=float),
        np.asarray(aggressive, dtype=np.int8),
        np.asarray(conservative, dtype=np.int8),
    )


def classifier_diagnostics(
    classifiers: list[tuple[str, object]],
    left_holdings: np.ndarray,
    right_holdings: np.ndarray,
) -> list[dict[str, object]]:
    features = np.vstack([left_holdings, right_holdings])
    labels = np.concatenate(
        [np.zeros(len(left_holdings), dtype=int), np.ones(len(right_holdings), dtype=int)]
    )
    rows = []
    for name, estimator in classifiers:
        predicted = estimator.predict(features)
        probabilities = estimator.predict_proba(features)
        classes = np.asarray(estimator.classes_)
        class_one = np.flatnonzero(classes == 1)
        brier = (
            brier_score_loss(labels, probabilities[:, class_one[0]])
            if class_one.size
            else np.nan
        )
        rows.append(
            {
                "classifier": name,
                "in_sample_balanced_accuracy": balanced_accuracy_score(labels, predicted),
                "in_sample_brier": float(brier),
            }
        )
    return rows


def forecast_error_rows(
    model_name: str,
    window: int,
    predictions: np.ndarray,
    evaluation_returns: np.ndarray,
    calibration_count: int,
    trade_count: int,
) -> list[dict[str, object]]:
    realized = evaluation_returns[1:]
    slices = {
        "calibration": slice(0, calibration_count - 1),
        "trade": slice(calibration_count, calibration_count + trade_count - 1),
    }
    rows = []
    for segment, segment_slice in slices.items():
        predicted = predictions[segment_slice]
        actual = realized[segment_slice]
        error = predicted - actual
        rows.append(
            {
                "window": window,
                "model": model_name,
                "segment": segment,
                "observations": int(error.size),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "directional_accuracy": float(np.mean(np.sign(predicted) == np.sign(actual))),
            }
        )
    return rows


def load_or_fit_forecasts(
    model_name: str,
    window: int,
    train_returns: np.ndarray,
    evaluation_returns: np.ndarray,
    config: ForecastConfig,
    *,
    seed: int,
    cache_dir: Path,
    force: bool,
) -> tuple[np.ndarray, np.ndarray]:
    cache_path = cache_dir / f"window_{window:02d}_{model_name}.npz"
    if cache_path.exists() and not force:
        with np.load(cache_path) as arrays:
            return arrays["forecasts"], arrays["weights"]
    print(f"FORECAST window={window} model={model_name}: fitting")
    forecasts = fit_predict(
        model_name,
        train_returns,
        evaluation_returns,
        config,
        seed=seed,
    )
    weights = forecasts_to_weights(
        forecasts,
        train_returns,
        evaluation_returns,
        risk_window=config.risk_window,
        temperature=config.softmax_temperature,
        max_weight=config.max_weight,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, forecasts=forecasts, weights=weights)
    return forecasts, weights


def prepare_candidate_paths(
    full_data: pd.DataFrame,
    windows: list[dict[str, object]],
    options: dict[str, object],
    config: ForecastConfig,
    *,
    model_names: tuple[str, ...],
    pair_components: dict[str, tuple[str, str]],
    seed: int,
    output_dir: Path,
    force_forecasts: bool,
) -> tuple[
    list[dict[str, object]],
    pd.DataFrame,
    dict[str, np.ndarray],
    pd.DataFrame,
    pd.DataFrame,
]:
    prices = price_matrix(full_data)
    returns = log_return_matrix(prices)
    expected_assets = list(map(str, prices.columns))
    actual_assets = ordered_asset_labels(full_data)
    if expected_assets != actual_assets:
        raise ValueError("forecast and trading asset orders do not match")

    cache_dir = output_dir / "forecast_cache"
    base_curves = {model: pd.DataFrame() for model in model_names}
    base_last_states: dict[str, list[float] | None] = {
        model: None for model in model_names
    }
    records: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    dispersion_rows: list[dict[str, object]] = []

    for window_info in windows:
        window = int(window_info["window"])
        train_dates = [date for date in map(str, window_info["train_dates"]) if date in returns.index]
        calibration_dates = list(map(str, window_info["calibration_dates"]))
        trade_dates = list(map(str, window_info["trade_dates"]))
        evaluation_dates = calibration_dates + trade_dates
        train_returns = returns.loc[train_dates].to_numpy(dtype=float)
        evaluation_returns = returns.loc[evaluation_dates].to_numpy(dtype=float)
        calibration = frame_for_dates(full_data, calibration_dates)
        trade = frame_for_dates(full_data, trade_dates)
        calibration_count = len(calibration_dates)
        trade_count = len(trade_dates)
        calibration_holdings: dict[str, np.ndarray] = {}
        trade_holdings: dict[str, np.ndarray] = {}

        for model_name in model_names:
            model_seed = stable_seed(seed, "forecast", window, model_name)
            forecasts, weights = load_or_fit_forecasts(
                model_name,
                window,
                train_returns,
                evaluation_returns,
                config,
                seed=model_seed,
                cache_dir=cache_dir,
                force=force_forecasts,
            )
            error_rows.extend(
                forecast_error_rows(
                    model_name,
                    window,
                    forecasts,
                    evaluation_returns,
                    calibration_count,
                    trade_count,
                )
            )
            calibration_weights = weights[: calibration_count - 1]
            trade_weights = weights[
                calibration_count : calibration_count + trade_count - 1
            ]
            calibration_holdings[model_name], _account, _state = simulate_weight_strategy(
                calibration_weights,
                calibration,
                options,
                gross_exposure=config.gross_exposure,
            )
            (
                trade_holdings[model_name],
                base_account,
                base_last_states[model_name],
            ) = simulate_weight_strategy(
                trade_weights,
                trade,
                options,
                gross_exposure=config.gross_exposure,
                initial=base_last_states[model_name] is None,
                previous_state=base_last_states[model_name],
            )
            base_curves[model_name] = append_account_curve(
                base_curves[model_name], base_account
            )

        for pair, (left, right) in pair_components.items():
            calibration_candidates = np.stack(
                [calibration_holdings[left], calibration_holdings[right]], axis=1
            )
            trade_candidates = np.stack(
                [trade_holdings[left], trade_holdings[right]], axis=1
            )
            for segment, candidates in (
                ("calibration", calibration_candidates),
                ("trade", trade_candidates),
            ):
                dispersion_rows.append(
                    {
                        "window": window,
                        "pair": pair,
                        "segment": segment,
                        "mean_holding_l1": float(
                            np.abs(candidates[:, 0] - candidates[:, 1]).sum(axis=1).mean()
                        ),
                        "mean_dispersion": float(
                            np.mean([holding_dispersion(day) for day in candidates])
                        ),
                        "identical_holding_rate": float(
                            np.mean(np.all(np.isclose(candidates[:, 0], candidates[:, 1]), axis=1))
                        ),
                    }
                )

        records.append(
            {
                "window": window,
                "calibration": calibration,
                "trade": trade,
                "calibration_holdings": calibration_holdings,
                "trade_holdings": trade_holdings,
                "window_info": window_info,
            }
        )

    expected_dates = np.concatenate(
        [np.asarray(record["window_info"]["trade_dates"], dtype=str) for record in records]
    )
    base_values = {
        model: align_account_curve(curve, expected_dates) for model, curve in base_curves.items()
    }
    base_rows = [
        {"model": model, **metrics_from_account_values(values)}
        for model, values in base_values.items()
    ]
    return (
        records,
        pd.DataFrame(base_rows).sort_values("model"),
        base_values,
        pd.DataFrame(error_rows),
        pd.DataFrame(dispersion_rows),
    )


def run_repeat(
    repeat: int,
    *,
    records: list[dict[str, object]],
    options: dict[str, object],
    tau_values: np.ndarray,
    expected_dates: np.ndarray,
    master_seed: int,
    output_dir: Path,
    pair_components: dict[str, tuple[str, str]],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
    config_keys = [
        (pair, group) for pair in pair_components for group in range(1, 6)
    ]
    decision_inputs: dict[
        tuple[str, int], list[tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ] = {key: [] for key in config_keys}
    classifier_rows = []
    diagnostic_rows = []

    for record in records:
        window = int(record["window"])
        calibration_holdings = record["calibration_holdings"]
        trade_holdings = record["trade_holdings"]
        window_info = record["window_info"]
        for pair, (left, right) in pair_components.items():
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
                    (
                        record["trade"],
                        candidates,
                        dispersions,
                        aggressive,
                        conservative,
                    )
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
                for row in classifier_diagnostics(
                    classifiers,
                    calibration_holdings[left],
                    calibration_holdings[right],
                ):
                    diagnostic_rows.append(
                        {
                            "repeat": repeat,
                            "window": window,
                            "pair": pair,
                            "classifier_group": group,
                            **row,
                        }
                    )

    curves = np.empty(
        (len(config_keys), len(tau_values), len(expected_dates)), dtype=float
    )
    metric_rows = []
    for config_index, (pair, group) in enumerate(config_keys):
        inputs = decision_inputs[(pair, group)]
        signatures: dict[bytes, list[int]] = {}
        selected_by_tau: list[list[np.ndarray]] = []
        for tau_index, tau in enumerate(tau_values):
            selected_windows = []
            signature_parts = []
            for _trade, _candidates, dispersions, aggressive, conservative in inputs:
                selected = np.where(dispersions < float(tau), aggressive, conservative)
                selected_windows.append(selected)
                signature_parts.append(selected.tobytes())
            signatures.setdefault(b"|".join(signature_parts), []).append(tau_index)
            selected_by_tau.append(selected_windows)

        for tau_indices in signatures.values():
            representative = tau_indices[0]
            curve = pd.DataFrame()
            last_state: list[float] | None = None
            for input_index, (trade, candidates, _d, _a, _c) in enumerate(inputs):
                selected = selected_by_tau[representative][input_index]
                targets = candidates[np.arange(len(selected)), selected]
                account, last_state = simulate_selected_holdings(
                    targets,
                    trade,
                    options,
                    initial=last_state is None,
                    previous_state=last_state,
                )
                curve = append_account_curve(curve, account)
            values = align_account_curve(curve, expected_dates)
            metrics = metrics_from_account_values(values)
            for tau_index in tau_indices:
                curves[config_index, tau_index] = values
                metric_rows.append(
                    {
                        "repeat": repeat,
                        "pair": pair,
                        "classifier_group": group,
                        "tau": float(tau_values[tau_index]),
                        "equivalent_tau_paths": len(tau_indices),
                        **metrics,
                    }
                )

    run_dir = output_dir / "runs" / f"repeat_{repeat:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(metric_rows).sort_values(
        ["pair", "classifier_group", "tau"]
    )
    classifier_frame = pd.DataFrame(classifier_rows)
    diagnostics_frame = pd.DataFrame(diagnostic_rows)
    metrics_frame.to_csv(run_dir / "ensemble_metrics.csv", index=False)
    classifier_frame.to_csv(run_dir / "classifier_refit_audit.csv", index=False)
    diagnostics_frame.to_csv(run_dir / "classifier_diagnostics.csv", index=False)
    np.savez_compressed(run_dir / "account_curves.npz", ensemble=curves, dates=expected_dates)
    return metrics_frame, curves, classifier_frame, diagnostics_frame


def simple_average_baselines(
    records: list[dict[str, object]],
    options: dict[str, object],
    expected_dates: np.ndarray,
    pair_components: dict[str, tuple[str, str]],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    values_by_pair = {}
    for pair, (left, right) in pair_components.items():
        curve = pd.DataFrame()
        last_state: list[float] | None = None
        for record in records:
            targets = 0.5 * (
                record["trade_holdings"][left] + record["trade_holdings"][right]
            )
            account, last_state = simulate_selected_holdings(
                targets,
                record["trade"],
                options,
                initial=last_state is None,
                previous_state=last_state,
            )
            curve = append_account_curve(curve, account)
        values = align_account_curve(curve, expected_dates)
        values_by_pair[pair] = values
        rows.append(
            {
                "pair": pair,
                "strategy": "simple_holding_average",
                **metrics_from_account_values(values),
            }
        )
    return pd.DataFrame(rows).sort_values("pair"), values_by_pair


def completed_repeat_exists(repeat: int, output_dir: Path) -> bool:
    run_dir = output_dir / "runs" / f"repeat_{repeat:02d}"
    return all(
        (run_dir / name).exists()
        for name in (
            "ensemble_metrics.csv",
            "classifier_refit_audit.csv",
            "classifier_diagnostics.csv",
            "account_curves.npz",
        )
    )


def load_repeat(
    repeat: int, output_dir: Path
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
    run_dir = output_dir / "runs" / f"repeat_{repeat:02d}"
    metrics = pd.read_csv(run_dir / "ensemble_metrics.csv")
    classifier = pd.read_csv(run_dir / "classifier_refit_audit.csv")
    diagnostics = pd.read_csv(run_dir / "classifier_diagnostics.csv")
    with np.load(run_dir / "account_curves.npz") as arrays:
        curves = arrays["ensemble"]
    return metrics, curves, classifier, diagnostics


def aggregate_metrics(all_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in all_metrics.groupby(["pair", "classifier_group", "tau"]):
        pair, classifier_group, tau = key
        row: dict[str, object] = {
            "pair": pair,
            "classifier_group": classifier_group,
            "tau": tau,
            "n_backtests": len(group),
        }
        for metric in METRICS:
            mean, sd, low, high = mean_sd_ci(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["pair", "classifier_group", "tau"])


def select_tau_and_compare(
    mean_metrics: pd.DataFrame,
    all_metrics: pd.DataFrame,
    base_metrics: pd.DataFrame,
    average_metrics: pd.DataFrame,
    pair_components: dict[str, tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = (
        mean_metrics.sort_values(
            ["pair", "classifier_group", "sharpe_mean", "tau"],
            ascending=[True, True, False, True],
        )
        .groupby(["pair", "classifier_group"], as_index=False)
        .head(1)
        .sort_values(["pair", "classifier_group"])
    )
    base_lookup = base_metrics.set_index("model")
    average_lookup = average_metrics.set_index("pair")
    summary_rows = []
    paired_rows = []
    for row in selected.itertuples(index=False):
        left, right = pair_components[row.pair]
        stronger = max((left, right), key=lambda model: float(base_lookup.loc[model, "sharpe"]))
        stronger_sharpe = float(base_lookup.loc[stronger, "sharpe"])
        average_sharpe = float(average_lookup.loc[row.pair, "sharpe"])
        runs = all_metrics[
            (all_metrics["pair"] == row.pair)
            & (all_metrics["classifier_group"] == row.classifier_group)
            & np.isclose(all_metrics["tau"], row.tau)
        ].copy()
        runs["stronger_model"] = stronger
        runs["stronger_sharpe"] = stronger_sharpe
        runs["delta_sharpe"] = runs["sharpe"] - stronger_sharpe
        delta_mean, delta_sd, delta_low, delta_high = mean_sd_ci(runs["delta_sharpe"])
        summary_rows.append(
            {
                "pair": row.pair,
                "classifier_group": int(row.classifier_group),
                "selected_global_tau": float(row.tau),
                "ensemble_cumulative_return_mean": float(row.cumulative_return_mean),
                "ensemble_sharpe_mean": float(row.sharpe_mean),
                "ensemble_sharpe_sd": float(row.sharpe_sd),
                "ensemble_calmar_mean": float(row.calmar_mean),
                "ensemble_max_drawdown_mean": float(row.max_drawdown_mean),
                "stronger_model": stronger,
                "stronger_sharpe": stronger_sharpe,
                "simple_average_sharpe": average_sharpe,
                "delta_sharpe_vs_average": float(row.sharpe_mean) - average_sharpe,
                "delta_sharpe_mean": delta_mean,
                "delta_sharpe_sd": delta_sd,
                "delta_sharpe_ci_low": delta_low,
                "delta_sharpe_ci_high": delta_high,
                "win_rate_vs_stronger": float(np.mean(runs["delta_sharpe"] > 0.0)),
            }
        )
        paired_rows.append(runs)
    return pd.DataFrame(summary_rows), pd.concat(paired_rows, ignore_index=True)


def summarize_tau_robustness(
    mean_metrics: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for result in selected.itertuples(index=False):
        candidates = mean_metrics[
            (mean_metrics["pair"] == result.pair)
            & (mean_metrics["classifier_group"] == result.classifier_group)
        ]
        better_than_stronger = candidates[
            candidates["sharpe_mean"] > result.stronger_sharpe
        ]
        better_than_average = candidates[
            candidates["sharpe_mean"] > result.simple_average_sharpe
        ]
        rows.append(
            {
                "pair": result.pair,
                "classifier_group": int(result.classifier_group),
                "selected_global_tau": float(result.selected_global_tau),
                "tau_count": len(candidates),
                "tau_beating_stronger": len(better_than_stronger),
                "tau_beating_simple_average": len(better_than_average),
                "tau_beating_stronger_min": (
                    float(better_than_stronger["tau"].min())
                    if not better_than_stronger.empty
                    else np.nan
                ),
                "tau_beating_stronger_max": (
                    float(better_than_stronger["tau"].max())
                    if not better_than_stronger.empty
                    else np.nan
                ),
                "tau_within_0.01_sharpe_of_best": int(
                    (
                        candidates["sharpe_mean"]
                        >= result.ensemble_sharpe_mean - 0.01
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["pair", "classifier_group"])


def summarize_common_tau(
    mean_metrics: pd.DataFrame,
    base_metrics: pd.DataFrame,
    average_metrics: pd.DataFrame,
    pair_components: dict[str, tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_lookup = base_metrics.set_index("model")
    average_lookup = average_metrics.set_index("pair")
    rows = []
    for pair, (left, right) in pair_components.items():
        stronger_sharpe = max(
            float(base_lookup.loc[left, "sharpe"]),
            float(base_lookup.loc[right, "sharpe"]),
        )
        average_sharpe = float(average_lookup.loc[pair, "sharpe"])
        pair_metrics = mean_metrics[mean_metrics["pair"] == pair]
        for tau, candidates in pair_metrics.groupby("tau"):
            sharpes = candidates["sharpe_mean"]
            rows.append(
                {
                    "pair": pair,
                    "tau": float(tau),
                    "classifier_groups": len(candidates),
                    "sharpe_across_groups_mean": float(sharpes.mean()),
                    "sharpe_across_groups_min": float(sharpes.min()),
                    "sharpe_across_groups_max": float(sharpes.max()),
                    "min_delta_vs_stronger": float(sharpes.min() - stronger_sharpe),
                    "min_delta_vs_simple_average": float(
                        sharpes.min() - average_sharpe
                    ),
                    "groups_beating_stronger": int(
                        (sharpes > stronger_sharpe).sum()
                    ),
                    "groups_beating_simple_average": int(
                        (sharpes > average_sharpe).sum()
                    ),
                }
            )
    summary = pd.DataFrame(rows).sort_values(["pair", "tau"])
    selected = (
        summary.sort_values(
            [
                "pair",
                "min_delta_vs_simple_average",
                "sharpe_across_groups_mean",
                "tau",
            ],
            ascending=[True, False, False, True],
        )
        .groupby("pair", as_index=False)
        .head(1)
        .sort_values("pair")
    )
    return summary, selected


def markdown_table(frame: pd.DataFrame, columns: list[str], rows: int = 20) -> str:
    view = frame.loc[:, columns].head(rows).copy()
    for column in view.select_dtypes(include=[float]).columns:
        view[column] = view[column].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(view.columns) + " |"
    separator = "|" + "|".join(["---"] * len(view.columns)) + "|"
    body = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])


def write_report(
    output_dir: Path,
    *,
    spec: ExperimentSpec,
    args: argparse.Namespace,
    config: ForecastConfig,
    metadata: dict[str, object],
    windows_frame: pd.DataFrame,
    base_metrics: pd.DataFrame,
    average_metrics: pd.DataFrame,
    selected: pd.DataFrame,
    tau_robustness: pd.DataFrame,
    selected_common_tau: pd.DataFrame,
    forecast_summary: pd.DataFrame,
) -> None:
    wins = int((selected["delta_sharpe_mean"] > 0.0).sum())
    positive_intervals = int((selected["delta_sharpe_ci_low"] > 0.0).sum())
    average_wins = int((selected["delta_sharpe_vs_average"] > 0.0).sum())
    configuration_count = len(selected)
    trade_days = int(windows_frame["trade_dates"].sum())
    window_lengths = ", ".join(map(str, windows_frame["trade_dates"].tolist()))
    pair_summary = (
        selected.assign(
            beats_stronger=selected["delta_sharpe_mean"] > 0.0,
            positive_ci=selected["delta_sharpe_ci_low"] > 0.0,
            beats_simple_average=selected["delta_sharpe_vs_average"] > 0.0,
        )
        .groupby("pair", as_index=False)[
            ["beats_stronger", "positive_ci", "beats_simple_average"]
        ]
        .sum()
    )
    report = [
        f"# {spec.report_title}",
        "",
        "## Protocol",
        "",
        f"- Dataset: {args.dataset_label}; source `{args.data_dir}`; trade split `{args.trade_split}`; {metadata['stock_count']} assets are present in the supplied files.",
        f"- Evaluation span: {windows_frame.iloc[0]['trade_start']} to {windows_frame.iloc[-1]['trade_end']}, {trade_days} sessions in {len(windows_frame)} rolling blocks ({window_lengths} sessions).",
        f"- Base models: {spec.model_description}.",
        "- All models predict the next-session cross-section of log returns from close-price history only.",
        f"- Common portfolio map: trailing-{config.risk_window}-session volatility scaling, cross-sectional softmax, maximum weight {config.max_weight:.2f}, gross exposure {config.gross_exposure:.2f}.",
        "- Forecast-model fitting uses an expanding historical window. Classifiers use only the immediately preceding rolling calibration block.",
        f"- Every tau is fixed for a complete path; grid {args.tau_start:.2f}-{args.tau_stop:.2f} by {args.tau_step:.2f}.",
        f"- Repetitions: {args.repetitions}; candidate forecasts and holdings remain fixed while rolling classifiers are refitted.",
        "- Classifier groups and voting are unchanged from the RL experiment; no classifier grid search is used.",
        "",
        "## Single Models",
        "",
        markdown_table(
            base_metrics,
            ["model", "cumulative_return", "sharpe", "calmar", "max_drawdown"],
        ),
        "",
        "## Simple Average Controls",
        "",
        markdown_table(
            average_metrics,
            ["pair", "cumulative_return", "sharpe", "calmar", "max_drawdown"],
        ),
        "",
        "## Selected Global Tau",
        "",
        markdown_table(
            selected,
            [
                "pair",
                "classifier_group",
                "selected_global_tau",
                "ensemble_cumulative_return_mean",
                "ensemble_sharpe_mean",
                "stronger_model",
                "stronger_sharpe",
                "simple_average_sharpe",
                "delta_sharpe_vs_average",
                "delta_sharpe_mean",
                "delta_sharpe_ci_low",
                "delta_sharpe_ci_high",
                "win_rate_vs_stronger",
            ],
        ),
        "",
        "## Pair-Level Outcomes",
        "",
        markdown_table(
            pair_summary,
            ["pair", "beats_stronger", "positive_ci", "beats_simple_average"],
        ),
        "",
        "## Tau Robustness",
        "",
        markdown_table(
            tau_robustness,
            [
                "pair",
                "classifier_group",
                "selected_global_tau",
                "tau_beating_stronger",
                "tau_beating_simple_average",
                "tau_beating_stronger_min",
                "tau_beating_stronger_max",
                "tau_within_0.01_sharpe_of_best",
            ],
        ),
        "",
        "## Common Tau Across Classifier Groups",
        "",
        "The selected common tau maximizes the worst classifier group's Sharpe advantage over the simple holding average.",
        "",
        markdown_table(
            selected_common_tau,
            [
                "pair",
                "tau",
                "sharpe_across_groups_mean",
                "sharpe_across_groups_min",
                "min_delta_vs_stronger",
                "min_delta_vs_simple_average",
                "groups_beating_stronger",
                "groups_beating_simple_average",
            ],
        ),
        "",
        "## Forecast Diagnostics",
        "",
        markdown_table(
            forecast_summary,
            ["model", "segment", "mae", "rmse", "directional_accuracy"],
        ),
        "",
        "## Main Finding",
        "",
        f"At each pair-group configuration's mean-Sharpe-maximizing global tau, the classifier-assisted ensemble exceeds its stronger component in {wins}/{configuration_count} configurations and the simple holding average in {average_wins}/{configuration_count}; {positive_intervals}/{configuration_count} paired 95% intervals versus the stronger component are entirely positive.",
        "",
        "Tau is selected on the completed evaluation span, so this is a sensitivity experiment rather than an out-of-sample tau-selection claim. Confidence intervals are conditional on the selected tau and fixed candidate forecasts and therefore do not correct for tau-selection uncertainty.",
        "",
    ]
    (output_dir / spec.report_filename).write_text(
        "\n".join(report), encoding="utf-8"
    )


def main(default_experiment_group: str = "group1") -> None:
    args = parse_args(default_experiment_group)
    spec = EXPERIMENT_SPECS[args.experiment_group]
    pair_components = spec.pair_components
    apply_smoke_defaults(args)
    output_dir = Path(
        args.output_dir
        or f"results/forecasting_{spec.key}_{args.dataset_label.lower()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config_from_args(args)

    full_data, indicators, trade_start, metadata = load_trademaster_rolling_data(
        args.data_dir, trade_split=args.trade_split
    )
    options = env_kwargs(full_data, indicators)
    windows = build_rolling_windows(
        full_data,
        trade_start_date=trade_start,
        rebalance_window=args.rebalance_window,
        validation_window=args.validation_window,
        max_windows=args.max_windows,
    )
    windows_frame = rolling_window_summary(windows)
    windows_frame.to_csv(output_dir / "rolling_windows.csv", index=False)
    tau_values = tau_grid(args.tau_start, args.tau_stop, args.tau_step)
    if args.max_weight * int(options["stock_dim"]) < 1.0:
        raise ValueError("max_weight is infeasible for this dataset")

    manifest = {
        "dataset": metadata,
        "dataset_label": args.dataset_label,
        "experiment_group": spec.key,
        "data_dir": str(Path(args.data_dir).resolve()),
        "trade_split": args.trade_split,
        "forecast_config": manifest_forecast_config(config, spec),
        "models": list(spec.model_names),
        "pairs": list(pair_components),
        "classifier_groups": [1, 2, 3, 4, 5],
        "repetitions": args.repetitions,
        "seed": args.seed,
        "tau_values": tau_values.tolist(),
        "window_count": len(windows),
        "window_boundaries": windows_frame.to_dict(orient="records"),
        "fixed_global_tau_per_path": True,
        "forecast_models_retrained_by_window": True,
        "classifier_rolling": True,
        "classifier_grid_search": False,
    }
    manifest = json.loads(json.dumps(manifest))
    manifest_path = output_dir / "experiment_manifest.json"
    if manifest_path.exists() and not args.force_forecasts:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable_keys = (
            "data_dir",
            "trade_split",
            "forecast_config",
            "models",
            "pairs",
            "seed",
            "tau_values",
            "window_count",
        )
        if any(existing.get(key) != manifest.get(key) for key in comparable_keys):
            raise ValueError(
                "output directory contains an incompatible experiment; use a new directory"
            )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    (
        records,
        base_metrics,
        base_values,
        forecast_errors,
        candidate_diagnostics,
    ) = prepare_candidate_paths(
        full_data,
        windows,
        options,
        config,
        model_names=spec.model_names,
        pair_components=pair_components,
        seed=args.seed,
        output_dir=output_dir,
        force_forecasts=args.force_forecasts,
    )
    expected_dates = np.concatenate(
        [np.asarray(record["window_info"]["trade_dates"], dtype=str) for record in records]
    )
    average_metrics, average_values = simple_average_baselines(
        records, options, expected_dates, pair_components
    )
    base_metrics.to_csv(output_dir / "base_model_metrics.csv", index=False)
    average_metrics.to_csv(output_dir / "simple_average_metrics.csv", index=False)
    forecast_errors.to_csv(output_dir / "forecast_errors_by_window.csv", index=False)
    candidate_diagnostics.to_csv(output_dir / "candidate_holding_diagnostics.csv", index=False)
    np.savez_compressed(
        output_dir / "base_account_curves.npz",
        dates=expected_dates,
        **base_values,
    )
    np.savez_compressed(
        output_dir / "simple_average_account_curves.npz",
        dates=expected_dates,
        **average_values,
    )

    metric_frames = []
    curve_arrays = []
    classifier_frames = []
    diagnostic_frames = []
    for repeat in range(args.repetitions):
        if args.resume and completed_repeat_exists(repeat, output_dir):
            print(f"BACKTEST {repeat + 1}/{args.repetitions}: loading completed result")
            metrics, curves, classifier, diagnostics = load_repeat(repeat, output_dir)
        else:
            print(f"BACKTEST {repeat + 1}/{args.repetitions}: refitting classifiers")
            metrics, curves, classifier, diagnostics = run_repeat(
                repeat,
                records=records,
                options=options,
                tau_values=tau_values,
                expected_dates=expected_dates,
                master_seed=args.seed,
                output_dir=output_dir,
                pair_components=pair_components,
            )
        metric_frames.append(metrics)
        curve_arrays.append(curves)
        classifier_frames.append(classifier)
        diagnostic_frames.append(diagnostics)

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    mean_metrics = aggregate_metrics(all_metrics)
    selected, paired = select_tau_and_compare(
        mean_metrics,
        all_metrics,
        base_metrics,
        average_metrics,
        pair_components,
    )
    tau_robustness = summarize_tau_robustness(mean_metrics, selected)
    common_tau, selected_common_tau = summarize_common_tau(
        mean_metrics,
        base_metrics,
        average_metrics,
        pair_components,
    )
    classifier_audit = pd.concat(classifier_frames, ignore_index=True)
    classifier_diagnostics_frame = pd.concat(diagnostic_frames, ignore_index=True)
    forecast_summary = (
        forecast_errors.groupby(["model", "segment"], as_index=False)[
            ["mae", "rmse", "directional_accuracy"]
        ]
        .mean()
        .sort_values(["model", "segment"])
    )

    all_metrics.to_csv(output_dir / "all_classifier_refit_metrics.csv", index=False)
    mean_metrics.to_csv(output_dir / "mean_metrics_by_fixed_tau.csv", index=False)
    selected.to_csv(output_dir / "selected_tau_summary.csv", index=False)
    tau_robustness.to_csv(output_dir / "tau_robustness_summary.csv", index=False)
    common_tau.to_csv(output_dir / "common_tau_summary.csv", index=False)
    selected_common_tau.to_csv(
        output_dir / "selected_common_tau_summary.csv", index=False
    )
    paired.to_csv(output_dir / "selected_tau_paired_runs.csv", index=False)
    classifier_audit.to_csv(output_dir / "classifier_refit_audit.csv", index=False)
    classifier_diagnostics_frame.to_csv(
        output_dir / "classifier_diagnostics.csv", index=False
    )
    forecast_summary.to_csv(output_dir / "forecast_error_summary.csv", index=False)
    np.savez_compressed(
        output_dir / "all_ensemble_account_curves.npz",
        ensemble=np.stack(curve_arrays),
        dates=expected_dates,
        pairs=np.asarray(list(pair_components)),
        tau=tau_values,
    )
    write_report(
        output_dir,
        spec=spec,
        args=args,
        config=config,
        metadata=metadata,
        windows_frame=windows_frame,
        base_metrics=base_metrics,
        average_metrics=average_metrics,
        selected=selected,
        tau_robustness=tau_robustness,
        selected_common_tau=selected_common_tau,
        forecast_summary=forecast_summary,
    )
    print(base_metrics.to_string(index=False))
    print(selected.to_string(index=False))
    print(f"Saved forecasting {spec.key} experiment to {output_dir}")


if __name__ == "__main__":
    main()
