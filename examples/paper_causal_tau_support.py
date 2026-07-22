from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from finrl.reproduction.forecasting_ensemble import simulate_selected_holdings
from finrl.reproduction.metrics import metrics_from_account_values
from reproduce_classifier_ensemble import append_account_curve
from reproduce_classifier_ensemble import env_kwargs
from reproduce_classifier_ensemble import frame_for_dates
from reproduce_classifier_ensemble import load_trademaster_rolling_data
from run_forecasting_group1 import align_account_curve


PAIR_COMPONENTS = {
    "a2c_ppo": ("a2c", "ppo"),
    "a2c_sac": ("a2c", "sac"),
    "ppo_sac": ("ppo", "sac"),
}
WARMUP_PERIOD = "2019"
EVALUATION_PERIODS = ("2020",)
ALL_PERIODS = (WARMUP_PERIOD, *EVALUATION_PERIODS)
METRICS = (
    "cumulative_return",
    "annualized_return",
    "sharpe",
    "calmar",
    "max_drawdown",
)


@dataclass(frozen=True)
class PeriodInputs:
    period: str
    records: list[dict[str, object]]
    options: dict[str, object]
    expected_dates: np.ndarray
    master_seed: int
    seed_audit: dict[tuple[int, int, int], int]


@dataclass
class SequenceResult:
    metrics: pd.DataFrame
    curves: dict[str, np.ndarray]
    dates: dict[str, np.ndarray]
    decisions: pd.DataFrame
    classifier_audit: pd.DataFrame
    expert_state_audit: pd.DataFrame


def load_pair_manifest(root: Path, expected_pair: str) -> dict[str, object]:
    manifest_path = root / "experiment_manifest.json"
    audit_path = root / "experiment_validation_audit.json"
    if not manifest_path.exists() or not audit_path.exists():
        raise FileNotFoundError(f"missing validated candidate run under {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True:
        raise ValueError(f"candidate validation audit failed for {root}")
    if list(manifest.get("pair_components", {})) != [expected_pair]:
        raise ValueError(f"{root} does not contain only {expected_pair}")
    required = {
        "deterministic_rl_inference": True,
        "rl_training_window": "expanding",
        "classifier_training_window": "rolling_previous_block",
        "classifier_grid_search": False,
        "fixed_global_tau_per_path": True,
        "repetitions": 30,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"{root} manifest has {key}={manifest.get(key)!r}")
    if manifest.get("classifier_groups") != [1, 2, 3, 4, 5]:
        raise ValueError(f"{root} does not contain all classifier groups")
    return manifest


def selected_validation_model(
    history: pd.DataFrame, *, window: int, pair: str
) -> tuple[str, float, float]:
    left, right = PAIR_COMPONENTS[pair]
    scores: dict[str, float] = {}
    for model in (left, right):
        rows = history[
            (history["window"].astype(int) == window)
            & (history["model"].astype(str) == model)
        ]
        if rows.empty:
            raise ValueError(f"missing validation history for {pair} window {window} {model}")
        scores[model] = float(rows["sharpe"].max())
    selected = max((left, right), key=lambda model: (scores[model], model == left))
    return selected, scores[left], scores[right]


def load_seed_audit(root: Path, pair: str) -> dict[tuple[int, int, int], int]:
    audit = pd.read_csv(root / "classifier_refit_audit.csv")
    if set(audit["pair"].astype(str)) != {pair}:
        raise ValueError(f"unexpected pair in classifier audit under {root}")
    keys = ["repeat", "window", "classifier_group"]
    if audit.duplicated(keys).any():
        raise ValueError(f"duplicate classifier-fit keys under {root}")
    return {
        (int(row.repeat), int(row.window), int(row.classifier_group)): int(
            row.classifier_seed
        )
        for row in audit.itertuples(index=False)
    }


def prepare_records(
    root: Path,
    manifest: dict[str, object],
    pair: str,
) -> tuple[list[dict[str, object]], dict[str, object], np.ndarray]:
    model_names = PAIR_COMPONENTS[pair]
    full_data, indicators, _trade_start, metadata = load_trademaster_rolling_data(
        manifest["data_dir"], trade_split=str(manifest["trade_split"])
    )
    if int(metadata["stock_count"]) != int(manifest["dataset"]["stock_count"]):
        raise ValueError(f"stock dimension differs from the saved run in {root}")
    options = env_kwargs(full_data, indicators)
    windows = pd.read_csv(root / "rolling_windows.csv").sort_values("window")
    history = pd.read_csv(root / "rolling_rl_selection_history.csv")
    records: list[dict[str, object]] = []
    expected_dates: list[np.ndarray] = []
    offset = 0

    for window_row in windows.itertuples(index=False):
        window = int(window_row.window)
        with np.load(
            root / "candidate_holdings" / f"window_{window:02d}.npz"
        ) as arrays:
            calibration_dates = arrays["calibration_dates"].astype(str)
            trade_dates = arrays["trade_dates"].astype(str)
            calibration_holdings = {
                model: arrays[f"{model}_calibration"].astype(float)
                for model in model_names
            }
            trade_holdings = {
                model: arrays[f"{model}_trade"].astype(float)
                for model in model_names
            }
        if len(trade_dates) != int(window_row.trade_dates):
            raise ValueError(f"candidate dates differ from window {window} in {root}")
        expected_shape = (len(trade_dates) - 1, int(options["stock_dim"]))
        for model in model_names:
            calibration_shape = (
                len(calibration_dates) - 1,
                int(options["stock_dim"]),
            )
            if calibration_holdings[model].shape != calibration_shape:
                raise ValueError(f"invalid calibration holdings for {model} window {window}")
            if trade_holdings[model].shape != expected_shape:
                raise ValueError(f"invalid trade holdings for {model} window {window}")
        fallback, left_score, right_score = selected_validation_model(
            history, window=window, pair=pair
        )
        decision_indices = np.arange(offset, offset + len(trade_dates) - 1)
        records.append(
            {
                "window": window,
                "trade": frame_for_dates(full_data, trade_dates),
                "trade_dates": trade_dates,
                "calibration_dates": calibration_dates,
                "calibration_holdings": calibration_holdings,
                "trade_holdings": trade_holdings,
                "fallback_model": fallback,
                "left_validation_sharpe": left_score,
                "right_validation_sharpe": right_score,
                "decision_indices": decision_indices,
            }
        )
        expected_dates.append(trade_dates)
        offset += len(trade_dates)
    return records, options, np.concatenate(expected_dates)


def load_period_inputs(candidate_root: Path, period: str, pair: str) -> PeriodInputs:
    root = candidate_root / period / pair
    manifest = load_pair_manifest(root, pair)
    records, options, expected_dates = prepare_records(root, manifest, pair)
    return PeriodInputs(
        period=period,
        records=records,
        options=options,
        expected_dates=expected_dates,
        master_seed=int(manifest["master_seed"]),
        seed_audit=load_seed_audit(root, pair),
    )


def load_tau_grid(candidate_root: Path, pair: str) -> np.ndarray:
    grids = [
        np.asarray(
            load_pair_manifest(candidate_root / period / pair, pair)["tau_values"],
            dtype=float,
        )
        for period in ALL_PERIODS
    ]
    if any(not np.array_equal(grids[0], grid) for grid in grids[1:]):
        raise ValueError(f"tau grids differ across periods for {pair}")
    return grids[0]


def simulate_causal_single(inputs: PeriodInputs) -> tuple[dict[str, float], str]:
    curve = pd.DataFrame()
    last_state: list[float] | None = None
    selected_models: list[str] = []
    for record in inputs.records:
        model = str(record["fallback_model"])
        account, last_state = simulate_selected_holdings(
            record["trade_holdings"][model],
            record["trade"],
            inputs.options,
            initial=last_state is None,
            previous_state=last_state,
        )
        curve = append_account_curve(curve, account)
        selected_models.append(model)
    values = align_account_curve(curve, inputs.expected_dates)
    return metrics_from_account_values(values), ";".join(selected_models)


def add_paper_comparators(
    metrics: pd.DataFrame, *, candidate_root: Path
) -> pd.DataFrame:
    metrics = metrics.copy()
    metrics["period"] = metrics["period"].astype(str)
    rows: list[pd.DataFrame] = []
    for (period, pair), frame in metrics.groupby(["period", "pair"], sort=False):
        root = candidate_root / str(period) / pair
        base = pd.read_csv(root / "base_model_metrics.csv")
        stronger = base.sort_values(
            ["sharpe", "model"], ascending=[False, True]
        ).iloc[0]
        causal_metrics, sequence = simulate_causal_single(
            load_period_inputs(candidate_root, str(period), pair)
        )
        enriched = frame.copy()
        enriched["stronger_single_model"] = str(stronger["model"])
        enriched["causal_single_model_sequence"] = sequence
        for metric in ("cumulative_return", "sharpe", "calmar", "max_drawdown"):
            enriched[f"stronger_single_{metric}"] = float(stronger[metric])
            enriched[f"causal_single_{metric}"] = float(causal_metrics[metric])
        rows.append(enriched)
    result = pd.concat(rows, ignore_index=True)
    result["delta_sharpe_vs_stronger"] = (
        result["sharpe"] - result["stronger_single_sharpe"]
    )
    result["delta_sharpe_vs_causal_single"] = (
        result["sharpe"] - result["causal_single_sharpe"]
    )
    keys = ["period", "pair", "repeat", "classifier_group"]
    return result.sort_values(keys).reset_index(drop=True)


def votes_text(votes: np.ndarray) -> str:
    return ";".join(map(str, np.asarray(votes, dtype=int).tolist()))


def count_changes(values: pd.Series) -> int:
    array = values.dropna().to_numpy()
    return int(np.sum(array[1:] != array[:-1])) if len(array) > 1 else 0


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame.loc[:, columns].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(view.columns) + " |"
    separator = "|" + "|".join(["---"] * len(view.columns)) + "|"
    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])
