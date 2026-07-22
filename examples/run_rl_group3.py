from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/finrl-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from scipy import stats

from finrl.reproduction.classifier_ensemble import holding_dispersion
from finrl.reproduction.classifier_ensemble import tau_grid
from finrl.reproduction.metrics import metrics_from_account_values
from reproduce_classifier_ensemble import BASE_MODEL_PARAMS
from reproduce_classifier_ensemble import append_account_curve
from reproduce_classifier_ensemble import build_rolling_windows
from reproduce_classifier_ensemble import collect_holdings_and_account
from reproduce_classifier_ensemble import env_kwargs
from reproduce_classifier_ensemble import frame_for_dates
from reproduce_classifier_ensemble import load_trademaster_rolling_data
from reproduce_classifier_ensemble import rl_model_class
from reproduce_classifier_ensemble import train_or_load_validation_selected_model
from run_forecasting_group1 import aggregate_metrics
from run_forecasting_group1 import align_account_curve
from run_forecasting_group1 import completed_repeat_exists
from run_forecasting_group1 import load_repeat
from run_forecasting_group1 import markdown_table
from run_forecasting_group1 import rolling_window_summary
from run_forecasting_group1 import run_repeat
from run_forecasting_group1 import select_tau_and_compare
from run_forecasting_group1 import simple_average_baselines
from run_forecasting_group1 import summarize_common_tau
from run_forecasting_group1 import summarize_tau_robustness


DEFAULT_MODEL_NAMES = ("ppo", "tqc")
MODEL_CHOICES = tuple(BASE_MODEL_PARAMS)
MODEL_DISPLAY_NAMES = {
    "a2c": "Advantage Actor-Critic (A2C)",
    "ppo": "Proximal Policy Optimization (PPO)",
    "sac": "Soft Actor-Critic (SAC)",
    "td3": "Twin Delayed DDPG (TD3)",
    "tqc": "Truncated Quantile Critics (TQC)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a two-agent deep-RL extension under the paper-aligned expanding-RL, "
            "rolling-classifier, fixed-global-tau protocol."
        )
    )
    parser.add_argument("--data-dir", default="external_data/trademaster_dj30")
    parser.add_argument("--trade-split", choices=["valid", "test"], default="valid")
    parser.add_argument("--dataset-label", default="DJ30")
    parser.add_argument(
        "--output-dir", default="results/rl_group3_ppo_tqc_dj30_full253"
    )
    parser.add_argument(
        "--models",
        nargs=2,
        choices=MODEL_CHOICES,
        default=list(DEFAULT_MODEL_NAMES),
        metavar=("LEFT", "RIGHT"),
        help="Two distinct DRL agents to compare and ensemble.",
    )
    parser.add_argument(
        "--model-source",
        action="append",
        default=[],
        metavar="MODEL=DIR",
        help=(
            "Import audited best checkpoints and validation histories from DIR. "
            "DIR may be an experiment directory or its models directory."
        ),
    )
    parser.add_argument(
        "--experiment-role",
        choices=["main", "stress_test", "ablation"],
        default="main",
    )
    parser.add_argument(
        "--normalize-imported-checkpoints",
        action="store_true",
        help=(
            "Reserialize an imported SB3 checkpoint only when its NumPy-era "
            "metadata cannot be loaded directly; policy tensors must hash identically."
        ),
    )
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--eval-interval", type=int, default=20000)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--master-seed", type=int, default=250217518)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument(
        "--training-workers",
        type=int,
        default=4,
        help="Independent model-window training jobs to run concurrently.",
    )
    parser.add_argument("--rebalance-window", type=int, default=63)
    parser.add_argument("--validation-window", type=int, default=63)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--tau-start", type=float, default=0.01)
    parser.add_argument("--tau-stop", type=float, default=0.89)
    parser.add_argument("--tau-step", type=float, default=0.01)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one window, 2,000 RL steps, two refits, and three tau values.",
    )
    return parser.parse_args()


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.timesteps = min(args.timesteps, 2000)
    args.eval_interval = min(args.eval_interval, 1000)
    args.repetitions = min(args.repetitions, 2)
    args.training_workers = 1
    args.max_windows = 1
    args.tau_start = 0.20
    args.tau_stop = 0.60
    args.tau_step = 0.20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_distribution_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (pair, group), frame in paired.groupby(["pair", "classifier_group"]):
        delta = frame["delta_sharpe"].to_numpy(dtype=float)
        nonzero = delta[~np.isclose(delta, 0.0)]
        wins = int((nonzero > 0.0).sum())
        sign_p = (
            float(stats.binomtest(wins, len(nonzero), 0.5, alternative="greater").pvalue)
            if len(nonzero)
            else 1.0
        )
        rows.append(
            {
                "pair": pair,
                "classifier_group": int(group),
                "n_backtests": len(delta),
                "wins_vs_stronger": int((delta > 0.0).sum()),
                "win_rate_vs_stronger": float(np.mean(delta > 0.0)),
                "delta_sharpe_mean": float(delta.mean()),
                "delta_sharpe_q25": float(np.quantile(delta, 0.25)),
                "delta_sharpe_median": float(np.median(delta)),
                "delta_sharpe_q75": float(np.quantile(delta, 0.75)),
                "delta_sharpe_min": float(delta.min()),
                "delta_sharpe_max": float(delta.max()),
                "one_sided_sign_test_p": sign_p,
            }
        )
    return pd.DataFrame(rows).sort_values(["pair", "classifier_group"])


def build_validation_audit(
    *,
    model_names: tuple[str, str],
    windows: pd.DataFrame,
    checkpoints: pd.DataFrame,
    selection_history: pd.DataFrame,
    classifier_audit: pd.DataFrame,
    classifier_diagnostics: pd.DataFrame,
    all_metrics: pd.DataFrame,
    curves: np.ndarray,
    evaluation_dates: np.ndarray,
    expected_trade_dates: np.ndarray,
    tau_values: np.ndarray,
    repetitions: int,
    expected_window_count: int,
    expected_validation_nodes: int,
) -> dict[str, object]:
    expected_session_count = len(expected_trade_dates)
    later_calibration_matches = all(
        windows.loc[index, "calibration_start"]
        == windows.loc[index - 1, "trade_start"]
        and windows.loc[index, "calibration_end"]
        == windows.loc[index - 1, "trade_end"]
        for index in range(1, len(windows))
    )
    expected_fits = repetitions * len(windows) * 5
    expected_metrics = repetitions * len(tau_values) * 5
    expected_diagnostics = repetitions * len(windows) * (4 + 3 + 2 + 7 + 9)
    checks = {
        "rolling_window_count": len(windows) == expected_window_count,
        "evaluation_session_count": int(windows["trade_dates"].sum())
        == expected_session_count,
        "evaluation_dates_cover_trade_split": bool(
            np.array_equal(
                np.asarray(evaluation_dates, dtype=str),
                np.asarray(expected_trade_dates, dtype=str),
            )
        ),
        "strict_train_calibration_trade_order": bool(
            (
                (windows["train_end"] < windows["calibration_start"])
                & (windows["calibration_end"] < windows["trade_start"])
            ).all()
        ),
        "later_calibration_equals_previous_trade": later_calibration_matches,
        "checkpoint_count": len(checkpoints) == len(windows) * len(model_names),
        "checkpoint_model_window_coverage": set(
            zip(checkpoints["window"].astype(int), checkpoints["model"].astype(str))
        )
        == {
            (int(window), model)
            for window in windows["window"].astype(int)
            for model in model_names
        },
        "checkpoint_hashes_unique": checkpoints["sha256"].nunique()
        == len(checkpoints),
        "imported_policy_parameters_match": bool(
            checkpoints["imported_policy_parameters_match"].astype(bool).all()
        ),
        "validation_nodes_per_checkpoint": bool(
            (
                selection_history.groupby(["window", "model"]).size()
                == expected_validation_nodes
            ).all()
        ),
        "classifier_fit_count": len(classifier_audit) == expected_fits,
        "classifier_fit_keys_unique": not classifier_audit.duplicated(
            ["repeat", "window", "pair", "classifier_group"]
        ).any(),
        "classifier_diagnostic_count": len(classifier_diagnostics)
        == expected_diagnostics,
        "run_metric_count": len(all_metrics) == expected_metrics,
        "run_metric_keys_unique": not all_metrics.duplicated(
            ["repeat", "pair", "classifier_group", "tau"]
        ).any(),
        "all_financial_metrics_finite": bool(
            np.isfinite(
                all_metrics[
                    ["cumulative_return", "sharpe", "calmar", "max_drawdown"]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "curve_shape_matches_protocol": list(curves.shape)
        == [repetitions, 5, len(tau_values), expected_session_count],
        "all_account_values_finite": bool(np.isfinite(curves).all()),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "observed": {
            "rolling_windows": len(windows),
            "evaluation_sessions": int(windows["trade_dates"].sum()),
            "expected_trade_sessions": expected_session_count,
            "checkpoints": len(checkpoints),
            "validation_history_rows": len(selection_history),
            "classifier_fits": len(classifier_audit),
            "classifier_diagnostic_rows": len(classifier_diagnostics),
            "run_metric_rows": len(all_metrics),
            "curve_shape": list(curves.shape),
        },
    }


def trade_period_dates(full_data: pd.DataFrame, trade_start: str) -> np.ndarray:
    dates = np.asarray(sorted(full_data["date"].astype(str).unique()), dtype=str)
    return dates[dates >= str(trade_start)]


def selected_validation_step(history: pd.DataFrame) -> int:
    mask = history["is_best"].astype(str).str.lower().isin(["true", "1"])
    if not mask.any():
        raise ValueError("validation history has no selected checkpoint")
    return int(history.loc[mask, "timesteps"].iloc[-1])


def experiment_models(args: argparse.Namespace) -> tuple[str, str]:
    model_names = tuple(map(str, args.models))
    if len(set(model_names)) != 2:
        raise ValueError("--models must contain two distinct agents")
    return model_names  # type: ignore[return-value]


def experiment_pair(model_names: tuple[str, str]) -> str:
    return "_".join(model_names)


def parse_model_sources(values: list[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--model-source must use MODEL=DIR syntax")
        model, raw_path = value.split("=", 1)
        model = model.strip().lower()
        if model not in MODEL_CHOICES:
            raise ValueError(f"unsupported model source: {model}")
        if model in sources:
            raise ValueError(f"duplicate model source: {model}")
        root = Path(raw_path).expanduser().resolve()
        model_dir = root / "models" if (root / "models").is_dir() else root
        if not model_dir.is_dir():
            raise FileNotFoundError(f"model source directory does not exist: {root}")
        sources[model] = model_dir
    return sources


def validate_source_history(
    history: pd.DataFrame,
    *,
    model: str,
    model_tag: str,
    window_info: dict[str, object],
    timesteps: int,
    eval_interval: int,
) -> None:
    required = {"model", "model_tag", "timesteps", "is_best", "sharpe"}
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"{model} source history is missing columns: {missing}")
    expected_steps = list(range(eval_interval, timesteps + 1, eval_interval))
    if not expected_steps or expected_steps[-1] != timesteps:
        expected_steps.append(timesteps)
    actual_steps = history["timesteps"].astype(int).tolist()
    if actual_steps != expected_steps:
        raise ValueError(
            f"{model} source history has steps {actual_steps}, expected {expected_steps}"
        )
    if set(history["model"].astype(str)) != {model}:
        raise ValueError(f"{model} source history contains a different model")
    if set(history["model_tag"].astype(str)) != {model_tag}:
        raise ValueError(f"{model} source history has an incompatible model tag")
    best_step = selected_validation_step(history)
    max_step = int(history.loc[history["sharpe"].astype(float).idxmax(), "timesteps"])
    if best_step != max_step:
        raise ValueError(
            f"{model} selected step {best_step} is not the maximum-Sharpe step {max_step}"
        )

    expected_dates = {
        "train_start": str(window_info["train_start"]),
        "train_end": str(window_info["train_end"]),
        "calibration_start": str(window_info["calibration_start"]),
        "calibration_end": str(window_info["calibration_end"]),
    }
    for column, expected in expected_dates.items():
        history_column = column
        if column.startswith("calibration_") and column not in history.columns:
            history_column = column.replace("calibration_", "validation_")
        if history_column in history.columns:
            actual = set(history[history_column].astype(str))
            if actual != {expected}:
                raise ValueError(
                    f"{model} source history {history_column}={actual}, expected {expected}"
                )


def policy_parameter_sha256(model: object) -> str:
    digest = hashlib.sha256()
    state = model.policy.state_dict()  # type: ignore[attr-defined]
    for name, tensor in sorted(state.items()):
        values = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def load_imported_model(
    model_name: str,
    checkpoint: Path,
    options: dict[str, object],
    *,
    allow_normalization: bool,
) -> tuple[object, str]:
    # Checkpoints created under NumPy 2 pickle this module under its new name.
    # NumPy 1.26 provides the same implementation at the legacy import path.
    if "numpy._core.numeric" not in sys.modules:
        import numpy.core.numeric as numpy_core_numeric

        sys.modules["numpy._core.numeric"] = numpy_core_numeric
    model_class = rl_model_class(model_name)
    try:
        return model_class.load(str(checkpoint)), "byte_copy"
    except (ModuleNotFoundError, ValueError) as error:
        if not allow_normalization:
            raise ValueError(
                f"checkpoint {checkpoint} is incompatible with the current runtime; "
                "rerun with --normalize-imported-checkpoints or retrain the model"
            ) from error

    from gymnasium.spaces import Box

    custom_objects = {
        "observation_space": Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(options["state_space"]),),
            dtype=np.float32,
        ),
        "action_space": Box(
            low=-1.0,
            high=1.0,
            shape=(int(options["action_space"]),),
            dtype=np.float32,
        ),
        "_last_obs": None,
        "_last_episode_starts": None,
    }
    return (
        model_class.load(str(checkpoint), custom_objects=custom_objects),
        "runtime_reserialize",
    )


def import_pretrained_models(
    args: argparse.Namespace,
    windows: list[dict[str, object]],
    output_dir: Path,
    model_names: tuple[str, str],
    options: dict[str, object],
) -> dict[tuple[str, int], dict[str, object]]:
    sources = parse_model_sources(args.model_source)
    unknown = sorted(set(sources).difference(model_names))
    if unknown:
        raise ValueError(f"model sources are not part of --models: {unknown}")
    if sources and args.force_train:
        raise ValueError("--model-source cannot be combined with --force-train")

    destination = output_dir / "models"
    destination.mkdir(parents=True, exist_ok=True)
    provenance: dict[tuple[str, int], dict[str, object]] = {}
    for model, source_dir in sources.items():
        for window_info in windows:
            window = int(window_info["window"])
            model_tag = f"rolling_w{window}_{args.timesteps}_seed{args.model_seed}"
            stem = f"agent_{model}_{model_tag}"
            source_checkpoint = source_dir / f"{stem}_best.zip"
            source_history = source_dir / f"{stem}_validation_history.csv"
            if not source_checkpoint.is_file() or not source_history.is_file():
                raise FileNotFoundError(
                    f"incomplete source checkpoint pair for {model} window {window}: "
                    f"{source_checkpoint}, {source_history}"
                )
            history = pd.read_csv(source_history)
            validate_source_history(
                history,
                model=model,
                model_tag=model_tag,
                window_info=window_info,
                timesteps=args.timesteps,
                eval_interval=args.eval_interval,
            )
            imported_model, import_mode = load_imported_model(
                model,
                source_checkpoint,
                options,
                allow_normalization=args.normalize_imported_checkpoints,
            )
            source_parameter_hash = policy_parameter_sha256(imported_model)
            target_checkpoint = destination / source_checkpoint.name
            target_history = destination / source_history.name
            if target_checkpoint.exists():
                target_model = rl_model_class(model).load(str(target_checkpoint))
            elif import_mode == "byte_copy":
                shutil.copy2(source_checkpoint, target_checkpoint)
                target_model = rl_model_class(model).load(str(target_checkpoint))
            else:
                imported_model.save(str(target_checkpoint.with_suffix("")))
                target_model = rl_model_class(model).load(str(target_checkpoint))
            target_parameter_hash = policy_parameter_sha256(target_model)
            if source_parameter_hash != target_parameter_hash:
                raise ValueError(
                    f"policy parameters changed while importing {model} window {window}"
                )
            if target_history.exists():
                if sha256_file(target_history) != sha256_file(source_history):
                    raise ValueError(
                        f"existing validation history differs from source: {target_history}"
                    )
            else:
                shutil.copy2(source_history, target_history)
            provenance[(model, window)] = {
                "source_checkpoint": str(source_checkpoint.resolve()),
                "source_checkpoint_sha256": sha256_file(source_checkpoint),
                "source_history": str(source_history.resolve()),
                "source_history_sha256": sha256_file(source_history),
                "import_mode": import_mode,
                "policy_parameter_sha256": target_parameter_hash,
                "policy_parameters_match": True,
            }
            del imported_model, target_model
            gc.collect()
    return provenance


def candidate_diagnostic_rows(
    window: int,
    pair: str,
    segment: str,
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, object]:
    candidates = np.stack([left, right], axis=1)
    return {
        "window": window,
        "pair": pair,
        "segment": segment,
        "samples": len(candidates),
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


def train_model_window_job(job: dict[str, object]) -> dict[str, object]:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    full_data, indicators, trade_start, _metadata = load_trademaster_rolling_data(
        str(job["data_dir"]), trade_split=str(job["trade_split"])
    )
    options = env_kwargs(full_data, indicators)
    windows = build_rolling_windows(
        full_data,
        trade_start_date=trade_start,
        rebalance_window=int(job["rebalance_window"]),
        validation_window=int(job["validation_window"]),
        max_windows=(
            None if job["max_windows"] is None else int(job["max_windows"])
        ),
    )
    window = int(job["window"])
    window_info = windows[window - 1]
    train = frame_for_dates(full_data, window_info["train_dates"])
    calibration = frame_for_dates(full_data, window_info["calibration_dates"])
    model_name = str(job["model"])
    timesteps = int(job["timesteps"])
    model_seed = int(job["model_seed"])
    model_tag = f"rolling_w{window}_{timesteps}_seed{model_seed}"
    _model, history = train_or_load_validation_selected_model(
        model_name,
        train,
        calibration,
        options,
        Path(str(job["output_dir"])),
        timesteps,
        model_seed,
        bool(job["force_train"]),
        model_tag=model_tag,
        eval_interval=int(job["eval_interval"]),
    )
    return {
        "window": window,
        "model": model_name,
        "selected_validation_step": selected_validation_step(history),
    }


def pretrain_model_windows(
    args: argparse.Namespace,
    windows: list[dict[str, object]],
    output_dir: Path,
    model_names: tuple[str, str],
) -> None:
    workers = min(max(1, int(args.training_workers)), len(windows) * len(model_names))
    if workers == 1:
        return
    common: dict[str, object] = {
        "data_dir": args.data_dir,
        "trade_split": args.trade_split,
        "rebalance_window": args.rebalance_window,
        "validation_window": args.validation_window,
        "max_windows": args.max_windows,
        "output_dir": str(output_dir),
        "timesteps": args.timesteps,
        "eval_interval": args.eval_interval,
        "model_seed": args.model_seed,
        "force_train": args.force_train,
    }
    jobs = [
        {**common, "window": int(window["window"]), "model": model}
        for window in windows
        for model in model_names
    ]
    print(f"PRETRAINING {len(jobs)} independent model-window jobs with {workers} workers")
    context = mp.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        for completed in pool.imap_unordered(train_model_window_job, jobs):
            print(
                "PRETRAINED window {window} {model}: selected step {step}".format(
                    window=completed["window"],
                    model=completed["model"],
                    step=completed["selected_validation_step"],
                )
            )


def prepare_rl_candidates(
    full_data: pd.DataFrame,
    windows: list[dict[str, object]],
    options: dict[str, object],
    args: argparse.Namespace,
    output_dir: Path,
    model_names: tuple[str, str],
    pair: str,
    imported_sources: dict[tuple[str, int], dict[str, object]],
) -> tuple[
    list[dict[str, object]],
    pd.DataFrame,
    dict[str, np.ndarray],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    records: list[dict[str, object]] = []
    base_curves = {model: pd.DataFrame() for model in model_names}
    base_last_states: dict[str, list[float] | None] = {
        model: None for model in model_names
    }
    history_frames: list[pd.DataFrame] = []
    checkpoint_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    candidate_dir = output_dir / "candidate_holdings"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    for window_info in windows:
        window = int(window_info["window"])
        train = frame_for_dates(full_data, window_info["train_dates"])
        calibration = frame_for_dates(full_data, window_info["calibration_dates"])
        trade = frame_for_dates(full_data, window_info["trade_dates"])
        calibration_holdings: dict[str, np.ndarray] = {}
        trade_holdings: dict[str, np.ndarray] = {}
        print(
            f"RL WINDOW {window}/{len(windows)}: train "
            f"{window_info['train_start']}..{window_info['train_end']}; "
            f"calibration {window_info['calibration_start']}.."
            f"{window_info['calibration_end']}; trade "
            f"{window_info['trade_start']}..{window_info['trade_end']}"
        )

        for model_name in model_names:
            model_tag = (
                f"rolling_w{window}_{args.timesteps}_seed{args.model_seed}"
            )
            model, history = train_or_load_validation_selected_model(
                model_name,
                train,
                calibration,
                options,
                output_dir,
                args.timesteps,
                args.model_seed,
                args.force_train,
                model_tag=model_tag,
                eval_interval=args.eval_interval,
            )
            annotated = history.copy()
            annotated["window"] = window
            annotated["train_start"] = window_info["train_start"]
            annotated["train_end"] = window_info["train_end"]
            annotated["calibration_start"] = window_info["calibration_start"]
            annotated["calibration_end"] = window_info["calibration_end"]
            annotated["calibration_source"] = window_info["calibration_source"]
            history_frames.append(annotated)

            checkpoint = (
                output_dir
                / "models"
                / f"agent_{model_name}_{model_tag}_best.zip"
            )
            imported_source = imported_sources.get((model_name, window), {})
            checkpoint_hash = sha256_file(checkpoint)
            checkpoint_rows.append(
                {
                    "window": window,
                    "model": model_name,
                    "checkpoint": str(checkpoint.resolve()),
                    "sha256": checkpoint_hash,
                    "training_seed": args.model_seed,
                    "training_timesteps": args.timesteps,
                    "evaluation_interval": args.eval_interval,
                    "selected_validation_step": selected_validation_step(history),
                    "deterministic_inference": True,
                    "imported_checkpoint_source": imported_source.get(
                        "source_checkpoint", ""
                    ),
                    "imported_source_sha256": imported_source.get(
                        "source_checkpoint_sha256", ""
                    ),
                    "checkpoint_import_mode": imported_source.get(
                        "import_mode", "fresh_training"
                    ),
                    "policy_parameter_sha256": imported_source.get(
                        "policy_parameter_sha256", ""
                    ),
                    "imported_policy_parameters_match": imported_source.get(
                        "policy_parameters_match", True
                    ),
                }
            )

            calibration_holdings[model_name], _account, _state = (
                collect_holdings_and_account(model, calibration, options)
            )
            (
                trade_holdings[model_name],
                account,
                base_last_states[model_name],
            ) = collect_holdings_and_account(
                model,
                trade,
                options,
                initial=base_last_states[model_name] is None,
                previous_state=base_last_states[model_name],
            )
            base_curves[model_name] = append_account_curve(
                base_curves[model_name], account
            )
            del model
            gc.collect()

        for segment, holdings in (
            ("calibration", calibration_holdings),
            ("trade", trade_holdings),
        ):
            diagnostic_rows.append(
                candidate_diagnostic_rows(
                    window,
                    pair,
                    segment,
                    holdings[model_names[0]],
                    holdings[model_names[1]],
                )
            )

        candidate_payload: dict[str, np.ndarray] = {
            "calibration_dates": np.asarray(
                window_info["calibration_dates"], dtype=str
            ),
            "trade_dates": np.asarray(window_info["trade_dates"], dtype=str),
        }
        for model_name in model_names:
            candidate_payload[f"{model_name}_calibration"] = calibration_holdings[
                model_name
            ]
            candidate_payload[f"{model_name}_trade"] = trade_holdings[model_name]
        np.savez_compressed(
            candidate_dir / f"window_{window:02d}.npz",
            **candidate_payload,
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
        model: align_account_curve(curve, expected_dates)
        for model, curve in base_curves.items()
    }
    base_metrics = pd.DataFrame(
        [
            {"model": model, **metrics_from_account_values(values)}
            for model, values in base_values.items()
        ]
    ).sort_values("model")
    return (
        records,
        base_metrics,
        base_values,
        pd.concat(history_frames, ignore_index=True),
        pd.DataFrame(checkpoint_rows),
        pd.DataFrame(diagnostic_rows),
    )


def write_report(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    model_names: tuple[str, str],
    pair: str,
    metadata: dict[str, object],
    windows: pd.DataFrame,
    checkpoints: pd.DataFrame,
    base_metrics: pd.DataFrame,
    average_metrics: pd.DataFrame,
    selected: pd.DataFrame,
    tau_robustness: pd.DataFrame,
    selected_common_tau: pd.DataFrame,
    candidate_diagnostics: pd.DataFrame,
    paired_distribution: pd.DataFrame,
) -> None:
    wins = int((selected["delta_sharpe_mean"] > 0.0).sum())
    positive_intervals = int((selected["delta_sharpe_ci_low"] > 0.0).sum())
    average_wins = int((selected["delta_sharpe_vs_average"] > 0.0).sum())
    distribution_notes = [
        "Group {group}: mean paired Sharpe delta {mean:.6f}, median {median:.6f}, "
        "wins {wins}/{total}, one-sided sign-test p={p:.4f}.".format(
            group=int(row.classifier_group),
            mean=float(row.delta_sharpe_mean),
            median=float(row.delta_sharpe_median),
            wins=int(row.wins_vs_stronger),
            total=int(row.n_backtests),
            p=float(row.one_sided_sign_test_p),
        )
        for row in paired_distribution.itertuples(index=False)
    ]
    model_label = " + ".join(name.upper() for name in model_names)
    model_descriptions = " and ".join(
        MODEL_DISPLAY_NAMES[name] for name in model_names
    )
    tau_robustness_display = tau_robustness.copy()
    for column in ("tau_beating_stronger_min", "tau_beating_stronger_max"):
        tau_robustness_display[column] = tau_robustness_display[column].map(
            lambda value: "not applicable" if pd.isna(value) else value
        )
    report = [
        f"# Expanded Deep-RL Pair: {model_label}",
        "",
        f"Experiment role: **{args.experiment_role.replace('_', ' ')}**.",
        "",
        "## Protocol",
        "",
        f"- Dataset: {args.dataset_label}; `{args.data_dir}` `{args.trade_split}` split; {metadata['stock_count']} aligned assets.",
        f"- Evaluation: {windows.iloc[0]['trade_start']} to {windows.iloc[-1]['trade_end']}, {int(windows['trade_dates'].sum())} sessions in {len(windows)} blocks.",
        f"- Base agents: {model_descriptions}.",
        f"- Each selected checkpoint comes from seed {args.model_seed} expanding-window training for {args.timesteps:,} steps per agent and window.",
        f"- Calibration Sharpe is evaluated every {args.eval_interval:,} steps; the best prior-block checkpoint supplies both calibration and trade candidates.",
        "- Reused checkpoint source hashes are recorded. If serialization normalization is required, policy-tensor fingerprints must remain identical before and after migration.",
        "- Candidate generation and trading inference use `deterministic=True`.",
        "- The first classifier block is the training tail; each later classifier is fitted only on the immediately previous traded block.",
        "- Five fixed classifier groups, no classifier grid search, and the original voting mechanism are used.",
        f"- Every tau is fixed across the complete path; grid {args.tau_start:.2f}-{args.tau_stop:.2f} by {args.tau_step:.2f}.",
        f"- {args.repetitions} repetitions refit only rolling classifiers; selected RL checkpoints and deterministic candidates remain fixed.",
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
                "trade_start",
                "trade_end",
                "trade_dates",
            ],
        ),
        "",
        "## Selected RL Checkpoints",
        "",
        markdown_table(
            checkpoints,
            ["window", "model", "selected_validation_step", "training_seed"],
        ),
        "",
        "## Single Models",
        "",
        markdown_table(
            base_metrics,
            ["model", "cumulative_return", "sharpe", "calmar", "max_drawdown"],
        ),
        "",
        "## Simple Average Control",
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
        "## Tau Robustness",
        "",
        markdown_table(
            tau_robustness_display,
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
        "## Candidate Diversity",
        "",
        markdown_table(
            candidate_diagnostics,
            [
                "window",
                "segment",
                "samples",
                "mean_holding_l1",
                "mean_dispersion",
                "identical_holding_rate",
            ],
        ),
        "",
        "## Paired Distribution Audit",
        "",
        markdown_table(
            paired_distribution,
            [
                "classifier_group",
                "wins_vs_stronger",
                "win_rate_vs_stronger",
                "delta_sharpe_mean",
                "delta_sharpe_q25",
                "delta_sharpe_median",
                "delta_sharpe_q75",
                "delta_sharpe_min",
                "delta_sharpe_max",
                "one_sided_sign_test_p",
            ],
        ),
        "",
        *distribution_notes,
        "",
        "## Main Finding",
        "",
        f"At each classifier group's mean-Sharpe-maximizing global tau, {model_label} beats its stronger component in {wins}/5 groups and the simple holding average in {average_wins}/5; {positive_intervals}/5 paired 95% intervals versus the stronger component are entirely positive.",
        "",
        "Tau is selected after comparing complete evaluation paths, so this is a full-path sensitivity analysis rather than an out-of-sample tau-selection claim. The 30 repetitions measure classifier-refit variation conditional on one fixed checkpoint set and one realized market path; they do not measure variation across RL training seeds.",
        "",
    ]
    (output_dir / f"DRL_PAIR_{pair.upper()}_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    apply_smoke_defaults(args)
    model_names = experiment_models(args)
    pair = experiment_pair(model_names)
    pair_components = {pair: model_names}
    if args.force_train and args.resume:
        raise ValueError("--force-train and --resume cannot be combined")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    imported_sources = import_pretrained_models(
        args, windows, output_dir, model_names, options
    )

    manifest = {
        "dataset": metadata,
        "dataset_label": args.dataset_label,
        "data_dir": str(Path(args.data_dir).resolve()),
        "trade_split": args.trade_split,
        "experiment_group": "expanded_deep_rl_pair",
        "experiment_role": args.experiment_role,
        "models": list(model_names),
        "pair_components": {
            key: list(value) for key, value in pair_components.items()
        },
        "model_parameters": {name: BASE_MODEL_PARAMS[name] for name in model_names},
        "pretrained_model_sources": {
            model: str(path)
            for model, path in parse_model_sources(args.model_source).items()
        },
        "normalize_imported_checkpoints": args.normalize_imported_checkpoints,
        "model_seed": args.model_seed,
        "master_seed": args.master_seed,
        "training_workers": args.training_workers,
        "training_timesteps_per_model_window": args.timesteps,
        "rl_evaluation_interval": args.eval_interval,
        "deterministic_rl_inference": True,
        "rl_training_window": "expanding",
        "classifier_training_window": "rolling_previous_block",
        "classifier_groups": [1, 2, 3, 4, 5],
        "classifier_grid_search": False,
        "repetitions": args.repetitions,
        "tau_values": tau_values.tolist(),
        "fixed_global_tau_per_path": True,
        "window_count": len(windows),
        "window_boundaries": windows_frame.to_dict(orient="records"),
        "expected_classifier_fits": args.repetitions * len(windows) * 5,
        "expected_run_metric_rows": args.repetitions * len(tau_values) * 5,
    }
    manifest_path = output_dir / "experiment_manifest.json"
    if manifest_path.exists() and not args.force_train:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = (
            "data_dir",
            "trade_split",
            "experiment_role",
            "models",
            "model_parameters",
            "pretrained_model_sources",
            "normalize_imported_checkpoints",
            "model_seed",
            "master_seed",
            "training_timesteps_per_model_window",
            "rl_evaluation_interval",
            "repetitions",
            "tau_values",
            "window_count",
        )
        if any(existing.get(key) != manifest.get(key) for key in comparable):
            raise ValueError(
                "output directory contains an incompatible experiment; use a new directory"
            )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    pretrain_model_windows(args, windows, output_dir, model_names)
    if args.training_workers > 1:
        args.force_train = False

    (
        records,
        base_metrics,
        base_values,
        selection_history,
        checkpoint_manifest,
        candidate_diagnostics,
    ) = prepare_rl_candidates(
        full_data,
        windows,
        options,
        args,
        output_dir,
        model_names,
        pair,
        imported_sources,
    )
    expected_dates = np.concatenate(
        [np.asarray(record["window_info"]["trade_dates"], dtype=str) for record in records]
    )
    expected_trade_dates = trade_period_dates(full_data, trade_start)
    average_metrics, average_values = simple_average_baselines(
        records, options, expected_dates, pair_components
    )

    base_metrics.to_csv(output_dir / "base_model_metrics.csv", index=False)
    average_metrics.to_csv(output_dir / "simple_average_metrics.csv", index=False)
    selection_history.to_csv(
        output_dir / "rolling_rl_selection_history.csv", index=False
    )
    checkpoint_manifest.to_csv(output_dir / "checkpoint_manifest.csv", index=False)
    candidate_diagnostics.to_csv(
        output_dir / "candidate_holding_diagnostics.csv", index=False
    )
    np.savez_compressed(
        output_dir / "base_account_curves.npz", dates=expected_dates, **base_values
    )
    np.savez_compressed(
        output_dir / "simple_average_account_curves.npz",
        dates=expected_dates,
        **average_values,
    )

    manifest["checkpoints"] = checkpoint_manifest.to_dict(orient="records")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    metric_frames: list[pd.DataFrame] = []
    curve_arrays: list[np.ndarray] = []
    classifier_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
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
                master_seed=args.master_seed,
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
        mean_metrics, base_metrics, average_metrics, pair_components
    )
    classifier_audit = pd.concat(classifier_frames, ignore_index=True)
    classifier_diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    paired_distribution = paired_distribution_summary(paired)

    all_metrics.to_csv(output_dir / "all_classifier_refit_metrics.csv", index=False)
    mean_metrics.to_csv(output_dir / "mean_metrics_by_fixed_tau.csv", index=False)
    selected.to_csv(output_dir / "selected_tau_summary.csv", index=False)
    paired.to_csv(output_dir / "selected_tau_paired_runs.csv", index=False)
    tau_robustness.to_csv(output_dir / "tau_robustness_summary.csv", index=False)
    common_tau.to_csv(output_dir / "common_tau_summary.csv", index=False)
    selected_common_tau.to_csv(
        output_dir / "selected_common_tau_summary.csv", index=False
    )
    classifier_audit.to_csv(output_dir / "classifier_refit_audit.csv", index=False)
    classifier_diagnostics.to_csv(
        output_dir / "classifier_diagnostics.csv", index=False
    )
    paired_distribution.to_csv(
        output_dir / "selected_tau_distribution_audit.csv", index=False
    )
    curve_stack = np.stack(curve_arrays)
    np.savez_compressed(
        output_dir / "all_ensemble_account_curves.npz",
        ensemble=curve_stack,
        dates=expected_dates,
        pairs=np.asarray(list(pair_components)),
        tau=tau_values,
    )
    validation_audit = build_validation_audit(
        model_names=model_names,
        windows=windows_frame,
        checkpoints=checkpoint_manifest,
        selection_history=selection_history,
        classifier_audit=classifier_audit,
        classifier_diagnostics=classifier_diagnostics,
        all_metrics=all_metrics,
        curves=curve_stack,
        evaluation_dates=expected_dates,
        expected_trade_dates=expected_trade_dates,
        tau_values=tau_values,
        repetitions=args.repetitions,
        expected_window_count=(4 if args.max_windows is None else len(windows_frame)),
        expected_validation_nodes=int(np.ceil(args.timesteps / args.eval_interval)),
    )
    (output_dir / "experiment_validation_audit.json").write_text(
        json.dumps(validation_audit, indent=2), encoding="utf-8"
    )
    if not validation_audit["passed"]:
        raise ValueError("experiment validation audit failed")
    write_report(
        output_dir,
        args=args,
        model_names=model_names,
        pair=pair,
        metadata=metadata,
        windows=windows_frame,
        checkpoints=checkpoint_manifest,
        base_metrics=base_metrics,
        average_metrics=average_metrics,
        selected=selected,
        tau_robustness=tau_robustness,
        selected_common_tau=selected_common_tau,
        candidate_diagnostics=candidate_diagnostics,
        paired_distribution=paired_distribution,
    )
    print(base_metrics.to_string(index=False))
    print(selected.to_string(index=False))
    print(f"Saved expanded deep-RL {pair} experiment to {output_dir}")


if __name__ == "__main__":
    main()
