from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/finrl-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from finrl.reproduction.causal_crossover_tau import CausalCrossoverTau
from finrl.reproduction.classifier_ensemble import confidence_matrix
from finrl.reproduction.classifier_ensemble import holding_dispersion
from finrl.reproduction.classifier_ensemble import select_holding_from_confidence
from finrl.reproduction.classifier_ensemble import train_classifier_group
from finrl.reproduction.forecasting_ensemble import (
    initialize_selected_holding_environment,
)
from finrl.reproduction.forecasting_ensemble import step_target_holding_branch
from finrl.reproduction.metrics import metrics_from_account_values
from paper_causal_tau_support import ALL_PERIODS
from paper_causal_tau_support import EVALUATION_PERIODS
from paper_causal_tau_support import METRICS
from paper_causal_tau_support import PAIR_COMPONENTS
from paper_causal_tau_support import WARMUP_PERIOD
from paper_causal_tau_support import PeriodInputs
from paper_causal_tau_support import SequenceResult
from paper_causal_tau_support import add_paper_comparators
from paper_causal_tau_support import count_changes
from paper_causal_tau_support import load_period_inputs
from paper_causal_tau_support import load_tau_grid
from paper_causal_tau_support import markdown_table
from paper_causal_tau_support import votes_text
from run_forecasting_group1 import align_account_curve
from run_forecasting_group1 import mean_sd_ci
from run_forecasting_group1 import stable_seed
from reproduce_classifier_ensemble import append_account_curve


MAX_FIT_BLOCKS = 8
CDF_LOOKBACK_DAYS = 252
MIN_HISTORY_BLOCKS = 4
MIN_INFORMATIVE_DAYS = 63
MIN_SIDE_DAYS = 20
CONFIDENCE_LEVEL = 0.95
FEEDBACK_STRING_COLUMNS = (
    "period",
    "pair",
    "date",
    "selected_mode",
)
FEEDBACK_INTEGER_COLUMNS = (
    "repeat",
    "classifier_group",
    "window",
    "decision_index",
    "aggressive_agent_index",
    "conservative_agent_index",
)
FEEDBACK_BOOLEAN_COLUMNS = ("fallback_used", "branch_diverged")
FEEDBACK_FLOAT_COLUMNS = (
    "selected_tau",
    "holding_dispersion",
    "aggressive_return",
    "conservative_return",
    "fallback_return",
    "actual_master_return",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a causal blockwise tau selected from the zero crossing of "
            "same-state classifier-mode return advantages."
        )
    )
    parser.add_argument(
        "--candidate-root", default="work/causal_candidates"
    )
    parser.add_argument(
        "--output-dir", default="work/causal_tau_guarded"
    )
    parser.add_argument(
        "--pairs", nargs="+", choices=tuple(PAIR_COMPONENTS), default=list(PAIR_COMPONENTS)
    )
    parser.add_argument(
        "--groups", nargs="+", type=int, choices=range(1, 6), default=[1, 2, 3, 4, 5]
    )
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--disable-lcb-guardrail", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.smoke:
        args.pairs = args.pairs[:1]
        args.groups = args.groups[:1]
        args.repetitions = 1
        args.workers = 1
    if not 1 <= args.repetitions <= 30:
        raise ValueError("repetitions must be between 1 and 30")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if len(args.pairs) != len(set(args.pairs)):
        raise ValueError("pairs must be unique")
    if len(args.groups) != len(set(args.groups)):
        raise ValueError("classifier groups must be unique")


def make_controller(
    tau_values: np.ndarray, *, require_positive_lcb: bool
) -> CausalCrossoverTau:
    return CausalCrossoverTau(
        tau_values,
        max_fit_blocks=MAX_FIT_BLOCKS,
        cdf_lookback_days=CDF_LOOKBACK_DAYS,
        min_history_blocks=MIN_HISTORY_BLOCKS,
        min_informative_days=MIN_INFORMATIVE_DAYS,
        min_side_days=MIN_SIDE_DAYS,
        confidence_level=CONFIDENCE_LEVEL,
        require_positive_lcb=require_positive_lcb,
    )


def simulate_period(
    *,
    pair: str,
    repeat: int,
    group: int,
    inputs: PeriodInputs,
    controller: CausalCrossoverTau,
    save_outputs: bool,
) -> tuple[
    dict[str, object],
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    model_names = tuple(PAIR_COMPONENTS[pair])
    curve = pd.DataFrame()
    last_state: list[float] | None = None
    decisions: list[dict[str, object]] = []
    classifier_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []

    for record in inputs.records:
        window = int(record["window"])
        trade_start = str(record["trade_dates"][0])
        trade_end = str(record["trade_dates"][-1])
        history_before = controller.snapshot()
        if history_before["last_completed_date"]:
            if str(history_before["last_completed_date"]) >= trade_start:
                raise ValueError("tau training history reaches the current trade block")

        if inputs.period == WARMUP_PERIOD:
            selection = None
            selection_row: dict[str, object] = {
                "status": "warmup_collection",
                "selected_tau": np.nan,
                "threshold_quantile": np.nan,
                "raw_threshold": np.nan,
                "history_blocks": history_before["completed_blocks"],
                "history_days": history_before["completed_days"],
                "informative_days": np.nan,
                "cdf_days": np.nan,
                "low_mode_advantage": np.nan,
                "high_mode_advantage": np.nan,
                "policy_advantage_mean": np.nan,
                "policy_advantage_lcb": np.nan,
                "fit_start_date": "",
                "fit_end_date": str(history_before["last_completed_date"]),
            }
        else:
            selection = controller.select_for_next_block()
            selection_row = selection.to_dict()

        expected_seed = stable_seed(
            inputs.master_seed, "classifier", repeat, window, pair, group
        )
        saved_seed = inputs.seed_audit.get((repeat, window, group))
        if saved_seed != expected_seed:
            raise ValueError(
                f"classifier seed mismatch for {inputs.period}/{pair}/"
                f"repeat {repeat}/window {window}/group {group}"
            )
        classifiers = train_classifier_group(
            [
                record["calibration_holdings"][model_names[0]],
                record["calibration_holdings"][model_names[1]],
            ],
            group,
            random_state=expected_seed,
            grid_search=False,
        )
        classifier_rows.append(
            {
                "period": inputs.period,
                "pair": pair,
                "repeat": repeat,
                "classifier_group": group,
                "window": window,
                "classifier_seed": expected_seed,
                "classifier_count": len(classifiers),
                "classifier_names": ";".join(name for name, _model in classifiers),
                "calibration_start": str(record["calibration_dates"][0]),
                "calibration_end": str(record["calibration_dates"][-1]),
                "trade_start": trade_start,
            }
        )

        candidates = np.stack(
            [record["trade_holdings"][model] for model in model_names], axis=1
        )
        environment = initialize_selected_holding_environment(
            record["trade"],
            inputs.options,
            initial=last_state is None,
            previous_state=last_state,
        )
        fallback_index = model_names.index(str(record["fallback_model"]))
        block_dispersion: list[float] = []
        block_aggressive_return: list[float] = []
        block_conservative_return: list[float] = []
        block_fallback_return: list[float] = []
        block_diverged: list[bool] = []

        for local_step, decision_index in enumerate(record["decision_indices"]):
            day_candidates = candidates[local_step]
            dispersion = holding_dispersion(day_candidates)
            q_matrix = confidence_matrix(classifiers, day_candidates, [0, 1])
            aggressive = select_holding_from_confidence(
                day_candidates, q_matrix, 2.0, dispersion=dispersion
            )
            conservative = select_holding_from_confidence(
                day_candidates, q_matrix, -1.0, dispersion=dispersion
            )

            branches: dict[int, tuple[object, float]] = {}
            for candidate_index in {
                int(aggressive.selected_index),
                int(conservative.selected_index),
                int(fallback_index),
            }:
                branches[candidate_index] = step_target_holding_branch(
                    environment,
                    day_candidates[candidate_index],
                    hmax=float(inputs.options["hmax"]),
                )
            aggressive_branch, aggressive_return = branches[
                int(aggressive.selected_index)
            ]
            conservative_branch, conservative_return = branches[
                int(conservative.selected_index)
            ]
            fallback_branch, fallback_return = branches[int(fallback_index)]

            fallback_used = selection is None or not selection.selected
            if fallback_used:
                selected_index = int(fallback_index)
                selected_mode = "single_rl_fallback"
                environment = fallback_branch
                actual_return = fallback_return
            elif dispersion < selection.selected_tau:
                selected_index = int(aggressive.selected_index)
                selected_mode = "aggressive"
                environment = aggressive_branch
                actual_return = aggressive_return
            else:
                selected_index = int(conservative.selected_index)
                selected_mode = "conservative"
                environment = conservative_branch
                actual_return = conservative_return

            expected_return = branches[selected_index][1]
            if not np.isclose(actual_return, expected_return, rtol=0.0, atol=1e-15):
                raise ValueError("selected branch return differs from master return")

            diverged = int(aggressive.selected_index) != int(
                conservative.selected_index
            )
            block_dispersion.append(dispersion)
            block_aggressive_return.append(aggressive_return)
            block_conservative_return.append(conservative_return)
            block_fallback_return.append(fallback_return)
            block_diverged.append(diverged)

            decisions.append(
                {
                    "period": inputs.period,
                    "pair": pair,
                    "repeat": repeat,
                    "classifier_group": group,
                    "window": window,
                    "decision_index": int(decision_index),
                    "date": str(record["trade_dates"][local_step]),
                    "selection_status": selection_row["status"],
                    "selected_tau": selection_row["selected_tau"],
                    "threshold_quantile": selection_row["threshold_quantile"],
                    "fallback_used": fallback_used,
                    "fallback_model": record["fallback_model"],
                    "selected_mode": selected_mode,
                    "selected_agent_index": selected_index,
                    "selected_agent": model_names[selected_index],
                    "holding_dispersion": dispersion,
                    "aggressive_agent_index": aggressive.selected_index,
                    "conservative_agent_index": conservative.selected_index,
                    "aggressive_votes": votes_text(aggressive.votes),
                    "conservative_votes": votes_text(conservative.votes),
                    "branch_diverged": diverged,
                    "aggressive_return": aggressive_return,
                    "conservative_return": conservative_return,
                    "fallback_return": fallback_return,
                    "actual_master_return": actual_return,
                    "counterfactual_best_return": max(
                        aggressive_return,
                        conservative_return,
                        fallback_return,
                    ),
                    "counterfactual_regret": max(
                        aggressive_return,
                        conservative_return,
                        fallback_return,
                    )
                    - actual_return,
                    "history_blocks_before": history_before["completed_blocks"],
                    "history_end_before": history_before["last_completed_date"],
                }
            )

        account = environment.save_asset_memory()
        curve = append_account_curve(curve, account)
        last_state = list(environment.render())

        block_id = f"{inputs.period}-window-{window}"
        controller.add_completed_block(
            block_id=block_id,
            start_date=str(record["trade_dates"][0]),
            end_date=trade_end,
            dispersion=np.asarray(block_dispersion),
            aggressive_return=np.asarray(block_aggressive_return),
            conservative_return=np.asarray(block_conservative_return),
            fallback_return=np.asarray(block_fallback_return),
            branch_diverged=np.asarray(block_diverged),
        )
        history_after = controller.snapshot()
        block_rows.append(
            {
                "period": inputs.period,
                "pair": pair,
                "repeat": repeat,
                "classifier_group": group,
                "window": window,
                "block_id": block_id,
                "trade_start": trade_start,
                "trade_end": trade_end,
                "feedback_end": trade_end,
                "fallback_model": record["fallback_model"],
                **selection_row,
                "history_blocks_before": history_before["completed_blocks"],
                "history_days_before": history_before["completed_days"],
                "history_end_before": history_before["last_completed_date"],
                "history_blocks_after": history_after["completed_blocks"],
                "history_days_after": history_after["completed_days"],
                "history_end_after": history_after["last_completed_date"],
                "block_decision_days": len(block_dispersion),
                "block_informative_days": int(np.sum(block_diverged)),
                "outcomes_added_only_after_block": True,
            }
        )

    values = align_account_curve(curve, inputs.expected_dates)
    financial_metrics = metrics_from_account_values(values)
    decision_frame = pd.DataFrame(decisions)
    metric: dict[str, object] = {
        "period": inputs.period,
        "pair": pair,
        "repeat": repeat,
        "classifier_group": group,
        "decision_days": int(
            sum(len(record["decision_indices"]) for record in inputs.records)
        ),
        **financial_metrics,
    }
    if save_outputs:
        ensemble_days = ~decision_frame["fallback_used"].astype(bool)
        hit = np.isclose(
            decision_frame["actual_master_return"],
            decision_frame["counterfactual_best_return"],
            rtol=0.0,
            atol=1e-15,
        )
        selected_tau = decision_frame.loc[ensemble_days, "selected_tau"]
        metric.update(
            selected_tau_days=int(ensemble_days.sum()),
            fallback_days=int((~ensemble_days).sum()),
            fallback_day_rate=float((~ensemble_days).mean()),
            selected_blocks=int(
                pd.DataFrame(block_rows).query("period == @inputs.period")["status"]
                .eq("selected")
                .sum()
            ),
            mean_selected_tau=(
                float(selected_tau.mean()) if len(selected_tau) else np.nan
            ),
            mean_threshold_quantile=(
                float(
                    decision_frame.loc[ensemble_days, "threshold_quantile"].mean()
                )
                if ensemble_days.any()
                else np.nan
            ),
            tau_switches=count_changes(selected_tau),
            aggressive_day_rate=float(
                (decision_frame["selected_mode"] == "aggressive").mean()
            ),
            counterfactual_hit_rate=float(hit.mean()),
            ensemble_mode_hit_rate=(
                float(hit[ensemble_days].mean()) if ensemble_days.any() else np.nan
            ),
            branch_divergence_rate=float(decision_frame["branch_diverged"].mean()),
            mean_counterfactual_regret=float(
                decision_frame["counterfactual_regret"].mean()
            ),
        )
    return (
        metric,
        values,
        decision_frame,
        pd.DataFrame(classifier_rows),
        pd.DataFrame(block_rows),
    )


def completed_run_exists(run_dir: Path) -> bool:
    required = [
        run_dir / "metrics.csv",
        run_dir / "account_curves.npz",
        run_dir / "daily_crossover_tau_decisions.csv",
        run_dir / "daily_crossover_tau_feedback.npz",
        run_dir / "classifier_refit_audit.csv",
        run_dir / "block_tau_audit.csv",
    ]
    return all(path.exists() for path in required)


def save_feedback_payload(path: Path, decisions: pd.DataFrame) -> None:
    payload: dict[str, np.ndarray] = {}
    for column in FEEDBACK_STRING_COLUMNS:
        payload[column] = decisions[column].fillna("").astype(str).to_numpy(dtype=str)
    for column in FEEDBACK_INTEGER_COLUMNS:
        payload[column] = decisions[column].to_numpy(dtype=np.int64)
    for column in FEEDBACK_BOOLEAN_COLUMNS:
        payload[column] = decisions[column].to_numpy(dtype=bool)
    for column in FEEDBACK_FLOAT_COLUMNS:
        payload[column] = decisions[column].to_numpy(dtype=float)
    np.savez_compressed(path, **payload)


def restore_feedback_payload(path: Path, decisions: pd.DataFrame) -> pd.DataFrame:
    restored = decisions.copy()
    with np.load(path, allow_pickle=False) as arrays:
        for column in (
            *FEEDBACK_STRING_COLUMNS,
            *FEEDBACK_INTEGER_COLUMNS,
            *FEEDBACK_BOOLEAN_COLUMNS,
            *FEEDBACK_FLOAT_COLUMNS,
        ):
            if len(arrays[column]) != len(restored):
                raise ValueError("lossless feedback payload does not align with CSV rows")
            restored[column] = arrays[column]
    return restored


def save_sequence_result(run_dir: Path, result: SequenceResult) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    result.metrics.to_csv(run_dir / "metrics.csv", index=False)
    result.decisions.to_csv(run_dir / "daily_crossover_tau_decisions.csv", index=False)
    save_feedback_payload(
        run_dir / "daily_crossover_tau_feedback.npz", result.decisions
    )
    result.classifier_audit.to_csv(run_dir / "classifier_refit_audit.csv", index=False)
    result.expert_state_audit.to_csv(run_dir / "block_tau_audit.csv", index=False)
    payload: dict[str, np.ndarray] = {}
    for period in EVALUATION_PERIODS:
        payload[f"dates_{period}"] = result.dates[period]
        payload[f"account_value_{period}"] = result.curves[period]
    np.savez_compressed(run_dir / "account_curves.npz", **payload)


def load_sequence_result(run_dir: Path) -> SequenceResult:
    with np.load(run_dir / "account_curves.npz") as arrays:
        curves = {
            period: arrays[f"account_value_{period}"].astype(float)
            for period in EVALUATION_PERIODS
        }
        dates = {
            period: arrays[f"dates_{period}"].astype(str)
            for period in EVALUATION_PERIODS
        }
    decisions = pd.read_csv(run_dir / "daily_crossover_tau_decisions.csv")
    decisions = restore_feedback_payload(
        run_dir / "daily_crossover_tau_feedback.npz", decisions
    )
    return SequenceResult(
        metrics=pd.read_csv(run_dir / "metrics.csv"),
        curves=curves,
        dates=dates,
        decisions=decisions,
        classifier_audit=pd.read_csv(run_dir / "classifier_refit_audit.csv"),
        expert_state_audit=pd.read_csv(run_dir / "block_tau_audit.csv"),
    )


def simulate_sequence(
    *,
    pair: str,
    repeat: int,
    group: int,
    inputs_by_period: dict[str, PeriodInputs],
    tau_values: np.ndarray,
    require_positive_lcb: bool,
    run_dir: Path,
) -> SequenceResult:
    controller = make_controller(
        tau_values, require_positive_lcb=require_positive_lcb
    )
    metrics: list[dict[str, object]] = []
    curves: dict[str, np.ndarray] = {}
    dates: dict[str, np.ndarray] = {}
    decisions: list[pd.DataFrame] = []
    classifier_audits: list[pd.DataFrame] = []
    block_audits: list[pd.DataFrame] = []

    for period in ALL_PERIODS:
        save_outputs = period in EVALUATION_PERIODS
        metric, values, decision_frame, classifier_audit, block_audit = simulate_period(
            pair=pair,
            repeat=repeat,
            group=group,
            inputs=inputs_by_period[period],
            controller=controller,
            save_outputs=save_outputs,
        )
        if save_outputs:
            metrics.append(metric)
            curves[period] = values
            dates[period] = inputs_by_period[period].expected_dates
        decisions.append(decision_frame)
        classifier_audits.append(classifier_audit)
        block_audits.append(block_audit)

    result = SequenceResult(
        metrics=pd.DataFrame(metrics),
        curves=curves,
        dates=dates,
        decisions=pd.concat(decisions, ignore_index=True),
        classifier_audit=pd.concat(classifier_audits, ignore_index=True),
        expert_state_audit=pd.concat(block_audits, ignore_index=True),
    )
    save_sequence_result(run_dir, result)
    return result


def aggregate_results(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    comparisons = (
        "delta_sharpe_vs_stronger",
        "delta_sharpe_vs_causal_single",
    )
    for (period, pair, group), frame in metrics.groupby(
        ["period", "pair", "classifier_group"], sort=True
    ):
        row: dict[str, object] = {
            "period": period,
            "pair": pair,
            "classifier_group": int(group),
            "repetitions": len(frame),
            "stronger_single_model": frame["stronger_single_model"].iloc[0],
            "stronger_single_sharpe": float(frame["stronger_single_sharpe"].iloc[0]),
            "causal_single_sharpe": float(frame["causal_single_sharpe"].iloc[0]),
            "wins_vs_stronger_single": int(
                (frame["delta_sharpe_vs_stronger"] > 0.0).sum()
            ),
            "wins_vs_causal_single": int(
                (frame["delta_sharpe_vs_causal_single"] > 0.0).sum()
            ),
            "mean_fallback_day_rate": float(frame["fallback_day_rate"].mean()),
            "mean_selected_blocks": float(frame["selected_blocks"].mean()),
            "mean_selected_tau": float(frame["mean_selected_tau"].mean()),
            "mean_threshold_quantile": float(
                frame["mean_threshold_quantile"].mean()
            ),
            "mean_branch_divergence_rate": float(
                frame["branch_divergence_rate"].mean()
            ),
            "mean_ensemble_mode_hit_rate": float(
                frame["ensemble_mode_hit_rate"].mean()
            ),
        }
        for metric in METRICS:
            mean, sd, low, high = mean_sd_ci(frame[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        for comparison in comparisons:
            mean, sd, low, high = mean_sd_ci(frame[comparison])
            row[f"{comparison}_mean"] = mean
            row[f"{comparison}_sd"] = sd
            row[f"{comparison}_ci_low"] = low
            row[f"{comparison}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["period", "pair", "classifier_group"]
    ).reset_index(drop=True)


def aggregate_and_save(
    *,
    output: Path,
    results: dict[tuple[str, int, int], SequenceResult],
    args: argparse.Namespace,
    candidate_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_metrics = pd.concat(
        [result.metrics for result in results.values()], ignore_index=True
    )
    metrics = add_paper_comparators(raw_metrics, candidate_root=candidate_root)
    summary = aggregate_results(metrics)
    metrics.to_csv(output / "all_crossover_tau_metrics.csv", index=False)
    summary.to_csv(output / "crossover_tau_summary.csv", index=False)
    all_decisions = pd.concat(
        [result.decisions for result in results.values()], ignore_index=True
    )
    all_decisions.to_csv(output / "all_daily_crossover_tau_decisions.csv", index=False)
    save_feedback_payload(
        output / "all_daily_crossover_tau_feedback.npz", all_decisions
    )
    block_audit = pd.concat(
        [result.expert_state_audit for result in results.values()], ignore_index=True
    )
    block_audit.to_csv(output / "all_block_tau_audit.csv", index=False)
    pd.concat(
        [result.classifier_audit for result in results.values()], ignore_index=True
    ).to_csv(output / "all_classifier_refit_audit.csv", index=False)

    validation: dict[str, object] = {"passed": True, "pairs": {}}
    for period in EVALUATION_PERIODS:
        for pair in args.pairs:
            selected = [
                results[(pair, repeat, group)]
                for repeat in range(args.repetitions)
                for group in args.groups
            ]
            curves = np.asarray(
                [
                    [results[(pair, repeat, group)].curves[period] for group in args.groups]
                    for repeat in range(args.repetitions)
                ]
            )
            dates = selected[0].dates[period]
            pair_output = output / period / pair
            pair_output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                pair_output / "all_crossover_tau_account_curves.npz",
                crossover_tau=curves,
                dates=dates,
                classifier_groups=np.asarray(args.groups),
            )
            pair_metrics = metrics.loc[
                (metrics["period"].astype(str) == period) & (metrics["pair"] == pair)
            ]
            pair_metrics.to_csv(pair_output / "all_crossover_tau_metrics.csv", index=False)
            pair_decisions = pd.concat(
                [
                    result.decisions.loc[
                        result.decisions["period"].astype(str) == period
                    ]
                    for result in selected
                ],
                ignore_index=True,
            )
            pair_decisions.to_csv(
                pair_output / "all_daily_crossover_tau_decisions.csv", index=False
            )
            selected_return = np.where(
                pair_decisions["fallback_used"],
                pair_decisions["fallback_return"],
                np.where(
                    pair_decisions["selected_mode"] == "aggressive",
                    pair_decisions["aggressive_return"],
                    pair_decisions["conservative_return"],
                ),
            )
            mode_ok = (
                pair_decisions["fallback_used"]
                | (
                    (pair_decisions["holding_dispersion"] < pair_decisions["selected_tau"])
                    == (pair_decisions["selected_mode"] == "aggressive")
                )
            )
            history_ok = (
                pair_decisions["history_end_before"].fillna("").eq("")
                | (
                    pair_decisions["history_end_before"].astype(str)
                    < pair_decisions["date"].astype(str)
                )
            )
            audit = {
                "curve_shape": list(curves.shape),
                "metric_rows": len(pair_metrics),
                "decision_rows": len(pair_decisions),
                "max_selected_return_error": float(
                    np.max(
                        np.abs(
                            pair_decisions["actual_master_return"].to_numpy(dtype=float)
                            - selected_return
                        )
                    )
                ),
                "mode_mismatches": int((~mode_ok).sum()),
                "history_date_violations": int((~history_ok).sum()),
            }
            audit["passed"] = bool(
                curves.shape
                == (args.repetitions, len(args.groups), len(dates))
                and len(pair_metrics) == args.repetitions * len(args.groups)
                and audit["max_selected_return_error"] <= 1e-15
                and audit["mode_mismatches"] == 0
                and audit["history_date_violations"] == 0
            )
            validation["pairs"][f"{period}/{pair}"] = audit
            validation["passed"] = validation["passed"] and audit["passed"]
    (output / "experiment_validation_audit.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    if not validation["passed"]:
        raise ValueError("crossover tau validation audit failed")
    return metrics, summary


def write_report(
    output: Path,
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    block_audit: pd.DataFrame,
    *,
    require_positive_lcb: bool,
) -> None:
    comparison_columns = {
        "stronger full-period single RL": "delta_sharpe_vs_stronger",
        "causal rolling-validation single RL": "delta_sharpe_vs_causal_single",
    }
    comparison_lines = []
    for label, column in comparison_columns.items():
        delta = metrics[column]
        comparison_lines.append(
            f"- Beat {label} on {int((delta > 1e-12).sum())}/{len(delta)} paths; "
            f"lost {int((delta < -1e-12).sum())}; tied "
            f"{int(np.isclose(delta, 0.0, rtol=1e-10, atol=1e-10).sum())}."
        )
    evaluation_blocks = block_audit.loc[
        block_audit["period"].astype(str).isin(EVALUATION_PERIODS)
    ]
    status_counts = evaluation_blocks["status"].value_counts().sort_index()
    lines = [
        "# Causal Quantile-Crossover Tau Experiment",
        "",
        "## Protocol",
        "",
        "- 2019 is collection-only warm-up. Every warm-up decision uses the causal rolling-validation single RL while both classifier modes receive same-state one-step net-return feedback.",
        f"- At each later block start, at most {MAX_FIT_BLOCKS} completed blocks are used. Dispersion is mapped to its empirical percentile using at most {CDF_LOOKBACK_DAYS} past decisions.",
        "- A decreasing isotonic curve estimates aggressive-minus-conservative return as a function of the dispersion percentile. Its zero crossing is mapped to the original 0.01--0.89 tau grid and frozen for the complete next block.",
        f"- A fit needs at least {MIN_HISTORY_BLOCKS} completed blocks, {MIN_INFORMATIVE_DAYS} divergent-mode days, and {MIN_SIDE_DAYS} observations on each side of the crossing.",
        (
            f"- The guarded version additionally requires a positive one-sided {CONFIDENCE_LEVEL:.0%} Student-t lower bound for historical block-level log-return advantage over the causal single-RL fallback."
            if require_positive_lcb
            else "- The mechanism-only version does not use the policy-advantage LCB guardrail; it falls back only when no adequately supported mode crossover exists."
        ),
        "- Current-block outcomes are added only after the block finishes. RL checkpoints, deterministic holdings, rolling classifier fits, voting, account state, and transaction costs are unchanged.",
        "",
        "## Selection Status",
        "",
        *[f"- {status}: {int(count)} blocks" for status, count in status_counts.items()],
        "",
        "## Path Comparisons",
        "",
        *comparison_lines,
        "",
        "## Configuration Results",
        "",
        markdown_table(
            summary,
            [
                "period",
                "pair",
                "classifier_group",
                "sharpe_mean",
                "stronger_single_sharpe",
                "delta_sharpe_vs_stronger_mean",
                "delta_sharpe_vs_causal_single_mean",
                "mean_fallback_day_rate",
                "mean_selected_blocks",
                "mean_selected_tau",
                "mean_threshold_quantile",
            ],
        ),
        "",
        "The 30 repetitions vary rolling classifier fits conditional on fixed RL checkpoints and one market path; they are not independent market samples.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    candidate_root = Path(args.candidate_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    require_positive_lcb = not args.disable_lcb_guardrail
    results: dict[tuple[str, int, int], SequenceResult] = {}

    for pair in args.pairs:
        print(f"PREPARE {pair}")
        inputs_by_period = {
            period: load_period_inputs(candidate_root, period, pair)
            for period in ALL_PERIODS
        }
        tau_values = load_tau_grid(candidate_root, pair)
        pending: list[tuple[tuple[str, int, int], Path]] = []
        for repeat in range(args.repetitions):
            for group in args.groups:
                key = (pair, repeat, group)
                run_dir = (
                    output
                    / "runs"
                    / pair
                    / f"repeat_{repeat:02d}"
                    / f"group_{group}"
                )
                if args.resume and completed_run_exists(run_dir):
                    print(f"LOAD {pair} repeat {repeat:02d} group {group}")
                    results[key] = load_sequence_result(run_dir)
                else:
                    pending.append((key, run_dir))

        if args.workers == 1:
            for key, run_dir in pending:
                print(f"RUN  {key[0]} repeat {key[1]:02d} group {key[2]}")
                results[key] = simulate_sequence(
                    pair=key[0],
                    repeat=key[1],
                    group=key[2],
                    inputs_by_period=inputs_by_period,
                    tau_values=tau_values,
                    require_positive_lcb=require_positive_lcb,
                    run_dir=run_dir,
                )
        elif pending:
            print(f"PARALLEL {pair}: {len(pending)} sequences, {args.workers} workers")
            context = mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers, mp_context=context
            ) as executor:
                futures = {
                    executor.submit(
                        simulate_sequence,
                        pair=key[0],
                        repeat=key[1],
                        group=key[2],
                        inputs_by_period=inputs_by_period,
                        tau_values=tau_values,
                        require_positive_lcb=require_positive_lcb,
                        run_dir=run_dir,
                    ): key
                    for key, run_dir in pending
                }
                for future in concurrent.futures.as_completed(futures):
                    key = futures[future]
                    results[key] = future.result()
                    print(f"DONE {key[0]} repeat {key[1]:02d} group {key[2]}")

    metrics, summary = aggregate_and_save(
        output=output,
        results=results,
        args=args,
        candidate_root=candidate_root,
    )
    block_audit = pd.read_csv(output / "all_block_tau_audit.csv")
    manifest = {
        "experiment": "causal_quantile_mode_advantage_crossover_tau",
        "candidate_root": str(candidate_root),
        "warmup_period": WARMUP_PERIOD,
        "evaluation_periods": list(EVALUATION_PERIODS),
        "pairs": args.pairs,
        "classifier_groups": args.groups,
        "repetitions": args.repetitions,
        "max_fit_blocks": MAX_FIT_BLOCKS,
        "cdf_lookback_days": CDF_LOOKBACK_DAYS,
        "min_history_blocks": MIN_HISTORY_BLOCKS,
        "min_informative_days": MIN_INFORMATIVE_DAYS,
        "min_side_days": MIN_SIDE_DAYS,
        "confidence_level": CONFIDENCE_LEVEL,
        "require_positive_lcb": require_positive_lcb,
        "mode_gap_target": "same_master_state_aggressive_return_minus_conservative_return",
        "fit": "decreasing_isotonic_regression_on_causal_dispersion_percentile",
        "threshold": "zero_crossing_mapped_to_original_tau_grid_and_frozen_next_block",
        "fallback": "current_block_prior_validation_selected_single_rl",
        "policy_guardrail": "one_sided_block_log_return_advantage_lcb",
        "rl_checkpoints_retrained": False,
        "rl_inference": "deterministic",
        "classifier_refit": "rolling_previous_block_original_seed",
        "classifier_grid_search": False,
        "transaction_costs_preserved": True,
        "future_information_used": False,
        "tuned_hyperparameters": [],
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    write_report(
        output,
        summary,
        metrics,
        block_audit,
        require_positive_lcb=require_positive_lcb,
    )
    print(f"COMPLETE: {output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
