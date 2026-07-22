from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from finrl.reproduction.causal_crossover_tau import CausalCrossoverTau
from finrl.reproduction.metrics import metrics_from_account_values


DECISION_FLOAT_COLUMNS = (
    "selected_tau",
    "threshold_quantile",
    "raw_threshold",
    "low_mode_advantage",
    "high_mode_advantage",
    "policy_advantage_mean",
    "policy_advantage_lcb",
)
METRICS = ("cumulative_return", "annualized_return", "sharpe", "calmar", "max_drawdown")
FEEDBACK_COLUMNS = (
    "period",
    "pair",
    "date",
    "selected_mode",
    "repeat",
    "classifier_group",
    "window",
    "decision_index",
    "aggressive_agent_index",
    "conservative_agent_index",
    "fallback_used",
    "branch_diverged",
    "selected_tau",
    "holding_dispersion",
    "aggressive_return",
    "conservative_return",
    "fallback_return",
    "actual_master_return",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently replay and audit the causal crossover-tau outputs."
    )
    parser.add_argument(
        "--results-dir", default="work/causal_tau_guarded"
    )
    return parser.parse_args()


def equal_float(left: object, right: object, *, atol: float = 1e-12) -> bool:
    left_value = float(left)
    right_value = float(right)
    if np.isnan(left_value) and np.isnan(right_value):
        return True
    if np.isinf(left_value) or np.isinf(right_value):
        return left_value == right_value
    return bool(np.isclose(left_value, right_value, rtol=0.0, atol=atol))


def make_controller(manifest: dict[str, object], tau_values: np.ndarray) -> CausalCrossoverTau:
    return CausalCrossoverTau(
        tau_values,
        max_fit_blocks=int(manifest["max_fit_blocks"]),
        cdf_lookback_days=int(manifest["cdf_lookback_days"]),
        min_history_blocks=int(manifest["min_history_blocks"]),
        min_informative_days=int(manifest["min_informative_days"]),
        min_side_days=int(manifest["min_side_days"]),
        confidence_level=float(manifest["confidence_level"]),
        require_positive_lcb=bool(manifest["require_positive_lcb"]),
    )


def restore_lossless_feedback(results: Path, decisions: pd.DataFrame) -> pd.DataFrame:
    payload = results / "all_daily_crossover_tau_feedback.npz"
    restored = decisions.copy()
    with np.load(payload, allow_pickle=False) as arrays:
        for column in FEEDBACK_COLUMNS:
            if len(arrays[column]) != len(restored):
                raise ValueError("lossless feedback payload does not align with CSV rows")
            restored[column] = arrays[column]
    return restored


def audit_controller_replay(
    decisions: pd.DataFrame,
    blocks: pd.DataFrame,
    manifest: dict[str, object],
) -> dict[str, object]:
    tau_values = np.arange(0.01, 0.90, 0.01)
    replay_mismatches = 0
    selected_return_error = 0.0
    mode_mismatches = 0
    history_date_violations = 0
    chronology_violations = 0
    block_update_mismatches = 0
    audited_blocks = 0
    audited_days = 0
    mismatch_fields: Counter[str] = Counter()
    mismatch_examples: list[dict[str, object]] = []

    def record_mismatch(
        field: str,
        *,
        pair: str,
        repeat: int,
        group: int,
        period: str,
        window: int,
        replayed: object,
        saved: object,
    ) -> None:
        nonlocal replay_mismatches
        replay_mismatches += 1
        mismatch_fields[field] += 1
        if len(mismatch_examples) < 20:
            mismatch_examples.append(
                {
                    "pair": pair,
                    "repeat": int(repeat),
                    "classifier_group": int(group),
                    "period": period,
                    "window": window,
                    "field": field,
                    "replayed": replayed,
                    "saved": saved,
                }
            )

    sequence_keys = ["pair", "repeat", "classifier_group"]
    for key, sequence_blocks in blocks.groupby(sequence_keys, sort=True):
        pair, repeat, group = key
        controller = make_controller(manifest, tau_values)
        sequence_blocks = sequence_blocks.assign(
            period_order=sequence_blocks["period"].astype(int)
        ).sort_values(["period_order", "window"])
        sequence_days = decisions.loc[
            (decisions["pair"] == pair)
            & (decisions["repeat"] == repeat)
            & (decisions["classifier_group"] == group)
        ].copy()

        for _, block in sequence_blocks.iterrows():
            period = str(int(block["period"]))
            window = int(block["window"])
            day = sequence_days.loc[
                (sequence_days["period"].astype(str) == period)
                & (sequence_days["window"] == window)
            ].sort_values("decision_index")
            if len(day) != int(block["block_decision_days"]):
                block_update_mismatches += 1
                continue
            audited_blocks += 1
            audited_days += len(day)

            snapshot = controller.snapshot()
            if int(snapshot["completed_blocks"]) != int(block["history_blocks_before"]):
                chronology_violations += 1
            if int(snapshot["completed_days"]) != int(block["history_days_before"]):
                chronology_violations += 1
            saved_history_end = "" if pd.isna(block["history_end_before"]) else str(
                block["history_end_before"]
            )
            if str(snapshot["last_completed_date"]) != saved_history_end:
                chronology_violations += 1
            if saved_history_end and saved_history_end >= str(block["trade_start"]):
                history_date_violations += 1

            if period == str(manifest["warmup_period"]):
                if str(block["status"]) != "warmup_collection":
                    record_mismatch(
                        "status",
                        pair=pair,
                        repeat=repeat,
                        group=group,
                        period=period,
                        window=window,
                        replayed="warmup_collection",
                        saved=block["status"],
                    )
            else:
                replay = controller.select_for_next_block().to_dict()
                if replay["status"] != str(block["status"]):
                    record_mismatch(
                        "status",
                        pair=pair,
                        repeat=repeat,
                        group=group,
                        period=period,
                        window=window,
                        replayed=replay["status"],
                        saved=block["status"],
                    )
                for column in DECISION_FLOAT_COLUMNS:
                    if not equal_float(replay[column], block[column]):
                        record_mismatch(
                            column,
                            pair=pair,
                            repeat=repeat,
                            group=group,
                            period=period,
                            window=window,
                            replayed=replay[column],
                            saved=block[column],
                        )
                for column in (
                    "history_blocks",
                    "history_days",
                    "informative_days",
                    "cdf_days",
                ):
                    if int(replay[column]) != int(block[column]):
                        replay_mismatches += 1
                for column in ("fit_start_date", "fit_end_date"):
                    saved = "" if pd.isna(block[column]) else str(block[column])
                    if str(replay[column]) != saved:
                        record_mismatch(
                            column,
                            pair=pair,
                            repeat=repeat,
                            group=group,
                            period=period,
                            window=window,
                            replayed=replay[column],
                            saved=saved,
                        )

            expected_return = np.where(
                day["fallback_used"].astype(bool),
                day["fallback_return"],
                np.where(
                    day["selected_mode"] == "aggressive",
                    day["aggressive_return"],
                    day["conservative_return"],
                ),
            )
            selected_return_error = max(
                selected_return_error,
                float(
                    np.max(
                        np.abs(
                            expected_return
                            - day["actual_master_return"].to_numpy(dtype=float)
                        )
                    )
                ),
            )
            selected = ~day["fallback_used"].astype(bool)
            expected_mode = np.where(
                day["holding_dispersion"] < day["selected_tau"],
                "aggressive",
                "conservative",
            )
            mode_mismatches += int(
                (day.loc[selected, "selected_mode"] != expected_mode[selected]).sum()
            )
            diverged = (
                day["aggressive_agent_index"].to_numpy(dtype=int)
                != day["conservative_agent_index"].to_numpy(dtype=int)
            )
            if not np.array_equal(diverged, day["branch_diverged"].astype(bool)):
                block_update_mismatches += 1

            controller.add_completed_block(
                block_id=str(block["block_id"]),
                start_date=str(block["trade_start"]),
                end_date=str(block["feedback_end"]),
                dispersion=day["holding_dispersion"].to_numpy(dtype=float),
                aggressive_return=day["aggressive_return"].to_numpy(dtype=float),
                conservative_return=day["conservative_return"].to_numpy(dtype=float),
                fallback_return=day["fallback_return"].to_numpy(dtype=float),
                branch_diverged=day["branch_diverged"].to_numpy(dtype=bool),
            )
            after = controller.snapshot()
            if int(after["completed_blocks"]) != int(block["history_blocks_after"]):
                block_update_mismatches += 1
            if int(after["completed_days"]) != int(block["history_days_after"]):
                block_update_mismatches += 1
            if str(after["last_completed_date"]) != str(block["history_end_after"]):
                block_update_mismatches += 1

    passed = bool(
        replay_mismatches == 0
        and selected_return_error <= 1e-15
        and mode_mismatches == 0
        and history_date_violations == 0
        and chronology_violations == 0
        and block_update_mismatches == 0
    )
    return {
        "passed": passed,
        "audited_blocks": audited_blocks,
        "audited_days": audited_days,
        "selection_replay_mismatches": replay_mismatches,
        "selection_mismatch_fields": dict(mismatch_fields),
        "selection_mismatch_examples": mismatch_examples,
        "max_selected_return_error": selected_return_error,
        "mode_mismatches": mode_mismatches,
        "history_date_violations": history_date_violations,
        "chronology_violations": chronology_violations,
        "block_update_mismatches": block_update_mismatches,
    }


def audit_metrics(results: Path, metrics: pd.DataFrame) -> dict[str, object]:
    max_error = 0.0
    audited_paths = 0
    for period in sorted(metrics["period"].astype(str).unique()):
        for pair in sorted(metrics["pair"].unique()):
            path = results / period / pair / "all_crossover_tau_account_curves.npz"
            if not path.exists():
                continue
            with np.load(path) as arrays:
                curves = arrays["crossover_tau"].astype(float)
                groups = arrays["classifier_groups"].astype(int)
            for repeat in range(curves.shape[0]):
                for group_index, group in enumerate(groups):
                    saved = metrics.loc[
                        (metrics["period"].astype(str) == period)
                        & (metrics["pair"] == pair)
                        & (metrics["repeat"] == repeat)
                        & (metrics["classifier_group"] == group)
                    ]
                    if len(saved) != 1:
                        raise ValueError("metric key is missing or duplicated")
                    recomputed = metrics_from_account_values(curves[repeat, group_index])
                    for metric in METRICS:
                        max_error = max(
                            max_error,
                            abs(float(saved.iloc[0][metric]) - recomputed[metric]),
                        )
                    audited_paths += 1
    return {
        "passed": bool(max_error <= 1e-12),
        "audited_paths": audited_paths,
        "max_metric_error": max_error,
    }


def main() -> None:
    args = parse_args()
    results = Path(args.results_dir).expanduser().resolve()
    manifest = json.loads((results / "experiment_manifest.json").read_text())
    decisions = pd.read_csv(results / "all_daily_crossover_tau_decisions.csv")
    decisions = restore_lossless_feedback(results, decisions)
    blocks = pd.read_csv(results / "all_block_tau_audit.csv")
    metrics = pd.read_csv(results / "all_crossover_tau_metrics.csv")
    decisions["period"] = decisions["period"].astype(str)
    blocks["period"] = blocks["period"].astype(str)
    metrics["period"] = metrics["period"].astype(str)

    classifier = pd.read_csv(results / "all_classifier_refit_audit.csv")
    classifier_date_violations = int(
        (classifier["calibration_end"].astype(str) >= classifier["trade_start"].astype(str)).sum()
    )
    replay = audit_controller_replay(decisions, blocks, manifest)
    metric_audit = audit_metrics(results, metrics)
    audit = {
        "passed": bool(
            replay["passed"]
            and metric_audit["passed"]
            and classifier_date_violations == 0
        ),
        "controller_replay": replay,
        "metric_recomputation": metric_audit,
        "classifier_date_violations": classifier_date_violations,
        "causality_unit_test": (
            "unit_tests/test_causal_crossover_tau.py::"
            "test_unadded_future_outcomes_cannot_change_prior_decision"
        ),
    }
    (results / "independent_validation_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    report = [
        "# Independent Crossover-Tau Validation",
        "",
        f"- Overall audit passed: **{audit['passed']}**.",
        f"- Replayed blocks: **{replay['audited_blocks']}**.",
        f"- Replayed same-state daily feedback rows: **{replay['audited_days']}**.",
        f"- Selection replay mismatches: **{replay['selection_replay_mismatches']}**.",
        f"- History-date violations: **{replay['history_date_violations']}**.",
        f"- Classifier-date violations: **{classifier_date_violations}**.",
        f"- Maximum selected-return reconstruction error: **{replay['max_selected_return_error']:.3e}**.",
        f"- Recomputed annual paths: **{metric_audit['audited_paths']}**.",
        f"- Maximum financial-metric error: **{metric_audit['max_metric_error']:.3e}**.",
        "",
        "Every saved block-start decision was reconstructed from feedback ending before that block. Current-block outcomes were appended only after all decisions in the block had been executed.",
    ]
    (results / "VALIDATION_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    if not audit["passed"]:
        raise ValueError("independent crossover-tau audit failed")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
