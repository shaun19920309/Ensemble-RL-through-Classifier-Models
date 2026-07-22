from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PAIRS = {"a2c_ppo", "a2c_sac", "ppo_sac"}
MODELS = {"a2c", "ppo", "sac"}
GROUPS = {1, 2, 3, 4, 5}
METRICS = ["cumulative_return", "sharpe", "calmar", "max_drawdown"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently audit a fixed-RL 30-backtest result directory."
    )
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument(
        "--allow-omitted-checkpoints",
        action="store_true",
        help="Audit a compact public package whose checkpoint binaries are intentionally omitted.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    args = parse_args()
    root = Path(args.result_dir)
    tau_values = np.round(np.arange(0.01, 0.90, 0.01), 2)
    expected_metric_rows = args.repetitions * len(PAIRS) * len(GROUPS) * len(
        tau_values
    )

    metrics = pd.read_csv(root / "all_backtest_metrics.csv")
    base = pd.read_csv(root / "all_base_metrics.csv")
    classifiers = pd.read_csv(root / "classifier_refit_audit.csv")
    manifest = pd.read_csv(root / "fixed_rl_checkpoint_manifest.csv")
    windows = pd.read_csv(root / "rolling_windows.csv")
    selected = pd.read_csv(root / "selected_tau_summary.csv")
    comparison = pd.read_csv(root / "configuration_comparison.csv")
    coverage = pd.read_csv(root / "figure3_range_coverage.csv")
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))

    run_ids = set(range(args.repetitions))
    require(len(metrics) == expected_metric_rows, "ensemble metric row count mismatch")
    require(set(metrics["repeat"]) == run_ids, "ensemble repeat ids are incomplete")
    require(set(metrics["pair"]) == PAIRS, "ensemble RL pairs are incomplete")
    require(set(metrics["classifier_group"]) == GROUPS, "classifier groups are incomplete")
    require(
        np.array_equal(np.sort(metrics["tau"].unique()), tau_values),
        "tau grid is incomplete",
    )
    per_cell = metrics.groupby(["pair", "classifier_group", "tau"]).size()
    require(bool((per_cell == args.repetitions).all()), "tau cells are not fully paired")

    require(len(base) == args.repetitions * len(MODELS), "base metric row count mismatch")
    require(set(base["repeat"]) == run_ids, "base repeat ids are incomplete")
    require(set(base["model"]) == MODELS, "base models are incomplete")
    base_ranges = base.groupby("model")[METRICS].agg(lambda values: values.max() - values.min())
    require(float(base_ranges.to_numpy().max()) <= 1e-12, "deterministic base paths vary")

    expected_refits = args.repetitions * len(windows) * len(PAIRS) * len(GROUPS)
    require(len(classifiers) == expected_refits, "classifier refit row count mismatch")
    require(set(classifiers["repeat"]) == run_ids, "classifier repeat ids are incomplete")
    require(set(classifiers["pair"]) == PAIRS, "classifier RL pairs are incomplete")
    require(set(classifiers["classifier_group"]) == GROUPS, "classifier groups are incomplete")
    require(classifiers["classifier_seed"].is_unique, "classifier seed collision detected")

    require(len(windows) == 4, "expected four rolling windows")
    require(windows.iloc[0]["calibration_source"] == "train_tail", "window 1 source mismatch")
    require(
        bool((windows.iloc[1:]["calibration_source"] == "previous_trade").all()),
        "later calibration source mismatch",
    )
    for index in range(1, len(windows)):
        previous = windows.iloc[index - 1]
        current = windows.iloc[index]
        require(current["calibration_start"] == previous["trade_start"], "rolling start mismatch")
        require(current["calibration_end"] == previous["trade_end"], "rolling end mismatch")
        require(current["train_end"] < current["calibration_start"], "RL/calibration leakage")
        require(current["calibration_end"] < current["trade_start"], "calibration/trade leakage")

    require(len(manifest) == len(windows) * len(MODELS), "checkpoint manifest mismatch")
    require(set(manifest["model"]) == MODELS, "checkpoint models are incomplete")
    require(bool((manifest["training_seed"] == 42).all()), "unexpected RL training seed")
    require(
        not manifest["retrained_in_backtests"].astype(bool).any(),
        "RL was marked as retrained during repetitions",
    )
    for row in manifest.itertuples(index=False):
        checkpoint = Path(row.checkpoint)
        if checkpoint.exists():
            require(sha256_file(checkpoint) == row.sha256, f"checkpoint hash mismatch: {checkpoint}")
        else:
            require(
                args.allow_omitted_checkpoints,
                f"missing checkpoint: {checkpoint}",
            )

    require(len(selected) == len(PAIRS) * len(GROUPS), "selected result count mismatch")
    require(bool((selected["n_backtests"] == args.repetitions).all()), "selected n mismatch")
    require(set(np.round(selected["selected_global_tau"], 2)).issubset(set(tau_values)), "selected tau outside grid")
    require(len(comparison) == len(PAIRS) * len(GROUPS), "configuration comparison mismatch")
    require(len(coverage) == len(PAIRS), "Figure 3 range coverage mismatch")

    expected_run_files = {
        "ensemble_metrics.csv",
        "base_metrics.csv",
        "classifier_refit_audit.csv",
        "account_curves.npz",
    }
    for repeat in run_ids:
        run_dir = root / "runs" / f"repeat_{repeat:02d}"
        require(run_dir.is_dir(), f"missing run directory: {run_dir}")
        require(
            expected_run_files.issubset({path.name for path in run_dir.iterdir()}),
            f"incomplete run directory: {run_dir}",
        )

    figure_names = [
        "figure3_fixed_rl_30_backtests_yearly_performance",
        "figure4_fixed_rl_30_backtests_classifier_groups",
        "figure5_fixed_rl_30_backtests_group1",
    ]
    for stem in figure_names:
        for suffix in (".png", ".pdf"):
            path = root / "figures" / f"{stem}{suffix}"
            require(path.exists() and path.stat().st_size > 0, f"missing figure: {path}")

    require(metadata["repetitions"] == args.repetitions, "metadata repetition mismatch")
    require(metadata["rl_retrained_in_repetitions"] is False, "metadata says RL retrained")
    require(metadata["stochastic_rl_inference"] is False, "RL inference is not deterministic")
    require(metadata["classifier_refit_each_window"] is True, "classifier refit flag mismatch")
    require(metadata["fixed_global_tau_per_path"] is True, "global tau flag mismatch")
    require(metadata["original_dj30_comparison"] is False, "external comparison flag mismatch")

    audit = {
        "status": "PASS",
        "result_dir": str(root),
        "dataset_label": metadata["dataset_label"],
        "repetitions": args.repetitions,
        "windows": len(windows),
        "trade_dates": int(windows["trade_dates"].sum()),
        "fixed_checkpoints": len(manifest),
        "checkpoint_binaries_included": all(
            Path(path).exists() for path in manifest["checkpoint"]
        ),
        "unique_checkpoint_hashes": int(manifest["sha256"].nunique()),
        "ensemble_metric_rows": len(metrics),
        "classifier_group_refits": len(classifiers),
        "unique_classifier_seeds": int(classifiers["classifier_seed"].nunique()),
        "maximum_base_metric_range": float(base_ranges.to_numpy().max()),
        "selected_configurations": len(selected),
        "tau_count": len(tau_values),
        "figures_verified": len(figure_names) * 2,
    }
    (root / "independent_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
