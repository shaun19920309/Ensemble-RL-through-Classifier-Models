from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_COLUMNS = (
    "cumulative_return",
    "annualized_return",
    "sharpe",
    "calmar",
    "max_drawdown",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently audit completed model-family experiment outputs."
    )
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional combined JSON output. Per-directory audits are always written.",
    )
    return parser.parse_args()


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def expected_pairs(manifest: dict[str, object]) -> list[str]:
    pair_components = manifest.get("pair_components")
    if isinstance(pair_components, dict):
        return list(pair_components)
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("manifest does not define pairs")
    return [str(pair) for pair in pairs]


def audit_windows(
    windows: pd.DataFrame, manifest: dict[str, object], errors: list[str]
) -> None:
    require(len(windows) == 4, "rolling window count is not four", errors)
    if len(windows) != 4:
        return
    require(
        windows["train_start"].nunique() == 1,
        "expanding windows do not share one training start",
        errors,
    )
    require(
        windows["train_dates"].is_monotonic_increasing,
        "expanding training lengths are not monotone",
        errors,
    )
    for row_index, row in windows.reset_index(drop=True).iterrows():
        expected_source = "train_tail" if row_index == 0 else "previous_trade"
        require(
            row["calibration_source"] == expected_source,
            f"window {row_index + 1} has the wrong calibration source",
            errors,
        )
        require(
            str(row["train_end"]) < str(row["calibration_start"]),
            f"window {row_index + 1} training overlaps calibration",
            errors,
        )
        require(
            str(row["calibration_end"]) < str(row["trade_start"]),
            f"window {row_index + 1} calibration overlaps future trading",
            errors,
        )
        if row_index:
            previous = windows.iloc[row_index - 1]
            require(
                str(row["calibration_start"]) == str(previous["trade_start"])
                and str(row["calibration_end"]) == str(previous["trade_end"]),
                f"window {row_index + 1} does not calibrate on the previous trade block",
                errors,
            )
    manifest_windows = pd.DataFrame(manifest["window_boundaries"])
    require(
        windows.astype(str).equals(manifest_windows[windows.columns].astype(str)),
        "rolling_windows.csv differs from the manifest",
        errors,
    )


def audit_root(root: Path) -> dict[str, object]:
    root = root.resolve()
    errors: list[str] = []
    manifest = json.loads((root / "experiment_manifest.json").read_text())
    pairs = expected_pairs(manifest)
    checkpoint_hashes: dict[str, str] = {}
    repetitions = int(manifest["repetitions"])
    groups = [int(group) for group in manifest["classifier_groups"]]
    tau = np.asarray(manifest["tau_values"], dtype=float)
    expected_tau = np.arange(0.01, 0.90, 0.01)

    require(repetitions == 30, "repetition count is not 30", errors)
    require(groups == [1, 2, 3, 4, 5], "classifier groups are incomplete", errors)
    require(
        len(tau) == 89 and np.allclose(tau, expected_tau),
        "tau grid is not 0.01 through 0.89",
        errors,
    )
    require(
        bool(manifest.get("fixed_global_tau_per_path")),
        "tau is not declared fixed over the full path",
        errors,
    )
    require(
        manifest.get("classifier_grid_search") is False,
        "classifier grid search is enabled or unspecified",
        errors,
    )

    windows = pd.read_csv(root / "rolling_windows.csv")
    audit_windows(windows, manifest, errors)

    metrics = pd.read_csv(root / "all_classifier_refit_metrics.csv")
    expected_metric_rows = repetitions * len(pairs) * len(groups) * len(tau)
    require(
        len(metrics) == expected_metric_rows,
        f"metric row count {len(metrics)} != {expected_metric_rows}",
        errors,
    )
    require(
        sorted(metrics["repeat"].unique().tolist()) == list(range(repetitions)),
        "metric repetitions are incomplete",
        errors,
    )
    require(
        sorted(metrics["pair"].unique().tolist()) == sorted(pairs),
        "metric pairs differ from the manifest",
        errors,
    )
    require(
        sorted(metrics["classifier_group"].unique().tolist()) == groups,
        "metric classifier groups are incomplete",
        errors,
    )
    require(
        np.allclose(np.sort(metrics["tau"].unique()), expected_tau),
        "metric tau values are incomplete",
        errors,
    )
    require(
        np.isfinite(metrics[list(METRIC_COLUMNS)].to_numpy(dtype=float)).all(),
        "metrics contain non-finite values",
        errors,
    )

    fits = pd.read_csv(root / "classifier_refit_audit.csv")
    expected_fits = repetitions * len(pairs) * len(groups) * len(windows)
    require(
        len(fits) == expected_fits,
        f"classifier fit count {len(fits)} != {expected_fits}",
        errors,
    )
    fit_keys = ["repeat", "window", "pair", "classifier_group"]
    require(
        not fits.duplicated(fit_keys).any(),
        "classifier audit contains duplicate fit keys",
        errors,
    )
    require(
        (fits["calibration_end"].astype(str) < fits["trade_start"].astype(str)).all(),
        "classifier audit contains calibration/trade leakage",
        errors,
    )

    recomputed = (
        metrics.groupby(["pair", "classifier_group", "tau"], as_index=False)["sharpe"]
        .mean()
        .rename(columns={"sharpe": "recomputed_sharpe"})
        .sort_values(
            ["pair", "classifier_group", "recomputed_sharpe", "tau"],
            ascending=[True, True, False, True],
        )
        .groupby(["pair", "classifier_group"], as_index=False)
        .head(1)
        .sort_values(["pair", "classifier_group"])
        .reset_index(drop=True)
    )
    selected = (
        pd.read_csv(root / "selected_tau_summary.csv")
        .sort_values(["pair", "classifier_group"])
        .reset_index(drop=True)
    )
    require(
        len(selected) == len(pairs) * len(groups),
        "selected-tau summary has the wrong number of configurations",
        errors,
    )
    if len(selected) == len(recomputed):
        require(
            np.allclose(selected["selected_global_tau"], recomputed["tau"]),
            "selected tau is not the mean full-path Sharpe argmax",
            errors,
        )
        require(
            np.allclose(
                selected["ensemble_sharpe_mean"], recomputed["recomputed_sharpe"]
            ),
            "selected Sharpe differs from an independent aggregation",
            errors,
        )

    paired = pd.read_csv(root / "selected_tau_paired_runs.csv")
    require(
        len(paired) == repetitions * len(pairs) * len(groups),
        "selected-tau paired run count is incomplete",
        errors,
    )

    with np.load(root / "all_ensemble_account_curves.npz") as arrays:
        curves = arrays["ensemble"]
        dates = arrays["dates"]
    expected_curve_shape = (
        repetitions,
        len(pairs) * len(groups),
        len(tau),
        int(windows["trade_dates"].sum()),
    )
    require(
        curves.shape == expected_curve_shape,
        f"account curve shape {curves.shape} != {expected_curve_shape}",
        errors,
    )
    require(len(dates) == expected_curve_shape[-1], "curve dates are incomplete", errors)
    require(np.isfinite(curves).all(), "account curves contain non-finite values", errors)

    is_rl = manifest.get("experiment_group") == "expanded_deep_rl_pair"
    if is_rl:
        require(
            manifest.get("training_timesteps_per_model_window") == 100_000,
            "RL training is not 100,000 steps per model-window",
            errors,
        )
        require(
            manifest.get("rl_evaluation_interval") == 20_000,
            "RL checkpoint interval is not 20,000 steps",
            errors,
        )
        require(
            manifest.get("deterministic_rl_inference") is True,
            "RL inference is not deterministic",
            errors,
        )
        require(
            manifest.get("rl_training_window") == "expanding",
            "RL training is not expanding",
            errors,
        )
        require(
            manifest.get("classifier_training_window") == "rolling_previous_block",
            "RL classifier window is not the previous block",
            errors,
        )
        checkpoints = manifest.get("checkpoints", [])
        require(
            len(checkpoints) == len(manifest["models"]) * len(windows),
            "RL checkpoint count is incomplete",
            errors,
        )
        for checkpoint in checkpoints:
            checkpoint_key = f"{checkpoint['model']}:w{int(checkpoint['window'])}"
            checkpoint_hashes[checkpoint_key] = str(checkpoint["sha256"])
            step = int(checkpoint["selected_validation_step"])
            require(
                step in range(20_000, 100_001, 20_000),
                "RL selected checkpoint is outside the validation grid",
                errors,
            )
            require(
                checkpoint.get("deterministic_inference") is True,
                "RL checkpoint metadata is not deterministic",
                errors,
            )
            require(
                Path(checkpoint["checkpoint"]).exists(),
                "RL checkpoint file is missing",
                errors,
            )
            if "imported_policy_parameters_match" in checkpoint:
                require(
                    checkpoint["imported_policy_parameters_match"] is True,
                    "imported RL policy parameters do not match",
                    errors,
                )
        sources = manifest.get("pretrained_model_sources", {})
        if manifest.get("experiment_role") == "main":
            require(
                set(sources) == set(manifest["models"]),
                "main RL pair does not import both audited same-dataset candidates",
                errors,
            )
            require(
                all(item.get("imported_checkpoint_source") for item in checkpoints),
                "main RL pair contains a checkpoint without import provenance",
                errors,
            )
        if manifest.get("experiment_role") == "stress_test":
            require(not sources, "stress RL pair unexpectedly imports candidates", errors)
            require(
                all(item.get("checkpoint_import_mode") == "fresh_training" for item in checkpoints),
                "stress RL pair contains a non-fresh checkpoint",
                errors,
            )

    result = {
        "root": str(root),
        "dataset_label": manifest.get("dataset_label"),
        "experiment_group": manifest.get("experiment_group"),
        "pairs": pairs,
        "repetitions": repetitions,
        "tau_count": len(tau),
        "window_count": len(windows),
        "evaluation_sessions": int(windows["trade_dates"].sum()),
        "metric_rows": len(metrics),
        "classifier_fits": len(fits),
        "curve_shape": list(curves.shape),
        "checkpoint_hashes": checkpoint_hashes,
        "passed": not errors,
        "errors": errors,
    }
    (root / "independent_result_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    args = parse_args()
    results = [audit_root(root) for root in args.roots]
    collisions: list[dict[str, str]] = []
    for left_index, left in enumerate(results):
        if not left["checkpoint_hashes"]:
            continue
        for right in results[left_index + 1 :]:
            if left["dataset_label"] == right["dataset_label"]:
                continue
            common = set(left["checkpoint_hashes"]).intersection(
                right["checkpoint_hashes"]
            )
            for checkpoint_key in sorted(common):
                if (
                    left["checkpoint_hashes"][checkpoint_key]
                    == right["checkpoint_hashes"][checkpoint_key]
                ):
                    collisions.append(
                        {
                            "checkpoint": checkpoint_key,
                            "left_dataset": str(left["dataset_label"]),
                            "right_dataset": str(right["dataset_label"]),
                            "sha256": left["checkpoint_hashes"][checkpoint_key],
                        }
                    )
    combined = {
        "passed": all(result["passed"] for result in results) and not collisions,
        "cross_dataset_checkpoint_hash_collisions": collisions,
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(json.dumps(combined, indent=2))
    if not combined["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
