from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = (
    (
        "SSE50",
        {
            "group1": "results/forecasting_group1_sse50_full208",
            "group2": "results/forecasting_group2_sse50_full208",
            "main": "results/rl_group3_ppo_tqc_sse50_full208",
            "stress": "results/rl_group3_td3_tqc_sse50_full208",
        },
    ),
    (
        "HSTech10",
        {
            "group1": "results/forecasting_group1_hstech10_full206",
            "group2": "results/forecasting_group2_hstech10_full206",
            "main": "results/rl_group3_ppo_tqc_hstech10_full206",
            "stress": "results/rl_group3_td3_tqc_hstech10_full206",
        },
    ),
)

PAIR_ORDER = {
    "group1": ("arima_lstm", "arima_xgboost", "xgboost_lstm"),
    "group2": ("patchtst_itransformer",),
    "main": ("ppo_tqc",),
    "stress": ("td3_tqc",),
}

LABELS = {
    "arima": "ARIMA",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "patchtst": "PatchTST",
    "itransformer": "iTransformer",
    "ppo": "PPO",
    "td3": "TD3",
    "tqc": "TQC",
    "arima_lstm": "ARIMA+LSTM",
    "arima_xgboost": "ARIMA+XGBoost",
    "xgboost_lstm": "XGBoost+LSTM",
    "patchtst_itransformer": "PatchTST+iTransformer",
    "ppo_tqc": "PPO+TQC",
    "td3_tqc": "TD3+TQC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", default=".")
    parser.add_argument(
        "--output-dir",
        default=(
            "../paper/arxiv/arxiv_final/data/experiments/model_family_extensions"
        ),
    )
    return parser.parse_args()


def load_experiments(code_root: Path) -> dict[tuple[str, str], dict[str, object]]:
    experiments: dict[tuple[str, str], dict[str, object]] = {}
    for dataset, roots in DATASETS:
        for family, relative_root in roots.items():
            root = code_root / relative_root
            audit = json.loads((root / "independent_result_audit.json").read_text())
            if not audit["passed"]:
                raise ValueError(f"independent result audit failed: {root}")
            selected = pd.read_csv(root / "selected_tau_summary.csv")
            means = pd.read_csv(root / "mean_metrics_by_fixed_tau.csv")
            selected = selected.merge(
                means[
                    [
                        "pair",
                        "classifier_group",
                        "tau",
                        "cumulative_return_sd",
                        "calmar_sd",
                        "max_drawdown_sd",
                    ]
                ],
                left_on=["pair", "classifier_group", "selected_global_tau"],
                right_on=["pair", "classifier_group", "tau"],
                validate="one_to_one",
            ).drop(columns="tau")
            experiments[(dataset, family)] = {
                "root": root,
                "manifest": json.loads((root / "experiment_manifest.json").read_text()),
                "base": pd.read_csv(root / "base_model_metrics.csv"),
                "average": pd.read_csv(root / "simple_average_metrics.csv"),
                "selected": selected,
                "robustness": pd.read_csv(root / "tau_robustness_summary.csv"),
                "common": pd.read_csv(root / "selected_common_tau_summary.csv"),
            }
    return experiments


def format_metric(value: float) -> str:
    return f"{float(value):.4f}"


def metric_cells(row: pd.Series) -> list[str]:
    return [
        format_metric(row["cumulative_return"]),
        format_metric(row["sharpe"]),
        format_metric(row["calmar"]),
        format_metric(row["max_drawdown"]),
    ]


def selected_cells(row: pd.Series) -> list[str]:
    wins = int(round(30 * float(row["win_rate_vs_stronger"])))
    return [
        f"{float(row['selected_global_tau']):.2f}",
        f"${float(row['ensemble_cumulative_return_mean']):.4f}"
        f"\\pm{float(row['cumulative_return_sd']):.4f}$",
        f"${float(row['ensemble_sharpe_mean']):.4f}"
        f"\\pm{float(row['ensemble_sharpe_sd']):.4f}$",
        f"${float(row['ensemble_calmar_mean']):.4f}"
        f"\\pm{float(row['calmar_sd']):.4f}$",
        f"${float(row['ensemble_max_drawdown_mean']):.4f}"
        f"\\pm{float(row['max_drawdown_sd']):.4f}$",
        f"{float(row['delta_sharpe_mean']):.4f} "
        f"[{float(row['delta_sharpe_ci_low']):.4f}, "
        f"{float(row['delta_sharpe_ci_high']):.4f}]",
        str(wins),
    ]


def baseline_rows(
    experiments: dict[tuple[str, str], dict[str, object]]
) -> list[list[str]]:
    rows: list[list[str]] = []
    for dataset, _roots in DATASETS:
        for family, experiment_label in (
            ("group1", "I"),
            ("group2", "II"),
        ):
            experiment = experiments[(dataset, family)]
            for item in experiment["base"].itertuples(index=False):
                values = pd.Series(item._asdict())
                rows.append(
                    [dataset, experiment_label, LABELS[item.model], *metric_cells(values)]
                )
            for item in experiment["average"].itertuples(index=False):
                values = pd.Series(item._asdict())
                rows.append(
                    [
                        dataset,
                        experiment_label,
                        f"{LABELS[item.pair]} average",
                        *metric_cells(values),
                    ]
                )

        main = experiments[(dataset, "main")]
        stress = experiments[(dataset, "stress")]
        main_base = main["base"].set_index("model")
        stress_base = stress["base"].set_index("model")
        if not np.allclose(
            main_base.loc["tqc"].to_numpy(dtype=float),
            stress_base.loc["tqc"].to_numpy(dtype=float),
        ):
            raise ValueError(f"{dataset} TQC paths differ between main and stress runs")
        for model, source in (("ppo", main_base), ("td3", stress_base), ("tqc", main_base)):
            rows.append(
                [
                    dataset,
                    "III",
                    LABELS[model],
                    *metric_cells(source.loc[model]),
                ]
            )
        for family, experiment in (("main", main), ("stress", stress)):
            item = experiment["average"].iloc[0]
            rows.append(
                [
                    dataset,
                    "III",
                    f"{LABELS[PAIR_ORDER[family][0]]} average",
                    *metric_cells(item),
                ]
            )
    return rows


def selected_rows(
    experiments: dict[tuple[str, str], dict[str, object]], families: tuple[str, ...]
) -> list[list[str]]:
    rows: list[list[str]] = []
    for dataset, _roots in DATASETS:
        for family in families:
            selected = experiments[(dataset, family)]["selected"]
            for pair in PAIR_ORDER[family]:
                pair_frame = selected.loc[selected["pair"] == pair].sort_values(
                    "classifier_group"
                )
                for item in pair_frame.itertuples(index=False):
                    row = pd.Series(item._asdict())
                    rows.append(
                        [
                            dataset,
                            LABELS[pair],
                            str(int(item.classifier_group)),
                            *selected_cells(row),
                        ]
                    )
    return rows


def summary_rows(
    experiments: dict[tuple[str, str], dict[str, object]]
) -> tuple[list[list[str]], pd.DataFrame]:
    rows: list[list[str]] = []
    records: list[dict[str, object]] = []
    descriptions = {
        "group1": "Representative forecasting",
        "group2": "PatchTST+iTransformer",
        "main": "PPO+TQC (main)",
        "stress": "TD3+TQC (stress)",
    }
    for dataset, _roots in DATASETS:
        for family in ("group1", "group2", "main", "stress"):
            experiment = experiments[(dataset, family)]
            selected = experiment["selected"]
            count = len(selected)
            beats_stronger = int((selected["delta_sharpe_mean"] > 0).sum())
            positive_ci = int((selected["delta_sharpe_ci_low"] > 0).sum())
            beats_average = int((selected["delta_sharpe_vs_average"] > 0).sum())
            stable: str | int = "--"
            if family in {"main", "stress"}:
                distribution = pd.read_csv(
                    experiment["root"] / "selected_tau_distribution_audit.csv"
                )
                stable = int(
                    (
                        (distribution["delta_sharpe_median"] > 0)
                        & (distribution["one_sided_sign_test_p"] < 0.05)
                    ).sum()
                )
            rows.append(
                [
                    dataset,
                    descriptions[family],
                    str(count),
                    str(beats_stronger),
                    str(positive_ci),
                    str(beats_average),
                    str(stable),
                ]
            )
            records.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "description": descriptions[family],
                    "configurations": count,
                    "beats_stronger": beats_stronger,
                    "positive_95pct_ci": positive_ci,
                    "beats_average": beats_average,
                    "median_sign_stable": stable,
                }
            )
    return rows, pd.DataFrame(records)


def checkpoint_rows(
    experiments: dict[tuple[str, str], dict[str, object]]
) -> list[list[str]]:
    rows: list[list[str]] = []
    for dataset, _roots in DATASETS:
        checkpoints: dict[str, dict[int, int]] = {}
        for family in ("main", "stress"):
            manifest = experiments[(dataset, family)]["manifest"]
            for item in manifest["checkpoints"]:
                model = str(item["model"])
                checkpoints.setdefault(model, {})[int(item["window"])] = int(
                    item["selected_validation_step"]
                )
        for model in ("ppo", "td3", "tqc"):
            steps = checkpoints[model]
            if sorted(steps) != [1, 2, 3, 4]:
                raise ValueError(f"incomplete {dataset} {model} checkpoints")
            rows.append(
                [dataset, LABELS[model], *[str(steps[index] // 1000) for index in range(1, 5)]]
            )
    return rows


def common_tau_rows(
    experiments: dict[tuple[str, str], dict[str, object]]
) -> list[list[str]]:
    rows: list[list[str]] = []
    for dataset, _roots in DATASETS:
        for family in ("group1", "group2", "main", "stress"):
            common = experiments[(dataset, family)]["common"].set_index("pair")
            for pair in PAIR_ORDER[family]:
                item = common.loc[pair]
                rows.append(
                    [
                        dataset,
                        LABELS[pair],
                        f"{float(item['tau']):.2f}",
                        f"{float(item['sharpe_across_groups_mean']):.4f}",
                        f"{float(item['sharpe_across_groups_min']):.4f}",
                        f"{float(item['sharpe_across_groups_max']):.4f}",
                        str(int(item["groups_beating_stronger"])),
                        str(int(item["groups_beating_simple_average"])),
                    ]
                )
    return rows


def table(
    *,
    caption: str,
    label: str,
    columns: str,
    headers: list[str],
    rows: list[list[str]],
    size: str = "tiny",
) -> str:
    body = [
        "\\begin{table}[p]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\{size}",
        "\\resizebox{\\textwidth}{!}{",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    previous_dataset = None
    for row in rows:
        if previous_dataset is not None and row[0] != previous_dataset:
            body.append("\\midrule")
        body.append(" & ".join(row) + " \\\\")
        previous_dataset = row[0]
    body.extend(["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""])
    return "\n".join(body)


def main() -> None:
    args = parse_args()
    code_root = Path(args.code_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = load_experiments(code_root)

    summary, summary_frame = summary_rows(experiments)
    tex = [
        table(
            caption=(
                "Component and simple holding-average baselines for the complete "
                "SSE50 and HSTech10 model-family extensions."
            ),
            label="tab:extension_external_baselines",
            columns="lllrrrr",
            headers=[
                "\\textbf{Dataset}",
                "\\textbf{Experiment}",
                "\\textbf{Candidate}",
                "\\textbf{Return}",
                "\\textbf{Sharpe}",
                "\\textbf{Calmar}",
                "\\textbf{MDD}",
            ],
            rows=baseline_rows(experiments),
        ),
        table(
            caption=(
                "Complete external-market Experiment I selected-global-$\\tau$ "
                "results. Metrics are mean $\\pm$ sample standard deviation across "
                "30 rolling classifier refits. $\\Delta S$ compares the ensemble "
                "with the stronger component."
            ),
            label="tab:extension_external_group1",
            columns="lllccccccc",
            headers=[
                "\\textbf{Dataset}",
                "\\textbf{Pair}",
                "\\textbf{G}",
                "$\\boldsymbol{\\tau}$",
                "\\textbf{Return}",
                "\\textbf{Sharpe}",
                "\\textbf{Calmar}",
                "\\textbf{MDD}",
                "$\\boldsymbol{\\Delta S}$ \\textbf{[95\\% CI]}",
                "\\textbf{Wins/30}",
            ],
            rows=selected_rows(experiments, ("group1",)),
        ),
        table(
            caption=(
                "Complete external-market Experiments II and III selected-global-"
                "$\\tau$ results. Definitions follow Table~\\ref{tab:extension_external_group1}."
            ),
            label="tab:extension_external_groups23",
            columns="lllccccccc",
            headers=[
                "\\textbf{Dataset}",
                "\\textbf{Pair}",
                "\\textbf{G}",
                "$\\boldsymbol{\\tau}$",
                "\\textbf{Return}",
                "\\textbf{Sharpe}",
                "\\textbf{Calmar}",
                "\\textbf{MDD}",
                "$\\boldsymbol{\\Delta S}$ \\textbf{[95\\% CI]}",
                "\\textbf{Wins/30}",
            ],
            rows=selected_rows(experiments, ("group2", "main", "stress")),
        ),
        table(
            caption=(
                "Configuration-level external model-family summary. Median/sign "
                "stability requires a positive paired median and one-sided exact "
                "sign-test $p<0.05$ and is audited for the expanded deep RL pairs."
            ),
            label="tab:extension_external_summary",
            columns="llccccc",
            headers=[
                "\\textbf{Dataset}",
                "\\textbf{Experiment or pair}",
                "\\textbf{Configs.}",
                "\\textbf{Beats stronger}",
                "\\textbf{Positive 95\\% CI}",
                "\\textbf{Beats average}",
                "\\textbf{Median/sign stable}",
            ],
            rows=summary,
            size="scriptsize",
        ),
        table(
            caption=(
                "Prior-block-selected external-market deep RL checkpoints for "
                "Experiment III. Entries are training steps in thousands."
            ),
            label="tab:extension_external_rl_checkpoints",
            columns="llcccc",
            headers=[
                "\\textbf{Dataset}",
                "\\textbf{Agent}",
                "\\textbf{Window 1}",
                "\\textbf{Window 2}",
                "\\textbf{Window 3}",
                "\\textbf{Window 4}",
            ],
            rows=checkpoint_rows(experiments),
            size="small",
        ),
        table(
            caption=(
                "External-market common-threshold sensitivity. For each dataset and "
                "candidate pair, one fixed global $\\tau$ is selected by mean Sharpe "
                "across all five fixed classifier groups."
            ),
            label="tab:extension_external_common_tau",
            columns="llcccccc",
            headers=[
                "\\textbf{Dataset}",
                "\\textbf{Pair}",
                "$\\boldsymbol{\\tau}$",
                "\\textbf{Mean Sharpe}",
                "\\textbf{Min}",
                "\\textbf{Max}",
                "\\textbf{Groups $>$ stronger}",
                "\\textbf{Groups $>$ average}",
            ],
            rows=common_tau_rows(experiments),
            size="scriptsize",
        ),
    ]
    (output_dir / "external_model_family_tables.tex").write_text(
        "\n".join(tex), encoding="utf-8"
    )
    summary_frame.to_csv(
        output_dir / "external_model_family_configuration_summary.csv", index=False
    )
    robustness = []
    for dataset, _roots in DATASETS:
        for family in ("group1", "group2", "main", "stress"):
            frame = experiments[(dataset, family)]["robustness"].copy()
            frame.insert(0, "family", family)
            frame.insert(0, "dataset", dataset)
            robustness.append(frame)
    pd.concat(robustness, ignore_index=True).to_csv(
        output_dir / "external_model_family_threshold_robustness.csv", index=False
    )
    print(output_dir / "external_model_family_tables.tex")


if __name__ == "__main__":
    main()
