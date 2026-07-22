from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/finrl-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PERIOD = "2020"
PAIR_ORDER = ("a2c_ppo", "a2c_sac", "ppo_sac")
PAIR_LABELS = {
    "a2c_ppo": "A2C+PPO",
    "a2c_sac": "A2C+SAC",
    "ppo_sac": "PPO+SAC",
}
GROUP_LABELS = {
    1: "G1 SVM",
    2: "G2 Logistic",
    3: "G3 Trees",
    4: "G4 SVM+Logistic",
    5: "G5 All",
}
TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain successful and failed 2020 causal crossover-tau configurations."
    )
    parser.add_argument(
        "--daily-decisions",
        default="work/causal_tau_mechanism_only/all_daily_crossover_tau_decisions.csv",
    )
    parser.add_argument(
        "--block-audit",
        default="work/causal_tau_mechanism_only/all_block_tau_audit.csv",
    )
    parser.add_argument(
        "--run-metrics",
        default="work/causal_tau_mechanism_only/all_crossover_tau_metrics.csv",
    )
    parser.add_argument(
        "--config-features",
        default="work/causal_tau_statistics/configuration_features_15.csv",
    )
    parser.add_argument(
        "--candidate-root",
        default="work/causal_candidates/2020",
    )
    parser.add_argument(
        "--output-dir",
        default="work/causal_tau_attribution",
    )
    return parser.parse_args()


def safe_sharpe(returns: np.ndarray) -> float:
    values = np.asarray(returns, dtype=float)
    sd = float(values.std(ddof=1))
    return float(np.sqrt(252.0) * values.mean() / sd) if sd > 1e-12 else 0.0


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.4f}"
        )
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def load_base_returns(
    candidate_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []

    for pair in PAIR_ORDER:
        root = candidate_root / pair
        windows = pd.read_csv(root / "rolling_windows.csv").sort_values("window")
        base_metrics = pd.read_csv(root / "base_model_metrics.csv")
        stronger = str(base_metrics.sort_values("sharpe", ascending=False).iloc[0]["model"])
        weaker = str(base_metrics.sort_values("sharpe", ascending=False).iloc[-1]["model"])
        with np.load(root / "base_account_curves.npz") as arrays:
            dates = arrays["dates"].astype(str)
            model_names = [name for name in arrays.files if name != "dates"]
            curves = {name: arrays[name].astype(float) for name in model_names}

        offset = 0
        pair_return_parts: dict[str, list[np.ndarray]] = {name: [] for name in model_names}
        block_winners: list[str] = []
        oracle_parts: list[np.ndarray] = []
        for window_row in windows.itertuples(index=False):
            window = int(window_row.window)
            count = int(window_row.trade_dates)
            segment_dates = dates[offset : offset + count]
            if segment_dates[0] != str(window_row.trade_start) or segment_dates[-1] != str(
                window_row.trade_end
            ):
                raise ValueError(f"base dates do not match window {window} for {pair}")
            model_returns: dict[str, np.ndarray] = {}
            for model in model_names:
                values = curves[model][offset : offset + count]
                returns = values[1:] / values[:-1] - 1.0
                model_returns[model] = returns
                pair_return_parts[model].append(returns)
                for date, value in zip(segment_dates[:-1], returns):
                    daily_rows.append(
                        {
                            "pair": pair,
                            "window": window,
                            "date": date,
                            "model": model,
                            "base_return": float(value),
                        }
                    )
                block_rows.append(
                    {
                        "pair": pair,
                        "window": window,
                        "model": model,
                        "cumulative_return": float(np.prod(1.0 + returns) - 1.0),
                        "sharpe": safe_sharpe(returns),
                        "mean_daily_return": float(returns.mean()),
                        "daily_volatility": float(returns.std(ddof=1)),
                    }
                )
            block_winner = max(
                model_names,
                key=lambda model: float(np.prod(1.0 + model_returns[model]) - 1.0),
            )
            block_winners.append(block_winner)
            oracle_parts.append(model_returns[block_winner])
            offset += count

        if offset != len(dates):
            raise ValueError(f"unused base dates for {pair}")
        all_returns = {
            model: np.concatenate(parts) for model, parts in pair_return_parts.items()
        }
        oracle_returns = np.concatenate(oracle_parts)
        stronger_returns = all_returns[stronger]
        source_stronger = base_metrics.loc[base_metrics["model"] == stronger].iloc[0]
        pair_rows.append(
            {
                "pair": pair,
                "pair_label": PAIR_LABELS[pair],
                "stronger_model": stronger,
                "weaker_model": weaker,
                "stronger_fixed_sharpe": float(source_stronger["sharpe"]),
                "stronger_fixed_cumulative_return": float(
                    source_stronger["cumulative_return"]
                ),
                "stronger_intra_block_cumulative_return": float(
                    np.prod(1.0 + stronger_returns) - 1.0
                ),
                "base_daily_return_correlation": float(
                    np.corrcoef(all_returns[stronger], all_returns[weaker])[0, 1]
                ),
                "block_winner_sequence": ";".join(block_winners),
                "block_winner_switches": int(
                    sum(left != right for left, right in zip(block_winners, block_winners[1:]))
                ),
                "block_oracle_cumulative_return": float(
                    np.prod(1.0 + oracle_returns) - 1.0
                ),
                "block_oracle_sharpe": safe_sharpe(oracle_returns),
                "oracle_return_headroom": float(
                    np.prod(1.0 + oracle_returns)
                    - np.prod(1.0 + stronger_returns)
                ),
                "oracle_sharpe_headroom": float(
                    safe_sharpe(oracle_returns) - safe_sharpe(stronger_returns)
                ),
            }
        )

    base_daily = pd.DataFrame(daily_rows)
    base_blocks = pd.DataFrame(block_rows)
    pair_features = pd.DataFrame(pair_rows)
    block_winner = (
        base_blocks.sort_values(
            ["pair", "window", "cumulative_return"], ascending=[True, True, False]
        )
        .groupby(["pair", "window"], as_index=False)
        .first()[["pair", "window", "model"]]
        .rename(columns={"model": "block_winner"})
    )
    base_blocks = base_blocks.merge(block_winner, on=["pair", "window"])
    return base_daily, base_blocks, pair_features


def add_causal_baseline(
    pair_features: pd.DataFrame, metrics_path: Path
) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    metrics = metrics.loc[metrics["period"].astype(str) == PERIOD]
    baseline = (
        metrics.groupby("pair", sort=True)
        .agg(
            causal_single_model_sequence=("causal_single_model_sequence", "first"),
            causal_single_sharpe=("causal_single_sharpe", "first"),
        )
        .reset_index()
    )
    result = pair_features.merge(baseline, on="pair", validate="one_to_one")
    result["causal_correct_blocks"] = result.apply(
        lambda row: sum(
            selected == winner
            for selected, winner in zip(
                str(row["causal_single_model_sequence"]).split(";"),
                str(row["block_winner_sequence"]).split(";"),
            )
        ),
        axis=1,
    )
    return result


def add_selected_model_return(frame: pd.DataFrame, output: str, selector: str) -> None:
    frame[output] = np.nan
    for model in ("a2c", "ppo", "sac"):
        mask = frame[selector].astype(str) == model
        if model in frame.columns:
            frame.loc[mask, output] = frame.loc[mask, model]
    if frame[output].isna().any():
        raise ValueError(f"could not map {selector} to {output}")


def enrich_daily_decisions(
    daily_path: Path,
    base_daily: pd.DataFrame,
    base_blocks: pd.DataFrame,
    pair_features: pd.DataFrame,
) -> pd.DataFrame:
    daily = pd.read_csv(daily_path)
    daily = daily.loc[daily["period"].astype(str) == PERIOD].copy()
    if len(daily) != 112_050 or daily["repeat"].nunique() != 30:
        raise ValueError("unexpected 2020 daily decision sample")

    wide = base_daily.pivot(
        index=["pair", "window", "date"], columns="model", values="base_return"
    ).reset_index()
    daily["date"] = daily["date"].astype(str)
    daily = daily.merge(wide, on=["pair", "window", "date"], validate="many_to_one")
    pair_map = pair_features.set_index("pair")
    daily["stronger_model"] = daily["pair"].map(pair_map["stronger_model"])
    daily["weaker_model"] = daily["pair"].map(pair_map["weaker_model"])
    daily = daily.merge(
        base_blocks[["pair", "window", "block_winner"]].drop_duplicates(),
        on=["pair", "window"],
        validate="many_to_one",
    )
    add_selected_model_return(daily, "stronger_return", "stronger_model")
    add_selected_model_return(daily, "weaker_return", "weaker_model")
    add_selected_model_return(daily, "selected_base_return", "selected_agent")

    for column in (
        "actual_master_return",
        "fallback_return",
        "stronger_return",
        "selected_base_return",
    ):
        if (daily[column] <= -1.0).any():
            raise ValueError(f"invalid daily return in {column}")
    daily["actual_log_return"] = np.log1p(daily["actual_master_return"])
    daily["fallback_log_return"] = np.log1p(daily["fallback_return"])
    daily["stronger_log_return"] = np.log1p(daily["stronger_return"])
    daily["selected_base_log_return"] = np.log1p(daily["selected_base_return"])
    daily["log_advantage_vs_fallback"] = (
        daily["actual_log_return"] - daily["fallback_log_return"]
    )
    daily["log_advantage_vs_stronger"] = (
        daily["actual_log_return"] - daily["stronger_log_return"]
    )
    daily["path_transition_drag"] = (
        daily["actual_log_return"] - daily["selected_base_log_return"]
    )
    daily["active"] = ~daily["fallback_used"].astype(bool)
    daily["uses_stronger"] = daily["selected_agent"] == daily["stronger_model"]
    daily["uses_weaker"] = daily["selected_agent"] == daily["weaker_model"]
    daily["uses_block_winner"] = daily["selected_agent"] == daily["block_winner"]
    daily["wins_vs_stronger_day"] = (
        daily["actual_master_return"] > daily["stronger_return"] + TOLERANCE
    )
    daily["loses_vs_stronger_day"] = (
        daily["actual_master_return"] < daily["stronger_return"] - TOLERANCE
    )
    return daily


def aggregate_attribution(
    daily: pd.DataFrame,
    metrics_path: Path,
    config_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["pair", "classifier_group", "repeat", "window"]
    window_runs = (
        daily.groupby(keys, sort=True)
        .agg(
            decision_days=("date", "size"),
            active_day_rate=("active", "mean"),
            stronger_exposure_rate=("uses_stronger", "mean"),
            weaker_exposure_rate=("uses_weaker", "mean"),
            block_winner_exposure_rate=("uses_block_winner", "mean"),
            daily_win_vs_stronger_rate=("wins_vs_stronger_day", "mean"),
            daily_loss_vs_stronger_rate=("loses_vs_stronger_day", "mean"),
            actual_log_return=("actual_log_return", "sum"),
            fallback_log_return=("fallback_log_return", "sum"),
            stronger_log_return=("stronger_log_return", "sum"),
            log_advantage_vs_fallback=("log_advantage_vs_fallback", "sum"),
            log_advantage_vs_stronger=("log_advantage_vs_stronger", "sum"),
            path_transition_drag=("path_transition_drag", "sum"),
            mean_actual_return=("actual_master_return", "mean"),
            actual_volatility=("actual_master_return", "std"),
            stronger_volatility=("stronger_return", "std"),
        )
        .reset_index()
    )
    for column in ("actual", "fallback", "stronger"):
        window_runs[f"{column}_cumulative_return"] = (
            np.exp(window_runs[f"{column}_log_return"]) - 1.0
        )

    numeric = window_runs.select_dtypes(include=[np.number]).columns.difference(
        ["repeat", "classifier_group", "window"]
    )
    config_window = (
        window_runs.groupby(["pair", "classifier_group", "window"], sort=True)[numeric]
        .mean()
        .reset_index()
    )
    config_window["pair_label"] = config_window["pair"].map(PAIR_LABELS)

    annual_runs = (
        daily.groupby(["pair", "classifier_group", "repeat"], sort=True)
        .agg(
            active_day_rate=("active", "mean"),
            stronger_exposure_rate=("uses_stronger", "mean"),
            weaker_exposure_rate=("uses_weaker", "mean"),
            block_winner_exposure_rate=("uses_block_winner", "mean"),
            daily_win_vs_stronger_rate=("wins_vs_stronger_day", "mean"),
            daily_loss_vs_stronger_rate=("loses_vs_stronger_day", "mean"),
            log_advantage_vs_fallback=("log_advantage_vs_fallback", "sum"),
            log_advantage_vs_stronger=("log_advantage_vs_stronger", "sum"),
            path_transition_drag=("path_transition_drag", "sum"),
            mean_actual_return=("actual_master_return", "mean"),
            actual_volatility=("actual_master_return", "std"),
            stronger_mean_return=("stronger_return", "mean"),
            stronger_volatility=("stronger_return", "std"),
        )
        .reset_index()
    )
    for window in range(1, 5):
        block = window_runs.loc[
            window_runs["window"] == window,
            [
                "pair",
                "classifier_group",
                "repeat",
                "log_advantage_vs_stronger",
                "log_advantage_vs_fallback",
                "weaker_exposure_rate",
                "block_winner_exposure_rate",
            ],
        ].rename(
            columns={
                "log_advantage_vs_stronger": f"w{window}_log_adv_vs_stronger",
                "log_advantage_vs_fallback": f"w{window}_log_adv_vs_fallback",
                "weaker_exposure_rate": f"w{window}_weaker_exposure",
                "block_winner_exposure_rate": f"w{window}_winner_exposure",
            }
        )
        annual_runs = annual_runs.merge(
            block,
            on=["pair", "classifier_group", "repeat"],
            validate="one_to_one",
        )
    annual_runs["mean_return_delta_vs_stronger"] = (
        annual_runs["mean_actual_return"] - annual_runs["stronger_mean_return"]
    )
    annual_runs["volatility_delta_vs_stronger"] = (
        annual_runs["actual_volatility"] - annual_runs["stronger_volatility"]
    )

    run_metrics = pd.read_csv(metrics_path)
    run_metrics = run_metrics.loc[run_metrics["period"].astype(str) == PERIOD]
    annual_runs = annual_runs.merge(
        run_metrics[
            [
                "pair",
                "classifier_group",
                "repeat",
                "delta_sharpe_vs_causal_single",
                "delta_sharpe_vs_stronger",
            ]
        ],
        on=["pair", "classifier_group", "repeat"],
        validate="one_to_one",
    )
    annual_runs["run_beats_stronger"] = (
        annual_runs["delta_sharpe_vs_stronger"] > 0.0
    ).astype(float)
    annual_runs["run_beats_causal"] = (
        annual_runs["delta_sharpe_vs_causal_single"] > 0.0
    ).astype(float)

    annual_numeric = annual_runs.select_dtypes(include=[np.number]).columns.difference(
        ["repeat", "classifier_group"]
    )
    config = (
        annual_runs.groupby(["pair", "classifier_group"], sort=True)[annual_numeric]
        .mean()
        .reset_index()
    )
    source_config = pd.read_csv(config_path)
    source_columns = [
        "pair",
        "classifier_group",
        "sharpe",
        "delta_sharpe_vs_causal_single",
        "delta_sharpe_vs_stronger",
        "success_vs_causal",
        "success_vs_stronger",
        "branch_divergence_rate",
        "ensemble_mode_hit_rate",
        "mean_vote_margin",
        "selected_block_win_rate",
    ]
    config = config.drop(
        columns=[
            "delta_sharpe_vs_causal_single",
            "delta_sharpe_vs_stronger",
        ]
    ).merge(
        source_config[source_columns],
        on=["pair", "classifier_group"],
        validate="one_to_one",
    )
    config["pair_label"] = config["pair"].map(PAIR_LABELS)
    config["group_label"] = config["classifier_group"].map(GROUP_LABELS)
    return config_window, config


def status_and_group_summary(
    audit_path: Path, config: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = pd.read_csv(audit_path)
    audit = audit.loc[audit["period"].astype(str) == PERIOD].copy()
    status = (
        audit.groupby(["pair", "classifier_group", "status"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    status["total_blocks"] = status.filter(regex="^(selected|fallback_)").sum(axis=1)
    status["selected_block_rate"] = status.get("selected", 0) / status["total_blocks"]
    config = config.merge(
        status[["pair", "classifier_group", "selected_block_rate"]],
        on=["pair", "classifier_group"],
        validate="one_to_one",
        suffixes=("", "_audit"),
    )
    summary = (
        config.groupby("classifier_group", sort=True)
        .agg(
            configurations=("pair", "size"),
            configs_beating_stronger=("success_vs_stronger", "sum"),
            mean_delta_sharpe_vs_stronger=("delta_sharpe_vs_stronger", "mean"),
            mean_active_day_rate=("active_day_rate", "mean"),
            mean_selected_block_rate=("selected_block_rate", "mean"),
            mean_branch_divergence=("branch_divergence_rate", "mean"),
            mean_block_winner_exposure=("block_winner_exposure_rate", "mean"),
            mean_log_advantage_vs_fallback=("log_advantage_vs_fallback", "mean"),
            mean_log_advantage_vs_stronger=("log_advantage_vs_stronger", "mean"),
            mean_mode_hit_rate=("ensemble_mode_hit_rate", "mean"),
            mean_vote_margin=("mean_vote_margin", "mean"),
            mean_positive_refit_rate=("run_beats_stronger", "mean"),
        )
        .reset_index()
    )
    summary["group_label"] = summary["classifier_group"].map(GROUP_LABELS)
    return config, summary


def plot_causes(
    pair_features: pd.DataFrame,
    config_window: pd.DataFrame,
    config: pd.DataFrame,
    output: Path,
) -> None:
    row_labels = [
        f"{PAIR_LABELS[pair]} G{group}"
        for pair in PAIR_ORDER
        for group in range(1, 6)
    ]
    row_keys = [(pair, group) for pair in PAIR_ORDER for group in range(1, 6)]
    heat = np.full((15, 4), np.nan)
    for row_index, (pair, group) in enumerate(row_keys):
        subset = config_window.loc[
            (config_window["pair"] == pair)
            & (config_window["classifier_group"] == group)
        ].sort_values("window")
        heat[row_index] = subset["log_advantage_vs_stronger"].to_numpy(float)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)
    pair_positions = np.arange(len(PAIR_ORDER))
    pair_frame = pair_features.set_index("pair").loc[list(PAIR_ORDER)]
    axes[0, 0].bar(
        pair_positions,
        pair_frame["oracle_return_headroom"],
        color=["#4472C4", "#2E8B57", "#C55A11"],
    )
    axes[0, 0].set_xticks(pair_positions, [PAIR_LABELS[pair] for pair in PAIR_ORDER])
    axes[0, 0].set_ylabel("Block-oracle return headroom")
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].grid(axis="y", alpha=0.2)

    limit = float(np.nanmax(np.abs(heat)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    image = axes[0, 1].imshow(heat, cmap="RdYlGn", norm=norm, aspect="auto")
    axes[0, 1].set_xticks(np.arange(4), [f"W{window}" for window in range(1, 5)])
    axes[0, 1].set_yticks(np.arange(15), row_labels, fontsize=7)
    axes[0, 1].set_title("Mean log advantage vs stronger fixed model")
    fig.colorbar(image, ax=axes[0, 1], fraction=0.046, pad=0.04)

    colors = config["success_vs_stronger"].map({True: "#2E8B57", False: "#C00000"})
    axes[1, 0].scatter(
        config["block_winner_exposure_rate"],
        config["delta_sharpe_vs_stronger"],
        c=colors,
        s=55,
    )
    for row in config.itertuples(index=False):
        axes[1, 0].annotate(
            f"{PAIR_LABELS[row.pair]} G{int(row.classifier_group)}",
            (row.block_winner_exposure_rate, row.delta_sharpe_vs_stronger),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6.5,
        )
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_xlabel("Exposure to realized block winner")
    axes[1, 0].set_ylabel("Delta Sharpe vs stronger fixed model")
    axes[1, 0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#2E8B57",
                markeredgecolor="none",
                label="Mean beats stronger",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#C00000",
                markeredgecolor="none",
                label="Mean fails stronger",
            ),
        ],
        frameon=False,
        fontsize=7,
        loc="upper left",
    )
    axes[1, 0].grid(alpha=0.2)

    axes[1, 1].scatter(
        config["active_day_rate"],
        config["log_advantage_vs_fallback"],
        c=colors,
        s=55,
    )
    for row in config.itertuples(index=False):
        axes[1, 1].annotate(
            f"{PAIR_LABELS[row.pair]} G{int(row.classifier_group)}",
            (row.active_day_rate, row.log_advantage_vs_fallback),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=6.5,
        )
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_xlabel("Active day rate")
    axes[1, 1].set_ylabel("Annual log advantage vs causal fallback")
    axes[1, 1].grid(alpha=0.2)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"figure_2020_success_failure_causes.{suffix}", dpi=220)
    plt.close(fig)


def write_report(
    output: Path,
    pair_features: pd.DataFrame,
    base_blocks: pd.DataFrame,
    config_window: pd.DataFrame,
    config: pd.DataFrame,
    groups: pd.DataFrame,
) -> None:
    pair_table = pair_features[
        [
            "pair_label",
            "base_daily_return_correlation",
            "block_winner_sequence",
            "causal_single_model_sequence",
            "causal_correct_blocks",
            "stronger_intra_block_cumulative_return",
            "block_oracle_cumulative_return",
            "oracle_return_headroom",
            "oracle_sharpe_headroom",
        ]
    ].copy()
    pair_table.columns = [
        "RL组合",
        "日收益相关",
        "块赢家序列",
        "因果选择序列",
        "因果选对块数",
        "同口径最强收益",
        "块Oracle收益",
        "Oracle收益空间",
        "Oracle Sharpe空间",
    ]
    block_table = base_blocks[
        ["pair", "window", "model", "cumulative_return", "sharpe", "block_winner"]
    ].copy()
    block_table["pair"] = block_table["pair"].map(PAIR_LABELS)
    block_table.columns = ["RL组合", "窗口", "模型", "收益", "Sharpe", "块赢家"]

    config_table = config[
        [
            "pair_label",
            "classifier_group",
            "delta_sharpe_vs_stronger",
            "run_beats_stronger",
            "active_day_rate",
            "branch_divergence_rate",
            "block_winner_exposure_rate",
            "weaker_exposure_rate",
            "log_advantage_vs_fallback",
            "log_advantage_vs_stronger",
            "mean_return_delta_vs_stronger",
            "volatility_delta_vs_stronger",
        ]
    ].copy()
    config_table.columns = [
        "RL组合",
        "Group",
        "DeltaSharpe-最强",
        "30次胜率",
        "激活日比例",
        "分支分化率",
        "块赢家暴露",
        "弱模型暴露",
        "Log优势-fallback",
        "Log优势-最强",
        "日均收益差",
        "日波动差",
    ]
    group_table = groups[
        [
            "group_label",
            "configs_beating_stronger",
            "mean_delta_sharpe_vs_stronger",
            "mean_selected_block_rate",
            "mean_branch_divergence",
            "mean_block_winner_exposure",
            "mean_log_advantage_vs_fallback",
            "mean_positive_refit_rate",
        ]
    ].copy()
    group_table.columns = [
        "分类器组",
        "胜最强配置数",
        "平均DeltaSharpe",
        "选中块比例",
        "分支分化率",
        "块赢家暴露",
        "Log优势-fallback",
        "30次平均胜率",
    ]

    window_pivot = config_window.pivot_table(
        index=["pair_label", "classifier_group"],
        columns="window",
        values="log_advantage_vs_stronger",
    ).reset_index()
    window_pivot.columns = ["RL组合", "Group", "W1", "W2", "W3", "W4"]

    exposure = config_window.pivot_table(
        index=["pair_label", "classifier_group"],
        columns="window",
        values="weaker_exposure_rate",
    ).reset_index()
    exposure.columns = ["RL组合", "Group", "W1", "W2", "W3", "W4"]
    ideal = (
        config_window.groupby(["pair_label", "window"], sort=True)[
            "ideal_weaker_exposure"
        ]
        .first()
        .unstack()
        .reset_index()
    )
    ideal.columns = ["RL组合", "W1", "W2", "W3", "W4"]
    ideal.insert(1, "Group", "Ideal")
    exposure["Group"] = exposure["Group"].map(lambda value: f"G{int(value)}")
    exposure = pd.concat([ideal, exposure], ignore_index=True)
    exposure["pair_order"] = exposure["RL组合"].map(
        {PAIR_LABELS[pair]: index for index, pair in enumerate(PAIR_ORDER)}
    )
    exposure["group_order"] = exposure["Group"].map(
        {"Ideal": 0, **{f"G{group}": group for group in range(1, 6)}}
    )
    exposure = exposure.sort_values(["pair_order", "group_order"]).drop(
        columns=["pair_order", "group_order"]
    )

    lines = [
        "# 2020年集成有效与失败原因归因",
        "",
        "## 口径",
        "",
        "本报告只使用DJ30的2020年四个rolling交易块。归因表将集成日收益同时与因果fallback和事后最强固定RL比较。块Oracle只使用249个块内决策日，并与同样口径的最强固定模型比较；它在看完每块收益后选块赢家，只用于衡量已实现互补空间，不是可部署基准。",
        "",
        "30次路径共享同一市场和固定RL候选；`30次胜率`只表示分类器重拟合条件稳定性，不是30个独立市场样本。",
        "",
        "## RL组合的可切换空间",
        "",
        markdown_table(pair_table),
        "",
        "## 四个窗口的基模型表现",
        "",
        markdown_table(block_table),
        "",
        "## 15个配置的机制归因",
        "",
        markdown_table(config_table),
        "",
        "## 相对最强固定模型的块级Log优势",
        "",
        markdown_table(window_pivot),
        "",
        "## 弱模型的窗口暴露与事后理想序列",
        "",
        "`Ideal=1`表示全年较弱的模型在该窗口反而是块赢家；`Ideal=0`表示应使用全年较强模型。其他行是实际选择弱模型的日比例。",
        "",
        markdown_table(exposure),
        "",
        "## 分类器组归因",
        "",
        markdown_table(group_table),
        "",
        "## 有效与失败的具体原因",
        "",
        "- **A2C+PPO：互补空间较小且错误暴露发生在高代价窗口。** 两模型日收益相关为0.9032，同口径块Oracle收益空间只有0.0754。五个分类器组在W1相对PPO的Log损失均在0.093以上，W2/W3的收益不足以补回暴跌块和W4的错误A2C暴露。",
        "- **A2C+SAC：G1/G2/G4几乎学到了正确的块轮换。** 事后理想弱模型A2C暴露是`1,0,1,0`；G2恰好为`1,0,1,0`，G4为`1,0.03,0.97,0.03`，因而在W1和W3避开SAC、在W2/W4转向SAC。G3为`1,1,0.91,0.96`，几乎一直停在A2C，所以失去W4的SAC上涨。",
        "- **PPO+SAC：成功来自避开危机期SAC，再有限参与W4。** G2和G4在W1/W3的SAC暴露为0，分别在W4增加到0.22和0.51，所以稳定超过PPO。G1在W1/W3错误暴露SAC为0.25/0.47；G3在W3的SAC暴露高0.91，两者都在不利窗口切换。",
        "- **G3的失败是机制没有充分启动，而不是启动后命中率低。** 其分支分化率只有34.6%，选中块只有14.2%，大多数时间继承了本就可能选错块的causal fallback。",
        "- **G5的表现与投票稀释/抵消及重拟合不稳定一致，但不能由单年数据确定唯一因果。** 它的分支分化率75.9%、选中块58.9%，均低于G2/G4；A2C+SAC和PPO+SAC的30次胜最强比例分别只有36.7%和43.3%，因此平均结果为正但不稳定。",
        "",
        "## 结论",
        "",
        "1. 组合有效的第一条件是存在足够大的分阶段互补空间，而不只是两个模型曾经交替胜出。",
        "2. 第二条件是分类器投票必须产生足够的分支分化和激活覆盖。只在少数日期上高命中不足以改变全年结果。",
        "3. 第三条件是激活后要提高对已实现块赢家的暴露，同时不能在暴跌块过度暴露于高回撤弱模型。",
        "4. 以上是2020年事后机制归因，不能直接转化为事前阈值。",
    ]
    (output / "CAUSE_ANALYSIS_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base_daily, base_blocks, pair_features = load_base_returns(
        Path(args.candidate_root).resolve()
    )
    pair_features = add_causal_baseline(
        pair_features, Path(args.run_metrics).resolve()
    )
    daily = enrich_daily_decisions(
        Path(args.daily_decisions).resolve(),
        base_daily,
        base_blocks,
        pair_features,
    )
    config_window, config = aggregate_attribution(
        daily,
        Path(args.run_metrics).resolve(),
        Path(args.config_features).resolve(),
    )
    ideal = (
        base_blocks[["pair", "window", "block_winner"]]
        .drop_duplicates()
        .merge(
            pair_features[["pair", "weaker_model"]],
            on="pair",
            validate="many_to_one",
        )
    )
    ideal["ideal_weaker_exposure"] = (
        ideal["block_winner"] == ideal["weaker_model"]
    ).astype(float)
    config_window = config_window.merge(
        ideal[["pair", "window", "ideal_weaker_exposure"]],
        on=["pair", "window"],
        validate="many_to_one",
    )
    config, groups = status_and_group_summary(Path(args.block_audit).resolve(), config)

    pair_features.to_csv(output / "pair_complementarity.csv", index=False)
    base_blocks.to_csv(output / "base_model_window_metrics.csv", index=False)
    config_window.to_csv(output / "configuration_window_attribution.csv", index=False)
    config_window.pivot_table(
        index=["pair_label", "classifier_group"],
        columns="window",
        values="weaker_exposure_rate",
    ).to_csv(output / "weaker_exposure_by_window.csv")
    config.to_csv(output / "configuration_cause_features.csv", index=False)
    groups.to_csv(output / "classifier_group_cause_summary.csv", index=False)
    plot_causes(pair_features, config_window, config, output)
    write_report(output, pair_features, base_blocks, config_window, config, groups)

    manifest = {
        "period": PERIOD,
        "excludes_2021": True,
        "daily_rows": int(len(daily)),
        "configurations": int(len(config)),
        "configuration_windows": int(len(config_window)),
        "interpretation": "retrospective_mechanism_attribution_not_admission_rule",
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print((output / "CAUSE_ANALYSIS_ZH.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
