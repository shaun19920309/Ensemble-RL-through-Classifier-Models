from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


METRIC_COLUMNS = ["cumulative_return", "sharpe", "calmar", "max_drawdown"]


def daily_returns(values: pd.Series | np.ndarray) -> pd.Series:
    """Return the finite one-period returns from an account-value series."""
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def max_drawdown(values: pd.Series | np.ndarray) -> float:
    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        return float("nan")
    drawdown = series / series.cummax() - 1.0
    return float(drawdown.min())


def sharpe_ratio(
    values: pd.Series | np.ndarray,
    periods_per_year: int = 252,
    annual_risk_free_rate: float = 0.0,
) -> float:
    returns = daily_returns(values)
    if returns.empty or returns.std() == 0:
        return 0.0
    daily_risk_free_rate = (1.0 + annual_risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = returns - daily_risk_free_rate
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std())


def cumulative_return(values: pd.Series | np.ndarray) -> float:
    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        return float("nan")
    return float(series.iloc[-1] / series.iloc[0] - 1.0)


def annualized_return(
    values: pd.Series | np.ndarray, periods_per_year: int = 252
) -> float:
    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    periods = len(series) - 1
    if periods <= 0 or series.iloc[0] <= 0 or series.iloc[-1] <= 0:
        return float("nan")
    growth = float(series.iloc[-1] / series.iloc[0])
    return float(growth ** (periods_per_year / periods) - 1.0)


def calmar_ratio(values: pd.Series | np.ndarray, periods_per_year: int = 252) -> float:
    mdd = abs(max_drawdown(values))
    if mdd == 0:
        return 0.0
    return float(annualized_return(values, periods_per_year=periods_per_year) / mdd)


def metrics_from_account_values(values: pd.Series | np.ndarray) -> dict[str, float]:
    return {
        "cumulative_return": cumulative_return(values),
        "annualized_return": annualized_return(values),
        "sharpe": sharpe_ratio(values),
        "calmar": calmar_ratio(values),
        "max_drawdown": max_drawdown(values),
    }


def load_paper_reference_summary(results_dir: str | Path) -> pd.DataFrame:
    """Read the paper CSV exports and keep the best tau by ensemble Sharpe."""
    root = Path(results_dir)
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("*_average_metrics_results.csv")):
        stem = path.name.replace("_average_metrics_results.csv", "")
        pair, group_text = stem.rsplit("_", 1)
        df = pd.read_csv(path)
        if df.shape[1] < 5:
            continue
        # The local paper CSVs keep ensemble metrics in columns 1:5 even when
        # the header text says a2c_ppo for every pair.
        metric_block = df.iloc[:, :5].copy()
        metric_block.columns = ["tau", *METRIC_COLUMNS]
        best = metric_block.loc[metric_block["sharpe"].astype(float).idxmax()]
        rows.append(
            {
                "pair": pair.replace("a2cppo", "a2c_ppo")
                .replace("a2csac", "a2c_sac")
                .replace("pposac", "ppo_sac"),
                "classifier_group": int(group_text),
                "source_file": path.name,
                **{key: float(best[key]) for key in ["tau", *METRIC_COLUMNS]},
            }
        )
    return pd.DataFrame(rows).sort_values(["pair", "classifier_group"]).reset_index(drop=True)


def compare_with_reference(reproduced: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    keys = ["pair", "classifier_group"]
    merged = reproduced.merge(reference, on=keys, how="left", suffixes=("_repro", "_paper"))
    for metric in ["tau", *METRIC_COLUMNS]:
        repro_col = f"{metric}_repro"
        paper_col = f"{metric}_paper"
        if repro_col in merged and paper_col in merged:
            merged[f"{metric}_delta"] = merged[repro_col] - merged[paper_col]
    return merged
