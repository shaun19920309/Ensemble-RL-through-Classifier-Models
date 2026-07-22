from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finrl.reproduction.classifier_ensemble import holding_dispersion
from finrl.reproduction.classifier_ensemble import select_holding_from_confidence
from finrl.reproduction.classifier_ensemble import tau_grid
from finrl.reproduction.classifier_ensemble import train_classifier_group
from finrl.reproduction.metrics import annualized_return
from finrl.reproduction.metrics import metrics_from_account_values


def test_holding_dispersion_is_normalized_average_std():
    holdings = np.array([[10, 20, 30], [20, 20, 10]], dtype=float)
    assert holding_dispersion(holdings) == pytest.approx(0.5)


def test_strict_reproduction_uses_fixed_classifier_and_original_tau_grid():
    assert train_classifier_group.__kwdefaults__["grid_search"] is False
    values = tau_grid()
    assert len(values) == 89
    assert values[0] == pytest.approx(0.01)
    assert values[-1] == pytest.approx(0.89)


def test_decision_block_uses_high_probabilities_when_variance_is_low():
    candidates = np.array([[10, 10, 10], [11, 10, 10]], dtype=float)
    q_matrix = np.array([[0.8, 0.6], [0.7, 0.9], [0.75, 0.4]])
    decision = select_holding_from_confidence(candidates, q_matrix, tau=0.8)
    assert decision.mode == "aggressive"
    assert decision.selected_index == 0
    assert decision.votes.tolist() == [2, 1]


def test_decision_block_uses_low_probabilities_when_variance_is_high():
    candidates = np.array([[0, 100, 0], [100, 0, 80]], dtype=float)
    q_matrix = np.array([[0.8, 0.2], [0.7, 0.3], [0.9, 0.4]])
    decision = select_holding_from_confidence(candidates, q_matrix, tau=0.1)
    assert decision.mode == "conservative"
    assert decision.selected_index == 1
    assert decision.votes.tolist() == [0, 3]


def test_decision_block_breaks_vote_tie_with_branch_specific_mean_confidence():
    candidates = np.array([[10, 10], [11, 10]], dtype=float)
    aggressive_q = np.array([[0.90, 0.40], [0.49, 0.50]])
    aggressive = select_holding_from_confidence(
        candidates,
        aggressive_q,
        tau=1.0,
        dispersion=0.0,
    )
    assert aggressive.votes.tolist() == [1, 1]
    assert aggressive.selected_index == 0

    conservative_q = np.array([[0.10, 0.80], [0.70, 0.20]])
    conservative = select_holding_from_confidence(
        candidates,
        conservative_q,
        tau=0.0,
        dispersion=1.0,
    )
    assert conservative.votes.tolist() == [1, 1]
    assert conservative.selected_index == 0


def test_financial_metrics_match_expected_signs():
    values = pd.Series([100.0, 110.0, 105.0, 120.0])
    metrics = metrics_from_account_values(values)
    assert metrics["cumulative_return"] == pytest.approx(0.2)
    assert metrics["annualized_return"] == pytest.approx(annualized_return(values))
    assert metrics["max_drawdown"] == pytest.approx(105.0 / 110.0 - 1.0)
    assert metrics["sharpe"] > 0
    assert metrics["calmar"] == pytest.approx(
        metrics["annualized_return"] / abs(metrics["max_drawdown"])
    )


def test_rolling_windows_use_train_tail_then_previous_trade_for_calibration():
    script = Path(__file__).resolve().parents[1] / "examples" / "reproduce_classifier_ensemble.py"
    spec = importlib.util.spec_from_file_location("classifier_ensemble_repro", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    dates = pd.bdate_range("2020-01-01", periods=13).strftime("%Y-%m-%d")
    df = pd.DataFrame({"date": np.repeat(dates, 2), "tic": ["A", "B"] * len(dates)})
    windows = module.build_rolling_windows(
        df,
        trade_start_date=dates[6],
        rebalance_window=2,
        validation_window=2,
        max_windows=3,
    )

    assert windows[0]["calibration_source"] == "train_tail"
    assert list(windows[0]["calibration_dates"]) == list(dates[4:6])
    assert list(windows[0]["trade_dates"]) == list(dates[6:8])
    assert list(windows[1]["calibration_dates"]) == list(windows[0]["trade_dates"])
    assert windows[1]["calibration_source"] == "previous_trade"
    assert list(windows[2]["calibration_dates"]) == list(windows[1]["trade_dates"])
    assert list(windows[2]["trade_dates"]) == list(dates[10:13])
