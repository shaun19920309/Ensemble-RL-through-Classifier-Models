from __future__ import annotations

import numpy as np
import pytest

from finrl.reproduction.classifier_ensemble import holding_dispersion
from finrl.reproduction.disagreement import estimate_shrinkage_covariance
from finrl.reproduction.disagreement import holding_disagreement
from finrl.reproduction.disagreement import l1_portfolio_distance
from finrl.reproduction.disagreement import market_value_weights
from finrl.reproduction.disagreement import risk_weighted_portfolio_distance


def test_original_metric_is_bitwise_compatible_with_v1() -> None:
    holdings = np.asarray([[0.0, 4.0, 10.0], [6.0, 2.0, 1.0]])
    assert holding_disagreement("original", holdings) == holding_dispersion(holdings)


def test_l1_distance_includes_cash_and_reaches_one_for_disjoint_portfolios() -> None:
    shares = np.asarray([[0.0, 0.0], [10.0, 0.0]])
    risky, cash = market_value_weights(
        shares,
        prices=np.asarray([100.0, 50.0]),
        cash_balances=np.asarray([1000.0, 0.0]),
    )
    assert l1_portfolio_distance(risky, cash) == pytest.approx(1.0)


def test_l1_distance_is_zero_for_identical_portfolios() -> None:
    shares = np.asarray([[2.0, 3.0], [2.0, 3.0]])
    risky, cash = market_value_weights(
        shares,
        prices=np.asarray([10.0, 20.0]),
        cash_balances=np.asarray([40.0, 40.0]),
    )
    assert l1_portfolio_distance(risky, cash) == pytest.approx(0.0)


def test_risk_distance_is_normalized_and_covariance_scale_invariant() -> None:
    risky = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    covariance = np.eye(2)
    expected = np.sqrt(2.0) / 2.0
    assert risk_weighted_portfolio_distance(risky, covariance) == pytest.approx(expected)
    assert risk_weighted_portfolio_distance(risky, covariance * 17.0) == pytest.approx(
        expected
    )


def test_risk_distance_rejects_non_psd_covariance() -> None:
    risky = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        risk_weighted_portfolio_distance(risky, np.asarray([[1.0, 2.0], [2.0, 1.0]]))


def test_shrinkage_covariance_is_symmetric_psd() -> None:
    prices = np.asarray(
        [
            [100.0, 50.0, 80.0],
            [101.0, 49.0, 81.0],
            [100.5, 50.5, 82.0],
            [102.0, 51.0, 81.5],
            [103.0, 50.8, 83.0],
        ]
    )
    covariance = estimate_shrinkage_covariance(prices)
    assert np.allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12


def test_original_can_report_zero_for_large_uniform_share_disagreement() -> None:
    shares = np.asarray([[0.0, 0.0], [100.0, 100.0]])
    assert holding_disagreement("original", shares) == pytest.approx(0.0)
    assert holding_disagreement(
        "l1",
        shares,
        prices=np.asarray([10.0, 10.0]),
        cash_balances=np.asarray([2000.0, 0.0]),
    ) == pytest.approx(1.0)
