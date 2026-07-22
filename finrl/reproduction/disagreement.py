from __future__ import annotations

from itertools import combinations
from typing import Literal

import numpy as np
from sklearn.covariance import LedoitWolf

from finrl.reproduction.classifier_ensemble import holding_dispersion


DisagreementMetric = Literal["original", "l1", "risk_weighted"]
DISAGREEMENT_METRICS: tuple[DisagreementMetric, ...] = (
    "original",
    "l1",
    "risk_weighted",
)


def market_value_weights(
    share_holdings: np.ndarray,
    prices: np.ndarray,
    cash_balances: np.ndarray,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert candidate share holdings to risky-asset and cash weights."""
    shares = np.asarray(share_holdings, dtype=float)
    price_vector = np.asarray(prices, dtype=float)
    cash = np.asarray(cash_balances, dtype=float)
    if shares.ndim != 2:
        raise ValueError("share_holdings must be shaped (candidates, assets)")
    if price_vector.shape != (shares.shape[1],):
        raise ValueError("prices must contain one value per asset")
    if cash.shape != (shares.shape[0],):
        raise ValueError("cash_balances must contain one value per candidate")
    if not np.isfinite(shares).all() or not np.isfinite(price_vector).all():
        raise ValueError("holdings and prices must be finite")
    if not np.isfinite(cash).all():
        raise ValueError("cash balances must be finite")
    if (shares < -eps).any() or (price_vector <= 0).any() or (cash < -eps).any():
        raise ValueError("V2 portfolio weights require long-only holdings and positive prices")

    risky_notional = np.maximum(shares, 0.0) * price_vector[None, :]
    wealth = cash + risky_notional.sum(axis=1)
    if (wealth <= eps).any():
        raise ValueError("every candidate portfolio must have positive wealth")
    risky_weights = risky_notional / wealth[:, None]
    cash_weights = np.maximum(cash, 0.0) / wealth
    total = risky_weights.sum(axis=1) + cash_weights
    if not np.allclose(total, 1.0, atol=1e-9):
        raise ValueError("portfolio weights do not sum to one")
    return risky_weights, cash_weights


def l1_portfolio_distance(
    risky_weights: np.ndarray,
    cash_weights: np.ndarray,
) -> float:
    """Mean pairwise total-variation distance, including the cash allocation."""
    risky = np.asarray(risky_weights, dtype=float)
    cash = np.asarray(cash_weights, dtype=float)
    if risky.ndim != 2 or cash.shape != (risky.shape[0],):
        raise ValueError("portfolio weights have incompatible shapes")
    if risky.shape[0] < 2:
        raise ValueError("at least two candidate portfolios are required")
    complete = np.column_stack([cash, risky])
    distances = [
        0.5 * float(np.abs(complete[left] - complete[right]).sum())
        for left, right in combinations(range(len(complete)), 2)
    ]
    return float(np.clip(np.mean(distances), 0.0, 1.0))


def risk_weighted_portfolio_distance(
    risky_weights: np.ndarray,
    covariance: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """Mean pairwise covariance-norm distance normalized to the unit interval."""
    risky = np.asarray(risky_weights, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if risky.ndim != 2:
        raise ValueError("risky_weights must be shaped (candidates, assets)")
    if risky.shape[0] < 2:
        raise ValueError("at least two candidate portfolios are required")
    if cov.shape != (risky.shape[1], risky.shape[1]):
        raise ValueError("covariance shape does not match the asset dimension")
    if not np.isfinite(cov).all():
        raise ValueError("covariance must be finite")
    cov = 0.5 * (cov + cov.T)
    minimum_eigenvalue = float(np.linalg.eigvalsh(cov).min())
    if minimum_eigenvalue < -1e-9:
        raise ValueError("covariance must be positive semidefinite")

    def norm(vector: np.ndarray) -> float:
        return float(np.sqrt(max(float(vector @ cov @ vector), 0.0)))

    distances: list[float] = []
    for left, right in combinations(range(len(risky)), 2):
        numerator = norm(risky[left] - risky[right])
        denominator = norm(risky[left]) + norm(risky[right])
        distances.append(0.0 if denominator <= eps else numerator / denominator)
    return float(np.clip(np.mean(distances), 0.0, 1.0))


def estimate_shrinkage_covariance(price_history: np.ndarray) -> np.ndarray:
    """Estimate a stable return covariance from a strictly prior price block."""
    prices = np.asarray(price_history, dtype=float)
    if prices.ndim != 2 or prices.shape[0] < 4 or prices.shape[1] < 1:
        raise ValueError("price_history must contain at least four dates")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("price history must be finite and strictly positive")
    returns = prices[1:] / prices[:-1] - 1.0
    if not np.isfinite(returns).all():
        raise ValueError("price history produced non-finite returns")
    covariance = LedoitWolf(assume_centered=False).fit(returns).covariance_
    covariance = 0.5 * (covariance + covariance.T)
    if covariance.shape != (prices.shape[1], prices.shape[1]):
        raise ValueError("covariance estimator returned an unexpected shape")
    return covariance


def holding_disagreement(
    metric: DisagreementMetric,
    share_holdings: np.ndarray,
    *,
    prices: np.ndarray | None = None,
    cash_balances: np.ndarray | None = None,
    covariance: np.ndarray | None = None,
) -> float:
    """Evaluate one V2 disagreement metric without changing the V1 definition."""
    if metric == "original":
        return holding_dispersion(share_holdings)
    if metric not in DISAGREEMENT_METRICS:
        raise ValueError(f"unknown disagreement metric: {metric}")
    if prices is None or cash_balances is None:
        raise ValueError(f"{metric} requires prices and cash balances")
    risky, cash = market_value_weights(share_holdings, prices, cash_balances)
    if metric == "l1":
        return l1_portfolio_distance(risky, cash)
    if covariance is None:
        raise ValueError("risk_weighted requires a prior-block covariance matrix")
    return risk_weighted_portfolio_distance(risky, covariance)
