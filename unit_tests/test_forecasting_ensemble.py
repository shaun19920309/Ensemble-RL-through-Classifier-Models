from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finrl.reproduction.forecasting_ensemble import evaluation_inputs
from finrl.reproduction.forecasting_ensemble import ForecastConfig
from finrl.reproduction.forecasting_ensemble import fit_predict
from finrl.reproduction.forecasting_ensemble import forecasts_to_weights
from finrl.reproduction.forecasting_ensemble import log_return_matrix
from finrl.reproduction.forecasting_ensemble import ordered_asset_labels
from finrl.reproduction.forecasting_ensemble import price_matrix
from finrl.reproduction.forecasting_ensemble import supervised_samples
from finrl.reproduction.forecasting_ensemble import transformer_forecasts


def test_supervised_samples_use_only_history_for_next_return():
    returns = np.arange(12, dtype=float).reshape(6, 2)
    inputs, targets = supervised_samples(returns, lookback=3)

    np.testing.assert_array_equal(inputs[0], returns[0:3])
    np.testing.assert_array_equal(targets[0], returns[3])
    np.testing.assert_array_equal(inputs[-1], returns[2:5])
    np.testing.assert_array_equal(targets[-1], returns[5])


def test_evaluation_inputs_include_current_observation_but_not_future():
    train = np.arange(10, dtype=float).reshape(5, 2)
    evaluation = np.arange(10, 18, dtype=float).reshape(4, 2)
    inputs = evaluation_inputs(train, evaluation, lookback=3)

    np.testing.assert_array_equal(inputs[0], np.vstack([train[-2:], evaluation[0]]))
    np.testing.assert_array_equal(
        inputs[1], np.vstack([train[-1], evaluation[0], evaluation[1]])
    )
    assert len(inputs) == len(evaluation) - 1


def test_forecast_weights_are_long_only_capped_and_normalized():
    train = np.linspace(-0.02, 0.03, 60).reshape(20, 3)
    evaluation = np.linspace(-0.01, 0.02, 12).reshape(4, 3)
    forecasts = np.array(
        [[0.02, 0.01, -0.01], [0.01, 0.03, 0.00], [-0.02, 0.01, 0.04]]
    )
    weights = forecasts_to_weights(
        forecasts,
        train,
        evaluation,
        risk_window=5,
        temperature=1.0,
        max_weight=0.60,
    )

    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 0.60 + 1e-12)


def test_price_and_return_panels_are_complete_and_ordered():
    data = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"],
            "tic": ["B", "A", "B", "A"],
            "close": [20.0, 10.0, 22.0, 11.0],
        }
    )
    prices = price_matrix(data)
    returns = log_return_matrix(prices)

    assert list(prices.columns) == ["A", "B"]
    assert returns.shape == (1, 2)
    np.testing.assert_allclose(returns.iloc[0].to_numpy(), np.log([1.1, 1.1]))


def test_price_matrix_rejects_incomplete_panel():
    data = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01", "2020-01-02"],
            "tic": ["A", "B", "A"],
            "close": [10.0, 20.0, 11.0],
        }
    )
    with pytest.raises(ValueError, match="complete"):
        price_matrix(data)


def test_numeric_ticker_order_matches_the_trading_frame():
    data = pd.DataFrame(
        {
            "date": ["2020-01-01"] * 3 + ["2020-01-02"] * 3,
            "tic": [241, 700, 1347] * 2,
            "close": [10.0, 20.0, 30.0, 11.0, 19.0, 33.0],
        }
    )
    prices = price_matrix(data)

    assert list(map(str, prices.columns)) == ordered_asset_labels(data)
    assert ordered_asset_labels(data) == ["241", "700", "1347"]


@pytest.mark.parametrize("model_name", ["patchtst", "itransformer"])
def test_transformer_forecasters_produce_finite_next_day_cross_sections(model_name):
    rng = np.random.default_rng(123)
    train = rng.normal(0.0, 0.02, size=(48, 4))
    evaluation = rng.normal(0.0, 0.02, size=(8, 4))
    config = ForecastConfig(
        lookback=8,
        transformer_d_model=16,
        transformer_heads=4,
        transformer_layers=1,
        transformer_ffn=32,
        transformer_epochs=2,
        transformer_batch_size=8,
        transformer_patience=2,
        patch_length=4,
        patch_stride=2,
    )

    forecasts = fit_predict(model_name, train, evaluation, config, seed=77)

    assert forecasts.shape == (len(evaluation) - 1, train.shape[1])
    assert np.isfinite(forecasts).all()


def test_patchtst_rejects_patch_longer_than_lookback():
    returns = np.zeros((24, 3), dtype=float)
    with pytest.raises(ValueError, match="patch settings"):
        transformer_forecasts(
            "patchtst",
            returns,
            returns[:5],
            lookback=4,
            d_model=8,
            heads=2,
            layers=1,
            ffn=16,
            dropout=0.1,
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            weight_decay=1e-4,
            patience=1,
            validation_fraction=0.1,
            patch_length=5,
            patch_stride=1,
            seed=1,
        )
