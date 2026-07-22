from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import warnings
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


MODEL_NAMES = ("arima", "xgboost", "lstm", "patchtst", "itransformer")


@dataclass(frozen=True)
class ForecastConfig:
    lookback: int = 20
    arima_order: tuple[int, int, int] = (1, 0, 1)
    arima_method: str = "innovations_mle"
    xgb_estimators: int = 300
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.03
    lstm_hidden_size: int = 64
    lstm_layers: int = 2
    lstm_dropout: float = 0.10
    lstm_epochs: int = 50
    lstm_batch_size: int = 64
    lstm_learning_rate: float = 1e-3
    lstm_patience: int = 8
    transformer_d_model: int = 64
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_ffn: int = 128
    transformer_dropout: float = 0.10
    transformer_epochs: int = 50
    transformer_batch_size: int = 64
    transformer_learning_rate: float = 1e-3
    transformer_weight_decay: float = 1e-4
    transformer_patience: int = 8
    patch_length: int = 4
    patch_stride: int = 2
    validation_fraction: float = 0.10
    risk_window: int = 20
    softmax_temperature: float = 1.0
    max_weight: float = 0.20
    gross_exposure: float = 0.95


def ordered_asset_labels(data: pd.DataFrame) -> list[str]:
    if data.empty or not {"date", "tic"}.issubset(data.columns):
        raise ValueError("asset-order data must contain date and tic columns")
    dates = data["date"].astype(str)
    first_date = dates.min()
    labels = data.loc[dates == first_date, "tic"].astype(str).tolist()
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("first trading date must contain each asset exactly once")
    return labels


def price_matrix(
    data: pd.DataFrame,
    *,
    preferred_column: str = "adjcp",
) -> pd.DataFrame:
    required = {"date", "tic"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"price data is missing columns: {sorted(missing)}")
    price_column = preferred_column if preferred_column in data else "close"
    if price_column not in data:
        raise ValueError("price data must contain adjcp or close")
    panel = (
        data.pivot(index="date", columns="tic", values=price_column)
        .sort_index()
        .sort_index(axis=1)
        .astype(float)
    )
    if panel.empty or panel.isna().any().any():
        raise ValueError("price matrix must be non-empty and complete")
    values = panel.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("prices must be finite and strictly positive")
    panel.index = panel.index.astype(str)
    return panel


def log_return_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices).diff().dropna()
    if returns.empty or not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("log-return matrix must be non-empty and finite")
    return returns


def supervised_samples(
    returns: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2:
        raise ValueError("returns must be shaped (dates, assets)")
    if lookback < 2 or len(values) <= lookback:
        raise ValueError("training returns do not cover the requested lookback")
    inputs = np.stack(
        [values[target - lookback : target] for target in range(lookback, len(values))]
    )
    targets = values[lookback:].copy()
    return inputs, targets


def evaluation_inputs(
    train_returns: np.ndarray,
    evaluation_returns: np.ndarray,
    lookback: int,
) -> np.ndarray:
    train = np.asarray(train_returns, dtype=float)
    evaluation = np.asarray(evaluation_returns, dtype=float)
    if train.ndim != 2 or evaluation.ndim != 2 or train.shape[1] != evaluation.shape[1]:
        raise ValueError("train and evaluation returns must share an asset dimension")
    if len(evaluation) < 2:
        raise ValueError("evaluation requires at least two dates")
    combined = np.vstack([train, evaluation])
    first_decision = len(train)
    inputs = []
    for evaluation_index in range(len(evaluation) - 1):
        decision = first_decision + evaluation_index
        start = decision - lookback + 1
        if start < 0:
            raise ValueError("history does not cover the requested lookback")
        inputs.append(combined[start : decision + 1])
    return np.stack(inputs)


def _validate_forecasts(
    forecasts: np.ndarray,
    evaluation_returns: np.ndarray,
) -> np.ndarray:
    values = np.asarray(forecasts, dtype=float)
    expected = (len(evaluation_returns) - 1, evaluation_returns.shape[1])
    if values.shape != expected:
        raise ValueError(f"forecast shape {values.shape} does not match {expected}")
    if not np.isfinite(values).all():
        raise ValueError("forecasts must be finite")
    return values


def arima_forecasts(
    train_returns: np.ndarray,
    evaluation_returns: np.ndarray,
    *,
    order: tuple[int, int, int] = (1, 0, 1),
    method: str = "innovations_mle",
) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    train = np.asarray(train_returns, dtype=float)
    evaluation = np.asarray(evaluation_returns, dtype=float)
    predictions = np.empty((len(evaluation) - 1, train.shape[1]), dtype=float)
    for asset in range(train.shape[1]):
        specification = ARIMA(
            train[:, asset],
            order=order,
            trend="c" if order[1] == 0 else "t",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            fitted = specification.fit(method=method)
        if not np.isfinite(fitted.params).all() or not np.isfinite(fitted.aic):
            raise RuntimeError(f"ARIMA{order} produced invalid estimates for asset {asset}")
        state = fitted
        for step in range(len(evaluation) - 1):
            state = state.append(evaluation[step : step + 1, asset], refit=False)
            predictions[step, asset] = float(state.forecast(steps=1)[0])
    return _validate_forecasts(predictions, evaluation)


def xgboost_forecasts(
    train_returns: np.ndarray,
    evaluation_returns: np.ndarray,
    *,
    lookback: int,
    estimators: int,
    max_depth: int,
    learning_rate: float,
    seed: int,
) -> np.ndarray:
    train_x, train_y = supervised_samples(train_returns, lookback)
    predict_x = evaluation_inputs(train_returns, evaluation_returns, lookback)
    flat_train = train_x.reshape(len(train_x), -1)
    flat_predict = predict_x.reshape(len(predict_x), -1)
    target_mean = train_y.mean(axis=0)
    target_scale = train_y.std(axis=0, ddof=0)
    target_scale = np.where(target_scale > 1e-8, target_scale, 1.0)
    scaled_targets = (train_y - target_mean) / target_scale

    worker_config = {
        "estimators": int(estimators),
        "max_depth": int(max_depth),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
    }
    with tempfile.TemporaryDirectory(prefix="finrl-xgboost-") as temporary:
        root = Path(temporary)
        input_path = root / "input.npz"
        output_path = root / "output.npy"
        np.savez_compressed(
            input_path,
            train_x=flat_train,
            train_y=scaled_targets,
            predict_x=flat_predict,
        )
        environment = os.environ.copy()
        project_root = str(Path(__file__).resolve().parents[2])
        existing_path = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            project_root if not existing_path else f"{project_root}{os.pathsep}{existing_path}"
        )
        environment["OMP_NUM_THREADS"] = "1"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "finrl.reproduction.xgboost_worker",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--config",
                json.dumps(worker_config),
            ],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "isolated XGBoost worker failed: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )
        scaled_predictions = np.load(output_path)
    predictions = scaled_predictions * target_scale + target_mean
    return _validate_forecasts(predictions, evaluation_returns)


def lstm_forecasts(
    train_returns: np.ndarray,
    evaluation_returns: np.ndarray,
    *,
    lookback: int,
    hidden_size: int,
    layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    validation_fraction: float,
    seed: int,
) -> np.ndarray:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torch.utils.data import TensorDataset

    class LSTMRegressor(nn.Module):
        def __init__(self, asset_count: int) -> None:
            super().__init__()
            self.encoder = nn.LSTM(
                input_size=asset_count,
                hidden_size=int(hidden_size),
                num_layers=int(layers),
                dropout=float(dropout) if layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Linear(int(hidden_size), asset_count)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            encoded, _state = self.encoder(values)
            return self.head(encoded[:, -1, :])

    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    train = np.asarray(train_returns, dtype=float)
    mean = train.mean(axis=0)
    scale = train.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    scaled_train = (train - mean) / scale
    train_x, train_y = supervised_samples(scaled_train, lookback)

    validation_count = max(1, int(round(len(train_x) * validation_fraction)))
    if validation_count >= len(train_x):
        raise ValueError("LSTM validation split leaves no training samples")
    split = len(train_x) - validation_count
    x_fit = torch.as_tensor(train_x[:split], dtype=torch.float32)
    y_fit = torch.as_tensor(train_y[:split], dtype=torch.float32)
    x_validation = torch.as_tensor(train_x[split:], dtype=torch.float32)
    y_validation = torch.as_tensor(train_y[split:], dtype=torch.float32)

    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(x_fit, y_fit),
        batch_size=min(int(batch_size), len(x_fit)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = LSTMRegressor(train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    loss_function = nn.MSELoss()
    best_loss = np.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for _epoch in range(int(epochs)):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(x_validation), y_validation))
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                break

    if best_state is None:
        raise RuntimeError("LSTM training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    scaled_evaluation = (np.asarray(evaluation_returns, dtype=float) - mean) / scale
    predict_x = evaluation_inputs(scaled_train, scaled_evaluation, lookback)
    with torch.no_grad():
        scaled_predictions = model(
            torch.as_tensor(predict_x, dtype=torch.float32)
        ).numpy()
    predictions = scaled_predictions * scale + mean
    return _validate_forecasts(predictions, evaluation_returns)


def transformer_forecasts(
    model_name: str,
    train_returns: np.ndarray,
    evaluation_returns: np.ndarray,
    *,
    lookback: int,
    d_model: int,
    heads: int,
    layers: int,
    ffn: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    validation_fraction: float,
    patch_length: int,
    patch_stride: int,
    seed: int,
) -> np.ndarray:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torch.utils.data import TensorDataset

    if model_name not in {"patchtst", "itransformer"}:
        raise ValueError(f"unsupported Transformer forecaster: {model_name}")
    if d_model <= 0 or heads <= 0 or d_model % heads:
        raise ValueError("d_model must be positive and divisible by heads")
    if layers <= 0 or ffn <= 0 or epochs <= 0 or batch_size <= 0:
        raise ValueError("Transformer dimensions and training limits must be positive")
    if not 0.0 <= dropout < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("dropout and validation_fraction are outside valid ranges")
    if model_name == "patchtst" and (
        patch_length <= 0 or patch_stride <= 0 or patch_length > lookback
    ):
        raise ValueError("PatchTST patch settings are incompatible with lookback")

    class PatchTSTRegressor(nn.Module):
        def __init__(self, asset_count: int) -> None:
            super().__init__()
            self.asset_count = asset_count
            self.patch_count = 1 + (lookback - patch_length) // patch_stride
            self.patch_embedding = nn.Linear(patch_length, d_model)
            self.position = nn.Parameter(torch.zeros(1, self.patch_count, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=ffn,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=layers,
                norm=nn.LayerNorm(d_model),
                enable_nested_tensor=False,
            )
            self.head = nn.Linear(self.patch_count * d_model, 1)
            nn.init.normal_(self.position, mean=0.0, std=0.02)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            patches = values.transpose(1, 2).unfold(
                dimension=2,
                size=patch_length,
                step=patch_stride,
            )
            batch, assets, patch_count, _patch_size = patches.shape
            tokens = self.patch_embedding(patches).reshape(
                batch * assets, patch_count, d_model
            )
            encoded = self.encoder(tokens + self.position)
            return self.head(encoded.flatten(start_dim=1)).reshape(batch, assets)

    class ITransformerRegressor(nn.Module):
        def __init__(self, asset_count: int) -> None:
            super().__init__()
            self.asset_count = asset_count
            self.temporal_embedding = nn.Linear(lookback, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=ffn,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=layers,
                norm=nn.LayerNorm(d_model),
                enable_nested_tensor=False,
            )
            self.head = nn.Linear(d_model, 1)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            means = values.mean(dim=1, keepdim=True).detach()
            centered = values - means
            scale = torch.sqrt(
                centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            tokens = self.temporal_embedding(
                (centered / scale).transpose(1, 2)
            )
            encoded = self.encoder(tokens)
            normalized_forecast = self.head(encoded).squeeze(-1)
            return normalized_forecast * scale.squeeze(1) + means.squeeze(1)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    train = np.asarray(train_returns, dtype=float)
    mean = train.mean(axis=0)
    scale = train.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    scaled_train = (train - mean) / scale
    scaled_evaluation = (np.asarray(evaluation_returns, dtype=float) - mean) / scale
    train_x, train_y = supervised_samples(scaled_train, lookback)
    predict_x = evaluation_inputs(scaled_train, scaled_evaluation, lookback)

    validation_count = max(1, int(round(len(train_x) * validation_fraction)))
    if validation_count >= len(train_x):
        raise ValueError("Transformer validation split leaves no training samples")
    split = len(train_x) - validation_count
    x_fit = torch.as_tensor(train_x[:split], dtype=torch.float32)
    y_fit = torch.as_tensor(train_y[:split], dtype=torch.float32)
    x_validation = torch.as_tensor(train_x[split:], dtype=torch.float32)
    y_validation = torch.as_tensor(train_y[split:], dtype=torch.float32)

    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(x_fit, y_fit),
        batch_size=min(int(batch_size), len(x_fit)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model: nn.Module
    if model_name == "patchtst":
        model = PatchTSTRegressor(train.shape[1])
    else:
        model = ITransformerRegressor(train.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    loss_function = nn.MSELoss()
    best_loss = np.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for _epoch in range(int(epochs)):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(model(x_validation), y_validation))
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                break

    if best_state is None:
        raise RuntimeError(f"{model_name} training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(predict_x), int(batch_size)):
            batch = torch.as_tensor(
                predict_x[start : start + int(batch_size)], dtype=torch.float32
            )
            predictions.append(model(batch).numpy())
    scaled_predictions = np.concatenate(predictions, axis=0)
    restored = scaled_predictions * scale + mean
    return _validate_forecasts(restored, evaluation_returns)


def fit_predict(
    model_name: str,
    train_returns: np.ndarray,
    evaluation_returns: np.ndarray,
    config: ForecastConfig,
    *,
    seed: int,
) -> np.ndarray:
    if model_name == "arima":
        return arima_forecasts(
            train_returns,
            evaluation_returns,
            order=config.arima_order,
            method=config.arima_method,
        )
    if model_name == "xgboost":
        return xgboost_forecasts(
            train_returns,
            evaluation_returns,
            lookback=config.lookback,
            estimators=config.xgb_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            seed=seed,
        )
    if model_name == "lstm":
        return lstm_forecasts(
            train_returns,
            evaluation_returns,
            lookback=config.lookback,
            hidden_size=config.lstm_hidden_size,
            layers=config.lstm_layers,
            dropout=config.lstm_dropout,
            epochs=config.lstm_epochs,
            batch_size=config.lstm_batch_size,
            learning_rate=config.lstm_learning_rate,
            patience=config.lstm_patience,
            validation_fraction=config.validation_fraction,
            seed=seed,
        )
    if model_name in {"patchtst", "itransformer"}:
        return transformer_forecasts(
            model_name,
            train_returns,
            evaluation_returns,
            lookback=config.lookback,
            d_model=config.transformer_d_model,
            heads=config.transformer_heads,
            layers=config.transformer_layers,
            ffn=config.transformer_ffn,
            dropout=config.transformer_dropout,
            epochs=config.transformer_epochs,
            batch_size=config.transformer_batch_size,
            learning_rate=config.transformer_learning_rate,
            weight_decay=config.transformer_weight_decay,
            patience=config.transformer_patience,
            validation_fraction=config.validation_fraction,
            patch_length=config.patch_length,
            patch_stride=config.patch_stride,
            seed=seed,
        )
    raise ValueError(f"unknown forecasting model: {model_name}")


def _capped_weights(values: np.ndarray, max_weight: float) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    if raw.ndim != 1 or np.any(raw < 0.0) or not np.isfinite(raw).all():
        raise ValueError("raw weights must be a finite non-negative vector")
    asset_count = len(raw)
    if max_weight * asset_count < 1.0 - 1e-12:
        raise ValueError("max_weight is infeasible for the asset count")
    if raw.sum() <= 0.0:
        raw = np.ones(asset_count, dtype=float)
    weights = np.zeros(asset_count, dtype=float)
    active = np.ones(asset_count, dtype=bool)
    remaining = 1.0
    while active.any():
        active_raw = raw[active]
        if active_raw.sum() <= 0.0:
            proposal = np.full(active.sum(), remaining / active.sum())
        else:
            proposal = active_raw / active_raw.sum() * remaining
        over = proposal > max_weight + 1e-12
        active_indices = np.flatnonzero(active)
        if not over.any():
            weights[active_indices] = proposal
            break
        capped_indices = active_indices[over]
        weights[capped_indices] = max_weight
        active[capped_indices] = False
        remaining = 1.0 - weights.sum()
    weights /= weights.sum()
    return weights


def forecasts_to_weights(
    forecasts: np.ndarray,
    train_returns: np.ndarray,
    evaluation_returns: np.ndarray,
    *,
    risk_window: int,
    temperature: float,
    max_weight: float,
) -> np.ndarray:
    predictions = _validate_forecasts(forecasts, evaluation_returns)
    if risk_window < 2 or temperature <= 0.0:
        raise ValueError("risk_window and temperature must be positive")
    combined = np.vstack([train_returns, evaluation_returns])
    first_decision = len(train_returns)
    rows = []
    for step, forecast in enumerate(predictions):
        decision = first_decision + step
        history = combined[max(0, decision - risk_window + 1) : decision + 1]
        volatility = history.std(axis=0, ddof=1)
        positive = volatility[np.isfinite(volatility) & (volatility > 1e-8)]
        floor = float(np.median(positive)) if positive.size else 1.0
        volatility = np.where(
            np.isfinite(volatility) & (volatility > 1e-8), volatility, floor
        )
        score = forecast / volatility
        score_std = float(score.std(ddof=0))
        standardized = (
            (score - score.mean()) / score_std if score_std > 1e-12 else np.zeros_like(score)
        )
        logits = standardized / temperature
        logits -= logits.max()
        raw = np.exp(logits)
        rows.append(_capped_weights(raw, max_weight))
    return np.asarray(rows, dtype=float)


def shares_from_state(state: Iterable[float], stock_dim: int) -> np.ndarray:
    values = np.asarray(list(state), dtype=float)
    return values[1 + stock_dim : 1 + 2 * stock_dim]


def prices_from_state(state: Iterable[float], stock_dim: int) -> np.ndarray:
    values = np.asarray(list(state), dtype=float)
    return values[1 : 1 + stock_dim]


def initialize_selected_holding_environment(
    data: pd.DataFrame,
    env_options: dict[str, object],
    *,
    initial: bool = True,
    previous_state: list[float] | None = None,
) -> StockTradingEnv:
    options = dict(env_options)
    options.update(
        initial=initial,
        previous_state=[] if previous_state is None else previous_state,
    )
    environment = StockTradingEnv(df=data, **options)
    environment.reset()
    return environment


def step_target_holding_branch(
    environment: StockTradingEnv,
    target_holding: np.ndarray,
    *,
    hmax: float,
) -> tuple[StockTradingEnv, float]:
    """Execute one target on a lightweight branch of the same trading state."""

    stock_dim = int(environment.stock_dim)
    target = np.asarray(target_holding, dtype=float)
    if target.shape != (stock_dim,) or not np.isfinite(target).all():
        raise ValueError("target_holding must be a finite stock-dimensional vector")
    if hmax <= 0.0:
        raise ValueError("hmax must be positive")
    if environment.day >= len(environment.df.index.unique()) - 1:
        raise ValueError("cannot branch a terminal trading environment")

    branch = copy(environment)
    branch.state = list(environment.state)
    branch.asset_memory = list(environment.asset_memory)
    branch.rewards_memory = list(environment.rewards_memory)
    branch.actions_memory = list(environment.actions_memory)
    branch.state_memory = list(environment.state_memory)
    branch.date_memory = list(environment.date_memory)
    current = shares_from_state(branch.render(), stock_dim)
    action = np.clip((target - current) / float(hmax), -1.0, 1.0)
    branch.step(action)
    realized_return = float(branch.asset_memory[-1] / branch.asset_memory[-2] - 1.0)
    return branch, realized_return


def simulate_weight_strategy(
    weights: np.ndarray,
    data: pd.DataFrame,
    env_options: dict[str, object],
    *,
    gross_exposure: float,
    initial: bool = True,
    previous_state: list[float] | None = None,
) -> tuple[np.ndarray, pd.DataFrame, list[float]]:
    targets = np.asarray(weights, dtype=float)
    expected_steps = len(data.index.unique()) - 1
    stock_dim = int(env_options["stock_dim"])
    if targets.shape != (expected_steps, stock_dim):
        raise ValueError(
            f"weight shape {targets.shape} does not match {(expected_steps, stock_dim)}"
        )
    if not 0.0 < gross_exposure <= 1.0:
        raise ValueError("gross_exposure must be in (0, 1]")

    options = dict(env_options)
    options.update(
        initial=initial,
        previous_state=[] if previous_state is None else previous_state,
    )
    environment = StockTradingEnv(df=data, **options)
    vector_env, _observation = environment.get_sb_env()
    holdings = []
    for target_weights in targets:
        state = list(environment.render())
        prices = prices_from_state(state, stock_dim)
        current = shares_from_state(state, stock_dim)
        account_value = float(state[0] + prices @ current)
        target_shares = np.floor(
            account_value * gross_exposure * target_weights / prices
        )
        action = np.clip(
            (target_shares - current) / float(env_options["hmax"]), -1.0, 1.0
        )
        _observation, _reward, done, _info = vector_env.step(np.asarray([action]))
        holdings.append(shares_from_state(environment.render(), stock_dim))
        if done[0]:
            break
    return (
        np.asarray(holdings, dtype=float),
        environment.save_asset_memory(),
        list(environment.render()),
    )


def simulate_selected_holdings(
    selected_holdings: np.ndarray,
    data: pd.DataFrame,
    env_options: dict[str, object],
    *,
    initial: bool = True,
    previous_state: list[float] | None = None,
) -> tuple[pd.DataFrame, list[float]]:
    targets = np.asarray(selected_holdings, dtype=float)
    expected_steps = len(data.index.unique()) - 1
    stock_dim = int(env_options["stock_dim"])
    if targets.shape != (expected_steps, stock_dim):
        raise ValueError(
            f"holding shape {targets.shape} does not match {(expected_steps, stock_dim)}"
        )
    options = dict(env_options)
    options.update(
        initial=initial,
        previous_state=[] if previous_state is None else previous_state,
    )
    environment = StockTradingEnv(df=data, **options)
    vector_env, _observation = environment.get_sb_env()
    for target in targets:
        current = shares_from_state(environment.render(), stock_dim)
        action = np.clip(
            (target - current) / float(env_options["hmax"]), -1.0, 1.0
        )
        _observation, _reward, done, _info = vector_env.step(np.asarray([action]))
        if done[0]:
            break
    return environment.save_asset_memory(), list(environment.render())
