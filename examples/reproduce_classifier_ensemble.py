from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/finrl-matplotlib")

import numpy as np
import pandas as pd

from finrl.config import INDICATORS
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.reproduction.classifier_ensemble import select_holding
from finrl.reproduction.classifier_ensemble import tau_grid
from finrl.reproduction.classifier_ensemble import train_classifier_group
from finrl.reproduction.metrics import compare_with_reference
from finrl.reproduction.metrics import load_paper_reference_summary
from finrl.reproduction.metrics import metrics_from_account_values


PAPER_TRAIN_START = "2010-01-01"
PAPER_TRAIN_END = "2019-10-01"
PAPER_VALIDATION_START = "2019-10-01"
PAPER_TRADE_START = "2020-01-01"
PAPER_TRADE_END = "2021-01-01"

BASE_MODEL_PARAMS = {
    "a2c": {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007},
    "ppo": {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 64,
    },
    "sac": {
        "batch_size": 128,
        "buffer_size": 100000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    },
    "td3": {
        "batch_size": 128,
        "buffer_size": 100000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
    },
    "tqc": {
        "batch_size": 128,
        "buffer_size": 100000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
        "top_quantiles_to_drop_per_net": 2,
    },
}
METRIC_COLUMNS = ["cumulative_return", "sharpe", "calmar", "max_drawdown"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the classifier-ensemble RL trading experiment."
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full", "rolling"],
        default="rolling",
        help=(
            "smoke tests the decision block, full runs one-shot RL reproduction, "
            "and rolling follows the paper "
            "rebalancing/validation protocol."
        ),
    )
    parser.add_argument("--paper-results-dir", default="../data")
    parser.add_argument("--output-dir", default="work/core_dj30_candidates")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--groups", default="1,2,3,4,5")
    parser.add_argument("--pairs", default="a2c_sac,ppo_sac,a2c_ppo")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument(
        "--data-source",
        choices=["yahoo", "trademaster", "synthetic"],
        default="trademaster",
    )
    parser.add_argument("--trademaster-data-dir", default="external_data/trademaster_dj30")
    parser.add_argument("--trademaster-trade-split", choices=["valid", "test"], default="valid")
    parser.add_argument("--trademaster-validation-window", type=int, default=63)
    parser.add_argument("--rebalance-window", type=int, default=63)
    parser.add_argument("--rolling-validation-window", type=int, default=63)
    parser.add_argument(
        "--rl-eval-interval",
        type=int,
        default=20000,
        help="Evaluate and checkpoint RL models on the rolling validation window every N timesteps.",
    )
    parser.add_argument(
        "--rolling-max-windows",
        type=int,
        default=None,
        help="Limit rolling windows for partial/half reproduction runs.",
    )
    parser.add_argument("--tau-start", type=float, default=0.01)
    parser.add_argument("--tau-stop", type=float, default=0.89)
    parser.add_argument("--tau-step", type=float, default=0.01)
    parser.add_argument(
        "--save-decisions",
        action="store_true",
        help="Save rolling decision traces for every pair/group/tau.",
    )
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--synthetic-stocks", type=int, default=5)
    return parser.parse_args()


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class _FixedClassifier:
    classes_ = np.array([0, 1])

    def __init__(self, probabilities: np.ndarray):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if len(x) != len(self.probabilities):
            raise ValueError("fixed classifier received unexpected sample count")
        return self.probabilities


def run_smoke_mode(args: argparse.Namespace) -> None:
    output_dir = ensure_output_dir(args.output_dir)
    candidates = np.array([[10, 12, 9], [11, 10, 10]], dtype=float)
    classifiers = [
        ("fixed_1", _FixedClassifier([[0.82, 0.18], [0.35, 0.65]])),
        ("fixed_2", _FixedClassifier([[0.75, 0.25], [0.40, 0.60]])),
    ]
    decision = select_holding(classifiers, candidates, [0, 1], tau=0.50)
    df = pd.DataFrame(
        [
            {
                "selected_index": decision.selected_index,
                "dispersion": decision.dispersion,
                "mode": decision.mode,
                "votes": decision.votes.tolist(),
                "selected_holding": decision.selected_holding.tolist(),
            }
        ]
    )
    out_path = output_dir / "smoke_decision.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"Saved smoke decision to {out_path}")


def make_synthetic_processed_data(stock_count: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(PAPER_TRAIN_START, PAPER_TRADE_END)
    tickers = DOW_30_TICKER[:stock_count]
    rows = []
    market_turbulence = np.abs(rng.normal(30, 20, len(dates)))
    market_vix = np.abs(rng.normal(18, 8, len(dates)))
    for ticker_index, ticker in enumerate(tickers):
        drift = 0.0001 + ticker_index * 0.00002
        shocks = rng.normal(drift, 0.015, len(dates))
        close = 80 * np.exp(np.cumsum(shocks))
        open_price = close * (1 + rng.normal(0, 0.002, len(dates)))
        high = np.maximum(open_price, close) * (1 + rng.random(len(dates)) * 0.01)
        low = np.minimum(open_price, close) * (1 - rng.random(len(dates)) * 0.01)
        volume = rng.integers(1_000_000, 5_000_000, len(dates))
        frame = pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "tic": ticker,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "day": dates.dayofweek,
                "vix": market_vix,
                "turbulence": market_turbulence,
            }
        )
        for indicator in INDICATORS:
            frame[indicator] = rng.normal(0, 1, len(dates))
        rows.append(frame)
    return pd.concat(rows, ignore_index=True).sort_values(["date", "tic"]).reset_index(drop=True)


def load_or_prepare_data(
    cache_path: Path,
    *,
    synthetic: bool = False,
    synthetic_stocks: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    if synthetic:
        return make_synthetic_processed_data(synthetic_stocks, seed)
    if cache_path.exists():
        return pd.read_csv(cache_path)

    raw = YahooDownloader(
        start_date=PAPER_TRAIN_START,
        end_date=PAPER_TRADE_END,
        ticker_list=DOW_30_TICKER,
    ).fetch_data()
    engineer = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False,
    )
    processed = engineer.preprocess_data(raw)
    processed = processed.sort_values(["date", "tic"]).fillna(0).replace(np.inf, 0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(cache_path, index=False)
    return processed


def trademaster_indicator_list(df: pd.DataFrame) -> list[str]:
    indicators = [
        column
        for column in df.columns
        if column.startswith("z") or column.startswith("zd_")
    ]
    return [column for column in indicators if pd.api.types.is_numeric_dtype(df[column])]


def normalize_trademaster_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    unnamed = [column for column in data.columns if column.startswith("Unnamed")]
    data = data.drop(columns=unnamed, errors="ignore")
    if "date" not in data or "tic" not in data:
        raise ValueError("TradeMaster data must include date and tic columns")
    data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
    data["day"] = pd.to_datetime(data["date"]).dt.dayofweek
    if "turbulence" not in data:
        data["turbulence"] = 0.0
    if "vix" not in data:
        data["vix"] = 0.0
    indicators = trademaster_indicator_list(data)
    keep_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "tic",
        "day",
        "vix",
        "turbulence",
        *indicators,
    ]
    data = data[[column for column in keep_columns if column in data.columns]]
    data = data.replace([np.inf, -np.inf], 0).fillna(0)
    stock_count = data["tic"].nunique()
    complete_dates = data.groupby("date")["tic"].nunique()
    complete_dates = complete_dates[complete_dates == stock_count].index
    data = data[data["date"].isin(complete_dates)]
    data = data.sort_values(["date", "tic"], ignore_index=True)
    data.index = data["date"].factorize()[0]
    return data


def split_tail_window(df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = np.asarray(sorted(df["date"].unique()))
    if len(unique_dates) <= window + 1:
        raise ValueError(
            f"not enough dates ({len(unique_dates)}) for validation window {window}"
        )
    validation_dates = set(unique_dates[-window:])
    train = df[~df["date"].isin(validation_dates)].copy()
    validation = df[df["date"].isin(validation_dates)].copy()
    train.index = train["date"].factorize()[0]
    validation.index = validation["date"].factorize()[0]
    return train, validation


def load_trademaster_splits(
    data_dir: str | Path,
    *,
    trade_split: str = "valid",
    validation_window: int = 63,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    root = Path(data_dir)
    paths = {name: root / f"{name}.csv" for name in ("train", "valid", "test")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing TradeMaster split files: {missing}")
    frames = {
        name: normalize_trademaster_frame(pd.read_csv(path, on_bad_lines="skip"))
        for name, path in paths.items()
    }
    indicators = trademaster_indicator_list(frames["train"])
    if not indicators:
        raise ValueError("could not detect TradeMaster technical indicator columns")
    train, validation = split_tail_window(frames["train"], validation_window)
    trade = frames[trade_split]
    return train, validation, trade, indicators


def load_experiment_splits(
    args: argparse.Namespace, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    data_source = "synthetic" if args.synthetic_data else args.data_source
    if data_source == "trademaster":
        return load_trademaster_splits(
            args.trademaster_data_dir,
            trade_split=args.trademaster_trade_split,
            validation_window=args.trademaster_validation_window,
        )

    processed = load_or_prepare_data(
        output_dir / "cache" / "dow30_2010_2020_processed.csv",
        synthetic=data_source == "synthetic",
        synthetic_stocks=args.synthetic_stocks,
        seed=args.seed,
    )
    indicators = list(INDICATORS)
    return (
        data_split(processed, PAPER_TRAIN_START, PAPER_TRAIN_END),
        data_split(processed, PAPER_VALIDATION_START, PAPER_TRADE_START),
        data_split(processed, PAPER_TRADE_START, PAPER_TRADE_END),
        indicators,
    )


def env_kwargs(processed: pd.DataFrame, tech_indicators: list[str]) -> dict[str, object]:
    stock_dimension = len(processed.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(tech_indicators) * stock_dimension
    return {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [0.001] * stock_dimension,
        "sell_cost_pct": [0.001] * stock_dimension,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": tech_indicators,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "print_verbosity": 50,
    }


def build_env(
    df: pd.DataFrame, kwargs: dict[str, object], **overrides: object
) -> StockTradingEnv:
    env_options = dict(kwargs)
    env_options.update(overrides)
    return StockTradingEnv(df=df, **env_options)


def rl_model_class(model_name: str):
    from stable_baselines3 import A2C
    from stable_baselines3 import PPO
    from stable_baselines3 import SAC
    from stable_baselines3 import TD3

    model_classes = {"a2c": A2C, "ppo": PPO, "sac": SAC, "td3": TD3}
    if model_name == "tqc":
        try:
            from sb3_contrib import TQC
        except ImportError as error:
            raise ImportError(
                "TQC requires sb3-contrib; install requirements-reproduction.txt"
            ) from error
        return TQC
    if model_name not in model_classes:
        raise ValueError(f"unsupported RL model: {model_name}")
    return model_classes[model_name]


def initialize_rl_model(model_name: str, env, seed: int):
    from finrl.agents.stablebaselines3.models import DRLAgent

    if model_name == "tqc":
        return rl_model_class(model_name)(
            policy="MlpPolicy",
            env=env,
            verbose=0,
            seed=seed,
            **BASE_MODEL_PARAMS[model_name],
        )
    return DRLAgent(env=env).get_model(
        model_name,
        model_kwargs=BASE_MODEL_PARAMS[model_name],
        seed=seed,
        verbose=0,
    )


def train_or_load_model(
    model_name: str,
    train_data: pd.DataFrame,
    kwargs: dict[str, object],
    output_dir: Path,
    timesteps: int,
    seed: int,
    force_train: bool,
    model_tag: str | None = None,
):
    from finrl.agents.stablebaselines3.models import DRLAgent

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    tag = model_tag or str(timesteps)
    model_path = model_dir / f"agent_{model_name}_{tag}"
    zip_path = model_path.with_suffix(".zip")
    if zip_path.exists() and not force_train:
        return rl_model_class(model_name).load(str(model_path))

    env_train, _ = build_env(train_data, kwargs).get_sb_env()
    model = initialize_rl_model(model_name, env_train, seed)
    trained = DRLAgent.train_model(
        model=model,
        tb_log_name=f"classifier_ensemble_{model_name}",
        total_timesteps=timesteps,
    )
    trained.save(str(model_path))
    return trained


def train_or_load_validation_selected_model(
    model_name: str,
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    kwargs: dict[str, object],
    output_dir: Path,
    timesteps: int,
    seed: int,
    force_train: bool,
    *,
    model_tag: str,
    eval_interval: int,
) -> tuple[object, pd.DataFrame]:
    from finrl.agents.stablebaselines3.models import TensorboardCallback

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / f"agent_{model_name}_{model_tag}_best"
    best_zip_path = best_model_path.with_suffix(".zip")
    history_path = model_dir / f"agent_{model_name}_{model_tag}_validation_history.csv"

    if best_zip_path.exists() and history_path.exists() and not force_train:
        history = pd.read_csv(history_path)
        return rl_model_class(model_name).load(str(best_model_path)), history

    env_train, _ = build_env(train_data, kwargs).get_sb_env()
    model = initialize_rl_model(model_name, env_train, seed)

    rows = []
    best_sharpe = -np.inf
    best_step = 0
    completed = 0
    interval = max(1, int(eval_interval))
    while completed < timesteps:
        chunk = min(interval, timesteps - completed)
        model.learn(
            total_timesteps=chunk,
            tb_log_name=f"classifier_ensemble_{model_name}_{model_tag}",
            callback=TensorboardCallback(),
            reset_num_timesteps=completed == 0,
        )
        completed += chunk
        validation_holdings, validation_account, _ = collect_holdings_and_account(
            model, validation_data, kwargs
        )
        metrics = metrics_from_account_values(validation_account["account_value"])
        is_best = metrics["sharpe"] > best_sharpe
        if is_best:
            best_sharpe = metrics["sharpe"]
            best_step = completed
            model.save(str(best_model_path))
        rows.append(
            {
                "model": model_name,
                "model_tag": model_tag,
                "timesteps": completed,
                "is_best": bool(is_best),
                "best_step_so_far": best_step,
                "validation_samples": int(len(validation_holdings)),
                **metrics,
            }
        )
        print(
            f"{model_name} validation checkpoint {completed}/{timesteps}: "
            f"Sharpe={metrics['sharpe']:.4f}, "
            f"return={metrics['cumulative_return']:.4f}, "
            f"best_step={best_step}"
        )

    history = pd.DataFrame(rows)
    history.to_csv(history_path, index=False)
    if not best_zip_path.exists():
        model.save(str(best_model_path))
    return rl_model_class(model_name).load(str(best_model_path)), history


def shares_from_state(state: list[float], stock_dim: int) -> np.ndarray:
    return np.asarray(state[1 + stock_dim : 1 + 2 * stock_dim], dtype=float)


def collect_holdings_and_account(
    model,
    data: pd.DataFrame,
    kwargs: dict[str, object],
    *,
    initial: bool = True,
    previous_state: list[float] | None = None,
    diagnostics: list[dict[str, object]] | None = None,
    deterministic: bool = True,
    prediction_seed: int | None = None,
):
    environment = build_env(
        data,
        kwargs,
        initial=initial,
        previous_state=[] if previous_state is None else previous_state,
    )
    vec_env, obs = environment.get_sb_env()
    holdings = []
    for step in range(len(data.index.unique()) - 1):
        state_before = np.asarray(environment.render(), dtype=float)
        prices = state_before[1 : 1 + int(kwargs["stock_dim"])]
        shares = state_before[
            1 + int(kwargs["stock_dim"]) : 1 + 2 * int(kwargs["stock_dim"])
        ]
        account_before = float(state_before[0] + prices @ shares)
        decision_date = str(environment.date_memory[-1])
        cost_before = float(environment.cost)
        if prediction_seed is not None:
            step_seed = int((prediction_seed + step * 1_000_003) % (2**31 - 1))
            random.seed(step_seed)
            np.random.seed(step_seed)
            try:
                import torch

                torch.manual_seed(step_seed)
            except ImportError:
                pass
        action, _states = model.predict(obs, deterministic=deterministic)
        obs, _rewards, dones, _info = vec_env.step(action)
        holdings.append(shares_from_state(environment.render(), kwargs["stock_dim"]))
        if diagnostics is not None:
            executed = np.asarray(environment.actions_memory[-1], dtype=float)
            gross_notional = float(np.abs(executed) @ prices)
            diagnostics.append(
                {
                    "step": step,
                    "decision_date": decision_date,
                    "date": str(environment.date_memory[-1]),
                    "gross_trade_notional": gross_notional,
                    "turnover": gross_notional / account_before if account_before else 0.0,
                    "cost_increment": float(environment.cost) - cost_before,
                }
            )
        if dones[0]:
            break
    return (
        np.asarray(holdings, dtype=float),
        environment.save_asset_memory(),
        list(environment.render()),
    )


def run_pair_ensemble_from_candidates(
    candidates_a: np.ndarray,
    candidates_b: np.ndarray,
    classifiers,
    tau: float,
    trade_data: pd.DataFrame,
    kwargs: dict[str, object],
    *,
    initial: bool = True,
    previous_state: list[float] | None = None,
):
    environment = build_env(
        trade_data,
        kwargs,
        initial=initial,
        previous_state=[] if previous_state is None else previous_state,
    )
    vec_env, obs = environment.get_sb_env()
    decisions = []
    steps = min(len(candidates_a), len(candidates_b), len(trade_data.index.unique()) - 1)

    for step in range(steps):
        candidates = np.vstack([candidates_a[step], candidates_b[step]])
        decision = select_holding(classifiers, candidates, [0, 1], tau=tau)
        state_before = np.asarray(environment.render(), dtype=float)
        prices = state_before[1 : 1 + int(kwargs["stock_dim"])]
        current_holding = shares_from_state(state_before, kwargs["stock_dim"])
        account_before = float(state_before[0] + prices @ current_holding)
        decision_date = str(environment.date_memory[-1])
        cost_before = float(environment.cost)
        action = np.clip(
            (decision.selected_holding - current_holding) / kwargs["hmax"],
            -1.0,
            1.0,
        )
        obs, _rewards, dones, _info = vec_env.step(np.asarray([action]))
        executed = np.asarray(environment.actions_memory[-1], dtype=float)
        gross_notional = float(np.abs(executed) @ prices)
        decisions.append(
            {
                "step": step,
                "decision_date": decision_date,
                "date": environment.date_memory[-1],
                "selected_agent": decision.selected_index,
                "dispersion": decision.dispersion,
                "mode": decision.mode,
                "votes": decision.votes.tolist(),
                "gross_trade_notional": gross_notional,
                "turnover": gross_notional / account_before if account_before else 0.0,
                "cost_increment": float(environment.cost) - cost_before,
            }
        )
        if dones[0]:
            break
    return environment.save_asset_memory(), pd.DataFrame(decisions), list(environment.render())


def date_reindexed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["date", "tic"], ignore_index=True)
    out.index = out["date"].factorize()[0]
    return out


def frame_for_dates(df: pd.DataFrame, dates: list[str] | np.ndarray) -> pd.DataFrame:
    date_set = set(map(str, dates))
    return date_reindexed(df[df["date"].isin(date_set)])


def append_account_curve(existing: pd.DataFrame, piece: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return piece.copy()
    next_piece = piece.copy()
    if not next_piece.empty and str(existing["date"].iloc[-1]) == str(next_piece["date"].iloc[0]):
        next_piece = next_piece.iloc[1:].copy()
    return pd.concat([existing, next_piece], ignore_index=True)


def load_trademaster_rolling_data(
    data_dir: str | Path,
    *,
    trade_split: str = "valid",
) -> tuple[pd.DataFrame, list[str], str, dict[str, object]]:
    root = Path(data_dir)
    paths = {name: root / f"{name}.csv" for name in ("train", "valid", "test")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing TradeMaster split files: {missing}")
    frames = {
        name: normalize_trademaster_frame(pd.read_csv(path, on_bad_lines="skip"))
        for name, path in paths.items()
    }
    splits = ["train"]
    if trade_split == "test":
        splits.append("valid")
    splits.append(trade_split)
    full = date_reindexed(pd.concat([frames[name] for name in splits], ignore_index=True))
    indicators = trademaster_indicator_list(frames["train"])
    metadata = {
        "data_source": "trademaster",
        "trade_split": trade_split,
        "trade_start": str(frames[trade_split]["date"].min()),
        "trade_end": str(frames[trade_split]["date"].max()),
        "date_count": int(full["date"].nunique()),
        "stock_count": int(full["tic"].nunique()),
        "indicator_count": int(len(indicators)),
    }
    return full, indicators, str(frames[trade_split]["date"].min()), metadata


def load_rolling_data(
    args: argparse.Namespace, output_dir: Path
) -> tuple[pd.DataFrame, list[str], str, dict[str, object]]:
    data_source = "synthetic" if args.synthetic_data else args.data_source
    if data_source == "trademaster":
        return load_trademaster_rolling_data(
            args.trademaster_data_dir,
            trade_split=args.trademaster_trade_split,
        )

    processed = load_or_prepare_data(
        output_dir / "cache" / "dow30_2010_2020_processed.csv",
        synthetic=data_source == "synthetic",
        synthetic_stocks=args.synthetic_stocks,
        seed=args.seed,
    )
    metadata = {
        "data_source": data_source,
        "trade_split": "paper_2020",
        "trade_start": PAPER_TRADE_START,
        "trade_end": PAPER_TRADE_END,
        "date_count": int(processed["date"].nunique()),
        "stock_count": int(processed["tic"].nunique()),
        "indicator_count": int(len(INDICATORS)),
    }
    return date_reindexed(processed), list(INDICATORS), PAPER_TRADE_START, metadata


def build_rolling_windows(
    df: pd.DataFrame,
    *,
    trade_start_date: str,
    rebalance_window: int,
    validation_window: int,
    max_windows: int | None,
) -> list[dict[str, object]]:
    dates = np.asarray(sorted(map(str, df["date"].unique())))
    trade_start_index = int(np.searchsorted(dates, trade_start_date))
    windows: list[dict[str, object]] = []
    for trade_start in range(trade_start_index, len(dates) - 1, rebalance_window):
        trade_end = min(trade_start + rebalance_window, len(dates))
        if len(dates) - trade_end == 1:
            trade_end = len(dates)
        if not windows:
            calibration_start = trade_start - validation_window
            calibration_source = "train_tail"
        else:
            previous_trade_dates = windows[-1]["trade_dates"]
            calibration_start = int(np.where(dates == previous_trade_dates[0])[0][0])
            calibration_source = "previous_trade"
        calibration_end = trade_start
        if calibration_start <= 0 or trade_end - trade_start < 2:
            continue
        calibration_dates = dates[calibration_start:calibration_end]
        if calibration_source == "previous_trade":
            if list(calibration_dates) != list(windows[-1]["trade_dates"]):
                raise ValueError("calibration dates must match the previous traded window")
        if not (calibration_end <= trade_start and calibration_start < calibration_end):
            raise ValueError("calibration window must end before the current trade window")
        windows.append(
            {
                "window": len(windows) + 1,
                "train_dates": dates[:calibration_start],
                "validation_dates": calibration_dates,
                "calibration_dates": calibration_dates,
                "calibration_source": calibration_source,
                "trade_dates": dates[trade_start:trade_end],
                "train_start": dates[0],
                "train_end": dates[calibration_start - 1],
                "validation_start": dates[calibration_start],
                "validation_end": dates[calibration_end - 1],
                "calibration_start": dates[calibration_start],
                "calibration_end": dates[calibration_end - 1],
                "trade_start": dates[trade_start],
                "trade_end": dates[trade_end - 1],
            }
        )
        if max_windows is not None and len(windows) >= max_windows:
            break
    if not windows:
        raise ValueError("no rolling windows could be built from the available data")
    return windows


def run_pair_ensemble_from_models(
    left_model,
    right_model,
    classifiers,
    tau: float,
    trade_data: pd.DataFrame,
    kwargs: dict[str, object],
    *,
    initial: bool = True,
    previous_state: list[float] | None = None,
):
    start_state = [] if previous_state is None else previous_state
    ensemble_env = build_env(
        trade_data,
        kwargs,
        initial=initial,
        previous_state=start_state,
    )
    ensemble_vec, _ensemble_obs = ensemble_env.get_sb_env()

    candidate_runners = []
    for model in (left_model, right_model):
        candidate_env = build_env(
            trade_data,
            kwargs,
            initial=initial,
            previous_state=start_state,
        )
        candidate_vec, candidate_obs = candidate_env.get_sb_env()
        candidate_runners.append(
            {
                "model": model,
                "env": candidate_env,
                "vec": candidate_vec,
                "obs": candidate_obs,
            }
        )

    decisions = []
    for step in range(len(trade_data.index.unique()) - 1):
        candidates = []
        for runner in candidate_runners:
            action, _states = runner["model"].predict(runner["obs"], deterministic=True)
            runner["obs"], _rewards, dones, _info = runner["vec"].step(action)
            candidates.append(shares_from_state(runner["env"].render(), kwargs["stock_dim"]))
            if dones[0]:
                break
        if len(candidates) != 2:
            break
        decision = select_holding(classifiers, np.vstack(candidates), [0, 1], tau=tau)
        state_before = np.asarray(ensemble_env.render(), dtype=float)
        prices = state_before[1 : 1 + int(kwargs["stock_dim"])]
        current_holding = shares_from_state(state_before, kwargs["stock_dim"])
        account_before = float(state_before[0] + prices @ current_holding)
        decision_date = str(ensemble_env.date_memory[-1])
        cost_before = float(ensemble_env.cost)
        action = np.clip(
            (decision.selected_holding - current_holding) / kwargs["hmax"],
            -1.0,
            1.0,
        )
        _obs, _rewards, dones, _info = ensemble_vec.step(np.asarray([action]))
        executed = np.asarray(ensemble_env.actions_memory[-1], dtype=float)
        gross_notional = float(np.abs(executed) @ prices)
        decisions.append(
            {
                "step": step,
                "decision_date": decision_date,
                "date": ensemble_env.date_memory[-1],
                "selected_agent": decision.selected_index,
                "dispersion": decision.dispersion,
                "mode": decision.mode,
                "votes": decision.votes.tolist(),
                "gross_trade_notional": gross_notional,
                "turnover": gross_notional / account_before if account_before else 0.0,
                "cost_increment": float(ensemble_env.cost) - cost_before,
            }
        )
        if dones[0]:
            break
    return ensemble_env.save_asset_memory(), pd.DataFrame(decisions), list(ensemble_env.render())


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    view = df.loc[:, columns].head(max_rows).copy()
    for column in view.select_dtypes(include=[float]).columns:
        view[column] = view[column].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(view.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in view.to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def write_additional_rolling_checks(
    output_dir: Path,
    *,
    metrics: pd.DataFrame,
    best: pd.DataFrame,
    base_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    reference: pd.DataFrame | None,
) -> None:
    if reference is None or reference.empty:
        return

    paper = reference.rename(
        columns={
            "tau": "tau_paper",
            "cumulative_return": "cumulative_return_paper",
            "sharpe": "sharpe_paper",
            "calmar": "calmar_paper",
            "max_drawdown": "max_drawdown_paper",
        }
    ).copy()
    lookup = metrics.copy()
    lookup["tau_key"] = lookup["tau"].round(2)
    paper["tau_key"] = paper["tau_paper"].round(2)
    same_tau = paper.merge(
        lookup,
        on=["pair", "classifier_group", "tau_key"],
        how="left",
        suffixes=("", "_repro_at_paper_tau"),
    )
    same_tau = same_tau.rename(
        columns={
            "tau": "tau_repro_at_paper_tau",
            "cumulative_return": "cumulative_return_repro_at_paper_tau",
            "sharpe": "sharpe_repro_at_paper_tau",
            "calmar": "calmar_repro_at_paper_tau",
            "max_drawdown": "max_drawdown_repro_at_paper_tau",
        }
    )
    for metric in METRIC_COLUMNS:
        same_tau[f"{metric}_same_tau_delta"] = (
            same_tau[f"{metric}_repro_at_paper_tau"]
            - same_tau[f"{metric}_paper"]
        )
    same_tau.to_csv(output_dir / "rolling_same_paper_tau_comparison.csv", index=False)

    best_base_sharpe = float(base_metrics["sharpe"].max())
    best_row = best.sort_values(
        ["sharpe", "tau"], ascending=[False, True]
    ).iloc[0]
    summary = pd.DataFrame(
        [
            {
                "criterion": "best_tau_positive_return_rate",
                "value": float((best["cumulative_return"] > 0).mean()),
            },
            {
                "criterion": "same_paper_tau_positive_return_rate",
                "value": float(
                    (same_tau["cumulative_return_repro_at_paper_tau"] > 0).mean()
                ),
            },
            {
                "criterion": "best_tau_beats_best_base_sharpe_rate",
                "value": float((best["sharpe"] > best_base_sharpe).mean()),
            },
            {
                "criterion": "same_paper_tau_beats_best_base_sharpe_rate",
                "value": float(
                    (same_tau["sharpe_repro_at_paper_tau"] > best_base_sharpe).mean()
                ),
            },
            {
                "criterion": "mean_best_tau_sharpe_delta_vs_paper",
                "value": float(comparison["sharpe_delta"].mean()),
            },
            {
                "criterion": "mean_same_tau_sharpe_delta_vs_paper",
                "value": float(same_tau["sharpe_same_tau_delta"].mean()),
            },
            {
                "criterion": "best_repro_pair_group",
                "value": f"{best_row['pair']}/group{int(best_row['classifier_group'])}",
            },
            {
                "criterion": "best_repro_sharpe",
                "value": float(best_row["sharpe"]),
            },
        ]
    )
    summary.to_csv(output_dir / "rolling_credibility_summary.csv", index=False)

    report_path = output_dir / "ROLLING_FULL_VALID_REPRO_REPORT.md"
    if not report_path.exists():
        return
    report = report_path.read_text(encoding="utf-8")
    if "## Same Paper Tau Check" in report:
        return
    extra = [
        "",
        "## Same Paper Tau Check",
        "",
        "This table evaluates the current rolling run at the exact tau selected in the paper CSVs, rather than at this run's best tau.",
        "",
        markdown_table(
            same_tau,
            [
                "pair",
                "classifier_group",
                "tau_paper",
                "cumulative_return_repro_at_paper_tau",
                "cumulative_return_paper",
                "sharpe_repro_at_paper_tau",
                "sharpe_paper",
                "max_drawdown_repro_at_paper_tau",
                "max_drawdown_paper",
            ],
            max_rows=20,
        ),
        "",
        "Full table: `rolling_same_paper_tau_comparison.csv`.",
        "",
        "## Credibility Summary",
        "",
        markdown_table(summary, ["criterion", "value"], max_rows=20),
        "",
        "Additional file: `rolling_credibility_summary.csv`.",
        "",
    ]
    report_path.write_text(report.rstrip() + "\n" + "\n".join(extra), encoding="utf-8")


def write_rolling_report(
    output_dir: Path,
    *,
    metadata: dict[str, object],
    windows: list[dict[str, object]],
    base_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    group1: pd.DataFrame,
    group_summary: pd.DataFrame,
) -> None:
    protocol_rows = pd.DataFrame(
        [
            {
                "item": "train period",
                "paper": "2010-01-01 to 2019-10-01",
                "run": f"{windows[0]['train_start']} to {windows[-1]['train_end']}",
            },
            {
                "item": "calibration/rebalance",
                "paper": "60-day text; 63-day rebalancing/validation line",
                "run": (
                    f"{metadata['validation_window']} first train-tail calibration; "
                    f"then previous traded window / {metadata['rebalance_window']} rebalance"
                ),
            },
            {
                "item": "trade span",
                "paper": "2020 full year",
                "run": f"{windows[0]['trade_start']} to {windows[-1]['trade_end']}",
            },
            {
                "item": "classifier protocol",
                "paper": "fixed classifier definitions within each group",
                "run": "fixed definitions; rolling refit only; no grid search",
            },
            {
                "item": "tau protocol",
                "paper": "one global tau per complete backtest path",
                "run": "one global tau fixed across every rolling window",
            },
            {
                "item": "RL checkpoint selection",
                "paper": "validate agents with SR before classifier boundary training",
                "run": (
                    f"best calibration Sharpe checkpoint every "
                    f"{metadata['rl_eval_interval']} timesteps"
                ),
            },
            {
                "item": "no-leakage calibration",
                "paper": "Oct 2019 validation before Jan 2020 trade; sliding readjustment thereafter",
                "run": "window 1 uses train tail; later windows use the previous traded segment",
            },
            {
                "item": "backtest iterations",
                "paper": "30 averages in sensitivity figure",
                "run": f"1 seed ({metadata['seed']})",
            },
            {
                "item": "universe",
                "paper": "DJ30",
                "run": f"{metadata['stock_count']} TradeMaster tickers",
            },
        ]
    )
    protocol_rows.to_csv(output_dir / "rolling_protocol_checklist.csv", index=False)

    report = [
        "# Classifier ensemble rolling reproduction report",
        "",
        "## Run scope",
        "",
        f"- Data source: {metadata['data_source']} / {metadata['trade_split']}",
        f"- Windows completed: {len(windows)}",
        f"- Timesteps per model/window: {metadata['timesteps']}",
        f"- Global tau grid: {metadata['tau_start']} to {metadata['tau_stop']} step {metadata['tau_step']}",
        "- Each candidate tau is fixed for the complete annual path; tau is not reselected inside a rolling window.",
        "- Classifier membership and hyperparameters are fixed by group; only the fitted boundary is refreshed on the rolling calibration block.",
        f"- Output directory: `{output_dir}`",
        "",
        "This run follows the paper's rolling calibration/rebalancing structure. The first calibration block is the tail of the training set, and later classifier boundaries are readjusted on the immediately previous traded block. In the packaged configuration it covers all rolling windows in the selected TradeMaster validation split; it should still be read as a single-seed reproduction rather than the paper's 30-iteration average table.",
        "",
        "## Protocol checklist",
        "",
        markdown_table(protocol_rows, ["item", "paper", "run"], max_rows=20),
        "",
        "## Base models",
        "",
        markdown_table(
            base_metrics,
            ["model", "cumulative_return", "sharpe", "calmar", "max_drawdown"],
            max_rows=10,
        ),
        "",
        "## Main table comparison",
        "",
        markdown_table(
            comparison,
            [
                "pair",
                "classifier_group",
                "tau_repro",
                "tau_paper",
                "cumulative_return_repro",
                "cumulative_return_paper",
                "sharpe_repro",
                "sharpe_paper",
                "max_drawdown_repro",
                "max_drawdown_paper",
            ],
            max_rows=20,
        ),
        "",
        "Full table: `rolling_comparison_to_paper.csv`.",
        "",
        "## Classifier group 1",
        "",
        markdown_table(
            group1,
            [
                "pair",
                "tau",
                "cumulative_return",
                "sharpe",
                "calmar",
                "max_drawdown",
            ],
            max_rows=10,
        ),
        "",
        "## Classifier group comparison",
        "",
        markdown_table(
            group_summary,
            [
                "pair",
                "classifier_group",
                "tau",
                "cumulative_return",
                "sharpe",
                "max_drawdown",
            ],
            max_rows=20,
        ),
        "",
        "## Credibility notes",
        "",
        "- The rolling protocol gap from the earlier one-shot run is fixed here: models and classifier boundaries are recomputed by rolling window.",
        "- RL holdings for classifier training come from the calibration-Sharpe-selected checkpoint within each rolling window; calibration is train-tail for the first trade block and the immediately previous traded block thereafter.",
        "- Remaining non-paper differences are material: TradeMaster starts in 2012 rather than 2010, has 29 tickers rather than full DJ30, and this run uses one seed rather than the paper's 30-iteration averages.",
        "- The report therefore checks whether the implementation direction is credible under the full rolling validation span available in the packaged data.",
        "",
    ]
    (output_dir / "ROLLING_FULL_VALID_REPRO_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )


def run_rolling_mode(args: argparse.Namespace) -> None:
    output_dir = ensure_output_dir(args.output_dir)
    full_data, tech_indicators, trade_start_date, metadata = load_rolling_data(args, output_dir)
    kwargs = env_kwargs(full_data, tech_indicators)
    windows = build_rolling_windows(
        full_data,
        trade_start_date=trade_start_date,
        rebalance_window=args.rebalance_window,
        validation_window=args.rolling_validation_window,
        max_windows=args.rolling_max_windows,
    )
    tau_values = tau_grid(args.tau_start, args.tau_stop, args.tau_step)
    group_ids = [int(item) for item in args.groups.split(",") if item.strip()]
    pair_names = [item.strip() for item in args.pairs.split(",") if item.strip()]
    pair_map = {
        "a2c_sac": ("a2c", "sac"),
        "ppo_sac": ("ppo", "sac"),
        "a2c_ppo": ("a2c", "ppo"),
    }

    required_models = sorted(
        {model_name for pair_name in pair_names for model_name in pair_map[pair_name]}
    )
    base_curves = {name: pd.DataFrame() for name in required_models}
    base_last_states: dict[str, list[float] | None] = {name: None for name in base_curves}
    ensemble_curves: dict[tuple[str, int, float], pd.DataFrame] = {}
    ensemble_last_states: dict[tuple[str, int, float], list[float] | None] = {}
    decision_rows: list[pd.DataFrame] = []
    rl_selection_rows: list[pd.DataFrame] = []
    window_rows = []

    for window_info in windows:
        window_number = int(window_info["window"])
        train = frame_for_dates(full_data, window_info["train_dates"])
        calibration = frame_for_dates(full_data, window_info["calibration_dates"])
        trade = frame_for_dates(full_data, window_info["trade_dates"])
        print(
            "ROLLING WINDOW",
            window_number,
            "train",
            window_info["train_start"],
            window_info["train_end"],
            "calibration",
            window_info["calibration_start"],
            window_info["calibration_end"],
            window_info["calibration_source"],
            "trade",
            window_info["trade_start"],
            window_info["trade_end"],
        )

        models = {}
        validation_holdings = {}
        for name in required_models:
            model_tag = f"rolling_w{window_number}_{args.timesteps}_seed{args.seed}"
            model, selection_history = train_or_load_validation_selected_model(
                name,
                train,
                calibration,
                kwargs,
                output_dir,
                args.timesteps,
                args.seed,
                args.force_train,
                model_tag=model_tag,
                eval_interval=args.rl_eval_interval,
            )
            selection_history = selection_history.copy()
            selection_history["window"] = window_number
            selection_history["train_start"] = window_info["train_start"]
            selection_history["train_end"] = window_info["train_end"]
            selection_history["validation_start"] = window_info["validation_start"]
            selection_history["validation_end"] = window_info["validation_end"]
            selection_history["calibration_source"] = window_info["calibration_source"]
            rl_selection_rows.append(selection_history)
            models[name] = model
            validation_holdings[name], _, _ = collect_holdings_and_account(
                model, calibration, kwargs
            )
            initial_base = base_last_states[name] is None
            _, base_account, base_last_state = collect_holdings_and_account(
                model,
                trade,
                kwargs,
                initial=initial_base,
                previous_state=base_last_states[name],
            )
            base_last_states[name] = base_last_state
            base_curves[name] = append_account_curve(base_curves[name], base_account)

        for pair_name in pair_names:
            left, right = pair_map[pair_name]
            for group in group_ids:
                classifiers = train_classifier_group(
                    [validation_holdings[left], validation_holdings[right]],
                    group,
                    random_state=args.seed + window_number,
                    grid_search=False,
                )
                for tau in tau_values:
                    key = (pair_name, group, float(tau))
                    initial_ensemble = key not in ensemble_last_states
                    account, decisions, last_state = run_pair_ensemble_from_models(
                        models[left],
                        models[right],
                        classifiers,
                        float(tau),
                        trade,
                        kwargs,
                        initial=initial_ensemble,
                        previous_state=ensemble_last_states.get(key),
                    )
                    ensemble_last_states[key] = last_state
                    ensemble_curves[key] = append_account_curve(
                        ensemble_curves.get(key, pd.DataFrame()),
                        account,
                    )
                    if args.save_decisions:
                        decisions = decisions.copy()
                        decisions["window"] = window_number
                        decisions["pair"] = pair_name
                        decisions["classifier_group"] = group
                        decisions["tau"] = float(tau)
                        decision_rows.append(decisions)

        window_rows.append(
            {
                "window": window_number,
                "train_start": window_info["train_start"],
                "train_end": window_info["train_end"],
                "validation_start": window_info["validation_start"],
                "validation_end": window_info["validation_end"],
                "calibration_start": window_info["calibration_start"],
                "calibration_end": window_info["calibration_end"],
                "calibration_source": window_info["calibration_source"],
                "trade_start": window_info["trade_start"],
                "trade_end": window_info["trade_end"],
                "train_dates": int(train["date"].nunique()),
                "calibration_dates": int(calibration["date"].nunique()),
                "trade_dates": int(trade["date"].nunique()),
            }
        )

    window_summary = pd.DataFrame(window_rows)
    window_summary.to_csv(output_dir / "rolling_windows.csv", index=False)
    rl_selection_history = (
        pd.concat(rl_selection_rows, ignore_index=True)
        if rl_selection_rows
        else pd.DataFrame()
    )
    rl_selection_history.to_csv(output_dir / "rolling_rl_selection_history.csv", index=False)

    base_rows = []
    for name, curve in base_curves.items():
        curve.to_csv(output_dir / f"rolling_account_{name}.csv", index=False)
        base_rows.append({"model": name, **metrics_from_account_values(curve["account_value"])})
    base_metrics = pd.DataFrame(base_rows)
    base_metrics.to_csv(output_dir / "rolling_base_model_metrics.csv", index=False)

    metric_rows = []
    for (pair_name, group, tau), curve in ensemble_curves.items():
        metric_rows.append(
            {
                "pair": pair_name,
                "classifier_group": group,
                "tau": tau,
                **metrics_from_account_values(curve["account_value"]),
            }
        )
    metrics = pd.DataFrame(metric_rows).sort_values(["pair", "classifier_group", "tau"])
    metrics.to_csv(output_dir / "rolling_metrics_all_tau.csv", index=False)
    best = (
        metrics.sort_values(["sharpe", "tau"], ascending=[False, True])
        .groupby(["pair", "classifier_group"], as_index=False)
        .head(1)
        .sort_values(["pair", "classifier_group"])
    )
    best.to_csv(output_dir / "rolling_best_by_sharpe.csv", index=False)

    for _, row in best.iterrows():
        key = (row["pair"], int(row["classifier_group"]), float(row["tau"]))
        ensemble_curves[key].to_csv(
            output_dir
            / f"rolling_account_best_{row['pair']}_group{int(row['classifier_group'])}_tau{float(row['tau']):.2f}.csv",
            index=False,
        )

    if decision_rows:
        pd.concat(decision_rows, ignore_index=True).to_csv(
            output_dir / "rolling_decisions.csv", index=False
        )

    paper_dir = Path(args.paper_results_dir)
    reference = None
    if paper_dir.exists():
        reference = load_paper_reference_summary(paper_dir)
        comparison = compare_with_reference(best, reference)
        comparison.to_csv(output_dir / "rolling_comparison_to_paper.csv", index=False)
    else:
        comparison = best.copy()

    group1 = best[best["classifier_group"] == 1].sort_values("pair")
    group1.to_csv(output_dir / "rolling_classifier_group1_best.csv", index=False)
    group_summary = best.sort_values(["pair", "classifier_group"])
    group_summary.to_csv(output_dir / "rolling_classifier_group_summary.csv", index=False)
    sensitivity = metrics[metrics["classifier_group"] == 1].sort_values(["pair", "tau"])
    sensitivity.to_csv(output_dir / "rolling_tau_sensitivity_group1.csv", index=False)

    metadata.update(
        {
            "seed": args.seed,
            "timesteps": args.timesteps,
            "rebalance_window": args.rebalance_window,
            "validation_window": args.rolling_validation_window,
            "rl_eval_interval": args.rl_eval_interval,
            "tau_start": args.tau_start,
            "tau_stop": args.tau_stop,
            "tau_step": args.tau_step,
        }
    )
    write_rolling_report(
        output_dir,
        metadata=metadata,
        windows=windows,
        base_metrics=base_metrics,
        comparison=comparison,
        group1=group1,
        group_summary=group_summary,
    )
    write_additional_rolling_checks(
        output_dir,
        metrics=metrics,
        best=best,
        base_metrics=base_metrics,
        comparison=comparison,
        reference=reference,
    )
    print(comparison.to_string(index=False))
    print(f"Saved rolling report to {output_dir / 'ROLLING_FULL_VALID_REPRO_REPORT.md'}")


def run_full_mode(args: argparse.Namespace) -> None:
    output_dir = ensure_output_dir(args.output_dir)
    train, validation, trade, tech_indicators = load_experiment_splits(args, output_dir)
    kwargs = env_kwargs(pd.concat([train, validation, trade]), tech_indicators)

    models = {
        name: train_or_load_model(
            name,
            train,
            kwargs,
            output_dir,
            args.timesteps,
            args.seed,
            args.force_train,
        )
        for name in ("a2c", "ppo", "sac")
    }

    validation_holdings = {}
    trade_holdings = {}
    base_rows = []
    for name, model in models.items():
        validation_holdings[name], _, _ = collect_holdings_and_account(
            model, validation, kwargs
        )
        trade_holdings[name], account, _ = collect_holdings_and_account(
            model, trade, kwargs
        )
        base_rows.append({"model": name, **metrics_from_account_values(account["account_value"])})
    pd.DataFrame(base_rows).to_csv(output_dir / "base_model_metrics.csv", index=False)

    group_ids = [int(item) for item in args.groups.split(",") if item.strip()]
    pair_names = [item.strip() for item in args.pairs.split(",") if item.strip()]
    pair_map = {
        "a2c_sac": ("a2c", "sac"),
        "ppo_sac": ("ppo", "sac"),
        "a2c_ppo": ("a2c", "ppo"),
    }
    metric_rows = []
    for pair_name in pair_names:
        left, right = pair_map[pair_name]
        for group in group_ids:
            classifiers = train_classifier_group(
                [validation_holdings[left], validation_holdings[right]],
                group,
                random_state=args.seed,
                grid_search=False,
            )
            for tau in tau_grid():
                account, decisions, _ = run_pair_ensemble_from_candidates(
                    trade_holdings[left],
                    trade_holdings[right],
                    classifiers,
                    float(tau),
                    trade,
                    kwargs,
                )
                row = {
                    "pair": pair_name,
                    "classifier_group": group,
                    "tau": float(tau),
                    **metrics_from_account_values(account["account_value"]),
                }
                metric_rows.append(row)
                decisions.to_csv(
                    output_dir / f"decisions_{pair_name}_group{group}_tau{tau:.2f}.csv",
                    index=False,
                )

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "reproduction_metrics_all_tau.csv", index=False)
    best = (
        metrics.sort_values(["sharpe", "tau"], ascending=[False, True])
        .groupby(["pair", "classifier_group"], as_index=False)
        .head(1)
        .sort_values(["pair", "classifier_group"])
    )
    best.to_csv(output_dir / "reproduction_best_by_sharpe.csv", index=False)

    paper_dir = Path(args.paper_results_dir)
    if paper_dir.exists():
        reference = load_paper_reference_summary(paper_dir)
        comparison = compare_with_reference(best, reference)
        comparison.to_csv(output_dir / "comparison_to_paper.csv", index=False)
        print(comparison.to_string(index=False))
    else:
        print(best.to_string(index=False))


def main() -> None:
    args = parse_args()
    if args.mode == "smoke":
        run_smoke_mode(args)
    elif args.mode == "rolling":
        run_rolling_mode(args)
    else:
        run_full_mode(args)


if __name__ == "__main__":
    main()
