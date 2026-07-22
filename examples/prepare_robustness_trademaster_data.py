from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_HORIZONS = (5, 10, 15, 20, 25, 30)


def add_trademaster_like_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
    data["tic"] = data["tic"].astype(str)
    data = data.sort_values(["tic", "date"], ignore_index=True)
    if "adjcp" not in data:
        data["adjcp"] = data["close"]

    close = data["close"].replace(0, np.nan)
    data["zopen"] = data["open"] / close - 1.0
    data["zhigh"] = data["high"] / close - 1.0
    data["zlow"] = data["low"] / close - 1.0
    data["zadjcp"] = data["adjcp"] / close - 1.0
    data["zclose"] = data.groupby("tic")["close"].pct_change()
    for horizon in FEATURE_HORIZONS:
        data[f"zd_{horizon}"] = data.groupby("tic")["close"].pct_change(horizon)

    data = data.replace([np.inf, -np.inf], 0).fillna(0)
    data = data.sort_values(["date", "tic"], ignore_index=True)
    return data[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adjcp",
            "tic",
            "zopen",
            "zhigh",
            "zlow",
            "zadjcp",
            "zclose",
            *[f"zd_{horizon}" for horizon in FEATURE_HORIZONS],
        ]
    ]


def keep_complete_panel(df: pd.DataFrame) -> pd.DataFrame:
    stock_count = df["tic"].nunique()
    complete_dates = df.groupby("date")["tic"].nunique()
    complete_dates = complete_dates[complete_dates == stock_count].index
    return df[df["date"].isin(complete_dates)].copy()


def split_panel(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    train_ratio: float,
    valid_ratio: float,
) -> dict[str, object]:
    dates = np.asarray(sorted(df["date"].unique()))
    if len(dates) < 252:
        raise ValueError(f"not enough complete dates for rolling experiments: {len(dates)}")
    train_end = int(len(dates) * train_ratio)
    valid_end = int(len(dates) * (train_ratio + valid_ratio))
    train_end = max(train_end, 126)
    valid_end = max(valid_end, train_end + 63)
    valid_end = min(valid_end, len(dates) - 63)
    splits = {
        "train": dates[:train_end],
        "valid": dates[train_end:valid_end],
        "test": dates[valid_end:],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split_dates in splits.items():
        frame = df[df["date"].isin(split_dates)].copy()
        frame.to_csv(output_dir / f"{name}.csv", index=False)

    return {
        "output_dir": str(output_dir),
        "date_start": str(dates[0]),
        "date_end": str(dates[-1]),
        "date_count": int(len(dates)),
        "stock_count": int(df["tic"].nunique()),
        "tickers": sorted(df["tic"].unique().tolist()),
        "splits": {
            name: {
                "date_start": str(split_dates[0]),
                "date_end": str(split_dates[-1]),
                "date_count": int(len(split_dates)),
            }
            for name, split_dates in splits.items()
        },
    }


def prepare_sse50(raw_path: Path, output_dir: Path) -> dict[str, object]:
    raw = pd.read_csv(raw_path)
    raw = raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "ticker": "tic",
        }
    )
    raw["datetime"] = pd.to_datetime(raw["date"])
    raw["date"] = raw["datetime"].dt.strftime("%Y-%m-%d")
    raw["tic"] = raw["tic"].astype(str)
    raw = raw.sort_values(["tic", "datetime"], ignore_index=True)
    daily = (
        raw.groupby(["date", "tic"], as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .sort_values(["date", "tic"], ignore_index=True)
    )
    daily = keep_complete_panel(daily)
    processed = add_trademaster_like_features(daily)
    metadata = split_panel(processed, output_dir, train_ratio=0.60, valid_ratio=0.20)
    metadata.update(
        {
            "dataset": "sse50_daily",
            "source_path": str(raw_path),
            "preprocessing": "Aggregated five intraday observations per ticker/day to daily OHLCV; retained complete dates.",
        }
    )
    return metadata


def prepare_hstech10(raw_path: Path, output_dir: Path) -> dict[str, object]:
    raw = pd.read_csv(raw_path, on_bad_lines="skip")
    raw = raw.rename(columns={"ticker": "tic"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
    raw["tic"] = raw["tic"].astype(str)
    raw = raw[(raw["date"] >= "2016-06-01") & (raw["date"] <= "2020-08-31")].copy()

    coverage = raw.groupby("tic")["date"].nunique().sort_values(ascending=False)
    selected = coverage.head(10).index.tolist()
    panel = raw[raw["tic"].isin(selected)].copy()
    complete_dates = panel.groupby("date")["tic"].nunique()
    complete_dates = complete_dates[complete_dates == len(selected)].index
    panel = panel[panel["date"].isin(complete_dates)].copy()
    panel = panel[["date", "open", "high", "low", "close", "volume", "tic"]]
    panel = panel.sort_values(["date", "tic"], ignore_index=True)
    processed = add_trademaster_like_features(panel)
    metadata = split_panel(processed, output_dir, train_ratio=0.60, valid_ratio=0.20)
    metadata.update(
        {
            "dataset": "hstech10",
            "source_path": str(raw_path),
            "preprocessing": "Selected the 10 HSTech tickers with longest coverage over 2016-06-01 to 2020-08-31; retained complete dates.",
        }
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SSE50 and HSTech10 robustness data for the rolling classifier-ensemble experiment."
    )
    parser.add_argument(
        "--raw-root",
        default="../github/external_data",
        help="Directory containing trademaster_sse50/data.csv and trademaster_hstech30/hstech30.csv.",
    )
    parser.add_argument(
        "--output-root",
        default="external_data",
        help="Output root for processed TradeMaster-style split directories.",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    jobs = [
        (
            "sse50_daily",
            prepare_sse50,
            raw_root / "trademaster_sse50" / "data.csv",
            output_root / "trademaster_sse50_daily",
        ),
        (
            "hstech10",
            prepare_hstech10,
            raw_root / "trademaster_hstech30" / "hstech30.csv",
            output_root / "trademaster_hstech10",
        ),
    ]
    for name, func, raw_path, output_dir in jobs:
        if not raw_path.exists():
            raise FileNotFoundError(f"missing raw data for {name}: {raw_path}")
        metadata = func(raw_path, output_dir)
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
