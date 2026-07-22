from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit portfolio panels at date-ticker grain.")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="MARKET=directory containing train.csv, valid.csv, and test.csv",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def parse_dataset(value: str) -> tuple[str, Path]:
    market, separator, directory = value.partition("=")
    if not separator or not market.strip() or not directory.strip():
        raise ValueError(f"dataset must be MARKET=DIRECTORY, got {value!r}")
    return market.strip(), Path(directory.strip())


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for dataset in args.dataset:
        market, directory = parse_dataset(dataset)
        date_sets: dict[str, set[str]] = {}
        for split in ("train", "valid", "test"):
            path = directory / f"{split}.csv"
            frame = pd.read_csv(path)
            required = {"date", "tic", "close"}
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            dates = frame["date"].astype(str)
            daily_assets = frame.groupby("date")["tic"].nunique()
            date_sets[split] = set(dates.unique())
            rows.append(
                {
                    "market": market,
                    "split": split,
                    "rows": len(frame),
                    "dates": dates.nunique(),
                    "tickers": frame["tic"].nunique(),
                    "date_start": dates.min(),
                    "date_end": dates.max(),
                    "min_assets_per_day": int(daily_assets.min()),
                    "max_assets_per_day": int(daily_assets.max()),
                    "duplicate_date_ticker_keys": int(
                        frame.duplicated(["date", "tic"]).sum()
                    ),
                    "null_cells": int(frame.isna().sum().sum()),
                    "non_positive_close": int((frame["close"] <= 0).sum()),
                }
            )
        overlaps = {
            "train_valid": len(date_sets["train"] & date_sets["valid"]),
            "train_test": len(date_sets["train"] & date_sets["test"]),
            "valid_test": len(date_sets["valid"] & date_sets["test"]),
        }
        for row in rows[-3:]:
            row.update({f"overlap_dates_{key}": value for key, value in overlaps.items()})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(rows)
    audit.to_csv(output, index=False)
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
