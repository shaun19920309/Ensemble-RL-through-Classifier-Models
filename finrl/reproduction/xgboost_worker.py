from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit per-asset XGBoost regressors in an isolated native runtime."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config)
    with np.load(args.input) as arrays:
        train_x = arrays["train_x"]
        train_y = arrays["train_y"]
        predict_x = arrays["predict_x"]

    predictions = np.empty((len(predict_x), train_y.shape[1]), dtype=float)
    for asset in range(train_y.shape[1]):
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=int(config["estimators"]),
            max_depth=int(config["max_depth"]),
            learning_rate=float(config["learning_rate"]),
            min_child_weight=5,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_alpha=0.0,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=1,
            random_state=int(config["seed"]) + asset,
        )
        model.fit(train_x, train_y[:, asset], verbose=False)
        predictions[:, asset] = model.predict(predict_x)
    np.save(Path(args.output), predictions)


if __name__ == "__main__":
    main()
