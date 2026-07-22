from __future__ import annotations

__all__ = ["test", "trade", "train"]


def __getattr__(name):
    if name == "test":
        from finrl.test import test

        return test
    if name == "trade":
        from finrl.trade import trade

        return trade
    if name == "train":
        from finrl.train import train

        return train
    raise AttributeError(f"module 'finrl' has no attribute {name!r}")
