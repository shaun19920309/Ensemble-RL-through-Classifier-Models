from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True)
class CrossoverTauDecision:
    status: str
    selected_tau: float
    threshold_quantile: float
    raw_threshold: float
    history_blocks: int
    history_days: int
    informative_days: int
    cdf_days: int
    low_mode_advantage: float
    high_mode_advantage: float
    policy_advantage_mean: float
    policy_advantage_lcb: float
    fit_start_date: str
    fit_end_date: str

    @property
    def selected(self) -> bool:
        return self.status == "selected"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _ModeAdvantageBlock:
    block_id: str
    start_date: str
    end_date: str
    dispersion: np.ndarray
    aggressive_return: np.ndarray
    conservative_return: np.ndarray
    fallback_return: np.ndarray
    branch_diverged: np.ndarray


class CausalCrossoverTau:
    """Blockwise ex-ante tau from a causal mode-advantage crossover.

    Completed same-state counterfactual blocks are the only feedback. At the
    start of a new block, holding dispersion is converted to an empirical
    percentile, a decreasing isotonic curve is fitted to aggressive-minus-
    conservative one-step returns, and its zero crossing is mapped back to the
    original tau grid. The selected tau is immutable until the block ends.
    """

    def __init__(
        self,
        tau_values: np.ndarray,
        *,
        max_fit_blocks: int = 8,
        cdf_lookback_days: int = 252,
        min_history_blocks: int = 4,
        min_informative_days: int = 63,
        min_side_days: int = 20,
        confidence_level: float = 0.95,
        require_positive_lcb: bool = True,
    ) -> None:
        tau = np.asarray(tau_values, dtype=float)
        if (
            tau.ndim != 1
            or len(tau) == 0
            or not np.isfinite(tau).all()
            or np.any(np.diff(tau) <= 0.0)
        ):
            raise ValueError("tau_values must be finite and strictly increasing")
        if max_fit_blocks < 1:
            raise ValueError("max_fit_blocks must be positive")
        if cdf_lookback_days < 2:
            raise ValueError("cdf_lookback_days must be at least two")
        if min_history_blocks < 1 or min_history_blocks > max_fit_blocks:
            raise ValueError(
                "min_history_blocks must be positive and no larger than max_fit_blocks"
            )
        if min_informative_days < 2 or min_side_days < 1:
            raise ValueError("minimum sample counts must be positive")
        if not 0.5 < confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1")

        self.tau_values = tau.copy()
        self.max_fit_blocks = int(max_fit_blocks)
        self.cdf_lookback_days = int(cdf_lookback_days)
        self.min_history_blocks = int(min_history_blocks)
        self.min_informative_days = int(min_informative_days)
        self.min_side_days = int(min_side_days)
        self.confidence_level = float(confidence_level)
        self.require_positive_lcb = bool(require_positive_lcb)
        self._blocks: list[_ModeAdvantageBlock] = []

    @property
    def completed_blocks(self) -> int:
        return len(self._blocks)

    @property
    def completed_days(self) -> int:
        return int(sum(len(block.dispersion) for block in self._blocks))

    @property
    def last_completed_date(self) -> str:
        return self._blocks[-1].end_date if self._blocks else ""

    def snapshot(self) -> dict[str, object]:
        return {
            "completed_blocks": self.completed_blocks,
            "completed_days": self.completed_days,
            "last_completed_date": self.last_completed_date,
        }

    def add_completed_block(
        self,
        *,
        block_id: str,
        start_date: str,
        end_date: str,
        dispersion: np.ndarray,
        aggressive_return: np.ndarray,
        conservative_return: np.ndarray,
        fallback_return: np.ndarray,
        branch_diverged: np.ndarray,
    ) -> None:
        if not block_id:
            raise ValueError("block_id must not be empty")
        if any(block.block_id == block_id for block in self._blocks):
            raise ValueError(f"duplicate completed block: {block_id}")
        if not start_date or not end_date or start_date > end_date:
            raise ValueError("block dates must be non-empty and ordered")
        if self._blocks and self._blocks[-1].end_date >= start_date:
            raise ValueError("completed blocks must be strictly chronological")

        arrays = [
            np.asarray(dispersion, dtype=float),
            np.asarray(aggressive_return, dtype=float),
            np.asarray(conservative_return, dtype=float),
            np.asarray(fallback_return, dtype=float),
        ]
        diverged = np.asarray(branch_diverged, dtype=bool)
        lengths = {len(array) for array in [*arrays, diverged]}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError("completed-block arrays must be non-empty and aligned")
        if any(array.ndim != 1 for array in arrays) or diverged.ndim != 1:
            raise ValueError("completed-block arrays must be one-dimensional")
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("completed-block values must be finite")
        if np.any(np.concatenate(arrays[1:]) <= -1.0):
            raise ValueError("returns must be greater than -1")

        self._blocks.append(
            _ModeAdvantageBlock(
                block_id=str(block_id),
                start_date=str(start_date),
                end_date=str(end_date),
                dispersion=arrays[0].copy(),
                aggressive_return=arrays[1].copy(),
                conservative_return=arrays[2].copy(),
                fallback_return=arrays[3].copy(),
                branch_diverged=diverged.copy(),
            )
        )

    def _fallback_decision(
        self,
        status: str,
        *,
        history_blocks: int,
        history_days: int,
        informative_days: int,
        cdf_days: int,
        low_mode_advantage: float = np.nan,
        high_mode_advantage: float = np.nan,
        policy_advantage_mean: float = np.nan,
        policy_advantage_lcb: float = np.nan,
        threshold_quantile: float = np.nan,
        raw_threshold: float = np.nan,
        fit_start_date: str = "",
        fit_end_date: str = "",
    ) -> CrossoverTauDecision:
        return CrossoverTauDecision(
            status=status,
            selected_tau=np.nan,
            threshold_quantile=float(threshold_quantile),
            raw_threshold=float(raw_threshold),
            history_blocks=int(history_blocks),
            history_days=int(history_days),
            informative_days=int(informative_days),
            cdf_days=int(cdf_days),
            low_mode_advantage=float(low_mode_advantage),
            high_mode_advantage=float(high_mode_advantage),
            policy_advantage_mean=float(policy_advantage_mean),
            policy_advantage_lcb=float(policy_advantage_lcb),
            fit_start_date=fit_start_date,
            fit_end_date=fit_end_date,
        )

    @staticmethod
    def _empirical_percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
        ordered = np.sort(np.asarray(reference, dtype=float))
        return np.searchsorted(ordered, values, side="right") / len(ordered)

    def _snap_tau(self, value: float) -> float:
        distance = np.abs(self.tau_values - float(value))
        return float(self.tau_values[int(np.flatnonzero(distance == distance.min())[0])])

    def select_for_next_block(self) -> CrossoverTauDecision:
        fit_blocks = self._blocks[-self.max_fit_blocks :]
        history_blocks = len(fit_blocks)
        history_days = int(sum(len(block.dispersion) for block in fit_blocks))
        informative_days = int(
            sum(int(block.branch_diverged.sum()) for block in fit_blocks)
        )
        fit_start_date = fit_blocks[0].start_date if fit_blocks else ""
        fit_end_date = fit_blocks[-1].end_date if fit_blocks else ""
        cdf_days = min(history_days, self.cdf_lookback_days)

        if history_blocks < self.min_history_blocks:
            return self._fallback_decision(
                "fallback_insufficient_blocks",
                history_blocks=history_blocks,
                history_days=history_days,
                informative_days=informative_days,
                cdf_days=cdf_days,
                fit_start_date=fit_start_date,
                fit_end_date=fit_end_date,
            )
        if informative_days < self.min_informative_days:
            return self._fallback_decision(
                "fallback_insufficient_informative_days",
                history_blocks=history_blocks,
                history_days=history_days,
                informative_days=informative_days,
                cdf_days=cdf_days,
                fit_start_date=fit_start_date,
                fit_end_date=fit_end_date,
            )

        dispersion = np.concatenate([block.dispersion for block in fit_blocks])
        aggressive = np.concatenate([block.aggressive_return for block in fit_blocks])
        conservative = np.concatenate(
            [block.conservative_return for block in fit_blocks]
        )
        diverged = np.concatenate([block.branch_diverged for block in fit_blocks])
        cdf_reference = dispersion[-self.cdf_lookback_days :]
        percentile = self._empirical_percentile(dispersion, cdf_reference)
        mode_gap = aggressive - conservative

        fit_x = percentile[diverged]
        fit_y = mode_gap[diverged]
        isotonic = IsotonicRegression(increasing=False, out_of_bounds="clip")
        isotonic.fit(fit_x, fit_y)
        quantile_grid = np.arange(0.01, 1.0, 0.01)
        fitted_gap = isotonic.predict(quantile_grid)

        nonpositive = np.flatnonzero(fitted_gap <= 0.0)
        if fitted_gap[0] <= 0.0 or len(nonpositive) == 0:
            return self._fallback_decision(
                "fallback_no_mode_crossover",
                history_blocks=history_blocks,
                history_days=history_days,
                informative_days=informative_days,
                cdf_days=len(cdf_reference),
                low_mode_advantage=float(fitted_gap[0]),
                high_mode_advantage=float(fitted_gap[-1]),
                fit_start_date=fit_start_date,
                fit_end_date=fit_end_date,
            )
        crossing_index = int(nonpositive[0])
        if crossing_index == 0:
            return self._fallback_decision(
                "fallback_no_mode_crossover",
                history_blocks=history_blocks,
                history_days=history_days,
                informative_days=informative_days,
                cdf_days=len(cdf_reference),
                low_mode_advantage=float(fitted_gap[0]),
                high_mode_advantage=float(fitted_gap[-1]),
                fit_start_date=fit_start_date,
                fit_end_date=fit_end_date,
            )
        threshold_quantile = float(
            (quantile_grid[crossing_index - 1] + quantile_grid[crossing_index]) / 2.0
        )
        low_count = int(np.sum(fit_x < threshold_quantile))
        high_count = int(np.sum(fit_x >= threshold_quantile))
        if low_count < self.min_side_days or high_count < self.min_side_days:
            return self._fallback_decision(
                "fallback_insufficient_crossover_support",
                history_blocks=history_blocks,
                history_days=history_days,
                informative_days=informative_days,
                cdf_days=len(cdf_reference),
                low_mode_advantage=float(fitted_gap[0]),
                high_mode_advantage=float(fitted_gap[-1]),
                threshold_quantile=threshold_quantile,
                fit_start_date=fit_start_date,
                fit_end_date=fit_end_date,
            )

        raw_threshold = float(
            np.quantile(cdf_reference, threshold_quantile, method="linear")
        )
        selected_tau = self._snap_tau(raw_threshold)

        block_advantages = []
        for block in fit_blocks:
            selected_return = np.where(
                block.dispersion < selected_tau,
                block.aggressive_return,
                block.conservative_return,
            )
            daily_log_advantage = np.log1p(selected_return) - np.log1p(
                block.fallback_return
            )
            block_advantages.append(float(daily_log_advantage.mean()))
        advantage = np.asarray(block_advantages, dtype=float)
        advantage_mean = float(advantage.mean())
        if len(advantage) < 2:
            advantage_lcb = -np.inf
        else:
            standard_error = float(advantage.std(ddof=1) / np.sqrt(len(advantage)))
            critical = float(
                stats.t.ppf(self.confidence_level, df=len(advantage) - 1)
            )
            advantage_lcb = advantage_mean - critical * standard_error

        common = {
            "history_blocks": history_blocks,
            "history_days": history_days,
            "informative_days": informative_days,
            "cdf_days": len(cdf_reference),
            "low_mode_advantage": float(fitted_gap[0]),
            "high_mode_advantage": float(fitted_gap[-1]),
            "policy_advantage_mean": advantage_mean,
            "policy_advantage_lcb": advantage_lcb,
            "threshold_quantile": threshold_quantile,
            "raw_threshold": raw_threshold,
            "fit_start_date": fit_start_date,
            "fit_end_date": fit_end_date,
        }
        if self.require_positive_lcb and advantage_lcb <= 0.0:
            return self._fallback_decision(
                "fallback_nonpositive_policy_lcb",
                **common,
            )
        return CrossoverTauDecision(
            status="selected",
            selected_tau=selected_tau,
            **common,
        )
