from __future__ import annotations

import numpy as np

from finrl.reproduction.causal_crossover_tau import CausalCrossoverTau


TAU_GRID = np.arange(0.01, 0.90, 0.01)


def add_synthetic_block(
    controller: CausalCrossoverTau,
    block_index: int,
    *,
    fallback_return: float = -0.002,
    always_aggressive: bool = False,
) -> None:
    dispersion = np.linspace(0.05, 0.85, 63)
    if always_aggressive:
        gap = np.full(63, 0.002)
    else:
        gap = 0.01 * (0.45 - dispersion)
    aggressive = 0.001 + gap / 2.0
    conservative = 0.001 - gap / 2.0
    year = 2010 + block_index
    controller.add_completed_block(
        block_id=f"block-{block_index}",
        start_date=f"{year}-01-01",
        end_date=f"{year}-03-31",
        dispersion=dispersion,
        aggressive_return=aggressive,
        conservative_return=conservative,
        fallback_return=np.full(63, fallback_return),
        branch_diverged=np.ones(63, dtype=bool),
    )


def test_selects_causal_zero_crossing_and_maps_to_tau_grid() -> None:
    controller = CausalCrossoverTau(
        TAU_GRID,
        min_history_blocks=4,
        min_informative_days=63,
        require_positive_lcb=False,
    )
    for block in range(4):
        add_synthetic_block(controller, block)

    decision = controller.select_for_next_block()

    assert decision.selected
    assert 0.45 <= decision.threshold_quantile <= 0.55
    assert 0.40 <= decision.selected_tau <= 0.50
    assert decision.fit_end_date == "2013-03-31"
    assert decision.history_blocks == 4


def test_guardrail_allows_only_positive_block_level_advantage() -> None:
    passing = CausalCrossoverTau(TAU_GRID, min_history_blocks=4)
    failing = CausalCrossoverTau(TAU_GRID, min_history_blocks=4)
    for block in range(4):
        add_synthetic_block(passing, block, fallback_return=-0.002)
        add_synthetic_block(failing, block, fallback_return=0.02)

    assert passing.select_for_next_block().selected
    failed = failing.select_for_next_block()
    assert failed.status == "fallback_nonpositive_policy_lcb"
    assert np.isnan(failed.selected_tau)


def test_no_mode_crossover_uses_fallback() -> None:
    controller = CausalCrossoverTau(TAU_GRID, min_history_blocks=4)
    for block in range(4):
        add_synthetic_block(controller, block, always_aggressive=True)

    decision = controller.select_for_next_block()

    assert decision.status == "fallback_no_mode_crossover"
    assert not decision.selected


def test_unadded_future_outcomes_cannot_change_prior_decision() -> None:
    controller = CausalCrossoverTau(
        TAU_GRID,
        min_history_blocks=4,
        require_positive_lcb=False,
    )
    for block in range(4):
        add_synthetic_block(controller, block)
    before = controller.select_for_next_block()

    future_gap = np.full(63, -1.0)
    assert before == controller.select_for_next_block()
    assert future_gap.mean() == -1.0


def test_completed_blocks_must_be_strictly_chronological() -> None:
    controller = CausalCrossoverTau(TAU_GRID, min_history_blocks=1)
    add_synthetic_block(controller, 0)

    try:
        add_synthetic_block(controller, 0)
    except ValueError as error:
        assert "duplicate completed block" in str(error)
    else:
        raise AssertionError("duplicate block must fail")
