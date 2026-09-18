import pytest

from mm_sim.market_data import MINUTES_PER_TRADING_DAY
from mm_sim.simulation import MarketMakingSimulation, SimulationConfig


@pytest.fixture(scope="module")
def result():
    config = SimulationConfig(session_minutes=120, seed=7)
    return MarketMakingSimulation(config).run()


def test_tick_log_covers_full_session(result):
    assert len(result.ticks) == 120


def test_pnl_decomposition_reconciles_every_tick(result):
    ticks = result.ticks
    reconciled = ticks["cumulative_spread_capture"] - ticks["cumulative_hedging_cost"] + ticks["residual_risk_pnl"]
    assert (reconciled - ticks["total_pnl"]).abs().max() < 1e-6


def test_hedging_keeps_net_delta_bounded(result):
    config = result.config
    # Allow some slack for the single tick between a breach and the resulting hedge.
    assert result.ticks["net_delta_total"].abs().max() < config.hedging.delta_threshold * 3


def test_fills_reference_contracts_in_the_chain(result):
    chain_symbols = {c.symbol for c in result.chain}
    if len(result.fills):
        assert set(result.fills["contract"]).issubset(chain_symbols)


def test_breach_actions_are_one_of_known_types(result):
    known_actions = {"block_position_increasing_orders", "force_hedge", "lockout_new_risk", "halt_trading"}
    if len(result.breaches):
        assert set(result.breaches["action"]).issubset(known_actions)


def test_simulation_is_reproducible_with_same_seed():
    config = SimulationConfig(session_minutes=60, seed=99)
    result_a = MarketMakingSimulation(config).run()
    result_b = MarketMakingSimulation(config).run()
    assert result_a.ticks["total_pnl"].equals(result_b.ticks["total_pnl"])


def test_default_session_length_is_one_trading_day():
    assert SimulationConfig().session_minutes == MINUTES_PER_TRADING_DAY
