import numpy as np
import pytest

from mm_sim.market_data import (
    build_option_chain,
    build_vol_regime,
    minutes_to_years,
    rolling_realized_vol,
    simulate_gbm_path,
)
from mm_sim.pricing import OptionType


def test_gbm_path_starts_at_s0_and_has_correct_length():
    path = simulate_gbm_path(s0=100.0, mu=0.0, sigma=0.2, n_steps=50, dt=minutes_to_years(1), seed=1)
    assert len(path) == 51
    assert path[0] == pytest.approx(100.0)


def test_gbm_path_is_reproducible_with_seed():
    path_a = simulate_gbm_path(s0=100.0, mu=0.0, sigma=0.2, n_steps=50, dt=minutes_to_years(1), seed=7)
    path_b = simulate_gbm_path(s0=100.0, mu=0.0, sigma=0.2, n_steps=50, dt=minutes_to_years(1), seed=7)
    assert np.array_equal(path_a, path_b)


def test_gbm_path_prices_stay_positive():
    path = simulate_gbm_path(s0=100.0, mu=0.0, sigma=0.6, n_steps=500, dt=minutes_to_years(1), seed=3)
    assert np.all(path > 0)


def test_vol_regime_spike_is_higher_than_base():
    regime = build_vol_regime(n_steps=100, base_vol=0.20, spike_start_frac=0.4, spike_end_frac=0.6, spike_multiplier=2.0)
    assert np.isclose(regime[0], 0.20)
    assert np.isclose(regime[50], 0.40)
    assert np.isclose(regime[90], 0.20)


def test_rolling_realized_vol_matches_length_of_prices():
    path = simulate_gbm_path(s0=100.0, mu=0.0, sigma=0.2, n_steps=100, dt=minutes_to_years(1), seed=1)
    realized = rolling_realized_vol(path, window=10, dt=minutes_to_years(1))
    assert len(realized) == len(path)
    assert not np.any(np.isnan(realized))


def test_rolling_realized_vol_recovers_constant_vol_approximately():
    sigma = 0.25
    path = simulate_gbm_path(s0=100.0, mu=0.0, sigma=sigma, n_steps=2000, dt=minutes_to_years(1), seed=11)
    realized = rolling_realized_vol(path, window=200, dt=minutes_to_years(1))
    assert abs(realized[-1] - sigma) < 0.05


def test_build_option_chain_centers_on_atm_and_lists_both_sides():
    chain = build_option_chain(spot0=101.0, n_strikes=5, strike_spacing=5.0)
    assert len(chain) == 2 * 5  # calls + puts per strike
    calls = [c for c in chain if c.option_type == OptionType.CALL]
    puts = [c for c in chain if c.option_type == OptionType.PUT]
    assert len(calls) == len(puts) == 5
    call_strikes = sorted(c.strike for c in calls)
    assert 100.0 in call_strikes  # rounds to nearest strike_spacing


def test_option_contract_symbol_format():
    chain = build_option_chain(spot0=100.0, n_strikes=1, strike_spacing=5.0)
    symbols = {c.symbol for c in chain}
    assert symbols == {"100C", "100P"}
