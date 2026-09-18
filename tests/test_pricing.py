import math

import pytest

from mm_sim.pricing import OptionType, black_scholes_greeks, black_scholes_price

SPOT = 100.0
STRIKE = 100.0
T = 30 / 365
RATE = 0.04
VOL = 0.20


def test_put_call_parity():
    call = black_scholes_price(SPOT, STRIKE, T, RATE, VOL, OptionType.CALL)
    put = black_scholes_price(SPOT, STRIKE, T, RATE, VOL, OptionType.PUT)
    rhs = SPOT - STRIKE * math.exp(-RATE * T)
    assert math.isclose(call - put, rhs, abs_tol=1e-9)


def test_atm_call_delta_near_half():
    greeks = black_scholes_greeks(SPOT, STRIKE, T, RATE, VOL, OptionType.CALL)
    assert 0.45 < greeks.delta < 0.65


def test_atm_put_delta_near_negative_half():
    greeks = black_scholes_greeks(SPOT, STRIKE, T, RATE, VOL, OptionType.PUT)
    assert -0.55 < greeks.delta < -0.35


def test_call_and_put_share_gamma_and_vega():
    call_greeks = black_scholes_greeks(SPOT, STRIKE, T, RATE, VOL, OptionType.CALL)
    put_greeks = black_scholes_greeks(SPOT, STRIKE, T, RATE, VOL, OptionType.PUT)
    assert math.isclose(call_greeks.gamma, put_greeks.gamma, rel_tol=1e-9)
    assert math.isclose(call_greeks.vega, put_greeks.vega, rel_tol=1e-9)


def test_deep_itm_call_delta_near_one():
    greeks = black_scholes_greeks(spot=150.0, strike=STRIKE, time_to_expiry=T, rate=RATE, vol=VOL, option_type=OptionType.CALL)
    assert greeks.delta > 0.95


def test_deep_otm_call_delta_near_zero():
    greeks = black_scholes_greeks(spot=60.0, strike=STRIKE, time_to_expiry=T, rate=RATE, vol=VOL, option_type=OptionType.CALL)
    assert greeks.delta < 0.05


def test_price_positive_and_bounded_by_spot():
    call = black_scholes_price(SPOT, STRIKE, T, RATE, VOL, OptionType.CALL)
    assert 0.0 < call < SPOT


def test_zero_time_to_expiry_returns_intrinsic_value():
    call = black_scholes_price(spot=110.0, strike=100.0, time_to_expiry=0.0, rate=RATE, vol=VOL, option_type=OptionType.CALL)
    assert math.isclose(call, 10.0)
    put = black_scholes_price(spot=110.0, strike=100.0, time_to_expiry=0.0, rate=RATE, vol=VOL, option_type=OptionType.PUT)
    assert math.isclose(put, 0.0)


def test_negative_time_to_expiry_raises():
    with pytest.raises(ValueError):
        black_scholes_greeks(SPOT, STRIKE, -0.01, RATE, VOL, OptionType.CALL)
