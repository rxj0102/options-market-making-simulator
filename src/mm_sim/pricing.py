"""Black-Scholes option pricing and Greeks.

Everything here is pure, stateless math: given market/contract parameters,
return a theoretical fair value and risk sensitivities. The rest of the
simulator treats these numbers as the market maker's "theo" -- the anchor
that quotes are built around.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import norm


class OptionType(str, Enum):
    """Contract side. Values are used verbatim in logs and plots."""

    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class Greeks:
    """First-order (and gamma) risk sensitivities of a single option.

    Units: delta/gamma are per 1.0 move in the underlying, vega is per 1.00
    (100 vol points) change in volatility, theta is per calendar day, rho is
    per 1.00 (100%) change in the risk-free rate.
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _d1_d2(spot: float, strike: float, time_to_expiry: float, rate: float, vol: float) -> tuple[float, float]:
    """Shared intermediate terms used by both price and Greek formulas."""
    if time_to_expiry <= 0.0 or vol <= 0.0:
        raise ValueError("time_to_expiry and vol must be strictly positive")
    sqrt_t = np.sqrt(time_to_expiry)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * time_to_expiry) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return d1, d2


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    vol: float,
    option_type: OptionType,
) -> float:
    """Theoretical Black-Scholes fair value (no dividends).

    Args:
        spot: Current underlying price.
        strike: Option strike price.
        time_to_expiry: Time to expiration, in years.
        rate: Continuously-compounded risk-free rate (e.g. 0.05 for 5%).
        vol: Annualized implied volatility used to price the option (e.g. 0.20).
        option_type: OptionType.CALL or OptionType.PUT.

    Returns:
        The theoretical option price.
    """
    if time_to_expiry <= 0.0:
        intrinsic = spot - strike if option_type == OptionType.CALL else strike - spot
        return max(intrinsic, 0.0)

    d1, d2 = _d1_d2(spot, strike, time_to_expiry, rate, vol)
    discount = np.exp(-rate * time_to_expiry)

    if option_type == OptionType.CALL:
        return spot * norm.cdf(d1) - strike * discount * norm.cdf(d2)
    return strike * discount * norm.cdf(-d2) - spot * norm.cdf(-d1)


def black_scholes_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    vol: float,
    option_type: OptionType,
) -> Greeks:
    """Analytic Black-Scholes Greeks (no dividends).

    Args:
        spot: Current underlying price.
        strike: Option strike price.
        time_to_expiry: Time to expiration, in years. Must be > 0; the
            simulator marks an option as expired (flat, no Greeks) rather
            than calling this at expiry.
        rate: Continuously-compounded risk-free rate.
        vol: Annualized implied volatility used to price the option.
        option_type: OptionType.CALL or OptionType.PUT.

    Returns:
        A Greeks instance with delta, gamma, vega, theta (per day), rho.
    """
    d1, d2 = _d1_d2(spot, strike, time_to_expiry, rate, vol)
    sqrt_t = np.sqrt(time_to_expiry)
    discount = np.exp(-rate * time_to_expiry)
    pdf_d1 = norm.pdf(d1)

    gamma = pdf_d1 / (spot * vol * sqrt_t)
    vega = spot * pdf_d1 * sqrt_t / 100.0  # per 1 vol point (0.01), scaled to "per 1.00" below in callers if needed

    if option_type == OptionType.CALL:
        delta = norm.cdf(d1)
        theta_annual = -(spot * pdf_d1 * vol) / (2 * sqrt_t) - rate * strike * discount * norm.cdf(d2)
        rho = strike * time_to_expiry * discount * norm.cdf(d2) / 100.0
    else:
        delta = norm.cdf(d1) - 1.0
        theta_annual = -(spot * pdf_d1 * vol) / (2 * sqrt_t) + rate * strike * discount * norm.cdf(-d2)
        rho = -strike * time_to_expiry * discount * norm.cdf(-d2) / 100.0

    theta_per_day = theta_annual / 365.0

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta_per_day, rho=rho)


def price_and_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    vol: float,
    option_type: OptionType,
) -> tuple[float, Greeks]:
    """Convenience wrapper returning both price and Greeks in one call."""
    price = black_scholes_price(spot, strike, time_to_expiry, rate, vol, option_type)
    greeks = black_scholes_greeks(spot, strike, time_to_expiry, rate, vol, option_type)
    return price, greeks
