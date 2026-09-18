"""Underlying price simulation and options chain construction.

The underlying follows simulated geometric Brownian motion (GBM) rather
than real market data, so the simulation is fully reproducible and lets us
script a volatility regime shift mid-session. That vol shift is what makes
"realized vol suddenly disagrees with the implied vol the chain was priced
at" a real, visible event in the simulation rather than a hypothetical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mm_sim.pricing import OptionType

MINUTES_PER_TRADING_DAY = 390
TRADING_DAYS_PER_YEAR = 252


def minutes_to_years(n_minutes: float) -> float:
    """Convert a number of 1-minute bars to a year fraction (ACT/252-style)."""
    return n_minutes / (MINUTES_PER_TRADING_DAY * TRADING_DAYS_PER_YEAR)


def simulate_gbm_path(
    s0: float,
    mu: float,
    sigma: float | np.ndarray,
    n_steps: int,
    dt: float,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate a GBM price path.

    Args:
        s0: Starting underlying price.
        mu: Annualized drift.
        sigma: Annualized volatility. Either a scalar (constant vol) or an
            array of length ``n_steps`` giving the vol used for each step
            (this is how a mid-session vol regime shift is injected).
        n_steps: Number of time steps to simulate.
        dt: Length of one step, in years.
        seed: RNG seed for reproducibility.

    Returns:
        Array of length ``n_steps + 1`` (including the starting price) of
        simulated underlying prices.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_steps)
    sigma_arr = np.broadcast_to(np.asarray(sigma, dtype=float), (n_steps,))

    log_returns = (mu - 0.5 * sigma_arr**2) * dt + sigma_arr * np.sqrt(dt) * z
    log_prices = np.log(s0) + np.concatenate([[0.0], np.cumsum(log_returns)])
    return np.exp(log_prices)


def build_vol_regime(
    n_steps: int,
    base_vol: float,
    spike_start_frac: float = 0.40,
    spike_end_frac: float = 0.55,
    spike_multiplier: float = 1.8,
) -> np.ndarray:
    """Build a per-step volatility schedule with a temporary mid-session spike.

    This is what creates a realized-vs-implied vol mismatch during the
    simulation: the options chain is priced (once, at session start) off
    ``base_vol``, but the underlying actually realizes a higher vol for a
    stretch in the middle of the session.

    Args:
        n_steps: Number of simulation steps.
        base_vol: Annualized vol outside of the spike window.
        spike_start_frac: Fraction of the session (0-1) where the spike begins.
        spike_end_frac: Fraction of the session (0-1) where the spike ends.
        spike_multiplier: Multiplier applied to ``base_vol`` during the spike.

    Returns:
        Array of length ``n_steps`` with the vol to use at each step.
    """
    vol = np.full(n_steps, base_vol, dtype=float)
    start = int(n_steps * spike_start_frac)
    end = int(n_steps * spike_end_frac)
    vol[start:end] *= spike_multiplier
    return vol


def rolling_realized_vol(prices: np.ndarray, window: int, dt: float) -> np.ndarray:
    """Annualized rolling realized volatility from a price path.

    Args:
        prices: Array of underlying prices, length ``n_steps + 1``.
        window: Number of log-returns in the rolling window.
        dt: Length of one step, in years (used to annualize).

    Returns:
        Array the same length as ``prices``, with NaN until enough
        observations exist and forward-filled with the first valid estimate
        before that (so callers always have a usable number).
    """
    log_returns = np.diff(np.log(prices))
    realized = pd.Series(log_returns).rolling(window=window).std(ddof=1) * np.sqrt(1.0 / dt)
    realized = realized.bfill()
    # Prepend one value to align length with `prices` (no return exists at t=0).
    return np.concatenate([[realized.iloc[0]], realized.to_numpy()])


@dataclass(frozen=True)
class OptionContract:
    """A single listed contract in the simulated chain."""

    strike: float
    option_type: OptionType

    @property
    def symbol(self) -> str:
        side = "C" if self.option_type == OptionType.CALL else "P"
        return f"{self.strike:g}{side}"


def build_option_chain(
    spot0: float,
    n_strikes: int = 9,
    strike_spacing: float = 5.0,
) -> list[OptionContract]:
    """Build a single-expiry chain of calls and puts around the starting spot.

    Args:
        spot0: Underlying price at session start, used to center the chain.
        n_strikes: Number of strikes per side (should be odd so one strike
            lands ATM); calls and puts are listed at every strike.
        strike_spacing: Distance between adjacent strikes.

    Returns:
        A list of ``OptionContract`` (both calls and puts at each strike).
    """
    half = n_strikes // 2
    atm_strike = round(spot0 / strike_spacing) * strike_spacing
    strikes = [atm_strike + i * strike_spacing for i in range(-half, half + 1)]

    chain: list[OptionContract] = []
    for strike in strikes:
        chain.append(OptionContract(strike=strike, option_type=OptionType.CALL))
        chain.append(OptionContract(strike=strike, option_type=OptionType.PUT))
    return chain
