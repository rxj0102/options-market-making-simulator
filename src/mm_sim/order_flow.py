"""Simulated customer order flow.

Orders arrive as a Poisson process and are mostly directionless noise, but a
configurable fraction are "informed": their side is biased toward whichever
option would profit from the underlying's move over the next few ticks.
This is a deliberate look-ahead -- real market makers can't see the future
-- used *only* to generate order flow that is genuinely adversely selected,
so that the toxicity signal the quoting engine reacts to (see
``pnl_attribution.py`` / rolling markout) has something real to detect. The
market maker itself never receives this information; it only ever sees
side, size, and strike, and has to infer toxicity after the fact from how
price moves following its fills.

Wider quoted spreads also reduce fill probability here, which is what
creates the actual edge-vs-flow tradeoff a market maker faces: quote wide
enough to survive toxic flow and you trade less; quote tight and you trade
more but eat more adverse selection.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from mm_sim.market_data import OptionContract
from mm_sim.pricing import OptionType
from mm_sim.quoting_engine import Quote


class Side(str, Enum):
    """Customer's side of the trade (the market maker takes the other side)."""

    BUY = "buy"    # customer buys -> hits the market maker's ask
    SELL = "sell"  # customer sells -> hits the market maker's bid


@dataclass(frozen=True)
class CustomerOrder:
    """A single customer order attempting to trade against the MM's quote."""

    contract: OptionContract
    side: Side
    size: int


@dataclass(frozen=True)
class OrderFlowConfig:
    """Tunable parameters of the simulated order flow process."""

    arrival_rate_per_step: float = 0.6
    """Average number of orders (across the whole chain) generated per tick."""

    informed_probability: float = 0.15
    """Fraction of orders whose side is biased toward the near-term price move."""

    informed_lookahead_steps: int = 5
    """How many ticks ahead an "informed" order's bias looks."""

    informed_min_move: float = 0.0015
    """Minimum relative underlying move over the lookahead for an order to be informed;
    below this the move is treated as noise and the order is random."""

    min_size: int = 1
    max_size: int = 10
    """Order size bounds, in contracts."""

    atm_bias: float = 1.5
    """Higher values concentrate order flow more heavily on near-ATM strikes."""

    base_fill_prob: float = 0.9
    """Fill probability in the limit of a razor-thin (near-zero) spread."""

    fill_prob_spread_sensitivity: float = 8.0
    """How quickly fill probability decays as the quoted relative spread widens."""


class OrderFlowGenerator:
    """Generates customer orders and decides whether they fill against a quote."""

    def __init__(self, config: OrderFlowConfig | None = None, seed: int | None = None) -> None:
        self.config = config or OrderFlowConfig()
        self._rng = np.random.default_rng(seed)

    def generate_orders(
        self,
        chain: list[OptionContract],
        spot: float,
        future_spot: float,
        strike_spacing: float,
    ) -> list[CustomerOrder]:
        """Generate this tick's customer orders.

        Args:
            chain: The full listed options chain.
            spot: Current underlying price.
            future_spot: Underlying price ``informed_lookahead_steps`` ticks
                ahead. Used only to bias informed order direction -- never
                exposed to the market maker.
            strike_spacing: Distance between adjacent strikes, used to scale
                the ATM concentration weighting.

        Returns:
            List of customer orders generated this tick (possibly empty).
        """
        cfg = self.config
        n_orders = self._rng.poisson(cfg.arrival_rate_per_step)
        if n_orders == 0:
            return []

        weights = np.array(
            [math.exp(-cfg.atm_bias * abs(c.strike - spot) / strike_spacing) for c in chain]
        )
        weights /= weights.sum()

        relative_move = (future_spot - spot) / spot
        is_informed_move = abs(relative_move) >= cfg.informed_min_move
        bullish = relative_move > 0

        orders: list[CustomerOrder] = []
        for _ in range(n_orders):
            contract = chain[self._rng.choice(len(chain), p=weights)]
            size = int(self._rng.integers(cfg.min_size, cfg.max_size + 1))

            is_informed = is_informed_move and self._rng.random() < cfg.informed_probability
            if is_informed:
                # Trade the side that profits if the anticipated move happens.
                wants_call_like_exposure = bullish
                if contract.option_type == OptionType.CALL:
                    side = Side.BUY if wants_call_like_exposure else Side.SELL
                else:
                    side = Side.SELL if wants_call_like_exposure else Side.BUY
            else:
                side = Side.BUY if self._rng.random() < 0.5 else Side.SELL

            orders.append(CustomerOrder(contract=contract, side=side, size=size))

        return orders

    def fill_probability(self, quote: Quote) -> float:
        """Probability a customer trades at the given quote.

        Modeled as decaying exponentially in the *relative* half-spread
        (half-spread as a fraction of theo): tight, competitive markets
        trade often; wide, defensive markets trade rarely.
        """
        cfg = self.config
        relative_spread = quote.half_spread / max(quote.theo, 0.01)
        prob = cfg.base_fill_prob * math.exp(-cfg.fill_prob_spread_sensitivity * relative_spread)
        return float(np.clip(prob, 0.0, 1.0))

    def decide_fill(self, quote: Quote) -> bool:
        """Roll the dice on whether an order at this quote actually trades."""
        return self._rng.random() < self.fill_probability(quote)


@dataclass(frozen=True)
class MarkoutConfig:
    """Parameters for inferring order flow toxicity from post-fill markouts."""

    lookahead_steps: int = 5
    """How many ticks after a fill to check whether the market moved against it."""

    window_size: int = 30
    """Number of resolved fills kept in the rolling toxicity estimate."""


class ToxicityTracker:
    """Infers order flow toxicity from realized post-fill price markouts.

    The market maker never sees whether a customer order was "informed" --
    it only sees, after the fact, whether the underlying moved against the
    delta exposure a fill just handed it. This tracks exactly that: each
    fill is checked ``lookahead_steps`` later, and the score is the rate of
    adverse markouts *in excess of* the 50% rate pure noise would produce by
    chance, clipped to [0, 1]. Pure random flow reads as 0; consistently
    informed flow approaches 1.
    """

    def __init__(self, config: MarkoutConfig | None = None) -> None:
        self.config = config or MarkoutConfig()
        self._pending: deque[tuple[int, float, float]] = deque()
        self._recent_outcomes: deque[bool] = deque(maxlen=self.config.window_size)

    def record_fill(self, step_index: int, mm_delta_shares: float, spot_at_fill: float) -> None:
        """Queue a fill to be marked out once ``lookahead_steps`` have passed.

        Args:
            step_index: Tick at which the fill occurred.
            mm_delta_shares: Change in the market maker's shares-equivalent
                delta caused by this fill (positive = more bullish exposure).
            spot_at_fill: Underlying price at the moment of the fill.
        """
        self._pending.append((step_index, mm_delta_shares, spot_at_fill))

    def update(self, step_index: int, spot_now: float) -> float:
        """Resolve any fills old enough to mark out and return the current score."""
        cfg = self.config
        while self._pending and step_index - self._pending[0][0] >= cfg.lookahead_steps:
            _, mm_delta_shares, spot_at_fill = self._pending.popleft()
            markout_pnl = mm_delta_shares * (spot_now - spot_at_fill)
            self._recent_outcomes.append(markout_pnl < 0)
        return self.toxicity_score

    @property
    def toxicity_score(self) -> float:
        if not self._recent_outcomes:
            return 0.0
        adverse_fraction = sum(self._recent_outcomes) / len(self._recent_outcomes)
        return float(np.clip(2.0 * (adverse_fraction - 0.5), 0.0, 1.0))
