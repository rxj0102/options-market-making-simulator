"""Bid/ask quote construction.

A market maker does not quote "the theoretical price" -- they quote a price
they're willing to trade at, which is theo *adjusted* for the risk of being
wrong or being picked off. This module builds that adjustment explicitly as
a sum of named components, so every widening/skewing decision the engine
makes is traceable:

  mid  = theo + inventory_skew
  half_spread = base_half_spread
              + inventory_widen   (bigger position -> wider, to slow down further accumulation)
              + gamma_widen       (more gamma exposure -> wider, near-dated/ATM risk)
              + vol_mismatch_widen (realized vs. implied vol disagree -> less confidence in theo)
              + toxicity_widen    (recent fills have been adversely selected -> assume flow is informed)

  bid = mid - half_spread
  ask = mid + half_spread

This is deliberately a linear, inspectable model (in the spirit of
Avellaneda-Stoikov inventory skewing) rather than a black box, because the
point of this project is to show *why* a quote moves, not to fit a curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mm_sim.pricing import Greeks


@dataclass(frozen=True)
class QuotingConfig:
    """Tunable coefficients for the quoting engine.

    All "widen" coefficients are non-negative multipliers on a risk
    quantity; larger values produce wider markets. ``inventory_skew_coeff``
    controls how aggressively the mid is shifted (not just widened) to work
    off inventory.
    """

    min_half_spread: float = 0.02
    """Floor half-spread in price terms (e.g. exchange tick / minimum edge)."""

    max_half_spread: float = 2.00
    """Cap on half-spread so a single component can't blow the market out."""

    inventory_norm: float = 5_000.0
    """Dollar delta exposure treated as "one unit" of inventory risk."""

    inventory_skew_coeff: float = 0.15
    """Price shift per unit of normalized inventory (mid moves *against* position)."""

    inventory_widen_coeff: float = 0.10
    """Half-spread widening per unit of normalized (absolute) inventory."""

    gamma_widen_coeff: float = 0.05
    """Half-spread widening per unit of dollar gamma (P&L swing per 1% move)."""

    vol_mismatch_widen_coeff: float = 2.5
    """Half-spread widening per unit of (|realized - implied vol| * vega)."""

    toxicity_widen_coeff: float = 0.50
    """Half-spread widening per unit of toxicity score (expected in [0, 1])."""


@dataclass(frozen=True)
class Quote:
    """A two-sided market plus the fully decomposed reasoning behind it."""

    theo: float
    mid: float
    bid: float
    ask: float
    half_spread: float
    skew: float
    components: dict[str, float] = field(default_factory=dict)

    @property
    def spread(self) -> float:
        return self.ask - self.bid


class QuotingEngine:
    """Builds a bid/ask around a theoretical price, adjusted for risk."""

    def __init__(self, config: QuotingConfig | None = None) -> None:
        self.config = config or QuotingConfig()

    def compute_quote(
        self,
        theo: float,
        greeks: Greeks,
        position_contracts: float,
        spot: float,
        contract_multiplier: float,
        realized_vol: float,
        implied_vol: float,
        toxicity_score: float,
        min_price: float = 0.01,
    ) -> Quote:
        """Construct a bid/ask quote for a single option contract.

        Args:
            theo: Black-Scholes theoretical fair value for this contract.
            greeks: Greeks of a single contract (per-share, not position-scaled).
            position_contracts: Market maker's *current* signed position in
                this contract (positive = long, negative = short), before
                this quote is used to trade.
            spot: Current underlying price.
            contract_multiplier: Shares per contract (100 for standard equity options).
            realized_vol: Recently realized annualized volatility of the underlying.
            implied_vol: Implied volatility used to mark this contract's theo.
            toxicity_score: Rolling estimate in [0, 1] of how adversely
                selected recent fills in this name have been (0 = flow looks
                random, 1 = flow has been consistently picking off the MM).
            min_price: Floor so bid never quotes through zero.

        Returns:
            A Quote with bid/ask and a breakdown of each spread component,
            so callers (and the notebook) can show exactly why the market
            looks the way it does.
        """
        cfg = self.config

        dollar_delta = position_contracts * greeks.delta * contract_multiplier
        normalized_inventory = dollar_delta / cfg.inventory_norm

        # Skew the whole market down when long (encourage selling to us,
        # discourage more buying), up when short -- classic inventory control.
        skew = -cfg.inventory_skew_coeff * normalized_inventory

        inventory_widen = cfg.inventory_widen_coeff * abs(normalized_inventory)

        # Dollar gamma: approx P&L swing from a 1% underlying move, per contract.
        dollar_gamma = 0.5 * greeks.gamma * (spot * 0.01) ** 2 * contract_multiplier
        gamma_widen = cfg.gamma_widen_coeff * dollar_gamma

        vol_mismatch = abs(realized_vol - implied_vol)
        vol_mismatch_widen = cfg.vol_mismatch_widen_coeff * vol_mismatch * greeks.vega

        toxicity_widen = cfg.toxicity_widen_coeff * max(0.0, min(1.0, toxicity_score))

        half_spread = cfg.min_half_spread + inventory_widen + gamma_widen + vol_mismatch_widen + toxicity_widen
        half_spread = min(half_spread, cfg.max_half_spread)

        mid = theo + skew
        bid = max(min_price, mid - half_spread)
        ask = max(bid + min_price, mid + half_spread)

        components = {
            "base": cfg.min_half_spread,
            "inventory_widen": inventory_widen,
            "gamma_widen": gamma_widen,
            "vol_mismatch_widen": vol_mismatch_widen,
            "toxicity_widen": toxicity_widen,
            "inventory_skew": skew,
        }

        return Quote(theo=theo, mid=mid, bid=bid, ask=ask, half_spread=half_spread, skew=skew, components=components)
