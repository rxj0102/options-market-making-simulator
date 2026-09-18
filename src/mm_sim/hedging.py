"""Delta hedging.

The options book accumulates delta exposure as customer flow trades against
the market maker's quotes. This module decides when to trade the underlying
to flatten that exposure, and at what cost. Two triggers are supported and
can be combined:

  - threshold hedging: hedge as soon as net delta exceeds a band
  - time-based hedging: hedge at least every N ticks regardless of size

Hedging more often reduces residual risk (P&L noise from unhedged
gamma/vega drift) but pays more in transaction cost; hedging less often
saves on cost but leaves more directional exposure on the book. That
tradeoff is deliberately left as a config knob rather than "solved", since
showing the tradeoff is the point.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HedgeConfig:
    """Hedging trigger and cost parameters."""

    delta_threshold: float = 50.0
    """Net (options + underlying) delta, in shares-equivalent, that triggers an immediate hedge."""

    max_steps_between_hedges: int | None = 20
    """Force a hedge at least this often even if under threshold. None disables time-based hedging."""

    cost_per_share: float = 0.005
    """Assumed cost (slippage / half the underlying's own spread) per share hedged."""


@dataclass(frozen=True)
class HedgeTrade:
    """A single hedge execution in the underlying."""

    step_index: int
    shares: float
    """Signed shares traded; positive = bought underlying, negative = sold."""
    price: float
    cost: float
    reason: str
    """"threshold" or "time", i.e. which trigger fired."""


class DeltaHedger:
    """Tracks the underlying hedge position and decides when to trade."""

    def __init__(self, config: HedgeConfig | None = None) -> None:
        self.config = config or HedgeConfig()
        self.underlying_position: float = 0.0
        self._steps_since_hedge: int = 0

    def net_delta(self, options_delta_shares: float) -> float:
        """Total delta exposure: options book plus the current underlying hedge."""
        return options_delta_shares + self.underlying_position

    def maybe_hedge(self, step_index: int, options_delta_shares: float, spot: float) -> HedgeTrade | None:
        """Check triggers and execute a flattening hedge trade if warranted.

        Args:
            step_index: Current simulation tick index (for logging).
            options_delta_shares: Net delta from the options book alone, in
                shares-equivalent (i.e. sum of position_contracts * delta *
                contract_multiplier across the chain).
            spot: Current underlying price, used as the hedge execution price.

        Returns:
            The ``HedgeTrade`` executed, or ``None`` if no hedge was triggered.
        """
        cfg = self.config
        self._steps_since_hedge += 1

        net = self.net_delta(options_delta_shares)
        if abs(net) > cfg.delta_threshold:
            reason = "threshold"
        elif (
            cfg.max_steps_between_hedges is not None
            and self._steps_since_hedge >= cfg.max_steps_between_hedges
            and abs(net) > 1e-9
        ):
            reason = "time"
        else:
            return None

        return self._execute(step_index, net, spot, reason)

    def force_hedge(self, step_index: int, options_delta_shares: float, spot: float) -> HedgeTrade:
        """Immediately flatten net delta, bypassing the normal triggers.

        Used when a portfolio-level risk limit breach demands an immediate
        hedge regardless of the configured threshold or hedge cadence.
        """
        self._steps_since_hedge += 1
        net = self.net_delta(options_delta_shares)
        return self._execute(step_index, net, spot, reason="risk_limit_breach")

    def _execute(self, step_index: int, net_delta_shares: float, spot: float, reason: str) -> HedgeTrade:
        hedge_shares = -net_delta_shares  # trade that flattens net delta to zero
        cost = abs(hedge_shares) * self.config.cost_per_share

        self.underlying_position += hedge_shares
        self._steps_since_hedge = 0

        return HedgeTrade(step_index=step_index, shares=hedge_shares, price=spot, cost=cost, reason=reason)
