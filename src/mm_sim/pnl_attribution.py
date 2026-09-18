"""P&L decomposition.

A market maker's daily P&L is one number, but three very different things
produced it:

  1. spread capture   -- edge earned by trading away from theo (bid below
                          theo, ask above theo), the "reward" for making markets
  2. hedging cost      -- transaction cost paid crossing the underlying's own
                          spread/slippage to stay delta-flat, the direct cost
                          of risk control
  3. residual risk P&L -- everything else: mark-to-market swings from
                          unhedged gamma/vega/theta between hedges, i.e. the
                          P&L consequence of *not* being perfectly hedged
                          at every instant

Spread capture and hedging cost are both directly measurable from trade
prices vs. theo and from hedge trade costs. Residual risk P&L is defined as
whatever is left: ``total P&L - spread_capture + hedging_cost``. This makes
the three buckets reconcile to total P&L *by construction*, which is the
point -- it isolates "P&L from making markets well" from "P&L from taking
uncontrolled risk", the same split a real market-making desk's P&L
attribution is trying to answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from mm_sim.hedging import HedgeTrade
from mm_sim.market_data import OptionContract
from mm_sim.order_flow import CustomerOrder, Side


@dataclass(frozen=True)
class PnLSnapshot:
    """P&L state at a single simulation tick."""

    step_index: int
    cash: float
    portfolio_value: float
    total_pnl: float
    cumulative_spread_capture: float
    cumulative_hedging_cost: float
    residual_risk_pnl: float


class PnLAttributor:
    """Tracks cash, marks the book, and decomposes cumulative P&L."""

    def __init__(self) -> None:
        self.cash: float = 0.0
        self.cumulative_spread_capture: float = 0.0
        self.cumulative_hedging_cost: float = 0.0
        self.history: list[PnLSnapshot] = []

    def record_option_fill(
        self,
        order: CustomerOrder,
        trade_price: float,
        theo_at_trade: float,
        contract_multiplier: float,
    ) -> None:
        """Book a customer fill: update cash and cumulative spread capture.

        Args:
            order: The customer order that filled (its side is the
                *customer's* side; the market maker takes the other side).
            trade_price: Price the trade executed at (the MM's bid or ask).
            theo_at_trade: Theoretical fair value at the moment of the trade.
            contract_multiplier: Shares per contract.
        """
        size = order.size
        if order.side == Side.BUY:
            # Customer buys -> MM sells at the ask, receives cash.
            self.cash += trade_price * size * contract_multiplier
            edge_per_share = trade_price - theo_at_trade
        else:
            # Customer sells -> MM buys at the bid, pays cash.
            self.cash -= trade_price * size * contract_multiplier
            edge_per_share = theo_at_trade - trade_price

        self.cumulative_spread_capture += edge_per_share * size * contract_multiplier

    def record_hedge_trade(self, hedge: HedgeTrade) -> None:
        """Book a hedge trade in the underlying: cash flow plus its cost."""
        self.cash -= hedge.shares * hedge.price
        self.cash -= hedge.cost
        self.cumulative_hedging_cost += hedge.cost

    def mark_to_market(
        self,
        step_index: int,
        option_positions: dict[OptionContract, int],
        theo_by_contract: dict[OptionContract, float],
        underlying_position: float,
        spot: float,
        contract_multiplier: float,
    ) -> PnLSnapshot:
        """Mark the book and record this tick's P&L decomposition.

        Args:
            step_index: Current simulation tick.
            option_positions: Current signed position (contracts) per contract.
            theo_by_contract: Current theoretical price per contract.
            underlying_position: Current hedge position in the underlying (shares).
            spot: Current underlying price.
            contract_multiplier: Shares per contract.

        Returns:
            The ``PnLSnapshot`` for this tick (also appended to ``history``).
        """
        options_value = sum(
            qty * theo_by_contract[contract] * contract_multiplier
            for contract, qty in option_positions.items()
            if qty != 0
        )
        underlying_value = underlying_position * spot
        portfolio_value = self.cash + options_value + underlying_value
        total_pnl = portfolio_value  # started from zero cash/position, so this *is* total P&L

        residual = total_pnl - self.cumulative_spread_capture + self.cumulative_hedging_cost

        snapshot = PnLSnapshot(
            step_index=step_index,
            cash=self.cash,
            portfolio_value=portfolio_value,
            total_pnl=total_pnl,
            cumulative_spread_capture=self.cumulative_spread_capture,
            cumulative_hedging_cost=self.cumulative_hedging_cost,
            residual_risk_pnl=residual,
        )
        self.history.append(snapshot)
        return snapshot
