"""Tick-by-tick simulation orchestrator.

Wires every module together into a single simulated trading session:

  1. advance the underlying, refresh theo/Greeks for the whole chain
  2. check hard risk limits against the *start-of-tick* book and gate
     what new risk this tick's flow is allowed to add
  3. build quotes (inventory skew/widening, vol-mismatch widening,
     toxicity widening)
  4. generate customer orders and match them against those quotes
  5. hedge the resulting delta exposure (forced immediately on a delta
     limit breach, otherwise on the hedger's own threshold/cadence)
  6. mark the book to market and record the P&L decomposition

This module intentionally contains the "business logic" of how these
pieces interact each tick; each imported module stays independently
testable in isolation (see ``tests/``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from mm_sim.hedging import DeltaHedger, HedgeConfig
from mm_sim.market_data import (
    MINUTES_PER_TRADING_DAY,
    OptionContract,
    build_option_chain,
    build_vol_regime,
    minutes_to_years,
    rolling_realized_vol,
    simulate_gbm_path,
)
from mm_sim.order_flow import (
    MarkoutConfig,
    OrderFlowConfig,
    OrderFlowGenerator,
    Side,
    ToxicityTracker,
)
from mm_sim.pnl_attribution import PnLAttributor
from mm_sim.pricing import Greeks, price_and_greeks
from mm_sim.quoting_engine import QuotingConfig, QuotingEngine
from mm_sim.risk_limits import RiskLimitMonitor, RiskLimitsConfig


@dataclass
class SimulationConfig:
    """All parameters needed to run one simulated trading session."""

    s0: float = 100.0
    mu: float = 0.0
    base_vol: float = 0.20
    risk_free_rate: float = 0.04

    n_strikes: int = 9
    strike_spacing: float = 5.0
    dte_days_at_start: float = 30.0
    contract_multiplier: float = 100.0

    session_minutes: int = MINUTES_PER_TRADING_DAY
    realized_vol_window: int = 30

    vol_spike_start_frac: float = 0.40
    vol_spike_end_frac: float = 0.55
    vol_spike_multiplier: float = 1.8

    seed: int | None = 7

    quoting: QuotingConfig = field(default_factory=QuotingConfig)
    order_flow: OrderFlowConfig = field(default_factory=OrderFlowConfig)
    hedging: HedgeConfig = field(default_factory=HedgeConfig)
    risk_limits: RiskLimitsConfig = field(default_factory=RiskLimitsConfig)
    markout: MarkoutConfig = field(default_factory=MarkoutConfig)


@dataclass
class SimulationResult:
    """Everything a caller (or the notebook) needs to analyze a run."""

    ticks: pd.DataFrame
    fills: pd.DataFrame
    hedges: pd.DataFrame
    breaches: pd.DataFrame
    chain: list[OptionContract]
    config: SimulationConfig


def _aggregate_book_greeks(
    positions: dict[OptionContract, int],
    greeks_by_contract: dict[OptionContract, Greeks],
    spot: float,
    contract_multiplier: float,
) -> tuple[float, float, float]:
    """Aggregate net delta (shares), dollar gamma, and dollar vega across the book."""
    net_delta = 0.0
    net_gamma_dollars = 0.0
    net_vega_dollars = 0.0
    for contract, qty in positions.items():
        if qty == 0:
            continue
        g = greeks_by_contract[contract]
        net_delta += qty * g.delta * contract_multiplier
        net_gamma_dollars += qty * 0.5 * g.gamma * (spot * 0.01) ** 2 * contract_multiplier
        net_vega_dollars += qty * g.vega * contract_multiplier
    return net_delta, net_gamma_dollars, net_vega_dollars


class MarketMakingSimulation:
    """Runs a single simulated market-making session end to end."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()

    def run(self) -> SimulationResult:
        cfg = self.config
        dt = minutes_to_years(1)

        vol_regime = build_vol_regime(
            cfg.session_minutes, cfg.base_vol, cfg.vol_spike_start_frac, cfg.vol_spike_end_frac, cfg.vol_spike_multiplier
        )
        price_path = simulate_gbm_path(cfg.s0, cfg.mu, vol_regime, cfg.session_minutes, dt, seed=cfg.seed)
        realized_vol_path = rolling_realized_vol(price_path, cfg.realized_vol_window, dt)

        chain = build_option_chain(price_path[0], cfg.n_strikes, cfg.strike_spacing)
        atm_call = min(
            (c for c in chain if c.option_type.value == "call"),
            key=lambda c: abs(c.strike - price_path[0]),
        )

        positions: dict[OptionContract, int] = {c: 0 for c in chain}

        engine = QuotingEngine(cfg.quoting)
        flow = OrderFlowGenerator(cfg.order_flow, seed=cfg.seed)
        hedger = DeltaHedger(cfg.hedging)
        risk_monitor = RiskLimitMonitor(cfg.risk_limits)
        pnl = PnLAttributor()
        toxicity_tracker = ToxicityTracker(cfg.markout)

        initial_T_years = cfg.dte_days_at_start / 365.0
        min_T_years = 1.0 / 365.0 / 24.0  # floor so pricing never sees T <= 0 within a single session

        tick_rows: list[dict] = []
        fill_rows: list[dict] = []
        hedge_rows: list[dict] = []
        breach_rows: list[dict] = []

        for step in range(cfg.session_minutes):
            spot = float(price_path[step])
            T = max(initial_T_years - minutes_to_years(step), min_T_years)
            realized_vol = float(realized_vol_path[step])
            implied_vol = cfg.base_vol

            theo_by_contract: dict[OptionContract, float] = {}
            greeks_by_contract: dict[OptionContract, Greeks] = {}
            for c in chain:
                theo, greeks = price_and_greeks(spot, c.strike, T, cfg.risk_free_rate, implied_vol, c.option_type)
                theo_by_contract[c] = theo
                greeks_by_contract[c] = greeks

            toxicity_score = toxicity_tracker.update(step, spot)

            # --- Pre-trade risk check: gates what flow this tick is allowed to add. ---
            locked_contracts: set[OptionContract] = set()
            if not risk_monitor.halted:
                for c in chain:
                    breach = risk_monitor.check_position(step, c.symbol, positions[c])
                    if breach is not None:
                        breach_rows.append(vars(breach))
                        if breach.action == "block_position_increasing_orders":
                            locked_contracts.add(c)

                pre_delta_options, pre_gamma, pre_vega = _aggregate_book_greeks(positions, greeks_by_contract, spot, cfg.contract_multiplier)
                pre_delta_total = pre_delta_options + hedger.underlying_position
                portfolio_breaches = risk_monitor.check_portfolio(step, pre_delta_total, pre_gamma, pre_vega)
                for b in portfolio_breaches:
                    breach_rows.append(vars(b))
                portfolio_lockout = any(b.action == "lockout_new_risk" for b in portfolio_breaches)
                force_hedge = any(b.action == "force_hedge" for b in portfolio_breaches)
            else:
                portfolio_lockout = True
                force_hedge = False

            # --- Quote every contract using the pre-trade book. ---
            quotes = {
                c: engine.compute_quote(
                    theo_by_contract[c],
                    greeks_by_contract[c],
                    positions[c],
                    spot,
                    cfg.contract_multiplier,
                    realized_vol,
                    implied_vol,
                    toxicity_score,
                )
                for c in chain
            }

            # --- Generate and attempt to fill this tick's customer orders. ---
            n_fills = 0
            if not risk_monitor.halted:
                lookahead = cfg.order_flow.informed_lookahead_steps
                future_idx = min(step + lookahead, cfg.session_minutes)
                future_spot = float(price_path[future_idx])
                orders = flow.generate_orders(chain, spot, future_spot, cfg.strike_spacing)

                for order in orders:
                    contract = order.contract
                    mm_position_delta = -order.size if order.side == Side.BUY else order.size
                    prospective = positions[contract] + mm_position_delta
                    increases_risk = abs(prospective) > abs(positions[contract])

                    if increases_risk and (contract in locked_contracts or portfolio_lockout):
                        continue

                    quote = quotes[contract]
                    if not flow.decide_fill(quote):
                        continue

                    trade_price = quote.ask if order.side == Side.BUY else quote.bid
                    positions[contract] = prospective
                    pnl.record_option_fill(order, trade_price, theo_by_contract[contract], cfg.contract_multiplier)

                    mm_delta_shares_from_fill = mm_position_delta * greeks_by_contract[contract].delta * cfg.contract_multiplier
                    toxicity_tracker.record_fill(step, mm_delta_shares_from_fill, spot)

                    n_fills += 1
                    fill_rows.append(
                        {
                            "step": step,
                            "contract": contract.symbol,
                            "customer_side": order.side.value,
                            "size": order.size,
                            "trade_price": trade_price,
                            "theo": theo_by_contract[contract],
                        }
                    )

            # --- Hedge the resulting exposure. ---
            post_delta, post_gamma, post_vega = _aggregate_book_greeks(positions, greeks_by_contract, spot, cfg.contract_multiplier)
            if force_hedge:
                hedge_trade = hedger.force_hedge(step, post_delta, spot)
            else:
                hedge_trade = hedger.maybe_hedge(step, post_delta, spot)
            if hedge_trade is not None:
                pnl.record_hedge_trade(hedge_trade)
                hedge_rows.append(vars(hedge_trade))

            # --- Mark to market and record P&L attribution. ---
            snapshot = pnl.mark_to_market(step, positions, theo_by_contract, hedger.underlying_position, spot, cfg.contract_multiplier)

            atm_quote = quotes[atm_call]
            tick_rows.append(
                {
                    "step": step,
                    "spot": spot,
                    "time_to_expiry_years": T,
                    "realized_vol": realized_vol,
                    "implied_vol": implied_vol,
                    "toxicity_score": toxicity_score,
                    "net_delta_options": post_delta,
                    "net_delta_total": post_delta + hedger.underlying_position,
                    "net_gamma_dollars": post_gamma,
                    "net_vega_dollars": post_vega,
                    "underlying_position": hedger.underlying_position,
                    "n_fills": n_fills,
                    "halted": risk_monitor.halted,
                    "atm_theo": atm_quote.theo,
                    "atm_bid": atm_quote.bid,
                    "atm_ask": atm_quote.ask,
                    "atm_half_spread": atm_quote.half_spread,
                    "cash": snapshot.cash,
                    "portfolio_value": snapshot.portfolio_value,
                    "total_pnl": snapshot.total_pnl,
                    "cumulative_spread_capture": snapshot.cumulative_spread_capture,
                    "cumulative_hedging_cost": snapshot.cumulative_hedging_cost,
                    "residual_risk_pnl": snapshot.residual_risk_pnl,
                }
            )

        return SimulationResult(
            ticks=pd.DataFrame(tick_rows),
            fills=pd.DataFrame(fill_rows),
            hedges=pd.DataFrame(hedge_rows),
            breaches=pd.DataFrame(breach_rows),
            chain=chain,
            config=cfg,
        )
