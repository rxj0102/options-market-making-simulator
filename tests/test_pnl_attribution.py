import pytest

from mm_sim.hedging import HedgeTrade
from mm_sim.market_data import OptionContract
from mm_sim.order_flow import CustomerOrder, Side
from mm_sim.pnl_attribution import PnLAttributor
from mm_sim.pricing import OptionType

CONTRACT = OptionContract(strike=100.0, option_type=OptionType.CALL)
MULTIPLIER = 100.0


def test_customer_buy_captures_positive_edge_when_mm_sells_above_theo():
    attributor = PnLAttributor()
    order = CustomerOrder(contract=CONTRACT, side=Side.BUY, size=2)
    attributor.record_option_fill(order, trade_price=2.60, theo_at_trade=2.50, contract_multiplier=MULTIPLIER)

    assert attributor.cumulative_spread_capture == pytest.approx((2.60 - 2.50) * 2 * MULTIPLIER)
    assert attributor.cash == pytest.approx(2.60 * 2 * MULTIPLIER)


def test_customer_sell_captures_positive_edge_when_mm_buys_below_theo():
    attributor = PnLAttributor()
    order = CustomerOrder(contract=CONTRACT, side=Side.SELL, size=3)
    attributor.record_option_fill(order, trade_price=2.40, theo_at_trade=2.50, contract_multiplier=MULTIPLIER)

    assert attributor.cumulative_spread_capture == pytest.approx((2.50 - 2.40) * 3 * MULTIPLIER)
    assert attributor.cash == pytest.approx(-2.40 * 3 * MULTIPLIER)


def test_fill_at_theo_captures_zero_edge():
    attributor = PnLAttributor()
    order = CustomerOrder(contract=CONTRACT, side=Side.BUY, size=1)
    attributor.record_option_fill(order, trade_price=2.50, theo_at_trade=2.50, contract_multiplier=MULTIPLIER)
    assert attributor.cumulative_spread_capture == pytest.approx(0.0)


def test_hedge_trade_reduces_cash_and_tracks_cost():
    attributor = PnLAttributor()
    hedge = HedgeTrade(step_index=1, shares=50.0, price=100.0, cost=1.25, reason="threshold")
    attributor.record_hedge_trade(hedge)

    assert attributor.cash == pytest.approx(-(50.0 * 100.0) - 1.25)
    assert attributor.cumulative_hedging_cost == pytest.approx(1.25)


def test_pnl_decomposition_reconciles_to_total():
    attributor = PnLAttributor()
    order = CustomerOrder(contract=CONTRACT, side=Side.SELL, size=5)
    attributor.record_option_fill(order, trade_price=2.30, theo_at_trade=2.50, contract_multiplier=MULTIPLIER)

    hedge = HedgeTrade(step_index=1, shares=-250.0, price=100.0, cost=1.25, reason="threshold")
    attributor.record_hedge_trade(hedge)

    snapshot = attributor.mark_to_market(
        step_index=1,
        option_positions={CONTRACT: 5},
        theo_by_contract={CONTRACT: 2.55},
        underlying_position=-250.0,
        spot=100.5,
        contract_multiplier=MULTIPLIER,
    )

    reconciled = snapshot.cumulative_spread_capture - snapshot.cumulative_hedging_cost + snapshot.residual_risk_pnl
    assert reconciled == pytest.approx(snapshot.total_pnl)
    assert snapshot.portfolio_value == pytest.approx(snapshot.total_pnl)


def test_flat_book_has_zero_pnl():
    attributor = PnLAttributor()
    snapshot = attributor.mark_to_market(
        step_index=0,
        option_positions={CONTRACT: 0},
        theo_by_contract={CONTRACT: 2.50},
        underlying_position=0.0,
        spot=100.0,
        contract_multiplier=MULTIPLIER,
    )
    assert snapshot.total_pnl == pytest.approx(0.0)
