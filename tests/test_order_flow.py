from mm_sim.market_data import OptionContract
from mm_sim.order_flow import (
    MarkoutConfig,
    OrderFlowConfig,
    OrderFlowGenerator,
    ToxicityTracker,
)
from mm_sim.pricing import OptionType
from mm_sim.quoting_engine import Quote

CHAIN = [
    OptionContract(strike=95.0, option_type=OptionType.CALL),
    OptionContract(strike=100.0, option_type=OptionType.CALL),
    OptionContract(strike=105.0, option_type=OptionType.CALL),
    OptionContract(strike=95.0, option_type=OptionType.PUT),
    OptionContract(strike=100.0, option_type=OptionType.PUT),
    OptionContract(strike=105.0, option_type=OptionType.PUT),
]


def _quote(half_spread, theo=2.50):
    return Quote(theo=theo, mid=theo, bid=theo - half_spread, ask=theo + half_spread, half_spread=half_spread, skew=0.0)


def test_generate_orders_is_reproducible_with_seed():
    gen_a = OrderFlowGenerator(OrderFlowConfig(), seed=42)
    gen_b = OrderFlowGenerator(OrderFlowConfig(), seed=42)
    orders_a = gen_a.generate_orders(CHAIN, spot=100.0, future_spot=101.0, strike_spacing=5.0)
    orders_b = gen_b.generate_orders(CHAIN, spot=100.0, future_spot=101.0, strike_spacing=5.0)
    assert [(o.contract, o.side, o.size) for o in orders_a] == [(o.contract, o.side, o.size) for o in orders_b]


def test_order_sizes_within_configured_bounds():
    cfg = OrderFlowConfig(arrival_rate_per_step=5.0, min_size=2, max_size=4)
    gen = OrderFlowGenerator(cfg, seed=1)
    orders = gen.generate_orders(CHAIN, spot=100.0, future_spot=100.0, strike_spacing=5.0)
    for order in orders:
        assert 2 <= order.size <= 4


def test_fill_probability_decreases_with_wider_spread():
    gen = OrderFlowGenerator(OrderFlowConfig())
    tight = gen.fill_probability(_quote(half_spread=0.05))
    wide = gen.fill_probability(_quote(half_spread=1.00))
    assert tight > wide


def test_fill_probability_bounded_in_zero_one():
    gen = OrderFlowGenerator(OrderFlowConfig())
    assert 0.0 <= gen.fill_probability(_quote(half_spread=0.0)) <= 1.0
    assert 0.0 <= gen.fill_probability(_quote(half_spread=10.0)) <= 1.0


def test_toxicity_tracker_neutral_before_any_resolved_fills():
    tracker = ToxicityTracker(MarkoutConfig(lookahead_steps=5))
    tracker.record_fill(step_index=0, mm_delta_shares=100.0, spot_at_fill=100.0)
    score = tracker.update(step_index=1, spot_now=101.0)
    assert score == 0.0


def test_toxicity_tracker_scores_adverse_flow_highly():
    tracker = ToxicityTracker(MarkoutConfig(lookahead_steps=1, window_size=10))
    for step in range(5):
        # MM's delta went up (bought exposure), but the underlying then fell -- adverse every time.
        tracker.record_fill(step_index=step, mm_delta_shares=100.0, spot_at_fill=100.0)
        score = tracker.update(step_index=step + 1, spot_now=99.0)
    assert score > 0.9


def test_toxicity_tracker_scores_benign_flow_as_zero():
    tracker = ToxicityTracker(MarkoutConfig(lookahead_steps=1, window_size=10))
    for step in range(5):
        # MM bought exposure and the underlying rose afterward -- favorable, not adverse.
        tracker.record_fill(step_index=step, mm_delta_shares=100.0, spot_at_fill=100.0)
        score = tracker.update(step_index=step + 1, spot_now=101.0)
    assert score == 0.0
