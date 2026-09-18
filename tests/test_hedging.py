from mm_sim.hedging import DeltaHedger, HedgeConfig


def test_no_hedge_when_under_threshold_and_within_cadence():
    hedger = DeltaHedger(HedgeConfig(delta_threshold=100.0, max_steps_between_hedges=20))
    trade = hedger.maybe_hedge(step_index=1, options_delta_shares=10.0, spot=100.0)
    assert trade is None
    assert hedger.underlying_position == 0.0


def test_threshold_breach_triggers_flattening_hedge():
    hedger = DeltaHedger(HedgeConfig(delta_threshold=50.0, max_steps_between_hedges=None))
    trade = hedger.maybe_hedge(step_index=5, options_delta_shares=120.0, spot=100.0)
    assert trade is not None
    assert trade.reason == "threshold"
    assert trade.shares == -120.0
    assert hedger.underlying_position == -120.0
    assert hedger.net_delta(options_delta_shares=120.0) == 0.0


def test_time_based_hedge_fires_after_max_steps_even_under_threshold():
    hedger = DeltaHedger(HedgeConfig(delta_threshold=1000.0, max_steps_between_hedges=3))
    assert hedger.maybe_hedge(1, 10.0, 100.0) is None
    assert hedger.maybe_hedge(2, 10.0, 100.0) is None
    trade = hedger.maybe_hedge(3, 10.0, 100.0)
    assert trade is not None
    assert trade.reason == "time"


def test_hedge_cost_scales_with_shares_traded():
    hedger = DeltaHedger(HedgeConfig(delta_threshold=10.0, cost_per_share=0.02))
    trade = hedger.maybe_hedge(step_index=1, options_delta_shares=100.0, spot=50.0)
    assert trade.cost == 100.0 * 0.02


def test_force_hedge_ignores_threshold():
    hedger = DeltaHedger(HedgeConfig(delta_threshold=1_000_000.0, max_steps_between_hedges=None))
    trade = hedger.force_hedge(step_index=1, options_delta_shares=5.0, spot=100.0)
    assert trade is not None
    assert trade.reason == "risk_limit_breach"
    assert hedger.underlying_position == -5.0


def test_repeated_hedging_keeps_net_delta_near_zero():
    hedger = DeltaHedger(HedgeConfig(delta_threshold=25.0, max_steps_between_hedges=None))
    options_delta = 0.0
    for step, bump in enumerate([40.0, -30.0, 60.0, -55.0]):
        options_delta += bump
        trade = hedger.maybe_hedge(step, options_delta, spot=100.0)
        if trade is not None:
            assert abs(hedger.net_delta(options_delta)) < 1e-9
