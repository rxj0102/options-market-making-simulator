from mm_sim.pricing import Greeks
from mm_sim.quoting_engine import QuotingConfig, QuotingEngine

THEO = 2.50
GREEKS = Greeks(delta=0.5, gamma=0.07, vega=0.11, theta=-0.04, rho=0.01)
SPOT = 100.0
MULTIPLIER = 100.0


def _quote(engine, position=0, realized_vol=0.20, implied_vol=0.20, toxicity=0.0):
    return engine.compute_quote(
        theo=THEO,
        greeks=GREEKS,
        position_contracts=position,
        spot=SPOT,
        contract_multiplier=MULTIPLIER,
        realized_vol=realized_vol,
        implied_vol=implied_vol,
        toxicity_score=toxicity,
    )


def test_flat_position_has_no_skew_and_brackets_theo():
    engine = QuotingEngine(QuotingConfig())
    q = _quote(engine, position=0)
    assert q.skew == 0.0
    assert q.bid < THEO < q.ask


def test_long_position_skews_mid_down():
    engine = QuotingEngine(QuotingConfig())
    flat = _quote(engine, position=0)
    long = _quote(engine, position=50)
    assert long.mid < flat.mid
    assert long.half_spread > flat.half_spread


def test_short_position_skews_mid_up():
    engine = QuotingEngine(QuotingConfig())
    flat = _quote(engine, position=0)
    short = _quote(engine, position=-50)
    assert short.mid > flat.mid


def test_long_and_short_of_equal_size_are_symmetric():
    engine = QuotingEngine(QuotingConfig())
    long = _quote(engine, position=50)
    short = _quote(engine, position=-50)
    assert abs((long.mid - THEO) + (short.mid - THEO)) < 1e-9
    assert abs(long.half_spread - short.half_spread) < 1e-9


def test_vol_mismatch_widens_spread():
    engine = QuotingEngine(QuotingConfig())
    matched = _quote(engine, realized_vol=0.20, implied_vol=0.20)
    mismatched = _quote(engine, realized_vol=0.45, implied_vol=0.20)
    assert mismatched.half_spread > matched.half_spread


def test_toxicity_widens_spread():
    engine = QuotingEngine(QuotingConfig())
    clean = _quote(engine, toxicity=0.0)
    toxic = _quote(engine, toxicity=0.9)
    assert toxic.half_spread > clean.half_spread


def test_half_spread_never_below_configured_floor():
    engine = QuotingEngine(QuotingConfig(min_half_spread=0.05, gamma_widen_coeff=0.0))
    q = engine.compute_quote(
        theo=THEO,
        greeks=Greeks(delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=0.0),
        position_contracts=0,
        spot=SPOT,
        contract_multiplier=MULTIPLIER,
        realized_vol=0.20,
        implied_vol=0.20,
        toxicity_score=0.0,
    )
    assert q.half_spread == 0.05


def test_half_spread_respects_configured_cap():
    engine = QuotingEngine(QuotingConfig(max_half_spread=0.10))
    q = _quote(engine, position=500, toxicity=1.0, realized_vol=0.9, implied_vol=0.2)
    assert q.half_spread == 0.10


def test_bid_never_crosses_zero():
    engine = QuotingEngine(QuotingConfig(max_half_spread=100.0))
    q = _quote(engine, position=100_000, toxicity=1.0)
    assert q.bid >= 0.01
    assert q.ask > q.bid
