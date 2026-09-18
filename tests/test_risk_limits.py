from mm_sim.risk_limits import RiskLimitMonitor, RiskLimitsConfig


def _config(**overrides):
    defaults = dict(
        max_position_per_contract=100,
        max_net_delta_shares=200.0,
        max_net_gamma_dollars=300.0,
        max_net_vega_dollars=500.0,
        hard_stop_multiplier=2.0,
    )
    defaults.update(overrides)
    return RiskLimitsConfig(**defaults)


def test_position_within_limit_is_not_a_breach():
    monitor = RiskLimitMonitor(_config())
    assert monitor.check_position(step_index=1, contract_symbol="100C", position_contracts=50) is None
    assert not monitor.halted


def test_position_breach_blocks_increasing_orders():
    monitor = RiskLimitMonitor(_config())
    breach = monitor.check_position(step_index=1, contract_symbol="100C", position_contracts=150)
    assert breach is not None
    assert breach.action == "block_position_increasing_orders"
    assert not monitor.halted


def test_extreme_position_breach_halts_trading():
    monitor = RiskLimitMonitor(_config())
    breach = monitor.check_position(step_index=1, contract_symbol="100C", position_contracts=250)
    assert breach.action == "halt_trading"
    assert monitor.halted


def test_delta_breach_forces_hedge():
    monitor = RiskLimitMonitor(_config())
    breaches = monitor.check_portfolio(step_index=1, net_delta_shares=250.0, net_gamma_dollars=0.0, net_vega_dollars=0.0)
    assert len(breaches) == 1
    assert breaches[0].metric == "net_delta"
    assert breaches[0].action == "force_hedge"


def test_gamma_and_vega_breach_lockout_new_risk():
    monitor = RiskLimitMonitor(_config())
    breaches = monitor.check_portfolio(step_index=1, net_delta_shares=0.0, net_gamma_dollars=350.0, net_vega_dollars=600.0)
    actions = {b.metric: b.action for b in breaches}
    assert actions["net_gamma"] == "lockout_new_risk"
    assert actions["net_vega"] == "lockout_new_risk"


def test_extreme_delta_breach_halts_trading():
    monitor = RiskLimitMonitor(_config())
    breaches = monitor.check_portfolio(step_index=1, net_delta_shares=500.0, net_gamma_dollars=0.0, net_vega_dollars=0.0)
    assert breaches[0].action == "halt_trading"
    assert monitor.halted


def test_no_breach_within_all_limits():
    monitor = RiskLimitMonitor(_config())
    breaches = monitor.check_portfolio(step_index=1, net_delta_shares=10.0, net_gamma_dollars=10.0, net_vega_dollars=10.0)
    assert breaches == []
    assert not monitor.halted


def test_breaches_are_logged_on_the_monitor():
    monitor = RiskLimitMonitor(_config())
    monitor.check_position(step_index=1, contract_symbol="100C", position_contracts=150)
    monitor.check_portfolio(step_index=2, net_delta_shares=250.0, net_gamma_dollars=0.0, net_vega_dollars=0.0)
    assert len(monitor.breaches) == 2
