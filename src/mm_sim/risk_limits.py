"""Hard risk limits and stop-out rules.

Quoting and hedging are the *soft* controls on risk -- wider spreads and
periodic delta flattening. This module is the backstop: hard caps on
position size and portfolio Greeks that, when breached, force a specific
corrective action rather than a warning. Breaches escalate in severity:

  - a limit breach triggers a corrective action (reject risk-increasing
    orders in that contract, force an immediate hedge, or lock out new
    risk portfolio-wide)
  - a breach beyond ``hard_stop_multiplier`` times the limit triggers a
    full stop-out: trading halts entirely until the book is back under
    control.

Every breach is logged with the metric, its value, the limit, and the
action taken, so the notebook can show exactly when and why the simulated
desk's risk controls kicked in.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimitsConfig:
    """Hard caps on position size and portfolio Greek exposure."""

    max_position_per_contract: int = 200
    """Max absolute position (contracts) in any single listed contract."""

    max_net_delta_shares: float = 300.0
    """Max absolute net delta for the whole book, in shares-equivalent."""

    max_net_gamma_dollars: float = 250.0
    """Max absolute net dollar gamma (book P&L swing per 1% underlying move)."""

    max_net_vega_dollars: float = 700.0
    """Max absolute net dollar vega (book P&L swing per 1 vol point)."""

    hard_stop_multiplier: float = 2.0
    """A breach beyond limit * this multiplier triggers a full trading halt."""


@dataclass(frozen=True)
class RiskBreach:
    """A single risk limit breach and the action taken in response."""

    step_index: int
    metric: str
    value: float
    limit: float
    action: str


class RiskLimitMonitor:
    """Checks position and portfolio Greek exposure against hard limits."""

    def __init__(self, config: RiskLimitsConfig | None = None) -> None:
        self.config = config or RiskLimitsConfig()
        self.breaches: list[RiskBreach] = []
        self.halted: bool = False

    def _classify(self, value: float, limit: float, soft_action: str) -> str | None:
        """Return the action for a breach, or None if within limits."""
        if abs(value) > limit * self.config.hard_stop_multiplier:
            return "halt_trading"
        if abs(value) > limit:
            return soft_action
        return None

    def check_position(self, step_index: int, contract_symbol: str, position_contracts: float) -> RiskBreach | None:
        """Check a single contract's position against the per-contract limit.

        Returns a breach with action ``"block_position_increasing_orders"``
        (only orders that reduce the position are still accepted) or
        ``"halt_trading"`` for an extreme breach.
        """
        action = self._classify(position_contracts, self.config.max_position_per_contract, "block_position_increasing_orders")
        if action is None:
            return None
        if action == "halt_trading":
            self.halted = True
        breach = RiskBreach(step_index, f"position:{contract_symbol}", position_contracts, self.config.max_position_per_contract, action)
        self.breaches.append(breach)
        return breach

    def check_portfolio(
        self,
        step_index: int,
        net_delta_shares: float,
        net_gamma_dollars: float,
        net_vega_dollars: float,
    ) -> list[RiskBreach]:
        """Check aggregate book Greeks against portfolio-level limits.

        Delta breaches escalate to ``"force_hedge"`` (bypass the hedger's
        normal threshold and flatten immediately); gamma/vega breaches
        escalate to ``"lockout_new_risk"`` (reject any order that would
        increase that exposure). Extreme breaches of any kind escalate to
        ``"halt_trading"``.
        """
        cfg = self.config
        checks = [
            ("net_delta", net_delta_shares, cfg.max_net_delta_shares, "force_hedge"),
            ("net_gamma", net_gamma_dollars, cfg.max_net_gamma_dollars, "lockout_new_risk"),
            ("net_vega", net_vega_dollars, cfg.max_net_vega_dollars, "lockout_new_risk"),
        ]

        breaches: list[RiskBreach] = []
        for metric, value, limit, soft_action in checks:
            action = self._classify(value, limit, soft_action)
            if action is None:
                continue
            if action == "halt_trading":
                self.halted = True
            breaches.append(RiskBreach(step_index, metric, value, limit, action))

        self.breaches.extend(breaches)
        return breaches
