# options-market-making-simulator

A simulated options market-making engine: pricing, quoting, customer order flow, delta hedging, hard risk limits, and P&L attribution, wired together into one tick-by-tick trading session.

## What market making is

A market maker continuously quotes both a price it will buy at (the bid) and a price it will sell at (the ask), earning the spread between them in exchange for standing ready to trade with anyone who wants to. The job isn't computing a fair value — that's a formula — it's deciding, in real time, how much edge to demand for the risk a given quote exposes the desk to, and how aggressively to lay that risk off (via hedging) before it turns into a P&L problem.

## What this simulator does

It runs one simulated trading day (390 one-minute ticks) of a market maker quoting a single-expiry options chain (9 strikes of calls and puts) against a simulated underlying price path. Each tick:

1. **Prices the chain** with Black-Scholes, using a fixed implied-vol assumption while the underlying's *realized* vol is allowed to drift away from it (including a scripted mid-session vol regime shift) — a deliberate, visible source of model risk.
2. **Builds a two-sided quote** around theo, where the half-spread is a sum of named components (a base edge floor, inventory skew/widening, gamma/time-to-expiry widening, realized-vs-implied vol mismatch widening, and order-flow toxicity widening) and the whole market skews against the desk's current position — not just widens.
3. **Simulates customer order flow**: Poisson arrivals, mostly random, with a tunable fraction of "informed" flow biased toward the underlying's near-term move; wider quotes reduce fill probability, creating a real edge-vs-flow tradeoff.
4. **Delta-hedges** the resulting exposure in the underlying, on a configurable threshold/cadence, so the cost of staying flat is explicit and separable from the risk of not staying flat.
5. **Enforces hard risk limits** on position size and portfolio delta/gamma/vega, escalating from rejecting risk-increasing orders, to forcing an immediate hedge, to a full trading halt — every breach and the action it triggered is logged.
6. **Attributes P&L** into spread capture, hedging cost, and residual (unhedged) risk P&L, which reconcile exactly to the session's total.

See `notebooks/trading_session_walkthrough.ipynb` for a full run with commentary and visualizations.

## How to run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # installs the mm_sim package (src/ layout)
```

Run the test suite:

```bash
pytest tests/
```

Run the full walkthrough notebook:

```bash
jupyter notebook notebooks/trading_session_walkthrough.ipynb
```

Or run a session directly in Python:

```python
from mm_sim.simulation import MarketMakingSimulation, SimulationConfig

result = MarketMakingSimulation(SimulationConfig()).run()
print(result.ticks.tail())       # per-tick log: quotes, Greeks, P&L
print(result.fills)               # every customer fill
print(result.hedges)              # every hedge trade
print(result.breaches)            # every risk limit breach and the action taken
```

## Project layout

```
src/mm_sim/
    pricing.py          # Black-Scholes price + Greeks
    market_data.py      # simulated GBM underlying path + options chain
    quoting_engine.py   # bid/ask construction: inventory skew, gamma/vol/toxicity widening
    order_flow.py        # simulated customer orders + toxicity inference from markouts
    hedging.py           # threshold/time-based delta hedger
    risk_limits.py       # position and portfolio Greek limits, breach escalation
    pnl_attribution.py   # spread capture / hedging cost / residual risk P&L decomposition
    simulation.py         # tick-by-tick orchestrator
tests/                   # pytest unit + integration tests
notebooks/                # full session walkthrough with visualizations
```

## Disclaimer

This is an educational simulation, not a live trading system. The underlying price path, the options chain, and all customer order flow are synthetic. There is no real market data, no real execution, and no real money anywhere in this repository. It is intended to demonstrate market-making concepts (quoting, hedging, risk management, P&L attribution) in code, not to be used for actual trading.

## License

MIT — see `LICENSE`.
