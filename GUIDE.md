# simplex allocator – starter guide

## why this project exists

- we keep a crypto budget (usd) but want systematic decisions instead of gut feel.
- every allocation should respect three guardrails: total budget, per-asset cap, and an overall risk budget derived from 24 h price range.
- we want resilience: if binance is down, we still need to enter values by hand.
- we must log every rebalance for audits and coursework.

## what happens on each rebalance

1. **snapshot** – the “assets info table” is prefilled from binance (price, expected return = 24 h change %, risk = high–low span / price). you can override any cell; only edited values replace the live feed.
2. **lp build** – `trader.py` turns the snapshot into an lp: maximize `Σ wᵢ·rᵢ` subject to budget, per-asset caps, and risk budget. weights are fractions of total capital.
3. **simplex solve** – `simplex.py` runs the primal simplex method and returns optimal weights. dollars = `budget * weight`, quantity = `allocation / price`.
4. **state update + trades** – `portfolio_state.py` compares the new target quantities to the previous holdings, using current prices to compute buy/sell deltas. it writes `portfolio_state.json` so the next iteration knows what we currently own.
5. **ledger write** – the result is appended to `trades.csv` with timestamp, iteration, weight, allocation, quantity, price, and expected return.
6. **ui update** – streamlit shows:
   - metrics (objective, cash reserve, number of assets),
   - the latest allocation table,
   - “allocation rationale” (objective contributions + constraint checks),
   - charts (weights, allocations, objective over time),
   - “rebalance log” sourced directly from the csv.

because charts and logs read from the ledger, all prior allocations remain visible even after restarting the app. pressing **clear history** truncates `trades.csv` and resets the visuals.

## how to run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### dashboard tips

- **symbols** – select the trading pairs you care about; defaults to btc, eth, bnb, sol.
- **budget / caps / risk** – sliders define the lp constraints. keep risk budget ≤ 1 so the span constraint is meaningful.
- **assets info table** – edit expected returns or prices as needed. leaving a cell blank means “use binance value”. if binance is down, rows with both price and expected return act as fallback.
- **auto-refresh** – optional polling (10–600 s) to rerun without clicking.
- **manual overrides** – changing `expected_return` automatically rescales `risk` (unless you also edit risk) so the lp stays consistent.
- **warnings** – if binance fails, you’ll see a yellow banner and the run will use manual data only.

## files that matter

| file | role |
| --- | --- |
| `app.py` | streamlit entry point, manual override editor, ledger-backed charts, allocation rationale. |
| `market.py` | binance client + manual-data parser, computes risk proxy. |
| `simplex.py` | primal simplex solver (`max c^T x` with `Ax ≤ b`). |
| `trader.py` | builds lp from snapshots, maps solution to trade decisions. |
| `ledger.py` | append-only csv writer; clear history button recreates the file. |
| `portfolio_state.py` | load/save holdings and compute trade deltas for the “trade instructions” table. |
| `tests/test_simplex.py` | regression tests for solver and risk-budget logic. |
| `trades.csv` | persistent history consumed by charts/logs. |

## validating for the assignment

- run `pytest -q` to show the solver works on small deterministic problems (requirement: “test with a small number of variables”).
- demonstrate manual vs. solver vs. library by showing a hand-solved 2-asset lp, the same run in streamlit, and an external solver (e.g., `scipy.optimize.linprog`).
- use the report (`REPORT.md`) for background and references; this guide is the quick-start for teammates.

## best way to explain during defence

1. open the dashboard, show the assets info table prefilled from binance.
2. tweak an expected return to illustrate manual overrides; note how risk auto-updates.
3. hit “rebalance now” and walk through:
   - the toast with local time,
   - the allocation table,
   - the “trade instructions” table (highlighting buy/sell deltas),
   - the rationale tables (objective + constraints),
   - the charts,
   - the rebalance log row that just appeared (with iteration number).
4. open `trades.csv` to prove history persists.
5. mention tests + literature references for completeness.

after walking through these steps, your classmate should be able to operate the project and explain every requirement (problem context, model formulation, implementation, results, validation, and future work).
