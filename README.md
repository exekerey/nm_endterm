# Simplex Crypto Allocator

This repository reduces adaptive LP research down to its essentials: a clean simplex solver, a Binance/CSV data source, and a Streamlit dashboard that rebalances a portfolio while logging trades. No complex learners or orchestration layers—just the math needed to turn price snapshots into allocations.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the Streamlit UI and choose symbols, budgeting, and refresh cadence. Upload a CSV/JSON snapshot to override Binance, toggle auto-refresh for continuous rebalances, and watch real-time weight/return charts update as trades are appended to `trades.csv`.

## Project Structure

- `simplex.py` – pure-Python implementation of the primal simplex method for `max c^T x` with `A x <= b`.
- `market.py` – fetches Binance 24h ticker data or reads manual CSV/JSON snapshots into `AssetSnapshot` records.
- `trader.py` – converts snapshots into an LP (budget, max allocation, risk cap) and turns the simplex solution into trade instructions.
- `ledger.py` – minimal CSV writer that stores timestamped trades for later analysis.
- `portfolio_state.py` – persists per-symbol quantities and computes buy/sell deltas between rebalances.
- `app.py` – Streamlit dashboard with auto-refresh, live charts, and ledger integration.
- `tests/test_simplex.py` – regression tests for the simplex solver and allocation logic.

## Manual Snapshots

Provide a CSV with columns `symbol,price,expected_return,risk` or a JSON list of objects with the same keys. Example:

```csv
symbol,price,expected_return,risk
BTCUSDT,108000,0.015,0.06
ETHUSDT,3800,0.02,0.08
```

The engine treats `expected_return` as the linear objective coefficient and enforces `risk_budget` by summing `risk * weight`.

## Streamlit Dashboard

- **Symbols & Budget** – pick trading pairs, set portfolio budget, per-asset caps, and risk budget (weights·risk ≤ limit).
- **Data Sources** – pull Binance spot data or upload manual CSV/JSON files for custom expected returns.
- **Auto Refresh** – enable polling (10–600 s) to keep allocations and charts in sync without manual clicks.
- **Real-time Charts** – Altair plots visualize weights, dollar allocations, and objective value across runs.
- **Trade instructions** – a holdings state file converts the latest target weights into buy/sell deltas using current prices.
- **Ledger** – every rebalance appends to `trades.csv` so you can audit or backtest decisions elsewhere.

## Development

- Format: standard library + `numpy`, `pandas`, and Streamlit; the simplex implementation lives in `simplex.py`.
- Tests: run `pytest` after touching the solver or trading logic.
- Logging: inspect `trades.csv` to audit what the bot executed—no databases involved.
