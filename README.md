Simplex iterative crypto allocator
==================================

Authors: Kassiyet Bolat & Danial Baitakov (IT-2307)  
Code: https://github.com/exekerey/nm_endterm

1. Context
----------

Digital asset desks face the same pressures as manufacturing or logistics teams: finite budget, volatile demand, and the need for auditable decisions. Because crypto markets run 24/7, manual rebalancing quickly becomes slow and error prone. This project solves a linear program (LP) at every market snapshot so that allocations remain systematic, explainable, and easy to defend.

2. Literature review
--------------------

- H. Markowitz (1952) formalized the return–risk trade-off as an optimization problem.  
- G. B. Dantzig (1947) introduced the simplex algorithm, making linear optimization practical.  
- N. Devanur & K. Jain (2012) and S. Balseiro et al. (2021) analyzed online/repeated LPs that adapt to streaming inputs.  
- W. Powell (2019) described rolling-horizon planning, where models are re-solved whenever new information arrives.  
- A. Dalfi (2022) showed that simple LPs with volatility proxies work well in crypto markets.  
- D. Șerban & C. Dedu (2025) studied robust crypto portfolios using entropy-based risk controls.  

These works motivate a lightweight, repeatedly solved LP that keeps all constraints visible for audits and coursework.

3. Problem overview
-------------------

We allocate a fixed USD budget across a set of crypto pairs (BTC, ETH, BNB, SOL, etc.). The allocator must:

- keep total weights ≤ 1 (budget conservation)  
- cap each weight by a user slider (diversification)  
- limit weighted risk exposure `Σ wᵢ · riskᵢ ≤ β` (risk budget)  
- remain usable when Binance is offline by allowing manual overrides  
- log every rebalance with timestamps, prices, expected returns, and weights

4. Model formulation
--------------------

Objective: maximize `Σ wᵢ · rᵢ`

Constraints:

1. `Σ wᵢ ≤ 1` (budget)  
2. `wᵢ ≤ α` for each asset (max allocation slider)  
3. `Σ wᵢ · riskᵢ ≤ β` (risk budget slider, 0–1)  
4. `wᵢ ≥ 0`

Definitions:

- `wᵢ`: decision variables (portfolio weights)  
- `rᵢ`: expected returns from Binance (24 h percent change / 100) or manual input  
- `riskᵢ`: 24 h span `(high − low) / lastPrice`, interpreted as percent of capital exposed to that day’s move  
- `α`: per-asset cap  
- `β`: allowed weighted average span

Weights map to dollars via `allocationᵢ = budget * wᵢ` and to quantities via `allocationᵢ / priceᵢ`. Any unused weight becomes cash.

5. Implementation details
-------------------------

### Data ingest

- `market.py` pulls `/api/v3/ticker/24hr` for each symbol, capturing price, percent change, high/low, volume, and close time.  
- The Streamlit “assets info table” pre-fills with those values; editing a cell overrides only that field. Rows with both price and expected return act as fallback snapshots if Binance fails.

### LP construction and solve

- `trader.py` builds the LP matrices (budget row, identity rows for caps, risk row).  
- `simplex.py` runs the primal simplex method `max c^T x` with `Ax ≤ b`.  
- The solver output becomes weights, dollar allocations, quantities, and cash reserve.

### Trade instructions

- `portfolio_state.py` loads `portfolio_state.json`, compares old quantities to new targets, and emits buy/sell deltas with USD notionals (`Δqty × price`).  
- After each rebalance the state file is updated so the next run knows what the portfolio currently holds. Clearing history deletes both the ledger and this state.

### Ledger and visualization

- `ledger.py` appends each decision to `trades.csv` (timestamp, iteration, symbol, weight, allocation, quantity, price, expected return USD).  
- Streamlit reloads the CSV on every refresh to draw weight/ allocation charts, objective-over-time, and the rebalance log. Because the visuals rely on the ledger, history persists across restarts until you press “Clear history.”  
- Rationale tables show objective contributions (weight, allocation, expected return USD) and constraint evaluations (left-hand side, right-hand side, slack) so the math stays transparent.

### Workflow summary

1. Fetch Binance data (or rely on manual fallback).  
2. Formulate and solve the LP via simplex.  
3. Convert the target allocation into trade instructions relative to current holdings.  
4. Append results to the ledger and update charts/logs.  
5. Repeat manually or via auto-refresh.

6. Running and testing
----------------------

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Tests:

```
pytest -q
```

`tests/test_simplex.py` includes a 2-variable LP with a known optimum and a risk-budget case where `Σ w·risk` hits the configured limit.

7. Future improvements
----------------------

- replace the simple span proxy with factor-based risk while keeping the model linear  
- surface simplex dual variables to explain the “price” of each constraint  
- add scenario testing (e.g., +5% / −5% shocks) to compare runs side by side  
- schedule automatic cross-checks against `scipy.optimize.linprog` or other solvers

References
----------

1. H. Markowitz, “Portfolio Selection,” *The Journal of Finance*, 1952.  
2. G. B. Dantzig, “Maximization of a Linear Function of Variables Subject to Linear Inequalities,” 1947.  
3. N. Devanur and K. Jain, “Online Algorithms for Linear and Convex Programming,” *Mathematics of Operations Research*, 2012.  
4. S. Balseiro, O. Besbes, G. Brown, “Learning in Repeated Linear Programs,” *Operations Research*, 2021.  
5. W. Powell, *Sequential Decision Analytics and Modeling with Python*, 2019.  
6. A. Dalfi, “Evaluation of Portfolio Optimization Methods on Cryptocurrencies,” Dalarna University thesis, 2022.  
7. D. Șerban and C. Dedu, “Robust Portfolio Optimization in Crypto Markets,” *Risks*, 2025.

