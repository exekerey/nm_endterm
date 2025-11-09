# Simplex Crypto Allocator – Project Report

## 1. Context & Motivation

Digital asset desks face the same resource-allocation pressures as manufacturing or logistics teams: finite budget, volatile demand, and the need for auditable decisions. Crypto prices move around the clock, making manual rebalancing error prone. The Simplex Crypto Allocator addresses this by framing the portfolio problem as a linear program (LP) that can be solved repeatedly as market snapshots arrive. The tool targets three user personas:

- **Portfolio engineers** who need fast “what-if” runs with manual overrides when exchange data is stale.
- **Risk managers** who enforce budget, per-asset caps, and aggregate risk consumption.
- **Compliance/audit teams** who require a persistent, human-readable trade ledger.

By combining a transparent simplex implementation, reproducible data sources, and a Streamlit dashboard, the project demonstrates how core LP concepts power real-time allocation engines.

## 2. Literature Review

Foundational work on optimization-based investing dates back to Markowitz’s mean–variance portfolio theory, which formalized the trade-off between expected return and variance using quadratic and linear programs [1]. Dantzig’s introduction of the simplex algorithm showed that linear objectives with linear constraints can be solved efficiently in practice [2]. Modern LP texts such as Bertsimas and Tsitsiklis [3] and Boyd and Vandenberghe [4] emphasize modular model-building (decision variables, constraints, dual interpretation) along with sensitivity analysis—principles we mirror in the dashboard’s “Allocation Rationale”.

Recent case studies extend LP allocation to streaming finance. Borri (2019) measures cryptocurrency tail risks and shows why risk budgets are essential even in long-only strategies [5]. Liu and Tsyvinski (2021) document persistent cross-sectional return premia in crypto assets, motivating systematic optimizers over discretionary bets [6]. Ben-Tal et al. outline robust optimization techniques for uncertain returns [7], and financial data platforms (Bloomberg, Refinitiv) expose APIs for intraday rebalancing. Our approach aligns with this literature by:

1. Treating 24 h Binance changes as latest expected returns.
2. Constraining exposure through linear risk budgets (akin to factor risk limits).
3. Logging every simplex solution to a ledger for post-hoc auditing, reflecting best practices highlighted in compliance-oriented studies (e.g., Deloitte’s reports on algorithmic trading governance).

## 3. Problem Overview

### 3.1 Real-world Framing

We must decide how to allocate a fixed USD budget across a basket of crypto pairs (BTCUSDT, ETHUSDT, etc.). Decisions must respect:

- **Budget conservation:** weights sum to ≤ 1.
- **Per-asset caps:** avoid concentration in any single symbol.
- **Risk budget:** keep weighted exposure to price span (high–low range normalized by price) under a limit.
- **Operational transparency:** capture each rebalance with timestamps, prices, expected returns, and computed weights.

The system must remain usable when Binance is unavailable by allowing manual inputs and must surface the logic behind each allocation so teams can defend decisions in audits or coursework presentations.

### 3.2 Literature Connection

Our formulation can be viewed as a simplified, long-only mean–risk LP, where the “risk” proxy is derived from 24 h volatility (high–low span) instead of full covariance. This keeps the problem linear and aligns with practical production scheduling analogues where risk = resource consumption.

## 4. Model Formulation

Let `w_i` denote the fraction of capital allocated to asset `i`. Each snapshot provides:

- `r_i`: expected return (Binance 24 h price change percent / 100 or manual override).
- `risk_i`: normalized risk proxy ( (high − low)/price or override).

The LP is:

```
max   Σ_i w_i * r_i
subject to
       Σ_i w_i ≤ 1                                  (budget)
       w_i ≤ α  for all i                           (per-asset cap, α = max_allocation)
       Σ_i w_i * risk_i ≤ β                         (risk budget, β = risk_budget)
       w_i ≥ 0  for all i
```

Weights map to dollar allocations via `allocation_i = budget * w_i`; any unused budget becomes a cash reserve. The dashboard reports:

- Objective contributions `w_i * r_i` per asset.
- Constraint evaluations (left-hand side, right-hand side, slack).

## 5. Implementation Details

| Component | Role |
| --- | --- |
| `market.py` | Fetches Binance `/api/v3/ticker/24hr` data, computes expected returns (`priceChangePercent/100`) and risk `(high-low)/price`, or ingests manual rows. |
| `simplex.py` | Standalone primal simplex solver for `max c^T x` with `Ax ≤ b`, used by both CLI and dashboard. |
| `trader.py` | Builds LP matrices from snapshots (budget row, identity rows for caps, risk row) and converts solver output into `TradeDecision` objects (weights, dollars, quantities). |
| `ledger.py` | Appends each decision to `trades/trades.csv` so history survives restarts; a “Clear history” button truncates the file. |
| `app.py` | Streamlit UI: auto-refresh, manual override table (prefilled from Binance), rationale tables, Altair charts, rebalance log sourced from the ledger. |
| `tests/test_simplex.py` | Regression suite: (1) 2-variable LP with known optimum, (2) risk-budget scenario ensuring Σ w·risk equals the configured limit. |

### Data & Overrides

- Table pre-filling: on load, the “assets info table” fetches Binance values so users edit existing rows rather than typing everything.
- Partial overrides: editing only `expected_return` automatically rescales `risk` unless the risk cell is manually set, providing deterministic equations during defence.
- Offline mode: if Binance fails, rows with both price and expected return act as fallback snapshots (validated through exception handling paths in `app.py`).

### Tests

`pytest -q` executes the solver and trader regression suite, providing concrete evidence of correctness before scaling up.

## 6. Results & Analysis

Although the dashboard is interactive, we can summarize representative findings from typical runs (budget = $1,000, α = 0.4, β = 0.25):

- **Objective maximization:** Assets with stronger positive 24 h change (e.g., SOLUSDT at +2.2 %) acquire larger weights until limited by α or risk budget.
- **Risk binding cases:** When a high-return asset also has high span (`risk_i`), the risk budget becomes the active constraint. The “Constraint evaluations” table often shows `Σ w·risk = β`, mirroring the test in `tests/test_simplex.py`.
- **Ledger-backed charts:** After several rebalances, the weight chart highlights momentum shifts (e.g., BTC weight decreasing as expected return drops), while the allocation chart reveals dollar swings. Because charts read directly from `trades/trades.csv`, clearing history gives a fresh canvas for demos.
- **Manual overrides:** During Binance outages, the warning banner indicates fallback mode; the ledger still logs decisions, proving continuity.

## 7. Key Insights

1. **LP transparency scales:** By keeping the model linear and displaying each constraint’s slack, stakeholders can audit decisions without diving into solver internals.
2. **Manual overrides double as resilience:** Prefilled tables reduce friction, and partial overrides mean we can tweak a single asset without copy-pasting entire CSVs.
3. **Ledger-first analytics:** Persisting trades before plotting ensures reliable history, aligning with compliance best practices.
4. **Testing small before large:** The regression tests mirror the syllabus requirement to validate on low-dimensional LPs before scaling.

## 8. Summary & Future Improvements

The Simplex Crypto Allocator demonstrates how classical LP theory underpins a modern streaming allocator: data ingestion, model construction, simplex solve, and transparent reporting in one loop. For the course defence, the dashboard visualizations plus tests provide tangible evidence of understanding decision variables, objective, constraints, and solution interpretation.

Future enhancements:

- **Factor-aware risk:** Replace the scalar risk proxy with a factor matrix (still linear if factors are precomputed).
- **Dual reporting:** Surface simplex dual variables to discuss pricing of constraints during the defence.
- **Scenario testing:** Add a panel to batch-run manual scenarios (e.g., stress +5 %/-5 % returns) and compare outputs side by side.
- **Integration tests vs. external solvers:** Automate comparisons against `scipy.optimize.linprog` or CBC to satisfy the “manual vs. solver vs. library” requirement programmatically.

## References

1. H. Markowitz, “Portfolio Selection,” *The Journal of Finance*, vol. 7, no. 1, 1952, pp. 77–91.
2. G. B. Dantzig, “Maximization of a Linear Function of Variables Subject to Linear Inequalities,” 1947, USAF Project SCOOP.
3. D. Bertsimas and J. N. Tsitsiklis, *Introduction to Linear Optimization*, Athena Scientific, 1997.
4. S. Boyd and L. Vandenberghe, *Convex Optimization*, Cambridge University Press, 2004.
5. N. Borri, “Conditional Tail-Risk in Cryptocurrency Markets,” *Journal of Empirical Finance*, vol. 50, 2019, pp. 1–19.
6. Y. Liu and A. Tsyvinski, “Risks and Returns of Cryptocurrency,” *Review of Financial Studies*, vol. 34, no. 6, 2021, pp. 2689–2727.
7. A. Ben-Tal, L. El Ghaoui, and A. Nemirovski, *Robust Optimization*, Princeton University Press, 2009.
