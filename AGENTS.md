# Repository Guidelines

This codebase now centers on a simplex LP solver, a Binance/manual data interface, and a Streamlit dashboard with real-time charts. Keep contributions focused on these moving parts so the allocator stays understandable and auditable.

## Project Structure & Module Organization

- `simplex.py` hosts the primal simplex implementation; guard it with math-heavy comments only where the logic is non-obvious.
- `market.py` defines `AssetSnapshot`, the Binance REST client, and CSV/JSON loaders for manual overrides.
- `trader.py` converts snapshots into LP constraints (budget, per-asset cap, risk budget) and converts simplex outputs into trade decisions.
- `ledger.py` appends trades to a CSV ledger; nothing else writes to disk.
- `app.py` is the only entry point; keep it a thin Streamlit shell so the solver and market abstractions remain reusable.

## Build, Test, and Development Commands

- `python -m venv .venv && source .venv/bin/activate` then `pip install -r requirements.txt`.
- `streamlit run app.py` launches the dashboard (auto-refresh + charts).
- `pytest -q` executes `tests/test_simplex.py`, covering the solver and the trade engine.

## Coding Style & Naming Conventions

Stick to Python 3.10+, 4-space indentation, and type hints for public functions. Modules should stay under ~300 lines; factor helpers rather than introducing new packages. Use `snake_case` for functions/files, `PascalCase` for data classes, and prefer numpy vector math over manual loops. Print/log only inside the Streamlit layer so the solver and engine remain pure.

## Testing Guidelines

Add regression tests under `tests/` following the `test_*.py` glob. When touching `simplex.py`, craft deterministic problems with known optima; when modifying `trader.py`, synthesize snapshots with hand-calculable risk/return bounds. Mock Binance calls by injecting custom snapshots—no network access in tests.

## Commit & Pull Request Guidelines

Keep commit subjects short and imperative (`simplex: guard zero objective`). Reference issues when applicable, note dashboard or ledger format changes in the PR body, and attach `pytest` output or screenshots if they demonstrate new behavior.
