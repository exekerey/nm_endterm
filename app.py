"""Streamlit dashboard for the simplex-based crypto allocator."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from ledger import TradeLedger, TradeRecord
from market import DEFAULT_SYMBOLS, AssetSnapshot, BinanceAPIError, MarketDataSource
from simplex import SimplexError
from portfolio_state import PortfolioState, apply_target_state, compute_trades, load_state, save_state
from trader import SimplexTradeEngine, TradeDecision, TradingResult


USER_TZ = timezone(timedelta(hours=5))
STATE_PATH = Path("portfolio_state.json")


def _format_local_time(ts: datetime, *, include_date: bool = True) -> str:
    local_ts = ts.astimezone(USER_TZ)
    fmt = "%Y-%m-%d %H:%M:%S" if include_date else "%H:%M:%S"
    return local_ts.strftime(fmt)


def _init_session_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("summary_history", [])
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_snapshots", None)
    st.session_state.setdefault("last_timestamp", None)
    st.session_state.setdefault("auto_counter", -1)
    st.session_state.setdefault("manual_override_rows", None)
    st.session_state.setdefault("last_run_params", {})
    st.session_state.setdefault("manual_prefill_done", False)
    st.session_state.setdefault("manual_prefill_baseline", {})
    st.session_state.setdefault("manual_risk_overrides", {})
    st.session_state.setdefault("last_trades", [])


def _sync_manual_override_rows(symbols: Sequence[str]) -> List[dict]:
    rows = st.session_state.get("manual_override_rows")
    if rows is None:
        rows = [
            {"symbol": symbol, "price": None, "expected_return": None, "risk": None, "volume": None}
            for symbol in symbols
        ]
    else:
        existing = {str(row.get("symbol", "")).upper() for row in rows if str(row.get("symbol", "")).strip()}
        for symbol in symbols:
            if symbol.upper() not in existing:
                rows.append(
                    {"symbol": symbol, "price": None, "expected_return": None, "risk": None, "volume": None}
                )
    st.session_state["manual_override_rows"] = rows
    return rows


def _is_missing(value: object) -> bool:
    try:
        return value is None or pd.isna(value)
    except TypeError:
        return False


def _prefill_override_rows(symbols: Sequence[str]) -> List[dict]:
    source = MarketDataSource(symbols=symbols, enable_fallback=True)
    snapshots = source.fetch()
    rows: List[dict] = []
    for snap in snapshots:
        rows.append(
            {
                "symbol": snap.symbol,
                "price": snap.price,
                "expected_return": snap.expected_return,
                "risk": snap.risk,
                "volume": snap.volume,
            }
        )
    return rows


def _manual_editor(symbols: Sequence[str]) -> pd.DataFrame:
    if not st.session_state.get("manual_prefill_done"):
        try:
            prefilled = _prefill_override_rows(symbols)
            st.session_state["manual_override_rows"] = prefilled
            st.session_state["manual_prefill_baseline"] = {
                row["symbol"]: {
                    key: float(row[key])
                    for key in ("price", "expected_return", "risk", "volume")
                    if row.get(key) is not None
                }
                for row in prefilled
            }
        except BinanceAPIError as exc:
            st.warning(f"Failed to prefill overrides from Binance: {exc}")
            st.session_state["manual_override_rows"] = None
            st.session_state["manual_prefill_baseline"] = {}
        finally:
            st.session_state["manual_prefill_done"] = True
    rows = _sync_manual_override_rows(symbols)
    previous_rows = [row.copy() for row in rows]
    data = pd.DataFrame(rows, columns=["symbol", "price", "expected_return", "risk", "volume"])
    edited = st.data_editor(
        data,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "symbol": st.column_config.TextColumn("Symbol", help="Ticker symbol, e.g., BTCUSDT."),
            "price": st.column_config.NumberColumn("Price", min_value=0.0, format="%.6f"),
            "expected_return": st.column_config.NumberColumn(
                "Expected return", help="Decimal return (0.02 = 2%).", format="%.6f"
            ),
            "risk": st.column_config.NumberColumn("Risk", help="Optional risk score; defaults to |expected return|."),
            "volume": st.column_config.NumberColumn("Volume", help="Optional volume proxy."),
        },
        key="manual_override_editor",
    )
    _flag_manual_risk_overrides(previous_rows, edited)
    baseline = st.session_state.get("manual_prefill_baseline", {})
    edited = _auto_update_calculated_columns(edited, baseline)
    st.session_state["manual_override_rows"] = edited.to_dict("records")
    return edited


def _flag_manual_risk_overrides(previous_rows: List[dict], edited: pd.DataFrame) -> None:
    previous_map = {}
    for row in previous_rows or []:
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        previous_map[symbol] = row.get("risk")
    flags = st.session_state.setdefault("manual_risk_overrides", {})
    active_symbols: set[str] = set()
    for _, row in edited.iterrows():
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        active_symbols.add(symbol)
        current_risk = row.get("risk")
        previous_risk = previous_map.get(symbol)
        if _is_missing(current_risk):
            flags[symbol] = False
            continue
        if _is_missing(previous_risk):
            flags[symbol] = False
            continue
        if math.isclose(float(current_risk), float(previous_risk), rel_tol=1e-9, abs_tol=1e-12):
            flags[symbol] = False
            continue
        flags[symbol] = True
    for symbol in list(flags.keys()):
        if symbol not in active_symbols:
            flags.pop(symbol, None)
    st.session_state["manual_risk_overrides"] = flags


def _scaled_risk_value(
    baseline_expected: float | None,
    baseline_risk: float | None,
    new_expected: float,
) -> float:
    expected_abs = abs(float(new_expected))
    if baseline_expected is None or baseline_risk is None:
        return expected_abs
    baseline_abs = abs(float(baseline_expected))
    if baseline_abs < 1e-9:
        return expected_abs
    scale = expected_abs / baseline_abs if baseline_abs else 0.0
    return max(float(baseline_risk) * scale, 0.0)


def _auto_update_calculated_columns(
    edited: pd.DataFrame,
    baseline: dict[str, dict[str, float]],
) -> pd.DataFrame:
    if edited.empty:
        return edited
    manual_flags = st.session_state.setdefault("manual_risk_overrides", {})
    updated = edited.copy()
    for idx, row in updated.iterrows():
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        expected = row.get("expected_return")
        if _is_missing(expected):
            manual_flags[symbol] = manual_flags.get(symbol, False)
            continue
        baseline_row = baseline.get(symbol, {})
        baseline_expected = baseline_row.get("expected_return")
        baseline_risk = baseline_row.get("risk")
        expected_changed = (
            baseline_expected is None
            or not math.isclose(float(expected), float(baseline_expected), rel_tol=1e-9, abs_tol=1e-12)
        )
        if not expected_changed:
            continue
        if manual_flags.get(symbol):
            continue
        auto_value = _scaled_risk_value(baseline_expected, baseline_risk, float(expected))
        updated.at[idx, "risk"] = auto_value
    st.session_state["manual_risk_overrides"] = manual_flags
    return updated


def _collect_overrides(
    rows: pd.DataFrame,
    baseline: dict[str, dict[str, float]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    overrides: dict[str, dict[str, float]] = {}
    fallback_rows: dict[str, dict[str, float]] = {}
    seen: set[str] = set()
    for _, row in rows.iterrows():
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        if symbol in seen:
            raise ValueError(f"duplicate symbol '{symbol}' in manual overrides")
        seen.add(symbol)
        baseline_fields = baseline.get(symbol, {})
        override_fields: dict[str, float] = {}
        fallback_fields: dict[str, float] = {}
        for key in ("price", "expected_return", "risk", "volume"):
            value = row.get(key)
            if _is_missing(value):
                continue
            numeric_value = float(value)
            fallback_fields[key] = numeric_value
            baseline_value = baseline_fields.get(key)
            if baseline_value is None or not math.isclose(
                numeric_value, float(baseline_value), rel_tol=1e-9, abs_tol=1e-12
            ):
                override_fields[key] = numeric_value
        if fallback_fields:
            fallback_rows[symbol] = fallback_fields
        if override_fields:
            overrides[symbol] = override_fields
    return overrides, fallback_rows


def _apply_overrides_to_snapshots(
    snapshots: Sequence[AssetSnapshot],
    overrides: dict[str, dict[str, float]],
) -> tuple[List[AssetSnapshot], set[str]]:
    if not overrides:
        return list(snapshots), set()
    updated: List[AssetSnapshot] = []
    applied: set[str] = set()
    for snap in snapshots:
        fields = overrides.get(snap.symbol)
        if not fields:
            updated.append(snap)
            continue
        applied.add(snap.symbol)
        updated.append(
            AssetSnapshot(
                symbol=snap.symbol,
                price=fields.get("price", snap.price),
                expected_return=fields.get("expected_return", snap.expected_return),
                risk=max(fields.get("risk", snap.risk), 0.0),
                volume=max(fields.get("volume", snap.volume), 0.0),
                timestamp=snap.timestamp,
            )
        )
    remaining = set(overrides.keys()) - applied
    if remaining:
        now = datetime.now(timezone.utc)
        for symbol in sorted(remaining):
            fields = overrides[symbol]
            price = fields.get("price")
            expected = fields.get("expected_return")
            if price is None or expected is None:
                continue
            risk = fields.get("risk", abs(expected))
            volume = fields.get("volume", 0.0)
            updated.append(
                AssetSnapshot(
                    symbol=symbol,
                    price=float(price),
                    expected_return=float(expected),
                    risk=max(float(risk), 0.0),
                    volume=max(float(volume), 0.0),
                    timestamp=now,
                )
            )
            applied.add(symbol)
    return updated, applied


def _snapshots_from_overrides(
    overrides: dict[str, dict[str, float]]
) -> tuple[List[AssetSnapshot], List[str]]:
    now = datetime.now(timezone.utc)
    snapshots: List[AssetSnapshot] = []
    incomplete: List[str] = []
    for symbol, fields in overrides.items():
        price = fields.get("price")
        expected = fields.get("expected_return")
        if price is None or expected is None:
            incomplete.append(symbol)
            continue
        risk = fields.get("risk", abs(expected))
        volume = fields.get("volume", 0.0)
        snapshots.append(
            AssetSnapshot(
                symbol=symbol,
                price=float(price),
                expected_return=float(expected),
                risk=max(float(risk), 0.0),
                volume=max(float(volume), 0.0),
                timestamp=now,
            )
        )
    return snapshots, incomplete


def _lp_diagnostics(
    snapshots: Sequence[AssetSnapshot],
    decisions: Sequence[TradeDecision],
    max_allocation: float,
    risk_budget: float,
    budget: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not decisions:
        return pd.DataFrame(), pd.DataFrame()
    weights = np.array([decision.weight for decision in decisions], dtype=float)
    expected = np.array([snap.expected_return for snap in snapshots], dtype=float)
    allocations = budget * weights
    usd_returns = allocations * expected
    objective_rows = []
    for decision, allocation, usd_return in zip(decisions, allocations, usd_returns):
        objective_rows.append(
            {
                "symbol": decision.symbol,
                "weight": decision.weight,
                "allocation": allocation,
                "expected_return_usd": usd_return,
            }
        )
    objective_df = pd.DataFrame(objective_rows)

    constraint_rows: List[dict] = []
    total_weight = float(weights.sum())
    constraint_rows.append(
        {
            "constraint": "Budget Σw_i ≤ 1.0",
            "lhs": total_weight,
            "rhs": 1.0,
            "slack": 1.0 - total_weight,
            "tight": abs(1.0 - total_weight) < 1e-6,
        }
    )
    if max_allocation < 1.0:
        for decision, weight in zip(decisions, weights):
            constraint_rows.append(
                {
                    "constraint": f"{decision.symbol} cap w_i ≤ {max_allocation:.2f}",
                    "lhs": weight,
                    "rhs": max_allocation,
                    "slack": max_allocation - weight,
                    "tight": (max_allocation - weight) < 1e-6,
                }
            )
    risk_vector = np.array([snap.risk for snap in snapshots], dtype=float)
    if np.any(risk_vector > 0.0):
        risk_lhs = float(np.dot(weights, risk_vector))
        constraint_rows.append(
            {
                "constraint": "Risk Σw_i·risk_i ≤ risk_budget",
                "lhs": risk_lhs,
                "rhs": risk_budget,
                "slack": risk_budget - risk_lhs,
                "tight": (risk_budget - risk_lhs) < 1e-6,
            }
        )
    constraints_df = pd.DataFrame(constraint_rows)
    return objective_df, constraints_df


def _record_history(timestamp: datetime, decisions: Iterable[TradeDecision], result: TradingResult) -> None:
    history = st.session_state["history"]
    summary_history = st.session_state["summary_history"]
    iteration = len(summary_history)
    for decision in decisions:
        history.append(
            {
                "timestamp": timestamp,
                "iteration": iteration,
                "symbol": decision.symbol,
                "weight": decision.weight,
                "allocation": decision.allocation,
                "quantity": decision.quantity,
                "price": decision.price,
                "expected_return": decision.expected_return,
            }
        )
    st.session_state["history"] = history
    summary_history.append(
        {
            "timestamp": timestamp,
            "objective": result.objective,
            "cash_reserve": result.cash_reserve,
        }
    )
    st.session_state["summary_history"] = summary_history


def _decisions_dataframe(timestamp: datetime, decisions: Sequence[TradeDecision]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "symbol": d.symbol,
                "weight": d.weight,
                "allocation": d.allocation,
                "quantity": d.quantity,
                "price": d.price,
                "expected_return_usd": d.allocation * d.expected_return,
            }
            for d in decisions
        ]
    )


def _trades_dataframe(trades: Sequence[dict]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "symbol": trade["symbol"],
                "action": trade["action"],
                "quantity": trade["quantity"],
                "price": trade["price"],
                "notional_usd": trade["notional_usd"],
            }
            for trade in trades
        ]
    )


def _append_to_ledger(
    ledger_path: Path,
    timestamp: datetime,
    iteration: int,
    decisions: Sequence[TradeDecision],
) -> None:
    ledger = TradeLedger(ledger_path)
    records = [
        TradeRecord(
            timestamp=timestamp,
            iteration=iteration,
            symbol=decision.symbol,
            weight=decision.weight,
            allocation=decision.allocation,
            quantity=decision.quantity,
            price=decision.price,
            expected_return=decision.expected_return,
        )
        for decision in decisions
    ]
    ledger.extend(records)


def _clear_ledger_file(path: Path) -> None:
    if path.exists():
        path.unlink()
    TradeLedger(path)


def _load_ledger_history(ledger_path: Path) -> pd.DataFrame:
    if not ledger_path.exists():
        return pd.DataFrame(columns=TradeLedger.FIELDNAMES)
    try:
        return pd.read_csv(ledger_path, parse_dates=["timestamp"])
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame(columns=TradeLedger.FIELDNAMES)


def main() -> None:
    st.set_page_config(page_title="Simplex Crypto Allocator", layout="wide")
    _init_session_state()

    st.title("simplex crypto allocator")

    with st.sidebar:
        st.header("Parameters")
        symbols = st.multiselect(
            "Symbols",
            options=sorted(DEFAULT_SYMBOLS),
            default=list(DEFAULT_SYMBOLS),
            help="Trading pairs to include in the LP.",
        )
        budget = st.number_input("Budget (quote currency)", min_value=100.0, value=1_000.0, step=100.0)
        max_allocation = st.slider(
            "Max allocation per asset",
            min_value=0.05,
            max_value=1.0,
            value=0.4,
            step=0.05,
            help="Upper bound on each weight before scaling to dollars.",
        )
        risk_budget = st.slider(
            "Risk budget",
            min_value=0.05,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Caps the dot product of weights and per-asset risk scores.",
        )
        clear_ledger_clicked = st.button("Clear history", help="Truncate the ledger file", use_container_width=True)
        st.divider()
        auto_refresh = st.checkbox("Auto-refresh", value=False)
        refresh_interval = st.number_input(
            "Refresh interval (seconds)",
            min_value=10,
            max_value=600,
            value=60,
            step=10,
        )

    ledger_path = Path("trades.csv")
    if clear_ledger_clicked:
        _clear_ledger_file(ledger_path)
        if STATE_PATH.exists():
            STATE_PATH.unlink()
        st.session_state["history"] = []
        st.session_state["summary_history"] = []
        st.session_state["last_trades"] = []
        st.sidebar.success("Ledger cleared.")

    st.subheader("assets info table")
    manual_override_df = _manual_editor(symbols)

    run_click = st.button("Rebalance now", type="primary")
    should_run = run_click

    if auto_refresh:
        auto_counter = st_autorefresh(interval=int(refresh_interval * 1000), key="auto_refresh_counter")
        if auto_counter != st.session_state.get("auto_counter"):
            should_run = True
            st.session_state["auto_counter"] = auto_counter
    else:
        st.session_state["auto_counter"] = -1

    if should_run and symbols:
        try:
            baseline = st.session_state.get("manual_prefill_baseline", {})
            overrides_map: dict[str, dict[str, float]] = {}
            fallback_rows: dict[str, dict[str, float]] = {}
            if manual_override_df is not None:
                overrides_map, fallback_rows = _collect_overrides(manual_override_df, baseline)
            timestamp = datetime.now(timezone.utc)
            source = MarketDataSource(symbols=symbols, enable_fallback=False)
            binance_error: BinanceAPIError | None = None
            incomplete_symbols: List[str] = []
            applied_symbols: set[str] = set()
            try:
                base_snapshots = source.fetch()
            except BinanceAPIError as exc:
                binance_error = exc
            if binance_error is None:
                snapshots, applied_symbols = _apply_overrides_to_snapshots(base_snapshots, overrides_map)
            else:
                fallback_snapshots, incomplete_symbols = _snapshots_from_overrides(fallback_rows)
                if not fallback_snapshots:
                    raise ValueError(
                        "Binance data unavailable and manual overrides must include price and expected return for at least one symbol."
                    ) from binance_error
                snapshots = fallback_snapshots
                applied_symbols = {snap.symbol for snap in snapshots}
            engine = SimplexTradeEngine(
                budget=budget,
                max_allocation=max_allocation,
                risk_budget=risk_budget,
            )
            result = engine.allocate(snapshots)
            _append_to_ledger(ledger_path, timestamp, len(st.session_state["summary_history"]), result.decisions)
            previous_state = load_state(STATE_PATH)
            trades = compute_trades(previous_state, result.decisions)
            save_state(apply_target_state(result), STATE_PATH)
            st.session_state["last_trades"] = trades
            _record_history(timestamp, result.decisions, result)
            st.session_state["last_result"] = result
            st.session_state["last_snapshots"] = snapshots
            st.session_state["last_timestamp"] = timestamp
            st.session_state["last_run_params"] = {
                "budget": budget,
                "max_allocation": max_allocation,
                "risk_budget": risk_budget,
            }
            st.success(f"Rebalance completed at {_format_local_time(timestamp)} UTC+5")
            if binance_error is not None:
                message = "Binance data unavailable; allocation used manual overrides only."
                if incomplete_symbols:
                    unique = ", ".join(sorted(set(incomplete_symbols)))
                    message += f" Skipped symbols missing price or expected return: {unique}."
                st.warning(message)
        except BinanceAPIError as exc:
            st.error(
                f"Allocation failed: {exc} Enter price and expected return overrides to run when Binance is offline."
            )
        except (SimplexError, ValueError) as exc:
            st.error(f"Allocation failed: {exc}")

    last_result: TradingResult | None = st.session_state.get("last_result")
    last_snapshots: Sequence[AssetSnapshot] | None = st.session_state.get("last_snapshots")
    last_timestamp: datetime | None = st.session_state.get("last_timestamp")
    last_params = st.session_state.get("last_run_params") or {}

    if last_result and last_snapshots and last_timestamp:
        col1, col2, col3 = st.columns(3)
        col1.metric("Objective (Σ w·r)", f"{last_result.objective:.4f}")
        col2.metric("Cash reserve", f"${last_result.cash_reserve:,.2f}")
        col3.metric("Tracked assets", f"{len(last_result.decisions)}")

        st.subheader("Latest Allocation")
        decisions_df = _decisions_dataframe(last_timestamp, last_result.decisions)
        st.dataframe(decisions_df, hide_index=True, use_container_width=True)

        trades_df = _trades_dataframe(st.session_state.get("last_trades", []))
        if not trades_df.empty:
            st.subheader("trade instructions")
            st.dataframe(trades_df, hide_index=True, use_container_width=True)

        st.subheader("Allocation Rationale")
        budget_note = last_params.get("budget")
        max_alloc_note = last_params.get("max_allocation")
        risk_budget_note = last_params.get("risk_budget")
        rationale_parts = [
            "The LP maximizes Σ wᵢ·rᵢ where rᵢ is each asset's 24h price change from Binance (or manual overrides).",
            "Weights represent fractions of the total budget and must satisfy Σ wᵢ ≤ 1.",
        ]
        if max_alloc_note is not None:
            rationale_parts.append(f"Each asset weight is capped at wᵢ ≤ {max_alloc_note:.2f}.")
        if risk_budget_note is not None:
            rationale_parts.append(
                f"The risk guardrail enforces Σ wᵢ·riskᵢ ≤ {risk_budget_note:.2f}, using per-asset risk scores."
            )
        if budget_note is not None:
            rationale_parts.append(f"With the stored budget of ${budget_note:,.2f}, weights map directly to dollars.")
        st.markdown(" ".join(rationale_parts))

        diag_max_allocation = float(max_alloc_note) if max_alloc_note is not None else 1.0
        diag_risk_budget = float(risk_budget_note) if risk_budget_note is not None else 1.0
        diag_budget = (
            float(budget_note)
            if budget_note is not None
            else sum(d.allocation for d in last_result.decisions) + last_result.cash_reserve
        )
        objective_df, constraints_df = _lp_diagnostics(
            last_snapshots,
            last_result.decisions,
            diag_max_allocation,
            diag_risk_budget,
            diag_budget,
        )
        if not objective_df.empty:
            st.markdown("**Objective terms (per-asset contributions)**")
            st.dataframe(objective_df, hide_index=True, use_container_width=True)
        if not constraints_df.empty:
            st.markdown("**Constraint evaluations**")
            st.dataframe(constraints_df, hide_index=True, use_container_width=True)

    history_df = _load_ledger_history(ledger_path)

    st.subheader("Real-time Charts")
    if not history_df.empty:
        ts_series = history_df["timestamp"]
        if ts_series.dt.tz is None:
            ts_series = ts_series.dt.tz_localize(timezone.utc)
        history_df = history_df.assign(timestamp=ts_series)
        weight_chart = (
            alt.Chart(history_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("timestamp:T", title="Timestamp"),
                y=alt.Y("weight:Q", title="Portfolio weight"),
                color=alt.Color("symbol:N", title="Symbol"),
            )
            .properties(title="Weights over time")
        )
        st.altair_chart(weight_chart, use_container_width=True)

        allocation_chart = (
            alt.Chart(history_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("timestamp:T", title="Timestamp"),
                y=alt.Y("allocation:Q", title="Allocation ($)"),
                color=alt.Color("symbol:N", title="Symbol"),
            )
            .properties(title="Capital allocation over time")
        )
        st.altair_chart(allocation_chart, use_container_width=True)
    else:
        st.info("Run at least one allocation to populate the charts.")

    summary_history = st.session_state.get("summary_history", [])
    if summary_history:
        summary_df = pd.DataFrame(summary_history)
        summary_chart = (
            alt.Chart(summary_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("timestamp:T", title="Timestamp"),
                y=alt.Y("objective:Q", title="Objective value"),
            )
            .properties(title="Objective value over time")
        )
        st.altair_chart(summary_chart, use_container_width=True)

    if not history_df.empty:
        st.subheader("Rebalance Log")
        display_cols = [
            "timestamp",
            "iteration",
            "symbol",
            "weight",
            "allocation",
            "quantity",
            "price",
            "expected_return",
        ]
        ts_series = history_df["timestamp"]
        local_series = ts_series.dt.tz_convert(USER_TZ)
        log_df = (
            history_df.assign(timestamp=local_series)
            .sort_values(by=["timestamp", "symbol"], ascending=[False, True])
            .reindex(columns=display_cols)
        )
        log_df["timestamp"] = log_df["timestamp"].dt.strftime("%H:%M:%S")
        st.dataframe(
            log_df,
            hide_index=True,
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
