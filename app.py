from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Iterable

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from market import (
    BinanceAPIError,
    BinanceClient,
    DEFAULT_SYMBOLS,
    get_market_snapshot,
)
from portfolio import solve_trading_problem
from solver.lp_solver import LPSolverError
from storage import (
    append_trade,
    compute_price_returns,
    fetch_history,
    init_db,
    load_portfolio,
    record_price_history,
    reset_database,
    save_portfolio,
)


HISTORY_LIMIT = 200
REFRESH_MIN_SECONDS = 1.0
REFRESH_MAX_SECONDS = 10.0


st.set_page_config(page_title="Crypto LP Trading Sandbox", layout="wide")
st.title("Crypto LP Trading Sandbox")
st.caption(
    "Toy quant-trading example: pull spot data from Binance, infer feedback Z, "
    "and solve an LP to distribute capital across cryptocurrencies."
)

if "trade_history" not in st.session_state:
    st.session_state.trade_history = []
if "latest_record" not in st.session_state:
    st.session_state.latest_record = None
if "auto_trading" not in st.session_state:
    st.session_state.auto_trading = False
if "auto_interval" not in st.session_state:
    st.session_state.auto_interval = 60.0
if "last_trade_ts" not in st.session_state:
    st.session_state.last_trade_ts = None
if "auto_notice" not in st.session_state:
    st.session_state.auto_notice = ""

init_db()
if not st.session_state.trade_history:
    persisted_history = fetch_history(HISTORY_LIMIT)
    if persisted_history:
        st.session_state.trade_history = persisted_history
        st.session_state.latest_record = persisted_history[-1]
        last_ts = persisted_history[-1]["timestamp"]
        if isinstance(last_ts, datetime):
            st.session_state.last_trade_ts = last_ts.timestamp()
        else:
            st.session_state.last_trade_ts = None
    else:
        st.session_state.trade_history = []

if "portfolio_cash" not in st.session_state or "portfolio_holdings" not in st.session_state:
    cash_balance, holdings = load_portfolio()
    st.session_state.portfolio_cash = cash_balance
    st.session_state.portfolio_holdings = holdings
if "additional_cash" not in st.session_state:
    st.session_state.additional_cash = 0.0
if "manual_override_enabled" not in st.session_state:
    st.session_state.manual_override_enabled = False
if "manual_override_prompted" not in st.session_state:
    st.session_state.manual_override_prompted = False
if "manual_overrides" not in st.session_state:
    st.session_state.manual_overrides = {}
if "_reset_additional_cash" not in st.session_state:
    st.session_state._reset_additional_cash = False


def _format_usd(value: float) -> str:
    return f"${value:,.2f}"


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _holdings_overview(
    snapshots: Iterable,
    holdings: dict[str, float],
) -> tuple[float, list[dict[str, float]]]:
    price_lookup = {asset.symbol: float(asset.price) for asset in snapshots}
    rows: list[dict[str, float]] = []
    total_value = 0.0
    for symbol, units in holdings.items():
        price = price_lookup.get(symbol)
        if price is None:
            continue
        notional = float(units) * price
        total_value += notional
        rows.append(
            {
                "Symbol": symbol,
                "Units": float(units),
                "Price (USDT)": price,
                "Value (USDT)": notional,
            }
        )
    rows.sort(key=lambda item: item["Value (USDT)"], reverse=True)
    return total_value, rows


st.sidebar.header("Portfolio & LP Configuration")

if st.session_state._reset_additional_cash:
    st.session_state.additional_cash = 0.0
    st.session_state._reset_additional_cash = False

deposit = st.sidebar.number_input(
    "Add cash this run (USDT)",
    min_value=0.0,
    step=100.0,
    format="%.2f",
    key="additional_cash",
)
risk_multiplier = st.sidebar.slider(
    "Risk multiplier (× equity)",
    min_value=0.1,
    max_value=3.0,
    value=1.2,
    step=0.1,
)
max_weight = st.sidebar.slider(
    "Max single-asset weight",
    min_value=0.05,
    max_value=1.0,
    value=0.4,
    step=0.05,
)
interval_seconds = st.sidebar.number_input(
    "Auto-trade interval (seconds)",
    min_value=1.0,
    max_value=86400.0,
    value=float(st.session_state.auto_interval),
    step=1.0,
    format="%.0f",
)

with st.sidebar.expander("Maintenance", expanded=False):
    st.caption("Clear all locally stored trades, holdings, and price history (irreversible).")
    if st.button("Clear local database", type="primary"):
        reset_database()
        st.session_state.trade_history = []
        st.session_state.latest_record = None
        st.session_state.portfolio_cash = 0.0
        st.session_state.portfolio_holdings = {}
        st.session_state.previous_prices = {}
        st.session_state.manual_override_enabled = False
        st.session_state.manual_override_prompted = False
        st.session_state.manual_overrides = {}
        st.session_state.auto_trading = False
        st.session_state.auto_notice = "Database cleared."
        st.session_state.last_trade_ts = None
        st.session_state._reset_additional_cash = True
        st.success("Database cleared. Please rerun to refresh state.")
        st.experimental_rerun()

portfolio_symbols = set(st.session_state.portfolio_holdings.keys())
symbol_universe = sorted(set(DEFAULT_SYMBOLS) | portfolio_symbols)
if not symbol_universe:
    symbol_universe = list(DEFAULT_SYMBOLS)
default_selection = [sym for sym in DEFAULT_SYMBOLS if sym in symbol_universe][:4]
if not default_selection and symbol_universe:
    default_selection = symbol_universe[:4]
selected_symbols = st.sidebar.multiselect(
    "Tradable symbols",
    symbol_universe,
    default=default_selection,
)
custom_symbol = st.sidebar.text_input(
    "Add symbol (e.g. DOGEUSDT)",
    value="",
).strip().upper()

if custom_symbol:
    selected_symbols.append(custom_symbol)

symbols = list(
    dict.fromkeys(
        [
            symbol
            for symbol in (selected_symbols + list(portfolio_symbols))
            if symbol
        ]
    )
)
if not symbols:
    st.error("Select at least one trading symbol in the sidebar.")
    st.stop()

client = BinanceClient()
try:
    snapshots = get_market_snapshot(symbols, client=client, enable_fallback=False)
    source_label = "Binance live API"
except BinanceAPIError as exc:
    st.warning(
        f"Could not reach Binance live API ({exc}); falling back to embedded sample data."
    )
    snapshots = get_market_snapshot(symbols, client=client, enable_fallback=True)
    source_label = "Sample fallback"

previous_prices = st.session_state.get("previous_prices", {})
record_price_history(snapshots)
current_prices = {asset.symbol: float(asset.price) for asset in snapshots}
historical_returns = compute_price_returns(
    symbols,
    current_prices,
    horizons_days=[7, 30, 365],
)

for asset in snapshots:
    symbol_returns = historical_returns.get(asset.symbol, {})
    weekly = symbol_returns.get(7)
    monthly = symbol_returns.get(30)
    yearly = symbol_returns.get(365)
    if weekly is not None:
        asset.return_7d = weekly
    if monthly is not None:
        asset.return_30d = monthly
    if yearly is not None:
        asset.return_365d = yearly

    asset.feedback_z = asset.price_change_percent / 100.0

all_non_positive = all(asset.feedback_z <= 0 for asset in snapshots)
if all_non_positive and not st.session_state.manual_override_enabled and not st.session_state.manual_override_prompted:
    st.session_state.manual_override_enabled = True
    st.session_state.manual_override_prompted = True

manual_toggle = st.sidebar.checkbox(
    "Enable manual expected-return overrides",
    value=st.session_state.manual_override_enabled,
)
st.session_state.manual_override_enabled = manual_toggle

manual_box = st.sidebar.expander(
    "Manual expected-return overrides", expanded=manual_toggle
)
manual_overrides: dict[str, float] = {}
if manual_toggle:
    with manual_box:
        st.caption(
            "Adjust annualised return forecasts (in percent). Positive values encourage buys; negative encourage sells."
        )
        for asset in snapshots:
            key = f"return_override_{asset.symbol}"
            default_pct = float(st.session_state.get(key, asset.feedback_z * 100))
            override_pct = st.number_input(
                f"{asset.symbol} override (%)",
                value=default_pct,
                step=0.01,
                format="%.4f",
                key=key,
            )
            manual_overrides[asset.symbol] = override_pct / 100.0
    st.session_state.manual_overrides = manual_overrides
else:
    with manual_box:
        st.caption("Enable overrides to edit expected returns.")
    st.session_state.manual_overrides = {}

manual_overrides = st.session_state.manual_overrides
manual_override_active = manual_toggle and bool(manual_overrides)

if all_non_positive and not manual_toggle:
    st.warning(
        "All modelled returns for the selected universe are non-positive. Enable manual overrides to provide custom expectations."
    )

for asset in snapshots:
    override_value = manual_overrides.get(asset.symbol)
    if override_value is not None:
        asset.feedback_z = override_value

market_rows = []
for asset in snapshots:
    prior_price = previous_prices.get(asset.symbol)
    realized = (
        (asset.price - prior_price) / prior_price
        if prior_price and prior_price > 0
        else None
    )
    manual_override_pct = (
        manual_overrides.get(asset.symbol) * 100 if asset.symbol in manual_overrides else None
    )
    market_rows.append(
        {
            "Symbol": asset.symbol,
            "Price (USDT)": asset.price,
            "24h Change (%)": asset.price_change_percent,
            "Risk Score": asset.risk_score,
            "Feedback Z": asset.feedback_z,
            "Δ since last run": realized,
            "7d Change (%)": asset.return_7d * 100 if asset.return_7d is not None else None,
            "30d Change (%)": asset.return_30d * 100 if asset.return_30d is not None else None,
            "365d Change (%)": asset.return_365d * 100 if asset.return_365d is not None else None,
            "Manual Override (%)": manual_override_pct,
            "Expected Return (%)": asset.feedback_z * 100,
            "Timestamp (UTC)": asset.event_time.strftime("%Y-%m-%d %H:%M"),
        }
    )

st.markdown("### Market Snapshot")
st.caption(f"Data source: {source_label}")
st.dataframe(
    market_rows,
    
    hide_index=True,
    column_config={
        "Price (USDT)": st.column_config.NumberColumn(format="%.2f"),
        "24h Change (%)": st.column_config.NumberColumn(format="%.2f"),
        "Risk Score": st.column_config.NumberColumn(format="%.3f"),
        "Δ since last run": st.column_config.NumberColumn(format="%.4f"),
        "7d Change (%)": st.column_config.NumberColumn(format="%.2f"),
        "30d Change (%)": st.column_config.NumberColumn(format="%.2f"),
        "365d Change (%)": st.column_config.NumberColumn(format="%.2f"),
        "Feedback Z": st.column_config.NumberColumn(format="%.4f"),
        "Manual Override (%)": st.column_config.NumberColumn(format="%.2f"),
        "Expected Return (%)": st.column_config.NumberColumn(format="%.2f"),
    },
)

portfolio_cash = float(st.session_state.portfolio_cash)
holdings_value, holdings_rows = _holdings_overview(
    snapshots,
    st.session_state.portfolio_holdings,
)
current_equity = portfolio_cash + holdings_value
equity_after_deposit = current_equity + float(deposit)

st.markdown("### Portfolio Overview")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Cash on hand", _format_usd(portfolio_cash))
col_b.metric("Holdings value", _format_usd(holdings_value))
col_c.metric("Total equity", _format_usd(current_equity))
st.caption(
    f"Pending deposit this run: {_format_usd(deposit)} → equity if executed: "
    f"{_format_usd(equity_after_deposit)}"
)

if holdings_rows:
    st.dataframe(
        holdings_rows,
        
        hide_index=True,
        column_config={
            "Price (USDT)": st.column_config.NumberColumn(format="%.2f"),
            "Value (USDT)": st.column_config.NumberColumn(format="%.2f"),
            "Units": st.column_config.NumberColumn(format="%.6f"),
        },
    )
else:
    st.info(
        "No open positions yet. Run the optimizer to deploy capital or add a deposit."
    )


def _execute_trade(trigger: str) -> tuple[bool, str | dict]:
    run_ts = datetime.now(timezone.utc)
    holdings_before = dict(st.session_state.portfolio_holdings)
    cash_before = float(st.session_state.portfolio_cash)
    deposit_amount = float(st.session_state.get("additional_cash", 0.0))
    price_lookup = {asset.symbol: float(asset.price) for asset in snapshots}
    holdings_value = sum(
        float(units) * price_lookup.get(symbol, 0.0)
        for symbol, units in holdings_before.items()
    )
    equity_before = cash_before + holdings_value
    total_equity = equity_before + deposit_amount
    if total_equity <= 0:
        return False, "No equity available. Add cash before trading."

    try:
        result = solve_trading_problem(
            snapshots,
            total_equity=total_equity,
            risk_multiplier=risk_multiplier,
            max_weight=max_weight,
            current_holdings=holdings_before,
        )
    except (LPSolverError, ValueError) as exc:
        return False, str(exc)

    holdings_after = {
        decision.symbol: decision.target_units
        for decision in result.decisions
        if decision.target_units > 1e-6
    }
    cash_after = result.cash_remaining
    st.session_state.portfolio_cash = cash_after
    st.session_state.portfolio_holdings = holdings_after
    save_portfolio(cash_after, holdings_after)

    total_capital = result.total_capital
    total_expected_return = result.total_expected_return
    total_risk = result.total_risk
    roi_invested = total_expected_return / total_capital if total_capital > 0 else 0.0
    roi_equity = (
        total_expected_return / result.total_equity if result.total_equity > 0 else 0.0
    )
    risk_limit = risk_multiplier * result.total_equity
    risk_usage = total_risk / risk_limit if risk_limit else 0.0

    record = {
        "timestamp": run_ts,
        "trigger": trigger,
        "capital": total_capital,
        "expected_pnl": total_expected_return,
        "expected_roi": roi_invested,
        "expected_equity_roi": roi_equity,
        "risk_usage": risk_usage,
        "equity": result.total_equity,
        "cash_before": cash_before + deposit_amount,
        "cash_after": cash_after,
        "deposit": deposit_amount,
        "return_horizon": "24h",
        "manual_override_enabled": manual_toggle,
        "manual_overrides": {
            symbol: round(value * 100, 4) for symbol, value in manual_overrides.items()
        }
        if manual_override_active
        else {},
    }
    record["allocations"] = [
        {
            "symbol": decision.symbol,
            "action": decision.action,
            "price": decision.price,
            "current_units": decision.current_units,
            "target_units": decision.target_units,
            "trade_units": decision.trade_units,
            "trade_usd": decision.trade_usd,
            "allocation_usd": decision.allocation_usd,
            "expected_pnl_usd": decision.expected_return_usd,
            "expected_return_frac": (
                decision.expected_return_usd / decision.allocation_usd
                if decision.allocation_usd > 1e-6
                else 0.0
            ),
            "risk_contribution": decision.risk_contribution,
        }
        for decision in result.decisions
        if decision.allocation_usd > 1e-6 or abs(decision.trade_usd) > 1e-6
    ]
    record["objective_value"] = result.solution.objective
    record["decision_vector"] = [float(x) for x in result.solution.x]

    st.session_state.trade_history.append(record)
    if len(st.session_state.trade_history) > HISTORY_LIMIT:
        st.session_state.trade_history = st.session_state.trade_history[-HISTORY_LIMIT:]
    st.session_state.latest_record = record
    append_trade(record)
    st.session_state._reset_additional_cash = True
    return True, record


manual_run = st.button("Run LP Allocation", type="primary")

if manual_run:
    success, payload = _execute_trade("manual")
    if success:
        st.session_state.last_trade_ts = time.time()
        timestamp = payload["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")
        st.success(f"Trade executed at {timestamp}.")
    else:
        st.error(f"Optimization error: {payload}")

st.sidebar.markdown("---")
start_col, stop_col = st.sidebar.columns(2)
if start_col.button("Start"):
    st.session_state.auto_trading = True
    st.session_state.auto_interval = interval_seconds
    st.session_state.auto_notice = "Auto trading enabled."
if stop_col.button("Stop"):
    st.session_state.auto_trading = False
    st.session_state.auto_notice = "Auto trading stopped."

st.session_state.auto_interval = interval_seconds
auto_status = "Running" if st.session_state.auto_trading else "Stopped"
status_color = "success" if st.session_state.auto_trading else "info"
getattr(st.sidebar, status_color)(f"Auto trading: {auto_status}")

if st.session_state.auto_notice:
    st.sidebar.caption(st.session_state.auto_notice)

if st.session_state.auto_trading:
    last_trade_ts = st.session_state.last_trade_ts
    interval = st.session_state.auto_interval
    now = time.time()
    if last_trade_ts is None:
        due = True
        seconds_remaining = 0.0
    else:
        elapsed = now - last_trade_ts
        due = elapsed >= interval
        seconds_remaining = max(0.0, interval - elapsed)

    st.sidebar.caption(f"Next trade in ≈ {int(seconds_remaining)}s")

    if due:
        success, payload = _execute_trade("auto")
        st.session_state.last_trade_ts = now
        if success:
            timestamp = payload["timestamp"].strftime("%Y-%m-%d %H:%M:%S UTC")
            st.session_state.auto_notice = f"Auto trade executed at {timestamp}."
        else:
            st.session_state.auto_notice = f"Auto trade error: {payload}"

    refresh_seconds = min(
        max(interval / 5.0, REFRESH_MIN_SECONDS),
        REFRESH_MAX_SECONDS,
    )
    components.html(
        f"""
        <script type="text/javascript">
            setTimeout(function() {{
                window.location.reload();
            }}, {int(refresh_seconds * 1000)});
        </script>
        """,
        height=0,
    )

latest = st.session_state.get("latest_record")
if latest:
    record = latest
    record_manual_overrides = record.get("manual_overrides", {}) or {}
    record_manual_enabled = bool(record.get("manual_override_enabled", False)) and bool(
        record_manual_overrides
    )

    st.markdown("#### Latest Decision")
    equity_value = record.get("equity", record["capital"])
    cash_after = record.get("cash_after", 0.0)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total equity", _format_usd(equity_value))
    col2.metric("Capital invested", _format_usd(record["capital"]))
    col3.metric("Expected PnL", _format_usd(record["expected_pnl"]))
    col4.metric("Cash remaining", _format_usd(cash_after))
    row1, row2, row3 = st.columns(3)
    row1.metric("ROI (invested)", _format_pct(record["expected_roi"]))
    row2.metric("ROI (equity)", _format_pct(record.get("expected_equity_roi", 0.0)))
    row3.metric(
        "Risk usage",
        _format_pct(record["risk_usage"]),
        help="Risk usage compares LP risk to the configured limit.",
    )
    caption_text = "Return signal → 24h change"
    if record_manual_enabled:
        caption_text += " | Manual overrides active"
    st.caption(caption_text)

    allocation_rows = []
    for item in record.get("allocations", []):
        symbol = item.get("symbol") or item.get("Symbol")
        if not symbol:
            continue
        action = (item.get("action") or item.get("Action") or "hold").upper()
        price = item.get("price") or item.get("Price (USDT)") or 0.0
        current_units = (
            item.get("current_units") or item.get("Current Units") or item.get("Units") or 0.0
        )
        target_units = item.get("target_units") or item.get("Target Units") or 0.0
        trade_units = item.get("trade_units") or item.get("Trade Units") or 0.0
        trade_usd = item.get("trade_usd") or item.get("Trade (USDT)") or 0.0
        allocation_usd = (
            item.get("allocation_usd") or item.get("Allocation (USDT)") or 0.0
        )
        expected_pnl_usd = (
            item.get("expected_pnl_usd") or item.get("Expected PnL (USDT)") or 0.0
        )
        expected_return_frac = item.get("expected_return_frac")
        if expected_return_frac is None and allocation_usd:
            expected_return_frac = expected_pnl_usd / allocation_usd
        risk_contribution = (
            item.get("risk_contribution") or item.get("Risk Contribution") or 0.0
        )
        override_pct = record_manual_overrides.get(symbol) if record_manual_overrides else None
        allocation_rows.append(
            {
                "Action": action,
                "Symbol": symbol,
                "Price (USDT)": price,
                "Current Units": current_units,
                "Target Units": target_units,
                "Trade Units": trade_units,
                "Trade (USDT)": trade_usd,
                "Allocation (USDT)": allocation_usd,
                "Expected PnL (USDT)": expected_pnl_usd,
                "Expected Return (%)": (expected_return_frac or 0.0) * 100,
                "Manual Override (%)": override_pct,
                "Risk Contribution": risk_contribution,
            }
        )

    if allocation_rows:
        st.markdown("#### Allocation Detail")
        st.dataframe(
            allocation_rows,
            
            hide_index=True,
            column_config={
                "Price (USDT)": st.column_config.NumberColumn(format="%.2f"),
                "Current Units": st.column_config.NumberColumn(format="%.6f"),
                "Target Units": st.column_config.NumberColumn(format="%.6f"),
                "Trade Units": st.column_config.NumberColumn(format="%.6f"),
                "Trade (USDT)": st.column_config.NumberColumn(format="%.2f"),
                "Allocation (USDT)": st.column_config.NumberColumn(format="%.2f"),
                "Expected PnL (USDT)": st.column_config.NumberColumn(format="%.2f"),
                "Expected Return (%)": st.column_config.NumberColumn(format="%.2f"),
                "Manual Override (%)": st.column_config.NumberColumn(format="%.2f"),
                "Risk Contribution": st.column_config.NumberColumn(format="%.2f"),
            },
        )
    else:
        st.info("The LP chose not to allocate capital under the current settings.")

    with st.expander("Solver Diagnostics"):
        st.write("Objective value:", record.get("objective_value"))
        st.write(
            "Decision vector:",
            [round(float(x), 4) for x in record.get("decision_vector", [])],
        )

trade_history = st.session_state.get("trade_history", [])
if trade_history:
    history_df = pd.DataFrame(trade_history)
    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
    if "expected_equity_roi" not in history_df.columns:
        history_df["expected_equity_roi"] = float("nan")
    if "equity" not in history_df.columns:
        history_df["equity"] = float("nan")
    if "cash_before" not in history_df.columns:
        history_df["cash_before"] = float("nan")
    if "cash_after" not in history_df.columns:
        history_df["cash_after"] = float("nan")
    if "deposit" not in history_df.columns:
        history_df["deposit"] = float("nan")
    history_df["expected_equity_roi"] = history_df["expected_equity_roi"].fillna(0.0)
    history_df["cash_before"] = history_df["cash_before"].fillna(0.0)
    history_df["cash_after"] = history_df["cash_after"].fillna(0.0)
    history_df["deposit"] = history_df["deposit"].fillna(0.0)
    history_df["equity"] = history_df["equity"].fillna(history_df["capital"])

    chart_df = history_df.tail(50).copy()
    chart_df["expected_roi_pct"] = chart_df["expected_roi"] * 100.0
    chart_df["expected_equity_roi_pct"] = chart_df["expected_equity_roi"] * 100.0
    chart_df["risk_usage_pct"] = chart_df["risk_usage"] * 100.0
    metric_map = {
        "expected_roi_pct": "Expected ROI %",
        "expected_equity_roi_pct": "Equity ROI %",
        "risk_usage_pct": "Risk Usage %",
    }
    melted = chart_df.melt(
        id_vars=["timestamp"],
        value_vars=list(metric_map.keys()),
        var_name="metric",
        value_name="value",
    )
    melted["metric"] = melted["metric"].map(metric_map)

    roi_chart = (
        alt.Chart(melted)
        .mark_line(point=True)
        .encode(
            x=alt.X("timestamp:T", title="Timestamp (UTC)"),
            y=alt.Y("value:Q", title="Percent"),
            color=alt.Color("metric:N", title="Metric"),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Timestamp"),
                alt.Tooltip("metric:N", title="Metric"),
                alt.Tooltip("value:Q", title="Value", format=".2f"),
            ],
        )
        .properties(height=280)
    )

    pnl_chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("timestamp:T", title="Timestamp (UTC)"),
            y=alt.Y("expected_pnl:Q", title="Expected PnL (USDT)"),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Timestamp"),
                alt.Tooltip("expected_pnl:Q", title="Expected PnL", format=".2f"),
                alt.Tooltip("capital:Q", title="Capital", format=".2f"),
            ],
        )
        .properties(height=280)
    )

    st.markdown("#### Trading History")
    st.altair_chart(roi_chart, use_container_width=True)
    st.altair_chart(pnl_chart, use_container_width=True)

    display_df = history_df.copy()
    display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    display_df["expected_roi"] = display_df["expected_roi"].map(
        lambda x: f"{x * 100:.2f}%"
    )
    display_df["expected_equity_roi"] = display_df["expected_equity_roi"].map(
        lambda x: f"{x * 100:.2f}%"
    )
    display_df["risk_usage"] = display_df["risk_usage"].map(
        lambda x: f"{x * 100:.2f}%"
    )
    if "return_horizon" in display_df.columns:
        display_df["return_horizon"] = display_df["return_horizon"].fillna("24h").map(
            lambda value: "24h"
            if str(value).lower() in {"1d", "24h"}
            else str(value)
        )
    else:
        display_df["return_horizon"] = "24h"
    if "manual_override_enabled" not in display_df.columns:
        display_df["manual_override_enabled"] = False
    if "manual_overrides" not in display_df.columns:
        display_df["manual_overrides"] = ""
    display_df["manual_override_enabled"] = display_df["manual_override_enabled"].map(
        lambda flag: "Yes" if bool(flag) else "No"
    )
    display_df["manual_overrides"] = display_df["manual_overrides"].apply(
        lambda overrides: ", ".join(
            f"{symbol}:{value:.2f}%"
            for symbol, value in overrides.items()
        )
        if isinstance(overrides, dict) and overrides
        else ""
    )
    st.dataframe(
        display_df[
            [
                "timestamp",
                "trigger",
                "equity",
                "capital",
                "cash_before",
                "cash_after",
                "deposit",
                "expected_pnl",
                "expected_roi",
                "expected_equity_roi",
                "risk_usage",
                "return_horizon",
                "manual_override_enabled",
                "manual_overrides",
            ]
        ],
        hide_index=True,
        column_config={
            "equity": st.column_config.NumberColumn("Equity (USDT)", format="%.2f"),
            "capital": st.column_config.NumberColumn("Capital (USDT)", format="%.2f"),
            "cash_before": st.column_config.NumberColumn("Cash Before (USDT)", format="%.2f"),
            "cash_after": st.column_config.NumberColumn("Cash (USDT)", format="%.2f"),
            "deposit": st.column_config.NumberColumn("Deposit (USDT)", format="%.2f"),
            "expected_pnl": st.column_config.NumberColumn("Expected PnL (USDT)", format="%.2f"),
            "return_horizon": st.column_config.TextColumn("Return Horizon"),
            "manual_override_enabled": st.column_config.TextColumn("Manual Overrides?"),
            "manual_overrides": st.column_config.TextColumn("Override Values"),
        },
    )

st.session_state["previous_prices"] = {
    asset.symbol: asset.price for asset in snapshots
}
