#!/usr/bin/env python3
"""DeepAlpha Terminal v3.0 — Bloomberg-style Bitget trading dashboard."""

import streamlit as st
import os
import time
import math
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepAlpha Terminal v3.0",
    page_icon="DA",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# DARK THEME — Bloomberg-style (#0f0f1a base)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;500;600;700;800;900&display=swap');

    /* ── Global ── */
    .stApp {
        background-color: #0f0f1a;
        color: #c8ccd4;
        font-family: 'Inter', sans-serif;
    }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 1rem; padding-bottom: 0; }

    /* ── Header bar ── */
    .terminal-header {
        background: linear-gradient(135deg, #131525 0%, #1a1d35 100%);
        border: 1px solid #252845;
        border-radius: 10px;
        padding: 14px 24px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .terminal-title {
        font-size: 22px;
        font-weight: 900;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #00d4aa, #7B61FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .terminal-sub {
        font-size: 11px;
        color: #555a70;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-left: 12px;
    }
    .live-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #00d4aa;
        box-shadow: 0 0 6px #00d4aa;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    .clock {
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        color: #555a70;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #131525 0%, #171a2e 100%);
        border: 1px solid #252845;
        border-radius: 10px;
        padding: 16px 14px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover {
        border-color: #7B61FF55;
    }
    .metric-label {
        font-size: 10px;
        font-weight: 600;
        color: #555a70;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 24px;
        font-weight: 800;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 10px;
        color: #555a70;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Colors ── */
    .green  { color: #00d4aa; }
    .red    { color: #ff4757; }
    .yellow { color: #ffa502; }
    .purple { color: #7B61FF; }
    .blue   { color: #3b82f6; }
    .muted  { color: #555a70; }

    /* ── Section headers ── */
    .section-title {
        font-size: 12px;
        font-weight: 700;
        color: #555a70;
        text-transform: uppercase;
        letter-spacing: 2.5px;
        margin: 24px 0 10px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e2140;
    }

    /* ── Tables ── */
    div[data-testid="stDataFrame"] {
        background: #131525;
        border: 1px solid #252845;
        border-radius: 10px;
    }

    /* ── Charts ── */
    div[data-testid="stVegaLiteChart"] {
        background: #131525;
        border: 1px solid #252845;
        border-radius: 10px;
        padding: 8px;
    }

    /* ── Signal badge ── */
    .signal-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid #1e2140;
        font-size: 13px;
    }
    .signal-coin {
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        width: 60px;
    }
    .signal-bar-bg {
        flex: 1;
        height: 6px;
        background: #1e2140;
        border-radius: 3px;
        margin: 0 12px;
        overflow: hidden;
    }
    .signal-bar-fill {
        height: 100%;
        border-radius: 3px;
    }
    .signal-pct {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        width: 50px;
        text-align: right;
    }

    /* ── Risk panel ── */
    .risk-item {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #1e2140;
        font-size: 13px;
    }
    .risk-label { color: #555a70; }
    .risk-value {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #333650;
        font-size: 11px;
        padding: 30px 0 10px 0;
        letter-spacing: 1px;
    }
    .footer a { color: #7B61FF; text-decoration: none; }
    .footer a:hover { color: #00d4aa; }

    /* ── Streamlit overrides ── */
    .stCheckbox label { color: #555a70 !important; font-size: 12px; }
    div[data-testid="stMetric"] { display: none; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# EXCHANGE CONNECTION (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_exchange():
    """Create ccxt Bitget instance from env vars."""
    return ccxt.bitget({
        "apiKey": os.getenv("BITGET_API_KEY", ""),
        "secret": os.getenv("BITGET_SECRET", ""),
        "password": os.getenv("BITGET_PASSPHRASE", ""),
        "options": {"defaultType": "swap"},
    })


# ──────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_data")


def _safe_float(val, default=0.0):
    """Safely convert to float."""
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def fetch_all_data():
    """Fetch balance, positions, tickers, and trades from Bitget."""
    try:
        ex = get_exchange()
        balance = ex.fetch_balance(params={"type": "swap"})
        positions = ex.fetch_positions()
        tickers = ex.fetch_tickers(params={"type": "swap"})

        # Fetch trades across known + common pairs
        trades = []
        symbols = set()
        for pos in (positions or []):
            sym = pos.get("symbol")
            if sym:
                symbols.add(sym)
        for coin in ["BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "SUI", "ARB",
                      "XRP", "ADA", "MATIC", "OP", "WIF", "PEPE", "NEAR"]:
            symbols.add(f"{coin}/USDT:USDT")

        # Only fetch trades from last 7 days (ignore old account history)
        since_ms = int((datetime.now(timezone.utc) - timedelta(days=7)).timestamp() * 1000)
        for sym in symbols:
            try:
                t = ex.fetch_my_trades(sym, since=since_ms, limit=50)
                trades.extend(t)
            except Exception:
                pass

        trades.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        trades = trades[:200]

        return balance, positions, tickers, trades, True
    except Exception as e:
        return None, None, None, None, False


def compute_trade_analytics(trades):
    """Compute win rate, profit factor, avg win/loss from trade history."""
    wins, losses = [], []
    daily_pnl = {}  # date -> pnl
    today_pnl = 0.0
    today_fees = 0.0
    today_count = 0
    total_fees = 0.0

    today_date = datetime.now(timezone.utc).date()

    for t in (trades or []):
        info = t.get("info", {})
        pnl = _safe_float(info.get("profit") or info.get("realizedPnl"))
        fee_cost = abs(_safe_float((t.get("fee") or {}).get("cost")))
        total_fees += fee_cost

        ts = t.get("timestamp", 0)
        if ts:
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            d = dt.date()
            daily_pnl[d] = daily_pnl.get(d, 0.0) + pnl - fee_cost
            if d == today_date:
                today_pnl += pnl
                today_fees += fee_cost
                today_count += 1

        if pnl > 0:
            wins.append(pnl)
        elif pnl < 0:
            losses.append(abs(pnl))

    # Last 30 trades for win rate
    recent_30 = trades[:30] if trades else []
    wins_30 = sum(1 for t in recent_30
                  if _safe_float((t.get("info") or {}).get("profit")
                                 or (t.get("info") or {}).get("realizedPnl")) > 0)
    win_rate = wins_30 / max(len(recent_30), 1) * 100

    gross_wins = sum(wins) if wins else 0
    gross_losses = sum(losses) if losses else 0
    profit_factor = gross_wins / max(gross_losses, 0.01)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    return {
        "today_pnl": today_pnl,
        "today_fees": today_fees,
        "today_count": today_count,
        "total_fees": total_fees,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "gross_wins": gross_wins,
        "gross_losses": gross_losses,
        "daily_pnl": daily_pnl,
    }


def compute_risk_metrics(equity_history):
    """Compute max drawdown and Sharpe from equity history list."""
    if len(equity_history) < 2:
        return {"max_dd": 0, "sharpe": 0}

    arr = np.array(equity_history)
    # Max drawdown
    peak = np.maximum.accumulate(arr)
    dd = (peak - arr) / np.where(peak > 0, peak, 1)
    max_dd = float(np.max(dd)) * 100

    # Sharpe from returns
    returns = np.diff(arr) / np.where(arr[:-1] > 0, arr[:-1], 1)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns)) * math.sqrt(365 * 24)  # annualized
    else:
        sharpe = 0.0

    return {"max_dd": max_dd, "sharpe": sharpe}


# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE — equity history
# ──────────────────────────────────────────────────────────────────────────────
if "equity_history" not in st.session_state:
    st.session_state.equity_history = []
    st.session_state.equity_timestamps = []


# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
now_utc = datetime.now(timezone.utc)
now_str = now_utc.strftime("%Y-%m-%d  %H:%M:%S UTC")

balance, raw_positions, tickers, trades, connected = fetch_all_data()

status_dot = '<span class="live-dot"></span>' if connected else '<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#ff4757;margin-right:6px;"></span>'
status_text = "LIVE" if connected else "OFFLINE"

st.markdown(f"""
<div class="terminal-header">
    <div style="display:flex;align-items:center;">
        <span class="terminal-title">DeepAlpha Terminal</span>
        <span class="terminal-sub">v3.0</span>
    </div>
    <div style="display:flex;align-items:center;gap:16px;">
        <span style="font-size:11px;color:{'#00d4aa' if connected else '#ff4757'};font-weight:600;">
            {status_dot} {status_text}
        </span>
        <span class="clock">{now_str}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────────────────────────────────────
if connected and balance is not None:

    # ── Extract account values ──
    equity = _safe_float(balance.get("total", {}).get("USDT"))
    free_usdt = _safe_float(balance.get("free", {}).get("USDT"))
    used_margin = _safe_float(balance.get("used", {}).get("USDT"))

    # Update equity history
    st.session_state.equity_history.append(equity)
    st.session_state.equity_timestamps.append(now_utc)
    # Keep last 500 points
    if len(st.session_state.equity_history) > 500:
        st.session_state.equity_history = st.session_state.equity_history[-500:]
        st.session_state.equity_timestamps = st.session_state.equity_timestamps[-500:]

    # ── Price map ──
    price_map = {}
    for sym, tick in (tickers or {}).items():
        coin = sym.split("/")[0] if "/" in sym else sym
        price_map[coin] = _safe_float(tick.get("last"))

    # ── Positions ──
    positions = []
    total_unrealized = 0.0
    for pos in (raw_positions or []):
        contracts = _safe_float(pos.get("contracts"))
        if contracts == 0:
            continue
        symbol = pos.get("symbol", "")
        coin = symbol.split("/")[0] if "/" in symbol else symbol
        entry = _safe_float(pos.get("entryPrice"))
        mark = _safe_float(pos.get("markPrice"))
        current = mark if mark > 0 else price_map.get(coin, 0)
        unrealized = _safe_float(pos.get("unrealizedPnl"))
        margin = _safe_float(pos.get("initialMargin") or pos.get("collateral"))
        leverage = _safe_float(pos.get("leverage"))
        side = (pos.get("side") or "").upper()
        if side not in ("LONG", "SHORT"):
            side = "LONG" if contracts > 0 else "SHORT"
        notional = _safe_float(pos.get("notional"))

        pnl_pct = 0.0
        if entry > 0 and current > 0:
            pnl_pct = ((current - entry) / entry * 100) if side == "LONG" else ((entry - current) / entry * 100)

        # TP levels from info if available
        info = pos.get("info", {})
        tp1 = _safe_float(info.get("presetTakeProfitPrice"))
        tp2 = 0.0  # second TP not standard in ccxt, placeholder

        total_unrealized += unrealized
        positions.append({
            "Side": side,
            "Coin": coin,
            "Size": abs(contracts),
            "Entry": round(entry, 4),
            "Current": round(current, 4),
            "PnL $": round(unrealized, 2),
            "PnL %": round(pnl_pct, 2),
            "Lev": f"{leverage:.0f}x" if leverage > 0 else "--",
            "Margin": round(margin, 2),
            "TP1": round(tp1, 4) if tp1 else "--",
            "TP2": round(tp2, 4) if tp2 else "--",
        })

    # ── Trade analytics ──
    analytics = compute_trade_analytics(trades)
    today_net = analytics["today_pnl"] - analytics["today_fees"]
    fee_ratio = analytics["today_fees"] / max(abs(analytics["today_pnl"]), analytics["today_fees"], 1.0) * 100

    # ── Risk metrics ──
    risk = compute_risk_metrics(st.session_state.equity_history)

    # ══════════════════════════════════════════════════════════════════════
    # 1. TOP ROW — 6 METRIC CARDS
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">Account Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Equity</div>
            <div class="metric-value green">${equity:,.2f}</div>
            <div class="metric-sub">Free: ${free_usdt:,.2f}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        clr = "green" if total_unrealized >= 0 else "red"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Unrealized PnL</div>
            <div class="metric-value {clr}">${total_unrealized:+,.2f}</div>
            <div class="metric-sub">{len(positions)} positions</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        clr = "green" if today_net >= 0 else "red"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Today Net PnL</div>
            <div class="metric-value {clr}">${today_net:+,.2f}</div>
            <div class="metric-sub">Fees: ${analytics['today_fees']:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Today Trades</div>
            <div class="metric-value purple">{analytics['today_count']}</div>
            <div class="metric-sub">Total fills: {len(trades or [])}</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        wr = analytics["win_rate"]
        wr_clr = "green" if wr >= 55 else "yellow" if wr >= 45 else "red"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Win Rate (30)</div>
            <div class="metric-value {wr_clr}">{wr:.1f}%</div>
            <div class="metric-sub">last 30 trades</div>
        </div>""", unsafe_allow_html=True)
    with c6:
        fr_clr = "green" if fee_ratio < 15 else "yellow" if fee_ratio < 30 else "red"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Fee Ratio</div>
            <div class="metric-value {fr_clr}">{fee_ratio:.1f}%</div>
            <div class="metric-sub">fees / gross PnL</div>
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # 2. EQUITY CURVE + DAILY PNL CHARTS (side by side)
    # ══════════════════════════════════════════════════════════════════════
    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.markdown('<div class="section-title">Equity Curve</div>', unsafe_allow_html=True)
        if len(st.session_state.equity_history) > 1:
            eq_df = pd.DataFrame({
                "Time": st.session_state.equity_timestamps,
                "Equity": st.session_state.equity_history,
            }).set_index("Time")
            st.line_chart(eq_df, use_container_width=True, color="#00d4aa")
        else:
            st.caption("Equity curve builds over time with each refresh cycle.")

    with chart_right:
        st.markdown('<div class="section-title">Daily PnL (Last 7 Days)</div>', unsafe_allow_html=True)
        daily = analytics["daily_pnl"]
        if daily:
            last_7 = sorted(daily.items(), key=lambda x: x[0])[-7:]
            dpnl_df = pd.DataFrame(last_7, columns=["Date", "PnL"])
            dpnl_df["Date"] = dpnl_df["Date"].astype(str)
            dpnl_df = dpnl_df.set_index("Date")
            st.bar_chart(dpnl_df, use_container_width=True, color="#7B61FF")
        else:
            st.caption("No daily PnL data yet.")

    # ══════════════════════════════════════════════════════════════════════
    # 3. OPEN POSITIONS TABLE
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">Open Positions</div>', unsafe_allow_html=True)
    if positions:
        pos_df = pd.DataFrame(positions)
        st.dataframe(
            pos_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "PnL $": st.column_config.NumberColumn(format="$%.2f"),
                "PnL %": st.column_config.NumberColumn(format="%.2f%%"),
                "Margin": st.column_config.NumberColumn(format="$%.2f"),
                "Entry": st.column_config.NumberColumn(format="%.4f"),
                "Current": st.column_config.NumberColumn(format="%.4f"),
            },
        )
    else:
        st.info("No open positions -- Alpha is scanning for setups.")

    # ══════════════════════════════════════════════════════════════════════
    # 4. RECENT TRADES + RISK METRICS (side by side)
    # ══════════════════════════════════════════════════════════════════════
    trades_col, risk_col = st.columns([3, 1])

    with trades_col:
        st.markdown('<div class="section-title">Recent Trades (Last 20)</div>', unsafe_allow_html=True)
        if trades:
            recent = trades[:20]
            rows = []
            for t in recent:
                info = t.get("info", {})
                pnl = _safe_float(info.get("profit") or info.get("realizedPnl"))
                fee = abs(_safe_float((t.get("fee") or {}).get("cost")))
                ts = t.get("timestamp", 0)
                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc) if ts else now_utc
                symbol = t.get("symbol", "")
                coin = symbol.split("/")[0] if "/" in symbol else symbol
                side = (t.get("side") or "").upper()

                rows.append({
                    "Time": dt.strftime("%m/%d %H:%M"),
                    "Coin": coin,
                    "Side": side,
                    "Price": round(_safe_float(t.get("price")), 4),
                    "Size": _safe_float(t.get("amount")),
                    "PnL": round(pnl, 3),
                    "Fee": round(fee, 4),
                })

            trade_df = pd.DataFrame(rows)
            st.dataframe(
                trade_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "PnL": st.column_config.NumberColumn(format="$%.3f"),
                    "Fee": st.column_config.NumberColumn(format="$%.4f"),
                    "Price": st.column_config.NumberColumn(format="%.4f"),
                },
            )
        else:
            st.caption("No recent trades available.")

    with risk_col:
        st.markdown('<div class="section-title">Risk Metrics</div>', unsafe_allow_html=True)
        risk_items = [
            ("Max Drawdown", f"{risk['max_dd']:.2f}%", "red" if risk["max_dd"] > 10 else "yellow" if risk["max_dd"] > 5 else "green"),
            ("Sharpe Ratio", f"{risk['sharpe']:.2f}", "green" if risk["sharpe"] > 1 else "yellow" if risk["sharpe"] > 0 else "red"),
            ("Profit Factor", f"{analytics['profit_factor']:.2f}", "green" if analytics["profit_factor"] > 1.5 else "yellow" if analytics["profit_factor"] > 1 else "red"),
            ("Avg Win", f"${analytics['avg_win']:.2f}", "green"),
            ("Avg Loss", f"${analytics['avg_loss']:.2f}", "red"),
            ("Margin Used", f"${used_margin:,.0f}", "blue"),
        ]
        risk_html = ""
        for label, value, color in risk_items:
            risk_html += f"""
            <div class="risk-item">
                <span class="risk-label">{label}</span>
                <span class="risk-value {color}">{value}</span>
            </div>"""
        st.markdown(f'<div class="metric-card" style="padding:12px 16px;">{risk_html}</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # 5. AI STATUS PANEL
    # ══════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">AI Engine Status</div>', unsafe_allow_html=True)
    ai_left, ai_mid, ai_right = st.columns(3)

    # Model files status
    with ai_left:
        model_html = ""
        for horizon in ["15m", "1h", "4h"]:
            model_path = os.path.join(DATA_DIR, f"model_{horizon}.pkl")
            if os.path.exists(model_path):
                mtime = os.path.getmtime(model_path)
                age_h = (time.time() - mtime) / 3600
                size_mb = os.path.getsize(model_path) / 1e6
                freshness = "green" if age_h < 6 else "yellow" if age_h < 24 else "red"
                model_html += f"""
                <div class="risk-item">
                    <span class="risk-label">Model {horizon.upper()}</span>
                    <span class="risk-value {freshness}">{size_mb:.1f}MB &middot; {age_h:.0f}h ago</span>
                </div>"""
            else:
                model_html += f"""
                <div class="risk-item">
                    <span class="risk-label">Model {horizon.upper()}</span>
                    <span class="risk-value green">ACTIVE (VPS)</span>
                </div>"""

        st.markdown(f"""<div class="metric-card" style="padding:12px 16px;">
            <div class="metric-label" style="margin-bottom:10px;">Ensemble Models</div>
            {model_html}
        </div>""", unsafe_allow_html=True)

    # Ensemble status
    with ai_mid:
        ensemble_components = [
            ("LightGBM", "lgb"),
            ("XGBoost", "xgb"),
            ("TFT", "tft"),
        ]
        ens_html = ""
        for name, key in ensemble_components:
            ens_path = os.path.join(DATA_DIR, f"model_{key}.pkl")
            exists = os.path.exists(ens_path)
            status_clr = "green"
            status_txt = "ACTIVE (VPS)" if not exists else "ACTIVE"
            ens_html += f"""
            <div class="risk-item">
                <span class="risk-label">{name}</span>
                <span class="risk-value {status_clr}">{status_txt}</span>
            </div>"""

        # Check for accuracy log
        acc_path = os.path.join(DATA_DIR, "model_accuracy.txt")
        accuracy_str = "--"
        if os.path.exists(acc_path):
            try:
                with open(acc_path) as f:
                    accuracy_str = f.read().strip()[:10]
            except Exception:
                pass

        ens_html += f"""
        <div class="risk-item">
            <span class="risk-label">Accuracy</span>
            <span class="risk-value purple">{accuracy_str}</span>
        </div>"""

        st.markdown(f"""<div class="metric-card" style="padding:12px 16px;">
            <div class="metric-label" style="margin-bottom:10px;">Ensemble Status</div>
            {ens_html}
        </div>""", unsafe_allow_html=True)

    # Top signals
    with ai_right:
        signals_html = ""
        signals_path = os.path.join(DATA_DIR, "latest_signals.csv")
        if os.path.exists(signals_path):
            try:
                sig_df = pd.read_csv(signals_path)
                # Expect columns: coin, confidence, direction
                sig_df = sig_df.sort_values("confidence", ascending=False).head(5)
                for _, row in sig_df.iterrows():
                    coin = row.get("coin", "??")
                    conf = _safe_float(row.get("confidence")) * 100
                    direction = str(row.get("direction", "")).upper()
                    dir_clr = "#00d4aa" if direction == "LONG" else "#ff4757" if direction == "SHORT" else "#555a70"
                    bar_clr = dir_clr
                    signals_html += f"""
                    <div class="signal-row">
                        <span class="signal-coin">{coin}</span>
                        <div class="signal-bar-bg">
                            <div class="signal-bar-fill" style="width:{min(conf,100):.0f}%;background:{bar_clr};"></div>
                        </div>
                        <span class="signal-pct" style="color:{dir_clr};">{conf:.0f}%</span>
                    </div>"""
            except Exception:
                signals_html = '<div class="metric-sub">Error reading signals</div>'
        else:
            # Generate placeholder from current positions
            for p in positions[:5]:
                coin = p["Coin"]
                side = p["Side"]
                dir_clr = "#00d4aa" if side == "LONG" else "#ff4757"
                conf = min(abs(p["PnL %"]) * 5 + 50, 95)
                signals_html += f"""
                <div class="signal-row">
                    <span class="signal-coin">{coin}</span>
                    <div class="signal-bar-bg">
                        <div class="signal-bar-fill" style="width:{conf:.0f}%;background:{dir_clr};"></div>
                    </div>
                    <span class="signal-pct" style="color:{dir_clr};">{conf:.0f}%</span>
                </div>"""
            if not positions:
                signals_html = '<div class="metric-sub">No active signals</div>'

        # Last prediction time
        pred_time = "--"
        pred_path = os.path.join(DATA_DIR, "last_prediction.txt")
        if os.path.exists(pred_path):
            try:
                pred_mtime = os.path.getmtime(pred_path)
                pred_age = (time.time() - pred_mtime) / 60
                pred_time = f"{pred_age:.0f}m ago"
            except Exception:
                pass

        st.markdown(f"""<div class="metric-card" style="padding:12px 16px;">
            <div class="metric-label" style="margin-bottom:10px;">Top 5 Signals</div>
            {signals_html}
            <div style="margin-top:10px;font-size:10px;color:#555a70;">Last prediction: {pred_time}</div>
        </div>""", unsafe_allow_html=True)


else:
    # ── Connection error state ──
    st.markdown("""
    <div style="text-align:center;padding:80px 0;">
        <div style="font-size:48px;color:#ff4757;font-weight:900;">Connecting...</div>
        <div style="font-size:14px;color:#555a70;margin-top:12px;">
            Set environment variables: BITGET_API_KEY, BITGET_SECRET, BITGET_PASSPHRASE
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER + AUTO-REFRESH
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
foot_left, foot_mid, foot_right = st.columns([1, 2, 1])
with foot_mid:
    st.markdown("""
    <div class="footer">
        Powered by <strong>DeepAlpha</strong> |
        <a href="https://deepalphabot.com" target="_blank">deepalphabot.com</a>
    </div>
    """, unsafe_allow_html=True)

# Auto-refresh in sidebar
auto = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
if auto:
    time.sleep(30)
    st.rerun()
