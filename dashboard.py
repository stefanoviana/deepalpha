#!/usr/bin/env python3
"""DeepAlpha Dashboard v3.0 — Bitget trading terminal via ccxt."""
import streamlit as st
import os
import time
import ccxt
import pandas as pd
from datetime import datetime, timezone

st.set_page_config(
    page_title="DeepAlpha Terminal",
    page_icon="DA",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- DARK THEME CSS ---
st.markdown("""
<style>
    .stApp { background-color: #0a0e17; color: #e0e0e0; }
    .metric-card { background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 16px; text-align: center; }
    .metric-value { font-size: 28px; font-weight: 800; }
    .metric-label { font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; }
    .green { color: #00d4aa; }
    .red { color: #ff4757; }
    .yellow { color: #ffa502; }
    .blue { color: #3b82f6; }
    .header-bar { background: linear-gradient(90deg, #111827 0%, #1a1f2e 100%); padding: 12px 20px; border-radius: 8px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; }
    .section-title { font-size: 14px; font-weight: 700; color: #6b7280; text-transform: uppercase; letter-spacing: 2px; margin: 20px 0 10px 0; border-bottom: 1px solid #1f2937; padding-bottom: 8px; }
    div[data-testid="stDataFrame"] { background: #111827; border-radius: 8px; }
    .stProgress > div > div { background-color: #00d4aa; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_data")


@st.cache_resource
def get_exchange():
    """Create ccxt Bitget instance (cached)."""
    return ccxt.bitget({
        "apiKey": os.getenv("BITGET_API_KEY", ""),
        "secret": os.getenv("BITGET_SECRET", ""),
        "password": os.getenv("BITGET_PASSPHRASE", ""),
        "options": {"defaultType": "swap"},
    })


def fetch_data():
    """Fetch all live data from Bitget via ccxt."""
    try:
        ex = get_exchange()
        balance = ex.fetch_balance(params={"type": "swap"})
        positions = ex.fetch_positions()
        tickers = ex.fetch_tickers(params={"type": "swap"})
        # Fetch recent trades (last 100)
        trades = []
        try:
            # Bitget: fetch trades for positions we know about, plus common pairs
            symbols_to_check = set()
            for pos in positions:
                if pos.get("symbol"):
                    symbols_to_check.add(pos["symbol"])
            # Add common pairs
            for coin in ["BTC", "ETH", "SOL", "DOGE", "AVAX", "LINK", "SUI", "ARB"]:
                symbols_to_check.add(f"{coin}/USDT:USDT")
            for sym in symbols_to_check:
                try:
                    t = ex.fetch_my_trades(sym, limit=50)
                    trades.extend(t)
                except Exception:
                    pass
            # Sort by timestamp desc
            trades.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            trades = trades[:200]
        except Exception:
            pass
        return balance, positions, tickers, trades
    except Exception as e:
        st.error(f"API error: {e}")
        return None, None, None, None


# --- HEADER ---
st.markdown("""
<div class="header-bar">
    <div style="display:flex;align-items:center;gap:12px;">
        <span style="font-size:24px;font-weight:900;color:#00d4aa;">DeepAlpha</span>
        <span style="font-size:12px;color:#6b7280;">TERMINAL v3.0 &mdash; Bitget</span>
    </div>
    <div style="font-size:12px;color:#6b7280;">""" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</div>
</div>
""", unsafe_allow_html=True)

balance, raw_positions, tickers, trades = fetch_data()

if balance is not None:
    # Account values from balance
    equity = float(balance.get("total", {}).get("USDT", 0) or 0)
    free_usdt = float(balance.get("free", {}).get("USDT", 0) or 0)
    used_margin = float(balance.get("used", {}).get("USDT", 0) or 0)

    # Build price map from tickers
    price_map = {}
    for sym, tick in (tickers or {}).items():
        # Extract coin from symbol like "BTC/USDT:USDT"
        coin = sym.split("/")[0] if "/" in sym else sym
        price_map[coin] = float(tick.get("last", 0) or 0)

    # Positions
    positions = []
    total_pnl = 0
    for pos in (raw_positions or []):
        contracts = float(pos.get("contracts", 0) or 0)
        if contracts == 0:
            continue
        symbol = pos.get("symbol", "")
        coin = symbol.split("/")[0] if "/" in symbol else symbol
        entry = float(pos.get("entryPrice", 0) or 0)
        current = float(pos.get("markPrice", 0) or pos.get("liquidationPrice", 0) or price_map.get(coin, 0))
        unrealized = float(pos.get("unrealizedPnl", 0) or 0)
        margin = float(pos.get("initialMargin", 0) or pos.get("collateral", 0) or 0)
        side = pos.get("side", "").upper()
        if side not in ("LONG", "SHORT"):
            side = "LONG" if contracts > 0 else "SHORT"
        notional = float(pos.get("notional", 0) or 0)
        pnl_pct = 0
        if entry > 0 and current > 0:
            pnl_pct = (current - entry) / entry * 100 if side == "LONG" else (entry - current) / entry * 100
        total_pnl += unrealized
        positions.append({
            "Side": side, "Coin": coin, "Size": abs(contracts),
            "Entry": entry, "Current": current,
            "PnL $": round(unrealized, 2), "PnL %": round(pnl_pct, 2),
            "Margin": round(margin, 0)
        })

    # Today's trades
    today_pnl = 0
    today_fees = 0
    today_trades = 0
    total_hist_pnl = 0
    total_hist_fees = 0
    if trades:
        for t in trades:
            # ccxt trade structure
            fee_cost = float((t.get("fee") or {}).get("cost", 0) or 0)
            # PnL from info field (exchange-specific)
            info = t.get("info", {})
            pnl = float(info.get("profit", 0) or info.get("realizedPnl", 0) or 0)
            total_hist_pnl += pnl
            total_hist_fees += abs(fee_cost)
            ts = t.get("timestamp", 0)
            if ts:
                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                if dt.date() == datetime.now(timezone.utc).date():
                    today_pnl += pnl
                    today_fees += abs(fee_cost)
                    today_trades += 1

    fee_ratio = today_fees / max(abs(today_pnl), 1) * 100

    # --- TOP METRICS ---
    st.markdown('<div class="section-title">Account Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total USDT</div><div class="metric-value green">${equity:,.0f}</div></div>', unsafe_allow_html=True)
    with c2:
        color = "green" if total_pnl >= 0 else "red"
        st.markdown(f'<div class="metric-card"><div class="metric-label">Unrealized PnL</div><div class="metric-value {color}">${total_pnl:+,.2f}</div></div>', unsafe_allow_html=True)
    with c3:
        color = "green" if today_pnl - today_fees >= 0 else "red"
        st.markdown(f'<div class="metric-card"><div class="metric-label">Today Net PnL</div><div class="metric-value {color}">${today_pnl - today_fees:+,.2f}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Today Trades</div><div class="metric-value blue">{today_trades}</div></div>', unsafe_allow_html=True)
    with c5:
        fee_color = "green" if fee_ratio < 15 else "yellow" if fee_ratio < 30 else "red"
        st.markdown(f'<div class="metric-card"><div class="metric-label">Fee Ratio</div><div class="metric-value {fee_color}">{fee_ratio:.1f}%</div></div>', unsafe_allow_html=True)
    with c6:
        usage = used_margin / equity * 100 if equity > 0 else 0
        st.markdown(f'<div class="metric-card"><div class="metric-label">Margin Usage</div><div class="metric-value blue">{usage:.0f}%</div></div>', unsafe_allow_html=True)

    # --- POSITIONS ---
    st.markdown('<div class="section-title">Open Positions</div>', unsafe_allow_html=True)
    if positions:
        df = pd.DataFrame(positions)
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={
                         "PnL $": st.column_config.NumberColumn(format="$%.2f"),
                         "PnL %": st.column_config.NumberColumn(format="%.2f%%"),
                         "Margin": st.column_config.NumberColumn(format="$%.0f"),
                     })
    else:
        st.info("No open positions -- Alpha is scanning for opportunities")

    # --- ORDER FLOW ---
    st.markdown('<div class="section-title">Cross-Venue Order Flow (Live)</div>', unsafe_allow_html=True)
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from order_flow_analyzer import get_order_flow_signal

        ofi_cols = st.columns(4)
        for idx, coin in enumerate(["BTC", "ETH", "SOL", "HYPE"]):
            signal = get_order_flow_signal(coin)
            with ofi_cols[idx]:
                score = signal["cross_venue_score"]
                direction = signal["direction"]
                color = "green" if direction == "LONG" else "red" if direction == "SHORT" else "blue"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{coin} Order Flow</div>
                    <div class="metric-value {color}">{score:+.3f}</div>
                    <div style="font-size:11px;color:#6b7280;">HL: {signal['ofi_hl']:+.2f} | BN: {signal['ofi_binance']:+.2f}</div>
                    <div style="font-size:13px;font-weight:700;" class="{color}">{direction}</div>
                </div>
                """, unsafe_allow_html=True)
    except Exception:
        st.caption("Order flow analyzer not available")

    # --- PERFORMANCE ---
    st.markdown('<div class="section-title">Performance Summary</div>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        net = total_hist_pnl - total_hist_fees
        color = "green" if net >= 0 else "red"
        st.markdown(f'<div class="metric-card"><div class="metric-label">All-Time Net PnL</div><div class="metric-value {color}">${net:+,.2f}</div></div>', unsafe_allow_html=True)
    with p2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Fees Paid</div><div class="metric-value yellow">${total_hist_fees:,.2f}</div></div>', unsafe_allow_html=True)
    with p3:
        total_fee_ratio = total_hist_fees / max(abs(total_hist_pnl), 1) * 100
        fee_color = "green" if total_fee_ratio < 15 else "yellow" if total_fee_ratio < 30 else "red"
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Fee Ratio</div><div class="metric-value {fee_color}">{total_fee_ratio:.0f}%</div></div>', unsafe_allow_html=True)
    with p4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Fills</div><div class="metric-value blue">{len(trades) if trades else 0}</div></div>', unsafe_allow_html=True)

    # --- MODEL STATUS ---
    st.markdown('<div class="section-title">AI Model Status</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    for horizon, col in [("1h", m1), ("4h", m2), ("15m", m3)]:
        model_path = os.path.join(DATA_DIR, f"model_{horizon}.pkl")
        if os.path.exists(model_path):
            mtime = os.path.getmtime(model_path)
            age = (time.time() - mtime) / 3600
            size_mb = os.path.getsize(model_path) / 1e6
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Model {horizon.upper()}</div>
                    <div class="metric-value green">{size_mb:.1f} MB</div>
                    <div style="font-size:11px;color:#6b7280;">Updated {age:.0f}h ago</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Model {horizon.upper()}</div><div class="metric-value red">N/A</div></div>', unsafe_allow_html=True)

    # --- RECENT TRADES ---
    st.markdown('<div class="section-title">Recent Trades</div>', unsafe_allow_html=True)
    if trades:
        recent = trades[:15]  # already sorted desc
        trade_rows = []
        for t in recent:
            info = t.get("info", {})
            pnl = float(info.get("profit", 0) or info.get("realizedPnl", 0) or 0)
            ts = t.get("timestamp", 0)
            if ts:
                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            else:
                dt = datetime.now(timezone.utc)
            symbol = t.get("symbol", "")
            coin = symbol.split("/")[0] if "/" in symbol else symbol
            fee_cost = float((t.get("fee") or {}).get("cost", 0) or 0)
            trade_rows.append({
                "Time": dt.strftime("%m/%d %H:%M"),
                "Coin": coin,
                "Side": t.get("side", ""),
                "Size": t.get("amount", 0),
                "Price": t.get("price", 0),
                "PnL": f"${pnl:+.2f}" if pnl != 0 else "--",
                "Fee": f"${abs(fee_cost):.3f}",
            })
        if trade_rows:
            st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No recent trades")

else:
    st.error("Cannot connect to Bitget API. Set BITGET_API_KEY, BITGET_SECRET, BITGET_PASSPHRASE env vars.")

# --- FOOTER ---
st.markdown("---")
fc1, fc2, fc3, fc4 = st.columns(4)
fc1.markdown("[GitHub](https://github.com/stefanoviana/deepalpha)")
fc2.markdown("[Discord](https://discord.gg/P4yX686m)")
fc3.markdown("[Telegram](https://t.me/DeepAlphaVault)")
fc4.markdown("[Bitget Referral](https://www.bitget.com/referral/register?clacCode=LAM6LQEJ)")

# Auto-refresh
if st.sidebar.checkbox("Auto-refresh (30s)"):
    time.sleep(30)
    st.rerun()
