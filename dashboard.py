#!/usr/bin/env python3
"""DeepAlpha Dashboard - Streamlit web UI for monitoring and configuration."""
import streamlit as st
import json
import os
import time
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="DeepAlpha Dashboard", page_icon="DA", layout="wide")

# --- CONFIG ---
EXCHANGE = os.getenv("EXCHANGE", "hyperliquid")
WALLET = os.getenv("WALLET_ADDRESS", "")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_data")

# --- HEADER ---
st.title("DeepAlpha Dashboard")
st.caption("AI-Powered Trading Bot | " + EXCHANGE.capitalize())

# --- SIDEBAR ---
st.sidebar.header("Configuration")
st.sidebar.text("Exchange: " + EXCHANGE.capitalize())
st.sidebar.text("Wallet: " + (WALLET[:8] + "..." + WALLET[-4:] if WALLET else "Not set"))

# Refresh button
if st.sidebar.button("Refresh"):
    st.rerun()

# Auto-refresh
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)

# --- MAIN ---
col1, col2, col3, col4 = st.columns(4)


def get_hyperliquid_data():
    """Fetch live data from Hyperliquid."""
    try:
        r = requests.post("https://api.hyperliquid.xyz/info",
                         json={"type": "clearinghouseState", "user": WALLET}, timeout=5)
        state = r.json()
        r2 = requests.post("https://api.hyperliquid.xyz/info",
                          json={"type": "spotClearinghouseState", "user": WALLET}, timeout=5)
        spot = r2.json()
        return state, spot
    except Exception:
        return None, None


def get_positions(state):
    """Extract positions from state."""
    positions = []
    if not state:
        return positions
    for pos in state.get("assetPositions", []):
        p = pos.get("position", {})
        size = float(p.get("szi", 0))
        if size == 0:
            continue
        positions.append({
            "Coin": p.get("coin", ""),
            "Side": "SHORT" if size < 0 else "LONG",
            "Size": abs(size),
            "Entry": float(p.get("entryPx", 0)),
            "Margin": float(p.get("marginUsed", 0)),
            "PnL": float(p.get("unrealizedPnl", 0)),
            "Leverage": p.get("leverage", {}).get("value", "?"),
        })
    return positions


# Fetch data
if WALLET and EXCHANGE == "hyperliquid":
    state, spot = get_hyperliquid_data()

    if state and spot:
        equity = float(state.get("marginSummary", {}).get("accountValue", 0))
        usdc = float([b for b in spot["balances"] if b["coin"] == "USDC"][0]["total"])
        margin_used = float(state.get("marginSummary", {}).get("totalMarginUsed", 0))
        positions = get_positions(state)
        total_pnl = sum(p["PnL"] for p in positions)

        # Metrics
        col1.metric("USDC Total", f"${usdc:,.2f}")
        col2.metric("Perp Equity", f"${equity:,.2f}")
        col3.metric("Unrealized PnL", f"${total_pnl:+,.2f}",
                    delta=f"{total_pnl:+.2f}", delta_color="normal")
        col4.metric("Open Positions", str(len(positions)))

        # Positions table
        st.subheader("Open Positions")
        if positions:
            df = pd.DataFrame(positions)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions")

        # Margin usage
        st.subheader("Margin Usage")
        if usdc > 0:
            usage_pct = margin_used / usdc * 100
            st.progress(min(usage_pct / 100, 1.0))
            st.caption(f"${margin_used:,.0f} / ${usdc:,.0f} ({usage_pct:.1f}%)")
    else:
        st.error("Could not connect to Hyperliquid API")
else:
    st.warning("Set WALLET_ADDRESS and EXCHANGE in .env to see live data")

# --- MODEL INFO ---
st.subheader("AI Model Status")
col_m1, col_m2, col_m3 = st.columns(3)

for horizon, col in [("1h", col_m1), ("4h", col_m2), ("15m", col_m3)]:
    model_path = os.path.join(DATA_DIR, f"model_{horizon}.pkl")
    if os.path.exists(model_path):
        mtime = os.path.getmtime(model_path)
        age_hours = (time.time() - mtime) / 3600
        size_mb = os.path.getsize(model_path) / 1e6
        col.metric(f"Model {horizon.upper()}", f"{size_mb:.1f} MB")
        col.caption(f"Updated {age_hours:.1f}h ago")
    else:
        col.metric(f"Model {horizon.upper()}", "Not found")

# --- TRADE LOG ---
st.subheader("Recent Trades")
trades_path = os.path.join(DATA_DIR, "trades_log_vps.json")
if not os.path.exists(trades_path):
    trades_path = os.path.join(os.path.dirname(DATA_DIR), "trades_log.json")

if os.path.exists(trades_path):
    try:
        with open(trades_path) as f:
            trades = json.load(f)
        if trades:
            recent = trades[-20:]
            trade_data = []
            for t in reversed(recent):
                trade_data.append({
                    "Coin": t.get("coin", ""),
                    "Side": t.get("side", ""),
                    "Strategy": t.get("strategy", ""),
                    "PnL": f"${t.get('pnl', 0):+.2f}",
                    "Hold": f"{t.get('hold_time_min', 0):.0f}m",
                    "Reason": t.get("reason", "")[:30],
                })
            st.dataframe(pd.DataFrame(trade_data), use_container_width=True, hide_index=True)

            # Stats
            wins = len([t for t in trades if t.get("pnl", 0) > 0])
            losses = len([t for t in trades if t.get("pnl", 0) < 0])
            total = sum(t.get("pnl", 0) for t in trades)
            wr = wins / max(wins + losses, 1) * 100

            st.caption(f"Total: {len(trades)} trades | Win Rate: {wr:.1f}% | Net PnL: ${total:+.2f}")
    except Exception:
        st.error("Could not load trade log")
else:
    st.info("No trade log found. Run the bot first.")

# --- FOOTER ---
st.divider()
col_f1, col_f2, col_f3 = st.columns(3)
col_f1.markdown("[GitHub](https://github.com/stefanoviana/deepalpha)")
col_f2.markdown("[Docs](https://stefanoviana.github.io/deepalpha/)")
col_f3.markdown("[Discord](https://discord.gg/P4yX686m)")

# Auto-refresh
if auto_refresh:
    time.sleep(30)
    st.rerun()
