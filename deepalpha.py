"""
DeepAlpha — AI Trading Bot (Free Version)
Autonomous trading on Hyperliquid using a LightGBM model.

Usage:
    python download_data.py   # download historical data
    python train.py           # train the AI model
    python deepalpha.py       # start trading

Environment variables: see .env.example
"""

import json
import os
import pickle
import time
import traceback
import numpy as np
import requests
import lightgbm as lgb
from eth_account import Account
from hyperliquid.utils import constants
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info

import config
from features import build_features, FEATURE_NAMES
from risk_manager import RiskManager

# ─── Hyperliquid API ────────────────────────────────────────────────────────
HL_INFO_URL = "https://api.hyperliquid.xyz/info"


# ─── Telegram ───────────────────────────────────────────────────────────────

def send_telegram(message: str) -> None:
    """Send a Telegram notification (if configured)."""
    if not config.TELEGRAM_TOKEN or not config.TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id": config.TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
        }, timeout=10)
    except Exception:
        pass  # Never crash because of Telegram


# ─── Data fetching ──────────────────────────────────────────────────────────

def fetch_candles(coin: str, limit: int = 200) -> list[dict] | None:
    """Fetch recent 1h candles from Hyperliquid."""
    try:
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - (limit * 3600 * 1000)

        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": "1h",
                "startTime": start_ms,
                "endTime": end_ms,
            },
        }
        resp = requests.post(HL_INFO_URL, json=payload, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        return [
            {"o": float(c["o"]), "h": float(c["h"]),
             "l": float(c["l"]), "c": float(c["c"]), "v": float(c["v"])}
            for c in raw
        ]
    except Exception as e:
        print(f"  [ERROR] Candles {coin}: {e}")
        return None


def fetch_mid_price(info: Info, coin: str) -> float | None:
    """Get current mid price from Hyperliquid orderbook."""
    try:
        book = info.l2_snapshot(coin)
        if book and book["levels"]:
            best_bid = float(book["levels"][0][0]["px"])
            best_ask = float(book["levels"][1][0]["px"])
            return (best_bid + best_ask) / 2.0
    except Exception:
        pass
    return None


def fetch_funding_rate(coin: str) -> float:
    """Fetch the current funding rate for a coin."""
    try:
        payload = {"type": "metaAndAssetCtxs"}
        resp = requests.post(HL_INFO_URL, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # data is [meta, [asset_ctx, ...]]
        meta = data[0]
        ctxs = data[1]
        for i, asset in enumerate(meta["universe"]):
            if asset["name"] == coin:
                return float(ctxs[i]["funding"])
    except Exception:
        pass
    return 0.0


def fetch_equity(info: Info, wallet: str) -> float:
    """Fetch current account equity."""
    try:
        state = info.user_state(wallet)
        return float(state["marginSummary"]["accountValue"])
    except Exception:
        return 0.0


def fetch_positions(info: Info, wallet: str) -> list[dict]:
    """Fetch open positions from Hyperliquid."""
    try:
        state = info.user_state(wallet)
        positions = []
        for pos in state.get("assetPositions", []):
            p = pos["position"]
            size = float(p["szi"])
            if size != 0:
                positions.append({
                    "coin": p["coin"],
                    "size": size,
                    "entry": float(p["entryPx"]),
                    "side": "long" if size > 0 else "short",
                    "unrealized_pnl": float(p["unrealizedPnl"]),
                })
        return positions
    except Exception:
        return []


# ─── Trading ────────────────────────────────────────────────────────────────

def open_position(exchange: Exchange, coin: str, side: str, qty: float,
                  price: float) -> bool:
    """
    Open a position on Hyperliquid.

    Uses limit order slightly above/below mid for quick fill.
    """
    try:
        is_buy = (side == "long")
        # Slight slippage allowance for quick fill
        slippage = 0.001
        if is_buy:
            limit_price = price * (1 + slippage)
        else:
            limit_price = price * (1 - slippage)

        # Round price to reasonable precision
        if price > 1000:
            limit_price = round(limit_price, 1)
        elif price > 1:
            limit_price = round(limit_price, 4)
        else:
            limit_price = round(limit_price, 6)

        # Round quantity
        if price > 100:
            qty = round(qty, 4)
        else:
            qty = round(qty, 2)

        if qty <= 0:
            return False

        result = exchange.order(
            coin, is_buy, qty, limit_price,
            {"limit": {"tif": "Ioc"}},  # Immediate-or-cancel
        )

        if result["status"] == "ok":
            fills = result["response"]["data"]["statuses"]
            if any("filled" in str(s).lower() or "resting" in str(s).lower() for s in fills):
                return True
        return False

    except Exception as e:
        print(f"  [ERROR] Order {side} {coin}: {e}")
        return False


def close_position(exchange: Exchange, coin: str, size: float,
                   price: float) -> bool:
    """Close an existing position."""
    try:
        is_buy = (size < 0)  # Buy to close short, sell to close long
        qty = abs(size)

        slippage = 0.002
        if is_buy:
            limit_price = price * (1 + slippage)
        else:
            limit_price = price * (1 - slippage)

        if price > 1000:
            limit_price = round(limit_price, 1)
        elif price > 1:
            limit_price = round(limit_price, 4)
        else:
            limit_price = round(limit_price, 6)

        result = exchange.order(
            coin, is_buy, qty, limit_price,
            {"limit": {"tif": "Ioc"}},
            reduce_only=True,
        )

        if result["status"] == "ok":
            return True
        return False

    except Exception as e:
        print(f"  [ERROR] Close {coin}: {e}")
        return False


# ─── AI Prediction ──────────────────────────────────────────────────────────

def predict_signal(
    model: lgb.Booster,
    candles: list[dict],
    btc_candles: list[dict] | None = None,
    funding: float = 0.0,
) -> tuple[str, float]:
    """
    Generate a trading signal from the AI model.

    Returns
    -------
    (signal, confidence)
        signal: "long", "short", or "neutral"
        confidence: float between 0 and 1
    """
    open_ = np.array([c["o"] for c in candles])
    high = np.array([c["h"] for c in candles])
    low = np.array([c["l"] for c in candles])
    close = np.array([c["c"] for c in candles])
    volume = np.array([c["v"] for c in candles])

    btc_close = None
    if btc_candles:
        btc_close = np.array([c["c"] for c in btc_candles])
        # Align lengths
        min_len = min(len(close), len(btc_close))
        close_aligned = close[-min_len:]
        btc_close = btc_close[-min_len:]
        open_ = open_[-min_len:]
        high = high[-min_len:]
        low = low[-min_len:]
        volume = volume[-min_len:]
        close = close_aligned

    features = build_features(open_, high, low, close, volume, btc_close, funding)

    # Use the last row (most recent candle)
    X = features[-1:, :]

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0)

    # Predict probability of price going up
    prob_up = model.predict(X)[0]

    if prob_up > config.MIN_CONFIDENCE:
        return "long", prob_up
    elif prob_up < (1 - config.MIN_CONFIDENCE):
        return "short", 1 - prob_up
    else:
        return "neutral", max(prob_up, 1 - prob_up)


# ─── Main loop ──────────────────────────────────────────────────────────────

class DeepAlpha:
    """Main trading bot orchestrator."""

    def __init__(self):
        # Validate config
        if not config.PRIVATE_KEY:
            raise ValueError("PRIVATE_KEY not set. Copy .env.example to .env and configure it.")
        if not config.WALLET_ADDRESS:
            raise ValueError("WALLET_ADDRESS not set in .env")

        # Load model
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {config.MODEL_PATH}. "
                "Run `python train.py` first."
            )
        with open(config.MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
            if isinstance(model_data, dict):
                self.model = model_data["model"]
            else:
                self.model = model_data  # legacy raw Booster

        # Initialise Hyperliquid connection
        account = Account.from_key(config.PRIVATE_KEY)
        self.info = Info(constants.MAINNET_API_URL, skip_ws=True)
        self.exchange = Exchange(account, constants.MAINNET_API_URL)
        self.wallet = config.WALLET_ADDRESS

        # Set leverage for all coins
        self._set_leverage()

        # Risk manager
        self.risk = RiskManager()

        # Track last retrain time
        self.last_retrain = time.time()

        print("DeepAlpha initialised successfully")
        print(f"  Wallet:    {self.wallet[:8]}...{self.wallet[-4:]}")
        print(f"  Leverage:  {config.LEVERAGE}x")
        print(f"  Max pos:   {config.MAX_POSITIONS}")
        print(f"  Risk/trade:{config.RISK_PER_TRADE*100:.0f}%")
        send_telegram("DeepAlpha bot started")

    def _set_leverage(self) -> None:
        """Set leverage for all traded coins."""
        for coin in config.COINS:
            try:
                self.exchange.update_leverage(
                    config.LEVERAGE, coin, is_cross=True
                )
            except Exception:
                pass

    def _sync_positions(self) -> None:
        """Sync internal position tracker with actual exchange positions."""
        positions = fetch_positions(self.info, self.wallet)
        # Add any positions we don't know about
        for pos in positions:
            coin = pos["coin"]
            if coin not in self.risk.open_positions:
                self.risk.register_open(
                    coin, pos["side"], pos["entry"], abs(pos["size"])
                )
        # Remove positions that no longer exist
        active_coins = {p["coin"] for p in positions}
        for coin in list(self.risk.open_positions.keys()):
            if coin not in active_coins:
                self.risk.open_positions.pop(coin, None)

    def _check_exits(self) -> None:
        """Check all open positions for SL/TP exits."""
        for coin in list(self.risk.open_positions.keys()):
            price = fetch_mid_price(self.info, coin)
            if price is None:
                continue

            exit_reason = self.risk.check_exit(coin, price)
            if exit_reason:
                pos = self.risk.open_positions[coin]
                size = pos["qty"] if pos["side"] == "long" else -pos["qty"]
                success = close_position(self.exchange, coin, size, price)
                if success:
                    pnl = self.risk.register_close(coin, price)
                    emoji = "+" if pnl >= 0 else ""
                    msg = (
                        f"{'STOP LOSS' if exit_reason == 'sl' else 'TAKE PROFIT'} "
                        f"{coin} | {pos['side'].upper()} | "
                        f"Entry: {pos['entry']:.4f} | Exit: {price:.4f} | "
                        f"PnL: {emoji}{pnl:.2f} USD"
                    )
                    print(f"  [EXIT] {msg}")
                    send_telegram(msg)

    def _scan_for_entries(self) -> None:
        """Scan all coins for new entry signals."""
        equity = fetch_equity(self.info, self.wallet)
        if equity <= 0:
            print("  [WARN] Could not fetch equity")
            return

        can_open, reason = self.risk.can_open(equity)
        if not can_open:
            print(f"  [RISK] {reason}")
            return

        # Fetch BTC candles once for correlation feature
        btc_candles = fetch_candles("BTC", 200)

        for coin in config.COINS:
            # Skip if already in a position
            if coin in self.risk.open_positions:
                continue

            # Check again — might have hit max during this scan
            can_open, _ = self.risk.can_open(equity)
            if not can_open:
                break

            candles = fetch_candles(coin, 200)
            if not candles or len(candles) < 50:
                continue

            funding = fetch_funding_rate(coin)
            signal, confidence = predict_signal(
                self.model, candles, btc_candles, funding
            )

            if signal == "neutral":
                continue

            # Get current price
            price = fetch_mid_price(self.info, coin)
            if price is None:
                continue

            # Calculate position size
            qty = self.risk.calc_quantity(equity, price)

            # Execute trade
            success = open_position(self.exchange, coin, signal, qty, price)
            if success:
                self.risk.register_open(coin, signal, price, qty)
                msg = (
                    f"OPEN {signal.upper()} {coin} | "
                    f"Price: {price:.4f} | Qty: {qty:.4f} | "
                    f"Confidence: {confidence:.1%}"
                )
                print(f"  [TRADE] {msg}")
                send_telegram(msg)

            # Small delay between orders
            time.sleep(0.5)

    def _maybe_retrain(self) -> None:
        """Retrain the model periodically (every 2 hours)."""
        if time.time() - self.last_retrain < config.RETRAIN_INTERVAL:
            return

        print("  [RETRAIN] Starting automatic retrain...")
        try:
            # Import and run training inline
            from train import prepare_dataset, train_model
            X, y = prepare_dataset()
            model = train_model(X, y)
            with open(config.MODEL_PATH, "wb") as f:
                pickle.dump(model, f)
            self.model = model
            self.last_retrain = time.time()
            print("  [RETRAIN] Complete")
            send_telegram("Model retrained successfully")
        except Exception as e:
            print(f"  [RETRAIN] Failed: {e}")

    def run(self) -> None:
        """Main trading loop."""
        print("\n" + "=" * 60)
        print("DeepAlpha — Starting trading loop")
        print(f"Scanning {len(config.COINS)} coins every {config.MAIN_LOOP_SECONDS}s")
        print("=" * 60 + "\n")

        while True:
            try:
                loop_start = time.time()
                now = time.strftime("%Y-%m-%d %H:%M:%S")
                equity = fetch_equity(self.info, self.wallet)
                n_pos = len(self.risk.open_positions)

                print(f"[{now}] Equity: ${equity:,.2f} | "
                      f"Positions: {n_pos}/{config.MAX_POSITIONS} | "
                      f"Daily PnL: ${self.risk.daily_pnl:,.2f}")

                # 1. Sync positions with exchange
                self._sync_positions()

                # 2. Check exits (SL/TP)
                self._check_exits()

                # 3. Scan for new entries
                self._scan_for_entries()

                # 4. Maybe retrain
                self._maybe_retrain()

                # Sleep until next iteration
                elapsed = time.time() - loop_start
                sleep_time = max(1, config.MAIN_LOOP_SECONDS - elapsed)
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                print("\nShutting down...")
                send_telegram("DeepAlpha bot stopped")
                break

            except Exception as e:
                print(f"[ERROR] {e}")
                traceback.print_exc()
                time.sleep(30)  # Wait before retrying on error


# ─── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(r"""
    ____                  ___    __      __
   / __ \___  ___  ____  /   |  / /___  / /_  ____ _
  / / / / _ \/ _ \/ __ \/ /| | / / __ \/ __ \/ __ `/
 / /_/ /  __/  __/ /_/ / ___ |/ / /_/ / / / / /_/ /
/_____/\___/\___/ .___/_/  |_/_/ .___/_/ /_/\__,_/
               /_/            /_/
                              Free Edition
    """)
    bot = DeepAlpha()
    bot.run()
