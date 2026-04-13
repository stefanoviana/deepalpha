"""
DeepAlpha — AI Trading Bot (Free Version)
Autonomous trading on Hyperliquid, Binance & Bybit using a LightGBM model.

Usage:
    python download_data.py   # download historical data
    python train.py           # train the AI model
    python deepalpha.py       # start trading

Environment variables: see .env.example
Set EXCHANGE=hyperliquid|binance|bybit in .env to choose your exchange.
"""

import os
import pickle
import time
import traceback
import numpy as np
import requests
import lightgbm as lgb

import config
from features import build_features, FEATURE_NAMES
from risk_manager import RiskManager
from exchange_adapter import ExchangeAdapter, get_exchange


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
    """Main trading bot orchestrator.  Works with any supported exchange."""

    def __init__(self):
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

        # Initialise exchange via the adapter layer
        self.exchange: ExchangeAdapter = get_exchange(config.EXCHANGE)
        self.exchange.connect()

        # Set leverage for all coins
        self._set_leverage()

        # Risk manager
        self.risk = RiskManager()

        # Track last retrain time
        self.last_retrain = time.time()

        print("DeepAlpha initialised successfully")
        print(f"  Exchange:  {config.EXCHANGE}")
        print(f"  Leverage:  {config.LEVERAGE}x")
        print(f"  Max pos:   {config.MAX_POSITIONS}")
        print(f"  Risk/trade:{config.RISK_PER_TRADE*100:.0f}%")
        send_telegram(f"DeepAlpha bot started ({config.EXCHANGE})")

    def _set_leverage(self) -> None:
        """Set leverage for all traded coins."""
        for coin in config.COINS:
            try:
                self.exchange.set_leverage(coin, config.LEVERAGE)
            except Exception:
                pass

    def _sync_positions(self) -> None:
        """Sync internal position tracker with actual exchange positions."""
        positions = self.exchange.get_positions()
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
            try:
                book = self.exchange.get_orderbook(coin)
                price = book["mid"]
            except Exception:
                continue

            exit_reason = self.risk.check_exit(coin, price)
            if exit_reason:
                pos = self.risk.open_positions[coin]
                result = self.exchange.close_position(coin)
                if result.get("success"):
                    pnl = self.risk.register_close(coin, price)
                    sign = "+" if pnl >= 0 else ""
                    msg = (
                        f"{'STOP LOSS' if exit_reason == 'sl' else 'TAKE PROFIT'} "
                        f"{coin} | {pos['side'].upper()} | "
                        f"Entry: {pos['entry']:.4f} | Exit: {price:.4f} | "
                        f"PnL: {sign}{pnl:.2f} USD"
                    )
                    print(f"  [EXIT] {msg}")
                    send_telegram(msg)

    def _scan_for_entries(self) -> None:
        """Scan all coins for new entry signals."""
        equity = self.exchange.get_balance()
        if equity <= 0:
            print("  [WARN] Could not fetch equity")
            return

        can_open, reason = self.risk.can_open(equity)
        if not can_open:
            print(f"  [RISK] {reason}")
            return

        # Fetch BTC candles once for correlation feature
        try:
            btc_candles = self.exchange.get_candles("BTC", "1h", 200)
        except Exception:
            btc_candles = None

        for coin in config.COINS:
            # Skip if already in a position
            if coin in self.risk.open_positions:
                continue

            # Check again — might have hit max during this scan
            can_open, _ = self.risk.can_open(equity)
            if not can_open:
                break

            try:
                candles = self.exchange.get_candles(coin, "1h", 200)
            except Exception:
                continue
            if not candles or len(candles) < 50:
                continue

            try:
                funding = self.exchange.get_funding_rate(coin)
            except Exception:
                funding = 0.0

            signal, confidence = predict_signal(
                self.model, candles, btc_candles, funding
            )

            if signal == "neutral":
                continue

            # Get current price
            try:
                book = self.exchange.get_orderbook(coin)
                price = book["mid"]
            except Exception:
                continue

            # Calculate position size
            qty = self.risk.calc_quantity(equity, price)

            # Execute trade via adapter
            side_str = "buy" if signal == "long" else "sell"
            result = self.exchange.place_market_order(coin, side_str, qty)
            success = result.get("success", False)

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
        print(f"DeepAlpha — Starting trading loop ({config.EXCHANGE})")
        print(f"Scanning {len(config.COINS)} coins every {config.MAIN_LOOP_SECONDS}s")
        print("=" * 60 + "\n")

        while True:
            try:
                loop_start = time.time()
                now = time.strftime("%Y-%m-%d %H:%M:%S")
                equity = self.exchange.get_balance()
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
