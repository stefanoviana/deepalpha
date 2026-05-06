"""
DeepAlpha V11.0 — AI Trading Bot
Autonomous crypto trading on Bybit & Binance.

Website:  https://deepalphabot.com
Cloud:    https://deepalphabot.com/cloud (7-day free trial)
GitHub:   https://github.com/stefanoviana/deepalpha
Discord:  https://discord.gg/P4yX686m
Support:  https://deepalphabot.com/cloud

Usage:
    1. Create .env file with your API keys (see .env.example)
    2. python deepalpha.py
"""

# --- Dependency Check --------------------------------------------------------
import sys

_REQUIRED = {
    "lightgbm": "lightgbm",
    "ccxt": "ccxt",
    "numpy": "numpy",
    "xgboost": "xgboost",
    "sklearn": "scikit-learn",
    "dotenv": "python-dotenv",
}
_missing = []
for _mod, _pkg in _REQUIRED.items():
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_pkg)
if _missing:
    print("[ERROR] Missing dependencies: " + ", ".join(_missing))
    print("[ERROR] Run:  pip install -r requirements.txt")
    print("[ERROR] Or:   pip install " + " ".join(_missing))
    sys.exit(1)

# --- .env Check --------------------------------------------------------------
import os as _os
if not _os.path.exists(".env") and not _os.environ.get("EXCHANGE"):
    print("[ERROR] No .env file found in current directory.")
    print("[ERROR] Copy the example:  cp .env.example .env")
    print("[ERROR] Then edit it with your API keys (see .env.example for details).")
    print("[ERROR] Docs: https://deepalphabot.com/setup-guide")
    sys.exit(1)

import os
import pickle
import time
import traceback
import numpy as np
import requests
import lightgbm as lgb

import hashlib
import platform
import uuid

import config
from features import build_features, FEATURE_NAMES
from risk_manager import RiskManager
from exchange_adapter import ExchangeAdapter, get_exchange


def _show_banner():
    print("\n" + "=" * 55)
    print("  DeepAlpha V11.0 — AI Crypto Trading Bot")
    print("  70.9% accuracy | 72 ML features | Walk-forward validated")
    print("=" * 55)
    print("  Website:  https://deepalphabot.com")
    print("  Cloud:    https://deepalphabot.com/cloud")
    print("  GitHub:   https://github.com/stefanoviana/deepalpha")
    print("  Discord:  https://discord.gg/P4yX686m")
    print("-" * 55)
    print("  Pro: $39/mo | Lifetime: $199 | Free 7-day trial")
    print("=" * 55 + "\n")

_show_banner()


# ─── License & Model Update ───────────────────────────────────────────────

def get_machine_id() -> str:
    """Generate a unique machine identifier for license binding."""
    node = uuid.getnode()
    system = platform.system()
    return hashlib.sha256(f"{node}-{system}".encode()).hexdigest()[:32]


def verify_license() -> dict:
    """Verify license key with the server. Returns license info or exits."""
    if not config.LICENSE_KEY:
        print("[LICENSE] No LICENSE_KEY in .env — running in free mode (limited features)")
        return {"valid": False, "plan": "free"}
    try:
        resp = requests.post(
            f"{config.LICENSE_SERVER}/verify",
            json={"key": config.LICENSE_KEY, "machine_id": get_machine_id()},
            timeout=15,
        )
        data = resp.json()
        if data.get("valid"):
            print(f"[LICENSE] Valid — plan: {data['plan']}, expires in {data.get('days_remaining', '?')} days")
            return data
        else:
            print(f"[LICENSE] Invalid: {data.get('error', 'unknown')}")
            print("[LICENSE] Get a license at https://deepalphabot.com")
            return {"valid": False, "plan": "free"}
    except Exception as e:
        print(f"[LICENSE] Server unreachable ({e}) — continuing with local model")
        return {"valid": True, "plan": "offline"}


def _ping_usage():
    """Anonymous usage ping — helps us understand adoption. No personal data sent."""
    if config.DISABLE_TELEMETRY:
        return
    try:
        data = {
            "v": "11.0",
            "os": platform.system(),
            "py": platform.python_version(),
            "mid": get_machine_id()[:8],
            "exchange": getattr(config, "EXCHANGE", "unknown"),
            "plan": "free",
        }
        requests.post("https://deepalphabot.com/cloud/api/health", json=data, timeout=5)
        # Notify via server (no tokens in client code)
        requests.post("https://deepalphabot.com/api/telemetry", json=data, timeout=5)
        print("[INFO] Usage ping sent (anonymous, no personal data)")
        print("[INFO] Need help? https://deepalphabot.com (live chat) | https://t.me/DeepAlphaVault_bot")
    except Exception:
        pass  # Never crash for telemetry


def update_model(horizon: str = "1h") -> bool:
    """Download latest AI model from the license server."""
    if not config.LICENSE_KEY:
        return False
    try:
        print(f"[MODEL] Checking for {horizon} model update...")
        resp = requests.post(
            f"{config.LICENSE_SERVER}/model/{horizon}",
            json={"key": config.LICENSE_KEY, "machine_id": get_machine_id()},
            timeout=30,
        )
        if resp.status_code == 200:
            model_path = config.MODEL_PATH
            # Backup old model
            if os.path.exists(model_path):
                os.rename(model_path, model_path + ".bak")
            with open(model_path, "wb") as f:
                f.write(resp.content)
            print(f"[MODEL] Updated {horizon} model ({len(resp.content)//1024}KB)")
            return True
        else:
            print(f"[MODEL] No update available ({resp.status_code})")
            return False
    except Exception as e:
        print(f"[MODEL] Update failed: {e}")
        return False


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
        # Verify license
        self.license = verify_license()

        # Anonymous usage ping
        _ping_usage()

        # Try to download latest model from server
        if self.license.get("valid") and config.LICENSE_KEY:
            update_model("1h")

        # Load model
        if not os.path.exists(config.MODEL_PATH):
            print("")
            print("=" * 55)
            print("  MODEL NOT FOUND")
            print("=" * 55)
            print(f"  File: {config.MODEL_PATH}")
            print("")
            print("  Options:")
            print("  1. Train your own: python train.py")
            print("  2. Use the cloud platform (no setup, AI ready):")
            print("     https://deepalphabot.com  (7-day free trial)")
            print("")
            print("  The cloud version includes:")
            print("  - Pre-trained AI model (70.9% accuracy)")
            print("  - Grid Bot + DCA Bot + 10 strategies")
            print("  - 12 exchanges, Telegram bot control")
            print("  - No installation needed")
            print("")
            print("  Need help?")
            print("  Chat:     https://deepalphabot.com (live chat)")
            print("  Telegram: https://t.me/DeepAlphaVault_bot")
            print("  Discord:  https://discord.gg/P4yX686m")
            print("=" * 55)
            raise FileNotFoundError(f"Model not found at {config.MODEL_PATH}")
        with open(config.MODEL_PATH, "rb") as f:
            # SECURITY WARNING: Loading models via pickle is insecure.
            # Only use models from trusted sources (api.deepalphabot.com).
            model_data = pickle.load(f)
            if isinstance(model_data, dict):
                self.model = model_data["model"]
                self.selected_features = model_data.get("selected_feature_indices")
            else:
                self.model = model_data
                self.selected_features = None

        # Initialise exchange via the adapter layer
        self.exchange: ExchangeAdapter = get_exchange(config.EXCHANGE)
        self.exchange.connect()

        # Set leverage for all coins
        self._set_leverage()

        # Risk manager
        self.risk = RiskManager()

        # Track model update time
        self.last_model_check = time.time()

        print("DeepAlpha initialised successfully")
        print(f"  Exchange:  {config.EXCHANGE}")
        print(f"  Leverage:  {config.LEVERAGE}x")
        print(f"  Max pos:   {config.MAX_POSITIONS}")
        print(f"  Coins:     {len(config.COINS)}")
        print(f"  License:   {self.license.get('plan', 'free')}")
        send_telegram(f"DeepAlpha started ({config.EXCHANGE}, {self.license.get('plan', 'free')})")

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

    def _maybe_update_model(self) -> None:
        """Check for model updates from the license server."""
        update_interval = config.MODEL_UPDATE_HOURS * 3600
        if time.time() - self.last_model_check < update_interval:
            return
        self.last_model_check = time.time()
        if not config.LICENSE_KEY:
            return
        try:
            if update_model("1h"):
                with open(config.MODEL_PATH, "rb") as f:
                    # SECURITY WARNING: Loading models via pickle is insecure.
                    # Only use models from trusted sources (api.deepalphabot.com).
                    model_data = pickle.load(f)
                    if isinstance(model_data, dict):
                        self.model = model_data["model"]
                        self.selected_features = model_data.get("selected_feature_indices")
                    else:
                        self.model = model_data
                send_telegram("AI model auto-updated to latest version")
        except Exception as e:
            print(f"  [MODEL] Update check failed: {e}")

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

                # 4. Check for model updates
                self._maybe_update_model()

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
    try:
        bot = DeepAlpha()
        bot.run()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("")
        print("=" * 60)
        print("  HOW TO FIX: Download or train a model first.")
        print("  Option 1:  python train.py")
        print("  Option 2:  Use cloud (no setup): https://deepalphabot.com")
        print("=" * 60)
        time.sleep(30)
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Bot crashed: {e}")
        traceback.print_exc()
        print("")
        print("=" * 60)
        print("  COMMON FIXES:")
        print("  1. Check your .env file (see .env.example)")
        print("  2. Run: pip install -r requirements.txt")
        print("  3. Make sure you have model files (.pkl) in the same directory")
        print("  4. Need help?")
        print("     - Setup guide: https://deepalphabot.com/setup-guide")
        print("     - Telegram:    https://t.me/DeepAlphaVault_bot")
        print("     - Discord:     https://discord.gg/P4yX686m")
        print("=" * 60)
        # Sleep before exit to prevent crash-loop spam (PM2, Docker, systemd)
        print("")
        print("[INFO] Waiting 30s before exit to prevent crash-loop spam...")
        time.sleep(30)
        sys.exit(1)
