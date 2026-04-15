"""
DeepAlpha — Configuration
Loads all settings from environment variables with sensible defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── Exchange selection ───────────────────────────────────────────────────
# Supported values: "hyperliquid", "bitget", "binance", "bybit"
EXCHANGE: str = os.getenv("EXCHANGE", "bitget")

# ─── Hyperliquid credentials ───────────────────────────────────────────────
PRIVATE_KEY: str = os.getenv("PRIVATE_KEY", "")
WALLET_ADDRESS: str = os.getenv("WALLET_ADDRESS", "")

# ─── Binance Futures credentials ──────────────────────────────────────────
BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")
BINANCE_TESTNET: bool = os.getenv("BINANCE_TESTNET", "false").lower() == "true"

# ─── Bitget credentials ──────────────────────────────────────────────────
BITGET_API_KEY: str = os.getenv("BITGET_API_KEY", "")
BITGET_SECRET: str = os.getenv("BITGET_SECRET", "")
BITGET_PASSPHRASE: str = os.getenv("BITGET_PASSPHRASE", "")

# ─── Bybit credentials ───────────────────────────────────────────────────
BYBIT_API_KEY: str = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET: str = os.getenv("BYBIT_API_SECRET", "")
BYBIT_TESTNET: bool = os.getenv("BYBIT_TESTNET", "false").lower() == "true"

# ─── Telegram notifications (optional) ─────────────────────────────────────
TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ─── Fee rates (per side) ─────────────────────────────────────────────────
FEE_RATE: float = 0.0006 if EXCHANGE == "bitget" else 0.00035       # taker fee
MAKER_FEE_RATE: float = 0.0002 if EXCHANGE == "bitget" else 0.0002  # maker fee

# ─── Trading parameters ────────────────────────────────────────────────────
LEVERAGE: int = int(os.getenv("LEVERAGE", "5"))
MAX_POSITIONS: int = int(os.getenv("MAX_POSITIONS", "3"))
RISK_PER_TRADE: float = float(os.getenv("RISK_PER_TRADE", "0.10"))       # 10% of equity
MAX_DAILY_LOSS_PCT: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.05"))  # 5% daily max loss

# ─── Risk management ───────────────────────────────────────────────────────
STOP_LOSS_PCT: float = float(os.getenv("STOP_LOSS_PCT", "0.02"))        # 2% fixed SL
TAKE_PROFIT_PCT: float = float(os.getenv("TAKE_PROFIT_PCT", "0.03"))     # 3.0% fixed TP
CIRCUIT_BREAKER_LOSSES: int = int(os.getenv("CIRCUIT_BREAKER_LOSSES", "3"))
CIRCUIT_BREAKER_COOLDOWN: int = int(os.getenv("CIRCUIT_BREAKER_COOLDOWN", "3600"))  # seconds

# ─── License & Model Server ───────────────────────────────────────────────
LICENSE_KEY: str = os.getenv("LICENSE_KEY", "")
LICENSE_SERVER: str = os.getenv("LICENSE_SERVER", "http://217.15.163.134:8090")
MODEL_UPDATE_HOURS: int = int(os.getenv("MODEL_UPDATE_HOURS", "24"))  # check for new models every N hours

# ─── Model / AI ────────────────────────────────────────────────────────────
MODEL_PATH: str = os.getenv("MODEL_PATH", "model.pkl")
MIN_CONFIDENCE: float = float(os.getenv("MIN_CONFIDENCE", "0.58"))       # minimum prediction confidence

# ─── Data ───────────────────────────────────────────────────────────────────
CANDLE_INTERVAL: str = os.getenv("CANDLE_INTERVAL", "1h")
DATA_DIR: str = os.getenv("DATA_DIR", "data")

# ─── Coins to trade (must match trained model) ───────────────────────────
COINS: list[str] = [
    "BTC", "ETH", "SOL", "BNB", "DOGE", "AVAX", "LINK", "ARB", "OP", "APT",
    "SUI", "INJ", "TIA", "WLD", "NEAR", "FET", "AAVE", "DOT", "ADA", "XRP",
    "LTC", "BCH", "CRV", "ONDO", "ENA", "JUP", "RENDER",
]

# ─── Loop timing ───────────────────────────────────────────────────────────
MAIN_LOOP_SECONDS: int = int(os.getenv("MAIN_LOOP_SECONDS", "60"))
