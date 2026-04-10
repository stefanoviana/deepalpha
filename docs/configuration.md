# Configuration

All configuration is managed through environment variables loaded from a `.env` file. Default values are defined in `config.py`.

---

## Credentials

| Variable | Default | Description |
|----------|---------|-------------|
| `PRIVATE_KEY` | `""` | Hyperliquid API private key (required) |
| `WALLET_ADDRESS` | `""` | Your Hyperliquid wallet address (required) |
| `TELEGRAM_TOKEN` | `""` | Telegram bot token for notifications (optional) |
| `TELEGRAM_CHAT_ID` | `""` | Telegram chat ID to receive notifications (optional) |

---

## Trading Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `LEVERAGE` | `5` | Leverage multiplier applied to all positions. Higher leverage amplifies both gains and losses. |
| `MAX_POSITIONS` | `3` | Maximum number of simultaneous open positions. Limits total exposure. |
| `RISK_PER_TRADE` | `0.10` | Fraction of equity allocated per trade (before leverage). At 10% with 5x leverage, each trade uses 50% of equity in notional. |
| `MAX_DAILY_LOSS_PCT` | `0.05` | Maximum daily loss as a fraction of equity. Trading pauses if this threshold is hit. |

---

## Risk Management

| Variable | Default | Description |
|----------|---------|-------------|
| `STOP_LOSS_PCT` | `0.02` | Fixed stop-loss distance from entry price (2%). A long entered at $100 triggers SL at $98. |
| `TAKE_PROFIT_PCT` | `0.03` | Fixed take-profit distance from entry price (3%). A long entered at $100 triggers TP at $103. |
| `CIRCUIT_BREAKER_LOSSES` | `3` | Number of consecutive losing trades before the circuit breaker activates. |
| `CIRCUIT_BREAKER_COOLDOWN` | `3600` | Cooldown period in seconds (1 hour) after the circuit breaker triggers. No new trades during this time. |

---

## AI Model

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `model.pkl` | File path to the trained model. Updated automatically on retrain. |
| `MIN_CONFIDENCE` | `0.58` | Minimum prediction confidence to open a trade. Predictions between 0.42 and 0.58 are classified as "neutral" and skipped. |
| `RETRAIN_INTERVAL` | `7200` | Seconds between automatic retraining cycles (default: 2 hours). The bot retrains using the latest downloaded data. |

---

## Data

| Variable | Default | Description |
|----------|---------|-------------|
| `CANDLE_INTERVAL` | `1h` | Candle timeframe used for both data download and prediction. |
| `DATA_DIR` | `data` | Directory where candle JSON files are stored. |
| `MAIN_LOOP_SECONDS` | `60` | Seconds between each scan iteration. Lower values scan more frequently but increase API calls. |

---

## Coin Universe

The list of coins to scan is defined directly in `config.py`:

```python
COINS: list[str] = [
    "BTC", "ETH", "SOL", "DOGE", "AVAX",
    "LINK", "ARB", "OP", "SUI", "INJ",
    "MATIC", "APT", "SEI", "TIA", "WIF",
    "PEPE", "ONDO", "RENDER", "FET", "JUP",
]
```

To add or remove coins, edit this list and re-run `download_data.py` and `train.py` to incorporate the new data.

---

## Example Configurations

### Conservative

Best for beginners or accounts under $1,000. Minimizes drawdown at the cost of slower growth.

```bash
LEVERAGE=3
MAX_POSITIONS=2
RISK_PER_TRADE=0.05
STOP_LOSS_PCT=0.015
TAKE_PROFIT_PCT=0.025
MAX_DAILY_LOSS_PCT=0.03
MIN_CONFIDENCE=0.62
CIRCUIT_BREAKER_LOSSES=2
CIRCUIT_BREAKER_COOLDOWN=7200
```

Key traits:

- 3x leverage limits max loss per trade
- Only 2 positions open at once
- 5% equity per trade (15% notional at 3x)
- Higher confidence threshold filters out marginal signals
- Circuit breaker trips after just 2 losses, with a 2-hour cooldown

---

### Moderate (Default)

Balanced setup suitable for accounts of $1,000-$5,000.

```bash
LEVERAGE=5
MAX_POSITIONS=3
RISK_PER_TRADE=0.10
STOP_LOSS_PCT=0.02
TAKE_PROFIT_PCT=0.03
MAX_DAILY_LOSS_PCT=0.05
MIN_CONFIDENCE=0.58
CIRCUIT_BREAKER_LOSSES=3
CIRCUIT_BREAKER_COOLDOWN=3600
```

This is the default configuration shipped with DeepAlpha. It provides a 1.5:1 reward-to-risk ratio with moderate position sizing.

---

### Aggressive

For experienced traders with accounts above $5,000 who accept higher drawdowns for faster growth.

```bash
LEVERAGE=7
MAX_POSITIONS=5
RISK_PER_TRADE=0.15
STOP_LOSS_PCT=0.025
TAKE_PROFIT_PCT=0.04
MAX_DAILY_LOSS_PCT=0.08
MIN_CONFIDENCE=0.55
CIRCUIT_BREAKER_LOSSES=4
CIRCUIT_BREAKER_COOLDOWN=1800
```

Key traits:

- 7x leverage amplifies returns (and losses)
- Up to 5 simultaneous positions
- Lower confidence threshold opens more trades
- Wider stop-loss gives trades more room to breathe
- Higher daily loss limit before stopping

!!! warning
    Aggressive settings can produce significant drawdowns. Only use these if you fully understand leverage risk and have tested on paper first.

---

## Pro Version Parameters

The [Pro version](https://stefanocrypto.gumroad.com/l/ezilv) adds these additional configuration options:

| Variable | Description |
|----------|-------------|
| `MAX_RISK_PER_TRADE_PCT` | Dynamic position sizing as % of equity (replaces fixed margin) |
| `MAX_TOTAL_EXPOSURE_PCT` | Maximum total portfolio exposure as % of equity |
| `HARD_SL_PCT` | Hard price stop-loss -- absolute max loss per position (e.g., 1% price = 5% at 5x) |
| `TRAIL_TIER1_PCT` | Trailing TP tier 1: move SL to breakeven when profit exceeds this level |
| `TRAIL_TIER2_PCT` | Trailing TP tier 2: lock partial profit when profit exceeds this level |
| `COIN_BLACKLIST` | Comma-separated list of coins to exclude from scanning |
| `GLOBAL_OPEN_COOLDOWN` | Minimum seconds between opening any two positions |
