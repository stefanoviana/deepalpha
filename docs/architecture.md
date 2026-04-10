# Architecture

This page describes the system design of DeepAlpha, including data flow, component responsibilities, and the execution model.

---

## System Overview

```
+-------------------+      +-------------------+      +------------------+
|   Hyperliquid     |      |    DeepAlpha       |      |   Outputs        |
|   Public API      |----->|    Main Loop       |----->|                  |
|                   |      |                    |      |  Trade Execution |
|  - Candles (1h)   |      |  1. Sync positions |      |  Telegram Alerts |
|  - Orderbook L2   |      |  2. Check SL/TP    |      |  Console Logs    |
|  - Funding rates  |      |  3. Scan entries   |      |                  |
|  - Account state  |      |  4. Auto-retrain   |      +------------------+
+-------------------+      +-------------------+
                                    |
                    +---------------+---------------+
                    |               |               |
              +-----v----+   +-----v----+   +------v-----+
              | features |   |  train   |   |    risk     |
              |   .py    |   |   .py    |   |  manager   |
              |          |   |          |   |    .py     |
              | 15 tech  |   | LightGBM |   | SL/TP     |
              | features |   | training |   | Sizing    |
              +----------+   +----------+   | Circuit   |
                                            | breaker   |
                                            +------------+
```

---

## Data Flow

The system follows a linear pipeline from raw market data to trade execution:

```
Market Data (Hyperliquid API)
         |
         v
   OHLCV Candles (200 x 1h per coin)
         |
         v
   Feature Engineering (15 features per candle)
         |
         v
   ML Model Prediction (LightGBM)
         |
         v
   Signal: LONG / SHORT / NEUTRAL + confidence %
         |
         v
   Risk Check (max positions, daily loss, circuit breaker)
         |
         v
   Position Sizing (equity * risk_per_trade * leverage)
         |
         v
   Order Execution (limit IOC via Hyperliquid SDK)
         |
         v
   Position Monitoring (SL/TP check every 60s)
```

---

## Components

### `deepalpha.py` -- Main Trading Bot

The central orchestrator. Contains the `DeepAlpha` class which:

- Initializes the Hyperliquid SDK connection and loads the trained model
- Sets leverage for all configured coins on startup
- Runs the main trading loop with four phases per iteration:
    1. **Sync positions** -- reconcile internal state with the exchange
    2. **Check exits** -- evaluate all open positions against SL/TP levels
    3. **Scan for entries** -- fetch candles for each coin, generate predictions, open trades
    4. **Auto-retrain** -- periodically retrain the model on fresh data

Key functions:

| Function | Purpose |
|----------|---------|
| `fetch_candles(coin, limit)` | Fetch 1h OHLCV data from Hyperliquid |
| `fetch_mid_price(info, coin)` | Get current mid price from L2 orderbook |
| `fetch_funding_rate(coin)` | Get the current funding rate |
| `fetch_equity(info, wallet)` | Get account equity |
| `fetch_positions(info, wallet)` | Get all open positions |
| `predict_signal(model, candles, ...)` | Generate LONG/SHORT/NEUTRAL signal from AI model |
| `open_position(exchange, coin, side, qty, price)` | Place a limit IOC order |
| `close_position(exchange, coin, size, price)` | Close with reduce-only limit IOC |

---

### `train.py` -- Training Pipeline

Trains a LightGBM binary classifier to predict whether price will go up or down over the next 3 candles.

Process:

1. Load JSON candle files from `data/`
2. Compute features using `build_features()` from `features.py`
3. Generate labels: `1` if `close[i + 3] > close[i]`, else `0`
4. Trim warmup rows (first 30) and invalid trailing labels
5. Split chronologically: 70% train / 15% validation / 15% test
6. Train with early stopping on validation loss (max 2000 rounds, stops after 50 rounds of no improvement)
7. Report test accuracy and feature importance
8. Save model to `model.pkl`

LightGBM hyperparameters:

```python
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 50,
}
```

---

### `features.py` -- Feature Engineering

Computes 15 technical features from raw OHLCV arrays. All features are normalized (ratios, percentages, or bounded values) to help the model generalize across different price scales.

Input: arrays of open, high, low, close, volume (all same length), optional BTC close prices, and a scalar funding rate.

Output: a `(N, 15)` NumPy matrix.

See [Features](features.md) for the full list and descriptions.

---

### `risk_manager.py` -- Risk Management

The `RiskManager` class enforces all risk rules:

- **Position sizing**: `equity * RISK_PER_TRADE * LEVERAGE / price` = quantity
- **Stop-loss / Take-profit**: Fixed percentage from entry, calculated at position open
- **Max positions**: Blocks new trades when `MAX_POSITIONS` is reached
- **Daily loss limit**: Blocks new trades when daily realized PnL exceeds `MAX_DAILY_LOSS_PCT` of equity
- **Circuit breaker**: After `CIRCUIT_BREAKER_LOSSES` consecutive losses, pauses all trading for `CIRCUIT_BREAKER_COOLDOWN` seconds
- **Daily reset**: PnL counters reset every 24 hours

---

### `download_data.py` -- Data Downloader

Downloads historical 1h candle data from Hyperliquid's public API for all coins in `config.COINS`. Saves each coin as a separate JSON file in the `data/` directory.

- Downloads 365 days of history by default
- No API key required
- Rate-limited with 0.5s delay between coins

---

### `config.py` -- Configuration

Loads all settings from environment variables via `python-dotenv`. Provides sensible defaults for every parameter so the bot can run with just `PRIVATE_KEY` and `WALLET_ADDRESS` set.

---

## Execution Model

DeepAlpha runs as a single-threaded Python process with a synchronous main loop:

```
while True:
    sync_positions()      # ~1 API call
    check_exits()         # ~1 API call per open position
    scan_for_entries()    # ~2 API calls per coin (candles + price)
    maybe_retrain()       # CPU-bound, every 2 hours
    sleep(remaining_time) # Fill up to MAIN_LOOP_SECONDS
```

Each iteration takes approximately 30-45 seconds depending on the number of coins being scanned and network latency. The default 60-second loop interval provides ample time for completion.

### API Call Budget

Per iteration (worst case with 20 coins):

| Operation | API Calls |
|-----------|-----------|
| Sync positions | 1 |
| Check exits (3 positions) | 3 |
| BTC candles (once) | 1 |
| Coin candles (20 coins) | 20 |
| Funding rates (20 coins) | 20 |
| Mid prices (20 coins) | 20 |
| Equity check | 1 |
| **Total** | **~66** |

At 60-second intervals, this is roughly 1 request per second -- well within Hyperliquid's rate limits.

---

## File Structure

```
deepalpha/
├── deepalpha.py          # Main trading bot (DeepAlpha class + main loop)
├── train.py              # AI training pipeline (LightGBM)
├── download_data.py      # Historical data downloader
├── features.py           # Feature engineering (15 features)
├── risk_manager.py       # Position sizing, SL/TP, circuit breaker
├── config.py             # Configuration (env vars + defaults)
├── requirements.txt      # Python dependencies
├── .env.example          # Template for credentials
├── model.pkl             # Trained model (generated by train.py)
└── data/                 # Candle data (generated by download_data.py)
    ├── BTC_1h.json
    ├── ETH_1h.json
    └── ...
```
