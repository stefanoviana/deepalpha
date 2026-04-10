# Strategies

DeepAlpha uses a single strategy in the free version: **AI Autonomous**. This page explains how the strategy works end-to-end, from data ingestion to trade execution.

---

## AI Autonomous Strategy

The AI Autonomous strategy is fully data-driven. It does not rely on hard-coded rules like "buy when RSI < 30." Instead, a machine learning model learns which combinations of features historically predict upward or downward price movement, and the bot acts on those predictions.

### How Predictions Work

The prediction pipeline runs once per coin per scan cycle (every 60 seconds by default):

1. **Fetch candles** -- Download the most recent 200 hourly candles for the coin.
2. **Build features** -- Compute 15 technical features from the OHLCV data (see [Features](features.md)).
3. **Extract the latest row** -- Take only the most recent candle's feature vector (shape: `1 x 15`).
4. **Model inference** -- Pass the feature vector to the LightGBM model, which outputs `prob_up`: the predicted probability that the price will rise over the next 3 candles.
5. **Classify signal**:
    - If `prob_up > MIN_CONFIDENCE` (default 0.58): signal = **LONG**
    - If `prob_up < 1 - MIN_CONFIDENCE` (i.e., < 0.42): signal = **SHORT**
    - Otherwise: signal = **NEUTRAL** (no trade)

```python
prob_up = model.predict(features[-1:, :])

if prob_up > 0.58:
    signal = "long"
elif prob_up < 0.42:
    signal = "short"
else:
    signal = "neutral"
```

The confidence threshold creates a dead zone around 0.50, filtering out low-conviction predictions. Only clear directional signals trigger trades.

---

### LightGBM Model

DeepAlpha uses [LightGBM](https://lightgbm.readthedocs.io), a gradient boosting framework that builds an ensemble of decision trees.

Why LightGBM:

- **Fast training** -- Trains on 170,000+ samples in under 5 minutes.
- **Handles mixed feature types** -- Works well with the combination of bounded (RSI), unbounded (momentum), and ratio features.
- **Built-in regularization** -- Feature fraction, bagging, and min child samples prevent overfitting.
- **Early stopping** -- Training halts automatically when validation loss stops improving.

Training configuration:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `objective` | `binary` | Binary classification (up vs. down) |
| `metric` | `binary_logloss` | Optimize log loss |
| `num_leaves` | `63` | Tree complexity (moderate) |
| `learning_rate` | `0.05` | Conservative step size |
| `feature_fraction` | `0.8` | Use 80% of features per tree (reduces overfitting) |
| `bagging_fraction` | `0.8` | Use 80% of data per tree (reduces overfitting) |
| `min_child_samples` | `50` | Minimum samples per leaf (prevents noise fitting) |
| `early_stopping` | `50 rounds` | Stop if validation loss doesn't improve for 50 rounds |

---

### Walk-Forward Validation

DeepAlpha uses chronological (walk-forward) data splitting, not random splitting. This is critical for financial data where temporal patterns exist.

```
|<---- 70% Train ---->|<-- 15% Val -->|<-- 15% Test -->|
  oldest data                                newest data
       |                    |                    |
  Model learns         Early stopping      Final evaluation
  from this data       monitors this        on unseen data
```

Random splitting would leak future information into training data (e.g., the model could learn that "BTC at $60,000 goes up" using data from after the prediction target). Walk-forward splitting prevents this by ensuring the model only trains on data older than its validation and test sets.

---

### Self-Learning (Auto-Retrain)

The bot periodically retrains its model to adapt to changing market conditions:

1. Every `RETRAIN_INTERVAL` seconds (default: 7200 = 2 hours), the bot checks if a retrain is due.
2. It imports the training functions from `train.py` and runs the full pipeline inline.
3. The new model replaces the old one in memory and is saved to disk.
4. A Telegram notification confirms the retrain.

This keeps the model current without manual intervention. If retraining fails (e.g., corrupt data), the bot continues using the previous model.

---

## Order Execution

### Limit IOC Orders

All orders are placed as **Limit Immediate-or-Cancel (IOC)** with a small slippage allowance:

- **Entry orders**: limit price = mid price +/- 0.1% (buys slightly above mid, sells slightly below)
- **Exit orders**: limit price = mid price +/- 0.2% (wider slippage for urgent exits)

IOC means the order fills immediately at the limit price or better, and any unfilled portion is cancelled. This avoids leaving stale resting orders on the book.

```python
result = exchange.order(
    coin, is_buy, qty, limit_price,
    {"limit": {"tif": "Ioc"}},
)
```

### Position Sizing

Each trade is sized based on account equity:

```
notional = equity * RISK_PER_TRADE * LEVERAGE
quantity = notional / current_price
```

With defaults (10% risk, 5x leverage), a $1,000 account opens positions worth $500 in notional value ($100 in margin).

---

## Exit Logic

### Fixed Stop-Loss and Take-Profit

Every position gets a fixed SL and TP at open time:

| Side | Stop-Loss | Take-Profit |
|------|-----------|-------------|
| Long | `entry * (1 - STOP_LOSS_PCT)` | `entry * (1 + TAKE_PROFIT_PCT)` |
| Short | `entry * (1 + STOP_LOSS_PCT)` | `entry * (1 - TAKE_PROFIT_PCT)` |

With defaults (2% SL, 3% TP), the reward-to-risk ratio is 1.5:1.

### Monitoring

The bot checks all open positions against their SL/TP levels every 60 seconds. If the current mid price crosses either level, the position is closed with a reduce-only limit IOC order.

---

## Circuit Breaker

After `CIRCUIT_BREAKER_LOSSES` consecutive losing trades (default: 3), the bot activates a cooldown:

- No new positions are opened for `CIRCUIT_BREAKER_COOLDOWN` seconds (default: 3600 = 1 hour).
- Existing positions continue to be monitored for SL/TP exits.
- After the cooldown, the consecutive loss counter resets and trading resumes.
- Any winning trade resets the consecutive loss counter immediately.

---

## Pro Version Strategies

The [Pro version](https://stefanocrypto.gumroad.com/l/ezilv) adds additional strategies:

| Strategy | Description |
|----------|-------------|
| **XGBoost + LightGBM Ensemble** | Averages predictions from two different model architectures for higher accuracy (72.9% on test data) |
| **PPO Reinforcement Learning** | Proximal Policy Optimization agent that learns optimal position sizing and entry timing from simulated trading |
| **BTC Regime Filter** | Analyzes BTC trend to filter trades. In strong BTC downtrends, only SHORT signals are allowed. In strong uptrends, only LONG. Prevents trading against the dominant market direction. |
| **ATR Dynamic Trailing Stop** | Replaces fixed SL/TP with ATR-based levels that adapt to current volatility. Tight stops in calm markets, wide stops in volatile markets. |
| **Trailing Take-Profit Tiers** | Multi-tier system: at +1.5% profit, move SL to breakeven. At +3% profit, lock +1.5% profit. Lets winners run while securing gains. |
| **Order Book Scalping** | WebSocket-based L2 order book analysis with CVD confirmation and anti-spoofing filters for short-term trades |
| **Maker-Only Orders** | Posts limit orders at favorable prices instead of IOC, reducing fees from taker (0.035%) to maker (0.01%) or even negative maker rebates |
