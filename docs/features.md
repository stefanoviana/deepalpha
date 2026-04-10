# Features

DeepAlpha's free version computes 15 technical features from raw OHLCV candle data. All features are normalized so the model can generalize across coins with different price scales.

---

## Feature List

### Basic Price Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | `rsi_14` | Relative Strength Index over 14 periods. Measures whether an asset is overbought (>70) or oversold (<30). Bounded between 0 and 100. |
| 2 | `atr_14` | Average True Range over 14 periods, normalized by close price. Measures current volatility as a fraction of price. Higher values indicate more volatile conditions. |
| 3 | `ema_12_26_diff` | Difference between the 12-period and 26-period exponential moving averages, normalized by close price. Positive values indicate bullish momentum (fast EMA above slow EMA), negative values indicate bearish momentum. This is essentially a normalized MACD line. |

---

### Momentum Features

| # | Feature | Description |
|---|---------|-------------|
| 4 | `price_momentum_3` | Percentage price change over the last 3 candles. Captures short-term directional momentum. |
| 5 | `price_momentum_7` | Percentage price change over the last 7 candles. Captures medium-term directional momentum. Comparing 3-period and 7-period momentum reveals whether a move is accelerating or decelerating. |

---

### Volume Features

| # | Feature | Description |
|---|---------|-------------|
| 6 | `volume_ma_ratio` | Current volume divided by the 20-period simple moving average of volume. Values above 1.0 indicate above-average volume (potential breakout or capitulation). Values below 1.0 indicate quiet markets. |
| 7 | `volume_change_pct` | Percentage change in volume from the prior candle. Sudden volume spikes often precede large price moves. |

---

### Volatility Features

| # | Feature | Description |
|---|---------|-------------|
| 8 | `high_low_range` | `(High - Low) / Close` for each candle. Measures intra-candle volatility. Wide ranges suggest high volatility or potential reversal. |
| 9 | `candle_body_ratio` | `|Close - Open| / (High - Low)`. Ratio of the candle body to the total range (body + wicks). Values near 1.0 indicate strong directional moves. Values near 0.0 indicate indecision (long wicks, small body -- doji patterns). |

---

### Price Structure Features

| # | Feature | Description |
|---|---------|-------------|
| 10 | `close_vs_open` | `(Close - Open) / Open`. The directional movement of the candle as a percentage. Positive = green candle, negative = red candle. |
| 11 | `price_vs_vwap` | Price relative to the session VWAP (Volume Weighted Average Price). `(Close - VWAP) / VWAP`. Positive values mean price is above average transaction price (bullish), negative means below (bearish). VWAP is approximated using a cumulative calculation. |
| 12 | `dist_from_24h_high` | `(Close - 24h_Rolling_High) / 24h_Rolling_High`. How far the current price is from the rolling 24-hour high. Always negative or zero. Values near zero indicate price is at recent highs. |
| 13 | `dist_from_24h_low` | `(Close - 24h_Rolling_Low) / 24h_Rolling_Low`. How far the current price is above the rolling 24-hour low. Always positive or zero. Values near zero indicate price is at recent lows. |

---

### Advanced Features

| # | Feature | Description |
|---|---------|-------------|
| 14 | `btc_correlation_20` | 20-period rolling Pearson correlation between the coin's close price and BTC's close price. Values near 1.0 mean the coin moves in lockstep with BTC. Values near 0.0 indicate independent movement. Low correlation coins offer diversification when BTC dominates the market. |
| 15 | `funding_rate` | The current Hyperliquid funding rate for the asset. Positive funding means longs pay shorts (market is bullish/overleveraged long). Negative funding means shorts pay longs. Extreme funding rates often precede reversals. |

---

## Why These Features?

The feature set was chosen based on several principles:

1. **Normalization** -- Every feature is expressed as a ratio or percentage, not an absolute value. This allows the model to learn patterns that generalize across BTC ($60,000) and PEPE ($0.00001) without scale bias.

2. **Complementary signals** -- The features cover different aspects of market microstructure: trend (EMA, momentum), mean reversion (RSI), volatility (ATR, range), volume activity, price structure (VWAP, 24h range), and cross-asset correlation.

3. **No lookahead bias** -- Every feature is computed using only past data available at prediction time. The training pipeline enforces chronological splitting to prevent data leakage.

4. **Computational efficiency** -- All features are computed using pure NumPy without external TA libraries. This keeps the dependency footprint minimal and allows fast computation during live trading.

---

## Pro Version Features

The [Pro version](https://stefanocrypto.gumroad.com/l/ezilv) expands the feature set to 40+, adding:

| Category | Features |
|----------|----------|
| Funding dynamics | `funding_rate_delta_1h`, `funding_rate_delta_4h`, `funding_rate_delta_8h`, `funding_oi_weighted` |
| Open Interest | `oi_change_pct`, `oi_value` |
| Order flow | `order_flow_ratio`, `obi_proxy`, `obi_momentum` |
| Cumulative Volume Delta | `cvd_5`, `cvd_20` |
| Multi-timeframe | `rsi_4h` |
| Sentiment | `fear_greed_index` |
| Liquidation | `liquidation_pressure` |
| Statistical | `price_skewness_24`, `price_kurtosis_24`, `linear_trend_slope_24` |
| Time | `hour_of_day`, `day_of_week` |
| Streak | `consecutive_green`, `consecutive_red` |
| Divergence | `rsi_divergence` |
| Volume momentum | `volume_momentum_3` |

These additional features capture order flow dynamics, market sentiment, and cross-timeframe signals that the free version does not include.
