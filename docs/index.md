# DeepAlpha

**AI-Powered Autonomous Trading on Hyperliquid, Binance, and Bybit**

---

## What is DeepAlpha?

DeepAlpha is an open-source AI trading bot that uses machine learning to trade perpetual futures on multiple exchanges. It supports **Hyperliquid**, **Binance Futures**, and **Bybit** out of the box. It trains a LightGBM + XGBoost + PPO ensemble on 50 engineered features, and executes trades autonomously with maker-only orders and built-in risk management.

This is not a toy. DeepAlpha is a production system designed to run 24/7, learn from its own mistakes, and adapt through continuous retraining.

---

## Supported Exchanges

| Exchange | Markets | Leverage | Status |
|----------|---------|----------|--------|
| **Hyperliquid** | 229 perps | up to 40x | Full support |
| **Binance Futures** | 300+ USDT-M | up to 125x | Supported via ccxt |
| **Bybit** | 200+ USDT perps | up to 100x | Supported via ccxt |

---

## Key Features

- **LightGBM + XGBoost + PPO Ensemble** -- 3 ML models working together. 50 engineered features including skewness, kurtosis, trend slope, and area ratio.
- **229+ Markets** -- Trades all available perpetual markets on your chosen exchange.
- **Multi-Exchange** -- Switch between Hyperliquid, Binance, and Bybit with one config change.
- **Self-Learning** -- Learns from its own trades. Losses are weighted 5x in retraining so the model avoids past mistakes.
- **Maker-Only Orders** -- Limit orders with 30s wait and re-pricing. Near-zero fees.
- **50 Advanced Features** -- RSI, ATR, EMA, skewness, kurtosis, linear trend slope, area ratio, range position, volume-price trend, and more.
- **Risk Management** -- ATR trailing stops, BTC regime filter, position sizing, daily loss limits, circuit breaker.
- **Telegram Notifications** -- Real-time alerts for trade entries, exits, and errors.
- **Walk-Forward Validation** -- Chronological data split prevents overfitting. No look-ahead bias.

---

## Quick Links

| Page | Description |
|------|-------------|
| [Getting Started](getting-started.md) | Install, train, and run DeepAlpha in under 10 minutes |
| [Configuration](configuration.md) | All configuration parameters with examples |
| [Architecture](architecture.md) | System design, data flow, and component overview |
| [Features](features.md) | The 15 technical features used by the AI model |
| [Strategies](strategies.md) | How the AI generates and executes trading signals |
| [FAQ](faq.md) | Common questions and troubleshooting |

---

## Pro Version

The free version covers the core trading loop. [DeepAlpha Pro](https://stefanocrypto.gumroad.com/l/ezilv) adds:

- XGBoost + LightGBM ensemble (72.9% accuracy)
- 40 features including OBI, CVD, Fear & Greed Index, and RSI divergence
- ATR-based dynamic trailing stops
- BTC regime filter
- PPO reinforcement learning module
- Grafana monitoring dashboard
- 229 markets via dynamic coin selection (Forager)

---

## Disclaimer

Trading involves significant risk of loss. This software is provided as-is, with no guarantees of profitability. Past performance does not indicate future results. Only trade with money you can afford to lose.

---

## License

MIT License -- see [LICENSE](https://github.com/stefanoviana/deepalpha/blob/main/LICENSE).
