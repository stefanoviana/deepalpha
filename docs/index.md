# DeepAlpha

**AI-Powered Autonomous Trading on Hyperliquid**

---

## What is DeepAlpha?

DeepAlpha is an open-source AI trading bot that uses machine learning to trade perpetual futures on [Hyperliquid L1](https://hyperliquid.xyz). It downloads historical market data, trains a LightGBM gradient boosting model on 15 technical features, and executes trades autonomously with built-in risk management.

This is not a toy or a proof-of-concept. DeepAlpha is a production system designed to run 24/7, survive volatile markets, and adapt through periodic retraining.

---

## Key Features

- **LightGBM AI Model** -- Gradient boosting classifier trained on 1M+ samples with walk-forward validation to prevent overfitting.
- **20 Markets** -- Scans BTC, ETH, SOL, DOGE, AVAX, LINK, ARB, OP, SUI, INJ, MATIC, APT, SEI, TIA, WIF, PEPE, ONDO, RENDER, FET, and JUP for trading opportunities.
- **Self-Learning** -- Automatically retrains the model every 2 hours using the latest market data.
- **Risk Management** -- Fixed stop-loss/take-profit, position sizing as a percentage of equity, daily loss limits, and a circuit breaker that pauses trading after consecutive losses.
- **Telegram Notifications** -- Real-time alerts for trade entries, exits, stop-losses, and retraining events.
- **Zero API Keys Required for Data** -- Hyperliquid's info API is fully public. No exchange API key needed to download data or train the model.

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
