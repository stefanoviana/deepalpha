<div align="center">

# DeepAlpha

### AI-Powered Autonomous Trading on Hyperliquid, Binance & Bybit

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hyperliquid](https://img.shields.io/badge/Exchange-Hyperliquid-purple.svg)](https://hyperliquid.xyz)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM-orange.svg)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-red.svg)](https://xgboost.readthedocs.io)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Discord](https://img.shields.io/badge/Discord-Join-7289da.svg)](https://discord.gg/P4yX686m)

[Telegram](https://t.me/DeepAlphaVault) · [Discord](https://discord.gg/P4yX686m) · [Blog](blog/how-i-built-deepalpha.md) · [Docs](docs/index.md) · [Get Pro](https://stefanocrypto.gumroad.com/l/ezilv)

**If you find this useful, please give it a star — it helps a lot!**

[![GitHub stars](https://img.shields.io/github/stars/stefanoviana/deepalpha?style=social)](https://github.com/stefanoviana/deepalpha/stargazers)

</div>

---

## What is DeepAlpha?

DeepAlpha is an open-source AI trading bot that uses machine learning to trade perpetual futures on [Hyperliquid L1](https://hyperliquid.xyz). It trains on years of historical data, generates predictions, and executes trades autonomously.

**This is not a toy.** DeepAlpha is a production system designed to survive bear markets.

## Features

| Feature | Free | [Pro](https://stefanocrypto.gumroad.com/l/ezilv) |
|---------|------|-----|
| AI Model (LightGBM) | ✅ | ✅ |
| XGBoost Ensemble | ❌ | ✅ |
| Walk-Forward Validation | ✅ | ✅ |
| Technical Features | 15 | 40 |
| OBI + CVD Features | ❌ | ✅ |
| Strategies | 1 (AI Autonomous) | 3 |
| ATR Dynamic Trailing Stop | ❌ | ✅ |
| BTC Regime Filter | ❌ | ✅ |
| Fear & Greed Index | ❌ | ✅ |
| Auto-Retrain | Every 2h | Every 15min |
| Grafana Dashboard | ❌ | ✅ |
| Telegram Notifications | Basic | Full |
| Support | Community | Private TG |

## How It Works

```
Market Data (Hyperliquid API)
    ↓
Feature Engineering (RSI, ATR, EMA, Momentum, Volume)
    ↓
LightGBM Model (trained on 1M+ samples)
    ↓
Signal Generation (LONG/SHORT with confidence %)
    ↓
Risk Management (position sizing, stop-loss, max positions)
    ↓
Trade Execution (Hyperliquid API)
```

## What's New

**v1.2** — Chronological walk-forward split now sorts by timestamp across all coins (fixes forward-looking bias). Updated feature table.

**v1.1** — Improved prediction target (0.5% threshold), wider stop-loss (2%), faster execution.

The Pro version (v8.2) includes: 3 focused strategies, 40 features, XGBoost ensemble (72.9% accuracy), ATR trailing stops, BTC regime filter.


## How the AI Works

DeepAlpha uses a LightGBM gradient boosting model trained on historical crypto data with walk-forward validation.

**Training Pipeline:**
1. Download 1 year of hourly candles for 20 coins
2. Compute 15 technical features (RSI, ATR, EMA, momentum, etc.)
3. Generate labels: does price move significantly in the next hour?
4. Split data chronologically: 70% train, 15% validation, 15% test
5. Train with early stopping to prevent overfitting
6. Evaluate on test set (never seen during training)

**Key Design Decisions:**
- Walk-forward validation (not random split) — prevents overfitting
- Conservative features — only battle-tested technical indicators
- Early stopping — model stops training when validation loss increases
- Symmetric labels — model predicts both up and down moves

The Pro version adds XGBoost ensemble, PPO reinforcement learning, 50 features, and ATR-based prediction targets.

## Quick Start

### Option A: Docker (recommended)
```bash
git clone https://github.com/stefanoviana/deepalpha.git
cd deepalpha
cp .env.example .env
# Edit .env with your exchange credentials

# Train the model
docker compose run --rm trainer

# Start trading
docker compose up -d deepalpha

# Open dashboard (optional)
docker compose --profile dashboard up -d
# Visit http://localhost:8501
```

### Option B: Manual Install
```bash
git clone https://github.com/stefanoviana/deepalpha.git
cd deepalpha
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your exchange credentials
```

### 3. Train the AI
```bash
python download_data.py    # Download 1 year of candle data
python train.py            # Train the model (~5 min)
```

### 4. Run
```bash
python deepalpha.py        # Start trading
```

## Architecture

```
deepalpha/
├── deepalpha.py          # Main trading bot
├── exchange_adapter.py   # Multi-exchange adapter layer
├── train.py              # AI training pipeline
├── download_data.py      # Data downloader
├── features.py           # Feature engineering
├── risk_manager.py       # Position sizing & risk
├── config.py             # Configuration
└── requirements.txt      # Dependencies
```

## Supported Exchanges

DeepAlpha supports multiple exchanges through a unified adapter layer. Set the `EXCHANGE` variable in your `.env` file to switch between them.

| Exchange | Type | Status |
|----------|------|--------|
| [Hyperliquid](https://hyperliquid.xyz) | L1 Perps | Default |
| [Binance Futures](https://www.binance.com/en/futures) | USDT-M Perpetual | Supported |
| [Bybit](https://www.bybit.com) | USDT Perpetual | Supported |

### Switching Exchanges

Set the `EXCHANGE` environment variable in your `.env` file:

```bash
EXCHANGE=hyperliquid   # default
EXCHANGE=binance
EXCHANGE=bybit
```

### Environment Variables per Exchange

**Hyperliquid** (default -- no extra dependency):
```bash
EXCHANGE=hyperliquid
PRIVATE_KEY=0xYOUR_PRIVATE_KEY
WALLET_ADDRESS=0xYOUR_WALLET_ADDRESS
```

**Binance Futures** (requires `pip install ccxt`):
```bash
EXCHANGE=binance
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
BINANCE_TESTNET=false          # set to true for testnet
```

**Bybit** (requires `pip install ccxt`):
```bash
EXCHANGE=bybit
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
BYBIT_TESTNET=false            # set to true for testnet
```

### Example `.env` for Binance

```bash
# Exchange
EXCHANGE=binance
BINANCE_API_KEY=abc123
BINANCE_API_SECRET=secret456

# Trading
LEVERAGE=5
MAX_POSITIONS=3
RISK_PER_TRADE=0.10

# Telegram (optional)
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

> **Note:** Binance and Bybit adapters require the `ccxt` library. Install it with `pip install ccxt`. Hyperliquid uses its own SDK and does not need ccxt.

## Risk Management

- **Leverage**: 5x (configurable, max 10x)
- **Position size**: 5% of equity per trade
- **Stop-loss**: Fixed 1.5% from entry
- **Take-profit**: 3% from entry (2:1 R:R minimum)
- **Max positions**: 3 simultaneous
- **Daily loss limit**: 5% of equity
- **Circuit breaker**: Pauses after 3 consecutive losses

## Model Performance

The AI model uses walk-forward validation (no overfitting):
- Training: 70% of data (oldest)
- Validation: 15% (for early stopping)
- Test: 15% (newest, never seen during training)

Typical accuracy: **55-80%** on out-of-sample data. Combined with 2:1+ reward-to-risk, this generates positive expected value.

## Disclaimer

**Trading involves significant risk of loss.** This software is provided as-is, with no guarantees of profitability. Past performance does not indicate future results. Only trade with money you can afford to lose.

## Contributing

We welcome contributions! Whether it's new features, bug fixes, documentation, or exchange adapters — every PR helps.

1. Read the [Contributing Guide](CONTRIBUTING.md)
2. Check [open issues](https://github.com/stefanoviana/deepalpha/issues) for ideas
3. Fork, branch, code, and submit a PR

High-impact areas: new technical indicators, exchange adapters (Binance, Bybit, dYdX), tests, and documentation.

## Community

- [Discord](https://discord.gg/P4yX686m) — General discussion, help, feature ideas
- [Telegram](https://t.me/DeepAlphaVault) — Announcements and trade signals
- [Blog](blog/how-i-built-deepalpha.md) — Technical deep dives on the architecture
- [Docs](https://stefanoviana.github.io/deepalpha/) — Full documentation (coming soon)

## License

MIT License — see [LICENSE](LICENSE)

## Links

- [Discord Server](https://discord.gg/P4yX686m)
- [Telegram Channel](https://t.me/DeepAlphaVault)
- [Blog: How I Built DeepAlpha](blog/how-i-built-deepalpha.md)
- [Documentation](https://stefanoviana.github.io/deepalpha/)
- [DeepAlpha Pro](https://stefanocrypto.gumroad.com/l/ezilv) — 3 strategies, ensemble model, 50 features, PPO agent, ATR stops
# Tests: 52 passing
