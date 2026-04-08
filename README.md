<div align="center">

# DeepAlpha

### AI-Powered Autonomous Trading on Hyperliquid

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hyperliquid](https://img.shields.io/badge/Exchange-Hyperliquid-purple.svg)](https://hyperliquid.xyz)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM-orange.svg)](https://lightgbm.readthedocs.io)

[Telegram](https://t.me/DeepAlphaVault) · [Get Pro](https://stefanocrypto.gumroad.com/l/ezilv)

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
| Technical Features | 15 | 38 |
| OBI + CVD Features | ❌ | ✅ |
| Strategies | 1 (AI Autonomous) | 5 |
| ATR Dynamic Trailing Stop | ❌ | ✅ |
| BTC Regime Filter | ❌ | ✅ |
| LLM Market Analysis | ❌ | ✅ |
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

## Quick Start

### 1. Install
```bash
git clone https://github.com/stefanoviana/deepalpha.git
cd deepalpha
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env with your Hyperliquid private key
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
├── train.py              # AI training pipeline
├── download_data.py      # Data downloader
├── features.py           # Feature engineering
├── risk_manager.py       # Position sizing & risk
├── config.py             # Configuration
└── requirements.txt      # Dependencies
```

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

Typical accuracy: **58-62%** on out-of-sample data. Combined with 2:1+ reward-to-risk, this generates positive expected value.

## Disclaimer

**Trading involves significant risk of loss.** This software is provided as-is, with no guarantees of profitability. Past performance does not indicate future results. Only trade with money you can afford to lose.

## License

MIT License — see [LICENSE](LICENSE)

## Links

- [Telegram Channel](https://t.me/DeepAlphaVault)
- [DeepAlpha Pro](https://stefanocrypto.gumroad.com/l/ezilv) — 5 strategies, ensemble model, 38 features, ATR stops
