<div align="center">

# DeepAlpha V11.0

### AI-Powered Crypto Trading Bot for Bybit

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Bybit](https://img.shields.io/badge/Exchange-Bybit-F7A600.svg)](https://www.bybit.com)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM-orange.svg)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-red.svg)](https://xgboost.readthedocs.io)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Discord](https://img.shields.io/badge/Discord-Join-7289da.svg)](https://discord.gg/P4yX686m)

[Website](https://deepalphabot.com) · [Dashboard](https://deepalphabot.com/cloud) · [Discord](https://discord.gg/P4yX686m) · [Telegram](https://t.me/DeepAlphaVault)

</div>

---

## What is DeepAlpha?

DeepAlpha is an open-source ML trading system that predicts crypto price direction on Bybit perpetual futures. It uses 72 engineered features from L2 orderbook data, funding rates, and market microstructure signals.

The core model achieves **70.9% directional accuracy** on walk-forward validated out-of-sample data.

## Plans

| Feature | Free | [Pro $39/mo](https://deepalphabot.com/cloud) | [Lifetime $199](https://deepalphabot.com/cloud) |
|---------|------|------|----------|
| AI Model (LightGBM) | ✅ | ✅ | ✅ |
| XGBoost + RF Ensemble | ❌ | ✅ | ✅ |
| Walk-Forward Validation | ✅ | ✅ | ✅ |
| Features | 15 | 72 (V11) | 72 (V11) |
| TFT + TransformerGRU | ❌ | ✅ | ✅ |
| HMM Regime Detection | ❌ | ✅ | ✅ |
| Meta-Labeling | ❌ | ✅ | ✅ |
| ATR Dynamic TP/SL | ❌ | ✅ | ✅ |
| Cloud Dashboard | ❌ | ✅ | ✅ |
| Auto Retraining (daily) | ❌ | ✅ | ✅ |
| Telegram Alerts | ❌ | ✅ | ✅ |
| Source Code Download | ❌ | ✅ | ✅ |
| Support | Community | Discord | Direct Developer |
| Future Updates | — | ✅ | ✅ Forever |

## How It Works

```
Market Data (Bybit API)
    ↓
Feature Engineering (RSI, ATR, EMA, Momentum, Volume)
    ↓
LightGBM Model (trained on 1M+ samples)
    ↓
Signal Generation (LONG/SHORT with confidence %)
    ↓
Risk Management (position sizing, stop-loss, max positions)
    ↓
Trade Execution (Bybit API)
```

## What's New

**V11.0** (April 2026) — Major accuracy upgrade:
- 72 features (10 new: Hurst exponent, VPIN, volatility regime, fractal efficiency, multi-timeframe alignment)
- 70.9% walk-forward validated accuracy (up from 60%)
- TFT (Temporal Fusion Transformer) + TransformerGRU neural models
- HMM 3-state regime detection (bull/bear/sideways)
- Dynamic ATR-based TP/SL with multi-target take profit
- Pump scanner for new listing detection
- Cloud dashboard with backtest and live signals
- Daily LSTM auto-retraining


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

## Quick Start (Pro — 2 minutes)

**Windows:** double-click `setup.bat`
**Mac/Linux:** run `bash setup.sh`

The setup wizard asks for your license key and Bybit API keys, then starts trading automatically. The AI model is downloaded from our server — no training needed.

### Manual Install
```bash
git clone https://github.com/stefanoviana/deepalpha.git
cd deepalpha
pip install -r requirements.txt
cp .env.example .env        # edit with your keys
python deepalpha.py          # start trading
```

### Docker
```bash
git clone https://github.com/stefanoviana/deepalpha.git
cd deepalpha
cp .env.example .env        # edit with your keys
docker compose up -d deepalpha
```

### Free Version (train your own model)
```bash
python download_data.py      # download historical data
python train.py              # train model (~5 min)
python deepalpha.py           # start trading
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

## Dashboard

Professional Bloomberg-style dark terminal built with Streamlit.

- Real-time USDC balance, PnL, fee ratio, margin usage
- Live cross-venue order flow (Hyperliquid + Binance)
- AI model status with timestamps
- Recent trades with PnL
- Auto-refresh every 30 seconds

```bash
export WALLET_ADDRESS=your_wallet
streamlit run dashboard.py
# Open http://localhost:8501
```

---

## FreqAI Plugin

Already using Freqtrade? Use DeepAlpha's ML pipeline as a FreqAI plugin — no need to switch platforms.

```bash
# Copy the plugin into your Freqtrade project
cp freqai-plugin/deepalpha_model.py your_freqtrade/freqai/prediction_models/
```

Features included:
- **Triple Barrier Labeling** (68.4% accuracy vs ~55-60% standard FreqAI)
- **SHAP Feature Selection** (auto-removes noise features)
- **Meta-Labeling** (filters bad trades, 21% rejection rate)
- **Purged Walk-Forward CV** (no overfitting)

See [freqai-plugin/README.md](freqai-plugin/README.md) for full setup guide.

---

## Supported Exchange

DeepAlpha is optimized for **Bybit** perpetual futures. We focus on one exchange to maximize prediction accuracy.

| Exchange | Type | Status |
|----------|------|--------|
| [Bybit](https://www.bybit.com) | USDT Perpetual | Supported |

### Configuration

```bash
EXCHANGE=bybit
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
BYBIT_TESTNET=false
LEVERAGE=5
MAX_POSITIONS=5
TELEGRAM_TOKEN=your_bot_token     # optional
TELEGRAM_CHAT_ID=your_chat_id     # optional
```

## Risk Management

- **Leverage**: 5x (configurable, max 10x)
- **Position size**: 5% of equity per trade
- **Stop-loss**: Fixed 1.5% from entry
- **Take-profit**: 3% from entry (2:1 R:R minimum)
- **Max positions**: 3 simultaneous
- **Daily loss limit**: 5% of equity
- **Circuit breaker**: Pauses after 3 consecutive losses

## Performance

Walk-forward validated results (V11 model, out-of-sample):

| Metric | Value |
|--------|-------|
| Directional Accuracy | 70.9% |
| Profit Factor | 2.91 |
| Max Drawdown | 20.7% |
| Sharpe Ratio | 0.97 |
| Avg Win / Avg Loss | 4.6:1 |

> **Methodology:** Walk-forward validation with expanding window and 24h embargo between train/test. All metrics are on strictly out-of-sample data, never seen during training.

## Model Architecture

| Model | Accuracy | Role |
|-------|----------|------|
| LightGBM | 70.9% | Primary |
| XGBoost | 70.8% | Ensemble |
| Random Forest | 69.3% | Ensemble |
| TransformerGRU | 65.8% | Neural |
| TFT | 63.7% | Neural |

72 features including: L2 orderbook proxies, funding rate momentum, cross-asset correlation, volatility regime detection, Hurst exponent, VPIN, multi-timeframe alignment.

## Disclaimer

**Trading involves significant risk of loss.** This software is provided as-is, with no guarantees of profitability. Past backtest performance does not indicate future results. The performance metrics above are from historical backtests and may not reflect live trading conditions. Only trade with money you can afford to lose.

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

## Integrations

- **[FreqAI](https://www.freqtrade.io/en/stable/freqai/)** — Use DeepAlpha's ML pipeline as a drop-in FreqAI prediction model. See [freqai-plugin/](freqai-plugin/) for setup.
- **[Bybit Copy Trading](https://www.bybit.com/invite?ref=LN1XOX)** — Follow DeepAlpha trades directly on Bybit with one click.
- **Telegram** — Real-time trade alerts and portfolio updates via bot.

## Links

- [Discord Server](https://discord.gg/P4yX686m)
- [Telegram Channel](https://t.me/DeepAlphaVault)
- [Blog: How I Built DeepAlpha](blog/how-i-built-deepalpha.md)
- [Documentation](https://stefanoviana.github.io/deepalpha/)
- [DeepAlpha Pro](https://deepalphabot.com/cloud) — 3 strategies, ensemble model, 50 features, PPO agent, ATR stops

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=stefanoviana/deepalpha&type=Date)](https://star-history.com/#stefanoviana/deepalpha&Date)

<!-- Keywords: crypto trading bot, ai trading, machine learning trading, lightgbm crypto, bybit bot, bitget bot, algorithmic trading, freqai, xgboost crypto, automated trading, quant trading bot -->
