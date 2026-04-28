<div align="center">

# DeepAlpha V11.0

### AI-Powered Crypto Trading Bot for Bybit & Binance Futures

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Bybit](https://img.shields.io/badge/Exchange-Bybit-F7A600.svg)](https://www.bybit.com)
[![Binance](https://img.shields.io/badge/Exchange-Binance-F3BA2F.svg)](https://www.binance.com)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM-orange.svg)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-red.svg)](https://xgboost.readthedocs.io)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Discord](https://img.shields.io/badge/Discord-Join-7289da.svg)](https://discord.gg/P4yX686m)

[Website](https://deepalphabot.com) · [Dashboard](https://deepalphabot.com/cloud) · [Discord](https://discord.gg/P4yX686m) · [Telegram](https://t.me/DeepAlphaVault)

</div>

---

## What is DeepAlpha?

DeepAlpha is an open-source ML trading system that predicts crypto price direction on **Bybit** and **Binance** perpetual futures. It uses 72 engineered features from L2 orderbook data, funding rates, and market microstructure signals.

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
Market Data (Bybit / Binance API via ccxt)
    ↓
Feature Engineering (RSI, ATR, EMA, Momentum, Volume)
    ↓
LightGBM Model (trained on 1M+ samples)
    ↓
Signal Generation (LONG/SHORT with confidence %)
    ↓
Risk Management (position sizing, stop-loss, max positions)
    ↓
Trade Execution (Bybit / Binance API via ccxt)
```

## What's New

**V11.1** (April 2026) — Liquidation map + pump scanner:
- **Liquidation level estimator** — detects where stop-loss clusters are, optimizes TP/SL placement
- **Pump scanner** — real-time detection of volume spikes and new listings on Bybit + Binance
- **Binance cross-exchange alerts** — detects Binance listings that pump on Bybit

**V11.0** (April 2026) — Major accuracy upgrade:
- 72 features (10 new: Hurst exponent, VPIN, volatility regime, fractal efficiency, multi-timeframe alignment)
- 70.9% walk-forward validated accuracy (up from 60%)
- TFT (Temporal Fusion Transformer) + TransformerGRU neural models
- HMM 3-state regime detection (bull/bear/sideways)
- Dynamic ATR-based TP/SL with multi-target take profit
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

The setup wizard asks for your license key and exchange API keys (Bybit, Binance, or Bitget), then starts trading automatically. The AI model is downloaded from our server — no training needed.

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

## Trading Strategies

DeepAlpha combines multiple strategies for maximum edge:

### 1. AI Directional Prediction (Primary)
XGBoost + LightGBM ensemble predicts price direction with 70.9% accuracy using 72 features from L2 orderbook data, funding rates, and market microstructure.

### 2. HMM Regime Detection
Hidden Markov Model identifies bull/bear/sideways regimes. The bot adapts: wider TP in bull, tighter SL in bear, reduced activity in sideways.

### 3. Liquidation Level Analysis
Estimates where stop-loss clusters are based on open interest and leverage distribution. Uses liquidation zones to:
- Place TP before liquidation cascades (take profit before bounce)
- Place SL beyond danger zones (avoid getting caught in cascades)
- Boost confidence when liquidation cascades favor our direction

### 4. Pump Scanner
Real-time detection of volume spikes and new listings across Bybit and Binance. Auto-enters pumps with tight risk management (5% equity, 2h max hold).

### 5. Multi-Target Take Profit
- **T1** (0.8x ATR): Close 33% — lock some profit early
- **T2** (1.3x ATR): Close 33% — let winner run
- **Trailing**: ATR-based trailing stop on remaining 34%

### 6. Auto-Unstuck
Graduated exit when trade goes wrong: -2% close 25%, -3% close 25%, -4% close 25%, -5% hard cap close all.

### 4. Run
```bash
python deepalpha.py        # Start AI trading bot
python pump_scanner.py     # Start pump scanner (standalone)
```

---

## Pump Scanner

Real-time pump detection system that monitors **all 500+ Bybit USDT perpetual pairs** every 3 seconds.

### What it detects
- **Volume spikes**: 5x+ normal volume in a single candle
- **Price momentum**: +3% move with 3+ consecutive green candles
- **New listings**: Monitors Bybit announcements API for new perpetual contracts
- **Dump exhaustion**: RSI > 80 + volume decline for short entries

### How it trades
1. Detects pump signal (volume + price + RSI + buy ratio confirmation)
2. Opens long with 5x leverage, ATR-based stop loss
3. **TP cascade**: TP1 (+5%) close 40% | TP2 (+10%) close 30% + trailing | TP3 (+20%) close rest
4. Circuit breaker: stops after -$50 daily loss

### Quick Start
```bash
# 1. Add to your .env file:
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret

# 2. Optional: Telegram alerts
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# 3. Run standalone:
pip install ccxt numpy requests python-dotenv
python pump_scanner.py
```

### Configuration
See [`pump_config_example.env`](pump_config_example.env) for all tunable parameters:
- Detection thresholds (volume multiplier, price spike %, RSI range)
- Position sizing (leverage, risk budget, max positions)
- TP/SL levels (ATR-based SL, cascade TP at 5%/10%/20%)
- Fakeout filters (consecutive candles, buy ratio, minimum volume)

### Run with Docker
```bash
docker compose up -d pump-scanner
```

### Run alongside the AI bot
```python
from pump_scanner import create_pump_scanner_from_config

scanner = create_pump_scanner_from_config()
if scanner:
    scanner.start()  # runs in background thread
```

---

## TradingView Webhook

Execute your TradingView alerts automatically on any supported exchange. No coding needed.

### Setup (2 minutes)
1. Get your webhook key from the [Dashboard](https://deepalphabot.com/cloud/dashboard)
2. In TradingView, create an alert and set the webhook URL:
   ```
   https://deepalphabot.com/api/webhook/tradingview
   ```
3. Set the alert message (JSON):
   ```json
   {
     "action": "buy",
     "symbol": "BTC",
     "qty": 0.01,
     "key": "your_webhook_key"
   }
   ```

### Supported Actions
| Action | Description |
|--------|-------------|
| `buy` | Open long / buy market order |
| `sell` | Open short / sell market order |
| `close` | Close entire position |

### Auto-sizing
Set `qty` to `0` and DeepAlpha will automatically size the position (5% of equity at 5x leverage).

### Works with all 12 exchanges
Bybit, Binance, OKX, Gate.io, KuCoin, Bitget, HTX, MEXC, BingX, Phemex, BitMart, WhiteBIT — same webhook URL, trades on whichever exchange you connected.

---

## Architecture

```
deepalpha/
├── deepalpha.py          # Main AI trading bot
├── pump_scanner.py       # Real-time pump detection (standalone)
├── exchange_adapter.py   # Multi-exchange adapter layer
├── train.py              # AI training pipeline
├── download_data.py      # Data downloader
├── features.py           # Feature engineering (72 features)
├── risk_manager.py       # Position sizing & risk
├── config.py             # Configuration
├── pump_config_example.env  # Pump scanner config template
└── requirements.txt      # Dependencies
```

## Cloud Platform

Don't want to self-host? Use DeepAlpha Cloud — we run the bot for you.

**[deepalphabot.com/cloud](https://deepalphabot.com/cloud)**

### What you get:
- **Live AI signals** — see every prediction in real-time
- **Auto-trading** — connect your Bybit or Binance API, the bot trades for you
- **Backtest viewer** — test the AI on historical data with custom parameters
- **Equity curve** — track your portfolio performance over time
- **Trade history** — every trade logged with PnL and exit reason
- **Email alerts** — welcome sequence + trial expiry notifications

### How it works:
1. Register at [deepalphabot.com/cloud](https://deepalphabot.com/cloud) (7-day free trial)
2. Connect your exchange API keys (Bybit, Binance, Bitget — read-only supported)
3. Configure risk settings (leverage, max positions, confidence threshold)
4. The AI trades automatically — check your dashboard anytime

### Security:
- API keys encrypted with **AES-256 Fernet** encryption
- **Read-only API** supported — the bot can trade but never withdraw
- Keys stored only on our server, never shared
- SSL/TLS 1.3 + HSTS on all connections

### Cloud Stack:
- **Backend**: FastAPI + PostgreSQL + JWT auth
- **AI Engine**: Same V11 models as self-hosted (LightGBM + XGBoost + TFT)
- **Hosting**: Dedicated VPS with PM2 process management
- **Email**: Resend transactional email service

### Pricing:
| Plan | Price | Features |
|------|-------|----------|
| Free Trial | $0 for 7 days | Full access, no credit card |
| Pro | $39/month | Auto-trading + dashboard + signals |
| Lifetime | $199 one-time | Everything forever + source code |

---

## FreqAI Plugin

Already using Freqtrade? Use DeepAlpha's ML pipeline as a FreqAI plugin.

```bash
cp freqai-plugin/deepalpha_model.py your_freqtrade/freqai/prediction_models/
```

Features:
- **Triple Barrier Labeling** (70.9% accuracy vs ~55-60% standard FreqAI)
- **SHAP Feature Selection** (auto-removes noise features)
- **Meta-Labeling** (filters bad trades)
- **Walk-Forward CV** with 24h embargo

See [freqai-plugin/README.md](freqai-plugin/README.md) for setup guide.

---

## Supported Exchanges

DeepAlpha supports multiple exchanges via the [ccxt](https://github.com/ccxt/ccxt) library. Switch exchanges with a single env var.

| Exchange | Type | Status | Markets |
|----------|------|--------|---------|
| [Bybit](https://www.bybit.com) | USDT Perpetual | Supported | 3,200+ |
| [Binance](https://www.binance.com) | USDT-M Futures | Supported | 4,300+ |
| [OKX](https://www.okx.com) | USDT Perpetual | **New** | 3,000+ |
| [Gate.io](https://www.gate.io) | USDT Perpetual | **New** | 6,300+ |
| [KuCoin](https://www.kucoin.com) | USDT Perpetual | **New** | 1,600+ |
| [Bitget](https://www.bitget.com) | USDT-M Futures | Supported | 1,300+ |
| [Hyperliquid](https://hyperliquid.xyz) | Perpetual (L1) | Supported | 100+ |

### Configuration — Bybit (default)

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

### Configuration — Binance Futures

```bash
EXCHANGE=binance
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
BINANCE_TESTNET=false
LEVERAGE=5
MAX_POSITIONS=5
```

### Configuration — OKX

```bash
EXCHANGE=okx
OKX_API_KEY=your_api_key
OKX_API_SECRET=your_api_secret
OKX_PASSPHRASE=your_passphrase
LEVERAGE=5
MAX_POSITIONS=5
```

### Configuration — Gate.io

```bash
EXCHANGE=gateio
GATEIO_API_KEY=your_api_key
GATEIO_API_SECRET=your_api_secret
LEVERAGE=5
MAX_POSITIONS=5
```

### Configuration — KuCoin

```bash
EXCHANGE=kucoin
KUCOIN_API_KEY=your_api_key
KUCOIN_API_SECRET=your_api_secret
KUCOIN_PASSPHRASE=your_passphrase
LEVERAGE=5
MAX_POSITIONS=5
```

### Configuration — Bitget

```bash
EXCHANGE=bitget
BITGET_API_KEY=your_api_key
BITGET_SECRET=your_api_secret
BITGET_PASSPHRASE=your_passphrase
LEVERAGE=5
MAX_POSITIONS=5
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

High-impact areas: new technical indicators, exchange adapters (dYdX, OKX), tests, and documentation.

## Community

- [Discord](https://discord.gg/P4yX686m) — General discussion, help, feature ideas
- [Telegram](https://t.me/DeepAlphaVault) — Announcements and trade signals
- [Blog](blog/how-i-built-deepalpha.md) — Technical deep dives on the architecture
- [Docs](https://stefanoviana.github.io/deepalpha/) — Full documentation (coming soon)

## License

MIT License — see [LICENSE](LICENSE)

## Integrations

- **[Crypto Pump Scanner](https://github.com/stefanoviana/crypto-pump-scanner)** — Real-time pump detection for Bybit. Monitors 500+ pairs, auto-trades volume spikes. Standalone or integrated.
- **[FreqAI](https://www.freqtrade.io/en/stable/freqai/)** — Use DeepAlpha's ML pipeline as a drop-in FreqAI prediction model. See [freqai-plugin/](freqai-plugin/) for setup.
- **[Binance Futures](https://www.binance.com)** — Full support for Binance USDT-M perpetual futures via ccxt.
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

<!-- Keywords: crypto trading bot, ai trading, machine learning trading, lightgbm crypto, bybit bot, binance bot, binance futures, bitget bot, algorithmic trading, freqai, xgboost crypto, automated trading, quant trading bot -->
