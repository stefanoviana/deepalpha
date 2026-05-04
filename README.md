<div align="center">

# DeepAlpha V11.0

### AI-Powered Crypto Trading Bot — 18 Exchanges, 81.3% Accuracy

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Exchanges](https://img.shields.io/badge/Exchanges-12-00d4aa.svg)](https://deepalphabot.com)
[![Free Trial](https://img.shields.io/badge/Free_Trial-7_Days-brightgreen.svg)](https://deepalphabot.com/cloud)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM-orange.svg)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-red.svg)](https://xgboost.readthedocs.io)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Discord](https://img.shields.io/badge/Discord-Join-7289da.svg)](https://discord.gg/P4yX686m)
[![Telegram Bot](https://img.shields.io/badge/Telegram-Bot-0088cc.svg)](https://t.me/DeepAlphaVault_bot)

[Website](https://deepalphabot.com) · [Dashboard](https://deepalphabot.com/cloud) · [Telegram Bot](https://t.me/DeepAlphaVault_bot) · [Discord](https://discord.gg/P4yX686m) · [Channel](https://t.me/DeepAlphaVault)

</div>

---

## 🚀 Two Ways to Use DeepAlpha

### ☁️ Cloud Version (Recommended) — No Setup Required
> **The easiest way to start. Bot runs 24/7 on our servers.**

| | |
|---|---|
| ✅ No VPS, no Docker, no Python | ✅ 12 exchanges supported |
| ✅ Start in Paper Mode (zero risk) | ✅ Real-time dashboard |
| ✅ Pump scanner included | ✅ TradingView webhooks |
| ✅ Exchange-side stop-loss | ✅ Telegram bot control |

<div align="center">

### [**Start Free 7-Day Trial →**](https://deepalphabot.com/cloud/login)
*No credit card required. Start in Paper Mode — switch to Live when ready.*

</div>

---

### 🖥️ Self-Hosted Version — Full Control
> **For developers who want to customize the code.**

```bash
git clone https://github.com/stefanoviana/deepalpha.git
cd deepalpha
pip install -r requirements.txt
cp .env.example .env  # add your API keys
python deepalpha.py
```

**Supports 12 exchanges:** Bybit, Binance, OKX, Gate.io, KuCoin, Bitget, HTX, MEXC, BingX, Phemex, BitMart, WhiteBIT



---

### About AI Models

The self-hosted version includes **training scripts** but does NOT include pre-trained models. You have two options:

#### Option A: Train Your Own Models (Free)
```bash
python download_data.py   # Download historical data from Bybit
python train.py           # Train XGBoost + LightGBM models (~2-4 hours)
```
This requires 8GB+ RAM and takes several hours. Your accuracy depends on data quality and parameters.

#### Option B: Use Our Pre-Trained Models (Cloud Pro)
Our production models achieve **81.3% walk-forward validated accuracy** after months of optimization:
- 72 engineered features
- 12 months of curated training data
- Optuna hyperparameter tuning (200+ trials)
- 4-window walk-forward validation
- Weekly retraining with fresh market data

These models are **exclusive to Cloud Pro** users — the same models powering our live trading.

**[Get Pre-Trained Models — 7 Days Free (no credit card)](https://deepalphabot.com/cloud/login)**

---

---

### Cloud vs Self-Hosted

| Feature | Cloud (Pro) | Self-Hosted (Free) |
|---|---|---|
| **Setup time** | 30 seconds | 30+ minutes |
| **AI Bot** | ✅ 24/7 managed | ✅ You manage |
| **Pump Scanner** | ✅ Included | ✅ Included |
| **Dashboard** | ✅ Web + Mobile | ❌ CLI only |
| **Telegram Bot** | ✅ Full control | ❌ |
| **TradingView Webhooks** | ✅ | ❌ |
| **Auto-restart on crash** | ✅ | ❌ |
| **Exchange-side SL** | ✅ | ❌ |
| **Paper Mode** | ✅ | ❌ |
| **Price** | $39/mo (7 days free) | Free forever |

<div align="center">

### [**Try Cloud Free →**](https://deepalphabot.com/cloud/login) &nbsp;&nbsp;|&nbsp;&nbsp; [**Self-Host Guide →**](https://github.com/stefanoviana/deepalpha#-self-hosted-version--full-control)

</div>

> Cloud-hosted. No installation. No VPS. No coding. The AI trades 24/7 for you.

## Telegram Bot — Trade from Your Phone

Manage everything from Telegram with **[@DeepAlphaVault_bot](https://t.me/DeepAlphaVault_bot)**:

| Command | What it does |
|---------|-------------|
| `/start` | Welcome & account status |
| `/login` | Link your DeepAlpha account |
| `/start_bot` | Start AI / Grid / DCA Bot |
| `/stop_bot` | Stop bot |
| `/status` | Open positions & PnL |
| `/balance` | Exchange balance |
| `/pnl` | PnL summary |
| `/trades` | Recent trades |
| `/keys` | Setup exchange API keys |

Start/stop bots, check positions, get real-time trade notifications — all from Telegram. [Open Bot →](https://t.me/DeepAlphaVault_bot)

## Pump Scanner — Catch Explosive Moves

The built-in pump scanner monitors **500+ coins in real-time** for volume spikes and automatically trades the pump.

**How it works:**
1. Scans all coins every 2 minutes for volume spikes (5x+ above average)
2. Opens a LONG position automatically when pump detected
3. Takes profit in 3 tiers: TP1 (+5%), TP2 (+10%), TP3 (+20%)
4. Trailing stop locks in gains after TP1
5. Stop loss at -3% for protection

**Recent results:**
- ZEREBRO: +$33 (+7.7%) in 72 minutes
- B Token: +$66 (+12.7%) in 2 minutes

The pump scanner runs automatically alongside the AI Bot. No extra configuration needed.

---

## What is DeepAlpha?

DeepAlpha is an open-source ML trading system that predicts crypto price direction on **18 exchanges** including Bybit, Binance, OKX, Gate.io, KuCoin, Bitget, HTX, MEXC, BingX, Phemex, BitMart, and WhiteBIT. It uses 72 engineered features from L2 orderbook data, funding rates, and market microstructure signals.

The core model achieves **81.3% directional accuracy** on walk-forward validated out-of-sample data.

## Plans

| Feature | Free (GitHub) | [Pro $39/mo](https://deepalphabot.com/cloud) | [Lifetime $199](https://deepalphabot.com/cloud) |
|---------|:---:|:---:|:---:|
| AI Model (LightGBM) | Train yourself | ✅ Pre-trained (81.3%) | ✅ Pre-trained |
| XGBoost + RF Ensemble | ❌ | ✅ | ✅ |
| **Grid Bot** (range trading) | ❌ | ✅ 5 strategies | ✅ 5 strategies |
| **DCA Bot** (safety orders) | ❌ | ✅ 5 strategies | ✅ 5 strategies |
| Pump Scanner (500+ coins) | ❌ | ✅ Auto | ✅ Auto |
| Features | 15 basic | 78 (V11 full) | 78 (V11 full) |
| TFT + TransformerGRU | ❌ | ✅ | ✅ |
| HMM Regime Detection | ❌ | ✅ | ✅ |
| Cloud Dashboard | ❌ | ✅ Real-time | ✅ Real-time |
| **Telegram Bot Control** | ❌ | ✅ @DeepAlphaVault_bot | ✅ |
| Auto Retraining (daily) | ❌ | ✅ | ✅ |
| Auto Restart & Monitoring | ❌ | ✅ 24/7 | ✅ 24/7 |
| 12 Exchange Support | Manual setup | ✅ 1-click | ✅ 1-click |
| VPS Required | ✅ ($20-50/mo) | ❌ Cloud hosted | ❌ Cloud hosted |
| Source Code | View only | ❌ | ✅ Full download |
| Support | GitHub Issues | Discord | Direct Developer |

### 💡 Why Most Users Choose Cloud

> **The free version requires you to:**
> - Set up your own VPS ($20-50/month)
> - Train the ML model yourself (hours of compute)
> - Monitor and restart the bot manually
> - Handle updates, crashes, and maintenance
>
> **The cloud version gives you everything ready in 2 minutes:**
> - Pre-trained AI model with 81.3% accuracy
> - Grid Bot + DCA Bot + 10 pre-built strategies
> - Manage from Telegram (@DeepAlphaVault_bot)
> - Auto-restart, daily retraining, 24/7 monitoring
> - No VPS, no installation, no maintenance
>
> **The VPS alone costs $20-50/month — our Pro plan is $39/month with everything included.**
>
> 👉 [**Start Free 7-Day Trial — No Credit Card**](https://deepalphabot.com/cloud)

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
- 81.3% walk-forward validated accuracy (up from 60%)
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
XGBoost + LightGBM ensemble predicts price direction with 81.3% accuracy using 72 features from L2 orderbook data, funding rates, and market microstructure.

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

### Works with all 18 exchanges
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

## Cloud Platform — Premium Features

The [DeepAlpha Cloud Platform](https://deepalphabot.com) includes additional features not available in the open-source version:

| Feature | Open Source | Cloud Platform |
|---------|-----------|----------------|
| AI Trading Bot | ✅ | ✅ |
| Backtesting | ✅ | ✅ Enhanced |
| Grid Trading Bot | ❌ | ✅ 5 strategies |
| DCA Bot + Safety Orders | ❌ | ✅ 5 strategies |
| 12 Exchange Support | Bybit only | ✅ All 12 |
| Real-time Dashboard | ❌ | ✅ |
| TradingView Webhooks | ❌ | ✅ |
| Auto-restart | ❌ | ✅ |
| Telegram Alerts | Basic | ✅ Advanced |

**Try free for 7 days — no credit card required:** [deepalphabot.com](https://deepalphabot.com)

### What you get:
- **Live AI signals** — see every prediction in real-time
- **Auto-trading** — connect your Bybit or Binance API, the bot trades for you
- **Grid Bot** — 5 pre-built grid strategies for range-bound markets
- **DCA Bot** — dollar-cost averaging with safety orders for volatile markets
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
- **Triple Barrier Labeling** (81.3% accuracy vs ~55-60% standard FreqAI)
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
| [OKX](https://www.okx.com) | USDT Perpetual | Supported | 3,000+ |
| [Gate.io](https://www.gate.io) | USDT Perpetual | Supported | 6,300+ |
| [KuCoin](https://www.kucoin.com) | USDT Perpetual | Supported | 1,600+ |
| [Bitget](https://www.bitget.com) | USDT-M Futures | Supported | 1,300+ |
| [HTX](https://www.htx.com) | USDT Perpetual | Supported | 900+ |
| [MEXC](https://www.mexc.com) | USDT Perpetual | Supported | 2,400+ |
| [BingX](https://www.bingx.com) | USDT Perpetual | Supported | 800+ |
| [Phemex](https://www.phemex.com) | USDT Perpetual | Supported | 300+ |
| [BitMart](https://www.bitmart.com) | USDT Perpetual | Supported | 400+ |
| [WhiteBIT](https://www.whitebit.com) | USDT Perpetual | Supported | 200+ |

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
| Directional Accuracy | 81.3% |
| Profit Factor | 2.91 |
| Max Drawdown | 20.7% |
| Sharpe Ratio | 0.97 |
| Avg Win / Avg Loss | 4.6:1 |

> **Methodology:** Walk-forward validation with expanding window and 24h embargo between train/test. All metrics are on strictly out-of-sample data, never seen during training.

## Model Architecture

| Model | Accuracy | Role |
|-------|----------|------|
| LightGBM | 81.3% | Primary |
| XGBoost | 70.8% | Ensemble |
| Random Forest | 69.3% | Ensemble |
| TransformerGRU | 65.8% | Neural |
| TFT | 63.7% | Neural |

72 features including: L2 orderbook proxies, funding rate momentum, cross-asset correlation, volatility regime detection, Hurst exponent, VPIN, multi-timeframe alignment.


## Troubleshooting

Having issues? Here are the most common problems and how to fix them.

### Missing dependencies

```
[ERROR] Missing dependencies: lightgbm, ccxt
```

**Fix:** Install all required packages:
```bash
pip install -r requirements.txt
```

If you get permission errors, use `pip install --user -r requirements.txt` or a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### Missing .env file

```
[ERROR] No .env file found in current directory.
```

**Fix:** Copy the example and fill in your API keys:
```bash
cp .env.example .env
nano .env   # or any text editor
```

You need at minimum:
- `EXCHANGE=bybit` (or binance, okx, bitget, etc.)
- The matching API key + secret for your exchange

### Missing model files

```
[ERROR] Model not found at model_1h.pkl
```

**Fix:** You need a trained ML model. Two options:
1. **Train your own** (free):
   ```bash
   python download_data.py   # download historical data
   python train.py           # train model (~5 min)
   ```
2. **Use the cloud** (no model needed): [deepalphabot.com](https://deepalphabot.com) - 7-day free trial

### API key errors

```
AuthenticationError: Invalid API key
```

**Fix:**
- Double-check your API key and secret in `.env` (no extra spaces or quotes)
- Make sure the API key has **trading permissions** enabled
- For Bybit: enable "Contract - Trade" permission
- For Binance: enable "Futures" permission and whitelist your IP
- For Bitget/OKX/KuCoin: make sure you also set the passphrase

### Exchange not supported

```
ValueError: Unsupported exchange: xxx
```

**Fix:** Set `EXCHANGE` in your `.env` to one of the supported values:
`bybit`, `binance`, `okx`, `bitget`, `gateio`, `kucoin`, `htx`, `mexc`, `bingx`, `phemex`, `bitmart`, `whitebit`, `hyperliquid`

### Bot crashes in a loop

The bot waits 30 seconds before exiting after a crash to prevent restart-loop spam (PM2, Docker, systemd). Check the error message above the crash for the root cause.

### Still stuck?

- **Setup guide:** [deepalphabot.com/setup-guide](https://deepalphabot.com/setup-guide)
- **Telegram support:** [@DeepAlphaVault_bot](https://t.me/DeepAlphaVault_bot)
- **Discord:** [discord.gg/P4yX686m](https://discord.gg/P4yX686m)
- **GitHub Issues:** [github.com/stefanoviana/deepalpha/issues](https://github.com/stefanoviana/deepalpha/issues)

Or just use the **cloud version** — no installation, no setup, no crashes: [deepalphabot.com](https://deepalphabot.com)

---

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

<!-- Keywords: crypto trading bot, ai trading, machine learning trading, lightgbm crypto, bybit bot, binance bot, binance futures, bitget bot, okx bot, gate.io bot, kucoin bot, htx bot, mexc bot, bingx bot, phemex bot, bitmart bot, whitebit bot, huobi trading bot, algorithmic trading, freqai, xgboost crypto, automated trading, quant trading bot, 12 exchange trading bot, multi-exchange trading bot, cloud trading bot, free crypto trading bot, free trial trading bot, tradingview webhook bot, ccxt trading bot, best AI trading bot 2026, crypto trading bot 2026, non-custodial trading bot, open source trading bot, pump scanner, crypto pump detector, volume spike bot -->

---

## Blog & Resources

- [Best AI Crypto Trading Bot 2026](https://deepalphabot.com/blog/article-6)
- [Bybit Trading Bot Free Trial](https://deepalphabot.com/blog/article-7)
- [Automated Crypto Trading for Beginners](https://deepalphabot.com/blog/article-8)
- [TradingView Webhook Crypto Bot Setup](https://deepalphabot.com/blog/article-9)
- [How to Use AI for Crypto Trading](https://deepalphabot.com/blog/article-10)

---

<div align="center">
<sub>Built with Python, XGBoost, LightGBM, CCXT | <a href="https://deepalphabot.com">deepalphabot.com</a> | <a href="https://pypi.org/project/deepalpha-freqai/">PyPI</a></sub>
</div>
