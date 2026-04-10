# Getting Started

This guide walks you through installing DeepAlpha, training the AI model, and running the bot for the first time.

---

## Requirements

- **Python 3.10 or higher**
- **A Hyperliquid account** with funds deposited -- see [hyperliquid.xyz](https://hyperliquid.xyz)
- **A Hyperliquid API private key** -- generated from the Hyperliquid web app under Settings > API
- **Minimum recommended capital**: $500 USD (to allow proper position sizing at 5x leverage)

---

## Step 1: Clone and Install

```bash
git clone https://github.com/stefanoviana/deepalpha.git
cd deepalpha
pip install -r requirements.txt
```

Dependencies installed:

| Package | Purpose |
|---------|---------|
| `lightgbm>=4.0` | Gradient boosting ML model |
| `numpy>=1.24` | Numerical computation |
| `requests>=2.28` | HTTP requests to Hyperliquid API |
| `python-dotenv>=1.0` | Load environment variables from `.env` |
| `hyperliquid-python-sdk>=0.5` | Official Hyperliquid Python SDK |
| `eth-account>=0.11` | Ethereum account signing |

---

## Step 2: Download Historical Data

Download 1 year of hourly candle data from Hyperliquid for all 20 configured coins. No API key is required -- Hyperliquid's info endpoint is public.

```bash
python download_data.py
```

Expected output:

```
Downloading 365 days of 1h candles for 20 coins...
Saving to: data/

  [ 1/20] BTC    — 8,760 candles saved
  [ 2/20] ETH    — 8,760 candles saved
  ...
  [20/20] JUP    — 8,760 candles saved

Done! Data saved to data/ directory.
```

This creates JSON files in the `data/` directory (one per coin), typically totaling around 50 MB.

---

## Step 3: Train the AI Model

Train the LightGBM classifier using walk-forward validation:

```bash
python train.py
```

The training pipeline:

1. Loads candle data for all 20 coins
2. Computes 15 technical features per candle (RSI, ATR, EMA crossover, momentum, etc.)
3. Generates binary labels: will price go up or down over the next 3 candles?
4. Splits chronologically: 70% train, 15% validation, 15% test
5. Trains with early stopping (stops when validation loss plateaus)
6. Reports test accuracy and saves the model to `model.pkl`

Expected output:

```
============================================================
DeepAlpha — Training Pipeline
============================================================

[1/3] Loading data and building features...
  BTC    — 8,700 samples
  ETH    — 8,700 samples
  ...

  Total dataset: 174,000 samples, 15 features
  Label balance: 51.2% positive

[2/3] Training LightGBM model...
  Train:  121,800 samples
  Val:    26,100 samples
  Test:   26,100 samples

  Test accuracy: 0.5812 (58.1%)

[3/3] Saving model to model.pkl...
  Model saved (245 KB)

============================================================
Training complete! Run `python deepalpha.py` to start trading.
============================================================
```

!!! note
    Accuracy above 55% on unseen test data is considered good for financial prediction. Combined with a 2:1 reward-to-risk ratio (3% TP vs 2% SL), this produces positive expected value.

---

## Step 4: Configure Environment

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your Hyperliquid API credentials:

```bash
# Required
PRIVATE_KEY=0xYOUR_PRIVATE_KEY_HERE
WALLET_ADDRESS=0xYOUR_WALLET_ADDRESS_HERE

# Optional: Telegram notifications
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

!!! warning
    Never commit your `.env` file to version control. The `.gitignore` file already excludes it, but always double-check.

---

## Step 5: Run the Bot

```bash
python deepalpha.py
```

You should see:

```
    ____                  ___    __      __
   / __ \___  ___  ____  /   |  / /___  / /_  ____ _
  / / / / _ \/ _ \/ __ \/ /| | / / __ \/ __ \/ __ `/
 / /_/ /  __/  __/ /_/ / ___ |/ / /_/ / / / / /_/ /
/_____/\___/\___/ .___/_/  |_/_/ .___/_/ /_/\__,_/
               /_/            /_/
                              Free Edition

DeepAlpha initialised successfully
  Wallet:    0xYour...addr
  Leverage:  5x
  Max pos:   3
  Risk/trade:10%

============================================================
DeepAlpha — Starting trading loop
Scanning 20 coins every 60s
============================================================

[2026-04-10 12:00:00] Equity: $1,000.00 | Positions: 0/3 | Daily PnL: $0.00
```

---

## Step 6: Verify It Works

Check these indicators to confirm the bot is running correctly:

1. **Console output** updates every 60 seconds with equity, positions, and daily PnL.
2. **Telegram notifications** arrive when the bot opens or closes a trade (if configured).
3. **Hyperliquid web app** shows your open positions under the Portfolio tab.
4. **No error messages** in the console output.

If you see `[RISK] Max positions reached (3)`, that means the bot is fully allocated -- this is normal behavior, not an error.

---

## Running on a VPS (Recommended)

For 24/7 operation, run DeepAlpha on a Linux VPS. A $5/month server (1 CPU, 1 GB RAM) is sufficient.

```bash
# Using screen to keep the bot running after SSH disconnect
screen -S deepalpha
python deepalpha.py

# Detach: Ctrl+A, then D
# Reattach: screen -r deepalpha
```

Alternatively, use `systemd` for automatic restart:

```ini
# /etc/systemd/system/deepalpha.service
[Unit]
Description=DeepAlpha Trading Bot
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/deepalpha
ExecStart=/usr/bin/python3 deepalpha.py
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable deepalpha
sudo systemctl start deepalpha
sudo journalctl -u deepalpha -f  # View logs
```

---

## Next Steps

- Read [Configuration](configuration.md) to tune risk parameters for your capital size.
- Read [Architecture](architecture.md) to understand how the system works.
- Read [Strategies](strategies.md) to understand how the AI makes trading decisions.
