# FAQ

Frequently asked questions about DeepAlpha.

---

### Why does it only trade SHORT?

**The AI model adapts to market conditions.** If the market is in a sustained downtrend, the model learns that most coins have a higher probability of going down, so it generates SHORT signals almost exclusively.

This is not a bug -- it reflects the model's assessment of current conditions. The Pro version includes a **BTC Regime Filter** that explicitly detects bull/bear regimes and restricts signal direction accordingly, but even the free version naturally tilts toward the dominant trend through its training data.

If you want to force the bot to take both directions equally, you can lower `MIN_CONFIDENCE` (e.g., to 0.52), but this increases the number of marginal trades and may reduce overall profitability.

---

### Why are fees so high?

DeepAlpha's free version uses **Limit IOC (Immediate-or-Cancel)** orders, which are classified as **taker** orders on Hyperliquid. Taker fees are currently 0.035% per side (0.07% round trip).

On a $500 notional position:

- Entry fee: $0.175
- Exit fee: $0.175
- Total: $0.35 per round trip

This adds up quickly with frequent trading. The Pro version uses **maker orders** (posted to the book at a favorable price and waiting for a fill), which have fees of 0.01% or less -- in some cases, maker orders earn a rebate (negative fees).

To minimize fee impact:

- Increase `TAKE_PROFIT_PCT` so each winning trade exceeds the fee cost by a wider margin.
- Reduce `MAIN_LOOP_SECONDS` only if you are on maker orders; more frequent scanning with taker orders just generates more fees.

---

### How much capital do I need?

**Minimum recommended: $500 USD.**

Here is why: with default settings (5x leverage, 10% risk per trade), each position uses $50 of margin and $250 of notional value. At 3 maximum positions, you need $150 in margin plus a buffer for unrealized losses and fees.

Below $500, position sizes become very small and fees consume a disproportionate share of profits.

| Account Size | Margin per Trade | Notional per Trade | Notes |
|-------------|-----------------|-------------------|-------|
| $500 | $50 | $250 | Minimum viable |
| $1,000 | $100 | $500 | Comfortable |
| $5,000 | $500 | $2,500 | Optimal for default settings |

---

### Can I use it on Binance?

**Currently, DeepAlpha only supports Hyperliquid.** The SDK, order types, and API structure are specific to Hyperliquid's L1 exchange.

Reasons for choosing Hyperliquid:

- **On-chain orderbook** -- fully transparent, no exchange manipulation.
- **Low fees** -- 0.035% taker, 0.01% maker (lower than most CEXes for small accounts).
- **No KYC required** -- trade with just a wallet.
- **Public API** -- candle data, funding rates, and orderbook snapshots require no API key.
- **Fast execution** -- sub-second order fills on L1.

Porting to Binance would require replacing the Hyperliquid SDK calls with Binance API equivalents and adapting the order types. The AI model and feature engineering would remain the same.

---

### How do I set up Telegram notifications?

1. Open Telegram and search for [@BotFather](https://t.me/BotFather).
2. Send `/newbot` and follow the instructions to create a bot. You will receive a bot token.
3. Start a conversation with your new bot, then visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` to find your chat ID.
4. Add both values to your `.env` file:

```bash
TELEGRAM_TOKEN=1234567890:ABCdef...
TELEGRAM_CHAT_ID=123456789
```

The bot sends notifications for:

- Trade entries (coin, side, price, quantity, confidence)
- Trade exits (SL or TP, entry/exit prices, realized PnL)
- Bot start/stop events
- Model retrain completions

---

### What happens if the bot crashes?

DeepAlpha handles errors gracefully:

- **API errors** are caught per-coin and logged. The bot continues scanning other coins.
- **Unhandled exceptions** trigger a 30-second pause, then the loop resumes.
- **Open positions are not affected** by a bot restart. On startup, `_sync_positions()` detects any existing exchange positions and tracks them.

For production use, run the bot under `systemd` (Linux) or `screen` with automatic restart. See the [Getting Started](getting-started.md#running-on-a-vps-recommended) guide.

---

### How often does it retrain?

Every **2 hours** by default (`RETRAIN_INTERVAL=7200`). The retrain uses the same data in the `data/` directory, so it will only see new data if you also run `download_data.py` periodically.

For continuous learning, you can set up a cron job:

```bash
# Crontab: download fresh data every hour
0 * * * * cd /root/deepalpha && python download_data.py >> /var/log/deepalpha-data.log 2>&1
```

The Pro version retrains every 15 minutes and downloads data automatically.

---

### What accuracy should I expect?

Typical test accuracy is **55-62%** on out-of-sample data. This may seem low, but combined with a favorable reward-to-risk ratio (3% TP vs 2% SL = 1.5:1 R:R), even 55% accuracy produces positive expected value:

```
Expected value per trade = (win_rate * avg_win) - (loss_rate * avg_loss)
                         = (0.55 * 3%) - (0.45 * 2%)
                         = 1.65% - 0.90%
                         = +0.75% per trade (before fees)
```

The Pro version with XGBoost ensemble achieves 65-73% accuracy on test data.

---

### Can I add my own coins?

Yes. Edit the `COINS` list in `config.py`:

```python
COINS: list[str] = [
    "BTC", "ETH", "SOL", "DOGE", "AVAX",
    # Add your coins here:
    "NEAR", "FIL", "ATOM",
]
```

Then re-download data and retrain:

```bash
python download_data.py
python train.py
```

Any coin listed on Hyperliquid perpetuals can be added. Use the ticker symbol without the "-PERP" suffix.

---

### Why is the model file so small?

`model.pkl` is typically 200-400 KB. LightGBM models are compact because gradient boosted trees store only split points and leaf values, not raw data. Even a model trained on 170,000+ samples compresses into a small file.

---

### Is this profitable?

DeepAlpha is a tool, not a guaranteed money printer. Profitability depends on:

- Market conditions (trending markets favor the model; choppy sideways markets do not)
- Configuration (leverage, risk per trade, confidence threshold)
- Fees (taker vs maker)
- Capital size (fees are proportionally larger on small accounts)

The model has positive expected value based on backtesting and walk-forward validation, but past performance does not guarantee future results. Always start with small capital and monitor performance before scaling up.
