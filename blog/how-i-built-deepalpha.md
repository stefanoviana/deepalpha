# How I Built a Self-Learning AI Trading Bot for 229 Crypto Markets

*Published April 2026 | ~8 min read*

Most trading bots fail. Not because the predictions are wrong, but because they never learn *why* they're wrong. I built DeepAlpha to fix that.

After burning through several iterations, losing money on fees, and watching "profitable" backtests blow up in production, I built a system that actually works: an AI trading bot that trades 229 perpetual futures markets on Hyperliquid, retrains itself every 15 minutes, and treats every losing trade as a lesson worth 5x more than a winner.

This post is a technical walkthrough of the architecture, the mistakes I made, and what I'd do differently.

---

## The Problem: Why Most Trading Bots Lose Money

Traditional quant bots have three fatal flaws:

1. **They predict direction, not profit.** A model that says "price goes up" with 60% accuracy sounds good — until you realize that fees, slippage, and stop-losses eat your edge alive. Accuracy is not the same as profitability.

2. **They don't learn from mistakes.** You train a model, deploy it, and it keeps making the same errors. If it consistently gets stopped out on low-liquidity pairs at 3am UTC, it has no mechanism to adapt.

3. **They ignore execution costs.** A backtest that assumes instant fills at mid-price is fiction. In real markets, you're either paying taker fees (which destroy thin edges) or you're posting limit orders that may never fill.

I wanted a system that optimizes for *net profit after fees*, learns from its own trading history, and adapts execution to minimize costs.

---

## The Architecture

DeepAlpha runs as a single Python process with four core loops:

```
Market Data (229 pairs via Hyperliquid WebSocket)
    |
Feature Engineering (50 features per pair)
    |
ML Ensemble (LightGBM + XGBoost)
    |
PPO Reinforcement Learning (position sizing + timing)
    |
Maker-Only Execution Engine
    |
Self-Learning Retrain Loop (every 15 min)
```

Each component was built to solve a specific failure mode I encountered in production.

---

## Feature Engineering: Beyond RSI and Moving Averages

The first version of DeepAlpha used 15 standard technical indicators — RSI, ATR, EMA crossovers, momentum. It worked, but the signal was weak. The model was seeing what everyone else sees.

I expanded to 50 features by drawing from academic quantitative finance and from the [intelligent-trading-bot](https://github.com/intelligent-trading-bot) project. The features that moved the needle most:

- **Distribution features**: Skewness and kurtosis of returns over rolling windows. Markets that are negatively skewed tend to crash harder; high kurtosis means fat tails are in play. These features capture regime information that simple moving averages miss entirely.

- **Area ratio**: The ratio of price area above vs. below a moving average over a lookback window. This is a more stable trend signal than a simple crossover because it integrates over time rather than reacting to a single crossing point.

- **Order Book Imbalance (OBI)**: The ratio of bid-side to ask-side depth in the L2 orderbook. When bids massively outweigh asks, short-term price pressure is upward. This feature alone improved precision on short-term signals by ~4%.

- **Cumulative Volume Delta (CVD)**: The running sum of (buy volume - sell volume). CVD divergence from price is one of the strongest mean-reversion signals I've found — when price makes a new high but CVD doesn't, smart money is selling into the rally.

- **Fear & Greed Index**: An external sentiment feature pulled from the API. I was skeptical, but it turns out that extreme fear readings (below 20) are genuinely predictive of short-term bounces in crypto.

All features are computed on multiple timeframes (5m, 15m, 1h, 4h) and z-score normalized before feeding to the model.

---

## The ML Ensemble: LightGBM + XGBoost

The prediction engine is an ensemble of LightGBM and XGBoost gradient-boosted trees. Why trees instead of a neural network?

1. **Tabular data.** For structured/tabular features, gradient-boosted trees consistently outperform deep learning. This isn't controversial — it's the consensus in the ML competition community (Kaggle, etc.).

2. **Fast retraining.** A full retrain on 1M+ samples takes ~90 seconds on CPU. Neural networks would take 10-50x longer, making the 15-minute retrain loop impractical.

3. **Interpretability.** I can inspect feature importances after every retrain. When the model suddenly starts weighting a feature heavily, I want to know about it.

The ensemble combines predictions with learned weights. LightGBM tends to be better on trending markets; XGBoost handles mean-reversion regimes better. The ensemble outperforms either model alone by 2-3% in walk-forward tests.

**Walk-forward validation** is critical. I never do random train/test splits — that leaks future information and inflates accuracy. Data is split chronologically: 70% train (oldest), 15% validation (middle), 15% test (newest). The model only sees the past to predict the future, exactly as it will in production.

---

## The Self-Learning Loop: Weighting Losses 5x

This is the feature I'm most proud of. Every 15 minutes, DeepAlpha retrains on its own recent trading data. But here's the key: **losing trades are weighted 5x more than winning trades in the loss function.**

Why? Because in trading, losses are more informative than wins. A winning trade could be luck. A losing trade tells you something specific: the model was wrong in this exact market condition, with these exact features, at this exact time. By overweighting losses, the model aggressively corrects its weak spots.

The implementation uses LightGBM's sample weight parameter:

```python
weights = np.where(trade_outcomes < 0, 5.0, 1.0)
train_data = lgb.Dataset(X_train, label=y_train, weight=weights)
```

After implementing this, the model's drawdown-to-peak ratio improved by roughly 30%. It didn't increase raw accuracy much, but it significantly reduced the *magnitude* of losing trades — the model learned to avoid the conditions where it loses big.

---

## PPO Reinforcement Learning: Optimizing Profit, Not Accuracy

The ML ensemble tells DeepAlpha *what* to trade. The PPO (Proximal Policy Optimization) agent decides *how much* and *when*.

Why add RL on top of ML predictions? Because position sizing and entry timing are sequential decision problems. The optimal position size depends on your current exposure, recent volatility, and how much drawdown you can tolerate. This is exactly what RL is designed for.

The PPO agent's reward function is **net PnL after fees** — not accuracy, not Sharpe ratio, not any proxy metric. The agent learns that:

- Taking smaller positions in choppy markets reduces drawdown
- Scaling into positions (rather than entering all at once) improves average entry price
- Skipping marginal signals (confidence below a threshold) is often the highest-EV play

The PPO layer added roughly 15% to risk-adjusted returns in backtesting, primarily by reducing position sizes during volatile regimes.

---

## Maker-Only Execution: From -$115 to Near-Zero Fees

This was the most painful lesson. In the first week of live trading, DeepAlpha generated $80 in gross profit and paid $115 in taker fees. Net result: -$35.

Hyperliquid charges 0.035% for taker orders but offers 0.01% rebates for maker orders (limit orders that add liquidity). On 229 markets with frequent rebalancing, the difference between taker and maker is the difference between profit and loss.

I rewrote the execution engine to be 100% maker-only:

1. Place limit orders 1-2 ticks inside the spread
2. Wait up to 30 seconds for fills
3. If unfilled, adjust price to the current best bid/ask
4. Cancel and retry if the market moves beyond a threshold

This reduced effective fees from 0.035% to approximately -0.01% (we actually get paid to trade). On a system that executes hundreds of trades per day, this single change turned a losing system into a profitable one.

---

## Results: Real Trading Data

DeepAlpha has been trading live with approximately $1,100 in capital across 229 perpetual futures markets on Hyperliquid. Some key metrics from production:

- **Markets traded**: 229 perpetual futures pairs
- **Retrain frequency**: Every 15 minutes
- **Features**: 50 per pair
- **Execution**: 100% maker orders
- **Average fee per trade**: Near zero (maker rebates offset costs)

The system runs autonomously on a VPS with Telegram notifications for all trades, retrains, and anomalies.

---

## What's Next

DeepAlpha is far from finished. The roadmap includes:

1. **L2 Orderbook Recording**: Storing full orderbook snapshots to build better microstructure features. The goal is capturing supply/demand imbalances before they show up in price.

2. **HFT Backtesting Framework**: Building a tick-level backtester that accounts for queue position, latency, and partial fills. Most crypto backtesting tools are bar-based and miss execution dynamics entirely.

3. **Multi-Exchange Expansion**: Extending execution to Binance, Bybit, and dYdX for cross-exchange arbitrage and better liquidity access.

4. **Vault Structure**: Creating a DeFi vault where external capital can benefit from DeepAlpha's strategies, with transparent on-chain performance tracking.

---

## Try It Yourself

DeepAlpha is open source. The free version includes the LightGBM model, walk-forward validation, and the full training pipeline.

- **GitHub**: [github.com/stefanoviana/deepalpha](https://github.com/stefanoviana/deepalpha)
- **Telegram**: [t.me/DeepAlphaVault](https://t.me/DeepAlphaVault)
- **Pro Version**: [DeepAlpha Pro on Gumroad](https://stefanocrypto.gumroad.com/l/ezilv) — XGBoost ensemble, 50 features, PPO agent, ATR trailing stops, Grafana dashboard

If you build something with it, I'd love to hear about it. Open an issue or reach out on Telegram.

---

*Written by Stefano — building quantitative trading systems since 2024.*
