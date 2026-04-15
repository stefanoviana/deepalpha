# Changelog

## v1.5.0 (2026-04-15)
- **HMM regime detection** — 3-state Hidden Markov Model (bull/sideways/bear) corrects direction bias
- **SHORT bias fix** — filters >90% same-direction clusters in bearish regimes
- **Asymmetric reversal removed** — disabled reversal strategy that conflicted with AI signals
- **Telegram hold decisions** — interactive hold/close prompt at 4h trade expiry
- **SHORT bias protection** — >90% same-direction signals filtered as noise

## v1.3.0 (2026-04-15)
- **Auto-update models** — AI models auto-download from license server every 24h (no manual retraining)
- **Setup wizard** — `setup.bat` (Windows) / `setup.sh` (Mac/Linux) for 2-minute install with guided prompts
- **License key integration** — Gumroad webhook auto-provisions license keys on purchase
- **Bitget copy trading referral** — referral integration for copy trading signups

## v1.2.0 (2026-04-14)
- **Clean confidence**: removed 5 ad-hoc modifiers (mean-rev, PPO, OFI, regime, VPIN/liq sizing). Model probability is now used raw
- **Cost-aware trading**: trades only open when expected edge > 1.5x transaction costs
- **Conservative sizing**: risk per trade reduced from 40% to 15% of equity
- **L2 feature fix**: live features now use same candle proxies as training (was using real L2 = train/serve skew)
- **TP multi-target fix**: TP raised to 4x ATR so T2 (3.5x ATR) can trigger before full close
- **AI coin filter**: only 27 trained crypto, removed FTM/MKR (not on Bitget)

## v1.1.0 (2026-04-14)
- **Bitget exchange support** — full adapter via ccxt (copy trading compatible)
- **Fee rate fix** — correct 0.06% taker fee for Bitget (was hardcoded at 0.035%)
- **16 critical bug fixes** in Pro version (fee tracking, PnL calculations, exchange compatibility)
- **AI coin filter** — model only trades on 27 trained crypto assets
- **Backtest report** — verified walk-forward results page at deepalphabot.com/results.html
- **License server** — auto license creation via Gumroad webhook

## v1.0.0 (2026-04-08)
- Initial open-source release
- LightGBM model with walk-forward validation
- 15 technical features
- Risk management (5x leverage, SL/TP, circuit breaker)
- Telegram notifications
- Hyperliquid L1 perpetuals support

## Roadmap
- [x] More exchange support (Bitget added)
- [ ] XGBoost ensemble (available in Pro)
- [ ] ATR dynamic trailing stops (available in Pro)
- [ ] BTC regime filter (available in Pro)
- [ ] Web dashboard
- [ ] Backtesting improvements
