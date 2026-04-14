# Changelog

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
