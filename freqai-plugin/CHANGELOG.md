# Changelog

All notable changes to `deepalpha-freqai` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.1] - 2026-04-16

Docs-only release. No functional or API changes.

### Changed
- Project URLs updated to the new canonical GitHub organisation: `quantumApha/deepalpha` (previously `stefanoviana/deepalpha`)
- README badges and links refreshed

No upgrade required if you are on 1.1.0 — behaviour is identical. `pip install --upgrade deepalpha-freqai` to pull the updated metadata.

---

## [1.1.0] - 2026-04-16

First release with **honest, reproducible profitability** on both bull and bear backtests.

### Honest backtest results

| Market regime | Period | Market change | Bot profit | Win rate | Max drawdown |
|---|---|---|---|---|---|
| Bull | Q1 2024 | +67.66% | **+6.93%** | 62.5% | 2.37% |
| Bear (LUNA crash) | May-Aug 2022 | -33.60% | **+0.41%** | 58.0% | 7.74% |

Tested on 5 majors (BTC/ETH/SOL/BNB/XRP) on Binance Futures with 90-day rolling training windows and 21-day forward tests. Your mileage will vary on different asset baskets and timeframes.

### Added

- **Regime-aware Triple Barrier labels** in `example_strategy.py`. Upper and lower barriers scale asymmetrically based on an EMA24 vs EMA96 regime filter:
  - Bull: profit_taking=1.2 ATR, stop_loss=2.0 ATR (favour longs)
  - Bear: profit_taking=2.0 ATR, stop_loss=1.2 ATR (favour shorts)
  - Sideways: symmetric (1.5 / 1.5)
- **Sequential / memory features** (a.k.a. "poor-man's LSTM") in `feature_engineering_expand_basic`:
  - Rolling log-return stats (mean/std/sum/skew) over horizons 4/12/24/48 bars
  - Return z-score vs 24-bar rolling std
  - Volatility-of-volatility (48-bar)
  - Momentum-of-momentum (close-lag and volume-lag at 1/3/6/12)
  - High-Low range-ratio
- **Hard regime gate** in `populate_entry_trend`: longs only in uptrend, shorts only in downtrend. Counter-trend entries are always rejected (previous soft override produced too many false shorts in bull markets).
- **Dynamic position sizing** via `custom_stake_amount`: stake scales 0.25x-1.75x with the winning-class probability. High-confidence signals get more capital, marginal calls get less.
- **`counter_trend_override` DecimalParameter**: kept for hyperopt compatibility; currently unused by the hard regime gate (see roadmap).

### Changed

- **Default timeframe: 5m -> 1h**. 5m backtests suffered from intrabar noise that consistently triggered stops before signals played out; 1h gives Triple Barrier room to breathe and lifts Sharpe from -21 to +5.98 on the same 2024 bull period.
- **ROI table rescaled** for 1h bars: 6% -> 4% -> 2.5% -> 1.5% across increasing durations.
- **Stop loss**: -3% -> -4% (accommodates 1h bar range without tripping on benign drawdowns).
- **Trailing stop**: disabled by default. The tighter trailing on 5m closed winners too early; you can still re-enable in config.
- **Triple Barrier horizon**: 12 bars on the trading timeframe. On 1h this is 12 hours; on 5m it was only 1 hour.
- **3-class multiclass classifier** by default (SHORT=0, FLAT=1, LONG=2) with `class_weight="balanced"`. The previous binary mode collapsed FLAT into the majority class and introduced hard direction bias.
- **Sell threshold default raised** from 0.55 to 0.60 to compensate for residual downside skew in volatile intraday data.

### Fixed

- Short trades in bull markets no longer generate a stop-loss bloodbath: the previous 0.70 counter-trend override let the model take high-confidence shorts during uptrends that were systematically wrong. The hard regime gate plus 1h timeframe eliminates this failure mode.
- Example strategy now explicitly sets `can_short = True`.

### Breaking changes

- **Default timeframe is now 1h.** If you relied on 5m defaults, set `"timeframe": "5m"` in your config explicitly and lower the training window (`train_period_days`) accordingly. Profitability on 5m is NOT guaranteed by this release.
- **`stoploss` default changed** from -0.03 to -0.04. If you had overridden it in your own strategy subclass, nothing changes.
- **`feature_engineering_expand_basic` emits ~40 new features.** Enable `principal_component_analysis` or raise `DI_threshold` if you were near FreqAI's feature cap.

### Migration from 1.0.x

```bash
pip install --upgrade deepalpha-freqai
```

Then in your config:

```json
{
  "timeframe": "1h",
  "freqai": {
    "feature_parameters": {
      "include_timeframes": ["1h", "4h"],
      "indicator_periods_candles": [10, 20, 50],
      "label_period_candles": 12,
      "include_shifted_candles": 3
    },
    "train_period_days": 90,
    "backtest_period_days": 21
  }
}
```

Copy `example_strategy.py` from the repo or update your own strategy to include the new `feature_engineering_expand_basic` block and the regime gate in `populate_entry_trend`.

---

## [1.0.5] - 2026-04-16

### Fixed

- **Critical** - `predict()` now returns the correct schema expected by FreqAI:
  - `pred_df`: DataFrame with `dk.label_list` columns first, then one column per class (as strings, so `remove_features_from_df` does not choke on int column names).
  - `do_predict`: 1D `numpy.ndarray` of ints (not a DataFrame).
- Fixes the `ValueError: array length 0 does not match index length ...` crash in `FreqaiDataKitchen.get_predictions_to_append()` during backtesting.
- Meta-labeling filter no longer appends extra columns to `pred_df` - it only modifies `do_predict`, keeping `append_dict` lengths in sync.

### Changed

- Class-probability columns are now stringified (`str(c)`) to avoid downstream `col.startswith(...)` failures on integer class labels.
- `dk.DI_values` is explicitly set to zeros when Dissimilarity Index is not computed, matching `BaseClassifierModel` conventions.

### Notes

This is the first **production-ready** release. Upgrade from any earlier 1.0.x is strongly recommended - prior versions would crash during backtesting or dry-run on FreqAI 2024+.

---

## [1.0.4] - 2026-04-15

### Fixed

- Additional `predict()` safety: fall back to `self.model` if `self.primary_model` was not restored after FreqAI's model persistence reload.
- Raise a clear `RuntimeError` if `predict()` is called before `fit()`/load, instead of silently returning garbage.

### Known issues

- Predict schema still mismatched in some edge cases (fully resolved in 1.0.5).

---

## [1.0.3] - 2026-04-14

### Changed

- `predict()` now reuses `BaseClassifierModel`'s feature preparation path (`dk.find_features` -> `dk.filter_features`) before applying SHAP-selected columns, ensuring the inference pipeline mirrors training.
- `dk.data_dictionary["prediction_features"]` is populated so meta-labeling and downstream FreqAI code can reference the exact frame used for inference.

---

## [1.0.2] - 2026-04-13

### Added

- `self.model = primary` is now set inside `fit()`, so FreqAI's model persistence layer (`dd.load_data`) can serialise and restore DeepAlphaModel across restarts.

### Fixed

- Plugin no longer loses its primary model after a Freqtrade restart - previously `self.primary_model` was not serialised.

---

## [1.0.1] - 2026-04-12

### Changed

- **Package structure** - moved from flat-layout (`deepalpha_model.py` at root) to proper package folder (`deepalpha_freqai/deepalpha_model.py`). Fixes conflicts with other top-level `deepalpha_model` modules in user environments.
- **Lazy Freqtrade import** - `DeepAlphaModel` is now imported on attribute access via `__getattr__`, so `import deepalpha_freqai` no longer crashes on systems without Freqtrade. Utilities (`PurgedWalkForwardCV`, `apply_triple_barrier`, `select_features_by_shap`) work standalone.

### Fixed

- `pip install deepalpha-freqai` on a Freqtrade-less environment now succeeds (useful for research notebooks / CI).

---

## [1.0.0] - 2026-04-11

### Added

- Initial public release on PyPI.
- `DeepAlphaModel` - FreqAI-compatible classifier wrapping:
  - Triple Barrier Labeling (`apply_triple_barrier`)
  - SHAP-based top-k feature selection (`select_features_by_shap`)
  - Meta-labeling via a second LightGBM model
  - Purged walk-forward cross-validation (`PurgedWalkForwardCV`)
- Example Freqtrade strategy (`example_strategy.py`).
- Example config (`config_example.json`).
- Unit tests under `tests/`.

---

## Upgrade guide

### From 1.0.x (any) to 1.0.5

Just run:

```bash
pip install --upgrade deepalpha-freqai
```

No config changes required. If you were seeing `array length 0` crashes or `KeyError` on class columns, they are fixed.

### From source install to PyPI install

1. Remove the manual copy of `deepalpha_model.py` from `freqtrade/freqaimodels/`.
2. `pip install deepalpha-freqai`.
3. Keep your existing `config.json` as-is.
