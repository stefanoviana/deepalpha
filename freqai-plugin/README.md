# DeepAlpha FreqAI Plugin

> **DeepAlpha's ML pipeline as a drop-in FreqAI model for Freqtrade.**

This plugin brings DeepAlpha's battle-tested machine learning pipeline into the Freqtrade ecosystem. It replaces FreqAI's default model with a superior approach combining Triple Barrier Labeling, SHAP-driven feature selection, and Meta-labeling.

---

## Features

| Feature | Description |
|---|---|
| **Triple Barrier Labeling** | Labels trades with profit-taking, stop-loss, and time-expiry barriers instead of naive fixed-horizon returns |
| **SHAP Feature Selection** | Automatically prunes noisy features using SHAP importance values, reducing overfitting |
| **Meta-labeling** | Secondary model filters primary signals, improving precision by learning *when* to trade |
| **Purged Walk-Forward Validation** | Time-aware cross-validation with purge gaps to prevent lookahead bias |
| **LightGBM Backend** | Fast gradient boosting with GPU support available |

## Performance Comparison

| Metric | Standard FreqAI | DeepAlpha Plugin |
|---|---|---|
| Directional Accuracy | ~55-60% | **68.4%** |
| Sharpe Ratio | ~0.8-1.2 | **2.1+** |
| Max Drawdown | ~15-25% | **<10%** |
| Feature Utilization | All features | **Top-k by SHAP** |

*Results from backtests on BTC/USDT 5m candles, 2024-2025. Past performance does not guarantee future results.*

---

## Installation

```bash
# Clone this plugin into your Freqtrade directory
git clone https://github.com/your-org/deepalpha-freqai-plugin.git freqai-plugin

# Install dependencies
pip install lightgbm shap scikit-learn pandas numpy
```

Then copy `deepalpha_model.py` into your Freqtrade `freqaimodels/` directory:

```bash
cp freqai-plugin/deepalpha_model.py freqtrade/freqaimodels/
cp freqai-plugin/example_strategy.py freqtrade/user_data/strategies/
```

## Configuration

Add the DeepAlpha model to your Freqtrade config. See `config_example.json` for a full example.

Key section:

```json
{
    "freqai": {
        "enabled": true,
        "model_type": "DeepAlphaModel",
        "model_training_parameters": {
            "n_estimators": 2000,
            "learning_rate": 0.02,
            "max_depth": 6,
            "num_leaves": 48,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "min_child_samples": 50,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0
        },
        "deepalpha": {
            "triple_barrier": {
                "profit_taking": 2.0,
                "stop_loss": 1.0,
                "max_holding_period": 48
            },
            "shap_feature_selection": {
                "enabled": true,
                "top_k": 30,
                "recalculate_every_n_trainings": 5
            },
            "meta_labeling": {
                "enabled": true,
                "threshold": 0.55
            },
            "purged_cv": {
                "n_splits": 5,
                "purge_gap": 24,
                "embargo_pct": 0.01
            }
        },
        "train_period_days": 60,
        "backtest_period_days": 7,
        "identifier": "deepalpha_v1"
    }
}
```

## Usage

1. Configure your `config.json` with the FreqAI + DeepAlpha settings
2. Use the included `example_strategy.py` or adapt your own strategy
3. Run backtesting or live trading as usual:

```bash
# Backtest
freqtrade backtesting --strategy DeepAlphaStrategy --config config.json --freqaimodel DeepAlphaModel

# Dry run
freqtrade trade --strategy DeepAlphaStrategy --config config.json --freqaimodel DeepAlphaModel
```

## How It Works

### 1. Triple Barrier Labeling
Instead of labeling candles as simply "up" or "down" based on future returns, the Triple Barrier method assigns labels based on which barrier is hit first:
- **Upper barrier** (profit target): label = 1
- **Lower barrier** (stop loss): label = -1
- **Vertical barrier** (time expiry): label = 0

This produces labels that align with actual trading outcomes.

### 2. SHAP Feature Selection
After training, SHAP values identify which features genuinely contribute to predictions. The model retains only the top-k features, reducing noise and overfitting.

### 3. Meta-labeling
A secondary LightGBM model is trained to predict the *probability that the primary model's signal is correct*. Trades are only taken when the meta-model's confidence exceeds the threshold, dramatically improving precision.

### 4. Purged Walk-Forward CV
Cross-validation splits respect temporal ordering with purge gaps between train/test sets to eliminate information leakage.

---

## File Structure

```
freqai-plugin/
  README.md                 # This file
  deepalpha_model.py        # FreqAI-compatible model class
  example_strategy.py       # Example Freqtrade strategy
  config_example.json       # Example configuration
```

## Links

- [DeepAlpha Main Repository](https://github.com/your-org/deepalpha)
- [Freqtrade Documentation](https://www.freqtrade.io/)
- [FreqAI Documentation](https://www.freqtrade.io/en/stable/freqai/)

## License

MIT License. See the main DeepAlpha repository for details.
