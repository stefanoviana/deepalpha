[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![FreqAI Compatible](https://img.shields.io/badge/FreqAI-compatible-orange.svg)](https://www.freqtrade.io/en/stable/freqai/)
[![PyPI version](https://img.shields.io/pypi/v/deepalpha-freqai.svg)](https://pypi.org/project/deepalpha-freqai/)

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

### Option 1: Install via pip (recommended)

```bash
pip install deepalpha-freqai
```

### Option 2: Install from source

```bash
# Clone the repository
git clone https://github.com/stefanoviana/deepalpha.git
cd deepalpha

# Install in development mode
pip install -e .
```

### Option 3: Manual setup

```bash
# Clone this plugin into your Freqtrade directory
git clone https://github.com/stefanoviana/deepalpha.git freqai-plugin

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
                "max_holding_period": 48,
                "volatility_window": 20
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
  setup.py                  # PyPI packaging
  __init__.py               # Package init
  deepalpha_model.py        # FreqAI-compatible model class
  example_strategy.py       # Example Freqtrade strategy
  config_example.json       # Example configuration
  tests/
    test_deepalpha_model.py # Unit tests
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Contributing to Freqtrade

We welcome contributions and are working towards submitting DeepAlpha as an official FreqAI model within Freqtrade. Here is how you can help or submit your own changes upstream:

### Submitting as a Freqtrade PR

1. **Fork** the [Freqtrade repository](https://github.com/freqtrade/freqtrade) on GitHub.
2. **Create a feature branch** from the `develop` branch:
   ```bash
   git checkout develop
   git checkout -b feat/deepalpha-freqai-model
   ```
3. **Add the model file** to `freqtrade/freqai/prediction_models/DeepAlphaModel.py`.
4. **Add the example strategy** to `freqtrade/templates/DeepAlphaStrategy.py`.
5. **Add unit tests** to `tests/freqai/test_deepalpha_model.py`.
6. **Update the FreqAI documentation** in `docs/freqai.md` to reference the new model.
7. **Ensure all tests pass**:
   ```bash
   pytest tests/freqai/ -v
   ```
8. **Submit a Pull Request** against the `develop` branch with:
   - A clear description of the model and its advantages
   - Backtest results demonstrating improved performance
   - Links to the academic references (Triple Barrier from *Advances in Financial Machine Learning* by Marcos Lopez de Prado)

### Contributing to this Plugin

1. Fork this repository at [github.com/stefanoviana/deepalpha](https://github.com/stefanoviana/deepalpha).
2. Create a feature branch: `git checkout -b feat/your-feature`.
3. Write tests for any new functionality.
4. Ensure all tests pass: `pytest tests/ -v`.
5. Submit a Pull Request with a clear description of your changes.

### Code Style

- Follow PEP 8 and the existing code conventions.
- All public functions and classes must have docstrings.
- Type hints are required for all function signatures.

---

## Links

- [DeepAlpha Repository](https://github.com/stefanoviana/deepalpha)
- [Freqtrade Documentation](https://www.freqtrade.io/)
- [FreqAI Documentation](https://www.freqtrade.io/en/stable/freqai/)
- [Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) (Triple Barrier reference)

## License

MIT License. See the main DeepAlpha repository for details.
