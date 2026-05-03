# Contributing to DeepAlpha

Thank you for your interest in contributing! This guide will help you get started.

## Project Structure

```
deepalpha/
├── deepalpha.py          # Main bot entry point (self-hosted)
├── config.py             # Configuration loader (.env)
├── features.py           # ML feature engineering (72 features)
├── train.py              # Model training script
├── risk_manager.py       # Position sizing, SL/TP, drawdown limits
├── exchange_adapter.py   # Exchange abstraction layer (CCXT)
├── pump_scanner.py       # Real-time pump detection
├── order_flow_analyzer.py # L2 orderbook analysis
├── regime_detector.py    # Market regime (bull/bear/sideways)
├── liquidation_levels.py # Liquidation heatmap
├── gnn_model.py          # Graph Neural Network (experimental)
├── tft_model.py          # Temporal Fusion Transformer
├── transformer_gru_model.py # Transformer-GRU hybrid
├── requirements.txt      # Python dependencies
├── .env.example          # Config template
├── Dockerfile            # Container setup
└── tests/                # Unit tests (needs work!)
```

## How to Set Up Your Dev Environment

```bash
# 1. Fork the repo on GitHub, then:
git clone https://github.com/YOUR_USERNAME/deepalpha.git
cd deepalpha

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy config
cp .env.example .env
# Edit .env with your exchange API keys

# 5. Run tests (when available)
pytest tests/ -v

# 6. Run the bot
python deepalpha.py
```

## How to Contribute

### Step 1: Pick an Issue
- Look for issues labeled `help wanted` or `good first issue`
- Comment "I'd like to work on this" so others know
- Wait for a maintainer to assign it to you

### Step 2: Create a Branch
```bash
git checkout -b feature/your-feature-name
```

### Step 3: Write Your Code
- Follow the existing code style
- Add docstrings to new functions
- Handle errors gracefully (no bare `except:`)
- Use `logger` for logging, not `print()`

### Step 4: Test Your Changes
```bash
# Run the bot in paper mode to test
# In .env, set: PAPER_MODE=true
python deepalpha.py

# Run unit tests
pytest tests/ -v
```

### Step 5: Submit a Pull Request
```bash
git add .
git commit -m "Add: brief description of your change"
git push origin feature/your-feature-name
```
Then go to GitHub and create a Pull Request.

## Contribution Areas

### 1. ML Features (features.py)
Add new features to improve prediction accuracy. Current: 72 features.

```python
# Example: adding a new feature
def build_features(candles, indicators):
    features = {}
    # ... existing features ...

    # YOUR NEW FEATURE:
    features["my_new_feature"] = calculate_something(candles)

    return features
```

**How to validate:** Run `python train.py` and compare accuracy before/after.

### 2. Exchange Support (exchange_adapter.py)
Add support for new exchanges via CCXT.

```python
# Test your exchange:
import ccxt
ex = ccxt.your_exchange({"apiKey": "...", "secret": "..."})
ex.load_markets()
print(ex.fetch_ticker("BTC/USDT"))
print(ex.fetch_ohlcv("BTC/USDT", "1h", limit=10))
```

### 3. Unit Tests (tests/)
We need tests! Use pytest + pytest-asyncio.

```python
# tests/test_features.py
import pytest
from features import build_features

def test_feature_count():
    candles = [...]  # mock data
    features = build_features(candles, {})
    assert len(features) == 72
```

### 4. Documentation
- Translate README to other languages
- Improve inline code comments
- Write blog posts about strategies

### 5. Docker & DevOps
- Improve Dockerfile
- Add docker-compose.yml
- CI/CD pipeline with GitHub Actions

## Code Style

- Python 3.9+
- Use type hints where possible
- Use `async/await` for exchange calls
- Constants in UPPER_CASE
- Max line length: 120 chars
- Use f-strings for formatting

## Rewards

| Contribution | Reward |
|---|---|
| Bug fix (PR merged) | Credit in README |
| Small feature (PR merged) | 1 month Pro access |
| Major feature (PR merged) | Lifetime access |
| Translation | Credit in README |

## Questions?

- Open a [Discussion](https://github.com/stefanoviana/deepalpha/discussions)
- Join [Discord](https://discord.gg/P4yX686m)
- Message [@DeepAlphaVault_bot](https://t.me/DeepAlphaVault_bot) on Telegram

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
