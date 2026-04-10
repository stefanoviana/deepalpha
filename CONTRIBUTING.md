# Contributing to DeepAlpha

Thanks for your interest in contributing to DeepAlpha! This document explains how to get involved.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Areas We Need Help](#areas-we-need-help)

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Be respectful, constructive, and professional.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/deepalpha.git
   cd deepalpha
   ```
3. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Contribute

### Reporting Bugs

Use the [Bug Report](https://github.com/stefanoviana/deepalpha/issues/new?template=bug_report.md) template. Include:
- Steps to reproduce
- Expected vs actual behavior
- Error logs (redact any API keys or private data)
- Your environment (OS, Python version, package versions)

### Suggesting Features

Use the [Feature Request](https://github.com/stefanoviana/deepalpha/issues/new?template=feature_request.md) template. Explain:
- What problem the feature solves
- How it would work
- Whether you're willing to implement it

### Submitting Code

1. Check existing issues and PRs to avoid duplicate work
2. For large changes, **open an issue first** to discuss the approach
3. Write clean, tested code
4. Submit a pull request

## Development Setup

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/deepalpha.git
cd deepalpha
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Download sample data for testing
python download_data.py

# Train a model locally
python train.py

# Run the bot in paper mode (no real trades)
python deepalpha.py --paper
```

## Pull Request Process

1. **Branch naming**: Use `feature/`, `fix/`, or `docs/` prefixes
   - `feature/add-bollinger-bands`
   - `fix/stop-loss-calculation`
   - `docs/update-architecture-diagram`

2. **Commit messages**: Use clear, descriptive messages
   - Good: `Add kurtosis feature to rolling window calculations`
   - Bad: `update stuff`

3. **PR description**: Explain what changed and why. Include:
   - What problem this solves
   - How you tested it
   - Any backtest results (if applicable)

4. **Review**: A maintainer will review your PR. Be patient and responsive to feedback.

5. **Merge**: Once approved, the PR will be squash-merged into `main`.

## Coding Standards

### Python Style

- **Python 3.10+** required
- **Type hints** encouraged for function signatures
- **Docstrings** for public functions and classes
- Keep functions focused and under 50 lines when possible
- Use descriptive variable names (`rolling_window_size`, not `rws`)

### Project Conventions

- All features go in `features.py`
- Risk management logic goes in `risk_manager.py`
- Configuration constants go in `config.py`
- Never hardcode API keys or secrets — use environment variables

### Data and Model Safety

- Never commit trained model files (`.pkl`, `.joblib`, `.h5`)
- Never commit API keys, private keys, or `.env` files
- Never commit large data files — use `download_data.py` to fetch them
- Always use walk-forward (chronological) data splits, never random splits

### Testing

- Test with historical data before submitting ML changes
- Include backtest results in your PR if you modify features or model logic
- Compare against the baseline model to show improvement

## Areas We Need Help

Here are the highest-impact areas where contributions are welcome:

### Features & Indicators
- New technical indicators (Bollinger Bands, Ichimoku, VWAP)
- Alternative data sources (social sentiment, on-chain metrics)
- Cross-pair correlation features

### Infrastructure
- Unit tests and integration tests
- CI/CD pipeline setup
- Docker containerization
- Logging improvements

### Documentation
- Architecture diagrams
- Feature engineering guide
- Deployment tutorials (AWS, GCP, self-hosted)

### Exchange Support
- Binance Futures adapter
- Bybit adapter
- dYdX v4 adapter

### Research
- Alternative ML models (CatBoost, TabNet, temporal fusion transformers)
- Improved walk-forward validation schemes
- Fee-aware backtesting framework

## Questions?

- Open an [issue](https://github.com/stefanoviana/deepalpha/issues)
- Join the [Discord](https://discord.gg/deepalpha) (coming soon)
- Reach out on [Telegram](https://t.me/DeepAlphaVault)

---

Thank you for helping make DeepAlpha better!
