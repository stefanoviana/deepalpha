---
name: Bug Report
about: Report a bug, error, or unexpected behavior
title: "[BUG] "
labels: bug
assignees: ''
---

## Describe the Bug

A clear and concise description of what the bug is.

## Steps to Reproduce

1. Run `python ...`
2. With configuration: ...
3. On market(s): ...
4. Observe error

## Expected Behavior

What you expected to happen.

## Actual Behavior

What actually happened.

## Error Log

```
Paste the full error traceback here.
IMPORTANT: Redact any API keys, private keys, or wallet addresses.
```

## Environment

- **OS**: (e.g., Ubuntu 22.04, Windows 11, macOS 14)
- **Python version**: (e.g., 3.10.12)
- **DeepAlpha version**: (free / Pro, and version number if known)
- **LightGBM version**: (run `pip show lightgbm`)
- **Running on**: (local machine / VPS / Docker)

## Configuration

Relevant settings from your `config.py` or `.env` (redact secrets):

```python
LEVERAGE = ...
MAX_POSITIONS = ...
RETRAIN_INTERVAL = ...
```

## Additional Context

- Did this work before? If so, what changed?
- Does this happen on specific markets or all markets?
- Is this reproducible or intermittent?
- Screenshots or Grafana dashboards (if applicable)
