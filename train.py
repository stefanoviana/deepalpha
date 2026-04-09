"""
DeepAlpha — Training Pipeline (Free Version)
Trains a LightGBM classifier using walk-forward validation.

Usage:
    python download_data.py   # first, download data
    python train.py           # then train the model

The model predicts whether the price will go up or down over the next
N candles, using 15 technical features.
"""

import json
import os
import pickle
import numpy as np
import lightgbm as lgb
import config
from features import build_features, FEATURE_NAMES

# ─── Configuration ──────────────────────────────────────────────────────────
LOOKAHEAD = 3          # Predict price change over next 3 candles
MIN_CANDLES = 200      # Minimum candles required to use a coin
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15       # Remaining 0.15 is test


def load_candles(coin: str) -> dict | None:
    """Load candle data from JSON file."""
    path = os.path.join(config.DATA_DIR, f"{coin}_1h.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def make_labels(close: np.ndarray, lookahead: int = LOOKAHEAD) -> np.ndarray:
    """
    Create binary labels: 1 if price goes up over the next `lookahead`
    candles, 0 otherwise.
    """
    labels = np.zeros(len(close))
    for i in range(len(close) - lookahead):
        future_price = close[i + lookahead]
        labels[i] = 1.0 if future_price > close[i] else 0.0
    # Last `lookahead` labels are invalid — will be trimmed
    return labels


def prepare_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Load all coin data, compute features and labels, return combined X, y.
    """
    all_X = []
    all_y = []

    # Load BTC data for correlation feature
    btc_data = load_candles("BTC")
    btc_close = None
    if btc_data:
        btc_close = np.array([c["c"] for c in btc_data])

    for coin in config.COINS:
        data = load_candles(coin)
        if data is None or len(data) < MIN_CANDLES:
            print(f"  Skipping {coin} — insufficient data")
            continue

        open_ = np.array([c["o"] for c in data])
        high = np.array([c["h"] for c in data])
        low = np.array([c["l"] for c in data])
        close = np.array([c["c"] for c in data])
        volume = np.array([c["v"] for c in data])

        # Align BTC close to same length
        coin_btc = None
        if btc_close is not None and len(btc_close) >= len(close):
            coin_btc = btc_close[-len(close):]

        X = build_features(open_, high, low, close, volume, coin_btc)
        y = make_labels(close)

        # Trim: remove first 30 rows (warmup) and last LOOKAHEAD (invalid labels)
        warmup = 30
        X = X[warmup: -LOOKAHEAD]
        y = y[warmup: -LOOKAHEAD]

        # Remove rows with NaN
        valid = ~np.isnan(X).any(axis=1)
        X = X[valid]
        y = y[valid]

        all_X.append(X)
        all_y.append(y)
        print(f"  {coin:6s} — {len(X):,} samples")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y


def train_model(X: np.ndarray, y: np.ndarray) -> lgb.Booster:
    """
    Train a LightGBM model with walk-forward split.

    Split:
      - 70% training (oldest data)
      - 15% validation (early stopping)
      - 15% test (final evaluation)
    """
    n = len(X)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"\n  Train:  {len(X_train):,} samples")
    print(f"  Val:    {len(X_val):,} samples")
    print(f"  Test:   {len(X_test):,} samples")

    # LightGBM parameters
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "verbose": -1,
    }

    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    val_ds = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_ds)

    # Train with early stopping
    model = lgb.train(
        params,
        train_ds,
        num_boost_round=2000,
        valid_sets=[val_ds],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_pred_bin = (y_pred > 0.5).astype(int)
    accuracy = np.mean(y_pred_bin == y_test)

    print(f"\n  Test accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  Test samples:  {len(X_test):,}")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    sorted_idx = np.argsort(importance)[::-1]
    print("\n  Feature importance (top 10):")
    for rank, idx in enumerate(sorted_idx[:10], 1):
        print(f"    {rank:2d}. {FEATURE_NAMES[idx]:25s} {importance[idx]:,.0f}")

    return model, accuracy


def main() -> None:
    """Full training pipeline."""
    print("=" * 60)
    print("DeepAlpha — Training Pipeline")
    print("=" * 60)

    # 1. Prepare data
    print("\n[1/3] Loading data and building features...")
    X, y = prepare_dataset()
    print(f"\n  Total dataset: {len(X):,} samples, {X.shape[1]} features")
    print(f"  Label balance: {y.mean():.2%} positive")

    # 2. Train model
    print("\n[2/3] Training LightGBM model...")
    model, accuracy = train_model(X, y)

    # 3. Save model
    print(f"\n[3/3] Saving model to {config.MODEL_PATH}...")
    with open(config.MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved ({os.path.getsize(config.MODEL_PATH) / 1024:.0f} KB)")

    print("\n" + "=" * 60)
    print("Training complete! Run `python deepalpha.py` to start trading.")
    print("=" * 60)


if __name__ == "__main__":
    main()
