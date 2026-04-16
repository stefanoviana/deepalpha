"""
DeepAlpha FreqAI - standalone demo
==================================

Runs the core DeepAlpha pipeline on synthetic OHLCV data, WITHOUT
requiring a Freqtrade install. Proves the package works out of the box.

Usage:
    pip install deepalpha-freqai
    python demo.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from deepalpha_freqai import apply_triple_barrier, select_features_by_shap


def make_synthetic_ohlcv(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Geometric Brownian motion OHLCV with a few indicator-ish features."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, n)
    close = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({"close": close})
    df["ret_1"] = df["close"].pct_change()
    df["ret_5"] = df["close"].pct_change(5)
    df["sma_10"] = df["close"].rolling(10).mean() / df["close"] - 1
    df["sma_50"] = df["close"].rolling(50).mean() / df["close"] - 1
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df["rsi_like"] = df["ret_1"].rolling(14).apply(
        lambda x: (x[x > 0].sum() / (abs(x).sum() + 1e-9)) * 100, raw=False
    )
    # Noise features (SHAP should prune these)
    for k in range(10):
        df[f"noise_{k}"] = rng.normal(0, 1, n)
    return df.dropna().reset_index(drop=True)


def main() -> None:
    print("=" * 60)
    print("DeepAlpha FreqAI - Standalone Demo")
    print("=" * 60)

    # 1. Synthetic data
    df = make_synthetic_ohlcv(n=5000)
    print(f"\n[1] Generated {len(df)} bars with {df.shape[1]} features.")

    # 2. Triple Barrier labels
    labels = apply_triple_barrier(
        df, close_col="close", profit_taking=2.0, stop_loss=1.0,
        max_holding_period=48, volatility_window=20,
    )
    print(f"\n[2] Triple Barrier labels: {labels.value_counts().to_dict()}")

    # 3. Train primary LightGBM
    feature_cols = [c for c in df.columns if c != "close"]
    X, y = df[feature_cols], labels
    split = int(len(X) * 0.8)
    X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

    model = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X_tr, y_tr)
    train_acc = model.score(X_tr, y_tr)
    test_acc = model.score(X_te, y_te)
    print(f"\n[3] Primary LGBM - train acc: {train_acc:.3f}  test acc: {test_acc:.3f}")

    # 4. SHAP feature selection
    top_features = select_features_by_shap(model, X_tr, top_k=5)
    print(f"\n[4] SHAP top-5 features: {top_features}")

    # 5. Predict
    preds = model.predict(X_te.head(10))
    print(f"\n[5] Sample predictions on test set: {preds.tolist()}")
    print(f"    True labels:                   {y_te.head(10).tolist()}")

    print("\n" + "=" * 60)
    print("Done. Install deepalpha-freqai into your Freqtrade env to use")
    print("DeepAlphaModel as a FreqAI plugin. See README for config.")
    print("=" * 60)


if __name__ == "__main__":
    main()
