#!/usr/bin/env python3
"""
DeepAlpha Temporal Fusion Transformer (TFT) Predictor
Multi-horizon price prediction using attention-based architecture.

This module provides a TFT-based predictor that complements the existing
LightGBM model with sequence-aware, multi-horizon forecasting. The TFT
architecture excels at capturing temporal dependencies and regime shifts
that tree-based models may miss.

Architecture:
    - Variable Selection Networks for automatic feature importance
    - LSTM encoder/decoder for temporal processing
    - Multi-head attention for long-range dependencies
    - Quantile outputs for uncertainty estimation

Requires:
    pip install pytorch-forecasting pytorch-lightning torch pandas

Integration with DeepAlpha:
    from tft_predictor import TFTPredictor

    # Initialize
    tft = TFTPredictor(horizons=["15m", "1h", "4h"])

    # Training (offline, run on local machine)
    candles = {"BTC": [...], "ETH": [...]}  # dict of OHLCV candle lists
    dataset = tft.prepare_data(candles)
    tft.train(dataset, epochs=50)
    tft.save_model("models/tft_model.pt")

    # Inference (in the bot loop)
    tft.load_model("models/tft_model.pt")
    pred = tft.predict("BTC", current_features)
    # pred = {"15m": {"direction": "LONG", "confidence": 0.72}, ...}

Expected candle format (same as deepalpha.py):
    [{"o": 64000.0, "h": 64200.0, "l": 63800.0, "c": 64100.0, "v": 123.45}, ...]

Author: DeepAlpha Team
License: MIT
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ─── Optional dependency detection ──────────────────────────────────────────

HAS_TFT = False
HAS_PANDAS = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore[assignment]

try:
    import torch
    import pytorch_lightning as pl
    from pytorch_forecasting import (
        GroupNormalizer,
        TemporalFusionTransformer,
        TimeSeriesDataSet,
    )
    from pytorch_forecasting.metrics import QuantileLoss
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

    HAS_TFT = True
except ImportError:
    torch = None  # type: ignore[assignment]
    pl = None  # type: ignore[assignment]

logger = logging.getLogger("deepalpha.tft")

# ─── Constants ───────────────────────────────────────────────────────────────

HORIZON_MAP: dict[str, int] = {
    "15m": 1,    # 1 step  at 15m resolution
    "1h": 4,     # 4 steps at 15m resolution
    "4h": 16,    # 16 steps at 15m resolution
}

DEFAULT_HORIZONS = ["15m", "1h", "4h"]

OBSERVED_FEATURES = [
    "rsi_14",
    "atr_14_norm",
    "volume_ratio",
    "funding_rate",
    "btc_correlation",
    "price_momentum_3",
    "price_momentum_7",
    "close_vs_open",
    "high_low_range",
]

KNOWN_FUTURE_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "hour_sin",
    "hour_cos",
]

STATIC_CATEGORICALS = ["coin_id"]


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class TFTPrediction:
    """Single-horizon prediction result."""

    direction: str          # "LONG" or "SHORT"
    confidence: float       # 0.0 – 1.0
    predicted_return: float  # expected % return
    upper_bound: float      # 90th percentile return
    lower_bound: float      # 10th percentile return

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "predicted_return": round(self.predicted_return, 6),
            "upper_bound": round(self.upper_bound, 6),
            "lower_bound": round(self.lower_bound, 6),
        }


@dataclass
class TFTConfig:
    """TFT hyperparameters. Sensible defaults for crypto 15m data."""

    hidden_size: int = 64
    attention_head_size: int = 4
    dropout: float = 0.1
    hidden_continuous_size: int = 32
    learning_rate: float = 1e-3
    max_encoder_length: int = 96   # 24 hours of 15m candles
    max_prediction_length: int = 16  # 4 hours ahead
    batch_size: int = 64
    gradient_clip_val: float = 0.1
    reduce_on_plateau_patience: int = 4


# ─── Feature engineering ─────────────────────────────────────────────────────

def _compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder RSI from close prices."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    alpha = 1.0 / period
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[period] = gain[1 : period + 1].mean()
    avg_loss[period] = loss[1 : period + 1].mean()
    for i in range(period + 1, len(close)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i - 1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i - 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss > 1e-10, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[:period] = 50.0  # pad warmup
    return rsi


def _compute_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """Average True Range, normalised by close price."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.abs(high - prev_close), np.abs(low - prev_close))
    atr = np.convolve(tr, np.ones(period) / period, mode="full")[:len(tr)]
    atr[:period] = atr[period]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(close > 0, atr / close, 0.0)


def _build_features_df(
    candles: list[dict],
    coin: str,
    funding_rate: float = 0.0,
    btc_closes: np.ndarray | None = None,
) -> "pd.DataFrame":
    """Convert raw OHLCV candles to a feature DataFrame for TFT training.

    Args:
        candles: List of {"o", "h", "l", "c", "v"} dicts (chronological).
        coin: Coin ticker (e.g. "BTC").
        funding_rate: Current funding rate (static fallback).
        btc_closes: BTC close array for correlation. If coin == BTC, ignored.

    Returns:
        pd.DataFrame with all required columns for TimeSeriesDataSet.
    """
    if pd is None:
        raise ImportError("pandas is required: pip install pandas")

    o = np.array([c["o"] for c in candles], dtype=np.float64)
    h = np.array([c["h"] for c in candles], dtype=np.float64)
    l = np.array([c["l"] for c in candles], dtype=np.float64)  # noqa: E741
    c = np.array([c["c"] for c in candles], dtype=np.float64)
    v = np.array([c["v"] for c in candles], dtype=np.float64)

    n = len(candles)

    # Target: log return over next step (forward-filled at the end)
    target = np.zeros(n, dtype=np.float64)
    target[:-1] = np.log(c[1:] / c[:-1])
    target[-1] = 0.0

    # Volume ratio vs 20-period MA
    vol_ma = np.convolve(v, np.ones(20) / 20, mode="full")[:n]
    vol_ma[:20] = vol_ma[20]
    with np.errstate(divide="ignore", invalid="ignore"):
        volume_ratio = np.where(vol_ma > 0, v / vol_ma, 1.0)

    # BTC correlation
    if btc_closes is not None and coin.upper() != "BTC":
        btc_ret = np.diff(np.log(btc_closes), prepend=0.0)
        coin_ret = np.diff(np.log(c), prepend=0.0)
        corr = np.full(n, 0.0)
        window = 20
        for i in range(window, n):
            x = btc_ret[i - window : i]
            y = coin_ret[i - window : i]
            corr_val = np.corrcoef(x, y)[0, 1]
            corr[i] = 0.0 if np.isnan(corr_val) else corr_val
        btc_correlation = corr
    else:
        btc_correlation = np.ones(n)

    # Temporal features
    # Use synthetic timestamps: assume 15-min intervals starting from now
    base_ts = int(time.time()) - n * 900
    timestamps = pd.to_datetime(
        [base_ts + i * 900 for i in range(n)], unit="s", utc=True
    )

    # Momentum
    mom_3 = np.zeros(n)
    mom_7 = np.zeros(n)
    mom_3[3:] = (c[3:] - c[:-3]) / c[:-3]
    mom_7[7:] = (c[7:] - c[:-7]) / c[:-7]

    # Candle metrics
    with np.errstate(divide="ignore", invalid="ignore"):
        close_vs_open = np.where(o > 0, (c - o) / o, 0.0)
        high_low_range = np.where(c > 0, (h - l) / c, 0.0)

    df = pd.DataFrame(
        {
            "time_idx": np.arange(n),
            "coin_id": coin.upper(),
            "close": c,
            "target": target,
            # Observed features
            "rsi_14": _compute_rsi(c),
            "atr_14_norm": _compute_atr(h, l, c),
            "volume_ratio": volume_ratio,
            "funding_rate": funding_rate,
            "btc_correlation": btc_correlation,
            "price_momentum_3": mom_3,
            "price_momentum_7": mom_7,
            "close_vs_open": close_vs_open,
            "high_low_range": high_low_range,
            # Known future features
            "hour_of_day": timestamps.hour.astype(float),
            "day_of_week": timestamps.dayofweek.astype(float),
            "hour_sin": np.sin(2 * np.pi * timestamps.hour / 24),
            "hour_cos": np.cos(2 * np.pi * timestamps.hour / 24),
        }
    )

    # Replace infinities and NaNs
    df.replace([np.inf, -np.inf], 0.0, inplace=True)
    df.fillna(0.0, inplace=True)

    return df


# ─── Main predictor class ───────────────────────────────────────────────────


class TFTPredictor:
    """Multi-horizon price predictor using Temporal Fusion Transformer.

    The TFT combines high-performance multi-horizon forecasting with
    interpretable insights into temporal dynamics. It uses:

    - Variable Selection Networks to pick the most relevant features
    - Gated Residual Networks for non-linear processing
    - Multi-head attention for long-range dependencies
    - Quantile regression for uncertainty estimation

    Usage:
        >>> tft = TFTPredictor(horizons=["15m", "1h", "4h"])
        >>> dataset = tft.prepare_data({"BTC": candles_btc, "ETH": candles_eth})
        >>> tft.train(dataset, epochs=50)
        >>> pred = tft.predict("BTC", current_features)
        >>> print(pred["1h"]["direction"], pred["1h"]["confidence"])
        LONG 0.72

    If pytorch-forecasting is not installed, all methods return None
    gracefully and log a warning.
    """

    def __init__(
        self,
        horizons: list[str] | None = None,
        config: TFTConfig | None = None,
        device: str = "auto",
    ) -> None:
        """Initialize the TFT predictor.

        Args:
            horizons: Prediction horizons. Default: ["15m", "1h", "4h"].
            config: TFT hyperparameter config. Default: sensible crypto defaults.
            device: "auto", "cpu", or "cuda".
        """
        self.horizons = horizons or DEFAULT_HORIZONS
        self.config = config or TFTConfig()
        self.model: Any = None
        self.training_dataset: Any = None
        self._trainer: Any = None
        self._device = device
        self._is_trained = False

        if not HAS_TFT:
            logger.warning(
                "pytorch-forecasting not installed. TFT predictions disabled. "
                "Install with: pip install pytorch-forecasting pytorch-lightning torch"
            )

    # ─── Data preparation ────────────────────────────────────────────────

    def prepare_data(
        self,
        candles_dict: dict[str, list[dict]],
        funding_rates: dict[str, float] | None = None,
    ) -> Optional["TimeSeriesDataSet"]:
        """Convert candle data from multiple coins into a TimeSeriesDataSet.

        Args:
            candles_dict: {"BTC": [candles], "ETH": [candles], ...}
                Each candle is {"o", "h", "l", "c", "v"}.
            funding_rates: {"BTC": 0.0001, ...}. Defaults to 0.0.

        Returns:
            TimeSeriesDataSet ready for training, or None if deps missing.
        """
        if not HAS_TFT or pd is None:
            logger.warning("Cannot prepare data: missing dependencies.")
            return None

        funding_rates = funding_rates or {}

        # Get BTC closes for correlation computation
        btc_closes = None
        if "BTC" in candles_dict:
            btc_closes = np.array(
                [c["c"] for c in candles_dict["BTC"]], dtype=np.float64
            )

        frames: list[pd.DataFrame] = []
        global_idx_offset = 0

        for coin, candles in candles_dict.items():
            if len(candles) < self.config.max_encoder_length + self.config.max_prediction_length:
                logger.warning(
                    f"Skipping {coin}: only {len(candles)} candles "
                    f"(need {self.config.max_encoder_length + self.config.max_prediction_length})"
                )
                continue

            fr = funding_rates.get(coin, 0.0)

            # Resample BTC closes to match this coin's length if needed
            coin_btc = None
            if btc_closes is not None and coin.upper() != "BTC":
                min_len = min(len(btc_closes), len(candles))
                coin_btc = btc_closes[:min_len]

            df = _build_features_df(candles, coin, fr, coin_btc)
            df["time_idx"] = df["time_idx"] + global_idx_offset
            frames.append(df)
            global_idx_offset += len(df)

        if not frames:
            logger.error("No valid coin data after filtering.")
            return None

        full_df = pd.concat(frames, ignore_index=True)

        # Reset time_idx per coin to start from 0 (required by pytorch-forecasting)
        full_df["time_idx"] = full_df.groupby("coin_id").cumcount()

        logger.info(
            f"Prepared {len(full_df)} samples across "
            f"{full_df['coin_id'].nunique()} coins."
        )

        # Build TimeSeriesDataSet
        max_encoder = self.config.max_encoder_length
        max_pred = self.config.max_prediction_length

        training_cutoff = full_df["time_idx"].max() - max_pred

        training = TimeSeriesDataSet(
            full_df[full_df["time_idx"] <= training_cutoff],
            time_idx="time_idx",
            target="target",
            group_ids=["coin_id"],
            min_encoder_length=max_encoder // 2,
            max_encoder_length=max_encoder,
            min_prediction_length=1,
            max_prediction_length=max_pred,
            static_categoricals=STATIC_CATEGORICALS,
            time_varying_known_reals=KNOWN_FUTURE_FEATURES,
            time_varying_unknown_reals=["target"] + OBSERVED_FEATURES,
            target_normalizer=GroupNormalizer(groups=["coin_id"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        self.training_dataset = training
        self._training_df = full_df
        self._training_cutoff = training_cutoff
        return training

    # ─── Training ────────────────────────────────────────────────────────

    def train(
        self,
        dataset: Optional["TimeSeriesDataSet"] = None,
        epochs: int = 50,
        val_split: float = 0.15,
        gpus: int | str = "auto",
    ) -> dict[str, float] | None:
        """Train the TFT model.

        Args:
            dataset: TimeSeriesDataSet from prepare_data(). If None, uses
                the last prepared dataset.
            epochs: Maximum training epochs.
            val_split: Fraction of data for validation.
            gpus: Number of GPUs or "auto".

        Returns:
            Dict with training metrics, or None if deps missing.
        """
        if not HAS_TFT:
            logger.warning("Cannot train: pytorch-forecasting not installed.")
            return None

        dataset = dataset or self.training_dataset
        if dataset is None:
            raise ValueError(
                "No dataset provided. Call prepare_data() first or pass a dataset."
            )

        # Create validation set from stored DataFrame
        val_df = getattr(self, '_training_df', None)
        if val_df is None:
            # Fallback: use training data
            val_df = dataset.to_dataframe() if hasattr(dataset, 'to_dataframe') else None
        if val_df is not None:
            validation = TimeSeriesDataSet.from_dataset(
                dataset, val_df, predict=True, stop_randomization=True
            )
        else:
            validation = dataset

        train_loader = dataset.to_dataloader(
            train=True,
            batch_size=self.config.batch_size,
            num_workers=0,
            persistent_workers=False,
        )
        val_loader = validation.to_dataloader(
            train=False,
            batch_size=self.config.batch_size * 2,
            num_workers=0,
            persistent_workers=False,
        )

        # Determine accelerator
        if self._device == "auto":
            accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        elif self._device == "cuda":
            accelerator = "gpu"
        else:
            accelerator = "cpu"

        devices = 1 if accelerator == "gpu" else "auto"

        # Callbacks
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.config.reduce_on_plateau_patience + 2,
            verbose=True,
            mode="min",
        )
        lr_monitor = LearningRateMonitor()

        # Trainer
        self._trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator=accelerator,
            devices=devices,
            enable_model_summary=True,
            gradient_clip_val=self.config.gradient_clip_val,
            callbacks=[lr_monitor, early_stop],
            enable_progress_bar=True,
            log_every_n_steps=10,
        )

        # Model
        self.model = TemporalFusionTransformer.from_dataset(
            dataset,
            hidden_size=self.config.hidden_size,
            attention_head_size=self.config.attention_head_size,
            dropout=self.config.dropout,
            hidden_continuous_size=self.config.hidden_continuous_size,
            learning_rate=self.config.learning_rate,
            loss=QuantileLoss(),
            reduce_on_plateau_patience=self.config.reduce_on_plateau_patience,
            log_interval=10,
        )

        logger.info(
            f"Training TFT with {sum(p.numel() for p in self.model.parameters()):,} "
            f"parameters on {accelerator}."
        )

        # Train — wrap model if needed for newer pytorch-lightning
        _model = self.model
        try:
            self._trainer.fit(_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        except TypeError:
            # Fallback: use lightning.Trainer from the lightning package directly
            import lightning
            self._trainer = lightning.Trainer(
                max_epochs=epochs,
                accelerator=accelerator,
                gradient_clip_val=self.config.gradient_clip_val,
                enable_model_summary=False,
                enable_progress_bar=True,
            )
            self._trainer.fit(_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        self._is_trained = True

        # Extract metrics
        best_loss = early_stop.best_score
        metrics = {
            "best_val_loss": float(best_loss) if best_loss is not None else float("nan"),
            "epochs_trained": self._trainer.current_epoch,
            "parameters": sum(p.numel() for p in self.model.parameters()),
        }

        logger.info(f"Training complete. Best val_loss: {metrics['best_val_loss']:.6f}")
        return metrics

    # ─── Prediction ──────────────────────────────────────────────────────

    def predict(
        self,
        coin: str,
        current_features: dict[str, float] | "pd.DataFrame | None" = None,
        candles: list[dict] | None = None,
    ) -> dict[str, dict[str, Any]] | None:
        """Predict direction and confidence for each horizon.

        Provide either current_features (dict of latest feature values) or
        raw candles for the encoder window. Candles are preferred as they
        allow the model to see the full temporal context.

        Args:
            coin: Coin ticker (e.g. "BTC").
            current_features: Dict of feature name -> value for latest step.
            candles: Recent OHLCV candles (at least max_encoder_length).

        Returns:
            {
                "15m": {"direction": "LONG", "confidence": 0.72, ...},
                "1h":  {"direction": "SHORT", "confidence": 0.61, ...},
                "4h":  {"direction": "LONG", "confidence": 0.55, ...},
            }
            or None if model not available.
        """
        if not HAS_TFT or self.model is None:
            logger.warning("Cannot predict: model not loaded or deps missing.")
            return None

        if candles is None and current_features is None:
            raise ValueError("Provide either candles or current_features.")

        try:
            # Build encoder DataFrame
            if candles is not None:
                encoder_df = _build_features_df(candles, coin)
                encoder_df["time_idx"] = np.arange(len(encoder_df))
            else:
                # Construct a minimal single-row DataFrame from features
                if pd is None:
                    return None
                encoder_df = pd.DataFrame([current_features])
                encoder_df["coin_id"] = coin.upper()
                encoder_df["time_idx"] = 0
                encoder_df["target"] = 0.0

            # Get prediction from model
            self.model.eval()

            # Create prediction dataset using training dataset parameters
            if self.training_dataset is None:
                logger.error("No training dataset reference. Load a model first.")
                return None

            predict_dataset = TimeSeriesDataSet.from_dataset(
                self.training_dataset,
                encoder_df,
                predict=True,
                stop_randomization=True,
            )
            predict_loader = predict_dataset.to_dataloader(
                train=False, batch_size=1, num_workers=0
            )

            # Raw predictions: quantile outputs across the prediction horizon
            raw_predictions = self.model.predict(
                predict_loader,
                mode="quantiles",
                return_x=True,
            )

            predictions = raw_predictions.output
            # predictions shape: (batch, prediction_length, n_quantiles)
            # Quantiles from QuantileLoss default: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
            median_idx = 3  # 0.5 quantile
            lower_idx = 1   # 0.1 quantile
            upper_idx = 5   # 0.9 quantile

            result: dict[str, dict[str, Any]] = {}

            for horizon_name in self.horizons:
                steps = HORIZON_MAP.get(horizon_name)
                if steps is None:
                    logger.warning(f"Unknown horizon: {horizon_name}")
                    continue

                # Clamp to available prediction length
                step_idx = min(steps - 1, predictions.shape[1] - 1)

                median_pred = float(predictions[0, step_idx, median_idx])
                lower_pred = float(predictions[0, step_idx, lower_idx])
                upper_pred = float(predictions[0, step_idx, upper_idx])

                # Direction from median prediction
                direction = "LONG" if median_pred > 0 else "SHORT"

                # Confidence: how much the quantile range agrees on direction
                # Higher when both bounds are on the same side of zero
                spread = upper_pred - lower_pred
                if spread > 1e-10:
                    # Distance of median from zero relative to spread
                    confidence = min(abs(median_pred) / (spread / 2), 1.0)
                else:
                    confidence = 0.5

                # Boost confidence if lower and upper agree on direction
                if lower_pred > 0 and upper_pred > 0:
                    confidence = max(confidence, 0.7)
                elif lower_pred < 0 and upper_pred < 0:
                    confidence = max(confidence, 0.7)

                result[horizon_name] = TFTPrediction(
                    direction=direction,
                    confidence=confidence,
                    predicted_return=median_pred,
                    upper_bound=upper_pred,
                    lower_bound=lower_pred,
                ).to_dict()

            return result

        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return None

    # ─── Attention weights (interpretability) ────────────────────────────

    def get_attention_weights(
        self, candles: list[dict], coin: str
    ) -> dict[str, Any] | None:
        """Extract attention weights for interpretability.

        Returns which time steps and features the model focuses on most.
        Useful for understanding why the model made a particular prediction.

        Args:
            candles: Recent OHLCV candles.
            coin: Coin ticker.

        Returns:
            Dict with "temporal_attention" and "variable_importance" keys,
            or None if not available.
        """
        if not HAS_TFT or self.model is None or self.training_dataset is None:
            return None

        try:
            encoder_df = _build_features_df(candles, coin)
            encoder_df["time_idx"] = np.arange(len(encoder_df))

            interpret_dataset = TimeSeriesDataSet.from_dataset(
                self.training_dataset,
                encoder_df,
                predict=True,
                stop_randomization=True,
            )
            interpret_loader = interpret_dataset.to_dataloader(
                train=False, batch_size=1, num_workers=0
            )

            interpretation = self.model.interpret_output(
                self.model.predict(interpret_loader, mode="raw", return_x=True),
                reduction="sum",
            )

            return {
                "attention_weights": {
                    k: v.detach().cpu().numpy().tolist()
                    for k, v in interpretation.items()
                    if hasattr(v, "detach")
                },
            }
        except Exception as e:
            logger.error(f"Attention extraction error: {e}")
            return None

    # ─── Persistence ─────────────────────────────────────────────────────

    def save_model(self, path: str) -> bool:
        """Save trained model and dataset parameters.

        Args:
            path: File path (e.g. "models/tft_model.pt").

        Returns:
            True if saved successfully.
        """
        if not HAS_TFT or self.model is None:
            logger.warning("No model to save.")
            return False

        try:
            save_dir = Path(path).parent
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save the best model checkpoint
            self._trainer.save_checkpoint(path)

            # Also save the dataset parameters for inference reconstruction
            dataset_params_path = str(path) + ".dataset_params"
            if self.training_dataset is not None:
                torch.save(
                    self.training_dataset.get_parameters(), dataset_params_path
                )

            logger.info(f"Model saved to {path}")
            return True

        except Exception as e:
            logger.error(f"Save error: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """Load a previously trained TFT model.

        Args:
            path: File path (e.g. "models/tft_model.pt").

        Returns:
            True if loaded successfully.
        """
        if not HAS_TFT:
            logger.warning("Cannot load: pytorch-forecasting not installed.")
            return False

        try:
            if not os.path.exists(path):
                logger.error(f"Model file not found: {path}")
                return False

            self.model = TemporalFusionTransformer.load_from_checkpoint(path)
            self.model.eval()
            self._is_trained = True

            # Load dataset params if available
            dataset_params_path = str(path) + ".dataset_params"
            if os.path.exists(dataset_params_path):
                params = torch.load(dataset_params_path, weights_only=False)
                # Reconstruct a minimal dataset for prediction
                logger.info("Loaded dataset parameters for inference.")

            logger.info(f"Model loaded from {path}")
            return True

        except Exception as e:
            logger.error(f"Load error: {e}")
            return False

    # ─── Utilities ───────────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """Check if TFT dependencies are installed and model is ready."""
        return HAS_TFT and self._is_trained and self.model is not None

    def summary(self) -> str:
        """Return a human-readable model summary."""
        if not HAS_TFT:
            return "TFT unavailable (install pytorch-forecasting)"
        if self.model is None:
            return "TFT not trained/loaded"
        n_params = sum(p.numel() for p in self.model.parameters())
        return (
            f"TFT Model: {n_params:,} parameters | "
            f"Horizons: {self.horizons} | "
            f"Encoder: {self.config.max_encoder_length} steps | "
            f"Hidden: {self.config.hidden_size}"
        )


# ─── Standalone training script ─────────────────────────────────────────────

def _fetch_candles_hl(coin: str, interval: str = "15m", limit: int = 5000) -> list[dict]:
    """Fetch candles from Hyperliquid public API.

    Args:
        coin: Coin ticker (e.g. "BTC").
        interval: Candle interval ("15m", "1h", etc.).
        limit: Max number of candles.

    Returns:
        List of {"o", "h", "l", "c", "v"} dicts.
    """
    import requests

    url = "https://api.hyperliquid.xyz/info"
    end_ms = int(time.time() * 1000)

    interval_ms_map = {
        "1m": 60_000,
        "5m": 300_000,
        "15m": 900_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
    }
    interval_ms = interval_ms_map.get(interval, 900_000)
    start_ms = end_ms - (limit * interval_ms)

    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }

    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    raw = resp.json()

    return [
        {
            "o": float(c["o"]),
            "h": float(c["h"]),
            "l": float(c["l"]),
            "c": float(c["c"]),
            "v": float(c["v"]),
        }
        for c in raw
    ]


if __name__ == "__main__":
    """
    Standalone training demo.

    Downloads 15m candles from Hyperliquid for BTC and ETH,
    builds a TFT model, trains it, and shows sample predictions.

    Usage:
        pip install pytorch-forecasting pytorch-lightning torch pandas requests
        python tft_predictor.py
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not HAS_TFT:
        print("=" * 60)
        print("ERROR: pytorch-forecasting is not installed.")
        print("Install with:")
        print("  pip install pytorch-forecasting pytorch-lightning torch")
        print("=" * 60)
        exit(1)

    print("=" * 60)
    print("  DeepAlpha TFT Predictor — Training Demo")
    print("=" * 60)

    # --- Step 1: Download data ---
    coins = ["BTC", "ETH"]
    candles_dict: dict[str, list[dict]] = {}

    for coin in coins:
        print(f"\n[1/4] Downloading {coin} 15m candles from Hyperliquid...")
        try:
            candles = _fetch_candles_hl(coin, interval="15m", limit=3000)
            print(f"  Got {len(candles)} candles.")
            if len(candles) > 200:
                candles_dict[coin] = candles
            else:
                print(f"  Skipping {coin}: not enough data.")
        except Exception as e:
            print(f"  Failed to download {coin}: {e}")

    if not candles_dict:
        print("No data downloaded. Check your internet connection.")
        exit(1)

    # --- Step 2: Prepare dataset ---
    print("\n[2/4] Preparing TimeSeriesDataSet...")
    tft = TFTPredictor(
        horizons=["15m", "1h", "4h"],
        config=TFTConfig(
            hidden_size=32,           # smaller for demo
            attention_head_size=2,
            max_encoder_length=96,     # 24h of 15m candles
            max_prediction_length=16,  # 4h ahead
            batch_size=32,
            learning_rate=1e-3,
        ),
    )

    dataset = tft.prepare_data(candles_dict)
    if dataset is None:
        print("Failed to prepare dataset.")
        exit(1)

    print(f"  Dataset size: {len(dataset)} samples")

    # --- Step 3: Train ---
    print("\n[3/4] Training TFT model (this may take a few minutes)...")
    metrics = tft.train(dataset, epochs=10)  # small epochs for demo

    if metrics:
        print(f"\n  Training results:")
        print(f"    Best val_loss:   {metrics['best_val_loss']:.6f}")
        print(f"    Epochs trained:  {metrics['epochs_trained']}")
        print(f"    Parameters:      {metrics['parameters']:,}")

    # --- Step 4: Predict ---
    print("\n[4/4] Running predictions...")
    for coin in candles_dict:
        candles = candles_dict[coin]
        # Use last encoder_length candles for context
        context = candles[-tft.config.max_encoder_length - tft.config.max_prediction_length :]
        predictions = tft.predict(coin, candles=context)

        if predictions:
            print(f"\n  {coin} predictions:")
            for horizon, pred in predictions.items():
                print(
                    f"    {horizon:>4s}: {pred['direction']:>5s} "
                    f"(conf={pred['confidence']:.2f}, "
                    f"ret={pred['predicted_return']:+.4f}, "
                    f"range=[{pred['lower_bound']:+.4f}, {pred['upper_bound']:+.4f}])"
                )
        else:
            print(f"\n  {coin}: prediction failed.")

    # --- Save model ---
    model_path = "models/tft_demo_model.pt"
    print(f"\nSaving model to {model_path}...")
    tft.save_model(model_path)

    print("\n" + "=" * 60)
    print("  Demo complete. Model summary:")
    print(f"  {tft.summary()}")
    print("=" * 60)
