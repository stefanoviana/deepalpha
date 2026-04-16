"""
DeepAlpha FreqAI Plugin
=======================

A FreqAI-compatible model for Freqtrade implementing:
- Triple Barrier Labeling
- SHAP-based feature selection
- Meta-labeling for trade filtering
- Purged walk-forward cross-validation

Usage:
    from deepalpha_freqai import DeepAlphaModel

Note: DeepAlphaModel requires freqtrade to be installed.
Utilities (PurgedWalkForwardCV, apply_triple_barrier, select_features_by_shap)
are available without freqtrade.
"""

__version__ = "1.0.8"
__author__ = "DeepAlpha Team"


def __getattr__(name):
    """Lazy import to avoid crashing when freqtrade isn't installed."""
    if name == "DeepAlphaModel":
        from .deepalpha_model import DeepAlphaModel
        return DeepAlphaModel
    if name in ("PurgedWalkForwardCV", "apply_triple_barrier", "select_features_by_shap"):
        from .deepalpha_model import (
            PurgedWalkForwardCV,
            apply_triple_barrier,
            select_features_by_shap,
        )
        return {
            "PurgedWalkForwardCV": PurgedWalkForwardCV,
            "apply_triple_barrier": apply_triple_barrier,
            "select_features_by_shap": select_features_by_shap,
        }[name]
    raise AttributeError(f"module 'deepalpha_freqai' has no attribute {name!r}")


__all__ = [
    "DeepAlphaModel",
    "PurgedWalkForwardCV",
    "apply_triple_barrier",
    "select_features_by_shap",
]
