"""
DeepAlpha FreqAI Plugin
=======================

A FreqAI-compatible model for Freqtrade implementing:
- Triple Barrier Labeling
- SHAP-based feature selection
- Meta-labeling for trade filtering
- Purged walk-forward cross-validation

Usage:
    from deepalpha_model import DeepAlphaModel
"""

__version__ = "1.0.0"
__author__ = "DeepAlpha Team"

from deepalpha_model import (  # noqa: F401
    DeepAlphaModel,
    PurgedWalkForwardCV,
    apply_triple_barrier,
    select_features_by_shap,
)
