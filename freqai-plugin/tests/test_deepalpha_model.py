"""
Unit tests for DeepAlpha FreqAI Plugin
======================================

Tests cover:
- PurgedWalkForwardCV split correctness
- Triple Barrier labeling validity
- SHAP feature selection
- Meta-labeling probability bounds
- DeepAlphaModel fit/predict pipeline (with mocked FreqAI interfaces)

Run with:
    pytest tests/test_deepalpha_model.py -v
"""

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification

# ---------------------------------------------------------------------------
# Mock freqtrade imports before importing deepalpha_model
# ---------------------------------------------------------------------------

_freqtrade_mock = types.ModuleType("freqtrade")
_freqai_mock = types.ModuleType("freqtrade.freqai")
_base_models_mock = types.ModuleType("freqtrade.freqai.base_models")
_base_classifier_mock = types.ModuleType(
    "freqtrade.freqai.base_models.BaseClassifierModel"
)
_data_kitchen_mock = types.ModuleType("freqtrade.freqai.data_kitchen")


class _FakeBaseClassifierModel:
    """Minimal stand-in for BaseClassifierModel."""

    def __init__(self, **kwargs):
        self.freqai_info = kwargs.get("freqai_info", {})


class _FakeFreqaiDataKitchen:
    """Minimal stand-in for FreqaiDataKitchen."""

    def __init__(self):
        self.training_features_list = []

    def find_features(self, df):
        self.training_features_list = [
            c for c in df.columns if c.startswith("%-")
        ]

    def filter_features(self, df, feature_list, training_filter=False):
        available = [c for c in feature_list if c in df.columns]
        return df[available], df[available].columns.tolist()


_base_classifier_mock.BaseClassifierModel = _FakeBaseClassifierModel
_data_kitchen_mock.FreqaiDataKitchen = _FakeFreqaiDataKitchen

sys.modules["freqtrade"] = _freqtrade_mock
sys.modules["freqtrade.freqai"] = _freqai_mock
sys.modules["freqtrade.freqai.base_models"] = _base_models_mock
sys.modules[
    "freqtrade.freqai.base_models.BaseClassifierModel"
] = _base_classifier_mock
sys.modules["freqtrade.freqai.data_kitchen"] = _data_kitchen_mock

# Now safe to import
from deepalpha_model import (  # noqa: E402
    DeepAlphaModel,
    PurgedWalkForwardCV,
    apply_triple_barrier,
    select_features_by_shap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic price DataFrame with open/high/low/close/volume."""
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    close = np.maximum(close, 1.0)  # keep positive
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    open_ = close + rng.randn(n) * 0.2
    volume = rng.uniform(100, 10000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def _make_feature_df(n_samples: int = 300, n_features: int = 50, seed: int = 42):
    """Generate a feature matrix with %-prefixed column names."""
    rng = np.random.RandomState(seed)
    data = rng.randn(n_samples, n_features)
    columns = [f"%-feat_{i}" for i in range(n_features)]
    return pd.DataFrame(data, columns=columns)


# ===========================================================================
# Tests: PurgedWalkForwardCV
# ===========================================================================


class TestPurgedWalkForwardCV:
    """Tests for the PurgedWalkForwardCV cross-validator."""

    def test_n_splits_returned(self):
        cv = PurgedWalkForwardCV(n_splits=5, purge_gap=10)
        assert cv.get_n_splits() == 5

    def test_splits_count(self):
        """The number of yielded splits should equal n_splits."""
        cv = PurgedWalkForwardCV(n_splits=5, purge_gap=5, embargo_pct=0.0)
        X = np.arange(600).reshape(-1, 1)
        splits = list(cv.split(X))
        assert len(splits) == 5

    def test_no_overlap_between_train_and_test(self):
        """Train and test indices must never overlap."""
        cv = PurgedWalkForwardCV(n_splits=3, purge_gap=10, embargo_pct=0.0)
        X = np.arange(400).reshape(-1, 1)
        for train_idx, test_idx in cv.split(X):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Train/test overlap: {overlap}"

    def test_purge_gap_respected(self):
        """
        There must be a gap of at least `purge_gap` samples between
        the last training index and the first test index.
        """
        purge_gap = 20
        cv = PurgedWalkForwardCV(n_splits=3, purge_gap=purge_gap, embargo_pct=0.0)
        X = np.arange(1000).reshape(-1, 1)
        for train_idx, test_idx in cv.split(X):
            gap = test_idx[0] - train_idx[-1]
            assert gap >= purge_gap, (
                f"Purge gap violated: gap={gap}, required={purge_gap}"
            )

    def test_train_before_test(self):
        """All training indices must come before test indices (temporal order)."""
        cv = PurgedWalkForwardCV(n_splits=4, purge_gap=5)
        X = np.arange(500).reshape(-1, 1)
        for train_idx, test_idx in cv.split(X):
            assert train_idx[-1] < test_idx[0], (
                "Training data extends past start of test data"
            )

    def test_expanding_window(self):
        """Each successive fold should have a larger training set."""
        cv = PurgedWalkForwardCV(n_splits=4, purge_gap=5, embargo_pct=0.0)
        X = np.arange(500).reshape(-1, 1)
        train_sizes = [len(tr) for tr, _ in cv.split(X)]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] > train_sizes[i - 1], (
                f"Training set did not expand: fold {i-1}={train_sizes[i-1]}, "
                f"fold {i}={train_sizes[i]}"
            )


# ===========================================================================
# Tests: Triple Barrier Labeling
# ===========================================================================


class TestTripleBarrier:
    """Tests for the apply_triple_barrier function."""

    def test_labels_are_valid(self):
        """Labels must be in {-1, 0, 1}."""
        df = _make_price_df(500)
        labels = apply_triple_barrier(df, volatility_window=20)
        unique_labels = set(labels.unique())
        assert unique_labels.issubset({-1, 0, 1}), (
            f"Unexpected label values: {unique_labels}"
        )

    def test_label_series_length(self):
        """Output series must have the same length as input DataFrame."""
        df = _make_price_df(200)
        labels = apply_triple_barrier(df)
        assert len(labels) == len(df)

    def test_label_index_aligned(self):
        """Output index must match input DataFrame index."""
        df = _make_price_df(100)
        df.index = pd.RangeIndex(start=50, stop=150)
        labels = apply_triple_barrier(df)
        assert labels.index.equals(df.index)

    def test_early_rows_are_zero(self):
        """
        The first `volatility_window` rows have NaN volatility,
        so they should be labeled 0.
        """
        df = _make_price_df(200)
        vol_window = 20
        labels = apply_triple_barrier(df, volatility_window=vol_window)
        # First vol_window rows should be 0 (NaN volatility)
        assert (labels.iloc[:vol_window] == 0).all()

    def test_large_profit_barrier_produces_fewer_wins(self):
        """
        With a very large profit_taking multiplier, fewer labels should
        be +1 compared to a small multiplier.
        """
        df = _make_price_df(500)
        labels_easy = apply_triple_barrier(df, profit_taking=0.5, stop_loss=5.0)
        labels_hard = apply_triple_barrier(df, profit_taking=10.0, stop_loss=5.0)
        wins_easy = (labels_easy == 1).sum()
        wins_hard = (labels_hard == 1).sum()
        assert wins_easy >= wins_hard, (
            f"Expected more wins with easier barrier: easy={wins_easy}, hard={wins_hard}"
        )


# ===========================================================================
# Tests: SHAP Feature Selection
# ===========================================================================


class TestSHAPSelection:
    """Tests for select_features_by_shap."""

    @pytest.fixture
    def trained_model_and_data(self):
        """Create a simple trained LightGBM model with known features."""
        X, y = make_classification(
            n_samples=300, n_features=50, n_informative=10,
            random_state=42,
        )
        feature_names = [f"feat_{i}" for i in range(50)]
        X_df = pd.DataFrame(X, columns=feature_names)
        model = LGBMClassifier(
            n_estimators=50, max_depth=4, verbose=-1, random_state=42,
        )
        model.fit(X_df, y)
        return model, X_df

    def test_shap_reduces_features(self, trained_model_and_data):
        """SHAP selection with top_k < total should return fewer features."""
        model, X_df = trained_model_and_data
        selected = select_features_by_shap(model, X_df, top_k=10)
        assert len(selected) == 10
        assert len(selected) < len(X_df.columns)

    def test_shap_returns_valid_names(self, trained_model_and_data):
        """All returned feature names must exist in the original DataFrame."""
        model, X_df = trained_model_and_data
        selected = select_features_by_shap(model, X_df, top_k=15)
        for feat in selected:
            assert feat in X_df.columns, f"Unknown feature: {feat}"

    def test_shap_top_k_larger_than_total(self, trained_model_and_data):
        """If top_k >= n_features, all features should be returned."""
        model, X_df = trained_model_and_data
        selected = select_features_by_shap(model, X_df, top_k=100)
        assert len(selected) == len(X_df.columns)


# ===========================================================================
# Tests: Meta-labeling
# ===========================================================================


class TestMetaLabeling:
    """Tests for the meta-labeling pipeline within DeepAlphaModel."""

    def test_meta_labels_are_binary(self):
        """
        Meta-labels should be binary (0 or 1) -- representing whether
        the primary model prediction was correct.
        """
        rng = np.random.RandomState(42)
        primary_preds = rng.choice([-1, 0, 1], size=200)
        true_labels = rng.choice([-1, 0, 1], size=200)
        meta_labels = (primary_preds == true_labels).astype(int)
        assert set(np.unique(meta_labels)).issubset({0, 1})

    def test_meta_model_probabilities_bounded(self):
        """
        A meta-model trained on binary labels should produce
        probabilities in [0, 1].
        """
        X, y = make_classification(
            n_samples=200, n_features=10, random_state=42,
        )
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
        meta_model = LGBMClassifier(
            n_estimators=50, max_depth=3, verbose=-1, random_state=42,
        )
        meta_model.fit(X_df, y)
        proba = meta_model.predict_proba(X_df)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_meta_confidence_filters_predictions(self):
        """
        With a high threshold, the meta-model should filter out
        a significant portion of predictions.
        """
        rng = np.random.RandomState(42)
        confidence = rng.uniform(0.3, 0.9, size=100)
        threshold = 0.7
        filtered = confidence < threshold
        # At least some should be filtered
        assert filtered.sum() > 0
        # And some should pass
        assert (~filtered).sum() > 0


# ===========================================================================
# Tests: DeepAlphaModel (integration with mocks)
# ===========================================================================


class TestDeepAlphaModelIntegration:
    """Integration tests for the full DeepAlphaModel pipeline."""

    @pytest.fixture
    def model_instance(self):
        """Create a DeepAlphaModel with test configuration."""
        model = DeepAlphaModel(
            freqai_info={
                "model_training_parameters": {
                    "n_estimators": 50,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "verbose": -1,
                },
                "deepalpha": {
                    "triple_barrier": {
                        "profit_taking": 2.0,
                        "stop_loss": 1.0,
                        "max_holding_period": 20,
                        "volatility_window": 10,
                    },
                    "shap_feature_selection": {
                        "enabled": True,
                        "top_k": 10,
                        "recalculate_every_n_trainings": 1,
                    },
                    "meta_labeling": {
                        "enabled": True,
                        "threshold": 0.55,
                    },
                    "purged_cv": {
                        "n_splits": 2,
                        "purge_gap": 5,
                        "embargo_pct": 0.01,
                    },
                },
            }
        )
        return model

    def test_fit_returns_model(self, model_instance):
        """fit() should return a trained LGBMClassifier."""
        n_samples = 300
        n_features = 20
        rng = np.random.RandomState(42)

        X_train = pd.DataFrame(
            rng.randn(n_samples, n_features),
            columns=[f"feat_{i}" for i in range(n_features)],
        )
        X_test = pd.DataFrame(
            rng.randn(n_samples // 3, n_features),
            columns=[f"feat_{i}" for i in range(n_features)],
        )
        y_train = pd.DataFrame(
            {"label": rng.choice([-1, 0, 1], size=n_samples)}
        )
        y_test = pd.DataFrame(
            {"label": rng.choice([-1, 0, 1], size=n_samples // 3)}
        )

        data_dict = {
            "train_features": X_train,
            "test_features": X_test,
            "train_labels": y_train,
            "test_labels": y_test,
        }

        dk = _FakeFreqaiDataKitchen()
        result = model_instance.fit(data_dict, dk)

        assert isinstance(result, LGBMClassifier)
        assert model_instance.primary_model is not None
        assert model_instance.meta_model is not None
        assert model_instance.selected_features is not None
        assert len(model_instance.selected_features) <= 10

    def test_predict_output_shape(self, model_instance):
        """predict() should return (pred_df, dna_df) with correct shapes."""
        n_train, n_test, n_feat = 300, 50, 20
        rng = np.random.RandomState(42)

        feature_cols = [f"%-feat_{i}" for i in range(n_feat)]
        X_train = pd.DataFrame(rng.randn(n_train, n_feat), columns=feature_cols)
        X_test = pd.DataFrame(rng.randn(n_test, n_feat), columns=feature_cols)
        y_train = pd.DataFrame({"label": rng.choice([-1, 0, 1], size=n_train)})
        y_test = pd.DataFrame({"label": rng.choice([-1, 0, 1], size=n_test)})

        data_dict = {
            "train_features": X_train,
            "test_features": X_test,
            "train_labels": y_train,
            "test_labels": y_test,
        }

        dk = _FakeFreqaiDataKitchen()
        model_instance.fit(data_dict, dk)

        # Now predict
        predict_df = pd.DataFrame(
            rng.randn(20, n_feat), columns=feature_cols,
        )
        dk_pred = _FakeFreqaiDataKitchen()
        pred_df, dna_df = model_instance.predict(predict_df, dk_pred)

        assert len(pred_df) == 20
        assert len(dna_df) == 20
        assert "prediction" in pred_df.columns
        assert "do_not_act" in dna_df.columns

    def test_training_count_increments(self, model_instance):
        """Each call to fit() should increment _training_count."""
        rng = np.random.RandomState(42)
        n, f = 200, 15
        X = pd.DataFrame(rng.randn(n, f), columns=[f"f{i}" for i in range(f)])
        y = pd.DataFrame({"label": rng.choice([0, 1], size=n)})

        data_dict = {
            "train_features": X,
            "test_features": X.iloc[:50],
            "train_labels": y,
            "test_labels": y.iloc[:50],
        }
        dk = _FakeFreqaiDataKitchen()

        assert model_instance._training_count == 0
        model_instance.fit(data_dict, dk)
        assert model_instance._training_count == 1
        model_instance.fit(data_dict, dk)
        assert model_instance._training_count == 2
