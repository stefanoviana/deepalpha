"""
DeepAlpha FreqAI Model
======================

A FreqAI-compatible model that implements:
- Triple Barrier Labeling for realistic trade outcome labels
- LightGBM with SHAP-based feature selection
- Meta-labeling for trade filtering
- Purged walk-forward cross-validation

Usage:
    Set "model_type": "DeepAlphaModel" in your Freqtrade config's freqai section.

Author: DeepAlpha Team
License: MIT
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import accuracy_score, log_loss

# Freqtrade imports — only needed for DeepAlphaModel class.
# Kept as try/except so utilities (PurgedWalkForwardCV, apply_triple_barrier, select_features_by_shap)
# work without freqtrade installed.
try:
    from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
    from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
    _HAS_FREQTRADE = True
except ImportError:
    _HAS_FREQTRADE = False
    # Create stub base class so module still parses. DeepAlphaModel will raise at instantiation.
    class BaseClassifierModel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "DeepAlphaModel requires freqtrade. Install with: pip install freqtrade"
            )
    class FreqaiDataKitchen:  # type: ignore[no-redef]
        pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Purged Walk-Forward Cross-Validator
# ---------------------------------------------------------------------------

class PurgedWalkForwardCV(BaseCrossValidator):
    """
    Time-series cross-validator with purge gaps between train and test sets.

    Prevents information leakage by inserting a gap (purge window) between
    the training and testing periods, and an optional embargo period after
    the test set.

    Parameters
    ----------
    n_splits : int
        Number of CV folds.
    purge_gap : int
        Number of samples to purge between train and test sets.
    embargo_pct : float
        Fraction of test samples to embargo after each test set.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 24,
        embargo_pct: float = 0.01,
    ):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)
        embargo_size = int(test_size * self.embargo_pct)

        for i in range(self.n_splits):
            test_start = (i + 1) * test_size
            test_end = min(test_start + test_size, n_samples)

            train_end = max(0, test_start - self.purge_gap)
            train_indices = np.arange(0, train_end)

            test_indices = np.arange(test_start, test_end)

            # Apply embargo: exclude post-test samples from future training
            if embargo_size > 0 and test_end + embargo_size < n_samples:
                pass  # Embargo handled by not including post-test in train

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices


# ---------------------------------------------------------------------------
# Triple Barrier Labeling
# ---------------------------------------------------------------------------

def apply_triple_barrier(
    df: pd.DataFrame,
    close_col: str = "close",
    profit_taking: float = 1.5,
    stop_loss: float = 1.5,
    max_holding_period: int = 48,
    volatility_window: int = 20,
) -> pd.Series:
    """
    Apply the Triple Barrier labeling method.

    NOTE (SHORT-bias fix, 2026-04-16): defaults were previously asymmetric
    (profit_taking=2.0, stop_loss=1.0). With pt=2*sigma, sl=1*sigma and a
    random walk, the lower barrier is hit first roughly twice as often as
    the upper barrier, producing a strong DOWN/SHORT label skew even in
    bull markets. Defaults are now symmetric (1.5 / 1.5).

    For each bar, compute a volatility-scaled upper (profit) and lower (loss)
    barrier.  The label is determined by which barrier is touched first:
      +1  upper barrier hit  (profitable trade)
      -1  lower barrier hit  (losing trade)
       0  vertical barrier hit (time expiry, neutral)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a column with closing prices.
    close_col : str
        Name of the close-price column.
    profit_taking : float
        Upper barrier multiplier in units of rolling volatility.
    stop_loss : float
        Lower barrier multiplier in units of rolling volatility.
    max_holding_period : int
        Maximum number of bars before the vertical barrier triggers.
    volatility_window : int
        Lookback window for computing rolling volatility (std of returns).

    Returns
    -------
    pd.Series
        Labels aligned with the input DataFrame index.
    """
    close = df[close_col].values
    returns = pd.Series(close).pct_change()
    volatility = returns.rolling(volatility_window).std().values

    labels = np.zeros(len(close), dtype=int)

    for i in range(len(close)):
        if np.isnan(volatility[i]) or volatility[i] == 0:
            labels[i] = 0
            continue

        upper = close[i] * (1 + profit_taking * volatility[i])
        lower = close[i] * (1 - stop_loss * volatility[i])
        end_idx = min(i + max_holding_period, len(close))

        label = 0  # default: vertical barrier
        for j in range(i + 1, end_idx):
            if close[j] >= upper:
                label = 1
                break
            elif close[j] <= lower:
                label = -1
                break

        labels[i] = label

    return pd.Series(labels, index=df.index, name="triple_barrier_label")


# ---------------------------------------------------------------------------
# SHAP Feature Selection
# ---------------------------------------------------------------------------

def select_features_by_shap(
    model: LGBMClassifier,
    X: pd.DataFrame,
    top_k: int = 30,
) -> "list[str]":
    """
    Select the top-k most important features using SHAP values.

    Parameters
    ----------
    model : LGBMClassifier
        A fitted LightGBM model.
    X : pd.DataFrame
        Feature matrix (a sample is sufficient for speed).
    top_k : int
        Number of features to retain.

    Returns
    -------
    list[str]
        Names of the selected features.
    """
    sample = X.sample(n=min(1000, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # For multi-class, shap_values may be a list of arrays or a 3D array;
    # reduce to a 1D importance vector (one value per feature).
    if isinstance(shap_values, list):
        # List of (n_samples, n_features) arrays, one per class
        importance = np.mean(
            [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
        )
    else:
        shap_arr = np.array(shap_values)
        if shap_arr.ndim == 3:
            # Shape (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
            # Average over samples and classes
            importance = np.abs(shap_arr).mean(axis=(0, 2)) if shap_arr.shape[2] != len(X.columns) \
                else np.abs(shap_arr).mean(axis=(0, 1))
        elif shap_arr.ndim == 2:
            importance = np.abs(shap_arr).mean(axis=0)
        else:
            importance = np.abs(shap_arr)
    # Ensure 1D
    importance = np.asarray(importance).flatten()[:len(X.columns)]

    feature_importance = pd.Series(importance, index=X.columns)
    top_features = feature_importance.nlargest(top_k).index.tolist()

    logger.info(
        "SHAP feature selection: kept %d / %d features. "
        "Top 5: %s",
        len(top_features),
        len(X.columns),
        top_features[:5],
    )
    return top_features


# ---------------------------------------------------------------------------
# DeepAlpha FreqAI Model
# ---------------------------------------------------------------------------

class DeepAlphaModel(BaseClassifierModel):
    """
    FreqAI model implementing the DeepAlpha ML pipeline.

    Pipeline stages:
    1. Generate labels via Triple Barrier method
    2. Train primary LightGBM classifier
    3. Select features via SHAP importance
    4. Retrain primary model on selected features
    5. Train meta-labeling model to filter signals
    6. Validate with purged walk-forward CV

    Configuration lives under config["freqai"]["deepalpha"].
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.primary_model: Optional[LGBMClassifier] = None
        self.meta_model: Optional[LGBMClassifier] = None
        self.selected_features: Optional[List[str]] = None
        self._training_count: int = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_deepalpha_config(self) -> Dict[str, Any]:
        """Return the deepalpha sub-config with defaults."""
        da_cfg = self.freqai_info.get("deepalpha", {})
        defaults = {
            "triple_barrier": {
                # SHORT-bias fix: symmetric barriers (was 2.0 / 1.0).
                "profit_taking": 1.5,
                "stop_loss": 1.5,
                "max_holding_period": 48,
                "volatility_window": 20,
            },
            "shap_feature_selection": {
                "enabled": True,
                "top_k": 30,
                "recalculate_every_n_trainings": 5,
            },
            "meta_labeling": {
                "enabled": True,
                "threshold": 0.55,
            },
            "purged_cv": {
                "n_splits": 5,
                "purge_gap": 24,
                "embargo_pct": 0.01,
            },
        }
        for key, val in defaults.items():
            if key not in da_cfg:
                da_cfg[key] = val
            elif isinstance(val, dict):
                for k2, v2 in val.items():
                    da_cfg[key].setdefault(k2, v2)
        return da_cfg

    def _build_lgbm(self, params: Dict[str, Any]) -> LGBMClassifier:
        """Instantiate a LightGBM classifier from training parameters.

        Configured as 3-class multiclass (SHORT=0, FLAT=1, LONG=2) so the
        model can output P_short / P_flat / P_long probabilities instead
        of a collapsed binary {0,1} signal.
        """
        default_params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "n_estimators": 2000,
            "learning_rate": 0.02,
            "max_depth": 6,
            "num_leaves": 48,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "min_child_samples": 50,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
            # Keeps training balanced across the 3 classes in regimes where
            # one class dominates (e.g. FLAT in low-volatility periods).
            "class_weight": "balanced",
        }
        default_params.update(params)
        return LGBMClassifier(**default_params)

    def _generate_labels(
        self,
        df: pd.DataFrame,
        cfg: Dict[str, Any],
    ) -> pd.Series:
        """Generate Triple Barrier labels for the training data."""
        tb_cfg = cfg["triple_barrier"]
        labels = apply_triple_barrier(
            df,
            close_col="close",
            profit_taking=tb_cfg["profit_taking"],
            stop_loss=tb_cfg["stop_loss"],
            max_holding_period=tb_cfg["max_holding_period"],
            volatility_window=tb_cfg.get("volatility_window", 20),
        )
        label_counts = labels.value_counts().to_dict()
        logger.info("Triple Barrier label distribution: %s", label_counts)
        return labels

    def _run_purged_cv(
        self,
        model: LGBMClassifier,
        X: pd.DataFrame,
        y: pd.Series,
        cfg: Dict[str, Any],
    ) -> Dict[str, float]:
        """Evaluate the model with purged walk-forward cross-validation."""
        cv_cfg = cfg["purged_cv"]
        cv = PurgedWalkForwardCV(
            n_splits=cv_cfg["n_splits"],
            purge_gap=cv_cfg["purge_gap"],
            embargo_pct=cv_cfg["embargo_pct"],
        )

        accuracies = []
        losses = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            fold_model = self._build_lgbm(
                self.freqai_info.get("model_training_parameters", {})
            )
            fold_model.fit(
                X_tr, y_tr,
                eval_set=[(X_te, y_te)],
                callbacks=[],
            )

            preds = fold_model.predict(X_te)
            proba = fold_model.predict_proba(X_te)

            acc = accuracy_score(y_te, preds)
            try:
                ll = log_loss(y_te, proba, labels=fold_model.classes_)
            except ValueError:
                ll = float("nan")

            accuracies.append(acc)
            losses.append(ll)
            logger.info(
                "Purged CV fold %d: accuracy=%.4f  log_loss=%.4f",
                fold, acc, ll,
            )

        results = {
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "mean_log_loss": float(np.nanmean(losses)),
        }
        logger.info("Purged CV results: %s", results)
        return results

    # ------------------------------------------------------------------
    # FreqAI interface: fit
    # ------------------------------------------------------------------

    def fit(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        Train the DeepAlpha pipeline.

        Steps:
        1. Generate Triple Barrier labels from training data
        2. Train a primary LightGBM classifier
        3. (Optional) Select top-k features via SHAP
        4. Retrain primary model on selected features
        5. (Optional) Train meta-labeling model
        6. Run purged walk-forward CV for diagnostics

        Parameters
        ----------
        data_dictionary : dict
            Contains "train_features", "train_labels", "train_weights",
            and their test counterparts, as prepared by FreqaiDataKitchen.
        dk : FreqaiDataKitchen
            The data kitchen instance managing data transformations.

        Returns
        -------
        Any
            The fitted model (self is stored by FreqAI framework).
        """
        da_cfg = self._get_deepalpha_config()
        model_params = self.freqai_info.get("model_training_parameters", {})

        X_train = data_dictionary["train_features"]
        X_test = data_dictionary["test_features"]

        # --- 1. Triple Barrier Labels ---
        # If the data kitchen already has labels (from strategy), use them.
        # Otherwise, generate labels from close prices in the training data.
        if "train_labels" in data_dictionary and len(data_dictionary["train_labels"].columns) > 0:
            y_train = data_dictionary["train_labels"].iloc[:, 0]
            y_test = data_dictionary["test_labels"].iloc[:, 0]
            logger.info("Using labels provided by strategy / data kitchen.")
        else:
            logger.info("Generating Triple Barrier labels from training data.")
            y_train = self._generate_labels(
                data_dictionary.get("train_dataframe", X_train), da_cfg
            )
            y_test = self._generate_labels(
                data_dictionary.get("test_dataframe", X_test), da_cfg
            )

        # Align indices
        y_train = y_train.loc[X_train.index] if hasattr(y_train, "loc") else y_train
        y_test = y_test.loc[X_test.index] if hasattr(y_test, "loc") else y_test

        # --- 2. Train primary model (full feature set) ---
        logger.info(
            "Training primary LightGBM on %d samples, %d features.",
            len(X_train), X_train.shape[1],
        )
        primary = self._build_lgbm(model_params)
        primary.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[],
        )

        # --- 3. SHAP feature selection ---
        shap_cfg = da_cfg["shap_feature_selection"]
        if shap_cfg["enabled"]:
            recalc_interval = shap_cfg.get("recalculate_every_n_trainings", 5)
            if (
                self.selected_features is None
                or self._training_count % recalc_interval == 0
            ):
                self.selected_features = select_features_by_shap(
                    primary, X_train, top_k=shap_cfg["top_k"]
                )

            X_train_sel = X_train[self.selected_features]
            X_test_sel = X_test[self.selected_features]

            # --- 4. Retrain on selected features ---
            logger.info(
                "Retraining primary model on %d SHAP-selected features.",
                len(self.selected_features),
            )
            primary = self._build_lgbm(model_params)
            primary.fit(
                X_train_sel, y_train,
                eval_set=[(X_test_sel, y_test)],
                callbacks=[],
            )
        else:
            X_train_sel = X_train
            X_test_sel = X_test

        self.primary_model = primary
        self.model = primary  # FreqAI stores the fitted object as self.model

        # --- 5. Meta-labeling ---
        meta_cfg = da_cfg["meta_labeling"]
        if meta_cfg["enabled"]:
            primary_preds_train = primary.predict(X_train_sel)
            meta_labels = (primary_preds_train == y_train.values).astype(int)

            # Meta features: original features + primary model prediction
            meta_X_train = X_train_sel.copy()
            meta_X_train["primary_prediction"] = primary_preds_train

            meta_X_test = X_test_sel.copy()
            meta_X_test["primary_prediction"] = primary.predict(X_test_sel)
            meta_y_test = (
                primary.predict(X_test_sel) == y_test.values
            ).astype(int)

            logger.info("Training meta-labeling model on %d samples.", len(meta_X_train))
            meta_model = LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=4,
                num_leaves=16,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            meta_model.fit(
                meta_X_train, meta_labels,
                eval_set=[(meta_X_test, meta_y_test)],
                callbacks=[],
            )
            self.meta_model = meta_model

            meta_acc = accuracy_score(meta_y_test, meta_model.predict(meta_X_test))
            logger.info("Meta-labeling model accuracy: %.4f", meta_acc)

        # --- 6. Purged walk-forward CV (diagnostics) ---
        cv_results = self._run_purged_cv(primary, X_train_sel, y_train, da_cfg)
        self.cv_results = cv_results

        self._training_count += 1
        logger.info(
            "DeepAlpha training #%d complete. CV accuracy: %.4f +/- %.4f",
            self._training_count,
            cv_results["mean_accuracy"],
            cv_results["std_accuracy"],
        )

        return self.primary_model

    # ------------------------------------------------------------------
    # FreqAI interface: predict
    # ------------------------------------------------------------------

    def predict(
        self, unfiltered_df: pd.DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate predictions using the trained DeepAlpha pipeline.

        This method mirrors the schema produced by BaseClassifierModel.predict():
          * pred_df columns = dk.label_list + list(primary_model.classes_)
          * do_predict = 1D numpy int array (1 = trust, 0 = skip)

        Returning any other shape (e.g. a DataFrame for do_predict, or extra
        columns on pred_df) breaks FreqaiDataKitchen.get_predictions_to_append(),
        which builds a dict-of-columns and calls DataFrame(append_dict).
        """
        da_cfg = self._get_deepalpha_config()

        # Prepare features through the data kitchen (same pipeline as fit)
        dk.find_features(unfiltered_df)
        filtered_df, _ = dk.filter_features(
            unfiltered_df,
            dk.training_features_list,
            training_filter=False,
        )

        # Apply SHAP feature selection so we feed the primary model the same
        # columns it was trained on.
        if self.selected_features is not None:
            available = [f for f in self.selected_features if f in filtered_df.columns]
            filtered_df = filtered_df[available]

        # Persist the prediction feature frame so the framework / meta-labeling
        # can reference it (matches BaseClassifierModel behaviour).
        dk.data_dictionary["prediction_features"] = filtered_df

        # FreqAI persists only self.model across runs (via dd.load_data).
        # Fall back to self.model if self.primary_model wasn't restored.
        primary_model = self.primary_model or self.model
        if primary_model is None:
            raise RuntimeError("DeepAlphaModel.predict called before fit/load of primary model")

        # Primary predictions and probabilities
        predictions = primary_model.predict(filtered_df)
        probabilities = primary_model.predict_proba(filtered_df)

        # Build pred_df with the EXACT schema BaseClassifierModel.predict returns:
        #   first column(s) = dk.label_list, then one column per class.
        label_list = list(dk.label_list) if dk.label_list else ["&label"]
        pred_df = pd.DataFrame(
            np.reshape(predictions, (-1, len(label_list))),
            columns=label_list,
        )
        # Convert class labels to strings so downstream freqtrade code
        # (e.g. remove_features_from_df -> col.startswith) doesn't choke
        # on integer column names produced by int-valued classifiers.
        pred_df_prob = pd.DataFrame(
            probabilities,
            columns=[str(c) for c in primary_model.classes_],
        )
        pred_df = pd.concat([pred_df, pred_df_prob], axis=1)

        # do_predict must be a 1D numpy array of ints (1 = use, 0 = skip).
        do_predict = np.ones(len(pred_df), dtype=np.int_)

        # DI values – we don't compute DI here, so provide zeros (same shape
        # convention BaseClassifierModel uses when DI is disabled).
        dk.DI_values = np.zeros(len(pred_df))
        dk.do_predict = do_predict

        # Meta-labeling filter: ONLY modify do_predict. Do NOT add extra
        # columns to pred_df — that would desynchronise append_dict lengths
        # in get_predictions_to_append and raise
        # "ValueError: array length 0 does not match index length ...".
        meta_cfg = da_cfg["meta_labeling"]
        if meta_cfg["enabled"] and self.meta_model is not None:
            try:
                meta_features = filtered_df.copy()
                meta_features["primary_prediction"] = predictions
                meta_proba = self.meta_model.predict_proba(meta_features)
                confidence = (
                    meta_proba[:, 1] if meta_proba.shape[1] > 1 else meta_proba[:, 0]
                )

                threshold = meta_cfg.get("threshold", 0.55)
                low_conf = confidence < threshold
                do_predict = np.where(low_conf, 0, do_predict).astype(np.int_)
                dk.do_predict = do_predict

                logger.info(
                    "Meta-labeling filtered %d / %d predictions (threshold=%.2f).",
                    int(low_conf.sum()), len(pred_df), threshold,
                )
            except Exception as e:
                logger.warning(
                    "Meta-labeling post-processing failed, using base predictions: %s", e
                )

        return pred_df, do_predict
