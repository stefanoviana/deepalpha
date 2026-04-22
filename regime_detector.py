#!/usr/bin/env python3
"""
regime_detector.py - Hidden Markov Model for Market Regime Detection
====================================================================
Identifies 3 market regimes:
  0 = CALM   (low vol, trending)
  1 = NORMAL (moderate vol, mixed)
  2 = VOLATILE (high vol, choppy)

Uses returns, volatility, and volume as observable features.
Can be used as:
  - A standalone feature (regime_score) fed into LightGBM
  - A strategy selector (different thresholds per regime)

Dependencies: numpy, scipy (for Gaussian PDF). No heavy ML libs required.
"""
import numpy as np
import pickle
import os
import logging

log = logging.getLogger("DeepAlpha")

N_STATES = 3
STATE_NAMES = {0: 'CALM', 1: 'NORMAL', 2: 'VOLATILE'}


class GaussianHMM:
    """Minimal Gaussian HMM with 3 states, trained via Baum-Welch (EM)."""

    def __init__(self, n_states=N_STATES, n_features=3, n_iter=50, tol=1e-4):
        self.n_states = n_states
        self.n_features = n_features
        self.n_iter = n_iter
        self.tol = tol

        # Initialize parameters
        self.pi = np.ones(n_states) / n_states  # initial state probs
        self.A = None   # transition matrix (n_states x n_states)
        self.means = None   # emission means (n_states x n_features)
        self.covars = None  # emission covariances (n_states x n_features x n_features)
        self.trained = False

    def _init_params(self, X):
        """Initialize parameters using K-means-like heuristic."""
        n = len(X)
        # Sort by volatility (column 1) to get natural regime ordering
        vol_idx = np.argsort(X[:, 1])
        third = n // 3

        # Assign initial clusters by volatility quantile
        clusters = [vol_idx[:third], vol_idx[third:2*third], vol_idx[2*third:]]

        self.means = np.zeros((self.n_states, self.n_features))
        self.covars = np.zeros((self.n_states, self.n_features, self.n_features))

        for s in range(self.n_states):
            if len(clusters[s]) > 0:
                self.means[s] = X[clusters[s]].mean(axis=0)
                cov = np.cov(X[clusters[s]].T)
                if cov.ndim == 0:
                    cov = np.array([[float(cov)]])
                # Ensure positive definite
                self.covars[s] = cov + np.eye(self.n_features) * 1e-6
            else:
                self.means[s] = X.mean(axis=0)
                self.covars[s] = np.eye(self.n_features)

        # Transition matrix: high self-transition probability (regimes are sticky)
        self.A = np.array([
            [0.90, 0.07, 0.03],  # CALM stays calm
            [0.05, 0.85, 0.10],  # NORMAL
            [0.03, 0.12, 0.85],  # VOLATILE stays volatile
        ])

    def _log_gaussian_pdf(self, x, mean, covar):
        """Log probability of x under multivariate Gaussian."""
        k = len(mean)
        diff = x - mean
        try:
            # Use Cholesky for numerical stability
            L = np.linalg.cholesky(covar)
            solve = np.linalg.solve(L, diff)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            log_prob = -0.5 * (k * np.log(2 * np.pi) + log_det + np.dot(solve, solve))
        except np.linalg.LinAlgError:
            # Fallback: diagonal approximation
            diag = np.maximum(np.diag(covar), 1e-10)
            log_prob = -0.5 * (k * np.log(2 * np.pi) + np.sum(np.log(diag)) +
                               np.sum(diff ** 2 / diag))
        return log_prob

    def _compute_log_emission(self, X):
        """Compute log emission probabilities for all observations."""
        n = len(X)
        log_B = np.zeros((n, self.n_states))
        for s in range(self.n_states):
            for t in range(n):
                log_B[t, s] = self._log_gaussian_pdf(X[t], self.means[s], self.covars[s])
        return log_B

    def _forward(self, log_B):
        """Forward algorithm (log-space)."""
        n = len(log_B)
        log_alpha = np.full((n, self.n_states), -np.inf)

        # Init
        for s in range(self.n_states):
            log_alpha[0, s] = np.log(self.pi[s] + 1e-300) + log_B[0, s]

        # Recurse
        log_A = np.log(self.A + 1e-300)
        for t in range(1, n):
            for s in range(self.n_states):
                log_alpha[t, s] = _logsumexp(log_alpha[t-1] + log_A[:, s]) + log_B[t, s]

        return log_alpha

    def _backward(self, log_B):
        """Backward algorithm (log-space)."""
        n = len(log_B)
        log_beta = np.full((n, self.n_states), -np.inf)
        log_beta[-1, :] = 0.0  # log(1)

        log_A = np.log(self.A + 1e-300)
        for t in range(n - 2, -1, -1):
            for s in range(self.n_states):
                log_beta[t, s] = _logsumexp(log_A[s, :] + log_B[t+1, :] + log_beta[t+1, :])

        return log_beta

    def fit(self, X):
        """Train HMM using Baum-Welch (EM) algorithm."""
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        if n < 100:
            log.warning("[HMM] Too few samples for training")
            return self

        self._init_params(X)
        prev_ll = -np.inf

        for iteration in range(self.n_iter):
            # E-step
            log_B = self._compute_log_emission(X)
            log_alpha = self._forward(log_B)
            log_beta = self._backward(log_B)

            # Log-likelihood
            ll = _logsumexp(log_alpha[-1])
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

            # Posterior: gamma[t,s] = P(state=s at time t | observations)
            log_gamma = log_alpha + log_beta
            log_gamma -= _logsumexp_axis(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            # Xi: transition posteriors
            log_A = np.log(self.A + 1e-300)
            xi_sum = np.zeros((self.n_states, self.n_states))
            for t in range(n - 1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi_sum[i, j] += np.exp(
                            log_alpha[t, i] + log_A[i, j] + log_B[t+1, j] + log_beta[t+1, j] - ll
                        )

            # M-step
            # Update pi
            self.pi = gamma[0] / gamma[0].sum()

            # Update A
            for i in range(self.n_states):
                denom = gamma[:-1, i].sum()
                if denom > 1e-10:
                    self.A[i, :] = xi_sum[i, :] / denom
                # Ensure row sums to 1
                row_sum = self.A[i, :].sum()
                if row_sum > 0:
                    self.A[i, :] /= row_sum

            # Update means and covariances
            for s in range(self.n_states):
                gamma_s = gamma[:, s]
                total_gamma = gamma_s.sum()
                if total_gamma > 1e-10:
                    self.means[s] = (gamma_s[:, np.newaxis] * X).sum(axis=0) / total_gamma
                    diff = X - self.means[s]
                    self.covars[s] = (gamma_s[:, np.newaxis, np.newaxis] *
                                      (diff[:, :, np.newaxis] * diff[:, np.newaxis, :])).sum(axis=0) / total_gamma
                    # Regularize
                    self.covars[s] += np.eye(self.n_features) * 1e-4

        self.trained = True
        return self

    def predict(self, X):
        """Viterbi decoding: find most likely state sequence."""
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        if not self.trained or n == 0:
            return np.ones(n, dtype=int)  # default NORMAL

        log_B = self._compute_log_emission(X)
        log_A = np.log(self.A + 1e-300)

        # Viterbi
        log_delta = np.zeros((n, self.n_states))
        psi = np.zeros((n, self.n_states), dtype=int)

        log_delta[0] = np.log(self.pi + 1e-300) + log_B[0]

        for t in range(1, n):
            for s in range(self.n_states):
                trans = log_delta[t-1] + log_A[:, s]
                psi[t, s] = np.argmax(trans)
                log_delta[t, s] = trans[psi[t, s]] + log_B[t, s]

        # Backtrack
        states = np.zeros(n, dtype=int)
        states[-1] = np.argmax(log_delta[-1])
        for t in range(n - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def predict_proba(self, X):
        """Return state probabilities for each observation."""
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        if not self.trained or n == 0:
            probs = np.zeros((n, self.n_states))
            probs[:, 1] = 1.0  # default NORMAL
            return probs

        log_B = self._compute_log_emission(X)
        log_alpha = self._forward(log_B)
        log_beta = self._backward(log_B)

        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp_axis(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)

    def save(self, path):
        """Save trained HMM to pickle."""
        data = {
            'n_states': self.n_states,
            'n_features': self.n_features,
            'pi': self.pi,
            'A': self.A,
            'means': self.means,
            'covars': self.covars,
            'trained': self.trained,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        """Load trained HMM from pickle."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        hmm = cls(n_states=data['n_states'], n_features=data['n_features'])
        hmm.pi = data['pi']
        hmm.A = data['A']
        hmm.means = data['means']
        hmm.covars = data['covars']
        hmm.trained = data['trained']
        return hmm


def _logsumexp(x):
    """Numerically stable log-sum-exp."""
    x = np.asarray(x)
    mx = x.max()
    if mx == -np.inf:
        return -np.inf
    return mx + np.log(np.sum(np.exp(x - mx)))


def _logsumexp_axis(x, axis=1, keepdims=False):
    """Log-sum-exp along an axis."""
    mx = x.max(axis=axis, keepdims=True)
    result = mx + np.log(np.sum(np.exp(x - mx), axis=axis, keepdims=True))
    if not keepdims:
        result = result.squeeze(axis=axis)
    return result


# =========================================================================
# Helper: Build HMM observation features from OHLCV
# =========================================================================
def build_hmm_observations(closes, volumes, window=24):
    """
    Build observation matrix for HMM from OHLCV data.

    Returns (n, 3) array:
      col 0: returns (1-period log returns)
      col 1: realized volatility (rolling std of returns)
      col 2: volume change (log volume ratio to rolling mean)
    """
    n = len(closes)
    obs = np.zeros((n, 3))

    # Log returns
    for i in range(1, n):
        if closes[i-1] > 0 and closes[i] > 0:
            obs[i, 0] = np.log(closes[i] / closes[i-1])

    # Rolling volatility
    for i in range(window, n):
        obs[i, 1] = obs[i-window:i, 0].std()

    # Volume ratio
    for i in range(window, n):
        vol_mean = volumes[i-window:i].mean()
        if vol_mean > 0 and volumes[i] > 0:
            obs[i, 2] = np.log(volumes[i] / vol_mean)

    return obs


def get_regime_features(states, proba):
    """
    Convert HMM output to features for LightGBM.

    Args:
        states: array of regime indices (0, 1, 2)
        proba: array of shape (n, 3) with regime probabilities

    Returns:
        dict with:
          regime_state: current regime (0=CALM, 1=NORMAL, 2=VOLATILE)
          regime_calm_prob: probability of CALM
          regime_volatile_prob: probability of VOLATILE
          regime_transition: 1 if regime changed from previous candle
    """
    n = len(states)
    regime_state = np.array(states, dtype=np.float64)
    calm_prob = proba[:, 0] if proba.shape[1] > 0 else np.zeros(n)
    volatile_prob = proba[:, 2] if proba.shape[1] > 2 else np.zeros(n)

    transition = np.zeros(n)
    for i in range(1, n):
        if states[i] != states[i-1]:
            transition[i] = 1.0

    return {
        'regime_state': regime_state,
        'regime_calm_prob': calm_prob,
        'regime_volatile_prob': volatile_prob,
        'regime_transition': transition,
    }


# =========================================================================
# Convenience: train + predict in one call (for training pipeline)
# =========================================================================
def train_and_predict(closes, volumes, save_path=None):
    """
    Train HMM on full history and return regime features.

    Args:
        closes: 1D array of close prices
        volumes: 1D array of volumes
        save_path: optional path to save trained HMM

    Returns:
        dict of regime features (same length as closes)
    """
    obs = build_hmm_observations(closes, volumes)

    # Train on observations after warmup period
    warmup = 48
    hmm = GaussianHMM(n_states=3, n_features=3, n_iter=50)
    hmm.fit(obs[warmup:])

    # Predict on full sequence
    states = hmm.predict(obs)
    proba = hmm.predict_proba(obs)

    if save_path:
        hmm.save(save_path)

    return get_regime_features(states, proba)


if __name__ == '__main__':
    print("[HMM] Regime Detector smoke test")

    # Generate synthetic data
    np.random.seed(42)
    n = 1000
    closes = np.cumsum(np.random.randn(n) * 0.01) + 100
    closes = np.exp(np.log(100) + np.cumsum(np.random.randn(n) * 0.01))
    volumes = np.abs(np.random.randn(n)) * 1000 + 500

    features = train_and_predict(closes, volumes)

    for k, v in features.items():
        print(f"  {k}: shape={v.shape}, unique={np.unique(v[:50])}")

    # Check regime distribution
    states = features['regime_state']
    for s in [0, 1, 2]:
        pct = (states == s).mean() * 100
        print(f"  {STATE_NAMES[s]}: {pct:.1f}%")

    print("[HMM] Smoke test passed")
