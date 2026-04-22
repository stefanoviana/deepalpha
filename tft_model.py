#!/usr/bin/env python3
"""
tft_model.py - Temporal Fusion Transformer for Trading
=======================================================
Simplified TFT implementation optimized for crypto trading.
Replaces LSTM with attention-based architecture.

Architecture:
  1. Variable Selection Network (learns which features matter)
  2. LSTM encoder (captures temporal patterns)
  3. Multi-head Attention (focuses on important time steps)
  4. Gated output (multi-horizon prediction: 1h, 4h, 12h)

Input: sequence of feature vectors (seq_len x n_features)
Output: probability of price going UP for each horizon

Dependencies: torch (PyTorch)
"""
import numpy as np
import pickle
import os
import logging

log = logging.getLogger("DeepAlpha")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class GatedLinearUnit(nn.Module):
    """GLU activation: splits input, applies sigmoid gate."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        out = self.fc(x)
        a, b = out.chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GateAddNorm(nn.Module):
    """Gated residual connection with layer norm."""
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.glu = GatedLinearUnit(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual=None):
        out = self.dropout(self.glu(x))
        if residual is not None:
            out = out + residual
        return self.norm(out)


class VariableSelectionNetwork(nn.Module):
    """Learns which input features are most important."""
    def __init__(self, n_features, hidden_dim, dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim

        # Per-feature transform
        self.feature_transforms = nn.ModuleList([
            nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())
            for _ in range(n_features)
        ])

        # Softmax weights over features
        self.weight_net = nn.Sequential(
            nn.Linear(n_features * hidden_dim, n_features),
            nn.Softmax(dim=-1)
        )

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        batch, seq_len, n_feat = x.shape

        # Transform each feature independently
        transformed = []
        for i in range(self.n_features):
            feat_i = x[:, :, i:i+1]  # (batch, seq_len, 1)
            t = self.feature_transforms[i](feat_i)  # (batch, seq_len, hidden)
            transformed.append(t)

        # Stack: (batch, seq_len, n_features, hidden)
        stacked = torch.stack(transformed, dim=2)

        # Compute feature weights
        flat = stacked.reshape(batch * seq_len, n_feat * self.hidden_dim)
        weights = self.weight_net(flat)  # (batch*seq_len, n_features)
        weights = weights.reshape(batch, seq_len, n_feat, 1)

        # Weighted sum
        selected = (stacked * weights).sum(dim=2)  # (batch, seq_len, hidden)
        return self.output_proj(selected), weights.squeeze(-1)


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention that returns interpretable attention weights."""
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch, seq_len, d_model = q.shape

        Q = self.W_q(q).reshape(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).reshape(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).reshape(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).reshape(batch, seq_len, d_model)
        output = self.W_o(context)

        # Average attention weights across heads for interpretability
        avg_attn = attn_weights.mean(dim=1)  # (batch, seq_len, seq_len)

        return output, avg_attn


class TemporalFusionTransformer(nn.Module):
    """
    Simplified TFT for crypto trading signals.

    Args:
        n_features: number of input features per timestep
        hidden_dim: hidden dimension (default 64)
        n_heads: number of attention heads (default 4)
        lstm_layers: number of LSTM layers (default 2)
        dropout: dropout rate (default 0.2)
        n_horizons: number of prediction horizons (default 3: 1h, 4h, 12h)
    """
    def __init__(self, n_features, hidden_dim=64, n_heads=4, lstm_layers=2,
                 dropout=0.2, n_horizons=3):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_horizons = n_horizons

        # Variable Selection
        self.vsn = VariableSelectionNetwork(n_features, hidden_dim, dropout)

        # Temporal processing: LSTM encoder
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=lstm_layers,
                           batch_first=True, dropout=dropout if lstm_layers > 1 else 0)

        # Post-LSTM gate
        self.post_lstm_gate = GateAddNorm(hidden_dim, hidden_dim, dropout)

        # Self-attention
        self.attention = InterpretableMultiHeadAttention(hidden_dim, n_heads, dropout)
        self.post_attn_gate = GateAddNorm(hidden_dim, hidden_dim, dropout)

        # Output heads (one per horizon)
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            for _ in range(n_horizons)
        ])

    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch, seq_len, n_features)
            return_attention: if True, also return attention weights

        Returns:
            predictions: (batch, n_horizons) - UP probability for each horizon
            feature_weights: (batch, seq_len, n_features) - feature importance
            attn_weights: (batch, seq_len, seq_len) - temporal attention (optional)
        """
        # Variable selection
        selected, feature_weights = self.vsn(x)

        # LSTM encoding
        lstm_out, _ = self.lstm(selected)

        # Gated skip connection
        temporal = self.post_lstm_gate(lstm_out, selected)

        # Self-attention
        attn_out, attn_weights = self.attention(temporal, temporal, temporal)
        enriched = self.post_attn_gate(attn_out, temporal)

        # Use last timestep for prediction
        last = enriched[:, -1, :]  # (batch, hidden_dim)

        # Multi-horizon output
        predictions = []
        for head in self.output_heads:
            predictions.append(head(last))
        predictions = torch.cat(predictions, dim=-1)  # (batch, n_horizons)

        if return_attention:
            return predictions, feature_weights, attn_weights
        return predictions, feature_weights


# =========================================================================
# Training utilities
# =========================================================================
def prepare_sequences(X, y_dict, seq_len=48):
    """
    Convert feature matrix + labels into sequences for TFT.

    Args:
        X: (n_samples, n_features) - all features sorted by time
        y_dict: dict with keys '1h', '4h' and optionally '12h'
        seq_len: sequence length (default 48 = 2 days of 1h candles)

    Returns:
        X_seq: (n_sequences, seq_len, n_features)
        y_seq: (n_sequences, n_horizons)
    """
    n = len(X)
    n_horizons = len(y_dict)
    horizon_keys = sorted(y_dict.keys())

    sequences = []
    labels = []

    for i in range(seq_len, n):
        seq = X[i - seq_len:i]
        label = [y_dict[k][i] for k in horizon_keys]

        # Skip if any label is invalid
        if any(l < 0 for l in label):
            continue

        sequences.append(seq)
        labels.append(label)

    if not sequences:
        return np.array([]), np.array([])

    return np.array(sequences), np.array(labels)


def train_tft(X_train, y_train, X_val, y_val, n_features,
              seq_len=48, hidden_dim=64, n_heads=4, n_horizons=2,
              epochs=100, lr=0.001, batch_size=256, patience=15,
              device='cpu'):
    """
    Train TFT model.

    Args:
        X_train, y_train: training sequences and labels
        X_val, y_val: validation sequences and labels
        n_features: number of input features
        Other args: hyperparameters

    Returns:
        model: trained TFT model
        history: dict with training metrics
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for TFT training")

    model = TemporalFusionTransformer(
        n_features=n_features, hidden_dim=hidden_dim,
        n_heads=n_heads, n_horizons=n_horizons, dropout=0.2
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    n_train = len(X_train)
    n_val = len(X_val)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0

        indices = np.random.permutation(n_train)
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch_idx = indices[start:end]

            x_batch = torch.FloatTensor(X_train[batch_idx]).to(device)
            y_batch = torch.FloatTensor(y_train[batch_idx]).to(device)

            optimizer.zero_grad()
            pred, _ = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        n_val_batches = 0

        with torch.no_grad():
            for start in range(0, n_val, batch_size):
                end = min(start + batch_size, n_val)
                x_batch = torch.FloatTensor(X_val[start:end]).to(device)
                y_batch = torch.FloatTensor(y_val[start:end]).to(device)

                pred, _ = model(x_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()
                n_val_batches += 1

                # Accuracy (average across horizons)
                pred_class = (pred > 0.5).float()
                val_correct += (pred_class == y_batch).float().sum().item()
                val_total += y_batch.numel()

        avg_val_loss = val_loss / max(n_val_batches, 1)
        val_acc = val_correct / max(val_total, 1) * 100

        scheduler.step(avg_val_loss)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            log.info(f"[TFT] Epoch {epoch+1}/{epochs} | "
                     f"train_loss={avg_train_loss:.4f} | "
                     f"val_loss={avg_val_loss:.4f} | "
                     f"val_acc={val_acc:.1f}%")

        if patience_counter >= patience:
            log.info(f"[TFT] Early stopping at epoch {epoch+1}")
            break

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    return model, history


def save_tft(model, config, normalization, accuracy, path):
    """Save TFT model in same format as LSTM."""
    data = {
        'model_state': model.state_dict(),
        'config': config,
        'normalization': normalization,
        'accuracy': accuracy,
        'model_type': 'tft',
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_tft(path):
    """Load trained TFT model."""
    if not HAS_TORCH:
        return None, None, None

    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)

        cfg = data['config']
        model = TemporalFusionTransformer(
            n_features=cfg['n_features'],
            hidden_dim=cfg.get('hidden_dim', 64),
            n_heads=cfg.get('n_heads', 4),
            n_horizons=cfg.get('n_horizons', 2),
        )
        model.load_state_dict(data['model_state'])
        model.eval()

        return model, cfg, data.get('normalization', {})
    except Exception as e:
        log.warning(f"[TFT] Failed to load: {e}")
        return None, None, None


def predict_tft(model, features_seq, normalization=None):
    """
    Predict using TFT model.

    Args:
        model: trained TFT model
        features_seq: (seq_len, n_features) array
        normalization: dict with 'mean' and 'std' arrays

    Returns:
        predictions: (n_horizons,) array of UP probabilities
        feature_importance: (n_features,) array of importance scores
    """
    if not HAS_TORCH or model is None:
        return None, None

    try:
        seq = np.array(features_seq, dtype=np.float32)

        if normalization:
            mean = np.array(normalization['mean'], dtype=np.float32)
            std = np.array(normalization['std'], dtype=np.float32)
            std = np.where(std < 1e-8, 1.0, std)
            seq = (seq - mean) / std

        with torch.no_grad():
            x = torch.FloatTensor(seq).unsqueeze(0)
            pred, feat_weights = model(x)

        predictions = pred[0].numpy()
        # Average feature weights across time steps
        importance = feat_weights[0].mean(dim=0).numpy()

        return predictions, importance

    except Exception as e:
        log.warning(f"[TFT] Predict error: {e}")
        return None, None


if __name__ == '__main__':
    if not HAS_TORCH:
        print("[TFT] PyTorch not available, skipping test")
    else:
        print("[TFT] Temporal Fusion Transformer smoke test")

        # Synthetic data
        np.random.seed(42)
        n = 2000
        n_features = 20
        seq_len = 48

        X = np.random.randn(n, n_features).astype(np.float32)
        y = {'1h': (np.random.rand(n) > 0.5).astype(np.float32),
             '4h': (np.random.rand(n) > 0.5).astype(np.float32)}

        X_seq, y_seq = prepare_sequences(X, y, seq_len=seq_len)
        print(f"  Sequences: {X_seq.shape}, Labels: {y_seq.shape}")

        split = int(len(X_seq) * 0.8)
        model, history = train_tft(
            X_seq[:split], y_seq[:split],
            X_seq[split:], y_seq[split:],
            n_features=n_features, seq_len=seq_len,
            epochs=5, batch_size=64
        )

        pred, importance = predict_tft(model, X_seq[0])
        print(f"  Prediction: {pred}")
        print(f"  Feature importance: {importance[:5]}...")

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")
        print("[TFT] Smoke test passed")
