#!/usr/bin/env python3
"""
transformer_gru_model.py - Transformer + GRU Hybrid for Trading
================================================================
Combines:
  - Transformer encoder: captures long-range dependencies (what happened 30 days ago)
  - GRU decoder: captures short-range dynamics (momentum in last few hours)
  - Combined prediction head

This is the state-of-the-art architecture for financial time series.

Input: (batch, seq_len, n_features)
Output: (batch, 1) probability of UP

Dependencies: torch (PyTorch)
"""
import numpy as np
import pickle
import logging

log = logging.getLogger("DeepAlpha")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])  # handle odd d_model
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerGRU(nn.Module):
    """
    Hybrid Transformer + GRU model.

    Architecture:
      Input → Linear projection → Positional Encoding
        → Transformer Encoder (long-range context)
        → GRU (short-range dynamics)
        → Attention pooling
        → Prediction head
    """
    def __init__(self, n_features, d_model=64, n_heads=4, n_transformer_layers=2,
                 gru_hidden=64, gru_layers=1, dropout=0.2):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

        # GRU for short-range patterns
        self.gru = nn.GRU(d_model, gru_hidden, num_layers=gru_layers,
                         batch_first=True, dropout=dropout if gru_layers > 1 else 0)

        # Attention pooling (learn which time steps matter most)
        self.attn_pool = nn.Sequential(
            nn.Linear(gru_hidden, 1),
        )

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            prob_up: (batch, 1)
        """
        # Project input
        h = self.input_proj(x)  # (batch, seq_len, d_model)
        h = self.pos_enc(h)

        # Transformer: long-range patterns
        h = self.transformer(h)  # (batch, seq_len, d_model)

        # GRU: short-range dynamics
        h, _ = self.gru(h)  # (batch, seq_len, gru_hidden)

        # Attention pooling
        attn_scores = self.attn_pool(h).squeeze(-1)  # (batch, seq_len)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, seq_len)
        context = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)  # (batch, gru_hidden)

        # Predict
        return self.head(context)


def train_transformer_gru(X_train, y_train, X_val, y_val, n_features,
                           d_model=64, n_heads=4, epochs=100, lr=0.0005,
                           batch_size=256, patience=15, device='cpu'):
    """
    Train Transformer+GRU hybrid.

    Args:
        X_train: (n_samples, seq_len, n_features) training sequences
        y_train: (n_samples,) binary labels
        X_val, y_val: validation data
        Other: hyperparameters

    Returns:
        model: trained model
        history: training metrics
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required")

    model = TransformerGRU(
        n_features=n_features, d_model=d_model, n_heads=n_heads,
        n_transformer_layers=2, gru_hidden=d_model, dropout=0.2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    n_train = len(X_train)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

        indices = np.random.permutation(n_train)
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch_idx = indices[start:end]

            x_b = torch.FloatTensor(X_train[batch_idx]).to(device)
            y_b = torch.FloatTensor(y_train[batch_idx]).unsqueeze(1).to(device)

            optimizer.zero_grad()
            pred = model(x_b)
            loss = criterion(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        n_val_batches = 0

        with torch.no_grad():
            for start in range(0, len(X_val), batch_size):
                end = min(start + batch_size, len(X_val))
                x_b = torch.FloatTensor(X_val[start:end]).to(device)
                y_b = torch.FloatTensor(y_val[start:end]).unsqueeze(1).to(device)

                pred = model(x_b)
                loss = criterion(pred, y_b)
                val_loss += loss.item()
                n_val_batches += 1

                pred_class = (pred > 0.5).float()
                val_correct += (pred_class == y_b).float().sum().item()
                val_total += len(y_b)

        avg_val_loss = val_loss / max(n_val_batches, 1)
        val_acc = val_correct / max(val_total, 1) * 100

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
            log.info(f"[TrGRU] Epoch {epoch+1}/{epochs} | "
                     f"train={avg_train_loss:.4f} | val={avg_val_loss:.4f} | "
                     f"acc={val_acc:.1f}%")

        if patience_counter >= patience:
            log.info(f"[TrGRU] Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    return model, history


def save_transformer_gru(model, config, normalization, accuracy, path):
    """Save model."""
    data = {
        'model_state': model.state_dict(),
        'config': config,
        'normalization': normalization,
        'accuracy': accuracy,
        'model_type': 'transformer_gru',
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_transformer_gru(path):
    """Load model."""
    if not HAS_TORCH:
        return None, None, None
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        cfg = data['config']
        model = TransformerGRU(
            n_features=cfg['n_features'],
            d_model=cfg.get('d_model', 64),
            n_heads=cfg.get('n_heads', 4),
        )
        model.load_state_dict(data['model_state'])
        model.eval()
        return model, cfg, data.get('normalization', {})
    except Exception as e:
        log.warning(f"[TrGRU] Load failed: {e}")
        return None, None, None


def predict_transformer_gru(model, features_seq, normalization=None):
    """
    Predict UP probability.

    Args:
        model: trained TransformerGRU
        features_seq: (seq_len, n_features)
        normalization: dict with 'mean', 'std'

    Returns:
        float: probability of UP (0-1)
    """
    if not HAS_TORCH or model is None:
        return None
    try:
        seq = np.array(features_seq, dtype=np.float32)
        if normalization:
            mean = np.array(normalization['mean'], dtype=np.float32)
            std = np.array(normalization['std'], dtype=np.float32)
            std = np.where(std < 1e-8, 1.0, std)
            seq = (seq - mean) / std

        with torch.no_grad():
            x = torch.FloatTensor(seq).unsqueeze(0)
            prob = model(x).item()
        return prob
    except Exception as e:
        log.warning(f"[TrGRU] Predict error: {e}")
        return None


if __name__ == '__main__':
    if not HAS_TORCH:
        print("[TrGRU] PyTorch not available")
    else:
        print("[TrGRU] Transformer+GRU smoke test")

        np.random.seed(42)
        n = 2000
        n_features = 20
        seq_len = 48

        X = np.random.randn(n, n_features).astype(np.float32)
        y = (np.random.rand(n) > 0.5).astype(np.float32)

        # Build sequences
        X_seq = np.array([X[i-seq_len:i] for i in range(seq_len, n)])
        y_seq = y[seq_len:]

        split = int(len(X_seq) * 0.8)
        model, history = train_transformer_gru(
            X_seq[:split], y_seq[:split],
            X_seq[split:], y_seq[split:],
            n_features=n_features, epochs=5, batch_size=64
        )

        prob = predict_transformer_gru(model, X_seq[0])
        print(f"  Prediction: {prob:.4f}")

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")
        print("[TrGRU] Smoke test passed")
