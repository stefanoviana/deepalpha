#!/usr/bin/env python3
"""
gnn_model.py - Graph Neural Network for Cross-Asset Trading
=============================================================
Models the crypto market as a graph where:
  - Each coin is a NODE with its features
  - EDGES represent correlations between coins
  - Message passing captures lead-lag relationships

Architecture:
  1. Graph construction from correlation matrix
  2. GCN (Graph Convolutional Network) layers for message passing
  3. Node-level prediction (per-coin UP/DOWN probability)

Key insight: if BTC drops, the GNN learns that ETH follows in ~5min,
SOL in ~10min, DOGE in ~30min. This gives earlier signals.

Dependencies: torch (PyTorch). No torch_geometric needed.
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


class GraphConvLayer(nn.Module):
    """Simple Graph Convolutional Layer (Kipf & Welling, 2017)."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        Args:
            x: (batch, n_nodes, in_features)
            adj: (n_nodes, n_nodes) normalized adjacency matrix

        Returns:
            (batch, n_nodes, out_features)
        """
        # Message passing: A * X * W + b
        support = torch.matmul(x, self.weight)  # (batch, n_nodes, out_features)
        output = torch.matmul(adj, support)  # (batch, n_nodes, out_features)
        return output + self.bias


class CryptoGNN(nn.Module):
    """
    Graph Neural Network for crypto market.

    Takes features for ALL coins simultaneously and predicts
    UP/DOWN probability for each coin, using cross-asset information.
    """
    def __init__(self, n_features, hidden_dim=64, n_gcn_layers=3, dropout=0.2):
        super().__init__()
        self.n_features = n_features

        # Node feature encoder (shared across all coins)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # GCN layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_norms = nn.ModuleList()
        for i in range(n_gcn_layers):
            self.gcn_layers.append(GraphConvLayer(hidden_dim, hidden_dim))
            self.gcn_norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Node-level prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # concat local + global
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, adj):
        """
        Args:
            x: (batch, n_nodes, n_features) - features for all coins
            adj: (n_nodes, n_nodes) - normalized adjacency matrix

        Returns:
            pred: (batch, n_nodes, 1) - UP probability per coin
        """
        # Encode node features
        h = self.encoder(x)  # (batch, n_nodes, hidden)

        # GCN message passing
        for gcn, norm in zip(self.gcn_layers, self.gcn_norms):
            h_new = gcn(h, adj)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            h = norm(h_new + h)  # residual connection

        # Global graph context (mean pooling)
        global_ctx = h.mean(dim=1, keepdim=True).expand_as(h)  # (batch, n_nodes, hidden)

        # Concatenate local + global for prediction
        combined = torch.cat([h, global_ctx], dim=-1)  # (batch, n_nodes, hidden*2)
        pred = self.head(combined)  # (batch, n_nodes, 1)

        return pred


def build_correlation_graph(returns_dict, threshold=0.3):
    """
    Build adjacency matrix from return correlations.

    Args:
        returns_dict: dict of {coin_name: np.array of returns}
        threshold: minimum absolute correlation for edge (default 0.3)

    Returns:
        adj: (n_coins, n_coins) normalized adjacency matrix
        coin_order: list of coin names in matrix order
    """
    coins = sorted(returns_dict.keys())
    n = len(coins)

    # Compute correlation matrix
    returns_matrix = np.column_stack([returns_dict[c] for c in coins])
    min_len = min(len(returns_dict[c]) for c in coins)
    returns_matrix = returns_matrix[:min_len]

    corr = np.corrcoef(returns_matrix.T)
    corr = np.nan_to_num(corr, nan=0.0)

    # Build adjacency: threshold + self-loops
    adj = np.zeros((n, n))
    for i in range(n):
        adj[i, i] = 1.0  # self-loop
        for j in range(i + 1, n):
            if abs(corr[i, j]) > threshold:
                adj[i, j] = abs(corr[i, j])
                adj[j, i] = abs(corr[i, j])

    # Normalize: D^{-1/2} A D^{-1/2}
    degree = adj.sum(axis=1)
    d_inv_sqrt = np.zeros_like(degree)
    nonzero = degree > 0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
    D = np.diag(d_inv_sqrt)
    adj_norm = D @ adj @ D

    return adj_norm.astype(np.float32), coins


def build_lead_lag_graph(returns_dict, max_lag=5):
    """
    Build directed graph based on lead-lag relationships.

    If coin A's returns at time t predict coin B at time t+lag,
    then A→B edge exists.

    Args:
        returns_dict: dict of {coin: returns_array}
        max_lag: maximum lag to check (in candles)

    Returns:
        adj: (n_coins, n_coins) normalized adjacency (DIRECTED)
        coin_order: list of coin names
    """
    coins = sorted(returns_dict.keys())
    n = len(coins)
    min_len = min(len(returns_dict[c]) for c in coins)

    adj = np.eye(n, dtype=np.float32)  # self-loops

    for i, ci in enumerate(coins):
        ri = returns_dict[ci][:min_len]
        for j, cj in enumerate(coins):
            if i == j:
                continue
            rj = returns_dict[cj][:min_len]

            # Check if coin i leads coin j
            best_corr = 0.0
            for lag in range(1, max_lag + 1):
                if lag >= min_len:
                    break
                corr = np.corrcoef(ri[:-lag], rj[lag:])[0, 1]
                if not np.isnan(corr):
                    best_corr = max(best_corr, abs(corr))

            if best_corr > 0.15:
                adj[i, j] = best_corr

    # Row-normalize
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    adj_norm = adj / row_sums

    return adj_norm, coins


def train_gnn(features_dict, labels_dict, adj, coin_order,
              n_features=None, hidden_dim=64, epochs=100,
              lr=0.001, batch_size=32, patience=15, device='cpu'):
    """
    Train GNN model.

    Args:
        features_dict: {coin: (n_samples, n_features)} aligned by time
        labels_dict: {coin: (n_samples,)} binary labels
        adj: (n_coins, n_coins) adjacency matrix
        coin_order: list of coin names matching adj rows

    Returns:
        model: trained CryptoGNN
        history: training metrics
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required")

    # Build aligned data matrix: (n_times, n_coins, n_features)
    n_coins = len(coin_order)
    min_len = min(len(features_dict[c]) for c in coin_order)
    if n_features is None:
        n_features = features_dict[coin_order[0]].shape[1]

    X_all = np.zeros((min_len, n_coins, n_features), dtype=np.float32)
    y_all = np.zeros((min_len, n_coins), dtype=np.float32)

    for idx, coin in enumerate(coin_order):
        X_all[:, idx, :] = features_dict[coin][:min_len, :n_features]
        y_all[:, idx] = labels_dict[coin][:min_len]

    # Split train/val (time-based)
    split = int(min_len * 0.8)
    X_train = X_all[:split]
    y_train = y_all[:split]
    X_val = X_all[split:]
    y_val = y_all[split:]

    model = CryptoGNN(n_features=n_features, hidden_dim=hidden_dim).to(device)
    adj_tensor = torch.FloatTensor(adj).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
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
            y_b = torch.FloatTensor(y_train[batch_idx]).unsqueeze(-1).to(device)

            optimizer.zero_grad()
            pred = model(x_b, adj_tensor)
            loss = criterion(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train = train_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            x_v = torch.FloatTensor(X_val).to(device)
            y_v = torch.FloatTensor(y_val).unsqueeze(-1).to(device)
            pred_v = model(x_v, adj_tensor)
            val_loss = criterion(pred_v, y_v).item()
            val_acc = ((pred_v > 0.5).float() == y_v).float().mean().item() * 100

        history['train_loss'].append(avg_train)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            log.info(f"[GNN] Epoch {epoch+1}/{epochs} | train={avg_train:.4f} | "
                     f"val={val_loss:.4f} | acc={val_acc:.1f}%")

        if patience_counter >= patience:
            log.info(f"[GNN] Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    return model, history


def save_gnn(model, adj, coin_order, config, normalization, accuracy, path):
    """Save GNN model + graph structure."""
    data = {
        'model_state': model.state_dict(),
        'adj': adj,
        'coin_order': coin_order,
        'config': config,
        'normalization': normalization,
        'accuracy': accuracy,
        'model_type': 'gnn',
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_gnn(path):
    """Load GNN model."""
    if not HAS_TORCH:
        return None, None, None, None
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        cfg = data['config']
        model = CryptoGNN(
            n_features=cfg['n_features'],
            hidden_dim=cfg.get('hidden_dim', 64),
        )
        model.load_state_dict(data['model_state'])
        model.eval()
        return model, data['adj'], data['coin_order'], data.get('normalization', {})
    except Exception as e:
        log.warning(f"[GNN] Load failed: {e}")
        return None, None, None, None


def predict_gnn(model, features_dict, adj, coin_order, normalization=None, target_coin=None):
    """
    Predict UP probability for all coins (or a specific coin).

    Args:
        model: trained CryptoGNN
        features_dict: {coin: (n_features,) array} current features for each coin
        adj: adjacency matrix
        coin_order: list of coin names
        normalization: optional {'mean': array, 'std': array}
        target_coin: optional specific coin to get prediction for

    Returns:
        dict of {coin: probability} or single float if target_coin specified
    """
    if not HAS_TORCH or model is None:
        return None

    try:
        n_coins = len(coin_order)
        n_features = model.n_features

        # Build input tensor
        x = np.zeros((1, n_coins, n_features), dtype=np.float32)
        for idx, coin in enumerate(coin_order):
            if coin in features_dict:
                feat = np.array(features_dict[coin][:n_features], dtype=np.float32)
                if normalization:
                    mean = np.array(normalization['mean'][:n_features], dtype=np.float32)
                    std = np.array(normalization['std'][:n_features], dtype=np.float32)
                    std = np.where(std < 1e-8, 1.0, std)
                    feat = (feat - mean) / std
                x[0, idx, :len(feat)] = feat

        adj_tensor = torch.FloatTensor(adj)
        x_tensor = torch.FloatTensor(x)

        with torch.no_grad():
            pred = model(x_tensor, adj_tensor)  # (1, n_coins, 1)

        probs = pred[0, :, 0].numpy()
        result = {coin: float(probs[idx]) for idx, coin in enumerate(coin_order)}

        if target_coin:
            return result.get(target_coin, 0.5)
        return result

    except Exception as e:
        log.warning(f"[GNN] Predict error: {e}")
        return None


if __name__ == '__main__':
    if not HAS_TORCH:
        print("[GNN] PyTorch not available")
    else:
        print("[GNN] Graph Neural Network smoke test")

        np.random.seed(42)
        coins = ['BTC', 'ETH', 'SOL', 'DOGE', 'LINK']
        n = 500
        n_features = 20

        # Synthetic correlated returns
        btc_returns = np.random.randn(n) * 0.01
        returns_dict = {'BTC': btc_returns}
        for coin in coins[1:]:
            lag = np.random.randint(1, 4)
            noise = np.random.randn(n) * 0.005
            r = np.roll(btc_returns, lag) * (0.5 + np.random.rand() * 0.5) + noise
            returns_dict[coin] = r

        # Build graph
        adj, order = build_correlation_graph(returns_dict, threshold=0.1)
        print(f"  Adjacency matrix:\n{adj}")

        # Synthetic features and labels
        features_dict = {c: np.random.randn(n, n_features).astype(np.float32) for c in coins}
        labels_dict = {c: (np.random.rand(n) > 0.5).astype(np.float32) for c in coins}

        model, history = train_gnn(
            features_dict, labels_dict, adj, order,
            n_features=n_features, epochs=5, batch_size=32
        )

        # Predict
        current = {c: np.random.randn(n_features) for c in coins}
        result = predict_gnn(model, current, adj, order)
        for c, p in result.items():
            print(f"  {c}: {p:.4f}")

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # Lead-lag graph
        adj_ll, _ = build_lead_lag_graph(returns_dict, max_lag=3)
        print(f"  Lead-lag adjacency:\n{adj_ll}")

        print("[GNN] Smoke test passed")
