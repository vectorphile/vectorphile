"""Graph Neural Networks — GCN, GAT, GraphSAGE, GIN."""
from __future__ import annotations
import numpy as np
from typing import Optional, List, Tuple
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor


def _relu(x): return np.maximum(x, 0)
def _softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / (e.sum(axis=axis, keepdims=True) + 1e-9)


def _normalize_adj(adj: np.ndarray) -> np.ndarray:
    """Symmetric normalization: D^{-1/2} A D^{-1/2} (Kipf & Welling 2017).

    Args:
        adj: (N, N) adjacency matrix (can be sparse — dense for simplicity)
    Returns:
        normalized adjacency (N, N)
    """
    # Add self-loops: Ã = A + I
    A = adj + np.eye(adj.shape[0], dtype=np.float32)
    D = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-6))
    return D_inv_sqrt @ A @ D_inv_sqrt


def _degree_norm_adj(adj: np.ndarray) -> np.ndarray:
    """Row-wise D^{-1} normalization for GraphSAGE mean aggregation."""
    A = adj + np.eye(adj.shape[0], dtype=np.float32)
    D = A.sum(axis=1, keepdims=True)
    return A / (D + 1e-6)


# ── GCN ───────────────────────────────────────────────────────────────────────

class GCNLayer(Module):
    """Single Graph Convolutional layer — Kipf & Welling 2017.

    H' = σ(D̃^{-½} Ã D̃^{-½} H W)

    where Ã = A + I (self-loops added).
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, activation: str = 'relu'):
        super().__init__()
        self.activation = activation
        scale = np.sqrt(2.0 / in_features)
        self.weight = Tensor(np.random.randn(in_features, out_features).astype(np.float32) * scale)
        self.weight.requires_grad = True
        self._parameters['weight'] = self.weight
        if bias:
            self.bias = Tensor(np.zeros(out_features, dtype=np.float32))
            self.bias.requires_grad = True
            self._parameters['bias'] = self.bias
        else:
            self.bias = None

    def forward(self, x: Tensor, adj: np.ndarray) -> Tensor:
        """
        Args:
            x: (N, in_features) node feature matrix
            adj: (N, N) adjacency matrix (unnormalized; will be normalized internally)
        Returns:
            (N, out_features)
        """
        adj_norm = _normalize_adj(adj)
        out = adj_norm @ x._data @ self.weight._data
        if self.bias is not None:
            out = out + self.bias._data
        if self.activation == 'relu':
            out = _relu(out)
        return Tensor(out.astype(np.float32))


class GCN(Module):
    """Graph Convolutional Network — Kipf & Welling 2017.

    Stacks GCN layers for node classification or graph-level tasks.

    Args:
        in_features: input node feature dimension
        hidden_dims: list of hidden layer sizes
        out_features: output classes/dimension
        dropout: dropout rate
    """

    def __init__(self, in_features: int, hidden_dims: List[int],
                 out_features: int, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        dims = [in_features] + hidden_dims
        self.layers = []
        for i in range(len(dims) - 1):
            layer = GCNLayer(dims[i], dims[i + 1])
            self.layers.append(layer)
            self._modules[f'gcn_{i}'] = layer
        self.out_layer = GCNLayer(dims[-1], out_features, activation='none')
        self._modules['gcn_out'] = self.out_layer

    def forward(self, x: Tensor, adj: np.ndarray) -> Tensor:
        """
        Args:
            x: (N, in_features)
            adj: (N, N) adjacency matrix
        Returns:
            (N, out_features) logits
        """
        d = x._data
        for layer in self.layers:
            d = layer(Tensor(d), adj)._data
            if self.dropout > 0 and self.training:
                mask = np.random.binomial(1, 1 - self.dropout, d.shape).astype(np.float32)
                d = d * mask / (1 - self.dropout)
        return self.out_layer(Tensor(d), adj)


# ── GAT ───────────────────────────────────────────────────────────────────────

class GATLayer(Module):
    """Graph Attention Network layer — Veličković et al. 2018.

    Uses multi-head attention to weight neighbor contributions.

    h_i' = ‖_{k=1}^{K}  σ( Σ_{j∈N_i} α_{ij}^k W^k h_j )

    where α_{ij} = softmax over neighbors of LeakyReLU(aᵀ [Whi ‖ Whj]).
    """

    def __init__(self, in_features: int, out_features: int,
                 num_heads: int = 4, dropout: float = 0.6,
                 alpha: float = 0.2, concat: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.num_heads    = num_heads
        self.dropout      = dropout
        self.alpha_lr     = alpha
        self.concat       = concat

        # W: (num_heads, in_features, out_features)
        self.W = Tensor(np.random.randn(num_heads, in_features, out_features).astype(np.float32)
                        * np.sqrt(2.0 / in_features))
        self.W.requires_grad = True
        self._parameters['W'] = self.W
        # Attention vector a: (num_heads, 2*out_features)
        self.a = Tensor(np.random.randn(num_heads, 2 * out_features).astype(np.float32) * 0.01)
        self.a.requires_grad = True
        self._parameters['a'] = self.a

    def forward(self, x: Tensor, adj: np.ndarray) -> Tensor:
        """
        Args:
            x: (N, in_features)
            adj: (N, N) binary adjacency matrix (1 = connected, 0 = not)
        Returns:
            (N, num_heads * out_features) if concat else (N, out_features)
        """
        N = x._data.shape[0]
        h = x._data  # (N, in_features)

        head_outputs = []
        for k in range(self.num_heads):
            Wh = h @ self.W._data[k]   # (N, out_features)
            a_k = self.a._data[k]      # (2*out_features,)
            a1  = a_k[:self.out_features]   # source attention
            a2  = a_k[self.out_features:]   # target attention

            # e_{ij} = LeakyReLU(a1·h_i + a2·h_j)
            src_scores = Wh @ a1   # (N,)
            tgt_scores = Wh @ a2   # (N,)
            # Broadcast: (N, N) attention logits
            e = src_scores[:, None] + tgt_scores[None, :]  # (N, N)
            e = np.where(e > 0, e, self.alpha_lr * e)       # LeakyReLU

            # Mask non-edges (set to -inf before softmax)
            mask = adj == 0
            e = np.where(mask, -1e9, e)
            attn = _softmax(e, axis=-1)   # (N, N)

            if self.dropout > 0 and self.training:
                drop_mask = np.random.binomial(1, 1 - self.dropout, attn.shape).astype(np.float32)
                attn = attn * drop_mask / (1 - self.dropout)

            # Aggregate
            out_k = attn @ Wh   # (N, out_features)
            if self.concat:
                out_k = _relu(out_k)
            head_outputs.append(out_k)

        if self.concat:
            return Tensor(np.concatenate(head_outputs, axis=-1).astype(np.float32))
        # Average heads (last layer)
        return Tensor(np.stack(head_outputs, axis=0).mean(0).astype(np.float32))


class GAT(Module):
    """Graph Attention Network — multi-layer with configurable heads.

    Args:
        in_features: input node feature size
        hidden_dim: hidden layer feature size per head
        out_features: output classes
        num_heads: attention heads in hidden layers
        num_layers: total number of GAT layers
        dropout: dropout applied to attention weights and features
    """

    def __init__(self, in_features: int, hidden_dim: int,
                 out_features: int, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.6):
        super().__init__()
        self.dropout = dropout
        self.layers = []
        in_dim = in_features
        for i in range(num_layers - 1):
            layer = GATLayer(in_dim, hidden_dim, num_heads, dropout, concat=True)
            self.layers.append(layer)
            self._modules[f'gat_{i}'] = layer
            in_dim = hidden_dim * num_heads
        self.out_layer = GATLayer(in_dim, out_features, num_heads=1, dropout=dropout, concat=False)
        self._modules['gat_out'] = self.out_layer

    def forward(self, x: Tensor, adj: np.ndarray) -> Tensor:
        d = x._data
        for layer in self.layers:
            if self.dropout > 0 and self.training:
                mask = np.random.binomial(1, 1 - self.dropout, d.shape).astype(np.float32)
                d = d * mask / (1 - self.dropout)
            d = layer(Tensor(d), adj)._data
        return self.out_layer(Tensor(d), adj)


# ── GraphSAGE ─────────────────────────────────────────────────────────────────

class SAGELayer(Module):
    """Single GraphSAGE layer — Hamilton, Ying & Leskovec 2017.

    h_v = σ(W · [h_v ‖ AGG({h_u : u ∈ N(v)})])

    Supports 'mean', 'max', and 'lstm' (approximated as mean here) aggregators.
    """

    def __init__(self, in_features: int, out_features: int,
                 aggregator: str = 'mean', bias: bool = True):
        super().__init__()
        self.aggregator = aggregator
        # W: concatenated (self + aggregated) → out
        self.W = Tensor(np.random.randn(in_features * 2, out_features).astype(np.float32)
                        * np.sqrt(2.0 / (in_features * 2)))
        self.W.requires_grad = True; self._parameters['W'] = self.W
        if bias:
            self.b = Tensor(np.zeros(out_features, dtype=np.float32))
            self.b.requires_grad = True; self._parameters['b'] = self.b
        else:
            self.b = None

    def _aggregate(self, h: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """Aggregate neighborhood features."""
        # Mask self-connections for aggregation
        A = adj.copy()
        np.fill_diagonal(A, 0)
        counts = A.sum(axis=1, keepdims=True).clip(min=1)
        if self.aggregator == 'mean':
            return (A @ h) / counts
        elif self.aggregator == 'max':
            N = h.shape[0]
            agg = np.full_like(h, -1e9)
            for i in range(N):
                nbr = np.where(A[i] > 0)[0]
                if len(nbr) > 0:
                    agg[i] = h[nbr].max(axis=0)
                else:
                    agg[i] = np.zeros(h.shape[1], dtype=np.float32)
            return agg
        return (A @ h) / counts  # fallback to mean

    def forward(self, x: Tensor, adj: np.ndarray) -> Tensor:
        h = x._data
        h_agg = self._aggregate(h, adj)
        h_cat = np.concatenate([h, h_agg], axis=-1)
        out = h_cat @ self.W._data
        if self.b is not None:
            out = out + self.b._data
        # L2 normalize (standard GraphSAGE)
        norm = np.linalg.norm(out, axis=-1, keepdims=True).clip(min=1e-6)
        return Tensor(_relu(out / norm).astype(np.float32))


class GraphSAGE(Module):
    """GraphSAGE — Hamilton, Ying & Leskovec 2017.

    Inductive representation learning via neighborhood sampling and aggregation.

    Args:
        in_features: input feature size
        hidden_dims: hidden layer sizes
        out_features: output dimension
        aggregator: 'mean' or 'max'
        dropout: feature dropout
    """

    def __init__(self, in_features: int, hidden_dims: List[int],
                 out_features: int, aggregator: str = 'mean',
                 dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        dims = [in_features] + hidden_dims
        self.layers = []
        for i in range(len(dims) - 1):
            layer = SAGELayer(dims[i], dims[i + 1], aggregator)
            self.layers.append(layer)
            self._modules[f'sage_{i}'] = layer
        self.out_w = Tensor(np.random.randn(dims[-1], out_features).astype(np.float32) * 0.02)
        self.out_b = Tensor(np.zeros(out_features, dtype=np.float32))
        self.out_w.requires_grad = True; self.out_b.requires_grad = True
        self._parameters['out_w'] = self.out_w; self._parameters['out_b'] = self.out_b

    def forward(self, x: Tensor, adj: np.ndarray) -> Tensor:
        d = x._data
        for layer in self.layers:
            d = layer(Tensor(d), adj)._data
            if self.dropout > 0 and self.training:
                mask = np.random.binomial(1, 1 - self.dropout, d.shape).astype(np.float32)
                d = d * mask / (1 - self.dropout)
        return Tensor(d @ self.out_w._data + self.out_b._data)


# ── GIN ───────────────────────────────────────────────────────────────────────

class GINLayer(Module):
    """Graph Isomorphism Network layer — Xu et al. 2019.

    h_v^{(k)} = MLP^{(k)}((1 + ε) h_v^{(k-1)} + Σ_{u ∈ N(v)} h_u^{(k-1)})

    GIN achieves the same discriminative power as the Weisfeiler-Lehman test.
    """

    def __init__(self, in_features: int, out_features: int,
                 eps: float = 0.0, train_eps: bool = True,
                 mlp_hidden: Optional[int] = None):
        super().__init__()
        mlp_hidden = mlp_hidden or out_features
        self.train_eps_flag = train_eps
        self.epsilon = Tensor(np.array([eps], dtype=np.float32))
        if train_eps:
            self.epsilon.requires_grad = True
            self._parameters['epsilon'] = self.epsilon
        # 2-layer MLP
        self.w1 = Tensor(np.random.randn(in_features, mlp_hidden).astype(np.float32) * np.sqrt(2.0 / in_features))
        self.b1 = Tensor(np.zeros(mlp_hidden, dtype=np.float32))
        self.w2 = Tensor(np.random.randn(mlp_hidden, out_features).astype(np.float32) * np.sqrt(2.0 / mlp_hidden))
        self.b2 = Tensor(np.zeros(out_features, dtype=np.float32))
        # BN
        self.bn_w = Tensor(np.ones(mlp_hidden, dtype=np.float32))
        self.bn_b = Tensor(np.zeros(mlp_hidden, dtype=np.float32))
        self._rm = np.zeros(mlp_hidden, dtype=np.float32)
        self._rv = np.ones(mlp_hidden, dtype=np.float32)
        for n, p in [('w1',self.w1),('b1',self.b1),('w2',self.w2),('b2',self.b2),
                     ('bn_w',self.bn_w),('bn_b',self.bn_b)]:
            p.requires_grad = True; self._parameters[n] = p

    def _bn(self, x):
        if self.training:
            m, v = x.mean(0), x.var(0)
            self._rm[:] = 0.9 * self._rm + 0.1 * m
            self._rv[:] = 0.9 * self._rv + 0.1 * v
        else:
            m, v = self._rm, self._rv
        return (x - m) / np.sqrt(v + 1e-5) * self.bn_w._data + self.bn_b._data

    def forward(self, x: Tensor, adj: np.ndarray) -> Tensor:
        """
        Args:
            x: (N, in_features)
            adj: (N, N) adjacency matrix (no self-loops expected; added internally)
        Returns:
            (N, out_features)
        """
        h = x._data
        # Sum aggregation (no self-loop — ε handles self)
        A_no_self = adj.copy(); np.fill_diagonal(A_no_self, 0)
        agg = A_no_self @ h    # (N, in_features)
        eps = self.epsilon._data[0]
        h_new = (1 + eps) * h + agg
        # MLP
        h_new = _relu(self._bn(h_new @ self.w1._data + self.b1._data))
        h_new = h_new @ self.w2._data + self.b2._data
        return Tensor(h_new.astype(np.float32))


class GIN(Module):
    """Graph Isomorphism Network — Xu et al. 2019.

    Provably most expressive GNN variant up to 1-WL isomorphism test.
    Useful for graph classification tasks.

    Args:
        in_features: input node feature size
        hidden_dim: hidden dimension per layer
        out_features: output classes (graph-level)
        num_layers: number of GIN layers
        dropout: feature dropout
        readout: graph-level pooling ('sum', 'mean', 'max')
    """

    def __init__(self, in_features: int, hidden_dim: int,
                 out_features: int, num_layers: int = 5,
                 dropout: float = 0.5, readout: str = 'sum'):
        super().__init__()
        self.dropout = dropout
        self.readout = readout
        self.num_layers = num_layers
        self.layers = []
        in_dim = in_features
        for i in range(num_layers):
            layer = GINLayer(in_dim, hidden_dim)
            self.layers.append(layer)
            self._modules[f'gin_{i}'] = layer
            in_dim = hidden_dim
        # Readout MLP: concatenate all layer representations
        self.mlp_w1 = Tensor(np.random.randn(hidden_dim * num_layers, hidden_dim).astype(np.float32) * 0.02)
        self.mlp_b1 = Tensor(np.zeros(hidden_dim, dtype=np.float32))
        self.mlp_w2 = Tensor(np.random.randn(hidden_dim, out_features).astype(np.float32) * 0.02)
        self.mlp_b2 = Tensor(np.zeros(out_features, dtype=np.float32))
        for n, p in [('mlp_w1',self.mlp_w1),('mlp_b1',self.mlp_b1),
                     ('mlp_w2',self.mlp_w2),('mlp_b2',self.mlp_b2)]:
            p.requires_grad = True; self._parameters[n] = p

    def _pool(self, h: np.ndarray) -> np.ndarray:
        if self.readout == 'sum':  return h.sum(0, keepdims=True)
        if self.readout == 'max':  return h.max(0, keepdims=True)
        return h.mean(0, keepdims=True)

    def forward(self, x: Tensor, adj: np.ndarray,
                batch: Optional[np.ndarray] = None) -> Tensor:
        """
        Args:
            x: (N, in_features) node features
            adj: (N, N) adjacency matrix
            batch: (N,) int array mapping nodes to graphs (None = single graph)
        Returns:
            (num_graphs, out_features) logits
        """
        d = x._data
        layer_reprs = []
        for layer in self.layers:
            d = layer(Tensor(d), adj)._data
            if self.dropout > 0 and self.training:
                mask = np.random.binomial(1, 1 - self.dropout, d.shape).astype(np.float32)
                d = d * mask / (1 - self.dropout)
            layer_reprs.append(d)

        # Graph-level readout (concatenate representations from all layers)
        if batch is None:
            # Single graph
            pooled = [self._pool(h) for h in layer_reprs]  # each (1, hidden_dim)
            graph_repr = np.concatenate(pooled, axis=-1)    # (1, hidden_dim * num_layers)
        else:
            # Batched graphs
            num_graphs = batch.max() + 1
            graph_reprs = []
            for g in range(num_graphs):
                mask_g = batch == g
                pooled_g = [self._pool(h[mask_g]) for h in layer_reprs]
                graph_reprs.append(np.concatenate(pooled_g, axis=-1))
            graph_repr = np.concatenate(graph_reprs, axis=0)  # (G, hidden_dim * num_layers)

        # Classification MLP
        out = _relu(graph_repr @ self.mlp_w1._data + self.mlp_b1._data)
        if self.dropout > 0 and self.training:
            mask = np.random.binomial(1, 1 - self.dropout, out.shape).astype(np.float32)
            out = out * mask / (1 - self.dropout)
        return Tensor(out @ self.mlp_w2._data + self.mlp_b2._data)
