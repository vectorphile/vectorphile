"""Graph Neural Networks — GCN, GAT, GraphSAGE, GIN."""
from __future__ import annotations
import math as _math
import random as _random
from typing import Optional, List, Tuple
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor
from vectorphile._core.ndarray import array as _nd_arr, NDArray


def _nd_randn(shape, scale=1.0):
    n = 1
    for s in shape: n *= s
    flat = [_math.sqrt(-2*_math.log(max(_random.random(),1e-10)))*_math.cos(2*_math.pi*_random.random())*scale
            for _ in range(n)]
    return _nd_arr(flat, dtype='float32').reshape(shape)

def _nd_zeros(shape):
    n = 1
    for s in shape: n *= s
    return _nd_arr([0.0]*n, dtype='float32').reshape(shape)

def _nd_ones(shape):
    n = 1
    for s in shape: n *= s
    return _nd_arr([1.0]*n, dtype='float32').reshape(shape)


def _relu(x: NDArray) -> NDArray:
    fl = x.to_flat_list()
    return _nd_arr([max(0.0, v) for v in fl], dtype='float32').reshape(x.shape)

def _softmax_rows(x: NDArray) -> NDArray:
    """Row-wise softmax for 2D (N, N) NDArray."""
    N = x.shape[0]
    fl = x.to_flat_list()
    result = []
    for i in range(N):
        row = fl[i*N:(i+1)*N]
        mx = max(row)
        exps = [_math.exp(v - mx) for v in row]
        s = sum(exps) + 1e-9
        result.extend([v / s for v in exps])
    return _nd_arr(result, dtype='float32').reshape((N, N))


def _normalize_adj(adj: NDArray) -> NDArray:
    """Symmetric normalization: D^{-1/2} A D^{-1/2}."""
    N = adj.shape[0]
    af = adj.to_flat_list()
    # Add self-loops
    a2 = [af[i*N+j] + (1.0 if i == j else 0.0) for i in range(N) for j in range(N)]
    # Row sums (degrees)
    d = [sum(a2[i*N:(i+1)*N]) for i in range(N)]
    d_inv_sqrt = [1.0 / _math.sqrt(di + 1e-6) for di in d]
    # D^{-1/2} A D^{-1/2}
    result = [d_inv_sqrt[i] * a2[i*N+j] * d_inv_sqrt[j] for i in range(N) for j in range(N)]
    return _nd_arr(result, dtype='float32').reshape((N, N))


def _degree_norm_adj(adj: NDArray) -> NDArray:
    """Row-wise D^{-1} normalization."""
    N = adj.shape[0]
    af = adj.to_flat_list()
    a2 = [af[i*N+j] + (1.0 if i == j else 0.0) for i in range(N) for j in range(N)]
    d = [sum(a2[i*N:(i+1)*N]) for i in range(N)]
    result = [a2[i*N+j] / (d[i] + 1e-6) for i in range(N) for j in range(N)]
    return _nd_arr(result, dtype='float32').reshape((N, N))


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
        scale = _math.sqrt(2.0 / in_features)
        self.weight = Tensor(_nd_randn((in_features, out_features), scale))
        self.weight.requires_grad = True
        self._parameters['weight'] = self.weight
        if bias:
            self.bias = Tensor(_nd_zeros((out_features,)))
            self.bias.requires_grad = True
            self._parameters['bias'] = self.bias
        else:
            self.bias = None

    def forward(self, x: Tensor, adj: NDArray) -> Tensor:
        adj_norm = _normalize_adj(adj)
        out = adj_norm @ x._data @ self.weight._data
        if self.bias is not None:
            out = out + self.bias._data
        if self.activation == 'relu':
            out = _relu(out)
        return Tensor(out)


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

    def forward(self, x: Tensor, adj: NDArray) -> Tensor:
        d = x._data
        for layer in self.layers:
            d = layer(Tensor(d), adj)._data
            if self.dropout > 0 and self.training:
                df = d.to_flat_list()
                df = [v * (1.0 if _random.random() < (1 - self.dropout) else 0.0) / (1 - self.dropout) for v in df]
                d = _nd_arr(df, dtype='float32').reshape(d.shape)
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
        self.W = Tensor(_nd_randn((num_heads, in_features, out_features), _math.sqrt(2.0 / in_features)))
        self.W.requires_grad = True
        self._parameters['W'] = self.W
        # Attention vector a: (num_heads, 2*out_features)
        self.a = Tensor(_nd_randn((num_heads, 2 * out_features), 0.01))
        self.a.requires_grad = True
        self._parameters['a'] = self.a

    def forward(self, x: Tensor, adj: NDArray) -> Tensor:
        N, F = x._data.shape
        hf = x._data.to_flat_list()
        af = adj.to_flat_list()
        Wf = self.W._data.to_flat_list()   # (heads, in_features, out_features)
        attn_f = self.a._data.to_flat_list()  # (heads, 2*out_features)
        O = self.out_features

        head_outputs = []
        for k in range(self.num_heads):
            # Extract W[k]: (F, O)
            Wk = Wf[k*F*O:(k+1)*F*O]
            # Wh = h @ W[k]: (N, O)
            Wh = []
            for i in range(N):
                for o in range(O):
                    s = sum(hf[i*F + f] * Wk[f*O + o] for f in range(F))
                    Wh.append(s)
            # a1 = attn[k, :O], a2 = attn[k, O:]
            ak = attn_f[k*2*O:(k+1)*2*O]
            a1 = ak[:O]; a2 = ak[O:]
            # src_scores = Wh @ a1: (N,)
            src = [sum(Wh[i*O+o] * a1[o] for o in range(O)) for i in range(N)]
            tgt = [sum(Wh[i*O+o] * a2[o] for o in range(O)) for i in range(N)]
            # e[i,j] = LeakyReLU(src[i] + tgt[j]), masked where adj==0
            e = []
            for i in range(N):
                for j in range(N):
                    raw = src[i] + tgt[j]
                    lrelu = raw if raw > 0 else self.alpha_lr * raw
                    # mask: if adj[i,j]==0, set to -1e9
                    e.append(lrelu if af[i*N+j] > 0.5 else -1e9)
            e_nd = _nd_arr(e, dtype='float32').reshape((N, N))
            attn = _softmax_rows(e_nd)   # (N, N)
            if self.dropout > 0 and self.training:
                atf = attn.to_flat_list()
                atf = [v * (1.0 if _random.random() < (1-self.dropout) else 0.0) / (1-self.dropout) for v in atf]
                attn = _nd_arr(atf, dtype='float32').reshape((N, N))
            # out_k = attn @ Wh: (N, O)
            Wh_nd = _nd_arr(Wh, dtype='float32').reshape((N, O))
            out_k = attn @ Wh_nd
            if self.concat:
                out_k = _relu(out_k)
            head_outputs.append(out_k)

        if self.concat:
            # concatenate along last axis: (N, heads*O)
            from vectorphile.core.tensor import _nd_concatenate as _gat_cat
            return Tensor(_gat_cat(head_outputs, axis=-1))
        # Average heads
        total = head_outputs[0]
        for h_out in head_outputs[1:]:
            total = total + h_out
        hf2 = total.to_flat_list()
        avg = _nd_arr([v / self.num_heads for v in hf2], dtype='float32').reshape((N, O))
        return Tensor(avg)


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

    def forward(self, x: Tensor, adj: NDArray) -> Tensor:
        d = x._data
        for layer in self.layers:
            if self.dropout > 0 and self.training:
                df = d.to_flat_list()
                df = [v * (1.0 if _random.random() < (1-self.dropout) else 0.0) / (1-self.dropout) for v in df]
                d = _nd_arr(df, dtype='float32').reshape(d.shape)
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
        self.W = Tensor(_nd_randn((in_features * 2, out_features), _math.sqrt(2.0 / (in_features * 2))))
        self.W.requires_grad = True; self._parameters['W'] = self.W
        if bias:
            self.b = Tensor(_nd_zeros((out_features,)))
            self.b.requires_grad = True; self._parameters['b'] = self.b
        else:
            self.b = None

    def _aggregate(self, h: NDArray, adj: NDArray) -> NDArray:
        N, F = h.shape
        af = adj.to_flat_list()
        hf = h.to_flat_list()
        # Remove diagonal
        A_no_self = [af[i*N+j] if i != j else 0.0 for i in range(N) for j in range(N)]
        if self.aggregator == 'mean':
            agg = []
            for i in range(N):
                row = A_no_self[i*N:(i+1)*N]
                cnt = max(sum(row), 1.0)
                for f in range(F):
                    agg.append(sum(row[j] * hf[j*F+f] for j in range(N)) / cnt)
            return _nd_arr(agg, dtype='float32').reshape((N, F))
        elif self.aggregator == 'max':
            agg = []
            for i in range(N):
                row = A_no_self[i*N:(i+1)*N]
                nbrs = [j for j in range(N) if row[j] > 0.5]
                for f in range(F):
                    if nbrs:
                        agg.append(max(hf[j*F+f] for j in nbrs))
                    else:
                        agg.append(0.0)
            return _nd_arr(agg, dtype='float32').reshape((N, F))
        return self._aggregate_mean(h, adj)

    def forward(self, x: Tensor, adj: NDArray) -> Tensor:
        h = x._data
        h_agg = self._aggregate(h, adj)
        from vectorphile.core.tensor import _nd_concatenate as _sage_cat
        h_cat = _sage_cat([h, h_agg], axis=-1)
        out = h_cat @ self.W._data
        if self.b is not None:
            out = out + self.b._data
        # L2 normalize (standard GraphSAGE)
        N, O = out.shape
        of = out.to_flat_list()
        norm_flat = []
        for i in range(N):
            row = of[i*O:(i+1)*O]
            n_val = max(_math.sqrt(sum(v**2 for v in row)), 1e-6)
            norm_flat.extend([v / n_val for v in row])
        normed = _nd_arr(norm_flat, dtype='float32').reshape((N, O))
        return Tensor(_relu(normed))


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
        self.out_w = Tensor(_nd_randn((dims[-1], out_features), 0.02))
        self.out_b = Tensor(_nd_zeros((out_features,)))
        self.out_w.requires_grad = True; self.out_b.requires_grad = True
        self._parameters['out_w'] = self.out_w; self._parameters['out_b'] = self.out_b

    def forward(self, x: Tensor, adj: NDArray) -> Tensor:
        d = x._data
        for layer in self.layers:
            d = layer(Tensor(d), adj)._data
            if self.dropout > 0 and self.training:
                df = d.to_flat_list()
                df = [v * (1.0 if _random.random() < (1-self.dropout) else 0.0) / (1-self.dropout) for v in df]
                d = _nd_arr(df, dtype='float32').reshape(d.shape)
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
        self.epsilon = Tensor(_nd_arr([eps], dtype='float32').reshape((1,)))
        if train_eps:
            self.epsilon.requires_grad = True
            self._parameters['epsilon'] = self.epsilon
        # 2-layer MLP
        self.w1 = Tensor(_nd_randn((in_features, mlp_hidden), _math.sqrt(2.0 / in_features)))
        self.b1 = Tensor(_nd_zeros((mlp_hidden,)))
        self.w2 = Tensor(_nd_randn((mlp_hidden, out_features), _math.sqrt(2.0 / mlp_hidden)))
        self.b2 = Tensor(_nd_zeros((out_features,)))
        # BN
        self.bn_w = Tensor(_nd_ones((mlp_hidden,)))
        self.bn_b = Tensor(_nd_zeros((mlp_hidden,)))
        self._rm = [0.0] * mlp_hidden
        self._rv = [1.0] * mlp_hidden
        for n, p in [('w1',self.w1),('b1',self.b1),('w2',self.w2),('b2',self.b2),
                     ('bn_w',self.bn_w),('bn_b',self.bn_b)]:
            p.requires_grad = True; self._parameters[n] = p

    def _bn(self, x: NDArray) -> NDArray:
        N, F = x.shape
        xf = x.to_flat_list()
        wf = self.bn_w._data.to_flat_list()
        bf = self.bn_b._data.to_flat_list()
        if self.training:
            # per-feature mean/var across N
            for f in range(F):
                col = [xf[i*F+f] for i in range(N)]
                mu = sum(col) / N
                va = sum((v-mu)**2 for v in col) / N
                self._rm[f] = 0.9 * self._rm[f] + 0.1 * mu
                self._rv[f] = 0.9 * self._rv[f] + 0.1 * va
            m_list = [sum(xf[i*F+f] for i in range(N))/N for f in range(F)]
            v_list = [sum((xf[i*F+f] - m_list[f])**2 for i in range(N))/N for f in range(F)]
        else:
            m_list = self._rm[:]
            v_list = self._rv[:]
        result = []
        for i in range(N):
            for f in range(F):
                inv = 1.0 / _math.sqrt(v_list[f] + 1e-5)
                result.append((xf[i*F+f] - m_list[f]) * inv * wf[f] + bf[f])
        return _nd_arr(result, dtype='float32').reshape((N, F))

    def forward(self, x: Tensor, adj: NDArray) -> Tensor:
        h = x._data
        N, F = h.shape
        af = adj.to_flat_list()
        hf = h.to_flat_list()
        # Sum aggregation (no self-loop — ε handles self)
        agg_flat = []
        for i in range(N):
            for f in range(F):
                s = sum(af[i*N+j] * hf[j*F+f] for j in range(N) if j != i)
                agg_flat.append(s)
        agg = _nd_arr(agg_flat, dtype='float32').reshape((N, F))
        eps_v = self.epsilon._data.to_flat_list()[0]
        h_new = h + h * eps_v + agg
        # MLP
        h_new = _relu(self._bn(h_new @ self.w1._data + self.b1._data))
        h_new = h_new @ self.w2._data + self.b2._data
        return Tensor(h_new)


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
        self.mlp_w1 = Tensor(_nd_randn((hidden_dim * num_layers, hidden_dim), 0.02))
        self.mlp_b1 = Tensor(_nd_zeros((hidden_dim,)))
        self.mlp_w2 = Tensor(_nd_randn((hidden_dim, out_features), 0.02))
        self.mlp_b2 = Tensor(_nd_zeros((out_features,)))
        for n, p in [('mlp_w1',self.mlp_w1),('mlp_b1',self.mlp_b1),
                     ('mlp_w2',self.mlp_w2),('mlp_b2',self.mlp_b2)]:
            p.requires_grad = True; self._parameters[n] = p

    def _pool(self, h: NDArray) -> NDArray:
        N, F = h.shape
        hf = h.to_flat_list()
        if self.readout == 'sum':
            row = [sum(hf[i*F+f] for i in range(N)) for f in range(F)]
        elif self.readout == 'max':
            row = [max(hf[i*F+f] for i in range(N)) for f in range(F)]
        else:
            row = [sum(hf[i*F+f] for i in range(N)) / N for f in range(F)]
        return _nd_arr(row, dtype='float32').reshape((1, F))

    def forward(self, x: Tensor, adj: NDArray, batch=None) -> Tensor:
        d = x._data
        layer_reprs = []
        for layer in self.layers:
            d = layer(Tensor(d), adj)._data
            if self.dropout > 0 and self.training:
                df = d.to_flat_list()
                df = [v * (1.0 if _random.random() < (1-self.dropout) else 0.0) / (1-self.dropout) for v in df]
                d = _nd_arr(df, dtype='float32').reshape(d.shape)
            layer_reprs.append(d)

        from vectorphile.core.tensor import _nd_concatenate as _gin_cat
        if batch is None:
            pooled = [self._pool(h_rep) for h_rep in layer_reprs]
            graph_repr = _gin_cat(pooled, axis=-1)
        else:
            # batch is a list of ints
            if isinstance(batch, NDArray):
                batch = [int(v) for v in batch.to_flat_list()]
            num_graphs = max(batch) + 1
            graph_reprs = []
            for g in range(num_graphs):
                node_ids = [i for i, b in enumerate(batch) if b == g]
                pooled_g = []
                for h_rep in layer_reprs:
                    hf = h_rep.to_flat_list()
                    N_tot, F = h_rep.shape
                    rows = [hf[i*F:(i+1)*F] for i in node_ids]
                    sub_flat = [v for row in rows for v in row]
                    sub = _nd_arr(sub_flat, dtype='float32').reshape((len(node_ids), F))
                    pooled_g.append(self._pool(sub))
                graph_reprs.append(_gin_cat(pooled_g, axis=-1))
            graph_repr = _gin_cat(graph_reprs, axis=0)

        out = _relu(graph_repr @ self.mlp_w1._data + self.mlp_b1._data)
        if self.dropout > 0 and self.training:
            df = out.to_flat_list()
            df = [v * (1.0 if _random.random() < (1-self.dropout) else 0.0) / (1-self.dropout) for v in df]
            out = _nd_arr(df, dtype='float32').reshape(out.shape)
        return Tensor(out @ self.mlp_w2._data + self.mlp_b2._data)
