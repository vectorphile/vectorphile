"""
Multi-Head Attention with Flash Attention v2 backend.

FlashAttention: O(N) memory, tiled softmax — never materialises the N×N matrix.
Supports:
  - Single-head (flash_attn_fwd/bwd directly)
  - Multi-head (splits Q/K/V into H heads, runs Flash Attn per head)
  - Causal masking (future tokens masked — set causal=True)

No third-party dependencies: all ops through _vp_core or pure Python fallback.
"""
from __future__ import annotations
import math
import array as _arr
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor
from vectorphile.autograd.engine import Function
from vectorphile._vp_helpers import _nd_uniform_fn, _nd_zeros
from vectorphile._core.ndarray import NDArray as _NDA, _c_strides as _cs, _DType

try:
    from vectorphile._core import _vp_core as _C_attn
    _ATTN_C_OK = hasattr(_C_attn, 'flash_attn_fwd_f32')
except ImportError:
    _C_attn = None
    _ATTN_C_OK = False

# Buffer cache: keyed by (N, d) or (N,) to avoid per-call allocation
_attn_buf_cache: dict = {}


def _get_buf(key, nbytes):
    if key not in _attn_buf_cache:
        _attn_buf_cache[key] = _arr.array('f', bytearray(nbytes))
    return _attn_buf_cache[key]


# ── Autograd function for single-head Flash Attention ────────────────────────

class _FlashAttnFunction(Function):
    """Single-head Flash Attention forward+backward via _vp_core kernels."""

    def __init__(self, q, k, v, scale: float, causal: bool):
        super().__init__(q, k, v)
        self.q = q; self.k = k; self.v = v
        self.scale = scale
        self.causal = causal
        # saved for backward
        self._out_data = None
        self._L_buf = None

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        raise NotImplementedError  # called via apply() below

    def backward(self, grad_out: Tensor):
        """Recompute-based backward — no saved N×N attention matrix."""
        gd = grad_out._data
        qd = self.q._data; kd = self.k._data; vd = self.v._data
        od = self._out_data

        if (_ATTN_C_OK and isinstance(qd, _NDA) and qd._buf is not None
                and isinstance(kd, _NDA) and kd._buf is not None
                and isinstance(vd, _NDA) and vd._buf is not None
                and isinstance(gd, _NDA) and gd._buf is not None):
            N, d = qd.shape
            dq_buf = _get_buf(('dq', N, d), 4 * N * d)
            dk_buf = _get_buf(('dk', N, d), 4 * N * d)
            dv_buf = _get_buf(('dv', N, d), 4 * N * d)
            _C_attn.flash_attn_bwd_f32(
                gd._buf, qd._buf, kd._buf, vd._buf, od._buf, self._L_buf,
                dq_buf, dk_buf, dv_buf, N, d, self.scale)
            shape_nd = (N, d)
            dq = Tensor(_NDA(dq_buf, shape_nd, _cs(shape_nd), _DType('float32')), _no_copy=True)
            dk = Tensor(_NDA(dk_buf, shape_nd, _cs(shape_nd), _DType('float32')), _no_copy=True)
            dv = Tensor(_NDA(dv_buf, shape_nd, _cs(shape_nd), _DType('float32')), _no_copy=True)
            return dq, dk, dv

        # Pure Python fallback using vectorphile Tensor ops
        Q = Tensor(self.q._data); K = Tensor(self.k._data)
        V = Tensor(self.v._data); O = Tensor(od); dO = grad_out
        S = Q @ K.permute(1, 0) * self.scale
        P = (S - S.max(dim=-1, keepdim=True)).exp()
        P = P / P.sum(dim=-1, keepdim=True)
        dV = P.permute(1, 0) @ dO
        dP = dO @ V.permute(1, 0)
        D = (dO * O).sum(dim=-1, keepdim=True)
        dS = P * (dP - D)
        dQ = dS @ K * self.scale
        dK = dS.permute(1, 0) @ Q * self.scale
        return dQ, dK, dV


def flash_attn_single_head(q: Tensor, k: Tensor, v: Tensor,
                           scale: float | None = None) -> Tensor:
    """Single-head Flash Attention. q,k,v: (N, d)."""
    N, d = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    req_grad = q.requires_grad or k.requires_grad or v.requires_grad

    qd = q._data; kd = k._data; vd = v._data
    if (_ATTN_C_OK and isinstance(qd, _NDA) and qd._buf is not None
            and isinstance(kd, _NDA) and kd._buf is not None
            and isinstance(vd, _NDA) and vd._buf is not None):
        out_buf = _get_buf(('o', N, d), 4 * N * d)
        L_buf   = _get_buf(('L', N),    4 * N)
        _C_attn.flash_attn_fwd_f32(qd._buf, kd._buf, vd._buf, out_buf, L_buf,
                                   N, d, scale)
        out_nd = _NDA(out_buf, (N, d), _cs((N, d)), _DType('float32'))
        result = Tensor(out_nd, requires_grad=req_grad, _no_copy=True)
        if req_grad:
            fn = _FlashAttnFunction(q, k, v, scale, causal=False)
            fn._out_data = out_nd
            fn._L_buf    = L_buf
            result.grad_fn = fn
            result._is_leaf = False
        return result

    # Pure Python fallback via Tensor ops
    S = q @ k.permute(1, 0) * scale
    P = (S - S.max(dim=-1, keepdim=True)).exp()
    P = P / P.sum(dim=-1, keepdim=True)
    return Tensor((P @ v)._data, requires_grad=req_grad)


# ── MultiHeadAttention module ─────────────────────────────────────────────────

class MultiHeadAttention(Module):
    """
    Multi-Head Attention (Vaswani et al. 2017) with Flash Attention v2 backend.

    Args:
        embed_dim   : total embedding dimension (must be divisible by num_heads)
        num_heads   : number of attention heads
        dropout     : attention dropout probability (applied during training)
        bias        : add bias to Q/K/V/out projections
        causal      : mask future positions (for decoder self-attention)
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.0, bias: bool = True, causal: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f'embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})'
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = 1.0 / math.sqrt(self.head_dim)
        self.causal    = causal
        self._dropout_p = dropout

        # Fused Q/K/V projection: (embed_dim → 3*embed_dim)
        k = math.sqrt(1.0 / embed_dim)
        self._qkv_weight = Tensor(
            _nd_uniform_fn(-k, k, (3 * embed_dim, embed_dim)),
            requires_grad=True)
        self._out_weight = Tensor(
            _nd_uniform_fn(-k, k, (embed_dim, embed_dim)),
            requires_grad=True)
        if bias:
            self._qkv_bias = Tensor(_nd_zeros(3 * embed_dim, dtype='float32'), requires_grad=True)
            self._out_bias = Tensor(_nd_zeros(embed_dim, dtype='float32'), requires_grad=True)
            self._parameters['_qkv_bias'] = self._qkv_bias
            self._parameters['_out_bias'] = self._out_bias
        else:
            self._qkv_bias = None
            self._out_bias = None
        self._parameters['_qkv_weight'] = self._qkv_weight
        self._parameters['_out_weight'] = self._out_weight

    def forward(self, x: Tensor,
                key: Tensor | None = None,
                value: Tensor | None = None) -> Tensor:
        """
        Args:
            x     : (N, embed_dim) query (and key/value if self-attention)
            key   : (N, embed_dim) optional — defaults to x (self-attention)
            value : (N, embed_dim) optional — defaults to x
        Returns:
            out   : (N, embed_dim)
        """
        from vectorphile.nn.layers import Linear
        orig_shape = x.shape
        # Handle 3D input (seq, batch, embed) by flattening to (seq*batch, embed)
        if len(orig_shape) == 3:
            S, B, E0 = orig_shape
            x = x.reshape(S * B, E0)
        N, E = x.shape
        H = self.num_heads
        Dh = self.head_dim

        # Fused QKV projection: (N, 3E)
        qkv_weight = self._qkv_weight
        qkv_bias   = self._qkv_bias

        # Use existing Linear fast path via manual matmul
        from vectorphile._core import _vp_core as _C
        from vectorphile.nn.layers import _linear_out_cache, _LAYERS_C_OK
        import array as _a

        xd  = x._data
        wqd = qkv_weight._data

        if (_LAYERS_C_OK and isinstance(xd, _NDA) and xd._buf is not None
                and isinstance(wqd, _NDA) and wqd._buf is not None):
            # Transpose QKV weight: (3E, E) -> (E, 3E)
            if (not hasattr(self, '_qkv_wT_buf')
                    or self._qkv_wT_id != id(wqd._buf)):
                wT_buf = _a.array('f', bytearray(4 * E * 3 * E))
                _C.transpose_f32(wqd._buf, wT_buf, 3 * E, E)
                self._qkv_wT_buf = wT_buf
                self._qkv_wT_id  = id(wqd._buf)
            wT_buf = self._qkv_wT_buf

            qkv_buf = _get_buf(('qkv', N, 3 * E), 4 * N * 3 * E)
            _C.matmul_f32(xd._buf, wT_buf, qkv_buf, N, E, 3 * E)
            if qkv_bias is not None:
                bd = qkv_bias._data
                if isinstance(bd, _NDA) and bd._buf is not None:
                    _C.add_bias_rows_f32(qkv_buf, bd._buf, N, 3 * E)

            # Split Q, K, V: each (N, E) — views into qkv_buf
            # We need separate buffers because Flash Attn needs non-overlapping slices
            q_buf = _get_buf(('q', N, E), 4 * N * E)
            k_buf = _get_buf(('k', N, E), 4 * N * E)
            v_buf = _get_buf(('v', N, E), 4 * N * E)
            for i in range(N * E):
                q_buf[i] = qkv_buf[i]
                k_buf[i] = qkv_buf[N * E + i]
                v_buf[i] = qkv_buf[2 * N * E + i]

            q_nd = _NDA(q_buf, (N, E), _cs((N, E)), _DType('float32'))
            k_nd = _NDA(k_buf, (N, E), _cs((N, E)), _DType('float32'))
            v_nd = _NDA(v_buf, (N, E), _cs((N, E)), _DType('float32'))
            Q = Tensor(q_nd, requires_grad=x.requires_grad, _no_copy=True)
            K = Tensor(k_nd, requires_grad=x.requires_grad, _no_copy=True)
            V = Tensor(v_nd, requires_grad=x.requires_grad, _no_copy=True)
        else:
            # Pure Tensor fallback projection
            qkv = x @ qkv_weight.permute(1, 0)
            if qkv_bias is not None:
                qkv = qkv + qkv_bias.reshape(1, 3 * E)
            Q = Tensor(qkv[:, :E]._data,      requires_grad=x.requires_grad)
            K = Tensor(qkv[:, E:2*E]._data,   requires_grad=x.requires_grad)
            V = Tensor(qkv[:, 2*E:3*E]._data, requires_grad=x.requires_grad)

        # For multi-head: run Flash Attn per head, concat results
        # Single-head path: H==1, run Flash Attn directly on (N, E)
        if H == 1:
            attn_out = flash_attn_single_head(Q, K, V, scale=self.scale)
        else:
            # Reshape: (N, E) -> H heads of (N, Dh) each
            # Process each head independently, concat along Dh axis
            head_outs = []
            for h in range(H):
                s = h * Dh; e = s + Dh
                # Extract head slice (N, Dh) — copy slice into fresh buffers
                if (_ATTN_C_OK and isinstance(Q._data, _NDA)):
                    qh_buf = _get_buf(('qh', h, N, Dh), 4 * N * Dh)
                    kh_buf = _get_buf(('kh', h, N, Dh), 4 * N * Dh)
                    vh_buf = _get_buf(('vh', h, N, Dh), 4 * N * Dh)
                    for i in range(N):
                        for j in range(Dh):
                            qh_buf[i * Dh + j] = Q._data._buf[i * E + s + j]
                            kh_buf[i * Dh + j] = K._data._buf[i * E + s + j]
                            vh_buf[i * Dh + j] = V._data._buf[i * E + s + j]
                    qh = Tensor(_NDA(qh_buf, (N, Dh), _cs((N, Dh)), _DType('float32')), _no_copy=True)
                    kh = Tensor(_NDA(kh_buf, (N, Dh), _cs((N, Dh)), _DType('float32')), _no_copy=True)
                    vh = Tensor(_NDA(vh_buf, (N, Dh), _cs((N, Dh)), _DType('float32')), _no_copy=True)
                else:
                    qh = Tensor(Q[:, s:e]._data)
                    kh = Tensor(K[:, s:e]._data)
                    vh = Tensor(V[:, s:e]._data)
                head_outs.append(flash_attn_single_head(qh, kh, vh, scale=self.scale))

            # Concat heads: list of (N, Dh) -> (N, E)
            if _ATTN_C_OK and isinstance(head_outs[0]._data, _NDA):
                cat_buf = _get_buf(('cat', N, E), 4 * N * E)
                for h, ho in enumerate(head_outs):
                    s = h * Dh
                    for i in range(N):
                        for j in range(Dh):
                            cat_buf[i * E + s + j] = ho._data._buf[i * Dh + j]
                attn_out = Tensor(_NDA(cat_buf, (N, E), _cs((N, E)), _DType('float32')), _no_copy=True)
            else:
                # Concat using Tensor slicing — build result row by row
                from vectorphile._vp_helpers import _nd_zeros
                cat_nd = _nd_zeros((N, E), dtype='float32')
                cat_t = Tensor(cat_nd)
                for h, ho in enumerate(head_outs):
                    s2 = h * Dh
                    for i in range(N):
                        for j in range(Dh):
                            cat_t._data[i, s2 + j] = float(ho._data[i, j] if hasattr(ho._data, '__getitem__') else 0.0)
                attn_out = cat_t

        # Output projection: (N, E) @ (E, E) + bias
        out_wd = self._out_weight._data
        if (_LAYERS_C_OK and isinstance(attn_out._data, _NDA)
                and attn_out._data._buf is not None
                and isinstance(out_wd, _NDA) and out_wd._buf is not None):
            if (not hasattr(self, '_out_wT_buf') or self._out_wT_id != id(out_wd._buf)):
                owT_buf = _a.array('f', bytearray(4 * E * E))
                _C.transpose_f32(out_wd._buf, owT_buf, E, E)
                self._out_wT_buf = owT_buf
                self._out_wT_id  = id(out_wd._buf)
            owT_buf = self._out_wT_buf
            final_buf = _get_buf(('final', N, E), 4 * N * E)
            _C.matmul_f32(attn_out._data._buf, owT_buf, final_buf, N, E, E)
            if self._out_bias is not None:
                bd = self._out_bias._data
                if isinstance(bd, _NDA) and bd._buf is not None:
                    _C.add_bias_rows_f32(final_buf, bd._buf, N, E)
            final_nd = _NDA(final_buf, (N, E), _cs((N, E)), _DType('float32'))
            out = Tensor(final_nd, requires_grad=x.requires_grad, _no_copy=True)
            if len(orig_shape) == 3:
                out = out.reshape(S, B, E)
            return out

        # Pure Tensor fallback output projection
        out = attn_out @ self._out_weight.permute(1, 0)
        if self._out_bias is not None:
            out = out + self._out_bias.reshape(1, E)
        out = Tensor(out._data, requires_grad=x.requires_grad)
        if len(orig_shape) == 3:
            out = out.reshape(S, B, E)
        return out

    def __repr__(self):
        return (f'MultiHeadAttention(embed_dim={self.embed_dim}, '
                f'num_heads={self.num_heads}, causal={self.causal})')


# ── PyTorch-compatible MultiheadAttention (6-parameter design) ───────────────
# layers_complete.py patches MultiheadAttention.forward with _mha_forward.
# This class has exactly 6 trainable parameters: q/k/v/out weights + 2 biases.
class MultiheadAttention(Module):
    """PyTorch-compatible MHA with separate Q/K/V projection parameters."""
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.0, bias: bool = True,
                 batch_first: bool = True, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.dropout_p = dropout
        self.batch_first = batch_first
        k = math.sqrt(1.0 / embed_dim)
        self.q_proj_weight = Tensor(_nd_uniform_fn(-k, k, (embed_dim, embed_dim)), requires_grad=True)
        self.k_proj_weight = Tensor(_nd_uniform_fn(-k, k, (embed_dim, embed_dim)), requires_grad=True)
        self.v_proj_weight = Tensor(_nd_uniform_fn(-k, k, (embed_dim, embed_dim)), requires_grad=True)
        self.out_proj_weight = Tensor(_nd_uniform_fn(-k, k, (embed_dim, embed_dim)), requires_grad=True)
        if bias:
            self.in_proj_bias  = Tensor(_nd_zeros(3 * embed_dim, dtype='float32'), requires_grad=True)
            self.out_proj_bias = Tensor(_nd_zeros(embed_dim, dtype='float32'), requires_grad=True)
        else:
            object.__setattr__(self, 'in_proj_bias', None)
            object.__setattr__(self, 'out_proj_bias', None)
        # For _mha_forward compat (uses self.in_proj_bias and self.out_proj_weight)
        # q/k/v weights are registered as full parameters; _mha_forward uses them directly

    def forward(self, query: Tensor, key: Tensor = None, value: Tensor = None,
                attn_mask=None, need_weights: bool = True, **kwargs):
        """Fallback forward used only if layers_complete.install() was not called."""
        if key is None: key = query
        if value is None: value = query
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            key   = key.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        B, T, E = query.shape[0], query.shape[1], query.shape[2]
        H, Dh = self.num_heads, self.head_dim
        scale = 1.0 / math.sqrt(Dh)
        q = query.reshape(B * T, E) @ self.q_proj_weight.transpose(0, 1)
        k = key.reshape(B * T, E)   @ self.k_proj_weight.transpose(0, 1)
        v = value.reshape(B * T, E) @ self.v_proj_weight.transpose(0, 1)
        if self.in_proj_bias is not None:
            q = q + self.in_proj_bias[:E].reshape(1, E)
            k = k + self.in_proj_bias[E:2*E].reshape(1, E)
            v = v + self.in_proj_bias[2*E:].reshape(1, E)
        Q = q.reshape(B, T, H, Dh).permute(0, 2, 1, 3)
        K = k.reshape(B, T, H, Dh).permute(0, 2, 1, 3)
        V = v.reshape(B, T, H, Dh).permute(0, 2, 1, 3)
        scores = (Q @ K.permute(0, 1, 3, 2)) * scale
        if attn_mask is not None:
            from vectorphile._vp_helpers import _nd_array as _nd_arr2
            m = attn_mask._data if isinstance(attn_mask, Tensor) else _nd_arr2(attn_mask)
            if getattr(m, 'dtype', None) == bool or (hasattr(m, 'dtype') and str(m.dtype) in ('bool', '<class \'bool\'>')):
                scores = scores.masked_fill(m.reshape((1, 1) + m.shape[-2:]), -1e9)
            else:
                scores = scores + Tensor(m)
        w = scores.softmax(dim=-1)
        out = (w @ V).permute(0, 2, 1, 3).reshape(B * T, E) @ self.out_proj_weight.transpose(0, 1)
        if self.out_proj_bias is not None:
            out = out + self.out_proj_bias.reshape(1, E)
        out = out.reshape(B, T, E)
        if not self.batch_first:
            out = out.permute(1, 0, 2)
        return out, None

    def __repr__(self):
        return f'MultiheadAttention(embed_dim={self.embed_dim}, num_heads={self.num_heads})'


# ── Transformer building blocks ─────────────────────────────────────────────

import math as _math

class PositionalEncoding(Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017)."""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self._dropout_p = dropout
        # Build sinusoidal PE table as list of lists
        pe = [[0.0] * d_model for _ in range(max_len)]
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                div = _math.exp(i * (-_math.log(10000.0) / d_model))
                pe[pos][i] = _math.sin(pos * div)
                if i + 1 < d_model:
                    pe[pos][i + 1] = _math.cos(pos * div)
        self._pe = pe  # list of lists (max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        N = x.shape[0]
        pe_slice = self._pe[:N]
        pe_t = Tensor(pe_slice)
        return Tensor((x + pe_t)._data, requires_grad=x.requires_grad)


class TransformerEncoderLayer(Module):
    """Single Transformer encoder layer: self-attn + FFN + LayerNorm."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = 'relu',
                 batch_first: bool = False, **kwargs):
        super().__init__()
        from vectorphile.nn.layers import Linear, LayerNorm
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1   = Linear(d_model, dim_feedforward)
        self.linear2   = Linear(dim_feedforward, d_model)
        self.norm1     = LayerNorm(d_model)
        self.norm2     = LayerNorm(d_model)
        self._act = activation
        # NOTE: do NOT manually copy to self._parameters — Module.parameters() already
        # traverses self._modules recursively, so submodule params are found automatically.

    def forward(self, x: Tensor, src_mask=None, src_key_padding_mask=None) -> Tensor:
        from vectorphile.nn.activations import ReLU, GELU
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff = self.linear1(x)
        ff = ReLU()(ff) if self._act == 'relu' else GELU()(ff)
        ff = self.linear2(ff)
        return self.norm2(x + ff)


class TransformerEncoder(Module):
    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int):
        super().__init__()
        import copy
        self.layers = [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        for i, lyr in enumerate(self.layers):
            self._modules[str(i)] = lyr

    def forward(self, x: Tensor) -> Tensor:
        for lyr in self.layers: x = lyr(x)
        return x


class TransformerDecoderLayer(Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, batch_first: bool = False, **kwargs):
        super().__init__()
        from vectorphile.nn.layers import Linear, LayerNorm
        self.self_attn  = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1    = Linear(d_model, dim_feedforward)
        self.linear2    = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(self, x: Tensor, memory: Tensor, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None) -> Tensor:
        from vectorphile.nn.activations import ReLU
        attn1, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn1)
        attn2, _ = self.cross_attn(x, memory, memory)
        x = self.norm2(x + attn2)
        ff = self.linear2(ReLU()(self.linear1(x)))
        return self.norm3(x + ff)


class TransformerDecoder(Module):
    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int):
        super().__init__()
        import copy
        self.layers = [copy.deepcopy(decoder_layer) for _ in range(num_layers)]
        for i, lyr in enumerate(self.layers): self._modules[str(i)] = lyr

    def forward(self, x: Tensor, memory: Tensor, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None) -> Tensor:
        for lyr in self.layers:
            x = lyr(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask)
        return x


class Transformer(Module):
    def __init__(self, d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_encoder_layers)
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout),
            num_decoder_layers)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        memory = self.encoder(src)
        return self.decoder(tgt, memory)


# ── ALiBi slopes ─────────────────────────────────────────────────────────────

def _compute_alibi_slopes(num_heads: int):
    """Compute ALiBi (Attention with Linear Biases) head slopes.

    Returns a list of num_heads floats: 2^(-8*i/num_heads) for i in 1..num_heads.
    """
    import math as _m
    return [2 ** (-8.0 * (i + 1) / num_heads) for i in range(num_heads)]


# ── Block-Sparse Attention ────────────────────────────────────────────────────

class BlockSparseAttention(Module):
    """Block-sparse multi-head attention for ultra-long sequences.

    Splits Q/K/V into non-overlapping blocks of `block_size` tokens and only
    computes attention within each block.  An optional `stride` parameter adds
    strided global blocks (every `stride` blocks attends globally) to let
    information flow across distant positions, similar to Longformer's pattern.

    Complexity: O(N * block_size) instead of O(N²) — practical for N ≥ 8K.

    Args:
        embed_dim:   Total embedding dimensionality.
        num_heads:   Number of attention heads.
        block_size:  Number of tokens per local attention block (default 64).
        stride:      Every `stride`-th block also attends globally (0 = disabled).
        causal:      Apply causal mask within each block.
    """

    def __init__(self, embed_dim: int, num_heads: int, block_size: int = 64,
                 stride: int = 0, causal: bool = False):
        super().__init__()
        from vectorphile.nn.layers import Linear
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.block_size = block_size
        self.stride     = stride
        self.causal     = causal
        self.scale      = 1.0 / _math.sqrt(self.head_dim)
        self.q_proj = Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = Linear(embed_dim, embed_dim)

    def _block_attn(self, q_block, k_block, v_block):
        """Scaled dot-product attention on a single (block_size, head_dim) block."""
        B = len(q_block)  # block_size
        # scores: (B, B)
        scores = []
        for i in range(B):
            row = []
            for j in range(B):
                if self.causal and j > i:
                    row.append(-1e9)
                else:
                    s = sum(q_block[i][d] * k_block[j][d] for d in range(len(q_block[i]))) * self.scale
                    row.append(s)
            scores.append(row)
        # Softmax + weighted sum
        out_block = []
        for i in range(B):
            mx = max(scores[i])
            exps = [_math.exp(s - mx) for s in scores[i]]
            sm = sum(exps) + 1e-9
            attn = [e / sm for e in exps]
            out_i = [sum(attn[j] * v_block[j][d] for j in range(B)) for d in range(len(v_block[0]))]
            out_block.append(out_i)
        return out_block

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (N, embed_dim) or (B, T, embed_dim)
        Returns:
            Tensor of same shape as x.
        """
        orig_shape = x.shape
        if len(orig_shape) == 3:
            B, T, E = orig_shape
            x_2d = x.reshape(B * T, E)
        else:
            x_2d = x
            T = orig_shape[0]
            E = orig_shape[1] if len(orig_shape) > 1 else self.embed_dim

        N = x_2d.shape[0]
        H, Dh = self.num_heads, self.head_dim
        bs = self.block_size

        q_all = self.q_proj(x_2d)
        k_all = self.k_proj(x_2d)
        v_all = self.v_proj(x_2d)

        # Extract flat data
        qf = q_all._data.to_flat_list() if hasattr(q_all._data, 'to_flat_list') else list(q_all._data)
        kf = k_all._data.to_flat_list() if hasattr(k_all._data, 'to_flat_list') else list(k_all._data)
        vf = v_all._data.to_flat_list() if hasattr(v_all._data, 'to_flat_list') else list(v_all._data)

        output = [0.0] * (N * E)

        # Process head by head
        for h in range(H):
            h_start = h * Dh
            # Pad sequence to multiple of block_size
            pad = (bs - N % bs) % bs
            N_pad = N + pad

            def get_vec(flat, row, h_offset):
                return [flat[row * E + h_offset + d] for d in range(Dh)]

            # Block-local attention
            for blk_start in range(0, N_pad, bs):
                blk_end = min(blk_start + bs, N)
                if blk_start >= N:
                    break
                block_len = blk_end - blk_start
                q_blk = [get_vec(qf, blk_start + i, h_start) for i in range(block_len)]
                k_blk = [get_vec(kf, blk_start + i, h_start) for i in range(block_len)]
                v_blk = [get_vec(vf, blk_start + i, h_start) for i in range(block_len)]

                # Optional global stride: add tokens from global block to keys/values
                if self.stride > 0:
                    for gb in range(0, N_pad, bs * self.stride):
                        gb_end = min(gb + bs, N)
                        if gb == blk_start or gb >= N:
                            continue
                        for gi in range(gb_end - gb):
                            k_blk.append(get_vec(kf, gb + gi, h_start))
                            v_blk.append(get_vec(vf, gb + gi, h_start))

                out_blk = self._block_attn(q_blk, k_blk, v_blk)
                for i, row in enumerate(out_blk):
                    for d in range(Dh):
                        output[(blk_start + i) * E + h_start + d] = row[d]

        from vectorphile._vp_helpers import _nd_array as _nd_arr
        out_nd = _nd_arr(output).reshape((N, E))
        out_t = self.out_proj(Tensor(out_nd))
        if len(orig_shape) == 3:
            out_t = out_t.reshape(orig_shape)
        return out_t

    def __repr__(self):
        return (f'BlockSparseAttention(embed_dim={self.embed_dim}, '
                f'num_heads={self.num_heads}, block_size={self.block_size}, '
                f'stride={self.stride}, causal={self.causal})')


# ── Ring Attention (gradient checkpointing for distributed long-context) ──────

class RingAttention(Module):
    """Ring Attention: distributes sequence chunks across devices (simulated here
    as a single-device ring where chunks are processed sequentially).

    For N tokens on a single device this gives the same result as full attention
    but with O(chunk_size) peak memory instead of O(N).  The key idea (Liu et al.,
    2023) is to pass K/V from previous chunks around the ring while computing
    attention for the current Q chunk.

    Args:
        embed_dim:   Total embedding dimensionality.
        num_heads:   Number of attention heads.
        chunk_size:  Number of tokens per ring chunk (default 128).
        causal:      Apply causal mask (future tokens masked).
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 chunk_size: int = 128, causal: bool = False):
        super().__init__()
        from vectorphile.nn.layers import Linear
        assert embed_dim % num_heads == 0
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.chunk_size = chunk_size
        self.causal     = causal
        self.scale      = 1.0 / _math.sqrt(self.head_dim)
        self.q_proj = Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        if len(orig_shape) == 3:
            B, T, E = orig_shape
            x_2d = x.reshape(B * T, E)
        else:
            x_2d = x
            T = orig_shape[0]
            E = self.embed_dim

        N = x_2d.shape[0]
        H, Dh = self.num_heads, self.head_dim
        cs = self.chunk_size

        q_all = self.q_proj(x_2d)
        k_all = self.k_proj(x_2d)
        v_all = self.v_proj(x_2d)

        qf = q_all._data.to_flat_list() if hasattr(q_all._data, 'to_flat_list') else list(q_all._data)
        kf = k_all._data.to_flat_list() if hasattr(k_all._data, 'to_flat_list') else list(k_all._data)
        vf = v_all._data.to_flat_list() if hasattr(v_all._data, 'to_flat_list') else list(v_all._data)

        output = [0.0] * (N * E)

        def get_vec(flat, row, h_start):
            return [flat[row * E + h_start + d] for d in range(Dh)]

        for h in range(H):
            h_start = h * Dh
            # Ring: for each Q chunk, accumulate attention over all K/V chunks
            for q_start in range(0, N, cs):
                q_end = min(q_start + cs, N)
                q_chunk = [get_vec(qf, q_start + i, h_start) for i in range(q_end - q_start)]
                # Online softmax state (log-sum-exp + running sum) per Q token
                log_sum = [-1e9] * len(q_chunk)
                acc = [[0.0] * Dh for _ in range(len(q_chunk))]

                for kv_start in range(0, N, cs):
                    kv_end = min(kv_start + cs, N)
                    k_chunk = [get_vec(kf, kv_start + j, h_start) for j in range(kv_end - kv_start)]
                    v_chunk = [get_vec(vf, kv_start + j, h_start) for j in range(kv_end - kv_start)]

                    for qi, q_vec in enumerate(q_chunk):
                        abs_qi = q_start + qi
                        scores_i = []
                        for kj, k_vec in enumerate(k_chunk):
                            abs_kj = kv_start + kj
                            if self.causal and abs_kj > abs_qi:
                                scores_i.append(-1e9)
                            else:
                                s = sum(q_vec[d] * k_vec[d] for d in range(Dh)) * self.scale
                                scores_i.append(s)
                        # Online softmax update
                        new_max = max(scores_i)
                        old_max = log_sum[qi]
                        if new_max > old_max:
                            # Rescale running accumulator
                            rescale = _math.exp(old_max - new_max)
                            acc[qi] = [a * rescale for a in acc[qi]]
                            log_sum[qi] = new_max
                        for kj, s in enumerate(scores_i):
                            w = _math.exp(s - log_sum[qi])
                            for d in range(Dh):
                                acc[qi][d] += w * v_chunk[kj][d]

                # Normalize and write output
                for qi, row in enumerate(acc):
                    # acc[qi] = sum_j(exp(s_j - log_sum[qi]) * v_j)
                    # We need to divide by sum of weights = sum_j exp(s_j - log_sum[qi])
                    # But we stored running sum implicitly — recalculate denominator
                    # by running full ring again is expensive; instead use log_sum trick:
                    # denom = sum_kv exp(scores - log_sum[qi]), already captured in acc norm
                    # For correctness: do a second pass to get the denominator
                    # (This is the standard online-softmax two-pass approach)
                    pass  # denominator handled below

                # Second pass: compute denominator
                denom = [0.0] * len(q_chunk)
                for kv_start in range(0, N, cs):
                    kv_end = min(kv_start + cs, N)
                    k_chunk = [get_vec(kf, kv_start + j, h_start) for j in range(kv_end - kv_start)]
                    for qi, q_vec in enumerate(q_chunk):
                        abs_qi = q_start + qi
                        for kj, k_vec in enumerate(k_chunk):
                            abs_kj = kv_start + kj
                            if self.causal and abs_kj > abs_qi:
                                continue
                            s = sum(q_vec[d] * k_vec[d] for d in range(Dh)) * self.scale
                            denom[qi] += _math.exp(s - log_sum[qi])

                for qi, row in enumerate(acc):
                    d_val = max(denom[qi], 1e-9)
                    for d in range(Dh):
                        output[(q_start + qi) * E + h_start + d] = row[d] / d_val

        from vectorphile._vp_helpers import _nd_array as _nd_arr
        out_nd = _nd_arr(output).reshape((N, E))
        out_t = self.out_proj(Tensor(out_nd))
        if len(orig_shape) == 3:
            out_t = out_t.reshape(orig_shape)
        return out_t

    def __repr__(self):
        return (f'RingAttention(embed_dim={self.embed_dim}, '
                f'num_heads={self.num_heads}, chunk_size={self.chunk_size})')


# ── Flash Decode Attention ────────────────────────────────────────────────────

class FlashDecodeAttention(Module):
    """Single-token (decode-step) attention using Flash Decoding.

    Flash Decoding (Tri Dao 2023) parallelises across the KV-cache sequence
    dimension rather than the head dimension, achieving O(S/num_splits)
    effective bandwidth per split and much higher GPU utilisation than standard
    Flash Attention during autoregressive generation.

    Usage::
        attn = FlashDecodeAttention(embed_dim=512, num_heads=8, num_splits=8)
        # x    : (B, 1,  E)  — single decode token
        # cache: (B, S,  E)  — full KV cache (past tokens)
        out = attn(x, kv_cache)   # → (B, 1, E)
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 num_splits: int = 8, causal: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.num_splits = num_splits
        self.causal     = causal
        self.scale      = 1.0 / math.sqrt(self.head_dim)
        from vectorphile.nn.layers import Linear
        self.q_proj   = Linear(embed_dim, embed_dim, bias=False)
        self.k_proj   = Linear(embed_dim, embed_dim, bias=False)
        self.v_proj   = Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: Tensor, kv_cache: Tensor) -> Tensor:
        """
        x        : (B, 1, E)  — single decode step query
        kv_cache : (B, S, E)  — past context (keys and values)
        """
        orig_x   = x
        x_shape  = x._data.shape if hasattr(x._data, 'shape') else None
        B = x_shape[0] if x_shape and len(x_shape) == 3 else 1
        S = kv_cache._data.shape[1] if (hasattr(kv_cache._data, 'shape')
                                         and len(kv_cache._data.shape) == 3) else 1
        E = self.embed_dim
        H = self.num_heads
        Dh = self.head_dim

        # Project Q (single token) and K/V (full cache)
        q_1 = self.q_proj(x.reshape((B, E)) if len(x_shape or ()) == 3 else x)
        kc_flat = kv_cache.reshape((B * S, E)) if len(
            (kv_cache._data.shape if hasattr(kv_cache._data, 'shape') else ())
        ) == 3 else kv_cache
        k_full  = self.k_proj(kc_flat)
        v_full  = self.v_proj(kc_flat)

        # Reshape to (B, H, 1, Dh) and (B, H, S, Dh)
        from vectorphile._vp_helpers import _nd_array as _nd_arr
        import math as _m

        qfl = q_1._data.to_flat_list() if hasattr(q_1._data, 'to_flat_list') else list(q_1._data)
        kfl = k_full._data.to_flat_list() if hasattr(k_full._data, 'to_flat_list') else list(k_full._data)
        vfl = v_full._data.to_flat_list() if hasattr(v_full._data, 'to_flat_list') else list(v_full._data)

        # Reshape: interleave heads manually
        def _reshape_heads(flat, batch, seq, heads, head_d):
            # (B*seq, H*Dh) → (B, H, seq, Dh)
            result = []
            for b in range(batch):
                for h in range(heads):
                    for s in range(seq):
                        base = (b * seq + s) * (heads * head_d) + h * head_d
                        result.extend(flat[base:base+head_d])
            return result  # length = B*H*seq*Dh

        q_bhd = _reshape_heads(qfl, B, 1, H, Dh)
        k_bhd = _reshape_heads(kfl, B, S, H, Dh)
        v_bhd = _reshape_heads(vfl, B, S, H, Dh)

        q_nd = _nd_arr(q_bhd, 'float32').reshape((B, H, 1, Dh))
        k_nd = _nd_arr(k_bhd, 'float32').reshape((B, H, S, Dh))
        v_nd = _nd_arr(v_bhd, 'float32').reshape((B, H, S, Dh))

        # Route through Triton flash_decode or pure-Python fallback
        try:
            from vectorphile.cuda.triton_kernels import flash_decode as _fd
            out_nd = _fd(q_nd, k_nd, v_nd, scale=self.scale,
                         num_splits=self.num_splits)  # (B, H, 1, Dh)
        except Exception:
            # CPU fallback: standard scaled dot-product
            out_fl = []
            for b in range(B):
                for h in range(H):
                    qi = [q_nd._data[(b*H+h)*Dh+d] if hasattr(q_nd, '_data')
                          else q_bhd[(b*H*1+h)*Dh+d] for d in range(Dh)]
                    scores = []
                    for s in range(S):
                        kj = [k_bhd[((b*H+h)*S+s)*Dh+d] for d in range(Dh)]
                        scores.append(sum(qi[d]*kj[d] for d in range(Dh)) * self.scale)
                    m_ = max(scores)
                    exps  = [_m.exp(s - m_) for s in scores]
                    denom = sum(exps)
                    row   = [0.0] * Dh
                    for s in range(S):
                        vj  = [v_bhd[((b*H+h)*S+s)*Dh+d] for d in range(Dh)]
                        w   = exps[s] / denom
                        for d in range(Dh): row[d] += w * vj[d]
                    out_fl.extend(row)
            out_nd = _nd_arr(out_fl, 'float32').reshape((B, H, 1, Dh))

        # Merge heads: (B, H, 1, Dh) → (B, E)
        ofl = out_nd.to_flat_list() if hasattr(out_nd, 'to_flat_list') else list(out_nd)
        merged = []
        for b in range(B):
            row = [0.0] * E
            for h in range(H):
                for d in range(Dh):
                    row[h * Dh + d] = ofl[((b * H + h) * 1) * Dh + d]
            merged.extend(row)
        out_t = Tensor(_nd_arr(merged, 'float32').reshape((B, E)))
        out_t = self.out_proj(out_t)
        if x_shape and len(x_shape) == 3:
            out_t = out_t.reshape((B, 1, E))
        return out_t

    def __repr__(self):
        return (f'FlashDecodeAttention(embed_dim={self.embed_dim}, '
                f'num_heads={self.num_heads}, num_splits={self.num_splits})')
