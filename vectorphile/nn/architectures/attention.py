"""Advanced attention mechanisms — RoPE, ALiBi, GQA, MQA, sliding-window, causal."""
from __future__ import annotations
import math as _math
import random as _random
from typing import Optional, Tuple
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor
from vectorphile._core.ndarray import array as _nd_arr, NDArray
import vectorphile.backend as _backend


def _nd_randn(shape, scale=1.0):
    n = 1
    for s in shape: n *= s
    flat = [_math.sqrt(-2*_math.log(max(_random.random(),1e-10)))*_math.cos(2*_math.pi*_random.random())*scale
            for _ in range(n)]
    return _nd_arr(flat, dtype='float32').reshape(shape)


# ── Positional Encoding helpers ───────────────────────────────────────────────

def rotate_half(x: NDArray) -> NDArray:
    """Rotate last dim by swapping and negating halves (RoPE helper)."""
    sh = x.shape
    half = sh[-1] // 2
    flat = x.to_flat_list()
    n_rows = 1
    for s in sh[:-1]: n_rows *= s
    d = sh[-1]
    result = []
    for r in range(n_rows):
        row = flat[r*d:(r+1)*d]
        result.extend([-v for v in row[half:]] + row[:half])
    return _nd_arr(result, dtype='float32').reshape(sh)


def apply_rotary_pos_emb(q: NDArray, k: NDArray,
                          cos: NDArray, sin: NDArray) -> Tuple[NDArray, NDArray]:
    """Apply RoPE to query and key tensors."""
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


class RotaryEmbedding(Module):
    """Rotary Position Embedding (RoPE) — Su et al. 2021."""

    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self._inv_freq = [1.0 / (base ** (i / dim)) for i in range(0, dim, 2)]
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        half = len(self._inv_freq)
        # freqs[t, i] = t * inv_freq[i]
        freqs = [[t * self._inv_freq[i] for i in range(half)] for t in range(seq_len)]
        # emb = concat(freqs, freqs) → (seq, dim)
        emb = [freqs[t] + freqs[t] for t in range(seq_len)]
        cos_flat = [_math.cos(v) for row in emb for v in row]
        sin_flat = [_math.sin(v) for row in emb for v in row]
        d = self.dim
        # shape (1, 1, seq, dim)
        self._cos = _nd_arr(cos_flat, dtype='float32').reshape((1, 1, seq_len, d))
        self._sin = _nd_arr(sin_flat, dtype='float32').reshape((1, 1, seq_len, d))

    def forward(self, q: Tensor, k: Tensor, seq_offset: int = 0) -> Tuple[Tensor, Tensor]:
        seq_len = q._data.shape[-2] + seq_offset
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        # Slice along axis 2: [seq_offset : seq_offset + T]
        T = q._data.shape[-2]
        d = self.dim
        cos_flat = self._cos.to_flat_list()
        sin_flat = self._sin.to_flat_list()
        # _cos shape (1,1,max_seq,d); take [seq_offset:seq_offset+T]
        cos_slice = cos_flat[seq_offset*d:(seq_offset+T)*d]
        sin_slice = sin_flat[seq_offset*d:(seq_offset+T)*d]
        cos = _nd_arr(cos_slice, dtype='float32').reshape((1, 1, T, d))
        sin = _nd_arr(sin_slice, dtype='float32').reshape((1, 1, T, d))
        # Triton fused RoPE fast path (inference only, no grad)
        if not q.requires_grad and not k.requires_grad:
            try:
                from vectorphile.cuda.triton_kernels import fused_rope as _fr
                q_nd = _nd_arr(list(q._data.to_flat_list() if hasattr(q._data, 'to_flat_list') else q._data),
                               dtype='float32').reshape(q._data.shape)
                k_nd = _nd_arr(list(k._data.to_flat_list() if hasattr(k._data, 'to_flat_list') else k._data),
                               dtype='float32').reshape(k._data.shape)
                q_rope = _fr(q_nd, cos, sin)
                k_rope = _fr(k_nd, cos, sin)
                return Tensor(q_rope), Tensor(k_rope)
            except Exception:
                pass
        q_rot, k_rot = apply_rotary_pos_emb(q._data, k._data, cos, sin)
        return Tensor(q_rot), Tensor(k_rot)


class ALiBiPositionBias(Module):
    """ALiBi (Attention with Linear Biases) — Press et al. 2021."""

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self._slopes = [2.0 ** (-8.0 * (i+1) / num_heads) for i in range(num_heads)]

    def get_bias(self, seq_len: int):
        """Return (num_heads, seq_len, seq_len) bias as NDArray."""
        result = []
        for h in range(self.num_heads):
            for i in range(seq_len):
                for j in range(seq_len):
                    result.append(-self._slopes[h] * abs(i - j))
        return _nd_arr(result, dtype='float32').reshape((self.num_heads, seq_len, seq_len))


# ── Core attention functions ──────────────────────────────────────────────────

def scaled_dot_product_attention(q, k, v, mask=None, dropout_p: float = 0.0,
                                  training: bool = False):
    """Scaled dot-product attention — Triton Flash Attention v2 fast path, then backend."""
    # Flash Attention v2: use when no mask, no dropout, ndim==4 (B,H,N,D)
    if mask is None and dropout_p == 0.0 and not training:
        try:
            qsh = q.shape if hasattr(q, 'shape') else None
            if qsh and len(qsh) == 4:
                from vectorphile.cuda.triton_kernels import flash_attention_v2 as _fa2
                out_nd = _fa2(q, k, v)
                # Return (out_nd, None) to match (output, weights) contract
                return out_nd, None
        except Exception:
            pass
    return _backend.attention(q, k, v, mask=mask, dropout_p=dropout_p, training=training)


def _softmax_nd(x: NDArray, axis: int = -1) -> NDArray:
    flat = x.to_flat_list()
    sh = list(x.shape)
    ax = axis % len(sh)
    ax_size = sh[ax]
    n_outer = 1
    for s in sh[:ax]: n_outer *= s
    n_inner = 1
    for s in sh[ax+1:]: n_inner *= s
    result = flat[:]
    for outer in range(n_outer):
        for inner in range(n_inner):
            row = [flat[outer*ax_size*n_inner + k*n_inner + inner] for k in range(ax_size)]
            mx = max(row)
            exps = [_math.exp(v - mx) for v in row]
            s = sum(exps) + 1e-9
            for k in range(ax_size):
                result[outer*ax_size*n_inner + k*n_inner + inner] = exps[k] / s
    return _nd_arr(result, dtype='float32').reshape(tuple(sh))


def causal_mask(seq_len: int) -> NDArray:
    """Upper-triangular causal mask — True (1.0) means masked."""
    flat = [1.0 if j > i else 0.0 for i in range(seq_len) for j in range(seq_len)]
    return _nd_arr(flat, dtype='float32').reshape((seq_len, seq_len))


def sliding_window_mask(seq_len: int, window_size: int) -> NDArray:
    """Sliding window attention mask — True means masked."""
    flat = [1.0 if (i - j < 0 or i - j > window_size) else 0.0
            for i in range(seq_len) for j in range(seq_len)]
    return _nd_arr(flat, dtype='float32').reshape((seq_len, seq_len))


# ── Attention layer variants ──────────────────────────────────────────────────

class GroupedQueryAttention(Module):
    """Grouped Query Attention (GQA) — Ainslie et al. 2023."""

    def __init__(self, embed_dim: int, num_q_heads: int, num_kv_heads: int,
                 dropout: float = 0.0, bias: bool = False,
                 rope: Optional[RotaryEmbedding] = None):
        super().__init__()
        assert embed_dim % num_q_heads == 0
        assert num_q_heads % num_kv_heads == 0
        self.embed_dim = embed_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_q_heads // num_kv_heads
        self.head_dim = embed_dim // num_q_heads
        self.dropout = dropout
        self.rope = rope
        scale = _math.sqrt(2.0 / (embed_dim + self.head_dim))
        self.w_q = self._param(embed_dim, num_q_heads * self.head_dim, scale)
        self.w_k = self._param(embed_dim, num_kv_heads * self.head_dim, scale)
        self.w_v = self._param(embed_dim, num_kv_heads * self.head_dim, scale)
        self.w_o = self._param(num_q_heads * self.head_dim, embed_dim, scale)

    def _param(self, in_f, out_f, scale):
        w = Tensor(_nd_randn((in_f, out_f), scale))
        w.requires_grad = True
        self._parameters[f'w_{len(self._parameters)}'] = w
        return w

    def _split_heads(self, x: NDArray, num_heads: int) -> NDArray:
        B, T, _ = x.shape
        return x.reshape((B, T, num_heads, self.head_dim)).transpose([0, 2, 1, 3])

    def forward(self, x: Tensor, mask=None, kv_cache=None, seq_offset: int = 0) -> Tensor:
        B, T, _ = x._data.shape
        # Route Q/K/V projections through VP CUDA when available
        try:
            from vectorphile._core import _vp_cuda as _cu
            import ctypes
            from vectorphile._core.ndarray import array as _nd_vp
            if hasattr(_cu, 'gemm_f32'):
                def _vp_mm(x_nd, w_nd):
                    xfl = x_nd.to_flat_list(); wfl = w_nd.to_flat_list()
                    sh_x = x_nd.shape; sh_w = w_nd.shape
                    M, K_ = sh_x[0]*sh_x[1], sh_x[2]
                    K2, N_ = sh_w
                    xa = (ctypes.c_float * len(xfl))(*xfl)
                    wa = (ctypes.c_float * len(wfl))(*wfl)
                    xbuf = _cu.alloc(len(xfl)*4); _cu.h2d(xa, xbuf)
                    wbuf = _cu.alloc(len(wfl)*4); _cu.h2d(wa, wbuf)
                    out_buf = _cu.gemm_f32(xbuf, wbuf, M, K_, N_)
                    out_arr = (ctypes.c_float * (M*N_))()
                    _cu.d2h(out_buf, out_arr)
                    return _nd_vp(list(out_arr), 'float32').reshape((B, T, N_))
                q = _vp_mm(x._data, self.w_q._data)
                k = _vp_mm(x._data, self.w_k._data)
                v = _vp_mm(x._data, self.w_v._data)
            else:
                raise AttributeError
        except Exception:
            q = x._data @ self.w_q._data
            k = x._data @ self.w_k._data
            v = x._data @ self.w_v._data
        q = self._split_heads(q, self.num_q_heads)
        k = self._split_heads(k, self.num_kv_heads)
        v = self._split_heads(v, self.num_kv_heads)
        if self.rope is not None:
            q_t, k_t = self.rope(Tensor(q.reshape((B, self.num_q_heads, T, self.head_dim))),
                                   Tensor(k.reshape((B, self.num_kv_heads, T, self.head_dim))),
                                   seq_offset)
            q, k = q_t._data, k_t._data
        if kv_cache is not None:
            if 'k' in kv_cache:
                from vectorphile.core.tensor import _nd_concatenate
                k = _nd_concatenate([kv_cache['k'], k], axis=2)
                v = _nd_concatenate([kv_cache['v'], v], axis=2)
            kv_cache['k'] = k
            kv_cache['v'] = v
        if self.num_groups > 1:
            # Repeat KV heads
            Hq, Hkv, Tk, d = self.num_q_heads, self.num_kv_heads, k.shape[2], self.head_dim
            k_flat = k.to_flat_list()
            v_flat = v.to_flat_list()
            k_rep, v_rep = [], []
            for b in range(B):
                for hq in range(Hq):
                    hkv = hq // self.num_groups
                    for t in range(Tk):
                        for di in range(d):
                            k_rep.append(k_flat[b*Hkv*Tk*d + hkv*Tk*d + t*d + di])
                            v_rep.append(v_flat[b*Hkv*Tk*d + hkv*Tk*d + t*d + di])
            k = _nd_arr(k_rep, dtype='float32').reshape((B, Hq, Tk, d))
            v = _nd_arr(v_rep, dtype='float32').reshape((B, Hq, Tk, d))
        out, _ = scaled_dot_product_attention(q, k, v, mask=mask,
                                               dropout_p=self.dropout, training=self.training)
        out = out.transpose([0, 2, 1, 3]).reshape((B, T, -1))
        return Tensor(out @ self.w_o._data)


class MultiQueryAttention(Module):
    """Multi-Query Attention (MQA) — Shazeer 2019."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        scale = 0.02
        self.w_q = Tensor(_nd_randn((embed_dim, embed_dim), scale))
        self.w_k = Tensor(_nd_randn((embed_dim, self.head_dim), scale))
        self.w_v = Tensor(_nd_randn((embed_dim, self.head_dim), scale))
        self.w_o = Tensor(_nd_randn((embed_dim, embed_dim), scale))
        for name, p in [('w_q', self.w_q), ('w_k', self.w_k),
                         ('w_v', self.w_v), ('w_o', self.w_o)]:
            p.requires_grad = True
            self._parameters[name] = p

    def forward(self, x: Tensor, mask=None) -> Tensor:
        B, T, _ = x._data.shape
        q = (x._data @ self.w_q._data).reshape((B, T, self.num_heads, self.head_dim)).transpose([0, 2, 1, 3])
        # k, v: (B, T, head_dim) → insert heads dim
        k_2d = x._data @ self.w_k._data  # (B, T, head_dim)
        v_2d = x._data @ self.w_v._data
        kf, vf = k_2d.to_flat_list(), v_2d.to_flat_list()
        d = self.head_dim
        # Broadcast to (B, num_heads, T, head_dim)
        k_rep = [kf[b*T*d + t*d + di] for b in range(B) for _ in range(self.num_heads) for t in range(T) for di in range(d)]
        v_rep = [vf[b*T*d + t*d + di] for b in range(B) for _ in range(self.num_heads) for t in range(T) for di in range(d)]
        k = _nd_arr(k_rep, dtype='float32').reshape((B, self.num_heads, T, d))
        v = _nd_arr(v_rep, dtype='float32').reshape((B, self.num_heads, T, d))
        out, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        out = out.transpose([0, 2, 1, 3]).reshape((B, T, -1))
        return Tensor(out @ self.w_o._data)


class SlidingWindowAttention(Module):
    """Sliding-window local attention (Mistral / Longformer-style)."""

    def __init__(self, embed_dim: int, num_heads: int, window_size: int = 4096,
                 dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.dropout = dropout
        scale = 0.02
        for name in ['w_q', 'w_k', 'w_v', 'w_o']:
            p = Tensor(_nd_randn((embed_dim, embed_dim), scale))
            p.requires_grad = True
            setattr(self, name, p)
            self._parameters[name] = p

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        B, T, _ = x._data.shape
        q = (x._data @ self.w_q._data).reshape((B, T, self.num_heads, self.head_dim)).transpose([0, 2, 1, 3])
        k = (x._data @ self.w_k._data).reshape((B, T, self.num_heads, self.head_dim)).transpose([0, 2, 1, 3])
        v = (x._data @ self.w_v._data).reshape((B, T, self.num_heads, self.head_dim)).transpose([0, 2, 1, 3])
        mask = sliding_window_mask(T, self.window_size)
        if causal:
            cm = causal_mask(T)
            mf = mask.to_flat_list()
            cf = cm.to_flat_list()
            combined = [1.0 if (mf[i] > 0.5 or cf[i] > 0.5) else 0.0 for i in range(len(mf))]
            mask = _nd_arr(combined, dtype='float32').reshape((T, T))
        # Reshape mask to (1, 1, T, T) for broadcasting
        mf = mask.to_flat_list()
        mask_4d = _nd_arr(mf, dtype='float32').reshape((1, 1, T, T))
        out, _ = scaled_dot_product_attention(q, k, v, mask=mask_4d)
        out = out.transpose([0, 2, 1, 3]).reshape((B, T, -1))
        return Tensor(out @ self.w_o._data)


# =============================================================================
# Session 11 - CrossAttention, LinearAttention, PerceiverAttention
# =============================================================================
from vectorphile.nn.layers import Linear  # noqa: E402  (placed after forward refs resolved)

class CrossAttention(Module):
    """Cross-attention: query from x, key/value from context.

    Used in encoder-decoder architectures (T5, Stable Diffusion UNet, etc.)
    where x attends over a separate context sequence.

    Args:
        d_model: query/output dimension
        d_context: key/value source dimension (default same as d_model)
        num_heads: number of attention heads
        bias: add bias to projections

    Usage::
        from vectorphile.nn.architectures import CrossAttention
        attn = CrossAttention(d_model=512, d_context=768, num_heads=8)
        y = attn(x, context)
    """

    def __init__(self, d_model: int, d_context: int = None, num_heads: int = 8,
                 bias: bool = True):
        super().__init__()
        d_context = d_context or d_model
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.scale     = self.head_dim ** -0.5
        self.w_q  = Linear(d_model,   d_model,   bias=bias)
        self.w_k  = Linear(d_context, d_model,   bias=bias)
        self.w_v  = Linear(d_context, d_model,   bias=bias)
        self.w_o  = Linear(d_model,   d_model,   bias=bias)

    def forward(self, x: Tensor, context: Tensor, mask: Tensor = None) -> Tensor:
        q = self.w_q(x)
        k = self.w_k(context)
        v = self.w_v(context)
        qf  = list(q._data.to_flat_list() if hasattr(q._data, 'to_flat_list') else q._data)
        kf  = list(k._data.to_flat_list() if hasattr(k._data, 'to_flat_list') else k._data)
        vf  = list(v._data.to_flat_list() if hasattr(v._data, 'to_flat_list') else v._data)
        sh  = x._data.shape; B, T, D = sh
        _, S, _ = context._data.shape
        import math as _m
        out = []
        for b in range(B):
            for t in range(T):
                qi = qf[b*T*D + t*D : b*T*D + t*D + D]
                scores = []
                for s in range(S):
                    kj = kf[b*S*D + s*D : b*S*D + s*D + D]
                    sc = sum(qi[d]*kj[d] for d in range(D)) * self.scale
                    scores.append(sc)
                m_ = max(scores)
                exps = [_m.exp(s - m_) for s in scores]
                denom = sum(exps)
                row = [0.0] * D
                for s in range(S):
                    vj = vf[b*S*D + s*D : b*S*D + s*D + D]
                    w  = exps[s] / denom
                    for d in range(D): row[d] += w * vj[d]
                out.extend(row)
        import array as _a2
        from vectorphile._core.ndarray import NDArray, _c_strides, _DType
        t_out = Tensor(NDArray(_a2.array('f', [float(v) for v in out]),
                               sh, _c_strides(sh), _DType('float32')))
        return self.w_o(t_out)


class LinearAttention(Module):
    """Linear Attention (Katharopoulos et al. 2020) - O(N) complexity.

    Uses elu+1 kernel: phi(Q)(phi(K)^T V) / (phi(Q) sum(phi(K)))

    Args:
        d_model: input/output dimension
        num_heads: number of attention heads
        eps: denominator stabiliser
    """

    def __init__(self, d_model: int, num_heads: int = 4, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.eps       = eps
        self.w_q = Linear(d_model, d_model, bias=False)
        self.w_k = Linear(d_model, d_model, bias=False)
        self.w_v = Linear(d_model, d_model, bias=False)
        self.w_o = Linear(d_model, d_model, bias=False)

    @staticmethod
    def _phi(x):
        import math as _m
        return [max(v, _m.exp(v) - 1) + 1 for v in x]

    def forward(self, x: Tensor) -> Tensor:
        import array as _a2
        from vectorphile._core.ndarray import NDArray, _c_strides, _DType
        q = self.w_q(x); k = self.w_k(x); v = self.w_v(x)
        sh = x._data.shape; B, N, D = sh; Dh = self.head_dim
        qf = list(q._data.to_flat_list() if hasattr(q._data, 'to_flat_list') else q._data)
        kf = list(k._data.to_flat_list() if hasattr(k._data, 'to_flat_list') else k._data)
        vf = list(v._data.to_flat_list() if hasattr(v._data, 'to_flat_list') else v._data)
        out = [0.0] * (B * N * D)
        for b in range(B):
            kv = [[0.0]*Dh for _ in range(Dh)]
            k_sum = [0.0] * Dh
            for n in range(N):
                offset = b*N*D + n*D
                kn = self._phi(kf[offset:offset+Dh])
                vn = vf[offset:offset+Dh]
                for i in range(Dh):
                    for j in range(Dh): kv[i][j] += kn[i] * vn[j]
                for i in range(Dh): k_sum[i] += kn[i]
            for n in range(N):
                offset = b*N*D + n*D
                qn = self._phi(qf[offset:offset+Dh])
                denom = sum(qn[i] * k_sum[i] for i in range(Dh)) + self.eps
                for j in range(Dh):
                    num = sum(qn[i] * kv[i][j] for i in range(Dh))
                    out[offset + j] = num / denom
        t_out = Tensor(NDArray(_a2.array('f', [float(v) for v in out]),
                               sh, _c_strides(sh), _DType('float32')))
        return self.w_o(t_out)
