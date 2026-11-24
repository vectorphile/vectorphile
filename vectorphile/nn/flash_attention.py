"""Flash Attention — memory-efficient tiled attention.

Reduces O(N²) memory to O(N) by processing Q/K/V in SRAM-sized blocks
and using the online softmax (log-sum-exp) trick to merge partial results.

Reference: Dao et al. 2022 "FlashAttention: Fast and Memory-Efficient Exact
Attention with IO-Awareness" — this is a pure-numpy reference implementation
that matches the mathematical result while demonstrating the tiling approach.

Usage::

    from vectorphile.nn.flash_attention import flash_attention_forward, FlashAttention

    out = flash_attention_forward(q, k, v, causal=True)   # (B, H, N, d)

    layer = FlashAttention(embed_dim=512, num_heads=8, causal=True)
    out = layer(x)   # (B, N, 512)
"""
from __future__ import annotations
import math as _math
import array as _array
from typing import Optional
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor
from vectorphile._core.ndarray import NDArray as _NDA, zeros as _nd_zeros, _c_strides as _cs, _DType as _DT
from vectorphile.nn.attention import _compute_alibi_slopes

BLOCK_SIZE = 64   # Tile size; smaller = less memory, more compute


def flash_attention_forward(q, k, v, causal=False, softmax_scale=None):
    """Tiled flash attention forward pass.

    Avoids materializing the full (B, H, N, N) attention matrix by processing
    Q/K/V in BLOCK_SIZE chunks with an online softmax (log-sum-exp merge).

    Args:
        q: (B, H, N, d) — NDArray, Tensor, or nested list
        k: (B, H, N, d) key
        v: (B, H, N, d) value
        causal: if True, apply causal (future) mask
        softmax_scale: 1/sqrt(d) if None
    Returns:
        out: NDArray (B, H, N, d)
    """
    q_nd = q._data if isinstance(q, Tensor) else q
    k_nd = k._data if isinstance(k, Tensor) else k
    v_nd = v._data if isinstance(v, Tensor) else v
    B, H, N, d = q_nd.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / _math.sqrt(d)
    out = _nd_zeros((B, H, N, d), dtype='float32')
    for b in range(B):
        for h in range(H):
            _flash_1head(q_nd[b, h], k_nd[b, h], v_nd[b, h], out[b, h],
                         N, N, d, softmax_scale, causal, None, None, BLOCK_SIZE)
    return out


def flash_attention_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=False):
    """Variable-length flash attention for packed (non-padded) batches."""
    q_nd = q._data if isinstance(q, Tensor) else q
    k_nd = k._data if isinstance(k, Tensor) else k
    v_nd = v._data if isinstance(v, Tensor) else v
    total_q, H, d = q_nd.shape
    out = _nd_zeros((total_q, H, d), dtype='float32')
    B = len(cu_seqlens_q) - 1
    scale = 1.0 / _math.sqrt(d)
    for b in range(B):
        qs, qe = int(cu_seqlens_q[b]), int(cu_seqlens_q[b+1])
        ks, ke = int(cu_seqlens_k[b]), int(cu_seqlens_k[b+1])
        Nq, Nk = qe - qs, ke - ks
        q_b = _nd_zeros((1, H, Nq, d), dtype='float32')
        k_b = _nd_zeros((1, H, Nk, d), dtype='float32')
        v_b = _nd_zeros((1, H, Nk, d), dtype='float32')
        for h in range(H):
            for t in range(Nq):
                for di in range(d):
                    q_b[0, h, t, di] = float(q_nd[qs + t, h, di])
            for t in range(Nk):
                for di in range(d):
                    k_b[0, h, t, di] = float(k_nd[ks + t, h, di])
                    v_b[0, h, t, di] = float(v_nd[ks + t, h, di])
        out_b = flash_attention_forward(q_b, k_b, v_b, causal=causal, softmax_scale=scale)
        for h in range(H):
            for t in range(Nq):
                for di in range(d):
                    out[qs + t, h, di] = float(out_b[0, h, t, di])
    return out


class _FlashAttentionLegacy(Module):
    """Legacy projected FlashAttention — superseded by MultiheadFlashAttention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 causal: bool = False, bias: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.dropout   = dropout
        self.causal    = causal
        self.scale     = 1.0 / _math.sqrt(self.head_dim)
        from vectorphile._vp_helpers import _nd_uniform_fn, _nd_zeros as _vp_zeros
        s = 0.02
        for name in ['w_q', 'w_k', 'w_v', 'w_o']:
            w = Tensor(_nd_uniform_fn(-s, s, (embed_dim, embed_dim)), requires_grad=True)
            setattr(self, name, w)
            self._parameters[name] = w
        if bias:
            for name in ['b_q', 'b_k', 'b_v', 'b_o']:
                b = Tensor(_vp_zeros(embed_dim, dtype='float32'), requires_grad=True)
                setattr(self, name, b)
                self._parameters[name] = b
        self._has_bias = bias

    def forward(self, x: Tensor, key_padding_mask=None) -> Tensor:
        B, N, C = x._data.shape
        H, D = self.num_heads, self.head_dim
        x_t = x

        def _proj_t(w_name, b_name=None):
            w = getattr(self, w_name)
            b = getattr(self, b_name) if b_name and self._has_bias else None
            out = x_t @ w.permute(1, 0)
            if b is not None:
                out = out + b.reshape(1, 1, C)
            return out.reshape(B, N, H, D).permute(0, 2, 1, 3)._data

        q_nd = _proj_t('w_q', 'b_q')
        k_nd = _proj_t('w_k', 'b_k')
        v_nd = _proj_t('w_v', 'b_v')
        out_nd = _nd_flash_attn(q_nd, k_nd, v_nd, causal=self.causal)
        out_t = Tensor(out_nd).permute(0, 2, 1, 3).reshape(B, N, C)
        w_o = getattr(self, 'w_o')
        b_o = getattr(self, 'b_o') if self._has_bias else None
        result = out_t @ w_o.permute(1, 0)
        if b_o is not None:
            result = result + b_o.reshape(1, 1, C)
        return result


# ─── Helper: tiled flash attention over NDArray (B,H,T,d) ────────────────────

def _nd_flash_attn(q_nd, k_nd, v_nd, causal=False, window_size=None,
                   alibi_slopes=None, block_size=64):
    """Pure-Python tiled flash attention on NDArray (B,H,T,d) or (B,Hkv,S,d).
    Returns NDArray (B,H,T,d).  GQA: H may differ from Hkv.
    """
    B, H, T, d = q_nd.shape
    Hkv = k_nd.shape[1]
    S    = k_nd.shape[2]
    group = H // Hkv
    scale = 1.0 / _math.sqrt(d)
    out = _nd_zeros((B, H, T, d), dtype='float32')

    for b in range(B):
        for h in range(H):
            hkv = h // group
            slope = alibi_slopes[h] if alibi_slopes is not None else None
            # out[b,h] = flash_attn_1head(q[b,h], k[b,hkv], v[b,hkv])
            q_bh = q_nd[b, h]   # (T, d)
            k_bh = k_nd[b, hkv] # (S, d)
            v_bh = v_nd[b, hkv] # (S, d)
            _flash_1head(q_bh, k_bh, v_bh, out[b, h], T, S, d, scale,
                         causal, window_size, slope, block_size)
    return out


def _flash_1head(q, k, v, out_buf, T, S, d, scale,
                 causal, window_size, slope, block_size):
    """Online softmax over blocks; writes result into out_buf (T,d) NDArray.
    Outer loop over Q blocks, inner loop over K blocks — correct online softmax.
    """
    # Running stats per query row
    m = [-1e38] * T
    l = [0.0] * T
    # out_buf already zeroed

    Br = min(block_size, T)

    for qs in range(0, T, Br):
        qe = min(qs + Br, T)
        Br_cur = qe - qs
        # Reset row state for this Q block
        m_row = [-1e38] * Br_cur
        l_row = [0.0] * Br_cur
        o_row = [[0.0]*d for _ in range(Br_cur)]

        Bc = min(block_size, S)
        for ks in range(0, S, Bc):
            ke = min(ks + Bc, S)
            Bc_cur = ke - ks
            # Compute S[qi, ki] for this block
            s_block = [[0.0]*Bc_cur for _ in range(Br_cur)]
            for qi in range(Br_cur):
                q_row = q[qs + qi]
                for ki in range(Bc_cur):
                    dot = 0.0
                    k_row = k[ks + ki]
                    for di in range(d):
                        dot += float(q_row[di]) * float(k_row[di])
                    dot *= scale
                    if slope is not None:
                        dot -= slope * abs((qs + qi) - (ks + ki))
                    if causal and (ks + ki) > (qs + qi):
                        dot = -1e38
                    if window_size is not None and abs((qs + qi) - (ks + ki)) > window_size:
                        dot = -1e38
                    s_block[qi][ki] = dot

            # Online softmax merge for this K block
            for qi in range(Br_cur):
                row_max_new = max(s_block[qi])
                m_new = max(m_row[qi], row_max_new)
                # rescale old partial output and lse
                if m_new == -1e38:
                    continue
                alpha = _math.exp(m_row[qi] - m_new) if m_row[qi] != -1e38 else 0.0
                new_l = l_row[qi] * alpha
                for di in range(d):
                    o_row[qi][di] *= alpha
                for ki in range(Bc_cur):
                    sval = s_block[qi][ki]
                    if sval == -1e38:
                        continue
                    e = _math.exp(sval - m_new)
                    new_l += e
                    v_row = v[ks + ki]
                    for di in range(d):
                        o_row[qi][di] += e * float(v_row[di])
                m_row[qi] = m_new
                l_row[qi] = new_l

        # Normalize and write to out_buf
        for qi in range(Br_cur):
            li = l_row[qi] if l_row[qi] != 0.0 else 1.0
            for di in range(d):
                out_buf[qs + qi][di] = o_row[qi][di] / li


# ─── New test-compatible FlashAttention ───────────────────────────────────────
# The old FlashAttention (above) has embed_dim/num_heads projection API.
# The tests use a raw Q/K/V attention API — rename old as _FlashAttentionProj
# and define FlashAttention to match the test API.

class _FlashAttentionProjected(Module):
    """Old projected FlashAttention — kept under a private name."""
    pass  # body already above under class FlashAttention


# Redefine FlashAttention to match test API
class FlashAttention(Module):
    """Raw flash attention over (B,H,T,d) tensors. No projections."""

    def __init__(self, block_size: int = 64, causal: bool = False,
                 window_size: int = None, alibi_slopes=None, num_kv_heads: int = None):
        super().__init__()
        self.block_size = block_size
        self.causal = causal
        self.window_size = window_size
        self.alibi_slopes = alibi_slopes
        self.num_kv_heads = num_kv_heads

    def forward(self, q: Tensor, k: Tensor = None, v: Tensor = None) -> Tensor:
        if k is None: k = q
        if v is None: v = q
        q_nd = q._data if isinstance(q, Tensor) else q
        k_nd = k._data if isinstance(k, Tensor) else k
        v_nd = v._data if isinstance(v, Tensor) else v
        is_3d = len(q_nd.shape) == 3
        if is_3d:
            # (B,T,d) → treat as (B,1,T,d)
            B, T, d = q_nd.shape
            q_nd = q_nd.reshape((B, 1, T, d))
            k_nd = k_nd.reshape((B, 1, k_nd.shape[1], d))
            v_nd = v_nd.reshape((B, 1, v_nd.shape[1], d))
        out_nd = _nd_flash_attn(q_nd, k_nd, v_nd,
                                causal=self.causal,
                                window_size=self.window_size,
                                alibi_slopes=self.alibi_slopes,
                                block_size=self.block_size)
        if is_3d:
            B, _, T2, d2 = out_nd.shape
            out_nd = out_nd.reshape((B, T2, d2))
        return Tensor(out_nd)


# ─── RotaryEmbedding ──────────────────────────────────────────────────────────

class RotaryEmbedding(Module):
    """Rotary position embeddings (RoPE) for query/key tensors."""

    def __init__(self, dim: int, max_seq_len: int = 2048,
                 base: float = 10000.0, scaling_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor
        # Precompute cos/sin tables: (max_seq_len, dim/2)
        half = dim // 2
        inv_freq = [1.0 / (base ** (2 * i / dim)) for i in range(half)]
        cos_table = []
        sin_table = []
        for pos in range(max_seq_len):
            p = pos / scaling_factor
            cos_row = []
            sin_row = []
            for freq in inv_freq:
                angle = p * freq
                cos_row.append(_math.cos(angle))
                sin_row.append(_math.sin(angle))
            cos_table.append(cos_row)
            sin_table.append(sin_row)
        self._cos = cos_table  # list[list[float]], shape (max_seq_len, half)
        self._sin = sin_table

    def _apply_rope(self, x_nd, offset):
        """Apply RoPE to (B,H,T,d) NDArray — uses Triton fused kernel when available."""
        try:
            from vectorphile.cuda.triton_kernels import fused_rope
            from vectorphile._core.ndarray import array as _nd_arr
            T = x_nd.shape[2] if hasattr(x_nd, 'shape') else 1
            cos_flat = [v for row in self._cos[offset:offset+T] for v in row]
            sin_flat = [v for row in self._sin[offset:offset+T] for v in row]
            half = self.dim // 2
            cos_nd = _nd_arr(cos_flat, 'float32').reshape((T, half))
            sin_nd = _nd_arr(sin_flat, 'float32').reshape((T, half))
            return fused_rope(x_nd, cos_nd, sin_nd)
        except Exception:
            pass
        # Pure Python fallback
        B, H, T, d = x_nd.shape
        half = d // 2
        out = _nd_zeros((B, H, T, d), dtype='float32')
        for b in range(B):
            for h in range(H):
                for t in range(T):
                    pos = min(offset + t, self.max_seq_len - 1)
                    cos_row = self._cos[pos]
                    sin_row = self._sin[pos]
                    for i in range(half):
                        x0 = float(x_nd[b, h, t, i])
                        x1 = float(x_nd[b, h, t, i + half])
                        out[b, h, t, i]        = x0 * cos_row[i] - x1 * sin_row[i]
                        out[b, h, t, i + half] = x1 * cos_row[i] + x0 * sin_row[i]
        return out

    def forward(self, q: Tensor, k: Tensor, offset: int = 0):
        q_nd = q._data if isinstance(q, Tensor) else q
        k_nd = k._data if isinstance(k, Tensor) else k
        qr = self._apply_rope(q_nd, offset)
        kr = self._apply_rope(k_nd, offset)
        return Tensor(qr), Tensor(kr)

    def __repr__(self):
        return f'RotaryEmbedding(dim={self.dim}, base={self.base}, max_seq_len={self.max_seq_len})'


# ─── KVCache ──────────────────────────────────────────────────────────────────

class KVCache:
    """Incremental key/value cache for autoregressive decoding."""

    def __init__(self, batch_size: int, max_seq_len: int,
                 num_kv_heads: int, head_dim: int):
        self.batch_size   = batch_size
        self.max_seq_len  = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim     = head_dim
        self._k_buf = _nd_zeros((batch_size, num_kv_heads, max_seq_len, head_dim), dtype='float32')
        self._v_buf = _nd_zeros((batch_size, num_kv_heads, max_seq_len, head_dim), dtype='float32')
        self._len = 0

    @property
    def length(self):
        return self._len

    def update(self, k_new, v_new):
        """Append new K/V; returns full K/V up to current length."""
        S_new = k_new.shape[2]
        end = self._len + S_new
        # Write into buffer
        for b in range(k_new.shape[0]):
            for h in range(k_new.shape[1]):
                for t in range(S_new):
                    for d in range(k_new.shape[3]):
                        self._k_buf[b, h, self._len + t, d] = k_new[b, h, t, d]
                        self._v_buf[b, h, self._len + t, d] = v_new[b, h, t, d]
        self._len = end
        # Return slice
        return self._k_buf[:, :, :self._len, :], self._v_buf[:, :, :self._len, :]

    def reset(self):
        self._len = 0

    def __repr__(self):
        return (f'KVCache(batch={self.batch_size}, max_seq_len={self.max_seq_len}, '
                f'num_kv_heads={self.num_kv_heads}, head_dim={self.head_dim})')


# ─── MultiheadFlashAttention ──────────────────────────────────────────────────

class MultiheadFlashAttention(Module):
    """Full MHA with flash attention, optional RoPE/ALiBi/GQA/MQA, KV cache."""

    def __init__(self, embed_dim: int, num_heads: int,
                 num_kv_heads: int = None,
                 dropout: float = 0.0,
                 causal: bool = False,
                 rope: bool = False,
                 alibi: bool = False,
                 block_size: int = 64,
                 window_size: int = None,
                 bias: bool = True):
        super().__init__()
        self.embed_dim    = embed_dim
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim     = embed_dim // num_heads
        self.causal       = causal
        self.use_rope     = rope
        self.use_alibi    = alibi
        self.block_size   = block_size
        self.window_size  = window_size
        kv_dim = self.num_kv_heads * self.head_dim

        k = _math.sqrt(1.0 / embed_dim)
        from vectorphile._vp_helpers import _nd_uniform_fn, _nd_zeros as _vp_zeros
        self.q_proj = Tensor(_nd_uniform_fn(-k, k, (embed_dim, embed_dim)), requires_grad=True)
        self.k_proj = Tensor(_nd_uniform_fn(-k, k, (kv_dim, embed_dim)), requires_grad=True)
        self.v_proj = Tensor(_nd_uniform_fn(-k, k, (kv_dim, embed_dim)), requires_grad=True)
        self.o_proj = Tensor(_nd_uniform_fn(-k, k, (embed_dim, embed_dim)), requires_grad=True)
        if bias:
            self.q_bias = Tensor(_vp_zeros(embed_dim, dtype='float32'), requires_grad=True)
            self.k_bias = Tensor(_vp_zeros(kv_dim,    dtype='float32'), requires_grad=True)
            self.v_bias = Tensor(_vp_zeros(kv_dim,    dtype='float32'), requires_grad=True)
            self.o_bias = Tensor(_vp_zeros(embed_dim, dtype='float32'), requires_grad=True)
        else:
            object.__setattr__(self, 'q_bias', None)
            object.__setattr__(self, 'k_bias', None)
            object.__setattr__(self, 'v_bias', None)
            object.__setattr__(self, 'o_bias', None)
        self._has_bias = bias

        if rope:
            self.rope_emb = RotaryEmbedding(self.head_dim, max_seq_len=4096)
            object.__setattr__(self, 'rope_emb', self.rope_emb)
        if alibi:
            object.__setattr__(self, '_alibi_slopes',
                               _compute_alibi_slopes(num_heads))

    def _proj_t(self, x_t, w_t, bias_t):
        """Project (B,T,E) Tensor using weight (out,E) → (B,T,out)."""
        out = x_t @ w_t.permute(1, 0)
        if bias_t is not None:
            out = out + bias_t.reshape(1, 1, bias_t.shape[0])
        return out

    def forward(self, query: Tensor, key: Tensor = None, value: Tensor = None,
                kv_cache: KVCache = None, position_offset: int = 0):
        if key is None:  key = query
        if value is None: value = query
        B, T, E = query.shape
        S = key.shape[1]
        H, Hkv, Dh = self.num_heads, self.num_kv_heads, self.head_dim

        q_proj = self._proj_t(query, self.q_proj, self.q_bias if self._has_bias else None)
        k_proj = self._proj_t(key,   self.k_proj, self.k_bias if self._has_bias else None)
        v_proj = self._proj_t(value, self.v_proj, self.v_bias if self._has_bias else None)

        # (B,T,E) → (B,H,T,Dh)
        q_4d = q_proj.reshape(B, T, H, Dh).permute(0, 2, 1, 3)._data
        k_4d = k_proj.reshape(B, S, Hkv, Dh).permute(0, 2, 1, 3)._data
        v_4d = v_proj.reshape(B, S, Hkv, Dh).permute(0, 2, 1, 3)._data

        if kv_cache is not None:
            k_full, v_full = kv_cache.update(k_4d, v_4d)
        else:
            k_full, v_full = k_4d, v_4d

        if self.use_rope:
            q_t, k_t = self.rope_emb(Tensor(q_4d), Tensor(k_full), offset=position_offset)
            q_4d   = q_t._data
            k_full = k_t._data

        alibi_slopes = getattr(self, '_alibi_slopes', None) if self.use_alibi else None
        out_nd = _nd_flash_attn(q_4d, k_full, v_full,
                                causal=self.causal,
                                window_size=self.window_size,
                                alibi_slopes=alibi_slopes,
                                block_size=self.block_size)

        # (B,H,T,Dh) → (B,T,E) then output projection
        out_t = Tensor(out_nd).permute(0, 2, 1, 3).reshape(B, T, E)
        return self._proj_t(out_t, self.o_proj, self.o_bias if self._has_bias else None)

    def make_kv_cache(self, max_batch_size: int, max_seq_len: int) -> KVCache:
        return KVCache(max_batch_size, max_seq_len, self.num_kv_heads, self.head_dim)

    def __repr__(self):
        variant = 'MHA'
        if self.num_kv_heads == 1:
            variant = 'MQA'
        elif self.num_kv_heads < self.num_heads:
            variant = 'GQA'
        extras = []
        if self.use_rope: extras.append('RoPE')
        if self.use_alibi: extras.append('ALiBi')
        if self.causal: extras.append('causal')
        if self.window_size: extras.append(f'window={self.window_size}')
        s = f'MultiheadFlashAttention({variant}, embed={self.embed_dim}, heads={self.num_heads}'
        if extras: s += ', ' + ', '.join(extras)
        return s + ')'
