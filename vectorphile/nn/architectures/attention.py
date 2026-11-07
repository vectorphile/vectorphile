"""Advanced attention mechanisms — RoPE, ALiBi, GQA, MQA, sliding-window, causal."""
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor
import vectorphile.backend as _backend


# ── Positional Encoding helpers ───────────────────────────────────────────────

def rotate_half(x: np.ndarray) -> np.ndarray:
    """Rotate the last dim by swapping and negating halves (RoPE helper)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q: np.ndarray, k: np.ndarray,
                          cos: np.ndarray, sin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply RoPE to query and key tensors."""
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class RotaryEmbedding(Module):
    """Rotary Position Embedding (RoPE) — Su et al. 2021.

    Used in LLaMA, Mistral, Falcon, and most modern LLMs.
    Encodes relative positions into Q/K projections without additive biases.
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        # θ_i = base^(-2i/d)
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        self._inv_freq = inv_freq
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = np.arange(seq_len, dtype=np.float32)
        freqs = np.outer(t, self._inv_freq)           # (seq, dim/2)
        emb = np.concatenate([freqs, freqs], axis=-1)  # (seq, dim)
        self._cos = np.cos(emb)[None, None, :, :]      # (1, 1, seq, dim)
        self._sin = np.sin(emb)[None, None, :, :]

    def forward(self, q: Tensor, k: Tensor, seq_offset: int = 0) -> Tuple[Tensor, Tensor]:
        seq_len = q._data.shape[-2] + seq_offset
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        cos = self._cos[:, :, seq_offset:seq_offset + q._data.shape[-2], :]
        sin = self._sin[:, :, seq_offset:seq_offset + q._data.shape[-2], :]
        q_rot, k_rot = apply_rotary_pos_emb(q._data, k._data, cos, sin)
        return Tensor(q_rot), Tensor(k_rot)


class ALiBiPositionBias(Module):
    """ALiBi (Attention with Linear Biases) — Press et al. 2021.

    Adds a linear position penalty to attention scores instead of embeddings.
    Strong length generalization beyond training sequence length.
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        # Slopes: 2^(-8/n) * [1..n] pattern
        m = np.arange(1, num_heads + 1, dtype=np.float32)
        slopes = 2.0 ** (-8.0 * m / num_heads)
        self._slopes = slopes  # (num_heads,)

    def get_bias(self, seq_len: int) -> np.ndarray:
        """Return (num_heads, seq_len, seq_len) bias matrix."""
        positions = np.arange(seq_len, dtype=np.float32)
        # relative distances: |i - j|
        dist = np.abs(positions[:, None] - positions[None, :])  # (seq, seq)
        # shape: (heads, seq, seq)
        bias = -self._slopes[:, None, None] * dist[None, :, :]
        return bias.astype(np.float32)


# ── Core attention functions ──────────────────────────────────────────────────

def scaled_dot_product_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    mask: Optional[np.ndarray] = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Standard scaled dot-product attention — dispatched through backend.

    Uses torch SDPA (which internally selects FlashAttention-2 on Ampere+)
    when the torch backend is active, otherwise the tiled numpy implementation.

    Args:
        q: (batch, heads, seq_q, head_dim)
        k: (batch, heads, seq_k, head_dim)
        v: (batch, heads, seq_k, head_dim)
        mask: broadcastable bool mask; True positions are MASKED OUT
    Returns:
        output (batch, heads, seq_q, head_dim), weights (batch, heads, seq_q, seq_k)
    """
    return _backend.attention(q, k, v, mask=mask, dropout_p=dropout_p, training=training)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / (e.sum(axis=axis, keepdims=True) + 1e-9)


def causal_mask(seq_len: int) -> np.ndarray:
    """Upper-triangular causal mask — True means masked (do not attend)."""
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)


def sliding_window_mask(seq_len: int, window_size: int) -> np.ndarray:
    """Sliding window attention mask (Mistral-style).

    Token i can only attend to tokens in [i - window_size, i].
    """
    idx = np.arange(seq_len)
    dist = idx[:, None] - idx[None, :]      # i - j
    # mask positions outside the window OR future positions
    mask = (dist < 0) | (dist > window_size)
    return mask


# ── Attention layer variants ──────────────────────────────────────────────────

class GroupedQueryAttention(Module):
    """Grouped Query Attention (GQA) — Ainslie et al. 2023.

    Used in LLaMA 2/3, Mistral, Gemma. Reduces KV cache memory by sharing
    key/value heads across groups of query heads.

    Args:
        embed_dim: total embedding dimension
        num_q_heads: number of query heads
        num_kv_heads: number of key/value heads (must divide num_q_heads evenly)
        dropout: attention dropout probability
        bias: whether to use bias in projections (LLaMA: False)
    """

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

        scale = np.sqrt(2.0 / (embed_dim + self.head_dim))
        self.w_q = self._param(embed_dim, num_q_heads * self.head_dim, scale, bias)
        self.w_k = self._param(embed_dim, num_kv_heads * self.head_dim, scale, bias)
        self.w_v = self._param(embed_dim, num_kv_heads * self.head_dim, scale, bias)
        self.w_o = self._param(num_q_heads * self.head_dim, embed_dim, scale, bias)

    def _param(self, in_f, out_f, scale, use_bias):
        w = Tensor(np.random.randn(in_f, out_f).astype(np.float32) * scale)
        w.requires_grad = True
        self._parameters[f'w_{len(self._parameters)}'] = w
        return w

    def _split_heads(self, x: np.ndarray, num_heads: int) -> np.ndarray:
        B, T, _ = x.shape
        return x.reshape(B, T, num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def forward(self, x: Tensor, mask: Optional[np.ndarray] = None,
                kv_cache: Optional[dict] = None, seq_offset: int = 0) -> Tensor:
        B, T, _ = x._data.shape
        q = x._data @ self.w_q._data   # (B, T, Hq*d)
        k = x._data @ self.w_k._data   # (B, T, Hkv*d)
        v = x._data @ self.w_v._data

        q = self._split_heads(q, self.num_q_heads)   # (B, Hq, T, d)
        k = self._split_heads(k, self.num_kv_heads)
        v = self._split_heads(v, self.num_kv_heads)

        if self.rope is not None:
            q_t, k_t = self.rope(Tensor(q.reshape(B, self.num_q_heads, T, self.head_dim)),
                                   Tensor(k.reshape(B, self.num_kv_heads, T, self.head_dim)),
                                   seq_offset)
            q, k = q_t._data, k_t._data

        # KV cache for inference
        if kv_cache is not None:
            if 'k' in kv_cache:
                k = np.concatenate([kv_cache['k'], k], axis=2)
                v = np.concatenate([kv_cache['v'], v], axis=2)
            kv_cache['k'] = k
            kv_cache['v'] = v

        # Expand KV heads to match Q heads
        if self.num_groups > 1:
            k = np.repeat(k, self.num_groups, axis=1)   # (B, Hq, T_kv, d)
            v = np.repeat(v, self.num_groups, axis=1)

        out, _ = scaled_dot_product_attention(q, k, v, mask=mask,
                                               dropout_p=self.dropout,
                                               training=self.training)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return Tensor(out @ self.w_o._data)


class MultiQueryAttention(Module):
    """Multi-Query Attention (MQA) — Shazeer 2019.

    Single KV head shared across all Q heads. Extreme KV cache reduction.
    Used in Falcon, PaLM, some variants of Gemini.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        scale = 0.02
        self.w_q = Tensor(np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale)
        self.w_k = Tensor(np.random.randn(embed_dim, self.head_dim).astype(np.float32) * scale)
        self.w_v = Tensor(np.random.randn(embed_dim, self.head_dim).astype(np.float32) * scale)
        self.w_o = Tensor(np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale)
        for name, p in [('w_q', self.w_q), ('w_k', self.w_k),
                         ('w_v', self.w_v), ('w_o', self.w_o)]:
            p.requires_grad = True
            self._parameters[name] = p

    def forward(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        B, T, _ = x._data.shape
        q = (x._data @ self.w_q._data).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = (x._data @ self.w_k._data)[:, None, :, :]   # (B, 1, T, d)
        v = (x._data @ self.w_v._data)[:, None, :, :]
        # Broadcast K/V across heads
        k = np.broadcast_to(k, (B, self.num_heads, T, self.head_dim))
        v = np.broadcast_to(v, (B, self.num_heads, T, self.head_dim))
        out, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return Tensor(out @ self.w_o._data)


class SlidingWindowAttention(Module):
    """Sliding-window local attention (Mistral / Longformer-style).

    Tokens attend only within a fixed window. O(n * w) instead of O(n^2).
    """

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
            p = Tensor(np.random.randn(embed_dim, embed_dim).astype(np.float32) * scale)
            p.requires_grad = True
            setattr(self, name, p)
            self._parameters[name] = p

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        B, T, _ = x._data.shape
        q = (x._data @ self.w_q._data).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = (x._data @ self.w_k._data).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = (x._data @ self.w_v._data).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        mask = sliding_window_mask(T, self.window_size)
        if causal:
            mask = mask | causal_mask(T)
        out, _ = scaled_dot_product_attention(q, k, v, mask=mask[None, None, :, :])
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return Tensor(out @ self.w_o._data)
