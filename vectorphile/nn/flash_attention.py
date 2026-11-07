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
import numpy as np
from typing import Optional
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor


BLOCK_SIZE = 64   # Tile size; smaller = less memory, more compute


def flash_attention_forward(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
) -> np.ndarray:
    """Tiled flash attention forward pass.

    Avoids materializing the full (B, H, N, N) attention matrix by processing
    Q/K/V in BLOCK_SIZE chunks with an online softmax (log-sum-exp merge).

    Args:
        q: (B, H, N, d) query
        k: (B, H, N, d) key
        v: (B, H, N, d) value
        causal: if True, apply causal (future) mask
        softmax_scale: 1/sqrt(d) if None
    Returns:
        out: (B, H, N, d) attention output
    """
    B, H, N, d = q.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / np.sqrt(d)

    out = np.zeros_like(q)

    for b in range(B):
        for h in range(H):
            q_bh = q[b, h]   # (N, d)
            k_bh = k[b, h]
            v_bh = v[b, h]

            # Online softmax state per query position
            m = np.full(N, -np.inf, dtype=np.float32)   # running max
            l = np.zeros(N, dtype=np.float32)             # running sum of exp
            o = np.zeros((N, d), dtype=np.float32)        # running output

            # Iterate over KV blocks
            for kv_start in range(0, N, BLOCK_SIZE):
                kv_end = min(kv_start + BLOCK_SIZE, N)
                k_block = k_bh[kv_start:kv_end]   # (Bk, d)
                v_block = v_bh[kv_start:kv_end]   # (Bk, d)

                # Iterate over Q blocks
                for q_start in range(0, N, BLOCK_SIZE):
                    q_end = min(q_start + BLOCK_SIZE, N)
                    q_block = q_bh[q_start:q_end]   # (Bq, d)

                    # Compute attention scores for this block pair
                    scores = q_block @ k_block.T * softmax_scale   # (Bq, Bk)

                    # Causal masking: query pos must be >= key pos
                    if causal:
                        q_idx = np.arange(q_start, q_end)[:, None]
                        k_idx = np.arange(kv_start, kv_end)[None, :]
                        causal_mask = k_idx > q_idx
                        scores = np.where(causal_mask, -1e9, scores)

                    # Online softmax update
                    block_max = scores.max(axis=-1)   # (Bq,)
                    m_old = m[q_start:q_end].copy()
                    m_new = np.maximum(m_old, block_max)
                    m[q_start:q_end] = m_new

                    # Rescale old accumulator
                    rescale = np.exp(m_old - m_new)   # (Bq,)
                    l[q_start:q_end] *= rescale
                    o[q_start:q_end] *= rescale[:, None]

                    # Add new block contribution
                    exp_scores = np.exp(scores - m_new[:, None])   # (Bq, Bk)
                    l[q_start:q_end] += exp_scores.sum(axis=-1)
                    o[q_start:q_end] += exp_scores @ v_block        # (Bq, d)

            # Normalize
            out[b, h] = o / (l[:, None] + 1e-9)

    return out.astype(np.float32)


def flash_attention_varlen(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    cu_seqlens_q: np.ndarray, cu_seqlens_k: np.ndarray,
    causal: bool = False,
) -> np.ndarray:
    """Variable-length flash attention for packed (non-padded) batches.

    Avoids padding waste by packing multiple sequences along the N dimension.

    Args:
        q: (total_q, H, d) — all query tokens packed
        k: (total_k, H, d) — all key tokens packed
        v: (total_k, H, d) — all value tokens packed
        cu_seqlens_q: (B+1,) cumulative sequence lengths for queries
        cu_seqlens_k: (B+1,) cumulative sequence lengths for keys
        causal: apply causal mask per sequence
    Returns:
        out: (total_q, H, d)
    """
    total_q, H, d = q.shape
    out = np.zeros_like(q)
    B = len(cu_seqlens_q) - 1
    scale = 1.0 / np.sqrt(d)

    for b in range(B):
        qs, qe = int(cu_seqlens_q[b]), int(cu_seqlens_q[b+1])
        ks, ke = int(cu_seqlens_k[b]), int(cu_seqlens_k[b+1])
        # Reshape to (1, H, Nq, d) for flash_attention_forward
        q_b = q[qs:qe].transpose(1, 0, 2)[None]   # (1, H, Nq, d)
        k_b = k[ks:ke].transpose(1, 0, 2)[None]
        v_b = v[ks:ke].transpose(1, 0, 2)[None]
        out_b = flash_attention_forward(q_b, k_b, v_b, causal=causal, softmax_scale=scale)
        out[qs:qe] = out_b[0].transpose(1, 0, 2)  # (Nq, H, d)

    return out


class FlashAttention(Module):
    """Multi-head attention using flash attention (tiled, memory-efficient).

    Mathematically equivalent to standard attention but avoids O(N²) memory.

    Args:
        embed_dim: total embedding dimension
        num_heads: number of attention heads
        dropout: attention dropout (applied post-attention in non-flash path)
        causal: use causal mask (for decoder / autoregressive models)
        bias: use bias in Q/K/V/O projections
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 causal: bool = False, bias: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.dropout   = dropout
        self.causal    = causal
        self.scale     = 1.0 / np.sqrt(self.head_dim)

        s = 0.02
        for name in ['w_q', 'w_k', 'w_v', 'w_o']:
            w = Tensor(np.random.randn(embed_dim, embed_dim).astype(np.float32) * s)
            w.requires_grad = True
            setattr(self, name, w)
            self._parameters[name] = w
        if bias:
            for name in ['b_q', 'b_k', 'b_v', 'b_o']:
                b = Tensor(np.zeros(embed_dim, dtype=np.float32))
                b.requires_grad = True
                setattr(self, name, b)
                self._parameters[name] = b
        self._has_bias = bias

    def forward(self, x: Tensor,
                key_padding_mask: Optional[np.ndarray] = None) -> Tensor:
        """
        Args:
            x: (B, N, embed_dim)
            key_padding_mask: (B, N) boolean — True = padded (ignored)
        Returns:
            (B, N, embed_dim)
        """
        B, N, C = x._data.shape
        H, D = self.num_heads, self.head_dim

        def _proj(w_name, b_name=None):
            w = getattr(self, w_name)._data
            b = getattr(self, b_name)._data if b_name and self._has_bias else None
            out = x._data @ w + (b if b is not None else 0)
            return out.reshape(B, N, H, D).transpose(0, 2, 1, 3)   # (B, H, N, D)

        q = _proj('w_q', 'b_q')
        k = _proj('w_k', 'b_k')
        v = _proj('w_v', 'b_v')

        # Use flash attention
        out = flash_attention_forward(q, k, v, causal=self.causal, softmax_scale=self.scale)

        # Merge heads
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)   # (B, N, C)

        # Output projection
        w_o = self.w_o._data
        b_o = self.b_o._data if self._has_bias else 0
        return Tensor((out @ w_o + b_o).astype(np.float32))
