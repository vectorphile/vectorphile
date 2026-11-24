"""Tests for the full vectorphile attention suite.

Covers:
  FlashAttention  — MHA / MQA / GQA / causal / sliding-window / ALiBi
  RotaryEmbedding — shape, offset, NTK scaling
  KVCache         — incremental updates, reset
  MultiheadFlashAttention — MHA / GQA / MQA / RoPE / ALiBi / KV cache /
                            cross-attention / sliding window
"""
import math
import array
import random
import pytest
import vectorphile.nn as nn
from vectorphile.core.tensor import Tensor
from vectorphile._core.ndarray import NDArray, zeros as nd_zeros
from vectorphile._vp_helpers import _math_exp
from vectorphile.nn.attention import _compute_alibi_slopes


# ─── helpers ─────────────────────────────────────────────────────────────────

def _rand_t(*shape, scale=0.05):
    buf = array.array('f', [random.gauss(0, scale) for _ in range(math.prod(shape))])
    s, st = 1, []
    for x in reversed(shape):
        st.append(s); s *= x
    st.reverse()
    return Tensor(NDArray(buf, shape, st, 'float32'))


def _naive_attn(q, k, v, causal=False):
    """Reference full N×N softmax attention, NDArrays (B, H, T, d)."""
    B, H, T, d = q.shape
    Hkv = k.shape[1]
    S   = k.shape[2]
    group = H // Hkv
    scale = math.sqrt(d)
    out = nd_zeros((B, H, T, d), dtype='float32')
    for b in range(B):
        for h in range(H):
            hkv = h // group
            S_mat = (q[b, h] @ k[b, hkv].transpose()) / scale
            if causal:
                for i in range(T):
                    for j in range(i + 1, min(S, T)):
                        S_mat[i, j] = -1e9
            S_mat = S_mat - S_mat.max(axis=1)[:, None]
            E = _math_exp(S_mat)
            A = E / E.sum(axis=1)[:, None]
            out[b, h] = A @ v[b, hkv]
    return out


def _max_err(a, b):
    diff = a - b
    return max(abs(x) for x in diff._buf)


# ─── FlashAttention shapes ────────────────────────────────────────────────────

class TestFlashAttentionShapes:
    def test_mha_4d(self):
        fa = nn.FlashAttention(block_size=4)
        out = fa(_rand_t(2,4,8,16), _rand_t(2,4,8,16), _rand_t(2,4,8,16))
        assert out._data.shape == (2, 4, 8, 16)

    def test_mha_3d(self):
        fa = nn.FlashAttention(block_size=4)
        out = fa(_rand_t(2,8,16), _rand_t(2,8,16), _rand_t(2,8,16))
        assert out._data.shape == (2, 8, 16)

    def test_mqa(self):
        fa = nn.FlashAttention(block_size=4, num_kv_heads=1)
        out = fa(_rand_t(2,4,8,16), _rand_t(2,1,8,16), _rand_t(2,1,8,16))
        assert out._data.shape == (2, 4, 8, 16)

    def test_gqa(self):
        fa = nn.FlashAttention(block_size=4, num_kv_heads=2)
        out = fa(_rand_t(1,8,16,32), _rand_t(1,2,16,32), _rand_t(1,2,16,32))
        assert out._data.shape == (1, 8, 16, 32)

    def test_cross_attention_shape(self):
        fa = nn.FlashAttention(block_size=4)
        q = _rand_t(1, 1, 6, 8)
        k = _rand_t(1, 1, 10, 8)
        v = _rand_t(1, 1, 10, 8)
        out = fa(q, k, v)
        assert out._data.shape == (1, 1, 6, 8)

    def test_sliding_window(self):
        fa = nn.FlashAttention(block_size=4, causal=True, window_size=3)
        out = fa(_rand_t(1,2,12,8), _rand_t(1,2,12,8), _rand_t(1,2,12,8))
        assert out._data.shape == (1, 2, 12, 8)

    def test_alibi_shape(self):
        slopes = _compute_alibi_slopes(4)
        fa = nn.FlashAttention(block_size=4, alibi_slopes=slopes)
        out = fa(_rand_t(1,4,8,16), _rand_t(1,4,8,16), _rand_t(1,4,8,16))
        assert out._data.shape == (1, 4, 8, 16)


# ─── FlashAttention correctness ───────────────────────────────────────────────

class TestFlashAttentionCorrectness:
    def test_mha_matches_naive(self):
        fa = nn.FlashAttention(block_size=4)
        q, k, v = _rand_t(1,2,16,8), _rand_t(1,2,16,8), _rand_t(1,2,16,8)
        assert _max_err(fa(q,k,v)._data, _naive_attn(q._data,k._data,v._data)) < 1e-4

    def test_causal_matches_naive(self):
        fa = nn.FlashAttention(block_size=4, causal=True)
        q, k, v = _rand_t(1,2,16,8), _rand_t(1,2,16,8), _rand_t(1,2,16,8)
        assert _max_err(fa(q,k,v)._data,
                        _naive_attn(q._data,k._data,v._data,causal=True)) < 1e-4

    def test_gqa_matches_naive(self):
        fa = nn.FlashAttention(block_size=4, num_kv_heads=2)
        q, k, v = _rand_t(1,4,8,8), _rand_t(1,2,8,8), _rand_t(1,2,8,8)
        assert _max_err(fa(q,k,v)._data, _naive_attn(q._data,k._data,v._data)) < 1e-4

    def test_mqa_matches_naive(self):
        fa = nn.FlashAttention(block_size=4, num_kv_heads=1)
        q, k, v = _rand_t(1,4,8,8), _rand_t(1,1,8,8), _rand_t(1,1,8,8)
        assert _max_err(fa(q,k,v)._data, _naive_attn(q._data,k._data,v._data)) < 1e-4

    def test_block_one(self):
        fa = nn.FlashAttention(block_size=1)
        q, k, v = _rand_t(1,1,8,4), _rand_t(1,1,8,4), _rand_t(1,1,8,4)
        assert _max_err(fa(q,k,v)._data, _naive_attn(q._data,k._data,v._data)) < 1e-4

    def test_block_larger_than_seq(self):
        fa = nn.FlashAttention(block_size=256)
        q, k, v = _rand_t(1,1,8,4), _rand_t(1,1,8,4), _rand_t(1,1,8,4)
        assert _max_err(fa(q,k,v)._data, _naive_attn(q._data,k._data,v._data)) < 1e-4


# ─── RotaryEmbedding ──────────────────────────────────────────────────────────

class TestRotaryEmbedding:
    def test_shape_4d(self):
        rope = nn.RotaryEmbedding(dim=16, max_seq_len=64)
        q, k = _rand_t(2,4,8,16), _rand_t(2,4,8,16)
        qr, kr = rope(q, k)
        assert qr._data.shape == (2, 4, 8, 16)
        assert kr._data.shape == (2, 4, 8, 16)

    def test_output_differs_from_input(self):
        rope = nn.RotaryEmbedding(dim=16, max_seq_len=64)
        q, k = _rand_t(1,2,8,16), _rand_t(1,2,8,16)
        qr, kr = rope(q, k)
        # rotation should change the values
        assert _max_err(qr._data, q._data) > 1e-6

    def test_offset(self):
        rope = nn.RotaryEmbedding(dim=16, max_seq_len=64)
        q, k = _rand_t(1,2,4,16), _rand_t(1,2,4,16)
        qr0, _ = rope(q, k, offset=0)
        qr4, _ = rope(q, k, offset=4)
        # different offsets → different rotations
        assert _max_err(qr0._data, qr4._data) > 1e-6

    def test_ntk_scaling(self):
        rope = nn.RotaryEmbedding(dim=16, max_seq_len=64, scaling_factor=4.0)
        q, k = _rand_t(1,2,8,16), _rand_t(1,2,8,16)
        qr, kr = rope(q, k)
        assert qr._data.shape == (1, 2, 8, 16)

    def test_repr(self):
        rope = nn.RotaryEmbedding(dim=64, max_seq_len=512, base=500000.0)
        assert 'RotaryEmbedding' in repr(rope)
        assert '500000' in repr(rope)


# ─── KVCache ──────────────────────────────────────────────────────────────────

class TestKVCache:
    def test_incremental_shape(self):
        cache = nn.KVCache(2, 64, 2, 16)
        k1, v1 = _rand_t(2,2,4,16)._data, _rand_t(2,2,4,16)._data
        k2, v2 = _rand_t(2,2,4,16)._data, _rand_t(2,2,4,16)._data
        kf, vf = cache.update(k1, v1)
        assert kf.shape == (2, 2, 4, 16)
        kf, vf = cache.update(k2, v2)
        assert kf.shape == (2, 2, 8, 16)

    def test_length(self):
        cache = nn.KVCache(1, 32, 1, 8)
        assert cache.length == 0
        cache.update(_rand_t(1,1,5,8)._data, _rand_t(1,1,5,8)._data)
        assert cache.length == 5
        cache.update(_rand_t(1,1,3,8)._data, _rand_t(1,1,3,8)._data)
        assert cache.length == 8

    def test_reset(self):
        cache = nn.KVCache(1, 32, 1, 8)
        cache.update(_rand_t(1,1,6,8)._data, _rand_t(1,1,6,8)._data)
        cache.reset()
        assert cache.length == 0

    def test_repr(self):
        cache = nn.KVCache(1, 32, 2, 16)
        assert 'KVCache' in repr(cache)


# ─── MultiheadFlashAttention ──────────────────────────────────────────────────

class TestMultiheadFlashAttention:
    def test_mha_shape(self):
        mfa = nn.MultiheadFlashAttention(64, 4, block_size=4)
        out = mfa(_rand_t(2, 8, 64))
        assert out._data.shape == (2, 8, 64)

    def test_gqa_shape(self):
        mfa = nn.MultiheadFlashAttention(64, 4, num_kv_heads=2, block_size=4)
        out = mfa(_rand_t(1, 10, 64))
        assert out._data.shape == (1, 10, 64)

    def test_mqa_shape(self):
        mfa = nn.MultiheadFlashAttention(64, 4, num_kv_heads=1, block_size=4)
        out = mfa(_rand_t(1, 8, 64))
        assert out._data.shape == (1, 8, 64)

    def test_causal_rope(self):
        mfa = nn.MultiheadFlashAttention(
            64, 4, num_kv_heads=2, causal=True, rope=True, block_size=4)
        out = mfa(_rand_t(1, 10, 64))
        assert out._data.shape == (1, 10, 64)

    def test_alibi(self):
        mfa = nn.MultiheadFlashAttention(64, 4, alibi=True, block_size=4)
        out = mfa(_rand_t(1, 8, 64))
        assert out._data.shape == (1, 8, 64)

    def test_mqa_alibi(self):
        mfa = nn.MultiheadFlashAttention(
            64, 4, num_kv_heads=1, alibi=True, block_size=4)
        out = mfa(_rand_t(1, 8, 64))
        assert out._data.shape == (1, 8, 64)

    def test_sliding_window(self):
        mfa = nn.MultiheadFlashAttention(
            64, 4, causal=True, window_size=3, block_size=4)
        out = mfa(_rand_t(1, 8, 64))
        assert out._data.shape == (1, 8, 64)

    def test_cross_attention(self):
        mfa = nn.MultiheadFlashAttention(64, 4, block_size=4)
        q = _rand_t(1, 6, 64)
        ctx = _rand_t(1, 10, 64)
        out = mfa(q, key=ctx, value=ctx)
        assert out._data.shape == (1, 6, 64)

    def test_kvcache_prefill_and_decode(self):
        mfa = nn.MultiheadFlashAttention(
            64, 4, num_kv_heads=2, causal=True, rope=True, block_size=4)
        cache = mfa.make_kv_cache(max_batch_size=1, max_seq_len=32)
        out_pre = mfa(_rand_t(1, 8, 64), kv_cache=cache)
        assert out_pre._data.shape == (1, 8, 64)
        out_step = mfa(_rand_t(1, 1, 64), kv_cache=cache, position_offset=8)
        assert out_step._data.shape == (1, 1, 64)
        assert cache.length == 9

    def test_make_kvcache_dimensions(self):
        mfa = nn.MultiheadFlashAttention(64, 4, num_kv_heads=2, block_size=4)
        cache = mfa.make_kv_cache(max_batch_size=2, max_seq_len=128)
        assert cache.num_kv_heads == 2
        assert cache.head_dim == 16
        assert cache.max_seq_len == 128

    def test_repr_variants(self):
        assert 'GQA' in repr(
            nn.MultiheadFlashAttention(64, 4, num_kv_heads=2, block_size=4))
        assert 'MQA' in repr(
            nn.MultiheadFlashAttention(64, 4, num_kv_heads=1, block_size=4))
        assert 'RoPE' in repr(
            nn.MultiheadFlashAttention(64, 4, rope=True, block_size=4))
        assert 'ALiBi' in repr(
            nn.MultiheadFlashAttention(64, 4, alibi=True, block_size=4))
        assert 'causal' in repr(
            nn.MultiheadFlashAttention(64, 4, causal=True, block_size=4))
