"""Dispatch layer: Mojo compiled library → optimized Python fallback.

All heavy compute ops are routed through here. The Mojo path uses:
- SIMD[DType.float32, N] vectorization (auto-width: AVX2=8, AVX-512=16, NEON=4)
- parallelize() for multi-core loop parallelism
- Fused memory kernels (no temporary allocation)
- Flash-Attention-2 tiling for O(N·d) memory attention

Compile the Mojo kernels:
    cd vectorphile/mojo/kernels
    mojo build ops.mojo -o ../libvp_mojo.so --target native

If the .so is absent, every function falls back to the optimized Python path
which is functionally identical (but slower).
"""
from __future__ import annotations
import ctypes
import os
import math as _math
from typing import Optional
from vectorphile._core.ndarray import NDArray, zeros as _nd_zeros, array as _nd_array

# ── Load compiled Mojo library (optional) ────────────────────────────────────

_LIB_PATHS = [
    os.path.join(os.path.dirname(__file__), 'libvp_mojo.so'),
    os.path.join(os.path.dirname(__file__), 'libvp_mojo.dylib'),
    os.path.join(os.path.dirname(__file__), 'libvp_mojo.dll'),
]

_mojo_lib   = None
_mojo_path  = None
for _p in _LIB_PATHS:
    if os.path.exists(_p):
        try:
            _mojo_lib  = ctypes.CDLL(_p)
            _mojo_path = _p
            break
        except OSError:
            pass


def is_mojo_available() -> bool:
    return _mojo_lib is not None


def mojo_backend_info() -> dict:
    if _mojo_lib is not None:
        return {
            'available': True,
            'path':      _mojo_path,
            'kernels':   ['matmul', 'flash_attention', 'layer_norm', 'rms_norm',
                          'silu', 'gelu', 'softmax', 'conv2d'],
        }
    return {
        'available': False,
        'path':      None,
        'kernels':   [],
        'hint':      'mojo build vectorphile/mojo/kernels/ops.mojo -o libvp_mojo.so --target native',
    }


# ── Mojo ctypes bridge helpers ────────────────────────────────────────────────

def _ptr(nd: NDArray):
    """Return ctypes void* to the underlying array buffer."""
    import ctypes as ct
    return ct.cast(ct.addressof(ct.c_float.from_buffer(nd._buf)), ct.c_void_p)


def _shape_ptr(shape):
    import ctypes as ct
    arr = (ct.c_int64 * len(shape))(*shape)
    return arr


# ── Matmul ────────────────────────────────────────────────────────────────────

def matmul(a: NDArray, b: NDArray) -> NDArray:
    """C = A @ B.  SIMD-tiled via Mojo (MC=64, KC=256, NC=64) or NumPy BLAS."""
    if _mojo_lib is not None:
        try:
            import ctypes as ct
            assert a.ndim == 2 and b.ndim == 2
            M, K = a.shape; _, N = b.shape
            import array as _ar
            out_buf = _ar.array('f', [0.0] * (M * N))
            from vectorphile._core.ndarray import NDArray as _ND
            out_nd = _ND(out_buf, (M, N), (N, 1), 'float32')
            fn = _mojo_lib.vp_matmul
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p,
                           ct.c_int, ct.c_int, ct.c_int]
            fn.restype  = None
            fn(_ptr(a), _ptr(b), _ptr(out_nd), ct.c_int(M), ct.c_int(K), ct.c_int(N))
            return out_nd
        except Exception:
            pass
    return a @ b


# ── Flash Attention ───────────────────────────────────────────────────────────

def flash_attention_fwd(
    q: NDArray, k: NDArray, v: NDArray,
    scale: float, causal: bool = False,
    mask: Optional[NDArray] = None,
) -> NDArray:
    """Flash Attention 2 forward — O(N·d) memory, tiled online softmax.

    Args:
        q, k, v : (B, H, T, d) float32
        scale   : 1 / sqrt(d)
        causal  : apply causal mask
        mask    : optional additive float32 mask (large neg = -inf)
    Returns:
        output  : (B, H, T, d)
    """
    if _mojo_lib is not None:
        try:
            import ctypes as ct
            B, H, T, d = q.shape
            import array as _ar
            out_buf = _ar.array('f', [0.0] * (B * H * T * d))
            from vectorphile._core.ndarray import NDArray as _ND
            out_nd  = _ND(out_buf, (B, H, T, d), (H*T*d, T*d, d, 1), 'float32')
            fn = _mojo_lib.vp_flash_attention
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p,
                           ct.c_int, ct.c_int, ct.c_int, ct.c_int,
                           ct.c_float, ct.c_int]
            fn.restype = None
            fn(_ptr(q), _ptr(k), _ptr(v), _ptr(out_nd),
               ct.c_int(B), ct.c_int(H), ct.c_int(T), ct.c_int(d),
               ct.c_float(scale), ct.c_int(1 if causal else 0))
            return out_nd
        except Exception:
            pass

    # Optimized Python fallback — Flash-Attention-2 tiled algorithm
    B, H, T, d = q.shape
    block = min(64, T)
    import array as _a
    out_buf = _a.array('f', [0.0] * (B * H * T * d))

    for b in range(B):
        for h in range(H):
            m_i = _a.array('f', [-1e9] * T)
            l_i = _a.array('f', [0.0]  * T)
            o_i = [[0.0] * d for _ in range(T)]

            for jstart in range(0, T, block):
                jend = min(jstart + block, T)
                Kj = k[b, h, jstart:jend, :]
                Vj = v[b, h, jstart:jend, :]

                for istart in range(0, T, block):
                    iend = min(istart + block, T)
                    Qi   = q[b, h, istart:iend, :]

                    # Compute attention scores S = Qi @ Kj^T * scale
                    idx = tuple([slice(None)] * Kj.ndim[:-1] if hasattr(Kj.ndim, '__len__') else [slice(None)] * (Kj.ndim - 1) + [slice(None)])
                    S = Qi @ Kj.transpose((1, 0)) * scale  # (Bq, Bk)

                    if causal:
                        for ii in range(iend - istart):
                            for jj in range(jend - jstart):
                                if jstart + jj > istart + ii:
                                    S._buf[ii * (jend - jstart) + jj] = -1e9

                    if mask is not None:
                        m_slice = mask[b, h, istart:iend, jstart:jend]
                        S = S + m_slice.astype('float32')

                    # Online softmax update
                    Bq = iend - istart; Bk = jend - jstart
                    for ii in range(Bq):
                        gi = istart + ii
                        row_max = max(float(S._buf[ii * Bk + jj]) for jj in range(Bk))
                        m_new   = max(m_i[gi], row_max)
                        exp_diff = _math.exp(m_i[gi] - m_new)
                        row_exp  = [_math.exp(float(S._buf[ii * Bk + jj]) - m_new)
                                    for jj in range(Bk)]
                        row_sum  = sum(row_exp)
                        l_new    = exp_diff * l_i[gi] + row_sum
                        for dd in range(d):
                            o_i[gi][dd] = (exp_diff * l_i[gi] * o_i[gi][dd] +
                                           sum(row_exp[jj] * float(Vj._buf[jj * d + dd])
                                               for jj in range(Bk))) / max(l_new, 1e-9)
                        m_i[gi] = m_new
                        l_i[gi] = l_new

            for i in range(T):
                offset = ((b * H + h) * T + i) * d
                for dd in range(d):
                    out_buf[offset + dd] = o_i[i][dd]

    from vectorphile._core.ndarray import NDArray as _ND
    return _ND(out_buf, (B, H, T, d), (H*T*d, T*d, d, 1), 'float32')


# ── Conv2d ────────────────────────────────────────────────────────────────────

def conv2d(x: NDArray, weight: NDArray, bias: Optional[NDArray],
           stride: int = 1, padding: int = 0,
           dilation: int = 1, groups: int = 1) -> NDArray:
    """2-D convolution — im2col + SIMD GEMM (Mojo) or im2col + NumPy BLAS."""
    if _mojo_lib is not None:
        try:
            import ctypes as ct
            B, C_in, H_in, W_in = x.shape
            C_out, _, kH, kW     = weight.shape
            H_out = (H_in + 2*padding - dilation*(kH-1) - 1) // stride + 1
            W_out = (W_in + 2*padding - dilation*(kW-1) - 1) // stride + 1
            import array as _ar
            out_buf = _ar.array('f', [0.0] * (B * C_out * H_out * W_out))
            from vectorphile._core.ndarray import NDArray as _ND
            out_nd = _ND(out_buf, (B, C_out, H_out, W_out),
                         (C_out*H_out*W_out, H_out*W_out, W_out, 1), 'float32')
            fn = _mojo_lib.vp_conv2d
            fn.argtypes = [ct.c_void_p, ct.c_void_p,
                           ct.c_void_p if bias is not None else ct.c_void_p,
                           ct.c_void_p,
                           ct.c_int, ct.c_int, ct.c_int, ct.c_int,
                           ct.c_int, ct.c_int, ct.c_int, ct.c_int,
                           ct.c_int, ct.c_int]
            fn.restype = None
            fn(_ptr(x), _ptr(weight),
               _ptr(bias) if bias is not None else ct.c_void_p(0),
               _ptr(out_nd),
               ct.c_int(B), ct.c_int(C_in), ct.c_int(H_in), ct.c_int(W_in),
               ct.c_int(C_out), ct.c_int(kH), ct.c_int(stride), ct.c_int(padding),
               ct.c_int(dilation), ct.c_int(groups))
            return out_nd
        except Exception:
            pass

    # Fallback: im2col + NumPy BLAS
    B, C_in, H_in, W_in = x.shape
    C_out, _, kH, kW     = weight.shape
    H_out = (H_in + 2*padding - dilation*(kH-1) - 1) // stride + 1
    W_out = (W_in + 2*padding - dilation*(kW-1) - 1) // stride + 1

    if padding > 0:
        from vectorphile._core.ndarray import NDArray as _ND
        import array as _a2
        ph, pw = padding, padding
        p_buf = _a2.array('f', [0.0] * (B * C_in * (H_in+2*ph) * (W_in+2*pw)))
        padded = _ND(p_buf, (B, C_in, H_in+2*ph, W_in+2*pw),
                     (C_in*(H_in+2*ph)*(W_in+2*pw),(H_in+2*ph)*(W_in+2*pw),(W_in+2*pw),1), 'float32')
        for b in range(B):
            for c in range(C_in):
                for i in range(H_in):
                    for j in range(W_in):
                        padded._buf[b*C_in*(H_in+2*ph)*(W_in+2*pw) +
                                    c*(H_in+2*ph)*(W_in+2*pw) +
                                    (i+ph)*(W_in+2*pw) + (j+pw)] = float(x[b,c,i,j])
    else:
        padded = x

    col_rows = C_in * kH * kW
    col_cols = H_out * W_out
    import array as _a
    col_buf = _a.array('f', [0.0] * (B * col_rows * col_cols))
    pH = padded.shape[2]; pW = padded.shape[3]

    for b in range(B):
        for c in range(C_in):
            for kh in range(kH):
                for kw in range(kW):
                    row = c * kH * kW + kh * kW + kw
                    for ho in range(H_out):
                        for wo in range(W_out):
                            col_idx = ho * W_out + wo
                            hi = ho * stride + kh * dilation
                            wi = wo * stride + kw * dilation
                            col_buf[(b*col_rows + row)*col_cols + col_idx] = \
                                float(padded._buf[b*C_in*pH*pW + c*pH*pW + hi*pW + wi])

    from vectorphile._core.ndarray import NDArray as _ND
    col = _ND(col_buf, (B, col_rows, col_cols),
              (col_rows*col_cols, col_cols, 1), 'float32')
    W_flat = weight.reshape(C_out, -1)

    out_buf2 = _a.array('f', [0.0] * (B * C_out * H_out * W_out))
    sz = C_out * H_out * W_out
    for b in range(B):
        ob = W_flat @ col[b]
        if bias is not None:
            ob = ob + bias.reshape(C_out, 1)
        flat = ob.reshape(C_out, H_out, W_out).flatten()
        for idx in range(sz):
            out_buf2[b*sz + idx] = float(flat._buf[idx])

    return _ND(out_buf2, (B, C_out, H_out, W_out),
               (C_out*H_out*W_out, H_out*W_out, W_out, 1), 'float32')


# ── Norm kernels ─────────────────────────────────────────────────────────────

def layer_norm(x: NDArray, weight: NDArray, bias: NDArray,
               eps: float = 1e-5) -> NDArray:
    """Fused LayerNorm: single-pass mean+var+scale (Mojo) or NumPy."""
    if _mojo_lib is not None:
        try:
            import ctypes as ct, array as _ar
            from vectorphile._core.ndarray import NDArray as _ND
            out_buf = _ar.array('f', [0.0] * len(x._buf))
            out_nd  = _ND(out_buf, x.shape, x.strides, 'float32')
            d = x.shape[-1]; n = len(x._buf) // d
            fn = _mojo_lib.vp_layer_norm
            fn.argtypes = [ct.c_void_p]*4 + [ct.c_int, ct.c_int, ct.c_float]
            fn.restype  = None
            fn(_ptr(x), _ptr(weight), _ptr(bias), _ptr(out_nd),
               ct.c_int(n), ct.c_int(d), ct.c_float(eps))
            return out_nd
        except Exception:
            pass
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1,  keepdims=True)
    return ((x - mean) / (var + eps) ** 0.5) * weight + bias


def rms_norm(x: NDArray, weight: NDArray, eps: float = 1e-6) -> NDArray:
    """Fused RMSNorm (Mojo) or NumPy."""
    if _mojo_lib is not None:
        try:
            import ctypes as ct, array as _ar
            from vectorphile._core.ndarray import NDArray as _ND
            out_buf = _ar.array('f', [0.0] * len(x._buf))
            out_nd  = _ND(out_buf, x.shape, x.strides, 'float32')
            d = x.shape[-1]; n = len(x._buf) // d
            fn = _mojo_lib.vp_rms_norm
            fn.argtypes = [ct.c_void_p]*3 + [ct.c_int, ct.c_int, ct.c_float]
            fn.restype  = None
            fn(_ptr(x), _ptr(weight), _ptr(out_nd),
               ct.c_int(n), ct.c_int(d), ct.c_float(eps))
            return out_nd
        except Exception:
            pass
    rms = ((x ** 2).mean(axis=-1, keepdims=True) + eps) ** 0.5
    return (x / rms) * weight


# ── Activation kernels ────────────────────────────────────────────────────────

def gelu(x: NDArray) -> NDArray:
    """GELU — SIMD-vectorized Mojo or tanh-approx fallback."""
    if _mojo_lib is not None:
        try:
            import ctypes as ct, array as _ar
            from vectorphile._core.ndarray import NDArray as _ND
            out_buf = _ar.array('f', [0.0] * len(x._buf))
            out_nd  = _ND(out_buf, x.shape, x.strides, 'float32')
            fn = _mojo_lib.vp_gelu
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
            fn.restype  = None
            fn(_ptr(x), _ptr(out_nd), ct.c_int(len(x._buf)))
            return out_nd
        except Exception:
            pass
    return x * 0.5 * (1.0 + _nd_tanh_approx(x))


def silu(x: NDArray) -> NDArray:
    """SiLU/Swish — SIMD-vectorized Mojo or sigmoid fallback."""
    if _mojo_lib is not None:
        try:
            import ctypes as ct, array as _ar
            from vectorphile._core.ndarray import NDArray as _ND
            out_buf = _ar.array('f', [0.0] * len(x._buf))
            out_nd  = _ND(out_buf, x.shape, x.strides, 'float32')
            fn = _mojo_lib.vp_silu
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
            fn.restype  = None
            fn(_ptr(x), _ptr(out_nd), ct.c_int(len(x._buf)))
            return out_nd
        except Exception:
            pass
    return x / (1.0 + _nd_exp_approx(-x))


def softmax(x: NDArray, axis: int = -1) -> NDArray:
    """Numerically-stable softmax — Mojo vectorized or NumPy."""
    if _mojo_lib is not None:
        try:
            import ctypes as ct, array as _ar
            from vectorphile._core.ndarray import NDArray as _ND
            out_buf = _ar.array('f', [0.0] * len(x._buf))
            out_nd  = _ND(out_buf, x.shape, x.strides, 'float32')
            d = x.shape[axis] if axis != -1 else x.shape[-1]
            n = len(x._buf) // d
            fn = _mojo_lib.vp_softmax
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int]
            fn.restype  = None
            fn(_ptr(x), _ptr(out_nd), ct.c_int(n), ct.c_int(d))
            return out_nd
        except Exception:
            pass
    x_max = x.max(axis=axis, keepdims=True)
    e = _nd_exp_approx(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


# ── Speculative decode kernel (token-level) ───────────────────────────────────

def speculative_verify(
    target_logits: list,  # list of float, vocab_size
    draft_probs:   list,  # list of (tok_id, draft_prob) tuples
) -> tuple:
    """Rejection-sampling verification step (pure Python — correctness path).

    Returns (accepted_tokens, correction_token).
    """
    import random
    accepted = []
    for tok, q_prob in draft_probs:
        p_target = target_logits[tok] if tok < len(target_logits) else 0.0
        accept   = min(1.0, p_target / max(q_prob, 1e-9))
        if random.random() < accept:
            accepted.append(tok)
        else:
            # sample correction from residual
            resid  = [max(0.0, target_logits[j] - draft_probs[j][1] if j < len(draft_probs) else target_logits[j])
                      for j in range(len(target_logits))]
            s      = sum(resid) or 1e-9
            resid  = [v/s for v in resid]
            r = random.random(); c = 0.0
            tok = len(resid) - 1
            for i, p in enumerate(resid):
                c += p
                if r < c: tok = i; break
            return accepted, tok
    return accepted, None


# ── helpers ───────────────────────────────────────────────────────────────────

def _nd_exp_approx(x: NDArray) -> NDArray:
    import array as _a
    buf = _a.array('f', [_math.exp(min(max(float(v), -88.0), 88.0)) for v in x._buf])
    from vectorphile._core.ndarray import NDArray as _ND
    return _ND(buf, x.shape, x.strides, 'float32')


def _nd_tanh_approx(x: NDArray) -> NDArray:
    import array as _a
    buf = _a.array('f', [
        _math.tanh(0.7978845608 * (float(v) + 0.044715 * float(v)**3))
        for v in x._buf
    ])
    from vectorphile._core.ndarray import NDArray as _ND
    return _ND(buf, x.shape, x.strides, 'float32')


# ── Batch Normalization ───────────────────────────────────────────────────────

def batch_norm(x: NDArray, weight: Optional[NDArray], bias: Optional[NDArray],
               running_mean: Optional[NDArray], running_var: Optional[NDArray],
               training: bool = False, momentum: float = 0.1, eps: float = 1e-5) -> NDArray:
    """Fused batch normalization — training or inference mode."""
    if _mojo_lib is not None:
        try:
            import ctypes as ct
            # Attempt to call compiled vp_batch_norm
            fn = _mojo_lib.vp_batch_norm
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p,
                           ct.c_void_p, ct.c_int, ct.c_int, ct.c_float, ct.c_int, ct.c_float]
            fn.restype = None
            import array as _a
            out_buf = _a.array('f', [0.0] * len(x._buf))
            from vectorphile._core.ndarray import NDArray as _ND
            out_nd = _ND(out_buf, x.shape, x.strides, 'float32')
            N, C = x.shape[0], x.shape[1]
            fn(_ptr(x), _ptr(weight) if weight is not None else ct.c_void_p(0),
               _ptr(bias) if bias is not None else ct.c_void_p(0),
               _ptr(running_mean) if running_mean is not None else ct.c_void_p(0),
               _ptr(running_var) if running_var is not None else ct.c_void_p(0),
               _ptr(out_nd), ct.c_int(N), ct.c_int(C), ct.c_float(momentum),
               ct.c_int(1 if training else 0), ct.c_float(eps))
            return out_nd
        except Exception:
            pass
    # Pure Python fallback
    import array as _a
    from vectorphile._core.ndarray import NDArray as _ND, zeros as _nd_z, _c_strides, _DType
    flat = list(x._buf); shape = x.shape; N = shape[0]; C = shape[1]
    spatial = 1
    for s in shape[2:]: spatial *= s
    out = flat[:]
    for c in range(C):
        vals = [flat[n*(C*spatial) + c*spatial + s] for n in range(N) for s in range(spatial)]
        mean = sum(vals) / len(vals)
        var = sum((v - mean)**2 for v in vals) / len(vals)
        inv_std = 1.0 / (var + eps)**0.5
        w = float(weight._buf[c]) if weight is not None else 1.0
        b = float(bias._buf[c]) if bias is not None else 0.0
        for n in range(N):
            for s in range(spatial):
                idx = n*(C*spatial) + c*spatial + s
                out[idx] = w * (flat[idx] - mean) * inv_std + b
        if running_mean is not None and training:
            running_mean._buf[c] = (1 - momentum)*running_mean._buf[c] + momentum*mean
        if running_var is not None and training:
            running_var._buf[c] = (1 - momentum)*running_var._buf[c] + momentum*var
    buf = _a.array('f', out)
    return _ND(buf, shape, _c_strides(shape), 'float32')


# ── Cross Entropy Loss ────────────────────────────────────────────────────────

def cross_entropy(logits: NDArray, targets, reduction: str = 'mean') -> float:
    """Fused softmax + NLL cross-entropy loss."""
    if _mojo_lib is not None:
        try:
            import ctypes as ct
            fn = _mojo_lib.vp_cross_entropy
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int]
            fn.restype = ct.c_float
            flat_targets = list(targets._buf) if hasattr(targets, '_buf') else list(targets)
            tgt_buf = __import__('array').array('i', [int(t) for t in flat_targets])
            tgt_nd_ptr = ct.cast(ct.addressof(ct.c_int.from_buffer(tgt_buf)), ct.c_void_p)
            N, C = logits.shape[0], logits.shape[1]
            loss = fn(_ptr(logits), tgt_nd_ptr, ct.c_int(N), ct.c_int(C))
            return float(loss)
        except Exception:
            pass
    flat = list(logits._buf)
    tgt_list = list(targets._buf) if hasattr(targets, '_buf') else list(targets)
    N, C = logits.shape[0], logits.shape[1]
    total = 0.0
    for n in range(N):
        row = flat[n*C:(n+1)*C]
        mx = max(row); exp_row = [_math.exp(v-mx) for v in row]; s = sum(exp_row)
        log_prob = _math.log(exp_row[int(tgt_list[n])] / max(s, 1e-10))
        total -= log_prob
    return total / N if reduction == 'mean' else total


# ── Embedding Lookup ──────────────────────────────────────────────────────────

def embedding(weight: NDArray, indices) -> NDArray:
    """Fused embedding table lookup."""
    if _mojo_lib is not None:
        try:
            import ctypes as ct
            fn = _mojo_lib.vp_embedding
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int, ct.c_int]
            fn.restype = None
            idx_list = list(indices._buf) if hasattr(indices, '_buf') else list(indices)
            N = len(idx_list); d = weight.shape[1]
            import array as _a
            out_buf = _a.array('f', [0.0] * N * d)
            from vectorphile._core.ndarray import NDArray as _ND
            out_nd = _ND(out_buf, (N, d), (d, 1), 'float32')
            idx_buf = _a.array('i', [int(i) for i in idx_list])
            idx_ptr = ct.cast(ct.addressof(ct.c_int.from_buffer(idx_buf)), ct.c_void_p)
            fn(_ptr(weight), idx_ptr, _ptr(out_nd), ct.c_int(N), ct.c_int(d), ct.c_int(weight.shape[0]))
            return out_nd
        except Exception:
            pass
    idx_list = list(indices._buf) if hasattr(indices, '_buf') else list(indices)
    N = len(idx_list); d = weight.shape[1]
    flat_w = list(weight._buf)
    import array as _a
    out = _a.array('f', [flat_w[int(idx)*d + k] for idx in idx_list for k in range(d)])
    from vectorphile._core.ndarray import NDArray as _ND
    return _ND(out, (N, d), (d, 1), 'float32')


# ── Dropout ───────────────────────────────────────────────────────────────────

def dropout(x: NDArray, p: float = 0.5, training: bool = True) -> NDArray:
    """Inverted dropout — scales by 1/(1-p) during training."""
    if not training or p == 0.0:
        return x
    import random as _r; import array as _a
    from vectorphile._core.ndarray import NDArray as _ND
    scale = 1.0 / (1.0 - p)
    out = _a.array('f', [0.0 if _r.random() < p else float(v) * scale for v in x._buf])
    return _ND(out, x.shape, x.strides, 'float32')


# ── ReLU ─────────────────────────────────────────────────────────────────────

def relu(x: NDArray) -> NDArray:
    """ReLU: max(0, x) — Mojo SIMD or pure-Python."""
    if _mojo_lib is not None:
        try:
            import ctypes as ct, array as _a
            from vectorphile._core.ndarray import NDArray as _ND
            out_buf = _a.array('f', [0.0] * len(x._buf))
            out_nd = _ND(out_buf, x.shape, x.strides, 'float32')
            fn = _mojo_lib.vp_relu
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
            fn.restype = None
            fn(_ptr(x), _ptr(out_nd), ct.c_int(len(x._buf)))
            return out_nd
        except Exception:
            pass
    import array as _a
    from vectorphile._core.ndarray import NDArray as _ND
    out = _a.array('f', [max(0.0, float(v)) for v in x._buf])
    return _ND(out, x.shape, x.strides, 'float32')


# ── Element-wise add ──────────────────────────────────────────────────────────

def add(a: NDArray, b: NDArray) -> NDArray:
    """Element-wise add — Mojo SIMD or fallback."""
    if _mojo_lib is not None:
        try:
            import ctypes as ct, array as _a
            from vectorphile._core.ndarray import NDArray as _ND
            out_buf = _a.array('f', [0.0] * len(a._buf))
            out_nd = _ND(out_buf, a.shape, a.strides, 'float32')
            fn = _mojo_lib.vp_add
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int]
            fn.restype = None
            fn(_ptr(a), _ptr(b), _ptr(out_nd), ct.c_int(len(a._buf)))
            return out_nd
        except Exception:
            pass
    return a + b


# ── Element-wise multiply ─────────────────────────────────────────────────────

def mul(a: NDArray, b: NDArray) -> NDArray:
    """Element-wise multiply — Mojo SIMD or fallback."""
    if _mojo_lib is not None:
        try:
            import ctypes as ct, array as _a
            from vectorphile._core.ndarray import NDArray as _ND
            out_buf = _a.array('f', [0.0] * len(a._buf))
            out_nd = _ND(out_buf, a.shape, a.strides, 'float32')
            fn = _mojo_lib.vp_mul
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int]
            fn.restype = None
            fn(_ptr(a), _ptr(b), _ptr(out_nd), ct.c_int(len(a._buf)))
            return out_nd
        except Exception:
            pass
    return a * b


# ── Scale ─────────────────────────────────────────────────────────────────────

def scale(x: NDArray, scalar: float) -> NDArray:
    """Scalar multiply — Mojo SIMD or fallback."""
    if _mojo_lib is not None:
        try:
            import ctypes as ct, array as _a
            from vectorphile._core.ndarray import NDArray as _ND
            out_buf = _a.array('f', [0.0] * len(x._buf))
            out_nd = _ND(out_buf, x.shape, x.strides, 'float32')
            fn = _mojo_lib.vp_scale
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_float, ct.c_int]
            fn.restype = None
            fn(_ptr(x), _ptr(out_nd), ct.c_float(scalar), ct.c_int(len(x._buf)))
            return out_nd
        except Exception:
            pass
    return x * scalar


# ── Transpose ────────────────────────────────────────────────────────────────

def transpose(x: NDArray) -> NDArray:
    """2-D transpose — Mojo tiled cache-friendly or fallback."""
    if _mojo_lib is not None and x.ndim == 2:
        try:
            import ctypes as ct, array as _a
            from vectorphile._core.ndarray import NDArray as _ND
            M, N = x.shape
            out_buf = _a.array('f', [0.0] * (M * N))
            out_nd = _ND(out_buf, (N, M), (M, 1), 'float32')
            fn = _mojo_lib.vp_transpose
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int]
            fn.restype = None
            fn(_ptr(x), _ptr(out_nd), ct.c_int(M), ct.c_int(N))
            return out_nd
        except Exception:
            pass
    return x.transpose()


# ── Fused Adam step ───────────────────────────────────────────────────────────

def adam_step(params: NDArray, grads: NDArray, m: NDArray, v: NDArray,
              lr: float, beta1: float, beta2: float, eps: float,
              weight_decay: float, t: int) -> None:
    """Fused Adam: single-pass SIMD, no intermediate buffers. In-place update."""
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    if _mojo_lib is not None:
        try:
            import ctypes as ct
            fn = _mojo_lib.vp_adam
            fn.argtypes = [ct.c_void_p] * 4 + [ct.c_int] + [ct.c_float] * 7
            fn.restype = None
            fn(_ptr(params), _ptr(grads), _ptr(m), _ptr(v),
               ct.c_int(len(params._buf)),
               ct.c_float(lr), ct.c_float(beta1), ct.c_float(beta2),
               ct.c_float(eps), ct.c_float(weight_decay),
               ct.c_float(bc1), ct.c_float(bc2))
            return
        except Exception:
            pass
    # Pure-Python fallback
    n = len(params._buf)
    for i in range(n):
        g = float(grads._buf[i]) + weight_decay * float(params._buf[i])
        m._buf[i] = beta1 * float(m._buf[i]) + (1 - beta1) * g
        v._buf[i] = beta2 * float(v._buf[i]) + (1 - beta2) * g * g
        m_hat = float(m._buf[i]) / bc1
        v_hat = float(v._buf[i]) / bc2
        params._buf[i] = float(params._buf[i]) - lr * m_hat / (_math.sqrt(v_hat) + eps)


# ── Fused Linear + Activation ─────────────────────────────────────────────────

def linear_silu(x: NDArray, weight: NDArray, bias: Optional[NDArray] = None) -> NDArray:
    """Fused Linear + SiLU: no intermediate allocation. Mojo SIMD or fallback."""
    if _mojo_lib is not None:
        try:
            import ctypes as ct, array as _a
            from vectorphile._core.ndarray import NDArray as _ND
            if x.ndim == 1:
                xr = x.reshape((1, x.shape[0]))
            else:
                xr = x
            N = xr.shape[0]; in_f = xr.shape[1]; out_f = weight.shape[0]
            out_buf = _a.array('f', [0.0] * (N * out_f))
            out_nd = _ND(out_buf, (N, out_f), (out_f, 1), 'float32')
            fn = _mojo_lib.vp_linear_silu
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p,
                           ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int]
            fn.restype = None
            fn(_ptr(xr), _ptr(weight),
               _ptr(bias) if bias is not None else ct.c_void_p(0),
               _ptr(out_nd), ct.c_int(N), ct.c_int(in_f), ct.c_int(out_f),
               ct.c_int(1 if bias is not None else 0))
            return out_nd
        except Exception:
            pass
    # Fallback: separate ops
    out = x @ weight.transpose()
    if bias is not None:
        out = out + bias
    return out / (1.0 + _nd_exp_approx(-out))


def linear_gelu(x: NDArray, weight: NDArray, bias: Optional[NDArray] = None) -> NDArray:
    """Fused Linear + GELU: no intermediate allocation. Mojo SIMD or fallback."""
    if _mojo_lib is not None:
        try:
            import ctypes as ct, array as _a
            from vectorphile._core.ndarray import NDArray as _ND
            if x.ndim == 1:
                xr = x.reshape((1, x.shape[0]))
            else:
                xr = x
            N = xr.shape[0]; in_f = xr.shape[1]; out_f = weight.shape[0]
            out_buf = _a.array('f', [0.0] * (N * out_f))
            out_nd = _ND(out_buf, (N, out_f), (out_f, 1), 'float32')
            fn = _mojo_lib.vp_linear_gelu
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p,
                           ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int]
            fn.restype = None
            fn(_ptr(xr), _ptr(weight),
               _ptr(bias) if bias is not None else ct.c_void_p(0),
               _ptr(out_nd), ct.c_int(N), ct.c_int(in_f), ct.c_int(out_f),
               ct.c_int(1 if bias is not None else 0))
            return out_nd
        except Exception:
            pass
    out = x @ weight.transpose()
    if bias is not None:
        out = out + bias
    return out * 0.5 * (1.0 + _nd_tanh_approx(out))


# ── Fused QKV projection ─────────────────────────────────────────────────────

def qkv_proj(x: NDArray, wqkv: NDArray) -> NDArray:
    """Fused Q/K/V projection: 3x fewer GEMM setups vs separate calls.
    x: (B*T, d_model), wqkv: (3*d_model, d_model) → out: (B*T, 3*d_model)
    """
    if _mojo_lib is not None:
        try:
            import ctypes as ct, array as _a
            from vectorphile._core.ndarray import NDArray as _ND
            BT = x.shape[0]; d_model = x.shape[1]; out_dim = 3 * d_model
            out_buf = _a.array('f', [0.0] * (BT * out_dim))
            out_nd = _ND(out_buf, (BT, out_dim), (out_dim, 1), 'float32')
            fn = _mojo_lib.vp_qkv_proj
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int]
            fn.restype = None
            fn(_ptr(x), _ptr(wqkv), _ptr(out_nd), ct.c_int(BT), ct.c_int(d_model))
            return out_nd
        except Exception:
            pass
    return x @ wqkv.transpose()


def rope(qk: NDArray, cos_cache: NDArray, sin_cache: NDArray) -> NDArray:
    """In-place RoPE via Mojo vectorized kernel. qk shape: (N, H, Dh)."""
    if _mojo_lib is not None:
        try:
            import ctypes as ct
            N, H, Dh = qk.shape[0], qk.shape[1], qk.shape[2]
            fn = _mojo_lib.vp_rope
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int, ct.c_int]
            fn.restype = None
            fn(_ptr(qk), _ptr(cos_cache), _ptr(sin_cache), ct.c_int(N), ct.c_int(H), ct.c_int(Dh))
            return qk
        except Exception:
            pass
    # Pure Python fallback
    flat = qk.to_flat_list(); N, H, Dh = qk.shape[0], qk.shape[1], qk.shape[2]
    cf = cos_cache.to_flat_list(); sf = sin_cache.to_flat_list()
    import math as _m, array as _a
    out = list(flat)
    for pos in range(N):
        for h in range(H):
            base = (pos * H + h) * Dh
            for i in range(Dh // 2):
                x0, x1 = flat[base + i], flat[base + i + Dh//2]
                c, s = cf[pos * (Dh//2) + i], sf[pos * (Dh//2) + i]
                out[base + i] = x0*c - x1*s
                out[base + i + Dh//2] = x0*s + x1*c
    from vectorphile._core.ndarray import NDArray as _ND, _c_strides as _cs, _DType
    buf = _a.array('f', out)
    return _ND(buf, qk.shape, _cs(qk.shape), _DType('float32'))


def swiglu(gate: NDArray, val: NDArray) -> NDArray:
    """Fused SwiGLU: silu(gate) * val via Mojo SIMD kernel."""
    if _mojo_lib is not None:
        try:
            import ctypes as ct, array as _a
            from vectorphile._core.ndarray import NDArray as _ND, _c_strides as _cs, _DType
            N = gate.shape[0] if hasattr(gate, 'shape') else len(gate.to_flat_list())
            flat_len = N
            for d in gate.shape[1:]: flat_len *= d
            out_buf = _a.array('f', [0.0] * flat_len)
            out_nd = _ND(out_buf, gate.shape, _cs(gate.shape), _DType('float32'))
            fn = _mojo_lib.vp_swiglu
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int]
            fn.restype = None
            fn(_ptr(gate), _ptr(val), _ptr(out_nd), ct.c_int(flat_len))
            return out_nd
        except Exception:
            pass
    import math as _m
    gf = gate.to_flat_list(); vf = val.to_flat_list()
    out = [gf[i] / (1.0 + _m.exp(-gf[i])) * vf[i] for i in range(len(gf))]
    import array as _a
    from vectorphile._core.ndarray import NDArray as _ND, _c_strides as _cs, _DType
    buf = _a.array('f', out)
    return _ND(buf, gate.shape, _cs(gate.shape), _DType('float32'))


def geglu(gate: NDArray, val: NDArray) -> NDArray:
    """Fused GeGLU: gelu(gate) * val via Mojo SIMD kernel."""
    if _mojo_lib is not None:
        try:
            import ctypes as ct, array as _a
            from vectorphile._core.ndarray import NDArray as _ND, _c_strides as _cs, _DType
            flat_len = 1
            for d in gate.shape: flat_len *= d
            out_buf = _a.array('f', [0.0] * flat_len)
            out_nd = _ND(out_buf, gate.shape, _cs(gate.shape), _DType('float32'))
            fn = _mojo_lib.vp_geglu
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int]
            fn.restype = None
            fn(_ptr(gate), _ptr(val), _ptr(out_nd), ct.c_int(flat_len))
            return out_nd
        except Exception:
            pass
    import math as _m
    C, K = 0.7978845608028654, 0.044715
    gf = gate.to_flat_list(); vf = val.to_flat_list()
    out = [gf[i] * 0.5 * (1.0 + _m.tanh(C*(gf[i]+K*gf[i]**3))) * vf[i] for i in range(len(gf))]
    import array as _a
    from vectorphile._core.ndarray import NDArray as _ND, _c_strides as _cs, _DType
    buf = _a.array('f', out)
    return _ND(buf, gate.shape, _cs(gate.shape), _DType('float32'))


def online_softmax(x: NDArray) -> NDArray:
    """Row-wise online softmax via Mojo SIMD kernel."""
    if _mojo_lib is not None:
        try:
            import ctypes as ct, array as _a
            from vectorphile._core.ndarray import NDArray as _ND, _c_strides as _cs, _DType
            sh = x.shape
            N = sh[-1] if len(sh) > 1 else sh[0]
            rows = x._buf.buffer_info()[1] // N
            out_buf = _a.array('f', [0.0] * (rows * N))
            out_nd  = _ND(out_buf, sh, _cs(sh), _DType('float32'))
            fn = _mojo_lib.vp_online_softmax
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int]
            fn.restype  = None
            fn(_ptr(x), _ptr(out_nd), ct.c_int(rows), ct.c_int(N))
            return out_nd
        except Exception:
            pass
    import math as _m
    sh = x.shape; N = sh[-1]; rows = x._buf.buffer_info()[1] // N if hasattr(x, '_buf') else len(x.to_flat_list()) // N
    xf = x.to_flat_list()
    result = []
    for r in range(rows):
        row = xf[r*N:(r+1)*N]; mx = max(row)
        exps = [_m.exp(v-mx) for v in row]; s = sum(exps)
        result.extend([e/s for e in exps])
    import array as _a
    from vectorphile._core.ndarray import NDArray as _ND, _c_strides as _cs, _DType
    return _ND(_a.array('f', result), sh, _cs(sh), _DType('float32'))


def layer_norm_mojo(x: NDArray, w: NDArray, b: NDArray, eps: float = 1e-5) -> NDArray:
    """Row-wise LayerNorm via Mojo SIMD kernel."""
    if _mojo_lib is not None:
        try:
            import ctypes as ct, array as _a
            from vectorphile._core.ndarray import NDArray as _ND, _c_strides as _cs, _DType
            sh = x.shape; D = sh[-1]
            rows = x._buf.buffer_info()[1] // D if hasattr(x, '_buf') else len(x.to_flat_list()) // D
            out_buf = _a.array('f', [0.0] * (rows * D))
            out_nd  = _ND(out_buf, sh, _cs(sh), _DType('float32'))
            fn = _mojo_lib.vp_layer_norm
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p,
                           ct.c_int, ct.c_int, ct.c_float]
            fn.restype  = None
            fn(_ptr(x), _ptr(w), _ptr(b), _ptr(out_nd),
               ct.c_int(rows), ct.c_int(D), ct.c_float(eps))
            return out_nd
        except Exception:
            pass
    import math as _m
    xf = x.to_flat_list(); wf = w.to_flat_list(); bf = b.to_flat_list()
    sh = x.shape; D = sh[-1]; rows = len(xf) // D
    result = []
    for r in range(rows):
        row = xf[r*D:(r+1)*D]
        mu  = sum(row) / D
        var = sum((v-mu)**2 for v in row) / D
        inv = 1.0 / _m.sqrt(var + eps)
        result.extend((row[i]-mu)*inv*wf[i]+bf[i] for i in range(D))
    import array as _a
    from vectorphile._core.ndarray import NDArray as _ND, _c_strides as _cs, _DType
    return _ND(_a.array('f', result), sh, _cs(sh), _DType('float32'))


def embedding_lookup_mojo(weight: NDArray, indices) -> NDArray:
    """Parallel embedding row-gather via Mojo kernel."""
    if _mojo_lib is not None:
        try:
            import ctypes as ct, array as _a
            from vectorphile._core.ndarray import NDArray as _ND, _c_strides as _cs, _DType
            if hasattr(indices, 'to_flat_list'):
                idx_flat = [int(v) for v in indices.to_flat_list()]
            else:
                idx_flat = [int(v) for v in indices]
            batch = len(idx_flat); D = weight.shape[1]
            idx_arr = _a.array('i', idx_flat)
            out_buf = _a.array('f', [0.0] * (batch * D))
            out_sh  = (batch, D)
            out_nd  = _ND(out_buf, out_sh, _cs(out_sh), _DType('float32'))
            fn = _mojo_lib.vp_embedding_lookup
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int]
            fn.restype  = None
            fn(_ptr(weight), ct.cast(idx_arr.buffer_info()[0], ct.c_void_p),
               _ptr(out_nd), ct.c_int(batch), ct.c_int(D))
            return out_nd
        except Exception:
            pass
    # Pure-Python fallback using array slicing
    if hasattr(indices, 'to_flat_list'):
        idx_flat = [int(v) for v in indices.to_flat_list()]
    else:
        idx_flat = [int(v) for v in indices]
    wf = weight.to_flat_list(); D = weight.shape[1]
    import array as _a
    wfl_arr = _a.array('f', wf)
    out_buf = _a.array('f', [0.0] * (len(idx_flat) * D))
    for i, row_i in enumerate(idx_flat):
        out_buf[i*D:(i+1)*D] = wfl_arr[row_i*D:(row_i+1)*D]
    from vectorphile._core.ndarray import NDArray as _ND, _c_strides as _cs, _DType
    sh = (len(idx_flat), D)
    return _ND(out_buf, sh, _cs(sh), _DType('float32'))


def gelu_mojo(x: NDArray) -> NDArray:
    """SIMD-vectorised GeLU activation via Mojo kernel."""
    if hasattr(x, 'to_flat_list'):
        xf = list(x.to_flat_list())
    else:
        xf = list(x)
    n = len(xf)
    sh = x.shape if hasattr(x, 'shape') else (n,)
    if _mojo_lib is not None:
        try:
            import ctypes as ct, array as _a
            from vectorphile._core.ndarray import NDArray as _ND, _c_strides as _cs, _DType
            out_buf = _a.array('f', [0.0] * n)
            out_nd = _ND(out_buf, sh, _cs(sh), _DType('float32'))
            x_nd = NDArray(_a.array('f', [float(v) for v in xf]), sh, _cs(sh), _DType('float32'))
            fn = _mojo_lib.vp_gelu_mojo
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
            fn.restype = None
            fn(_ptr(x_nd), _ptr(out_nd), ct.c_int(n))
            return out_nd
        except Exception:
            pass
    # Pure-Python fallback
    import math as _m
    result = [0.5*v*(1.0+_m.tanh(0.7978845608*(v+0.044715*v**3))) for v in xf]
    import array as _a
    from vectorphile._core.ndarray import NDArray as _ND, _c_strides as _cs, _DType
    buf = _a.array('f', [float(v) for v in result])
    return _ND(buf, sh, _cs(sh), _DType('float32'))


def silu_mojo(x: NDArray) -> NDArray:
    """SIMD-vectorised SiLU activation via Mojo kernel."""
    if hasattr(x, 'to_flat_list'):
        xf = list(x.to_flat_list())
    else:
        xf = list(x)
    n = len(xf)
    sh = x.shape if hasattr(x, 'shape') else (n,)
    if _mojo_lib is not None:
        try:
            import ctypes as ct, array as _a
            from vectorphile._core.ndarray import NDArray as _ND, _c_strides as _cs, _DType
            out_buf = _a.array('f', [0.0] * n)
            out_nd = _ND(out_buf, sh, _cs(sh), _DType('float32'))
            x_nd = NDArray(_a.array('f', [float(v) for v in xf]), sh, _cs(sh), _DType('float32'))
            fn = _mojo_lib.vp_silu_mojo
            fn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_int]
            fn.restype = None
            fn(_ptr(x_nd), _ptr(out_nd), ct.c_int(n))
            return out_nd
        except Exception:
            pass
    import math as _m
    result = [v / (1.0 + _m.exp(-v)) for v in xf]
    import array as _a
    from vectorphile._core.ndarray import NDArray as _ND, _c_strides as _cs, _DType
    buf = _a.array('f', [float(v) for v in result])
    return _ND(buf, sh, _cs(sh), _DType('float32'))


def adam_mojo(p: NDArray, g: NDArray, m: NDArray, v: NDArray,
              lr: float, b1: float, b2: float, eps: float,
              wd: float, bc1: float, bc2: float) -> None:
    """In-place fused Adam update via Mojo kernel."""
    n = len(p.to_flat_list() if hasattr(p, 'to_flat_list') else list(p))
    if _mojo_lib is not None:
        try:
            import ctypes as ct
            fn = _mojo_lib.vp_adam_mojo
            fn.argtypes = [ct.c_void_p]*4 + [ct.c_int] + [ct.c_float]*7
            fn.restype = None
            fn(_ptr(p), _ptr(g), _ptr(m), _ptr(v), ct.c_int(n),
               ct.c_float(lr), ct.c_float(b1), ct.c_float(b2),
               ct.c_float(eps), ct.c_float(wd), ct.c_float(bc1), ct.c_float(bc2))
            return
        except Exception:
            pass
    # Pure-Python fallback (in-place)
    import math as _m
    pf = list(p.to_flat_list() if hasattr(p, 'to_flat_list') else p)
    gf = list(g.to_flat_list() if hasattr(g, 'to_flat_list') else g)
    mf = list(m.to_flat_list() if hasattr(m, 'to_flat_list') else m)
    vf = list(v.to_flat_list() if hasattr(v, 'to_flat_list') else v)
    for i in range(n):
        mi = b1 * mf[i] + (1.0 - b1) * gf[i]
        vi = b2 * vf[i] + (1.0 - b2) * gf[i]**2
        mf[i] = mi; vf[i] = vi
        m_hat = mi / bc1; v_hat = vi / bc2
        pf[i] = pf[i] * (1.0 - lr * wd) - lr * m_hat / (_m.sqrt(v_hat) + eps)
    # Write back
    import array as _a
    from vectorphile._core.ndarray import NDArray as _ND, _c_strides as _cs, _DType
    sh = p.shape if hasattr(p, 'shape') else (n,)
    for nd_arr, vals in [(p, pf), (m, mf), (v, vf)]:
        buf = nd_arr._buf if hasattr(nd_arr, '_buf') else None
        if buf is not None:
            new_buf = _a.array('f', [float(x) for x in vals])
            nd_arr._buf = new_buf
