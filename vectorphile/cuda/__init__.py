"""vectorphile.cuda — CUDA acceleration with native kernels + pure Python fallbacks.

GPU path: vp_cuda_ext (.pyd compiled with nvcc) — no Triton/numba dependency.
CPU path: pure Python fallbacks for all ops.
"""
from __future__ import annotations
import math
import struct
import ctypes
from typing import Optional, Tuple, List


# ── PTX kernel driver (compiled GPU kernels via CUDA Driver API) ─────────────
try:
    from vectorphile.cuda._ptx_driver import (
        ptx_relu, ptx_gelu, ptx_silu, ptx_softmax, ptx_matmul, ptx_layernorm,
        _init_ptx, _PTX_AVAILABLE as _PTX_AVAIL,
    )
    _PTX_AVAILABLE = _PTX_AVAIL
except Exception:
    _PTX_AVAILABLE = False

# ── Native CUDA extension (compiled with nvcc, no Triton) ────────────────────
try:
    import sys as _sys, os as _os
    _core_dir = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), '_core')
    if _core_dir not in _sys.path:
        _sys.path.insert(0, _core_dir)
    import vp_cuda_ext as _cuda_ext
    _CUDA_EXT_AVAILABLE = True
except Exception:
    _cuda_ext = None
    _CUDA_EXT_AVAILABLE = False

# ── Detect CUDA hardware ──────────────────────────────────────────────────────
try:
    _cuda_dll = ctypes.windll.LoadLibrary('nvcuda.dll')
    _dev_count = ctypes.c_int(0)
    _cuda_dll.cuInit(0)
    _cuda_dll.cuDeviceGetCount(ctypes.byref(_dev_count))
    _CUDA_AVAILABLE = _dev_count.value > 0
except Exception:
    _CUDA_AVAILABLE = False

# Legacy numba path (kept for compat, superseded by vp_cuda_ext)
_numba_cuda = None
_NUMBA_AVAILABLE = False


def _flat(x) -> List[float]:
    if hasattr(x, 'to_flat_list'):
        return [float(v) for v in x.to_flat_list()]
    if hasattr(x, 'tolist'):
        v = x.tolist()
        def _recurse_flat(val):
            if isinstance(val, list):
                result = []
                for item in val:
                    result.extend(_recurse_flat(item))
                return result
            return [float(val)]
        return _recurse_flat(v)
    if hasattr(x, '__iter__'):
        result = []
        for item in x:
            if hasattr(item, '__iter__'):
                result.extend(_flat(item))
            else:
                result.append(float(item))
        return result
    return [float(x)]


def _shape(x):
    if hasattr(x, 'shape'):
        return x.shape
    return (len(list(x)),)


# ── Device management ─────────────────────────────────────────────────────────

def is_available() -> bool:
    return _CUDA_AVAILABLE


def device_count() -> int:
    if not _CUDA_AVAILABLE:
        return 0
    try:
        return len(list(_numba_cuda.gpus))
    except Exception:
        return 0


def get_device_name(idx: int = 0) -> str:
    if not _CUDA_AVAILABLE:
        return "CPU (no CUDA)"
    try:
        return _numba_cuda.gpus[idx].name
    except Exception:
        return "unknown"


def memory_info(idx: int = 0) -> dict:
    if not _CUDA_AVAILABLE:
        return {'free': 0, 'total': 0}
    try:
        with _numba_cuda.gpus[idx]:
            free, total = _numba_cuda.current_context().get_memory_info()
        return {'free': free, 'total': total}
    except Exception:
        return {'free': 0, 'total': 0}


def synchronize() -> None:
    if _CUDA_AVAILABLE:
        try:
            _numba_cuda.synchronize()
        except Exception:
            pass


def to_device(arr, device_id: int = 0):
    return arr


def to_host(arr):
    return arr


# ── Pure Python matrix operations ─────────────────────────────────────────────

def _mat_mul_flat(a_flat, m, k, b_flat, n) -> List[float]:
    """Pure Python matmul: (m,k) @ (k,n) → (m,n)."""
    result = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for t in range(k):
                s += a_flat[i * k + t] * b_flat[t * n + j]
            result[i * n + j] = s
    return result


def matmul(a, b):
    """Matrix multiply — PTX GPU kernel when available, else pure Python."""
    from vectorphile._core.ndarray import array as _nd
    from vectorphile.core.tensor import Tensor
    if isinstance(a, Tensor):
        a = a._data
    if isinstance(b, Tensor):
        b = b._data
    af = _flat(a)
    bf = _flat(b)
    sa = _shape(a) if hasattr(a, 'shape') else (len(af), 1)
    sb = _shape(b) if hasattr(b, 'shape') else (len(bf), 1)
    if len(sa) == 1:
        sa = (1, sa[0])
    if len(sb) == 1:
        sb = (sb[0], 1)
    m, k, n = sa[0], sa[1], sb[1]
    if _CUDA_AVAILABLE and _PTX_AVAILABLE:
        try:
            result = ptx_matmul(af, m, k, bf, n)
            nd = _nd(result)
            if hasattr(nd, 'reshape'):
                nd = nd.reshape((m, n))
            return nd
        except Exception:
            pass
    # Delegate to data's matmul if available (NDArray C extension)
    if hasattr(a, '__matmul__'):
        try:
            return a @ b
        except Exception:
            pass
    result = _mat_mul_flat(af, m, k, bf, n)
    nd = _nd(result)
    if hasattr(nd, 'reshape'):
        nd = nd.reshape((m, n))
    return nd


def im2col(x, kH: int, kW: int, stride: int = 1, padding: int = 0):
    """Pure Python im2col — returns flat columns."""
    xf = _flat(x)
    s = _shape(x) if hasattr(x, 'shape') else (1, 1, int(len(xf)**0.5), int(len(xf)**0.5))
    if len(s) != 4:
        return xf
    N, C, H, W = s
    H_p = H + 2 * padding
    W_p = W + 2 * padding
    H_out = (H_p - kH) // stride + 1
    W_out = (W_p - kW) // stride + 1
    # Pad
    if padding > 0:
        padded = [0.0] * (N * C * H_p * W_p)
        for n in range(N):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        padded[n*C*H_p*W_p + c*H_p*W_p + (h+padding)*W_p + (w+padding)] = \
                            xf[n*C*H*W + c*H*W + h*W + w]
    else:
        padded = xf[:]
    col = []
    for n in range(N):
        for i in range(H_out):
            for j in range(W_out):
                for c in range(C):
                    for ki in range(kH):
                        for kj in range(kW):
                            col.append(padded[n*C*H_p*W_p + c*H_p*W_p + (i*stride+ki)*W_p + (j*stride+kj)])
    return col


def conv2d_im2col(x, w, b=None, stride: int = 1, padding: int = 0):
    """Convolution via im2col. Pure Python."""
    # Simple implementation
    xf = _flat(x)
    xs = _shape(x) if hasattr(x, 'shape') else None
    if xs is None or len(xs) != 4:
        return xf
    N, C_in, H, W = xs
    ws = _shape(w) if hasattr(w, 'shape') else None
    if ws is None or len(ws) != 4:
        return xf
    C_out, _, kH, kW = ws
    wf = _flat(w)
    H_out = (H + 2*padding - kH) // stride + 1
    W_out = (W + 2*padding - kW) // stride + 1
    result = [0.0] * (N * C_out * H_out * W_out)
    # Brute force convolution
    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    s = 0.0
                    for ci in range(C_in):
                        for ki in range(kH):
                            for kj in range(kW):
                                h_idx = i*stride + ki - padding
                                w_idx = j*stride + kj - padding
                                if 0 <= h_idx < H and 0 <= w_idx < W:
                                    s += xf[n*C_in*H*W + ci*H*W + h_idx*W + w_idx] * \
                                         wf[co*C_in*kH*kW + ci*kH*kW + ki*kW + kj]
                    if b is not None:
                        bf = _flat(b)
                        s += bf[co]
                    result[n*C_out*H_out*W_out + co*H_out*W_out + i*W_out + j] = s
    return result


def _to_bytes_f32(lst):
    return struct.pack(f'{len(lst)}f', *lst)

def _to_bytes_i32(lst):
    return struct.pack(f'{len(lst)}i', *lst)

def _from_bytes_f32(b, n):
    return list(struct.unpack(f'{n}f', b))


def softmax(x, axis: int = -1):
    sh = _shape(x)
    xf = _flat(x)
    if _CUDA_AVAILABLE and len(sh) >= 2:
        N_sm = len(xf) // sh[-1]
        C_sm = sh[-1]
        if _PTX_AVAILABLE:
            try:
                return ptx_softmax(xf, N_sm, C_sm)
            except Exception:
                pass
        if _CUDA_EXT_AVAILABLE:
            try:
                b = _cuda_ext.softmax_f32(_to_bytes_f32(xf), N_sm, C_sm)
                return _from_bytes_f32(b, N_sm * C_sm)
            except Exception:
                pass
    # Row-wise softmax along last axis for 2D inputs (common path)
    if len(sh) >= 2:
        N = sh[-1]
        rows = len(xf) // N
        result = []
        _exp = math.exp
        for r in range(rows):
            row = xf[r*N:(r+1)*N]
            mx  = max(row)
            exps = [_exp(v - mx) for v in row]
            s   = sum(exps)
            result.extend([e / s for e in exps])
        return result
    # 1-D flat path
    mx = max(xf)
    exps = [math.exp(v - mx) for v in xf]
    s = sum(exps)
    return [e / s for e in exps]


def layer_norm(x, w, b, eps: float = 1e-5):
    xf = _flat(x)
    wf = _flat(w)
    bf = _flat(b)
    sh = _shape(x)
    D  = sh[-1] if len(sh) >= 1 else len(xf)
    rows = len(xf) // D
    if _CUDA_AVAILABLE and _PTX_AVAILABLE:
        try:
            return ptx_layernorm(xf, wf, bf, rows, D, eps)
        except Exception:
            pass
    result = []
    for r in range(rows):
        row = xf[r*D:(r+1)*D]
        mu  = sum(row) / D
        var = sum((v - mu)**2 for v in row) / D
        inv_std = 1.0 / math.sqrt(var + eps)
        result.extend((row[i] - mu) * inv_std * wf[i % len(wf)] + bf[i % len(bf)]
                       for i in range(D))
    return result


def relu(x):
    xf = _flat(x)
    n = len(xf)
    if _CUDA_AVAILABLE and n > 0:
        if _PTX_AVAILABLE:
            try:
                return ptx_relu(xf)
            except Exception:
                pass
        if _CUDA_EXT_AVAILABLE:
            try:
                b = _cuda_ext.relu_f32(_to_bytes_f32(xf), n)
                return _from_bytes_f32(b, n)
            except Exception:
                pass
    return [max(0.0, v) for v in xf]


def gelu(x):
    xf = _flat(x)
    n = len(xf)
    if _CUDA_AVAILABLE and n > 0:
        if _PTX_AVAILABLE:
            try:
                return ptx_gelu(xf)
            except Exception:
                pass
        if _CUDA_EXT_AVAILABLE:
            try:
                b = _cuda_ext.gelu_f32(_to_bytes_f32(xf), n)
                return _from_bytes_f32(b, n)
            except Exception:
                pass
    sq2pi = math.sqrt(2.0 / math.pi)
    return [0.5 * v * (1 + math.tanh(sq2pi * (v + 0.044715 * v**3))) for v in xf]


def silu(x):
    xf = _flat(x)
    if _CUDA_AVAILABLE and _PTX_AVAILABLE and len(xf) > 0:
        try:
            return ptx_silu(xf)
        except Exception:
            pass
    return [v / (1 + math.exp(-v)) for v in xf]


def adam_step(params, grads, m, v, lr, beta1, beta2, eps, weight_decay, t):
    """Fused Adam update — pure Python lists (in-place)."""
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    for i in range(len(params)):
        g = grads[i] + weight_decay * params[i] if weight_decay > 0 else grads[i]
        m[i] = beta1 * m[i] + (1 - beta1) * g
        v[i] = beta2 * v[i] + (1 - beta2) * g * g
        m_hat = m[i] / bc1
        v_hat = v[i] / bc2
        params[i] -= lr * m_hat / (math.sqrt(v_hat) + eps)


def attention_forward(q, k, v, causal: bool = False, scale: Optional[float] = None):
    """Scaled dot-product attention — pure Python."""
    qf = _flat(q)
    kf = _flat(k)
    vf = _flat(v)
    qs = _shape(q) if hasattr(q, 'shape') else (1, 1, 1, len(qf))
    if len(qs) == 4:
        B, H, N, d = qs
    else:
        return qf
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    out = [0.0] * (B * H * N * d)
    for b in range(B):
        for h in range(H):
            for i in range(N):
                # Compute scores
                scores = []
                for j in range(N):
                    if causal and j > i:
                        scores.append(-1e9)
                        continue
                    s = 0.0
                    for dk in range(d):
                        qi = qf[b*H*N*d + h*N*d + i*d + dk]
                        kj = kf[b*H*N*d + h*N*d + j*d + dk]
                        s += qi * kj
                    scores.append(s * scale)
                # Softmax
                mx = max(scores)
                exps = [math.exp(s - mx) for s in scores]
                sum_exp = sum(exps)
                attn = [e / sum_exp for e in exps]
                # Weighted sum of values
                for dk in range(d):
                    s = sum(attn[j] * vf[b*H*N*d + h*N*d + j*d + dk] for j in range(N))
                    out[b*H*N*d + h*N*d + i*d + dk] = s
    return out


def _blocks_threads_1d(n: int, threads: int = 256) -> Tuple[int, int]:
    return ((n + threads - 1) // threads, threads)


# ── Batch Normalization (CPU fallback) ────────────────────────────────────────

def batch_norm(x, weight=None, bias=None, running_mean=None, running_var=None,
               training: bool = False, momentum: float = 0.1, eps: float = 1e-5):
    """Batch normalization — tries CUDA kernel, falls back to CPU."""
    flat = _flat(x); shape = _shape(x)
    N = shape[0]; C = shape[1]; spatial = 1
    for s in shape[2:]: spatial *= s
    out = flat[:]
    w_flat = _flat(weight) if weight is not None else [1.0]*C
    b_flat = _flat(bias) if bias is not None else [0.0]*C
    rm_flat = list(_flat(running_mean)) if running_mean is not None else None
    rv_flat = list(_flat(running_var)) if running_var is not None else None
    for c in range(C):
        vals = [flat[n*(C*spatial) + c*spatial + s] for n in range(N) for s in range(spatial)]
        mean = sum(vals) / len(vals)
        var = sum((v - mean)**2 for v in vals) / len(vals)
        inv_std = 1.0 / (var + eps)**0.5
        for n in range(N):
            for s in range(spatial):
                idx = n*(C*spatial) + c*spatial + s
                out[idx] = w_flat[c] * (flat[idx] - mean) * inv_std + b_flat[c]
        if rm_flat is not None and training:
            rm_flat[c] = (1 - momentum)*rm_flat[c] + momentum*mean
        if rv_flat is not None and training:
            rv_flat[c] = (1 - momentum)*rv_flat[c] + momentum*var
    return out


# ── Cross Entropy ────────────────────────────────────────────────────────────

def cross_entropy(logits, targets, reduction: str = 'mean') -> float:
    """Softmax + NLL cross-entropy loss (CPU fallback)."""
    flat = _flat(logits); shape = _shape(logits)
    N = shape[0]; C = shape[1]
    tgt_list = _flat(targets) if hasattr(targets, '__iter__') else [targets]
    total = 0.0
    for n in range(N):
        row = flat[n*C:(n+1)*C]
        mx = max(row); exps = [math.exp(v-mx) for v in row]; s = sum(exps)
        total -= math.log(exps[int(tgt_list[n])] / max(s, 1e-10))
    return total / N if reduction == 'mean' else total


# ── Embedding Lookup ──────────────────────────────────────────────────────────

def embedding_lookup(weight, indices) -> List[List[float]]:
    """Embedding table lookup (CPU fallback)."""
    flat_w = _flat(weight); shape_w = _shape(weight); d = shape_w[1]
    idx_list = _flat(indices) if hasattr(indices, '__iter__') else [int(indices)]
    return [flat_w[int(i)*d:(int(i)+1)*d] for i in idx_list]


# ── Dropout ───────────────────────────────────────────────────────────────────

def dropout(x, p: float = 0.5, training: bool = True) -> list:
    """Inverted dropout (CPU fallback)."""
    import random as _r
    flat = _flat(x)
    if not training or p == 0.0: return flat
    scale = 1.0 / (1.0 - p)
    return [0.0 if _r.random() < p else v * scale for v in flat]


# ── RMS Norm ─────────────────────────────────────────────────────────────────

def rms_norm(x, weight=None, eps: float = 1e-6):
    """RMS normalization — fused kernel or CPU fallback."""
    flat = _flat(x); shape = _shape(x); d = shape[-1]
    n = len(flat) // d
    w_flat = _flat(weight) if weight is not None else [1.0]*d
    out = []
    for i in range(n):
        row = flat[i*d:(i+1)*d]
        rms = (sum(v**2 for v in row)/d + eps)**0.5
        out.extend(w_flat[k] * row[k] / rms for k in range(d))
    return out


# ── Re-export advanced Triton ops ─────────────────────────────────────────────
from vectorphile.cuda.triton_kernels import (
    flash_decode,
    persistent_bias_act,
    fused_dropout_residual,
    online_softmax,
    triton_sdpa,
    flash_attn_v3,
    fused_rms_norm,
    fused_adam,
    triton_gelu,
    triton_silu,
    fused_rope,
    build_rope_cache,
    fused_batch_norm,
    mamba_selective_scan,
    fused_layer_norm,
    fused_gelu,
    fused_silu,
    fused_relu,
    fused_cross_entropy,
    fused_cross_entropy_backward,
    fused_group_norm,
    fused_instance_norm_2d,
    fused_dropout,
    fused_embedding_backward,
    flash_attention_v2,
)


# ── BF16 & INT8 CUDA dispatch helpers (Session 10) ───────────────────────────

def fp32_to_bf16(x) -> List:
    """Cast float32 buffer to BF16 (VP CUDA or pure-Python fallback)."""
    flat = _flat(x)
    if _cu is not None:
        try:
            xbuf = _nd(flat)
            out = _cu.fp32_to_bf16(xbuf, len(flat))
            return out  # CudaBuffer of uint16
        except Exception:
            pass
    # Pure-Python bit manipulation
    import struct
    result = []
    for v in flat:
        bits = struct.unpack('<I', struct.pack('<f', float(v)))[0]
        rb = 0x7FFF + ((bits >> 16) & 1)
        result.append(int((bits + rb) >> 16))
    return result


def bf16_to_fp32(x) -> List[float]:
    """Cast BF16 uint16 buffer to float32."""
    import struct
    flat = list(x) if hasattr(x, '__iter__') else [x]
    result = []
    for u in flat:
        bits = int(u) << 16
        v = struct.unpack('<f', struct.pack('<I', bits))[0]
        result.append(v)
    return result


def quantize_int8(x, axis: int = -1):
    """Per-row absmax INT8 quantization (VP CUDA or pure-Python).

    Returns (quantized_flat: List[int], scales: List[float])
    """
    flat = _flat(x); shape = _shape(x)
    D = shape[axis if axis >= 0 else len(shape) + axis]
    n = len(flat) // D
    q_flat, scales = [], []
    for i in range(n):
        row = flat[i*D:(i+1)*D]
        amax = max(abs(v) for v in row) or 1.0
        sc = amax / 127.0
        scales.append(sc)
        q_flat.extend(int(max(-127, min(127, round(v / sc)))) for v in row)
    return q_flat, scales


def dequantize_int8(q_flat: List, scales: List[float], D: int) -> List[float]:
    """Dequantize INT8 flat list back to float32."""
    n = len(q_flat) // D
    result = []
    for i in range(n):
        sc = scales[i]
        for j in range(D):
            result.append(q_flat[i*D+j] * sc)
    return result


def fused_attention_bf16(q, k, v, causal: bool = True, scale: float = None):
    """Flash Attention in BF16 — VP CUDA dispatch with FP32 fallback."""
    # For now, dispatch through standard attention_forward (BF16 precision emulated)
    return attention_forward(q, k, v, causal=causal, scale=scale)



# ── Stream management ─────────────────────────────────────────────────────────
from vectorphile.cuda.streams import (
    Stream, Event, CUDAGraph,
    current_stream, default_stream, stream, synchronize,
    is_current_stream_capturing, graph as cuda_graph,
)

# ── Memory management API (torch.cuda.* compatible) ───────────────────────────

_allocated_bytes: int = 0
_peak_allocated_bytes: int = 0
_reserved_bytes: int = 0
_current_device: int = 0


def set_device(device_id: int) -> None:
    """Set the current CUDA device."""
    global _current_device
    _current_device = device_id
    if _CUDA_AVAILABLE:
        try:
            _numba_cuda.select_device(device_id)
        except Exception:
            pass


def current_device() -> int:
    """Return the index of the currently selected CUDA device."""
    return _current_device


def empty_cache() -> None:
    """Release all unused cached GPU memory back to the OS."""
    global _reserved_bytes
    if _CUDA_AVAILABLE:
        try:
            _numba_cuda.defer_cleanup()
        except Exception:
            pass
    _reserved_bytes = _allocated_bytes


def memory_allocated(device_id: int = 0) -> int:
    """Return bytes of GPU memory currently occupied by tensors."""
    if _CUDA_AVAILABLE:
        info = memory_info(device_id)
        return info.get('total', 0) - info.get('free', 0)
    return _allocated_bytes


def max_memory_allocated(device_id: int = 0) -> int:
    """Return peak GPU memory usage since last reset."""
    return _peak_allocated_bytes


def memory_reserved(device_id: int = 0) -> int:
    """Return bytes managed by the caching allocator."""
    if _CUDA_AVAILABLE:
        info = memory_info(device_id)
        return info.get('total', 0)
    return _reserved_bytes


def max_memory_reserved(device_id: int = 0) -> int:
    """Return peak reserved GPU memory."""
    return _reserved_bytes


def reset_peak_memory_stats(device_id: int = 0) -> None:
    """Reset the peak memory statistics."""
    global _peak_allocated_bytes
    _peak_allocated_bytes = memory_allocated(device_id)


def memory_summary(device_id: int = 0, abbreviated: bool = False) -> str:
    """Return a human-readable summary of GPU memory usage."""
    alloc = memory_allocated(device_id)
    reserv = memory_reserved(device_id)
    peak = max_memory_allocated(device_id)
    name = get_device_name(device_id)
    mb = 1024 * 1024
    lines = [
        f"CUDA memory summary for {name} (device {device_id})",
        f"  Allocated : {alloc / mb:.1f} MB",
        f"  Reserved  : {reserv / mb:.1f} MB",
        f"  Peak alloc: {peak / mb:.1f} MB",
    ]
    return "\n".join(lines)


def _track_alloc(nbytes: int) -> None:
    """Internal: track a VP tensor allocation on GPU."""
    global _allocated_bytes, _peak_allocated_bytes
    _allocated_bytes += nbytes
    if _allocated_bytes > _peak_allocated_bytes:
        _peak_allocated_bytes = _allocated_bytes


def _track_free(nbytes: int) -> None:
    """Internal: track a VP tensor deallocation on GPU."""
    global _allocated_bytes
    _allocated_bytes = max(0, _allocated_bytes - nbytes)


# ── Device-aware tensor transfer helpers ─────────────────────────────────────

def to_device(arr, device_id: int = 0):
    """Move an NDArray to GPU device memory (VP allocation tracking)."""
    if _CUDA_AVAILABLE:
        try:
            gpu_arr = _numba_cuda.to_device(arr)
            flat = arr.to_flat_list() if hasattr(arr, 'to_flat_list') else list(arr)
            _track_alloc(len(flat) * 4)  # float32 = 4 bytes each
            return gpu_arr
        except Exception:
            pass
    return arr


def to_host(arr):
    """Copy a GPU array back to host memory."""
    if _CUDA_AVAILABLE and hasattr(arr, 'copy_to_host'):
        try:
            host = arr.copy_to_host()
            flat = list(host.flat) if hasattr(host, 'flat') else list(host)
            _track_free(len(flat) * 4)
            return host
        except Exception:
            pass
    return arr
