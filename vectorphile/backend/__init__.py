"""vectorphile.backend — Unified hardware backend abstraction.

Dispatch priority (auto mode):
  Triton fused kernels  (requires triton + torch + CUDA)
  → PyTorch CUDA/MPS    (requires torch, fastest general-purpose path)
  → Numba CUDA kernels  (requires numba + CUDA, no torch required)
  → im2col + BLAS       (always available, 10-100× faster than loop conv)
  → pure Python         (universal fallback)
"""
from __future__ import annotations
import math
import time
from typing import Optional, Tuple, Union

# ── Optional backends — deferred to avoid slow cuda.is_available() at import ──
_torch = None
_TORCH_AVAILABLE = False
_TORCH_CUDA = False
_TORCH_CHECKED = False


def _ensure_torch():
    global _torch, _TORCH_AVAILABLE, _TORCH_CUDA, _TORCH_CHECKED
    if _TORCH_CHECKED:
        return
    _TORCH_CHECKED = True
    try:
        import torch as _t
        _torch = _t
        _TORCH_AVAILABLE = True
        _TORCH_CUDA = _t.cuda.is_available()
    except ImportError:
        pass

try:
    import cupy as _cp
    _CUPY_AVAILABLE = True
except ImportError:
    _cp = None
    _CUPY_AVAILABLE = False

try:
    from numba import cuda as _numba_cuda
    _NUMBA_AVAILABLE = True
    _NUMBA_CUDA = _numba_cuda.is_available()
except Exception:
    _numba_cuda = None
    _NUMBA_AVAILABLE = False
    _NUMBA_CUDA = False

try:
    import triton as _triton
    _TRITON_AVAILABLE = True and _TORCH_AVAILABLE and _TORCH_CUDA
except Exception:
    _triton = None
    _TRITON_AVAILABLE = False

# Metal (Apple Silicon MPS) detection
try:
    _METAL_AVAILABLE = (_TORCH_AVAILABLE and
                        hasattr(_torch, 'backends') and
                        hasattr(_torch.backends, 'mps') and
                        _torch.backends.mps.is_available())
except Exception:
    _METAL_AVAILABLE = False

# ARM NEON: detected via platform module (aarch64/arm64 CPUs)
import platform as _platform
_ARM_NEON = _platform.machine().lower() in ('aarch64', 'arm64', 'armv8', 'armv7l')

# Sub-backends (lazy import)
_torch_be = None
_cuda_be  = None
_triton_be = None

def _get_torch_be():
    global _torch_be
    if _torch_be is None:
        from vectorphile.backend import _torch_backend
        _torch_be = _torch_backend
    return _torch_be

def _get_cuda_be():
    global _cuda_be
    if _cuda_be is None:
        from vectorphile import cuda as _cuda_mod
        _cuda_be = _cuda_mod
    return _cuda_be

def _get_triton_be():
    global _triton_be
    if _triton_be is None:
        from vectorphile.cuda import triton_kernels
        _triton_be = triton_kernels
    return _triton_be


# ── Pure Python helpers ───────────────────────────────────────────────────────

def _flat(x):
    if hasattr(x, 'to_flat_list'):
        return [float(v) for v in x.to_flat_list()]
    if hasattr(x, 'tolist'):
        v = x.tolist()
        if isinstance(v, list) and v and isinstance(v[0], list):
            return [float(item) for row in v for item in row]
        return [float(item) for item in v]
    if hasattr(x, '__iter__'):
        result = []
        for item in x:
            if hasattr(item, '__iter__'):
                result.extend(float(v) for v in item)
            else:
                result.append(float(item))
        return result
    return [float(x)]

def _shape(x):
    if hasattr(x, 'shape'):
        return tuple(x.shape)
    return (len(_flat(x)),)

def _py_matmul(a, b):
    af = _flat(a); bf = _flat(b)
    sa = _shape(a); sb = _shape(b)
    if len(sa) == 1: sa = (1, sa[0])
    if len(sb) == 1: sb = (sb[0], 1)
    m, k, n = sa[0], sa[1], sb[1]
    out = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for t in range(k):
                s += af[i*k+t] * bf[t*n+j]
            out[i*n+j] = s
    return out, (m, n)

def _py_bn(x, w, b, rm, rv, training, momentum, eps):
    xf = _flat(x); wf = _flat(w); bf = _flat(b)
    n = len(xf)
    if training:
        mu = sum(xf) / max(n, 1)
        var = sum((v - mu)**2 for v in xf) / max(n, 1)
    else:
        mu = sum(_flat(rm)) / max(len(_flat(rm)), 1)
        var = sum(_flat(rv)) / max(len(_flat(rv)), 1)
    inv = 1.0 / math.sqrt(var + eps)
    return [(xf[i] - mu) * inv * wf[i % len(wf)] + bf[i % len(bf)] for i in range(n)]


# ── Global state ──────────────────────────────────────────────────────────────
_BACKEND: str = 'python'
_DEVICE: str  = 'cpu'


# ── Configuration ─────────────────────────────────────────────────────────────

def set_backend(name: str) -> None:
    """Set active backend. 'auto' picks the fastest available."""
    _ensure_torch()
    global _BACKEND, _DEVICE, _TRITON_AVAILABLE, _METAL_AVAILABLE
    if not _TRITON_AVAILABLE:
        try:
            import triton as _t2
            _TRITON_AVAILABLE = _TORCH_AVAILABLE and _TORCH_CUDA
        except Exception:
            pass
    if not _METAL_AVAILABLE and _TORCH_AVAILABLE:
        try:
            import torch as _t3
            _METAL_AVAILABLE = (hasattr(_t3, 'backends') and
                                hasattr(_t3.backends, 'mps') and
                                _t3.backends.mps.is_available())
        except Exception:
            pass
    if name == 'auto':
        # Check for vectorphile's own CUDA C extension first (fastest on NVIDIA GPU)
        try:
            from vectorphile._core import _vp_cuda as _vp_cu
            if _vp_cu.device_count() > 0:
                _BACKEND = 'vp_cuda'; _DEVICE = 'cuda'
                return
        except Exception:
            pass
        if _TORCH_AVAILABLE and _TORCH_CUDA:
            _BACKEND = 'torch'; _DEVICE = 'cuda'
        elif _METAL_AVAILABLE:
            _BACKEND = 'metal'; _DEVICE = 'mps'
        elif _TORCH_AVAILABLE:
            _BACKEND = 'torch'; _DEVICE = 'cpu'
        elif _NUMBA_CUDA:
            _BACKEND = 'cuda'; _DEVICE = 'cuda'
        elif _CUPY_AVAILABLE:
            _BACKEND = 'cupy'; _DEVICE = 'cuda'
        elif _ARM_NEON:
            _BACKEND = 'neon'; _DEVICE = 'cpu'
        else:
            # Try Mojo (CPU SIMD) before pure Python
            try:
                from vectorphile.mojo import is_mojo_available
                if is_mojo_available():
                    _BACKEND = 'mojo'; _DEVICE = 'cpu'
                    return
            except Exception:
                pass
            _BACKEND = 'python'; _DEVICE = 'cpu'
    elif name == 'torch':
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed. pip install torch")
        _BACKEND = 'torch'
        _DEVICE  = 'cuda' if _TORCH_CUDA else 'cpu'
    elif name == 'metal':
        if not _METAL_AVAILABLE:
            raise RuntimeError("Metal/MPS not available. Requires Apple Silicon Mac with PyTorch.")
        _BACKEND = 'metal'; _DEVICE = 'mps'
    elif name in ('neon', 'arm'):
        if not _ARM_NEON:
            raise RuntimeError("ARM NEON not detected on this platform.")
        _BACKEND = 'neon'; _DEVICE = 'cpu'
    elif name == 'cuda':
        if not _NUMBA_CUDA:
            raise RuntimeError("Numba CUDA not available.")
        _BACKEND = 'cuda'; _DEVICE = 'cuda'
    elif name == 'cupy':
        if not _CUPY_AVAILABLE:
            raise RuntimeError("CuPy not installed.")
        _BACKEND = 'cupy'; _DEVICE = 'cuda'
    elif name in ('numpy', 'python'):
        _BACKEND = 'python'; _DEVICE = 'cpu'
    elif name == 'vp_cuda':
        _BACKEND = 'vp_cuda'; _DEVICE = 'cuda'
    elif name == 'mojo':
        _BACKEND = 'mojo'; _DEVICE = 'cpu'
    else:
        raise ValueError(f"Unknown backend {name!r}. Valid: 'auto','torch','metal','neon','cuda','cupy','vp_cuda','mojo','python'")


def get_backend() -> str:
    return _BACKEND

def get_device() -> str:
    return _DEVICE

def set_device(device: str) -> None:
    global _DEVICE
    _DEVICE = device


# ── Capability queries ────────────────────────────────────────────────────────

def is_cuda_available() -> bool:
    _ensure_torch()
    return _TORCH_CUDA or _NUMBA_CUDA or _CUPY_AVAILABLE

def is_torch_available() -> bool:
    _ensure_torch()
    return _TORCH_AVAILABLE

def is_triton_available() -> bool:
    _ensure_torch()
    return _TRITON_AVAILABLE

def is_numba_available() -> bool:
    return _NUMBA_AVAILABLE

def is_metal_available() -> bool:
    """True when Apple Metal/MPS backend is available (Apple Silicon)."""
    return _METAL_AVAILABLE

def is_arm_neon() -> bool:
    """True when running on an ARM processor with NEON SIMD support."""
    return _ARM_NEON

def device_count() -> int:
    if _TORCH_AVAILABLE and _TORCH_CUDA:
        return _torch.cuda.device_count()
    if _NUMBA_CUDA:
        return _get_cuda_be().device_count()
    return 0

def memory_info(device_idx: int = 0) -> dict:
    if _TORCH_AVAILABLE and _TORCH_CUDA:
        free, total = _torch.cuda.mem_get_info(device_idx)
        return {'free': free, 'total': total, 'used': total - free}
    if _NUMBA_CUDA:
        info = _get_cuda_be().memory_info(device_idx)
        info['used'] = info['total'] - info['free']
        return info
    return {'free': 0, 'total': 0, 'used': 0}

def synchronize() -> None:
    if _TORCH_AVAILABLE and _TORCH_CUDA:
        _torch.cuda.synchronize()
    elif _NUMBA_CUDA:
        _get_cuda_be().synchronize()


# ── Tensor ops ────────────────────────────────────────────────────────────────

def _metal_matmul(a, b):
    """Metal/MPS matrix multiply via PyTorch MPS device."""
    af = _flat(a); bf = _flat(b)
    sa = _shape(a); sb = _shape(b)
    if len(sa) == 1: sa = (1, sa[0])
    if len(sb) == 1: sb = (sb[0], 1)
    import array as _ar
    buf_a = _ar.array('f', [float(v) for v in af])
    buf_b = _ar.array('f', [float(v) for v in bf])
    ta = _torch.frombuffer(buf_a, dtype=_torch.float32).reshape(sa).to('mps')
    tb = _torch.frombuffer(buf_b, dtype=_torch.float32).reshape(sb).to('mps')
    tc = _torch.matmul(ta, tb).cpu()
    return tc.tolist(), tuple(tc.shape)


def _neon_matmul(a, b):
    """ARM NEON-optimized GEMM: attempts ctypes/accelerate on macOS, else Python fallback."""
    import ctypes, sys
    if sys.platform == 'darwin':
        try:
            # Apple Accelerate BLAS (available on all macOS)
            _accel = ctypes.CDLL('/System/Library/Frameworks/Accelerate.framework/Accelerate')
            af = _flat(a); bf = _flat(b)
            sa = _shape(a); sb = _shape(b)
            if len(sa) == 1: sa = (1, sa[0])
            if len(sb) == 1: sb = (sb[0], 1)
            m, k, n = sa[0], sa[1], sb[1]
            import array as _ar
            ca = (ctypes.c_float * len(af))(*af)
            cb = (ctypes.c_float * len(bf))(*bf)
            cc = (ctypes.c_float * (m * n))(*([0.0] * (m * n)))
            # cblas_sgemm: row-major, no-trans, no-trans
            CblasRowMajor, CblasNoTrans = 101, 111
            _accel.cblas_sgemm(
                ctypes.c_int(CblasRowMajor), ctypes.c_int(CblasNoTrans), ctypes.c_int(CblasNoTrans),
                ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k),
                ctypes.c_float(1.0), ca, ctypes.c_int(k),
                cb, ctypes.c_int(n),
                ctypes.c_float(0.0), cc, ctypes.c_int(n)
            )
            return list(cc), (m, n)
        except Exception:
            pass
    return _py_matmul(a, b)


def matmul(a, b):
    """Matrix multiply dispatched to fastest available backend."""
    if _BACKEND == 'torch':
        return _get_torch_be().matmul(a, b)
    if _BACKEND == 'metal':
        try:
            return _metal_matmul(a, b)
        except Exception:
            pass
    if _BACKEND == 'neon':
        return _neon_matmul(a, b)
    if _BACKEND in ('cuda', 'cupy'):
        return _get_cuda_be().matmul(a, b)
    result, shape = _py_matmul(a, b)
    return result


def conv2d(x, weight, bias=None, stride: int = 1, padding: int = 0):
    if _BACKEND == 'torch':
        return _get_torch_be().conv2d(x, weight, bias, stride, padding)
    return _get_cuda_be().conv2d_im2col(x, weight, bias, stride, padding)


def batch_norm(x, weight, bias, running_mean, running_var,
               training: bool, momentum: float = 0.1, eps: float = 1e-5):
    if _BACKEND in ('torch', 'metal'):
        return _get_torch_be().batch_norm(x, weight, bias, running_mean, running_var,
                                          training, momentum, eps)
    return _py_bn(x, weight, bias, running_mean, running_var, training, momentum, eps)


def layer_norm(x, weight, bias, eps: float = 1e-5):
    if _TRITON_AVAILABLE:
        try:
            return _get_triton_be().fused_layer_norm(x, weight, bias, eps)
        except Exception:
            pass
    if _BACKEND == 'torch':
        return _get_torch_be().layer_norm(x, weight, bias, eps)
    return _get_cuda_be().layer_norm(x, weight, bias, eps)


def softmax(x, axis: int = -1):
    if _TRITON_AVAILABLE and axis == -1:
        try:
            return _get_triton_be().fused_softmax(x)
        except Exception:
            pass
    if _BACKEND == 'torch':
        return _get_torch_be().softmax(x, axis)
    return _get_cuda_be().softmax(x, axis)


def relu(x):
    if _BACKEND == 'torch':
        return _get_torch_be().relu(x)
    return _get_cuda_be().relu(x)

def gelu(x):
    if _BACKEND == 'torch':
        return _get_torch_be().gelu(x)
    return _get_cuda_be().gelu(x)

def silu(x):
    if _BACKEND == 'torch':
        return _get_torch_be().silu(x)
    return _get_cuda_be().silu(x)


def embedding(weight, ids):
    if _BACKEND == 'torch':
        return _get_torch_be().embedding_lookup(weight, ids)
    wf = _flat(weight)
    ws = _shape(weight)
    id_list = [int(v) for v in _flat(ids)]
    if len(ws) >= 2:
        d = ws[1]
        return [wf[i*d:(i+1)*d] for i in id_list]
    return [wf[i] for i in id_list]


def residual_layer_norm(x, residual, gamma, beta, eps: float = 1e-5):
    """Fused residual add + LayerNorm. Routes to VP CUDA kernel when available."""
    try:
        from vectorphile._core import _vp_cuda as _cu
        if hasattr(_cu, 'residual_layernorm_f32'):
            return _cu.residual_layernorm_f32(x, residual, gamma, beta,
                                               _shape(x)[0], _shape(x)[-1], eps)
    except Exception:
        pass
    # Pure Python fallback
    xf = _flat(x); rf = _flat(residual); gf = _flat(gamma); bf = _flat(beta)
    xs = _shape(x); N = xs[0]; D = xs[-1]
    out = []
    for i in range(N):
        row = [xf[i*D+j] + rf[i*D+j] for j in range(D)]
        mu = sum(row) / D
        var = sum((v-mu)**2 for v in row) / D
        inv = 1.0 / math.sqrt(var + eps)
        out.extend(gf[j] * (row[j]-mu) * inv + bf[j] for j in range(D))
    from vectorphile._core.ndarray import array as _nd_arr
    return _nd_arr(out, dtype='float32').reshape(xs)


def residual_rms_norm(x, residual, weight, eps: float = 1e-6):
    """Fused residual add + RMSNorm. Routes to VP CUDA kernel when available."""
    try:
        from vectorphile._core import _vp_cuda as _cu
        if hasattr(_cu, 'residual_rmsnorm_f32'):
            return _cu.residual_rmsnorm_f32(x, residual, weight,
                                             _shape(x)[0], _shape(x)[-1], eps)
    except Exception:
        pass
    xf = _flat(x); rf = _flat(residual); wf = _flat(weight)
    xs = _shape(x); N = xs[0]; D = xs[-1]
    out = []
    for i in range(N):
        row = [xf[i*D+j] + rf[i*D+j] for j in range(D)]
        rms = math.sqrt(sum(v*v for v in row) / D + eps)
        out.extend(wf[j] * row[j] / rms for j in range(D))
    from vectorphile._core.ndarray import array as _nd_arr
    return _nd_arr(out, dtype='float32').reshape(xs)


def apply_rope(qk, cos_cache, sin_cache):
    """Apply Rotary Position Embedding in-place. Routes to VP CUDA kernel when available."""
    try:
        from vectorphile._core import _vp_cuda as _cu
        if hasattr(_cu, 'rope_f32'):
            sh = _shape(qk)
            return _cu.rope_f32(qk, cos_cache, sin_cache, sh[0], sh[1], sh[2])
    except Exception:
        pass
    # Pure Python fallback: qk shape (N, H, Dh)
    xf = _flat(qk); sh = _shape(qk); N, H, Dh = sh[0], sh[1], sh[2]
    cf = _flat(cos_cache); sf = _flat(sin_cache)
    out = list(xf)
    for pos in range(N):
        for h in range(H):
            base = (pos * H + h) * Dh
            for i in range(Dh // 2):
                x0 = xf[base + i]; x1 = xf[base + i + Dh//2]
                c  = cf[pos * (Dh//2) + i]; s = sf[pos * (Dh//2) + i]
                out[base + i]           = x0*c - x1*s
                out[base + i + Dh//2]   = x0*s + x1*c
    from vectorphile._core.ndarray import array as _nd_arr
    return _nd_arr(out, dtype='float32').reshape(sh)


def _attention_pure(q, k, v, mask=None, causal: bool = False):
    """Pure VP NDArray scaled dot-product attention fallback."""
    from vectorphile._core.ndarray import array as _nd_arr
    qs = _shape(q)
    B, H, T, d = qs[0], qs[1], qs[2], qs[3]
    scale = 1.0 / math.sqrt(d)
    # q: (B,H,T,d), k: (B,H,T,d) → scores: (B,H,T,T)
    kt = k.transpose([0, 1, 3, 2])  # (B,H,d,T)
    scores = q @ kt  # (B,H,T,T)
    scores_flat = scores.to_flat_list()
    # scale + softmax
    n = B * H * T * T
    # Apply scale
    scores_flat = [v * scale for v in scores_flat]
    # Softmax over last dim
    out_scores = []
    for b in range(B):
        for h in range(H):
            for t in range(T):
                base = (b * H * T + h * T + t) * T
                row = scores_flat[base:base + T]
                mx = max(row)
                exps = [math.exp(x - mx) for x in row]
                s = sum(exps)
                out_scores.extend([e / s for e in exps])
    attn_w = _nd_arr(out_scores, dtype='float32').reshape((B, H, T, T))
    out = attn_w @ v  # (B,H,T,d)
    return out, attn_w


def attention(q, k, v, mask=None, causal: bool = False,
              dropout_p: float = 0.0, training: bool = False):
    if _BACKEND == 'torch':
        try:
            return _get_torch_be().scaled_dot_product_attention(q, k, v, mask, dropout_p, training)
        except Exception:
            pass
        # Torch failed — fall through to pure VP path
        from vectorphile._core.ndarray import NDArray as _ND
        q_nd = q._data if hasattr(q, '_data') else q
        k_nd = k._data if hasattr(k, '_data') else k
        v_nd = v._data if hasattr(v, '_data') else v
        if isinstance(q_nd, _ND):
            return _attention_pure(q_nd, k_nd, v_nd, mask, causal)
    out = _get_cuda_be().attention_forward(q, k, v, causal=causal)
    # Synthesize weights for API compat
    qf = _flat(q); kf = _flat(k)
    qs = _shape(q)
    d = qs[-1] if qs else 1
    scale = 1.0 / math.sqrt(d)
    N = qs[-2] if len(qs) >= 2 else 1
    scores = [[sum(qf[i*d+t]*kf[j*d+t] for t in range(d))*scale for j in range(N)] for i in range(N)]
    if causal:
        for i in range(N):
            for j in range(i+1, N):
                scores[i][j] = -1e9
    weights = []
    for row in scores:
        mx = max(row)
        exps = [math.exp(s - mx) for s in row]
        s = sum(exps)
        weights.append([e/s for e in exps])
    return out, weights


def adam_step(params, grads, m, v, lr, beta1, beta2, eps, weight_decay, t) -> None:
    if _BACKEND == 'torch':
        _get_torch_be().adam_step(params, grads, m, v, lr, beta1, beta2, eps, weight_decay, t)
        return
    _get_cuda_be().adam_step(params, grads, m, v, lr, beta1, beta2, eps, weight_decay, t)


# ── Model utilities ───────────────────────────────────────────────────────────

def compile_model(model, example_input=None, mode: str = 'default'):
    if _BACKEND == 'torch':
        return _get_torch_be().compile_model(model, example_input, mode)
    return model


def to_device(array, device: str = 'cpu'):
    if device == 'cpu':
        return array
    if _TORCH_AVAILABLE and _TORCH_CUDA:
        import array as _ar
        idx = int(device.split(':')[1]) if ':' in device else 0
        af = _flat(array)
        buf = _ar.array('f', af)
        t = _torch.frombuffer(buf, dtype=_torch.float32)
        return t.cuda(idx)
    if _NUMBA_CUDA:
        return _get_cuda_be().to_device(array)
    return array


def benchmark(model, input_shape: Tuple,
              n_runs: int = 100, warmup: int = 10) -> dict:
    import random as _rand
    from vectorphile.core.tensor import Tensor
    from vectorphile._core.ndarray import array as _nd
    flat = [_rand.gauss(0, 1) for _ in range(1)]
    for s in input_shape:
        flat = flat * s
    flat = flat[:input_shape[0] if input_shape else 1]
    # Simple random input
    total = 1
    for s in input_shape:
        total *= s
    flat = [_rand.gauss(0, 1) for _ in range(total)]
    x = Tensor(_nd(flat))
    for _ in range(warmup):
        try:
            model(x)
        except Exception:
            break
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        try:
            model(x)
        except Exception:
            break
        times.append(time.perf_counter() - t0)
    if not times:
        return {'latency_ms': float('nan'), 'throughput': 0, 'memory_mb': 0, 'backend': _BACKEND}
    lat = sum(times) / len(times) * 1000
    tput = input_shape[0] / (sum(times) / len(times))
    mem = memory_info()
    return {
        'latency_ms':   round(lat, 3),
        'throughput':   round(tput, 1),
        'memory_mb':    round(mem.get('used', 0) / 1e6, 1),
        'backend':      _BACKEND,
        'device':       _DEVICE,
    }


# ── BackendArray ──────────────────────────────────────────────────────────────

class BackendArray:
    """Thin wrapper tracking which device holds the data."""

    def __init__(self, data, device: str = 'cpu'):
        if hasattr(data, 'to_flat_list'):
            self._data = list(data.to_flat_list())
        elif hasattr(data, 'tolist'):
            self._data = data.tolist()
        elif isinstance(data, list):
            self._data = data
        else:
            self._data = _flat(data)
        self._shape = _shape(data) if hasattr(data, 'shape') else (len(self._data),)
        self._device = device
        self._gpu = None
        if device.startswith('cuda'):
            self._move_to_gpu(device)

    def _move_to_gpu(self, device: str):
        if device == 'mps' and _METAL_AVAILABLE:
            import array as _ar
            buf = _ar.array('f', [float(v) for v in self._data])
            self._gpu = _torch.frombuffer(buf, dtype=_torch.float32).reshape(self._shape).to('mps')
        elif _TORCH_AVAILABLE and _TORCH_CUDA:
            import array as _ar
            idx = int(device.split(':')[1]) if ':' in device else 0
            buf = _ar.array('f', [float(v) for v in self._data])
            self._gpu = _torch.frombuffer(buf, dtype=_torch.float32).cuda(idx)
        elif _NUMBA_CUDA:
            self._gpu = _numba_cuda.to_device(self._data)

    @property
    def shape(self): return self._shape
    @property
    def device(self) -> str: return self._device

    def numpy(self):
        if self._gpu is not None:
            if _TORCH_AVAILABLE and hasattr(self._gpu, 'detach'):
                return self._gpu.detach().cpu().tolist()
            if hasattr(self._gpu, 'copy_to_host'):
                return self._gpu.copy_to_host()
        return self._data

    def cuda(self, device_idx: int = 0) -> 'BackendArray':
        self._device = f'cuda:{device_idx}'
        self._move_to_gpu(self._device)
        return self

    def mps(self) -> 'BackendArray':
        """Move to Apple Metal/MPS device."""
        self._device = 'mps'
        self._move_to_gpu('mps')
        return self

    def cpu(self) -> 'BackendArray':
        d = self.numpy()
        self._data = d if isinstance(d, list) else _flat(d)
        self._gpu = None
        self._device = 'cpu'
        return self

    def __matmul__(self, other: 'BackendArray') -> 'BackendArray':
        result, shape = _py_matmul(self._data, other._data)
        ba = BackendArray.__new__(BackendArray)
        ba._data = result; ba._shape = shape; ba._device = 'cpu'; ba._gpu = None
        return ba

    def __add__(self, other) -> 'BackendArray':
        od = other._data if isinstance(other, BackendArray) else _flat(other)
        return BackendArray([a+b for a,b in zip(self._data, od)])

    def __mul__(self, other) -> 'BackendArray':
        od = other._data if isinstance(other, BackendArray) else _flat(other)
        return BackendArray([a*b for a,b in zip(self._data, od)])

    def __sub__(self, other) -> 'BackendArray':
        od = other._data if isinstance(other, BackendArray) else _flat(other)
        return BackendArray([a-b for a,b in zip(self._data, od)])

    def __truediv__(self, other) -> 'BackendArray':
        od = other._data if isinstance(other, BackendArray) else _flat(other)
        return BackendArray([a/b for a,b in zip(self._data, od)])

    def __repr__(self) -> str:
        return f"BackendArray(shape={self._shape}, device={self._device!r}, backend={_BACKEND!r})"


# ── vectorphile native CUDA backend (our own _vp_cuda extension) ─────────────
try:
    from vectorphile._core import _vp_cuda as _vp_cuda_ext
    _VP_CUDA = _vp_cuda_ext.device_count() > 0
except Exception:
    _vp_cuda_ext = None
    _VP_CUDA = False


def is_vp_cuda_available() -> bool:
    """True when vectorphile's own CUDA extension is built and a GPU is present."""
    return _VP_CUDA


# Override set_backend to prefer our own CUDA ext over torch
_orig_set_backend = set_backend

def set_backend(name: str) -> None:
    global _BACKEND, _DEVICE
    if name == 'auto':
        if _VP_CUDA:
            _BACKEND = 'vp_cuda'; _DEVICE = 'cuda'; return
    _orig_set_backend(name)


# ── Auto-detect on import ─────────────────────────────────────────────────────
set_backend('auto')


def online_softmax(x):
    """Chunked online softmax — VP CUDA → Triton → pure Python."""
    try:
        from vectorphile._core import _vp_cuda as _cu
        if hasattr(_cu, 'online_softmax_f32'):
            sh = _shape(x); xf = _flat(x)
            rows = 1; N = sh[-1]
            if len(sh) > 1:
                rows = 1
                for d in sh[:-1]: rows *= d
            from vectorphile._core.ndarray import array as _nd_arr, NDArray
            # build CudaBuffer
            import ctypes
            arr = (ctypes.c_float * len(xf))(*xf)
            buf = _cu.alloc(len(xf) * 4)
            _cu.h2d(arr, buf)
            out_buf = _cu.online_softmax_f32(buf, rows, N)
            out_arr = (ctypes.c_float * len(xf))()
            _cu.d2h(out_buf, out_arr)
            return _nd_arr(list(out_arr), 'float32').reshape(sh)
    except Exception:
        pass
    try:
        from vectorphile.cuda.triton_kernels import online_softmax as _ts
        return _ts(x)
    except Exception:
        pass
    # pure Python
    import math as _m
    from vectorphile._core.ndarray import array as _nd_arr
    xf = _flat(x); sh = _shape(x); N = sh[-1]
    rows = len(xf) // N
    result = []
    for r in range(rows):
        row = xf[r*N:(r+1)*N]
        m_ = max(row); exps = [_m.exp(v-m_) for v in row]; s = sum(exps)
        result.extend([e/s for e in exps])
    return _nd_arr(result, 'float32').reshape(sh)


def fused_embed_pos(token_emb, pos_emb, input_ids, dropout_p: float = 0.0):
    """Fused embedding gather + positional bias + optional dropout.

    Routes to VP CUDA kernel when available; pure-Python fallback otherwise.
    token_emb: NDArray (vocab, D), pos_emb: NDArray (max_seq, D)
    input_ids : NDArray (B, T) of int indices
    Returns   : NDArray (B, T, D)
    """
    try:
        from vectorphile._core import _vp_cuda as _cu
        if hasattr(_cu, 'embed_pos_f32'):
            import ctypes, array as _a
            from vectorphile._core.ndarray import array as _nd_ep
            tok_fl  = _flat(token_emb)
            pos_fl  = _flat(pos_emb)
            ids_fl  = _flat(input_ids)
            sh_ids  = _shape(input_ids)
            B, T    = sh_ids[0], sh_ids[1] if len(sh_ids)>1 else 1
            D       = token_emb.shape[1] if hasattr(token_emb, 'shape') else len(tok_fl) // token_emb.shape[0]
            vocab   = token_emb.shape[0] if hasattr(token_emb, 'shape') else len(tok_fl) // D
            tok_arr = (ctypes.c_float * len(tok_fl))(*tok_fl)
            pos_arr = (ctypes.c_float * len(pos_fl))(*pos_fl)
            ids_arr = (ctypes.c_int   * len(ids_fl))(*[int(v) for v in ids_fl])
            tok_buf = _cu.alloc(len(tok_fl)*4); _cu.h2d(tok_arr, tok_buf)
            pos_buf = _cu.alloc(len(pos_fl)*4); _cu.h2d(pos_arr, pos_buf)
            ids_buf = _cu.alloc(len(ids_fl)*4); _cu.h2d(ids_arr, ids_buf)
            out_buf = _cu.embed_pos_f32(tok_buf, pos_buf, ids_buf, B, T, D, vocab, dropout_p, 0)
            out_arr = (ctypes.c_float * (B*T*D))()
            _cu.d2h(out_buf, out_arr)
            return _nd_ep(list(out_arr), 'float32').reshape((B, T, D))
    except Exception:
        pass
    # Pure Python fallback
    from vectorphile._core.ndarray import array as _nd_ep
    tok_fl = _flat(token_emb); pos_fl = _flat(pos_emb)
    ids_fl = [int(v) for v in _flat(input_ids)]
    sh_ids = _shape(input_ids)
    B, T   = sh_ids[0], sh_ids[1] if len(sh_ids)>1 else 1
    D      = token_emb.shape[1] if hasattr(token_emb, 'shape') else len(tok_fl) // (len(tok_fl)//max(D,1))
    # fallback: simple D guess
    try: D = token_emb.shape[-1]
    except Exception: D = len(pos_fl) // (len(pos_fl) // max(len(pos_fl), 1))
    result = []
    for b in range(B):
        for t in range(T):
            tok_id = ids_fl[b*T+t]
            for d in range(D):
                result.append(tok_fl[tok_id*D+d] + pos_fl[t*D+d])
    return _nd_ep(result, 'float32').reshape((B, T, D))


def warp_reduce_sum(x):
    """Warp-level reduce-sum across all elements. Returns scalar NDArray."""
    try:
        from vectorphile._core import _vp_cuda as _cu
        if hasattr(_cu, 'warp_reduce_sum_f32'):
            xf = _flat(x); n = len(xf)
            import ctypes
            arr = (ctypes.c_float * n)(*xf)
            buf = _cu.alloc(n * 4)
            _cu.h2d(arr, buf)
            out_buf = _cu.warp_reduce_sum_f32(buf, n)
            out_arr = (ctypes.c_float * 1)()
            _cu.d2h(out_buf, out_arr)
            from vectorphile._core.ndarray import array as _nd_arr
            return _nd_arr([out_arr[0]], 'float32')
    except Exception:
        pass
    xf = _flat(x)
    from vectorphile._core.ndarray import array as _nd_arr
    return _nd_arr([sum(xf)], 'float32')
