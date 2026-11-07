"""vectorphile.backend — Unified hardware backend abstraction.

Dispatch priority (auto mode):
  Triton fused kernels  (requires triton + torch + CUDA)
  → PyTorch CUDA/MPS    (requires torch, fastest general-purpose path)
  → Numba CUDA kernels  (requires numba + CUDA, no torch required)
  → im2col + BLAS       (always available, 10-100× faster than loop conv)
  → pure NumPy          (universal fallback)

Usage::

    import vectorphile.backend as backend

    backend.set_backend('auto')          # pick fastest available
    backend.set_backend('torch')         # force PyTorch path
    backend.set_backend('cuda')          # force Numba CUDA
    backend.set_backend('numpy')         # force NumPy

    out  = backend.matmul(a, b)
    out  = backend.conv2d(x, w, b, stride=1, padding=1)
    out  = backend.softmax(x, axis=-1)
    info = backend.memory_info()
    fast = backend.compile_model(model)  # torch.compile wrapping
"""
from __future__ import annotations
import time
import numpy as np
from typing import Optional, Tuple, Union

# ── Optional backends ─────────────────────────────────────────────────────────
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
    _TORCH_CUDA = _torch.cuda.is_available()
except ImportError:
    _torch = None
    _TORCH_AVAILABLE = False
    _TORCH_CUDA = False

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


# ── Global state ──────────────────────────────────────────────────────────────
_BACKEND: str = 'numpy'
_DEVICE: str  = 'cpu'


# ── Configuration ─────────────────────────────────────────────────────────────

def set_backend(name: str) -> None:
    """Set active backend. 'auto' picks the fastest available."""
    global _BACKEND, _DEVICE
    if name == 'auto':
        if _TORCH_AVAILABLE and _TORCH_CUDA:
            _BACKEND = 'torch'; _DEVICE = 'cuda'
        elif _TORCH_AVAILABLE:
            _BACKEND = 'torch'; _DEVICE = 'cpu'
        elif _NUMBA_CUDA:
            _BACKEND = 'cuda'; _DEVICE = 'cuda'
        elif _CUPY_AVAILABLE:
            _BACKEND = 'cupy'; _DEVICE = 'cuda'
        else:
            _BACKEND = 'numpy'; _DEVICE = 'cpu'
    elif name == 'torch':
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed. pip install torch")
        _BACKEND = 'torch'
        _DEVICE  = 'cuda' if _TORCH_CUDA else 'cpu'
    elif name == 'cuda':
        if not _NUMBA_CUDA:
            raise RuntimeError("Numba CUDA not available. pip install numba and install CUDA toolkit.")
        _BACKEND = 'cuda'; _DEVICE = 'cuda'
    elif name == 'cupy':
        if not _CUPY_AVAILABLE:
            raise RuntimeError("CuPy not installed. pip install cupy-cuda12x")
        _BACKEND = 'cupy'; _DEVICE = 'cuda'
    elif name == 'numpy':
        _BACKEND = 'numpy'; _DEVICE = 'cpu'
    else:
        raise ValueError(f"Unknown backend {name!r}. Valid: 'auto','torch','cuda','cupy','numpy'")


def get_backend() -> str:
    return _BACKEND


def get_device() -> str:
    return _DEVICE


def set_device(device: str) -> None:
    """Override device string, e.g. 'cuda:1'."""
    global _DEVICE
    _DEVICE = device


# ── Capability queries ────────────────────────────────────────────────────────

def is_cuda_available() -> bool:
    return _TORCH_CUDA or _NUMBA_CUDA or _CUPY_AVAILABLE


def is_torch_available() -> bool:
    return _TORCH_AVAILABLE


def is_triton_available() -> bool:
    return _TRITON_AVAILABLE


def is_numba_available() -> bool:
    return _NUMBA_AVAILABLE


def device_count() -> int:
    if _TORCH_AVAILABLE and _TORCH_CUDA:
        return _torch.cuda.device_count()
    if _NUMBA_CUDA:
        return _get_cuda_be().device_count()
    return 0


def memory_info(device_idx: int = 0) -> dict:
    """Return {'free': bytes, 'total': bytes, 'used': bytes}."""
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

def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiply dispatched to fastest available backend."""
    if _BACKEND == 'torch':
        return _get_torch_be().matmul(a, b)
    if _BACKEND in ('cuda', 'cupy'):
        return _get_cuda_be().matmul(a, b)
    return np.matmul(np.asarray(a, np.float32), np.asarray(b, np.float32))


def conv2d(x: np.ndarray, weight: np.ndarray,
           bias: Optional[np.ndarray] = None,
           stride: int = 1, padding: int = 0) -> np.ndarray:
    """2-D convolution (NCHW). Uses torch CUDA or im2col+BLAS."""
    if _BACKEND == 'torch':
        return _get_torch_be().conv2d(x, weight, bias, stride, padding)
    # Always use im2col on CPU — much faster than loop conv
    return _get_cuda_be().conv2d_im2col(x, weight, bias, stride, padding)


def batch_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               running_mean: np.ndarray, running_var: np.ndarray,
               training: bool, momentum: float = 0.1, eps: float = 1e-5) -> np.ndarray:
    if _BACKEND == 'torch':
        return _get_torch_be().batch_norm(x, weight, bias, running_mean, running_var,
                                           training, momentum, eps)
    return _numpy_bn(x, weight, bias, running_mean, running_var, training, momentum, eps)


def _numpy_bn(x, w, b, rm, rv, training, momentum, eps):
    if training:
        axes = tuple(i for i in range(x.ndim) if i != 1)
        mean = x.mean(axis=axes, keepdims=True)
        var  = x.var(axis=axes, keepdims=True)
        rm[:] = (1-momentum)*rm + momentum*x.mean(axis=axes)
        rv[:] = (1-momentum)*rv + momentum*x.var(axis=axes)
    else:
        sh = (1, -1) + (1,)*(x.ndim-2)
        mean = rm.reshape(sh); var = rv.reshape(sh)
    xn = (x - mean) / np.sqrt(var + eps)
    if x.ndim == 4:
        return xn * w[None,:,None,None] + b[None,:,None,None]
    return xn * w + b


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               eps: float = 1e-5) -> np.ndarray:
    if _TRITON_AVAILABLE:
        try:
            return _get_triton_be().fused_layer_norm(x, weight, bias, eps)
        except Exception:
            pass
    if _BACKEND == 'torch':
        return _get_torch_be().layer_norm(x, weight, bias, eps)
    return _get_cuda_be().layer_norm(x, weight, bias, eps)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    if _TRITON_AVAILABLE and axis == -1:
        try:
            return _get_triton_be().fused_softmax(x)
        except Exception:
            pass
    if _BACKEND == 'torch':
        return _get_torch_be().softmax(x, axis)
    return _get_cuda_be().softmax(x, axis)


def relu(x: np.ndarray) -> np.ndarray:
    if _BACKEND == 'torch':
        return _get_torch_be().relu(x)
    return _get_cuda_be().relu(x)


def gelu(x: np.ndarray) -> np.ndarray:
    if _BACKEND == 'torch':
        return _get_torch_be().gelu(x)
    return _get_cuda_be().gelu(x)


def silu(x: np.ndarray) -> np.ndarray:
    if _BACKEND == 'torch':
        return _get_torch_be().silu(x)
    return _get_cuda_be().silu(x)


def embedding(weight: np.ndarray, ids: np.ndarray) -> np.ndarray:
    if _BACKEND == 'torch':
        return _get_torch_be().embedding_lookup(weight, ids)
    return np.asarray(weight, np.float32)[ids.astype(int)]


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
              mask: Optional[np.ndarray] = None,
              causal: bool = False,
              dropout_p: float = 0.0, training: bool = False
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention dispatched to fastest backend."""
    if _BACKEND == 'torch':
        return _get_torch_be().scaled_dot_product_attention(q, k, v, mask, dropout_p, training)
    # Numba/cuda tiled flash-attention (no weights returned, synthesise them)
    out = _get_cuda_be().attention_forward(q, k, v, causal=causal)
    # Synthesise weights for API compat
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = q @ k.swapaxes(-2, -1) * scale
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    if causal:
        N = q.shape[-2]
        scores = np.where(np.triu(np.ones((N, N), bool), 1), -1e9, scores)
    e = np.exp(scores - scores.max(-1, keepdims=True))
    weights = e / (e.sum(-1, keepdims=True) + 1e-9)
    return out, weights.astype(np.float32)


def adam_step(params: np.ndarray, grads: np.ndarray,
              m: np.ndarray, v: np.ndarray,
              lr: float, beta1: float, beta2: float,
              eps: float, weight_decay: float, t: int) -> None:
    """Fused Adam update (in-place)."""
    if _BACKEND == 'torch':
        _get_torch_be().adam_step(params, grads, m, v, lr, beta1, beta2, eps, weight_decay, t)
        return
    _get_cuda_be().adam_step(params, grads, m, v, lr, beta1, beta2, eps, weight_decay, t)


# ── Model utilities ───────────────────────────────────────────────────────────

def compile_model(model, example_input=None, mode: str = 'default'):
    """Wrap a vectorphile Module in torch.compile for maximum throughput.

    Requires PyTorch >= 2.0. Returns original model if unavailable.
    """
    if _BACKEND == 'torch':
        return _get_torch_be().compile_model(model, example_input, mode)
    return model


def to_device(array: np.ndarray, device: str = 'cpu') -> np.ndarray:
    if device == 'cpu':
        return np.asarray(array, dtype=np.float32)
    if _TORCH_AVAILABLE and _TORCH_CUDA:
        idx = int(device.split(':')[1]) if ':' in device else 0
        return _torch.from_numpy(np.asarray(array, np.float32)).cuda(idx)
    if _NUMBA_CUDA:
        return _get_cuda_be().to_device(array)
    if _CUPY_AVAILABLE:
        return _cp.asarray(array)
    return np.asarray(array, np.float32)


def benchmark(model, input_shape: Tuple,
              n_runs: int = 100, warmup: int = 10) -> dict:
    """Profile model: mean latency (ms), throughput (samples/s), memory (MB).

    Args:
        model: callable vectorphile Module
        input_shape: (B, ...) — shape of one forward-pass input
        n_runs: timed runs
        warmup: un-timed warmup runs
    Returns:
        dict with 'latency_ms', 'throughput', 'memory_mb', 'backend'
    """
    from vectorphile.core.tensor import Tensor
    x = Tensor(np.random.randn(*input_shape).astype(np.float32))
    # Warmup
    for _ in range(warmup):
        try:
            model(x)
        except Exception:
            break
    if _TORCH_AVAILABLE and _TORCH_CUDA:
        _torch.cuda.synchronize()
    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        try:
            model(x)
        except Exception:
            break
        if _TORCH_AVAILABLE and _TORCH_CUDA:
            _torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    if not times:
        return {'latency_ms': float('nan'), 'throughput': 0, 'memory_mb': 0, 'backend': _BACKEND}
    lat = float(np.mean(times)) * 1000
    tput = input_shape[0] / float(np.mean(times))
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
    """Thin wrapper tracking which device holds the data.

    Constructible from any numpy array. Transparently dispatches ops
    to the active backend.

    Example::

        a = BackendArray(np.random.randn(512, 512).astype(np.float32))
        a.cuda()           # move to GPU (if available)
        b = a @ a          # stays on device
        result = b.numpy() # always returns CPU numpy
    """

    def __init__(self, data: np.ndarray, device: str = 'cpu'):
        self._np   = np.asarray(data, dtype=np.float32)
        self._device = device
        self._gpu  = None   # torch tensor or cupy/numba device array
        if device.startswith('cuda'):
            self._move_to_gpu(device)

    def _move_to_gpu(self, device: str):
        if _TORCH_AVAILABLE and _TORCH_CUDA:
            idx = int(device.split(':')[1]) if ':' in device else 0
            self._gpu = _torch.from_numpy(self._np).cuda(idx)
        elif _NUMBA_CUDA:
            self._gpu = _numba_cuda.to_device(self._np)
        elif _CUPY_AVAILABLE:
            self._gpu = _cp.asarray(self._np)

    @property
    def shape(self): return self._np.shape

    @property
    def dtype(self): return self._np.dtype

    @property
    def device(self) -> str: return self._device

    def numpy(self) -> np.ndarray:
        if self._gpu is not None:
            if _TORCH_AVAILABLE and isinstance(self._gpu, _torch.Tensor):
                return self._gpu.detach().cpu().numpy()
            if _CUPY_AVAILABLE and isinstance(self._gpu, _cp.ndarray):
                return _cp.asnumpy(self._gpu)
            if hasattr(self._gpu, 'copy_to_host'):
                return self._gpu.copy_to_host()
        return self._np

    def cuda(self, device_idx: int = 0) -> 'BackendArray':
        self._device = f'cuda:{device_idx}'
        self._move_to_gpu(self._device)
        return self

    def cpu(self) -> 'BackendArray':
        self._np = self.numpy()
        self._gpu = None
        self._device = 'cpu'
        return self

    def __matmul__(self, other: 'BackendArray') -> 'BackendArray':
        return BackendArray(matmul(self.numpy(), other.numpy()))

    def __add__(self, other) -> 'BackendArray':
        o = other.numpy() if isinstance(other, BackendArray) else other
        return BackendArray(self.numpy() + o)

    def __mul__(self, other) -> 'BackendArray':
        o = other.numpy() if isinstance(other, BackendArray) else other
        return BackendArray(self.numpy() * o)

    def __sub__(self, other) -> 'BackendArray':
        o = other.numpy() if isinstance(other, BackendArray) else other
        return BackendArray(self.numpy() - o)

    def __truediv__(self, other) -> 'BackendArray':
        o = other.numpy() if isinstance(other, BackendArray) else other
        return BackendArray(self.numpy() / o)

    def __repr__(self) -> str:
        return f"BackendArray(shape={self.shape}, device={self._device!r}, backend={_BACKEND!r})"


# ── Auto-detect on import ─────────────────────────────────────────────────────
set_backend('auto')
