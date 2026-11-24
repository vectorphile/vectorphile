"""vectorphile.metal — Apple Silicon Metal/MPS backend.

Provides GPU acceleration on Apple Silicon (M1/M2/M3/M4) via PyTorch's MPS
backend or MLX. Falls back gracefully to CPU when neither is available.

Usage::
    import vectorphile as vp
    if vp.metal.is_available():
        x = x.to('mps')
        model = model.to('mps')
"""
from __future__ import annotations

_MPS_AVAILABLE = False
_MLX_AVAILABLE = False
_DEVICE = 'cpu'

# ── PyTorch MPS (Metal Performance Shaders) ───────────────────────────────────
try:
    import torch as _torch
    if hasattr(_torch.backends, 'mps') and _torch.backends.mps.is_available():
        _MPS_AVAILABLE = True
        _DEVICE = 'mps'
except ImportError:
    _torch = None

# ── MLX (Apple's ML framework for Apple Silicon) ──────────────────────────────
try:
    import mlx.core as _mlx
    _MLX_AVAILABLE = True
    if _DEVICE == 'cpu':
        _DEVICE = 'mlx'
except ImportError:
    _mlx = None


def is_available() -> bool:
    """Return True if any Apple GPU backend (MPS or MLX) is available."""
    return _MPS_AVAILABLE or _MLX_AVAILABLE


def device_name() -> str:
    """Return the best available Apple GPU device name."""
    if _MLX_AVAILABLE:
        try:
            return _mlx.default_device().__repr__()
        except Exception:
            pass
    if _MPS_AVAILABLE:
        return 'Apple MPS (Metal Performance Shaders)'
    return 'CPU (no Metal/MPS available)'


def memory_allocated() -> int:
    """Return bytes currently allocated on Metal/MPS device."""
    if _MPS_AVAILABLE and _torch is not None:
        try:
            return _torch.mps.current_allocated_memory()
        except Exception:
            pass
    return 0


def empty_cache() -> None:
    """Free the MPS/MLX cached memory pool."""
    if _MPS_AVAILABLE and _torch is not None:
        try:
            _torch.mps.empty_cache()
        except Exception:
            pass


def synchronize() -> None:
    """Synchronize the Metal command queue (wait for all GPU ops to finish)."""
    if _MPS_AVAILABLE and _torch is not None:
        try:
            _torch.mps.synchronize()
        except Exception:
            pass


def to_mps(tensor_or_module):
    """Move a Tensor or Module to MPS device (Apple Silicon GPU)."""
    if not _MPS_AVAILABLE:
        return tensor_or_module
    if hasattr(tensor_or_module, 'to'):
        return tensor_or_module.to('mps')
    return tensor_or_module


def to_mlx(array):
    """Convert an NDArray to an MLX array for Apple Silicon compute."""
    if not _MLX_AVAILABLE:
        return array
    from vectorphile._core.ndarray import NDArray
    if isinstance(array, NDArray):
        flat = list(array.to_flat_list())
        return _mlx.array(flat, dtype=_mlx.float32).reshape(array.shape)
    return _mlx.array(array)


def from_mlx(mlx_array):
    """Convert an MLX array back to an NDArray."""
    if not _MLX_AVAILABLE:
        return mlx_array
    from vectorphile._core.ndarray import array as _nd_array
    flat = mlx_array.flatten().tolist()
    sh = tuple(mlx_array.shape)
    return _nd_array(flat, dtype='float32').reshape(sh)


__all__ = [
    'is_available', 'device_name', 'memory_allocated', 'empty_cache',
    'synchronize', 'to_mps', 'to_mlx', 'from_mlx',
]
