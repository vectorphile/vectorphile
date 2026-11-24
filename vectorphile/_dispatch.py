"""
vectorphile backend dispatch layer.

Selected at import time based on available hardware and OS.
Priority order:
  1. CUDA     — NVIDIA GPU, any OS
  2. MPS      — Apple Silicon GPU (macOS only)
  3. CPU-C    — vectorphile's AVX2/FMA C extension (default, all platforms)

The active backend is a module exposing the same interface as _core.
"""
import sys
import platform

_backend = None
_backend_name = "unknown"

def _select():
    global _backend, _backend_name

    # ── CUDA ─────────────────────────────────────────────────────────────────
    try:
        import vectorphile.cuda as _cuda_backend
        if _cuda_backend.is_available():
            _backend = _cuda_backend
            _backend_name = "cuda"
            return
    except ImportError:
        pass

    # ── Apple MPS (Metal) ────────────────────────────────────────────────────
    if sys.platform == "darwin":
        try:
            import vectorphile._mps as _mps_backend
            if _mps_backend.is_available():
                _backend = _mps_backend
                _backend_name = "mps"
                return
        except ImportError:
            pass

    # ── CPU — platform-optimised C extension ─────────────────────────────────
    # Phase 1 (current): AVX2/FMA + OpenMP LLVM threading
    # Phase 2 (planned): ISPC multi-target (AVX2 / AVX-512 / NEON)
    # Phase 3 (planned): Apple Accelerate + AMX on macOS
    # Phase 6 (planned): Mojo/MAX on Linux
    try:
        from vectorphile._core import _vp_core as _c_backend
        _backend = _c_backend
        _backend_name = "cpu-avx2"
        return
    except ImportError:
        pass

    # ── Pure-Python scalar fallback (no compiled extension) ──────────────────
    import vectorphile._scalar as _scalar_backend
    _backend = _scalar_backend
    _backend_name = "cpu-scalar"

_select()

def get_backend():
    return _backend

def get_backend_name() -> str:
    return _backend_name

def has_gpu() -> bool:
    return _backend_name in ("cuda", "mps", "rocm")

def has_avx2() -> bool:
    return getattr(_backend, "has_avx2", lambda: False)()

def has_blas() -> bool:
    return getattr(_backend, "has_blas", lambda: False)()

def has_vml() -> bool:
    return getattr(_backend, "has_vml", lambda: False)()

def info() -> dict:
    return {
        "backend":    _backend_name,
        "platform":   sys.platform,
        "machine":    platform.machine(),
        "python":     sys.version.split()[0],
        "avx2":       has_avx2(),
        "blas":       has_blas(),
        "vml":        has_vml(),
        "gpu":        has_gpu(),
    }


# ── Compatibility shims for vectorphile.__init__ ──────────────────────────────

def best_device() -> str:
    """Return the best available compute device string."""
    if has_gpu():
        return _backend_name  # "cuda", "mps", "rocm"
    return "cpu"


def to_device(tensor, device: str):
    """Move tensor to target device (no-op for CPU-only tensors)."""
    return tensor


def compile_model(model, backend: str = "auto", **kwargs):
    """Accelerate a model using the best available backend.
    Currently: returns model unchanged on CPU (C/AVX2 kernels active by default).
    Phase 5 (GPU): emit CUDA/Metal graphs.
    """
    return model


class _HWInfo:
    """Hardware capability namespace exposed as vp.hw.*"""
    @property
    def backend(self): return _backend_name
    @property
    def avx2(self): return has_avx2()
    @property
    def blas(self): return has_blas()
    @property
    def vml(self): return has_vml()
    @property
    def gpu(self): return has_gpu()
    @property
    def cuda(self): return _backend_name == "cuda"
    @property
    def mps(self): return _backend_name == "mps"
    @property
    def cpu_threads(self):
        try:
            import os
            return os.cpu_count() or 1
        except Exception:
            return 1
    def __repr__(self): return f"HWInfo(backend={_backend_name!r})"

hw = _HWInfo()
