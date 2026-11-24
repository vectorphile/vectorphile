"""
vectorphile GPU kernel loader — ctypes bridge for GPU ops.

Loaded by vectorphile._core.ndarray when a CUDA GPU is present.
Falls back gracefully when no GPU or no libvpkernels_gpu.so.

Usage (internal):
    from vectorphile._mojo.gpu_loader import GPU
    if GPU.available():
        ctx = GPU.default_ctx()
        GPU.lib().vp_gpu_sgemm(ctx, C_ptr, A_ptr, B_ptr, M, K, N)
        GPU.sync(ctx)
"""
from __future__ import annotations
import ctypes
import sys
from pathlib import Path

_LIB: ctypes.CDLL | None = None
_CTX = None          # opaque DeviceContext handle
_AVAILABLE = False

_LIB_DIR = Path(__file__).parent
_LIB_NAMES = {
    "linux":  "libvpkernels_gpu.so",
    "darwin": "libvpkernels_gpu.dylib",
    "win32":  "vpkernels_gpu.dll",
}

c_float_p = ctypes.POINTER(ctypes.c_float)
c_void_p  = ctypes.c_void_p


def _load() -> bool:
    global _LIB, _CTX, _AVAILABLE
    lib_name = _LIB_NAMES.get(sys.platform, "libvpkernels_gpu.so")
    lib_path = _LIB_DIR / lib_name
    if not lib_path.exists():
        return False
    try:
        _LIB = ctypes.CDLL(str(lib_path))
        _setup_signatures(_LIB)
        # Check device count
        n_dev = _LIB.vp_gpu_device_count()
        if n_dev == 0:
            return False
        _CTX = _LIB.vp_gpu_init(ctypes.c_int32(0))
        _AVAILABLE = True
        return True
    except (OSError, AttributeError):
        return False


def _setup_signatures(lib: ctypes.CDLL):
    # Device management
    lib.vp_gpu_device_count.restype = ctypes.c_int32
    lib.vp_gpu_device_count.argtypes = []

    lib.vp_gpu_init.restype = c_void_p
    lib.vp_gpu_init.argtypes = [ctypes.c_int32]

    lib.vp_gpu_destroy.restype = None
    lib.vp_gpu_destroy.argtypes = [c_void_p]

    lib.vp_gpu_alloc.restype = c_float_p
    lib.vp_gpu_alloc.argtypes = [c_void_p, ctypes.c_int64]

    lib.vp_gpu_free.restype = None
    lib.vp_gpu_free.argtypes = [c_void_p, c_float_p]

    lib.vp_gpu_h2d.restype = None
    lib.vp_gpu_h2d.argtypes = [c_void_p, c_float_p, c_float_p, ctypes.c_int64]

    lib.vp_gpu_d2h.restype = None
    lib.vp_gpu_d2h.argtypes = [c_void_p, c_float_p, c_float_p, ctypes.c_int64]

    lib.vp_gpu_zeros.restype = None
    lib.vp_gpu_zeros.argtypes = [c_void_p, c_float_p, ctypes.c_int64]

    lib.vp_gpu_fill.restype = None
    lib.vp_gpu_fill.argtypes = [c_void_p, c_float_p, ctypes.c_float, ctypes.c_int64]

    lib.vp_gpu_sync.restype = None
    lib.vp_gpu_sync.argtypes = [c_void_p]

    lib.vp_gpu_memory_info.restype = None
    lib.vp_gpu_memory_info.argtypes = [c_void_p,
                                        ctypes.POINTER(ctypes.c_int64),
                                        ctypes.POINTER(ctypes.c_int64)]

    # Elementwise
    for name in ("vp_gpu_exp", "vp_gpu_log", "vp_gpu_sqrt", "vp_gpu_relu",
                 "vp_gpu_sigmoid", "vp_gpu_tanh", "vp_gpu_gelu", "vp_gpu_silu",
                 "vp_gpu_neg", "vp_gpu_abs"):
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = [c_void_p, c_float_p, c_float_p, ctypes.c_int]

    for name in ("vp_gpu_add", "vp_gpu_sub", "vp_gpu_mul", "vp_gpu_div"):
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = [c_void_p, c_float_p, c_float_p, c_float_p, ctypes.c_int]

    for name in ("vp_gpu_add_scalar", "vp_gpu_mul_scalar"):
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = [c_void_p, c_float_p, c_float_p, ctypes.c_float, ctypes.c_int]

    lib.vp_gpu_clamp.restype = None
    lib.vp_gpu_clamp.argtypes = [c_void_p, c_float_p, c_float_p,
                                   ctypes.c_float, ctypes.c_float, ctypes.c_int]

    lib.vp_gpu_fused_scale_add.restype = None
    lib.vp_gpu_fused_scale_add.argtypes = [c_void_p, c_float_p, c_float_p,
                                             c_float_p, ctypes.c_float, ctypes.c_int]

    lib.vp_gpu_fused_relu_add.restype = None
    lib.vp_gpu_fused_relu_add.argtypes = [c_void_p, c_float_p, c_float_p,
                                            c_float_p, ctypes.c_int]

    # GEMM
    lib.vp_gpu_sgemm.restype = None
    lib.vp_gpu_sgemm.argtypes = [c_void_p, c_float_p, c_float_p, c_float_p,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int]

    lib.vp_gpu_sgemm_transB.restype = None
    lib.vp_gpu_sgemm_transB.argtypes = [c_void_p, c_float_p, c_float_p, c_float_p,
                                          ctypes.c_int, ctypes.c_int, ctypes.c_int]

    lib.vp_gpu_sgemm_batched.restype = None
    lib.vp_gpu_sgemm_batched.argtypes = [c_void_p, c_float_p, c_float_p, c_float_p,
                                           ctypes.c_int, ctypes.c_int,
                                           ctypes.c_int, ctypes.c_int]

    # Reductions
    for name in ("vp_gpu_sum", "vp_gpu_max", "vp_gpu_min"):
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = [c_void_p, c_float_p, c_float_p, ctypes.c_int]

    lib.vp_gpu_softmax_rows.restype = None
    lib.vp_gpu_softmax_rows.argtypes = [c_void_p, c_float_p, ctypes.c_int, ctypes.c_int]

    lib.vp_gpu_log_softmax_rows.restype = None
    lib.vp_gpu_log_softmax_rows.argtypes = [c_void_p, c_float_p, ctypes.c_int, ctypes.c_int]

    lib.vp_gpu_layernorm_rows.restype = None
    lib.vp_gpu_layernorm_rows.argtypes = [c_void_p, c_float_p, c_float_p,
                                            c_float_p, c_float_p,
                                            ctypes.c_int, ctypes.c_int, ctypes.c_float]

    lib.vp_gpu_rmsnorm_rows.restype = None
    lib.vp_gpu_rmsnorm_rows.argtypes = [c_void_p, c_float_p, c_float_p,
                                          c_float_p,
                                          ctypes.c_int, ctypes.c_int, ctypes.c_float]


class GPU:
    """Static accessor — use GPU.available() before calling anything."""

    @staticmethod
    def available() -> bool:
        return _AVAILABLE

    @staticmethod
    def lib() -> ctypes.CDLL | None:
        return _LIB

    @staticmethod
    def default_ctx():
        return _CTX

    @staticmethod
    def sync():
        if _LIB and _CTX:
            _LIB.vp_gpu_sync(_CTX)

    @staticmethod
    def memory_info() -> dict:
        if not (_LIB and _CTX):
            return {"free": 0, "total": 0}
        free  = ctypes.c_int64(0)
        total = ctypes.c_int64(0)
        _LIB.vp_gpu_memory_info(_CTX, ctypes.byref(free), ctypes.byref(total))
        return {"free": free.value, "total": total.value,
                "used": total.value - free.value}


_load()
