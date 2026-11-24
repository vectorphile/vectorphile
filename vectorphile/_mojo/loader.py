"""
vectorphile Mojo kernel loader — ctypes bridge.

Loaded at import time by vectorphile._core.ndarray.
If libvpkernels.so is absent (Windows native, no build yet),
all functions return None and NDArray falls back to pure Python.
"""
from __future__ import annotations
import ctypes
import os
import sys
from pathlib import Path

_LIB: ctypes.CDLL | None = None
_GPU_LIB: ctypes.CDLL | None = None
_AVAILABLE = False
_GPU_AVAILABLE = False

_LIB_DIR = Path(__file__).parent
_LIB_NAMES = {
    "linux":  "libvpkernels.so",
    "darwin": "libvpkernels.dylib",
    "win32":  "vpkernels.dll",
}
_GPU_LIB_NAMES = {
    "linux":  "libvpkernels_gpu.so",
    "darwin": "libvpkernels_gpu.dylib",
    "win32":  "vpkernels_gpu.dll",
}

c_float_p = ctypes.POINTER(ctypes.c_float)
c_int_p   = ctypes.POINTER(ctypes.c_int32)
c_void_p  = ctypes.c_void_p


def _load() -> bool:
    global _LIB, _AVAILABLE, _GPU_LIB, _GPU_AVAILABLE
    platform = sys.platform

    # CPU kernels
    lib_name = _LIB_NAMES.get(platform, "libvpkernels.so")
    lib_path = _LIB_DIR / lib_name
    if lib_path.exists():
        try:
            _LIB = ctypes.CDLL(str(lib_path))
            _setup_signatures(_LIB)
            _AVAILABLE = True
        except OSError:
            pass

    # GPU kernels (optional)
    gpu_name = _GPU_LIB_NAMES.get(platform, "libvpkernels_gpu.so")
    gpu_path = _LIB_DIR / gpu_name
    if gpu_path.exists():
        try:
            _GPU_LIB = ctypes.CDLL(str(gpu_path))
            _setup_gpu_signatures(_GPU_LIB)
            _GPU_AVAILABLE = True
        except OSError:
            pass

    return _AVAILABLE


def _sym(lib: ctypes.CDLL, name: str):
    """Return lib.name if it exists, else None (graceful missing-symbol skip)."""
    try:
        return getattr(lib, name)
    except AttributeError:
        return None


def _setup_signatures(lib: ctypes.CDLL):
    """Register ctypes signatures for all known symbols. Missing symbols are silently skipped."""

    def _sig(name, restype, argtypes):
        fn = _sym(lib, name)
        if fn is not None:
            fn.restype = restype
            fn.argtypes = argtypes

    def _sigs(names, restype, argtypes):
        for name in names:
            _sig(name, restype, argtypes)

    # ── GEMM ──────────────────────────────────────────────────────────────────
    _gemm_args = [c_float_p, c_float_p, c_float_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _sig("vp_gemm_f32",    None, [c_float_p, c_float_p, c_float_p,
                                   ctypes.c_int32, ctypes.c_int32, ctypes.c_int32])
    _sig("vp_sgemm",       None, _gemm_args)
    _sig("vp_sgemm_transB",None, _gemm_args)
    _sig("vp_sgemm_batched",None, _gemm_args + [ctypes.c_int])
    _sig("vp_nd_matmul",   None, _gemm_args)
    # alias: if only vp_gemm_f32 exists, expose it as vp_sgemm too
    if _sym(lib, "vp_sgemm") is None and _sym(lib, "vp_gemm_f32") is not None:
        lib.vp_sgemm = _sym(lib, "vp_gemm_f32")  # type: ignore[attr-defined]

    # ── Elementwise (old per-file names) ──────────────────────────────────────
    _sigs(("vp_exp", "vp_log", "vp_sqrt", "vp_abs", "vp_neg",
           "vp_relu", "vp_sigmoid", "vp_tanh", "vp_gelu", "vp_silu",
           "vp_log1p", "vp_expm1"),
          None, [c_float_p, c_float_p, ctypes.c_int])
    _sigs(("vp_add", "vp_sub", "vp_mul", "vp_div"),
          None, [c_float_p, c_float_p, c_float_p, ctypes.c_int])
    _sigs(("vp_add_scalar", "vp_mul_scalar"),
          None, [c_float_p, c_float_p, ctypes.c_float, ctypes.c_int])
    _sig("vp_clamp", None, [c_float_p, c_float_p, ctypes.c_float, ctypes.c_float, ctypes.c_int])
    _sig("vp_pow",   None, [c_float_p, c_float_p, ctypes.c_float, ctypes.c_int])

    # ── Elementwise (new _f32 names) ──────────────────────────────────────────
    _sigs(("vp_exp_f32", "vp_log_f32", "vp_sqrt_f32",
           "vp_relu_f32", "vp_sigmoid_f32", "vp_tanh_f32",
           "vp_gelu_f32", "vp_silu_f32"),
          None, [c_float_p, c_float_p, ctypes.c_int32])
    _sigs(("vp_add_f32", "vp_sub_f32", "vp_mul_f32", "vp_div_f32"),
          None, [c_float_p, c_float_p, c_float_p, ctypes.c_int32])
    _sig("vp_scalar_mul_f32", None, [c_float_p, ctypes.c_float, c_float_p, ctypes.c_int32])
    _sig("vp_scalar_add_f32", None, [c_float_p, ctypes.c_float, c_float_p, ctypes.c_int32])

    # ── Reductions ────────────────────────────────────────────────────────────
    _sigs(("vp_sum", "vp_mean", "vp_max", "vp_min", "vp_norm_l2"),
          ctypes.c_float, [c_float_p, ctypes.c_int])
    _sig("vp_dot",    ctypes.c_float, [c_float_p, c_float_p, ctypes.c_int])
    _sigs(("vp_argmax", "vp_argmin"), ctypes.c_int, [c_float_p, ctypes.c_int])
    _sigs(("vp_sum_axis", "vp_max_axis", "vp_min_axis"),
          None, [c_float_p, c_float_p, ctypes.c_int, ctypes.c_int, ctypes.c_int])
    _sigs(("vp_softmax_inplace", "vp_log_softmax_inplace"),
          None, [c_float_p, ctypes.c_int])

    # ── Reductions (_f32 names) ───────────────────────────────────────────────
    _sig("vp_sum_f32",    None, [c_float_p, ctypes.c_int32, c_float_p])
    _sig("vp_max_f32",    None, [c_float_p, ctypes.c_int32, c_float_p])
    _sig("vp_min_f32",    None, [c_float_p, ctypes.c_int32, c_float_p])
    _sig("vp_argmax_f32", None, [c_float_p, ctypes.c_int32, c_int_p])

    # ── Random ────────────────────────────────────────────────────────────────
    _sig("vp_rand_uniform",  None, [c_float_p, ctypes.c_int, ctypes.c_uint64])
    _sig("vp_rand_normal",   None, [c_float_p, ctypes.c_int, ctypes.c_uint64,
                                    ctypes.c_float, ctypes.c_float])
    _sig("vp_rand_int",      None, [c_int_p, ctypes.c_int, ctypes.c_uint64,
                                    ctypes.c_int32, ctypes.c_int32])
    _sig("vp_rand_bernoulli",None, [c_float_p, ctypes.c_int, ctypes.c_uint64, ctypes.c_float])

    # ── Conv / Pooling ────────────────────────────────────────────────────────
    _conv_args = [c_float_p, c_float_p] + [ctypes.c_int] * 14
    _sig("vp_im2col",    None, _conv_args)
    _sig("vp_col2im",    None, _conv_args)
    _sig("vp_max_pool2d",None, [c_float_p, c_int_p, c_float_p] + [ctypes.c_int] * 12)
    _sig("vp_avg_pool2d",None, [c_float_p, c_float_p] + [ctypes.c_int] * 12)

    # ── Fused kernels ─────────────────────────────────────────────────────────
    _sigs(("vp_fused_linear_relu", "vp_fused_linear_gelu", "vp_fused_linear_silu"),
          None, [c_float_p, c_float_p, c_float_p, c_float_p] + [ctypes.c_int] * 3)
    _sig("vp_fused_scale_add",       None, [c_float_p, c_float_p, c_float_p, ctypes.c_float, ctypes.c_int])
    _sig("vp_fused_rmsnorm",         None, [c_float_p, c_float_p, c_float_p, ctypes.c_int, ctypes.c_float])
    _sig("vp_fused_layernorm",       None, [c_float_p, c_float_p, c_float_p, c_float_p, ctypes.c_int, ctypes.c_float])
    _sig("vp_fused_rope",            None, [c_float_p, c_float_p, c_float_p] + [ctypes.c_int] * 4)
    _sig("vp_fused_attention_scores",None, [c_float_p, c_float_p, c_float_p, ctypes.c_float] + [ctypes.c_int] * 4)
    _sig("vp_fused_dropout_add",     None, [c_float_p, c_float_p, c_float_p, ctypes.c_float, ctypes.c_int, ctypes.c_uint64])
    _sig("vp_fused_adam_step",       None, [c_float_p, c_float_p, c_float_p, c_float_p,
                                            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                                            ctypes.c_int, ctypes.c_int])
    _sig("vp_adam_step",             None, [c_float_p, c_float_p, c_float_p, c_float_p,
                                            ctypes.c_int32, ctypes.c_float, ctypes.c_float,
                                            ctypes.c_float, ctypes.c_float, ctypes.c_int32])

    # ── NDArray buffer ops ────────────────────────────────────────────────────
    _sig("vp_nd_alloc",  c_float_p,       [ctypes.c_int64, ctypes.c_int32])
    _sig("vp_nd_free",   None,             [c_float_p])
    _sig("vp_nd_zeros",  None,             [c_float_p, ctypes.c_int64])
    _sig("vp_nd_fill",   None,             [c_float_p, ctypes.c_float, ctypes.c_int64])
    _sig("vp_nd_copy",   None,             [c_float_p, c_float_p, ctypes.c_int64])
    _sig("vp_nd_set",    None,             [c_float_p, ctypes.c_int64, ctypes.c_float])
    _sig("vp_nd_get",    ctypes.c_float,   [c_float_p, ctypes.c_int64])
    _sigs(("vp_nd_add", "vp_nd_sub", "vp_nd_mul", "vp_nd_div"),
          c_float_p, [c_float_p, c_float_p, ctypes.c_int64])
    _sigs(("vp_nd_add_scalar", "vp_nd_mul_scalar"),
          c_float_p, [c_float_p, ctypes.c_float, ctypes.c_int64])
    _sigs(("vp_nd_sum", "vp_nd_max", "vp_nd_min"),
          ctypes.c_float, [c_float_p, ctypes.c_int64])
    _sig("vp_nd_dot",    ctypes.c_float,   [c_float_p, c_float_p, ctypes.c_int64])
    _sigs(("vp_nd_argmax", "vp_nd_argmin"), ctypes.c_int64, [c_float_p, ctypes.c_int64])
    _sigs(("vp_nd_exp", "vp_nd_log", "vp_nd_sqrt", "vp_nd_abs", "vp_nd_neg",
           "vp_nd_relu", "vp_nd_sigmoid", "vp_nd_tanh", "vp_nd_gelu", "vp_nd_silu"),
          c_float_p, [c_float_p, ctypes.c_int64])
    _sig("vp_nd_softmax_inplace", None, [c_float_p, ctypes.c_int64])


def _setup_gpu_signatures(lib: ctypes.CDLL):
    """Set ctypes signatures for GPU kernel library."""
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

    lib.vp_gpu_sync.restype = None
    lib.vp_gpu_sync.argtypes = [c_void_p]

    # Elementwise GPU ops
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

    # GPU GEMM
    lib.vp_gpu_sgemm.restype = None
    lib.vp_gpu_sgemm.argtypes = [c_void_p, c_float_p, c_float_p, c_float_p,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_int]

    lib.vp_gpu_sgemm_transB.restype = None
    lib.vp_gpu_sgemm_transB.argtypes = [c_void_p, c_float_p, c_float_p, c_float_p,
                                         ctypes.c_int, ctypes.c_int, ctypes.c_int]

    lib.vp_gpu_sgemm_batched.restype = None
    lib.vp_gpu_sgemm_batched.argtypes = [c_void_p, c_float_p, c_float_p, c_float_p,
                                          ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

    # GPU reductions
    for name in ("vp_gpu_sum", "vp_gpu_max", "vp_gpu_min"):
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = [c_void_p, c_float_p, c_float_p, ctypes.c_int]

    lib.vp_gpu_softmax_rows.restype = None
    lib.vp_gpu_softmax_rows.argtypes = [c_void_p, c_float_p, ctypes.c_int, ctypes.c_int]

    lib.vp_gpu_log_softmax_rows.restype = None
    lib.vp_gpu_log_softmax_rows.argtypes = [c_void_p, c_float_p, ctypes.c_int, ctypes.c_int]

    lib.vp_gpu_layernorm_rows.restype = None
    lib.vp_gpu_layernorm_rows.argtypes = [c_void_p, c_float_p, c_float_p, c_float_p,
                                           c_float_p, ctypes.c_int, ctypes.c_int, ctypes.c_float]

    lib.vp_gpu_rmsnorm_rows.restype = None
    lib.vp_gpu_rmsnorm_rows.argtypes = [c_void_p, c_float_p, c_float_p, c_float_p,
                                         ctypes.c_int, ctypes.c_int, ctypes.c_float]

    # Fused GPU kernels
    lib.vp_gpu_fused_scale_add.restype = None
    lib.vp_gpu_fused_scale_add.argtypes = [c_void_p, c_float_p, c_float_p, c_float_p,
                                            ctypes.c_float, ctypes.c_int]

    lib.vp_gpu_fused_relu_add.restype = None
    lib.vp_gpu_fused_relu_add.argtypes = [c_void_p, c_float_p, c_float_p, c_float_p, ctypes.c_int]


# ── Convenience accessors ─────────────────────────────────────────────────────

def available() -> bool:
    return _AVAILABLE

def gpu_available() -> bool:
    return _GPU_AVAILABLE

def lib() -> ctypes.CDLL | None:
    return _LIB

def gpu_lib() -> ctypes.CDLL | None:
    return _GPU_LIB

def get(name: str):
    """Return the ctypes function or None if Mojo kernels not loaded."""
    if _LIB is None:
        return None
    return getattr(_LIB, name, None)

def get_gpu(name: str):
    """Return a GPU ctypes function or None if GPU kernels not loaded."""
    if _GPU_LIB is None:
        return None
    return getattr(_GPU_LIB, name, None)


# Auto-load on import
_load()
