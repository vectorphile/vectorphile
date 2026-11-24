"""
Mojo/MAX backend for vectorphile.

Loads the compiled vp_mojo.mojopkg (built via scripts/build_vp_mojo_linux.sh)
and exposes the same interface as _vp_core so callers can swap backends.

On import, tries in order:
  1. _vp_mojo shared library (if mojo package compiled to .so via MAX)
  2. max.engine Python API (inference path only)
  3. Falls back gracefully — _MOJO_OK = False
"""
import ctypes
import os
import array as _arr

_MOJO_OK = False
_lib = None

# --- Try loading compiled Mojo shared lib -----------------------------------
def _try_load_so():
    global _lib, _MOJO_OK
    _here = os.path.dirname(__file__)
    for name in ('_vp_mojo.so', 'vp_mojo.so', '_vp_mojo.dylib'):
        path = os.path.join(_here, name)
        if os.path.isfile(path):
            try:
                _lib = ctypes.CDLL(path)
                _setup_signatures()
                _MOJO_OK = True
                return True
            except OSError:
                pass
    return False


def _setup_signatures():
    """Bind ctypes argument/return types for each kernel."""
    F32P = ctypes.POINTER(ctypes.c_float)
    for name in ('relu_f32', 'gelu_f32', 'silu_f32', 'sigmoid_f32', 'tanh_f32'):
        fn = getattr(_lib, name, None)
        if fn:
            fn.argtypes = [F32P, F32P, ctypes.c_int]
            fn.restype = None
    for name in ('relu_bwd_f32',):
        fn = getattr(_lib, name, None)
        if fn:
            fn.argtypes = [F32P, F32P, F32P, ctypes.c_int]
            fn.restype = None
    for name in ('gemm_f32',):
        fn = getattr(_lib, name, None)
        if fn:
            fn.argtypes = [F32P, F32P, F32P, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            fn.restype = None
    for name in ('gemm_act_f32',):
        fn = getattr(_lib, name, None)
        if fn:
            fn.argtypes = [F32P, F32P, F32P,
                           ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            fn.restype = None
    for name in ('transpose_f32', 'sum_rows_f32', 'add_bias_rows_f32'):
        fn = getattr(_lib, name, None)
        if fn:
            fn.argtypes = [F32P, F32P, ctypes.c_int, ctypes.c_int]
            fn.restype = None


_try_load_so()


# --- Public API (mirrors _vp_core interface) ---------------------------------

def _buf_ptr(buf):
    """Return ctypes float* from an array.array('f', ...) buffer."""
    addr, _ = buf.buffer_info()
    return ctypes.cast(addr, ctypes.POINTER(ctypes.c_float))


def matmul_f32(a_buf, b_buf, c_buf, M: int, K: int, N: int):
    if not _MOJO_OK:
        raise RuntimeError('Mojo backend not loaded')
    _lib.gemm_f32(_buf_ptr(a_buf), _buf_ptr(b_buf), _buf_ptr(c_buf),
                  ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(N))


def matmul_act_f32(a_buf, b_buf, c_buf, M: int, K: int, N: int, post_op: int):
    if not _MOJO_OK:
        raise RuntimeError('Mojo backend not loaded')
    _lib.gemm_act_f32(_buf_ptr(a_buf), _buf_ptr(b_buf), _buf_ptr(c_buf),
                     ctypes.c_int(M), ctypes.c_int(K), ctypes.c_int(N),
                     ctypes.c_int(post_op))


def relu_f32(x_buf, out_buf, n: int = 0):
    if not n:
        n = len(x_buf)
    _lib.relu_f32(_buf_ptr(x_buf), _buf_ptr(out_buf), ctypes.c_int(n))


def relu_backward_f32(grad_buf, out_buf, dx_buf):
    n = len(grad_buf)
    _lib.relu_bwd_f32(_buf_ptr(grad_buf), _buf_ptr(out_buf), _buf_ptr(dx_buf),
                      ctypes.c_int(n))


def transpose_f32(a_buf, b_buf, M: int, N: int):
    _lib.transpose_f32(_buf_ptr(a_buf), _buf_ptr(b_buf),
                       ctypes.c_int(M), ctypes.c_int(N))


def add_bias_rows_f32(c_buf, b_buf, M: int, N: int):
    _lib.add_bias_rows_f32(_buf_ptr(c_buf), _buf_ptr(b_buf),
                           ctypes.c_int(M), ctypes.c_int(N))


def sum_axis_f32(x_buf, out_buf, M: int, N: int, axis: int = 0):
    if axis != 0:
        raise ValueError('Mojo sum_axis_f32 only supports axis=0')
    _lib.sum_rows_f32(_buf_ptr(x_buf), _buf_ptr(out_buf),
                      ctypes.c_int(M), ctypes.c_int(N))


def is_available() -> bool:
    return _MOJO_OK
