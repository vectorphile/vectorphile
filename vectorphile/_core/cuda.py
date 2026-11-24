"""CUDA device helpers — imports _vp_cuda and exposes CudaBuffer + transfer ops."""
import array as _arr

try:
    from vectorphile._core import _vp_cuda as _C
    _CUDA_OK = True
    _CudaBuffer = _C.CudaBuffer
except ImportError:
    _C = None
    _CUDA_OK = False
    _CudaBuffer = None


def is_available() -> bool:
    return _CUDA_OK and _C.device_count() > 0


def device_name(idx: int = 0) -> str:
    if not _CUDA_OK:
        return 'no CUDA'
    return _C.device_name(idx)


def alloc(nbytes: int):
    """Allocate a CudaBuffer of nbytes on the current device."""
    if not _CUDA_OK:
        raise RuntimeError('CUDA not available — build scripts/build_vp_cuda_win.bat first')
    return _C.alloc(nbytes)


def to_device(cpu_buf) -> '_CudaBuffer':
    """Copy a CPU array.array buffer to a new CudaBuffer."""
    n = len(cpu_buf) * 4  # float32
    cbuf = _C.alloc(n)
    _C.h2d(cpu_buf, cbuf)
    return cbuf


def to_host(cuda_buf, out_buf=None, n_floats: int = 0):
    """Copy a CudaBuffer back to a CPU array.array('f', ...)."""
    if out_buf is None:
        out_buf = _arr.array('f', bytearray(cuda_buf.nbytes))
    _C.d2h(cuda_buf, out_buf)
    return out_buf


def synchronize():
    if _CUDA_OK:
        _C.synchronize()


def gemm_f32(a_flat, b_flat, m: int, k: int, n: int):
    """GEMM on GPU: C[m×n] = A[m×k] @ B[k×n]. Returns CPU array.array('f',...)."""
    if not _CUDA_OK:
        raise RuntimeError('CUDA not available')
    ga = _C.alloc(m * k * 4); gb = _C.alloc(k * n * 4)
    _C.h2d(a_flat, ga); _C.h2d(b_flat, gb)
    gc = _C.gemm_f32(ga, gb, m, k, n)
    _C.synchronize()
    out = _arr.array('f', bytearray(m * n * 4))
    _C.d2h(gc, out)
    return out


def relu_f32(a_flat, n: int):
    """Element-wise ReLU on GPU. Returns CPU array.array('f',...)."""
    if not _CUDA_OK:
        raise RuntimeError('CUDA not available')
    ga = _C.alloc(n * 4); _C.h2d(a_flat, ga)
    gc = _C.relu_f32(ga, n)
    _C.synchronize()
    out = _arr.array('f', bytearray(n * 4))
    _C.d2h(gc, out)
    return out


def softmax_f32(a_flat, rows: int, cols: int):
    """Row-wise softmax on GPU. Returns CPU array.array('f',...)."""
    if not _CUDA_OK:
        raise RuntimeError('CUDA not available')
    ga = _C.alloc(rows * cols * 4); _C.h2d(a_flat, ga)
    gc = _C.softmax_f32(ga, rows, cols)
    _C.synchronize()
    out = _arr.array('f', bytearray(rows * cols * 4))
    _C.d2h(gc, out)
    return out
