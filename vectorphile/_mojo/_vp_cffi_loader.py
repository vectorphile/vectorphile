"""Auto-generated cffi loader for vectorphile C kernels."""
from __future__ import annotations
import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
try:
    import _vp_kernels
    _ffi = _vp_kernels.ffi
    _lib = _vp_kernels.lib

    def available(): return True

    def get(name: str):
        try:
            return getattr(_lib, name)
        except AttributeError:
            return None

    def ptr(arr):
        """Return cffi float* pointing at array.array buffer."""
        import ctypes
        addr, _ = arr.buffer_info()
        return _ffi.cast("float*", addr)

    def iptr(arr):
        """Return cffi int* pointing at array.array buffer."""
        import ctypes
        addr, _ = arr.buffer_info()
        return _ffi.cast("int*", addr)

    def i64ptr(arr):
        """Return cffi int64_t* pointing at array.array buffer."""
        import ctypes
        addr, _ = arr.buffer_info()
        return _ffi.cast("int64_t*", addr)

except ImportError:
    _ffi = None; _lib = None
    def available(): return False
    def get(name): return None
    def ptr(arr): return None
    def iptr(arr): return None
    def i64ptr(arr): return None
