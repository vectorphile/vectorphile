"""Numba-accelerated parallel kernels for hot paths.

Provides CPU-parallel versions of element-wise ops via numba prange.
Falls back gracefully when numba is unavailable.
"""
from __future__ import annotations
import math as _math
import sys as _sys

# Numba crashes on Python 3.14 when platform.machine() spawns a subprocess
# that triggers an access violation. Skip it entirely on 3.14+.
_NUMBA = False
_nb = None
_np = None
_THREADS = 1

if _sys.version_info < (3, 14):
  try:
    import numba as _nb
    import numpy as _np
    _NUMBA = True
    _THREADS = _nb.get_num_threads()
  except Exception:
    _nb = None
    _np = None
    _NUMBA = False
    _THREADS = 1

if _NUMBA:
    @_nb.njit(parallel=True, fastmath=True, cache=True)
    def _relu(x: _np.ndarray, out: _np.ndarray) -> None:
        xf = x.ravel(); of = out.ravel()
        for i in _nb.prange(xf.shape[0]):
            v = xf[i]; of[i] = v if v > 0.0 else 0.0

    @_nb.njit(parallel=True, fastmath=True, cache=True)
    def _mul(a: _np.ndarray, b: _np.ndarray, out: _np.ndarray) -> None:
        af = a.ravel(); bf = b.ravel(); of = out.ravel()
        for i in _nb.prange(af.shape[0]):
            of[i] = af[i] * bf[i]

    @_nb.njit(parallel=True, fastmath=True, cache=True)
    def _add(a: _np.ndarray, b: _np.ndarray, out: _np.ndarray) -> None:
        af = a.ravel(); bf = b.ravel(); of = out.ravel()
        for i in _nb.prange(af.shape[0]):
            of[i] = af[i] + bf[i]

    @_nb.njit(parallel=True, fastmath=True, cache=True)
    def _sub(a: _np.ndarray, b: _np.ndarray, out: _np.ndarray) -> None:
        af = a.ravel(); bf = b.ravel(); of = out.ravel()
        for i in _nb.prange(af.shape[0]):
            of[i] = af[i] - bf[i]

    @_nb.njit(parallel=True, fastmath=True, cache=True)
    def _fma(a: _np.ndarray, b: _np.ndarray, out: _np.ndarray) -> None:
        """out = a * b + a  (fused multiply-add)"""
        af = a.ravel(); bf = b.ravel(); of = out.ravel()
        for i in _nb.prange(af.shape[0]):
            v = af[i]; of[i] = v * bf[i] + v

    @_nb.njit(parallel=True, fastmath=True, cache=True)
    def _mul_scalar(a: _np.ndarray, s: float, out: _np.ndarray) -> None:
        af = a.ravel(); of = out.ravel()
        for i in _nb.prange(af.shape[0]):
            of[i] = af[i] * s

    @_nb.njit(parallel=True, fastmath=True, cache=True)
    def _add_scalar(a: _np.ndarray, s: float, out: _np.ndarray) -> None:
        af = a.ravel(); of = out.ravel()
        for i in _nb.prange(af.shape[0]):
            of[i] = af[i] + s

    @_nb.njit(parallel=True, fastmath=True, cache=True)
    def _neg(a: _np.ndarray, out: _np.ndarray) -> None:
        af = a.ravel(); of = out.ravel()
        for i in _nb.prange(af.shape[0]):
            of[i] = -af[i]

    @_nb.njit(parallel=True, fastmath=True, cache=True)
    def _maximum(a: _np.ndarray, b: _np.ndarray, out: _np.ndarray) -> None:
        af = a.ravel(); bf = b.ravel(); of = out.ravel()
        for i in _nb.prange(af.shape[0]):
            va = af[i]; vb = bf[i]; of[i] = va if va > vb else vb

    @_nb.njit(parallel=True, fastmath=True, cache=True)
    def _minimum(a: _np.ndarray, b: _np.ndarray, out: _np.ndarray) -> None:
        af = a.ravel(); bf = b.ravel(); of = out.ravel()
        for i in _nb.prange(af.shape[0]):
            va = af[i]; vb = bf[i]; of[i] = va if va < vb else vb

    @_nb.njit(parallel=True, fastmath=True, cache=True)
    def _truediv(a: _np.ndarray, b: _np.ndarray, out: _np.ndarray) -> None:
        af = a.ravel(); bf = b.ravel(); of = out.ravel()
        for i in _nb.prange(af.shape[0]):
            of[i] = af[i] / bf[i]

    @_nb.njit(parallel=True, fastmath=True, cache=True)
    def _truediv_scalar(a: _np.ndarray, s: float, out: _np.ndarray) -> None:
        af = a.ravel(); of = out.ravel(); inv = 1.0 / s
        for i in _nb.prange(af.shape[0]):
            of[i] = af[i] * inv

    # Softmax row-wise: fused (max, exp, sum, div) in one pass per row
    @_nb.njit(parallel=True, fastmath=True, cache=True)
    def _softmax_rows(x: _np.ndarray, out: _np.ndarray) -> None:
        rows = x.shape[0]; cols = x.shape[1]
        for r in _nb.prange(rows):
            # find max
            m = x[r, 0]
            for c in range(1, cols):
                v = x[r, c]; m = v if v > m else m
            # exp and sum
            s = 0.0
            for c in range(cols):
                v = _math.exp(x[r, c] - m); out[r, c] = v; s += v
            # normalize
            inv_s = 1.0 / s
            for c in range(cols):
                out[r, c] *= inv_s

    @_nb.njit(parallel=True, fastmath=True, cache=True)
    def _log_softmax_rows(x: _np.ndarray, out: _np.ndarray) -> None:
        rows = x.shape[0]; cols = x.shape[1]
        for r in _nb.prange(rows):
            m = x[r, 0]
            for c in range(1, cols):
                v = x[r, c]; m = v if v > m else m
            s = 0.0
            for c in range(cols):
                s += _math.exp(x[r, c] - m)
            log_sum = _math.log(s) + m
            for c in range(cols):
                out[r, c] = x[r, c] - log_sum

else:
    # Stubs so callers can always import
    def _relu(x, out): import numpy as np; np.maximum(x, 0, out=out.reshape(x.shape))
    def _mul(a, b, out): import numpy as np; np.multiply(a, b, out=out.reshape(a.shape))
    def _add(a, b, out): import numpy as np; np.add(a, b, out=out.reshape(a.shape))
    def _sub(a, b, out): import numpy as np; np.subtract(a, b, out=out.reshape(a.shape))
    def _fma(a, b, out): import numpy as np; np.multiply(a, b, out=out.reshape(a.shape)); np.add(out.reshape(a.shape), a, out=out.reshape(a.shape))
    def _mul_scalar(a, s, out): import numpy as np; np.multiply(a, s, out=out.reshape(a.shape))
    def _add_scalar(a, s, out): import numpy as np; np.add(a, s, out=out.reshape(a.shape))
    def _neg(a, out): import numpy as np; np.negative(a, out=out.reshape(a.shape))
    def _maximum(a, b, out): import numpy as np; np.maximum(a, b, out=out.reshape(a.shape))
    def _minimum(a, b, out): import numpy as np; np.minimum(a, b, out=out.reshape(a.shape))
    def _truediv(a, b, out): import numpy as np; np.true_divide(a, b, out=out.reshape(a.shape))
    def _truediv_scalar(a, s, out): import numpy as np; np.true_divide(a, s, out=out.reshape(a.shape))
    def _softmax_rows(x, out): import numpy as np; m=x.max(1,keepdims=True); e=np.exp(x-m); out[:]=e/e.sum(1,keepdims=True)
    def _log_softmax_rows(x, out): import numpy as np; m=x.max(1,keepdims=True); lse=np.log(np.exp(x-m).sum(1,keepdims=True))+m; out[:]=x-lse
