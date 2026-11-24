"""
Pure-Python NDArray — the foundation of vectorphile's array engine.

Storage: Python's built-in `array.array` (typed flat buffer, no numpy).
Layout:  row-major (C order), stride-based views.
Math:    all ops implemented in pure Python with optional numpy fast-path.
"""
from __future__ import annotations
import array as _array
import math
import operator
import itertools
import builtins as _builtins
builtins_all = _builtins.all
builtins_sum = _builtins.sum
import ctypes
from typing import Union, Tuple, List, Optional, Iterable

try:
    import numpy as _npx
except Exception:
    _npx = None

try:
    from vectorphile._core import numba_kernels as _nbk
    _NUMBA = _nbk._NUMBA
except Exception:
    _nbk = None
    _NUMBA = False

try:
    from vectorphile._core import buf_pool as _bp
    _BP = True
except Exception:
    _bp = None
    _BP = False

try:
    from vectorphile._core import arr_pool as _ap
    _AP = True
except Exception:
    _ap = None
    _AP = False

# ── C extension: AVX2/FMA kernels — primary fast path ────────────────────────
# Use LazyLoader so the DLL scan is deferred to first actual C-extension call.
# This keeps `import vectorphile` fast in spawned processes (e.g. distributed
# workers) that never call AVX2 kernels.
try:
    import sys as _sys_lazy
    import importlib.util as _iutil_lazy
    _spec_lazy = _iutil_lazy.find_spec('vectorphile._core._vp_core')
    if _spec_lazy is not None:
        _spec_lazy.loader = _iutil_lazy.LazyLoader(_spec_lazy.loader)
        _C = _iutil_lazy.module_from_spec(_spec_lazy)
        _sys_lazy.modules['vectorphile._core._vp_core'] = _C
        _spec_lazy.loader.exec_module(_C)
        _C_EXT = True
    else:
        _C = None
        _C_EXT = False
    del _sys_lazy, _iutil_lazy, _spec_lazy
except Exception:
    try:
        from vectorphile._core import _vp_core as _C
        _C_EXT = True
    except Exception:
        _C = None
        _C_EXT = False

# Torch bridge removed — vectorphile uses only its own C/CUDA extension + NumPy.
_torch = None
_TORCH = False

# ── Kernel bridge: try cffi C kernels first, then Mojo .so ───────────────────
try:
    from vectorphile._mojo import _vp_cffi_loader as _cffi_kern
    _CFFI = _cffi_kern.available()
except Exception:
    _cffi_kern = None  # type: ignore
    _CFFI = False

try:
    from vectorphile._mojo import loader as _mojo
    _MOJO = _mojo.available()
except Exception:
    _mojo = None  # type: ignore
    _MOJO = False

_KERN = _CFFI or _MOJO

def _kern_fn(name: str):
    if name is None: return None
    if _CFFI: return _cffi_kern.get(name)
    if _MOJO: return _mojo.get(name)
    return None

def _mojo_ptr(arr: '_array.array') -> ctypes.c_void_p:
    """Return ctypes pointer to the flat buffer of a Python array.array."""
    addr, _ = arr.buffer_info()
    return ctypes.cast(addr, ctypes.POINTER(ctypes.c_float))

def _alloc_f32(n: int) -> '_array.array':
    """Allocate a zero-filled float32 array.array of length n, as fast as possible."""
    return _array.array('f', bytearray(4 * n))

def _kern_ptr(arr: '_array.array'):
    """Return kernel-compatible float* for the given array.array buffer."""
    if _CFFI:
        return _cffi_kern.ptr(arr)
    return _mojo_ptr(arr)


# ── dtype helpers ─────────────────────────────────────────────────────────────

class _DType(str):
    """String subclass for dtype names; also compares equal to numpy type/dtype objects."""
    _np = None
    _np_checked = False
    def __eq__(self, other):
        if isinstance(other, (str, _DType)): return str.__eq__(self, other)
        if not _DType._np_checked:
            _DType._np_checked = True
            try:
                import numpy as _np_mod
                _DType._np = _np_mod
            except ImportError:
                pass
        if _DType._np is not None:
            try:
                _np = _DType._np
                if isinstance(other, type): other = _np.dtype(other)
                if isinstance(other, _np.dtype): return str.__eq__(self, other.name)
            except TypeError: pass
        return NotImplemented
    def __hash__(self): return str.__hash__(self)
    def __contains__(self, item): return str.__contains__(self, item)


_TYPECODES = {
    'float32': 'f', 'float64': 'd', 'int8': 'b', 'int16': 'h',
    'int32': 'i', 'int64': 'l', 'uint8': 'B', 'uint16': 'H',
    'uint32': 'I', 'uint64': 'L', 'bool': 'B',
    # bfloat16 is stored as uint16; handled via _bf16_to_f32 / _f32_to_bf16
    'bfloat16': 'H',
}
_DTYPES = {v: k for k, v in _TYPECODES.items()}
_DTYPE_DEFAULTS = {
    'float32': 0.0, 'float64': 0.0, 'int8': 0, 'int16': 0,
    'int32': 0, 'int64': 0, 'uint8': 0, 'uint16': 0, 'uint32': 0, 'uint64': 0, 'bool': 0,
    'bfloat16': 0,
}


def _f32_to_bf16_int(f: float) -> int:
    """Round float32 to bfloat16 (truncate lower 16 bits of mantissa)."""
    import struct
    bits = struct.unpack('I', struct.pack('f', float(f)))[0]
    # Round to nearest even
    round_bit = (bits >> 16) & 1
    sticky = bits & 0x7FFF
    if sticky > 0x4000 or (sticky == 0x4000 and round_bit):
        bits += 0x8000
    return (bits >> 16) & 0xFFFF


def _bf16_int_to_f32(bf: int) -> float:
    """Convert bfloat16 (uint16) back to float32."""
    import struct
    bits = (int(bf) & 0xFFFF) << 16
    return struct.unpack('f', struct.pack('I', bits))[0]


def _erf_scalar(x: float) -> float:
    """Horner-form rational approximation, max error 1.5e-7 (Abramowitz & Stegun 7.1.26)."""
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * math.exp(-x * x))


def _resolve_dtype(dtype):
    if dtype is None: return _DType('float32')
    if isinstance(dtype, _DType): return dtype
    if isinstance(dtype, str): return _DType(dtype)
    # Python built-in types
    _PYTYPE_MAP = {bool: 'bool', int: 'int64', float: 'float64'}
    if dtype in _PYTYPE_MAP: return _DType(_PYTYPE_MAP[dtype])
    # numpy type/class (e.g., np.float64, np.float32) or dtype instance
    try:
        if _npx is not None:
            d = _npx.dtype(dtype)
            if d.name in _TYPECODES: return _DType(d.name)
            # map numpy kind
            _KIND_MAP = {'f': 'float32', 'd': 'float64', 'i': 'int32', 'l': 'int64', '?': 'bool'}
            if d.kind + str(d.itemsize) in _TYPECODES: return _DType(d.kind + str(d.itemsize))
    except Exception: pass
    # numpy dtype objects — string fallback
    try:
        s = str(dtype)  # e.g. 'float32', 'int64'
        if s in _TYPECODES: return _DType(s)
        import re as _re
        m = _re.search(r"'([^']+)'", s)
        if m and m.group(1) in _TYPECODES: return _DType(m.group(1))
        # strip numpy. prefix
        if s.startswith('numpy.'): s = s[6:]
        if s in _TYPECODES: return _DType(s)
        return _DType(s)
    except Exception: return _DType('float32')


def _prod(shape):
    p = 1
    for s in shape: p *= s
    return p


_c_strides_cache = {}

def _c_strides(shape):
    """Row-major strides in element counts."""
    if not shape: return ()
    cached = _c_strides_cache.get(shape)
    if cached is not None: return cached
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i+1] * shape[i+1]
    result = tuple(strides)
    if len(_c_strides_cache) < 512:  # limit cache size
        _c_strides_cache[shape] = result
    return result


# ── NDArray ────────────────────────────────────────────────────────────────────

class NDArray:
    """
    Pure-Python N-dimensional array with typed flat storage.

    Attributes
    ----------
    _buf   : array.array — flat typed buffer
    shape  : tuple[int, ...]
    strides: tuple[int, ...] — element strides (not bytes)
    dtype  : str
    offset : int — starting element in _buf
    """

    __slots__ = ('_buf', '_np_buf', '_cuda_buf', 'shape', 'strides', 'dtype', 'offset', '_owns_buf', '__weakref__')

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self, buf: _array.array, shape: tuple, strides: tuple,
                 dtype: str, offset: int = 0, owns_buf: bool = True):
        self._buf = buf
        self._cuda_buf = None  # CudaBuffer when tensor lives on GPU
        self._np_buf = None  # optional numpy backing; set by _np_wrap for lazy materialization
        self.shape = tuple(shape)
        self.strides = tuple(strides)
        self.dtype = dtype if isinstance(dtype, _DType) else _DType(dtype) if dtype else _DType('float32')
        self.offset = offset
        self._owns_buf = owns_buf

    def __del__(self):
        """Return buffers to pool when this NDArray is the sole owner."""
        try:
            import sys as _sys
            if self._owns_buf:
                # Return array.array buffer to arr_pool
                if _AP and self._buf is not None and self.offset == 0:
                    if _sys.getrefcount(self._buf) <= 2:
                        _ap.release(self._buf)
                # Return numpy backing to numpy pool
                if _BP and self._np_buf is not None:
                    if _sys.getrefcount(self._np_buf) <= 2:
                        _bp.release(self._np_buf)
        except Exception:
            pass

    def _get_buf(self) -> '_array.array':
        """Return array.array buffer, materializing from _np_buf if needed (cached).
        Clears _np_buf after materializing to prevent dual-buffer stale state."""
        if self._buf is None:
            tc = _TYPECODES.get(self.dtype, 'd')
            arr = self._np_buf
            if arr is None:
                raise ValueError("NDArray has no buffer")
            c = arr if arr.flags['C_CONTIGUOUS'] else _npx.ascontiguousarray(arr)
            self._buf = _array.array(tc)
            self._buf.frombytes(c.astype(tc).tobytes())
            self._np_buf = None  # prevent stale dual-buffer: _buf is now sole source of truth
        return self._buf

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def ndim(self) -> int: return len(self.shape)

    @property
    def size(self) -> int: return _prod(self.shape)

    @property
    def nbytes(self) -> int: return self.itemsize * self.size

    @property
    def itemsize(self) -> int:
        if self._buf is not None:
            return self._buf.itemsize
        return _ITEMSIZES.get(_TYPECODES.get(self.dtype, 'd'), 8)

    @property
    def T(self) -> 'NDArray': return self.transpose()

    # ── Flat indexing helper ──────────────────────────────────────────────────

    def _flat_idx(self, idx: tuple) -> int:
        pos = self.offset
        for i, (ix, st) in enumerate(zip(idx, self.strides)):
            pos += ix * st
        return pos

    def _iter_indices(self):
        """Iterate all multi-indices in C order."""
        return itertools.product(*[range(s) for s in self.shape])

    # ── Getitem / Setitem ─────────────────────────────────────────────────────

    def __getitem__(self, key):
        # normalize numpy scalar types to Python builtins
        _type_name = type(key).__name__
        if _type_name.startswith('int') and hasattr(key, 'item'):
            key = int(key)
        elif _type_name.startswith('float') and hasattr(key, 'item'):
            key = float(key)
        elif _type_name.startswith('bool') and hasattr(key, 'item'):
            key = bool(key)
        if key is None:
            # np.newaxis — insert a size-1 axis at front
            return self[(None,)]
        if isinstance(key, int):
            if self.ndim == 1:
                k = key if key >= 0 else self.shape[0] + key
                # Prefer numpy path to avoid materializing _buf (which disconnects views)
                if self._np_buf is not None and self.offset == 0:
                    return float(self._np_buf.flat[k * self.strides[0]])
                return self._get_buf()[self.offset + k * self.strides[0]]
            # first axis slice — must use _get_buf (NDArray constructor requires _buf)
            if self._np_buf is not None:
                # keep numpy-backed; return a numpy-backed view slice
                new_np = self._np_buf[key]
                nd = NDArray.__new__(NDArray)
                nd._buf = None; nd._np_buf = new_np if new_np.ndim > 0 else new_np.reshape((1,))
                nd.shape = self.shape[1:]
                nd.strides = self.strides[1:]
                nd.dtype = self.dtype; nd.offset = 0; nd._owns_buf = False
                # For 0-d result (scalar slice), return float
                if not self.shape[1:]:
                    return float(new_np)
                nd._np_buf = new_np
                return nd
            new_offset = self.offset + key * self.strides[0]
            return NDArray(self._get_buf(), self.shape[1:], self.strides[1:],
                           self.dtype, new_offset, owns_buf=False)
        if isinstance(key, slice):
            start, stop, step = key.indices(self.shape[0])
            new_shape = (len(range(start, stop, step)),) + self.shape[1:]
            new_strides = (self.strides[0] * step,) + self.strides[1:]
            new_offset = self.offset + start * self.strides[0]
            return NDArray(self._get_buf(), new_shape, new_strides, self.dtype,
                           new_offset, owns_buf=False)
        if isinstance(key, tuple):
            if len(key) == 0: return self
            # Expand ellipsis into the appropriate number of full slices
            if Ellipsis in key:
                n_ellipsis = sum(1 for k in key if k is Ellipsis)
                n_other = len(key) - n_ellipsis
                n_expand = max(0, len(self.shape) - n_other)
                new_key = []
                for k in key:
                    if k is Ellipsis:
                        new_key.extend([slice(None)] * n_expand)
                        n_expand = 0  # only first ellipsis expands
                    else:
                        new_key.append(k)
                key = tuple(new_key)
            # apply each index along its corresponding axis
            result = self
            axis = 0
            for k in key:
                if isinstance(k, (int, float)) or (hasattr(k, 'item') and not isinstance(k, bool)):
                    _k = int(k)
                    _k = _k if _k >= 0 else result.shape[axis] + _k
                    new_offset = result.offset + _k * result.strides[axis]
                    new_shape = result.shape[:axis] + result.shape[axis+1:]
                    new_strides = result.strides[:axis] + result.strides[axis+1:]
                    if not new_shape:
                        result = result._get_buf()[new_offset]
                        axis = 0
                        break
                    result = NDArray(result._get_buf(), new_shape, new_strides,
                                     result.dtype, new_offset, owns_buf=False)
                    # axis stays same since dim was removed
                elif isinstance(k, slice):
                    start, stop, step = k.indices(result.shape[axis])
                    n = len(range(start, stop, step))
                    new_shape = result.shape[:axis] + (n,) + result.shape[axis+1:]
                    new_strides = result.strides[:axis] + (result.strides[axis] * step,) + result.strides[axis+1:]
                    new_offset = result.offset + start * result.strides[axis]
                    result = NDArray(result._get_buf(), new_shape, new_strides,
                                     result.dtype, new_offset, owns_buf=False)
                    axis += 1
                elif k is None:
                    # np.newaxis — insert size-1 axis at current position
                    new_shape = result.shape[:axis] + (1,) + result.shape[axis:]
                    new_strides = result.strides[:axis] + (0,) + result.strides[axis:]
                    result = NDArray(result._get_buf(), new_shape, new_strides,
                                     result.dtype, result.offset, owns_buf=False)
                    axis += 1
                else:
                    # fall back for other types
                    result = result[k]
                    axis += 1
            return result
        # Boolean mask indexing — select elements where mask is True
        if isinstance(key, NDArray) and 'bool' in key.dtype:
            mask_flat = key.to_flat_list()
            tc = _TYPECODES.get(self.dtype, 'd')
            flat = []
            for i, m in enumerate(mask_flat):
                if m:
                    row = self[i]
                    if isinstance(row, NDArray):
                        flat.extend(row.to_flat_list())
                    else:
                        flat.append(row)
            count = sum(1 for m in mask_flat if m)
            out_shape = (count,) + self.shape[1:]
            if 'float' in self.dtype: flat = [float(v) for v in flat]
            else: flat = [int(v) for v in flat]
            buf = _array.array(tc, flat)
            return NDArray(buf, out_shape, _c_strides(out_shape), self.dtype)
        # NDArray integer fancy indexing (gather along axis 0)
        if isinstance(key, NDArray) and 'int' in key.dtype:
            idx_flat = key.to_flat_list()
            idx_shape = key.shape
            # each index selects one row along axis 0; result shape = idx_shape + self.shape[1:]
            out_shape = idx_shape + self.shape[1:]
            tc = _TYPECODES.get(self.dtype, 'd')
            flat = []
            for flat_i in idx_flat:
                row = self[int(flat_i)]
                if isinstance(row, NDArray):
                    flat.extend(row.to_flat_list())
                else:
                    flat.append(row)
            if 'float' in self.dtype: flat = [float(v) for v in flat]
            else: flat = [int(v) for v in flat]
            buf = _array.array(tc, flat)
            return NDArray(buf, out_shape, _c_strides(out_shape), self.dtype)
        # boolean mask / fancy indexing — fall back to list
        if hasattr(key, '__iter__'):
            indices = list(key)
            flat = self.to_flat_list()
            shape = self.shape
            result_data = []
            for idx in indices:
                if isinstance(idx, bool):
                    continue  # handled below
                result_data.append(self[int(idx)])
            return array([self[int(i)] for i in indices], dtype=self.dtype)
        raise IndexError(f"Unsupported index type: {type(key)}")

    def __setitem__(self, key, value):
        _type_name = type(key).__name__
        if _type_name.startswith('int') and hasattr(key, 'item'):
            key = int(key)
        # For numpy-backed arrays, modify _np_buf in-place (preserves views/sharing)
        if self._buf is None and self._np_buf is not None:
            if isinstance(value, NDArray):
                import numpy as _np_si
                self._np_buf[key] = _np_si.array(value)
            else:
                self._np_buf[key] = value
            return
        buf = self._get_buf()    # materialize first (may need _np_buf)
        self._np_buf = None      # now safe to clear — _buf is live
        if isinstance(key, int):
            k = key if key >= 0 else self.shape[0] + key
            if self.ndim == 1:
                buf[self.offset + k * self.strides[0]] = self.dtype == 'bool' and bool(value) or value
            else:
                target = self[k]
                target_flat = list(target._iter_indices())
                if isinstance(value, NDArray):
                    for idx in target_flat:
                        buf[target._flat_idx(idx)] = value._get_buf()[value._flat_idx(idx)]
                else:
                    for idx in target_flat:
                        buf[target._flat_idx(idx)] = value
        elif isinstance(key, tuple) and builtins_all(isinstance(k, int) for k in key) and len(key) == self.ndim:
            buf[self._flat_idx(key)] = value
        elif isinstance(key, tuple) and builtins_all(isinstance(k, (int, slice)) for k in key):
            view = self[key]
            if not isinstance(view, NDArray):
                buf[self._flat_idx(key)] = float(value) if 'float' in self.dtype else int(value)
            else:
                if isinstance(value, NDArray):
                    val_nd = value
                else:
                    try:
                        import numpy as _np_si
                        if isinstance(value, _np_si.ndarray):
                            val_nd = array(value)
                        else:
                            val_nd = _broadcast_scalar(value, view.shape, view.dtype)
                    except ImportError:
                        val_nd = _broadcast_scalar(value, view.shape, view.dtype)
                for idx in view._iter_indices():
                    view._get_buf()[view._flat_idx(idx)] = val_nd._get_buf()[val_nd._flat_idx(idx)] if isinstance(val_nd, NDArray) else val_nd
        else:
            try:
                import numpy as _np
                arr = _np.array(self)
                if isinstance(value, NDArray): value = _np.array(value)
                arr[key] = value
                flat = arr.ravel().tolist()
                for i, idx in enumerate(self._iter_indices()):
                    v = flat[i]
                    buf[self._flat_idx(idx)] = float(v) if 'float' in self.dtype else int(v)
            except ImportError:
                raise IndexError(f"Complex setitem requires numpy: {type(key)}")

    # ── Conversion ────────────────────────────────────────────────────────────

    def to_flat_list(self) -> list:
        # Fast path for numpy-backed arrays: convert directly without materializing array.array
        if self._buf is None and self._np_buf is not None and self.offset == 0:
            return self._np_buf.ravel().tolist()
        expected = _c_strides(self.shape)
        n = _prod(self.shape)
        buf = self._get_buf()
        if self.strides == expected and self.offset == 0 and len(buf) == n:
            return buf.tolist()
        if self.strides == expected and len(buf) >= self.offset + n:
            return buf[self.offset:self.offset + n].tolist()
        result = []
        for idx in self._iter_indices():
            result.append(buf[self._flat_idx(idx)])
        return result

    def tolist(self):
        _is_bool = 'bool' in str(self.dtype)
        if self.ndim == 0:
            v = self._get_buf()[self.offset]
            return bool(v) if _is_bool else v
        if self.ndim == 1:
            buf = self._get_buf()
            vals = [buf[self.offset + i*self.strides[0]] for i in range(self.shape[0])]
            return [bool(v) for v in vals] if _is_bool else vals
        return [self[i].tolist() for i in range(self.shape[0])]

    def _to_list(self):
        return self.to_flat_list()

    @classmethod
    def _from_numpy(cls, arr) -> 'NDArray':
        dtype = str(arr.dtype)
        tc = _TYPECODES.get(dtype, 'd')
        flat = arr.ravel().tolist()
        buf = _array.array(tc, flat)
        shape = tuple(arr.shape)
        return cls(buf, shape, _c_strides(shape), dtype)

    # ── Reshape / View ────────────────────────────────────────────────────────

    def reshape(self, *new_shape) -> 'NDArray':
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            new_shape = tuple(new_shape[0])
        # handle -1
        new_shape = list(new_shape)
        neg = [i for i, s in enumerate(new_shape) if s == -1]
        if neg:
            total = self.size
            known = _prod(s for s in new_shape if s != -1)
            new_shape[neg[0]] = total // known
        new_shape = tuple(new_shape)
        assert _prod(new_shape) == self.size
        # return a view sharing the same buffer when possible (contiguous)
        expected = _c_strides(self.shape)
        is_contiguous = (self.strides == expected and self.offset == 0)
        if is_contiguous:
            if self._np_buf is not None:
                # numpy-backed (or has both): prefer numpy path to keep views sharing _np_buf
                nd = NDArray.__new__(NDArray)
                nd._buf = None
                nd._np_buf = self._np_buf.reshape(new_shape)
                nd.shape = new_shape
                nd.strides = _c_strides(new_shape)
                nd.dtype = self.dtype
                nd.offset = 0
                nd._owns_buf = False
                return nd
            return NDArray(self._get_buf(), new_shape, _c_strides(new_shape),
                           self.dtype, self.offset, owns_buf=False)
        # non-contiguous: compact first, then reshape
        flat = self.to_flat_list()
        tc = _TYPECODES.get(self.dtype, 'd')
        buf = _array.array(tc, flat)
        return NDArray(buf, new_shape, _c_strides(new_shape), self.dtype)

    def flatten(self) -> 'NDArray':
        return self.reshape(self.size)

    def ravel(self) -> 'NDArray':
        return self.flatten()

    def squeeze(self, axis=None) -> 'NDArray':
        if axis is None:
            new_shape = tuple(s for s in self.shape if s != 1)
        else:
            new_shape = tuple(s for i, s in enumerate(self.shape) if i != axis or s != 1)
        return self.reshape(new_shape) if new_shape != self.shape else self

    def unsqueeze(self, axis: int) -> 'NDArray':
        new_shape = list(self.shape); new_shape.insert(axis, 1)
        return self.reshape(tuple(new_shape))

    def expand_dims(self, axis: int) -> 'NDArray':
        return self.unsqueeze(axis)

    def transpose(self, axes=None) -> 'NDArray':
        if axes is None:
            axes = tuple(range(self.ndim - 1, -1, -1))
        new_shape = tuple(self.shape[i] for i in axes)
        # Fast path for 2D matrix transpose
        if self.ndim == 2 and axes == (1, 0):
            # Try C extension blocked transpose first
            result_c = _transpose_c(self.contiguous())
            if result_c is not None:
                return result_c
            m, n = self.shape
            cont = self.contiguous()
            tc = _TYPECODES.get(cont.dtype, 'd')
            is_z = _ITEMSIZES.get(tc, 8)
            res_buf = _array.array(tc, bytearray(is_z * m * n))
            src = cont._get_buf(); dst = res_buf
            for i in range(m):
                for j in range(n):
                    dst[j * m + i] = src[i * n + j]
            return NDArray(res_buf, (n, m), _c_strides((n, m)), self.dtype)
        # General case
        result = zeros(new_shape, dtype=self.dtype)
        src_buf = self._get_buf(); dst_buf = result._get_buf()
        for idx in self._iter_indices():
            new_idx = tuple(idx[i] for i in axes)
            dst_buf[result._flat_idx(new_idx)] = src_buf[self._flat_idx(idx)]
        return result

    def copy(self) -> 'NDArray':
        tc = _TYPECODES.get(self.dtype, 'd')
        expected = _c_strides(self.shape)
        if self.strides == expected and self.offset == 0:
            if self._buf is None and self._np_buf is not None:
                nd = NDArray.__new__(NDArray)
                nd._buf = None
                nd._np_buf = self._np_buf.copy()
                nd.shape = self.shape
                nd.strides = self.strides
                nd.dtype = self.dtype
                nd.offset = 0
                nd._owns_buf = True
                return nd
            new_buf = _array.array(tc)
            new_buf.frombytes(self._get_buf().tobytes())
            return NDArray(new_buf, self.shape, _c_strides(self.shape), self.dtype)
        # Non-contiguous: fast path for 2D float32 transpose via C AVX2 kernel
        if (_C_EXT and self.dtype == 'float32' and self.ndim == 2
                and self._buf is not None):
            rows, cols = self.shape
            sr, sc = self.strides
            # Detect pure transpose: strides are (1, rows) in element units
            if sr == 1 and sc == rows:
                try:
                    out_buf = _array.array('f', bytearray(4 * rows * cols))
                    _C.transpose_f32(self._buf, out_buf, cols, rows)
                    return NDArray(out_buf, (rows, cols), _c_strides((rows, cols)), self.dtype)
                except Exception:
                    pass
        # numpy fast path for non-contiguous arrays
        if _npx is not None and self.dtype in _TYPECODES:
            try:
                view = _np_view(self)
                arr = _npx.ascontiguousarray(view, dtype=_TYPECODES.get(self.dtype, 'd'))
                return _np_wrap(arr, self.dtype)
            except Exception:
                pass
        flat = self.to_flat_list()
        buf = _array.array(tc, flat)
        return NDArray(buf, self.shape, _c_strides(self.shape), self.dtype)

    def astype(self, dtype) -> 'NDArray':
        dtype = _resolve_dtype(dtype)
        if dtype == self.dtype:
            return self.copy()
        tc = _TYPECODES.get(dtype, 'd')
        # BF16 → float32: decode stored uint16 back to float
        if self.dtype == 'bfloat16' and dtype == 'float32':
            flat_u16 = self.to_flat_list()
            flat = [_bf16_int_to_f32(int(x)) for x in flat_u16]
            buf = _array.array('f', flat)
            return NDArray(buf, self.shape, _c_strides(self.shape), 'float32')
        # float32 → BF16: encode to uint16
        if dtype == 'bfloat16':
            flat_f32 = [float(v) for v in self.to_flat_list()]
            flat = [_f32_to_bf16_int(v) for v in flat_f32]
            buf = _array.array('H', flat)
            return NDArray(buf, self.shape, _c_strides(self.shape), 'bfloat16')
        if _npx is not None and dtype in _TYPECODES and self.dtype in _TYPECODES:
            try:
                v = _np_view(self)
                with _npx.errstate(all='ignore'):
                    return _np_wrap(v != 0 if dtype == 'bool' else v, dtype)
            except Exception:
                pass
        if dtype == 'bool':
            flat = [int(bool(v)) for v in self.to_flat_list()]
        elif 'float' in dtype:
            flat = [float(v) for v in self.to_flat_list()]
        else:
            flat = [int(v) for v in self.to_flat_list()]
        buf = _array.array(tc, flat)
        return NDArray(buf, self.shape, _c_strides(self.shape), dtype)

    def contiguous(self) -> 'NDArray':
        expected = _c_strides(self.shape)
        if self.strides == expected and self.offset == 0:
            return self  # already contiguous, no copy
        return self.copy()

    # ── Reductions ────────────────────────────────────────────────────────────

    def sum(self, axis=None, keepdims=False, **kwargs):
        r = _np_axis_reduce(self, 'sum', axis, keepdims)
        if r is not None:
            return r
        # Global reduce via C extension sum_f32
        if axis is None and _C_EXT and str(self.dtype) == 'float32' and _is_c_contiguous_f32(self) and self._buf is not None:
            try:
                total = _C.sum_f32(self._buf)
                result = NDArray(_array.array('f', [total]), (1,) * self.ndim if keepdims else (), _c_strides((1,) * self.ndim) if keepdims else (), _DType('float32'))
                return result
            except Exception:
                pass
        return _reduce(self, math.fsum if 'float' in self.dtype else builtins_sum, axis, keepdims, 0.0)

    def __array__(self, dtype=None, copy=None):
        """Allow numpy to consume NDArray via np.asarray / np.sum etc."""
        flat = self.to_flat_list()
        try:
            import numpy as _np
            arr = _np.array(flat, dtype=dtype).reshape(self.shape) if self.shape else _np.array(flat[0] if flat else 0, dtype=dtype)
            return arr
        except ImportError:
            raise TypeError("numpy is not available for __array__ conversion")

    def mean(self, axis=None, keepdims=False, **kwargs):
        # promote bool/int to float for mean computation
        a = self.astype('float64') if 'int' in self.dtype or self.dtype == 'bool' else self
        # C extension fast path: global mean on contiguous float32
        if _C_EXT and axis is None and str(a.dtype) == 'float32' and _is_c_contiguous_f32(a):
            try:
                total = _C.sum_f32(a._buf)
                n = _prod(a.shape) if a.shape else 1
                val = float(total) / n
                scalar_buf = _array.array('f', [val])
                shape = (1,) * self.ndim if keepdims else ()
                return NDArray(scalar_buf, shape, _c_strides(shape), _DType('float32'))
            except Exception:
                pass
        # Axis reduce: torch/numpy fast path (uses pool for non-scalar outputs)
        r = _np_axis_reduce(a, 'mean', axis, keepdims, float_only=True)
        if r is not None:
            return r
        # Global reduce: torch, no pool (scalar result avoids aliasing)
        if axis is None and _TORCH and a.dtype in _FLOAT_DTYPES:
            try:
                view = _np_view(a)
                if view.flags['C_CONTIGUOUS']:
                    r_t = _torch.mean(_torch.from_numpy(view))
                    scalar = _npx.asarray(r_t.item(), dtype=_TYPECODES.get(a.dtype, 'd'))
                    if keepdims:
                        return _nd_from_numpy(scalar.reshape((1,) * self.ndim), a.dtype)
                    return _nd_from_numpy(scalar.reshape(()), a.dtype)
            except Exception:
                pass
        s = a.sum(axis=axis, keepdims=keepdims)
        if axis is None:
            n = self.size
        elif isinstance(axis, int):
            n = self.shape[axis % self.ndim]
        else:
            n = _prod(self.shape[ax % self.ndim] for ax in axis)
        return s / n

    def max(self, axis=None, keepdims=False, **kwargs):
        if _KERN and self.dtype == 'float32':
            kern_full = _kern_fn('vp_max')
            kern_axis = _kern_fn('vp_max_axis')
            if axis is None and kern_full is not None:
                cont = self.contiguous(); cont_buf = cont._get_buf()
                result = float(kern_full(_kern_ptr(cont_buf), cont_buf.__len__()))
                if keepdims: return _broadcast_scalar(result, (1,) * self.ndim, self.dtype)
                return _broadcast_scalar(result, (), self.dtype)
            if axis is not None and kern_axis is not None and isinstance(axis, int):
                ax = axis % self.ndim
                cont = self.contiguous(); cont_buf = cont._get_buf()
                outer = _prod(self.shape[:ax]) if ax > 0 else 1
                axis_len = self.shape[ax]
                inner = _prod(self.shape[ax+1:]) if ax < self.ndim-1 else 1
                out_shape = tuple(s for i,s in enumerate(self.shape) if i != ax)
                out_n = outer * inner
                dst_buf = _alloc_f32(out_n)
                kern_axis(_kern_ptr(cont_buf), _kern_ptr(dst_buf), outer, axis_len, inner)
                if keepdims:
                    ks = tuple(1 if i == ax else s for i,s in enumerate(self.shape))
                    return NDArray(dst_buf, ks, _c_strides(ks), 'float32')
                return NDArray(dst_buf, out_shape, _c_strides(out_shape), 'float32')
        r = _np_axis_reduce(self, 'max', axis, keepdims)
        if r is not None:
            return r
        return _reduce_fn(self, max, axis, keepdims)

    def min(self, axis=None, keepdims=False, **kwargs):
        if _KERN and self.dtype == 'float32':
            kern_full = _kern_fn('vp_min')
            kern_axis = _kern_fn('vp_min_axis')
            if axis is None and kern_full is not None:
                cont = self.contiguous(); cont_buf = cont._get_buf()
                result = float(kern_full(_kern_ptr(cont_buf), cont_buf.__len__()))
                if keepdims: return _broadcast_scalar(result, (1,) * self.ndim, self.dtype)
                return _broadcast_scalar(result, (), self.dtype)
            if axis is not None and kern_axis is not None and isinstance(axis, int):
                ax = axis % self.ndim
                cont = self.contiguous(); cont_buf = cont._get_buf()
                outer = _prod(self.shape[:ax]) if ax > 0 else 1
                axis_len = self.shape[ax]
                inner = _prod(self.shape[ax+1:]) if ax < self.ndim-1 else 1
                out_shape = tuple(s for i,s in enumerate(self.shape) if i != ax)
                out_n = outer * inner
                dst_buf = _alloc_f32(out_n)
                kern_axis(_kern_ptr(cont_buf), _kern_ptr(dst_buf), outer, axis_len, inner)
                if keepdims:
                    ks = tuple(1 if i == ax else s for i,s in enumerate(self.shape))
                    return NDArray(dst_buf, ks, _c_strides(ks), 'float32')
                return NDArray(dst_buf, out_shape, _c_strides(out_shape), 'float32')
        r = _np_axis_reduce(self, 'min', axis, keepdims)
        if r is not None:
            return r
        return _reduce_fn(self, min, axis, keepdims)

    def prod(self, axis=None, keepdims=False):
        def _prod_list(lst):
            r = 1
            for x in lst: r *= x
            return r
        r = _np_axis_reduce(self, 'prod', axis, keepdims, float_only=False)
        if r is not None:
            return r
        return _reduce(self, _prod_list, axis, keepdims, 1.0)

    def std(self, axis=None, keepdims=False, ddof=0):
        # Fast path: global std via C extension two-pass kernel
        if axis is None and str(self.dtype) == 'float32' and _is_c_contiguous_f32(self):
            result = _var_c(self, unbiased=(ddof == 1))
            if result is not None:
                _, var = result
                v = NDArray(_array.array('f', [math.sqrt(var)]), (1,), (1,), 'float32')
                if keepdims:
                    return v.reshape(tuple(1 for _ in self.shape))
                return v.item() if not keepdims else v
        m = self.mean(axis=axis, keepdims=True)
        diff = _elementwise_binary(self, m if isinstance(m, NDArray) else _broadcast_scalar(m, self.shape, self.dtype), operator.sub)
        sq = _elementwise_unary(diff, lambda x: x*x)
        v = sq.mean(axis=axis, keepdims=keepdims)
        return _elementwise_unary(v if isinstance(v, NDArray) else _broadcast_scalar(v, (1,), self.dtype), math.sqrt)

    def var(self, axis=None, keepdims=False, ddof=0):
        # Fast path: global var via C extension two-pass kernel
        if axis is None and str(self.dtype) == 'float32' and _is_c_contiguous_f32(self):
            result = _var_c(self, unbiased=(ddof == 1))
            if result is not None:
                _, var = result
                v = NDArray(_array.array('f', [var]), (1,), (1,), 'float32')
                if keepdims:
                    return v.reshape(tuple(1 for _ in self.shape))
                return var
        m = self.mean(axis=axis, keepdims=True)
        diff = _elementwise_binary(self, m if isinstance(m, NDArray) else _broadcast_scalar(m, self.shape, self.dtype), operator.sub)
        sq = _elementwise_unary(diff, lambda x: x*x)
        return sq.mean(axis=axis, keepdims=keepdims)

    def argmax(self, axis=None):
        if axis is None:
            flat = self.to_flat_list(); return flat.index(max(flat))
        return _argreduce(self, axis, max)

    def argmin(self, axis=None):
        if axis is None:
            flat = self.to_flat_list(); return flat.index(min(flat))
        return _argreduce(self, axis, min)

    def all(self, axis=None, **kwargs):
        return _reduce(self, _builtins.all, axis, False, True)

    def any(self, axis=None, **kwargs):
        return _reduce(self, _builtins.any, axis, False, False)

    def nonzero(self):
        indices = [[] for _ in range(self.ndim)]
        buf = self._get_buf()
        for idx in self._iter_indices():
            if buf[self._flat_idx(idx)] != 0:
                for i, ix in enumerate(idx): indices[i].append(ix)
        return tuple(array(ind, dtype='int64') for ind in indices)

    # ── Arithmetic ────────────────────────────────────────────────────────────

    def __add__(self, other): return _binary_op(self, other, operator.add)
    def __radd__(self, other): return _binary_op(other, self, operator.add)
    def __sub__(self, other): return _binary_op(self, other, operator.sub)
    def __rsub__(self, other): return _binary_op(other, self, operator.sub)
    def __mul__(self, other): return _binary_op(self, other, operator.mul)
    def __rmul__(self, other): return _binary_op(other, self, operator.mul)
    def __truediv__(self, other): return _binary_op(self, other, operator.truediv)
    def __rtruediv__(self, other): return _binary_op(other, self, operator.truediv)
    def __floordiv__(self, other): return _binary_op(self, other, operator.floordiv)
    def __mod__(self, other): return _binary_op(self, other, operator.mod)
    def __pow__(self, other): return _binary_op(self, other, operator.pow)
    def __rpow__(self, other): return _binary_op(other, self, operator.pow)
    def __neg__(self): return _elementwise_unary(self, operator.neg)
    def __abs__(self): return _elementwise_unary(self, abs)

    def erf(self) -> 'NDArray':
        return _elementwise_unary(self, _erf_scalar)

    def item(self):
        """Return Python scalar for single-element arrays."""
        assert self.size == 1, f"item() on array of size {self.size}"
        v = self._get_buf()[self.offset]
        if 'float' in self.dtype: return float(v)
        if 'bool' == self.dtype: return bool(v)
        return int(v)
    def __invert__(self): return _elementwise_unary(self, operator.not_)
    def __lshift__(self, n): return _elementwise_unary(self, lambda x: int(x) << int(n))
    def __rshift__(self, n): return _elementwise_unary(self, lambda x: int(x) >> int(n))
    def __and__(self, n):
        if isinstance(n, NDArray): return _binary_op(self, n, lambda a, b: int(a) & int(b))
        return _elementwise_unary(self, lambda x: int(x) & int(n))
    def __or__(self, other):
        if isinstance(other, NDArray): return _binary_op(self, other, lambda a, b: int(a) | int(b))
        return _elementwise_unary(self, lambda x: int(x) | int(other))

    # comparison
    def __eq__(self, other): return _binary_op(self, other, operator.eq, out_dtype='bool')
    def __ne__(self, other): return _binary_op(self, other, operator.ne, out_dtype='bool')
    def __lt__(self, other): return _binary_op(self, other, operator.lt, out_dtype='bool')
    def __le__(self, other): return _binary_op(self, other, operator.le, out_dtype='bool')
    def __gt__(self, other): return _binary_op(self, other, operator.gt, out_dtype='bool')
    def __ge__(self, other): return _binary_op(self, other, operator.ge, out_dtype='bool')

    def __matmul__(self, other): return matmul(self, other)

    # ── In-place ─────────────────────────────────────────────────────────────

    def _inplace_update(self, result):
        """Write result back into self, respecting views (non-contiguous buffers)."""
        if self is result:
            return self  # C kernel already modified the buffer in-place
        s_buf = self._get_buf()  # materialize first (may need _np_buf)
        self._np_buf = None      # now safe to clear — _buf is live
        r_buf = result._get_buf()
        # Fast path: both are contiguous and the same number of owned elements.
        if (self._is_contiguous() and result._is_contiguous()
                and s_buf.__len__() == r_buf.__len__()):
            s_buf[:] = r_buf
            return self
        flat = result.to_flat_list()
        tc = s_buf.typecode
        for i, idx in enumerate(self._iter_indices()):
            v = flat[i]
            s_buf[self._flat_idx(idx)] = float(v) if tc in ('f', 'd') else int(v)
        return self

    def _is_contiguous(self) -> bool:
        """True when strides match row-major (C) order with no offset."""
        if self.offset != 0:
            return False
        expected = 1
        for i in range(len(self.shape) - 1, -1, -1):
            if self.strides[i] != expected:
                return False
            expected *= self.shape[i]
        return True

    def __iadd__(self, other):
        result = self + other
        return self._inplace_update(result)

    def __isub__(self, other):
        result = self - other
        return self._inplace_update(result)

    def __imul__(self, other):
        return self._inplace_update(self * other)

    def __itruediv__(self, other):
        return self._inplace_update(self / other)

    # ── String ────────────────────────────────────────────────────────────────

    def __repr__(self):
        return f"NDArray({self.tolist()}, dtype={self.dtype}, shape={self.shape})"

    def __len__(self): return self.shape[0] if self.shape else 0

    def __iter__(self):
        for i in range(self.shape[0]): yield self[i]

    def __float__(self):
        assert self.size == 1, "Only size-1 arrays can be converted to float"
        return float(self.to_flat_list()[0])

    def __int__(self):
        assert self.size == 1
        return int(self.to_flat_list()[0])

    def __bool__(self):
        assert self.size == 1
        return bool(self.to_flat_list()[0])

    def __format__(self, format_spec):
        return format(float(self), format_spec) if self.size == 1 else format(repr(self), format_spec)

    # ── Utility ───────────────────────────────────────────────────────────────

    def clip(self, a_min, a_max) -> 'NDArray':
        return _elementwise_unary(self, lambda x: max(a_min, min(a_max, x)) if a_min is not None and a_max is not None
                                  else (max(a_min, x) if a_min is not None else min(a_max, x)))

    def fill(self, value):
        tc = _TYPECODES.get(self.dtype, 'd')
        if tc in ('b', 'h', 'i', 'l', 'q', 'B', 'H', 'I', 'L', 'Q'):
            value = int(value)
        else:
            value = float(value)
        self._np_buf = None
        buf = self._get_buf()
        for idx in self._iter_indices():
            buf[self._flat_idx(idx)] = value

    def diagonal(self, offset=0) -> 'NDArray':
        assert self.ndim == 2
        m, n = self.shape
        result = []
        for i in range(min(m, n)):
            r, c = (i, i + offset) if offset >= 0 else (i - offset, i)
            if 0 <= r < m and 0 <= c < n:
                result.append(self._get_buf()[self._flat_idx((r, c))])
        return array(result, dtype=self.dtype)

    def trace(self) -> float:
        return float(self.diagonal().sum())


# ── Broadcasting ──────────────────────────────────────────────────────────────

def _broadcast_shapes(a_shape, b_shape):
    """Compute broadcast output shape."""
    # pad shorter shape with 1s on the left
    la, lb = len(a_shape), len(b_shape)
    if la < lb: a_shape = (1,) * (lb - la) + tuple(a_shape)
    elif lb < la: b_shape = (1,) * (la - lb) + tuple(b_shape)
    out = []
    for a, b in zip(a_shape, b_shape):
        if a == b: out.append(a)
        elif a == 1: out.append(b)
        elif b == 1: out.append(a)
        else: raise ValueError(f"Cannot broadcast shapes {a_shape} and {b_shape}")
    return tuple(out)


def _broadcast_idx(idx, shape):
    """Map output index to input index given input shape (with broadcasting)."""
    offset = len(idx) - len(shape)
    result = []
    for i, (ix, s) in enumerate(zip(idx[offset:], shape)):
        result.append(0 if s == 1 else ix)
    return tuple(result)


def _broadcast_scalar(val, shape, dtype='float32'):
    tc = _TYPECODES.get(dtype, 'd')
    n = _prod(shape)
    # Cast val to match the typecode (int typecodes can't hold floats)
    if tc in ('b', 'h', 'i', 'l', 'q', 'B', 'H', 'I', 'L', 'Q'):
        try:
            cast_val = int(val)
        except (OverflowError, ValueError):
            cast_val = 0
    else:
        cast_val = float(val)
    buf = _array.array(tc, [cast_val] * n)
    return NDArray(buf, shape, _c_strides(shape), dtype)


# ── Vectorized fast paths on zero-copy NumPy views of the buffer ─────────────

def _relu_scalar(x): return x if x > 0.0 else 0.0
def _sigmoid_scalar(x): return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
def _sign_scalar(x): return 0.0 if x == 0 else (1.0 if x > 0 else -1.0)
def _log_eps_scalar(x): return math.log(x + 1e-300)

_FLOAT_DTYPES = ('float32', 'float64')


def _np_view(a):
    # Fast path: numpy-backed, contiguous — return _np_buf directly (zero copy)
    if a._np_buf is not None and a.offset == 0:
        expected = _c_strides(a.shape)
        if a.strides == expected:
            v = a._np_buf
            return v if v.shape == a.shape else v.reshape(a.shape)
    c = a.contiguous()
    if c._np_buf is not None:
        return c._np_buf
    buf = c._get_buf()
    return _npx.frombuffer(buf, dtype=buf.typecode, count=_prod(a.shape)).reshape(a.shape)


def _nd_from_numpy(arr, dtype):
    """Create NDArray backed directly by a numpy array (zero copy, lazy array.array)."""
    nd = NDArray.__new__(NDArray)
    nd._buf = None
    nd._np_buf = arr
    nd.shape = tuple(arr.shape) if arr.ndim > 0 else ()
    nd.strides = _c_strides(nd.shape)
    nd.dtype = dtype if isinstance(dtype, _DType) else _DType(dtype)
    nd.offset = 0
    nd._owns_buf = True
    return nd


def _np_wrap(r, dtype):
    tc = _TYPECODES.get(dtype, 'd')
    shape = tuple(_npx.shape(r))  # ascontiguousarray promotes 0-d to 1-d, so keep the real shape
    # Store as numpy-backed NDArray — no array.array allocation needed
    arr = _npx.ascontiguousarray(r, dtype=tc)
    if arr.ndim == 0:
        arr = arr.reshape(())
    return _nd_from_numpy(arr, dtype)


_ITEMSIZES = {'f': 4, 'd': 8, 'b': 1, 'h': 2, 'i': 4, 'l': 4, 'B': 1, 'H': 2, 'I': 4, 'L': 4}


def _alloc_out(shape, tc):
    """Allocate array.array and return (buf, writable_numpy_view) via ctypes — zero extra copies."""
    n = _prod(shape)
    itemsize = _ITEMSIZES.get(tc, 8)
    buf = _array.array(tc, bytearray(itemsize * n))
    addr, _ = buf.buffer_info()
    ctype = ctypes.c_float if tc == 'f' else ctypes.c_double
    view = _npx.ctypeslib.as_array((ctype * n).from_address(addr)).reshape(shape)
    return buf, view


def _sigmoid_np(x):
    # exp(-|x|) never overflows, so both tails stay exact
    e = _npx.exp(-_npx.abs(x))
    return _npx.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


_NP_UNARY = {}
_NP_UNARY_UFUNC = {}  # subset that support out= parameter
_NP_BINARY = {}
_NP_COMPARE = {}
if _npx is not None:
    _NP_UNARY = {
        math.exp: _npx.exp, math.log: _npx.log, math.log2: _npx.log2, math.log10: _npx.log10,
        math.sqrt: _npx.sqrt, math.tanh: _npx.tanh, math.sin: _npx.sin, math.cos: _npx.cos,
        math.tan: _npx.tan, math.fabs: _npx.abs, abs: _npx.abs, operator.neg: _npx.negative,
        math.floor: _npx.floor, math.ceil: _npx.ceil,
        _relu_scalar: lambda x: _npx.maximum(x, 0),
        _sigmoid_scalar: _sigmoid_np,
        _sign_scalar: _npx.sign,
        _log_eps_scalar: lambda x: _npx.log(x.astype(_npx.float64) + 1e-300),
    }
    def _sigmoid_ufunc(x, out=None):
        """1 / (1 + exp(-x)), numerically stable for float32 (clips at ±88)."""
        if out is None: out = _npx.empty(x.shape, dtype=x.dtype)
        _npx.clip(x, -88.0, 88.0, out=out)
        _npx.negative(out, out=out)
        _npx.exp(out, out=out)
        out += 1.0
        _npx.reciprocal(out, out=out)
        return out

    # ufuncs that accept out= for direct-write into pre-allocated buffer
    _NP_UNARY_UFUNC = {
        math.exp: _npx.exp, math.log: _npx.log, math.log2: _npx.log2, math.log10: _npx.log10,
        math.sqrt: _npx.sqrt, math.tanh: _npx.tanh, math.sin: _npx.sin, math.cos: _npx.cos,
        math.tan: _npx.tan, math.fabs: _npx.abs, abs: _npx.abs, operator.neg: _npx.negative,
        math.floor: _npx.floor, math.ceil: _npx.ceil, _sign_scalar: _npx.sign,
        _relu_scalar: lambda x, out=None: _npx.maximum(x, 0, out=out),
        _sigmoid_scalar: _sigmoid_ufunc,
    }
    _NP_BINARY = {
        operator.add: _npx.add, operator.sub: _npx.subtract, operator.mul: _npx.multiply,
        operator.truediv: _npx.true_divide, operator.pow: _npx.power,
        max: _npx.maximum, min: _npx.minimum,
    }
    _NP_COMPARE = {
        operator.gt: _npx.greater, operator.ge: _npx.greater_equal,
        operator.lt: _npx.less, operator.le: _npx.less_equal,
        operator.eq: _npx.equal, operator.ne: _npx.not_equal,
    }


_NB_UNARY = {}   # maps scalar fn → numba kernel(x, out)
_NB_BINARY = {}  # maps operator fn → numba kernel(a, b, out)
_NB_BINARY_S = {} # maps operator fn → numba kernel(a, scalar, out)

if _NUMBA:
    _NB_UNARY[_relu_scalar]   = _nbk._relu
    _NB_UNARY[operator.neg]   = _nbk._neg
    _NB_BINARY[operator.mul]  = _nbk._mul
    _NB_BINARY[operator.add]  = _nbk._add
    _NB_BINARY[operator.sub]  = _nbk._sub
    _NB_BINARY[operator.truediv] = _nbk._truediv
    _NB_BINARY[max]           = _nbk._maximum
    _NB_BINARY[min]           = _nbk._minimum
    _NB_BINARY_S[operator.mul]    = _nbk._mul_scalar
    _NB_BINARY_S[operator.add]    = _nbk._add_scalar
    _NB_BINARY_S[operator.truediv] = _nbk._truediv_scalar

# Maps scalar fn → torch op accepting (input, out=) for float32/float64
_TORCH_UNARY = {}  # fn → callable(x_torch, out=out_torch)
if _TORCH:
    _TORCH_UNARY[math.exp]        = _torch.exp
    _TORCH_UNARY[math.log]        = _torch.log
    _TORCH_UNARY[math.sqrt]       = _torch.sqrt
    _TORCH_UNARY[math.tanh]       = _torch.tanh
    _TORCH_UNARY[math.sin]        = _torch.sin
    _TORCH_UNARY[math.cos]        = _torch.cos
    _TORCH_UNARY[math.tan]        = _torch.tan
    _TORCH_UNARY[_sigmoid_scalar] = _torch.sigmoid
    # relu: torch.clamp(min=0) used only for float64 where numba path is inactive
    _TORCH_UNARY[_relu_scalar]    = lambda xt, out: _torch.clamp(xt, min=0.0, out=out)
    _TORCH_UNARY[math.fabs]       = _torch.abs
    _TORCH_UNARY[abs]             = _torch.abs
    _TORCH_UNARY[operator.neg]    = _torch.neg
    _TORCH_UNARY[_sign_scalar]    = _torch.sign


# ── C extension dispatch map ──────────────────────────────────────────────────
# Maps scalar function → C extension function name (float32 kernels)
# Populated lazily on first use to avoid triggering the DLL scan at import time.
_C_UNARY_F32 = {}
_C_BINARY_F32 = {}
_C_SCALAR_F32 = {}
_C_DISPATCH_POPULATED = False


def _populate_c_dispatch():
    """Populate C dispatch tables — called once, on first actual C-extension use."""
    global _C_DISPATCH_POPULATED
    if _C_DISPATCH_POPULATED or not _C_EXT:
        return
    _C_DISPATCH_POPULATED = True
    try:
        _C_UNARY_F32[_relu_scalar]    = _C.relu_f32
        _C_UNARY_F32[math.exp]        = _C.exp_f32
        _C_UNARY_F32[math.log]        = _C.log_f32
        _C_UNARY_F32[_sigmoid_scalar] = _C.sigmoid_f32
        _C_UNARY_F32[math.tanh]       = _C.tanh_f32
        _C_UNARY_F32[math.sqrt]       = _C.sqrt_f32
        _C_UNARY_F32[math.fabs]       = _C.abs_f32
        _C_UNARY_F32[abs]             = _C.abs_f32
        _C_UNARY_F32[operator.neg]    = _C.neg_f32
        _C_UNARY_F32[_sign_scalar]    = _C.sign_f32
        _C_BINARY_F32[operator.add]       = _C.add_f32
        _C_BINARY_F32[operator.sub]       = _C.sub_f32
        _C_BINARY_F32[operator.mul]       = _C.mul_f32
        _C_BINARY_F32[operator.truediv]   = _C.div_f32
        _C_BINARY_F32[max]                = _C.maximum_f32
        _C_BINARY_F32[min]                = _C.minimum_f32
        _C_SCALAR_F32[operator.add]       = _C.add_scalar_f32
        _C_SCALAR_F32[operator.sub]       = _C.sub_scalar_f32
        _C_SCALAR_F32[operator.mul]       = _C.mul_scalar_f32
        _C_SCALAR_F32[operator.truediv]   = _C.div_scalar_f32
        _C_UNARY_F32[_erf_scalar]         = _C.erf_f32
    except Exception:
        pass


def _is_c_contiguous_f32(a: NDArray) -> bool:
    """True when the NDArray is contiguous float32 starting at offset 0."""
    if str(a.dtype) != 'float32':
        return False
    if a.offset != 0:
        return False
    if not a.shape:
        return True
    expected = _c_strides(a.shape)
    return a.strides == expected


def _c_alloc_f32(shape: tuple) -> _array.array:
    """Allocate a float32 array.array for C extension output (pooled when available)."""
    n = _prod(shape) if shape else 1
    if _AP:
        return _ap.alloc_f32(n)
    return _array.array('f', bytes(4 * n))


def _nd_from_array(buf: _array.array, shape: tuple, dtype: str) -> 'NDArray':
    """Wrap a flat array.array into an NDArray."""
    strides = _c_strides(shape)
    return NDArray(buf, shape, strides, _DType(dtype))


def _np_unary(a, fn):
    if a.dtype not in _FLOAT_DTYPES:
        return None
    np_dtype = a.dtype
    # C extension AVX2 fast path — primary path, zero third-party deps
    if _C_EXT and np_dtype == 'float32' and _is_c_contiguous_f32(a):
        _populate_c_dispatch()
        c_fn = _C_UNARY_F32.get(fn)
        if c_fn is not None:
            try:
                out = _c_alloc_f32(a.shape)
                src = a._buf if a._buf is not None else _npx.frombuffer(a._np_buf, dtype='float32')
                c_fn(src, out)
                return _nd_from_array(out, a.shape, 'float32')
            except Exception:
                pass
    # PyTorch SVML fast path: uses Intel MKL/SVML for exp/log/trig, ~15x faster than numpy
    if _TORCH and np_dtype in _FLOAT_DTYPES:
        torch_fn = _TORCH_UNARY.get(fn)
        if torch_fn is not None:
            try:
                x = _np_view(a)
                if x.flags['C_CONTIGUOUS']:
                    out = _bp.alloc(a.shape, str(np_dtype)) if _BP else _npx.empty(a.shape, dtype=np_dtype)
                    xt = _torch.from_numpy(x)
                    outt = _torch.from_numpy(out)
                    torch_fn(xt, out=outt)
                    return _nd_from_numpy(out, np_dtype)
            except Exception:
                pass
    # Numba parallel fast path (relu, neg, fused ops)
    if _NUMBA and np_dtype == 'float32':
        nb_fn = _NB_UNARY.get(fn)
        if nb_fn is not None:
            try:
                x = _np_view(a)
                if x.flags['C_CONTIGUOUS']:
                    out = _bp.alloc(a.shape, 'float32') if _BP else _npx.empty(a.shape, dtype='float32')
                    nb_fn(x, out)
                    return _nd_from_numpy(out, np_dtype)
            except Exception:
                pass
    # Numpy ufunc fast path
    uf = _NP_UNARY_UFUNC.get(fn)
    if uf is not None:
        try:
            x = _np_view(a)
            out = _bp.alloc(a.shape, str(np_dtype)) if _BP else _npx.empty(a.shape, dtype=np_dtype)
            with _npx.errstate(all='ignore'):
                uf(x, out=out)
            return _nd_from_numpy(out, np_dtype)
        except Exception:
            pass
    f = _NP_UNARY.get(fn)
    if f is None:
        return None
    try:
        with _npx.errstate(all='ignore'):
            return _np_wrap(f(_np_view(a)), np_dtype)
    except Exception:
        return None


def _np_binary(a, b, fn, out_dtype):
    # C extension AVX2 fast path for float32 tensor-tensor and tensor-scalar ops
    if _C_EXT and out_dtype is None:
        _populate_c_dispatch()
        a_nd = isinstance(a, NDArray); b_nd = isinstance(b, NDArray)
        if a_nd and b_nd and a.dtype == 'float32' and b.dtype == 'float32':
            if _is_c_contiguous_f32(a) and _is_c_contiguous_f32(b) and a.shape == b.shape:
                c_fn = _C_BINARY_F32.get(fn)
                if c_fn is not None:
                    try:
                        out = _c_alloc_f32(a.shape)
                        c_fn(a._buf, b._buf, out)
                        return _nd_from_array(out, a.shape, 'float32')
                    except Exception:
                        pass
        elif a_nd and isinstance(b, (int, float)) and a.dtype == 'float32':
            if _is_c_contiguous_f32(a):
                c_fn = _C_SCALAR_F32.get(fn)
                if c_fn is not None:
                    try:
                        out = _c_alloc_f32(a.shape)
                        c_fn(a._buf, float(b), out)
                        return _nd_from_array(out, a.shape, 'float32')
                    except Exception:
                        pass

    if out_dtype is None:
        f = _NP_BINARY.get(fn)
    elif out_dtype == 'bool':
        f = _NP_COMPARE.get(fn)
    else:
        f = None
    if f is None:
        return None
    a_nd = isinstance(a, NDArray); b_nd = isinstance(b, NDArray)
    try:
        if a_nd and b_nd:
            if a.dtype != b.dtype or a.dtype not in _FLOAT_DTYPES:
                return None
            dtype = a.dtype; x = _np_view(a); y = _np_view(b)
            # Numba parallel fast path: same shape, contiguous, float32, not comparison
            if (_NUMBA and dtype == 'float32' and out_dtype != 'bool'
                    and x.shape == y.shape
                    and x.flags['C_CONTIGUOUS'] and y.flags['C_CONTIGUOUS']):
                nb_fn = _NB_BINARY.get(fn)
                if nb_fn is not None:
                    out = _bp.alloc(x.shape, 'float32') if _BP else _npx.empty(x.shape, dtype='float32')
                    nb_fn(x, y, out)
                    return _nd_from_numpy(out, dtype)
        elif a_nd and isinstance(b, (int, float)):
            if a.dtype not in _FLOAT_DTYPES:
                return None
            dtype = a.dtype; x = _np_view(a); y = b
            # Numba scalar fast path
            if (_NUMBA and dtype == 'float32' and out_dtype != 'bool'
                    and x.flags['C_CONTIGUOUS']):
                nb_fn = _NB_BINARY_S.get(fn)
                if nb_fn is not None:
                    out = _bp.alloc(x.shape, 'float32') if _BP else _npx.empty(x.shape, dtype='float32')
                    nb_fn(x, float(b), out)
                    return _nd_from_numpy(out, dtype)
        elif b_nd and isinstance(a, (int, float)):
            if b.dtype not in _FLOAT_DTYPES:
                return None
            dtype = b.dtype; x = a; y = _np_view(b)
        else:
            return None
        final_dtype = out_dtype or dtype
        # Numpy ufunc fast path
        if out_dtype != 'bool' and isinstance(f, _npx.ufunc):
            try:
                out_shape = _npx.broadcast_shapes(
                    x.shape if hasattr(x, 'shape') else (),
                    y.shape if hasattr(y, 'shape') else ())
                out = _bp.alloc(out_shape, str(final_dtype)) if _BP else _npx.empty(out_shape, dtype=final_dtype)
                with _npx.errstate(all='ignore'):
                    f(x, y, out=out)
                return _nd_from_numpy(out, final_dtype)
            except Exception:
                pass
        with _npx.errstate(all='ignore'):
            r = f(x, y)
        return _np_wrap(r, final_dtype)
    except Exception:
        return None


def _np_softmax(a: 'NDArray', axis: int) -> 'NDArray':
    """Softmax — C extension AVX2 (over last axis / 2D) or numpy fallback."""
    if a.dtype not in _FLOAT_DTYPES:
        return None
    # C extension: handles 2D row-wise softmax and 1D arrays
    if _C_EXT and str(a.dtype) == 'float32' and _is_c_contiguous_f32(a):
        try:
            nd = a.ndim
            if nd == 1:
                out = _c_alloc_f32(a.shape)
                _C.softmax_rows_f32(a._buf, out, 1, a.shape[0])
                return _nd_from_array(out, a.shape, 'float32')
            elif nd == 2 and axis in (-1, 1):
                rows, cols = a.shape
                out = _c_alloc_f32(a.shape)
                _C.softmax_rows_f32(a._buf, out, rows, cols)
                return _nd_from_array(out, a.shape, 'float32')
        except Exception:
            pass
    if _npx is None:
        return None
    try:
        x = _np_view(a)
        if _TORCH and x.flags['C_CONTIGUOUS']:
            out = _bp.alloc(a.shape, str(a.dtype)) if _BP else _npx.empty(a.shape, dtype=a.dtype)
            _torch.softmax(_torch.from_numpy(x), dim=axis, out=_torch.from_numpy(out))
            return _nd_from_numpy(out, a.dtype)
        m = x.max(axis=axis, keepdims=True)
        out = _bp.alloc(a.shape, str(a.dtype)) if _BP else _npx.empty(a.shape, dtype=a.dtype)
        _npx.subtract(x, m, out=out)
        _npx.exp(out, out=out)
        s = out.sum(axis=axis, keepdims=True)
        _npx.true_divide(out, s, out=out)
        return _nd_from_numpy(out, a.dtype)
    except Exception:
        return None


def _np_log_softmax(a: 'NDArray', axis: int) -> 'NDArray':
    """log_softmax — C extension AVX2 or numpy fallback."""
    if a.dtype not in _FLOAT_DTYPES:
        return None
    if _C_EXT and str(a.dtype) == 'float32' and _is_c_contiguous_f32(a):
        try:
            nd = a.ndim
            if nd == 1:
                out = _c_alloc_f32(a.shape)
                _C.log_softmax_rows_f32(a._buf, out, 1, a.shape[0])
                return _nd_from_array(out, a.shape, 'float32')
            elif nd == 2 and axis in (-1, 1):
                rows, cols = a.shape
                out = _c_alloc_f32(a.shape)
                _C.log_softmax_rows_f32(a._buf, out, rows, cols)
                return _nd_from_array(out, a.shape, 'float32')
        except Exception:
            pass
    if _npx is None:
        return None
    try:
        x = _np_view(a)
        if _TORCH and x.flags['C_CONTIGUOUS']:
            out = _bp.alloc(a.shape, str(a.dtype)) if _BP else _npx.empty(a.shape, dtype=a.dtype)
            _torch.log_softmax(_torch.from_numpy(x), dim=axis, out=_torch.from_numpy(out))
            return _nd_from_numpy(out, a.dtype)
        m = x.max(axis=axis, keepdims=True)
        out = _bp.alloc(a.shape, str(a.dtype)) if _BP else _npx.empty(a.shape, dtype=a.dtype)
        _npx.subtract(x, m, out=out)
        tmp = _bp.alloc(a.shape, str(a.dtype)) if _BP else _npx.empty(a.shape, dtype=a.dtype)
        _npx.exp(out, out=tmp)
        log_sum = _npx.log(tmp.sum(axis=axis, keepdims=True))
        _npx.subtract(out, log_sum, out=out)
        return _nd_from_numpy(out, a.dtype)
    except Exception:
        return None


# ── Layer Norm + RMS Norm (C extension fast path) ────────────────────────────

def layer_norm(x: 'NDArray', normalized_shape, weight=None, bias=None,
               eps: float = 1e-5) -> 'NDArray':
    """Layer normalization — AVX2 vectorized, one pass per row."""
    if _C_EXT and str(x.dtype) == 'float32' and _is_c_contiguous_f32(x):
        try:
            # normalized_shape gives the last N dims; flatten preceding dims into rows
            if isinstance(normalized_shape, int):
                norm_size = normalized_shape
            else:
                norm_size = 1
                for s in normalized_shape: norm_size *= s
            total = _prod(x.shape)
            rows = total // norm_size
            out = _c_alloc_f32(x.shape)
            gamma_buf = weight._buf if (weight is not None and _is_c_contiguous_f32(weight)) else None
            beta_buf  = bias._buf   if (bias   is not None and _is_c_contiguous_f32(bias))   else None
            _C.layer_norm_f32(x._buf, out,
                              gamma_buf if gamma_buf is not None else None,
                              beta_buf  if beta_buf  is not None else None,
                              rows, norm_size, eps)
            return _nd_from_array(out, x.shape, 'float32')
        except Exception:
            pass
    # Python fallback
    flat = x.to_flat_list(); n_total = len(flat)
    ns = normalized_shape if isinstance(normalized_shape, int) else _prod(normalized_shape)
    rows = n_total // ns
    out_flat = []
    wf = weight.to_flat_list() if weight is not None else None
    bf = bias.to_flat_list()   if bias   is not None else None
    for r in range(rows):
        row = flat[r*ns:(r+1)*ns]
        mean = sum(row) / ns
        var  = sum((v - mean)**2 for v in row) / ns
        inv  = 1.0 / math.sqrt(var + eps)
        for j, v in enumerate(row):
            xn = (v - mean) * inv
            if wf: xn *= wf[j]
            if bf: xn += bf[j]
            out_flat.append(xn)
    buf = _array.array('f', out_flat)
    return NDArray(buf, x.shape, _c_strides(x.shape), 'float32')


def rms_norm(x: 'NDArray', weight=None, eps: float = 1e-6) -> 'NDArray':
    """RMS normalization — used in LLaMA-style transformers."""
    if _C_EXT and str(x.dtype) == 'float32' and _is_c_contiguous_f32(x):
        try:
            norm_size = x.shape[-1] if x.shape else _prod(x.shape)
            total = _prod(x.shape)
            rows = total // norm_size
            out = _c_alloc_f32(x.shape)
            w_buf = weight._buf if (weight is not None and _is_c_contiguous_f32(weight)) else None
            _C.rms_norm_f32(x._buf, out,
                            w_buf if w_buf is not None else None,
                            rows, norm_size, eps)
            return _nd_from_array(out, x.shape, 'float32')
        except Exception:
            pass
    flat = x.to_flat_list(); n = len(flat)
    ns = x.shape[-1] if x.shape else n
    rows = n // ns
    wf = weight.to_flat_list() if weight is not None else None
    out_flat = []
    for r in range(rows):
        row = flat[r*ns:(r+1)*ns]
        ss = sum(v*v for v in row) / ns
        inv = 1.0 / math.sqrt(ss + eps)
        for j, v in enumerate(row):
            xn = v * inv
            if wf: xn *= wf[j]
            out_flat.append(xn)
    buf = _array.array('f', out_flat)
    return NDArray(buf, x.shape, _c_strides(x.shape), 'float32')


def _var_c(a: 'NDArray', unbiased: bool = False):
    """Fast variance using C extension two-pass kernel."""
    if _C_EXT and str(a.dtype) == 'float32' and _is_c_contiguous_f32(a):
        try:
            return _C.var_f32(a._buf, int(unbiased))
        except Exception:
            pass
    return None


def _transpose_c(a: 'NDArray') -> 'NDArray':
    """Blocked cache-oblivious transpose via C extension."""
    if (_C_EXT and a.ndim == 2 and str(a.dtype) == 'float32'
            and _is_c_contiguous_f32(a)):
        try:
            rows, cols = a.shape
            out = _c_alloc_f32((cols, rows))
            _C.transpose_f32(a._buf, out, rows, cols)
            return _nd_from_array(out, (cols, rows), 'float32')
        except Exception:
            pass
    return None


def _sum_axis_c(a: 'NDArray', axis: int) -> 'NDArray':
    """AVX2-vectorized axial sum for 2D contiguous float32 arrays."""
    if (_C_EXT and a.ndim == 2 and str(a.dtype) == 'float32'
            and _is_c_contiguous_f32(a)):
        try:
            rows, cols = a.shape
            if axis == 1:
                out = _c_alloc_f32((rows,))
            else:
                out = _c_alloc_f32((cols,))
            _C.sum_axis_f32(a._buf, out, rows, cols, axis)
            return _nd_from_array(out, (rows,) if axis == 1 else (cols,), 'float32')
        except Exception:
            pass
    return None


def where_c(cond: 'NDArray', a: 'NDArray', b: 'NDArray') -> 'NDArray':
    """AVX2 conditional select: out[i] = a[i] if cond[i]>0 else b[i]."""
    if (_C_EXT and str(a.dtype) == 'float32'
            and _is_c_contiguous_f32(cond) and _is_c_contiguous_f32(a)
            and _is_c_contiguous_f32(b)):
        try:
            out = _c_alloc_f32(a.shape)
            _C.where_f32(cond._buf, a._buf, b._buf, out)
            return _nd_from_array(out, a.shape, 'float32')
        except Exception:
            pass
    flat_c = cond.to_flat_list()
    flat_a = a.to_flat_list()
    flat_b = b.to_flat_list()
    result = [av if cv > 0.0 else bv for cv, av, bv in zip(flat_c, flat_a, flat_b)]
    buf = _array.array('f', result)
    return NDArray(buf, a.shape, _c_strides(a.shape), 'float32')


# ── Elementwise operations ────────────────────────────────────────────────────

_KERN_UNARY_MAP = {
    math.exp:   'vp_exp',
    math.log:   'vp_log',
    math.sqrt:  'vp_sqrt',
    math.fabs:  'vp_abs',
    abs:        'vp_abs',
    operator.neg: 'vp_neg',
}
# Kept for backward compat
_MOJO_UNARY_MAP = _KERN_UNARY_MAP


def _elementwise_unary(a, fn, out_dtype=None):
    if type(a).__name__ == 'GArray':
        # Delegate to GArray-aware dispatch in _vp_helpers
        try:
            from vectorphile._vp_helpers import _nd_elementwise as _vph_ew
            return _vph_ew(a, fn, out_dtype)
        except Exception:
            pass
        return a  # fallback: identity (shouldn't reach here)
    if out_dtype is None and _npx is not None:
        r = _np_unary(a, fn)
        if r is not None:
            return r
    dtype = out_dtype or a.dtype
    # ── C/Mojo kernel fast path for float32 ──────────────────────────────────
    if _KERN and dtype == 'float32' and out_dtype is None and a.dtype == 'float32':
        kern_name = _KERN_UNARY_MAP.get(fn)
        if kern_name:
            kern = _kern_fn(kern_name)
            if kern is not None:
                cont = a.contiguous()
                cont_buf = cont._get_buf()
                dst_buf = _alloc_f32(len(cont_buf))
                kern(_kern_ptr(dst_buf), _kern_ptr(cont_buf), len(cont_buf))
                return NDArray(dst_buf, a.shape, _c_strides(a.shape), 'float32')
    # ── Pure-Python fallback ──────────────────────────────────────────────────
    tc = _TYPECODES.get(dtype, 'd')
    flat = [fn(v) for v in a.to_flat_list()]
    if 'float' in dtype: flat = [float(v) for v in flat]
    elif 'int' in dtype: flat = [int(v) for v in flat]
    buf = _array.array(tc, flat)
    return NDArray(buf, a.shape, _c_strides(a.shape), dtype)


_KERN_BINARY_MAP = {
    operator.add:      'vp_add',
    operator.sub:      'vp_sub',
    operator.mul:      'vp_mul',
    operator.truediv:  'vp_div',
}


def _elementwise_binary(a, b, fn, out_dtype=None) -> NDArray:
    # Normalize scalars first so C kernel paths can check shapes
    # Use the operand's dtype for scalar storage (not out_dtype) to avoid overflow
    if isinstance(a, (int, float)):
        if isinstance(b, NDArray):
            a = _broadcast_scalar(a, b.shape, b.dtype)
        else:
            a = array([a])
    elif not isinstance(a, NDArray):
        a = array(a)
    if isinstance(b, (int, float)):
        b = _broadcast_scalar(b, a.shape, a.dtype)
    elif not isinstance(b, NDArray):
        b = array(b)

    out_shape = _broadcast_shapes(a.shape, b.shape)
    dtype = out_dtype or a.dtype

    # ── C/Mojo kernel fast path: contiguous float32 — BEFORE numpy ───────────
    if (_KERN and dtype == 'float32' and out_dtype is None
            and a.dtype == 'float32' and b.dtype == 'float32'):
        kern_name = _KERN_BINARY_MAP.get(fn)
        if kern_name and a.shape == out_shape and b.shape == out_shape:
            kern = _kern_fn(kern_name)
            if kern is not None:
                ca = a.contiguous(); cb = b.contiguous()
                ca_buf = ca._get_buf(); cb_buf = cb._get_buf()
                n = ca_buf.__len__()
                dst_buf = _alloc_f32(n)
                kern(_kern_ptr(dst_buf), _kern_ptr(ca_buf), _kern_ptr(cb_buf), n)
                return NDArray(dst_buf, out_shape, _c_strides(out_shape), 'float32')
        # Scalar broadcasting: one side is a single value
        if kern_name and (a.size == 1 or b.size == 1):
            kern_scalar = _kern_fn('vp_add_scalar' if kern_name == 'vp_add'
                                   else 'vp_mul_scalar' if kern_name == 'vp_mul'
                                   else None)
            if kern_scalar and kern_name in ('vp_add', 'vp_mul'):
                if a.size == 1 and b.size != 1:
                    scalar_val = float(a._get_buf()[a.offset])
                    cb = b.contiguous(); cb_buf = cb._get_buf(); n = cb_buf.__len__()
                    dst_buf = _alloc_f32(n)
                    kern_scalar(_kern_ptr(dst_buf), _kern_ptr(cb_buf), scalar_val, n)
                    return NDArray(dst_buf, out_shape, _c_strides(out_shape), 'float32')
                elif b.size == 1 and a.size != 1:
                    scalar_val = float(b._get_buf()[b.offset])
                    ca = a.contiguous(); ca_buf = ca._get_buf(); n = ca_buf.__len__()
                    dst_buf = _alloc_f32(n)
                    kern_scalar(_kern_ptr(dst_buf), _kern_ptr(ca_buf), scalar_val, n)
                    return NDArray(dst_buf, out_shape, _c_strides(out_shape), 'float32')

    # Broadcast fast path: use vp_broadcast_binary for float32 ops with any shapes
    if (_KERN and dtype == 'float32' and out_dtype is None
            and a.dtype == 'float32' and b.dtype == 'float32'):
        kern_name = _KERN_BINARY_MAP.get(fn)
        if kern_name:
            kern_bc = _kern_fn('vp_broadcast_binary')
            op_id = {'vp_add': 0, 'vp_sub': 1, 'vp_mul': 2, 'vp_div': 3}.get(kern_name, -1)
            if kern_bc is not None and op_id >= 0:
                ndim = len(out_shape)
                # Compute broadcast strides (0 where dim was 1 or missing)
                def _bcast_strides(shape, out_shape):
                    # Pad shape on left with 1s
                    pad = len(out_shape) - len(shape)
                    padded = (1,) * pad + tuple(shape)
                    strides = _c_strides(padded)
                    # Zero out axes where the original dim is 1
                    return tuple(0 if padded[i] == 1 else strides[i] for i in range(len(out_shape)))
                a_st = _bcast_strides(a.shape, out_shape)
                b_st = _bcast_strides(b.shape, out_shape)
                ca = a.contiguous(); cb = b.contiguous()
                ca_buf = ca._get_buf(); cb_buf = cb._get_buf()
                n_out = _prod(out_shape)
                dst_buf = _alloc_f32(n_out)
                shape_buf = _array.array('i', list(out_shape))
                ast_buf = _array.array('i', list(a_st))
                bst_buf = _array.array('i', list(b_st))
                if _CFFI:
                    kern_bc(_kern_ptr(dst_buf), _kern_ptr(ca_buf), _kern_ptr(cb_buf),
                            _cffi_kern.iptr(shape_buf), _cffi_kern.iptr(ast_buf),
                            _cffi_kern.iptr(bst_buf), ndim, op_id)
                else:
                    kern_bc(_kern_ptr(dst_buf), _kern_ptr(ca_buf), _kern_ptr(cb_buf),
                            _kern_ptr(shape_buf), _kern_ptr(ast_buf),
                            _kern_ptr(bst_buf), ndim, op_id)
                return NDArray(dst_buf, out_shape, _c_strides(out_shape), 'float32')

    # ── Numpy fallback (handles all dtypes, broadcasting, comparisons) ────────
    if _npx is not None:
        r = _np_binary(a, b, fn, out_dtype)
        if r is not None:
            return r

    tc = _TYPECODES.get(dtype, 'd')
    flat = []
    for idx in itertools.product(*[range(s) for s in out_shape]):
        ai = _broadcast_idx(idx, a.shape)
        bi = _broadcast_idx(idx, b.shape)
        va = a._get_buf()[a._flat_idx(ai)]
        vb = b._get_buf()[b._flat_idx(bi)]
        v = fn(va, vb)
        if 'float' in dtype: v = float(v)
        elif 'int' in dtype: v = int(v)
        elif dtype == 'bool': v = int(bool(v))
        flat.append(v)

    buf = _array.array(tc, flat)
    return NDArray(buf, out_shape, _c_strides(out_shape), dtype)


def _binary_op(a, b, fn, out_dtype=None) -> NDArray:
    if isinstance(a, NDArray) and isinstance(b, NDArray):
        return _elementwise_binary(a, b, fn, out_dtype)
    elif isinstance(a, NDArray):
        return _elementwise_binary(a, b, fn, out_dtype)
    elif isinstance(b, NDArray):
        return _elementwise_binary(a, b, fn, out_dtype)
    return fn(a, b)


# ── Reduction helpers ─────────────────────────────────────────────────────────

def _np_axis_reduce(a: NDArray, op: str, axis, keepdims, float_only=True):
    """Axis reduction — C extension AVX2 or NumPy fallback."""
    # C extension fast path: global sum/max/min on contiguous float32
    if _C_EXT and axis is None and str(a.dtype) == 'float32' and _is_c_contiguous_f32(a):
        try:
            if op == 'sum':
                v = _C.sum_f32(a._buf)
            elif op == 'max' or op == 'amax':
                v = _C.max_f32(a._buf)
            elif op == 'min' or op == 'amin':
                v = _C.min_f32(a._buf)
            else:
                v = None
            if v is not None:
                scalar_buf = _array.array('f', [float(v)])
                shape = (1,) if keepdims else ()
                return NDArray(scalar_buf, shape, _c_strides(shape), _DType('float32'))
        except Exception:
            pass

    # C extension axis-reduce for 2D float32 sum/max/min
    if (_C_EXT and axis is not None and not isinstance(axis, (tuple, list)) and a.ndim == 2
            and str(a.dtype) == 'float32' and _is_c_contiguous_f32(a)):
        ax = int(axis) % 2
        rows, cols = a.shape
        if op == 'sum':
            try:
                out_shape = (rows,) if ax == 1 else (cols,)
                out_buf = _c_alloc_f32(out_shape)
                _C.sum_axis_f32(a._buf, out_buf, rows, cols, ax)
                result = NDArray(out_buf, out_shape, _c_strides(out_shape), _DType('float32'))
                if keepdims:
                    result = result.reshape((rows, 1) if ax == 1 else (1, cols))
                return result
            except Exception:
                pass

    if _npx is None:
        return None
    try:
        c = a.contiguous()
        tc = _TYPECODES.get(c.dtype, 'd')
        if float_only and tc not in ('f', 'd'):
            return None
        np_dtype = str(a.dtype)
        view = c._np_buf if c._np_buf is not None else _npx.frombuffer(
            c._get_buf(), dtype=_npx.dtype(c._get_buf().typecode), count=_prod(a.shape)).reshape(a.shape)
        if axis is None:
            return None  # handled by caller (NDArray.sum/mean) to avoid pool entanglement
        ax = int(axis) if isinstance(axis, int) else tuple(int(x) for x in axis)
        r = getattr(view, op)(axis=ax, keepdims=keepdims)
        return _nd_from_numpy(_npx.ascontiguousarray(r, dtype=tc), a.dtype)
    except Exception:
        return None


def _reduce(a: NDArray, fn, axis, keepdims, identity):
    if axis is None:
        result = fn(a.to_flat_list())
        if keepdims: return _broadcast_scalar(result, (1,) * a.ndim, a.dtype)
        return _broadcast_scalar(result, (), a.dtype)
    # normalize to sorted tuple of non-negative axes
    if isinstance(axis, int):
        axes = (axis % a.ndim,)
    else:
        axes = tuple(sorted(ax % a.ndim for ax in axis))
    # reduce one axis at a time
    out = a
    for ax_offset, ax in enumerate(axes):
        # when keepdims=False, removed dims shift later axes left; when True, shape stays
        real_ax = ax if keepdims else ax - ax_offset
        out_shape = list(out.shape); out_shape[real_ax] = 1
        result = zeros(tuple(out_shape), dtype=out.dtype)
        result.fill(identity)
        for idx in out._iter_indices():
            out_idx = list(idx); out_idx[real_ax] = 0
            cur = result._buf[result._flat_idx(tuple(out_idx))]
            v = out._buf[out._flat_idx(idx)]
            result._buf[result._flat_idx(tuple(out_idx))] = cur + v
        if not keepdims:
            final_shape = tuple(s for i, s in enumerate(out.shape) if i != real_ax)
            out = result.reshape(final_shape) if final_shape else result
        else:
            out = result
    return out


def _reduce_fn(a: NDArray, fn, axis, keepdims):
    if axis is None:
        result = fn(a.to_flat_list())
        if keepdims: return _broadcast_scalar(result, (1,) * a.ndim, a.dtype)
        return result
    if isinstance(axis, (tuple, list)):
        result = a
        for ax in sorted(set(int(x % a.ndim) for x in axis), reverse=True):
            result = _reduce_fn(result, fn, ax, keepdims=True)
        if not keepdims:
            final_shape = tuple(s for i, s in enumerate(result.shape) if s != 1 or i not in [int(x % a.ndim) for x in axis])
            result = result.reshape(final_shape)
        return result
    axis = axis % a.ndim
    out_shape = list(a.shape); out_shape[axis] = 1
    result_data = {}
    for idx in a._iter_indices():
        out_idx = tuple(0 if i == axis else v for i, v in enumerate(idx))
        v = a._buf[a._flat_idx(idx)]
        if out_idx not in result_data: result_data[out_idx] = v
        else: result_data[out_idx] = fn([result_data[out_idx], v])
    flat = [result_data[idx] for idx in sorted(result_data.keys())]
    out_arr = NDArray(_array.array(_TYPECODES.get(a.dtype, 'd'), flat),
                      tuple(out_shape), _c_strides(tuple(out_shape)), a.dtype)
    if not keepdims:
        final_shape = tuple(s for i, s in enumerate(a.shape) if i != axis)
        return out_arr.reshape(final_shape)
    return out_arr


def _argreduce(a: NDArray, axis: int, fn):
    axis = axis % a.ndim
    out_shape = tuple(s for i, s in enumerate(a.shape) if i != axis)
    result_vals = {}; result_args = {}
    a_buf = a._get_buf()
    for idx in a._iter_indices():
        out_idx = tuple(v for i, v in enumerate(idx) if i != axis)
        v = a_buf[a._flat_idx(idx)]
        if out_idx not in result_vals:
            result_vals[out_idx] = v; result_args[out_idx] = idx[axis]
        elif fn([v, result_vals[out_idx]]) == v:
            result_vals[out_idx] = v; result_args[out_idx] = idx[axis]
    flat = [result_args[idx] for idx in sorted(result_args.keys())]
    buf = _array.array('l', flat)
    return NDArray(buf, out_shape, _c_strides(out_shape), 'int64')


# ── Matmul ────────────────────────────────────────────────────────────────────

def matmul(a: NDArray, b: NDArray) -> NDArray:
    """Matrix multiply -- C/Mojo GEMM if available, else pure Python."""
    # Handle 1D right operand: (m,k) @ (k,) -> (m,)
    if b.ndim == 1:
        b_col = b.reshape(b.shape[0], 1)
        result = matmul(a, b_col)
        return result.reshape(result.shape[:-1])
    # Handle 1D left operand: (k,) @ (k,n) -> (n,)
    if a.ndim == 1:
        a_row = a.reshape(1, a.shape[0])
        result = matmul(a_row, b)
        return result.reshape(result.shape[1:])
    assert a.shape[-1] == b.shape[-2], f"Matmul shape mismatch: {a.shape} @ {b.shape}"
    if a.ndim == 2 and b.ndim == 2:
        m, k, n = a.shape[0], a.shape[1], b.shape[1]
        # ── vectorphile CUDA GEMM (our own kernel, no torch) ─────────────────
        if a.dtype == 'float32' and b.dtype == 'float32' and m * k * n > 65536:
            try:
                from vectorphile._core import _vp_cuda as _vpcuda
                if _vpcuda.device_count() > 0:
                    flat_a = _array.array('f', [float(v) for v in a.to_flat_list()])
                    flat_b = _array.array('f', [float(v) for v in b.to_flat_list()])
                    nbytes_a = m * k * 4; nbytes_b = k * n * 4
                    ga = _vpcuda.alloc(nbytes_a); gb = _vpcuda.alloc(nbytes_b)
                    _vpcuda.h2d(flat_a, ga); _vpcuda.h2d(flat_b, gb)
                    gc = _vpcuda.gemm_f32(ga, gb, m, k, n)
                    _vpcuda.synchronize()
                    out_buf = _array.array('f', bytearray(m * n * 4))
                    _vpcuda.d2h(gc, out_buf)
                    return NDArray(out_buf, (m, n), _c_strides((m, n)), _DType('float32'))
            except Exception:
                pass
        # ── C extension AVX2/FMA GEMM — primary path ─────────────────────────
        if _C_EXT and a.dtype == 'float32' and b.dtype == 'float32':
            try:
                ca = a.contiguous(); cb = b.contiguous()
                # Ensure array-backed buffers (numpy-backed NDArrays have _buf=None)
                def _ensure_buf(nd):
                    if nd._buf is not None:
                        return nd._buf
                    # numpy-backed: copy to array.array
                    flat = nd._np_buf.ravel().astype('float32', copy=False)
                    buf = _array.array('f', flat.tobytes())
                    return buf
                a_buf = _ensure_buf(ca); b_buf = _ensure_buf(cb)
                out_buf = _c_alloc_f32((m, n))
                _C.matmul_f32(a_buf, b_buf, out_buf, m, k, n)
                return NDArray(out_buf, (m, n), _c_strides((m, n)), _DType('float32'))
            except Exception:
                pass
        if _C_EXT and a.dtype == 'float64' and b.dtype == 'float64':
            try:
                ca = a.contiguous(); cb = b.contiguous()
                def _ensure_buf_f64(nd):
                    if nd._np_buf is not None:
                        flat = nd._np_buf.ravel().astype('float64', copy=False)
                        return _array.array('d', flat.tobytes())
                    if nd._buf is not None:
                        return nd._buf
                    raise ValueError("no buffer")
                a_buf = _ensure_buf_f64(ca); b_buf = _ensure_buf_f64(cb)
                out_buf = _array.array('d', bytearray(8 * m * n))
                _C.matmul_f64(a_buf, b_buf, out_buf, m, k, n)
                return NDArray(out_buf, (m, n), _c_strides((m, n)), _DType('float64'))
            except Exception:
                pass
        # ── C/Mojo GEMM fast path ─────────────────────────────────────────────
        if _KERN and a.dtype == 'float32' and b.dtype == 'float32':
            kern = _kern_fn('vp_sgemm')
            if kern is not None:
                ca = a.contiguous(); cb = b.contiguous()
                ca_buf = ca._get_buf(); cb_buf = cb._get_buf()
                res_buf = _alloc_f32(m * n)
                kern(_kern_ptr(res_buf), _kern_ptr(ca_buf), _kern_ptr(cb_buf), m, k, n)
                return NDArray(res_buf, (m, n), _c_strides((m, n)), 'float32')
        # ── BF16 matmul: upcast to float32, compute, store as bfloat16 ──────────
        if (a.dtype == 'bfloat16' or b.dtype == 'bfloat16'):
            a32 = a.astype('float32') if a.dtype == 'bfloat16' else a
            b32 = b.astype('float32') if b.dtype == 'bfloat16' else b
            result32 = matmul(a32, b32)
            return result32.astype('bfloat16')
        # ── INT8 quantized GEMM: accumulate int32, return float32 ────────────
        if a.dtype == 'int8' and b.dtype == 'int8':
            ac = a.contiguous(); bc = b.contiguous()
            ab = ac._get_buf(); bb = bc._get_buf()
            res_buf = _array.array('i', [0] * (m * n))  # int32 accumulator
            for i in range(m):
                a_row = i * k
                c_row = i * n
                for p in range(k):
                    av = ab[a_row + p]
                    b_row = p * n
                    for j in range(n):
                        res_buf[c_row + j] += av * bb[b_row + j]
            # Dequantize to float32
            out_buf = _array.array('f', [float(v) for v in res_buf])
            return NDArray(out_buf, (m, n), _c_strides((m, n)), _DType('float32'))
        # ── Pure Python (blocked for cache efficiency) ────────────────────────
        ac = a.contiguous(); bc = b.contiguous()
        ab = ac._get_buf(); bb = bc._get_buf()
        res_buf = _array.array('f' if a.dtype == 'float32' else 'd', [0.0] * (m * n))
        # Tile sizes tuned for L1 cache (32KB ≈ 2048 floats per tile)
        BM, BK, BN = 32, 64, 32
        bm0 = 0
        while bm0 < m:
            bm1 = min(bm0 + BM, m)
            bk0 = 0
            while bk0 < k:
                bk1 = min(bk0 + BK, k)
                bn0 = 0
                while bn0 < n:
                    bn1 = min(bn0 + BN, n)
                    for i in range(bm0, bm1):
                        a_row = i * k
                        c_row = i * n
                        for p in range(bk0, bk1):
                            av = ab[a_row + p]
                            b_row = p * n
                            for j in range(bn0, bn1):
                                res_buf[c_row + j] += av * bb[b_row + j]
                    bn0 += BN
                bk0 += BK
            bm0 += BM
        return NDArray(res_buf, (m, n), _c_strides((m, n)), a.dtype)
    # batch matmul
    batch_a = a.shape[:-2]; batch_b = b.shape[:-2]
    out_batch = _broadcast_shapes(batch_a, batch_b)
    m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
    batch = _prod(out_batch) if out_batch else 1
    if (_KERN and a.dtype == 'float32' and b.dtype == 'float32'
            and len(out_batch) == 1 and batch_a == batch_b):
        kern = _kern_fn('vp_sgemm_batched')
        if kern is not None:
            ca = a.contiguous(); cb = b.contiguous()
            ca_buf = ca._get_buf(); cb_buf = cb._get_buf()
            res_buf = _alloc_f32(batch * m * n)
            kern(_kern_ptr(res_buf), _kern_ptr(ca_buf), _kern_ptr(cb_buf), batch, m, k, n)
            return NDArray(res_buf, out_batch + (m, n), _c_strides(out_batch + (m, n)), 'float32')
    result = zeros(out_batch + (m, n), dtype=a.dtype)
    for bidx in itertools.product(*[range(s) for s in out_batch]):
        ai = _broadcast_idx(bidx, batch_a)
        bi = _broadcast_idx(bidx, batch_b)
        a_slice = a[ai]
        b_slice = b[bi]
        r_slice = matmul(a_slice, b_slice)
        for ridx in itertools.product(range(m), range(n)):
            result._buf[result._flat_idx(bidx + ridx)] = r_slice._buf[r_slice._flat_idx(ridx)]
    return result


# ── Constructors ──────────────────────────────────────────────────────────────

def array(data, dtype=None, copy=True) -> NDArray:
    # Handle Tensor by extracting its _data NDArray
    if hasattr(data, '_data') and isinstance(getattr(data, '_data', None), NDArray):
        return array(data._data, dtype=dtype, copy=copy)
    if isinstance(data, NDArray):
        # preserve original dtype when none specified
        effective = _resolve_dtype(dtype) if dtype is not None else data.dtype
        return data.astype(effective) if effective != data.dtype else (data.copy() if copy else data)
    _dtype_explicit = dtype  # remember if user passed None vs an explicit type
    dtype = _resolve_dtype(dtype)
    # NumPy array — convert via tolist() to get plain Python scalars
    _np_ndarray = type(None)
    try:
        import numpy as _np
        _np_ndarray = _np.ndarray
    except ImportError:
        pass
    if isinstance(data, _np_ndarray):
        # Infer dtype from numpy array when not specified (preserve float64, bool, int64, etc.)
        if _dtype_explicit is None:
            np_dt = str(data.dtype)
            if np_dt in _TYPECODES:
                dtype = np_dt
            elif np_dt == 'bool':
                dtype = 'bool'
        shape = tuple(data.shape)
        flat = data.ravel().tolist()
        tc = _TYPECODES.get(dtype, 'd')
        if 'float' in dtype: flat = [float(x) for x in flat]
        elif 'int' in dtype: flat = [int(x) for x in flat]
        elif dtype == 'bool': flat = [int(bool(x)) for x in flat]
        buf = _array.array(tc, flat)
        return NDArray(buf, shape, _c_strides(shape), dtype)
    # Python scalar
    if isinstance(data, (int, float, bool)):
        tc = _TYPECODES.get(dtype, 'd')
        buf = _array.array(tc, [data])
        return NDArray(buf, (), (), dtype)
    # Python nested list / tuple — flatten
    def _shape(d):
        if isinstance(d, NDArray): return d.shape
        if hasattr(d, '_data') and isinstance(getattr(d, '_data', None), NDArray): return d._data.shape
        if not isinstance(d, (list, tuple)): return ()
        return (len(d),) + _shape(d[0]) if d else (0,)
    def _flatten(d):
        if isinstance(d, NDArray): return d.to_flat_list()
        if hasattr(d, '_data') and isinstance(getattr(d, '_data', None), NDArray): return d._data.to_flat_list()
        if not isinstance(d, (list, tuple)): return [d]
        result = []
        for x in d: result.extend(_flatten(x))
        return result
    shape = _shape(data)
    flat = _flatten(data)
    tc = _TYPECODES.get(dtype, 'd')
    if 'float' in dtype: flat = [float(x) for x in flat]
    elif 'int' in dtype: flat = [int(x) for x in flat]
    elif dtype == 'bool': flat = [int(bool(x)) for x in flat]
    buf = _array.array(tc, flat)
    return NDArray(buf, shape, _c_strides(shape), dtype)


def zeros(shape, dtype='float32') -> NDArray:
    if isinstance(shape, int): shape = (shape,)
    dtype = _resolve_dtype(dtype)
    tc = _TYPECODES.get(dtype, 'd')
    n = _prod(shape)
    itemsize = _ITEMSIZES.get(tc, 8)
    buf = _array.array(tc, bytearray(itemsize * n))  # bytearray = all zeros, O(n) in C
    return NDArray(buf, tuple(shape), _c_strides(shape), dtype)


def ones(shape, dtype='float32') -> NDArray:
    if isinstance(shape, int): shape = (shape,)
    dtype = _resolve_dtype(dtype)
    tc = _TYPECODES.get(dtype, 'd')
    n = _prod(shape)
    itemsize = _ITEMSIZES.get(tc, 8)
    buf = _array.array(tc, bytearray(itemsize * n))
    v = 1.0 if 'float' in dtype else 1
    buf[:] = _array.array(tc, [v]) * n  # single-element repeat, then slice-assign
    return NDArray(buf, tuple(shape), _c_strides(shape), dtype)


def full(shape, fill_value, dtype=None) -> NDArray:
    if isinstance(shape, int): shape = (shape,)
    dtype = _resolve_dtype(dtype or ('float32' if isinstance(fill_value, float) else 'int64'))
    tc = _TYPECODES.get(dtype, 'd')
    n = _prod(shape)
    v = float(fill_value) if 'float' in dtype else int(fill_value)
    buf = _array.array(tc, [v]) * n
    return NDArray(buf, tuple(shape), _c_strides(shape), dtype)


def empty(shape, dtype='float32') -> NDArray:
    return zeros(shape, dtype=dtype)


def eye(n: int, m: int = None, k: int = 0, dtype='float32') -> NDArray:
    if m is None: m = n
    result = zeros((n, m), dtype=dtype)
    v = 1.0 if 'float' in dtype else 1
    for i in range(n):
        j = i + k
        if 0 <= j < m: result._buf[result._flat_idx((i, j))] = v
    return result


def uniform(low: float, high: float, shape, dtype='float32') -> NDArray:
    """Create NDArray with values drawn from Uniform(low, high) using stdlib random."""
    import random as _rng
    if isinstance(shape, int): shape = (shape,)
    tc = _TYPECODES.get(dtype, 'f')
    n = _prod(shape)
    rng = high - low
    buf = _array.array(tc, (low + _rng.random() * rng for _ in range(n)))
    return NDArray(buf, tuple(shape), _c_strides(shape), dtype)


def randn_nd(*shape, dtype='float32') -> NDArray:
    """Standard-normal NDArray via Box-Muller using stdlib random — no numpy."""
    import random as _rng, math as _math
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    tc = _TYPECODES.get(dtype, 'f')
    n = _prod(shape)
    vals = []
    for i in range(0, n, 2):
        u1 = _rng.random() or 1e-10
        u2 = _rng.random()
        z0 = _math.sqrt(-2.0 * _math.log(u1)) * _math.cos(2.0 * _math.pi * u2)
        z1 = _math.sqrt(-2.0 * _math.log(u1)) * _math.sin(2.0 * _math.pi * u2)
        vals.append(float(z0))
        if i + 1 < n:
            vals.append(float(z1))
    buf = _array.array(tc, vals)
    return NDArray(buf, tuple(shape), _c_strides(shape), dtype)


def arange(start, stop=None, step=1, dtype=None) -> NDArray:
    if stop is None: start, stop = 0, start
    vals = []
    x = float(start)
    while (x < stop if step > 0 else x > stop):
        vals.append(x); x += step
    dtype = dtype or ('int64' if isinstance(start, int) and isinstance(step, int) else 'float32')
    tc = _TYPECODES.get(dtype, 'd')
    if 'int' in dtype: vals = [int(v) for v in vals]
    else: vals = [float(v) for v in vals]
    buf = _array.array(tc, vals)
    return NDArray(buf, (len(vals),), (_TYPECODES.get(dtype, 'd') and 1,), dtype)


def linspace(start: float, stop: float, num: int = 50, endpoint: bool = True,
             dtype: str = 'float32') -> NDArray:
    if num == 1: return array([start], dtype=dtype)
    n = num - 1 if endpoint else num
    step = (stop - start) / n
    vals = [start + i * step for i in range(num)]
    return array(vals, dtype=dtype)


# ── Kernel-accelerated math functions ────────────────────────────────────────

def _unary_kern(a: NDArray, kern_name: str, py_fn) -> NDArray:
    """Apply unary function using C kernel if available, else pure Python."""
    if _KERN and a.dtype == 'float32':
        kern = _kern_fn(kern_name)
        if kern is not None:
            cont = a.contiguous()
            n = cont._buf.__len__()
            dst_buf = _alloc_f32(n)
            kern(_kern_ptr(dst_buf), _kern_ptr(cont._buf), n)
            return NDArray(dst_buf, a.shape, _c_strides(a.shape), 'float32')
    return _elementwise_unary(a, py_fn)


def exp(a: NDArray) -> NDArray:
    return _unary_kern(a, 'vp_exp', math.exp)

def log(a: NDArray) -> NDArray:
    return _unary_kern(a, 'vp_log', math.log)

def sqrt(a: NDArray) -> NDArray:
    return _unary_kern(a, 'vp_sqrt', math.sqrt)

def abs_arr(a: NDArray) -> NDArray:
    return _unary_kern(a, 'vp_abs', abs)

def relu(a: NDArray) -> NDArray:
    return _unary_kern(a, 'vp_relu', lambda x: max(0.0, x))

def sigmoid(a: NDArray) -> NDArray:
    return _unary_kern(a, 'vp_sigmoid', lambda x: 1.0 / (1.0 + math.exp(-x)))

def tanh_arr(a: NDArray) -> NDArray:
    return _unary_kern(a, 'vp_tanh', math.tanh)

def gelu(a: NDArray) -> NDArray:
    def _gelu(x): return 0.5*x*(1.0+math.tanh(0.7978845608028654*(x+0.044715*x**3)))
    return _unary_kern(a, 'vp_gelu', _gelu)

def silu(a: NDArray) -> NDArray:
    def _silu(x): return x / (1.0 + math.exp(-x))
    return _unary_kern(a, 'vp_silu', _silu)

def log1p(a: NDArray) -> NDArray:
    return _unary_kern(a, 'vp_log1p', math.log1p)

def expm1(a: NDArray) -> NDArray:
    return _unary_kern(a, 'vp_expm1', math.expm1)

def erf(a: NDArray) -> NDArray:
    return _unary_kern(a, 'vp_erf', _erf_scalar)


def dot(a: NDArray, b: NDArray) -> float:
    if _KERN and a.dtype == 'float32' and b.dtype == 'float32':
        kern = _kern_fn('vp_dot')
        if kern is not None:
            ca = a.contiguous(); cb = b.contiguous()
            ca_buf = ca._get_buf(); cb_buf = cb._get_buf()
            return float(kern(_kern_ptr(ca_buf), _kern_ptr(cb_buf), ca_buf.__len__()))
    return float(sum(x*y for x, y in zip(a.to_flat_list(), b.to_flat_list())))


def norm_l2(a: NDArray) -> float:
    if _KERN and a.dtype == 'float32':
        kern = _kern_fn('vp_norm_l2')
        if kern is not None:
            cont = a.contiguous()
            return float(kern(_kern_ptr(cont._buf), cont._buf.__len__()))
    return math.sqrt(sum(x*x for x in a.to_flat_list()))


def softmax_inplace(a: NDArray) -> None:
    """In-place softmax along last axis (1D arrays only)."""
    if _KERN and a.dtype == 'float32':
        kern = _kern_fn('vp_softmax_inplace')
        if kern is not None and a.ndim == 1:
            cont = a.contiguous()
            kern(_kern_ptr(cont._buf), cont._buf.__len__())
            # write back (array may be a view)
            for i, idx in enumerate(a._iter_indices()):
                a._buf[a._flat_idx(idx)] = cont._buf[i]
            return
    flat = a.to_flat_list()
    m = max(flat); s = sum(math.exp(x - m) for x in flat)
    normed = [math.exp(x - m) / s for x in flat]
    for i, idx in enumerate(a._iter_indices()):
        a._buf[a._flat_idx(idx)] = normed[i]


# ── Fused kernel convenience wrappers ─────────────────────────────────────────

def fused_rmsnorm(x: NDArray, weight: NDArray, eps: float = 1e-6) -> NDArray:
    if _KERN and x.dtype == 'float32':
        kern = _kern_fn('vp_fused_rmsnorm')
        if kern is not None:
            cx = x.contiguous(); cw = weight.contiguous()
            n = cx._buf.__len__()
            dst = _alloc_f32(n)
            kern(_kern_ptr(dst), _kern_ptr(cx._buf), _kern_ptr(cw._buf), n, eps)
            return NDArray(dst, x.shape, _c_strides(x.shape), 'float32')
    flat = x.to_flat_list()
    ss = sum(v*v for v in flat) / len(flat)
    inv = 1.0 / math.sqrt(ss + eps)
    wf = weight.to_flat_list()
    out = [flat[i] * inv * wf[i] for i in range(len(flat))]
    buf = _array.array('f', out)
    return NDArray(buf, x.shape, _c_strides(x.shape), 'float32')


def fused_layernorm(x: NDArray, weight: NDArray, bias: NDArray, eps: float = 1e-5) -> NDArray:
    if _KERN and x.dtype == 'float32':
        kern = _kern_fn('vp_fused_layernorm')
        if kern is not None:
            cx = x.contiguous(); cw = weight.contiguous()
            cb_arr = bias.contiguous() if bias is not None else None
            n = cx._buf.__len__()
            dst = _alloc_f32(n)
            kern(_kern_ptr(dst), _kern_ptr(cx._buf), _kern_ptr(cw._buf),
                 _kern_ptr(cb_arr._buf) if cb_arr else _cffi_kern._ffi.NULL if _CFFI else None,
                 n, eps)
            return NDArray(dst, x.shape, _c_strides(x.shape), 'float32')
    flat = x.to_flat_list(); n = len(flat)
    mean = sum(flat) / n
    var = sum((v - mean)**2 for v in flat) / n
    inv = 1.0 / math.sqrt(var + eps)
    wf = weight.to_flat_list()
    bf = bias.to_flat_list() if bias is not None else [0.0] * n
    out = [(flat[i] - mean) * inv * wf[i] + bf[i] for i in range(n)]
    buf = _array.array('f', out)
    return NDArray(buf, x.shape, _c_strides(x.shape), 'float32')


def adam_step_kern(param: NDArray, grad: NDArray, m: NDArray, v: NDArray,
                   lr: float, b1: float, b2: float, eps: float, t: int) -> None:
    """Fused Adam parameter update (in-place on param, m, v)."""
    if param.dtype == 'float32':
        try:
            from vectorphile._core import _vp_core as _C
            cp = param.contiguous(); cg = grad.contiguous()
            cm = m.contiguous(); cv = v.contiguous()
            bc1 = 1.0 - b1**t; bc2 = 1.0 - b2**t
            _C.adam_step_f32(cp._buf, cg._buf, cm._buf, cv._buf,
                             lr, b1, b2, eps, 0.0, bc1, bc2)
            param._inplace_update(cp)
            m._inplace_update(cm)
            v._inplace_update(cv)
            return
        except Exception:
            pass
    # Pure Python fallback
    pf = param.to_flat_list(); gf = grad.to_flat_list()
    mf = m.to_flat_list(); vf = v.to_flat_list()
    import builtins as _bi
    bc1 = 1.0 - b1**t; bc2 = 1.0 - b2**t
    lr_c = lr * math.sqrt(bc2) / bc1
    for i in range(len(pf)):
        mf[i] = b1 * mf[i] + (1.0 - b1) * gf[i]
        vf[i] = b2 * vf[i] + (1.0 - b2) * gf[i]**2
        pf[i] -= lr_c * mf[i] / (math.sqrt(vf[i]) + eps)
    for i, idx in enumerate(param._iter_indices()):
        param._buf[param._flat_idx(idx)] = pf[i]
    for i, idx in enumerate(m._iter_indices()):
        m._buf[m._flat_idx(idx)] = mf[i]
    for i, idx in enumerate(v._iter_indices()):
        v._buf[v._flat_idx(idx)] = vf[i]


def fused_linear_act(x: NDArray, weight: NDArray, bias: NDArray, act: str = 'gelu') -> NDArray:
    """Fused linear + activation: out = act(x @ W.T + b). AVX2 C kernel when available."""
    kern_map = {'gelu': 'vp_fused_linear_gelu', 'relu': 'vp_fused_linear_relu', 'silu': 'vp_fused_linear_silu'}
    kern_name = kern_map.get(act)
    if kern_name and _KERN and x.dtype == 'float32':
        kern = _kern_fn(kern_name)
        if kern is not None:
            xc = x.contiguous(); wc = weight.contiguous()
            bc = bias.contiguous() if bias is not None else None
            xflat = xc.reshape(-1, xc.shape[-1])
            N, in_f = xflat.shape; out_f = wc.shape[0]
            dst = _alloc_f32(N * out_f)
            bias_buf = bc._buf if bc is not None else _alloc_f32(out_f)
            kern(_kern_ptr(dst), _kern_ptr(xflat._buf), _kern_ptr(wc._buf), _kern_ptr(bias_buf), N, in_f, out_f)
            out_shape = x.shape[:-1] + (out_f,)
            return NDArray(dst, out_shape, _c_strides(out_shape), 'float32')
    # fallback: separate matmul + act
    y = matmul(x, weight.transpose((1, 0)))
    if bias is not None:
        y = y + bias
    if act == 'gelu':
        return gelu(y)
    if act == 'relu':
        return relu(y)
    if act == 'silu':
        return silu(y)
    return y


def fused_attention_scores(Q: NDArray, K: NDArray, scale: float) -> NDArray:
    """Fused Q @ K.T * scale for multi-head attention. Shape: (B, H, S, S)."""
    if _KERN and Q.dtype == 'float32':
        kern = _kern_fn('vp_fused_attention_scores')
        if kern is not None and Q.ndim == 4:
            Qc = Q.contiguous(); Kc = K.contiguous()
            B, H, S, D = Qc.shape
            dst = _alloc_f32(B * H * S * S)
            kern(_kern_ptr(dst), _kern_ptr(Qc._buf), _kern_ptr(Kc._buf), scale, B, H, S, D)
            out_shape = (B, H, S, S)
            return NDArray(dst, out_shape, _c_strides(out_shape), 'float32')
    # fallback
    scores = matmul(Q, Q.transpose(tuple(range(Q.ndim - 2)) + (Q.ndim - 1, Q.ndim - 2)))
    return scores * scale


# ═══════════════════════════════════════════════════════════════════════════════
# ComplexNDArray — complex64 / complex128 support (pure Python, no dependency)
# ═══════════════════════════════════════════════════════════════════════════════

class ComplexNDArray:
    """Complex-valued N-D array backed by two NDArrays (real + imag).

    dtype is 'complex64' (float32 parts) or 'complex128' (float64 parts).
    """

    def __init__(self, real: NDArray, imag: NDArray, dtype: str = 'complex64'):
        assert real.shape == imag.shape, "real and imag must have the same shape"
        self.real = real
        self.imag = imag
        self._dtype = _DType(dtype)

    @staticmethod
    def from_scalars(data, dtype='complex64') -> 'ComplexNDArray':
        """Build from a flat list of Python complex or (re, im) tuples."""
        part_dtype = 'float32' if dtype == 'complex64' else 'float64'
        reals = [v.real if isinstance(v, complex) else float(v[0]) for v in data]
        imags = [v.imag if isinstance(v, complex) else float(v[1]) for v in data]
        re = array(reals, dtype=part_dtype)
        im = array(imags, dtype=part_dtype)
        return ComplexNDArray(re, im, dtype)

    @staticmethod
    def zeros(shape, dtype='complex64') -> 'ComplexNDArray':
        part_dtype = 'float32' if dtype == 'complex64' else 'float64'
        return ComplexNDArray(zeros(shape, dtype=part_dtype), zeros(shape, dtype=part_dtype), dtype)

    @staticmethod
    def from_real(real: NDArray, dtype='complex64') -> 'ComplexNDArray':
        part_dtype = 'float32' if dtype == 'complex64' else 'float64'
        return ComplexNDArray(real.astype(part_dtype), zeros(real.shape, dtype=part_dtype), dtype)

    @property
    def shape(self): return self.real.shape
    @property
    def ndim(self): return self.real.ndim
    @property
    def size(self): return self.real.size
    @property
    def dtype(self): return self._dtype

    def conj(self) -> 'ComplexNDArray':
        return ComplexNDArray(self.real.copy(), self.imag * -1.0, self._dtype)

    def abs(self) -> NDArray:
        """Element-wise magnitude |z| = sqrt(re² + im²)."""
        from vectorphile._core.ndarray import sqrt as _sqrt
        return _sqrt(self.real * self.real + self.imag * self.imag)

    def angle(self) -> NDArray:
        """Element-wise phase angle in radians."""
        def _atan2_flat(y_flat, x_flat):
            return _array.array('f', [math.atan2(float(y), float(x))
                                      for y, x in zip(y_flat, x_flat)])
        yf = list(self.imag.to_flat_list())
        xf = list(self.real.to_flat_list())
        buf = _atan2_flat(yf, xf)
        return NDArray(buf, self.shape, _c_strides(self.shape), _DType('float32'))

    def __add__(self, other):
        if isinstance(other, ComplexNDArray):
            return ComplexNDArray(self.real + other.real, self.imag + other.imag, self._dtype)
        return ComplexNDArray(self.real + other, self.imag.copy(), self._dtype)

    def __sub__(self, other):
        if isinstance(other, ComplexNDArray):
            return ComplexNDArray(self.real - other.real, self.imag - other.imag, self._dtype)
        return ComplexNDArray(self.real - other, self.imag.copy(), self._dtype)

    def __mul__(self, other):
        if isinstance(other, ComplexNDArray):
            # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            re = self.real * other.real - self.imag * other.imag
            im = self.real * other.imag + self.imag * other.real
            return ComplexNDArray(re, im, self._dtype)
        return ComplexNDArray(self.real * other, self.imag * other, self._dtype)

    def __rmul__(self, other): return self.__mul__(other)
    def __radd__(self, other): return self.__add__(other)

    def __truediv__(self, other):
        if isinstance(other, ComplexNDArray):
            denom = other.real * other.real + other.imag * other.imag
            re = (self.real * other.real + self.imag * other.imag) / denom
            im = (self.imag * other.real - self.real * other.imag) / denom
            return ComplexNDArray(re, im, self._dtype)
        return ComplexNDArray(self.real / other, self.imag / other, self._dtype)

    def __neg__(self):
        return ComplexNDArray(self.real * -1.0, self.imag * -1.0, self._dtype)

    def exp(self) -> 'ComplexNDArray':
        """e^(a+bi) = e^a * (cos(b) + i*sin(b))"""
        ea = exp(self.real)
        cos_b = _unary_kern(self.imag, None, math.cos)
        sin_b = _unary_kern(self.imag, None, math.sin)
        return ComplexNDArray(ea * cos_b, ea * sin_b, self._dtype)

    def matmul(self, other: 'ComplexNDArray') -> 'ComplexNDArray':
        """Complex matrix multiply: (A_re + i*A_im) @ (B_re + i*B_im)."""
        re = matmul(self.real, other.real) - matmul(self.imag, other.imag)
        im = matmul(self.real, other.imag) + matmul(self.imag, other.real)
        return ComplexNDArray(re, im, self._dtype)

    def __matmul__(self, other): return self.matmul(other)

    def reshape(self, *shape) -> 'ComplexNDArray':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return ComplexNDArray(self.real.reshape(*shape), self.imag.reshape(*shape), self._dtype)

    def to_flat_list(self):
        """Interleaved real, imag pairs."""
        re = list(self.real.to_flat_list())
        im = list(self.imag.to_flat_list())
        return [v for pair in zip(re, im) for v in pair]

    def __repr__(self):
        re_l = list(self.real.to_flat_list())[:4]
        im_l = list(self.imag.to_flat_list())[:4]
        pairs = [f'{r:.4f}+{i:.4f}j' for r, i in zip(re_l, im_l)]
        return f'ComplexNDArray({pairs}..., shape={self.shape}, dtype={self._dtype})'


# ── FFT using the Cooley-Tukey algorithm (pure Python, power-of-2) ────────────

def _fft_1d(x_re: list, x_im: list, inverse: bool = False) -> tuple:
    """Cooley-Tukey radix-2 DIT FFT on flat lists."""
    n = len(x_re)
    if n == 1:
        return x_re[:], x_im[:]
    # Bit-reversal permutation
    bits = int(math.log2(n))
    rev = [0] * n
    for i in range(1, n):
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bits - 1))
    xr = [x_re[rev[i]] for i in range(n)]
    xi = [x_im[rev[i]] for i in range(n)]
    sign = 1.0 if inverse else -1.0
    length = 2
    while length <= n:
        half = length // 2
        ang = sign * 2.0 * math.pi / length
        wr, wi = math.cos(ang), math.sin(ang)
        for i in range(0, n, length):
            cur_wr, cur_wi = 1.0, 0.0
            for j in range(half):
                u_r = xr[i + j]; u_i = xi[i + j]
                v_r = xr[i + j + half] * cur_wr - xi[i + j + half] * cur_wi
                v_i = xr[i + j + half] * cur_wi + xi[i + j + half] * cur_wr
                xr[i + j] = u_r + v_r; xi[i + j] = u_i + v_i
                xr[i + j + half] = u_r - v_r; xi[i + j + half] = u_i - v_i
                new_wr = cur_wr * wr - cur_wi * wi
                cur_wi = cur_wr * wi + cur_wi * wr
                cur_wr = new_wr
        length <<= 1
    if inverse:
        xr = [v / n for v in xr]
        xi = [v / n for v in xi]
    return xr, xi


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def fft(a: 'NDArray | ComplexNDArray', axis: int = -1) -> 'ComplexNDArray':
    """1-D FFT along the given axis."""
    if isinstance(a, NDArray):
        a = ComplexNDArray.from_real(a)
    nd = a.ndim
    ax = axis % nd
    # Only 1-D and 2-D last-axis supported natively
    if nd == 1:
        n = a.shape[0]
        p2 = _next_pow2(n)
        re = list(a.real.to_flat_list()) + [0.0] * (p2 - n)
        im = list(a.imag.to_flat_list()) + [0.0] * (p2 - n)
        or_, oi_ = _fft_1d(re, im)
        or_ = or_[:n]; oi_ = oi_[:n]
        re_nd = array(or_, dtype='float32')
        im_nd = array(oi_, dtype='float32')
        return ComplexNDArray(re_nd, im_nd, a.dtype)
    # For 2-D or higher, apply 1-D FFT along rows/cols by reshaping
    if nd == 2 and ax == 1:
        rows, cols = a.shape
        p2 = _next_pow2(cols)
        re_out = []
        im_out = []
        re_flat = list(a.real.to_flat_list())
        im_flat = list(a.imag.to_flat_list())
        for r in range(rows):
            row_re = re_flat[r * cols:(r + 1) * cols] + [0.0] * (p2 - cols)
            row_im = im_flat[r * cols:(r + 1) * cols] + [0.0] * (p2 - cols)
            or_, oi_ = _fft_1d(row_re, row_im)
            re_out.extend(or_[:cols])
            im_out.extend(oi_[:cols])
        re_nd = array(re_out, dtype='float32').reshape(rows, cols)
        im_nd = array(im_out, dtype='float32').reshape(rows, cols)
        return ComplexNDArray(re_nd, im_nd, a.dtype)
    if nd == 2 and ax == 0:
        # FFT along columns: transpose, fft along rows, transpose back
        a_t = ComplexNDArray(a.real.transpose(), a.imag.transpose(), a.dtype)
        result = fft(a_t, axis=1)
        re_nd = result.real.transpose()
        im_nd = result.imag.transpose()
        return ComplexNDArray(re_nd, im_nd, a.dtype)
    # For higher-dimensional arrays: apply along specified axis via reshape trick
    # Reshape to (batch, axis_size), fft, reshape back
    sh = a.shape
    ax_size = sh[ax]
    batch_size = 1
    for i, s in enumerate(sh):
        if i != ax:
            batch_size *= s
    re_flat = list(a.real.to_flat_list())
    im_flat = list(a.imag.to_flat_list())
    # Build index mapping: iterate over all positions, group by axis position
    p2 = _next_pow2(ax_size)
    # Use strides to extract slices along axis
    strides = [1] * nd
    for i in range(nd - 2, -1, -1):
        strides[i] = strides[i + 1] * sh[i + 1]
    re_out = [0.0] * len(re_flat)
    im_out = [0.0] * len(im_flat)
    # Iterate over all batch positions (all indices except the ax dimension)
    from itertools import product as _iproduct
    other_ranges = [range(sh[i]) for i in range(nd) if i != ax]
    for other_idx in _iproduct(*other_ranges):
        base = 0
        oi = 0
        for i in range(nd):
            if i != ax:
                base += other_idx[oi] * strides[i]
                oi += 1
        seq_re = [re_flat[base + k * strides[ax]] for k in range(ax_size)]
        seq_im = [im_flat[base + k * strides[ax]] for k in range(ax_size)]
        seq_re += [0.0] * (p2 - ax_size)
        seq_im += [0.0] * (p2 - ax_size)
        or_, oi_ = _fft_1d(seq_re, seq_im)
        for k in range(ax_size):
            idx = base + k * strides[ax]
            re_out[idx] = or_[k]
            im_out[idx] = oi_[k]
    re_nd = array(re_out, dtype='float32').reshape(sh)
    im_nd = array(im_out, dtype='float32').reshape(sh)
    return ComplexNDArray(re_nd, im_nd, a.dtype)


def ifft(a: 'NDArray | ComplexNDArray', axis: int = -1) -> 'ComplexNDArray':
    """Inverse FFT along the given axis (supports N-D)."""
    if isinstance(a, NDArray):
        a = ComplexNDArray.from_real(a)
    nd = a.ndim
    ax = axis % nd
    if nd == 1:
        n = a.shape[0]
        p2 = _next_pow2(n)
        re = list(a.real.to_flat_list()) + [0.0] * (p2 - n)
        im = list(a.imag.to_flat_list()) + [0.0] * (p2 - n)
        or_, oi_ = _fft_1d(re, im, inverse=True)
        re_nd = array(or_[:n], dtype='float32')
        im_nd = array(oi_[:n], dtype='float32')
        return ComplexNDArray(re_nd, im_nd, a.dtype)
    # Use conjugation trick: ifft(x) = conj(fft(conj(x))) / N
    neg_imag = a.imag * (-1.0)
    conj_a = ComplexNDArray(a.real, neg_imag, a.dtype)
    fft_conj = fft(conj_a, axis=ax)
    ax_size = a.shape[ax]
    scale = 1.0 / ax_size
    re_nd = fft_conj.real * scale
    im_nd = fft_conj.imag * (-scale)
    return ComplexNDArray(re_nd, im_nd, a.dtype)


def fft2(a: 'NDArray | ComplexNDArray') -> 'ComplexNDArray':
    """2-D FFT: FFT along rows then columns."""
    if isinstance(a, NDArray):
        a = ComplexNDArray.from_real(a)
    # Rows
    temp = fft(a, axis=1)
    # Cols: transpose, fft along axis=1, transpose back
    temp_t = ComplexNDArray(temp.real.transpose(), temp.imag.transpose(), a.dtype)
    result_t = fft(temp_t, axis=1)
    return ComplexNDArray(result_t.real.transpose(), result_t.imag.transpose(), a.dtype)
