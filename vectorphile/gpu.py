"""vectorphile.gpu — GPU arrays backed by vectorphile's own _vp_cuda kernels.

GArray wraps a CudaBuffer and implements the full ndarray-compatible interface
using our own CUDA C extension (_vp_cuda.pyd). No torch or external ML libs.

If _vp_cuda is not built yet (run scripts/build_vp_cuda_win.bat), all ops
fall back transparently to CPU NDArray so the rest of the library still works.
"""
from __future__ import annotations
import array as _arr
import operator
from typing import Optional, Tuple, Any

from vectorphile._core.ndarray import (
    NDArray, array as _nd_array, zeros as _nd_zeros, ones as _nd_ones,
    full as _nd_full, arange as _nd_arange,
)

# ── Load our own CUDA extension ───────────────────────────────────────────────
try:
    from vectorphile._core import _vp_cuda as _C
    _CUDA_OK = _C.device_count() > 0
except ImportError:
    _C = None
    _CUDA_OK = False

CUDA_AVAILABLE = _CUDA_OK
MPS_AVAILABLE = False      # Metal handled separately in vectorphile.metal
AVAILABLE = CUDA_AVAILABLE
_DEFAULT_DEVICE = 'cuda' if CUDA_AVAILABLE else 'cpu'


def _flat_f32(nd: NDArray) -> '_arr.array':
    """Extract flat float32 array.array from NDArray."""
    flat = nd.to_flat_list() if hasattr(nd, 'to_flat_list') else list(nd)
    return _arr.array('f', [float(v) for v in flat])


def _nd_from_flat(flat, shape) -> NDArray:
    """Reconstruct NDArray from flat array.array and shape."""
    buf = _arr.array('f', flat) if not isinstance(flat, _arr.array) else flat
    from vectorphile._core.ndarray import _c_strides, _DType
    strides = _c_strides(shape)
    return NDArray(buf, shape, strides, _DType('float32'))


# ═══════════════════════════════════════════════════════════════════════════════
# GArray — device-resident float32 array
# ═══════════════════════════════════════════════════════════════════════════════

class GArray:
    """GPU-resident float32 array backed by vectorphile's CUDA extension.

    When CUDA is unavailable, all ops fall back to NDArray on CPU so code
    written against GArray still runs correctly everywhere.
    """
    __array_priority__ = 1000

    def __init__(self, data, shape: tuple, device: str = 'cuda'):
        """data: CudaBuffer (on GPU) or NDArray/array.array (on CPU fallback)."""
        self._shape = tuple(shape)
        self._device = device if _CUDA_OK else 'cpu'
        if _CUDA_OK and hasattr(data, 'nbytes'):
            # data is a CudaBuffer
            self._buf = data
            self._cpu: Optional[NDArray] = None
        else:
            # CPU fallback: store as NDArray
            self._buf = None
            if isinstance(data, NDArray):
                self._cpu = data
            elif isinstance(data, _arr.array):
                self._cpu = _nd_from_flat(data, shape)
            else:
                flat = _arr.array('f', [float(v) for v in data])
                self._cpu = _nd_from_flat(flat, shape)

    # ── constructors ─────────────────────────────────────────────────────────

    @staticmethod
    def from_nd(nd: NDArray, device: str = 'cuda') -> 'GArray':
        if not _CUDA_OK:
            return GArray(nd, nd.shape, 'cpu')
        flat = _flat_f32(nd)
        nbytes = len(flat) * 4
        buf = _C.alloc(nbytes)
        _C.h2d(flat, buf)
        return GArray(buf, nd.shape, device)

    @staticmethod
    def from_list(data: list, shape: tuple, device: str = 'cuda') -> 'GArray':
        flat = _arr.array('f', [float(v) for v in data])
        if not _CUDA_OK:
            return GArray(_nd_from_flat(flat, shape), shape, 'cpu')
        nbytes = len(flat) * 4
        buf = _C.alloc(nbytes)
        _C.h2d(flat, buf)
        return GArray(buf, shape, device)

    def get(self) -> NDArray:
        """Copy device data back to a CPU NDArray."""
        if self._cpu is not None:
            return self._cpu
        n_floats = 1
        for d in self._shape:
            n_floats *= d
        cpu_buf = _arr.array('f', bytearray(n_floats * 4))
        _C.d2h(self._buf, cpu_buf)
        _C.synchronize()
        return _nd_from_flat(cpu_buf, self._shape)

    def _ensure_cpu(self) -> NDArray:
        if self._cpu is not None:
            return self._cpu
        self._cpu = self.get()
        return self._cpu

    def _cuda_buf(self):
        """Return CudaBuffer, uploading from cpu cache if needed."""
        if self._buf is not None:
            return self._buf
        flat = _flat_f32(self._cpu)
        nbytes = len(flat) * 4
        buf = _C.alloc(nbytes)
        _C.h2d(flat, buf)
        self._buf = buf
        return buf

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        n = 1
        for d in self._shape:
            n *= d
        return n

    @property
    def dtype(self) -> str:
        return 'float32'

    @property
    def device(self) -> str:
        return self._device

    @property
    def nbytes(self) -> int:
        return self.size * 4

    @property
    def T(self) -> 'GArray':
        return self.transpose()

    # ── shape ops ────────────────────────────────────────────────────────────

    def reshape(self, *shape) -> 'GArray':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        # Infer -1
        shape = list(shape)
        neg_idx = [i for i, d in enumerate(shape) if d == -1]
        if neg_idx:
            known = 1
            for i, d in enumerate(shape):
                if d != -1:
                    known *= d
            shape[neg_idx[0]] = self.size // known
        new_shape = tuple(shape)
        g = GArray(self._buf or self._cpu, new_shape, self._device)
        g._buf = self._buf
        g._cpu = self._cpu
        return g

    def flatten(self) -> 'GArray':
        return self.reshape(self.size)

    ravel = flatten

    def transpose(self, *axes) -> 'GArray':
        nd = self._ensure_cpu()
        if axes:
            result = nd.transpose(axes[0] if len(axes) == 1 and isinstance(axes[0], (tuple, list)) else axes)
        else:
            result = nd.transpose()
        return GArray(result, result.shape, self._device)

    def squeeze(self, axis=None) -> 'GArray':
        nd = self._ensure_cpu().squeeze(axis) if axis is not None else self._ensure_cpu().squeeze()
        return GArray(nd, nd.shape, self._device)

    def unsqueeze(self, axis: int) -> 'GArray':
        nd = self._ensure_cpu()
        new_shape = list(self._shape)
        new_shape.insert(axis, 1)
        return self.reshape(tuple(new_shape))

    def astype(self, dtype, copy=True) -> 'GArray':
        nd = self._ensure_cpu().astype(dtype)
        return GArray.from_nd(nd, self._device)

    def copy(self) -> 'GArray':
        if self._buf is not None and _CUDA_OK:
            # Allocate new buffer and memcpy
            nbytes = self.size * 4
            new_buf = _C.alloc(nbytes)
            cpu = self.get()
            flat = _flat_f32(cpu)
            _C.h2d(flat, new_buf)
            return GArray(new_buf, self._shape, self._device)
        cpu = self._ensure_cpu()
        return GArray(cpu.copy(), self._shape, self._device)

    def item(self):
        return float(self._ensure_cpu().item())

    def tolist(self):
        return list(self._ensure_cpu().to_flat_list())

    def to_flat_list(self):
        return self._ensure_cpu().to_flat_list()

    # ── arithmetic — dispatch through GPU kernels when available ─────────────

    def _broadcast_op(self, other, op_name: str) -> 'GArray':
        """Binary op via CPU NDArray (we have add/mul CUDA kernels for same-shape)."""
        a = self._ensure_cpu()
        b = other._ensure_cpu() if isinstance(other, GArray) else other
        result = getattr(a, f'__{op_name}__')(b)
        return GArray(result, result.shape, self._device)

    def __add__(self, other):
        # Use CUDA add kernel for same-shape float32
        if (_CUDA_OK and isinstance(other, GArray) and self._shape == other._shape):
            gc = _C.add_f32(self._cuda_buf(), other._cuda_buf(), self.size)
            _C.synchronize()
            return GArray(gc, self._shape, self._device)
        return self._broadcast_op(other, 'add')

    def __radd__(self, other): return self.__add__(other)
    def __iadd__(self, other): r = self.__add__(other); self._buf = r._buf; self._cpu = r._cpu; return self

    def __sub__(self, other):
        # a - b = a + (-1)*b
        if (_CUDA_OK and isinstance(other, GArray) and self._shape == other._shape):
            neg_b = _C.scale_f32(other._cuda_buf(), -1.0, other.size)
            gc = _C.add_f32(self._cuda_buf(), neg_b, self.size)
            _C.synchronize()
            return GArray(gc, self._shape, self._device)
        return self._broadcast_op(other, 'sub')

    def __rsub__(self, other):
        return GArray(other, self._shape, self._device).__sub__(self) if isinstance(other, (int, float)) else self._broadcast_op(other, 'rsub')

    def __mul__(self, other):
        if (_CUDA_OK and isinstance(other, (int, float))):
            gc = _C.scale_f32(self._cuda_buf(), float(other), self.size)
            _C.synchronize()
            return GArray(gc, self._shape, self._device)
        return self._broadcast_op(other, 'mul')

    def __rmul__(self, other): return self.__mul__(other)
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self.__mul__(1.0 / other)
        return self._broadcast_op(other, 'truediv')
    def __rtruediv__(self, other): return self._broadcast_op(other, 'rtruediv')
    def __neg__(self): return self.__mul__(-1.0)
    def __pow__(self, other): return self._broadcast_op(other, 'pow')
    def __rpow__(self, other): return self._broadcast_op(other, 'rpow')
    def __mod__(self, other): return self._broadcast_op(other, 'mod')
    def __floordiv__(self, other): return self._broadcast_op(other, 'floordiv')

    def __matmul__(self, other: 'GArray') -> 'GArray':
        a_nd = self._ensure_cpu()
        b_nd = other._ensure_cpu() if isinstance(other, GArray) else other
        if _CUDA_OK and self.ndim == 2 and isinstance(other, GArray) and other.ndim == 2:
            m, k, n = self._shape[0], self._shape[1], other._shape[1]
            gc = _C.gemm_f32(self._cuda_buf(), other._cuda_buf(), m, k, n)
            _C.synchronize()
            return GArray(gc, (m, n), self._device)
        result = a_nd @ b_nd
        return GArray(result, result.shape, self._device)

    def __rmatmul__(self, other):
        return GArray.from_nd(other, self._device).__matmul__(self)

    # Comparison ops
    def __eq__(self, other): nd = self._ensure_cpu(); r = nd == (other._ensure_cpu() if isinstance(other, GArray) else other); return GArray(r, r.shape, self._device)
    def __ne__(self, other): nd = self._ensure_cpu(); r = nd != (other._ensure_cpu() if isinstance(other, GArray) else other); return GArray(r, r.shape, self._device)
    def __lt__(self, other): nd = self._ensure_cpu(); r = nd < (other._ensure_cpu() if isinstance(other, GArray) else other); return GArray(r, r.shape, self._device)
    def __le__(self, other): nd = self._ensure_cpu(); r = nd <= (other._ensure_cpu() if isinstance(other, GArray) else other); return GArray(r, r.shape, self._device)
    def __gt__(self, other): nd = self._ensure_cpu(); r = nd > (other._ensure_cpu() if isinstance(other, GArray) else other); return GArray(r, r.shape, self._device)
    def __ge__(self, other): nd = self._ensure_cpu(); r = nd >= (other._ensure_cpu() if isinstance(other, GArray) else other); return GArray(r, r.shape, self._device)
    __hash__ = None

    def __abs__(self): nd = self._ensure_cpu().abs(); return GArray(nd, nd.shape, self._device)
    def __invert__(self): nd = ~self._ensure_cpu(); return GArray(nd, nd.shape, self._device)
    def __bool__(self): return bool(self.item())
    def __float__(self): return self.item()
    def __int__(self): return int(self.item())
    def __len__(self): return self._shape[0]
    def __repr__(self): return f'GArray({self.get()!r}, device={self._device})'

    def __getitem__(self, idx):
        nd = self._ensure_cpu()[idx]
        if isinstance(nd, NDArray):
            return GArray(nd, nd.shape, self._device)
        return nd

    def __setitem__(self, idx, value):
        nd = self._ensure_cpu()
        nd[idx] = value._ensure_cpu() if isinstance(value, GArray) else value
        self._cpu = nd
        self._buf = None  # invalidate GPU buffer

    def __iter__(self):
        nd = self._ensure_cpu()
        for i in range(self._shape[0]):
            row = nd[i]
            yield GArray(row, row.shape, self._device)

    # ── reduction ops ────────────────────────────────────────────────────────
    def sum(self, axis=None, keepdims=False, dtype=None):
        nd = self._ensure_cpu().sum(axis=axis, keepdims=keepdims)
        return GArray(nd, nd.shape, self._device)

    def mean(self, axis=None, keepdims=False, dtype=None):
        nd = self._ensure_cpu().mean(axis=axis, keepdims=keepdims)
        return GArray(nd, nd.shape, self._device)

    def max(self, axis=None, keepdims=False):
        nd = self._ensure_cpu().max(axis=axis, keepdims=keepdims)
        if isinstance(nd, NDArray):
            return GArray(nd, nd.shape, self._device)
        return nd

    def min(self, axis=None, keepdims=False):
        nd = self._ensure_cpu().min(axis=axis, keepdims=keepdims)
        if isinstance(nd, NDArray):
            return GArray(nd, nd.shape, self._device)
        return nd

    def std(self, axis=None, keepdims=False, ddof=0):
        nd = self._ensure_cpu().std(axis=axis, keepdims=keepdims, ddof=ddof)
        return GArray(nd, nd.shape if isinstance(nd, NDArray) else (), self._device)

    def var(self, axis=None, keepdims=False, ddof=0):
        nd = self._ensure_cpu().var(axis=axis, keepdims=keepdims, ddof=ddof)
        return GArray(nd, nd.shape if isinstance(nd, NDArray) else (), self._device)

    def argmax(self, axis=None):
        nd = self._ensure_cpu()
        idx = nd.argmax(axis=axis)
        if isinstance(idx, NDArray):
            return GArray(idx, idx.shape, self._device)
        return int(idx)

    def argmin(self, axis=None):
        nd = self._ensure_cpu()
        idx = nd.argmin(axis=axis)
        if isinstance(idx, NDArray):
            return GArray(idx, idx.shape, self._device)
        return int(idx)

    def cumsum(self, axis=None):
        nd = self._ensure_cpu().cumsum(axis=axis)
        return GArray(nd, nd.shape, self._device)

    def any(self, axis=None, keepdims=False):
        nd = self._ensure_cpu()
        r = nd.any(axis=axis, keepdims=keepdims) if hasattr(nd, 'any') else nd.sum() > 0
        if isinstance(r, NDArray):
            return GArray(r, r.shape, self._device)
        return bool(r)

    def all(self, axis=None, keepdims=False):
        nd = self._ensure_cpu()
        r = nd.all(axis=axis, keepdims=keepdims) if hasattr(nd, 'all') else nd.min() > 0
        if isinstance(r, NDArray):
            return GArray(r, r.shape, self._device)
        return bool(r)

    def clip(self, lo=None, hi=None):
        nd = self._ensure_cpu().clip(lo, hi)
        return GArray(nd, nd.shape, self._device)

    # ── elementwise math (CPU for now; major ops have CUDA kernels) ──────────
    def _unary(self, fn_name: str) -> 'GArray':
        nd = getattr(self._ensure_cpu(), fn_name)()
        return GArray(nd, nd.shape, self._device)

    def exp(self):   return self._unary('exp')
    def log(self):   return self._unary('log')
    def sqrt(self):  return self._unary('sqrt')
    def abs(self):   return self._unary('abs')
    def tanh(self):  return self._unary('tanh')
    def sin(self):   return self._unary('sin')
    def cos(self):   return self._unary('cos')
    def sign(self):  return self._unary('sign')

    def relu(self) -> 'GArray':
        if _CUDA_OK:
            gc = _C.relu_f32(self._cuda_buf(), self.size)
            _C.synchronize()
            return GArray(gc, self._shape, self._device)
        return self._unary('relu')

    def softmax(self, axis: int = -1) -> 'GArray':
        if _CUDA_OK and self.ndim == 2:
            rows, cols = self._shape if self.ndim == 2 else (self.size, 1)
            gc = _C.softmax_f32(self._cuda_buf(), rows, cols)
            _C.synchronize()
            return GArray(gc, (rows, cols), self._device)
        nd = self._ensure_cpu()
        # manual softmax
        mx = nd.max(axis=axis, keepdims=True)
        e = (nd - mx).exp()
        s = e.sum(axis=axis, keepdims=True)
        result = e / s
        return GArray(result, result.shape, self._device)

    def sigmoid(self):
        nd = self._ensure_cpu()
        one = _nd_ones(nd.shape, dtype='float32')
        result = one / (one + (-nd).exp())
        return GArray(result, result.shape, self._device)

    # ── NumPy protocol ───────────────────────────────────────────────────────
    def __array__(self, dtype=None, copy=None):
        return self._ensure_cpu()

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kw):
        from vectorphile._core.ndarray import NDArray
        if method == '__call__':
            cpu_inputs = [i._ensure_cpu() if isinstance(i, GArray) else i for i in inputs]
            result = ufunc(*cpu_inputs, **kw)
            if isinstance(result, NDArray):
                return GArray(result, result.shape, self._device)
            return result
        return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        return NotImplemented


# ── High-level GPU tensor creation helpers ────────────────────────────────────

def zeros(shape: tuple, device: str = 'cuda') -> GArray:
    nd = _nd_zeros(shape, dtype='float32')
    return GArray.from_nd(nd, device)

def ones(shape: tuple, device: str = 'cuda') -> GArray:
    nd = _nd_ones(shape, dtype='float32')
    return GArray.from_nd(nd, device)

def full(shape: tuple, fill_value: float, device: str = 'cuda') -> GArray:
    nd = _nd_full(shape, fill_value, dtype='float32')
    return GArray.from_nd(nd, device)

def to_gpu(a, device: str = 'cuda') -> GArray:
    if isinstance(a, GArray):
        return a
    if isinstance(a, NDArray):
        return GArray.from_nd(a, device)
    nd = _nd_array(list(a) if hasattr(a, '__iter__') else [a], dtype='float32')
    return GArray.from_nd(nd, device)

def to_cpu(a) -> NDArray:
    if isinstance(a, GArray):
        return a.get()
    return a


# ── Wire into Tensor / Module via install() ───────────────────────────────────

def install(vp_module=None):
    """Patch Tensor and Module with .to(), .cuda(), .cpu() etc."""
    from vectorphile.core.tensor import Tensor, _DTYPE_MAP
    from vectorphile.nn.module import Module

    orig_init = Tensor.__init__

    def _patched_init(self, data, dtype=None, device='cpu', requires_grad=False):
        if isinstance(data, GArray):
            self._data = data
            self.device = data.device
            self.requires_grad = requires_grad
            self.grad = None; self.grad_fn = None; self._is_leaf = True
            return
        orig_init(self, data, dtype=dtype, device=device, requires_grad=requires_grad)
        dev = str(device)
        if AVAILABLE and (dev.startswith('cuda') or dev == 'auto'):
            target = 'cuda' if dev.startswith('cuda') else _DEFAULT_DEVICE
            if isinstance(self._data, NDArray):
                self._data = to_gpu(self._data, target)
                self.device = target

    Tensor.__init__ = _patched_init

    def _tensor_to(self, device=None, dtype=None, **_):
        if dtype is not None:
            d = _DTYPE_MAP.get(dtype, dtype)
            if isinstance(self._data, GArray):
                self._data = self._data.astype(d)
            else:
                self._data = self._data.astype(d)
        if device is not None:
            dev = str(device)
            if dev.startswith('cuda') or dev == 'auto':
                if not AVAILABLE:
                    return self  # graceful: stay on CPU
                target = 'cuda'
                if not isinstance(self._data, GArray):
                    self._data = to_gpu(self._data, target)
                self.device = target
                if self.grad is not None and not isinstance(self.grad._data, GArray):
                    self.grad._data = to_gpu(self.grad._data, target)
            elif dev == 'cpu':
                if isinstance(self._data, GArray):
                    self._data = self._data.get()
                self.device = 'cpu'
                if self.grad is not None and isinstance(self.grad._data, GArray):
                    self.grad._data = self.grad._data.get()
        return self

    Tensor.to = _tensor_to
    Tensor.cuda = lambda self, *a, **k: _tensor_to(self, 'cuda')
    Tensor.cpu = lambda self, *a, **k: _tensor_to(self, 'cpu')
    Tensor.numpy = lambda self: (self._data.get() if isinstance(self._data, GArray) else self._data)
    Tensor.item = lambda self: float(self._data.item() if hasattr(self._data, 'item') else list(self._data.to_flat_list())[0])
    Tensor.tolist = lambda self: list(self._data.to_flat_list())
    Tensor.is_cuda = property(lambda self: isinstance(self._data, GArray))

    def _module_to(self, device=None, *_, **__):
        if device is None:
            return self
        for p in self.parameters():
            p.to(device)
        return self

    Module.to = _module_to
    Module.cuda = lambda self, *a, **k: _module_to(self, 'cuda')
    Module.cpu = lambda self, *a, **k: _module_to(self, 'cpu')

    if vp_module is not None:
        # Patch vp.cuda module-level helpers to use our _vp_cuda
        import types as _types
        c = getattr(vp_module, 'cuda', None)
        if c is not None and isinstance(c, _types.ModuleType):
            c.is_available = lambda: CUDA_AVAILABLE
            c.device_count = lambda: (_C.device_count() if _C else 0)
            c.device_name = lambda i=0: (_C.device_name(i) if _C else 'CPU')
            c.synchronize = lambda: (_C.synchronize() if _C else None)
            c.memory_allocated = lambda i=0: 0  # TODO: implement in _vp_cuda
            c.empty_cache = lambda: None
