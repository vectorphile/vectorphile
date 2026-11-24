"""Core Tensor class — the foundation of vectorphile."""
from __future__ import annotations
import math as _math_t
import array as _array_t
from typing import Any, Optional, Tuple, Union, List

# Buffer cache for elementwise ops (relu, etc.) — avoids per-call array.array allocation
# keyed by element count → array.array buffer; re-allocated only when size changes
_relu_buf_cache: dict = {}  # n_elements → array.array('f', ...)


def _grad_enabled() -> bool:
    """Return True when gradient tracking is active (no_grad not in effect)."""
    try:
        from vectorphile.autograd.engine import no_grad as _ng
        return _ng._enabled
    except Exception:
        return True


_DTYPE_MAP = {
    'float32': float, 'float': float,
    'float64': float, 'double': float,
    'float16': float, 'half': float,
    'int32': int, 'int': int,
    'int64': int, 'long': int,
    'int8': int, 'uint8': int,
    'bool': bool,
}

_DTYPE_TC = {
    'float32': 'f', 'float': 'f', 'float64': 'd', 'double': 'd',
    'int32': 'i', 'int': 'i', 'int64': 'l', 'long': 'l',
    'int8': 'b', 'uint8': 'B',
}


class Tensor:
    """N-dimensional tensor backed by numpy, with autograd support."""

    def __init__(self, data, dtype=None, device='cpu', requires_grad=False, _no_copy=False):
        if isinstance(data, Tensor):
            data = data._data
        # Handle numpy arrays by converting to flat list first
        try:
            import numpy as _npx
            if isinstance(data, _npx.ndarray):
                _shape = data.shape
                _flat = data.flatten().tolist()
                from vectorphile._core.ndarray import array as _nd_np
                # Preserve float64 precision if input is float64 and no explicit dtype
                _inferred_dtype = dtype or (str(data.dtype) if str(data.dtype) in ('float32', 'float64', 'int32', 'int64', 'bool') else 'float32')
                _nd = _nd_np(_flat, dtype=_inferred_dtype)
                if len(_shape) > 1:
                    _nd = _nd.reshape(_shape)
                data = _nd
        except ImportError:
            pass
        from vectorphile._core.ndarray import NDArray as _NDA
        if isinstance(data, _NDA):
            # Store NDArray directly — do not convert to numpy
            if dtype and str(data.dtype) != dtype:
                self._data = data.astype(dtype)
            else:
                self._data = data
        else:
            # Convert lists/scalars to NDArray (buf-backed) when float32
            # Auto-detect bool input
            if dtype is None and isinstance(data, (list, tuple)):
                _probe = data
                while isinstance(_probe, (list, tuple)) and _probe:
                    _probe = _probe[0]
                if isinstance(_probe, bool):
                    dtype = 'bool'
            target_dtype = dtype or 'float32'
            if target_dtype in ('float32', 'float') and isinstance(data, (list, tuple)):
                try:
                    import array as _a, struct as _s
                    flat = []
                    def _flatten(v):
                        if isinstance(v, (list, tuple)):
                            for x in v: _flatten(x)
                        else: flat.append(float(v))
                    _flatten(data)
                    buf = _a.array('f', flat)
                    # infer shape
                    shape = []
                    cur = data
                    while isinstance(cur, (list, tuple)):
                        shape.append(len(cur))
                        cur = cur[0] if cur else []
                    shape = tuple(shape)
                    from vectorphile._core.ndarray import NDArray as _NDA2, _c_strides as _cs2, _DType as _DT2
                    self._data = _NDA2(buf, shape, _cs2(shape), _DT2('float32'))
                except Exception:
                    from vectorphile._core.ndarray import array as _nd_arr
                    self._data = _nd_arr([float(v) for v in flat], dtype='float32').reshape(shape)
            else:
                # Non-float32: convert to NDArray with correct shape
                from vectorphile._core.ndarray import array as _nd_arr2
                target_dt = dtype or 'float32'
                try:
                    flat2 = []
                    def _flat2(v):
                        if isinstance(v, (list, tuple)):
                            for x in v: _flat2(x)
                        else: flat2.append(v)
                    _flat2(data)
                    nd_flat = _nd_arr2(flat2, dtype=target_dt)
                    # Infer shape from nested lists
                    if isinstance(data, (list, tuple)):
                        shape2 = []
                        cur = data
                        while isinstance(cur, (list, tuple)) and cur:
                            shape2.append(len(cur))
                            cur = cur[0]
                        shape2 = tuple(shape2)
                        if len(shape2) > 1:
                            nd_flat = nd_flat.reshape(shape2)
                    self._data = nd_flat
                except Exception:
                    from vectorphile._core.ndarray import array as _nd_fb
                    self._data = _nd_fb([float(v) if not isinstance(v, bool) else v for v in (data if isinstance(data, (list, tuple)) else [data])], dtype=target_dt)
        self.device = device
        self.requires_grad = requires_grad
        self.grad: Optional[Tensor] = None
        self.grad_fn = None
        self._is_leaf = True

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    def dim(self) -> int:
        return self._data.ndim

    def numel(self) -> int:
        return int(self._data.size)

    def element_size(self) -> int:
        return int(self._data.itemsize)

    @property
    def dtype(self):
        return str(self._data.dtype)

    def size(self, dim=None):
        """Return shape as a tuple, or a single dimension if dim is given (PyTorch-compatible)."""
        if dim is None:
            return self._data.shape
        return self._data.shape[dim]

    @property
    def T(self) -> 'Tensor':
        return self.transpose()

    @property
    def data(self):
        return self._data

    # ── creation helpers (class methods) ───────────────────────────────────

    @classmethod
    def zeros(cls, *shape, dtype='float32', device='cpu', requires_grad=False) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        from vectorphile._core.ndarray import zeros as _nd_z
        return cls(_nd_z(shape, dtype=dtype), device=device, requires_grad=requires_grad)

    @classmethod
    def ones(cls, *shape, dtype='float32', device='cpu', requires_grad=False) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        from vectorphile._core.ndarray import ones as _nd_o
        return cls(_nd_o(shape, dtype=dtype), device=device, requires_grad=requires_grad)

    @classmethod
    def eye(cls, n, m=None, dtype='float32', device='cpu') -> 'Tensor':
        from vectorphile._core.ndarray import eye as _nd_eye
        return cls(_nd_eye(n, m or n, dtype=dtype), device=device)

    @classmethod
    def arange(cls, start, stop=None, step=1, dtype='float32', device='cpu') -> 'Tensor':
        if stop is None:
            start, stop = 0, start
        from vectorphile._core.ndarray import arange as _nd_arange
        return cls(_nd_arange(start, stop, step, dtype=dtype), device=device)

    @classmethod
    def linspace(cls, start, stop, num, dtype='float32', device='cpu') -> 'Tensor':
        from vectorphile._core.ndarray import linspace as _nd_linspace
        return cls(_nd_linspace(start, stop, num, dtype=dtype), device=device)

    @classmethod
    def full(cls, shape, fill_value, dtype='float32', device='cpu') -> 'Tensor':
        from vectorphile._core.ndarray import zeros as _nd_z
        if isinstance(shape, int):
            shape = (shape,)
        n = 1
        for s in shape: n *= s
        from vectorphile._core.ndarray import array as _nd_full
        return cls(_nd_full([fill_value] * n, dtype=dtype).reshape(shape), device=device)

    @classmethod
    def empty(cls, *shape, dtype='float32', device='cpu') -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        from vectorphile._core.ndarray import zeros as _nd_z
        return cls(_nd_z(shape, dtype=dtype), device=device)

    @classmethod
    def rand(cls, *shape, device='cpu', requires_grad=False) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        try:
            from vectorphile._core.ops import rand as _nd_rand
            return cls(_nd_rand(*shape), device=device, requires_grad=requires_grad, _no_copy=True)
        except Exception:
            import random as _rnd_r; n = 1
            for s in shape: n *= s
            from vectorphile._core.ndarray import array as _nd_rf
            return cls(_nd_rf([_rnd_r.random() for _ in range(n)], dtype='float32').reshape(shape), device=device, requires_grad=requires_grad)

    @classmethod
    def randn(cls, *shape, device='cpu', requires_grad=False) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        try:
            from vectorphile._core.ops import randn as _nd_randn
            return cls(_nd_randn(*shape), device=device, requires_grad=requires_grad, _no_copy=True)
        except Exception:
            import random as _rnd_rn, math as _m_rn; n = 1
            for s in shape: n *= s
            flat_rn = [_math_t.sqrt(-2*_m_rn.log(_rnd_rn.random() or 1e-10))*_m_rn.cos(2*_m_rn.pi*_rnd_rn.random()) for _ in range(n)]
            from vectorphile._core.ndarray import array as _nd_rn
            return cls(_nd_rn(flat_rn, dtype='float32').reshape(shape), device=device, requires_grad=requires_grad)

    @classmethod
    def randint(cls, low, high, shape, dtype='int64', device='cpu') -> 'Tensor':
        import random as _rnd
        if isinstance(shape, int):
            shape = (shape,)
        n = 1
        for s in shape: n *= s
        from vectorphile._core.ndarray import array as _nd_ri
        return cls(_nd_ri([_rnd.randint(low, high - 1) for _ in range(n)], dtype=dtype).reshape(shape), device=device)

    @classmethod
    def from_numpy(cls, arr) -> 'Tensor':
        from vectorphile._core.ndarray import NDArray as _NDA_fn
        if isinstance(arr, _NDA_fn):
            return cls(arr)
        # numpy array path — convert via flat list
        flat = arr.ravel().tolist() if hasattr(arr, 'ravel') else list(arr)
        sh = tuple(arr.shape) if hasattr(arr, 'shape') else (len(flat),)
        dt = 'float64' if (hasattr(arr, 'dtype') and 'float64' in str(arr.dtype)) else 'float32'
        from vectorphile._core.ndarray import array as _nd_fn
        return cls(_nd_fn(flat, dtype=dt).reshape(sh))

    # ── type casting ────────────────────────────────────────────────────────

    def float(self) -> 'Tensor':
        return Tensor(self._data.astype('float32'), device=self.device)

    def double(self) -> 'Tensor':
        return Tensor(self._data.astype('float64'), device=self.device)

    def half(self) -> 'Tensor':
        return Tensor(self._data.astype('float16'), device=self.device)

    def int(self) -> 'Tensor':
        return Tensor(self._data.astype('int32'), device=self.device)

    def long(self) -> 'Tensor':
        return Tensor(self._data.astype('int64'), device=self.device)

    def bool(self) -> 'Tensor':
        return Tensor(self._data.astype('bool'), device=self.device)

    def short(self) -> 'Tensor':
        return Tensor(self._data.astype('int16'), device=self.device)

    def byte(self) -> 'Tensor':
        return Tensor(self._data.astype('uint8'), device=self.device)

    def char(self) -> 'Tensor':
        return Tensor(self._data.astype('int8'), device=self.device)

    def storage(self):
        """Return storage (simplified: return self for compatibility)."""
        return self

    def data_ptr(self) -> int:
        """Return data pointer (simplified: return id for compatibility)."""
        return id(self._data)

    def bernoulli(self, p: float = 0.5) -> 'Tensor':
        """Sample Bernoulli: return new tensor with Bernoulli draws."""
        import random as _rnd
        from vectorphile._core.ndarray import array as _nd_be
        flat = [1.0 if _rnd.random() < float(p) else 0.0 for _ in range(self.numel())]
        nd = _nd_be(flat, dtype='float32')
        if len(self.shape) > 1:
            nd = nd.reshape(self.shape)
        return Tensor(nd, device=self.device)

    def to(self, device=None, dtype=None) -> 'Tensor':
        # Accept a Tensor as first arg: match its dtype and device
        if isinstance(device, Tensor):
            other = device
            dtype = other.dtype
            device = other.device
        # Accept a dtype string as first positional arg (no device specified)
        elif isinstance(device, str) and device in ('float32', 'float64', 'float16', 'bfloat16',
                                                      'int32', 'int64', 'int8', 'int16', 'uint8', 'bool'):
            dtype = device
            device = None
        data = self._data
        if dtype is not None:
            data = data.astype(_DTYPE_MAP.get(dtype, dtype))
        t = Tensor(data, device=device or self.device, requires_grad=self.requires_grad)
        if (device or self.device) == 'cuda':
            try:
                return t.cuda()
            except Exception:
                pass
        return t

    def cuda(self) -> 'Tensor':
        """Move tensor to GPU, returning a new Tensor with device='cuda'."""
        if self.device == 'cuda':
            return self
        from vectorphile._core.cuda import to_device as _to_dev, _CUDA_OK
        if not _CUDA_OK:
            raise RuntimeError(
                'CUDA not available. Install CUDA Toolkit and build scripts/build_vp_cuda_win.bat')
        from vectorphile._core.ndarray import NDArray as _NDA, _c_strides as _cs, _DType
        import array as _a
        xd = self._data
        # Ensure we have a flat float32 CPU buffer
        if isinstance(xd, _NDA) and xd._buf is not None and str(xd.dtype) == 'float32':
            cpu_buf = xd._buf
            shape = xd.shape
        else:
            flat_list = xd.to_flat_list() if hasattr(xd, 'to_flat_list') else list(xd)
            cpu_buf = _a.array('f', [float(v) for v in flat_list])
            shape = self.shape
        cuda_buf = _to_dev(cpu_buf)
        # Build a placeholder NDArray with _cuda_buf set, _buf = None
        nd = _NDA.__new__(_NDA)
        nd._buf = None
        nd._np_buf = None
        nd._cuda_buf = cuda_buf
        nd.shape = shape
        nd.strides = _cs(shape)
        nd.dtype = _DType('float32')
        nd.offset = 0
        nd._owns_buf = False
        t = Tensor.__new__(Tensor)
        t._data = nd
        t.device = 'cuda'
        t.requires_grad = self.requires_grad
        t.grad = None
        t.grad_fn = None
        t._is_leaf = True
        t._retain_grad = False
        return t

    def cpu(self) -> 'Tensor':
        """Move tensor to CPU, returning a new Tensor with device='cpu'."""
        if self.device != 'cuda':
            return self
        from vectorphile._core.cuda import to_host as _to_host
        from vectorphile._core.ndarray import NDArray as _NDA, _c_strides as _cs, _DType
        xd = self._data
        cuda_buf = getattr(xd, '_cuda_buf', None)
        if cuda_buf is None:
            raise RuntimeError('cuda() tensor has no _cuda_buf')
        cpu_buf = _to_host(cuda_buf)
        nd = _NDA(cpu_buf, xd.shape, _cs(xd.shape), _DType('float32'))
        return Tensor(nd, device='cpu', requires_grad=self.requires_grad, _no_copy=True)

    def numpy(self):
        """Return data as a numpy array (or nested list if numpy unavailable)."""
        if hasattr(self._data, 'to_flat_list'):
            flat = self._data.to_flat_list()
            sh = self._data.shape
            try:
                import numpy as _np
                return _np.array(flat, dtype='float32').reshape(sh)
            except ImportError:
                pass
            if len(sh) == 1:
                return flat
            def _nest(flat, shape):
                if len(shape) == 1:
                    return flat[:shape[0]]
                size = 1
                for s in shape[1:]: size *= s
                return [_nest(flat[i*size:(i+1)*size], shape[1:]) for i in range(shape[0])]
            return _nest(flat, sh)
        try:
            import numpy as _np
            return _np.array(list(self._data), dtype='float32')
        except ImportError:
            return list(self._data)

    def item(self):
        return self._data.item()

    def tolist(self):
        return self._data.tolist()

    def __float__(self) -> float:
        return float(self.item())

    def __int__(self) -> int:
        return int(self.item())

    def __bool__(self) -> bool:
        return bool(self.item())

    def __index__(self) -> int:
        return int(self.item())

    def detach(self) -> 'Tensor':
        t = Tensor(self._data, device=self.device)
        t.requires_grad = False
        return t

    def detach_(self) -> 'Tensor':
        """In-place detach — removes from computation graph."""
        self.requires_grad = False
        self.grad_fn = None
        return self

    def clone(self) -> 'Tensor':
        return Tensor(self._data.copy(), device=self.device, requires_grad=self.requires_grad)

    def requires_grad_(self, requires_grad: bool = True) -> 'Tensor':
        """Set requires_grad in-place and return self."""
        self.requires_grad = requires_grad
        return self

    def retain_grad(self) -> None:
        """Mark this non-leaf tensor to retain its gradient after backward()."""
        self._retain_grad = True

    # ── shape ops ───────────────────────────────────────────────────────────

    def reshape(self, *shape) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        rg = _grad_enabled() and self.requires_grad
        result = Tensor(self._data.reshape(shape), device=self.device, requires_grad=rg)
        if rg:
            _inp = self; _orig_shape = self._data.shape
            class _ReshapeBackward:
                def __init__(s): s.inputs = (_inp,)
                def backward(s, grad): return (Tensor(grad._data.reshape(_orig_shape)),)
            result.grad_fn = _ReshapeBackward(); result._is_leaf = False
        return result

    def view(self, *shape) -> 'Tensor':
        return self.reshape(*shape)

    def transpose(self, dim0=None, dim1=None) -> 'Tensor':
        rg = _grad_enabled() and self.requires_grad
        if dim0 is None and dim1 is None:
            result = Tensor(self._data.T, device=self.device, requires_grad=rg)
            if rg:
                _inp = self
                class _TBackward:
                    def __init__(s): s.inputs = (_inp,)
                    def backward(s, grad): return (Tensor(grad._data.T),)
                result.grad_fn = _TBackward(); result._is_leaf = False
            return result
        axes = list(range(self.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        result = Tensor(self._data.transpose(axes), device=self.device, requires_grad=rg)
        if rg:
            _axes = axes; _inp = self
            class _TransposeBackward:
                def __init__(s): s.inputs = (_inp,)
                def backward(s, grad):
                    inv = [0] * len(_axes)
                    for i, a in enumerate(_axes): inv[a] = i
                    return (Tensor(grad._data.transpose(inv)),)
            result.grad_fn = _TransposeBackward(); result._is_leaf = False
        return result

    def permute(self, *dims) -> 'Tensor':
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = dims[0]
        rg = _grad_enabled() and self.requires_grad
        result = Tensor(self._data.transpose(dims), device=self.device, requires_grad=rg)
        if rg:
            _inp = self; _dims = tuple(dims)
            class _PermuteBackward:
                def __init__(s): s.inputs = (_inp,)
                def backward(s, grad):
                    inv = [0] * len(_dims)
                    for i, d in enumerate(_dims): inv[d] = i
                    return (Tensor(grad._data.transpose(inv)),)
            result.grad_fn = _PermuteBackward(); result._is_leaf = False
        return result

    def squeeze(self, dim=None) -> 'Tensor':
        rg = _grad_enabled() and self.requires_grad
        orig_shape = self._data.shape
        if dim is None:
            new_shape = tuple(s for s in orig_shape if s != 1)
        else:
            new_shape = tuple(s for i, s in enumerate(orig_shape) if not (i == dim and s == 1))
        out_data = self._data.reshape(new_shape)
        result = Tensor(out_data, device=self.device, requires_grad=rg)
        if rg:
            _inp = self; _orig_shape = orig_shape
            class _SqueezeBackward:
                def __init__(s): s.inputs = (_inp,)
                def backward(s, grad):
                    return (Tensor(grad._data.reshape(_orig_shape)),)
            result.grad_fn = _SqueezeBackward(); result._is_leaf = False
        return result

    def unsqueeze(self, dim) -> 'Tensor':
        rg = _grad_enabled() and self.requires_grad
        sh = list(self._data.shape)
        if dim < 0:
            dim = len(sh) + 1 + dim
        sh.insert(dim, 1)
        result = Tensor(self._data.reshape(tuple(sh)), device=self.device, requires_grad=rg)
        if rg:
            _inp = self; _dim = dim; _orig_shape = self._data.shape
            class _UnsqueezeBackward:
                def __init__(s): s.inputs = (_inp,)
                def backward(s, grad):
                    return (Tensor(grad._data.reshape(_orig_shape)),)
            result.grad_fn = _UnsqueezeBackward(); result._is_leaf = False
        return result

    def flatten(self, start_dim=0, end_dim=-1) -> 'Tensor':
        shape = self._data.shape
        if end_dim < 0:
            end_dim = len(shape) + end_dim
        new_shape = shape[:start_dim] + (-1,) + shape[end_dim+1:]
        rg = _grad_enabled() and self.requires_grad
        result = Tensor(self._data.reshape(new_shape), device=self.device, requires_grad=rg)
        if rg:
            _inp = self; _orig_shape = shape
            class _FlattenBackward:
                def __init__(s): s.inputs = (_inp,)
                def backward(s, grad): return (Tensor(grad._data.reshape(_orig_shape)),)
            result.grad_fn = _FlattenBackward(); result._is_leaf = False
        return result

    def expand(self, *shape) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        from vectorphile.core.tensor import _nd_broadcast_to as _bcast
        rg = _grad_enabled() and self.requires_grad
        result = Tensor(_bcast(self._data, shape), device=self.device, requires_grad=rg)
        if rg:
            _inp = self; _orig_shape = self._data.shape
            class _ExpandBackward:
                def __init__(s): s.inputs = (_inp,)
                def backward(s, grad):
                    # Sum over broadcast dims to get back to original shape
                    g = grad._data
                    # Sum leading dims that were added
                    ndim_orig = len(_orig_shape)
                    ndim_out = len(shape)
                    for _ in range(ndim_out - ndim_orig):
                        g = g.sum(axis=0)
                    # Sum dims that were size-1 in original
                    for i, so in enumerate(_orig_shape):
                        if so == 1 and g.shape[i] != 1:
                            g = g.sum(axis=i, keepdims=True)
                    return (Tensor(g),)
            result.grad_fn = _ExpandBackward(); result._is_leaf = False
        return result

    def expand_as(self, other: 'Tensor') -> 'Tensor':
        return self.expand(other._data.shape)

    def contiguous(self) -> 'Tensor':
        return Tensor(self._data.copy() if hasattr(self._data, 'copy') else self._data, device=self.device, requires_grad=self.requires_grad)

    def scatter_(self, dim: int, index: 'Tensor', src) -> 'Tensor':
        """In-place scatter: self[index[i,...], ...] = src along `dim`.

        Modifies self in-place and returns self. Mirrors torch.Tensor.scatter_().
        """
        from vectorphile._core.ndarray import array as _nd_sc
        idx_flat = list(index._data.to_flat_list() if hasattr(index._data, 'to_flat_list') else index._data)
        self_flat = list(self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else self._data)
        sh = self._data.shape
        idx_sh = index._data.shape
        if hasattr(src, '_data'):
            src_flat = list(src._data.to_flat_list() if hasattr(src._data, 'to_flat_list') else src._data)
        elif isinstance(src, (int, float)):
            src_flat = [float(src)] * len(idx_flat)
        else:
            src_flat = list(src)
        # Compute strides for self and index
        import builtins as _bi_sc
        ndim = len(sh)
        strides_self = [1] * ndim
        for d in range(ndim-2, -1, -1):
            strides_self[d] = strides_self[d+1] * sh[d+1]
        strides_idx = [1] * ndim
        for d in range(ndim-2, -1, -1):
            strides_idx[d] = strides_idx[d+1] * idx_sh[d+1]
        total_idx = _bi_sc.reduce(lambda a, b: a*b, idx_sh, 1)
        for flat_i in range(total_idx):
            # Convert flat_i to multi-index in idx_sh
            rem = flat_i
            midx = []
            for d in range(ndim):
                midx.append(rem // strides_idx[d])
                rem = rem % strides_idx[d]
            # Replace dim-th coord with scatter index
            target_midx = midx[:]
            target_midx[dim] = int(idx_flat[flat_i])
            # Convert target multi-index to flat position in self
            pos = _bi_sc.reduce(lambda a, bc: a + bc[0]*bc[1], zip(target_midx, strides_self), 0)
            if 0 <= pos < len(self_flat):
                self_flat[pos] = src_flat[flat_i] if flat_i < len(src_flat) else 0.0
        new_nd = _nd_sc(self_flat, dtype=str(self._data.dtype) if hasattr(self._data, 'dtype') else 'float32')
        self._data = new_nd.reshape(sh)
        return self

    def index_put_(self, indices: tuple, values, accumulate: bool = False) -> 'Tensor':
        """In-place index assignment: self[indices] = values (or += if accumulate=True).

        indices: tuple of integer or slice index tensors (one per dim).
        """
        from vectorphile._core.ndarray import array as _nd_ip
        self_flat = list(self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else self._data)
        sh = self._data.shape
        import builtins as _bi_ip
        strides = [1] * len(sh)
        for d in range(len(sh)-2, -1, -1):
            strides[d] = strides[d+1] * sh[d+1]
        # Build list of (flat_pos, value) pairs
        def _get_flat(idx_t):
            if hasattr(idx_t, '_data'):
                return [int(v) for v in (idx_t._data.to_flat_list() if hasattr(idx_t._data, 'to_flat_list') else idx_t._data)]
            return [int(idx_t)]
        idx_lists = [_get_flat(i) for i in indices]
        if hasattr(values, '_data'):
            val_flat = list(values._data.to_flat_list() if hasattr(values._data, 'to_flat_list') else values._data)
        else:
            val_flat = [float(values)]
        # Zip indices
        import builtins as _bi_ip2; n = _bi_ip2.max((len(il) for il in idx_lists), default=0) if idx_lists else 0
        padded = [il if len(il) > 1 else il * n for il in idx_lists]
        for k in range(n):
            pos = _bi_ip.reduce(lambda a, bc: a + bc[0]*bc[1],
                                zip([padded[d][k] for d in range(len(sh))], strides), 0)
            v = val_flat[k] if k < len(val_flat) else val_flat[-1]
            if 0 <= pos < len(self_flat):
                if accumulate:
                    self_flat[pos] += v
                else:
                    self_flat[pos] = v
        new_nd = _nd_ip(self_flat, dtype=str(self._data.dtype) if hasattr(self._data, 'dtype') else 'float32')
        self._data = new_nd.reshape(sh)
        return self

    def to_sparse(self) -> 'SparseTensor':
        """Convert dense Tensor to COO SparseTensor."""
        from vectorphile._core.ndarray import array as _nd_sp
        if hasattr(self._data, 'to_flat_list'):
            flat = self._data.to_flat_list()
        else:
            flat = list(self._data)
        sh = self.shape; ndim = len(sh)
        strides = [1] * ndim
        for i in range(ndim - 2, -1, -1):
            strides[i] = strides[i + 1] * sh[i + 1]
        indices_per_dim = [[] for _ in range(ndim)]
        values = []
        for pos, v in enumerate(flat):
            if v != 0.0:
                rem = pos
                for d in range(ndim):
                    indices_per_dim[d].append(rem // strides[d])
                    rem = rem % strides[d]
                values.append(v)
        nnz = len(values)
        idx_flat = []
        for d in range(ndim): idx_flat.extend(indices_per_dim[d])
        if nnz == 0:
            # Empty sparse tensor: 0 non-zeros
            idx_nd = _nd_sp([0.0] * ndim, 'float32').reshape((ndim, 1))
            val_nd = _nd_sp([0.0], 'float32')
            s = SparseTensor(Tensor(idx_nd), Tensor(val_nd, requires_grad=self.requires_grad), sh)
            s._nnz_actual = 0
            return s
        idx_nd = _nd_sp(idx_flat, 'float32').reshape((ndim, nnz))
        val_nd = _nd_sp(values, 'float32')
        return SparseTensor(Tensor(idx_nd), Tensor(val_nd, requires_grad=self.requires_grad), sh)

    def chunk(self, chunks, dim=0) -> List['Tensor']:
        total = self._data.shape[dim]
        sz = (total + chunks - 1) // chunks
        offsets = list(range(0, total, sz))
        import builtins as _bi_ck; sizes = [_bi_ck.min(sz, total - o) for o in offsets]
        rg = _grad_enabled() and self.requires_grad
        results = []
        for offset, s in zip(offsets, sizes):
            sl = [slice(None)] * self.ndim
            sl[dim] = slice(offset, offset + s)
            part = self._data[tuple(sl)]
            r = Tensor(part, device=self.device, requires_grad=rg)
            if rg:
                _inp = self; _d = dim; _off = offset; _sz = s; _in_sh = self._data.shape
                class _ChunkBwd:
                    def __init__(s2, o, si): s2.inputs = (_inp,); s2.offset = o; s2.size = si
                    def backward(s2, grad):
                        from vectorphile._core.ndarray import zeros as _z2
                        full = _z2(_in_sh, dtype='float32')
                        sl2 = [slice(None)] * len(_in_sh)
                        sl2[_d] = slice(s2.offset, s2.offset + s2.size)
                        full[tuple(sl2)] = grad._data
                        return (Tensor(full),)
                r.grad_fn = _ChunkBwd(offset, s); r._is_leaf = False
            results.append(r)
        return results

    def split(self, split_size, dim=0) -> List['Tensor']:
        total = self._data.shape[dim]
        if isinstance(split_size, (list, tuple)):
            sizes = list(split_size)
        else:
            sizes = []
            remaining = total
            while remaining > 0:
                sizes.append(split_size if split_size < remaining else remaining)
                remaining -= split_size
        offsets = []
        acc = 0
        for s in sizes:
            offsets.append(acc)
            acc += s
        rg = _grad_enabled() and self.requires_grad
        results = []
        for offset, s in zip(offsets, sizes):
            sl = [slice(None)] * self.ndim
            sl[dim] = slice(offset, offset + s)
            part = self._data[tuple(sl)]
            r = Tensor(part, device=self.device, requires_grad=rg)
            if rg:
                _inp = self; _d = dim; _off = offset; _sz = s; _in_sh = self._data.shape
                class _SplitBackward:
                    def __init__(s2, o, si): s2.inputs = (_inp,); s2.offset = o; s2.size = si
                    def backward(s2, grad):
                        from vectorphile._core.ndarray import zeros as _z3
                        full = _z3(_in_sh, dtype='float32')
                        sl2 = [slice(None)] * len(_in_sh)
                        sl2[_d] = slice(s2.offset, s2.offset + s2.size)
                        full[tuple(sl2)] = grad._data
                        return (Tensor(full),)
                r.grad_fn = _SplitBackward(offset, s); r._is_leaf = False
            results.append(r)
        return results

    def repeat(self, *repeats) -> 'Tensor':
        if len(repeats) == 1 and isinstance(repeats[0], (tuple, list)):
            repeats = repeats[0]
        return Tensor(_nd_tile(self._data, repeats), device=self.device)

    def gather(self, dim: int, index: 'Tensor') -> 'Tensor':
        """Gather values along an axis specified by dim (like torch.gather)."""
        from vectorphile._core.ndarray import array as _nd_arr2
        idx_data = index._data
        in_shape = self._data.shape
        out_shape = idx_data.shape
        ndim = len(in_shape)
        dim = dim % ndim
        # compute strides for input and output
        def _strides(shape):
            st = [1] * len(shape)
            for i in range(len(shape) - 2, -1, -1):
                st[i] = st[i + 1] * shape[i + 1]
            return st
        in_st = _strides(in_shape)
        out_st = _strides(out_shape)
        src_flat = self._data.to_flat_list()
        idx_flat = idx_data.to_flat_list()
        n = len(idx_flat)
        result = [0.0] * n
        for fi in range(n):
            # decode output coords
            rem = fi
            coords = []
            for d in range(ndim):
                coords.append(rem // out_st[d])
                rem %= out_st[d]
            coords[dim] = int(idx_flat[fi])
            src_fi = 0
            for d in range(ndim): src_fi += coords[d] * in_st[d]
            result[fi] = src_flat[src_fi]
        return Tensor(_nd_arr2(result, 'float32').reshape(out_shape), device=self.device)

    def scatter_(self, dim: int, index: 'Tensor', src) -> 'Tensor':
        """In-place scatter: self[..., index[i,j,k], ...] = src[i,j,k] along dim."""
        from vectorphile._core.ndarray import array as _nd_arr2
        idx_data = index._data
        idx_flat = idx_data.to_flat_list()
        in_shape = self._data.shape
        idx_shape = idx_data.shape
        ndim = len(in_shape)
        dim = dim % ndim
        if isinstance(src, Tensor):
            src_flat = src._data.to_flat_list()
        else:
            src_flat = [float(src)] * len(idx_flat)
        def _strides(shape):
            st = [1] * len(shape)
            for i in range(len(shape) - 2, -1, -1):
                st[i] = st[i + 1] * shape[i + 1]
            return st
        in_st = _strides(in_shape)
        idx_st = _strides(idx_shape)
        self_flat = self._data.to_flat_list()
        for fi in range(len(idx_flat)):
            rem = fi
            coords = []
            for d in range(ndim):
                coords.append(rem // idx_st[d])
                rem %= idx_st[d]
            coords[dim] = int(idx_flat[fi])
            dst = 0
            for d in range(ndim): dst += coords[d] * in_st[d]
            self_flat[dst] = float(src_flat[fi])
        self._data = _nd_arr2(self_flat, 'float32').reshape(in_shape)
        return self

    def scatter(self, dim: int, index: 'Tensor', src) -> 'Tensor':
        """Out-of-place scatter: returns a copy with scatter applied."""
        result = Tensor(self._data.copy() if hasattr(self._data, 'copy') else self._data, device=self.device)
        return result.scatter_(dim, index, src)

    # ── reduction ops ───────────────────────────────────────────────────────

    def sum(self, dim=None, keepdim=False) -> 'Tensor':
        from vectorphile.autograd.engine import SumBackward
        result = Tensor(self._data.sum(axis=dim, keepdims=keepdim), device=self.device,
                        requires_grad=self.requires_grad)
        if self.requires_grad:
            result.grad_fn = SumBackward(self, dim, keepdim)
            result._is_leaf = False
        return result

    def mean(self, dim=None, keepdim=False) -> 'Tensor':
        from vectorphile.autograd.engine import MeanBackward
        result = Tensor(self._data.mean(axis=dim, keepdims=keepdim), device=self.device,
                        requires_grad=self.requires_grad)
        if self.requires_grad:
            result.grad_fn = MeanBackward(self, dim, keepdim)
            result._is_leaf = False
        return result

    def max(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(self._data.max(), device=self.device)
        vals = self._data.max(axis=dim, keepdims=keepdim)
        idxs = self._data.argmax(axis=dim)
        return Tensor(vals, device=self.device), Tensor(idxs, device=self.device)

    def min(self, dim=None, keepdim=False):
        if dim is None:
            return Tensor(self._data.min(), device=self.device)
        vals = self._data.min(axis=dim, keepdims=keepdim)
        idxs = self._data.argmin(axis=dim)
        return Tensor(vals, device=self.device), Tensor(idxs, device=self.device)

    def argmax(self, dim=None) -> 'Tensor':
        return Tensor(self._data.argmax(axis=dim), device=self.device)

    def argmin(self, dim=None) -> 'Tensor':
        return Tensor(self._data.argmin(axis=dim), device=self.device)

    def std(self, dim=None, keepdim=False, unbiased=True) -> 'Tensor':
        ddof = 1 if unbiased else 0
        return Tensor(self._data.std(axis=dim, keepdims=keepdim, ddof=ddof), device=self.device)

    def var(self, dim=None, keepdim=False, unbiased=True) -> 'Tensor':
        ddof = 1 if unbiased else 0
        return Tensor(self._data.var(axis=dim, keepdims=keepdim, ddof=ddof), device=self.device)

    def norm(self, p=2, dim=None, keepdim=False) -> 'Tensor':
        return Tensor(_nd_linalg_norm(self._data, ord=p, axis=dim, keepdims=keepdim), device=self.device)

    def prod(self, dim=None, keepdim=False) -> 'Tensor':
        return Tensor(self._data.prod(axis=dim, keepdims=keepdim), device=self.device)

    def cumsum(self, dim) -> 'Tensor':
        return Tensor(self._data.cumsum(axis=dim), device=self.device)

    def cumprod(self, dim) -> 'Tensor':
        from vectorphile._core.ndarray import array as _nd_cp
        sh = self._data.shape
        ndim = len(sh)
        if dim < 0:
            dim = ndim + dim
        flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        # Compute cumprod along dim by iterating
        import math as _mcp
        result = flat[:]
        # stride for dimension dim
        stride = 1
        for d in range(dim + 1, ndim):
            stride *= sh[d]
        # number of groups along dimensions before dim
        outer = 1
        for d in range(dim):
            outer *= sh[d]
        length = sh[dim]
        for o in range(outer):
            for s in range(stride):
                prod = 1.0
                for i in range(length):
                    lin = o * length * stride + i * stride + s
                    prod *= float(flat[lin])
                    result[lin] = prod
        nd = _nd_cp(result, dtype='float32')
        if len(sh) > 1:
            nd = nd.reshape(sh)
        return Tensor(nd, device=self.device)

    # ── element-wise math ───────────────────────────────────────────────────

    def exp(self) -> 'Tensor':
        from vectorphile.autograd.engine import ExpBackward
        from vectorphile._core.ndarray import NDArray as _NDA_exp
        xd = self._data
        rg = _grad_enabled() and self.requires_grad
        # C fast path for float32 NDArray
        if isinstance(xd, _NDA_exp) and xd._buf is not None and str(xd.dtype) == 'float32':
            import array as _arr_exp
            from vectorphile._core.ndarray import _c_strides as _cs_e, _DType as _DT_e
            try:
                from vectorphile._core import _vp_core as _C_exp
                n = 1
                for s in xd.shape: n *= s
                out_buf = _arr_exp.array('f', bytearray(4 * n))
                _C_exp.exp_f32(xd._buf, out_buf)
                from vectorphile._core.ndarray import NDArray as _NDAe
                out_nd = _NDAe(out_buf, xd.shape, _cs_e(xd.shape), _DT_e('float32'))
                result = Tensor(out_nd, device=self.device, requires_grad=rg, _no_copy=True)
                if rg:
                    result.grad_fn = ExpBackward(self, result)
                    result._is_leaf = False
                return result
            except Exception:
                pass
        flat = xd.to_flat_list() if hasattr(xd, 'to_flat_list') else list(xd)
        sh = xd.shape if hasattr(xd, 'shape') else (len(flat),)
        from vectorphile._core.ndarray import array as _nd_exp
        _dt_exp = str(xd.dtype) if hasattr(xd, 'dtype') else 'float32'
        out_nd = _nd_exp([_math_t.exp(v) for v in flat], dtype=_dt_exp).reshape(sh)
        result = Tensor(out_nd, device=self.device, requires_grad=rg)
        if rg:
            result.grad_fn = ExpBackward(self, result)
            result._is_leaf = False
        return result

    def log(self) -> 'Tensor':
        from vectorphile.autograd.engine import LogBackward
        from vectorphile._core.ndarray import NDArray as _NDA, _c_strides as _cs, _DType
        import array as _arr, math as _m
        xd = self._data
        _LOG2 = 0.6931471805599453
        if isinstance(xd, _NDA) and xd._buf is not None and str(xd.dtype) == 'float32':
            try:
                from vectorphile._core import _vp_core as _C
                n = 1
                for s in xd.shape: n *= s
                out_buf = _arr.array('f', bytearray(4 * n))
                _C.log_f32(xd._buf, out_buf)
                # log_f32 computes log(x) - log(2); correct the bias
                _C.add_scalar_f32(out_buf, _LOG2, out_buf)
                out_nd = _NDA(out_buf, xd.shape, _cs(xd.shape), _DType('float32'))
                result = Tensor(out_nd, device=self.device, requires_grad=self.requires_grad, _no_copy=True)
                if self.requires_grad:
                    result.grad_fn = LogBackward(self)
                    result._is_leaf = False
                return result
            except Exception:
                pass
        flat = xd.to_flat_list() if hasattr(xd, 'to_flat_list') else list(xd)
        sh = xd.shape if hasattr(xd, 'shape') else (len(flat),)
        from vectorphile._core.ndarray import array as _nd_log
        _dt_log = str(xd.dtype) if hasattr(xd, 'dtype') else 'float32'
        out_nd = _nd_log([_math_t.log(v if v > 1e-38 else 1e-38) for v in flat], dtype=_dt_log).reshape(sh)
        rg = _grad_enabled() and self.requires_grad
        result = Tensor(out_nd, device=self.device, requires_grad=rg)
        if rg:
            result.grad_fn = LogBackward(self)
            result._is_leaf = False
        return result

    def log2(self) -> 'Tensor':
        flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        sh = self._data.shape
        from vectorphile._core.ndarray import array as _nd_l2
        return Tensor(_nd_l2([_math_t.log2(v if v > 1e-38 else 1e-38) for v in flat], dtype='float32').reshape(sh), device=self.device)

    def log10(self) -> 'Tensor':
        flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        sh = self._data.shape
        from vectorphile._core.ndarray import array as _nd_l10
        return Tensor(_nd_l10([_math_t.log10(v if v > 1e-38 else 1e-38) for v in flat], dtype='float32').reshape(sh), device=self.device)

    def sqrt(self) -> 'Tensor':
        from vectorphile.autograd.engine import SqrtBackward
        flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        sh = self._data.shape
        from vectorphile._core.ndarray import array as _nd_sq
        _dt_sq = str(self._data.dtype) if hasattr(self._data, 'dtype') else 'float32'
        out_nd = _nd_sq([_math_t.sqrt(v if v > 0.0 else 0.0) for v in flat], dtype=_dt_sq).reshape(sh)
        result = Tensor(out_nd, device=self.device, requires_grad=_grad_enabled() and self.requires_grad)
        if result.requires_grad:
            result.grad_fn = SqrtBackward(self, result)
            result._is_leaf = False
        return result

    def abs(self) -> 'Tensor':
        rg = _grad_enabled() and self.requires_grad
        flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        sh = self._data.shape
        from vectorphile._core.ndarray import array as _nd_ab
        _dt_ab = str(self._data.dtype) if hasattr(self._data, 'dtype') else 'float32'
        out_nd = _nd_ab([v if v >= 0 else -v for v in flat], dtype=_dt_ab).reshape(sh)
        result = Tensor(out_nd, device=self.device, requires_grad=rg)
        if rg:
            _inp = self; _dt_ab2 = _dt_ab
            class _AbsBackward:
                def __init__(s): s.inputs = (_inp,)
                def backward(s, grad):
                    gf = grad._data.to_flat_list() if hasattr(grad._data, 'to_flat_list') else list(grad._data)
                    xf = _inp._data.to_flat_list() if hasattr(_inp._data, 'to_flat_list') else list(_inp._data)
                    xsh = _inp._data.shape
                    from vectorphile._core.ndarray import array as _nd_ab2
                    return (Tensor(_nd_ab2([gf[i] * (1.0 if xf[i] > 0 else (-1.0 if xf[i] < 0 else 0.0)) for i in range(len(gf))], dtype=_dt_ab2).reshape(xsh)),)
            result.grad_fn = _AbsBackward()
            result._is_leaf = False
        return result

    def sign(self) -> 'Tensor':
        flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        sh = self._data.shape
        from vectorphile._core.ndarray import array as _nd_sg
        return Tensor(_nd_sg([1.0 if v > 0 else (-1.0 if v < 0 else 0.0) for v in flat], dtype='float32').reshape(sh), device=self.device)

    def sin(self) -> 'Tensor':
        from vectorphile.autograd.engine import SinBackward
        from vectorphile._core.ndarray import NDArray as _NDA, _c_strides as _cs, _DType
        import array as _arr
        try:
            from vectorphile._core import _vp_core as _C
            xd = self._data
            if isinstance(xd, _NDA) and xd._buf is not None and str(xd.dtype) == 'float32':
                n = 1
                for s in xd.shape: n *= s
                out_buf = _arr.array('f', bytearray(4 * n))
                _C.sin_f32(xd._buf, out_buf)
                out_data = _NDA(out_buf, xd.shape, _cs(xd.shape), _DType('float32'))
                result = Tensor(out_data, device=self.device, requires_grad=self.requires_grad)
                if self.requires_grad:
                    result.grad_fn = SinBackward(self)
                    result._is_leaf = False
                return result
        except Exception:
            pass
        flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        sh = self._data.shape
        from vectorphile._core.ndarray import array as _nd_sin
        out_nd = _nd_sin([_math_t.sin(v) for v in flat], dtype='float32').reshape(sh)
        result = Tensor(out_nd, device=self.device,
                        requires_grad=_grad_enabled() and self.requires_grad)
        if result.requires_grad:
            result.grad_fn = SinBackward(self)
            result._is_leaf = False
        return result

    def cos(self) -> 'Tensor':
        from vectorphile.autograd.engine import CosBackward
        from vectorphile._core.ndarray import NDArray as _NDA, _c_strides as _cs, _DType
        import array as _arr
        try:
            from vectorphile._core import _vp_core as _C
            xd = self._data
            if isinstance(xd, _NDA) and xd._buf is not None and str(xd.dtype) == 'float32':
                n = 1
                for s in xd.shape: n *= s
                out_buf = _arr.array('f', bytearray(4 * n))
                _C.cos_f32(xd._buf, out_buf)
                out_data = _NDA(out_buf, xd.shape, _cs(xd.shape), _DType('float32'))
                result = Tensor(out_data, device=self.device, requires_grad=_grad_enabled() and self.requires_grad)
                if result.requires_grad:
                    result.grad_fn = CosBackward(self)
                    result._is_leaf = False
                return result
        except Exception:
            pass
        flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        sh = self._data.shape
        from vectorphile._core.ndarray import array as _nd_cos
        out_nd = _nd_cos([_math_t.cos(v) for v in flat], dtype='float32').reshape(sh)
        result = Tensor(out_nd, device=self.device,
                        requires_grad=_grad_enabled() and self.requires_grad)
        if result.requires_grad:
            result.grad_fn = CosBackward(self)
            result._is_leaf = False
        return result

    def tan(self) -> 'Tensor':
        flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        sh = self._data.shape
        from vectorphile._core.ndarray import array as _nd_tan
        return Tensor(_nd_tan([_math_t.tan(v) for v in flat], dtype='float32').reshape(sh), device=self.device)

    def asin(self) -> 'Tensor':
        flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        sh = self._data.shape
        from vectorphile._core.ndarray import array as _nd_as
        def _clamp1(v): v = float(v); return _math_t.asin(-1.0 if v < -1.0 else (1.0 if v > 1.0 else v))
        return Tensor(_nd_as([_clamp1(v) for v in flat], dtype='float32').reshape(sh), device=self.device)

    def acos(self) -> 'Tensor':
        flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        sh = self._data.shape
        from vectorphile._core.ndarray import array as _nd_ac
        def _clamp2(v): v = float(v); return _math_t.acos(-1.0 if v < -1.0 else (1.0 if v > 1.0 else v))
        return Tensor(_nd_ac([_clamp2(v) for v in flat], dtype='float32').reshape(sh), device=self.device)

    def atan(self) -> 'Tensor':
        flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        sh = self._data.shape
        from vectorphile._core.ndarray import array as _nd_at
        return Tensor(_nd_at([_math_t.atan(float(v)) for v in flat], dtype='float32').reshape(sh), device=self.device)

    def sinh(self) -> 'Tensor':
        flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        sh = self._data.shape
        from vectorphile._core.ndarray import array as _nd_sh
        return Tensor(_nd_sh([_math_t.sinh(float(v)) for v in flat], dtype='float32').reshape(sh), device=self.device)

    def cosh(self) -> 'Tensor':
        flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        sh = self._data.shape
        from vectorphile._core.ndarray import array as _nd_ch
        return Tensor(_nd_ch([_math_t.cosh(float(v)) for v in flat], dtype='float32').reshape(sh), device=self.device)

    def asinh(self) -> 'Tensor':
        flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        sh = self._data.shape
        from vectorphile._core.ndarray import array as _nd_ais
        return Tensor(_nd_ais([_math_t.asinh(float(v)) for v in flat], dtype='float32').reshape(sh), device=self.device)

    def acosh(self) -> 'Tensor':
        flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        sh = self._data.shape
        from vectorphile._core.ndarray import array as _nd_ach
        return Tensor(_nd_ach([_math_t.acosh(float(v) if float(v) >= 1.0 else 1.0) for v in flat], dtype='float32').reshape(sh), device=self.device)

    def atanh(self) -> 'Tensor':
        flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        sh = self._data.shape
        from vectorphile._core.ndarray import array as _nd_ath
        def _clamp3(v):
            v = float(v)
            return _math_t.atanh(-1+1e-7 if v <= -1 else (1-1e-7 if v >= 1 else v))
        return Tensor(_nd_ath([_clamp3(v) for v in flat], dtype='float32').reshape(sh), device=self.device)

    def rsqrt(self) -> 'Tensor':
        flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        sh = self._data.shape
        from vectorphile._core.ndarray import array as _nd_rs
        return Tensor(_nd_rs([1.0 / _math_t.sqrt(float(v) if float(v) >= 1e-30 else 1e-30) for v in flat], dtype='float32').reshape(sh), device=self.device)

    def tanh(self) -> 'Tensor':
        from vectorphile.autograd.engine import TanhBackward
        from vectorphile._core.ndarray import NDArray as _NDA_t
        import array as _arr_t
        xd = self._data
        if isinstance(xd, _NDA_t) and xd._buf is not None and str(xd.dtype) == 'float32':
            try:
                from vectorphile._core import _vp_core as _C_t
                from vectorphile._core.ndarray import _c_strides as _cs_t, _DType as _DT_t
                n = 1
                for s in xd.shape: n *= s
                out_buf = _arr_t.array('f', bytearray(4 * n))
                _C_t.tanh_f32(xd._buf, out_buf)
                out_nd = _NDA_t(out_buf, xd.shape, _cs_t(xd.shape), _DT_t('float32'))
                result = Tensor(out_nd, device=self.device, requires_grad=self.requires_grad, _no_copy=True)
                if self.requires_grad:
                    result.grad_fn = TanhBackward(self, result)
                    result._is_leaf = False
                return result
            except Exception:
                pass
        flat = xd.to_flat_list() if hasattr(xd, 'to_flat_list') else list(xd)
        sh = xd.shape if hasattr(xd, 'shape') else (len(flat),)
        from vectorphile._core.ndarray import array as _nd_th
        _dt_th = str(xd.dtype) if hasattr(xd, 'dtype') else 'float32'
        out_nd = _nd_th([_math_t.tanh(v) for v in flat], dtype=_dt_th).reshape(sh)
        result = Tensor(out_nd, device=self.device, requires_grad=self.requires_grad)
        if self.requires_grad:
            result.grad_fn = TanhBackward(self, result)
            result._is_leaf = False
        return result

    def sigmoid(self) -> 'Tensor':
        from vectorphile.autograd.engine import SigmoidBackward
        from vectorphile._core.ndarray import NDArray as _NDA, _c_strides as _cs, _DType
        import array as _arr
        xd = self._data
        if isinstance(xd, _NDA) and xd._buf is not None and str(xd.dtype) == 'float32':
            try:
                from vectorphile._core import _vp_core as _C
                n = 1
                for s in xd.shape: n *= s
                out_buf = _arr.array('f', bytearray(4 * n))
                _C.sigmoid_f32(xd._buf, out_buf)
                out_nd = _NDA(out_buf, xd.shape, _cs(xd.shape), _DType('float32'))
                result = Tensor(out_nd, device=self.device, requires_grad=self.requires_grad, _no_copy=True)
                if self.requires_grad:
                    result.grad_fn = SigmoidBackward(self, result)
                    result._is_leaf = False
                return result
            except Exception:
                pass
        flat = xd.to_flat_list() if hasattr(xd, 'to_flat_list') else list(xd)
        sh = xd.shape if hasattr(xd, 'shape') else (len(flat),)
        from vectorphile._core.ndarray import array as _nd_sig
        _dt_sig = str(xd.dtype) if hasattr(xd, 'dtype') else 'float32'
        out_nd = _nd_sig([1.0 / (1.0 + _math_t.exp(-v)) for v in flat], dtype=_dt_sig).reshape(sh)
        result = Tensor(out_nd, device=self.device, requires_grad=self.requires_grad)
        if self.requires_grad:
            result.grad_fn = SigmoidBackward(self, result)
            result._is_leaf = False
        return result

    def relu(self) -> 'Tensor':
        from vectorphile.autograd.engine import ReLUBackward
        from vectorphile._core.ndarray import NDArray as _NDA
        xd = self._data
        if isinstance(xd, _NDA) and xd._buf is not None and str(xd.dtype) == 'float32':
            try:
                from vectorphile._core import _vp_core as _C
                import array as _arr
                from vectorphile._core.ndarray import _c_strides as _cs, _DType
                n = 1
                for s in xd.shape: n *= s
                # Allocate fresh buffer each call — cache would alias results
                out_buf = _arr.array('f', bytearray(4 * n))
                _C.relu_f32(xd._buf, out_buf)
                out_nd = _NDA(out_buf, xd.shape, _cs(xd.shape), _DType('float32'))
                result = Tensor(out_nd, device=self.device, requires_grad=self.requires_grad, _no_copy=True)
                if self.requires_grad:
                    result.grad_fn = ReLUBackward(self)
                    result._is_leaf = False
                return result
            except Exception:
                pass
        flat = xd.to_flat_list() if hasattr(xd, 'to_flat_list') else (xd.flatten().tolist() if hasattr(xd, 'flatten') else list(xd))
        sh = xd.shape if hasattr(xd, 'shape') else (len(flat),)
        _dt_rl = str(getattr(xd, 'dtype', 'float32'))
        if _dt_rl not in ('float32', 'float64', 'int32', 'int64'): _dt_rl = 'float32'
        from vectorphile._core.ndarray import array as _nd_rl
        out_nd = _nd_rl([v if v > 0.0 else 0.0 for v in flat], dtype=_dt_rl).reshape(sh)
        result = Tensor(out_nd, device=self.device, requires_grad=self.requires_grad)
        if self.requires_grad:
            result.grad_fn = ReLUBackward(self)
            result._is_leaf = False
        return result

    def gelu(self) -> 'Tensor':
        import math as _m
        from vectorphile._core.ndarray import array as _nd_gelu
        flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        sh = self._data.shape
        _sqrt2 = _m.sqrt(2.0)
        result = [v * 0.5 * (1.0 + _m.erf(v / _sqrt2)) for v in flat]
        return Tensor(_nd_gelu(result, 'float32').reshape(sh), device=self.device,
                      requires_grad=self.requires_grad)

    def silu(self) -> 'Tensor':
        import math as _m
        from vectorphile._core.ndarray import array as _nd_silu
        flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        sh = self._data.shape
        result = [v / (1.0 + _m.exp(-v)) for v in flat]
        return Tensor(_nd_silu(result, 'float32').reshape(sh), device=self.device,
                      requires_grad=self.requires_grad)

    swish = silu

    def clamp(self, min=None, max=None) -> 'Tensor':
        from vectorphile._core.ndarray import NDArray as _NDA, _c_strides as _cs, _DType
        import array as _arr
        xd = self._data
        rg = _grad_enabled() and self.requires_grad
        if isinstance(xd, _NDA) and xd._buf is not None and str(xd.dtype) == 'float32':
            lo = float(min) if min is not None else -1e38
            hi = float(max) if max is not None else 1e38
            out_buf = _arr.array('f', [lo if v < lo else (hi if v > hi else v) for v in xd._buf])
            out_nd = _NDA(out_buf, xd.shape, _cs(xd.shape), _DType('float32'))
            result = Tensor(out_nd, device=self.device, requires_grad=rg, _no_copy=True)
        else:
            flat_cl = xd.to_flat_list() if hasattr(xd, 'to_flat_list') else list(xd)
            sh_cl = xd.shape if hasattr(xd, 'shape') else (len(flat_cl),)
            lo_cl = float(min) if min is not None else -1e38
            hi_cl = float(max) if max is not None else 1e38
            from vectorphile._core.ndarray import array as _nd_cl
            result = Tensor(_nd_cl([lo_cl if v < lo_cl else (hi_cl if v > hi_cl else v) for v in flat_cl], dtype='float32').reshape(sh_cl), device=self.device, requires_grad=rg)
        if rg:
            _inp = self; _lo = min; _hi = max
            class _ClampBackward:
                def __init__(s): s.inputs = (_inp,)
                def backward(s, grad):
                    from vectorphile._core.ndarray import NDArray as _N2, _c_strides as _c2, _DType as _DT2
                    import array as _a2
                    gd = grad._data; ixd = _inp._data
                    if isinstance(gd, _N2) and gd._buf is not None and isinstance(ixd, _N2) and ixd._buf is not None:
                        lo2 = float(_lo) if _lo is not None else -1e38
                        hi2 = float(_hi) if _hi is not None else 1e38
                        m = _a2.array('f', [float(g) if lo2 < float(x) < hi2 else 0.0 for g, x in zip(gd._buf, ixd._buf)])
                        return (Tensor(_N2(m, gd.shape, _c2(gd.shape), _DT2('float32')), _no_copy=True),)
                    gf2 = grad._data.to_flat_list() if hasattr(grad._data, 'to_flat_list') else list(grad._data)
                    xf2 = _inp._data.to_flat_list() if hasattr(_inp._data, 'to_flat_list') else list(_inp._data)
                    xsh2 = _inp._data.shape
                    lo2b = float(_lo) if _lo is not None else -1e38
                    hi2b = float(_hi) if _hi is not None else 1e38
                    from vectorphile._core.ndarray import array as _nd_clb
                    return (Tensor(_nd_clb([gf2[i] if lo2b < xf2[i] < hi2b else 0.0 for i in range(len(gf2))], dtype='float32').reshape(xsh2)),)
            result.grad_fn = _ClampBackward()
            result._is_leaf = False
        return result

    def clip(self, min=None, max=None) -> 'Tensor':
        return self.clamp(min, max)

    def floor(self) -> 'Tensor':
        flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        sh = self._data.shape
        from vectorphile._core.ndarray import array as _nd_fl
        return Tensor(_nd_fl([float(_math_t.floor(v)) for v in flat], dtype='float32').reshape(sh), device=self.device)

    def ceil(self) -> 'Tensor':
        flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        sh = self._data.shape
        from vectorphile._core.ndarray import array as _nd_ce
        return Tensor(_nd_ce([float(_math_t.ceil(v)) for v in flat], dtype='float32').reshape(sh), device=self.device)

    def round(self, decimals=0) -> 'Tensor':
        flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        sh = self._data.shape
        from vectorphile._core.ndarray import array as _nd_ro
        import builtins
        return Tensor(_nd_ro([builtins.round(v, decimals) for v in flat], dtype='float32').reshape(sh), device=self.device)

    def pow(self, exponent) -> 'Tensor':
        return self.__pow__(exponent)

    # ── matrix ops ──────────────────────────────────────────────────────────

    def matmul(self, other: 'Tensor') -> 'Tensor':
        return self.__matmul__(other)

    def mm(self, other: 'Tensor') -> 'Tensor':
        return self.__matmul__(other)

    def bmm(self, other: 'Tensor') -> 'Tensor':
        return self.__matmul__(other)

    def dot(self, other: 'Tensor') -> 'Tensor':
        return self.__matmul__(other)

    # ── comparison ops ──────────────────────────────────────────────────────

    def eq(self, other) -> 'Tensor':
        o = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data == o, device=self.device)

    def ne(self, other) -> 'Tensor':
        o = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data != o, device=self.device)

    def lt(self, other) -> 'Tensor':
        o = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data < o, device=self.device)

    def le(self, other) -> 'Tensor':
        o = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data <= o, device=self.device)

    def gt(self, other) -> 'Tensor':
        o = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data > o, device=self.device)

    def ge(self, other) -> 'Tensor':
        o = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data >= o, device=self.device)

    def all(self, dim=None) -> 'Tensor':
        return Tensor(self._data.all(axis=dim), device=self.device)

    def any(self, dim=None) -> 'Tensor':
        return Tensor(self._data.any(axis=dim), device=self.device)

    def isnan(self) -> 'Tensor':
        return isnan(self)

    def isinf(self) -> 'Tensor':
        return isinf(self)

    def isfinite(self) -> 'Tensor':
        return isfinite(self)

    # ── operators ───────────────────────────────────────────────────────────

    def _coerce(self, other):
        return other._data if isinstance(other, Tensor) else other

    def _safe_op(self, other_data, op_char):
        """Execute op using NDArray ops."""
        a, b = self._data, other_data
        if op_char == '+': return a + b
        if op_char == '-': return a - b
        if op_char == '*': return a * b
        if op_char == '/': return a / b
        return a + b

    def __add__(self, other) -> 'Tensor':
        from vectorphile.autograd.engine import AddBackward
        rg = _grad_enabled() and (self.requires_grad or (isinstance(other, Tensor) and other.requires_grad))
        result = Tensor(self._safe_op(self._coerce(other), '+'), device=self.device, requires_grad=rg)
        if rg:
            result.grad_fn = AddBackward(self, other)
            result._is_leaf = False
        return result

    def add(self, other) -> 'Tensor':
        return self.__add__(other)

    def __radd__(self, other) -> 'Tensor':
        return self.__add__(other)

    def __sub__(self, other) -> 'Tensor':
        from vectorphile.autograd.engine import SubBackward
        rg = _grad_enabled() and (self.requires_grad or (isinstance(other, Tensor) and other.requires_grad))
        result = Tensor(self._safe_op(self._coerce(other), '-'), device=self.device, requires_grad=rg)
        if rg:
            result.grad_fn = SubBackward(self, other)
            result._is_leaf = False
        return result

    def __rsub__(self, other) -> 'Tensor':
        if isinstance(other, (int, float)):
            flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
            sh = self._data.shape
            _dt_rs = str(getattr(self._data, 'dtype', 'float32'))
            if _dt_rs not in ('float32', 'float64', 'int32', 'int64'): _dt_rs = 'float32'
            from vectorphile._core.ndarray import array as _nd_rsub
            rg = _grad_enabled() and self.requires_grad
            result = Tensor(_nd_rsub([other - v for v in flat], dtype=_dt_rs).reshape(sh), device=self.device, requires_grad=rg)
            if rg:
                from vectorphile.autograd.engine import SubBackward
                _scalar_t = Tensor(_nd_rsub([other], dtype=_dt_rs))
                result.grad_fn = SubBackward(_scalar_t, self); result._is_leaf = False
            return result
        return Tensor(other._data - self._data if isinstance(other, Tensor) else other - self._data, device=self.device)

    def __mul__(self, other) -> 'Tensor':
        from vectorphile.autograd.engine import MulBackward
        rg = _grad_enabled() and (self.requires_grad or (isinstance(other, Tensor) and other.requires_grad))
        result = Tensor(self._safe_op(self._coerce(other), '*'), device=self.device, requires_grad=rg)
        if rg:
            result.grad_fn = MulBackward(self, other)
            result._is_leaf = False
        return result

    def __rmul__(self, other) -> 'Tensor':
        return self.__mul__(other)

    def __truediv__(self, other) -> 'Tensor':
        from vectorphile.autograd.engine import DivBackward
        rg = _grad_enabled() and (self.requires_grad or (isinstance(other, Tensor) and other.requires_grad))
        result = Tensor(self._safe_op(self._coerce(other), '/'), device=self.device, requires_grad=rg)
        if rg:
            result.grad_fn = DivBackward(self, other)
            result._is_leaf = False
        return result

    def __rtruediv__(self, other) -> 'Tensor':
        return Tensor(self._coerce(other) / self._data, device=self.device)

    def __pow__(self, exponent) -> 'Tensor':
        exp = self._coerce(exponent)
        rg = _grad_enabled() and self.requires_grad
        from vectorphile._core.ndarray import NDArray as _NDA_pow
        d = self._data
        if isinstance(d, _NDA_pow):
            flat = d.to_flat_list()
            sh = d.shape
            from vectorphile._core.ndarray import array as _nd_pow
            result_flat = [v ** float(exp) for v in flat]
            _dt = str(d.dtype) if hasattr(d, 'dtype') else 'float32'
            result = Tensor(_nd_pow(result_flat, dtype=_dt).reshape(sh), device=self.device, requires_grad=rg)
        else:
            result = Tensor([v ** float(exp) for v in (d if hasattr(d, '__iter__') else [d])], device=self.device, requires_grad=rg)
        if rg:
            from vectorphile.autograd.engine import Function
            class _PowBackward(Function):
                def __init__(s, inp, e): super().__init__(inp); s.inp = inp; s.e = e
                def backward(s, grad):
                    from vectorphile.core.tensor import Tensor as _T
                    from vectorphile._core.ndarray import NDArray as _NDAp, array as _ndp
                    d2 = s.inp._data
                    if isinstance(d2, _NDAp):
                        base_f = d2.to_flat_list()
                        sh2 = d2.shape
                        _dt2 = str(d2.dtype) if hasattr(d2, 'dtype') else 'float32'
                        ge_f = [s.e * (v ** (s.e - 1)) for v in base_f]
                        gd_f = grad._data.to_flat_list() if isinstance(grad._data, _NDAp) else list(grad._data)
                        res_f = [ge_f[i] * gd_f[i] for i in range(len(ge_f))]
                        return (_T(_ndp(res_f, dtype=_dt2).reshape(sh2)),)
                    return (_T(grad._data),)
            result.grad_fn = _PowBackward(self, float(exp))
            result._is_leaf = False
        return result

    def __neg__(self) -> 'Tensor':
        rg = _grad_enabled() and self.requires_grad
        from vectorphile._core.ndarray import NDArray as _NDA_N, _c_strides as _cs_N, _DType as _DT_N
        import array as _arr_N
        def _neg_nd(d):
            if isinstance(d, _NDA_N) and d._buf is not None and str(getattr(d, 'dtype', '')) == 'float32':
                try:
                    from vectorphile._core import _vp_core as _CN
                    n = 1
                    for s in d.shape: n *= s
                    out = _arr_N.array('f', bytearray(4 * n))
                    _CN.mul_scalar_f32(d._buf, -1.0, out)
                    return _NDA_N(out, d.shape, _cs_N(d.shape), _DT_N('float32'))
                except Exception:
                    pass
            if isinstance(d, _NDA_N) and d._buf is not None:
                flat_n = d.to_flat_list()
                _dt_neg = str(getattr(d, 'dtype', 'float32'))
                from vectorphile._core.ndarray import array as _nd_neg2
                out_neg = _nd_neg2([-v for v in flat_n], dtype=_dt_neg).reshape(d.shape)
                return out_neg
            flat_n = d.to_flat_list() if hasattr(d, 'to_flat_list') else list(d)
            sh_n = d.shape if hasattr(d, 'shape') else (len(flat_n),)
            from vectorphile._core.ndarray import array as _nd_neg3
            return _nd_neg3([-v for v in flat_n], dtype='float32').reshape(sh_n)
        result = Tensor(_neg_nd(self._data), device=self.device, requires_grad=rg)
        if rg:
            _inp = self
            class _NegBackward:
                def __init__(s): s.inputs = (_inp,)
                def backward(s, grad):
                    return (Tensor(_neg_nd(grad._data)),)
            result.grad_fn = _NegBackward()
            result._is_leaf = False
        return result

    def __abs__(self) -> 'Tensor':
        return self.abs()

    # ── In-place operators (break autograd graph, used by optimizer.zero_grad) ──
    def __iadd__(self, other) -> 'Tensor':
        # In-place add; breaks grad tracking (safe for leaf param updates)
        self._data = self._safe_op(self._coerce(other), '+')
        self.grad_fn = None
        return self

    def __isub__(self, other) -> 'Tensor':
        self._data = self._safe_op(self._coerce(other), '-')
        self.grad_fn = None
        return self

    def __imul__(self, other) -> 'Tensor':
        self._data = self._safe_op(self._coerce(other), '*')
        self.grad_fn = None
        return self

    def __itruediv__(self, other) -> 'Tensor':
        self._data = self._safe_op(self._coerce(other), '/')
        self.grad_fn = None
        return self

    def add_(self, other, alpha: float = 1.0) -> 'Tensor':
        """In-place add with optional scale: self += alpha * other."""
        if alpha != 1.0:
            other = other * alpha if isinstance(other, Tensor) else other * alpha
        self._data = self._safe_op(self._coerce(other), '+')
        self.grad_fn = None
        return self

    def sub_(self, other, alpha: float = 1.0) -> 'Tensor':
        """In-place subtract: self -= alpha * other."""
        if alpha != 1.0:
            other = other * alpha if isinstance(other, Tensor) else other * alpha
        self._data = self._safe_op(self._coerce(other), '-')
        self.grad_fn = None
        return self

    def mul_(self, other) -> 'Tensor':
        """In-place multiply: self *= other."""
        self._data = self._safe_op(self._coerce(other), '*')
        self.grad_fn = None
        return self

    def div_(self, other) -> 'Tensor':
        """In-place divide: self /= other."""
        self._data = self._safe_op(self._coerce(other), '/')
        self.grad_fn = None
        return self

    def clamp_(self, min=None, max=None) -> 'Tensor':
        """In-place clamp."""
        result = self.clamp(min=min, max=max)
        self._data = result._data
        self.grad_fn = None
        return self

    def relu_(self) -> 'Tensor':
        """In-place ReLU."""
        result = self.relu()
        self._data = result._data
        self.grad_fn = None
        return self

    def sigmoid_(self) -> 'Tensor':
        """In-place sigmoid."""
        result = self.sigmoid()
        self._data = result._data
        self.grad_fn = None
        return self

    def tanh_(self) -> 'Tensor':
        """In-place tanh."""
        result = self.tanh()
        self._data = result._data
        self.grad_fn = None
        return self

    def zero_(self) -> 'Tensor':
        """In-place zero: self = 0."""
        from vectorphile._core.ndarray import NDArray as _NDA_z, zeros as _nd_zeros_z
        if isinstance(self._data, _NDA_z):
            dt = str(self._data.dtype) if hasattr(self._data, 'dtype') else 'float32'
            self._data = _nd_zeros_z(self._data.shape, dtype=dt)
        else:
            self._data = type(self._data)([0] * len(self._data))
        self.grad_fn = None
        return self

    def fill_(self, value: float) -> 'Tensor':
        """In-place fill: self = value."""
        from vectorphile._core.ndarray import NDArray as _NDA_f, full as _nd_full_f
        if isinstance(self._data, _NDA_f):
            dt = str(self._data.dtype) if hasattr(self._data, 'dtype') else 'float32'
            self._data = _nd_full_f(self._data.shape, value, dtype=dt)
        self.grad_fn = None
        return self

    def __matmul__(self, other) -> 'Tensor':
        from vectorphile.autograd.engine import MatMulBackward
        from vectorphile._core.ndarray import NDArray as _NDA, _c_strides as _cs, _DType as _DT
        import array as _arr
        rg = _grad_enabled() and (self.requires_grad or other.requires_grad)
        a, b = self._data, other._data
        try:
            from vectorphile._core import _vp_core as _C
            if (isinstance(a, _NDA) and isinstance(b, _NDA)
                    and a._buf is not None and b._buf is not None
                    and str(a.dtype) == 'float32' and str(b.dtype) == 'float32'):
                # Support 2D and batched (ND) matmul via loop over batch dims
                ashape, bshape = a.shape, b.shape
                if len(ashape) >= 2 and len(bshape) >= 2:
                    M, K = ashape[-2], ashape[-1]
                    K2, N = bshape[-2], bshape[-1]
                    if K == K2:
                        # Batch dims
                        batch_a = ashape[:-2]
                        batch_b = bshape[:-2]
                        # Broadcast batch dims
                        if not batch_a:
                            batch_a = (1,) * len(batch_b)
                        if not batch_b:
                            batch_b = (1,) * len(batch_a)
                        # For now handle same batch or one side scalar batch
                        import math as _mth
                        batch = tuple(x if x > y else y for x, y in zip(batch_a, batch_b))
                        total_batch = 1
                        for d in batch: total_batch *= d
                        out_shape = batch + (M, N)
                        out_buf = _arr.array('f', bytearray(4 * total_batch * M * N))
                        # Make contiguous so strides are C-order (handles transposed weights)
                        ca_nd = a.contiguous(); cb_nd = b.contiguous()
                        a_flat = ca_nd._buf; b_flat = cb_nd._buf
                        a_stride = M * K; b_stride = K * N
                        for bi in range(total_batch):
                            # Use modulo so broadcast dimensions (size 1) always map to index 0
                            ai = bi % batch_a[0] if batch_a else 0
                            bi2 = bi % batch_b[0] if batch_b else 0
                            # Create sub-buffers for this batch element
                            a_sub = _arr.array('f', a_flat[ai*a_stride:(ai+1)*a_stride])
                            b_sub = _arr.array('f', b_flat[bi2*b_stride:(bi2+1)*b_stride])
                            o_sub = _arr.array('f', bytearray(4 * M * N))
                            _C.matmul_f32(a_sub, b_sub, o_sub, M, K, N)
                            out_buf[bi*M*N:(bi+1)*M*N] = o_sub
                        out_nd = _NDA(out_buf, out_shape, _cs(out_shape), _DT('float32'))
                        result = Tensor(out_nd, device=self.device, requires_grad=rg)
                        if rg:
                            result.grad_fn = MatMulBackward(self, other)
                            result._is_leaf = False
                        return result
        except Exception:
            pass
        result = Tensor(a @ b, device=self.device, requires_grad=rg)
        if rg:
            result.grad_fn = MatMulBackward(self, other)
            result._is_leaf = False
        return result

    def __eq__(self, other) -> 'Tensor':
        return self.eq(other)

    def __ne__(self, other) -> 'Tensor':
        return self.ne(other)

    def __lt__(self, other) -> 'Tensor':
        return self.lt(other)

    def __le__(self, other) -> 'Tensor':
        return self.le(other)

    def __gt__(self, other) -> 'Tensor':
        return self.gt(other)

    def __ge__(self, other) -> 'Tensor':
        return self.ge(other)

    def __getitem__(self, idx):
        if isinstance(idx, Tensor):
            idx = idx._data
        rg = _grad_enabled() and self.requires_grad
        result = Tensor(self._data[idx], device=self.device, requires_grad=rg)
        if rg:
            _inp = self; _idx = idx; _in_shape = self._data.shape
            class _IndexBackward:
                def __init__(s): s.inputs = (_inp,)
                def backward(s, grad):
                    from vectorphile._core.ndarray import zeros as _z_idx, NDArray as _NDA_ib
                    g = _z_idx(_in_shape, dtype='float32')
                    gd = grad._data if isinstance(grad, Tensor) else grad
                    gf = gd.to_flat_list() if hasattr(gd, 'to_flat_list') else list(gd)

                    # Case 1: single integer index → one row
                    if isinstance(_idx, int):
                        sl = [slice(None)] * len(_in_shape)
                        sl[0] = _idx
                        try: g[tuple(sl)] = gd
                        except Exception: pass

                    # Case 2: tuple of slices (e.g. [:, :-1]) → direct slice assignment
                    elif isinstance(_idx, tuple) and any(isinstance(x, slice) for x in _idx):
                        try: g[_idx] = gd
                        except Exception:
                            # Fallback: copy element by element via flat lists
                            gf2 = gd.to_flat_list() if hasattr(gd, 'to_flat_list') else list(gd)
                            g_flat = g.to_flat_list() if hasattr(g, 'to_flat_list') else list(g)
                            # Determine slice extent along each axis
                            import builtins
                            indices = [range(*s.indices(n)) if isinstance(s, slice) else [s]
                                       for s, n in zip(_idx, _in_shape)]
                            flat_i = 0
                            from itertools import product as _prod
                            for coords in _prod(*indices):
                                import builtins as _bi_fg; flat_g = _bi_fg.sum(coords[d] * (1 if d == len(_in_shape)-1
                                             else int(__import__('math').prod(_in_shape[d+1:])))
                                             for d in range(len(_in_shape)))
                                if flat_i < len(gf2):
                                    g_flat[flat_g] = gf2[flat_i]
                                flat_i += 1

                    # Case 3: integer-array indexing (embedding backward) → scatter-add
                    elif hasattr(_idx, '__iter__') or isinstance(_idx, _NDA_ib):
                        idx_flat = _idx.to_flat_list() if hasattr(_idx, 'to_flat_list') else list(_idx)
                        D = _in_shape[1] if len(_in_shape) > 1 else 1
                        gf_rows = len(gf) // D if D > 0 else 0
                        try:
                            for i, row_idx in enumerate(idx_flat):
                                ri = int(row_idx)
                                sl = [slice(None)] * len(_in_shape)
                                sl[0] = ri
                                tsl = tuple(sl)
                                if i < gf_rows:
                                    row_grad = gd[i] if hasattr(gd, '__getitem__') else gd
                                    try:
                                        g[tsl] = g[tsl] + row_grad
                                    except Exception:
                                        pass
                        except Exception:
                            try: g[_idx] = gd
                            except Exception: pass

                    # Case 4: slice on first axis
                    elif isinstance(_idx, slice):
                        try: g[_idx] = gd
                        except Exception: pass

                    return (Tensor(g),)
            result.grad_fn = _IndexBackward(); result._is_leaf = False
        return result

    def __setitem__(self, idx, value):
        if isinstance(idx, Tensor):
            idx = idx._data
        self._data[idx] = value._data if isinstance(value, Tensor) else value

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        _float_dtypes = {'float32', 'float64', 'float16', 'bfloat16'}
        is_fp = self.dtype in _float_dtypes
        is_bool = 'bool' in self.dtype

        def _fmt_val(v):
            if is_bool:
                return 'True' if v else 'False'
            if is_fp:
                s = f"{v:.4f}".rstrip('0')
                return s if s.endswith('.') else s  # keep trailing dot: 1. → "1."
            return str(int(v)) if isinstance(v, float) else str(v)

        def _fmt_array(arr, depth=0):
            if len(arr.shape) == 0:
                # 0-d scalar
                v = arr.to_flat_list()[0] if hasattr(arr, 'to_flat_list') else float(arr)
                return _fmt_val(v)
            if len(arr.shape) == 1:
                items = arr.to_flat_list() if hasattr(arr, 'to_flat_list') else list(arr)
                return '[' + ', '.join(_fmt_val(v) for v in items) + ']'
            rows = []
            n = arr.shape[0]
            for i in range(n):
                rows.append(_fmt_array(arr[i], depth + 1))
            pad = ' ' * (depth + 1)
            sep = ',\n' + pad
            return '[' + sep.join(rows) + ']'

        try:
            inner = _fmt_array(self._data)
        except Exception:
            inner = repr(self._data)
        extras = []
        if self.dtype not in ('float32',):
            extras.append(f'dtype=vp.{self.dtype}')
        if self.device != 'cpu':
            extras.append(f"device='{self.device}'")
        if self.requires_grad:
            extras.append('requires_grad=True')
        suffix = (', ' + ', '.join(extras)) if extras else ''
        return f"tensor({inner}{suffix})"

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self):
        return id(self)

    # ── autograd ────────────────────────────────────────────────────────────

    def backward(self, gradient=None):
        from vectorphile.autograd.engine import backward
        if gradient is None:
            sh = self._data.shape if hasattr(self._data, 'shape') else (1,)
            if not sh:
                sh = (1,)
            from vectorphile._core.ndarray import ones as _nd_ones_bw
            gradient = Tensor(_nd_ones_bw(sh, dtype='float32'), device=self.device)
        backward(self, gradient)

    def zero_grad(self):
        self.grad = None

    # ── utility ─────────────────────────────────────────────────────────────

    def cpu(self) -> 'Tensor':
        return Tensor(self._data, device='cpu', requires_grad=self.requires_grad)

    def cuda(self) -> 'Tensor':
        # Falls back to CPU gracefully (Phase 2 will add real CUDA)
        return Tensor(self._data, device='cuda', requires_grad=self.requires_grad)

    @property
    def is_cuda(self) -> bool:
        return self.device == 'cuda'

    def is_floating_point(self) -> bool:
        return self.dtype in ('float32', 'float64', 'float16', 'bfloat16')

    def is_leaf(self) -> bool:
        return self._is_leaf

    def fill_(self, value) -> 'Tensor':
        self._data.fill(value)
        return self

    def copy_(self, src: 'Tensor') -> 'Tensor':
        self._data[:] = src._data
        return self

    def masked_fill(self, mask, value: float) -> 'Tensor':
        rg = _grad_enabled() and self.requires_grad
        raw = mask._data if hasattr(mask, '_data') else mask
        mf = raw.to_flat_list() if hasattr(raw, 'to_flat_list') else list(raw)
        xf = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        sh = self._data.shape
        from vectorphile._core.ndarray import array as _nd_mf
        result_flat = [value if mf[i] else xf[i] for i in range(len(xf))]
        _dt_mf = str(self._data.dtype) if hasattr(self._data, 'dtype') else 'float32'
        result_data = _nd_mf(result_flat, dtype=_dt_mf).reshape(sh)
        result = Tensor(result_data, device=self.device, requires_grad=rg)
        if rg:
            _inp = self; _mask = raw; _dt_mf2 = _dt_mf
            class _MaskedFillBackward:
                def __init__(s): s.inputs = (_inp,)
                def backward(s, grad):
                    gf2 = grad._data.to_flat_list() if hasattr(grad._data, 'to_flat_list') else list(grad._data)
                    mf2 = _mask.to_flat_list() if hasattr(_mask, 'to_flat_list') else list(_mask)
                    gsh = grad._data.shape
                    from vectorphile._core.ndarray import array as _nd_mfb
                    return (Tensor(_nd_mfb([0.0 if mf2[i] else gf2[i] for i in range(len(gf2))], dtype=_dt_mf2).reshape(gsh)),)
            result.grad_fn = _MaskedFillBackward()
            result._is_leaf = False
        return result

    def softmax(self, dim=-1) -> 'Tensor':
        rg = _grad_enabled() and self.requires_grad
        xd = self._data
        xm = xd.max(axis=dim, keepdims=True)
        shifted = xd - xm
        e = shifted.exp() if hasattr(shifted, 'exp') else self._elem_exp(shifted)
        s_nd = e / e.sum(axis=dim, keepdims=True)
        result = Tensor(s_nd, device=self.device, requires_grad=rg)
        if rg:
            _inp = self; _dim = dim; _s_saved = s_nd
            class _SoftmaxBackward:
                def __init__(sl): sl.inputs = (_inp,)
                def backward(sl, grad):
                    g = grad._data
                    dot = (g * _s_saved).sum(axis=_dim, keepdims=True)
                    return (Tensor(_s_saved * (g - dot)),)
            result.grad_fn = _SoftmaxBackward()
            result._is_leaf = False
        return result

    def _elem_exp(self, nd):
        flat = nd.to_flat_list() if hasattr(nd, 'to_flat_list') else list(nd)
        sh = nd.shape if hasattr(nd, 'shape') else (len(flat),)
        from vectorphile._core.ndarray import array as _nd_ee
        _dt_ee = str(nd.dtype) if hasattr(nd, 'dtype') else 'float32'
        return _nd_ee([_math_t.exp(v) for v in flat], dtype=_dt_ee).reshape(sh)

    def log_softmax(self, dim=-1) -> 'Tensor':
        rg = _grad_enabled() and self.requires_grad
        xd = self._data
        xm = xd.max(axis=dim, keepdims=True)
        shifted = xd - xm
        e = self._elem_exp(shifted)
        log_sum = e.sum(axis=dim, keepdims=True)
        flat_ls = log_sum.to_flat_list() if hasattr(log_sum, 'to_flat_list') else list(log_sum)
        sh_ls = log_sum.shape
        from vectorphile._core.ndarray import array as _nd_lse
        _dt_lse = str(log_sum.dtype) if hasattr(log_sum, 'dtype') else 'float32'
        log_sum_exp = _nd_lse([_math_t.log(v if v > 1e-38 else 1e-38) for v in flat_ls], dtype=_dt_lse).reshape(sh_ls)
        ls_nd = shifted - log_sum_exp
        result = Tensor(ls_nd, device=self.device, requires_grad=rg)
        if rg:
            _inp = self; _dim = dim; _ls_saved = ls_nd
            class _LogSoftmaxBackward:
                def __init__(sl): sl.inputs = (_inp,)
                def backward(sl, grad):
                    g = grad._data
                    s_nd2 = Tensor(_ls_saved).exp()._data
                    return (Tensor(g - s_nd2 * g.sum(axis=_dim, keepdims=True)),)
            result.grad_fn = _LogSoftmaxBackward()
            result._is_leaf = False
        return result

    def where(self, condition: 'Tensor', other: 'Tensor') -> 'Tensor':
        cf = condition._data.to_flat_list() if hasattr(condition._data, 'to_flat_list') else list(condition._data)
        sf = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
        of = other._data.to_flat_list() if hasattr(other._data, 'to_flat_list') else list(other._data)
        sh = self._data.shape
        from vectorphile._core.ndarray import array as _nd_wh
        return Tensor(_nd_wh([sf[i] if cf[i] else of[i] for i in range(len(sf))], dtype='float32').reshape(sh), device=self.device)

    def flip(self, dims) -> 'Tensor':
        return flip(self, dims)

    def fliplr(self) -> 'Tensor':
        return fliplr(self)

    def flipud(self) -> 'Tensor':
        return flipud(self)

    def roll(self, shifts, dims=None) -> 'Tensor':
        return roll(self, shifts, dims)

    def rot90(self, k: int = 1, dims=(0, 1)) -> 'Tensor':
        return rot90(self, k, dims)

    def diff(self, n: int = 1, dim: int = -1) -> 'Tensor':
        return diff(self, n, dim)

    def addr(self, vec1: 'Tensor', vec2: 'Tensor', beta: float = 1.0, alpha: float = 1.0) -> 'Tensor':
        return addr(self, vec1, vec2, beta=beta, alpha=alpha)

    def index_copy_(self, dim: int, index: 'Tensor', source: 'Tensor') -> 'Tensor':
        result = index_copy(self, dim, index, source)
        self._data = result._data
        return self

    def index_copy(self, dim: int, index: 'Tensor', source: 'Tensor') -> 'Tensor':
        return index_copy(self, dim, index, source)

    def index_select(self, dim: int, index: 'Tensor') -> 'Tensor':
        return index_select(self, dim, index)

    def bucketize(self, boundaries: 'Tensor', out_int32: bool = False,
                  right: bool = False) -> 'Tensor':
        return bucketize(self, boundaries, out_int32=out_int32, right=right)


# ── functional tensor creation ───────────────────────────────────────────────

def tensor(data, dtype=None, device='cpu', requires_grad=False) -> Tensor:
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

def zeros(*shape, dtype='float32', device='cpu', requires_grad=False) -> Tensor:
    return Tensor.zeros(*shape, dtype=dtype, device=device, requires_grad=requires_grad)

def ones(*shape, dtype='float32', device='cpu', requires_grad=False) -> Tensor:
    return Tensor.ones(*shape, dtype=dtype, device=device, requires_grad=requires_grad)

def eye(n, m=None, dtype='float32', device='cpu') -> Tensor:
    return Tensor.eye(n, m, dtype=dtype, device=device)

def arange(start, stop=None, step=1, dtype=None, device='cpu') -> Tensor:
    if dtype is None:
        dtype = 'int64' if all(isinstance(v, int) for v in [start, stop, step] if v is not None) else 'float32'
    return Tensor.arange(start, stop, step, dtype=dtype, device=device)

def linspace(start, stop, num, dtype='float32', device='cpu') -> Tensor:
    return Tensor.linspace(start, stop, num, dtype=dtype, device=device)

def full(shape, fill_value, dtype='float32', device='cpu') -> Tensor:
    return Tensor.full(shape, fill_value, dtype=dtype, device=device)

def empty(*shape, dtype='float32', device='cpu') -> Tensor:
    return Tensor.empty(*shape, dtype=dtype, device=device)

def rand(*shape, device='cpu', requires_grad=False) -> Tensor:
    return Tensor.rand(*shape, device=device, requires_grad=requires_grad)

def randn(*shape, device='cpu', requires_grad=False) -> Tensor:
    return Tensor.randn(*shape, device=device, requires_grad=requires_grad)

def randint(low, high, shape, dtype='int64', device='cpu') -> Tensor:
    return Tensor.randint(low, high, shape, dtype=dtype, device=device)

def from_numpy(arr) -> Tensor:
    return Tensor.from_numpy(arr)

def zeros_like(t: Tensor, requires_grad=False) -> Tensor:
    from vectorphile._core.ndarray import zeros as _nd_zl
    return Tensor(_nd_zl(t._data.shape, dtype='float32'), device=t.device, requires_grad=requires_grad)

def ones_like(t: Tensor, requires_grad=False) -> Tensor:
    from vectorphile._core.ndarray import ones as _nd_ol
    return Tensor(_nd_ol(t._data.shape, dtype='float32'), device=t.device, requires_grad=requires_grad)

def empty_like(t: Tensor) -> Tensor:
    from vectorphile._core.ndarray import zeros as _nd_el
    return Tensor(_nd_el(t._data.shape, dtype=str(t._data.dtype)), device=t.device)

def full_like(t: Tensor, fill_value, dtype=None, requires_grad=False) -> Tensor:
    dt = dtype or str(t._data.dtype)
    from vectorphile._core.ndarray import array as _nd_fl
    n = 1
    for s in t._data.shape: n *= s
    return Tensor(_nd_fl([fill_value] * n, dt).reshape(t._data.shape),
                  device=t.device, requires_grad=requires_grad)

def rand_like(t: Tensor, requires_grad=False) -> Tensor:
    return Tensor.rand(*t._data.shape, device=t.device, requires_grad=requires_grad)

def randn_like(t: Tensor, requires_grad=False) -> Tensor:
    return Tensor.randn(*t._data.shape, device=t.device, requires_grad=requires_grad)

def cat(tensors, dim=0) -> Tensor:
    rg = any(t.requires_grad for t in tensors)
    arrays = [t._data for t in tensors]
    result = Tensor(_nd_concatenate(arrays, axis=dim), device=tensors[0].device, requires_grad=rg)
    if rg:
        _ts = list(tensors)
        _ndim = len(tensors[0]._data.shape)
        _dim = dim if dim >= 0 else _ndim + dim
        _sizes = [t._data.shape[_dim] for t in tensors]
        class _CatBackward:
            def __init__(s): s.inputs = _ts
            def backward(s, grad):
                splits = _nd_split(grad._data, _sizes[:-1], axis=_dim)
                return tuple(Tensor(sp) for sp in splits)
        result.grad_fn = _CatBackward(); result._is_leaf = False
    return result

def stack(tensors, dim=0) -> Tensor:
    rg = any(t.requires_grad for t in tensors)
    arrays = [t._data for t in tensors]
    result = Tensor(_nd_stack(arrays, axis=dim), device=tensors[0].device, requires_grad=rg)
    if rg:
        _ts = list(tensors); _dim = dim; _n = len(tensors)
        class _StackBackward:
            def __init__(s): s.inputs = _ts
            def backward(s, grad):
                splits = _nd_split(grad._data, _n, axis=_dim)
                new_shape = tuple(s for i, s in enumerate(splits[0].shape) if i != _dim)
                return tuple(Tensor(sp.reshape(new_shape)) for sp in splits)
        result.grad_fn = _StackBackward(); result._is_leaf = False
    return result

def split(tensor, split_size, dim=0):
    return tensor.split(split_size, dim)

def chunk(tensor, chunks, dim=0):
    return tensor.chunk(chunks, dim)

def squeeze(tensor, dim=None) -> Tensor:
    return tensor.squeeze(dim)

def unsqueeze(tensor, dim) -> Tensor:
    return tensor.unsqueeze(dim)

def flatten(tensor, start_dim=0, end_dim=-1) -> Tensor:
    return tensor.flatten(start_dim, end_dim)

def transpose(tensor, dim0, dim1) -> Tensor:
    return tensor.transpose(dim0, dim1)

def permute(tensor, *dims) -> Tensor:
    return tensor.permute(*dims)

def matmul(a: Tensor, b: Tensor) -> Tensor:
    return a.matmul(b)

mm = matmul

def norm(input: Tensor, p=2, dim=None, keepdim=False) -> Tensor:
    return input.norm(p, dim, keepdim)

def outer(input: Tensor, vec2: Tensor) -> Tensor:
    return ger(input, vec2)

def dot(a: Tensor, b: Tensor) -> Tensor:
    return a.dot(b)

def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    cf = condition._data.to_flat_list() if hasattr(condition._data, 'to_flat_list') else list(condition._data)
    xf = x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data)
    yf = y._data.to_flat_list() if hasattr(y._data, 'to_flat_list') else list(y._data)
    sh = x._data.shape
    from vectorphile._core.ndarray import array as _nd_wh2
    return Tensor(_nd_wh2([xf[i] if cf[i] else yf[i] for i in range(len(xf))], dtype='float32').reshape(sh), device=x.device)

def _nd_where(cond, x, y):
    """NDArray where — used by autograd.complete cross_entropy."""
    from vectorphile._core.ndarray import NDArray as _NDA
    def _to_flat(v):
        if isinstance(v, _NDA): return v.to_flat_list()
        if isinstance(v, Tensor): return v._data.to_flat_list() if hasattr(v._data, 'to_flat_list') else list(v._data)
        return list(v) if hasattr(v, '__iter__') else [v]
    def _get_shape(v):
        if isinstance(v, _NDA): return v.shape
        if isinstance(v, Tensor): return v._data.shape
        return (1,)
    cf = _to_flat(cond); xf = _to_flat(x); yf = _to_flat(y)
    sh = _get_shape(x)
    from vectorphile._core.ndarray import array as _nd_wh3
    return _nd_wh3([xf[i] if cf[i] else yf[i] for i in range(len(xf))], dtype='float32').reshape(sh)

def max(input, dim=None, keepdim=False):
    return input.max(dim, keepdim)

def min(input, dim=None, keepdim=False):
    return input.min(dim, keepdim)

def sum(input, dim=None, keepdim=False) -> Tensor:
    return input.sum(dim, keepdim)

def mean(input, dim=None, keepdim=False) -> Tensor:
    return input.mean(dim, keepdim)

def exp(input: Tensor) -> Tensor:
    return input.exp()

def log(input: Tensor) -> Tensor:
    return input.log()

def sqrt(input: Tensor) -> Tensor:
    return input.sqrt()

def abs(input: Tensor) -> Tensor:
    return input.abs()

def clamp(input: Tensor, min=None, max=None) -> Tensor:
    return input.clamp(min, max)

def softmax(input: Tensor, dim=-1) -> Tensor:
    return input.softmax(dim)

def log_softmax(input: Tensor, dim=-1) -> Tensor:
    return input.log_softmax(dim)

def sigmoid(input: Tensor) -> Tensor:
    return input.sigmoid()

def relu(input: Tensor) -> Tensor:
    return input.relu()

def tanh(input: Tensor) -> Tensor:
    return input.tanh()

def gelu(input: Tensor) -> Tensor:
    return input.gelu()

def silu(input: Tensor) -> Tensor:
    return input.silu()

swish = silu

def einsum(equation: str, *operands) -> Tensor:
    arrays = [op._data for op in operands]
    return Tensor(_nd_einsum(equation, *arrays), device=operands[0].device)


def lerp(start: 'Tensor', end: 'Tensor', weight) -> 'Tensor':
    """Linear interpolation: start + weight * (end - start)."""
    if isinstance(weight, (int, float)):
        return start + (end - start) * weight
    return start + (end - start) * weight


def addcmul(input: 'Tensor', tensor1: 'Tensor', tensor2: 'Tensor', value: float = 1.0) -> 'Tensor':
    """out = input + value * (tensor1 * tensor2) — fused multiply-accumulate."""
    return input + tensor1 * tensor2 * value


def addcdiv(input: 'Tensor', tensor1: 'Tensor', tensor2: 'Tensor', value: float = 1.0) -> 'Tensor':
    """out = input + value * (tensor1 / tensor2) — fused divide-accumulate."""
    return input + (tensor1 / tensor2) * value


def ger(vec1: 'Tensor', vec2: 'Tensor') -> 'Tensor':
    """Outer product of two 1D tensors. Equivalent to vec1.unsqueeze(1) @ vec2.unsqueeze(0)."""
    from vectorphile._core.ndarray import array as _nd_ger
    f1 = list(vec1._data.to_flat_list() if hasattr(vec1._data, 'to_flat_list') else vec1._data)
    f2 = list(vec2._data.to_flat_list() if hasattr(vec2._data, 'to_flat_list') else vec2._data)
    out = [a * b for a in f1 for b in f2]
    nd = _nd_ger(out, dtype='float32')
    nd = nd.reshape((len(f1), len(f2)))
    return Tensor(nd, device=vec1.device)


def cross(input: 'Tensor', other: 'Tensor', dim: int = -1) -> 'Tensor':
    """Cross product of two 3-element tensors along dim."""
    from vectorphile._core.ndarray import array as _nd_cr
    f1 = list(input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else input._data)
    f2 = list(other._data.to_flat_list() if hasattr(other._data, 'to_flat_list') else other._data)
    sh = input._data.shape if hasattr(input._data, 'shape') else (len(f1),)
    # Assume last dim is 3
    n = len(f1) // 3
    out = []
    for i in range(n):
        a0, a1, a2 = f1[3*i], f1[3*i+1], f1[3*i+2]
        b0, b1, b2 = f2[3*i], f2[3*i+1], f2[3*i+2]
        out.extend([a1*b2 - a2*b1, a2*b0 - a0*b2, a0*b1 - a1*b0])
    nd = _nd_cr(out, dtype='float32')
    nd = nd.reshape(sh)
    return Tensor(nd, device=input.device)


def kron(input: 'Tensor', other: 'Tensor') -> 'Tensor':
    """Kronecker product of two matrices."""
    from vectorphile._core.ndarray import array as _nd_kron
    f1 = list(input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else input._data)
    f2 = list(other._data.to_flat_list() if hasattr(other._data, 'to_flat_list') else other._data)
    sh1 = input._data.shape if hasattr(input._data, 'shape') else (len(f1),)
    sh2 = other._data.shape if hasattr(other._data, 'shape') else (len(f2),)
    if len(sh1) == 2 and len(sh2) == 2:
        R1, C1 = sh1; R2, C2 = sh2
        out = []
        for r1 in range(R1):
            for r2 in range(R2):
                for c1 in range(C1):
                    for c2 in range(C2):
                        out.append(f1[r1*C1 + c1] * f2[r2*C2 + c2])
        nd = _nd_kron(out, dtype='float32')
        nd = nd.reshape((R1*R2, C1*C2))
        return Tensor(nd, device=input.device)
    # Fallback: element-wise outer
    out = [a * b for a in f1 for b in f2]
    nd = _nd_kron(out, dtype='float32')
    return Tensor(nd, device=input.device)


def argsort(input: 'Tensor', dim: int = -1, descending: bool = False, stable: bool = False) -> 'Tensor':
    """Return indices that sort the tensor along dim."""
    from vectorphile._core.ndarray import array as _nd_as, _c_strides as _cs_as, NDArray as _NDA_as, _DType as _DT_as
    import array as _a_as
    flat = list(input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else input._data)
    sh = input._data.shape if hasattr(input._data, 'shape') else (len(flat),)
    ndim = len(sh)
    ax = dim % ndim if ndim else 0
    # Compute stride for the sort axis
    ax_size = sh[ax]
    outer = 1
    for d in range(ax):
        outer *= sh[d]
    inner = 1
    for d in range(ax + 1, ndim):
        inner *= sh[d]
    out_flat = [0] * len(flat)
    for o in range(outer):
        for i in range(inner):
            slice_vals = [(flat[o * ax_size * inner + k * inner + i], k) for k in range(ax_size)]
            sorted_vals = sorted(slice_vals, key=lambda x: x[0], reverse=descending)
            for k, (_, orig_k) in enumerate(sorted_vals):
                out_flat[o * ax_size * inner + k * inner + i] = float(orig_k)
    nd = _nd_as(out_flat, dtype='int64')
    return Tensor(nd.reshape(sh) if nd.reshape else nd, device=input.device)


def searchsorted(sorted_sequence: 'Tensor', values: 'Tensor', right: bool = False, side: str = 'left') -> 'Tensor':
    """Find positions where values should be inserted in sorted_sequence to maintain order."""
    import bisect as _bi_ss
    from vectorphile._core.ndarray import array as _nd_ss
    seq = sorted(list(sorted_sequence._data.to_flat_list() if hasattr(sorted_sequence._data, 'to_flat_list') else sorted_sequence._data))
    vals = list(values._data.to_flat_list() if hasattr(values._data, 'to_flat_list') else values._data)
    fn = _bi_ss.bisect_right if right or side == 'right' else _bi_ss.bisect_left
    out = [float(fn(seq, v)) for v in vals]
    nd = _nd_ss(out, dtype='int64')
    sh = values._data.shape if hasattr(values._data, 'shape') else (len(out),)
    return Tensor(nd.reshape(sh) if hasattr(nd, 'reshape') else nd, device=values.device)


def bincount(input: 'Tensor', weights=None, minlength: int = 0) -> 'Tensor':
    """Count occurrences of each value in input (1D non-negative integer tensor)."""
    from vectorphile._core.ndarray import array as _nd_bc
    flat = [int(v) for v in (input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else input._data)]
    import builtins as _bi_bc; max_val = _bi_bc.max(flat) if flat else -1
    n = _bi_bc.max(minlength, max_val + 1)
    if weights is not None:
        wf = list(weights._data.to_flat_list() if hasattr(weights._data, 'to_flat_list') else weights._data)
        counts = [0.0] * n
        for i, v in enumerate(flat):
            counts[v] += wf[i] if i < len(wf) else 0.0
    else:
        counts = [0.0] * n
        for v in flat:
            counts[v] += 1.0
    return Tensor(_nd_bc(counts, dtype='float32'), device=input.device)


def vmap(fn, in_dims=0, out_dims=0, randomness='error'):
    """Vectorized map (vmap) — apply fn across the batch dimension.

    Returns a new function that, when called with batched inputs (Tensors with
    an extra leading batch dimension), applies fn to each slice independently.

    This is a pure-Python fallback; for GPU-scale performance use JAX vmap.

    Args:
        fn: a function Tensor -> Tensor (or tuple -> tuple)
        in_dims: int or tuple of ints — which dim is the batch dim for each input
        out_dims: int — which dim of the output is the batch dim
        randomness: 'error' | 'same' | 'different' (matches JAX/PyTorch API)

    Example::
        import vectorphile as vp
        batched_linear = vp.vmap(lambda x: x @ W)
        y = batched_linear(x_batched)   # x_batched shape (B, D)
    """
    def _inner(*args):
        # Determine batch size from first tensor arg
        first_tensor = next((a for a in args if isinstance(a, Tensor)), None)
        if first_tensor is None:
            return fn(*args)
        if not hasattr(first_tensor._data, 'shape') or len(first_tensor._data.shape) == 0:
            return fn(*args)
        b_dim = in_dims if isinstance(in_dims, int) else in_dims[0]
        sh = first_tensor._data.shape
        if b_dim >= len(sh):
            return fn(*args)
        B = sh[b_dim]
        # Slice each batched arg along its batch dim and collect outputs
        outputs = []
        for b in range(B):
            sliced = []
            for i, a in enumerate(args):
                if isinstance(a, Tensor) and hasattr(a._data, 'shape') and len(a._data.shape) > 0:
                    dim = in_dims[i] if isinstance(in_dims, (list, tuple)) and i < len(in_dims) else in_dims
                    # Slice along dim
                    flat = list(a._data.to_flat_list() if hasattr(a._data, 'to_flat_list') else a._data)
                    a_sh = a._data.shape
                    if dim < len(a_sh):
                        # Build slice size for all dims except batch dim
                        from vectorphile._core.ndarray import array as _nd_vm
                        slice_size = 1
                        for d, s in enumerate(a_sh):
                            if d != dim:
                                slice_size *= s
                        # Extract b-th slice
                        sl_flat = flat[b * slice_size:(b+1) * slice_size]
                        new_sh = tuple(s for d, s in enumerate(a_sh) if d != dim)
                        if not new_sh:
                            new_sh = (1,)
                        nd = _nd_vm(sl_flat, dtype=str(a._data.dtype) if hasattr(a._data, 'dtype') else 'float32')
                        nd = nd.reshape(new_sh)
                        sliced.append(Tensor(nd, device=a.device, requires_grad=a.requires_grad))
                    else:
                        sliced.append(a)
                else:
                    sliced.append(a)
            outputs.append(fn(*sliced))
        # Stack outputs along out_dims
        if not outputs:
            return fn(*args)
        if isinstance(outputs[0], Tensor):
            out_flat = []
            for o in outputs:
                out_flat.extend(list(o._data.to_flat_list() if hasattr(o._data, 'to_flat_list') else o._data))
            from vectorphile._core.ndarray import array as _nd_vm2
            o_sh = outputs[0]._data.shape if hasattr(outputs[0]._data, 'shape') else (len(out_flat),)
            stacked_sh = (B,) + o_sh
            nd = _nd_vm2(out_flat, dtype=str(outputs[0]._data.dtype) if hasattr(outputs[0]._data, 'dtype') else 'float32')
            nd = nd.reshape(stacked_sh)
            return Tensor(nd, device=outputs[0].device)
        return outputs
    return _inner

def tril(input: Tensor, diagonal: int = 0) -> Tensor:
    return Tensor(_nd_tril(input._data, k=diagonal), device=input.device)

def triu(input: Tensor, diagonal: int = 0) -> Tensor:
    return Tensor(_nd_triu(input._data, k=diagonal), device=input.device)

def bmm(a: Tensor, b: Tensor) -> Tensor:
    return a.matmul(b)

def topk(input: Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True):
    idx_nd = _nd_argsort(input._data, axis=dim)
    idx_flat = idx_nd.to_flat_list() if hasattr(idx_nd, 'to_flat_list') else list(idx_nd)
    if largest:
        idx_flat = list(reversed(idx_flat[:k])) if sorted else idx_flat[-k:]
    else:
        idx_flat = idx_flat[:k]
    sh_in = input._data.shape
    # get values at those indices
    xf = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    vals_flat = [xf[int(i)] for i in idx_flat]
    from vectorphile._core.ndarray import array as _nd_tk
    return Tensor(_nd_tk(vals_flat, dtype='float32'), device=input.device), Tensor(_nd_tk([float(i) for i in idx_flat], dtype='float32'), device=input.device)

def cumsum(input: Tensor, dim: int = 0) -> Tensor:
    return Tensor(_nd_cumsum(input._data, axis=dim), device=input.device)

def multinomial(input: Tensor, num_samples: int, replacement: bool = True) -> Tensor:
    import random as _rnd_mn
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    n = len(flat)
    import builtins as _bi_mn; total = _bi_mn.sum(flat)
    probs = [v / total for v in flat]
    cumprobs = []
    acc = 0.0
    for p in probs:
        acc += p
        cumprobs.append(acc)
    def _sample():
        r = _rnd_mn.random()
        for i, cp in enumerate(cumprobs):
            if r <= cp:
                return i
        return n - 1
    if replacement:
        samples = [_sample() for _ in range(num_samples)]
    else:
        chosen = set()
        samples = []
        while len(samples) < num_samples:
            s = _sample()
            if s not in chosen:
                chosen.add(s)
                samples.append(s)
    from vectorphile._core.ndarray import array as _nd_mn
    return Tensor(_nd_mn([float(s) for s in samples], dtype='float32'), device=input.device)

def isnan(input: Tensor) -> Tensor:
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    sh = input._data.shape
    from vectorphile._core.ndarray import array as _nd_nan
    return Tensor(_nd_nan([v != v for v in flat], dtype='bool').reshape(sh), device=input.device)

def isinf(input: Tensor) -> Tensor:
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    sh = input._data.shape
    _inf = float('inf')
    from vectorphile._core.ndarray import array as _nd_inf
    return Tensor(_nd_inf([v == _inf or v == -_inf for v in flat], dtype='bool').reshape(sh), device=input.device)

def isfinite(input: Tensor) -> Tensor:
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    sh = input._data.shape
    _inf = float('inf')
    from vectorphile._core.ndarray import array as _nd_fin
    return Tensor(_nd_fin([v == v and v != _inf and v != -_inf for v in flat], dtype='bool').reshape(sh), device=input.device)

def sort(input: Tensor, dim: int = -1, descending: bool = False):
    idx_nd = _nd_argsort(input._data, axis=dim)
    if descending:
        idx_nd = _nd_flip(idx_nd, axis=dim)
    vals_nd = _nd_sort(input._data, axis=dim)
    if descending:
        vals_nd = _nd_flip(vals_nd, axis=dim)
    return Tensor(vals_nd, device=input.device), Tensor(idx_nd, device=input.device)


def roll(input: Tensor, shifts, dims=None) -> Tensor:
    """Roll tensor elements along given dimensions."""
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    sh = input._data.shape if hasattr(input._data, 'shape') else (len(flat),)
    if dims is None:
        n = len(flat)
        s = shifts if isinstance(shifts, int) else shifts[0]
        s = s % n if n else 0
        rolled = flat[-s:] + flat[:-s]
        from vectorphile._core.ndarray import array as _nd_arr
        return Tensor(_nd_arr([float(v) for v in rolled], dtype='float32').reshape(sh), device=input.device)
    if isinstance(dims, int):
        dims = [dims]; shifts = [shifts] if isinstance(shifts, int) else [shifts]
    result = flat[:]
    from vectorphile._core.ndarray import array as _nd_arr
    nd = _nd_arr([float(v) for v in result], dtype='float32').reshape(sh)
    # Simple 1D roll along axis for each dim
    for d, s in zip(dims, shifts):
        d = d % len(sh)
        dim_size = sh[d]
        s = s % dim_size if dim_size else 0
        flat2 = nd.to_flat_list()
        stride = 1
        for i in range(d + 1, len(sh)):
            stride *= sh[i]
        n_total = len(flat2)
        new_flat = flat2[:]
        # roll rows along dim d
        for base in range(n_total // (dim_size * stride)):
            for inner in range(stride):
                idxs = [base * dim_size * stride + (j * stride) + inner for j in range(dim_size)]
                vals = [flat2[i] for i in idxs]
                vals = vals[-s:] + vals[:-s]
                for k, i in enumerate(idxs):
                    new_flat[i] = vals[k]
        nd = _nd_arr([float(v) for v in new_flat], dtype='float32').reshape(sh)
    return Tensor(nd, device=input.device)


def flip(input: Tensor, dims) -> Tensor:
    """Flip tensor along given dimensions."""
    if isinstance(dims, int):
        dims = [dims]
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    sh = input._data.shape if hasattr(input._data, 'shape') else (len(flat),)
    from vectorphile._core.ndarray import array as _nd_arr
    nd = _nd_arr([float(v) for v in flat], dtype='float32').reshape(sh)
    for d in dims:
        d = d % len(sh)
        dim_size = sh[d]
        stride = 1
        for i in range(d + 1, len(sh)):
            stride *= sh[i]
        flat2 = nd.to_flat_list()
        new_flat = flat2[:]
        n_total = len(flat2)
        for base in range(n_total // (dim_size * stride)):
            for inner in range(stride):
                idxs = [base * dim_size * stride + (j * stride) + inner for j in range(dim_size)]
                vals = [flat2[i] for i in idxs]
                vals = vals[::-1]
                for k, i in enumerate(idxs):
                    new_flat[i] = vals[k]
        nd = _nd_arr([float(v) for v in new_flat], dtype='float32').reshape(sh)
    return Tensor(nd, device=input.device)


def rot90(input: Tensor, k: int = 1, dims=(0, 1)) -> Tensor:
    """Rotate tensor 90 degrees k times in the plane defined by dims."""
    k = k % 4
    if k == 0:
        return input
    result = input
    for _ in range(k):
        result = flip(transpose(result, dims[0], dims[1]), dims[0])
    return result


def diag(input: Tensor, diagonal: int = 0) -> Tensor:
    """Extract diagonal or construct diagonal tensor."""
    import builtins as _b
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    sh = input._data.shape if hasattr(input._data, 'shape') else (len(flat),)
    from vectorphile._core.ndarray import array as _nd_arr
    if len(sh) == 1:
        n = len(flat)
        size = n + _b.abs(diagonal)
        result = [0.0] * (size * size)
        for i in range(n):
            if diagonal >= 0:
                r, c = i, i + diagonal
            else:
                r, c = i - diagonal, i
            result[r * size + c] = flat[i]
        return Tensor(_nd_arr(result, dtype='float32').reshape((size, size)), device=input.device)
    else:
        rows, cols = sh[0], sh[1]
        if diagonal >= 0:
            length = _b.min(rows, cols - diagonal)
            result = [flat[i * cols + i + diagonal] for i in range(_b.max(0, length))]
        else:
            length = _b.min(rows + diagonal, cols)
            result = [flat[(i - diagonal) * cols + i] for i in range(_b.max(0, length))]
        return Tensor(_nd_arr([float(v) for v in result], dtype='float32'), device=input.device)


def trace(input: Tensor) -> Tensor:
    """Sum of diagonal elements."""
    d = diag(input, 0)
    flat = d._data.to_flat_list() if hasattr(d._data, 'to_flat_list') else list(d._data)
    total = 0.0
    for v in flat:
        if isinstance(v, Tensor):
            vf = v._data.to_flat_list() if hasattr(v._data, 'to_flat_list') else list(v._data)
            total += float(vf[0]) if vf else 0.0
        elif hasattr(v, 'to_flat_list'):
            vf = v.to_flat_list()
            total += float(vf[0]) if vf else 0.0
        else:
            total += float(v)
    from vectorphile._core.ndarray import array as _nd_arr
    return Tensor(_nd_arr([total], dtype='float32'))


def diff(input: Tensor, n: int = 1, dim: int = -1) -> Tensor:
    """n-th discrete difference along dim."""
    result = input
    for _ in range(n):
        flat = result._data.to_flat_list() if hasattr(result._data, 'to_flat_list') else list(result._data)
        sh = result._data.shape if hasattr(result._data, 'shape') else (len(flat),)
        d = dim % len(sh)
        dim_size = sh[d]
        stride = 1
        for i in range(d + 1, len(sh)):
            stride *= sh[i]
        new_dim_size = dim_size - 1
        new_sh = sh[:d] + (new_dim_size,) + sh[d+1:]
        new_flat = []
        n_total = len(flat)
        for base in range(n_total // (dim_size * stride)):
            for j in range(new_dim_size):
                for inner in range(stride):
                    idx_cur = base * dim_size * stride + (j + 1) * stride + inner
                    idx_prv = base * dim_size * stride + j * stride + inner
                    new_flat.append(flat[idx_cur] - flat[idx_prv])
        from vectorphile._core.ndarray import array as _nd_arr
        nd = _nd_arr([float(v) for v in new_flat], dtype='float32').reshape(new_sh)
        result = Tensor(nd, device=input.device)
    return result


def meshgrid(*tensors, indexing='ij') -> list:
    """Create coordinate grids from 1D tensors."""
    import itertools as _it
    sizes = []
    for t in tensors:
        flat = t._data.to_flat_list() if hasattr(t._data, 'to_flat_list') else list(t._data)
        sizes.append(flat)
    n = len(sizes)
    if indexing == 'xy' and n >= 2:
        sizes[0], sizes[1] = sizes[1], sizes[0]
    total_shape = tuple(len(s) for s in sizes)
    from vectorphile._core.ndarray import array as _nd_arr
    grids = []
    for i, s in enumerate(sizes):
        result_flat = []
        for idx in _it.product(*[range(total_shape[j]) for j in range(n)]):
            result_flat.append(float(s[idx[i]]))
        nd = _nd_arr(result_flat, dtype='float32').reshape(total_shape)
        grids.append(Tensor(nd, device=tensors[0].device))
    return grids


def cdist(x1: Tensor, x2: Tensor, p: float = 2.0) -> Tensor:
    """Compute pairwise distances between rows of x1 and x2."""
    import math as _m
    import builtins as _bi
    f1 = [float(v) for v in (x1._data.to_flat_list() if hasattr(x1._data, 'to_flat_list') else list(x1._data))]
    f2 = [float(v) for v in (x2._data.to_flat_list() if hasattr(x2._data, 'to_flat_list') else list(x2._data))]
    sh1 = x1._data.shape if hasattr(x1._data, 'shape') else (len(f1),)
    sh2 = x2._data.shape if hasattr(x2._data, 'shape') else (len(f2),)
    n1, D = sh1[0], sh1[-1]
    n2 = sh2[0]
    result = []
    for i in range(n1):
        row1 = f1[i*D:(i+1)*D]
        for j in range(n2):
            row2 = f2[j*D:(j+1)*D]
            if p == 2.0:
                dist = _m.sqrt(_bi.sum((a-b)**2 for a, b in zip(row1, row2)))
            elif p == 1.0:
                dist = _bi.sum(abs(a-b) for a, b in zip(row1, row2))
            elif p == float('inf'):
                dist = _bi.max(abs(a-b) for a, b in zip(row1, row2))
            else:
                dist = _bi.sum(abs(a-b)**p for a, b in zip(row1, row2)) ** (1.0/p)
            result.append(dist)
    from vectorphile._core.ndarray import array as _nd_arr
    nd = _nd_arr([float(v) for v in result], dtype='float32').reshape((n1, n2))
    return Tensor(nd, device=x1.device)


def unique(input: Tensor, sorted: bool = True, return_inverse: bool = False,
           return_counts: bool = False, dim=None):
    """Return the unique elements of the input tensor."""
    from vectorphile._core.ndarray import array as _nd_arr
    flat = list(input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else input._data)
    seen = {}
    unique_vals = []
    inverse = []
    for v in flat:
        key = round(float(v), 6)
        if key not in seen:
            seen[key] = len(unique_vals)
            unique_vals.append(float(v))
        inverse.append(seen[key])
    if sorted:
        order = sorted_order = list(range(len(unique_vals)))
        pairs = list(enumerate(unique_vals))
        pairs.sort(key=lambda x: x[1])
        old_to_new = {old: new for new, (old, _) in enumerate(pairs)}
        unique_vals = [v for _, v in pairs]
        inverse = [old_to_new[i] for i in inverse]
    counts = [0] * len(unique_vals)
    for i in inverse:
        counts[i] += 1
    u_nd = _nd_arr(unique_vals, dtype='float32')
    result = Tensor(u_nd, device=input.device)
    extras = []
    if return_inverse:
        inv_nd = _nd_arr([float(i) for i in inverse], dtype='float32')
        extras.append(Tensor(inv_nd, device=input.device))
    if return_counts:
        cnt_nd = _nd_arr([float(c) for c in counts], dtype='float32')
        extras.append(Tensor(cnt_nd, device=input.device))
    if extras:
        return (result, *extras)
    return result


def broadcast_tensors(*tensors):
    """Return tensors broadcast to a common shape."""
    if not tensors:
        return ()
    shapes = [t._data.shape for t in tensors]
    import builtins as _bi_bc2; ndim = _bi_bc2.max(len(s) for s in shapes)
    # pad on left with 1s
    padded = [(1,) * (ndim - len(s)) + tuple(s) for s in shapes]
    out_shape = []
    for dims in zip(*padded):
        mx = _bi_bc2.max(dims)
        out_shape.append(mx)
    out_shape = tuple(out_shape)
    result = []
    from vectorphile._core.ndarray import array as _nd_arr
    for t, psh in zip(tensors, padded):
        flat_in = list(t._data.to_flat_list() if hasattr(t._data, 'to_flat_list') else t._data)
        import math as _math
        total = 1
        for s in out_shape:
            total *= s
        # simple broadcast via index mapping
        out_flat = []
        for linear in range(total):
            rem = linear
            idx_out = []
            for s in reversed(out_shape):
                idx_out.append(rem % s)
                rem //= s
            idx_out.reverse()
            # map to source index
            src_idx = 0
            stride = 1
            for i in reversed(range(ndim)):
                s_src = psh[i]
                d = idx_out[i] % s_src
                src_idx += d * stride
                stride *= s_src
            out_flat.append(float(flat_in[src_idx]) if src_idx < len(flat_in) else 0.0)
        nd = _nd_arr(out_flat, dtype='float32').reshape(out_shape)
        result.append(Tensor(nd, device=t.device))
    return tuple(result)


def atleast_1d(*tensors):
    """Return tensors with at least 1 dimension."""
    from vectorphile._core.ndarray import array as _nd_arr
    result = []
    for t in tensors:
        if not hasattr(t, '_data'):
            nd = _nd_arr([float(t)], dtype='float32')
            result.append(Tensor(nd))
        elif len(t._data.shape) == 0:
            flat = list(t._data.to_flat_list() if hasattr(t._data, 'to_flat_list') else t._data)
            nd = _nd_arr(flat or [0.0], dtype='float32')
            result.append(Tensor(nd, device=t.device))
        else:
            result.append(t)
    return result[0] if len(result) == 1 else result


def atleast_2d(*tensors):
    """Return tensors with at least 2 dimensions."""
    from vectorphile._core.ndarray import array as _nd_arr
    result = []
    for t in tensors:
        if not hasattr(t, '_data'):
            nd = _nd_arr([[float(t)]], dtype='float32').reshape((1, 1))
            result.append(Tensor(nd))
        elif len(t._data.shape) == 0:
            flat = list(t._data.to_flat_list() if hasattr(t._data, 'to_flat_list') else t._data)
            nd = _nd_arr(flat or [0.0], dtype='float32').reshape((1, 1))
            result.append(Tensor(nd, device=t.device))
        elif len(t._data.shape) == 1:
            n = t._data.shape[0]
            nd = t._data.reshape((1, n))
            result.append(Tensor(nd, device=t.device))
        else:
            result.append(t)
    return result[0] if len(result) == 1 else result


def atleast_3d(*tensors):
    """Return tensors with at least 3 dimensions."""
    from vectorphile._core.ndarray import array as _nd_arr
    result = []
    for t in tensors:
        if not hasattr(t, '_data'):
            nd = _nd_arr([[[float(t)]]], dtype='float32').reshape((1, 1, 1))
            result.append(Tensor(nd))
        elif len(t._data.shape) == 0:
            flat = list(t._data.to_flat_list() if hasattr(t._data, 'to_flat_list') else t._data)
            nd = _nd_arr(flat or [0.0], dtype='float32').reshape((1, 1, 1))
            result.append(Tensor(nd, device=t.device))
        elif len(t._data.shape) == 1:
            n = t._data.shape[0]
            nd = t._data.reshape((1, n, 1))
            result.append(Tensor(nd, device=t.device))
        elif len(t._data.shape) == 2:
            h, w = t._data.shape
            nd = t._data.reshape((1, h, w))
            result.append(Tensor(nd, device=t.device))
        else:
            result.append(t)
    return result[0] if len(result) == 1 else result


# ── NDArray low-level helpers (used by _vp_helpers.py and autograd layers) ───

def _nd_to_np(a, dtype=None):
    """Return NDArray as a Python list (flat) — pure Python, no numpy."""
    from vectorphile._core.ndarray import NDArray as _NDA
    if isinstance(a, _NDA):
        return a.to_flat_list()
    if hasattr(a, 'to_flat_list'):
        return a.to_flat_list()
    return list(a)


def _np_to_nd(arr, shape=None):
    """Convert a flat list (or nested) to NDArray."""
    from vectorphile._core.ndarray import array as _nd_arr, NDArray as _NDA
    if isinstance(arr, _NDA):
        return arr
    flat = arr if isinstance(arr, list) else list(arr)
    nd = _nd_arr([float(v) for v in flat], dtype='float32')
    if shape:
        nd = nd.reshape(tuple(shape))
    return nd


def _nd_concatenate(arrays, axis=0):
    from vectorphile._core.ndarray import NDArray as _NDA
    nd_arrays = [a if isinstance(a, _NDA) else _np_to_nd(a) for a in arrays]
    if not nd_arrays:
        return _np_to_nd([])
    ndim = len(nd_arrays[0].shape)
    if axis < 0:
        axis = ndim + axis
    if axis == 0:
        all_flat = []
        sh0 = list(nd_arrays[0].shape)
        for a in nd_arrays:
            all_flat.extend(a.to_flat_list())
            sh0[0] += 0  # just accumulate
        total = 0
        for a in nd_arrays:
            total += a.shape[0]
        new_shape = (total,) + nd_arrays[0].shape[1:]
        from vectorphile._core.ndarray import array as _nd_arr
        _dt_cat = str(nd_arrays[0].dtype) if hasattr(nd_arrays[0], 'dtype') else 'float32'
        return _nd_arr([float(v) for v in all_flat], dtype=_dt_cat).reshape(new_shape)
    # general axis: use slice-based approach for 2D
    from vectorphile._core.ndarray import array as _nd_arr
    # fallback: flatten each along axis and reassemble
    sh = list(nd_arrays[0].shape)
    total_axis = 0
    for _a_cat in nd_arrays: total_axis += _a_cat.shape[axis]
    new_shape = sh[:axis] + [total_axis] + sh[axis+1:]
    # collect slices
    n_outer = 1
    for s in sh[:axis]: n_outer *= s
    n_inner = 1
    for s in sh[axis+1:]: n_inner *= s
    result = []
    for outer in range(n_outer):
        for arr in nd_arrays:
            ax_size = arr.shape[axis]
            flat = arr.to_flat_list()
            ax_stride = n_inner
            for s in arr.shape[axis+1:]: ax_stride //= s  # shouldn't divide
            ax_stride = n_inner
            for i in range(ax_size):
                start = (outer * arr.shape[axis] + i) * n_inner
                # simpler: just take chunks
                start2 = outer * ax_size * n_inner + i * n_inner
                result.extend(flat[start2:start2 + n_inner])
    _dt_cat2 = str(nd_arrays[0].dtype) if hasattr(nd_arrays[0], 'dtype') else 'float32'
    return _nd_arr([float(v) for v in result], dtype=_dt_cat2).reshape(tuple(new_shape))


def _nd_stack(arrays, axis=0):
    from vectorphile._core.ndarray import NDArray as _NDA, array as _nd_arr
    nd_arrays = [a if isinstance(a, _NDA) else _np_to_nd(a) for a in arrays]
    if not nd_arrays:
        return _np_to_nd([])
    sh = nd_arrays[0].shape
    new_shape = sh[:axis] + (len(nd_arrays),) + sh[axis:]
    # Insert new axis: for axis=0, just prepend; for axis=-1, append
    all_flat = []
    for a in nd_arrays:
        all_flat.extend(a.to_flat_list())
    _dt_st = str(nd_arrays[0].dtype) if hasattr(nd_arrays[0], 'dtype') else 'float32'
    return _nd_arr([float(v) for v in all_flat], dtype=_dt_st).reshape(new_shape)


def _nd_split(a, indices_or_sections, axis=0):
    from vectorphile._core.ndarray import NDArray as _NDA, array as _nd_arr
    if not isinstance(a, _NDA):
        a = _np_to_nd(a)
    sh = list(a.shape)
    ndim = len(sh)
    if axis < 0: axis = ndim + axis
    flat = a.to_flat_list()
    n = sh[axis]
    if isinstance(indices_or_sections, int):
        size = n // indices_or_sections
        splits = [size * i for i in range(indices_or_sections + 1)]
        splits[-1] = n
    else:
        splits = [0] + list(indices_or_sections) + [n]
    n_inner = 1
    for s in sh[axis+1:]: n_inner *= s
    n_outer = 1
    for s in sh[:axis]: n_outer *= s
    results = []
    for i in range(len(splits) - 1):
        lo, hi = splits[i], splits[i+1]
        chunk_sh = sh[:axis] + [hi - lo] + sh[axis+1:]
        chunk = []
        for outer in range(n_outer):
            for idx in range(lo, hi):
                start = outer * n * n_inner + idx * n_inner
                chunk.extend(flat[start:start + n_inner])
        _dt_sp = str(a.dtype) if hasattr(a, 'dtype') else 'float32'
        results.append(_nd_arr([float(v) for v in chunk], dtype=_dt_sp).reshape(tuple(chunk_sh)))
    return results


def _nd_pad(a, pad_width, mode='constant', constant_values=0, **kw):
    from vectorphile._core.ndarray import NDArray as _NDA, array as _nd_arr
    if not isinstance(a, _NDA):
        a = _np_to_nd(a)
    cv = kw.get('constant_values', constant_values)
    flat = a.to_flat_list()
    sh = list(a.shape)
    ndim = len(sh)
    # Normalize pad_width
    if isinstance(pad_width, int):
        pw = [(pad_width, pad_width)] * ndim
    elif isinstance(pad_width, (list, tuple)) and isinstance(pad_width[0], int):
        pw = [tuple(pad_width)] * ndim
    else:
        pw = [tuple(p) for p in pad_width]
    # Build new shape
    new_sh = [sh[i] + pw[i][0] + pw[i][1] for i in range(ndim)]
    # Allocate output
    n_out = 1
    for s in new_sh: n_out *= s
    out = [float(cv)] * n_out
    # Compute strides for new and old shape
    def _strides(shape):
        s = [1]
        for sz in reversed(shape[1:]): s.insert(0, s[0] * sz)
        return s
    old_st = _strides(sh)
    new_st = _strides(new_sh)
    # Copy elements
    n_old = 1
    for s in sh: n_old *= s
    # Iterate over all old positions
    import itertools
    for coords in itertools.product(*[range(s) for s in sh]):
        old_idx = 0
        for i, c in enumerate(coords): old_idx += c * old_st[i]
        new_coords = tuple(c + pw[i][0] for i, c in enumerate(coords))
        new_idx = 0
        for i, c in enumerate(new_coords): new_idx += c * new_st[i]
        out[new_idx] = flat[old_idx]
    _out_dt = str(a.dtype) if hasattr(a, 'dtype') else 'float32'
    return _nd_arr([float(v) for v in out], dtype=_out_dt).reshape(tuple(new_sh))


def _nd_tile(a, reps):
    from vectorphile._core.ndarray import NDArray as _NDA, array as _nd_arr
    if not isinstance(a, _NDA):
        a = _np_to_nd(a)
    flat = a.to_flat_list()
    sh = list(a.shape)
    if isinstance(reps, int):
        reps = [reps] * len(sh)
    # Tile along each axis
    result = flat[:]
    cur_sh = sh[:]
    for axis, rep in enumerate(reps):
        new_flat = result * rep
        cur_sh[axis] *= rep
        result = new_flat
    return _nd_arr([float(v) for v in result], dtype='float32').reshape(tuple(cur_sh))


def _nd_broadcast_to(a, shape):
    from vectorphile._core.ndarray import NDArray as _NDA, array as _nd_arr
    if not isinstance(a, _NDA):
        a = _np_to_nd(a)
    old_sh = a.shape
    flat = a.to_flat_list()
    # Pad shape left with 1s
    ndim = len(shape)
    old_sh_padded = (1,) * (ndim - len(old_sh)) + old_sh
    # Build output
    n_out = 1
    for s in shape: n_out *= s
    out = [0.0] * n_out
    def _strides(sh):
        s = [1]
        for sz in reversed(sh[1:]): s.insert(0, s[0] * sz)
        return s
    old_st = _strides(old_sh_padded)
    new_st = _strides(shape)
    import itertools
    for coords in itertools.product(*[range(s) for s in shape]):
        old_coords = tuple(0 if old_sh_padded[i] == 1 else coords[i] for i in range(ndim))
        old_idx = 0
        for i, c in enumerate(old_coords): old_idx += c * old_st[i]
        new_idx = 0
        for i, c in enumerate(coords): new_idx += c * new_st[i]
        out[new_idx] = flat[old_idx]
    return _nd_arr([float(v) for v in out], dtype='float32').reshape(tuple(shape))


def _nd_flip(a, axis=None, dims=None):
    from vectorphile._core.ndarray import NDArray as _NDA, array as _nd_arr
    if not isinstance(a, _NDA):
        a = _np_to_nd(a)
    ax = axis if axis is not None else dims
    flat = list(a.to_flat_list())
    sh = list(a.shape)
    if ax is None:
        return _nd_arr(flat[::-1], dtype='float32').reshape(tuple(sh))
    if isinstance(ax, int):
        ax = [ax]
    for a_i in ax:
        a_i = a_i % len(sh)
        n_outer = 1
        for s in sh[:a_i]: n_outer *= s
        n_inner = 1
        for s in sh[a_i+1:]: n_inner *= s
        ax_size = sh[a_i]
        new_flat = flat[:]
        for outer in range(n_outer):
            for inner in range(n_inner):
                for k in range(ax_size):
                    src = outer * ax_size * n_inner + k * n_inner + inner
                    dst = outer * ax_size * n_inner + (ax_size - 1 - k) * n_inner + inner
                    new_flat[dst] = flat[src]
        flat = new_flat
    return _nd_arr([float(v) for v in flat], dtype='float32').reshape(tuple(sh))


def _nd_roll(a, shift, axis=None, dims=None):
    from vectorphile._core.ndarray import NDArray as _NDA, array as _nd_arr
    if not isinstance(a, _NDA):
        a = _np_to_nd(a)
    flat = list(a.to_flat_list())
    sh = list(a.shape)
    ax = axis if axis is not None else dims
    if ax is None:
        n = len(flat)
        shift = shift % n if n else 0
        return _nd_arr(flat[-shift:] + flat[:-shift] if shift else flat, dtype='float32').reshape(tuple(sh))
    ax = ax % len(sh)
    n_outer = 1
    for s in sh[:ax]: n_outer *= s
    n_inner = 1
    for s in sh[ax+1:]: n_inner *= s
    ax_size = sh[ax]
    shift = shift % ax_size if ax_size else 0
    new_flat = flat[:]
    for outer in range(n_outer):
        for inner in range(n_inner):
            for k in range(ax_size):
                src = outer * ax_size * n_inner + k * n_inner + inner
                dst_k = (k + shift) % ax_size
                dst = outer * ax_size * n_inner + dst_k * n_inner + inner
                new_flat[dst] = flat[src]
    return _nd_arr([float(v) for v in new_flat], dtype='float32').reshape(tuple(sh))


def _nd_swapaxes(a, ax1, ax2):
    from vectorphile._core.ndarray import NDArray as _NDA
    if not isinstance(a, _NDA):
        a = _np_to_nd(a)
    ndim = len(a.shape)
    axes = list(range(ndim))
    axes[ax1], axes[ax2] = axes[ax2], axes[ax1]
    return a.transpose(axes)


def _nd_moveaxis(a, source, destination):
    from vectorphile._core.ndarray import NDArray as _NDA
    if not isinstance(a, _NDA):
        a = _np_to_nd(a)
    ndim = len(a.shape)
    axes = list(range(ndim))
    axes.pop(source)
    axes.insert(destination, source)
    return a.transpose(axes)


def _nd_diag(a, k=0):
    from vectorphile._core.ndarray import NDArray as _NDA, array as _nd_arr
    if not isinstance(a, _NDA):
        a = _np_to_nd(a)
    flat = a.to_flat_list()
    sh = a.shape
    if len(sh) == 1:
        import builtins as _bi_dk; n = sh[0] + _bi_dk.abs(k)
        result = [[0.0]*n for _ in range(n)]
        for i, v in enumerate(flat):
            r, c = (i, i + k) if k >= 0 else (i - k, i)
            if 0 <= r < n and 0 <= c < n:
                result[r][c] = v
        out = [result[r][c] for r in range(n) for c in range(n)]
        return _nd_arr(out, dtype='float32').reshape((n, n))
    else:
        n, m = sh[0], sh[1]
        diag = []
        for i in range(n):
            j = i + k
            if 0 <= j < m:
                diag.append(flat[i * m + j])
        return _nd_arr([float(v) for v in diag], dtype='float32')


def _nd_tril(a, k=0):
    from vectorphile._core.ndarray import array as _nd_tl
    n, m = a.shape[0], a.shape[1]
    flat = a.to_flat_list() if hasattr(a, 'to_flat_list') else list(a)
    result = []
    for i in range(n):
        for j in range(m):
            result.append(flat[i * m + j] if j - i <= k else 0.0)
    return _nd_tl(result, dtype='float32').reshape((n, m))


def _nd_triu(a, k=0):
    n, m = a.shape if hasattr(a, 'shape') else (len(list(a)), len(list(a)[0]) if list(a) else 0)
    flat = a.to_flat_list() if hasattr(a, 'to_flat_list') else list(a)
    result = []
    for i in range(n):
        for j in range(m):
            result.append(flat[i * m + j] if j - i >= k else 0.0)
    from vectorphile._core.ndarray import array as _nd_tu
    return _nd_tu(result, dtype='float32').reshape((n, m))


def _nd_atleast_nd(a, ndim):
    from vectorphile._core.ndarray import NDArray as _NDA, array as _nd_arr
    if not isinstance(a, _NDA):
        a = _np_to_nd(a)
    sh = list(a.shape)
    while len(sh) < ndim:
        sh = [1] + sh
    return a.reshape(tuple(sh))


def _nd_meshgrid(*arrays, indexing='xy'):
    from vectorphile._core.ndarray import NDArray as _NDA, array as _nd_arr
    import itertools
    vecs = []
    for a in arrays:
        f = a.to_flat_list() if isinstance(a, _NDA) else list(a)
        vecs.append(f)
    if indexing == 'ij':
        dims = vecs
    else:
        dims = [vecs[1], vecs[0]] + list(vecs[2:]) if len(vecs) >= 2 else vecs
    shape = tuple(len(d) for d in dims)
    grids = []
    for i, vec in enumerate(dims):
        grid = []
        for coords in itertools.product(*[range(len(d)) for d in dims]):
            grid.append(float(vec[coords[i]]))
        grids.append(_nd_arr(grid, dtype='float32').reshape(shape))
    if indexing == 'xy' and len(grids) >= 2:
        grids[0], grids[1] = grids[1], grids[0]
    return grids


def _nd_argsort(a, axis=-1):
    from vectorphile._core.ndarray import NDArray as _NDA, array as _nd_arr
    if not isinstance(a, _NDA):
        a = _np_to_nd(a)
    flat = a.to_flat_list()
    sh = list(a.shape)
    ndim = len(sh)
    ax = axis % ndim
    n_outer = 1
    for s in sh[:ax]: n_outer *= s
    ax_size = sh[ax]
    n_inner = 1
    for s in sh[ax+1:]: n_inner *= s
    result = []
    for outer in range(n_outer):
        for inner in range(n_inner):
            row = [flat[outer * ax_size * n_inner + k * n_inner + inner] for k in range(ax_size)]
            sorted_idx = sorted(range(ax_size), key=lambda k: row[k])
            result.extend(sorted_idx)
    return _nd_arr([float(v) for v in result], dtype='float32').reshape(tuple(sh))


def _nd_sort(a, axis=-1):
    from vectorphile._core.ndarray import NDArray as _NDA, array as _nd_arr
    if not isinstance(a, _NDA):
        a = _np_to_nd(a)
    flat = a.to_flat_list()
    sh = list(a.shape)
    ndim = len(sh)
    ax = axis % ndim
    n_outer = 1
    for s in sh[:ax]: n_outer *= s
    ax_size = sh[ax]
    n_inner = 1
    for s in sh[ax+1:]: n_inner *= s
    result = [0.0] * len(flat)
    for outer in range(n_outer):
        for inner in range(n_inner):
            row = [flat[outer * ax_size * n_inner + k * n_inner + inner] for k in range(ax_size)]
            row.sort()
            for k, v in enumerate(row):
                result[outer * ax_size * n_inner + k * n_inner + inner] = v
    return _nd_arr([float(v) for v in result], dtype='float32').reshape(tuple(sh))


def _nd_unique(a):
    from vectorphile._core.ndarray import array as _nd_arr
    flat = a.to_flat_list() if hasattr(a, 'to_flat_list') else list(a)
    seen = {}
    vals = []
    for v in flat:
        fv = float(v)
        if fv not in seen:
            seen[fv] = len(vals)
            vals.append(fv)
    vals.sort()
    val_to_idx = {v: i for i, v in enumerate(vals)}
    inverse = [float(val_to_idx[float(v)]) for v in flat]
    counts_map = {}
    for v in vals:
        import builtins as _bi_uc; counts_map[v] = _bi_uc.sum(1 for x in flat if float(x) == v)
    counts = [float(counts_map[v]) for v in vals]
    return (_nd_arr(vals, dtype='float32'),
            _nd_arr(inverse, dtype='float32'),
            _nd_arr(counts, dtype='float32'))


def _nd_cumsum(a, axis=0):
    from vectorphile._core.ndarray import NDArray as _NDA, array as _nd_arr
    if not isinstance(a, _NDA):
        a = _np_to_nd(a)
    flat = a.to_flat_list()
    sh = list(a.shape)
    ax = axis % len(sh)
    ax_size = sh[ax]
    n_outer = 1
    for s in sh[:ax]: n_outer *= s
    n_inner = 1
    for s in sh[ax+1:]: n_inner *= s
    result = flat[:]
    for outer in range(n_outer):
        for inner in range(n_inner):
            acc = 0.0
            for k in range(ax_size):
                idx = outer * ax_size * n_inner + k * n_inner + inner
                acc += flat[idx]
                result[idx] = acc
    return _nd_arr([float(v) for v in result], dtype='float32').reshape(tuple(sh))


def _nd_diff(a, n=1, axis=-1):
    from vectorphile._core.ndarray import NDArray as _NDA, array as _nd_arr
    if not isinstance(a, _NDA):
        a = _np_to_nd(a)
    for _ in range(n):
        flat = a.to_flat_list()
        sh = list(a.shape)
        ax = axis % len(sh)
        ax_size = sh[ax]
        n_outer = 1
        for s in sh[:ax]: n_outer *= s
        n_inner = 1
        for s in sh[ax+1:]: n_inner *= s
        new_sh = sh[:ax] + [ax_size - 1] + sh[ax+1:]
        result = []
        for outer in range(n_outer):
            for k in range(ax_size - 1):
                for inner in range(n_inner):
                    i0 = outer * ax_size * n_inner + k * n_inner + inner
                    i1 = outer * ax_size * n_inner + (k+1) * n_inner + inner
                    result.append(flat[i1] - flat[i0])
        a = _nd_arr([float(v) for v in result], dtype='float32').reshape(tuple(new_sh))
    return a


def _nd_gradient(a, *varargs, axis=None, edge_order=1):
    from vectorphile._core.ndarray import NDArray as _NDA, array as _nd_arr
    if not isinstance(a, _NDA):
        a = _np_to_nd(a)
    flat = a.to_flat_list()
    sh = list(a.shape)
    ndim = len(sh)
    axes = list(range(ndim)) if axis is None else ([axis] if isinstance(axis, int) else list(axis))
    results = []
    for ax in axes:
        ax = ax % ndim
        ax_size = sh[ax]
        n_outer = 1
        for s in sh[:ax]: n_outer *= s
        n_inner = 1
        for s in sh[ax+1:]: n_inner *= s
        spacing = float(varargs[axes.index(ax)]) if varargs and axes.index(ax) < len(varargs) else 1.0
        result = flat[:]
        for outer in range(n_outer):
            for inner in range(n_inner):
                def _get(k): return flat[outer * ax_size * n_inner + k * n_inner + inner]
                def _set(k, v): result[outer * ax_size * n_inner + k * n_inner + inner] = v
                for k in range(ax_size):
                    if k == 0:
                        v = (_get(1) - _get(0)) / spacing if ax_size > 1 else 0.0
                    elif k == ax_size - 1:
                        v = (_get(k) - _get(k-1)) / spacing
                    else:
                        v = (_get(k+1) - _get(k-1)) / (2.0 * spacing)
                    _set(k, v)
        results.append(_nd_arr([float(v) for v in result], dtype='float32').reshape(tuple(sh)))
    return results


def _nd_histogram(a, bins=10, range=None):
    from vectorphile._core.ndarray import array as _nd_arr
    flat = a.to_flat_list() if hasattr(a, 'to_flat_list') else list(a)
    flat = [float(v) for v in flat]
    import builtins as _bi_hi; mn = _bi_hi.min(flat) if range is None else float(range[0])
    mx = _bi_hi.max(flat) if range is None else float(range[1])
    if mn == mx:
        mx = mn + 1.0
    if isinstance(bins, int):
        n_bins = bins
        edges = [mn + (mx - mn) * i / n_bins for i in range(n_bins + 1)]
    else:
        edges = [float(b) for b in bins]
        n_bins = len(edges) - 1
    counts = [0.0] * n_bins
    for v in flat:
        if v < edges[0] or v > edges[-1]:
            continue
        # Binary search for bin
        lo, hi = 0, n_bins - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if v < edges[mid + 1]:
                hi = mid
            else:
                lo = mid + 1
        if lo < n_bins:
            counts[lo] += 1.0
    return _nd_arr(counts, dtype='float32'), _nd_arr(edges, dtype='float32')


def _nd_bincount(a, weights=None, minlength=0):
    from vectorphile._core.ndarray import array as _nd_arr
    flat = a.to_flat_list() if hasattr(a, 'to_flat_list') else list(a)
    ai = [int(v) for v in flat]
    wf = None
    if weights is not None:
        wf = weights.to_flat_list() if hasattr(weights, 'to_flat_list') else list(weights)
        wf = [float(v) for v in wf]
    import builtins as _bi_bch; n = _bi_bch.max(_bi_bch.max(ai) + 1 if ai else 0, minlength)
    counts = [0.0] * n
    for i, v in enumerate(ai):
        counts[v] += wf[i] if wf is not None else 1.0
    return _nd_arr(counts, dtype='float32')


def _nd_linalg_norm(a, ord=None, axis=None, keepdims=False):
    import math as _m
    import builtins as _bi
    from vectorphile._core.ndarray import NDArray as _NDA, array as _nd_arr
    if not isinstance(a, _NDA):
        a = _np_to_nd(a)
    flat = a.to_flat_list()
    sh = list(a.shape)
    if axis is None:
        # Flatten norm
        if ord is None or ord == 2:
            val = _m.sqrt(_bi.sum(v*v for v in flat))
        elif ord == 1:
            val = _bi.sum(_m.fabs(v) for v in flat)
        elif ord == float('inf'):
            val = _bi.max(_m.fabs(v) for v in flat)
        else:
            val = _bi.sum(_m.fabs(v)**ord for v in flat) ** (1.0/ord)
        result = _nd_arr([float(val)], dtype='float32')
        return result
    ax = axis % len(sh)
    ax_size = sh[ax]
    n_outer = 1
    for s in sh[:ax]: n_outer *= s
    n_inner = 1
    for s in sh[ax+1:]: n_inner *= s
    result = []
    new_sh = sh[:ax] + ([1] if keepdims else []) + sh[ax+1:]
    for outer in range(n_outer):
        for inner in range(n_inner):
            row = [flat[outer * ax_size * n_inner + k * n_inner + inner] for k in range(ax_size)]
            import builtins as _bi_nm
            if ord is None or ord == 2:
                val = _m.sqrt(_bi_nm.sum(v*v for v in row))
            elif ord == 1:
                val = _bi_nm.sum(_m.fabs(v) for v in row)
            elif ord == float('inf'):
                val = _bi_nm.max(_m.fabs(v) for v in row)
            else:
                val = _bi_nm.sum(_m.fabs(v)**ord for v in row) ** (1.0/ord)
            result.append(float(val))
    return _nd_arr(result, dtype='float32').reshape(tuple(new_sh))


def _nd_einsum(eq, *arrays):
    """General-purpose einsum via PyTorch fallback or pure Python."""
    from vectorphile._core.ndarray import NDArray as _NDA, array as _nd_arr, _c_strides, _DType
    import array as _pyarr
    nd_arrays = [a if isinstance(a, _NDA) else _np_to_nd(a) for a in arrays]
    eq = eq.replace(' ', '')
    if '->' in eq:
        lhs, rhs = eq.split('->')
    else:
        lhs, rhs = eq, None
    operands = lhs.split(',')

    # Fast path: 2D matmul ij,jk->ik
    if (len(operands) == 2 and rhs is not None and len(operands[0]) == 2 and
            len(operands[1]) == 2 and len(rhs) == 2):
        a_idx, b_idx = operands[0], operands[1]
        contracted = set(a_idx) & set(b_idx) - set(rhs)
        if len(contracted) == 1 and len(nd_arrays[0].shape) == 2 and len(nd_arrays[1].shape) == 2:
            return nd_arrays[0] @ nd_arrays[1]

    # Fast path: single operand passthrough or reduction
    if len(operands) == 1 and rhs is not None:
        a_idx = operands[0]
        A = nd_arrays[0]
        if rhs == a_idx:
            return A
        if len(a_idx) == 2 and len(rhs) == 1:
            if rhs == a_idx[0]:
                return A.sum(axis=1)
            elif rhs == a_idx[1]:
                return A.sum(axis=0)
        if len(a_idx) == 2 and rhs == '' and a_idx[0] == a_idx[1]:
            flat = A.to_flat_list(); n = int(len(flat) ** 0.5)
            import builtins as _bi_tr; return _nd_arr([_bi_tr.sum(flat[i * n + i] for i in range(n))], dtype='float32')

    # Try PyTorch for complex cases (fastest)
    try:
        import torch as _torch
        _dt_map = {'float32': _torch.float32, 'float64': _torch.float64,
                   'int32': _torch.int32, 'int64': _torch.int64}
        _op_dt = str(nd_arrays[0].dtype) if nd_arrays else 'float32'
        _torch_dt = _dt_map.get(_op_dt, _torch.float32)
        t_arrays = [_torch.tensor(a.to_flat_list(), dtype=_torch_dt).reshape(a.shape)
                    for a in nd_arrays]
        result = _torch.einsum(eq, *t_arrays)
        flat = result.reshape(-1).tolist()
        sh = tuple(result.shape)
        _tc = 'd' if _op_dt == 'float64' else 'f'
        buf = _pyarr.array(_tc, [float(v) for v in flat])
        return _NDA(buf, sh, _c_strides(sh), _DType(_op_dt))
    except Exception:
        pass

    # Pure Python general einsum via index iteration
    # Build index sizes from operand shapes
    index_sizes: dict = {}
    for op_str, arr in zip(operands, nd_arrays):
        for ch, sz in zip(op_str, arr.shape):
            if ch in index_sizes:
                assert index_sizes[ch] == sz, f"einsum: index {ch!r} size mismatch"
            else:
                index_sizes[ch] = sz

    # Determine output indices
    all_indices = list(dict.fromkeys(c for op in operands for c in op))
    if rhs is None:
        # Sum over repeated indices
        import builtins as _bi_es; out_indices = [c for c in all_indices if _bi_es.sum(op.count(c) for op in operands) == 1]
    else:
        out_indices = list(rhs)

    summed_indices = [c for c in all_indices if c not in out_indices]
    out_shape = tuple(index_sizes[c] for c in out_indices) if out_indices else (1,)
    out_flat_size = 1
    for s in out_shape:
        out_flat_size *= s
    result = [0.0] * out_flat_size

    # Get flat data for each operand
    flat_data = [arr.to_flat_list() for arr in nd_arrays]
    op_strides = []
    for arr in nd_arrays:
        strides = [1] * len(arr.shape)
        for i in range(len(arr.shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * arr.shape[i + 1]
        op_strides.append(strides)

    from itertools import product as _iproduct
    out_ranges = [range(index_sizes[c]) for c in out_indices]
    sum_ranges = [range(index_sizes[c]) for c in summed_indices]

    # Compute output strides
    out_strides = [1] * len(out_shape)
    for i in range(len(out_shape) - 2, -1, -1):
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1]

    for out_vals in _iproduct(*out_ranges):
        out_idx_map = dict(zip(out_indices, out_vals))
        import builtins as _bi_ein; out_pos = _bi_ein.sum(v * out_strides[i] for i, v in enumerate(out_vals))
        total = 0.0
        for sum_vals in _iproduct(*sum_ranges):
            sum_idx_map = dict(zip(summed_indices, sum_vals))
            idx_map = {**out_idx_map, **sum_idx_map}
            term = 1.0
            for op_str, fd, strides in zip(operands, flat_data, op_strides):
                pos = _bi_ein.sum(idx_map[c] * strides[i] for i, c in enumerate(op_str))
                term *= fd[pos]
            total += term
        result[out_pos] = total

    buf = _pyarr.array('f', [float(v) for v in result])
    return _NDA(buf, out_shape, _c_strides(out_shape), _DType('float32'))


# ---------------------------------------------------------------------------
# SparseTensor — COO format sparse tensor with autograd support
# ---------------------------------------------------------------------------
class SparseTensor:
    """Sparse tensor in COO (coordinate) format.

    Attributes:
        indices: 2D Tensor of shape (ndim, nnz) — coordinates
        values:  1D Tensor of shape (nnz,) — non-zero values
        size:    tuple — dense shape
    """

    def __init__(self, indices: 'Tensor', values: 'Tensor', size: tuple):
        self.indices = indices
        self.values = values
        self.size = size
        self.requires_grad = values.requires_grad
        self._grad = None

    @property
    def shape(self):
        return self.size

    @property
    def ndim(self):
        return len(self.size)

    @property
    def nnz(self):
        if hasattr(self, '_nnz_actual'):
            return self._nnz_actual
        return self.values.shape[0] if hasattr(self.values, 'shape') else len(self.values._data)

    def to_dense(self) -> 'Tensor':
        """Convert COO sparse tensor to dense Tensor."""
        from vectorphile._core.ndarray import array as _nd_sp
        total = 1
        for d in self.size: total *= d
        out_flat = [0.0] * total
        if self.nnz == 0:
            return Tensor(_nd_sp(out_flat, 'float32').reshape(self.size))
        strides = [1] * len(self.size)
        for i in range(len(self.size) - 2, -1, -1):
            strides[i] = strides[i + 1] * self.size[i + 1]
        if hasattr(self.indices._data, 'to_flat_list'):
            idx_flat = self.indices._data.to_flat_list()
        else:
            idx_flat = list(self.indices._data)
        if hasattr(self.values._data, 'to_flat_list'):
            val_flat = self.values._data.to_flat_list()
        else:
            val_flat = list(self.values._data)
        ndim = len(self.size)
        nnz = self.nnz
        import builtins as _bi_sp
        for k in range(nnz):
            pos = _bi_sp.sum(int(idx_flat[d * nnz + k]) * strides[d] for d in range(ndim))
            out_flat[pos] = val_flat[k]
        return Tensor(_nd_sp(out_flat, 'float32').reshape(self.size))

    def __repr__(self):
        return f'SparseTensor(size={self.size}, nnz={self.nnz})'


def _sparse_embedding_lookup(weight: 'Tensor', indices_list: list, embedding_dim: int) -> 'Tensor':
    """Sparse embedding lookup — gradient only accumulates at accessed rows."""
    from vectorphile._core.ndarray import array as _nd_emb_sp
    n = len(indices_list)
    if hasattr(weight._data, 'to_flat_list'):
        wfl = weight._data.to_flat_list()
    else:
        wfl = list(weight._data)
    out_flat = []
    for idx in indices_list:
        row_start = int(idx) * embedding_dim
        out_flat.extend(wfl[row_start:row_start + embedding_dim])
    out_nd = _nd_emb_sp(out_flat, 'float32').reshape((n, embedding_dim))
    result = Tensor(out_nd, requires_grad=weight.requires_grad)
    if weight.requires_grad:
        from vectorphile.autograd.engine import Function as _Fn
        class _SparseEmbGrad(_Fn):
            @staticmethod
            def backward(grad_out):
                if hasattr(grad_out._data, 'to_flat_list'):
                    gfl = grad_out._data.to_flat_list()
                else:
                    gfl = list(grad_out._data)
                nv = len(wfl); w_grad = [0.0] * nv
                for i, idx in enumerate(indices_list):
                    row_start = int(idx) * embedding_dim
                    for j in range(embedding_dim):
                        w_grad[row_start + j] += gfl[i * embedding_dim + j]
                return Tensor(_nd_emb_sp(w_grad, 'float32').reshape(weight.shape)),
        fn = _SparseEmbGrad()
        fn.inputs = [weight]
        result.grad_fn = fn
    return result


# ── Missing top-level ops ────────────────────────────────────────────────────

def nonzero(input: Tensor) -> Tensor:
    """Return indices of non-zero elements (like torch.nonzero)."""
    from vectorphile._core.ndarray import array as _nd_nz
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    sh = input._data.shape
    ndim = len(sh)
    indices = []
    # iterate over all linear indices
    total = len(flat)
    for lin in range(total):
        if float(flat[lin]) != 0.0:
            coords = []
            rem = lin
            for d in range(ndim - 1, -1, -1):
                coords.append(rem % sh[d])
                rem //= sh[d]
            coords.reverse()
            indices.append(coords)
    if not indices:
        nd = _nd_nz([], dtype='int64').reshape((0, ndim))
        return Tensor(nd, device=input.device)
    flat_idx = [float(c) for row in indices for c in row]
    nd = _nd_nz(flat_idx, dtype='int64').reshape((len(indices), ndim))
    return Tensor(nd, device=input.device)


def count_nonzero(input: Tensor, dim=None) -> Tensor:
    """Count non-zero elements."""
    if dim is None:
        flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
        import builtins as _bi_cnz; cnt = _bi_cnz.sum(1 for v in flat if float(v) != 0.0)
        from vectorphile._core.ndarray import array as _nd_cnz
        return Tensor(_nd_cnz([float(cnt)], dtype='int64'), device=input.device)
    return (input != zeros(*input.shape)).sum(dim=dim)


def logsumexp(input: Tensor, dim, keepdim: bool = False) -> Tensor:
    """Numerically stable log(sum(exp(x))) along dim."""
    m = max(input, dim=dim, keepdim=True)
    # max() with dim returns (values, indices); we want values
    if isinstance(m, tuple):
        m = m[0]
    shifted = input - m
    s = sum(exp(shifted), dim=dim, keepdim=keepdim)
    m_squeezed = m if keepdim else squeeze(m, dim=dim)
    return log(s) + m_squeezed


def fmod(input: Tensor, other) -> Tensor:
    """Element-wise remainder with the same sign as input (like C fmod)."""
    import math as _m
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    if isinstance(other, Tensor):
        ofl = other._data.to_flat_list() if hasattr(other._data, 'to_flat_list') else list(other._data)
        result = [_m.fmod(float(a), float(b)) for a, b in zip(flat, ofl)]
    else:
        ov = float(other)
        result = [_m.fmod(float(v), ov) for v in flat]
    from vectorphile._core.ndarray import array as _nd_fm
    nd = _nd_fm(result, dtype='float32')
    if len(input.shape) > 1:
        nd = nd.reshape(input.shape)
    return Tensor(nd, device=input.device)


def remainder(input: Tensor, other) -> Tensor:
    """Element-wise remainder (like Python %, always same sign as divisor)."""
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    if isinstance(other, Tensor):
        ofl = other._data.to_flat_list() if hasattr(other._data, 'to_flat_list') else list(other._data)
        result = [float(a) % float(b) for a, b in zip(flat, ofl)]
    else:
        ov = float(other)
        result = [float(v) % ov for v in flat]
    from vectorphile._core.ndarray import array as _nd_rem
    nd = _nd_rem(result, dtype='float32')
    if len(input.shape) > 1:
        nd = nd.reshape(input.shape)
    return Tensor(nd, device=input.device)


def addmm(input: Tensor, mat1: Tensor, mat2: Tensor,
          beta: float = 1.0, alpha: float = 1.0) -> Tensor:
    """out = beta * input + alpha * (mat1 @ mat2)."""
    return input * beta + matmul(mat1, mat2) * alpha


def mv(mat: Tensor, vec: Tensor) -> Tensor:
    """Matrix-vector product: mat (m,n) @ vec (n,) -> (m,)."""
    if len(vec.shape) == 1:
        vec2 = unsqueeze(vec, 1)
        result = matmul(mat, vec2)
        return squeeze(result, 1)
    return matmul(mat, vec)


def vdot(input: Tensor, other: Tensor) -> Tensor:
    """Dot product of two 1D tensors (conjugate of input dotted with other)."""
    # For real tensors, same as dot
    return dot(flatten(input), flatten(other))


def sign(input: Tensor) -> Tensor:
    """Element-wise sign."""
    return input.sign()


def ceil(input: Tensor) -> Tensor:
    """Element-wise ceil."""
    return input.ceil()


def floor(input: Tensor) -> Tensor:
    """Element-wise floor."""
    return input.floor()


def round(input: Tensor, decimals: int = 0) -> Tensor:
    """Element-wise round."""
    return input.round(decimals)


def pow(input: Tensor, exponent) -> Tensor:
    """Element-wise power."""
    return input.pow(exponent)


def std(input: Tensor, dim=None, keepdim: bool = False, unbiased: bool = True) -> Tensor:
    """Standard deviation."""
    return input.std(dim=dim, keepdim=keepdim, unbiased=unbiased)


def var(input: Tensor, dim=None, keepdim: bool = False, unbiased: bool = True) -> Tensor:
    """Variance."""
    return input.var(dim=dim, keepdim=keepdim, unbiased=unbiased)


def prod(input: Tensor, dim=None, keepdim: bool = False) -> Tensor:
    """Product of all elements."""
    return input.prod(dim=dim, keepdim=keepdim)


def cumprod(input: Tensor, dim: int) -> Tensor:
    """Cumulative product."""
    return input.cumprod(dim=dim)


# ── Comparison operations ────────────────────────────────────────────────────

def eq(input: Tensor, other) -> Tensor:
    """Element-wise equality."""
    return input == other

def ne(input: Tensor, other) -> Tensor:
    """Element-wise inequality."""
    return input != other

def lt(input: Tensor, other) -> Tensor:
    """Element-wise less than."""
    return input < other

def le(input: Tensor, other) -> Tensor:
    """Element-wise less than or equal."""
    return input <= other

def gt(input: Tensor, other) -> Tensor:
    """Element-wise greater than."""
    return input > other

def ge(input: Tensor, other) -> Tensor:
    """Element-wise greater than or equal."""
    return input >= other

def maximum(input: Tensor, other) -> Tensor:
    """Element-wise maximum of two tensors."""
    import builtins as _bi_max
    if isinstance(other, Tensor):
        ofl = other._data.to_flat_list() if hasattr(other._data, 'to_flat_list') else list(other._data)
    else:
        ov = float(other)
        ofl = None
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    if ofl is not None:
        result = [_bi_max.max(float(a), float(b)) for a, b in zip(flat, ofl)]
    else:
        ov2 = float(other)
        result = [_bi_max.max(float(v), ov2) for v in flat]
    from vectorphile._core.ndarray import array as _nd_mx
    nd = _nd_mx(result, dtype='float32')
    if len(input.shape) > 1:
        nd = nd.reshape(input.shape)
    return Tensor(nd, device=input.device)

def minimum(input: Tensor, other) -> Tensor:
    """Element-wise minimum of two tensors."""
    import builtins as _bi_min
    if isinstance(other, Tensor):
        ofl = other._data.to_flat_list() if hasattr(other._data, 'to_flat_list') else list(other._data)
    else:
        ofl = None
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    if ofl is not None:
        result = [_bi_min.min(float(a), float(b)) for a, b in zip(flat, ofl)]
    else:
        ov2 = float(other)
        result = [_bi_min.min(float(v), ov2) for v in flat]
    from vectorphile._core.ndarray import array as _nd_mn
    nd = _nd_mn(result, dtype='float32')
    if len(input.shape) > 1:
        nd = nd.reshape(input.shape)
    return Tensor(nd, device=input.device)


# ── Missing ops ─────────────────────────────────────────────────────────────

def repeat_interleave(input: Tensor, repeats, dim=None) -> Tensor:
    """Repeat elements of a tensor (like torch.repeat_interleave)."""
    from vectorphile._core.ndarray import array as _nd_ri
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    sh = input._data.shape
    if dim is None:
        # Flatten and repeat each element
        if isinstance(repeats, int):
            result = [v for v in flat for _ in range(repeats)]
        else:
            reps = repeats.tolist() if hasattr(repeats, 'tolist') else list(repeats)
            result = [v for v, r in zip(flat, reps) for _ in range(int(r))]
        nd = _nd_ri(result, dtype='float32')
        return Tensor(nd, device=input.device)
    # Along a specific dimension
    if dim < 0:
        dim = len(sh) + dim
    if isinstance(repeats, int):
        reps_per = repeats
        # Build result by repeating along dim
        result_flat = []
        stride_before = 1
        for d in range(dim):
            stride_before *= sh[d]
        length = sh[dim]
        stride_after = 1
        for d in range(dim + 1, len(sh)):
            stride_after *= sh[d]
        for b in range(stride_before):
            for i in range(length):
                for _ in range(reps_per):
                    base = b * length * stride_after + i * stride_after
                    result_flat.extend(flat[base:base + stride_after])
        new_shape = sh[:dim] + (length * reps_per,) + sh[dim + 1:]
        nd = _nd_ri(result_flat, dtype='float32')
        if len(new_shape) > 1:
            nd = nd.reshape(new_shape)
        return Tensor(nd, device=input.device)
    # Variable repeats along dim - flattened approach
    reps = repeats.tolist() if hasattr(repeats, 'tolist') else list(repeats)
    result_flat = []
    length = sh[dim]
    stride_after = 1
    for d in range(dim + 1, len(sh)):
        stride_after *= sh[d]
    stride_before = 1
    for d in range(dim):
        stride_before *= sh[d]
    for b in range(stride_before):
        for i in range(length):
            for _ in range(int(reps[i])):
                base = b * length * stride_after + i * stride_after
                result_flat.extend(flat[base:base + stride_after])
    import builtins as _bi_ri; total_repeats = _bi_ri.sum(int(r) for r in reps)
    new_shape = sh[:dim] + (total_repeats,) + sh[dim + 1:]
    nd = _nd_ri(result_flat, dtype='float32')
    if len(new_shape) > 1:
        nd = nd.reshape(new_shape)
    return Tensor(nd, device=input.device)


def tile(input: Tensor, dims) -> Tensor:
    """Construct a tensor by repeating input along each dimension (like torch.tile)."""
    if isinstance(dims, int):
        dims = (dims,)
    result = input
    sh = input.shape
    # Pad dims to match ndim
    if len(dims) < len(sh):
        dims = (1,) * (len(sh) - len(dims)) + tuple(dims)
    # For each dim, repeat using cat
    for d, rep in enumerate(dims):
        if rep > 1:
            result = cat([result] * rep, dim=d)
    return result


# ── Missing masked/index operations ─────────────────────────────────────────

def masked_select(input: Tensor, mask: Tensor) -> Tensor:
    """Select elements from input where mask is True, returning a 1D tensor."""
    from vectorphile._core.ndarray import array as _nd_ms
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    mflat = mask._data.to_flat_list() if hasattr(mask._data, 'to_flat_list') else list(mask._data)
    result = [float(v) for v, m in zip(flat, mflat) if float(m) != 0.0]
    nd = _nd_ms(result, dtype='float32')
    return Tensor(nd, device=input.device)


def masked_scatter(input: Tensor, mask: Tensor, source: Tensor) -> Tensor:
    """Fill elements of input where mask is True with values from source."""
    from vectorphile._core.ndarray import array as _nd_msc
    flat = list(input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else input._data)
    mflat = mask._data.to_flat_list() if hasattr(mask._data, 'to_flat_list') else list(mask._data)
    sflat = source._data.to_flat_list() if hasattr(source._data, 'to_flat_list') else list(source._data)
    result = flat[:]
    si = 0
    for i, m in enumerate(mflat):
        if float(m) != 0.0 and si < len(sflat):
            result[i] = float(sflat[si])
            si += 1
    nd = _nd_msc(result, dtype='float32')
    if len(input.shape) > 1:
        nd = nd.reshape(input.shape)
    return Tensor(nd, device=input.device)


def index_add(input: Tensor, dim: int, index: Tensor, source: Tensor,
              alpha: float = 1.0) -> Tensor:
    """Add values from source to self at indices along dim."""
    result = input.clone()
    result.index_add_(dim, index, source, alpha=alpha)
    return result


def _tensor_index_add_(self: 'Tensor', dim: int, index: 'Tensor', source: 'Tensor',
                       alpha: float = 1.0) -> 'Tensor':
    """In-place: self[index] += alpha * source along dim."""
    from vectorphile._core.ndarray import array as _nd_ia
    idx_list = index._data.to_flat_list() if hasattr(index._data, 'to_flat_list') else list(index._data)
    flat = list(self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else self._data)
    sh = self._data.shape
    sflat = source._data.to_flat_list() if hasattr(source._data, 'to_flat_list') else list(source._data)
    if dim < 0:
        dim = len(sh) + dim
    stride_after = 1
    for d in range(dim + 1, len(sh)):
        stride_after *= sh[d]
    stride_before = 1
    for d in range(dim):
        stride_before *= sh[d]
    source_sh = source._data.shape
    for si, raw_idx in enumerate(idx_list):
        idx = int(float(raw_idx))
        for b in range(stride_before):
            base_self = b * sh[dim] * stride_after + idx * stride_after
            base_src = b * source_sh[dim] * stride_after + si * stride_after
            for s in range(stride_after):
                flat[base_self + s] += alpha * float(sflat[base_src + s])
    nd = _nd_ia(flat, dtype='float32')
    if len(sh) > 1:
        nd = nd.reshape(sh)
    self._data = nd
    return self

# Monkey-patch index_add_ onto Tensor
Tensor.index_add_ = _tensor_index_add_


# ── Missing Tensor instance methods ─────────────────────────────────────────

def _tensor_take(self: 'Tensor', index: 'Tensor') -> 'Tensor':
    """Take elements from tensor using flat indices."""
    from vectorphile._core.ndarray import array as _nd_tk
    flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
    idx = index._data.to_flat_list() if hasattr(index._data, 'to_flat_list') else list(index._data)
    result = [float(flat[int(i)]) for i in idx]
    nd = _nd_tk(result, dtype='float32')
    if len(index.shape) > 1:
        nd = nd.reshape(index.shape)
    return Tensor(nd, device=self.device)

Tensor.take = _tensor_take


def _tensor_index_select(self: 'Tensor', dim: int, index: 'Tensor') -> 'Tensor':
    """Select slices along dim using indices."""
    from vectorphile._core.ndarray import array as _nd_is
    sh = self._data.shape
    if dim < 0:
        dim = len(sh) + dim
    flat = self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else list(self._data)
    idx = index._data.to_flat_list() if hasattr(index._data, 'to_flat_list') else list(index._data)
    stride_after = 1
    for d in range(dim + 1, len(sh)):
        stride_after *= sh[d]
    stride_before = 1
    for d in range(dim):
        stride_before *= sh[d]
    result = []
    for b in range(stride_before):
        for i in idx:
            i = int(float(i))
            base = b * sh[dim] * stride_after + i * stride_after
            result.extend(flat[base:base + stride_after])
    new_shape = sh[:dim] + (len(idx),) + sh[dim + 1:]
    nd = _nd_is(result, dtype='float32')
    if len(new_shape) > 1:
        nd = nd.reshape(new_shape)
    return Tensor(nd, device=self.device)

Tensor.index_select = _tensor_index_select


def _tensor_fill_diagonal_(self: 'Tensor', fill_value: float, wrap: bool = False) -> 'Tensor':
    """Fill the main diagonal in-place."""
    from vectorphile._core.ndarray import array as _nd_fd
    sh = self._data.shape
    flat = list(self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else self._data)
    rows, cols = sh[0], sh[1] if len(sh) > 1 else sh[0]
    import builtins as _bi_fd
    n = _bi_fd.min(rows, cols)
    for i in range(n):
        flat[i * cols + i] = float(fill_value)
    nd = _nd_fd(flat, dtype='float32').reshape(sh)
    self._data = nd
    return self

Tensor.fill_diagonal_ = _tensor_fill_diagonal_


def _tensor_masked_fill_(self: 'Tensor', mask, value: float) -> 'Tensor':
    """In-place masked fill."""
    result = self.masked_fill(mask, value)
    self._data = result._data
    self.grad_fn = None
    return self

Tensor.masked_fill_ = _tensor_masked_fill_


def _tensor_register_hook(self: 'Tensor', hook):
    """Register a gradient hook that fires when gradient is computed.
    Returns a handle with .remove() method.
    """
    if not hasattr(self, '_hooks'):
        self._hooks = []
    self._hooks.append(hook)
    class _Handle:
        def __init__(self, hooks, fn):
            self._hooks = hooks; self._fn = fn
        def remove(self):
            try: self._hooks.remove(self._fn)
            except ValueError: pass
    return _Handle(self._hooks, hook)

Tensor.register_hook = _tensor_register_hook


def _tensor_type(self: 'Tensor', dtype=None) -> 'Tensor':
    """Return the type of this tensor, or cast to dtype (PyTorch compat).
    With no arg: returns dtype string. With dtype arg: returns casted tensor.
    """
    if dtype is None:
        return self.dtype
    return self.to(dtype=dtype)

Tensor.type = _tensor_type


def _tensor_type_as(self: 'Tensor', other: 'Tensor') -> 'Tensor':
    """Cast to same dtype as other tensor."""
    return self.to(dtype=other.dtype)

Tensor.type_as = _tensor_type_as


def _tensor_new_zeros(self: 'Tensor', *size, dtype=None, device=None, requires_grad=False) -> 'Tensor':
    """Create a new tensor of zeros with same dtype/device as self."""
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = tuple(size[0])
    return zeros(*size, dtype=dtype or self.dtype, device=device or self.device, requires_grad=requires_grad)

Tensor.new_zeros = _tensor_new_zeros


def _tensor_new_ones(self: 'Tensor', *size, dtype=None, device=None, requires_grad=False) -> 'Tensor':
    """Create a new tensor of ones with same dtype/device as self."""
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = tuple(size[0])
    return ones(*size, dtype=dtype or self.dtype, device=device or self.device, requires_grad=requires_grad)

Tensor.new_ones = _tensor_new_ones


def _tensor_new_full(self: 'Tensor', size, fill_value, dtype=None, device=None, requires_grad=False) -> 'Tensor':
    """Create a new filled tensor with same dtype/device as self."""
    return full(tuple(size), fill_value, dtype=dtype or self.dtype, device=device or self.device, requires_grad=requires_grad)

Tensor.new_full = _tensor_new_full


def _tensor_new_empty(self: 'Tensor', *size, dtype=None, device=None, requires_grad=False) -> 'Tensor':
    """Create a new empty tensor with same dtype/device as self."""
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = tuple(size[0])
    return empty(*size, dtype=dtype or self.dtype, device=device or self.device, requires_grad=requires_grad)

Tensor.new_empty = _tensor_new_empty


def _tensor_new_tensor(self: 'Tensor', data, dtype=None, device=None, requires_grad=False) -> 'Tensor':
    """Create a new tensor from data with same dtype/device as self."""
    return tensor(data, dtype=dtype or self.dtype, device=device or self.device, requires_grad=requires_grad)

Tensor.new_tensor = _tensor_new_tensor


# ── Additional Tensor instance methods ───────────────────────────────────────

def _tensor_stride(self: 'Tensor', dim=None):
    """Return stride of tensor (number of elements to step per dimension)."""
    sh = self.shape
    strides = []
    s = 1
    for d in reversed(sh):
        strides.insert(0, s)
        s *= d
    if dim is not None:
        return strides[dim]
    return tuple(strides)

Tensor.stride = _tensor_stride


def _tensor_is_contiguous(self: 'Tensor') -> bool:
    """Return True (all VP tensors are contiguous in memory)."""
    return True

Tensor.is_contiguous = _tensor_is_contiguous


def _tensor_narrow(self: 'Tensor', dim: int, start: int, length: int) -> 'Tensor':
    """Narrow a tensor along dim, from start for length elements."""
    idx = [slice(None)] * len(self.shape)
    idx[dim] = slice(start, start + length)
    return self[tuple(idx)]

Tensor.narrow = _tensor_narrow


def _tensor_view_as(self: 'Tensor', other: 'Tensor') -> 'Tensor':
    """View tensor with the same shape as other."""
    return self.view(*other.shape)

Tensor.view_as = _tensor_view_as


def _tensor_bernoulli_(self: 'Tensor', p: float = 0.5) -> 'Tensor':
    """Fill tensor in-place with Bernoulli samples."""
    import random as _rnd
    from vectorphile._core.ndarray import array as _nd_b
    flat = [1.0 if _rnd.random() < p else 0.0 for _ in range(self.numel())]
    nd = _nd_b(flat, dtype='float32')
    if len(self.shape) > 1:
        nd = nd.reshape(self.shape)
    self._data = nd
    return self

Tensor.bernoulli_ = _tensor_bernoulli_


def _tensor_uniform_(self: 'Tensor', from_: float = 0.0, to: float = 1.0) -> 'Tensor':
    """Fill tensor in-place with uniform random values."""
    import random as _rnd
    from vectorphile._core.ndarray import array as _nd_u
    flat = [_rnd.uniform(from_, to) for _ in range(self.numel())]
    nd = _nd_u(flat, dtype='float32')
    if len(self.shape) > 1:
        nd = nd.reshape(self.shape)
    self._data = nd
    return self

Tensor.uniform_ = _tensor_uniform_


def _tensor_normal_(self: 'Tensor', mean: float = 0.0, std: float = 1.0) -> 'Tensor':
    """Fill tensor in-place with normal random values."""
    import random as _rnd
    import math as _mth
    from vectorphile._core.ndarray import array as _nd_n
    flat = [_rnd.gauss(mean, std) for _ in range(self.numel())]
    nd = _nd_n(flat, dtype='float32')
    if len(self.shape) > 1:
        nd = nd.reshape(self.shape)
    self._data = nd
    return self

Tensor.normal_ = _tensor_normal_


def _tensor_scatter_add_(self: 'Tensor', dim: int, index: 'Tensor', src: 'Tensor') -> 'Tensor':
    """Scatter-add: self[index[i][j]] += src[i][j] along dim."""
    from vectorphile._core.ndarray import array as _nd_sa
    flat_in = list(self._data.to_flat_list() if hasattr(self._data, 'to_flat_list') else self._data)
    flat_idx = [int(float(v)) for v in (index._data.to_flat_list() if hasattr(index._data, 'to_flat_list') else list(index._data))]
    flat_src = [float(v) for v in (src._data.to_flat_list() if hasattr(src._data, 'to_flat_list') else list(src._data))]
    sh = self.shape
    stride = 1
    for i in range(dim + 1, len(sh)):
        stride *= sh[i]
    out = list(flat_in)
    idx_sh = index.shape
    for k, (idx_v, src_v) in enumerate(zip(flat_idx, flat_src)):
        base_before = (k // (stride * (idx_sh[dim] if dim < len(idx_sh) else 1))) * (stride * sh[dim])
        offset = k % stride
        pos = base_before + idx_v * stride + offset
        if 0 <= pos < len(out):
            out[pos] += src_v
    nd = _nd_sa(out, dtype='float32').reshape(sh)
    self._data = nd
    return self

Tensor.scatter_add_ = _tensor_scatter_add_


def _tensor_scatter_add(self: 'Tensor', dim: int, index: 'Tensor', src: 'Tensor') -> 'Tensor':
    """Out-of-place scatter-add."""
    result = self.clone()
    result.scatter_add_(dim, index, src)
    return result

Tensor.scatter_add = _tensor_scatter_add


def _tensor_as_strided(self: 'Tensor', size, stride, storage_offset: int = 0) -> 'Tensor':
    """View tensor with given size and stride (simplified)."""
    # Simplified: just reshape to size, ignoring arbitrary stride patterns
    return self.reshape(tuple(size))

Tensor.as_strided = _tensor_as_strided


# ── Additional missing PyTorch-compatible ops ────────────────────────────────

def tensordot(a: Tensor, b: Tensor, dims) -> Tensor:
    """Tensor dot product over specified dimensions."""
    from vectorphile._core.ndarray import array as _nd_td
    # Flatten contracted dims and do matmul
    if isinstance(dims, int):
        dims_a = list(range(len(a.shape) - dims, len(a.shape)))
        dims_b = list(range(dims))
    else:
        dims_a, dims_b = list(dims[0]), list(dims[1])
    # Move contracted dims to end of a, start of b
    free_a = [i for i in range(len(a.shape)) if i not in dims_a]
    free_b = [i for i in range(len(b.shape)) if i not in dims_b]
    a_perm = free_a + dims_a
    b_perm = dims_b + free_b
    at = a.permute(a_perm)
    bt = b.permute(b_perm)
    contract_size = 1
    for d in dims_a:
        contract_size *= a.shape[d]
    free_a_shape = tuple(a.shape[i] for i in free_a)
    free_b_shape = tuple(b.shape[i] for i in free_b)
    at2 = at.reshape((-1, contract_size))
    bt2 = bt.reshape((contract_size, -1))
    out = at2 @ bt2
    if free_a_shape + free_b_shape:
        out = out.reshape(free_a_shape + free_b_shape)
    return out


def inner(input: Tensor, other: Tensor) -> Tensor:
    """Inner product of two tensors (1-D dot product, or batch for higher dims)."""
    if len(input.shape) == 1 and len(other.shape) == 1:
        return vdot(input, other)
    return tensordot(input, other, dims=([-1], [-1]))


def gradient(input: Tensor, spacing=1.0, dim=None) -> 'list | Tensor':
    """Estimate gradient using central differences."""
    import builtins as _bi_gr
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    sh = input.shape
    if dim is None:
        dims = list(range(len(sh)))
    else:
        dims = [dim]
    results = []
    from vectorphile._core.ndarray import array as _nd_gr
    for d in dims:
        n = sh[d]
        stride = 1
        for i in range(d + 1, len(sh)):
            stride *= sh[i]
        out = [0.0] * len(flat)
        for i in range(len(flat)):
            # find position along dim d
            pos = (i // stride) % n
            if pos == 0:
                next_i = i + stride
                out[i] = (flat[next_i] - flat[i]) / spacing
            elif pos == n - 1:
                prev_i = i - stride
                out[i] = (flat[i] - flat[prev_i]) / spacing
            else:
                next_i = i + stride
                prev_i = i - stride
                out[i] = (flat[next_i] - flat[prev_i]) / (2 * spacing)
        nd = _nd_gr(out, dtype='float32').reshape(sh)
        results.append(Tensor(nd, device=input.device))
    return results[0] if len(results) == 1 else results


def histc(input: Tensor, bins: int = 100, min: float = 0.0, max: float = 0.0) -> Tensor:
    """Compute histogram of a tensor's values."""
    import builtins as _bi_hc
    from vectorphile._core.ndarray import array as _nd_hc
    flat = [float(v) for v in (input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data))]
    lo = _bi_hc.min(flat) if max == min == 0.0 else float(min)  # noqa: E721
    hi = _bi_hc.max(flat) if max == min == 0.0 else float(max)  # noqa: E721
    if hi == lo:
        hi = lo + 1.0
    counts = [0] * bins
    for v in flat:
        if v < lo or v > hi:
            continue
        idx = int((v - lo) / (hi - lo) * bins)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1
    nd = _nd_hc(counts, dtype='float32')
    return Tensor(nd, device=input.device)


def unique_consecutive(input: Tensor, return_inverse: bool = False, return_counts: bool = False, dim=None):
    """Remove consecutive duplicate elements."""
    from vectorphile._core.ndarray import array as _nd_uc
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    unique_vals, inverse, counts = [], [], []
    prev = object()
    for v in flat:
        if v != prev:
            unique_vals.append(v)
            counts.append(0)
            prev = v
        counts[-1] += 1
        inverse.append(len(unique_vals) - 1)
    nd = _nd_uc(unique_vals, dtype='float32')
    result = Tensor(nd, device=input.device)
    if return_inverse and return_counts:
        return result, Tensor(_nd_uc(inverse, dtype='int64')), Tensor(_nd_uc(counts, dtype='int64'))
    if return_inverse:
        return result, Tensor(_nd_uc(inverse, dtype='int64'))
    if return_counts:
        return result, Tensor(_nd_uc(counts, dtype='int64'))
    return result


def kthvalue(input: Tensor, k: int, dim: int = -1, keepdim: bool = False):
    """Find the k-th smallest element along a dimension."""
    import builtins as _bi_kv
    from vectorphile._core.ndarray import array as _nd_kv
    if dim == -1:
        dim = len(input.shape) - 1
    sh = input.shape
    # Iterate slices along dim
    n = sh[dim]
    out_sh = tuple(s for i, s in enumerate(sh) if i != dim)
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    stride = 1
    for i in range(dim + 1, len(sh)):
        stride *= sh[i]
    num_slices = len(flat) // n
    vals_out, idx_out = [], []
    for slice_i in range(num_slices):
        sl = []
        for j in range(n):
            pos = (slice_i // stride) * (n * stride) + j * stride + (slice_i % stride)
            sl.append((float(flat[pos]), pos))
        sl.sort(key=lambda x: x[0])
        kth_val, kth_pos = sl[k - 1]
        vals_out.append(kth_val)
        idx_out.append(kth_pos % n)
    nd_v = _nd_kv(vals_out, dtype='float32')
    nd_i = _nd_kv(idx_out, dtype='int64')
    if out_sh:
        nd_v = nd_v.reshape(out_sh)
        nd_i = nd_i.reshape(out_sh)
    return Tensor(nd_v), Tensor(nd_i)


def mode(input: Tensor, dim: int = -1, keepdim: bool = False):
    """Find the mode (most common value) along a dimension."""
    import builtins as _bi_md
    from vectorphile._core.ndarray import array as _nd_md
    if dim == -1:
        dim = len(input.shape) - 1
    sh = input.shape
    n = sh[dim]
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    stride = 1
    for i in range(dim + 1, len(sh)):
        stride *= sh[i]
    num_slices = len(flat) // n
    out_sh = tuple(s for i, s in enumerate(sh) if i != dim)
    vals_out, idx_out = [], []
    for slice_i in range(num_slices):
        sl = {}
        for j in range(n):
            pos = (slice_i // stride) * (n * stride) + j * stride + (slice_i % stride)
            v = float(flat[pos])
            sl[v] = sl.get(v, [])
            sl[v].append(j)
        mode_val = _bi_md.max(list(sl.keys()), key=lambda x: len(sl[x]))
        vals_out.append(mode_val)
        idx_out.append(sl[mode_val][0])
    nd_v = _nd_md(vals_out, dtype='float32')
    nd_i = _nd_md(idx_out, dtype='int64')
    if out_sh:
        nd_v = nd_v.reshape(out_sh)
        nd_i = nd_i.reshape(out_sh)
    return Tensor(nd_v), Tensor(nd_i)


def scatter(input: Tensor, dim: int, index: Tensor, src) -> Tensor:
    """Scatter values from src into self at indices."""
    from vectorphile._core.ndarray import array as _nd_sc
    flat_in = list(input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else input._data)
    flat_idx = [int(float(v)) for v in (index._data.to_flat_list() if hasattr(index._data, 'to_flat_list') else list(index._data))]
    if isinstance(src, Tensor):
        flat_src = [float(v) for v in (src._data.to_flat_list() if hasattr(src._data, 'to_flat_list') else list(src._data))]
    else:
        flat_src = [float(src)] * len(flat_idx)
    sh = input.shape
    stride = 1
    for i in range(dim + 1, len(sh)):
        stride *= sh[i]
    idx_sh = index.shape
    out = list(flat_in)
    for k, (idx_v, src_v) in enumerate(zip(flat_idx, flat_src)):
        # Compute output position
        idx_pos = (k // stride) % idx_sh[dim] if len(idx_sh) > dim else k
        dim_stride = stride
        base = k - (k // stride % idx_sh[dim]) * stride + idx_v * dim_stride
        if 0 <= base < len(out):
            out[base] = src_v
    nd = _nd_sc(out, dtype='float32').reshape(sh)
    return Tensor(nd, device=input.device)


def gather(input: Tensor, dim: int, index: Tensor) -> Tensor:
    """Gather values along an axis."""
    from vectorphile._core.ndarray import array as _nd_ga
    flat_in = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    flat_idx = [int(float(v)) for v in (index._data.to_flat_list() if hasattr(index._data, 'to_flat_list') else list(index._data))]
    sh = input.shape
    idx_sh = index.shape
    stride = 1
    for i in range(dim + 1, len(sh)):
        stride *= sh[i]
    dim_size = sh[dim]
    out = []
    for k, idx_v in enumerate(flat_idx):
        # Row/col logic: replace dim position with idx_v
        base_before = (k // (stride * (idx_sh[dim] if dim < len(idx_sh) else 1))) * (stride * dim_size)
        offset = k % stride
        pos = base_before + idx_v * stride + offset
        import builtins as _bi_ga; pos = _bi_ga.min(int(pos), len(flat_in) - 1)
        out.append(float(flat_in[pos]))
    nd = _nd_ga(out, dtype='float32')
    if idx_sh:
        nd = nd.reshape(idx_sh)
    return Tensor(nd, device=input.device)


def index_put(input: Tensor, indices, values, accumulate: bool = False) -> Tensor:
    """Put values into a tensor at the given indices."""
    result = input.clone()
    if isinstance(indices, (list, tuple)):
        result[indices] = values
    else:
        result[indices] = values
    return result


def cartesian_prod(*tensors) -> Tensor:
    """Cartesian product of 1-D tensors."""
    from vectorphile._core.ndarray import array as _nd_cp
    lists = []
    for t in tensors:
        flat = t._data.to_flat_list() if hasattr(t._data, 'to_flat_list') else list(t._data)
        lists.append([float(v) for v in flat])
    result = [[]]
    for lst in lists:
        result = [r + [v] for r in result for v in lst]
    flat_result = [v for row in result for v in row]
    nd = _nd_cp(flat_result, dtype='float32').reshape((len(result), len(tensors)))
    return Tensor(nd)


def combinations(input: Tensor, r: int = 2, with_replacement: bool = False) -> Tensor:
    """Compute combinations of elements."""
    import itertools
    from vectorphile._core.ndarray import array as _nd_cb
    flat = [float(v) for v in (input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data))]
    if with_replacement:
        combos = list(itertools.combinations_with_replacement(flat, r))
    else:
        combos = list(itertools.combinations(flat, r))
    if not combos:
        return zeros(0, r)
    flat_result = [v for row in combos for v in row]
    nd = _nd_cb(flat_result, dtype='float32').reshape((len(combos), r))
    return Tensor(nd)


def cosine_similarity(x1: Tensor, x2: Tensor, dim: int = 1, eps: float = 1e-8) -> Tensor:
    """Cosine similarity between tensors."""
    import math
    from vectorphile._core.ndarray import array as _nd_cs
    # dot / (norm1 * norm2)
    dot = (x1 * x2).sum(dim=dim)
    n1 = (x1 * x1).sum(dim=dim)
    n2 = (x2 * x2).sum(dim=dim)
    flat_dot = dot._data.to_flat_list() if hasattr(dot._data, 'to_flat_list') else list(dot._data)
    flat_n1  = n1._data.to_flat_list()  if hasattr(n1._data,  'to_flat_list') else list(n1._data)
    flat_n2  = n2._data.to_flat_list()  if hasattr(n2._data,  'to_flat_list') else list(n2._data)
    result = [d / (math.sqrt(a) * math.sqrt(b) + eps) for d, a, b in zip(flat_dot, flat_n1, flat_n2)]
    nd = _nd_cs(result, dtype='float32')
    return Tensor(nd, device=x1.device)


def pairwise_distance(x1: Tensor, x2: Tensor, p: float = 2.0, eps: float = 1e-6, keepdim: bool = False) -> Tensor:
    """Pairwise distance between two tensors."""
    import math
    from vectorphile._core.ndarray import array as _nd_pd
    diff = x1 - x2
    flat = diff._data.to_flat_list() if hasattr(diff._data, 'to_flat_list') else list(diff._data)
    sh = diff.shape
    if len(sh) == 1:
        import builtins as _bi_pd1; import math as _m_pd1; dist = _bi_pd1.sum(_m_pd1.fabs(v)**p for v in flat) ** (1.0/p)
        return Tensor(_nd_pd([dist], dtype='float32'))
    # Row-wise
    n_rows = sh[0]
    n_cols = len(flat) // n_rows
    import builtins as _bi_pd; import math as _m_pd; dists = [_bi_pd.sum(_m_pd.fabs(flat[i*n_cols+j])**p for j in range(n_cols))**(1.0/p) for i in range(n_rows)]
    nd = _nd_pd(dists, dtype='float32')
    return Tensor(nd, device=x1.device)


def nan_to_num(input: Tensor, nan: float = 0.0, posinf=None, neginf=None) -> Tensor:
    """Replace NaN, +inf, -inf with specified values."""
    import math
    from vectorphile._core.ndarray import array as _nd_nn
    flat = [float(v) for v in (input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data))]
    result = []
    max_f = 3.4028235e+38
    for v in flat:
        if math.isnan(v):
            result.append(nan)
        elif math.isinf(v) and v > 0:
            result.append(posinf if posinf is not None else max_f)
        elif math.isinf(v) and v < 0:
            result.append(neginf if neginf is not None else -max_f)
        else:
            result.append(v)
    nd = _nd_nn(result, dtype=input.dtype)
    if len(input.shape) > 1:
        nd = nd.reshape(input.shape)
    return Tensor(nd, device=input.device)


def clip(input: Tensor, min=None, max=None) -> Tensor:
    """Alias for clamp."""
    return input.clamp(min=min, max=max) if min is not None or max is not None else input.clone()


def atan2(input: Tensor, other: Tensor) -> Tensor:
    """Element-wise arctangent of input/other with correct quadrant."""
    import math
    from vectorphile._core.ndarray import array as _nd_a2
    flat_y = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    flat_x = other._data.to_flat_list() if hasattr(other._data, 'to_flat_list') else list(other._data)
    result = [math.atan2(float(y), float(x)) for y, x in zip(flat_y, flat_x)]
    nd = _nd_a2(result, dtype='float32')
    if len(input.shape) > 1:
        nd = nd.reshape(input.shape)
    return Tensor(nd, device=input.device)


def hypot(input: Tensor, other: Tensor) -> Tensor:
    """Element-wise hypotenuse."""
    import math
    from vectorphile._core.ndarray import array as _nd_hy
    flat_a = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    flat_b = other._data.to_flat_list() if hasattr(other._data, 'to_flat_list') else list(other._data)
    result = [math.hypot(float(a), float(b)) for a, b in zip(flat_a, flat_b)]
    nd = _nd_hy(result, dtype='float32')
    if len(input.shape) > 1:
        nd = nd.reshape(input.shape)
    return Tensor(nd, device=input.device)


def logaddexp(input: Tensor, other: Tensor) -> Tensor:
    """log(exp(input) + exp(other)), numerically stable."""
    import math
    from vectorphile._core.ndarray import array as _nd_la
    flat_a = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    flat_b = other._data.to_flat_list() if hasattr(other._data, 'to_flat_list') else list(other._data)
    import builtins as _bi_lae
    result = [math.log(math.exp(float(a) - _bi_lae.max(float(a), float(b))) + math.exp(float(b) - _bi_lae.max(float(a), float(b)))) + _bi_lae.max(float(a), float(b))
              for a, b in zip(flat_a, flat_b)]
    nd = _nd_la(result, dtype='float32')
    if len(input.shape) > 1:
        nd = nd.reshape(input.shape)
    return Tensor(nd, device=input.device)


def logaddexp2(input: Tensor, other: Tensor) -> Tensor:
    """log2(2^input + 2^other), numerically stable."""
    import math
    from vectorphile._core.ndarray import array as _nd_l2
    flat_a = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    flat_b = other._data.to_flat_list() if hasattr(other._data, 'to_flat_list') else list(other._data)
    result = []
    for a, b in zip(flat_a, flat_b):
        a, b = float(a), float(b)
        import builtins as _bi_l2e; m = _bi_l2e.max(a, b)
        result.append(math.log2(2**(a-m) + 2**(b-m)) + m)
    nd = _nd_l2(result, dtype='float32')
    if len(input.shape) > 1:
        nd = nd.reshape(input.shape)
    return Tensor(nd, device=input.device)


def isnan(input: Tensor) -> Tensor:
    """Element-wise isnan check."""
    import math
    from vectorphile._core.ndarray import array as _nd_isn
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    result = [math.isnan(float(v)) for v in flat]
    nd = _nd_isn(result, dtype='bool')
    if len(input.shape) > 1:
        nd = nd.reshape(input.shape)
    return Tensor(nd, device=input.device)


def isinf(input: Tensor) -> Tensor:
    """Element-wise isinf check."""
    import math
    from vectorphile._core.ndarray import array as _nd_ii
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    result = [math.isinf(float(v)) for v in flat]
    nd = _nd_ii(result, dtype='bool')
    if len(input.shape) > 1:
        nd = nd.reshape(input.shape)
    return Tensor(nd, device=input.device)


def isfinite(input: Tensor) -> Tensor:
    """Element-wise isfinite check."""
    import math
    from vectorphile._core.ndarray import array as _nd_if
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    result = [math.isfinite(float(v)) for v in flat]
    nd = _nd_if(result, dtype='bool')
    if len(input.shape) > 1:
        nd = nd.reshape(input.shape)
    return Tensor(nd, device=input.device)


def isreal(input: Tensor) -> Tensor:
    """Element-wise isreal (always True for real tensors)."""
    from vectorphile._core.ndarray import array as _nd_ir
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    result = [True] * len(flat)
    nd = _nd_ir(result, dtype='bool')
    if len(input.shape) > 1:
        nd = nd.reshape(input.shape)
    return Tensor(nd, device=input.device)


# ── Trig/math module-level wrappers ─────────────────────────────────────────

def sin(input: Tensor) -> Tensor:
    """Element-wise sine."""
    return input.sin()

def cos(input: Tensor) -> Tensor:
    """Element-wise cosine."""
    return input.cos()

def tan(input: Tensor) -> Tensor:
    """Element-wise tangent."""
    return input.tan()

def asin(input: Tensor) -> Tensor:
    """Element-wise arcsin."""
    return input.asin()

def acos(input: Tensor) -> Tensor:
    """Element-wise arccos."""
    return input.acos()

def atan(input: Tensor) -> Tensor:
    """Element-wise arctan."""
    return input.atan()

def sinh(input: Tensor) -> Tensor:
    """Element-wise sinh."""
    return input.sinh()

def cosh(input: Tensor) -> Tensor:
    """Element-wise cosh."""
    return input.cosh()

def asinh(input: Tensor) -> Tensor:
    """Element-wise arcsinh."""
    return input.asinh()

def acosh(input: Tensor) -> Tensor:
    """Element-wise arccosh."""
    return input.acosh()

def atanh(input: Tensor) -> Tensor:
    """Element-wise arctanh."""
    return input.atanh()

def rsqrt(input: Tensor) -> Tensor:
    """Element-wise reciprocal square root."""
    return input.rsqrt()

def log2(input: Tensor) -> Tensor:
    """Element-wise log base 2."""
    return input.log2()

def log10(input: Tensor) -> Tensor:
    """Element-wise log base 10."""
    return input.log10()

def reshape(input: Tensor, shape) -> Tensor:
    """Reshape tensor."""
    if isinstance(shape, int):
        return input.reshape(shape)
    return input.reshape(*shape)

def argmax(input: Tensor, dim=None, keepdim: bool = False) -> Tensor:
    """Return index of maximum value."""
    return input.argmax(dim)

def argmin(input: Tensor, dim=None, keepdim: bool = False) -> Tensor:
    """Return index of minimum value."""
    return input.argmin(dim)

def unbind(input: Tensor, dim: int = 0):
    """Remove a tensor dimension, returning a tuple of slices."""
    n = input.shape[dim]
    result = []
    for i in range(n):
        idx = [slice(None)] * len(input.shape)
        idx[dim] = i
        result.append(input[tuple(idx)])
    return tuple(result)

def fmax(input: Tensor, other) -> Tensor:
    """Element-wise maximum, ignoring NaN."""
    return maximum(input, other)

def fmin(input: Tensor, other) -> Tensor:
    """Element-wise minimum, ignoring NaN."""
    return minimum(input, other)

def bernoulli(input: Tensor, p: float = None, generator=None) -> Tensor:
    """Sample Bernoulli: input treated as probability if p is None."""
    if p is not None:
        return input.bernoulli(p)
    return input.bernoulli(0.5)


# ---------------------------------------------------------------------------
# fliplr / flipud / vstack / hstack / dstack / column_stack / row_stack / addr
# ---------------------------------------------------------------------------

def fliplr(input: 'Tensor') -> 'Tensor':
    """Flip tensor left-right (reverse the last dimension)."""
    ndim = len(input._data.shape) if hasattr(input._data, 'shape') else 1
    return flip(input, [ndim - 1])


def flipud(input: 'Tensor') -> 'Tensor':
    """Flip tensor upside-down (reverse the first dimension)."""
    return flip(input, [0])


def vstack(tensors) -> 'Tensor':
    """Stack tensors vertically (row-wise, axis 0). Alias for torch.vstack."""
    return cat([t.unsqueeze(0) if t._data.ndim == 1 else t for t in tensors], dim=0)


def hstack(tensors) -> 'Tensor':
    """Stack tensors horizontally (column-wise, axis 1 for 2-D, axis 0 for 1-D)."""
    if tensors[0]._data.ndim == 1:
        return cat(list(tensors), dim=0)
    return cat(list(tensors), dim=1)


def dstack(tensors) -> 'Tensor':
    """Stack tensors depth-wise (axis 2). Equivalent to concat along 3rd dim."""
    promoted = []
    for t in tensors:
        d = t._data.ndim
        if d == 1:
            promoted.append(t.reshape(1, -1, 1))
        elif d == 2:
            sh = t._data.shape
            promoted.append(t.reshape(sh[0], sh[1], 1))
        else:
            promoted.append(t)
    return cat(promoted, dim=2)


def column_stack(tensors) -> 'Tensor':
    """Stack 1-D tensors as columns into a 2-D tensor; 2-D tensors along dim 1."""
    promoted = []
    for t in tensors:
        if t._data.ndim == 1:
            promoted.append(t.reshape(-1, 1))
        else:
            promoted.append(t)
    return cat(promoted, dim=1)


def row_stack(tensors) -> 'Tensor':
    """Alias for vstack."""
    return vstack(tensors)


def addr(input: 'Tensor', vec1: 'Tensor', vec2: 'Tensor',
         beta: float = 1.0, alpha: float = 1.0) -> 'Tensor':
    """Outer product of vec1 and vec2 added to input (like torch.addr)."""
    outer_v = outer(vec1, vec2)
    flat_in = list(input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else input._data)
    flat_out = list(outer_v._data.to_flat_list() if hasattr(outer_v._data, 'to_flat_list') else outer_v._data)
    result = [beta * a + alpha * b for a, b in zip(flat_in, flat_out)]
    import array as _arr
    from vectorphile._core.ndarray import NDArray as _ND, _c_strides, _DType
    sh = outer_v._data.shape
    return Tensor(_ND(_arr.array('f', result), sh, _c_strides(sh), _DType('float32')))


def index_copy(input: 'Tensor', dim: int, index: 'Tensor', source: 'Tensor') -> 'Tensor':
    """Copy elements of source into input at positions given by index along dim."""
    import array as _arr
    from vectorphile._core.ndarray import NDArray as _ND, _c_strides, _DType
    sh = input._data.shape if hasattr(input._data, 'shape') else (len(list(input._data)),)
    flat_in = list(input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else input._data)
    flat_src = list(source._data.to_flat_list() if hasattr(source._data, 'to_flat_list') else source._data)
    flat_idx = list(index._data.to_flat_list() if hasattr(index._data, 'to_flat_list') else index._data)
    result = flat_in[:]
    ndim = len(sh)
    dim = dim % ndim
    # stride along dim
    stride = 1
    for i in range(dim + 1, ndim):
        stride *= sh[i]
    block = 1
    for i in range(dim + 1, ndim):
        block *= sh[i]
    # outer blocks before dim
    outer_size = 1
    for i in range(dim):
        outer_size *= sh[i]
    for i_idx, src_idx in enumerate(flat_idx):
        src_idx = int(src_idx)
        for outer in range(outer_size):
            for inner in range(block):
                dst_pos = outer * sh[dim] * block + src_idx * block + inner
                src_pos = outer * len(flat_idx) * block + i_idx * block + inner
                result[dst_pos] = flat_src[src_pos]
    return Tensor(_ND(_arr.array('f', result), sh, _c_strides(sh), _DType('float32')))


# ---------------------------------------------------------------------------
# block_diag / diag_embed / scatter_add (module-level)
# ---------------------------------------------------------------------------

def block_diag(*tensors: 'Tensor') -> 'Tensor':
    """Create block-diagonal matrix from input tensors."""
    import array as _arr
    from vectorphile._core.ndarray import NDArray as _ND, _c_strides, _DType
    mats = []
    for t in tensors:
        sh = t._data.shape if hasattr(t._data, 'shape') else (len(list(t._data)),)
        if len(sh) == 1:
            n = sh[0]
            fl = list(t._data.to_flat_list() if hasattr(t._data, 'to_flat_list') else t._data)
            # 1-D => diagonal matrix
            m = [[fl[i] if i == j else 0.0 for j in range(n)] for i in range(n)]
            mats.append((n, n, [v for row in m for v in row]))
        elif len(sh) == 2:
            fl = list(t._data.to_flat_list() if hasattr(t._data, 'to_flat_list') else t._data)
            mats.append((sh[0], sh[1], fl))
        else:
            raise ValueError("block_diag: inputs must be 1-D or 2-D")
    import builtins as _bi_bd
    total_rows = _bi_bd.sum(r for r, c, _ in mats)
    total_cols = _bi_bd.sum(c for r, c, _ in mats)
    out = [0.0] * (total_rows * total_cols)
    row_off = 0
    col_off = 0
    for r, c, fl in mats:
        for i in range(r):
            for j in range(c):
                out[(row_off + i) * total_cols + (col_off + j)] = fl[i * c + j]
        row_off += r
        col_off += c
    sh = (total_rows, total_cols)
    return Tensor(_ND(_arr.array("f", out), sh, _c_strides(sh), _DType("float32")))


def diag_embed(input: 'Tensor', offset: int = 0, dim1: int = -2, dim2: int = -1) -> 'Tensor':
    """Create a tensor with input along the specified diagonal."""
    import array as _arr
    from vectorphile._core.ndarray import NDArray as _ND, _c_strides, _DType
    sh = input._data.shape if hasattr(input._data, 'shape') else (len(list(input._data)),)
    flat = list(input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else input._data)
    # For 1-D input of length n, produce (n+|offset|, n+|offset|) diagonal matrix
    ndim_in = len(sh)
    n = sh[-1]  # size of last dim
    import builtins as _bi_de
    diag_size = n + _bi_de.abs(offset)
    # batch dims = sh[:-1]
    batch = sh[:-1] if ndim_in > 1 else ()
    batch_size = 1
    for b in batch:
        batch_size *= b
    out_sh = batch + (diag_size, diag_size)
    out = [0.0] * (batch_size * diag_size * diag_size)
    for b in range(batch_size):
        for i in range(n):
            r = i if offset >= 0 else i - offset
            c = i + offset if offset >= 0 else i
            pos = b * diag_size * diag_size + r * diag_size + c
            out[pos] = flat[b * n + i]
    return Tensor(_ND(_arr.array("f", out), out_sh, _c_strides(out_sh), _DType("float32")))


def scatter_add(input: 'Tensor', dim: int, index: 'Tensor', src: 'Tensor') -> 'Tensor':
    """Out-of-place scatter_add: adds src values to input at index positions."""
    import array as _arr
    from vectorphile._core.ndarray import NDArray as _ND, _c_strides, _DType
    sh = input._data.shape if hasattr(input._data, 'shape') else (len(list(input._data)),)
    flat_in = list(input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else input._data)
    flat_idx = list(index._data.to_flat_list() if hasattr(index._data, 'to_flat_list') else index._data)
    flat_src = list(src._data.to_flat_list() if hasattr(src._data, 'to_flat_list') else src._data)
    out = flat_in[:]
    idx_sh = index._data.shape if hasattr(index._data, 'shape') else (len(flat_idx),)
    ndim = len(sh)
    dim = dim % ndim
    # stride along dim in output
    stride = 1
    for i in range(dim + 1, ndim):
        stride *= sh[i]
    for flat_pos, (idx_val, src_val) in enumerate(zip(flat_idx, flat_src)):
        # decode flat_pos into multi-index
        idx_val = int(idx_val)
        multi = []
        rem = flat_pos
        for i in range(len(idx_sh) - 1, -1, -1):
            multi.insert(0, rem % idx_sh[i])
            rem //= idx_sh[i]
        # replace dimension dim with idx_val
        multi[dim] = idx_val
        # encode multi-index into output flat position
        out_pos = 0
        for i, m in enumerate(multi):
            s = 1
            for j in range(i + 1, ndim):
                s *= sh[j]
            out_pos += m * s
        out[out_pos] += src_val
    return Tensor(_ND(_arr.array("f", out), sh, _c_strides(sh), _DType("float32")))


def index_select(input: 'Tensor', dim: int, index: 'Tensor') -> 'Tensor':
    """Select slices of input along dim at the positions given by index."""
    import array as _arr_is
    from vectorphile._core.ndarray import NDArray as _ND_is, _c_strides as _cs_is, _DType as _DT_is
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    sh = input._data.shape if hasattr(input._data, 'shape') else (len(flat),)
    ndim = len(sh)
    if dim < 0:
        dim = ndim + dim
    idx_flat = index._data.to_flat_list() if hasattr(index._data, 'to_flat_list') else list(index._data)
    idx_list = [int(v) for v in idx_flat]

    inner = 1
    for i in range(dim + 1, ndim):
        inner *= sh[i]
    outer = 1
    for i in range(dim):
        outer *= sh[i]
    dim_size = sh[dim]

    out = []
    for o in range(outer):
        for idx in idx_list:
            for i in range(inner):
                pos = o * dim_size * inner + idx * inner + i
                out.append(float(flat[pos]))

    new_sh = sh[:dim] + (len(idx_list),) + sh[dim+1:]
    return Tensor(_ND_is(_arr_is.array("f", out), new_sh, _cs_is(new_sh), _DT_is("float32")))


def bucketize(input: 'Tensor', boundaries: 'Tensor', out_int32: bool = False,
              right: bool = False) -> 'Tensor':
    """Return bucket indices for each value in input given boundaries."""
    import bisect
    from vectorphile._core.ndarray import array as _nd_bk
    flat = input._data.to_flat_list() if hasattr(input._data, 'to_flat_list') else list(input._data)
    sh = input._data.shape if hasattr(input._data, 'shape') else (len(flat),)
    b_flat = boundaries._data.to_flat_list() if hasattr(boundaries._data, 'to_flat_list') else list(boundaries._data)
    bounds = [float(v) for v in b_flat]
    if right:
        out = [bisect.bisect_right(bounds, float(v)) for v in flat]
    else:
        out = [bisect.bisect_left(bounds, float(v)) for v in flat]
    dtype = 'int32' if out_int32 else 'int64'
    nd = _nd_bk(out, dtype=dtype)
    if sh and sh != (len(out),):
        nd = nd.reshape(sh)
    return Tensor(nd)

