"""Core Tensor class — the foundation of vectorphile."""
from __future__ import annotations
import numpy as np
from typing import Any, Optional, Tuple, Union, List


_DTYPE_MAP = {
    'float32': np.float32, 'float': np.float32,
    'float64': np.float64, 'double': np.float64,
    'float16': np.float16, 'half': np.float16,
    'int32': np.int32, 'int': np.int32,
    'int64': np.int64, 'long': np.int64,
    'int8': np.int8, 'uint8': np.uint8,
    'bool': np.bool_,
}


class Tensor:
    """N-dimensional tensor backed by numpy, with autograd support."""

    def __init__(self, data, dtype=None, device='cpu', requires_grad=False):
        if isinstance(data, Tensor):
            data = data._data
        if isinstance(data, np.ndarray):
            self._data = data.astype(_DTYPE_MAP.get(dtype, data.dtype)) if dtype else data.copy()
        else:
            np_dtype = _DTYPE_MAP.get(dtype, np.float32) if dtype else None
            self._data = np.array(data, dtype=np_dtype)
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

    @property
    def dtype(self) -> str:
        return str(self._data.dtype)

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def T(self) -> 'Tensor':
        return self.transpose()

    @property
    def data(self) -> np.ndarray:
        return self._data

    # ── creation helpers (class methods) ───────────────────────────────────

    @classmethod
    def zeros(cls, *shape, dtype='float32', device='cpu', requires_grad=False) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return cls(np.zeros(shape, dtype=_DTYPE_MAP[dtype]), device=device, requires_grad=requires_grad)

    @classmethod
    def ones(cls, *shape, dtype='float32', device='cpu', requires_grad=False) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return cls(np.ones(shape, dtype=_DTYPE_MAP[dtype]), device=device, requires_grad=requires_grad)

    @classmethod
    def eye(cls, n, m=None, dtype='float32', device='cpu') -> 'Tensor':
        return cls(np.eye(n, m, dtype=_DTYPE_MAP[dtype]), device=device)

    @classmethod
    def arange(cls, start, stop=None, step=1, dtype='float32', device='cpu') -> 'Tensor':
        if stop is None:
            start, stop = 0, start
        return cls(np.arange(start, stop, step, dtype=_DTYPE_MAP[dtype]), device=device)

    @classmethod
    def linspace(cls, start, stop, num, dtype='float32', device='cpu') -> 'Tensor':
        return cls(np.linspace(start, stop, num, dtype=_DTYPE_MAP[dtype]), device=device)

    @classmethod
    def full(cls, shape, fill_value, dtype='float32', device='cpu') -> 'Tensor':
        return cls(np.full(shape, fill_value, dtype=_DTYPE_MAP[dtype]), device=device)

    @classmethod
    def empty(cls, *shape, dtype='float32', device='cpu') -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return cls(np.empty(shape, dtype=_DTYPE_MAP[dtype]), device=device)

    @classmethod
    def rand(cls, *shape, device='cpu', requires_grad=False) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return cls(np.random.rand(*shape).astype(np.float32), device=device, requires_grad=requires_grad)

    @classmethod
    def randn(cls, *shape, device='cpu', requires_grad=False) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return cls(np.random.randn(*shape).astype(np.float32), device=device, requires_grad=requires_grad)

    @classmethod
    def randint(cls, low, high, shape, dtype='int64', device='cpu') -> 'Tensor':
        return cls(np.random.randint(low, high, shape).astype(_DTYPE_MAP[dtype]), device=device)

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'Tensor':
        return cls(arr)

    # ── type casting ────────────────────────────────────────────────────────

    def float(self) -> 'Tensor':
        return Tensor(self._data.astype(np.float32), device=self.device)

    def double(self) -> 'Tensor':
        return Tensor(self._data.astype(np.float64), device=self.device)

    def half(self) -> 'Tensor':
        return Tensor(self._data.astype(np.float16), device=self.device)

    def int(self) -> 'Tensor':
        return Tensor(self._data.astype(np.int32), device=self.device)

    def long(self) -> 'Tensor':
        return Tensor(self._data.astype(np.int64), device=self.device)

    def bool(self) -> 'Tensor':
        return Tensor(self._data.astype(np.bool_), device=self.device)

    def to(self, device=None, dtype=None) -> 'Tensor':
        data = self._data
        if dtype is not None:
            data = data.astype(_DTYPE_MAP.get(dtype, data.dtype))
        return Tensor(data, device=device or self.device, requires_grad=self.requires_grad)

    def numpy(self) -> np.ndarray:
        return self._data.copy()

    def item(self):
        return self._data.item()

    def tolist(self):
        return self._data.tolist()

    def detach(self) -> 'Tensor':
        t = Tensor(self._data, device=self.device)
        t.requires_grad = False
        return t

    def clone(self) -> 'Tensor':
        return Tensor(self._data.copy(), device=self.device, requires_grad=self.requires_grad)

    # ── shape ops ───────────────────────────────────────────────────────────

    def reshape(self, *shape) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return Tensor(self._data.reshape(shape), device=self.device, requires_grad=self.requires_grad)

    def view(self, *shape) -> 'Tensor':
        return self.reshape(*shape)

    def transpose(self, dim0=None, dim1=None) -> 'Tensor':
        if dim0 is None and dim1 is None:
            return Tensor(self._data.T, device=self.device, requires_grad=self.requires_grad)
        axes = list(range(self.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return Tensor(np.transpose(self._data, axes), device=self.device, requires_grad=self.requires_grad)

    def permute(self, *dims) -> 'Tensor':
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = dims[0]
        return Tensor(np.transpose(self._data, dims), device=self.device, requires_grad=self.requires_grad)

    def squeeze(self, dim=None) -> 'Tensor':
        if dim is None:
            return Tensor(np.squeeze(self._data), device=self.device, requires_grad=self.requires_grad)
        return Tensor(np.squeeze(self._data, axis=dim), device=self.device, requires_grad=self.requires_grad)

    def unsqueeze(self, dim) -> 'Tensor':
        return Tensor(np.expand_dims(self._data, axis=dim), device=self.device, requires_grad=self.requires_grad)

    def flatten(self, start_dim=0, end_dim=-1) -> 'Tensor':
        shape = self._data.shape
        if end_dim < 0:
            end_dim = self.ndim + end_dim
        new_shape = shape[:start_dim] + (-1,) + shape[end_dim+1:]
        return Tensor(self._data.reshape(new_shape), device=self.device, requires_grad=self.requires_grad)

    def expand(self, *shape) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return Tensor(np.broadcast_to(self._data, shape).copy(), device=self.device, requires_grad=self.requires_grad)

    def contiguous(self) -> 'Tensor':
        return Tensor(np.ascontiguousarray(self._data), device=self.device, requires_grad=self.requires_grad)

    def chunk(self, chunks, dim=0) -> List['Tensor']:
        return [Tensor(a, device=self.device) for a in np.array_split(self._data, chunks, axis=dim)]

    def split(self, split_size, dim=0) -> List['Tensor']:
        indices = list(range(split_size, self.shape[dim], split_size))
        return [Tensor(a, device=self.device) for a in np.split(self._data, indices, axis=dim)]

    def repeat(self, *repeats) -> 'Tensor':
        if len(repeats) == 1 and isinstance(repeats[0], (tuple, list)):
            repeats = repeats[0]
        return Tensor(np.tile(self._data, repeats), device=self.device)

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
        return Tensor(np.linalg.norm(self._data, ord=p, axis=dim, keepdims=keepdim), device=self.device)

    def prod(self, dim=None, keepdim=False) -> 'Tensor':
        return Tensor(self._data.prod(axis=dim, keepdims=keepdim), device=self.device)

    def cumsum(self, dim) -> 'Tensor':
        return Tensor(self._data.cumsum(axis=dim), device=self.device)

    def cumprod(self, dim) -> 'Tensor':
        return Tensor(self._data.cumprod(axis=dim), device=self.device)

    # ── element-wise math ───────────────────────────────────────────────────

    def exp(self) -> 'Tensor':
        return Tensor(np.exp(self._data), device=self.device, requires_grad=self.requires_grad)

    def log(self) -> 'Tensor':
        return Tensor(np.log(self._data), device=self.device, requires_grad=self.requires_grad)

    def log2(self) -> 'Tensor':
        return Tensor(np.log2(self._data), device=self.device)

    def log10(self) -> 'Tensor':
        return Tensor(np.log10(self._data), device=self.device)

    def sqrt(self) -> 'Tensor':
        return Tensor(np.sqrt(self._data), device=self.device, requires_grad=self.requires_grad)

    def abs(self) -> 'Tensor':
        return Tensor(np.abs(self._data), device=self.device, requires_grad=self.requires_grad)

    def sign(self) -> 'Tensor':
        return Tensor(np.sign(self._data), device=self.device)

    def sin(self) -> 'Tensor':
        return Tensor(np.sin(self._data), device=self.device)

    def cos(self) -> 'Tensor':
        return Tensor(np.cos(self._data), device=self.device)

    def tan(self) -> 'Tensor':
        return Tensor(np.tan(self._data), device=self.device)

    def tanh(self) -> 'Tensor':
        from vectorphile.autograd.engine import TanhBackward
        out_data = np.tanh(self._data)
        result = Tensor(out_data, device=self.device, requires_grad=self.requires_grad)
        if self.requires_grad:
            result.grad_fn = TanhBackward(self, result)
            result._is_leaf = False
        return result

    def sigmoid(self) -> 'Tensor':
        from vectorphile.autograd.engine import SigmoidBackward
        out_data = 1.0 / (1.0 + np.exp(-self._data))
        result = Tensor(out_data, device=self.device, requires_grad=self.requires_grad)
        if self.requires_grad:
            result.grad_fn = SigmoidBackward(self, result)
            result._is_leaf = False
        return result

    def relu(self) -> 'Tensor':
        from vectorphile.autograd.engine import ReLUBackward
        result = Tensor(np.maximum(0, self._data), device=self.device, requires_grad=self.requires_grad)
        if self.requires_grad:
            result.grad_fn = ReLUBackward(self)
            result._is_leaf = False
        return result

    def clamp(self, min=None, max=None) -> 'Tensor':
        return Tensor(np.clip(self._data, min, max), device=self.device)

    def clip(self, min=None, max=None) -> 'Tensor':
        return self.clamp(min, max)

    def floor(self) -> 'Tensor':
        return Tensor(np.floor(self._data), device=self.device)

    def ceil(self) -> 'Tensor':
        return Tensor(np.ceil(self._data), device=self.device)

    def round(self, decimals=0) -> 'Tensor':
        return Tensor(np.round(self._data, decimals), device=self.device)

    def pow(self, exponent) -> 'Tensor':
        exp = exponent._data if isinstance(exponent, Tensor) else exponent
        return Tensor(np.power(self._data, exp), device=self.device, requires_grad=self.requires_grad)

    # ── matrix ops ──────────────────────────────────────────────────────────

    def matmul(self, other: 'Tensor') -> 'Tensor':
        return Tensor(np.matmul(self._data, other._data), device=self.device, requires_grad=self.requires_grad or other.requires_grad)

    def mm(self, other: 'Tensor') -> 'Tensor':
        return self.matmul(other)

    def bmm(self, other: 'Tensor') -> 'Tensor':
        return Tensor(np.matmul(self._data, other._data), device=self.device)

    def dot(self, other: 'Tensor') -> 'Tensor':
        return Tensor(np.dot(self._data, other._data), device=self.device)

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

    # ── operators ───────────────────────────────────────────────────────────

    def _coerce(self, other):
        return other._data if isinstance(other, Tensor) else other

    def __add__(self, other) -> 'Tensor':
        from vectorphile.autograd.engine import AddBackward
        result = Tensor(self._data + self._coerce(other), device=self.device,
                        requires_grad=self.requires_grad or (isinstance(other, Tensor) and other.requires_grad))
        if result.requires_grad:
            result.grad_fn = AddBackward(self, other)
            result._is_leaf = False
        return result

    def __radd__(self, other) -> 'Tensor':
        return self.__add__(other)

    def __sub__(self, other) -> 'Tensor':
        from vectorphile.autograd.engine import SubBackward
        result = Tensor(self._data - self._coerce(other), device=self.device,
                        requires_grad=self.requires_grad or (isinstance(other, Tensor) and other.requires_grad))
        if result.requires_grad:
            result.grad_fn = SubBackward(self, other)
            result._is_leaf = False
        return result

    def __rsub__(self, other) -> 'Tensor':
        return Tensor(self._coerce(other) - self._data, device=self.device)

    def __mul__(self, other) -> 'Tensor':
        from vectorphile.autograd.engine import MulBackward
        result = Tensor(self._data * self._coerce(other), device=self.device,
                        requires_grad=self.requires_grad or (isinstance(other, Tensor) and other.requires_grad))
        if result.requires_grad:
            result.grad_fn = MulBackward(self, other)
            result._is_leaf = False
        return result

    def __rmul__(self, other) -> 'Tensor':
        return self.__mul__(other)

    def __truediv__(self, other) -> 'Tensor':
        from vectorphile.autograd.engine import DivBackward
        result = Tensor(self._data / self._coerce(other), device=self.device,
                        requires_grad=self.requires_grad or (isinstance(other, Tensor) and other.requires_grad))
        if result.requires_grad:
            result.grad_fn = DivBackward(self, other)
            result._is_leaf = False
        return result

    def __rtruediv__(self, other) -> 'Tensor':
        return Tensor(self._coerce(other) / self._data, device=self.device)

    def __pow__(self, exponent) -> 'Tensor':
        exp = self._coerce(exponent)
        return Tensor(np.power(self._data, exp), device=self.device, requires_grad=self.requires_grad)

    def __neg__(self) -> 'Tensor':
        return Tensor(-self._data, device=self.device, requires_grad=self.requires_grad)

    def __abs__(self) -> 'Tensor':
        return self.abs()

    def __matmul__(self, other) -> 'Tensor':
        from vectorphile.autograd.engine import MatMulBackward
        result = Tensor(np.matmul(self._data, other._data), device=self.device,
                        requires_grad=self.requires_grad or other.requires_grad)
        if result.requires_grad:
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
        return Tensor(self._data[idx], device=self.device)

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
        return f"tensor({self._data}, dtype={self.dtype}, device='{self.device}')"

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self):
        return id(self)

    # ── autograd ────────────────────────────────────────────────────────────

    def backward(self, gradient=None):
        from vectorphile.autograd.engine import backward
        if gradient is None:
            gradient = Tensor(np.ones_like(self._data), device=self.device)
        backward(self, gradient)

    def zero_grad(self):
        self.grad = None

    # ── utility ─────────────────────────────────────────────────────────────

    def cpu(self) -> 'Tensor':
        return Tensor(self._data, device='cpu', requires_grad=self.requires_grad)

    def cuda(self) -> 'Tensor':
        # Falls back to CPU gracefully (Phase 2 will add real CUDA)
        return Tensor(self._data, device='cuda', requires_grad=self.requires_grad)

    def is_cuda(self) -> bool:
        return self.device == 'cuda'

    def is_leaf(self) -> bool:
        return self._is_leaf

    def fill_(self, value) -> 'Tensor':
        self._data.fill(value)
        return self

    def copy_(self, src: 'Tensor') -> 'Tensor':
        self._data[:] = src._data
        return self

    def masked_fill(self, mask: 'Tensor', value: float) -> 'Tensor':
        result = self._data.copy()
        result[mask._data] = value
        return Tensor(result, device=self.device)

    def where(self, condition: 'Tensor', other: 'Tensor') -> 'Tensor':
        return Tensor(np.where(condition._data, self._data, other._data), device=self.device)


# ── functional tensor creation ───────────────────────────────────────────────

def tensor(data, dtype=None, device='cpu', requires_grad=False) -> Tensor:
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

def zeros(*shape, dtype='float32', device='cpu', requires_grad=False) -> Tensor:
    return Tensor.zeros(*shape, dtype=dtype, device=device, requires_grad=requires_grad)

def ones(*shape, dtype='float32', device='cpu', requires_grad=False) -> Tensor:
    return Tensor.ones(*shape, dtype=dtype, device=device, requires_grad=requires_grad)

def eye(n, m=None, dtype='float32', device='cpu') -> Tensor:
    return Tensor.eye(n, m, dtype=dtype, device=device)

def arange(start, stop=None, step=1, dtype='float32', device='cpu') -> Tensor:
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

def from_numpy(arr: np.ndarray) -> Tensor:
    return Tensor.from_numpy(arr)

def zeros_like(t: Tensor, requires_grad=False) -> Tensor:
    return Tensor(np.zeros_like(t._data), device=t.device, requires_grad=requires_grad)

def ones_like(t: Tensor, requires_grad=False) -> Tensor:
    return Tensor(np.ones_like(t._data), device=t.device, requires_grad=requires_grad)

def empty_like(t: Tensor) -> Tensor:
    return Tensor(np.empty_like(t._data), device=t.device)

def cat(tensors, dim=0) -> Tensor:
    arrays = [t._data for t in tensors]
    return Tensor(np.concatenate(arrays, axis=dim), device=tensors[0].device)

def stack(tensors, dim=0) -> Tensor:
    arrays = [t._data for t in tensors]
    return Tensor(np.stack(arrays, axis=dim), device=tensors[0].device)

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

def dot(a: Tensor, b: Tensor) -> Tensor:
    return a.dot(b)

def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return Tensor(np.where(condition._data, x._data, y._data), device=x.device)

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
    data = input._data
    shifted = data - data.max(axis=dim, keepdims=True)
    e = np.exp(shifted)
    return Tensor(e / e.sum(axis=dim, keepdims=True), device=input.device)

def log_softmax(input: Tensor, dim=-1) -> Tensor:
    data = input._data
    shifted = data - data.max(axis=dim, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted).sum(axis=dim, keepdims=True))
    return Tensor(shifted - log_sum_exp, device=input.device)

def sigmoid(input: Tensor) -> Tensor:
    return input.sigmoid()

def relu(input: Tensor) -> Tensor:
    return input.relu()

def tanh(input: Tensor) -> Tensor:
    return input.tanh()

def einsum(equation: str, *operands) -> Tensor:
    arrays = [op._data for op in operands]
    return Tensor(np.einsum(equation, *arrays), device=operands[0].device)
