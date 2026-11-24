"""Masked arrays — pure Python implementation, no external dependencies."""
from __future__ import annotations
import math
from typing import Any, List, Optional, Union


_MASKED = object()  # sentinel for masked values
masked = _MASKED
nomask = False


class MaskedArray:
    """A list-based masked array."""

    def __init__(self, data, mask=None, fill_value=None):
        if hasattr(data, 'to_flat_list'):
            self._data = data.to_flat_list()
        elif hasattr(data, 'tolist'):
            d = data.tolist()
            self._data = [d] if not isinstance(d, list) else d
        else:
            self._data = list(data)
        n = len(self._data)
        if mask is None or mask is False:
            self._mask = [False] * n
        elif mask is True:
            self._mask = [True] * n
        elif isinstance(mask, (list, tuple)):
            self._mask = [bool(m) for m in mask]
        else:
            self._mask = [False] * n
        self.fill_value = fill_value if fill_value is not None else 0.0

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if self._mask[idx]:
            return _MASKED
        return self._data[idx]

    def _valid(self):
        return [v for v, m in zip(self._data, self._mask) if not m]

    def filled(self, fill_value=None):
        fv = fill_value if fill_value is not None else self.fill_value
        return [fv if m else v for v, m in zip(self._data, self._mask)]

    def compressed(self):
        return self._valid()

    def count(self):
        return sum(1 for m in self._mask if not m)

    def mean(self):
        v = self._valid()
        return sum(float(x) for x in v) / max(len(v), 1)

    def sum(self):
        return sum(float(x) for x in self._valid())

    def min(self):
        v = self._valid()
        return min(float(x) for x in v) if v else _MASKED

    def max(self):
        v = self._valid()
        return max(float(x) for x in v) if v else _MASKED

    def var(self, ddof=0):
        v = [float(x) for x in self._valid()]
        n = len(v)
        if n <= ddof:
            return 0.0
        mu = sum(v) / n
        return sum((x - mu)**2 for x in v) / (n - ddof)

    def std(self, ddof=0):
        return math.sqrt(self.var(ddof))

    def __repr__(self):
        items = ['--' if m else str(v) for v, m in zip(self._data, self._mask)]
        return f'MaskedArray([{", ".join(items)}])'


def masked_array(data, mask=None, dtype=None, copy=False, fill_value=None, **kwargs):
    return MaskedArray(data, mask=mask, fill_value=fill_value)


def masked_where(condition, a):
    if isinstance(a, MaskedArray):
        data, old_mask = a._data, a._mask
    else:
        data = list(a) if hasattr(a, '__iter__') else [a]
        old_mask = [False] * len(data)
    if isinstance(condition, (list, tuple)):
        new_mask = [bool(c) or old_mask[i] for i, c in enumerate(condition)]
    else:
        new_mask = [bool(condition)] * len(data)
    return MaskedArray(data, mask=new_mask)


def masked_equal(a, value):
    data = list(a._data if isinstance(a, MaskedArray) else a)
    return masked_where([v == value for v in data], a)


def masked_not_equal(a, value):
    data = list(a._data if isinstance(a, MaskedArray) else a)
    return masked_where([v != value for v in data], a)


def masked_less(a, value):
    data = list(a._data if isinstance(a, MaskedArray) else a)
    return masked_where([float(v) < float(value) for v in data], a)


def masked_less_equal(a, value):
    data = list(a._data if isinstance(a, MaskedArray) else a)
    return masked_where([float(v) <= float(value) for v in data], a)


def masked_greater(a, value):
    data = list(a._data if isinstance(a, MaskedArray) else a)
    return masked_where([float(v) > float(value) for v in data], a)


def masked_greater_equal(a, value):
    data = list(a._data if isinstance(a, MaskedArray) else a)
    return masked_where([float(v) >= float(value) for v in data], a)


def masked_inside(a, v1, v2):
    data = list(a._data if isinstance(a, MaskedArray) else a)
    return masked_where([float(v1) <= float(v) <= float(v2) for v in data], a)


def masked_outside(a, v1, v2):
    data = list(a._data if isinstance(a, MaskedArray) else a)
    return masked_where([not (float(v1) <= float(v) <= float(v2)) for v in data], a)


def masked_values(a, value, rtol=1e-5, atol=1e-8):
    data = list(a._data if isinstance(a, MaskedArray) else a)
    return masked_where([abs(float(v) - float(value)) <= atol + rtol * abs(float(value)) for v in data], a)


def masked_invalid(a):
    data = list(a._data if isinstance(a, MaskedArray) else a)
    def _is_invalid(v):
        try:
            f = float(v)
            return math.isnan(f) or math.isinf(f)
        except (TypeError, ValueError):
            return True
    return masked_where([_is_invalid(v) for v in data], a)


def masked_object(a, value):
    data = list(a._data if isinstance(a, MaskedArray) else a)
    return masked_where([v is value for v in data], a)


def getdata(a):
    return a._data if isinstance(a, MaskedArray) else list(a)


def getmask(a):
    return a._mask if isinstance(a, MaskedArray) else [False] * len(a)


def getmaskarray(a):
    return getmask(a)


def filled(a, fill_value=None):
    if isinstance(a, MaskedArray):
        return a.filled(fill_value)
    return list(a)


def compressed(a):
    if isinstance(a, MaskedArray):
        return a.compressed()
    return list(a)


def count(a, axis=None):
    if isinstance(a, MaskedArray):
        return a.count()
    return len(a)


def make_mask(m, copy=False, shrink=True):
    return [bool(v) for v in m]


def make_mask_none(shape):
    if isinstance(shape, int):
        return [False] * shape
    n = 1
    for s in shape:
        n *= s
    return [False] * n


def mask_or(m1, m2):
    return [a or b for a, b in zip(m1, m2)]


def _apply_op(op, a, b=None):
    if b is None:
        data = getdata(a)
        mask = getmask(a)
        result = []
        for v, m in zip(data, mask):
            if m:
                result.append(0.0)
            else:
                try:
                    result.append(op(float(v)))
                except (ValueError, ZeroDivisionError):
                    result.append(0.0)
                    mask = list(mask)
                    mask[len(result)-1] = True
        return MaskedArray(result, mask=mask)
    da, db = getdata(a), getdata(b)
    ma, mb = getmask(a), getmask(b)
    result = []
    new_mask = []
    for i in range(min(len(da), len(db))):
        m = ma[i] or mb[i]
        if m:
            result.append(0.0)
            new_mask.append(True)
        else:
            try:
                result.append(op(float(da[i]), float(db[i])))
                new_mask.append(False)
            except (ValueError, ZeroDivisionError):
                result.append(0.0)
                new_mask.append(True)
    return MaskedArray(result, mask=new_mask)


def add(a, b): return _apply_op(lambda x, y: x + y, a, b)
def subtract(a, b): return _apply_op(lambda x, y: x - y, a, b)
def multiply(a, b): return _apply_op(lambda x, y: x * y, a, b)
def divide(a, b): return _apply_op(lambda x, y: x / y if y != 0 else float('nan'), a, b)
def power(a, b): return _apply_op(lambda x, y: x ** y, a, b)
def sqrt(a): return _apply_op(lambda x: math.sqrt(max(x, 0)), a)
def log(a): return _apply_op(lambda x: math.log(x) if x > 0 else float('nan'), a)
def exp(a): return _apply_op(math.exp, a)
def sin(a): return _apply_op(math.sin, a)
def cos(a): return _apply_op(math.cos, a)
def arcsin(a): return _apply_op(lambda x: math.asin(max(-1, min(1, x))), a)
def arccos(a): return _apply_op(lambda x: math.acos(max(-1, min(1, x))), a)
def arctan(a): return _apply_op(math.atan, a)
def arctan2(a, b): return _apply_op(math.atan2, a, b)
def tan(a): return _apply_op(math.tan, a)
def hypot(a, b): return _apply_op(math.hypot, a, b)
def conjugate(a): return a
def absolute(a): return _apply_op(abs, a)


def mean(a, axis=None):
    if isinstance(a, MaskedArray):
        return a.mean()
    vals = [float(v) for v in a]
    return sum(vals) / max(len(vals), 1)


def median(a):
    if isinstance(a, MaskedArray):
        v = sorted(float(x) for x in a._valid())
    else:
        v = sorted(float(x) for x in a)
    n = len(v)
    return v[n//2] if n % 2 == 1 else (v[n//2-1] + v[n//2]) / 2.0 if n > 0 else 0.0


def var(a, ddof=0):
    if isinstance(a, MaskedArray):
        return a.var(ddof)
    vals = [float(v) for v in a]
    mu = sum(vals) / max(len(vals), 1)
    return sum((v - mu)**2 for v in vals) / max(len(vals) - ddof, 1)


def std(a, ddof=0):
    return math.sqrt(var(a, ddof))


def sum(a, axis=None):
    if isinstance(a, MaskedArray):
        return a.sum()
    return builtins_sum(float(v) for v in a)


import builtins as _builtins
builtins_sum = _builtins.sum


def prod(a):
    result = 1.0
    for v in (a._valid() if isinstance(a, MaskedArray) else a):
        result *= float(v)
    return result


def min(a):
    if isinstance(a, MaskedArray):
        return a.min()
    return _builtins.min(float(v) for v in a)


def max(a):
    if isinstance(a, MaskedArray):
        return a.max()
    return _builtins.max(float(v) for v in a)


def ptp(a):
    return float(max(a)) - float(min(a))


def cumsum(a):
    vals = a._valid() if isinstance(a, MaskedArray) else list(a)
    result, s = [], 0.0
    for v in vals:
        s += float(v)
        result.append(s)
    return result


def cumprod(a):
    vals = a._valid() if isinstance(a, MaskedArray) else list(a)
    result, p = [], 1.0
    for v in vals:
        p *= float(v)
        result.append(p)
    return result


def average(a, weights=None):
    vals = a._valid() if isinstance(a, MaskedArray) else list(a)
    if weights is None:
        return sum(float(v) for v in vals) / max(len(vals), 1)
    wv = list(weights)
    return sum(float(v) * float(w) for v, w in zip(vals, wv)) / max(sum(float(w) for w in wv), 1e-10)


def anom(a):
    m = mean(a)
    return _apply_op(lambda x: x - m, a)


def dot(a, b):
    av = a._valid() if isinstance(a, MaskedArray) else list(a)
    bv = b._valid() if isinstance(b, MaskedArray) else list(b)
    return sum(float(x)*float(y) for x, y in zip(av, bv))


def inner(a, b):
    return dot(a, b)


def outer(a, b):
    av = list(a._data if isinstance(a, MaskedArray) else a)
    bv = list(b._data if isinstance(b, MaskedArray) else b)
    return [[float(x)*float(y) for y in bv] for x in av]


def correlate(a, b, mode='valid'):
    from vectorphile.signal import correlate as _corr
    from vectorphile._core.ndarray import array as _nd
    from vectorphile.core.tensor import Tensor
    at = Tensor(_nd([float(v) for v in (a._valid() if isinstance(a, MaskedArray) else a)]))
    bt = Tensor(_nd([float(v) for v in (b._valid() if isinstance(b, MaskedArray) else b)]))
    return _corr(at, bt, mode)


def convolve(a, b, mode='full'):
    from vectorphile.signal import convolve as _conv
    from vectorphile._core.ndarray import array as _nd
    from vectorphile.core.tensor import Tensor
    at = Tensor(_nd([float(v) for v in (a._valid() if isinstance(a, MaskedArray) else a)]))
    bt = Tensor(_nd([float(v) for v in (b._valid() if isinstance(b, MaskedArray) else b)]))
    return _conv(at, bt, mode)


def concatenate(arrays, axis=0):
    result = []
    for arr in arrays:
        result.extend(arr._data if isinstance(arr, MaskedArray) else list(arr))
    return MaskedArray(result)


def stack(arrays, axis=0):
    return concatenate(arrays, axis)


def hstack(arrays):
    return concatenate(arrays)


def vstack(arrays):
    return concatenate(arrays)


def dstack(arrays):
    return concatenate(arrays)


def column_stack(arrays):
    return concatenate(arrays)


def row_stack(arrays):
    return concatenate(arrays)


def reshape(a, shape):
    return a


def resize(a, shape):
    return a


def swapaxes(a, ax1, ax2):
    return a


def transpose(a, axes=None):
    return a


def ravel(a):
    return a._data[:] if isinstance(a, MaskedArray) else list(a)


def sort(a, axis=-1, kind=None):
    vals = a._valid() if isinstance(a, MaskedArray) else list(a)
    return sorted(float(v) for v in vals)


def argsort(a):
    vals = a._data if isinstance(a, MaskedArray) else list(a)
    return sorted(range(len(vals)), key=lambda i: float(vals[i]))


def argmin(a):
    vals = a._data if isinstance(a, MaskedArray) else list(a)
    return min(range(len(vals)), key=lambda i: float(vals[i]))


def argmax(a):
    vals = a._data if isinstance(a, MaskedArray) else list(a)
    return max(range(len(vals)), key=lambda i: float(vals[i]))


def nonzero(a):
    vals = a._data if isinstance(a, MaskedArray) else list(a)
    return ([i for i, v in enumerate(vals) if float(v) != 0.0],)


def where(condition, a=None, b=None):
    cond = list(condition._data if isinstance(condition, MaskedArray) else condition)
    if a is None:
        return ([i for i, v in enumerate(cond) if v],)
    av = list(a._data if isinstance(a, MaskedArray) else ([a] * len(cond) if not hasattr(a, '__iter__') else a))
    bv = list(b._data if isinstance(b, MaskedArray) else ([b] * len(cond) if not hasattr(b, '__iter__') else b))
    return [av[i] if cond[i] else bv[i] for i in range(len(cond))]


def choose(a, choices):
    return [choices[int(v)][i] for i, v in enumerate(list(a))]


def round_(a, decimals=0):
    vals = a._data if isinstance(a, MaskedArray) else list(a)
    return [round(float(v), decimals) for v in vals]


def clip(a, a_min, a_max):
    vals = a._data if isinstance(a, MaskedArray) else list(a)
    return [max(float(a_min), min(float(a_max), float(v))) for v in vals]


def compress(condition, a):
    cond = list(condition)
    vals = a._data if isinstance(a, MaskedArray) else list(a)
    return [v for v, c in zip(vals, cond) if c]


def take(a, indices):
    vals = a._data if isinstance(a, MaskedArray) else list(a)
    return [vals[i] for i in indices]


def indices(dimensions, dtype=int):
    if len(dimensions) == 1:
        return list(range(dimensions[0]))
    return []


def ones(shape, dtype=float):
    if isinstance(shape, int):
        return MaskedArray([1.0] * shape)
    n = 1
    for s in shape:
        n *= s
    return MaskedArray([1.0] * n)


def zeros(shape, dtype=float):
    if isinstance(shape, int):
        return MaskedArray([0.0] * shape)
    n = 1
    for s in shape:
        n *= s
    return MaskedArray([0.0] * n)


def arange(start, stop=None, step=1):
    if stop is None:
        start, stop = 0, start
    vals = []
    v = float(start)
    while (step > 0 and v < stop) or (step < 0 and v > stop):
        vals.append(v)
        v += step
    return MaskedArray(vals)


def empty(shape, dtype=float):
    return zeros(shape, dtype)


def ones_like(a):
    n = len(a._data if isinstance(a, MaskedArray) else list(a))
    return MaskedArray([1.0] * n)


def zeros_like(a):
    n = len(a._data if isinstance(a, MaskedArray) else list(a))
    return MaskedArray([0.0] * n)


def empty_like(a):
    return zeros_like(a)


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    return func1d(arr._data if isinstance(arr, MaskedArray) else list(arr), *args, **kwargs)


def apply_over_axes(func, arr, axes):
    return func(arr, axes)


def mask_rowcols(a, axis=None):
    return a


def flatnotmasked_contiguous(a):
    if not isinstance(a, MaskedArray):
        return [slice(0, len(a))]
    result = []
    start = None
    for i, m in enumerate(a._mask):
        if not m and start is None:
            start = i
        elif m and start is not None:
            result.append(slice(start, i))
            start = None
    if start is not None:
        result.append(slice(start, len(a._mask)))
    return result


def flatnotmasked_edges(a):
    s = flatnotmasked_contiguous(a)
    if not s:
        return None
    return s[0].start, s[-1].stop - 1


def notmasked_contiguous(a, axis=None):
    return flatnotmasked_contiguous(a)


def notmasked_edges(a, axis=None):
    return flatnotmasked_edges(a)


def clump_masked(a):
    if not isinstance(a, MaskedArray):
        return []
    result = []
    start = None
    for i, m in enumerate(a._mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            result.append(slice(start, i))
            start = None
    if start is not None:
        result.append(slice(start, len(a._mask)))
    return result


def clump_unmasked(a):
    return flatnotmasked_contiguous(a)


shrink_mask = None

masked_print_option = '--'
