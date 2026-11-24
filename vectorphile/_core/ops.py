"""
vectorphile._core.ops — dispatches to C extension when available,
falls back to pure-Python NDArray otherwise.

This is the ONLY file that knows whether the C extension is present.
All higher-level code imports from here.
"""
from __future__ import annotations
import math
import operator
import itertools
from vectorphile._core.ndarray import (
    NDArray, array, zeros, ones, full, empty, eye, arange, linspace,
    _broadcast_shapes, _broadcast_idx, _broadcast_scalar,
    _elementwise_unary, _elementwise_binary, _prod, _TYPECODES, _c_strides, _DType,
    matmul as _py_matmul,
    exp as _nd_exp, log as _nd_log, sqrt as _nd_sqrt,
    relu as _nd_relu, sigmoid as _nd_sigmoid, silu as _nd_silu,
    tanh_arr as _nd_tanh, gelu as _nd_gelu, abs_arr as _nd_abs,
    log1p as _nd_log1p, expm1 as _nd_expm1, erf as _nd_erf,
)
import array as _array

# ── Try to load the C extension ───────────────────────────────────────────────

try:
    from vectorphile._C import vpcore as _C
    _HAS_C = True
except ImportError:
    _C = None
    _HAS_C = False


# ── Arithmetic ────────────────────────────────────────────────────────────────

def _ensure(x, dtype=None):
    if isinstance(x, NDArray): return x
    if isinstance(x, (int, float)): return _broadcast_scalar(x, (1,), dtype or 'float32')
    return array(x, dtype=dtype)


def _c_binop(fn_c, fn_py):
    def _op(a, b):
        a = _ensure(a); b = _ensure(b)
        if _HAS_C and a.dtype == b.dtype and a.shape == b.shape:
            out_buf = fn_c(a._buf if hasattr(a, '_buf') else a, b._buf if hasattr(b, '_buf') else b)
            # out_buf is a VPBuffer from the C extension
            # wrap it back into NDArray via to_list
            flat = _C.to_list(out_buf)
            tc = _TYPECODES.get(a.dtype, 'd')
            buf = _array.array(tc, flat)
            return NDArray(buf, a.shape, _c_strides(a.shape), a.dtype)
        return fn_py(a, b)
    return _op


def _c_binop_buf(fn_c, fn_py):
    """Variant that passes VPBuffer objects directly (requires contig buffer)."""
    def _op(a, b):
        a = _ensure(a); b = _ensure(b)
        # For now, always use pure Python (C integration needs buffer extraction)
        return fn_py(a, b)
    return _op


def add(a, b) -> NDArray:
    return _elementwise_binary(_ensure(a), _ensure(b), operator.add)

def subtract(a, b) -> NDArray:
    return _elementwise_binary(_ensure(a), _ensure(b), operator.sub)

def multiply(a, b) -> NDArray:
    return _elementwise_binary(_ensure(a), _ensure(b), operator.mul)

def divide(a, b) -> NDArray:
    return _elementwise_binary(_ensure(a), _ensure(b), operator.truediv)

def power(a, b) -> NDArray:
    return _elementwise_binary(_ensure(a), _ensure(b), operator.pow)

def maximum(a, b) -> NDArray:
    return _elementwise_binary(_ensure(a), _ensure(b), lambda x, y: x if x >= y else y)

def minimum(a, b) -> NDArray:
    return _elementwise_binary(_ensure(a), _ensure(b), lambda x, y: x if x <= y else y)

def fmax(a, b) -> NDArray:
    def _fmax(x, y):
        if math.isnan(x): return y
        if math.isnan(y): return x
        return max(x, y)
    return _elementwise_binary(_ensure(a), _ensure(b), _fmax)

def fmin(a, b) -> NDArray:
    def _fmin(x, y):
        if math.isnan(x): return y
        if math.isnan(y): return x
        return min(x, y)
    return _elementwise_binary(_ensure(a), _ensure(b), _fmin)

def where(cond, x, y) -> NDArray:
    cond = _ensure(cond); x = _ensure(x); y = _ensure(y)
    out_shape = _broadcast_shapes(_broadcast_shapes(cond.shape, x.shape), y.shape)
    result = zeros(out_shape, dtype=x.dtype)
    for idx in itertools.product(*[range(s) for s in out_shape]):
        ci = _broadcast_idx(idx, cond.shape)
        xi = _broadcast_idx(idx, x.shape)
        yi = _broadcast_idx(idx, y.shape)
        cv = cond._buf[cond._flat_idx(ci)]
        result._buf[result._flat_idx(idx)] = x._buf[x._flat_idx(xi)] if cv else y._buf[y._flat_idx(yi)]
    return result


# ── Math functions ────────────────────────────────────────────────────────────

def _unary(fn_name, math_fn):
    def _op(a) -> NDArray:
        a = _ensure(a)
        return _elementwise_unary(a, math_fn)
    _op.__name__ = fn_name
    return _op

def abs(a) -> NDArray: return _nd_abs(_ensure(a))
def sqrt(a) -> NDArray: return _nd_sqrt(_ensure(a))
def exp(a) -> NDArray: return _nd_exp(_ensure(a))
def log(a) -> NDArray: return _nd_log(_ensure(a))
def log2(a) -> NDArray: return _elementwise_unary(_ensure(a), math.log2)
def log10(a) -> NDArray: return _elementwise_unary(_ensure(a), math.log10)
def log1p(a) -> NDArray: return _nd_log1p(_ensure(a))
def expm1(a) -> NDArray: return _nd_expm1(_ensure(a))
def sin(a) -> NDArray: return _elementwise_unary(_ensure(a), math.sin)
def cos(a) -> NDArray: return _elementwise_unary(_ensure(a), math.cos)
def tan(a) -> NDArray: return _elementwise_unary(_ensure(a), math.tan)
def sinh(a) -> NDArray: return _elementwise_unary(_ensure(a), math.sinh)
def cosh(a) -> NDArray: return _elementwise_unary(_ensure(a), math.cosh)
def tanh(a) -> NDArray: return _nd_tanh(_ensure(a))
def arcsin(a) -> NDArray: return _elementwise_unary(_ensure(a), math.asin)
def arccos(a) -> NDArray: return _elementwise_unary(_ensure(a), math.acos)
def arctan(a) -> NDArray: return _elementwise_unary(_ensure(a), math.atan)
def arctan2(a, b) -> NDArray: return _elementwise_binary(_ensure(a), _ensure(b), math.atan2)
def arcsinh(a) -> NDArray: return _elementwise_unary(_ensure(a), math.asinh)
def arccosh(a) -> NDArray: return _elementwise_unary(_ensure(a), math.acosh)
def arctanh(a) -> NDArray: return _elementwise_unary(_ensure(a), math.atanh)
def cbrt(a) -> NDArray: return _elementwise_unary(_ensure(a), lambda x: x**(1/3) if x >= 0 else -((-x)**(1/3)))
def square(a) -> NDArray: return _elementwise_unary(_ensure(a), lambda x: x*x)
def reciprocal(a) -> NDArray: return _elementwise_unary(_ensure(a), lambda x: 1/x)
def sign(a) -> NDArray: return _elementwise_unary(_ensure(a), lambda x: 1.0 if x>0 else (-1.0 if x<0 else 0.0))
def floor(a) -> NDArray: return _elementwise_unary(_ensure(a), math.floor)
def ceil(a) -> NDArray: return _elementwise_unary(_ensure(a), math.ceil)
def trunc(a) -> NDArray: return _elementwise_unary(_ensure(a), math.trunc)
def rint(a) -> NDArray: return _elementwise_unary(_ensure(a), round)
def fix(a) -> NDArray: return _elementwise_unary(_ensure(a), math.trunc)
def fabs(a) -> NDArray: return _elementwise_unary(_ensure(a), math.fabs)
def degrees(a) -> NDArray: return _elementwise_unary(_ensure(a), math.degrees)
def radians(a) -> NDArray: return _elementwise_unary(_ensure(a), math.radians)
def hypot(a, b) -> NDArray: return _elementwise_binary(_ensure(a), _ensure(b), math.hypot)
def fmod(a, b) -> NDArray: return _elementwise_binary(_ensure(a), _ensure(b), math.fmod)
def remainder(a, b) -> NDArray: return _elementwise_binary(_ensure(a), _ensure(b), operator.mod)
def heaviside(a, h0=0.5) -> NDArray:
    return _elementwise_unary(_ensure(a), lambda x: 1.0 if x > 0 else (h0 if x == 0 else 0.0))
def sinc(a) -> NDArray:
    def _sinc(x): return 1.0 if x == 0 else math.sin(math.pi*x)/(math.pi*x)
    return _elementwise_unary(_ensure(a), _sinc)

def gcd(a: int, b: int) -> int: return math.gcd(int(a), int(b))
def lcm(a: int, b: int) -> int: return abs(int(a)*int(b))//math.gcd(int(a), int(b)) if a and b else 0

def relu(a) -> NDArray: return _nd_relu(_ensure(a))
def sigmoid(a) -> NDArray: return _nd_sigmoid(_ensure(a))
def silu(a) -> NDArray: return _nd_silu(_ensure(a))


# ── Reductions ────────────────────────────────────────────────────────────────

def sum(a, axis=None, keepdims=False) -> NDArray:
    a = _ensure(a); return a.sum(axis=axis, keepdims=keepdims)

def mean(a, axis=None, keepdims=False) -> NDArray:
    a = _ensure(a); return a.mean(axis=axis, keepdims=keepdims)

def max(a, axis=None, keepdims=False) -> NDArray:
    a = _ensure(a); return a.max(axis=axis, keepdims=keepdims)

def min(a, axis=None, keepdims=False) -> NDArray:
    a = _ensure(a); return a.min(axis=axis, keepdims=keepdims)

def prod(a, axis=None, keepdims=False) -> NDArray:
    a = _ensure(a); return a.prod(axis=axis, keepdims=keepdims)

def std(a, axis=None, keepdims=False, ddof=0) -> NDArray:
    a = _ensure(a); return a.std(axis=axis, keepdims=keepdims, ddof=ddof)

def var(a, axis=None, keepdims=False, ddof=0) -> NDArray:
    a = _ensure(a); return a.var(axis=axis, keepdims=keepdims, ddof=ddof)

def norm(a, ord=None, axis=None) -> NDArray:
    a = _ensure(a)
    if axis is None:
        flat = a.to_flat_list()
        if ord is None or ord == 2: return math.sqrt(builtins_sum(x*x for x in flat))
        if ord == 1: return builtins_sum(abs(x) for x in flat)
        if ord == float('inf'): return builtins_max(abs(x) for x in flat)
    # Axis reduction: compute norm along given axis
    sh = a.shape
    ndim = len(sh)
    ax = axis % ndim if axis is not None else None
    # Compute norm along ax dimension
    out_shape = tuple(s for i, s in enumerate(sh) if i != ax)
    if not out_shape:
        out_shape = (1,)
    flat = a.to_flat_list()
    # Group elements by their position along all axes except ax
    out_size = 1
    for s in out_shape:
        out_size *= s
    ax_size = sh[ax]
    # Build stride for axis indexing
    strides = _c_strides(sh)
    result = []
    for flat_out_idx in range(out_size):
        # Reconstruct multi-index excluding ax
        remaining = flat_out_idx
        multi = []
        for i, s in enumerate(out_shape):
            multi.append(remaining % s)
            remaining //= s
        multi.reverse()
        # Insert ax dimension
        vals = []
        for k in range(ax_size):
            full_idx = multi[:ax] + [k] + multi[ax:]
            pos = builtins_sum(full_idx[i] * strides[i] for i in range(ndim))
            vals.append(flat[pos])
        if ord is None or ord == 2:
            result.append(math.sqrt(builtins_sum(v*v for v in vals)))
        elif ord == 1:
            result.append(builtins_sum(abs(v) for v in vals))
        elif ord == float('inf'):
            result.append(builtins_max(abs(v) for v in vals))
        else:
            result.append(builtins_sum(abs(v)**ord for v in vals)**(1.0/ord))
    import array as _ar
    buf = _ar.array('f', [float(v) for v in result])
    return NDArray(buf, out_shape, _c_strides(out_shape), _DType('float32'))


import builtins
builtins_sum = builtins.sum
builtins_max = builtins.max


# ── Matrix multiply ───────────────────────────────────────────────────────────

def matmul(a, b) -> NDArray:
    a = _ensure(a); b = _ensure(b)
    return _py_matmul(a, b)


def dot(a, b) -> NDArray:
    a = _ensure(a); b = _ensure(b)
    if a.ndim == 1 and b.ndim == 1:
        flat_a = a.to_flat_list(); flat_b = b.to_flat_list()
        return type(a)(
            _array.array(_TYPECODES.get(a.dtype, 'd'), [builtins_sum(x*y for x,y in zip(flat_a, flat_b))]),
            (), (), a.dtype)
    return matmul(a, b)


# ── Shape ops ─────────────────────────────────────────────────────────────────

def reshape(a, *shape) -> NDArray:
    return _ensure(a).reshape(*shape)

def transpose(a, axes=None) -> NDArray:
    return _ensure(a).transpose(axes)

def clip(a, a_min, a_max) -> NDArray:
    return _ensure(a).clip(a_min, a_max)


# ── Concatenation / stacking ──────────────────────────────────────────────────

def concatenate(arrays, axis=0) -> NDArray:
    arrays = [_ensure(a) for a in arrays]
    if axis == 0:
        flat = []
        new_shape = list(arrays[0].shape)
        new_shape[0] = builtins_sum(a.shape[0] for a in arrays)
        for a in arrays: flat.extend(a.to_flat_list())
        tc = _TYPECODES.get(arrays[0].dtype, 'd')
        buf = _array.array(tc, flat)
        return NDArray(buf, tuple(new_shape), _c_strides(tuple(new_shape)), arrays[0].dtype)
    # General axis concatenation via numpy if available
    try:
        import numpy as _np_cat
        arrs = [_np_cat.array(a.to_flat_list(), dtype='float32').reshape(a.shape) for a in arrays]
        result = _np_cat.concatenate(arrs, axis=axis)
        new_shape = tuple(result.shape)
        tc = _TYPECODES.get(arrays[0].dtype, 'f')
        buf = _array.array(tc, result.flatten().tolist())
        return NDArray(buf, new_shape, _c_strides(new_shape), arrays[0].dtype)
    except ImportError:
        pass
    # Pure Python fallback: move axis to front, concat, move back
    import itertools
    ndim = len(arrays[0].shape)
    ax = axis if axis >= 0 else ndim + axis
    # Build output shape
    out_shape = list(arrays[0].shape)
    out_shape[ax] = builtins_sum(a.shape[ax] for a in arrays)
    out_flat = [0.0] * (builtins_sum(1 for _ in itertools.product(*[range(s) for s in out_shape])))
    # Strides for output
    def strides(sh):
        s = [1];
        for sz in reversed(sh[1:]): s.insert(0, s[0] * sz)
        return s
    out_st = strides(out_shape)
    ax_offset = 0
    for a in arrays:
        a_flat = a.to_flat_list()
        a_st = strides(list(a.shape))
        for coords in itertools.product(*[range(s) for s in a.shape]):
            src_idx = 0
            for i, c in enumerate(coords): src_idx += c * a_st[i]
            out_coords = list(coords)
            out_coords[ax] += ax_offset
            dst_idx = 0
            for i, c in enumerate(out_coords): dst_idx += c * out_st[i]
            out_flat[dst_idx] = a_flat[src_idx]
        ax_offset += a.shape[ax]
    tc = _TYPECODES.get(arrays[0].dtype, 'f')
    buf = _array.array(tc, out_flat)
    return NDArray(buf, tuple(out_shape), _c_strides(tuple(out_shape)), arrays[0].dtype)


def stack(arrays, axis=0) -> NDArray:
    arrays = [_ensure(a) for a in arrays]
    unsqueezed = [a.unsqueeze(axis) for a in arrays]
    return concatenate(unsqueezed, axis=axis)


# ── Random — xoshiro256+ C kernels (from scratch) ────────────────────────────

import os as _os
import time as _time

try:
    from vectorphile._core.ndarray import (_kern_fn, _kern_ptr, _KERN,
                                            NDArray as _NDArray,
                                            _c_strides as _cs, _TYPECODES as _TC)
    _RAND_KERN = _KERN
except Exception:
    _RAND_KERN = False


def _rand_seed() -> int:
    return int(_time.time_ns()) ^ (_os.getpid() << 17)


def rand(*shape, dtype='float32') -> NDArray:
    """Uniform random in [0, 1)."""
    n = _prod(shape)
    buf = _array.array('f', b'\x00' * (4 * n))
    if _RAND_KERN:
        kern = _kern_fn('vp_rand_uniform')
        if kern is not None:
            kern(_kern_ptr(buf), n, _rand_seed())
            return NDArray(buf, shape, _c_strides(shape), 'float32')
    import random
    for i in range(n): buf[i] = random.random()
    return NDArray(buf, shape, _c_strides(shape), 'float32')


def randn(*shape, dtype='float32') -> NDArray:
    """Standard normal random."""
    n = _prod(shape)
    buf = _array.array('f', b'\x00' * (4 * n))
    if _RAND_KERN:
        kern = _kern_fn('vp_rand_normal')
        if kern is not None:
            kern(_kern_ptr(buf), n, _rand_seed(), 0.0, 1.0)
            return NDArray(buf, shape, _c_strides(shape), 'float32')
    import random
    for i in range(0, n - 1, 2):
        import math as _m
        u1 = random.random() + 1e-10; u2 = random.random()
        r = _m.sqrt(-2 * _m.log(u1))
        buf[i] = r * _m.cos(2 * _m.pi * u2)
        if i + 1 < n: buf[i+1] = r * _m.sin(2 * _m.pi * u2)
    return NDArray(buf, shape, _c_strides(shape), 'float32')


def randint(low, high=None, size=None, dtype='int64') -> NDArray:
    """Random integers in [low, high)."""
    if high is None: low, high = 0, low
    if size is None: size = ()
    if isinstance(size, int): size = (size,)
    n = _prod(size) if size else 1
    tc = _TYPECODES.get(dtype, 'l')
    buf = _array.array(tc, [0] * n)
    if _RAND_KERN:
        kern = _kern_fn('vp_rand_int')
        if kern is not None:
            from vectorphile._mojo._vp_cffi_loader import iptr as _iptr
            int_buf = _array.array('i', b'\x00' * (4 * n))
            kern(_iptr(int_buf), n, _rand_seed(), low, high)
            for i in range(n): buf[i] = int_buf[i]
            return NDArray(buf, size or (1,), _c_strides(size or (1,)), dtype)
    import random
    for i in range(n): buf[i] = random.randint(low, high - 1)
    return NDArray(buf, size or (1,), _c_strides(size or (1,)), dtype)


def bernoulli(shape, p=0.5) -> NDArray:
    """Bernoulli random (0 or 1 with probability p)."""
    n = _prod(shape)
    buf = _array.array('f', b'\x00' * (4 * n))
    if _RAND_KERN:
        kern = _kern_fn('vp_rand_bernoulli')
        if kern is not None:
            kern(_kern_ptr(buf), n, _rand_seed(), p)
            return NDArray(buf, shape, _c_strides(shape), 'float32')
    import random
    for i in range(n): buf[i] = 1.0 if random.random() < p else 0.0
    return NDArray(buf, shape, _c_strides(shape), 'float32')


def normal(mean=0.0, std=1.0, size=None) -> NDArray:
    """Normal distribution N(mean, std^2)."""
    if size is None: size = ()
    if isinstance(size, int): size = (size,)
    n = _prod(size) if size else 1
    buf = _array.array('f', b'\x00' * (4 * max(n, 1)))
    if _RAND_KERN:
        kern = _kern_fn('vp_rand_normal')
        if kern is not None:
            kern(_kern_ptr(buf), max(n, 1), _rand_seed(), mean, std)
            return NDArray(buf, size or (1,), _c_strides(size or (1,)), 'float32')
    import random, math as _m
    for i in range(0, n, 2):
        u1 = random.random() + 1e-10; u2 = random.random()
        r = _m.sqrt(-2 * _m.log(u1))
        buf[i] = mean + std * r * _m.cos(2 * _m.pi * u2)
        if i + 1 < n: buf[i+1] = mean + std * r * _m.sin(2 * _m.pi * u2)
    return NDArray(buf, size or (1,), _c_strides(size or (1,)), 'float32')
