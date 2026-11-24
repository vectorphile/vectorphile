"""vectorphile.jax — JAX-compatible functional transforms.

Provides the same API surface as JAX so models can be written once and run
under either framework. Pure-Python implementation over vectorphile Tensors —
no JAX, no torch dependency.

    import vectorphile.jax as vjax

    @vjax.jit
    def f(x): return (x * x).sum()

    g = vjax.grad(f)
    jvp_val, tangent = vjax.jvp(f, (x,), (v,))
    val, vjp_fn = vjax.vjp(f, x)
    vmapped = vjax.vmap(f)
"""
from __future__ import annotations
import math
import copy
import random as _rnd
from typing import Callable, Tuple, Any, Optional, Sequence
from vectorphile.core.tensor import Tensor
from vectorphile._core.ndarray import (
    NDArray, array as _nd_array, zeros, ones, full, eye, arange, linspace,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_data(x):
    return x._data if isinstance(x, Tensor) else x


def _get_shape(d) -> tuple:
    if isinstance(d, NDArray):
        return d.shape
    if isinstance(d, (list, tuple)):
        return (len(d),)
    return ()


def _get_flat(d) -> list:
    if isinstance(d, NDArray):
        return d.to_flat_list()
    if isinstance(d, (list, tuple)):
        return [float(v) for v in d]
    return [float(d)]


def _nd_copy(d):
    if isinstance(d, NDArray):
        flat = d.to_flat_list()
        return _nd_array(flat, dtype='float32').reshape(d.shape)
    return d


def _nd_ones_like(d) -> NDArray:
    sh = _get_shape(d)
    return ones(sh if sh else (1,), dtype='float32')


def _nd_zeros_like(d) -> NDArray:
    sh = _get_shape(d)
    return zeros(sh if sh else (1,), dtype='float32')


def _elem_wise(x, fn):
    d = _get_data(x)
    flat = _get_flat(d)
    sh = _get_shape(d)
    result = [fn(v) for v in flat]
    return Tensor(_nd_array(result, dtype='float32').reshape(sh if sh else (len(result),)))


def _elem_wise2(x, y, fn):
    xd, yd = _get_data(x), _get_data(y)
    xf, yf = _get_flat(xd), _get_flat(yd)
    sh = _get_shape(xd) or _get_shape(yd)
    if len(xf) == 1:
        xf = xf * len(yf); sh = _get_shape(yd)
    if len(yf) == 1:
        yf = yf * len(xf)
    result = [fn(a, b) for a, b in zip(xf, yf)]
    return Tensor(_nd_array(result, dtype='float32').reshape(sh if sh else (len(result),)))


def _nd_stack(arrays: list, axis: int = 0) -> NDArray:
    if not arrays:
        return _nd_array([], dtype='float32')
    sh = _get_shape(arrays[0])
    n = len(arrays)
    flat_all = []
    for a in arrays:
        flat_all.extend(_get_flat(a))
    new_shape = (n,) + sh
    return _nd_array(flat_all, dtype='float32').reshape(new_shape)


def _nd_concat(arrays: list, axis: int = 0) -> NDArray:
    if not arrays:
        return _nd_array([], dtype='float32')
    flat_all = []
    for a in arrays:
        flat_all.extend(_get_flat(a))
    if axis == 0:
        sh0 = _get_shape(arrays[0])
        new_n = sum((_get_shape(a)[0] if _get_shape(a) else 1) for a in arrays)
        rest = sh0[1:] if len(sh0) > 1 else ()
        return _nd_array(flat_all, dtype='float32').reshape((new_n,) + rest)
    return _nd_array(flat_all, dtype='float32')


def _nd_take(a, indices, axis=None):
    flat = _get_flat(a)
    sh = _get_shape(a)
    idx = [int(indices)] if not hasattr(indices, '__iter__') else [int(i) for i in indices]
    if axis is None or len(sh) <= 1:
        return _nd_array([flat[i] for i in idx], dtype='float32')
    n = sh[axis]
    stride = 1
    for d in sh[axis + 1:]:
        stride *= d
    outer = 1
    for d in sh[:axis]:
        outer *= d
    result = []
    for oi in range(outer):
        for i in idx:
            start = oi * n * stride + i * stride
            result.extend(flat[start:start + stride])
    new_sh = sh[:axis] + (len(idx),) + sh[axis + 1:]
    return _nd_array(result, dtype='float32').reshape(new_sh)


# ── grad ─────────────────────────────────────────────────────────────────────

def grad(fun: Callable, argnums=0, has_aux=False) -> Callable:
    """Return gradient of fun w.r.t. the argument(s) in argnums."""
    def grad_fn(*args, **kwargs):
        nums = (argnums,) if isinstance(argnums, int) else tuple(argnums)
        cloned = list(args)
        targets = []
        for i in nums:
            t = Tensor(_nd_copy(_get_data(args[i])), requires_grad=True)
            cloned[i] = t
            targets.append(t)
        for t in targets:
            t.grad = None
        out = fun(*cloned, **kwargs)
        aux = None
        if has_aux:
            out, aux = out
        out.backward(Tensor(_nd_ones_like(_get_data(out))))
        grads = tuple(
            Tensor(_nd_copy(_get_data(t.grad))) if t.grad is not None
            else Tensor(_nd_zeros_like(_get_data(t)))
            for t in targets
        )
        result_grads = grads[0] if isinstance(argnums, int) else grads
        return (result_grads, aux) if has_aux else result_grads
    return grad_fn


# ── value_and_grad ────────────────────────────────────────────────────────────

def value_and_grad(fun: Callable, argnums=0, has_aux=False) -> Callable:
    """Return (value, gradient) of fun."""
    def vg_fn(*args, **kwargs):
        nums = (argnums,) if isinstance(argnums, int) else tuple(argnums)
        cloned = list(args)
        targets = []
        for i in nums:
            t = Tensor(_nd_copy(_get_data(args[i])), requires_grad=True)
            cloned[i] = t
            targets.append(t)
        for t in targets:
            t.grad = None
        out = fun(*cloned, **kwargs)
        aux = None
        if has_aux:
            out, aux = out
        out.backward(Tensor(_nd_ones_like(_get_data(out))))
        grads = tuple(
            Tensor(_nd_copy(_get_data(t.grad))) if t.grad is not None
            else Tensor(_nd_zeros_like(_get_data(t)))
            for t in targets
        )
        result_grads = grads[0] if isinstance(argnums, int) else grads
        return (out, result_grads, aux) if has_aux else (out, result_grads)
    return vg_fn


# ── jvp ──────────────────────────────────────────────────────────────────────

def jvp(fun: Callable, primals: tuple, tangents: tuple) -> Tuple[Any, Any]:
    """Forward-mode JVP via finite differences."""
    eps = 1e-5
    out0 = fun(*primals)
    perturbed = []
    for p, t in zip(primals, tangents):
        pd, td = _get_data(p), _get_data(t)
        pf, tf = _get_flat(pd), _get_flat(td)
        sh = _get_shape(pd)
        new_flat = [a + eps * b for a, b in zip(pf, tf)]
        perturbed.append(Tensor(_nd_array(new_flat, dtype='float32').reshape(sh)))
    out1 = fun(*perturbed)
    o0f = _get_flat(_get_data(out0))
    o1f = _get_flat(_get_data(out1))
    sh = _get_shape(_get_data(out0))
    tan_flat = [(b - a) / eps for a, b in zip(o0f, o1f)]
    return out0, Tensor(_nd_array(tan_flat, dtype='float32').reshape(sh))


# ── vjp ──────────────────────────────────────────────────────────────────────

def vjp(fun: Callable, *primals) -> Tuple[Any, Callable]:
    """Reverse-mode VJP."""
    targets = [Tensor(_nd_copy(_get_data(p)), requires_grad=True) for p in primals]
    for t in targets:
        t.grad = None
    out = fun(*targets)
    out_stored = Tensor(_nd_copy(_get_data(out)))

    def vjp_fn(v):
        for t in targets:
            t.grad = None
        fresh = [Tensor(_nd_copy(_get_data(p)), requires_grad=True) for p in primals]
        fresh_out = fun(*fresh)
        fresh_out.backward(v if isinstance(v, Tensor) else Tensor(_get_data(v)))
        return tuple(
            Tensor(_nd_copy(_get_data(t.grad))) if t.grad is not None
            else Tensor(_nd_zeros_like(_get_data(p)))
            for t, p in zip(fresh, primals)
        )
    return out_stored, vjp_fn


# ── vmap ──────────────────────────────────────────────────────────────────────

def vmap(fun: Callable, in_axes=0, out_axes=0) -> Callable:
    """Vectorised map over a batch axis."""
    def batched(*args, **kwargs):
        from vectorphile.core.tensor import stack as _t_stack
        axes = [in_axes] * len(args) if isinstance(in_axes, int) else list(in_axes)
        d0 = _get_data(args[0])
        sh0 = _get_shape(d0)
        B = sh0[axes[0]] if sh0 else 1
        outputs = [fun(*[Tensor(_nd_take(_get_data(a), i, axis=ax)) for a, ax in zip(args, axes)], **kwargs)
                   for i in range(B)]
        return _t_stack(outputs, dim=0) if isinstance(out_axes, int) and out_axes == 0 and outputs and isinstance(outputs[0], Tensor) else outputs
    return batched


# ── jit ──────────────────────────────────────────────────────────────────────

def jit(fun: Callable = None, *, static_argnums=(), donate_argnums=()) -> Callable:
    """JIT-compile fun (no-op in vectorphile; marks function for future backends)."""
    if fun is None:
        return lambda fn: jit(fn, static_argnums=static_argnums)
    def jitted(*args, **kwargs):
        return fun(*args, **kwargs)
    jitted.__wrapped__ = fun
    jitted.__name__ = getattr(fun, '__name__', 'jitted')
    return jitted


# ── scan ──────────────────────────────────────────────────────────────────────

def scan(f: Callable, init, xs, length=None, reverse=False, unroll=1):
    """Scan a function over leading array axes while carrying state."""
    from vectorphile.core.tensor import stack as _t_stack
    carry = init
    if xs is None:
        xs_list = [None] * (length or 0)
    else:
        d = _get_data(xs) if isinstance(xs, Tensor) else xs
        sh = _get_shape(d)
        n = sh[0] if sh else (length or 0)
        flat = _get_flat(d)
        rest_sz = len(flat) // n if n > 0 else 1
        rest_sh = sh[1:] if len(sh) > 1 else (rest_sz,)
        xs_list = [Tensor(_nd_array(flat[i * rest_sz:(i + 1) * rest_sz], dtype='float32').reshape(rest_sh))
                   for i in range(n)]
    if reverse:
        xs_list = xs_list[::-1]
    ys = []
    for i in range(0, len(xs_list), unroll):
        for x in xs_list[i:i + unroll]:
            carry, y = f(carry, x)
            ys.append(y)
    if reverse:
        ys = ys[::-1]
    if ys and isinstance(ys[0], Tensor):
        return carry, _t_stack(ys, dim=0)
    return carry, ys


# ── pmap ──────────────────────────────────────────────────────────────────────

def pmap(fun: Callable, axis_name: str = 'batch', in_axes=0, out_axes=0, devices=None) -> Callable:
    return vmap(fun, in_axes=in_axes, out_axes=out_axes)


# ── lax ──────────────────────────────────────────────────────────────────────

class lax:
    """Low-level operations, mirroring jax.lax."""

    @staticmethod
    def add(x, y): return x + y
    @staticmethod
    def mul(x, y): return x * y
    @staticmethod
    def sub(x, y): return x - y
    @staticmethod
    def div(x, y): return x / y
    @staticmethod
    def neg(x): return -x
    @staticmethod
    def abs(x): return _elem_wise(x, abs)
    @staticmethod
    def exp(x): return x.exp()
    @staticmethod
    def log(x): return x.log()
    @staticmethod
    def sqrt(x): return x.sqrt()
    @staticmethod
    def pow(x, exp): return _elem_wise(x, lambda v: v ** exp)
    @staticmethod
    def max(x, y): return _elem_wise2(x, y, max)
    @staticmethod
    def min(x, y): return _elem_wise2(x, y, min)
    @staticmethod
    def clamp(x, lo, hi): return _elem_wise(x, lambda v: max(lo, min(hi, v)))
    @staticmethod
    def dot(a, b): return a @ b

    @staticmethod
    def dot_general(a, b, dimension_numbers):
        return a @ b  # simplified

    @staticmethod
    def conv(lhs, rhs, window_strides, padding):
        import vectorphile.nn.functional as F
        from vectorphile.core.tensor import Tensor as T
        return F.conv2d(lhs if isinstance(lhs, T) else T(lhs),
                        rhs if isinstance(rhs, T) else T(rhs),
                        stride=window_strides, padding=0 if padding == 'VALID' else 1)

    @staticmethod
    def reduce_sum(x, axes):
        return Tensor(_get_data(x).sum(axis=tuple(axes)))

    @staticmethod
    def reduce_max(x, axes):
        return Tensor(_get_data(x).max(axis=tuple(axes)))

    @staticmethod
    def slice(x, starts, limits, strides=None):
        d = _get_data(x)
        sh = _get_shape(d)
        flat = _get_flat(d)
        if len(sh) == 1:
            st = strides[0] if strides else 1
            return Tensor(_nd_array(flat[starts[0]:limits[0]:st], dtype='float32'))
        return Tensor(d)

    @staticmethod
    def dynamic_slice(x, starts, slice_sizes):
        d = _get_data(x)
        sh = _get_shape(d)
        flat = _get_flat(d)
        if len(sh) == 1:
            s, sz = int(starts[0]), int(slice_sizes[0])
            return Tensor(_nd_array(flat[s:s + sz], dtype='float32'))
        return Tensor(d)

    @staticmethod
    def concatenate(arrays, dimension):
        return Tensor(_nd_concat([_get_data(a) for a in arrays], axis=dimension))

    @staticmethod
    def reshape(x, new_sizes):
        return Tensor(_get_data(x).reshape(new_sizes))

    @staticmethod
    def transpose(x, permutation):
        return Tensor(_get_data(x).transpose(permutation))

    @staticmethod
    def broadcast_to(x, shape):
        d = _get_data(x)
        flat = _get_flat(d)
        total = 1
        for s in shape:
            total *= s
        if len(flat) == 1:
            result = flat * total
        else:
            reps = total // len(flat) if total % len(flat) == 0 else 1
            result = flat * reps
        return Tensor(_nd_array(result, dtype='float32').reshape(shape))

    @staticmethod
    def iota(dtype, size):
        return Tensor(arange(0, size, 1, dtype='float32'))

    @staticmethod
    def stop_gradient(x):
        return Tensor(_nd_copy(_get_data(x)))

    @staticmethod
    def while_loop(cond_fn, body_fn, init_val):
        val = init_val
        while cond_fn(val):
            val = body_fn(val)
        return val

    @staticmethod
    def cond(pred, true_fn, false_fn, *operands):
        p = _get_data(pred) if isinstance(pred, Tensor) else pred
        p_val = _get_flat(p)[0] if isinstance(p, NDArray) else float(p)
        return true_fn(*operands) if bool(p_val) else false_fn(*operands)

    @staticmethod
    def switch(index, branches, *operands):
        idx_d = _get_data(index) if isinstance(index, Tensor) else index
        idx = int(_get_flat(idx_d)[0]) if isinstance(idx_d, NDArray) else int(idx_d)
        return branches[idx](*operands)

    @staticmethod
    def fori_loop(lower, upper, body_fn, init_val):
        val = init_val
        for i in range(int(lower), int(upper)):
            val = body_fn(i, val)
        return val

    @staticmethod
    def associative_scan(fn, elems, axis=0, reverse=False):
        from vectorphile.core.tensor import stack as _t_stack
        d = _get_data(elems)
        sh = _get_shape(d)
        n = sh[axis] if sh else 1
        flat = _get_flat(d)
        rest_sz = len(flat) // n if n > 0 else 1
        rest_sh = sh[1:] if len(sh) > 1 else (rest_sz,)
        indices = list(range(n - 1, -1, -1)) if reverse else list(range(n))
        results = []
        carry = None
        for i in indices:
            xi = Tensor(_nd_array(flat[i * rest_sz:(i + 1) * rest_sz], dtype='float32').reshape(rest_sh))
            carry = fn(carry, xi) if carry is not None else xi
            results.append(carry)
        if reverse:
            results = results[::-1]
        return _t_stack(results, dim=axis)


# ── numpy ─────────────────────────────────────────────────────────────────────

class numpy:
    """jax.numpy-like API over vectorphile Tensors."""

    @staticmethod
    def array(x, dtype=None):
        if isinstance(x, Tensor): return x
        if isinstance(x, NDArray): return Tensor(x)
        flat = list(x) if hasattr(x, '__iter__') else [float(x)]
        return Tensor(_nd_array([float(v) for v in flat], dtype=dtype or 'float32'))

    @staticmethod
    def zeros(shape, dtype='float32'): return Tensor(zeros(shape, dtype=dtype))

    @staticmethod
    def ones(shape, dtype='float32'): return Tensor(ones(shape, dtype=dtype))

    @staticmethod
    def arange(*args, **kw):
        start, stop, step = 0, None, 1
        if len(args) == 1: stop = args[0]
        elif len(args) == 2: start, stop = args
        elif len(args) >= 3: start, stop, step = args[:3]
        return Tensor(arange(start, stop, step, dtype='float32'))

    @staticmethod
    def linspace(start, stop, num=50):
        return Tensor(linspace(start, stop, num, dtype='float32'))

    @staticmethod
    def eye(n, dtype='float32'): return Tensor(eye(n, dtype=dtype))

    @staticmethod
    def concatenate(arrays, axis=0):
        return Tensor(_nd_concat([_get_data(a) for a in arrays], axis=axis))

    @staticmethod
    def stack(arrays, axis=0):
        return Tensor(_nd_stack([_get_data(a) for a in arrays], axis=axis))

    @staticmethod
    def reshape(x, shape): return Tensor(_get_data(x).reshape(shape))

    @staticmethod
    def transpose(x, axes=None): return Tensor(_get_data(x).transpose(axes))

    @staticmethod
    def expand_dims(x, axis): return Tensor(_get_data(x).expand_dims(axis))

    @staticmethod
    def squeeze(x, axis=None): return Tensor(_get_data(x).squeeze(axis))

    @staticmethod
    def sum(x, axis=None, keepdims=False):
        return Tensor(_get_data(x).sum(axis=axis, keepdims=keepdims))

    @staticmethod
    def mean(x, axis=None, keepdims=False):
        return Tensor(_get_data(x).mean(axis=axis, keepdims=keepdims))

    @staticmethod
    def max(x, axis=None, keepdims=False):
        return Tensor(_get_data(x).max(axis=axis, keepdims=keepdims))

    @staticmethod
    def min(x, axis=None, keepdims=False):
        return Tensor(_get_data(x).min(axis=axis, keepdims=keepdims))

    @staticmethod
    def dot(a, b): return a @ b

    @staticmethod
    def matmul(a, b): return a @ b

    @staticmethod
    def exp(x): return x.exp()

    @staticmethod
    def log(x): return x.log()

    @staticmethod
    def sqrt(x): return x.sqrt()

    @staticmethod
    def abs(x): return _elem_wise(x, abs)

    @staticmethod
    def clip(x, a_min, a_max):
        return _elem_wise(x, lambda v: max(a_min, min(a_max, v)))

    @staticmethod
    def where(cond, x, y):
        cf = _get_flat(_get_data(cond)) if isinstance(cond, Tensor) else [float(cond)]
        xf = _get_flat(_get_data(x)) if isinstance(x, Tensor) else [float(x)]
        yf = _get_flat(_get_data(y)) if isinstance(y, Tensor) else [float(y)]
        sh = _get_shape(_get_data(x)) if isinstance(x, Tensor) else _get_shape(_get_data(y))
        n = max(len(cf), len(xf), len(yf))
        if len(cf) == 1: cf = cf * n
        if len(xf) == 1: xf = xf * n
        if len(yf) == 1: yf = yf * n
        result = [xv if cv else yv for cv, xv, yv in zip(cf, xf, yf)]
        return Tensor(_nd_array(result, dtype='float32').reshape(sh if sh else (len(result),)))

    @staticmethod
    def einsum(subscripts, *operands):
        from vectorphile._core.ndarray import _nd_einsum
        return Tensor(_nd_einsum(subscripts, *[_get_data(o) for o in operands]))

    @staticmethod
    def take(a, indices, axis=None):
        d = _get_data(a)
        idx = int(indices) if not hasattr(indices, '__iter__') else [int(i) for i in (indices.to_flat_list() if isinstance(indices, NDArray) else indices)]
        return Tensor(_nd_take(d, [idx] if isinstance(idx, int) else idx, axis=axis))

    @staticmethod
    def argsort(x, axis=-1):
        flat = _get_flat(_get_data(x))
        idx = sorted(range(len(flat)), key=lambda i: flat[i])
        return Tensor(_nd_array([float(i) for i in idx], dtype='float32'))

    @staticmethod
    def sort(x, axis=-1):
        flat = sorted(_get_flat(_get_data(x)))
        return Tensor(_nd_array(flat, dtype='float32').reshape(_get_shape(_get_data(x))))

    @staticmethod
    def argmax(x, axis=None): return Tensor(_get_data(x).argmax(axis=axis))

    @staticmethod
    def argmin(x, axis=None): return Tensor(_get_data(x).argmin(axis=axis))

    @staticmethod
    def unique(x):
        flat = _get_flat(_get_data(x))
        seen = sorted(set(flat))
        return Tensor(_nd_array(seen, dtype='float32'))

    @staticmethod
    def pad(x, pad_width, mode='constant', constant_values=0):
        d = _get_data(x)
        sh = _get_shape(d)
        flat = _get_flat(d)
        if len(sh) == 1:
            if isinstance(pad_width, int):
                pl, pr = pad_width, pad_width
            else:
                pl, pr = (pad_width[0], pad_width[1]) if isinstance(pad_width[0], int) else (pad_width[0][0], pad_width[0][1])
            result = [float(constant_values)] * pl + flat + [float(constant_values)] * pr
            return Tensor(_nd_array(result, dtype='float32'))
        return Tensor(d)

    @staticmethod
    def broadcast_to(x, shape): return lax.broadcast_to(x, shape)

    @staticmethod
    def full(shape, fill_value, dtype='float32'):
        return Tensor(full(shape, fill_value, dtype=dtype))

    @staticmethod
    def zeros_like(x): return Tensor(_nd_zeros_like(_get_data(x)))

    @staticmethod
    def ones_like(x): return Tensor(_nd_ones_like(_get_data(x)))

    @staticmethod
    def isnan(x):
        flat = _get_flat(_get_data(x))
        sh = _get_shape(_get_data(x))
        return Tensor(_nd_array([1.0 if math.isnan(v) else 0.0 for v in flat], dtype='float32').reshape(sh))

    @staticmethod
    def isinf(x):
        flat = _get_flat(_get_data(x))
        sh = _get_shape(_get_data(x))
        return Tensor(_nd_array([1.0 if math.isinf(v) else 0.0 for v in flat], dtype='float32').reshape(sh))

    @staticmethod
    def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
        flat = _get_flat(_get_data(x))
        sh = _get_shape(_get_data(x))
        def _fix(v):
            if math.isnan(v): return nan
            if math.isinf(v) and v > 0: return posinf if posinf is not None else 3.4e38
            if math.isinf(v) and v < 0: return neginf if neginf is not None else -3.4e38
            return v
        return Tensor(_nd_array([_fix(v) for v in flat], dtype='float32').reshape(sh))

    @staticmethod
    def tril(x, k=0):
        d = _get_data(x)
        sh = _get_shape(d)
        if len(sh) == 2:
            H, W = sh
            flat = _get_flat(d)
            result = [flat[r * W + c] if c <= r + k else 0.0 for r in range(H) for c in range(W)]
            return Tensor(_nd_array(result, dtype='float32').reshape(sh))
        return Tensor(d)

    @staticmethod
    def triu(x, k=0):
        d = _get_data(x)
        sh = _get_shape(d)
        if len(sh) == 2:
            H, W = sh
            flat = _get_flat(d)
            result = [flat[r * W + c] if c >= r + k else 0.0 for r in range(H) for c in range(W)]
            return Tensor(_nd_array(result, dtype='float32').reshape(sh))
        return Tensor(d)

    @staticmethod
    def cumsum(x, axis=None):
        flat = _get_flat(_get_data(x))
        sh = _get_shape(_get_data(x))
        result = []
        acc = 0.0
        for v in flat:
            acc += v; result.append(acc)
        return Tensor(_nd_array(result, dtype='float32').reshape(sh))

    @staticmethod
    def cumprod(x, axis=None):
        flat = _get_flat(_get_data(x))
        sh = _get_shape(_get_data(x))
        result = []
        acc = 1.0
        for v in flat:
            acc *= v; result.append(acc)
        return Tensor(_nd_array(result, dtype='float32').reshape(sh))

    @staticmethod
    def any(x, axis=None): return any(v != 0.0 for v in _get_flat(_get_data(x)))

    @staticmethod
    def all(x, axis=None): return all(v != 0.0 for v in _get_flat(_get_data(x)))

    @staticmethod
    def logical_and(x, y): return _elem_wise2(x, y, lambda a, b: 1.0 if (a != 0 and b != 0) else 0.0)

    @staticmethod
    def logical_or(x, y): return _elem_wise2(x, y, lambda a, b: 1.0 if (a != 0 or b != 0) else 0.0)

    @staticmethod
    def logical_not(x): return _elem_wise(x, lambda v: 0.0 if v != 0 else 1.0)

    @staticmethod
    def maximum(x, y): return _elem_wise2(x, y, max)

    @staticmethod
    def minimum(x, y): return _elem_wise2(x, y, min)

    @staticmethod
    def floor(x): return _elem_wise(x, math.floor)

    @staticmethod
    def ceil(x): return _elem_wise(x, math.ceil)

    @staticmethod
    def round(x, decimals=0): return _elem_wise(x, lambda v: round(v, decimals))

    @staticmethod
    def sign(x): return _elem_wise(x, lambda v: 1.0 if v > 0 else (-1.0 if v < 0 else 0.0))

    @staticmethod
    def sin(x): return _elem_wise(x, math.sin)

    @staticmethod
    def cos(x): return _elem_wise(x, math.cos)

    @staticmethod
    def tan(x): return _elem_wise(x, math.tan)

    @staticmethod
    def arcsin(x): return _elem_wise(x, math.asin)

    @staticmethod
    def arccos(x): return _elem_wise(x, math.acos)

    @staticmethod
    def arctan(x): return _elem_wise(x, math.atan)

    @staticmethod
    def arctan2(y, x): return _elem_wise2(y, x, math.atan2)

    @staticmethod
    def sinh(x): return _elem_wise(x, math.sinh)

    @staticmethod
    def cosh(x): return _elem_wise(x, math.cosh)

    @staticmethod
    def tanh(x): return _elem_wise(x, math.tanh)

    @staticmethod
    def power(x, y): return _elem_wise2(x, y, lambda a, b: a ** b)

    @staticmethod
    def cross(a, b, axis=-1):
        af, bf = _get_flat(_get_data(a)), _get_flat(_get_data(b))
        if len(af) == 3 and len(bf) == 3:
            result = [af[1]*bf[2]-af[2]*bf[1], af[2]*bf[0]-af[0]*bf[2], af[0]*bf[1]-af[1]*bf[0]]
            return Tensor(_nd_array(result, dtype='float32'))
        return Tensor(_nd_array([0.0], dtype='float32'))

    @staticmethod
    def outer(a, b):
        af = _get_flat(_get_data(a)); bf = _get_flat(_get_data(b))
        result = [ai * bj for ai in af for bj in bf]
        return Tensor(_nd_array(result, dtype='float32').reshape((len(af), len(bf))))

    @staticmethod
    def trace(x, offset=0, axis1=0, axis2=1):
        d = _get_data(x)
        sh = _get_shape(d)
        if len(sh) < 2:
            return Tensor(_nd_array([sum(_get_flat(d))], dtype='float32'))
        H, W = sh[0], sh[1]
        flat = _get_flat(d)
        total = sum(flat[r * W + r + offset] for r in range(min(H, W - offset)) if 0 <= r + offset < W)
        return Tensor(_nd_array([total], dtype='float32'))

    @staticmethod
    def diag(x, k=0):
        d = _get_data(x)
        sh = _get_shape(d)
        flat = _get_flat(d)
        if len(sh) == 1:
            n = len(flat)
            size = n + abs(k)
            result = [0.0] * (size * size)
            for i in range(n):
                r = i if k >= 0 else i - k
                c = i + k if k >= 0 else i
                if 0 <= r < size and 0 <= c < size:
                    result[r * size + c] = flat[i]
            return Tensor(_nd_array(result, dtype='float32').reshape((size, size)))
        elif len(sh) == 2:
            H, W = sh
            n = min(H - max(0, -k), W - max(0, k))
            result = [flat[(i + max(0, -k)) * W + i + max(0, k)] for i in range(max(0, n))]
            return Tensor(_nd_array(result, dtype='float32'))
        return Tensor(d)

    @staticmethod
    def diagflat(x, k=0): return numpy.diag(x, k=k)


# ── random ────────────────────────────────────────────────────────────────────

class random:
    """jax.random-compatible random API (uses stdlib random)."""

    @staticmethod
    def PRNGKey(seed: int): return [int(seed), 0]

    @staticmethod
    def split(key, num=2):
        seed = int(key[0]) if hasattr(key, '__getitem__') else int(key)
        rng = _rnd.Random(seed)
        return [[rng.randint(0, 2**31), 0] for _ in range(num)]

    @staticmethod
    def normal(key, shape=(), dtype='float32'):
        seed = int(key[0]) if hasattr(key, '__getitem__') else int(key)
        rng = _rnd.Random(seed)
        total = 1
        for s in (shape if shape else (1,)):
            total *= s
        flat = [math.sqrt(-2 * math.log(max(rng.random(), 1e-10))) * math.cos(2 * math.pi * rng.random())
                for _ in range(total)]
        sh = shape if shape else (total,)
        return Tensor(_nd_array(flat, dtype='float32').reshape(sh))

    @staticmethod
    def uniform(key, shape=(), minval=0.0, maxval=1.0, dtype='float32'):
        seed = int(key[0]) if hasattr(key, '__getitem__') else int(key)
        rng = _rnd.Random(seed)
        total = 1
        for s in (shape if shape else (1,)):
            total *= s
        flat = [rng.uniform(minval, maxval) for _ in range(total)]
        sh = shape if shape else (total,)
        return Tensor(_nd_array(flat, dtype='float32').reshape(sh))

    @staticmethod
    def randint(key, shape, minval, maxval, dtype='int32'):
        seed = int(key[0]) if hasattr(key, '__getitem__') else int(key)
        rng = _rnd.Random(seed)
        total = 1
        for s in shape:
            total *= s
        flat = [float(rng.randint(minval, maxval - 1)) for _ in range(total)]
        return Tensor(_nd_array(flat, dtype='float32').reshape(shape))

    @staticmethod
    def permutation(key, x, axis=0, independent=False):
        seed = int(key[0]) if hasattr(key, '__getitem__') else int(key)
        rng = _rnd.Random(seed)
        if isinstance(x, int):
            perm = list(range(x)); rng.shuffle(perm)
            return Tensor(_nd_array([float(i) for i in perm], dtype='float32'))
        flat = _get_flat(_get_data(x)); rng.shuffle(flat)
        return Tensor(_nd_array(flat, dtype='float32').reshape(_get_shape(_get_data(x))))

    @staticmethod
    def choice(key, a, shape=(), replace=True, p=None, axis=0):
        seed = int(key[0]) if hasattr(key, '__getitem__') else int(key)
        rng = _rnd.Random(seed)
        n = a if isinstance(a, int) else len(_get_flat(_get_data(a)))
        total = 1
        for s in (shape if shape else (1,)):
            total *= s
        if replace:
            chosen = [float(rng.randint(0, n - 1)) for _ in range(total)]
        else:
            pop = list(range(n)); rng.shuffle(pop)
            chosen = [float(i) for i in pop[:total]]
        sh = shape if shape else (total,)
        return Tensor(_nd_array(chosen, dtype='float32').reshape(sh))

    @staticmethod
    def bernoulli(key, p=0.5, shape=()):
        seed = int(key[0]) if hasattr(key, '__getitem__') else int(key)
        rng = _rnd.Random(seed)
        total = 1
        for s in (shape if shape else (1,)):
            total *= s
        flat = [1.0 if rng.random() < p else 0.0 for _ in range(total)]
        sh = shape if shape else (total,)
        return Tensor(_nd_array(flat, dtype='float32').reshape(sh))

    @staticmethod
    def gumbel(key, shape=()):
        seed = int(key[0]) if hasattr(key, '__getitem__') else int(key)
        rng = _rnd.Random(seed)
        total = 1
        for s in (shape if shape else (1,)):
            total *= s
        flat = [-math.log(-math.log(max(rng.random(), 1e-20))) for _ in range(total)]
        sh = shape if shape else (total,)
        return Tensor(_nd_array(flat, dtype='float32').reshape(sh))

    @staticmethod
    def categorical(key, logits, axis=-1):
        seed = int(key[0]) if hasattr(key, '__getitem__') else int(key)
        rng = _rnd.Random(seed)
        flat = _get_flat(_get_data(logits))
        noise = [-math.log(-math.log(max(rng.random(), 1e-20))) for _ in flat]
        best = max(range(len(flat)), key=lambda i: flat[i] + noise[i])
        return Tensor(_nd_array([float(best)], dtype='float32'))

    @staticmethod
    def fold_in(key, data):
        k0 = int(key[0]) if hasattr(key, '__getitem__') else int(key)
        k1 = int(key[1]) if hasattr(key, '__getitem__') else 0
        return [k0 ^ int(data), k1]


# ── tree_util ─────────────────────────────────────────────────────────────────

class tree_util:
    """jax.tree_util-like helpers for nested containers of Tensors."""

    @staticmethod
    def tree_map(f, tree, *rest):
        if isinstance(tree, Tensor):
            return f(tree, *rest)
        elif isinstance(tree, dict):
            return {k: tree_util.tree_map(f, v, *[r[k] for r in rest]) for k, v in tree.items()}
        elif isinstance(tree, (list, tuple)):
            mapped = [tree_util.tree_map(f, v, *[r[i] for r in rest]) for i, v in enumerate(tree)]
            return type(tree)(mapped)
        return f(tree, *rest)

    @staticmethod
    def tree_leaves(tree):
        if isinstance(tree, Tensor): return [tree]
        elif isinstance(tree, dict):
            leaves = []
            for v in tree.values(): leaves.extend(tree_util.tree_leaves(v))
            return leaves
        elif isinstance(tree, (list, tuple)):
            leaves = []
            for v in tree: leaves.extend(tree_util.tree_leaves(v))
            return leaves
        return [tree]

    @staticmethod
    def tree_flatten(tree): return tree_util.tree_leaves(tree), tree

    @staticmethod
    def tree_unflatten(treedef, leaves): return leaves

    @staticmethod
    def tree_reduce(f, tree, initializer=None):
        leaves = tree_util.tree_leaves(tree)
        if not leaves: return initializer
        result = leaves[0] if initializer is None else f(initializer, leaves[0])
        for leaf in (leaves[1:] if initializer is None else leaves):
            result = f(result, leaf)
        return result


# ── nn ────────────────────────────────────────────────────────────────────────

def _erf(z):
    t = 1.0 / (1.0 + 0.5 * abs(z))
    tau = t * math.exp(-z*z - 1.26551223 + t*(1.00002368 + t*(0.37409196 + t*(
        0.09678418 + t*(-0.18628806 + t*(0.27886807 + t*(-1.13520398 + t*(
        1.48851587 + t*(-0.82215223 + t*0.17087294)))))))))
    return (1 - tau) if z >= 0 else tau - 1


class nn:
    """jax.nn-like operations and initializers."""

    @staticmethod
    def relu(x): return _elem_wise(x, lambda v: max(0.0, v))

    @staticmethod
    def leaky_relu(x, negative_slope=0.01):
        return _elem_wise(x, lambda v: v if v >= 0 else negative_slope * v)

    @staticmethod
    def elu(x, alpha=1.0):
        return _elem_wise(x, lambda v: v if v >= 0 else alpha * (math.exp(v) - 1))

    @staticmethod
    def gelu(x, approximate=False):
        if approximate:
            return _elem_wise(x, lambda v: 0.5 * v * (1 + math.tanh(math.sqrt(2 / math.pi) * (v + 0.044715 * v**3))))
        return _elem_wise(x, lambda v: v * 0.5 * (1 + _erf(v / math.sqrt(2))))

    @staticmethod
    def silu(x):
        return _elem_wise(x, lambda v: v / (1 + math.exp(-v)))

    @staticmethod
    def sigmoid(x):
        return _elem_wise(x, lambda v: 1 / (1 + math.exp(-v)))

    @staticmethod
    def softmax(x, axis=-1):
        flat = _get_flat(_get_data(x))
        sh = _get_shape(_get_data(x))
        m = max(flat)
        exps = [math.exp(v - m) for v in flat]
        s = sum(exps)
        return Tensor(_nd_array([e / s for e in exps], dtype='float32').reshape(sh))

    @staticmethod
    def log_softmax(x, axis=-1):
        flat = _get_flat(_get_data(x))
        sh = _get_shape(_get_data(x))
        m = max(flat)
        log_sum = math.log(sum(math.exp(v - m) for v in flat)) + m
        return Tensor(_nd_array([v - log_sum for v in flat], dtype='float32').reshape(sh))

    @staticmethod
    def softplus(x, beta=1):
        return _elem_wise(x, lambda v: (1 / beta) * math.log1p(math.exp(beta * v)))

    @staticmethod
    def one_hot(x, num_classes):
        flat = _get_flat(_get_data(x))
        sh = _get_shape(_get_data(x))
        result = []
        for idx in flat:
            row = [0.0] * num_classes
            c = int(idx)
            if 0 <= c < num_classes: row[c] = 1.0
            result.extend(row)
        new_sh = sh + (num_classes,) if sh else (len(flat), num_classes)
        return Tensor(_nd_array(result, dtype='float32').reshape(new_sh))

    @staticmethod
    def normalize(x, axis=-1, eps=1e-12):
        flat = _get_flat(_get_data(x))
        sh = _get_shape(_get_data(x))
        norm = max(math.sqrt(sum(v * v for v in flat)), eps)
        return Tensor(_nd_array([v / norm for v in flat], dtype='float32').reshape(sh))

    class initializers:
        """Weight initialization functions."""

        @staticmethod
        def glorot_uniform():
            def init(key, shape, dtype='float32'):
                seed = int(key[0]) if hasattr(key, '__getitem__') else int(key)
                rng = _rnd.Random(seed)
                fan_in = shape[-2] if len(shape) >= 2 else shape[-1]
                fan_out = shape[-1]
                limit = math.sqrt(6.0 / (fan_in + fan_out))
                total = 1
                for s in shape: total *= s
                flat = [rng.uniform(-limit, limit) for _ in range(total)]
                return Tensor(_nd_array(flat, dtype=dtype).reshape(shape))
            return init

        @staticmethod
        def glorot_normal():
            def init(key, shape, dtype='float32'):
                seed = int(key[0]) if hasattr(key, '__getitem__') else int(key)
                rng = _rnd.Random(seed)
                fan_in = shape[-2] if len(shape) >= 2 else shape[-1]
                fan_out = shape[-1]
                std = math.sqrt(2.0 / (fan_in + fan_out))
                total = 1
                for s in shape: total *= s
                flat = [std * math.sqrt(-2 * math.log(max(rng.random(), 1e-10))) * math.cos(2 * math.pi * rng.random())
                        for _ in range(total)]
                return Tensor(_nd_array(flat, dtype=dtype).reshape(shape))
            return init

        @staticmethod
        def lecun_normal():
            def init(key, shape, dtype='float32'):
                seed = int(key[0]) if hasattr(key, '__getitem__') else int(key)
                rng = _rnd.Random(seed)
                fan_in = shape[-2] if len(shape) >= 2 else shape[-1]
                std = math.sqrt(1.0 / fan_in)
                total = 1
                for s in shape: total *= s
                flat = [std * math.sqrt(-2 * math.log(max(rng.random(), 1e-10))) * math.cos(2 * math.pi * rng.random())
                        for _ in range(total)]
                return Tensor(_nd_array(flat, dtype=dtype).reshape(shape))
            return init

        @staticmethod
        def he_normal():
            def init(key, shape, dtype='float32'):
                seed = int(key[0]) if hasattr(key, '__getitem__') else int(key)
                rng = _rnd.Random(seed)
                fan_in = shape[-2] if len(shape) >= 2 else shape[-1]
                std = math.sqrt(2.0 / fan_in)
                total = 1
                for s in shape: total *= s
                flat = [std * math.sqrt(-2 * math.log(max(rng.random(), 1e-10))) * math.cos(2 * math.pi * rng.random())
                        for _ in range(total)]
                return Tensor(_nd_array(flat, dtype=dtype).reshape(shape))
            return init

        @staticmethod
        def zeros():
            def init(key, shape, dtype='float32'):
                return Tensor(zeros(shape, dtype=dtype))
            return init

        @staticmethod
        def ones():
            def init(key, shape, dtype='float32'):
                return Tensor(ones(shape, dtype=dtype))
            return init

        @staticmethod
        def constant(value):
            def init(key, shape, dtype='float32'):
                return Tensor(full(shape, value, dtype=dtype))
            return init


__all__ = [
    'grad', 'value_and_grad', 'jvp', 'vjp', 'vmap', 'jit', 'scan', 'pmap',
    'lax', 'numpy', 'random', 'tree_util', 'nn',
]
