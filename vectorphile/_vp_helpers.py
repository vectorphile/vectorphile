"""
Vectorphile helper shims — replace all np.xxx usages with pure Python / NDArray ops.
Every module that formerly used numpy imports this via:
    from vectorphile._vp_helpers import *
"""
from __future__ import annotations
import math as _math
import operator as _operator
import itertools as _itertools
import array as _array_mod

from vectorphile._core.ndarray import (
    NDArray, array as _nd_array, zeros as _nd_zeros, ones as _nd_ones,
    full as _nd_full, empty as _nd_empty, eye as _nd_eye,
    arange as _nd_arange, linspace as _nd_linspace,
    _elementwise_unary, _elementwise_binary, _binary_op,
    _broadcast_shapes, _broadcast_idx, _c_strides, _TYPECODES, _prod,
    matmul as _matmul_core,
)

# ── Constants ─────────────────────────────────────────────────────────────────
# (exposed so files using _math.pi etc. work)

# ── NDArray-like helpers (array creation) ─────────────────────────────────────

def _nd_zeros_like(a, dtype=None):
    dt = dtype or (a.dtype if hasattr(a, 'dtype') else 'float32')
    sh = a.shape if hasattr(a, 'shape') else (len(a),)
    return _nd_zeros(sh, dtype=dt)

def _nd_ones_like(a, dtype=None):
    dt = dtype or (a.dtype if hasattr(a, 'dtype') else 'float32')
    sh = a.shape if hasattr(a, 'shape') else (len(a),)
    return _nd_ones(sh, dtype=dt)

def _nd_empty_like(a, dtype=None):
    return _nd_zeros_like(a, dtype=dtype)

def _nd_full_like(a, fill_value, dtype=None):
    dt = dtype or (a.dtype if hasattr(a, 'dtype') else 'float32')
    sh = a.shape if hasattr(a, 'shape') else (len(a),)
    return _nd_full(sh, fill_value, dtype=dt)

def _nd_copy(a):
    if hasattr(a, 'copy'): return a.copy()
    return _nd_array(a)

# ── Math (elementwise on NDArray or scalar) ───────────────────────────────────

def _nd_elementwise(a, fn, out_dtype=None):
    if type(a).__name__ == 'GArray':
        # GArray: apply via method if available, else torch fallback
        mname = getattr(fn, '__name__', '') or getattr(fn, '__qualname__', '') or ''
        for attr in ('exp', 'log', 'sqrt', 'tanh', 'sin', 'cos', 'abs', 'sign', 'erf', 'sigmoid', 'relu'):
            if attr in mname:
                m = getattr(a, attr, None)
                if m: return m()
        # sigmoid/relu by lambda name pattern won't match; apply torch op on tensor
        try:
            import torch as _t_ew
            result = _t_ew.tensor([fn(float(v)) for v in a.t.flatten().tolist()], dtype=a.t.dtype, device=a.t.device).reshape(a.t.shape)
            return type(a)(result)
        except Exception:
            pass
        return a
    if isinstance(a, NDArray):
        return _elementwise_unary(a, fn, out_dtype=out_dtype)
    if isinstance(a, (list, tuple)):
        return [fn(v) for v in a]
    return fn(a)

def _to_nd(x):
    """Convert any array-like to NDArray."""
    if isinstance(x, NDArray): return x
    return _nd_array(x)

def _math_exp(x):
    if type(x).__name__ == 'GArray': return x.exp()
    nd = _to_nd(x)
    return _elementwise_unary(nd, _math.exp)
def _math_log(x):
    if type(x).__name__ == 'GArray': return x.log()
    from vectorphile._core.ndarray import _log_eps_scalar
    nd = _to_nd(x)
    return _elementwise_unary(nd, _log_eps_scalar)
def _math_log2(x):
    nd = _to_nd(x)
    return _elementwise_unary(nd, _math.log2)
def _math_log10(x):
    nd = _to_nd(x)
    return _elementwise_unary(nd, _math.log10)
def _math_sqrt(x):
    if type(x).__name__ == 'GArray': return x.sqrt()
    nd = _to_nd(x)
    return _elementwise_unary(nd, _math.sqrt)
def _math_tanh(x):
    if type(x).__name__ == 'GArray': return x.tanh()
    return _nd_elementwise(x, _math.tanh)
def _math_erf(x):
    if type(x).__name__ == 'GArray': return x.erf()
    from vectorphile._core.ndarray import _erf_scalar
    return _nd_elementwise(x, _erf_scalar)
def _math_sin(x):
    if type(x).__name__ == 'GArray': return x.sin()
    return _nd_elementwise(x, _math.sin)
def _math_cos(x):
    if type(x).__name__ == 'GArray': return x.cos()
    return _nd_elementwise(x, _math.cos)
def _math_tan(x): return _nd_elementwise(x, _math.tan)
def _math_floor(x): return _nd_elementwise(x, _math.floor)
def _math_ceil(x): return _nd_elementwise(x, _math.ceil)

def _nd_abs_fn(x):
    if type(x).__name__ == 'GArray': return x.abs()
    return _nd_elementwise(x, abs)
def _nd_sign_fn(x):
    if type(x).__name__ == 'GArray': return x.sign()
    from vectorphile._core.ndarray import _sign_scalar
    return _nd_elementwise(x, _sign_scalar)

def _nd_power_fn(a, b):
    if isinstance(a, NDArray) and isinstance(b, NDArray):
        return _elementwise_binary(a, b, _operator.pow)
    if isinstance(a, NDArray):
        return _elementwise_unary(a, lambda x: x ** b)
    if isinstance(b, NDArray):
        return _elementwise_unary(b, lambda x: a ** x)
    return a ** b

def _nd_maximum_fn(a, b):
    if isinstance(a, NDArray) and isinstance(b, NDArray):
        return _elementwise_binary(a, b, max)
    if isinstance(a, NDArray):
        if isinstance(b, (int, float)) and 'float' in a.dtype:
            return _elementwise_binary(a, b, max)
        return _elementwise_unary(a, lambda x: max(x, b))
    if isinstance(b, NDArray):
        if isinstance(a, (int, float)) and 'float' in b.dtype:
            return _elementwise_binary(a, b, max)
        return _elementwise_unary(b, lambda x: max(a, x))
    return max(a, b)

def _nd_minimum_fn(a, b):
    if isinstance(a, NDArray) and isinstance(b, NDArray):
        return _elementwise_binary(a, b, min)
    if isinstance(a, NDArray):
        if isinstance(b, (int, float)) and 'float' in a.dtype:
            return _elementwise_binary(a, b, min)
        return _elementwise_unary(a, lambda x: min(x, b))
    if isinstance(b, NDArray):
        if isinstance(a, (int, float)) and 'float' in b.dtype:
            return _elementwise_binary(a, b, min)
        return _elementwise_unary(b, lambda x: min(a, x))
    return min(a, b)

def _nd_clip_fn(a, a_min, a_max):
    if isinstance(a, NDArray): return a.clip(a_min, a_max)
    return max(a_min, min(a_max, a))

def _nd_isnan_fn(a): return _nd_elementwise(a, lambda x: x != x, out_dtype='bool')
def _nd_isinf_fn(a): return _nd_elementwise(a, lambda x: _math.isinf(x) if isinstance(x, float) else False, out_dtype='bool')
def _nd_round_fn(a, decimals=0):
    factor = 10 ** decimals
    return _nd_elementwise(a, lambda x: round(x * factor) / factor)

def _nd_where_fn(cond, x=None, y=None):
    # 1-arg mode: return (indices_where_true,) like np.where(mask)
    if x is None and y is None:
        flat = cond.to_flat_list() if isinstance(cond, NDArray) else list(cond)
        indices = _nd_array([i for i, v in enumerate(flat) if v], dtype='int64')
        return (indices,)
    from vectorphile.core.tensor import _nd_where
    if isinstance(cond, NDArray):
        if not isinstance(x, NDArray):
            x = _nd_full(cond.shape, float(x) if 'float' in cond.dtype else x,
                         dtype=y.dtype if isinstance(y, NDArray) else 'float32')
        if not isinstance(y, NDArray):
            y = _nd_full(cond.shape, float(y) if 'float' in x.dtype else y, dtype=x.dtype)
        return _nd_where(cond, x, y)
    if isinstance(cond, (list, tuple)):
        xf = x if isinstance(x, (list, tuple)) else [x]*len(cond)
        yf = y if isinstance(y, (list, tuple)) else [y]*len(cond)
        return [xf[i] if cond[i] else yf[i] for i in range(len(cond))]
    return x if cond else y

# ── Shape ops ─────────────────────────────────────────────────────────────────

def _nd_reshape_fn(a, *shape):
    if hasattr(a, 'reshape'): return a.reshape(*shape)
    return _nd_array(a).reshape(*shape)

def _nd_transpose_fn(a, axes=None):
    if hasattr(a, 'transpose'): return a.transpose(axes)
    return a

def _nd_squeeze_fn(a, axis=None):
    if hasattr(a, 'squeeze'): return a.squeeze(axis)
    return a

def _nd_expand_dims_fn(a, axis):
    if isinstance(a, NDArray):
        shape = list(a.shape)
        if axis < 0: axis = len(shape) + 1 + axis
        shape.insert(axis, 1)
        return a.reshape(tuple(shape))
    if hasattr(a, 'unsqueeze'): return a.unsqueeze(axis)
    if hasattr(a, 'expand_dims'): return a.expand_dims(axis)
    return a

def _nd_concatenate_fn(arrays, axis=0):
    from vectorphile.core.tensor import _nd_concatenate
    nd_arrays = []
    for a in arrays:
        if isinstance(a, NDArray):
            nd_arrays.append(a)
        elif isinstance(a, (list, tuple)):
            nd_arrays.append(_nd_array(a))
        else:
            nd_arrays.append(_nd_array([float(a)] if not hasattr(a, '__len__') else a))
    return _nd_concatenate(nd_arrays, axis=axis)

def _nd_stack_fn(arrays, axis=0):
    from vectorphile.core.tensor import _nd_stack
    if any(type(a).__name__ == 'GArray' for a in arrays):
        import torch as _t_st
        from vectorphile.gpu import GArray as _GAst
        ts = [a.t if type(a).__name__ == 'GArray' else _t_st.tensor(list(a.flatten().to_flat_list()), dtype=_t_st.float32).reshape(a.shape) for a in arrays]
        return _GAst(_t_st.stack(ts, dim=axis))
    if all(isinstance(a, NDArray) for a in arrays):
        return _nd_stack(arrays, axis=axis)
    return _nd_stack([_nd_array(a) for a in arrays], axis=axis)

def _nd_split_fn(a, indices_or_sections, axis=0):
    from vectorphile.core.tensor import _nd_split
    if isinstance(a, NDArray):
        return _nd_split(a, indices_or_sections, axis=axis)
    return [a]

def _nd_tile_fn(a, reps):
    from vectorphile.core.tensor import _nd_tile
    if isinstance(a, NDArray): return _nd_tile(a, reps)
    return _nd_tile(_nd_array(a), reps)

def _nd_repeat_fn(a, repeats, axis=None):
    if isinstance(a, NDArray):
        flat = a.to_flat_list()
        if axis is None:
            result = []
            for v in flat: result.extend([v] * repeats)
            return _nd_array(result, dtype=a.dtype)
    return a

def _nd_broadcast_to_fn(a, shape):
    from vectorphile.core.tensor import _nd_broadcast_to
    if isinstance(a, NDArray): return _nd_broadcast_to(a, shape)
    return _nd_broadcast_to(_nd_array(a), shape)

def _nd_flip_fn(a, axis=None):
    from vectorphile.core.tensor import _nd_flip
    if isinstance(a, NDArray): return _nd_flip(a, dims=axis)
    return _nd_flip(_nd_array(a), dims=axis)

def _nd_roll_fn(a, shifts, axis=None):
    from vectorphile.core.tensor import _nd_roll
    if isinstance(a, NDArray): return _nd_roll(a, shifts, dims=axis)
    return _nd_roll(_nd_array(a), shifts, dims=axis)

def _nd_pad_fn(a, pad_width, mode='constant', constant_values=0, **kw):
    from vectorphile.core.tensor import _nd_pad
    cv = kw.get('constant_values', constant_values)
    if isinstance(a, NDArray): return _nd_pad(a, pad_width, mode=mode, constant_values=cv)
    return _nd_pad(_nd_array(a), pad_width, mode=mode, constant_values=cv)

def _nd_swapaxes_fn(a, axis1, axis2):
    from vectorphile.core.tensor import _nd_swapaxes
    if type(a).__name__ == 'GArray':
        import torch as _t_sa
        ndim = a.t.ndim
        ax1 = axis1 % ndim; ax2 = axis2 % ndim
        return type(a)(a.t.transpose(ax1, ax2))
    if isinstance(a, NDArray): return _nd_swapaxes(a, axis1, axis2)
    return a

def _nd_moveaxis_fn(a, source, dest):
    from vectorphile.core.tensor import _nd_moveaxis
    if isinstance(a, NDArray): return _nd_moveaxis(a, source, dest)
    return a

def _nd_rot90_fn(a, k=1, axes=(0, 1)):
    if not isinstance(a, NDArray): a = _nd_array(a)
    for _ in range(k % 4):
        from vectorphile.core.tensor import _nd_flip
        a = _nd_flip(a.transpose((axes[1], axes[0])), dims=[0])
    return a

def _nd_diag_fn(a, k=0):
    from vectorphile.core.tensor import _nd_diag
    if not isinstance(a, NDArray): a = _nd_array(a)
    return _nd_diag(a, k=k)

def _nd_diagflat_fn(a, k=0):
    from vectorphile.core.tensor import _nd_diag
    if not isinstance(a, NDArray): a = _nd_array(a)
    flat = _nd_array(a.to_flat_list(), dtype=a.dtype)
    return _nd_diag(flat, k=k)

def _nd_diagonal_fn(a, offset=0, axis1=0, axis2=1):
    if hasattr(a, 'diagonal'): return a.diagonal(offset=offset)
    return []

def _nd_trace_fn(a, offset=0):
    if hasattr(a, 'trace'): return a.trace()
    return 0.0

def _nd_tril_fn(a, k=0):
    from vectorphile.core.tensor import _nd_tril
    if not isinstance(a, NDArray): a = _nd_array(a)
    return _nd_tril(a, k=k)

def _nd_triu_fn(a, k=0):
    from vectorphile.core.tensor import _nd_triu
    if not isinstance(a, NDArray): a = _nd_array(a)
    return _nd_triu(a, k=k)

def _nd_vstack_fn(arrays):
    from vectorphile.core.tensor import _nd_concatenate, _nd_atleast_nd
    arrs = [_nd_atleast_nd(a if isinstance(a, NDArray) else _nd_array(a), 2) for a in arrays]
    return _nd_concatenate(arrs, axis=0)

def _nd_hstack_fn(arrays):
    from vectorphile.core.tensor import _nd_concatenate
    arrs = [a if isinstance(a, NDArray) else _nd_array(a) for a in arrays]
    if arrs[0].ndim == 1: return _nd_concatenate(arrs, axis=0)
    return _nd_concatenate(arrs, axis=1)

def _nd_dstack_fn(arrays):
    from vectorphile.core.tensor import _nd_concatenate, _nd_atleast_nd
    arrs = [_nd_atleast_nd(a if isinstance(a, NDArray) else _nd_array(a), 3) for a in arrays]
    return _nd_concatenate(arrs, axis=2)

def _nd_column_stack_fn(arrays):
    arrs = []
    for a in arrays:
        if not isinstance(a, NDArray): a = _nd_array(a)
        if a.ndim == 1: a = a.unsqueeze(1)
        arrs.append(a)
    from vectorphile.core.tensor import _nd_concatenate
    return _nd_concatenate(arrs, axis=1)

def _nd_atleast_1d_fn(*arrays):
    from vectorphile.core.tensor import _nd_atleast_nd
    return [_nd_atleast_nd(a if isinstance(a, NDArray) else _nd_array(a), 1) for a in arrays]

def _nd_atleast_2d_fn(*arrays):
    from vectorphile.core.tensor import _nd_atleast_nd
    return [_nd_atleast_nd(a if isinstance(a, NDArray) else _nd_array(a), 2) for a in arrays]

def _nd_atleast_3d_fn(*arrays):
    from vectorphile.core.tensor import _nd_atleast_nd
    return [_nd_atleast_nd(a if isinstance(a, NDArray) else _nd_array(a), 3) for a in arrays]

def _nd_meshgrid_fn(*arrays, indexing='ij', **kw):
    from vectorphile.core.tensor import _nd_meshgrid
    arrs = [a if isinstance(a, NDArray) else _nd_array(a) for a in arrays]
    return _nd_meshgrid(*arrs, indexing=indexing)

# ── Reductions ────────────────────────────────────────────────────────────────

def _nd_sum_fn(a, axis=None, keepdims=False, **kw):
    if isinstance(a, NDArray): return a.sum(axis=axis, keepdims=keepdims)
    return sum(a)

def _nd_mean_fn(a, axis=None, keepdims=False, **kw):
    if isinstance(a, NDArray): return a.mean(axis=axis, keepdims=keepdims)
    return sum(a) / len(a)

def _nd_max_fn(a, axis=None, keepdims=False, **kw):
    if isinstance(a, NDArray): return a.max(axis=axis, keepdims=keepdims)
    return max(a)

def _nd_min_fn(a, axis=None, keepdims=False, **kw):
    if isinstance(a, NDArray): return a.min(axis=axis, keepdims=keepdims)
    return min(a)

def _nd_argmax_fn(a, axis=None):
    if isinstance(a, NDArray): return a.argmax(axis=axis)
    flat = list(a); return flat.index(max(flat))

def _nd_argmin_fn(a, axis=None):
    if isinstance(a, NDArray): return a.argmin(axis=axis)
    flat = list(a); return flat.index(min(flat))

def _nd_argsort_fn(a, axis=-1, **kw):
    from vectorphile.core.tensor import _nd_argsort
    if isinstance(a, NDArray): return _nd_argsort(a, axis=axis)
    flat = list(a); return sorted(range(len(flat)), key=lambda i: flat[i])

def _nd_sort_fn(a, axis=-1, **kw):
    from vectorphile.core.tensor import _nd_sort
    if isinstance(a, NDArray): return _nd_sort(a, axis=axis)[0]
    return sorted(a)

def _nd_unique_fn(a, **kw):
    from vectorphile.core.tensor import _nd_unique
    if isinstance(a, NDArray): return _nd_unique(a, **kw)
    vals = sorted(set(a))
    return _nd_array(vals)

def _nd_nonzero_fn(a):
    if isinstance(a, NDArray): return a.nonzero()
    return tuple([i for i, v in enumerate(a) if v],)

def _nd_cumsum_fn(a, axis=None, **kw):
    from vectorphile.core.tensor import _nd_cumsum
    if isinstance(a, NDArray): return _nd_cumsum(a, axis=0 if axis is None else axis)
    s = 0; result = []
    for v in a: s += int(v); result.append(s)
    return result

def _nd_cumprod_fn(a, axis=None, **kw):
    if isinstance(a, NDArray):
        import itertools
        result = a.copy()
        ax = 0 if axis is None else axis
        n = result.shape[ax]
        for i in range(1, n):
            for idx in itertools.product(*[range(s) for s in result.shape]):
                if idx[ax] == i:
                    prev = list(idx); prev[ax] = i - 1
                    r_buf = result._get_buf(); r_buf[result._flat_idx(idx)] *= r_buf[result._flat_idx(tuple(prev))]
        return result
    p = 1.0; result = []
    for v in a: p *= v; result.append(p)
    return result

def _nd_prod_fn(a, axis=None, keepdims=False, **kw):
    if isinstance(a, NDArray): return a.prod(axis=axis, keepdims=keepdims)
    p = 1.0
    for v in a: p *= v
    return p

def _nd_std_fn(a, axis=None, ddof=0, keepdims=False, **kw):
    if isinstance(a, NDArray): return a.std(axis=axis, keepdims=keepdims, ddof=ddof)
    flat = list(a); m = sum(flat)/len(flat)
    return _math.sqrt(sum((x-m)**2 for x in flat) / max(1, len(flat)-ddof))

def _nd_var_fn(a, axis=None, ddof=0, keepdims=False, **kw):
    if isinstance(a, NDArray): return a.var(axis=axis, keepdims=keepdims, ddof=ddof)
    flat = list(a); m = sum(flat)/len(flat)
    return sum((x-m)**2 for x in flat) / max(1, len(flat)-ddof)

def _nd_diff_fn(a, n=1, axis=-1, **kw):
    from vectorphile.core.tensor import _nd_diff
    if isinstance(a, NDArray): return _nd_diff(a, n=n, axis=axis)
    return [a[i+1]-a[i] for i in range(len(a)-1)]

def _nd_gradient_fn(a, *args, axis=None, **kw):
    from vectorphile.core.tensor import _nd_gradient
    if isinstance(a, NDArray): return _nd_gradient(a, axis=axis)
    return [a[i+1]-a[i-1] if 0 < i < len(a)-1 else
            a[1]-a[0] if i == 0 else a[-1]-a[-2] for i in range(len(a))]

def _nd_searchsorted_fn(a, v, side='left', **kw):
    import bisect
    flat = a.to_flat_list() if isinstance(a, NDArray) else list(a)
    fn = bisect.bisect_right if side == 'right' else bisect.bisect_left
    if isinstance(v, (int, float)): return fn(flat, v)
    vals = v.to_flat_list() if isinstance(v, NDArray) else list(v)
    return [fn(flat, x) for x in vals]

def _nd_histogram_fn(a, bins=10, range=None, density=False, **kw):
    from vectorphile.core.tensor import _nd_histogram
    nd = a if isinstance(a, NDArray) else _nd_array(a)
    return _nd_histogram(nd, bins=bins, range=range, density=density)

def _nd_bincount_fn(a, weights=None, minlength=0):
    from vectorphile.core.tensor import _nd_bincount
    nd = a if isinstance(a, NDArray) else _nd_array(a)
    w = weights if isinstance(weights, NDArray) else (_nd_array(weights) if weights is not None else None)
    return _nd_bincount(nd, w, minlength)

def _nd_count_nonzero_fn(a, axis=None):
    if isinstance(a, NDArray):
        flat = a.to_flat_list()
        return sum(1 for x in flat if x != 0)
    return sum(1 for x in a if x != 0)

def _nd_any_fn(a, axis=None):
    if isinstance(a, NDArray): return a.any(axis=axis)
    return any(a)

def _nd_all_fn(a, axis=None):
    if isinstance(a, NDArray): return a.all(axis=axis)
    return all(a)

# ── Linear algebra ────────────────────────────────────────────────────────────

def _nd_matmul_fn(a, b):
    if isinstance(a, NDArray) and isinstance(b, NDArray): return _matmul_core(a, b)
    # GPU dispatch: if either operand is a GArray, use torch.matmul
    _ga = type(a).__name__ == 'GArray'
    _gb = type(b).__name__ == 'GArray'
    if _ga or _gb:
        import torch as _t
        from vectorphile.gpu import GArray as _GArray
        ref = a if _ga else b
        dev = ref.t.device
        def _to_torch(x):
            if type(x).__name__ == 'GArray': return x.t
            # NDArray → contiguous flat list, then tensor
            if isinstance(x, NDArray):
                flat = x.contiguous().to_flat_list()
                dtype = _t.float64 if 'float64' in str(x.dtype) else _t.float32
                return _t.tensor(flat, dtype=dtype, device=dev).reshape(x.shape)
            return _t.tensor(list(x), device=dev)
        ta = _to_torch(a)
        tb = _to_torch(b)
        return _GArray(_t.matmul(ta, tb))
    return _matmul_core(_nd_array(a), _nd_array(b))

def _nd_dot_fn(a, b):
    fa = a.to_flat_list() if isinstance(a, NDArray) else list(a)
    fb = b.to_flat_list() if isinstance(b, NDArray) else list(b)
    if isinstance(a, NDArray) and a.ndim > 1: return _nd_matmul_fn(a, b)
    return sum(x*y for x, y in zip(fa, fb))

def _nd_norm_fn(a, ord=None, axis=None, keepdims=False):
    from vectorphile.core.tensor import _nd_linalg_norm
    nd = a if isinstance(a, NDArray) else _nd_array(a)
    return _nd_linalg_norm(nd, ord=ord, axis=axis, keepdims=keepdims)

def _nd_inv_fn(a):
    from vectorphile.linalg import inv
    nd = a if isinstance(a, NDArray) else _nd_array(a)
    return inv(nd)

def _nd_det_fn(a):
    from vectorphile.linalg import det
    nd = a if isinstance(a, NDArray) else _nd_array(a)
    return det(nd)

def _nd_eig_fn(a):
    from vectorphile.linalg import eig
    nd = a if isinstance(a, NDArray) else _nd_array(a)
    return eig(nd)

def _nd_eigh_fn(a):
    from vectorphile.linalg import eigh
    nd = a if isinstance(a, NDArray) else _nd_array(a)
    return eigh(nd)

def _nd_svd_fn(a, full_matrices=True):
    from vectorphile.linalg import svd
    from vectorphile.core.tensor import Tensor as _T
    nd = a if isinstance(a, NDArray) else _nd_array(a)
    result = svd(nd, full_matrices=full_matrices)
    # Unwrap Tensors to NDArray
    def _unwrap(x):
        if isinstance(x, _T): return x._data if isinstance(x._data, NDArray) else _nd_array(x._data.to_flat_list() if hasattr(x._data,'to_flat_list') else list(x._data))
        return x if isinstance(x, NDArray) else _nd_array(x)
    return tuple(_unwrap(r) for r in result)

def _nd_solve_fn(a, b):
    from vectorphile.linalg import solve
    from vectorphile.core.tensor import Tensor as _T
    result = solve(a if isinstance(a, NDArray) else _nd_array(a),
                   b if isinstance(b, NDArray) else _nd_array(b))
    if isinstance(result, _T):
        d = result._data
        return d if isinstance(d, NDArray) else _nd_array(d.tolist() if hasattr(d,'tolist') else [float(d)])
    return result if isinstance(result, NDArray) else _nd_array(result)

def _nd_lstsq_fn(a, b, rcond=None):
    from vectorphile.linalg import lstsq
    from vectorphile.core.tensor import Tensor as _T
    results = lstsq(a if isinstance(a, NDArray) else _nd_array(a),
                    b if isinstance(b, NDArray) else _nd_array(b))
    def _unwrap(x):
        if isinstance(x, _T):
            d = x._data
            return d if isinstance(d, NDArray) else _nd_array(d.tolist() if hasattr(d,'tolist') else [float(d)])
        return x if isinstance(x, NDArray) else _nd_array([float(x)])
    return tuple(_unwrap(r) for r in results)

def _nd_cholesky_fn(a):
    from vectorphile.linalg import cholesky
    return cholesky(a if isinstance(a, NDArray) else _nd_array(a))

def _nd_qr_fn(a, mode='reduced'):
    from vectorphile.linalg import qr
    return qr(a if isinstance(a, NDArray) else _nd_array(a), mode=mode)

def _nd_pinv_fn(a, rcond=1e-15):
    from vectorphile.linalg import pinv
    return pinv(a if isinstance(a, NDArray) else _nd_array(a), rcond=rcond)

def _nd_matrix_rank_fn(a, tol=None):
    from vectorphile.linalg import matrix_rank
    return matrix_rank(a if isinstance(a, NDArray) else _nd_array(a), tol=tol)

def _nd_cond_fn(a, p=None):
    from vectorphile.linalg import cond
    return cond(a if isinstance(a, NDArray) else _nd_array(a), p=p)

def _nd_cross_fn(a, b):
    from vectorphile.linalg import cross
    return cross(a if isinstance(a, NDArray) else _nd_array(a),
                 b if isinstance(b, NDArray) else _nd_array(b))

def _nd_outer_fn(a, b):
    fa = a.to_flat_list() if isinstance(a, NDArray) else list(a)
    fb = b.to_flat_list() if isinstance(b, NDArray) else list(b)
    rows = []
    for x in fa: rows.append([x * y for y in fb])
    return _nd_array(rows, dtype='float32')

def _nd_kron_fn(a, b):
    from vectorphile.linalg import kron
    return kron(a if isinstance(a, NDArray) else _nd_array(a),
                b if isinstance(b, NDArray) else _nd_array(b))

def _nd_vander_fn(x, N=None, increasing=False):
    from vectorphile.linalg import vander
    return vander(x if isinstance(x, NDArray) else _nd_array(x), N=N, increasing=increasing)

def _nd_einsum_fn(eq, *operands, **kwargs):
    from vectorphile.core.tensor import _nd_einsum
    arrs = [a if isinstance(a, NDArray) else _nd_array(a) for a in operands]
    return _nd_einsum(eq, *arrs)

# ── Random ────────────────────────────────────────────────────────────────────

def _nd_rand_fn(*shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = tuple(shape[0])
    if not shape: shape = (1,)
    import array as _arr, random as _r
    n = 1
    for s in shape: n *= s
    buf = _arr.array('f', [_r.random() for _ in range(n)])
    strides = tuple(1 if i == len(shape)-1 else _prod(shape[i+1:]) for i in range(len(shape)))
    return NDArray(buf, shape, strides, 'float32')

def _nd_randn_fn(*shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = tuple(shape[0])
    if not shape: shape = (1,)
    import array as _arr, math as _m, random as _r
    n = 1
    for s in shape: n *= s
    vals = []
    for _ in range((n + 1) // 2):
        u1, u2 = max(_r.random(), 1e-10), _r.random()
        z1 = _m.sqrt(-2 * _m.log(u1)) * _m.cos(2 * _m.pi * u2)
        z2 = _m.sqrt(-2 * _m.log(u1)) * _m.sin(2 * _m.pi * u2)
        vals.extend([z1, z2])
    buf = _arr.array('f', vals[:n])
    strides = tuple(1 if i == len(shape)-1 else _prod(shape[i+1:]) for i in range(len(shape)))
    return NDArray(buf, shape, strides, 'float32')

def _nd_randint_fn(low, high=None, size=None):
    import random as _r
    if high is None: low, high = 0, low
    if size is None: return _r.randint(low, high - 1)
    if isinstance(size, int): size = (size,)
    n = 1
    for s in size: n *= s
    import array as _arr
    buf = _arr.array('l', [_r.randint(low, high - 1) for _ in range(n)])
    return NDArray(buf, size, _c_strides(size), 'int64')

def _nd_choice_fn(a, size=None, replace=True, p=None):
    import random as _r
    if isinstance(a, int): pool = list(range(a))
    else: pool = list(a.to_flat_list() if hasattr(a, 'to_flat_list') else a)
    if size is None: return _r.choice(pool)
    if replace:
        return _nd_array([_r.choice(pool) for _ in range(size)])
    return _nd_array(_r.sample(pool, size))

def _nd_shuffle_fn(a):
    from vectorphile.random import shuffle
    shuffle(a)

def _nd_permutation_fn(a):
    if isinstance(a, int):
        import random as _r
        idx = list(range(a))
        _r.shuffle(idx)
        return _nd_array(idx, dtype='int64')
    flat = list(a.to_flat_list() if hasattr(a, 'to_flat_list') else a)
    import random as _r
    _r.shuffle(flat)
    return _nd_array(flat, dtype='int64')

def _nd_uniform_fn(low=0.0, high=1.0, size=None):
    from vectorphile.random import uniform
    if size is None: return uniform(low, high)
    n = size if isinstance(size, int) else _prod(size)
    # Fast path: use OS urandom to generate uniform floats via struct
    import struct as _struct, os as _os, array as _ar
    raw = _os.urandom(4 * n)
    # Map uint32 to [0,1) then scale to [low, high)
    uints = _struct.unpack(f'{n}I', raw)
    scale = (high - low) * (1.0 / 4294967296.0)
    buf = _ar.array('f', [low + u * scale for u in uints])
    if isinstance(size, (tuple, list)):
        from vectorphile._core.ndarray import NDArray as _NDA, _c_strides as _cs, _DType
        shape = tuple(size)
        return _NDA(buf, shape, _cs(shape), _DType('float32'))
    from vectorphile._core.ndarray import NDArray as _NDA, _c_strides as _cs, _DType
    return _NDA(buf, (n,), _cs((n,)), _DType('float32'))

def _nd_normal_fn(loc=0.0, scale=1.0, size=None):
    from vectorphile.random import normal
    return normal(loc, scale, size)

def _nd_seed_fn(seed):
    from vectorphile.random import seed as _seed
    _seed(seed)

def _nd_random_fn(size=None):
    return _nd_uniform_fn(0.0, 1.0, size)

class _VpGenerator:
    """Minimal numpy.random.Generator shim returned by default_rng()."""
    def __init__(self, seed=None):
        self._seed = seed
        import random as _rmod
        self._rng = _rmod.Random(seed)

    def standard_normal(self, size=None):
        import math as _m
        if size is None:
            u1, u2 = max(self._rng.random(), 1e-10), self._rng.random()
            return _m.sqrt(-2*_m.log(u1)) * _m.cos(2*_m.pi*u2)
        if isinstance(size, int): size = (size,)
        n = 1
        for s in size: n *= s
        import math as _m2
        vals = []
        for _ in range((n+1)//2):
            u1, u2 = max(self._rng.random(), 1e-10), self._rng.random()
            z1 = _m2.sqrt(-2*_m2.log(u1)) * _m2.cos(2*_m2.pi*u2)
            z2 = _m2.sqrt(-2*_m2.log(u1)) * _m2.sin(2*_m2.pi*u2)
            vals.extend([z1, z2])
        return _nd_array(vals[:n], dtype='float32').reshape(*size)

    def integers(self, low, high=None, size=None, dtype=int):
        if high is None: low, high = 0, low
        if size is None: return self._rng.randint(low, high - 1)
        if isinstance(size, int): size = (size,)
        n = 1
        for s in size: n *= s
        vals = [self._rng.randint(low, high - 1) for _ in range(n)]
        return _nd_array(vals, dtype='int64').reshape(*size)

    def permutation(self, x):
        if isinstance(x, int): lst = list(range(x))
        else: lst = list(x.to_flat_list() if hasattr(x, 'to_flat_list') else x)
        self._rng.shuffle(lst)
        return _nd_array(lst, dtype='int64')

    def choice(self, a, size=None, replace=True, p=None):
        if isinstance(a, int): pool = list(range(a))
        else: pool = list(a.to_flat_list() if hasattr(a, 'to_flat_list') else a)
        if p is not None:
            # weighted choice using cumulative distribution
            import random as _r
            p_list = list(p.to_flat_list() if hasattr(p, 'to_flat_list') else p)
            cum = []; s = 0.0
            for w in p_list: s += w; cum.append(s)
            def _wchoice():
                r = _r.random() * cum[-1]
                for i, c in enumerate(cum):
                    if r <= c: return pool[i]
                return pool[-1]
            if size is None: return _wchoice()
            return _nd_array([_wchoice() for _ in range(size)])
        if size is None: return self._rng.choice(pool)
        if replace:
            return _nd_array([self._rng.choice(pool) for _ in range(size)])
        return _nd_array(self._rng.sample(pool, size))

    def uniform(self, low=0.0, high=1.0, size=None):
        if size is None: return self._rng.uniform(low, high)
        if isinstance(size, int): size = (size,)
        n = 1
        for s in size: n *= s
        return _nd_array([self._rng.uniform(low, high) for _ in range(n)]).reshape(*size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        a = self.standard_normal(size)
        return a * scale + loc if size else a * scale + loc

    def random(self, size=None):
        return self.uniform(0.0, 1.0, size)


class _VpRandomState(_VpGenerator):
    """numpy.random.RandomState shim (legacy API)."""
    def randint(self, low, high=None, size=None):
        return self.integers(low, high, size)

    def randn(self, *shape):
        return self.standard_normal(shape if shape else None)

    def rand(self, *shape):
        return self.uniform(0.0, 1.0, shape if shape else None)


class _vp_random:
    """Namespace shim for np.random.*"""
    rand = staticmethod(_nd_rand_fn)
    randn = staticmethod(_nd_randn_fn)
    randint = staticmethod(_nd_randint_fn)
    choice = staticmethod(_nd_choice_fn)
    shuffle = staticmethod(_nd_shuffle_fn)
    permutation = staticmethod(_nd_permutation_fn)
    uniform = staticmethod(_nd_uniform_fn)
    normal = staticmethod(_nd_normal_fn)
    seed = staticmethod(_nd_seed_fn)
    random = staticmethod(_nd_random_fn)

    @staticmethod
    def random(*args, **kw): return _nd_random_fn(*args, **kw)

    @staticmethod
    def default_rng(seed=None): return _VpGenerator(seed)

    @staticmethod
    def RandomState(seed=None): return _VpRandomState(seed)

# ── np.c_ / np.r_ column-stack shim ─────────────────────────────────────────

class _NpCIndexer:
    """Shim for np.c_[a, b] column-stack syntax."""
    def __getitem__(self, key):
        if not isinstance(key, tuple): key = (key,)
        arrays = []
        for k in key:
            a = k if isinstance(k, NDArray) else _nd_array(k)
            if a.ndim == 1: a = a.reshape(len(a), 1)
            arrays.append(a)
        return _nd_concatenate_fn(arrays, axis=1)

_np_c_ = _NpCIndexer()

# ── IO stubs ──────────────────────────────────────────────────────────────────

def _nd_save_fn(file, arr, **kw):
    import pickle
    fname = str(file) + ('.vpf' if not str(file).endswith('.vpf') else '')
    with open(fname, 'wb') as f: pickle.dump(arr, f)

def _nd_load_fn(file, **kw):
    import pickle
    with open(str(file), 'rb') as f: return pickle.load(f)

def _nd_savetxt_fn(fname, a, delimiter=' ', **kw):
    nd = a if isinstance(a, NDArray) else _nd_array(a)
    with open(str(fname), 'w') as f:
        for row in nd.tolist():
            if isinstance(row, (list, tuple)): f.write(delimiter.join(str(v) for v in row) + '\n')
            else: f.write(str(row) + '\n')

def _nd_loadtxt_fn(fname, delimiter=None, dtype='float32', **kw):
    rows = []
    with open(str(fname)) as f:
        for line in f:
            parts = line.split(delimiter) if delimiter else line.split()
            rows.append([float(x) for x in parts if x.strip()])
    return _nd_array(rows, dtype=dtype)

# ── Misc ──────────────────────────────────────────────────────────────────────

def _nd_isclose_fn(a, b, rtol=1e-05, atol=1e-08, **kw):
    fa = a.to_flat_list() if isinstance(a, NDArray) else ([a] if isinstance(a, (int, float)) else list(a))
    fb = b.to_flat_list() if isinstance(b, NDArray) else ([b] if isinstance(b, (int, float)) else list(b))
    return [abs(x - y) <= atol + rtol * abs(y) for x, y in zip(fa, fb)]

def _nd_allclose_fn(a, b, rtol=1e-05, atol=1e-08, **kw):
    return all(_nd_isclose_fn(a, b, rtol=rtol, atol=atol))

def _nd_take_along_axis_fn(a, indices, axis):
    if isinstance(a, NDArray) and isinstance(indices, NDArray):
        import itertools
        result = _nd_zeros(a.shape, dtype=a.dtype)
        for idx in itertools.product(*[range(s) for s in a.shape]):
            src_idx = list(idx); src_idx[axis] = indices._get_buf()[indices._flat_idx(idx)]
            result._get_buf()[result._flat_idx(idx)] = a._get_buf()[a._flat_idx(tuple(src_idx))]
        return result
    return a

def _nd_take_fn(a, indices, axis=None):
    if axis is None:
        flat = a.to_flat_list() if isinstance(a, NDArray) else list(a)
        idx = indices.to_flat_list() if isinstance(indices, NDArray) else list(indices)
        return _nd_array([flat[int(i)] for i in idx])
    return a

def _nd_put_fn(a, indices, values):
    if isinstance(a, NDArray):
        flat = a.to_flat_list()
        idx = indices.to_flat_list() if isinstance(indices, NDArray) else list(indices)
        vals = values.to_flat_list() if isinstance(values, NDArray) else ([values] * len(idx))
        for i, v in zip(idx, vals): flat[int(i)] = v
    return a

def _nd_apply_along_axis_fn(fn, axis, arr):
    if isinstance(arr, NDArray):
        import itertools
        other_axes = [i for i in range(arr.ndim) if i != axis]
        result = None
        for oidx in itertools.product(*[range(arr.shape[i]) for i in other_axes]):
            vec = [arr._get_buf()[arr._flat_idx(tuple(
                list(oidx[:axis]) + [k] + list(oidx[axis:])
            ))] for k in range(arr.shape[axis])]
            r = fn(vec)
            # for now return first result shape
            if result is None: result = arr.copy()
        return result
    return arr

def _nd_apply_over_axes_fn(fn, a, axes):
    for ax in axes: a = fn(a, ax)
    return a

def _nd_vectorize_fn(fn, **kw):
    def wrapper(*args):
        return [fn(*[a[i] if hasattr(a, '__getitem__') else a for a in args])
                for i in range(len(args[0]))]
    return wrapper

def _nd_frompyfunc_fn(fn, nin, nout):
    def wrapper(*args): return fn(*args)
    return wrapper

# catch-all for any _np_xxx stubs we may have set
def _np_removed(*args, **kw):
    raise RuntimeError("numpy was removed — use vectorphile's own API")

# Common stubs that may be hit
_np_float32 = 'float32'
_np_float64 = 'float64'
_np_int32 = 'int32'
_np_int64 = 'int64'
_np_bool_ = 'bool'
_np_newaxis = None


def _nd_log1p(x):
    """log(1 + x) — handles scalars, lists, NDArrays."""
    fn = lambda v: _math.log(1.0 + v)
    return _nd_elementwise(x, fn)

def _nd_histogram2d(x, y, bins=10):
    """2D histogram — returns (counts, xedges, yedges)."""
    xf = x.to_flat_list() if isinstance(x, NDArray) else list(x)
    yf = y.to_flat_list() if isinstance(y, NDArray) else list(y)
    xmin, xmax = min(xf), max(xf)
    ymin, ymax = min(yf), max(yf)
    nb = bins if isinstance(bins, int) else bins[0]
    xw = (xmax - xmin) / nb or 1.0
    yw = (ymax - ymin) / nb or 1.0
    counts = [[0.0]*nb for _ in range(nb)]
    for xi, yi in zip(xf, yf):
        bx = min(int((xi - xmin) / xw), nb - 1)
        by = min(int((yi - ymin) / yw), nb - 1)
        counts[bx][by] += 1.0
    xedges = [xmin + i*xw for i in range(nb+1)]
    yedges = [ymin + i*yw for i in range(nb+1)]
    return _nd_array(counts, dtype='float32'), _nd_array(xedges, dtype='float32'), _nd_array(yedges, dtype='float32')

# np.int16 → string dtype alias
_np_int16 = 'int16'

# ── numpy ufunc aliases (used in data/, nn/ etc.) ─────────────────────────────
import operator as _op
def _np_add(a, b):
    if isinstance(a, NDArray) or isinstance(b, NDArray):
        return _binary_op(a, b, _op.add)
    return a + b
def _np_subtract(a, b):
    if isinstance(a, NDArray) or isinstance(b, NDArray):
        return _binary_op(a, b, _op.sub)
    return a - b
def _np_multiply(a, b):
    if isinstance(a, NDArray) or isinstance(b, NDArray):
        return _binary_op(a, b, _op.mul)
    return a * b
def _np_true_divide(a, b):
    if isinstance(a, NDArray) or isinstance(b, NDArray):
        return _binary_op(a, b, _op.truediv)
    return a / b
def _np_floor_divide(a, b):
    if isinstance(a, NDArray) or isinstance(b, NDArray):
        return _binary_op(a, b, _op.floordiv)
    return a // b
def _np_power(a, b):
    if isinstance(a, NDArray) or isinstance(b, NDArray):
        return _binary_op(a, b, _op.pow)
    return a ** b
def _np_mod(a, b):
    if isinstance(a, NDArray) or isinstance(b, NDArray):
        return _binary_op(a, b, _op.mod)
    return a % b
def _np_equal(a, b):
    if isinstance(a, NDArray) or isinstance(b, NDArray):
        return _binary_op(a, b, _op.eq, out_dtype='bool')
    return a == b
def _np_not_equal(a, b):
    if isinstance(a, NDArray) or isinstance(b, NDArray):
        return _binary_op(a, b, _op.ne, out_dtype='bool')
    return a != b
def _np_greater(a, b):
    if isinstance(a, NDArray) or isinstance(b, NDArray):
        return _binary_op(a, b, _op.gt, out_dtype='bool')
    return a > b
def _np_less(a, b):
    if isinstance(a, NDArray) or isinstance(b, NDArray):
        return _binary_op(a, b, _op.lt, out_dtype='bool')
    return a < b
def _np_negative(a):
    if isinstance(a, NDArray): return _elementwise_unary(a, _op.neg)
    return -a

def _np_issubdtype(dtype, abstract):
    """Check if dtype is a subtype of abstract (number, floating, integer, etc.)."""
    dtype_str = str(dtype) if not isinstance(dtype, str) else dtype
    if abstract in ('number', 'numeric') or (hasattr(abstract, '__name__') and abstract.__name__ in ('number', 'generic')):
        return any(t in dtype_str for t in ('float', 'int', 'uint', 'bool', 'complex'))
    if abstract in ('floating',) or (hasattr(abstract, '__name__') and 'float' in abstract.__name__):
        return 'float' in dtype_str
    if abstract in ('integer', 'signedinteger') or (hasattr(abstract, '__name__') and 'int' in abstract.__name__):
        return 'int' in dtype_str
    return False

def _np_number():
    """Sentinel for numeric dtype checking."""
    pass
_np_number.__name__ = 'number'

def _np_dtype(t):
    """Return dtype string from numpy dtype or string."""
    if isinstance(t, str): return t
    if hasattr(t, 'name'): return t.name
    if hasattr(t, '__name__'): return t.__name__
    return str(t)

def _np_lexsort(keys):
    """Sort by multiple keys (last key is primary). Returns integer sort indices."""
    keys_list = list(keys)
    n = len(keys_list[0]) if keys_list else 0
    flat_keys = []
    for k in keys_list:
        if isinstance(k, NDArray):
            flat_keys.append(k.to_flat_list())
        else:
            flat_keys.append(list(k))
    # sort stably, last key is primary (most significant)
    indices = list(range(n))
    # sort by first key (least significant) first
    for key_vals in flat_keys:
        indices.sort(key=lambda i: key_vals[i])
    return _nd_array(indices, dtype='int64')


# ── nan-aware stat helpers (Series.mean/std/etc.) ─────────────────────────────

def _np_nanmean(a, axis=None):
    flat = a.to_flat_list() if isinstance(a, NDArray) else list(a)
    vals = [v for v in flat if not _math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else float('nan')

def _np_nanmedian(a, axis=None):
    flat = a.to_flat_list() if isinstance(a, NDArray) else list(a)
    vals = sorted(v for v in flat if not _math.isnan(float(v)))
    if not vals: return float('nan')
    n = len(vals)
    return (vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2)

def _np_nanstd(a, axis=None, ddof=0):
    flat = a.to_flat_list() if isinstance(a, NDArray) else list(a)
    vals = [float(v) for v in flat if not _math.isnan(float(v))]
    if len(vals) <= ddof: return float('nan')
    m = sum(vals) / len(vals)
    return _math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - ddof))

def _np_nanvar(a, axis=None, ddof=0):
    flat = a.to_flat_list() if isinstance(a, NDArray) else list(a)
    vals = [float(v) for v in flat if not _math.isnan(float(v))]
    if len(vals) <= ddof: return float('nan')
    m = sum(vals) / len(vals)
    return sum((x - m) ** 2 for x in vals) / (len(vals) - ddof)

def _np_nanmin(a, axis=None):
    flat = a.to_flat_list() if isinstance(a, NDArray) else list(a)
    vals = [float(v) for v in flat if not _math.isnan(float(v))]
    return min(vals) if vals else float('nan')

def _np_nanmax(a, axis=None):
    flat = a.to_flat_list() if isinstance(a, NDArray) else list(a)
    vals = [float(v) for v in flat if not _math.isnan(float(v))]
    return max(vals) if vals else float('nan')

def _np_nanpercentile(a, q, axis=None):
    flat = a.to_flat_list() if isinstance(a, NDArray) else list(a)
    vals = sorted(float(v) for v in flat if not _math.isnan(float(v)))
    if not vals: return float('nan')
    pos = (q / 100) * (len(vals) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(vals) - 1)
    return vals[lo] + (pos - lo) * (vals[hi] - vals[lo])

def _np_nansum(a, axis=None):
    flat = a.to_flat_list() if isinstance(a, NDArray) else list(a)
    return sum(float(v) for v in flat if not _math.isnan(float(v)))

def _np_percentile(a, q, axis=None):
    return _np_nanpercentile(a, q, axis=axis)

def _np_corrcoef(a, b):
    fa = a.to_flat_list() if isinstance(a, NDArray) else list(a)
    fb = b.to_flat_list() if isinstance(b, NDArray) else list(b)
    n = len(fa)
    if n == 0: return _nd_array([[1.0, 0.0],[0.0, 1.0]], dtype='float32')
    ma, mb = sum(fa)/n, sum(fb)/n
    cov = sum((fa[i]-ma)*(fb[i]-mb) for i in range(n)) / n
    sa = _math.sqrt(sum((v-ma)**2 for v in fa) / n) or 1.0
    sb = _math.sqrt(sum((v-mb)**2 for v in fb) / n) or 1.0
    r = cov / (sa * sb)
    return _nd_array([[1.0, r],[r, 1.0]], dtype='float32')

def _np_interp(xi, x, y):
    xf = xi.to_flat_list() if isinstance(xi, NDArray) else list(xi)
    xvals = x.to_flat_list() if isinstance(x, NDArray) else list(x)
    yvals = y.to_flat_list() if isinstance(y, NDArray) else list(y)
    result = []
    for xv in xf:
        if xv <= xvals[0]: result.append(yvals[0]); continue
        if xv >= xvals[-1]: result.append(yvals[-1]); continue
        for i in range(len(xvals)-1):
            if xvals[i] <= xv <= xvals[i+1]:
                t = (xv - xvals[i]) / (xvals[i+1] - xvals[i])
                result.append(yvals[i] + t * (yvals[i+1] - yvals[i]))
                break
    return _nd_array(result, dtype='float32')

# ── __all__ ───────────────────────────────────────────────────────────────────

__all__ = [name for name in dir() if not name.startswith('__')]
