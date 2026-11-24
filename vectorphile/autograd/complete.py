"""Complete reverse-mode autograd coverage (v0.5.0).

Before 0.5.0 only add/sub/mul/div/2-D matmul/mean/relu/sigmoid/tanh and 2-D Linear
propagated gradients, so cross-entropy classifiers and transformers could not train.
This module provides differentiable versions of the missing tensor operations,
functional ops and layers, and installs them over the non-differentiable ones.
Every op here is verified against central finite differences in tests/test_autograd.py.
"""
from __future__ import annotations
from collections import OrderedDict
from vectorphile._core.ndarray import NDArray, array as _nd_array, zeros as _nd_zeros, ones as _nd_ones, full as _nd_full, empty as _nd_empty, eye as _nd_eye, arange as _nd_arange, linspace as _nd_linspace
import math as _math
from vectorphile.core.tensor import Tensor
from vectorphile.autograd.engine import Function
import vectorphile.core.tensor as _ct
from vectorphile._vp_helpers import *



class _Fn(Function):
    def __init__(self, inputs, fn):
        super().__init__(*inputs)
        self._fn = fn

    def backward(self, grad):
        return self._fn(grad._data if isinstance(grad, Tensor) else _nd_array(grad))


def _out(data, parents, fn, device=None):
    from vectorphile.autograd.engine import no_grad
    req = no_grad._enabled and any(isinstance(p, Tensor) and p.requires_grad for p in parents)
    dev = device or next((p.device for p in parents if isinstance(p, Tensor)), 'cpu')
    # Skip _nd_array copy when data is already an NDArray or GArray — avoids 1.4ms/op
    nd = data if (_is_gpu(data) or isinstance(data, NDArray)) else _nd_array(data)
    t = Tensor(nd, device=dev, requires_grad=req)
    if req:
        t.grad_fn = _Fn(parents, fn)
        t._is_leaf = False
    return t


def _is_gpu(a):
    return type(a).__name__ == 'GArray'


def _d(x):
    if isinstance(x, Tensor):
        return x._data
    return x if _is_gpu(x) else _nd_array(x)


def _shape(args):
    return tuple(args[0]) if len(args) == 1 and isinstance(args[0], (tuple, list)) else tuple(args)


# ── Tensor methods ───────────────────────────────────────────────────────────

def t_pow(self, e):
    if isinstance(e, Tensor):
        y = self._data ** e._data
        return _out(y, (self, e), lambda g: (g * e._data * self._data ** (e._data - 1),
                                              g * y * _math_log(_nd_maximum_fn(self._data, 1e-30))))
    return _out(self._data ** e, (self,), lambda g: (g * e * self._data ** (e - 1),))


def t_neg(self):
    return _out(-self._data, (self,), lambda g: (-g,))


def t_exp(self):
    y = _math_exp(self._data)
    return _out(y, (self,), lambda g: (g * y,))


def t_log(self):
    return _out(_math_log(self._data), (self,), lambda g: (g / self._data,))


def t_sqrt(self):
    y = _math_sqrt(self._data)
    return _out(y, (self,), lambda g: (g / (2 * y),))


def t_abs(self):
    return _out(_nd_abs_fn(self._data), (self,), lambda g: (g * _nd_sign_fn(self._data),))


def _expand_reduced(g, shape, dim, keepdim):
    if dim is None:
        return _nd_broadcast_to_fn(g, shape)
    if not keepdim:
        axes = (dim,) if isinstance(dim, int) else tuple(dim)
        for ax in sorted(axes):
            g = _nd_expand_dims_fn(g, ax)
    return _nd_broadcast_to_fn(g, shape)


def t_sum(self, dim=None, keepdim=False):
    y = self._data.sum(axis=dim, keepdims=keepdim)
    return _out(y, (self,), lambda g: (_expand_reduced(g, self._data.shape, dim, keepdim).copy(),))


def t_mean(self, dim=None, keepdim=False):
    y = self._data.mean(axis=dim, keepdims=keepdim)
    n = self._data.size / max(y.size, 1)
    return _out(y, (self,), lambda g: (_expand_reduced(g, self._data.shape, dim, keepdim) / n,))


def t_reshape(self, *shape):
    s = self._data.shape
    return _out(self._data.reshape(_shape(shape)), (self,), lambda g: (g.reshape(s),))


def t_flatten(self, start_dim=0, end_dim=-1):
    s = self._data.shape
    nd = len(s)
    e = end_dim % nd if nd else 0
    new = s[:start_dim] + (int(_nd_prod_fn(s[start_dim:e + 1])),) + s[e + 1:]
    return t_reshape(self, new)


def t_transpose(self, dim0=None, dim1=None):
    if dim0 is None:
        axes = tuple(reversed(range(self._data.ndim)))
    else:
        axes = list(range(self._data.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    return t_permute(self, *axes)


def t_permute(self, *dims):
    axes = _shape(dims)
    inv = _nd_argsort_fn(axes)
    return _out(_nd_transpose_fn(self._data, axes), (self,), lambda g: (_nd_transpose_fn(g, inv),))


def t_squeeze(self, dim=None):
    s = self._data.shape
    return _out(_nd_squeeze_fn(self._data, axis=dim), (self,), lambda g: (g.reshape(s),))


def t_unsqueeze(self, dim):
    s = self._data.shape
    return _out(_nd_expand_dims_fn(self._data, dim), (self,), lambda g: (g.reshape(s),))


def t_contiguous(self):
    return self


def t_expand(self, *shape):
    target = _shape(shape)
    target = tuple(self._data.shape[i - (len(target) - self._data.ndim)] if d == -1 else d for i, d in enumerate(target))
    return _out(_nd_broadcast_to_fn(self._data, target), (self,), lambda g: (g,))


def _scatter_add(z, idx, g):
    """z[idx] += g  — pure Python implementation."""
    target = z[idx]
    if hasattr(target, '_buf'):
        for _i in target._iter_indices():
            pos = target._flat_idx(_i)
            gv = g._buf[g._flat_idx(_i)] if hasattr(g, '_buf') else float(g)
            target._buf[pos] += gv
    else:
        z[idx] = (target + g)


def t_getitem(self, idx):
    if isinstance(idx, Tensor):
        idx = idx._data
    elif isinstance(idx, tuple):
        idx = tuple(i._data if isinstance(i, Tensor) else i for i in idx)
    s, dt = self._data.shape, self._data.dtype

    def back(g):
        z = _nd_zeros(s, dtype=dt)
        _scatter_add(z, idx, g)
        return (z,)
    return _out(self._data[idx], (self,), back)


def t_split(self, split_size, dim=0):
    n = self._data.shape[dim]
    sizes = [split_size] * (n // split_size) + ([n % split_size] if n % split_size else []) \
        if isinstance(split_size, int) else list(split_size)
    out, start = [], 0
    for sz in sizes:
        sl = [slice(None)] * self._data.ndim
        sl[dim] = slice(start, start + sz)
        out.append(t_getitem(self, tuple(sl)))
        start += sz
    return out


def t_chunk(self, chunks, dim=0):
    return t_split(self, -(-self._data.shape[dim] // chunks), dim)


def t_matmul(self, other):
    a, b = self._data, _d(other)
    if a.ndim < 2 or b.ndim < 2:
        a2 = a if a.ndim >= 2 else a[None, :]
        b2 = b if b.ndim >= 2 else b[:, None]
        y = _nd_matmul_fn(a, b)

        def back1(g):
            g2 = _nd_array(g).reshape(_nd_matmul_fn(a2, b2).shape)
            return (_nd_matmul_fn(g2, _nd_swapaxes_fn(b2, -1, -2)).reshape(a.shape) if a.ndim < 2 else _nd_matmul_fn(g2, _nd_swapaxes_fn(b2, -1, -2)),
                    _nd_matmul_fn(_nd_swapaxes_fn(a2, -1, -2), g2).reshape(b.shape) if b.ndim < 2 else _nd_matmul_fn(_nd_swapaxes_fn(a2, -1, -2), g2))
        return _out(y, (self, other), back1)
    return _out(_nd_matmul_fn(a, b), (self, other),
                lambda g: (_nd_matmul_fn(g, _nd_swapaxes_fn(b, -1, -2)), _nd_matmul_fn(_nd_swapaxes_fn(a, -1, -2), g)))


def t_masked_fill(self, mask, value):
    m = _d(mask).astype(bool)
    return _out(_nd_where_fn(m, _nd_array(value, dtype=self._data.dtype), self._data), (self,),
                lambda g: (_nd_where_fn(m, 0, g),))


def t_clamp(self, min=None, max=None):
    y = _nd_clip_fn(self._data, min, max)
    return _out(y, (self,), lambda g: (g * (y == self._data),))


def t_to(self, device=None, dtype=None, **_):
    if dtype is not None:
        return Tensor(self._data, dtype=dtype, device=device or self.device, requires_grad=self.requires_grad)
    if device is not None:
        self.device = str(device)
    return self


# ── free functions ───────────────────────────────────────────────────────────

def cat(tensors, dim=0):
    ts = list(tensors)
    sizes = [t._data.shape[dim] for t in ts]
    splits = _nd_cumsum_fn(sizes)[:-1]
    return _out(_nd_concatenate_fn([t._data for t in ts], axis=dim), tuple(ts),
                lambda g: tuple(_nd_split_fn(g, splits, axis=dim)))


def stack(tensors, dim=0):
    ts = list(tensors)
    n = len(ts)
    def _stack_back(g):
        # g has an extra dim at position `dim`; slice out each index and squeeze
        grads = []
        for i in range(n):
            idx = tuple(slice(None) if ax != dim else i for ax in range(g.ndim))
            grads.append(g[idx])
        return tuple(grads)
    return _out(_nd_stack_fn([t._data for t in ts], axis=dim), tuple(ts), _stack_back)


def softmax(x, dim=-1):
    from vectorphile._core.ndarray import _np_softmax as _fused_sm
    _ax = int(dim) if dim >= 0 else len(x._data.shape) + int(dim)
    s = _fused_sm(x._data, _ax)
    if s is None:
        z = x._data - x._data.max(axis=dim, keepdims=True)
        e = _math_exp(z)
        s = e / e.sum(axis=dim, keepdims=True)
    return _out(s, (x,), lambda g: (s * (g - (g * s).sum(axis=dim, keepdims=True)),))


def log_softmax(x, dim=-1):
    from vectorphile._core.ndarray import _np_log_softmax as _fused_lsm
    _ax = int(dim) if dim >= 0 else len(x._data.shape) + int(dim)
    ls = _fused_lsm(x._data, _ax)
    if ls is None:
        z = x._data - x._data.max(axis=dim, keepdims=True)
        ls = z - _math_log(_math_exp(z).sum(axis=dim, keepdims=True))
    return _out(ls, (x,), lambda g: (g - _math_exp(ls) * g.sum(axis=dim, keepdims=True),))


def where(cond, a, b):
    c = _d(cond).astype(bool)
    parents = tuple(p for p in (a, b) if isinstance(p, Tensor))

    def back(g):
        out = []
        if isinstance(a, Tensor):
            out.append(_nd_where_fn(c, g, 0))
        if isinstance(b, Tensor):
            out.append(_nd_where_fn(c, 0, g))
        return tuple(out)
    return _out(_nd_where_fn(c, _d(a), _d(b)), parents, back)


# ── functional ───────────────────────────────────────────────────────────────

def gelu(x, approximate='none'):
    if type(x._data).__name__ == 'GArray':
        import vectorphile.backend as _be
        import torch as _torch
        import math as _m
        d = x._data
        out = _be.gelu(d)
        # d is captured in closure; compute GELU gradient on GPU
        def _gelu_gpu_bwd(g):
            t = d.t  # raw torch tensor
            cdf = 0.5 * (1.0 + _torch.erf(t / _m.sqrt(2)))
            pdf = _torch.exp(-0.5 * t * t) / _m.sqrt(2 * _m.pi)
            from vectorphile.gpu import GArray as _GA
            return (g * _GA(cdf + t * pdf),)
        return _out(out, (x,), _gelu_gpu_bwd)
    v = _nd_array(x._data)  # copy to prevent mutation issues in closures
    if approximate == 'tanh':
        c = _math_sqrt(2 / _math.pi)
        u = c * (v + 0.044715 * v ** 3)
        t = _math_tanh(u)
        _t = _nd_array(t); _v = v
        return _out(0.5 * _v * (1 + _t), (x,),
                    lambda g: (g * (0.5 * (1 + _t) + 0.5 * _v * (1 - _t * _t) * c * (1 + 3 * 0.044715 * _v * _v)),))
    u = v / _math_sqrt(2)
    # use native erf (vectorphile's pure-Python polynomial approximation)
    cdf = 0.5 * (1 + (u.erf() if hasattr(u, 'erf') else _nd_array(u).erf()))
    pdf = _math_exp(-0.5 * v * v) / _math_sqrt(2 * _math.pi)
    _cdf = _nd_array(cdf); _pdf = _nd_array(pdf); _v = v
    return _out(_v * _cdf, (x,), lambda g: (g * (_cdf + _v * _pdf),))


def silu(x, inplace=False):
    if type(x._data).__name__ == 'GArray':
        import vectorphile.backend as _be
        import torch as _torch
        d = x._data
        out = _be.silu(d)
        def _silu_gpu_bwd(g):
            t = d.t
            sig = _torch.sigmoid(t)
            from vectorphile.gpu import GArray as _GA
            return (g * _GA(sig * (1 + t * (1 - sig))),)
        return _out(out, (x,), _silu_gpu_bwd)
    s = 1 / (1 + _math_exp(-x._data))
    return _out(x._data * s, (x,), lambda g: (g * (s + x._data * s * (1 - s)),))


def dropout(x, p=0.5, training=True, inplace=False):
    if not training or p == 0:
        return x
    if _is_gpu(x._data):
        import torch as _t_do
        from vectorphile.gpu import GArray as _GA_do
        m_t = (_t_do.rand_like(x._data.t) > p).float() / (1.0 - p)
        m = _GA_do(m_t)
        return _out(x._data * m, (x,), lambda g: (g * m,))
    from vectorphile._core.ndarray import NDArray as _NDArray
    _r = _nd_rand_fn(*x._data.shape)
    _r_data = _r._data if hasattr(_r, '_data') else _r
    m = (_r_data > p).astype(x._data.dtype) / (1.0 - p)
    return _out(x._data * m, (x,), lambda g: (g * m,))


def embedding(x, weight, padding_idx=None, *_, **__):
    idx = _d(x).astype('int64')

    def back(g):
        # Bring gradient to CPU NDArray if on GPU
        if type(g).__name__ == 'GArray':
            g_flat = g.t.cpu().float().tolist() if hasattr(g.t, 'tolist') else list(g.t.cpu().float())
            g = _nd_array(g_flat, dtype='float32').reshape(g.t.shape)
        # Bring idx to CPU NDArray if on GPU
        _idx = idx
        if type(_idx).__name__ == 'GArray':
            _idx = _nd_array(list(_idx.t.cpu().long()), dtype='int64').reshape(_idx.t.shape)
        z = _nd_zeros(weight._data.shape, dtype=weight._data.dtype)
        if padding_idx is not None:
            g = _nd_where_fn((_idx == padding_idx)[..., None], 0, g)
        _scatter_add(z, _idx, g)
        return (z,)
    out = weight._data[idx]
    if padding_idx is not None:
        out = _nd_where_fn((idx == padding_idx)[..., None], 0, out)
    return _out(out, (weight,), back)


def linear(x, weight, bias=None):
    y = t_matmul(x, t_transpose(weight, 0, 1))
    return y + bias if bias is not None else y


def _layer_norm_fused(x, axes, weight, bias, eps):
    """Vectorized forward plus the closed-form backward: one graph node instead of nine."""
    from vectorphile._core.ndarray import NDArray, _npx, _np_view, _np_wrap
    if _npx is None or not isinstance(x, Tensor):
        return None
    for p in (x, weight, bias):
        if p is not None and not (isinstance(p, Tensor) and isinstance(p._data, NDArray)
                                  and p._data.dtype == 'float32'):
            return None
    try:
        xv = _np_view(x._data)
        xc = xv - xv.mean(axis=axes, keepdims=True)
        inv = 1.0 / _npx.sqrt((xc * xc).mean(axis=axes, keepdims=True) + eps)
        xhat = xc * inv
        wv = _np_view(weight._data).copy() if weight is not None else None
        y = xhat * wv if wv is not None else xhat
        if bias is not None:
            y = y + _np_view(bias._data)
        lead = tuple(range(xv.ndim - len(axes)))
        if wv is not None and wv.shape != xv.shape[len(lead):]:
            return None
        if bias is not None and bias._data.shape != xv.shape[len(lead):]:
            return None
        out = _np_wrap(y, 'float32')
    except Exception:
        return None

    def back(g):
        gv = _npx.broadcast_to(_np_view(g), xhat.shape)
        gh = gv * wv if wv is not None else gv
        dx = (gh - gh.mean(axis=axes, keepdims=True)
              - xhat * (gh * xhat).mean(axis=axes, keepdims=True)) * inv
        grads = [_np_wrap(dx, 'float32')]
        if weight is not None:
            grads.append(_np_wrap((gv * xhat).sum(axis=lead), 'float32'))
        if bias is not None:
            grads.append(_np_wrap(gv.sum(axis=lead), 'float32'))
        return tuple(grads)

    return _out(out, tuple(p for p in (x, weight, bias) if p is not None), back)


def layer_norm(x, normalized_shape=None, weight=None, bias=None, eps=1e-5):
    nd = 1 if normalized_shape is None or isinstance(normalized_shape, int) else len(tuple(normalized_shape))
    axes = tuple(range(-nd, 0))
    fused = _layer_norm_fused(x, axes, weight, bias, eps)
    if fused is not None:
        return fused
    mu = t_mean(x, dim=axes, keepdim=True)
    xc = x - mu
    var = t_mean(xc * xc, dim=axes, keepdim=True)
    y = xc / t_sqrt(var + eps)
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


def cross_entropy(x, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
    """Logits [..., C] (class axis last, or [N, C]) and integer targets [...]. Differentiable."""
    # GPU fast path: use torch for numerically stable forward/backward
    if _is_gpu(x._data):
        import torch as _torch_ce
        from vectorphile.gpu import GArray as _GA_ce
        orig_shape = x._data.t.shape
        C_ce = orig_shape[-1]
        logits_t = x._data.t.reshape(-1, C_ce).float()
        tgt_raw = _d(target)
        tgt_t = (tgt_raw.t.reshape(-1).long() if _is_gpu(tgt_raw)
                 else _torch_ce.tensor(list(tgt_raw.reshape(-1).to_flat_list()),
                                       dtype=_torch_ce.long, device=logits_t.device))
        n_ce = logits_t.shape[0]
        ls = _torch_ce.nn.functional.log_softmax(logits_t, dim=-1)
        probs = ls.exp()
        ignore_mask = (tgt_t != ignore_index)
        safe_tgt = tgt_t.clone(); safe_tgt[~ignore_mask] = 0
        q_t = _torch_ce.zeros_like(logits_t)
        q_t[_torch_ce.arange(n_ce, device=logits_t.device), safe_tgt] = 1.0
        if label_smoothing > 0:
            q_t = q_t * (1 - label_smoothing) + label_smoothing / C_ce
        w_t = ignore_mask.float()
        per_t = -(q_t * ls).sum(dim=-1) * w_t
        denom_ce = max(float(w_t.sum()), 1e-12) if reduction == 'mean' else 1.0
        if reduction == 'none':
            tgt_shape = list(tgt_raw.t.shape) if _is_gpu(tgt_raw) else list(tgt_raw.shape)
            val_t = _GA_ce(per_t.reshape(tgt_shape))
        else:
            val_t = _GA_ce((per_t.sum() / denom_ce).unsqueeze(0))

        def _ce_gpu_back(g):
            gl_t = (probs * q_t.sum(dim=-1, keepdim=True) - q_t) * w_t[:, None]
            g_scalar = (float(g.t.reshape(-1)[0]) if _is_gpu(g)
                        else float(list(g.reshape(-1).to_flat_list())[0]))
            gl_t = gl_t * g_scalar / denom_ce
            return (_GA_ce(gl_t.reshape(orig_shape).to(x._data.t.dtype)),)
        return _out(val_t, (x,), _ce_gpu_back)
    C = x._data.shape[-1]
    logits = x._data.reshape(-1, C)
    tgt = _d(target).astype('int64').reshape(-1)
    valid = tgt != ignore_index
    safe = _nd_where_fn(valid, tgt, 0)
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - _math_log(_math_exp(z).sum(axis=1, keepdims=True))
    n = logits.shape[0]
    q = _nd_zeros_like(logp)
    q[_nd_arange(n), safe] = 1.0
    if label_smoothing > 0:
        q = q * (1 - label_smoothing) + label_smoothing / C
    w = _nd_ones(n, dtype=logp.dtype) if weight is None else _d(weight)[safe].astype(logp.dtype)
    w = w * valid
    per = -(q * logp).sum(axis=1) * w
    denom = max(float(w.sum()), 1e-12) if reduction == 'mean' else 1.0
    p = _math_exp(logp)

    def back(g):
        gl = (p * q.sum(axis=1, keepdims=True) - q) * w[:, None]
        if reduction == 'none':
            gl = gl * _nd_array(g).reshape(-1, 1)
        else:
            gl = gl * float(_nd_array(g).reshape(-1)[0]) / denom
        return (gl.reshape(x._data.shape).astype(x._data.dtype),)
    if reduction == 'none':
        val = per.reshape(_d(target).shape)
    else:
        val = _nd_array(per.sum() / denom, dtype=logp.dtype)
    return _out(val, (x,), back)


def nll_loss(x, target, weight=None, ignore_index=-100, reduction='mean'):
    # GPU fast path
    if _is_gpu(x._data):
        import torch as _torch_nll
        from vectorphile.gpu import GArray as _GA_nll
        C_n = x._data.shape[-1]
        lp_t = x._data.t.reshape(-1, C_n).float()
        tgt_raw = _d(target)
        tgt_t = (tgt_raw.t.reshape(-1).long() if _is_gpu(tgt_raw)
                 else _torch_nll.tensor(list(tgt_raw.reshape(-1).to_flat_list()),
                                        dtype=_torch_nll.long, device=lp_t.device))
        n_n = lp_t.shape[0]
        ignore_mask = (tgt_t != ignore_index)
        safe_tgt = tgt_t.clone(); safe_tgt[~ignore_mask] = 0
        w_t = ignore_mask.float()
        denom_n = max(float(w_t.sum()), 1e-12) if reduction == 'mean' else 1.0
        per_t = -lp_t[_torch_nll.arange(n_n, device=lp_t.device), safe_tgt] * w_t
        val_t = _GA_nll((per_t.sum() / denom_n).unsqueeze(0))
        def _nll_gpu_back(g):
            g_s = (float(g.t.reshape(-1)[0]) if _is_gpu(g)
                   else float(list(g.reshape(-1).to_flat_list())[0]))
            z_t = _torch_nll.zeros_like(lp_t)
            z_t[_torch_nll.arange(n_n, device=lp_t.device), safe_tgt] = -w_t / denom_n * g_s
            return (_GA_nll(z_t.reshape(x._data.t.shape).to(x._data.t.dtype)),)
        return _out(val_t, (x,), _nll_gpu_back)
    C = x._data.shape[-1]
    lp = x._data.reshape(-1, C)
    tgt = _d(target).astype('int64').reshape(-1)
    valid = tgt != ignore_index
    safe = _nd_where_fn(valid, tgt, 0)
    n = lp.shape[0]
    w = (_nd_ones(n) if weight is None else _d(weight)[safe]) * valid
    denom = max(float(w.sum()), 1e-12) if reduction == 'mean' else 1.0

    def back(g):
        z = _nd_zeros_like(lp)
        z[_nd_arange(n), safe] = -w / denom * float(_nd_array(g).reshape(-1)[0])
        return (z.reshape(x._data.shape),)
    return _out(_nd_array((-lp[_nd_arange(n), safe] * w).sum() / denom, dtype=lp.dtype), (x,), back)


def mse_loss(x, target, reduction='mean'):
    d = x - target
    sq = d * d
    return t_mean(sq) if reduction == 'mean' else (t_sum(sq) if reduction == 'sum' else sq)


def l1_loss(x, target, reduction='mean'):
    a = t_abs(x - target)
    return t_mean(a) if reduction == 'mean' else (t_sum(a) if reduction == 'sum' else a)


# ── installation ─────────────────────────────────────────────────────────────

def install(vp_module=None):
    T = Tensor
    for name, fn in {
        'pow': t_pow, '__pow__': t_pow, '__neg__': t_neg, 'exp': t_exp, 'log': t_log, 'sqrt': t_sqrt,
        'abs': t_abs, '__abs__': t_abs, 'sum': t_sum, 'mean': t_mean, 'reshape': t_reshape, 'view': t_reshape,
        'flatten': t_flatten, 'transpose': t_transpose, 'permute': t_permute, 'squeeze': t_squeeze,
        'unsqueeze': t_unsqueeze, 'contiguous': t_contiguous, 'expand': t_expand, '__getitem__': t_getitem,
        'split': t_split, 'chunk': t_chunk, 'matmul': t_matmul, '__matmul__': t_matmul, 'mm': t_matmul,
        'bmm': t_matmul, 'masked_fill': t_masked_fill, 'clamp': t_clamp, 'clip': t_clamp,
    }.items():
        setattr(T, name, fn)
    # Built-in arithmetic and activations also honour vp.no_grad()
    from vectorphile.autograd.engine import no_grad as _ng

    def _respect_no_grad(orig):
        def op(self, *a, **k):
            out = orig(self, *a, **k)
            if not _ng._enabled and isinstance(out, Tensor):
                out.grad_fn, out.requires_grad, out._is_leaf = None, False, True
            return out
        return op
    for _name in ('__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__', '__truediv__',
                  '__rtruediv__', 'add', 'relu', 'sigmoid', 'tanh'):
        if hasattr(T, _name) and not getattr(getattr(T, _name), '_ng', False):
            _w = _respect_no_grad(getattr(T, _name)); _w._ng = True
            setattr(T, _name, _w)
    T.softmax = lambda self, dim=-1: softmax(self, dim)
    T.log_softmax = lambda self, dim=-1: log_softmax(self, dim)
    T.square = lambda self: t_pow(self, 2)
    T.numel = lambda self: int(self._data.size)

    free = {
        'cat': cat, 'stack': stack, 'softmax': softmax, 'log_softmax': log_softmax, 'where': where,
        'exp': t_exp, 'log': t_log, 'sqrt': t_sqrt, 'matmul': t_matmul, 'bmm': t_matmul,
        'transpose': t_transpose, 'permute': lambda x, *d: t_permute(x, *d), 'squeeze': t_squeeze,
        'unsqueeze': t_unsqueeze, 'flatten': t_flatten, 'clamp': t_clamp,
        'sum': lambda x, dim=None, keepdim=False: t_sum(x, dim, keepdim),
        'mean': lambda x, dim=None, keepdim=False: t_mean(x, dim, keepdim),
        'split': lambda x, s, dim=0: t_split(x, s, dim), 'chunk': lambda x, c, dim=0: t_chunk(x, c, dim),
        'square': lambda x: t_pow(x, 2),
    }
    for name, fn in free.items():
        setattr(_ct, name, fn)
        if vp_module is not None:
            setattr(vp_module, name, fn)

    import vectorphile.nn.functional as F
    # Only overwrite ops that don't already have optimized C-backed implementations.
    # cross_entropy and linear have optimized versions with fused C kernels — don't replace.
    for name, fn in {'softmax': softmax, 'log_softmax': log_softmax, 'gelu': gelu, 'silu': silu,
                     'dropout': dropout, 'embedding': embedding, 'layer_norm': layer_norm,
                     'nll_loss': nll_loss, 'mse_loss': mse_loss,
                     'l1_loss': l1_loss}.items():
        setattr(F, name, fn)

    import vectorphile.nn as nn
    from vectorphile.nn import layers, losses, activations, container
    from vectorphile.nn.module import Module

    # layers.Linear.forward already uses _LinearBackward (C-backed matmul) — don't overwrite
    layers.Embedding.forward = lambda self, idx: embedding(idx, self.weight, self.padding_idx)
    layers.Dropout.forward = lambda self, x: dropout(x, self.p, self.training)
    layers.LayerNorm.forward = lambda self, x: layer_norm(
        x, self.normalized_shape,
        self.weight if self.elementwise_affine else None,
        self.bias_param if self.elementwise_affine else None, self.eps)
    losses.CrossEntropyLoss.forward = lambda self, logits, targets: cross_entropy(
        logits, targets, ignore_index=self.ignore_index, reduction=self.reduction,
        label_smoothing=self.label_smoothing)
    if hasattr(losses, 'NLLLoss'):
        losses.NLLLoss.forward = lambda self, x, t: nll_loss(x, t)
    if hasattr(losses, 'MAELoss'):
        losses.MAELoss.forward = lambda self, pred, target: l1_loss(pred, target, self.reduction)
    if hasattr(losses, 'HuberLoss'):
        def _huber_fwd(self, pred, target):
            d = pred - target
            da = t_abs(d)
            sq = t_pow(da, 2) * 0.5
            lin = da * self.delta - 0.5 * self.delta ** 2
            elem = where(da._data < self.delta, sq, lin)
            return t_mean(elem) if self.reduction == 'mean' else (t_sum(elem) if self.reduction == 'sum' else elem)
        losses.HuberLoss.forward = _huber_fwd
    if hasattr(losses, 'BCEWithLogitsLoss'):
        def _bce_logits_fwd(self, logits, target):
            relu_logits = _out(_nd_maximum_fn(logits._data, 0), (logits,),
                               lambda g: (_nd_where_fn(logits._data >= 0, g, 0),))
            neg_abs = _out(-_nd_abs_fn(logits._data), (logits,),
                           lambda g: (_nd_where_fn(logits._data >= 0, -g, g),))
            _neg_abs_v = -_nd_abs_fn(logits._data)  # -|x|
            _exp_neg_abs = _math_exp(_neg_abs_v)
            _log1p_val = _nd_elementwise(_exp_neg_abs, lambda v: _math.log1p(v))
            log1p_exp = _out(_log1p_val, (logits,),
                             lambda g: ((-_exp_neg_abs / (1 + _exp_neg_abs)) * _nd_sign_fn(logits._data) * g,))
            loss = relu_logits - logits * target + log1p_exp
            return t_mean(loss) if self.reduction == 'mean' else (t_sum(loss) if self.reduction == 'sum' else loss)
        losses.BCEWithLogitsLoss.forward = _bce_logits_fwd
    if hasattr(losses, 'KLDivLoss'):
        def _kldiv_fwd(self, inp, target):
            if self.log_target:
                t_exp = _out(_math_exp(target._data), (target,), lambda g: (_math_exp(target._data) * g,))
                loss = t_exp * (target - inp)
            else:
                td = target._data
                log_t = _out(_math_log(td + 1e-12), (target,), lambda g: (g / (td + 1e-12),))
                loss = target * (log_t - inp)
            return t_mean(loss) if self.reduction == 'mean' else (t_sum(loss) if self.reduction == 'sum' else loss)
        losses.KLDivLoss.forward = _kldiv_fwd
    if hasattr(losses, 'CosineSimilarityLoss'):
        def _cosine_fwd(self, x1, x2):
            dot = t_sum(x1 * x2, dim=self.dim)
            n1 = t_sqrt(t_sum(x1 * x1, dim=self.dim) + self.eps)
            n2 = t_sqrt(t_sum(x2 * x2, dim=self.dim) + self.eps)
            cos_sim = dot / (n1 * n2)
            return t_mean(Tensor(_nd_ones_like(cos_sim._data)) - cos_sim)
        losses.CosineSimilarityLoss.forward = _cosine_fwd
    for cls_name, fn in (('ReLU', lambda self, x: x.relu()),
                         ('LeakyReLU', lambda self, x: _out(
                             _nd_where_fn(x._data >= 0, x._data, getattr(self, 'negative_slope', 0.01) * x._data),
                             (x,), lambda g: (_nd_where_fn(x._data >= 0, g, getattr(self, 'negative_slope', 0.01) * g),))),
                         ('GELU', lambda self, x: gelu(x)), ('SiLU', lambda self, x: silu(x)),
                         ('Softmax', lambda self, x: softmax(x, getattr(self, 'dim', -1))),
                         ('LogSoftmax', lambda self, x: log_softmax(x, getattr(self, 'dim', -1)))):
        if hasattr(activations, cls_name):
            getattr(activations, cls_name).forward = fn

    # ModuleList: register children so nested parameters are visible to optimizers and state_dict
    ML = container.ModuleList

    def _sync(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict((str(i), m) for i, m in enumerate(self._modules_list))

    def _append(self, module):
        self._modules_list.append(module); _sync(self); return self

    def _insert(self, index, module):
        self._modules_list.insert(index, module); _sync(self)

    def _setitem(self, idx, module):
        self._modules_list[idx] = module; _sync(self)

    ML.append, ML.insert, ML.__setitem__ = _append, _insert, _setitem

    # Module: device moves are no-ops on the NumPy backend; buffers are reported
    Module.float = lambda self: self
    Module.requires_grad_ = lambda self, flag=True: ([setattr(p, 'requires_grad', flag) for p in self.parameters()], self)[1]

    def _load_state_dict(self, state_dict, strict=True):
        own = dict(self.named_parameters())
        missing = [k for k in own if k not in state_dict]
        unexpected = [k for k in state_dict if k not in own]
        if strict and (missing or unexpected):
            raise KeyError(f'load_state_dict: missing keys {missing[:5]}, unexpected keys {unexpected[:5]}')
        for k, v in state_dict.items():
            if k in own:
                v = _d(v)
                if own[k]._data.shape != v.shape:
                    raise ValueError(f'load_state_dict: shape mismatch for "{k}": {own[k]._data.shape} vs {v.shape}')
                own[k]._data[...] = v
        return missing, unexpected
    Module.load_state_dict = _load_state_dict

    # A parameter reachable by two routes (own dict and a child) is yielded once, so it is updated once
    def _named_parameters(self, prefix='', _seen=None):
        seen = _seen if _seen is not None else set()
        for name, p in self._parameters.items():
            if id(p) not in seen:
                seen.add(id(p))
                yield (f'{prefix}.{name}' if prefix else name), p
        for mname, module in self._modules.items():
            yield from _named_parameters(module, f'{prefix}.{mname}' if prefix else mname, seen)

    Module.named_parameters = lambda self, prefix='': _named_parameters(self, prefix)
    Module.parameters = lambda self: (p for _, p in _named_parameters(self))

    from vectorphile.autograd import layers_complete
    layers_complete.install()
    from vectorphile.autograd import layers_extra
    layers_extra.install()
    from vectorphile import gpu
    gpu.install(vp_module)
    if vp_module is not None:
        vp_module.gpu = gpu
