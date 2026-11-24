"""Differentiable attention, recurrent, convolution, normalisation and pooling layers (v0.5.0).

Installed by `vectorphile.autograd.complete.install`. Forward passes are rebuilt from
differentiable tensor operations (or explicit backward closures for conv / pooling),
keeping each layer's existing parameters and attribute names.
"""
from __future__ import annotations
import math
from collections import OrderedDict
from vectorphile._core.ndarray import NDArray, array as _nd_array, zeros as _nd_zeros, ones as _nd_ones, full as _nd_full, empty as _nd_empty, eye as _nd_eye, arange as _nd_arange, linspace as _nd_linspace
import math as _math

def sliding_window_view(x, window_shape, axis=None, **kw):
    n = len(x) if hasattr(x, '__len__') else 0
    w = window_shape if isinstance(window_shape, int) else window_shape[0]
    return [x[i:i+w] for i in range(max(0, n-w+1))]

from vectorphile.core.tensor import Tensor
from vectorphile._vp_helpers import *

from vectorphile.autograd import complete as C


def _pair(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


# ── convolution ──────────────────────────────────────────────────────────────

def _im2col(xp, kH, kW, d0, d1, s0, s1, H_out, W_out):
    """Build (N, Cin, kH, kW, H_out, W_out) patch tensor."""
    N, Cin = xp.shape[0], xp.shape[1]
    patches = _nd_empty((N, Cin, kH, kW, H_out, W_out), dtype=xp.dtype)
    for i in range(kH):
        for j in range(kW):
            src = xp[:, :, i*d0:i*d0+H_out*s0:s0, j*d1:j*d1+W_out*s1:s1]
            patches[:, :, i, j, :, :] = src
    return patches


def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if groups > 1:
        cin = x._data.shape[1] // groups
        cout = weight._data.shape[0] // groups
        parts = [conv2d(x[:, g*cin:(g+1)*cin], weight[g*cout:(g+1)*cout], None, stride, padding, dilation)
                 for g in range(groups)]
        y = C.cat(parts, dim=1)
    else:
        (s0, s1), (p0, p1), (d0, d1) = _pair(stride), _pair(padding), _pair(dilation)
        xd, wd = x._data, weight._data
        N, Cin, H, W = xd.shape
        Cout, _, kH, kW = wd.shape
        H_out = (H + 2*p0 - d0*(kH-1) - 1) // s0 + 1
        W_out = (W + 2*p1 - d1*(kW-1) - 1) // s1 + 1

        xp = _nd_pad_fn(xd, ((0,0),(0,0),(p0,p0),(p1,p1)))
        patches = _im2col(xp, kH, kW, d0, d1, s0, s1, H_out, W_out)

        out = _nd_einsum_fn('ncijhw,ocij->nohw', patches, wd)
        patches_save, wd_save = patches, wd

        def back(g):
            gw = _nd_einsum_fn('nohw,ncijhw->ocij', g, patches_save)
            gxp = _nd_zeros_like(xp)
            d_patches = _nd_einsum_fn('nohw,ocij->ncijhw', g, wd_save)
            for i in range(kH):
                for j in range(kW):
                    gxp[:, :, i*d0:i*d0+H_out*s0:s0, j*d1:j*d1+W_out*s1:s1] = \
                        gxp[:, :, i*d0:i*d0+H_out*s0:s0, j*d1:j*d1+W_out*s1:s1] + d_patches[:, :, i, j, :, :]
            gx = gxp[:, :, p0:p0+H, p1:p1+W] if (p0 or p1) else gxp
            return gx, gw

        y = C._out(out.astype(xd.dtype), (x, weight), back)
    return y + bias.reshape(1, -1, 1, 1) if bias is not None else y


def conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    y = conv2d(x.unsqueeze(2), weight.unsqueeze(2), None, (1, stride), (0, padding), (1, dilation), groups).squeeze(2)
    return y + bias.reshape(1, -1, 1) if bias is not None else y


# ── pooling ──────────────────────────────────────────────────────────────────

def _pool2d(x, kernel, stride, padding, mode):
    (kh, kw), (p0, p1) = _pair(kernel), _pair(padding)
    s0, s1 = _pair(stride if stride is not None else kernel)
    xd = x._data
    N_, Nc_, H, W = xd.shape
    fill = -float('inf') if mode == 'max' else 0.0
    xp = _nd_pad_fn(xd, ((0, 0), (0, 0), (p0, p0), (p1, p1)), constant_values=fill)
    Ho = (H + 2*p0 - kh) // s0 + 1
    Wo = (W + 2*p1 - kw) // s1 + 1

    # Build windows tensor (N_, Nc_, Ho, Wo, kh, kw) using _im2col-style slicing
    win = _nd_empty((N_, Nc_, Ho, Wo, kh, kw), dtype=xp.dtype)
    for pi in range(kh):
        for pj in range(kw):
            win[:, :, :, :, pi, pj] = xp[:, :, pi:pi + s0 * Ho:s0, pj:pj + s1 * Wo:s1]

    out = win.max(axis=(-1, -2)) if mode == 'max' else win.mean(axis=(-1, -2))

    def back(g):
        gxp = _nd_zeros(xp.shape, dtype=g.dtype)
        if mode == 'max':
            m_norm = _nd_zeros(win.shape, dtype=g.dtype)
            for bi in range(N_):
                for ci in range(Nc_):
                    for hi in range(Ho):
                        for wi in range(Wo):
                            ov = float(out[bi, ci, hi, wi])
                            s = sum(1.0 for ki in range(kh) for kj in range(kw)
                                    if float(win[bi, ci, hi, wi, ki, kj]) == ov)
                            if s > 0:
                                for ki in range(kh):
                                    for kj in range(kw):
                                        if float(win[bi, ci, hi, wi, ki, kj]) == ov:
                                            m_norm[bi, ci, hi, wi, ki, kj] = 1.0 / s
        for pi in range(kh):
            for pj in range(kw):
                if mode == 'max':
                    gxp[:, :, pi:pi + s0 * Ho:s0, pj:pj + s1 * Wo:s1] = (
                        gxp[:, :, pi:pi + s0 * Ho:s0, pj:pj + s1 * Wo:s1] + g * m_norm[:, :, :, :, pi, pj])
                else:
                    gxp[:, :, pi:pi + s0 * Ho:s0, pj:pj + s1 * Wo:s1] = (
                        gxp[:, :, pi:pi + s0 * Ho:s0, pj:pj + s1 * Wo:s1] + g * (1.0 / (kh * kw)))
        return (gxp[:, :, p0:p0 + H, p1:p1 + W],)
    return C._out(out.astype(xd.dtype), (x,), back)


def max_pool2d(x, kernel_size, stride=None, padding=0):
    return _pool2d(x, kernel_size, stride, padding, 'max')


def avg_pool2d(x, kernel_size, stride=None, padding=0):
    return _pool2d(x, kernel_size, stride, padding, 'avg')


def _pool1d(x, k, s, p, mode):
    k = k[0] if isinstance(k, (tuple, list)) else k
    s = (s[0] if isinstance(s, (tuple, list)) else s) if s is not None else k
    p = p[0] if isinstance(p, (tuple, list)) else p
    return _pool2d(x.unsqueeze(2), (1, k), (1, s), (0, p), mode).squeeze(2)


def adaptive_avg_pool2d(x, output_size):
    oh, ow = _pair(output_size)
    H, W = x._data.shape[2:]
    oh, ow = oh or H, ow or W
    if H % oh == 0 and W % ow == 0:
        return _pool2d(x, (H // oh, W // ow), (H // oh, W // ow), 0, 'avg')
    rows = []
    for i in range(oh):
        h0, h1 = (i * H) // oh, -(-((i + 1) * H) // oh)
        rows.append(C.stack([x[:, :, h0:h1, (j * W) // ow:-(-((j + 1) * W) // ow)].mean(dim=(2, 3)) for j in range(ow)], dim=-1))
    return C.stack(rows, dim=2)


def adaptive_avg_pool1d(x, output_size):
    return adaptive_avg_pool2d(x.unsqueeze(2), (1, output_size)).squeeze(2)


# ── normalisation ────────────────────────────────────────────────────────────

# Module-level buffer cache: keyed by (N, C) → (out_buf, xhat_buf, mean_buf, rstd_buf)
# Avoids 4MB+ of Python array.array allocation per BN call in training loops.
_bn_buf_cache = {}

def _batch_norm(self, x):
    import array as _arr
    from vectorphile._core.ndarray import NDArray as _NDA, _c_strides as _cs, _DType
    from vectorphile.autograd.engine import BatchNormBackward
    try:
        from vectorphile._core import _vp_core as _C
        _c_ok = True
    except ImportError:
        _c_ok = False

    xd = x._data
    C = xd.shape[1] if xd.ndim >= 2 else xd.shape[0]
    has_affine = getattr(self, 'affine', True)
    shape = (1, C) if xd.ndim == 2 else (1, C) + (1,) * (xd.ndim - 2)

    # C fast path: 2D float32 buf-backed training
    if (_c_ok and isinstance(xd, NDArray) and xd.ndim == 2 and str(xd.dtype) == 'float32'
            and xd._buf is not None and self.training):
        N = xd.shape[0]
        gamma_buf = beta_buf = None
        if has_affine:
            wd = self.weight._data; bd = self.bias_param._data
            gamma_buf = wd._buf if wd._buf is not None else _arr.array('f', wd._np_buf.ravel().astype('float32', copy=False).tobytes())
            beta_buf  = bd._buf if bd._buf is not None else _arr.array('f', bd._np_buf.ravel().astype('float32', copy=False).tobytes())
        _need_xhat = x.requires_grad
        _cache_key = (N, C, _need_xhat)
        if _cache_key in _bn_buf_cache:
            out_buf, xhat_buf, mean_buf, rstd_buf = _bn_buf_cache[_cache_key]
        else:
            out_buf  = _arr.array('f', bytearray(4 * N * C))
            mean_buf = _arr.array('f', bytearray(4 * C))
            rstd_buf = _arr.array('f', bytearray(4 * C))
            xhat_buf = _arr.array('f', bytearray(4 * N * C)) if _need_xhat else None
            _bn_buf_cache[_cache_key] = (out_buf, xhat_buf, mean_buf, rstd_buf)
        _C.batch_norm_f32(xd._buf, out_buf, mean_buf, rstd_buf,
                          gamma_buf, beta_buf, xhat_buf, N, C, float(self.eps))
        rm = self.running_mean; rv = self.running_var
        rm_buf = rm._buf; rv_buf = rv._buf
        if hasattr(_C, 'bn_update_running_f32') and rm_buf is not None and rv_buf is not None:
            _C.bn_update_running_f32(mean_buf, rstd_buf, rm_buf, rv_buf,
                                     C, float(self.momentum), float(self.eps), N)
        else:
            m = self.momentum; _1m = 1.0 - m
            scale_var = float(N) / max(N - 1, 1)
            eps = self.eps
            for i in range(C):
                var_i = max(0.0, 1.0 / (rstd_buf[i] * rstd_buf[i]) - eps)
                rm_buf[i] = _1m * rm_buf[i] + m * mean_buf[i]
                rv_buf[i] = _1m * rv_buf[i] + m * var_i * scale_var
        out_nd = _NDA(out_buf, (N, C), _cs((N, C)), _DType('float32'))
        result = Tensor(out_nd, device=x.device, _no_copy=True)
        if x.requires_grad:
            xhat_nd = _NDA(xhat_buf, (N, C), _cs((N, C)), _DType('float32'))
            rstd_nd = _NDA(rstd_buf, (C,), _cs((C,)), _DType('float32'))
            gamma_nd = self.weight._data if has_affine else None
            result.requires_grad = True
            result.grad_fn = BatchNormBackward(x, xhat_nd, rstd_nd, gamma_nd, has_affine)
            result._is_leaf = False
            if has_affine:
                result.grad_fn.inputs = (x, self.weight, self.bias_param)
        return result

    # eval path: use Tensor ops to preserve dtype and support gradcheck
    if not self.training:
        mu = x.mean(dim=tuple(i for i in range(xd.ndim) if i != 1), keepdim=True)
        xc = x - mu
        var = (xc * xc).mean(dim=tuple(i for i in range(xd.ndim) if i != 1), keepdim=True)
        y = xc / (var + self.eps).sqrt()
        if has_affine:
            y = y * self.weight.reshape(shape) + self.bias_param.reshape(shape)
        return y

    # training non-2D or non-buf fallback: use Tensor autograd ops
    mu = x.mean(dim=tuple(i for i in range(xd.ndim) if i != 1), keepdim=True)
    xc = x - mu
    var = (xc * xc).mean(dim=tuple(i for i in range(xd.ndim) if i != 1), keepdim=True)
    y = xc / (var + self.eps).sqrt()
    if has_affine:
        y = y * self.weight.reshape(shape) + self.bias_param.reshape(shape)
    return y


def _rms_norm(self, x):
    nd = len(self.normalized_shape)
    axes = tuple(range(-nd, 0))
    return x / ((x * x).mean(dim=axes, keepdim=True) + self.eps).sqrt() * self.weight


def _group_norm(self, x):
    s = x._data.shape
    g = x.reshape(s[0], self.num_groups, -1)
    mu = g.mean(dim=2, keepdim=True)
    gc = g - mu
    var = (gc * gc).mean(dim=2, keepdim=True)
    y = (gc / (var + self.eps).sqrt()).reshape(*s)
    if getattr(self, 'affine', True):
        shape = [1, -1] + [1] * (len(s) - 2)
        y = y * self.weight.reshape(*shape) + self.bias_param.reshape(*shape)
    return y


# ── attention ────────────────────────────────────────────────────────────────

def _mha_forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=True,
                 prefix_k=None, prefix_v=None, **_):
    """Scaled dot-product multi-head attention. Masks: bool True = blocked, or additive float.
    `prefix_k` / `prefix_v` ([P, E] tensors) are prepended to the projected keys/values (prefix tuning)."""
    if not self.batch_first:
        query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
    N, T, E = query._data.shape
    h, d = self.num_heads, self.head_dim
    q = query @ self.q_proj_weight.transpose(0, 1)
    k = key @ self.k_proj_weight.transpose(0, 1)
    v = value @ self.v_proj_weight.transpose(0, 1)
    if self.in_proj_bias is not None:
        q, k, v = q + self.in_proj_bias[:E], k + self.in_proj_bias[E:2 * E], v + self.in_proj_bias[2 * E:]
    P = 0
    if prefix_k is not None:
        P = prefix_k._data.shape[0]
        k = C.cat([prefix_k.unsqueeze(0).expand(N, P, E), k], dim=1)
        v = C.cat([prefix_v.unsqueeze(0).expand(N, P, E), v], dim=1)
    S = k._data.shape[1]
    q = q.reshape(N, T, h, d).transpose(1, 2)
    k = k.reshape(N, S, h, d).transpose(1, 2)
    v = v.reshape(N, S, h, d).transpose(1, 2)
    att = (q @ k.transpose(2, 3)) * (1.0 / math.sqrt(d))
    block = _nd_zeros((N, 1, T, S), dtype=bool)
    if attn_mask is not None:
        m = attn_mask._data if isinstance(attn_mask, Tensor) else _nd_array(attn_mask)
        m = _nd_pad_fn(m, [(0, 0)] * (m.ndim - 1) + [(P, 0)]) if P else m
        if m.dtype == 'bool' or m.dtype == bool:
            block = block | m.reshape((1, 1) + m.shape[-2:])
        else:
            att = att + Tensor(m.astype(att._data.dtype))
    if key_padding_mask is not None:
        kp = (key_padding_mask._data if isinstance(key_padding_mask, Tensor) else _nd_array(key_padding_mask)).astype(bool)
        kp = _nd_pad_fn(kp, ((0, 0), (P, 0))) if P else kp
        block = block | kp[:, None, None, :]
    if block.any():
        att = att.masked_fill(_nd_broadcast_to_fn(block, att._data.shape), -1e9)
    w = C.softmax(att, dim=-1)
    w = C.dropout(w, self.dropout_p, self.training)
    out = (w @ v).transpose(1, 2).reshape(N, T, E) @ self.out_proj_weight.transpose(0, 1)
    if self.out_proj_bias is not None:
        out = out + self.out_proj_bias
    if not self.batch_first:
        out = out.transpose(0, 1)
    return out, (w.mean(dim=1) if need_weights else None)


# ── recurrent ────────────────────────────────────────────────────────────────

def _affine(self, x, h):
    g = x @ self.weight_ih.transpose(0, 1) + h @ self.weight_hh.transpose(0, 1)
    for b in (getattr(self, 'bias_ih', None), getattr(self, 'bias_hh', None)):
        if b is not None:
            g = g + b
    return g


def _zeros(x, n):
    return Tensor(_nd_zeros((x._data.shape[0], n), dtype=x._data.dtype))


def _rnn_cell(self, x, hx=None):
    h = hx if hx is not None else _zeros(x, self.hidden_size)
    g = _affine(self, x, h)
    return g.relu() if getattr(self, 'nonlinearity', 'tanh') == 'relu' else g.tanh()


def _lstm_cell(self, x, hx=None):
    h, c = hx if hx is not None else (_zeros(x, self.hidden_size), _zeros(x, self.hidden_size))
    i, f, g, o = _affine(self, x, h).split(self.hidden_size, dim=1)
    c2 = f.sigmoid() * c + i.sigmoid() * g.tanh()
    return o.sigmoid() * c2.tanh(), c2


def _gru_cell(self, x, hx=None):
    h = hx if hx is not None else _zeros(x, self.hidden_size)
    H = self.hidden_size
    gi = x @ self.weight_ih.transpose(0, 1)
    gh = h @ self.weight_hh.transpose(0, 1)
    if getattr(self, 'bias_ih', None) is not None:
        gi, gh = gi + self.bias_ih, gh + self.bias_hh
    ir, iz, in_ = gi.split(H, dim=1)
    hr, hz, hn = gh.split(H, dim=1)
    r, z = (ir + hr).sigmoid(), (iz + hz).sigmoid()
    n = (in_ + r * hn).tanh()
    return (z * -1.0 + 1.0) * n + z * h


def _flat_cells(cells):
    out = []
    for c in cells:
        out.extend(_flat_cells(c) if isinstance(c, (list, tuple)) else [c])
    return out


def _register_cells(self):
    """Recurrent stacks kept their cells in a plain list; expose them as registered children."""
    cells = _flat_cells(self.cells)
    self._parameters = OrderedDict()
    self._modules = OrderedDict((k, v) for k, v in self._modules.items() if all(v is not c for c in cells))
    for i, c in enumerate(cells):
        self._modules[f'cells.{i}'] = c


def _seq_forward(self, x, hx=None, lstm=False):
    if self.batch_first:
        x = x.transpose(0, 1)
    T = x._data.shape[0]
    cells = _flat_cells(self.cells)
    dirs = 2 if getattr(self, 'bidirectional', False) else 1
    h_n, c_n = [], []
    inp = [x[t] for t in range(T)]
    for layer in range(self.num_layers):
        outs = []
        for d in range(dirs):
            cell = cells[layer * dirs + d]
            idx = layer * dirs + d
            if hx is None:
                state = None
            elif lstm:
                state = (hx[0][idx], hx[1][idx])
            else:
                state = hx[idx]
            seq, steps = [None] * T, (range(T) if d == 0 else range(T - 1, -1, -1))
            for t in steps:
                state = cell(inp[t], state)
                seq[t] = state[0] if lstm else state
            outs.append(seq)
            h_n.append(state[0] if lstm else state)
            if lstm:
                c_n.append(state[1])
        inp = [C.cat([outs[0][t], outs[1][t]], dim=1) for t in range(T)] if dirs == 2 else outs[0]
        if layer < self.num_layers - 1 and getattr(self, 'dropout_p', 0) > 0:
            inp = [C.dropout(v, self.dropout_p, self.training) for v in inp]
    out = C.stack(inp, dim=0)
    if self.batch_first:
        out = out.transpose(0, 1)
    h = C.stack(h_n, dim=0)
    return (out, (h, C.stack(c_n, dim=0))) if lstm else (out, h)


# ── installation ─────────────────────────────────────────────────────────────

def install():
    import vectorphile.nn.functional as F
    from vectorphile.nn import layers, pooling, norm, attention, recurrent

    for name, fn in {'conv1d': conv1d, 'conv2d': conv2d, 'max_pool2d': max_pool2d, 'avg_pool2d': avg_pool2d,
                     'adaptive_avg_pool2d': adaptive_avg_pool2d}.items():
        setattr(F, name, fn)

    layers.Conv2d.forward = lambda self, x: conv2d(x, self.weight, self.bias_param, self.stride, self.padding,
                                                    getattr(self, 'dilation', 1), getattr(self, 'groups', 1))
    layers.Conv1d.forward = lambda self, x: conv1d(x, self.weight, self.bias_param, self.stride, self.padding,
                                                    getattr(self, 'dilation', 1), getattr(self, 'groups', 1))
    layers.BatchNorm1d.forward = _batch_norm
    layers.BatchNorm2d.forward = _batch_norm
    # norm.py RMSNorm.forward and GroupNorm.forward use full Tensor ops with grad; keep them

    # Pooling forward/backward is handled natively in vectorphile/nn/pooling.py
    # (MaxPool1d/2d, AvgPool1d/2d all have grad_fn attached in their forward).
    # Do NOT override pool.forward here — the sliding_window_view path is broken.

    attention.MultiheadAttention.forward = _mha_forward

    # recurrent.py now has gradient-aware forward implementations; do not override them
