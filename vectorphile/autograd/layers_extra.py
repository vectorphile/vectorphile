"""Differentiable Transformer wrapper layers, InstanceNorm, 3-D / adaptive-max pooling, transposed convolution."""
from __future__ import annotations
from vectorphile._core.ndarray import NDArray, array as _nd_array, zeros as _nd_zeros, ones as _nd_ones, full as _nd_full, empty as _nd_empty, eye as _nd_eye, arange as _nd_arange, linspace as _nd_linspace
import math as _math
from vectorphile.core.tensor import Tensor
from vectorphile.autograd import complete as C
from vectorphile.autograd import layers_complete as L
from vectorphile._vp_helpers import *



def _act(self, x):
    a = getattr(self, 'activation', 'relu')
    if callable(a):
        return a(x)
    return C.gelu(x) if a == 'gelu' else x.relu()


def _ff(self, x):
    return self.linear2(self.dropout(_act(self, self.linear1(x))))


def _encoder_layer(self, src, src_mask=None, src_key_padding_mask=None, **_):
    sa = lambda h: self.self_attn(h, h, h, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=False)[0]
    if getattr(self, 'norm_first', False):
        src = src + self.dropout1(sa(self.norm1(src)))
        return src + self.dropout2(_ff(self, self.norm2(src)))
    src = self.norm1(src + self.dropout1(sa(src)))
    return self.norm2(src + self.dropout2(_ff(self, src)))


def _decoder_layer(self, tgt, memory, tgt_mask=None, memory_mask=None,
                   tgt_key_padding_mask=None, memory_key_padding_mask=None, **_):
    sa = lambda h: self.self_attn(h, h, h, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False)[0]
    ca = lambda h: self.cross_attn(h, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask, need_weights=False)[0]
    if getattr(self, 'norm_first', False):
        tgt = tgt + self.dropout1(sa(self.norm1(tgt)))
        tgt = tgt + self.dropout2(ca(self.norm2(tgt)))
        return tgt + self.dropout3(_ff(self, self.norm3(tgt)))
    tgt = self.norm1(tgt + self.dropout1(sa(tgt)))
    tgt = self.norm2(tgt + self.dropout2(ca(tgt)))
    return self.norm3(tgt + self.dropout3(_ff(self, tgt)))


def _encoder(self, src, mask=None, src_key_padding_mask=None, **_):
    for layer in self.layers:
        src = layer(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
    return self.norm(src) if self.norm is not None else src


def _decoder(self, tgt, memory, tgt_mask=None, memory_mask=None,
             tgt_key_padding_mask=None, memory_key_padding_mask=None, **_):
    for layer in self.layers:
        tgt = layer(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
    return self.norm(tgt) if self.norm is not None else tgt


def _instance_norm(self, x):
    # Reduce over all spatial dims (2..N) one at a time to get correct (N,C,1,1,...) shape
    mu = x
    for ax in range(x._data.ndim - 1, 1, -1):
        mu = mu.mean(dim=ax, keepdim=True)
    xc = x - mu
    xc2 = xc * xc
    var = xc2
    for ax in range(xc2._data.ndim - 1, 1, -1):
        var = var.mean(dim=ax, keepdim=True)
    y = xc / (var + self.eps).sqrt()
    if getattr(self, 'affine', False):
        shape = [1, -1] + [1] * (x._data.ndim - 2)
        y = y * self.weight.reshape(*shape) + self.bias_param.reshape(*shape)
    return y


def adaptive_max_pool2d(x, output_size):
    oh, ow = L._pair(output_size)
    H, W = x._data.shape[2:]
    oh, ow = oh or H, ow or W
    if H % oh == 0 and W % ow == 0:
        return L._pool2d(x, (H // oh, W // ow), (H // oh, W // ow), 0, 'max')
    rows = []
    for i in range(oh):
        h0, h1 = (i * H) // oh, -(-((i + 1) * H) // oh)
        cols = []
        for j in range(ow):
            w0, w1 = (j * W) // ow, -(-((j + 1) * W) // ow)
            cols.append(L._pool2d(x[:, :, h0:h1, w0:w1], (h1 - h0, w1 - w0), None, 0, 'max'))
        rows.append(C.cat(cols, dim=3))
    return C.cat(rows, dim=2)


def _triple(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v, v)


def _pool3d(x, kernel, stride, padding, mode):
    """Max and average pooling are separable: pool H x W, then depth."""
    (kd, kh, kw), (pd, ph, pw) = _triple(kernel), _triple(padding)
    sd, sh, sw = _triple(stride if stride is not None else kernel)
    N, Cc, D, H, W = x._data.shape
    y = L._pool2d(x.reshape(N, Cc * D, H, W), (kh, kw), (sh, sw), (ph, pw), mode)
    Ho, Wo = y._data.shape[2:]
    y = L._pool2d(y.reshape(N, Cc, D, Ho * Wo), (kd, 1), (sd, 1), (pd, 0), mode)
    return y.reshape(N, Cc, y._data.shape[2], Ho, Wo)


def conv_transpose2d(x, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    """weight: [in_channels, out_channels, kH, kW]."""
    if groups != 1 or L._pair(dilation) != (1, 1):
        raise NotImplementedError('conv_transpose2d supports groups=1 and dilation=1')
    (s0, s1), (p0, p1), (o0, o1) = L._pair(stride), L._pair(padding), L._pair(output_padding)
    xd, w = x._data, weight._data
    N, _, H, W = xd.shape
    _, O, kh, kw = w.shape
    fh, fw = (H - 1) * s0 + kh + o0, (W - 1) * s1 + kw + o1

    def crop(a):
        return a[:, :, p0:fh - p0, p1:fw - p1]
    full = _nd_zeros((N, O, fh, fw), dtype=xd.dtype)
    for i in range(kh):
        for j in range(kw):
            full[:, :, i:i + s0 * H:s0, j:j + s1 * W:s1] += _nd_einsum_fn('nchw,co->nohw', xd, w[:, :, i, j], optimize=True)

    def back(g):
        gf = _nd_zeros((N, O, fh, fw), dtype=g.dtype)
        crop(gf)[...] = g
        gx, gw = _nd_zeros_like(xd), _nd_zeros_like(w)
        for i in range(kh):
            for j in range(kw):
                sl = gf[:, :, i:i + s0 * H:s0, j:j + s1 * W:s1]
                gx += _nd_einsum_fn('nohw,co->nchw', sl, w[:, :, i, j], optimize=True)
                gw[:, :, i, j] = _nd_einsum_fn('nchw,nohw->co', xd, sl, optimize=True)
        return gx, gw
    y = C._out(crop(full).copy(), (x, weight), back)
    return y + bias.reshape(1, -1, 1, 1) if bias is not None else y


def _register_layers(cls):
    orig = cls.__init__

    def init(self, *a, __orig=orig, **k):
        # Convenience: TransformerEncoder(d_model, nhead, num_layers) shorthand
        if cls.__name__ == 'TransformerEncoder' and 'd_model' in k:
            from vectorphile.nn.attention import TransformerEncoderLayer as _TEL
            d_model = k.pop('d_model')
            nhead = k.pop('nhead', 1)
            num_layers = k.pop('num_layers', 1)
            dim_ff = k.pop('dim_feedforward', 2048)
            dropout = k.pop('dropout', 0.1)
            activation = k.pop('activation', 'relu')
            layer = _TEL(d_model, nhead, dim_ff, dropout, activation)
            __orig(self, layer, num_layers, **k)
        elif cls.__name__ == 'TransformerDecoder' and 'd_model' in k:
            from vectorphile.nn.attention import TransformerDecoderLayer as _TDL
            d_model = k.pop('d_model')
            nhead = k.pop('nhead', 1)
            num_layers = k.pop('num_layers', 1)
            dim_ff = k.pop('dim_feedforward', 2048)
            dropout = k.pop('dropout', 0.1)
            activation = k.pop('activation', 'relu')
            layer = _TDL(d_model, nhead, dim_ff, dropout, activation)
            __orig(self, layer, num_layers, **k)
        else:
            __orig(self, *a, **k)
        self._parameters.clear()
        for i, layer in enumerate(self.layers):
            self._modules[f'layers.{i}'] = layer
        if getattr(self, 'norm', None) is not None:
            self._modules['norm'] = self.norm
    cls.__init__ = init


def install():
    import vectorphile.nn.functional as F
    from vectorphile.nn import attention, norm, pooling

    attention.TransformerEncoderLayer.forward = _encoder_layer
    attention.TransformerDecoderLayer.forward = _decoder_layer
    attention.TransformerEncoder.forward = _encoder
    attention.TransformerDecoder.forward = _decoder
    _register_layers(attention.TransformerEncoder)
    _register_layers(attention.TransformerDecoder)

    for name in ('InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d'):
        if hasattr(norm, name):
            getattr(norm, name).forward = _instance_norm

    # MaxPool3d/AvgPool3d/AdaptiveMaxPool2d have native backward in pooling.py;
    # do NOT override their forward with the broken sliding_window_view path.
    F.conv_transpose2d = conv_transpose2d
    F.adaptive_max_pool2d = adaptive_max_pool2d
