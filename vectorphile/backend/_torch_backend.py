"""PyTorch backend — routes all vectorphile compute ops through torch for CUDA.

When ``backend.set_backend('torch')`` is active every heavy op (matmul, conv2d,
attention, etc.) converts its NDArray inputs to torch tensors, executes on the
best available device (CUDA > MPS > CPU), then returns an NDArray.
"""
from __future__ import annotations
import math as _math
import random as _random
from typing import Optional, Tuple
from vectorphile._core.ndarray import NDArray, array as _nd_array, zeros as _nd_zeros

try:
    import torch
    import torch.nn.functional as _F
    _AVAILABLE = True
    if torch.cuda.is_available():
        _DEFAULT_DEVICE = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        _DEFAULT_DEVICE = 'mps'
    else:
        _DEFAULT_DEVICE = 'cpu'
except ImportError:
    torch = None
    _F = None
    _AVAILABLE = False
    _DEFAULT_DEVICE = 'cpu'


def is_available() -> bool:
    return _AVAILABLE


def get_default_device() -> str:
    return _DEFAULT_DEVICE


def _to_torch(arr, device: str):
    """NDArray/list → torch float32 on device."""
    if isinstance(arr, NDArray):
        flat = arr.to_flat_list()
        t = torch.tensor(flat, dtype=torch.float32).reshape(arr.shape)
    elif isinstance(arr, (list, tuple)):
        t = torch.tensor(arr, dtype=torch.float32)
    else:
        t = torch.as_tensor(arr, dtype=torch.float32)
    return t.to(device)


def _to_nd(t) -> NDArray:
    """torch → NDArray float32 on CPU."""
    flat = t.detach().cpu().float().reshape(-1).tolist()
    return _nd_array(flat, dtype='float32').reshape(tuple(t.shape))


def matmul(a, b, device: Optional[str] = None):
    if not _AVAILABLE:
        return _nd_matmul_fallback(a, b)
    dev = device or _DEFAULT_DEVICE
    return _to_nd(_to_torch(a, dev) @ _to_torch(b, dev))


def _nd_matmul_fallback(a, b):
    if isinstance(a, NDArray) and isinstance(b, NDArray):
        return a @ b
    return _nd_array(a) @ _nd_array(b)


def conv2d(x, weight, bias=None, stride: int = 1, padding: int = 0,
           device: Optional[str] = None):
    if not _AVAILABLE:
        return _nd_conv2d_fallback(x, weight, bias, stride, padding)
    dev = device or _DEFAULT_DEVICE
    tx = _to_torch(x, dev)
    tw = _to_torch(weight, dev)
    tb = _to_torch(bias, dev) if bias is not None else None
    return _to_nd(_F.conv2d(tx, tw, tb, stride=stride, padding=padding))


def _nd_conv2d_fallback(x, w, b, stride, padding):
    from vectorphile.cuda import conv2d_im2col
    return conv2d_im2col(x, w, b, stride, padding)


def batch_norm(x, weight, bias, running_mean, running_var,
               training: bool, momentum: float = 0.1, eps: float = 1e-5,
               device: Optional[str] = None):
    if not _AVAILABLE:
        return _nd_bn(x, weight, bias, running_mean, running_var, training, momentum, eps)
    dev = device or _DEFAULT_DEVICE
    tx  = _to_torch(x, dev)
    tw  = _to_torch(weight, dev)
    tb  = _to_torch(bias, dev)
    rm_flat = running_mean.to_flat_list() if isinstance(running_mean, NDArray) else list(running_mean)
    rv_flat = running_var.to_flat_list() if isinstance(running_var, NDArray) else list(running_var)
    trm = torch.tensor(rm_flat, dtype=torch.float32).to(dev)
    trv = torch.tensor(rv_flat, dtype=torch.float32).to(dev)
    out = _F.batch_norm(tx, trm, trv, tw, tb, training, momentum, eps)
    # sync running stats back
    if isinstance(running_mean, NDArray):
        new_rm = trm.cpu().tolist()
        for i, v in enumerate(new_rm): running_mean._buf[running_mean.offset + i] = v
    if isinstance(running_var, NDArray):
        new_rv = trv.cpu().tolist()
        for i, v in enumerate(new_rv): running_var._buf[running_var.offset + i] = v
    return _to_nd(out)


def _nd_bn(x, w, b, rm, rv, training, momentum, eps):
    xf = x.to_flat_list() if isinstance(x, NDArray) else list(x)
    sh = x.shape if isinstance(x, NDArray) else (len(x),)
    # simplified 1D BN
    n = len(xf)
    if training:
        mu = sum(xf) / n
        var = sum((v - mu)**2 for v in xf) / n
    else:
        mu = float(rm.to_flat_list()[0]) if isinstance(rm, NDArray) else float(rm[0])
        var = float(rv.to_flat_list()[0]) if isinstance(rv, NDArray) else float(rv[0])
    inv = 1.0 / _math.sqrt(var + eps)
    wf = w.to_flat_list() if isinstance(w, NDArray) else list(w)
    bf = b.to_flat_list() if isinstance(b, NDArray) else list(b)
    result = [(xf[i] - mu) * inv * wf[i % len(wf)] + bf[i % len(bf)] for i in range(n)]
    return _nd_array(result, dtype='float32').reshape(sh)


def layer_norm(x, weight, bias, eps: float = 1e-5, device: Optional[str] = None):
    if not _AVAILABLE:
        from vectorphile.cuda import layer_norm as _cuda_ln
        return _cuda_ln(x, weight, bias, eps)
    dev = device or _DEFAULT_DEVICE
    tx = _to_torch(x, dev)
    sh = list(tx.shape)
    return _to_nd(_F.layer_norm(tx, [sh[-1]], _to_torch(weight, dev), _to_torch(bias, dev), eps))


def softmax(x, axis: int = -1, device: Optional[str] = None):
    if not _AVAILABLE:
        xf = x.to_flat_list() if isinstance(x, NDArray) else list(x)
        sh = x.shape if isinstance(x, NDArray) else (len(x),)
        mx = max(xf); e = [_math.exp(v - mx) for v in xf]
        s = sum(e); result = [v / s for v in e]
        return _nd_array(result, dtype='float32').reshape(sh)
    dev = device or _DEFAULT_DEVICE
    return _to_nd(_F.softmax(_to_torch(x, dev), dim=axis))


def relu(x, device: Optional[str] = None):
    if not _AVAILABLE:
        xf = x.to_flat_list() if isinstance(x, NDArray) else list(x)
        sh = x.shape if isinstance(x, NDArray) else (len(x),)
        return _nd_array([max(0.0, v) for v in xf], dtype='float32').reshape(sh)
    dev = device or _DEFAULT_DEVICE
    return _to_nd(_F.relu(_to_torch(x, dev)))


def gelu(x, device: Optional[str] = None):
    if not _AVAILABLE:
        xf = x.to_flat_list() if isinstance(x, NDArray) else list(x)
        sh = x.shape if isinstance(x, NDArray) else (len(x),)
        result = [0.5*v*(1+_math.tanh(_math.sqrt(2/_math.pi)*(v+0.044715*v**3))) for v in xf]
        return _nd_array(result, dtype='float32').reshape(sh)
    dev = device or _DEFAULT_DEVICE
    return _to_nd(_F.gelu(_to_torch(x, dev)))


def silu(x, device: Optional[str] = None):
    if not _AVAILABLE:
        xf = x.to_flat_list() if isinstance(x, NDArray) else list(x)
        sh = x.shape if isinstance(x, NDArray) else (len(x),)
        result = [v / (1 + _math.exp(-v)) for v in xf]
        return _nd_array(result, dtype='float32').reshape(sh)
    dev = device or _DEFAULT_DEVICE
    return _to_nd(_F.silu(_to_torch(x, dev)))


def embedding_lookup(weight, ids, device: Optional[str] = None):
    if not _AVAILABLE:
        wf = weight.to_flat_list() if isinstance(weight, NDArray) else list(weight)
        ws = weight.shape if isinstance(weight, NDArray) else (len(weight),)
        dim = ws[1] if len(ws) > 1 else 1
        ids_list = ids.to_flat_list() if isinstance(ids, NDArray) else list(ids)
        result = [wf[int(i)*dim:(int(i)+1)*dim] for i in ids_list]
        flat = [v for row in result for v in row]
        return _nd_array(flat, dtype='float32').reshape((len(ids_list), dim))
    dev = device or _DEFAULT_DEVICE
    tw = _to_torch(weight, dev)
    ids_list = ids.to_flat_list() if isinstance(ids, NDArray) else list(ids)
    ti = torch.tensor([int(i) for i in ids_list], dtype=torch.int64).to(dev)
    return _to_nd(_F.embedding(ti, tw))


def scaled_dot_product_attention(q, k, v, mask=None, dropout_p: float = 0.0,
                                  training: bool = False, device: Optional[str] = None):
    if not _AVAILABLE:
        return _nd_sdpa(q, k, v, mask, dropout_p, training)
    dev = device or _DEFAULT_DEVICE
    tq = _to_torch(q, dev); tk = _to_torch(k, dev); tv = _to_torch(v, dev)
    if hasattr(_F, 'scaled_dot_product_attention'):
        attn_mask = None
        if mask is not None:
            mask_flat = mask.to_flat_list() if isinstance(mask, NDArray) else list(mask)
            attn_mask = torch.tensor(mask_flat, dtype=torch.float32).reshape(
                mask.shape if isinstance(mask, NDArray) else (len(mask),)).to(dev)
        out = _F.scaled_dot_product_attention(tq, tk, tv, attn_mask=attn_mask,
                                               dropout_p=dropout_p if training else 0.0)
        scale = 1.0 / _math.sqrt(q.shape[-1] if isinstance(q, NDArray) else tq.shape[-1])
        scores = tq @ tk.transpose(-2, -1) * scale
        if attn_mask is not None:
            scores = scores + attn_mask
        weights = _F.softmax(scores, dim=-1)
        return _to_nd(out), _to_nd(weights)
    return _nd_sdpa(q, k, v, mask, dropout_p, training)


def _nd_sdpa(q, k, v, mask, dropout_p, training):
    dim = q.shape[-1] if isinstance(q, NDArray) else 1
    scale = 1.0 / _math.sqrt(dim)
    # NDArray matmul for scores
    if isinstance(q, NDArray) and isinstance(k, NDArray):
        scores = q @ k.T * scale
    else:
        scores = _nd_array(q) @ _nd_array(k).T * scale
    scores_flat = scores.to_flat_list()
    if mask is not None:
        mask_flat = mask.to_flat_list() if isinstance(mask, NDArray) else list(mask)
        scores_flat = [-1e9 if mask_flat[i] else scores_flat[i] for i in range(len(scores_flat))]
    mx = max(scores_flat)
    e = [_math.exp(v - mx) for v in scores_flat]
    s = sum(e)
    w_flat = [v / (s + 1e-9) for v in e]
    w = _nd_array(w_flat, dtype='float32').reshape(scores.shape)
    if isinstance(v, NDArray):
        out = w @ v
    else:
        out = w @ _nd_array(v)
    return out, w


def adam_step(params, grads, m, v, lr: float, beta1: float, beta2: float,
              eps: float, weight_decay: float, t: int, device: Optional[str] = None) -> None:
    if not _AVAILABLE:
        from vectorphile.cuda import adam_step as _cuda_adam
        _cuda_adam(params, grads, m, v, lr, beta1, beta2, eps, weight_decay, t)
        return
    dev = device or _DEFAULT_DEVICE
    pf = params.to_flat_list() if isinstance(params, NDArray) else list(params)
    gf = grads.to_flat_list() if isinstance(grads, NDArray) else list(grads)
    mf = m.to_flat_list() if isinstance(m, NDArray) else list(m)
    vf = v.to_flat_list() if isinstance(v, NDArray) else list(v)
    bc1 = 1 - beta1**t; bc2 = 1 - beta2**t
    for i in range(len(pf)):
        g = gf[i] + weight_decay * pf[i]
        mf[i] = beta1 * mf[i] + (1 - beta1) * g
        vf[i] = beta2 * vf[i] + (1 - beta2) * g * g
        step = (mf[i] / bc1) / (_math.sqrt(vf[i] / bc2) + eps)
        pf[i] -= lr * step
    if isinstance(params, NDArray):
        for i, val in enumerate(pf): params._buf[params.offset + i] = val
    if isinstance(m, NDArray):
        for i, val in enumerate(mf): m._buf[m.offset + i] = val
    if isinstance(v, NDArray):
        for i, val in enumerate(vf): v._buf[v.offset + i] = val


def compile_model(model, example_input=None, mode: str = 'default'):
    """Wrap a vectorphile Module in torch.compile for maximum throughput."""
    if not _AVAILABLE or not hasattr(torch, 'compile'):
        return model
    try:
        return torch.compile(model, mode=mode)
    except Exception:
        return model
