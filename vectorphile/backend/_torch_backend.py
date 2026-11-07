"""PyTorch backend — routes all vectorphile compute ops through torch for CUDA.

When ``backend.set_backend('torch')`` is active every heavy op (matmul, conv2d,
attention, etc.) converts its numpy inputs to torch tensors, executes on the
best available device (CUDA > MPS > CPU), then returns a numpy array.

All functions accept numpy arrays and return numpy arrays so the rest of
vectorphile never needs to know which backend is active.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple

try:
    import torch
    import torch.nn.functional as _F
    _AVAILABLE = True
    # Pick best device once at import time
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


def _t(arr: np.ndarray, device: str) -> 'torch.Tensor':
    """numpy → torch float32 on device."""
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(device)


def _n(t: 'torch.Tensor') -> np.ndarray:
    """torch → numpy float32 on CPU."""
    return t.detach().cpu().numpy()


# ── Core ops ──────────────────────────────────────────────────────────────────

def matmul(a: np.ndarray, b: np.ndarray,
           device: Optional[str] = None) -> np.ndarray:
    if not _AVAILABLE:
        return np.matmul(a, b)
    dev = device or _DEFAULT_DEVICE
    return _n(_t(a, dev) @ _t(b, dev))


def conv2d(x: np.ndarray, weight: np.ndarray,
           bias: Optional[np.ndarray] = None,
           stride: int = 1, padding: int = 0,
           device: Optional[str] = None) -> np.ndarray:
    if not _AVAILABLE:
        return _numpy_conv2d_fallback(x, weight, bias, stride, padding)
    dev = device or _DEFAULT_DEVICE
    tx = _t(x, dev)
    tw = _t(weight, dev)
    tb = _t(bias, dev) if bias is not None else None
    return _n(_F.conv2d(tx, tw, tb, stride=stride, padding=padding))


def _numpy_conv2d_fallback(x, w, b, stride, padding):
    """Pure-numpy GEMM conv via im2col (used when torch unavailable)."""
    from vectorphile.cuda import conv2d_im2col
    return conv2d_im2col(x, w, b, stride, padding)


def batch_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               running_mean: np.ndarray, running_var: np.ndarray,
               training: bool, momentum: float = 0.1, eps: float = 1e-5,
               device: Optional[str] = None) -> np.ndarray:
    if not _AVAILABLE:
        return _numpy_bn(x, weight, bias, running_mean, running_var, training, momentum, eps)
    dev = device or _DEFAULT_DEVICE
    tx  = _t(x, dev)
    tw  = _t(weight, dev)
    tb  = _t(bias, dev)
    trm = torch.from_numpy(running_mean.astype(np.float32)).to(dev)
    trv = torch.from_numpy(running_var.astype(np.float32)).to(dev)
    out = _F.batch_norm(tx, trm, trv, tw, tb, training, momentum, eps)
    # sync running stats back
    running_mean[:] = trm.cpu().numpy()
    running_var[:]  = trv.cpu().numpy()
    return _n(out)


def _numpy_bn(x, w, b, rm, rv, training, momentum, eps):
    if training:
        axes = tuple(i for i in range(x.ndim) if i != 1)
        mean = x.mean(axis=axes, keepdims=True)
        var  = x.var(axis=axes, keepdims=True)
        rm[:] = (1-momentum)*rm + momentum*x.mean(axis=axes)
        rv[:] = (1-momentum)*rv + momentum*x.var(axis=axes)
    else:
        sh = (1,-1) + (1,)*(x.ndim-2)
        mean = rm.reshape(sh); var = rv.reshape(sh)
    xn = (x - mean) / np.sqrt(var + eps)
    if x.ndim == 4:
        return xn * w[None,:,None,None] + b[None,:,None,None]
    return xn * w + b


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
               eps: float = 1e-5, device: Optional[str] = None) -> np.ndarray:
    if not _AVAILABLE:
        from vectorphile.cuda import layer_norm as _cuda_ln
        return _cuda_ln(x, weight, bias, eps)
    dev = device or _DEFAULT_DEVICE
    return _n(_F.layer_norm(_t(x, dev), [x.shape[-1]], _t(weight, dev), _t(bias, dev), eps))


def softmax(x: np.ndarray, axis: int = -1,
            device: Optional[str] = None) -> np.ndarray:
    if not _AVAILABLE:
        e = np.exp(x - x.max(axis=axis, keepdims=True))
        return (e / (e.sum(axis=axis, keepdims=True) + 1e-9)).astype(np.float32)
    dev = device or _DEFAULT_DEVICE
    return _n(_F.softmax(_t(x, dev), dim=axis))


def relu(x: np.ndarray, device: Optional[str] = None) -> np.ndarray:
    if not _AVAILABLE:
        return np.maximum(x, 0).astype(np.float32)
    dev = device or _DEFAULT_DEVICE
    return _n(_F.relu(_t(x, dev)))


def gelu(x: np.ndarray, device: Optional[str] = None) -> np.ndarray:
    if not _AVAILABLE:
        return (0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))).astype(np.float32)
    dev = device or _DEFAULT_DEVICE
    return _n(_F.gelu(_t(x, dev)))


def silu(x: np.ndarray, device: Optional[str] = None) -> np.ndarray:
    if not _AVAILABLE:
        return (x / (1 + np.exp(-x))).astype(np.float32)
    dev = device or _DEFAULT_DEVICE
    return _n(_F.silu(_t(x, dev)))


def embedding_lookup(weight: np.ndarray, ids: np.ndarray,
                     device: Optional[str] = None) -> np.ndarray:
    if not _AVAILABLE:
        return weight[ids.astype(int)].astype(np.float32)
    dev = device or _DEFAULT_DEVICE
    tw = _t(weight, dev)
    ti = torch.from_numpy(ids.astype(np.int64)).to(dev)
    return _n(_F.embedding(ti, tw))


def scaled_dot_product_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray,
    mask: Optional[np.ndarray] = None,
    dropout_p: float = 0.0, training: bool = False,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention dispatched to torch (supports flash-attn internally)."""
    if not _AVAILABLE:
        return _numpy_sdpa(q, k, v, mask, dropout_p, training)
    dev = device or _DEFAULT_DEVICE
    tq = _t(q, dev); tk = _t(k, dev); tv = _t(v, dev)
    # torch >= 2.0 has efficient SDPA
    if hasattr(_F, 'scaled_dot_product_attention'):
        attn_mask = None
        if mask is not None:
            attn_mask = torch.from_numpy(mask.astype(np.float32)).to(dev)
            attn_mask = attn_mask.masked_fill(attn_mask.bool(), float('-inf')) if mask.dtype == bool else attn_mask
        out = _F.scaled_dot_product_attention(tq, tk, tv, attn_mask=attn_mask,
                                               dropout_p=dropout_p if training else 0.0)
        # weights not returned by torch SDPA; compute separately for compatibility
        scale = 1.0 / np.sqrt(q.shape[-1])
        scores = tq @ tk.transpose(-2, -1) * scale
        if attn_mask is not None:
            scores = scores + attn_mask
        weights = _F.softmax(scores, dim=-1)
        return _n(out), _n(weights)
    # torch < 2.0 fallback
    return _numpy_sdpa(q, k, v, mask, dropout_p, training)


def _numpy_sdpa(q, k, v, mask, dropout_p, training):
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = q @ k.swapaxes(-2, -1) * scale
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    e = np.exp(scores - scores.max(-1, keepdims=True))
    w = e / (e.sum(-1, keepdims=True) + 1e-9)
    if dropout_p > 0 and training:
        keep = np.random.binomial(1, 1-dropout_p, w.shape).astype(np.float32)
        w = w * keep / (1-dropout_p)
    return (w @ v).astype(np.float32), w.astype(np.float32)


def adam_step(params: np.ndarray, grads: np.ndarray,
              m: np.ndarray, v: np.ndarray,
              lr: float, beta1: float, beta2: float,
              eps: float, weight_decay: float, t: int,
              device: Optional[str] = None) -> None:
    """Fused Adam update via torch (in-place, numpy arrays modified directly)."""
    if not _AVAILABLE:
        from vectorphile.cuda import adam_step as _cuda_adam
        _cuda_adam(params, grads, m, v, lr, beta1, beta2, eps, weight_decay, t)
        return
    dev = device or _DEFAULT_DEVICE
    tp = torch.from_numpy(params).to(dev)
    tg = torch.from_numpy(grads).to(dev)
    tm = torch.from_numpy(m).to(dev)
    tv_ = torch.from_numpy(v).to(dev)
    if weight_decay > 0:
        tg = tg + weight_decay * tp
    tm.mul_(beta1).add_(tg, alpha=1-beta1)
    tv_.mul_(beta2).addcmul_(tg, tg, value=1-beta2)
    bc1 = 1 - beta1**t; bc2 = 1 - beta2**t
    step = tm / bc1 / (tv_.sqrt() / np.sqrt(bc2) + eps)
    tp.sub_(step, alpha=lr)
    params[:] = tp.cpu().numpy()
    m[:]      = tm.cpu().numpy()
    v[:]      = tv_.cpu().numpy()


# ── torch.compile wrapping ────────────────────────────────────────────────────

def compile_model(model, example_input=None, mode: str = 'default'):
    """Wrap a vectorphile Module in torch.compile for maximum throughput.

    Requires PyTorch >= 2.0. Falls back silently if unavailable.

    Args:
        model: vectorphile Module
        example_input: ignored (torch.compile traces lazily on first call)
        mode: 'default', 'reduce-overhead', or 'max-autotune'
    Returns:
        torch.compile'd module wrapper or original model
    """
    if not _AVAILABLE or not hasattr(torch, 'compile'):
        return model  # torch < 2.0 or not installed

    class _TorchWrapper:
        """Wraps a vectorphile Module and routes forward() through torch.compile."""
        def __init__(self, vp_model):
            self._vp = vp_model
            # Build a thin torch.nn.Module that calls vp_model.forward with numpy
            import torch.nn as _nn

            class _Bridge(_nn.Module):
                def __init__(self_, vp):
                    super().__init__()
                    self_.vp = vp
                def forward(self_, *args):
                    from vectorphile.core.tensor import Tensor as _T
                    np_args = [a.cpu().numpy() if isinstance(a, torch.Tensor) else a for a in args]
                    vp_args = [_T(a) for a in np_args]
                    result = self_.vp(*vp_args)
                    if hasattr(result, '_data'):
                        return torch.from_numpy(result._data)
                    return result

            self._bridge = torch.compile(_Bridge(vp_model), mode=mode)

        def __call__(self, *args):
            import torch as _torch
            torch_args = [
                _torch.from_numpy(np.asarray(a._data if hasattr(a, '_data') else a, dtype=np.float32))
                for a in args
            ]
            out = self._bridge(*torch_args)
            if isinstance(out, _torch.Tensor):
                from vectorphile.core.tensor import Tensor as _T
                return _T(out.detach().cpu().numpy())
            return out

        def __getattr__(self, name):
            return getattr(self._vp, name)

    return _TorchWrapper(model)
