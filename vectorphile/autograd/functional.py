"""Higher-order autograd utilities: grad, jacobian, hessian, vmap, checkpoint."""
from __future__ import annotations
import numpy as np
from contextlib import contextmanager
from typing import Callable, List, Optional

_GRAD_ENABLED = True


@contextmanager
def enable_grad():
    """Force gradient tracking inside a no_grad context."""
    from vectorphile.autograd.engine import no_grad as _no_grad
    prev = _no_grad._enabled
    _no_grad._enabled = True
    try:
        yield
    finally:
        _no_grad._enabled = prev


def grad(f: Callable, x) -> 'Tensor':
    """Return gradient of scalar f(x) w.r.t. x.

    Args:
        f: callable Tensor -> scalar Tensor
        x: input Tensor
    Returns:
        gradient Tensor with same shape as x
    """
    from vectorphile.core.tensor import Tensor as CoreTensor
    x.requires_grad = True
    x.grad = None
    out = f(x)
    out.backward(CoreTensor(np.ones_like(out._data)))
    return CoreTensor(x.grad._data.copy() if x.grad is not None else np.zeros_like(x._data))


def jacobian(f: Callable, x) -> np.ndarray:
    """Full Jacobian of f(x) w.r.t. x via repeated backward passes.

    Args:
        f: callable Tensor[n] -> Tensor[m]
        x: 1-D input Tensor
    Returns:
        J: np.ndarray shape (m, n)
    """
    from vectorphile.core.tensor import Tensor as CoreTensor
    x_np = x._data.flatten()
    n = len(x_np)
    out0_data = f(CoreTensor(x_np))._data.flatten()
    m = len(out0_data)
    J = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        xi = CoreTensor(x_np, requires_grad=True)
        out = f(xi)
        e = np.zeros(m, dtype=np.float32); e[i] = 1.0
        out.backward(CoreTensor(e.reshape(out._data.shape)))
        J[i] = xi.grad._data.flatten() if xi.grad is not None else 0
    return J


def hessian(f: Callable, x) -> np.ndarray:
    """Hessian of scalar f(x) via central finite differences.

    Args:
        f: callable Tensor[n] -> scalar Tensor
        x: 1-D input Tensor
    Returns:
        H: np.ndarray shape (n, n)
    """
    from vectorphile.core.tensor import Tensor as CoreTensor
    x_np = x._data.flatten()
    n = len(x_np)
    H = np.zeros((n, n), dtype=np.float32)
    eps = 1e-4
    for i in range(n):
        for j in range(n):
            def _f(v): return f(CoreTensor(v))._data.item()
            xpp = x_np.copy(); xpp[i] += eps; xpp[j] += eps
            xpm = x_np.copy(); xpm[i] += eps; xpm[j] -= eps
            xmp = x_np.copy(); xmp[i] -= eps; xmp[j] += eps
            xmm = x_np.copy(); xmm[i] -= eps; xmm[j] -= eps
            H[i,j] = (_f(xpp) - _f(xpm) - _f(xmp) + _f(xmm)) / (4 * eps**2)
    return H


def vmap(f: Callable) -> Callable:
    """Vectorised map: apply f to each element of a batched Tensor.

    Usage::

        batched = vmap(f)
        out = batched(x)   # x: (B, ...) applies f to each [i, ...]

    Returns:
        stacked Tensor of shape (B, *output_shape)
    """
    def batched(x):
        from vectorphile.core.tensor import Tensor as CoreTensor
        results = [f(CoreTensor(x._data[i])) for i in range(x._data.shape[0])]
        return CoreTensor(np.stack([r._data for r in results], axis=0))
    return batched


def checkpoint(fn: Callable, *args):
    """Gradient checkpointing: recompute activations on backward instead of storing.

    Saves memory at the cost of extra compute. The forward pass is run once
    without grad to get the output; on backward it reruns with grad.

    Args:
        fn: a callable that takes Tensors and returns a Tensor
        *args: input Tensors
    Returns:
        output Tensor; backward will recompute fn
    """
    from vectorphile.core.tensor import Tensor as CoreTensor
    from vectorphile.autograd.engine import no_grad as _no_grad

    # Forward without grad to get output value
    saved_enabled = _no_grad._enabled
    _no_grad._enabled = False
    with_no_grad_args = [CoreTensor(a._data) for a in args]
    _no_grad._enabled = False
    out_data = fn(*with_no_grad_args)._data.copy()
    _no_grad._enabled = saved_enabled

    out = CoreTensor(out_data, requires_grad=True)

    # On backward: rerun fn with grads
    class _CheckpointFn:
        def __init__(self):
            self.inputs = list(args)
        def backward(self, grad_out):
            fresh = [CoreTensor(a._data, requires_grad=a.requires_grad) for a in args]
            result = fn(*fresh)
            result.backward(grad_out)
            return [a.grad for a in fresh]

    out.grad_fn = _CheckpointFn()
    return out


# ── Tensor construction helpers ───────────────────────────────────────────────

def zeros(shape, requires_grad: bool = False):
    from vectorphile.core.tensor import Tensor as CoreTensor
    return CoreTensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)


def ones(shape, requires_grad: bool = False):
    from vectorphile.core.tensor import Tensor as CoreTensor
    return CoreTensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)


def randn(*shape, requires_grad: bool = False):
    from vectorphile.core.tensor import Tensor as CoreTensor
    return CoreTensor(np.random.randn(*shape).astype(np.float32), requires_grad=requires_grad)


def cat(tensors: list, axis: int = 0):
    """Concatenate tensors."""
    from vectorphile.core.tensor import Tensor as CoreTensor
    return CoreTensor(np.concatenate([t._data for t in tensors], axis=axis))


def stack(tensors: list, axis: int = 0):
    """Stack tensors along new axis."""
    from vectorphile.core.tensor import Tensor as CoreTensor
    return CoreTensor(np.stack([t._data for t in tensors], axis=axis))
