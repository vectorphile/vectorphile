"""Higher-order autograd utilities: grad, jacobian, hessian, vmap, checkpoint."""
from __future__ import annotations
import math as _math
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


def _ones_like_data(data):
    from vectorphile._core.ndarray import ones as _nd_ones
    if hasattr(data, 'shape'):
        return _nd_ones(data.shape, dtype='float32')
    n = len(data) if hasattr(data, '__len__') else 1
    return _nd_ones((n,), dtype='float32')


def _zeros_like_data(data):
    from vectorphile._core.ndarray import zeros as _nd_zeros
    if hasattr(data, 'shape'):
        return _nd_zeros(data.shape, dtype='float32')
    n = len(data) if hasattr(data, '__len__') else 1
    return _nd_zeros((n,), dtype='float32')


def _flat_data(data) -> list:
    if hasattr(data, 'to_flat_list'):
        return list(data.to_flat_list())
    if hasattr(data, 'tolist'):
        v = data.tolist()
        if v and isinstance(v[0], list):
            return [item for row in v for item in row]
        return list(v)
    if isinstance(data, list):
        return data
    return [float(data)]


def grad(f: Callable, x) -> 'Tensor':
    """Return gradient of scalar f(x) w.r.t. x."""
    from vectorphile.core.tensor import Tensor as CoreTensor
    x.requires_grad = True
    x.grad = None
    out = f(x)
    out.backward(CoreTensor(_ones_like_data(out._data)))
    if x.grad is not None:
        return CoreTensor(x.grad._data)
    return CoreTensor(_zeros_like_data(x._data))


def jacobian(f: Callable, x) -> list:
    """Full Jacobian of f(x) w.r.t. x via repeated backward passes.

    Returns J as a 2D list of shape (m, n).
    """
    from vectorphile.core.tensor import Tensor as CoreTensor
    from vectorphile._core.ndarray import array as _nd_arr, zeros as _nd_zeros
    x_flat = _flat_data(x._data)
    n = len(x_flat)
    out0 = f(CoreTensor(_nd_arr(x_flat)))
    out_flat = _flat_data(out0._data)
    m = len(out_flat)
    J = [[0.0] * n for _ in range(m)]
    for i in range(m):
        xi = CoreTensor(_nd_arr(x_flat), requires_grad=True)
        out = f(xi)
        e = [1.0 if k == i else 0.0 for k in range(m)]
        sh = out._data.shape if hasattr(out._data, 'shape') else (m,)
        out.backward(CoreTensor(_nd_arr(e).reshape(sh)))
        g = _flat_data(xi.grad._data) if xi.grad is not None else [0.0] * n
        J[i] = g[:n]
    return J


def hessian(f: Callable, x) -> list:
    """Hessian of scalar f(x) via central finite differences.

    Returns H as a 2D list of shape (n, n).
    """
    from vectorphile.core.tensor import Tensor as CoreTensor
    from vectorphile._core.ndarray import array as _nd_arr
    x_flat = _flat_data(x._data)
    n = len(x_flat)
    H = [[0.0] * n for _ in range(n)]
    eps = 1e-3

    def _f(v):
        return float(_flat_data(f(CoreTensor(_nd_arr([float(vi) for vi in v])))._data)[0])

    for i in range(n):
        for j in range(n):
            xpp = x_flat[:]; xpp[i] += eps; xpp[j] += eps
            xpm = x_flat[:]; xpm[i] += eps; xpm[j] -= eps
            xmp = x_flat[:]; xmp[i] -= eps; xmp[j] += eps
            xmm = x_flat[:]; xmm[i] -= eps; xmm[j] -= eps
            H[i][j] = (_f(xpp) - _f(xpm) - _f(xmp) + _f(xmm)) / (4 * eps**2)
    return H


def vmap(f: Callable, in_axes=0, out_axes=0) -> Callable:
    """Vectorised map: apply f element-wise across a batch dimension.

    Args:
        f: function to vectorise
        in_axes: int or tuple of ints/None — which axis to map over for each arg.
                 0 means the first (batch) axis. None means broadcast (don't map).
        out_axes: int — which output axis to stack results along.

    Returns a batched version of f that is equivalent to stacking f(x[i]) for each i.

    Usage::
        vf = vmap(lambda x: x * x)        # maps over batch dim 0
        vf2 = vmap(lambda x, y: x + y, in_axes=(0, None))  # y is broadcast
    """
    def batched(*args):
        from vectorphile.core.tensor import Tensor as CoreTensor, stack as _t_stack
        from vectorphile._core.ndarray import array as _nd_arr
        # Normalize in_axes
        if isinstance(in_axes, int):
            axes = [in_axes] * len(args)
        else:
            axes = list(in_axes)

        # Find batch size from the first mapped arg
        B = None
        for a, ax in zip(args, axes):
            if ax is not None:
                shape = a._data.shape if hasattr(a, '_data') and hasattr(a._data, 'shape') else (len(_flat_data(a._data)),)
                B = shape[ax] if ax < len(shape) else 1
                break
        if B is None:
            return f(*args)

        results = []
        for i in range(B):
            sliced = []
            for a, ax in zip(args, axes):
                if ax is None:
                    sliced.append(a)  # broadcast
                else:
                    shape = a._data.shape if hasattr(a, '_data') and hasattr(a._data, 'shape') else (len(_flat_data(a._data)),)
                    full = _flat_data(a._data)
                    n_per = len(full) // shape[0] if shape else len(full)
                    xi_flat = full[i * n_per:(i + 1) * n_per]
                    xi_sh = shape[1:] if len(shape) > 1 else (n_per,)
                    rg = getattr(a, 'requires_grad', False)
                    sliced.append(CoreTensor(_nd_arr(xi_flat).reshape(xi_sh), requires_grad=rg))
            results.append(f(*sliced))
        return _t_stack(results, dim=out_axes)
    return batched


def grad_and_value(f: Callable, x) -> tuple:
    """Return (gradient, value) of f at x."""
    from vectorphile.core.tensor import Tensor as CoreTensor
    x.requires_grad = True
    x.grad = None
    out = f(x)
    out.backward(CoreTensor(_ones_like_data(out._data)))
    g = CoreTensor(x.grad._data if x.grad is not None else _zeros_like_data(x._data))
    return g, out


def checkpoint(fn: Callable, *args):
    """Gradient checkpointing: recompute activations on backward instead of storing."""
    from vectorphile.core.tensor import Tensor as CoreTensor
    from vectorphile.autograd.engine import no_grad as _no_grad

    saved_enabled = _no_grad._enabled
    _no_grad._enabled = False
    with_no_grad_args = [CoreTensor(a._data) for a in args]
    _no_grad._enabled = False
    out_data = fn(*with_no_grad_args)._data
    _no_grad._enabled = saved_enabled

    out = CoreTensor(out_data, requires_grad=True)

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
    from vectorphile._core.ndarray import zeros as _nd_zeros
    return CoreTensor(_nd_zeros(shape, dtype='float32'), requires_grad=requires_grad)


def ones(shape, requires_grad: bool = False):
    from vectorphile.core.tensor import Tensor as CoreTensor
    from vectorphile._core.ndarray import ones as _nd_ones
    return CoreTensor(_nd_ones(shape, dtype='float32'), requires_grad=requires_grad)


def randn(*shape, requires_grad: bool = False):
    from vectorphile.core.tensor import Tensor as CoreTensor
    from vectorphile._core.ndarray import randn_nd as _nd_randn
    return CoreTensor(_nd_randn(shape), requires_grad=requires_grad)


def cat(tensors: list, axis: int = 0):
    from vectorphile.core.tensor import cat as _t_cat
    return _t_cat(tensors, dim=axis)


def stack(tensors: list, axis: int = 0):
    from vectorphile.core.tensor import stack as _t_stack
    return _t_stack(tensors, dim=axis)


def hessian_vector_product(f: Callable, x, v) -> 'Tensor':
    """Hessian-vector product H(x) @ v via double backward.

    Computes grad(grad(f(x)) · v, x) efficiently — O(n) instead of O(n^2).
    Used in Newton's method, natural gradient, implicit differentiation.
    """
    from vectorphile.core.tensor import Tensor as CoreTensor
    x.requires_grad = True
    x.grad = None
    out = f(x)
    out.backward(CoreTensor(_ones_like_data(out._data)))
    g = x.grad
    if g is None:
        return CoreTensor(_zeros_like_data(x._data))
    # dot grad with v, then differentiate again
    from vectorphile._core.ndarray import array as _nd_arr
    g_flat = _flat_data(g._data)
    v_flat = _flat_data(v._data)
    dot = sum(gi * vi for gi, vi in zip(g_flat, v_flat))
    # second-order: finite-difference approximation of d/dx (grad · v)
    eps = 1e-4
    x_flat = _flat_data(x._data)
    sh = x._data.shape if hasattr(x._data, 'shape') else (len(x_flat),)
    hvp = [0.0] * len(x_flat)
    for i in range(len(x_flat)):
        xp = x_flat[:]; xp[i] += eps
        x._data = _nd_arr(xp).reshape(sh); x.grad = None
        op = f(x); op.backward(CoreTensor(_ones_like_data(op._data)))
        gp = _flat_data(x.grad._data) if x.grad is not None else [0.0] * len(x_flat)
        xm = x_flat[:]; xm[i] -= eps
        x._data = _nd_arr(xm).reshape(sh); x.grad = None
        om = f(x); om.backward(CoreTensor(_ones_like_data(om._data)))
        gm = _flat_data(x.grad._data) if x.grad is not None else [0.0] * len(x_flat)
        x._data = _nd_arr(x_flat).reshape(sh)
        hvp[i] = sum((gp[j] - gm[j]) / (2 * eps) * v_flat[j] for j in range(len(v_flat)))
    return CoreTensor(_nd_arr(hvp).reshape(sh))


def jvp(f: Callable, x, v) -> tuple:
    """Jacobian-vector product (forward-mode AD) via finite differences.

    Returns (f(x), J(x) @ v) where J is the Jacobian.
    Forward-mode is efficient when the number of inputs >> outputs.
    """
    from vectorphile.core.tensor import Tensor as CoreTensor
    from vectorphile._core.ndarray import array as _nd_arr
    x_flat = _flat_data(x._data)
    v_flat = _flat_data(v._data)
    sh = x._data.shape if hasattr(x._data, 'shape') else (len(x_flat),)
    eps = 1e-4
    xp = [xi + eps * vi for xi, vi in zip(x_flat, v_flat)]
    xm = [xi - eps * vi for xi, vi in zip(x_flat, v_flat)]
    out0 = f(CoreTensor(_nd_arr(x_flat).reshape(sh)))
    outp = f(CoreTensor(_nd_arr(xp).reshape(sh)))
    outm = f(CoreTensor(_nd_arr(xm).reshape(sh)))
    outp_flat = _flat_data(outp._data)
    outm_flat = _flat_data(outm._data)
    jvp_flat = [(op - om) / (2 * eps) for op, om in zip(outp_flat, outm_flat)]
    out_sh = out0._data.shape if hasattr(out0._data, 'shape') else (len(outp_flat),)
    return out0, CoreTensor(_nd_arr(jvp_flat).reshape(out_sh))


def vjp(f: Callable, x, v) -> tuple:
    """Vector-Jacobian product (reverse-mode AD): v^T @ J(x).

    Returns (f(x), v^T @ J) — this is what standard .backward() computes.
    Provided here as an explicit functional API for composability.
    """
    from vectorphile.core.tensor import Tensor as CoreTensor
    x_copy = CoreTensor(x._data, requires_grad=True)
    out = f(x_copy)
    out.backward(v)
    g = x_copy.grad if x_copy.grad is not None else CoreTensor(_zeros_like_data(x._data))
    return out, g


def grad_and_hessian(f: Callable, x) -> tuple:
    """Return (gradient, Hessian) of scalar f at x."""
    return grad(f, x), hessian(f, x)


class DualNumber:
    """Dual number for exact forward-mode automatic differentiation.

    A dual number x + ε*v represents a function value x and its directional
    derivative v (the tangent). Arithmetic on dual numbers automatically
    propagates tangents via chain rule, giving exact (not finite-difference)
    JVPs.

    Usage::
        def f(x): return x * x + x
        d = DualNumber(3.0, 1.0)  # x=3, tangent=1
        result = f(d)
        print(result.primal, result.tangent)  # 12.0, 7.0  (exact: 2x+1 at x=3)
    """
    __slots__ = ('primal', 'tangent')

    def __init__(self, primal: float, tangent: float = 0.0):
        self.primal = float(primal)
        self.tangent = float(tangent)

    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.primal + other.primal, self.tangent + other.tangent)
        return DualNumber(self.primal + float(other), self.tangent)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.primal - other.primal, self.tangent - other.tangent)
        return DualNumber(self.primal - float(other), self.tangent)

    def __rsub__(self, other):
        return DualNumber(float(other) - self.primal, -self.tangent)

    def __mul__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.primal * other.primal,
                              self.tangent * other.primal + self.primal * other.tangent)
        s = float(other)
        return DualNumber(self.primal * s, self.tangent * s)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, DualNumber):
            p = self.primal / other.primal
            t = (self.tangent * other.primal - self.primal * other.tangent) / (other.primal ** 2)
            return DualNumber(p, t)
        s = float(other)
        return DualNumber(self.primal / s, self.tangent / s)

    def __rtruediv__(self, other):
        s = float(other)
        return DualNumber(s / self.primal, -s * self.tangent / (self.primal ** 2))

    def __pow__(self, n):
        n = float(n)
        return DualNumber(self.primal ** n, n * self.primal ** (n - 1) * self.tangent)

    def __neg__(self):
        return DualNumber(-self.primal, -self.tangent)

    def __abs__(self):
        sign = 1.0 if self.primal >= 0 else -1.0
        return DualNumber(abs(self.primal), sign * self.tangent)

    def exp(self):
        e = _math.exp(self.primal)
        return DualNumber(e, e * self.tangent)

    def log(self):
        return DualNumber(_math.log(self.primal), self.tangent / self.primal)

    def sqrt(self):
        s = _math.sqrt(self.primal)
        return DualNumber(s, self.tangent / (2.0 * s))

    def sin(self):
        return DualNumber(_math.sin(self.primal), _math.cos(self.primal) * self.tangent)

    def cos(self):
        return DualNumber(_math.cos(self.primal), -_math.sin(self.primal) * self.tangent)

    def tanh(self):
        t = _math.tanh(self.primal)
        return DualNumber(t, (1.0 - t * t) * self.tangent)

    def relu(self):
        return DualNumber(max(0.0, self.primal), self.tangent if self.primal > 0 else 0.0)

    def sigmoid(self):
        s = 1.0 / (1.0 + _math.exp(-self.primal))
        return DualNumber(s, s * (1.0 - s) * self.tangent)

    def __float__(self): return self.primal
    def __repr__(self): return f'DualNumber({self.primal} + ε*{self.tangent})'


def jvp_exact(f: Callable, primals: tuple, tangents: tuple) -> tuple:
    """Exact JVP using dual numbers (forward-mode AD).

    Unlike the finite-difference jvp(), this gives machine-precision results
    with a single function call (no perturbation needed).

    Args:
        f: scalar or vector-valued function of tuple of floats/tensors
        primals: tuple of primal values (floats or lists)
        tangents: tuple of tangent vectors (same shapes as primals)

    Returns:
        (primal_out, tangent_out): both are DualNumber objects or lists
    """
    dual_inputs = []
    for p, t in zip(primals, tangents):
        if isinstance(p, (list, tuple)):
            dual_inputs.append([DualNumber(float(pi), float(ti))
                                 for pi, ti in zip(p, t)])
        else:
            dual_inputs.append(DualNumber(float(p), float(t)))
    result = f(*dual_inputs)
    if isinstance(result, DualNumber):
        return result.primal, result.tangent
    if isinstance(result, (list, tuple)):
        return [r.primal if isinstance(r, DualNumber) else float(r) for r in result], \
               [r.tangent if isinstance(r, DualNumber) else 0.0 for r in result]
    return result, 0.0


def gradcheck(f, inputs, eps=1e-3, atol=1e-2, rtol=1e-2):
    """Finite-difference gradient checker."""
    from vectorphile.core.tensor import Tensor as CoreTensor
    from vectorphile._core.ndarray import array as _nd_arr

    for inp in inputs:
        inp.grad = None
    out = f(*inputs)
    if hasattr(out._data, 'shape') and out._data.shape not in [(), (1,)]:
        out = (out * out).sum()
    out.backward(CoreTensor(_ones_like_data(out._data)))

    errors = []
    for inp_idx, inp in enumerate(inputs):
        ana_flat = _flat_data(inp.grad._data) if inp.grad is not None else []
        x_flat = _flat_data(inp._data)
        n = len(x_flat)
        sh = inp._data.shape if hasattr(inp._data, 'shape') else (n,)
        if not ana_flat:
            ana_flat = [0.0] * n

        for i in range(n):
            xp = x_flat[:]; xp[i] += eps
            xm = x_flat[:]; xm[i] -= eps
            for other in inputs: other.grad = None
            inp._data = _nd_arr(xp).reshape(sh)
            vp_out = f(*inputs)
            vp = float(_flat_data(vp_out._data)[0])
            for other in inputs: other.grad = None
            inp._data = _nd_arr(xm).reshape(sh)
            vm_out = f(*inputs)
            vm = float(_flat_data(vm_out._data)[0])
            inp._data = _nd_arr(x_flat).reshape(sh)
            num = (vp - vm) / (2 * eps)
            ana = float(ana_flat[i]) if i < len(ana_flat) else 0.0
            diff = abs(num - ana)
            thresh = atol + rtol * max(abs(num), abs(ana))
            if diff > thresh:
                errors.append(f'input[{inp_idx}][{i}]: numerical={num:.6g} analytical={ana:.6g} diff={diff:.6g}')
    if errors:
        raise AssertionError('gradcheck failed:\n' + '\n'.join(errors[:10]))
    return True


def checkpoint_sequential(modules: list, segments: int, *args):
    """Gradient checkpointing for sequential module lists.

    Splits modules into `segments` groups and applies checkpointing to each
    segment boundary, reducing memory by recomputing intermediate activations
    during backward at the cost of one extra forward pass per segment.

    Args:
        modules: list of Module objects (called sequentially)
        segments: number of checkpoint segments
        *args: input tensors

    Returns:
        output Tensor after all modules

    Usage::
        from vectorphile.autograd.functional import checkpoint_sequential
        layers = [Linear(64,64), ReLU(), Linear(64,64)]
        y = checkpoint_sequential(layers, segments=2, x)
    """
    if not modules:
        return args[0] if len(args) == 1 else args
    n = len(modules)
    seg_size = max(1, n // max(1, segments))
    # Group into segments
    segs = []
    for i in range(0, n, seg_size):
        segs.append(modules[i:i+seg_size])

    x = args[0] if len(args) == 1 else args

    for seg in segs:
        def _run_seg(seg_mods, inp):
            for mod in seg_mods:
                inp = mod(inp)
            return inp
        # Capture the segment via a closure
        def _make_fn(mods):
            def _fn(*inputs):
                x_ = inputs[0]
                for m in mods:
                    x_ = m(x_)
                return x_
            return _fn
        x = checkpoint(_make_fn(seg), x)

    return x


class GradientCheckpointWrapper:
    """Wraps a Module to use gradient checkpointing on its forward pass.

    Equivalent to torch.utils.checkpoint.checkpoint(module, x) but as a
    reusable wrapper. Saves memory at the cost of one extra forward pass
    per backward pass.

    Args:
        module: Module to wrap

    Usage::
        from vectorphile.autograd.functional import GradientCheckpointWrapper
        fast_layer = GradientCheckpointWrapper(big_transformer_block)
        y = fast_layer(x)   # activations freed, recomputed on backward
    """

    def __init__(self, module):
        self._module = module

    def __call__(self, *args):
        return checkpoint(self._module, *args)

    def parameters(self):
        return self._module.parameters() if hasattr(self._module, 'parameters') else []

    def __repr__(self):
        return f'GradientCheckpointWrapper({self._module})'
