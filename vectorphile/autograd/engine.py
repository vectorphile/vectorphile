"""Reverse-mode automatic differentiation engine."""
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorphile.core.tensor import Tensor


def _topo_sort(root: 'Tensor'):
    visited, order = set(), []
    def visit(node):
        if id(node) not in visited:
            visited.add(id(node))
            if node.grad_fn is not None:
                for inp in node.grad_fn.inputs:
                    if hasattr(inp, 'grad_fn'):
                        visit(inp)
            order.append(node)
    visit(root)
    return reversed(order)


def backward(root: 'Tensor', gradient: 'Tensor'):
    from vectorphile.core.tensor import Tensor
    root.grad = gradient
    for node in _topo_sort(root):
        if node.grad_fn is None:
            continue
        grads = node.grad_fn.backward(node.grad)
        for inp, g in zip(node.grad_fn.inputs, grads):
            if not isinstance(inp, Tensor) or not inp.requires_grad:
                continue
            if g is None:
                continue
            if inp.grad is None:
                inp.grad = Tensor(np.zeros_like(inp._data))
            # Sum gradient over broadcast dimensions
            g_data = _sum_to_shape(g._data if isinstance(g, Tensor) else g, inp._data.shape)
            inp.grad._data += g_data


def _sum_to_shape(g: np.ndarray, shape) -> np.ndarray:
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, (gs, s) in enumerate(zip(g.shape, shape)):
        if s == 1 and gs != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class Function:
    def __init__(self, *inputs):
        self.inputs = inputs

    def backward(self, grad):
        raise NotImplementedError


class AddBackward(Function):
    def backward(self, grad):
        return grad, grad


class SubBackward(Function):
    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        neg = Tensor(-grad._data, device=grad.device)
        return grad, neg


class MulBackward(Function):
    def __init__(self, a, b):
        super().__init__(a, b)
        self.a, self.b = a, b

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        a_data = self.a._data if hasattr(self.a, '_data') else np.array(self.a)
        b_data = self.b._data if hasattr(self.b, '_data') else np.array(self.b)
        return Tensor(grad._data * b_data), Tensor(grad._data * a_data)


class DivBackward(Function):
    def __init__(self, a, b):
        super().__init__(a, b)
        self.a, self.b = a, b

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        a_data = self.a._data if hasattr(self.a, '_data') else np.array(self.a)
        b_data = self.b._data if hasattr(self.b, '_data') else np.array(self.b)
        da = grad._data / b_data
        db = -grad._data * a_data / (b_data ** 2)
        return Tensor(da), Tensor(db)


class MatMulBackward(Function):
    def __init__(self, a, b):
        super().__init__(a, b)
        self.a, self.b = a, b

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        return Tensor(np.matmul(grad._data, self.b._data.T)), Tensor(np.matmul(self.a._data.T, grad._data))


class ExpBackward(Function):
    def __init__(self, inp, out):
        super().__init__(inp)
        self.out = out

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        return (Tensor(grad._data * self.out._data),)


class LogBackward(Function):
    def __init__(self, inp):
        super().__init__(inp)
        self.inp = inp

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        return (Tensor(grad._data / (self.inp._data + 1e-12)),)


class SqrtBackward(Function):
    def __init__(self, inp, out):
        super().__init__(inp)
        self.out = out

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        return (Tensor(0.5 * grad._data / (self.out._data + 1e-12)),)


class SumBackward(Function):
    def __init__(self, inp, dim, keepdim):
        super().__init__(inp)
        self.inp = inp
        self.dim = dim
        self.keepdim = keepdim

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        g = grad._data
        if self.dim is not None and not self.keepdim:
            g = np.expand_dims(g, axis=self.dim)
        return (Tensor(np.broadcast_to(g, self.inp._data.shape).copy()),)


class MeanBackward(Function):
    def __init__(self, inp, dim, keepdim):
        super().__init__(inp)
        self.inp = inp
        self.dim = dim
        self.keepdim = keepdim

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        g = grad._data
        if self.dim is not None and not self.keepdim:
            g = np.expand_dims(g, axis=self.dim)
        n = self.inp._data.size if self.dim is None else self.inp._data.shape[self.dim]
        return (Tensor(np.broadcast_to(g / n, self.inp._data.shape).copy()),)


class ReLUBackward(Function):
    def __init__(self, inp):
        super().__init__(inp)
        self.inp = inp

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        return (Tensor(grad._data * (self.inp._data > 0).astype(np.float32)),)


class SigmoidBackward(Function):
    def __init__(self, inp, out):
        super().__init__(inp)
        self.out = out

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        s = self.out._data
        return (Tensor(grad._data * s * (1 - s)),)


class TanhBackward(Function):
    def __init__(self, inp, out):
        super().__init__(inp)
        self.out = out

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        return (Tensor(grad._data * (1 - self.out._data ** 2)),)


class no_grad:
    """Context manager to disable gradient tracking."""
    _enabled = True

    def __enter__(self):
        no_grad._enabled = False

    def __exit__(self, *args):
        no_grad._enabled = True
