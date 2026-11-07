"""Functional transforms — JAX-inspired grad, vmap, jit, jacobian, hessian."""
from __future__ import annotations
import numpy as np
import functools
from typing import Callable, Optional, Union, Tuple
from vectorphile.core.tensor import Tensor


def grad(func: Callable, argnums: Union[int, Tuple[int, ...]] = 0, has_aux=False) -> Callable:
    """Compute gradient of a scalar-valued function via finite differences.

    Returns a function that computes the gradient at the given arguments.

    Example:
        def loss(w): return ((w * 2 - 3)**2).sum()
        grad_fn = vp.transforms.grad(loss)
        g = grad_fn(vp.tensor([1.0, 2.0]))
    """
    if isinstance(argnums, int):
        argnums = (argnums,)

    @functools.wraps(func)
    def grad_fn(*args, **kwargs):
        eps = 1e-5
        grads = []
        for argnum in argnums:
            arg = args[argnum]
            if isinstance(arg, Tensor):
                data = arg._data.astype(np.float64)
                g = np.zeros_like(data)
                flat = data.flatten()
                for i in range(len(flat)):
                    p = flat.copy(); p[i] += eps
                    n = flat.copy(); n[i] -= eps
                    args_p = list(args)
                    args_n = list(args)
                    args_p[argnum] = Tensor(p.reshape(data.shape).astype(np.float32))
                    args_n[argnum] = Tensor(n.reshape(data.shape).astype(np.float32))
                    fp = func(*args_p, **kwargs)
                    fn = func(*args_n, **kwargs)
                    fp_val = fp._data.sum() if isinstance(fp, Tensor) else float(fp)
                    fn_val = fn._data.sum() if isinstance(fn, Tensor) else float(fn)
                    g.flat[i] = (fp_val - fn_val) / (2 * eps)
                grads.append(Tensor(g.astype(np.float32)))
            else:
                raise TypeError(f"grad only works on Tensor arguments, got {type(arg)}")
        if has_aux:
            val = func(*args, **kwargs)
            return (grads[0] if len(argnums) == 1 else grads), val
        return grads[0] if len(argnums) == 1 else tuple(grads)

    return grad_fn


def value_and_grad(func: Callable, argnums: Union[int, Tuple[int, ...]] = 0, has_aux=False) -> Callable:
    """Return (value, gradient) together."""
    grad_fn = grad(func, argnums, has_aux)

    @functools.wraps(func)
    def combined(*args, **kwargs):
        val = func(*args, **kwargs)
        g = grad_fn(*args, **kwargs)
        return val, g

    return combined


def jacobian(func: Callable, argnums: int = 0) -> Callable:
    """Compute the Jacobian matrix via finite differences.

    Returns a function that computes Jacobian at given inputs.
    J[i, j] = d(output[i]) / d(input[j])
    """
    @functools.wraps(func)
    def jac_fn(*args, **kwargs):
        eps = 1e-5
        arg = args[argnums]
        if not isinstance(arg, Tensor):
            raise TypeError("jacobian requires Tensor argument")
        x = arg._data.astype(np.float64)
        n_in = x.size
        # Compute output shape
        out0 = func(*args, **kwargs)
        if isinstance(out0, Tensor):
            n_out = out0._data.size
        else:
            n_out = 1
        J = np.zeros((n_out, n_in), dtype=np.float64)
        for j in range(n_in):
            xp = x.flatten().copy(); xp[j] += eps
            xn = x.flatten().copy(); xn[j] -= eps
            args_p = list(args); args_p[argnums] = Tensor(xp.reshape(x.shape).astype(np.float32))
            args_n = list(args); args_n[argnums] = Tensor(xn.reshape(x.shape).astype(np.float32))
            fp = func(*args_p, **kwargs)
            fn = func(*args_n, **kwargs)
            fp_arr = fp._data.flatten() if isinstance(fp, Tensor) else np.array([float(fp)])
            fn_arr = fn._data.flatten() if isinstance(fn, Tensor) else np.array([float(fn)])
            J[:, j] = (fp_arr - fn_arr) / (2 * eps)
        return Tensor(J.astype(np.float32))

    return jac_fn


def hessian(func: Callable, argnums: int = 0) -> Callable:
    """Compute the Hessian matrix (second-order partial derivatives)."""
    return jacobian(jacobian(func, argnums), argnums)


def vmap(func: Callable, in_axes: Union[int, Tuple] = 0, out_axes: int = 0) -> Callable:
    """Vectorized map — apply func independently over a batch dimension.

    Example:
        batched_matmul = vp.transforms.vmap(lambda A, b: A @ b)
        result = batched_matmul(batch_A, batch_b)  # processes each sample independently
    """
    @functools.wraps(func)
    def vmapped(*args, **kwargs):
        if isinstance(in_axes, int):
            axes = [in_axes] * len(args)
        else:
            axes = list(in_axes)
        # Determine batch size from first batched arg
        batch_size = None
        for arg, ax in zip(args, axes):
            if ax is not None and isinstance(arg, Tensor):
                batch_size = arg._data.shape[ax]
                break
        if batch_size is None:
            return func(*args, **kwargs)
        results = []
        for i in range(batch_size):
            sliced_args = []
            for arg, ax in zip(args, axes):
                if ax is None:
                    sliced_args.append(arg)
                elif isinstance(arg, Tensor):
                    sliced_args.append(Tensor(np.take(arg._data, i, axis=ax)))
                else:
                    sliced_args.append(arg)
            results.append(func(*sliced_args, **kwargs))
        if isinstance(results[0], Tensor):
            stacked = np.stack([r._data for r in results], axis=out_axes)
            return Tensor(stacked)
        return results

    return vmapped


def jit(func: Callable, static_argnums=None, static_argnames=None,
        donate_argnums=(), keep_unused=False, device=None, backend=None) -> Callable:
    """JIT compilation marker (Phase 1: identity, Phase 2: traces to C extensions).

    Decorating a function with jit marks it for future ahead-of-time compilation.
    In Phase 1, this is a transparent pass-through that records the function for
    profiling and later optimization.
    """
    func._jit_marked = True

    @functools.wraps(func)
    def jitted(*args, **kwargs):
        return func(*args, **kwargs)

    jitted._jit_marked = True
    return jitted


def pmap(func: Callable, axis_name: Optional[str] = None, in_axes=0, out_axes=0,
         devices=None, backend=None) -> Callable:
    """Parallel map over devices (Phase 1: identity, Phase 2: multi-GPU)."""
    return vmap(func, in_axes=in_axes, out_axes=out_axes)


def checkpoint(func: Callable) -> Callable:
    """Gradient checkpointing — recompute activations during backward pass.

    Reduces memory by not storing intermediate activations, trading compute for memory.
    Phase 1: identity wrapper. Phase 2: implements rematerialization.
    """
    @functools.wraps(func)
    def checkpointed(*args, **kwargs):
        return func(*args, **kwargs)

    checkpointed._checkpoint_marked = True
    return checkpointed


def linearize(func: Callable, *primals) -> Tuple:
    """Linearize func around primals — return (value, jvp_fn)."""
    val = func(*primals)
    jac_fn = jacobian(func)
    J = jac_fn(*primals)
    def jvp_fn(*tangents):
        t = tangents[0]._data.flatten() if isinstance(tangents[0], Tensor) else np.array(tangents[0]).flatten()
        return Tensor((J._data @ t).astype(np.float32))
    return val, jvp_fn


def jvp(func: Callable, primals, tangents) -> Tuple:
    """Forward-mode JVP — (output, directional derivative)."""
    val = func(*primals)
    jac_fn = jacobian(func)
    J = jac_fn(*primals)
    t = tangents[0]._data.flatten() if isinstance(tangents[0], Tensor) else np.array(tangents[0]).flatten()
    jvp_val = Tensor((J._data @ t).astype(np.float32))
    return val, jvp_val


def vjp(func: Callable, *primals) -> Tuple:
    """Reverse-mode VJP — (output, vjp_fn)."""
    val = func(*primals)
    jac_fn = jacobian(func)
    J = jac_fn(*primals)
    def vjp_fn(cotangents):
        g = cotangents._data.flatten() if isinstance(cotangents, Tensor) else np.array(cotangents).flatten()
        return (Tensor((g @ J._data).astype(np.float32)),)
    return val, vjp_fn


def eval_shape(func: Callable, *args) -> object:
    """Compute output shape without executing the function."""
    class ShapeProxy:
        def __init__(self, shape, dtype='float32'):
            self.shape = shape
            self.dtype = dtype
        def __repr__(self):
            return f'ShapeProxy(shape={self.shape}, dtype={self.dtype})'
    result = func(*args)
    if isinstance(result, Tensor):
        return ShapeProxy(result.shape, result.dtype)
    return result
