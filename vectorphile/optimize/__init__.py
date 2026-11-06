"""Numerical optimization routines."""
import numpy as np
from typing import Callable, Optional, Tuple
from vectorphile.core.tensor import Tensor


def minimize(fun: Callable, x0, method='bfgs', jac=None, tol=1e-6, options=None):
    """Minimize a scalar function. Wraps scipy.optimize.minimize when available."""
    try:
        from scipy.optimize import minimize as sp_minimize
        x0_arr = x0._data.flatten() if isinstance(x0, Tensor) else np.array(x0).flatten()
        def fn(x):
            t = Tensor(x.reshape(x0_arr.shape) if isinstance(x0, Tensor) else x, requires_grad=True)
            result = fun(t)
            return float(result._data) if isinstance(result, Tensor) else float(result)
        result = sp_minimize(fn, x0_arr, method=method.upper(), jac=jac, tol=tol, options=options)
        result.x = Tensor(result.x.reshape(x0._data.shape if isinstance(x0, Tensor) else x0.shape))
        return result
    except ImportError:
        return _gradient_descent(fun, x0, tol=tol, options=options or {})


def _gradient_descent(fun: Callable, x0, lr=0.01, tol=1e-6, max_iter=1000, options=None):
    if options:
        lr = options.get('lr', lr)
        max_iter = options.get('maxiter', max_iter)
    if isinstance(x0, Tensor):
        x = Tensor(x0._data.copy(), requires_grad=True)
    else:
        x = Tensor(np.array(x0, dtype=float), requires_grad=True)
    for i in range(max_iter):
        if x.grad is not None:
            x.grad = None
        result = fun(x)
        result.backward()
        if x.grad is None:
            break
        grad_norm = float(np.linalg.norm(x.grad._data))
        if grad_norm < tol:
            break
        x._data -= lr * x.grad._data

    class Result:
        pass
    r = Result()
    r.x = x
    r.fun = float(fun(x)._data)
    r.success = True
    return r


def root(fun: Callable, x0, method='hybr', tol=1e-6):
    """Find root of a function."""
    try:
        from scipy.optimize import root as sp_root
        x0_arr = x0._data.flatten() if isinstance(x0, Tensor) else np.array(x0).flatten()
        def fn(x):
            t = Tensor(x)
            result = fun(t)
            return result._data.flatten() if isinstance(result, Tensor) else np.array(result).flatten()
        return sp_root(fn, x0_arr, method=method, tol=tol)
    except ImportError:
        raise ImportError("root requires scipy: pip install scipy")


def linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, method='highs'):
    """Linear programming."""
    try:
        from scipy.optimize import linprog as sp_linprog
        c_arr = c._data if isinstance(c, Tensor) else np.array(c)
        return sp_linprog(c_arr, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=method)
    except ImportError:
        raise ImportError("linprog requires scipy: pip install scipy")


def curve_fit(f: Callable, xdata, ydata, p0=None):
    """Curve fitting."""
    try:
        from scipy.optimize import curve_fit as sp_curve_fit
        xd = xdata._data if isinstance(xdata, Tensor) else np.array(xdata)
        yd = ydata._data if isinstance(ydata, Tensor) else np.array(ydata)
        popt, pcov = sp_curve_fit(f, xd, yd, p0=p0)
        return Tensor(popt.astype(np.float32)), Tensor(pcov.astype(np.float32))
    except ImportError:
        raise ImportError("curve_fit requires scipy: pip install scipy")


def differential_evolution(fun: Callable, bounds, maxiter=1000, tol=0.01, seed=None):
    """Global optimization via differential evolution."""
    try:
        from scipy.optimize import differential_evolution as sp_de
        def fn(x):
            result = fun(Tensor(x))
            return float(result._data) if isinstance(result, Tensor) else float(result)
        return sp_de(fn, bounds, maxiter=maxiter, tol=tol, seed=seed)
    except ImportError:
        raise ImportError("differential_evolution requires scipy: pip install scipy")
