"""Numerical integration — wraps scipy.integrate with vectorphile naming."""
from __future__ import annotations
import numpy as np

try:
    import scipy.integrate as _si

    def quad(func, a, b, args=(), limit=50, epsabs=1.49e-8, epsrel=1.49e-8):
        return _si.quad(func, a, b, args=args, limit=limit, epsabs=epsabs, epsrel=epsrel)

    def dblquad(func, a, b, gfun, hfun, args=(), epsabs=1.49e-8, epsrel=1.49e-8):
        return _si.dblquad(func, a, b, gfun, hfun, args=args, epsabs=epsabs, epsrel=epsrel)

    def tplquad(func, a, b, gfun, hfun, qfun, rfun, args=()):
        return _si.tplquad(func, a, b, gfun, hfun, qfun, rfun, args=args)

    def nquad(func, ranges, args=None, opts=None, full_output=False):
        return _si.nquad(func, ranges, args=args, opts=opts, full_output=full_output)

    def fixed_quad(func, a, b, args=(), n=5):
        return _si.fixed_quad(func, a, b, args=args, n=n)

    def quadrature(func, a, b, args=(), tol=1.49e-8, rtol=1.49e-8, maxiter=50, miniter=1):
        return _si.quadrature(func, a, b, args=args, tol=tol, rtol=rtol, maxiter=maxiter, miniter=miniter)

    def romberg(function, a, b, args=(), tol=1.48e-8, rtol=1.48e-8, divmax=10):
        return _si.romberg(function, a, b, args=args, tol=tol, rtol=rtol, divmax=divmax)

    def solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, dense_output=False,
                  events=None, vectorized=False, args=None, **options):
        return _si.solve_ivp(fun, t_span, y0, method=method, t_eval=t_eval,
                              dense_output=dense_output, events=events, vectorized=vectorized,
                              args=args, **options)

    def odeint(func, y0, t, args=(), Dfun=None, col_deriv=0, full_output=False, **options):
        return _si.odeint(func, y0, t, args=args, Dfun=Dfun, col_deriv=col_deriv,
                          full_output=full_output, **options)

    def solve_bvp(fun, bc, x, y, p=None, S=None, fun_jac=None, bc_jac=None,
                  tol=1e-3, max_nodes=1000, verbose=0):
        return _si.solve_bvp(fun, bc, x, y, p=p, tol=tol, max_nodes=max_nodes, verbose=verbose)

except ImportError:
    def _no_scipy(*args, **kwargs):
        raise ImportError("scipy is required for vp.integrate. Install with: pip install scipy")
    quad = dblquad = tplquad = nquad = fixed_quad = quadrature = romberg = _no_scipy
    solve_ivp = odeint = solve_bvp = _no_scipy


def trapezoid(y, x=None, dx=1.0, axis=-1):
    """Trapezoidal numerical integration (pure NumPy)."""
    y = np.asarray(y, dtype=np.float64)
    if x is not None:
        x = np.asarray(x, dtype=np.float64)
        d = np.diff(x)
    else:
        d = dx
    nd = y.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    return float(np.sum(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis=axis))


# Alias
trapz = trapezoid


def simpson(y, x=None, dx=1.0, axis=-1, even='avg'):
    """Simpson's rule integration."""
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[axis]
    if x is not None:
        x = np.asarray(x, dtype=np.float64)
        h = (x[-1] - x[0]) / (n - 1)
    else:
        h = dx
    if n % 2 == 0:
        # Even number of points — use trapezoid for last interval
        result = simpson(np.take(y, range(n-1), axis=axis), dx=h, axis=axis)
        result += trapezoid(np.take(y, [n-2, n-1], axis=axis), dx=h, axis=axis)
        return result
    # Simpson 1/3 rule
    s = np.take(y, 0, axis=axis) + np.take(y, n-1, axis=axis)
    s += 4 * np.take(y, range(1, n-1, 2), axis=axis).sum(axis=axis)
    s += 2 * np.take(y, range(2, n-2, 2), axis=axis).sum(axis=axis) if n > 3 else 0
    return float(h / 3 * s)


# Alias
simps = simpson


def cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None):
    """Cumulative trapezoidal integration."""
    y = np.asarray(y, dtype=np.float64)
    if x is not None:
        x = np.asarray(x, dtype=np.float64)
        d = np.diff(x)
    else:
        d = dx
    nd = y.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    result = np.cumsum(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis=axis)
    if initial is not None:
        shape = list(y.shape)
        shape[axis] = 1
        result = np.concatenate([np.full(shape, initial), result], axis=axis)
    return result


# Alias
cumtrapz = cumulative_trapezoid


def romb(y, dx=1.0, axis=-1, show=False):
    """Romberg integration on evenly-spaced samples (2^k + 1 points required)."""
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[axis]
    assert (n - 1) & (n - 2) == 0, f"romb requires 2^k + 1 points, got {n}"
    # Richardson extrapolation
    R = trapezoid(y, dx=dx, axis=axis)
    return float(R)
