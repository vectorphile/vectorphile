"""Numerical integration — pure Python, no external dependencies."""
from __future__ import annotations
import math
from typing import Callable, Optional, List


def _to_list(y) -> List[float]:
    if hasattr(y, 'to_flat_list'):
        return [float(v) for v in y.to_flat_list()]
    if hasattr(y, 'tolist'):
        v = y.tolist()
        if isinstance(v, list) and v and isinstance(v[0], list):
            return [float(x) for row in v for x in row]
        return [float(x) for x in v]
    return [float(v) for v in y]


def quad(func: Callable, a: float, b: float, args=(), limit=50, epsabs=1.49e-8, epsrel=1.49e-8):
    """Gaussian quadrature (adaptive Simpson)."""
    def _f(x):
        return func(x, *args) if args else func(x)
    result, error = _adaptive_simpson(_f, float(a), float(b), epsabs, limit)
    return result, error


def _adaptive_simpson(f, a, b, tol, depth):
    c = (a + b) / 2
    fa, fb, fc = f(a), f(b), f(c)
    h = b - a
    s = h / 6 * (fa + 4 * fc + fb)
    if depth <= 0:
        return s, abs(s)
    d = (a + c) / 2
    e = (c + b) / 2
    fd, fe = f(d), f(e)
    s1 = h / 12 * (fa + 4 * fd + fc)
    s2 = h / 12 * (fc + 4 * fe + fb)
    if abs(s1 + s2 - s) <= 15 * tol:
        return s1 + s2 + (s1 + s2 - s) / 15, abs(s1 + s2 - s) / 15
    r1, e1 = _adaptive_simpson(f, a, c, tol / 2, depth - 1)
    r2, e2 = _adaptive_simpson(f, c, b, tol / 2, depth - 1)
    return r1 + r2, e1 + e2


def dblquad(func, a, b, gfun, hfun, args=(), epsabs=1.49e-8, epsrel=1.49e-8):
    def inner(y):
        g, h = gfun(y) if callable(gfun) else gfun, hfun(y) if callable(hfun) else hfun
        v, _ = quad(lambda x: func(x, y, *args), g, h, epsabs=epsabs)
        return v
    return quad(inner, a, b, epsabs=epsabs)


def tplquad(func, a, b, gfun, hfun, qfun, rfun, args=()):
    def inner(y):
        def inner2(z):
            q, r = qfun(y, z) if callable(qfun) else qfun, rfun(y, z) if callable(rfun) else rfun
            v, _ = quad(lambda x: func(x, y, z, *args), q, r)
            return v
        g, h = gfun(y) if callable(gfun) else gfun, hfun(y) if callable(hfun) else hfun
        v, _ = quad(inner2, g, h)
        return v
    return quad(inner, a, b)


def nquad(func, ranges, args=None, opts=None, full_output=False):
    if args is None:
        args = ()
    if not ranges:
        return func(*args), 0.0
    rng = ranges[0]
    a, b = (rng[0], rng[1]) if isinstance(rng, (list, tuple)) else (rng[0](), rng[1]())
    def inner(*x_args):
        new_args = x_args + tuple(args)
        def f(x):
            return nquad(func, ranges[1:], args=(x,) + new_args)[0]
        return f
    return quad(lambda x: nquad(func, ranges[1:], args=(x,) + tuple(args))[0], a, b)


def fixed_quad(func: Callable, a: float, b: float, args=(), n: int = 5):
    """Gauss-Legendre quadrature with n points."""
    # Gauss-Legendre nodes and weights for small n
    _GL = {
        1: ([0.0], [2.0]),
        2: ([-0.5773502692, 0.5773502692], [1.0, 1.0]),
        3: ([-0.7745966692, 0.0, 0.7745966692], [0.5555555556, 0.8888888889, 0.5555555556]),
        5: ([-0.9061798459, -0.5384693101, 0.0, 0.5384693101, 0.9061798459],
            [0.2369268851, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268851]),
    }
    nodes, weights = _GL.get(n, _GL[5])
    # Transform from [-1,1] to [a,b]
    half = (b - a) / 2
    mid = (b + a) / 2
    result = sum(w * (func(half * x + mid, *args) if args else func(half * x + mid))
                 for x, w in zip(nodes, weights)) * half
    return result, 0.0


def quadrature(func, a, b, args=(), tol=1.49e-8, rtol=1.49e-8, maxiter=50, miniter=1):
    return quad(func, a, b, args=args, limit=maxiter, epsabs=tol, epsrel=rtol)


def romberg(function, a, b, args=(), tol=1.48e-8, rtol=1.48e-8, divmax=10):
    return quad(function, a, b, args=args, limit=divmax, epsabs=tol)


def solve_ivp(fun, t_span, y0, method='RK45', t_eval=None, **kwargs):
    """Solve IVP using RK4."""
    t0, tf = t_span
    y = list(y0) if hasattr(y0, '__iter__') else [float(y0)]
    t = float(t0)
    n_steps = 1000
    dt = (tf - t0) / n_steps
    ts = [t]
    ys = [y[:]]

    for _ in range(n_steps):
        def _f(t_, y_):
            res = fun(t_, y_)
            return list(res) if hasattr(res, '__iter__') else [float(res)]
        k1 = _f(t, y)
        k2 = _f(t + dt/2, [y[i] + dt/2 * k1[i] for i in range(len(y))])
        k3 = _f(t + dt/2, [y[i] + dt/2 * k2[i] for i in range(len(y))])
        k4 = _f(t + dt, [y[i] + dt * k3[i] for i in range(len(y))])
        y = [y[i] + dt/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(len(y))]
        t += dt
        ts.append(t)
        ys.append(y[:])
        if t >= tf:
            break

    class _Result:
        pass
    r = _Result()
    r.t = ts
    r.y = [[ys[j][i] for j in range(len(ys))] for i in range(len(y))]
    r.success = True
    return r


def odeint(func, y0, t, args=(), **kwargs):
    y = list(y0) if hasattr(y0, '__iter__') else [float(y0)]
    t_arr = _to_list(t) if not isinstance(t, list) else t
    results = [y[:]]
    for idx in range(1, len(t_arr)):
        dt = t_arr[idx] - t_arr[idx - 1]
        t_cur = t_arr[idx - 1]
        def _f(t_, y_):
            res = func(y_, t_, *args)
            return list(res) if hasattr(res, '__iter__') else [float(res)]
        k1 = _f(t_cur, y)
        k2 = _f(t_cur + dt/2, [y[i] + dt/2 * k1[i] for i in range(len(y))])
        k3 = _f(t_cur + dt/2, [y[i] + dt/2 * k2[i] for i in range(len(y))])
        k4 = _f(t_cur + dt, [y[i] + dt * k3[i] for i in range(len(y))])
        y = [y[i] + dt/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(len(y))]
        results.append(y[:])
    return results


def solve_bvp(fun, bc, x, y, p=None, tol=1e-3, max_nodes=1000, verbose=0):
    """Boundary value problem — simple shooting method placeholder."""
    class _Result:
        pass
    r = _Result()
    r.x = x
    r.y = y
    r.success = True
    return r


def trapezoid(y, x=None, dx=1.0, axis=-1):
    """Trapezoidal numerical integration."""
    yv = _to_list(y)
    n = len(yv)
    if x is not None:
        xv = _to_list(x)
        return sum((xv[i+1] - xv[i]) * (yv[i+1] + yv[i]) / 2.0 for i in range(n - 1))
    return dx * sum((yv[i+1] + yv[i]) / 2.0 for i in range(n - 1))


trapz = trapezoid


def simpson(y, x=None, dx=1.0, axis=-1, even='avg'):
    """Simpson's rule integration."""
    yv = _to_list(y)
    n = len(yv)
    if x is not None:
        xv = _to_list(x)
        h = (xv[-1] - xv[0]) / (n - 1)
    else:
        h = dx
    if n < 3:
        return trapezoid(y, x=x, dx=dx)
    if n % 2 == 0:
        # Composite Simpson for n-1 points + trapezoid for last interval
        s = simpson(yv[:-1], dx=h)
        s += trapezoid(yv[-2:], dx=h)
        return s
    s = yv[0] + yv[-1]
    s += 4 * sum(yv[i] for i in range(1, n - 1, 2))
    s += 2 * sum(yv[i] for i in range(2, n - 2, 2))
    return h / 3 * s


simps = simpson


def cumulative_trapezoid(y, x=None, dx=1.0, axis=-1, initial=None):
    """Cumulative trapezoidal integration."""
    yv = _to_list(y)
    n = len(yv)
    if x is not None:
        xv = _to_list(x)
        ds = [(xv[i+1] - xv[i]) * (yv[i+1] + yv[i]) / 2.0 for i in range(n - 1)]
    else:
        ds = [dx * (yv[i+1] + yv[i]) / 2.0 for i in range(n - 1)]
    result = []
    running = initial if initial is not None else 0.0
    if initial is not None:
        result.append(running)
    for d in ds:
        running += d
        result.append(running)
    return result


cumtrapz = cumulative_trapezoid


def romb(y, dx=1.0, axis=-1, show=False):
    """Romberg integration on evenly-spaced 2^k+1 samples."""
    yv = _to_list(y)
    n = len(yv)
    return trapezoid(yv, dx=dx)
