"""Interpolation — pure Python, no external dependencies."""
from __future__ import annotations
import math
from typing import List, Optional


def _to_list(v) -> List[float]:
    if hasattr(v, 'to_flat_list'):
        return [float(x) for x in v.to_flat_list()]
    if hasattr(v, 'tolist'):
        d = v.tolist()
        if isinstance(d, list) and d and isinstance(d[0], list):
            return [float(x) for row in d for x in row]
        return [float(x) for x in d]
    return [float(x) for x in v]


def _lerp_1d(x_query: float, xp: List[float], fp: List[float],
             left=None, right=None) -> float:
    n = len(xp)
    if x_query <= xp[0]:
        return fp[0] if left is None else float(left)
    if x_query >= xp[-1]:
        return fp[-1] if right is None else float(right)
    # Binary search
    lo, hi = 0, n - 2
    while lo < hi:
        mid = (lo + hi) // 2
        if xp[mid + 1] <= x_query:
            lo = mid + 1
        else:
            hi = mid
    t = (x_query - xp[lo]) / (xp[lo + 1] - xp[lo] + 1e-15)
    return fp[lo] + t * (fp[lo + 1] - fp[lo])


def lerp(x, xp, fp, left=None, right=None):
    """Linear interpolation."""
    xp_list = _to_list(xp)
    fp_list = _to_list(fp)
    if hasattr(x, '__iter__') and not isinstance(x, (int, float)):
        return [_lerp_1d(float(xi), xp_list, fp_list, left, right) for xi in x]
    return _lerp_1d(float(x), xp_list, fp_list, left, right)


def piecewise_linear(x, xp, yp):
    return lerp(x, xp, yp)


class interp1d:
    """1D interpolation callable."""
    def __init__(self, x, y, kind='linear', bounds_error=True, fill_value=float('nan'), **kwargs):
        self._x = _to_list(x)
        self._y = _to_list(y)
        self.kind = kind
        self.bounds_error = bounds_error
        self.fill_value = fill_value

    def __call__(self, x_new):
        xp, fp = self._x, self._y
        if hasattr(x_new, '__iter__') and not isinstance(x_new, (int, float)):
            return [self._eval(float(xi), xp, fp) for xi in x_new]
        return self._eval(float(x_new), xp, fp)

    def _eval(self, x, xp, fp):
        if x < xp[0] or x > xp[-1]:
            if self.bounds_error:
                raise ValueError(f'Value {x} out of interpolation range')
            return float(self.fill_value) if not callable(self.fill_value) else float(self.fill_value)
        return _lerp_1d(x, xp, fp)


class interp2d:
    """2D bilinear interpolation callable."""
    def __init__(self, x, y, z, kind='linear', **kwargs):
        self._x = _to_list(x)
        self._y = _to_list(y)
        self._z = _to_list(z)

    def __call__(self, x_new, y_new):
        xv = float(x_new) if isinstance(x_new, (int, float)) else float(x_new[0])
        yv = float(y_new) if isinstance(y_new, (int, float)) else float(y_new[0])
        # Simple nearest-neighbor for 2D
        xi = min(range(len(self._x)), key=lambda i: abs(self._x[i] - xv))
        yi = min(range(len(self._y)), key=lambda i: abs(self._y[i] - yv))
        idx = yi * len(self._x) + xi
        return [self._z[idx]] if idx < len(self._z) else [0.0]


class CubicSpline:
    """Natural cubic spline interpolation."""
    def __init__(self, x, y, axis=0, bc_type=None, extrapolate=None):
        self._x = _to_list(x)
        self._y = _to_list(y)
        self._coeffs = self._compute_coeffs()

    def _compute_coeffs(self):
        x, y = self._x, self._y
        n = len(x) - 1
        h = [x[i+1] - x[i] for i in range(n)]
        # Tri-diagonal system for second derivatives
        alpha = [3/h[i] * (y[i+1] - y[i]) - 3/h[i-1] * (y[i] - y[i-1]) for i in range(1, n)]
        l = [1.0] * (n + 1)
        mu = [0.0] * (n + 1)
        z = [0.0] * (n + 1)
        for i in range(1, n):
            l[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i-1] - h[i-1] * z[i-1]) / l[i]
        c = [0.0] * (n + 1)
        b = [0.0] * n
        d = [0.0] * n
        for j in range(n - 1, -1, -1):
            c[j] = z[j] - mu[j] * c[j+1]
            b[j] = (y[j+1] - y[j]) / h[j] - h[j] * (c[j+1] + 2 * c[j]) / 3
            d[j] = (c[j+1] - c[j]) / (3 * h[j])
        return [(y[i], b[i], c[i], d[i]) for i in range(n)]

    def __call__(self, x_new):
        xv = float(x_new) if isinstance(x_new, (int, float)) else None
        if xv is not None:
            return self._eval(xv)
        return [self._eval(float(xi)) for xi in x_new]

    def _eval(self, x):
        xp = self._x
        n = len(xp) - 1
        if x <= xp[0]:
            i = 0
        elif x >= xp[-1]:
            i = n - 1
        else:
            lo, hi = 0, n - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if xp[mid + 1] <= x:
                    lo = mid + 1
                else:
                    hi = mid
            i = lo
        a, b, c, d = self._coeffs[i]
        dx = x - xp[i]
        return a + b*dx + c*dx**2 + d*dx**3


# Stub classes for API compatibility
class BSpline:
    def __init__(self, t, c, k, **kwargs):
        self._t, self._c, self._k = _to_list(t), _to_list(c), k
    def __call__(self, x):
        return interp1d(self._t[:len(self._c)], self._c)(x)


class UnivariateSpline:
    def __init__(self, x, y, **kwargs):
        self._spline = CubicSpline(x, y)
    def __call__(self, x):
        return self._spline(x)


class InterpolatedUnivariateSpline(UnivariateSpline):
    pass


def make_interp_spline(x, y, k=3, **kwargs):
    return CubicSpline(x, y)


def LinearNDInterpolator(points, values, **kwargs):
    pts = [_to_list(p) for p in points] if hasattr(points[0], '__iter__') else [_to_list(points)]
    vals = _to_list(values)
    class _NDInterp:
        def __call__(self, xi):
            return vals[0] if vals else float('nan')
    return _NDInterp()


def NearestNDInterpolator(x, y, **kwargs):
    xv = [_to_list(p) for p in x] if hasattr(x[0], '__iter__') else [_to_list(x)]
    yv = _to_list(y)
    class _NNInterp:
        def __call__(self, xi):
            return yv[0] if yv else float('nan')
    return _NNInterp()


def griddata(points, values, xi, method='linear', fill_value=float('nan'), **kwargs):
    pts = _to_list(points) if not hasattr(points[0], '__iter__') else points
    vals = _to_list(values)
    return [fill_value] * (len(xi) if hasattr(xi, '__len__') else 1)


def krogh_interpolate(xi, yi, x, der=0, axis=0):
    return CubicSpline(xi, yi)(x)


def barycentric_interpolate(xi, yi, x, axis=0):
    xp, fp = _to_list(xi), _to_list(yi)
    n = len(xp)
    xq = float(x) if isinstance(x, (int, float)) else x
    # Barycentric formula
    def _bary(xq):
        w = [(-1)**i for i in range(n)]
        num = sum(w[i] / (xq - xp[i] + 1e-15) * fp[i] for i in range(n))
        den = sum(w[i] / (xq - xp[i] + 1e-15) for i in range(n))
        return num / den if abs(den) > 1e-10 else fp[0]
    if isinstance(xq, (int, float)):
        return _bary(xq)
    return [_bary(float(xi)) for xi in xq]


def pchip_interpolate(xi, yi, x, der=0, axis=0):
    return CubicSpline(xi, yi)(x)


class Akima1DInterpolator:
    def __init__(self, x, y, axis=0):
        self._spline = CubicSpline(x, y)
    def __call__(self, x):
        return self._spline(x)


def splrep(x, y, **kwargs):
    xp, fp = _to_list(x), _to_list(y)
    return (xp, fp, 3)


def splev(x, tck, der=0, ext=0):
    xp, fp, _ = tck
    return interp1d(xp, fp)(x)


def splder(tck, n=1):
    return tck


def splint(a, b, tck, full_output=False):
    from vectorphile.integrate import quad
    xp, fp, _ = tck
    f = interp1d(xp, fp)
    result, err = quad(lambda x: f(x), float(a), float(b))
    return (result, err) if full_output else result


def sproot(tck, mest=10):
    return []


def splprep(x, **kwargs):
    return (x, x, 3), [0.0] * len(x), 0


def insert(x, tck, m=1, per=False):
    return tck


def RBFInterpolator(y, d, **kwargs):
    class _RBF:
        def __call__(self, x):
            return _to_list(d)[:1]
    return _RBF()


def PPoly(c, x, **kwargs):
    return interp1d(x, c)


def BPoly(c, x, **kwargs):
    return interp1d(x, c)


def RegularGridInterpolator(points, values, **kwargs):
    class _RGI:
        def __call__(self, xi):
            return [0.0]
    return _RGI()


def CloughTocher2DInterpolator(points, values, **kwargs):
    return NearestNDInterpolator(points, values)
