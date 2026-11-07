"""Interpolation — wraps scipy.interpolate with vectorphile naming."""
from __future__ import annotations
import numpy as np

try:
    import scipy.interpolate as _sp

    # 1D interpolation
    def interp1d(x, y, kind='linear', axis=-1, copy=True, bounds_error=True,
                 fill_value=np.nan, assume_sorted=False):
        return _sp.interp1d(x, y, kind=kind, axis=axis, copy=copy,
                            bounds_error=bounds_error, fill_value=fill_value,
                            assume_sorted=assume_sorted)

    def interp2d(x, y, z, kind='linear', copy=True, bounds_error=True, fill_value=None):
        return _sp.interp2d(x, y, z, kind=kind, copy=copy,
                            bounds_error=bounds_error, fill_value=fill_value)

    # Splines
    def CubicSpline(x, y, axis=0, bc_type=None, extrapolate=None):
        return _sp.CubicSpline(x, y, axis=axis, bc_type=bc_type, extrapolate=extrapolate)

    def BSpline(t, c, k, extrapolate=True, axis=0):
        return _sp.BSpline(t, c, k, extrapolate=extrapolate, axis=axis)

    def UnivariateSpline(x, y, w=None, bbox=None, k=3, s=None, ext=0, check_finite=False):
        return _sp.UnivariateSpline(x, y, w=w, bbox=bbox or [None, None], k=k, s=s, ext=ext, check_finite=check_finite)

    def InterpolatedUnivariateSpline(x, y, w=None, bbox=None, k=3, ext=0, check_finite=False):
        return _sp.InterpolatedUnivariateSpline(x, y, w=w, bbox=bbox or [None, None], k=k, ext=ext, check_finite=check_finite)

    def make_interp_spline(x, y, k=3, t=None, bc_type=None, axis=0, check_finite=True):
        return _sp.make_interp_spline(x, y, k=k, t=t, bc_type=bc_type, axis=axis, check_finite=check_finite)

    def PPoly(c, x, extrapolate=None, axis=0):
        return _sp.PPoly(c, x, extrapolate=extrapolate, axis=axis)

    def BPoly(c, x, extrapolate=None, axis=0):
        return _sp.BPoly(c, x, extrapolate=extrapolate, axis=axis)

    # N-D interpolation
    def RegularGridInterpolator(points, values, method='linear', bounds_error=True, fill_value=np.nan):
        return _sp.RegularGridInterpolator(points, values, method=method,
                                            bounds_error=bounds_error, fill_value=fill_value)

    def LinearNDInterpolator(points, values, fill_value=np.nan, rescale=False):
        return _sp.LinearNDInterpolator(points, values, fill_value=fill_value, rescale=rescale)

    def NearestNDInterpolator(x, y, rescale=False, tree_options=None):
        return _sp.NearestNDInterpolator(x, y, rescale=rescale, tree_options=tree_options)

    def CloughTocher2DInterpolator(points, values, fill_value=np.nan, tol=1e-6, maxiter=400, rescale=False):
        return _sp.CloughTocher2DInterpolator(points, values, fill_value=fill_value, tol=tol, maxiter=maxiter, rescale=rescale)

    def griddata(points, values, xi, method='linear', fill_value=np.nan, rescale=False):
        return _sp.griddata(points, values, xi, method=method, fill_value=fill_value, rescale=rescale)

    def RBFInterpolator(y, d, neighbors=None, smoothing=0.0, kernel='thin_plate_spline',
                        epsilon=None, degree=None):
        return _sp.RBFInterpolator(y, d, neighbors=neighbors, smoothing=smoothing,
                                    kernel=kernel, epsilon=epsilon, degree=degree)

    # Barycentric / Krogh
    def krogh_interpolate(xi, yi, x, der=0, axis=0):
        return _sp.krogh_interpolate(xi, yi, x, der=der, axis=axis)

    def barycentric_interpolate(xi, yi, x, axis=0):
        return _sp.barycentric_interpolate(xi, yi, x, axis=axis)

    def pchip_interpolate(xi, yi, x, der=0, axis=0):
        return _sp.pchip_interpolate(xi, yi, x, der=der, axis=axis)

    def Akima1DInterpolator(x, y, axis=0):
        return _sp.Akima1DInterpolator(x, y, axis=axis)

    # Spline helpers
    def splrep(x, y, w=None, xb=None, xe=None, k=3, s=None, full_output=False):
        return _sp.splrep(x, y, w=w, xb=xb, xe=xe, k=k, s=s, full_output=full_output)

    def splev(x, tck, der=0, ext=0):
        return _sp.splev(x, tck, der=der, ext=ext)

    def splder(tck, n=1):
        return _sp.splder(tck, n=n)

    def splint(a, b, tck, full_output=False):
        return _sp.splint(a, b, tck, full_output=full_output)

    def sproot(tck, mest=10):
        return _sp.sproot(tck, mest=mest)

    def splprep(x, w=None, u=None, ub=None, ue=None, k=3, task=0, s=None, t=None, full_output=False, nest=None, per=False, quiet=1):
        return _sp.splprep(x, w=w, u=u, ub=ub, ue=ue, k=k, task=task, s=s, t=t, full_output=full_output, nest=nest, per=per, quiet=quiet)

    def insert(x, tck, m=1, per=False):
        return _sp.insert(x, tck, m=m, per=per)

except ImportError:
    def _no_scipy(*args, **kwargs):
        raise ImportError("scipy is required for vp.interpolate. Install with: pip install scipy")
    interp1d = interp2d = CubicSpline = BSpline = UnivariateSpline = _no_scipy
    InterpolatedUnivariateSpline = make_interp_spline = PPoly = BPoly = _no_scipy
    RegularGridInterpolator = LinearNDInterpolator = NearestNDInterpolator = _no_scipy
    CloughTocher2DInterpolator = griddata = RBFInterpolator = _no_scipy
    krogh_interpolate = barycentric_interpolate = pchip_interpolate = Akima1DInterpolator = _no_scipy
    splrep = splev = splder = splint = sproot = splprep = insert = _no_scipy


# Pure-NumPy fallback: linear 1D interpolation
def lerp(x, xp, fp, left=None, right=None):
    """Linear interpolation (alias for np.interp)."""
    return np.interp(x, xp, fp, left=left, right=right)


def piecewise_linear(x, xp, yp):
    """Piecewise linear interpolation."""
    return np.interp(x, xp, yp)
