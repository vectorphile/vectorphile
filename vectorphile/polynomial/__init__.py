"""Polynomial operations — wraps numpy.polynomial with extra utilities."""
from __future__ import annotations
import numpy as np
from numpy.polynomial import (
    Polynomial, Chebyshev, Legendre, Laguerre, Hermite, HermiteE,
    polynomial, chebyshev, legendre, laguerre, hermite, hermite_e,
)
from numpy.polynomial.polynomial import (
    polyadd, polysub, polymul, polydiv, polyval, polyroots,
    polyfit, polyder, polyint, polypow, polycompanion, polyvander,
    polyvander2d, polyvander3d, polyval2d, polyval3d, polygrid2d, polygrid3d,
)
from numpy.polynomial.chebyshev import (
    chebval, chebfit, chebder, chebint, chebroots, chebpts1, chebpts2,
    chebfromroots, chebgauss, chebweight, chebvander,
)
from numpy.polynomial.legendre import (
    legval, legfit, legder, legint, legroots, leggauss, legweight, legvander,
)
from numpy.polynomial.laguerre import (
    lagval, lagfit, lagder, lagint, lagroots, laggauss, lagweight, lagvander,
)
from numpy.polynomial.hermite import (
    hermval, hermfit, hermder, hermint, hermroots, hermgauss, hermweight, hermvander,
)
from numpy.polynomial.hermite_e import (
    hermeval, hermefit, hermeder, hermeint, hermeroots, hermegauss, hermeweight, hermevander,
)

# Additional utilities
def polyeval_horner(coeffs, x):
    """Evaluate polynomial using Horner's method."""
    result = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        result = result * x + c
    return result


def poly_compose(p1, p2):
    """Compose polynomials: compute p1(p2(x))."""
    result = np.zeros(1)
    for coef in reversed(p1):
        result = np.polymul(result, p2)
        result[0] += coef
    return result


def poly_gcd(p1, p2):
    """Compute GCD of two polynomials via Euclidean algorithm."""
    p1, p2 = np.array(p1, dtype=np.float64), np.array(p2, dtype=np.float64)
    # Trim leading zeros
    p1 = np.trim_zeros(p1, 'f')
    p2 = np.trim_zeros(p2, 'f')
    while len(p2) > 0 and not np.allclose(p2, 0):
        _, r = np.polydiv(p1, p2)
        p1, p2 = p2, np.trim_zeros(r, 'f')
    return p1 / p1[0] if len(p1) > 0 and p1[0] != 0 else p1


def poly_lcm(p1, p2):
    """Least common multiple of two polynomials."""
    gcd = poly_gcd(p1, p2)
    prod = np.polymul(p1, p2)
    return np.polydiv(prod, gcd)[0]


def poly_interpolate(x_points, y_points):
    """Lagrange interpolation — return coefficients of interpolating polynomial."""
    x, y = np.array(x_points, dtype=np.float64), np.array(y_points, dtype=np.float64)
    n = len(x)
    result = np.zeros(n)
    for i in range(n):
        # Compute basis polynomial L_i
        basis = np.array([1.0])
        denom = 1.0
        for j in range(n):
            if j != i:
                basis = np.polymul(basis, [1, -x[j]])
                denom *= (x[i] - x[j])
        result = result + y[i] * basis / denom
    return result


def chebyshev_nodes(n, a=-1.0, b=1.0):
    """Chebyshev nodes of the first kind on [a, b]."""
    k = np.arange(1, n + 1)
    nodes = np.cos((2*k - 1) * np.pi / (2*n))
    return 0.5 * (a + b) + 0.5 * (b - a) * nodes


def companion_matrix(coeffs):
    """Companion matrix — eigenvalues are roots of the polynomial."""
    coeffs = np.array(coeffs, dtype=np.float64)
    n = len(coeffs) - 1
    C = np.zeros((n, n), dtype=np.float64)
    C[1:, :-1] = np.eye(n - 1)
    C[:, -1] = -coeffs[:-1] / coeffs[-1]
    return C


def sturm_sequence(p):
    """Sturm sequence for counting real roots in intervals."""
    sturm = [np.array(p, dtype=np.float64)]
    deriv = np.polyder(sturm[-1])
    if len(deriv) == 0:
        return sturm
    sturm.append(deriv)
    while len(sturm[-1]) > 1:
        _, r = np.polydiv(sturm[-2], sturm[-1])
        r = -np.trim_zeros(r, 'f')
        if len(r) == 0:
            break
        sturm.append(r)
    return sturm


def count_real_roots(p, a, b):
    """Count real roots of polynomial p in interval [a, b] via Sturm's theorem."""
    seq = sturm_sequence(p)
    def sign_changes(x):
        vals = [float(np.polyval(s, x)) for s in seq if len(s) > 0]
        changes = sum(1 for i in range(len(vals)-1) if vals[i]*vals[i+1] < 0)
        return changes
    return sign_changes(a) - sign_changes(b)
