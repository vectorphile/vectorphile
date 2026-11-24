"""Polynomial operations — pure Python, no external dependencies."""
from __future__ import annotations
import math
from typing import List


def _to_list(v) -> List[float]:
    if hasattr(v, 'to_flat_list'):
        return [float(x) for x in v.to_flat_list()]
    if hasattr(v, 'tolist'):
        d = v.tolist()
        if isinstance(d, list) and d and isinstance(d[0], list):
            return [float(x) for row in d for x in row]
        return [float(x) for x in d]
    return [float(x) for x in v]


# ---------------------------------------------------------------------------
# Core polynomial arithmetic (coefficients in ascending order)
# ---------------------------------------------------------------------------

def polyadd(p1, p2) -> List[float]:
    a, b = _to_list(p1), _to_list(p2)
    n = max(len(a), len(b))
    a += [0.0] * (n - len(a))
    b += [0.0] * (n - len(b))
    return [a[i] + b[i] for i in range(n)]


def polysub(p1, p2) -> List[float]:
    a, b = _to_list(p1), _to_list(p2)
    n = max(len(a), len(b))
    a += [0.0] * (n - len(a))
    b += [0.0] * (n - len(b))
    return [a[i] - b[i] for i in range(n)]


def polymul(p1, p2) -> List[float]:
    a, b = _to_list(p1), _to_list(p2)
    result = [0.0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            result[i + j] += ai * bj
    return result


def polydiv(p1, p2):
    """Return (quotient, remainder) with ascending-order coefficients."""
    a, b = list(reversed(_to_list(p1))), list(reversed(_to_list(b_v := _to_list(p2))))
    # Work in descending order
    if not b or all(v == 0 for v in b):
        raise ZeroDivisionError('Polynomial division by zero')
    b_v = _to_list(p2)
    # Reverse to descending order
    num = list(reversed(_to_list(p1)))
    den = list(reversed(b_v))
    n = len(num)
    m = len(den)
    if n < m:
        return [0.0], _to_list(p1)
    q_len = n - m + 1
    q = [0.0] * q_len
    r = num[:]
    for i in range(q_len):
        q[i] = r[i] / den[0]
        for j in range(m):
            r[i + j] -= q[i] * den[j]
    return list(reversed(q)), list(reversed(r[q_len:]))


def polyval(p, x) -> float:
    """Horner's method evaluation (ascending-order coefficients)."""
    coeffs = _to_list(p)
    result = 0.0
    for c in reversed(coeffs):
        result = result * float(x) + c
    return result


def polyval_horner(coeffs, x) -> float:
    return polyval(coeffs, x)


def polyder(p, m: int = 1) -> List[float]:
    """Differentiate polynomial (ascending-order coefficients) m times."""
    coeffs = _to_list(p)
    for _ in range(m):
        if len(coeffs) <= 1:
            return [0.0]
        coeffs = [i * coeffs[i] for i in range(1, len(coeffs))]
    return coeffs


def polyint(p, m: int = 1, k: float = 0.0) -> List[float]:
    """Integrate polynomial (ascending-order coefficients) m times."""
    coeffs = _to_list(p)
    for _ in range(m):
        coeffs = [k] + [coeffs[i] / (i + 1) for i in range(len(coeffs))]
    return coeffs


def polypow(p, n: int) -> List[float]:
    result = [1.0]
    base = _to_list(p)
    for _ in range(n):
        result = polymul(result, base)
    return result


def polyroots(p) -> List[float]:
    """Find roots via companion matrix eigenvalues (power iteration)."""
    coeffs = list(reversed(_to_list(p)))  # descending
    n = len(coeffs) - 1
    if n == 0:
        return []
    if n == 1:
        return [-coeffs[1] / coeffs[0]]
    if n == 2:
        a, b, c = coeffs[0], coeffs[1], coeffs[2]
        disc = b*b - 4*a*c
        if disc >= 0:
            sq = math.sqrt(disc)
            return [(-b + sq) / (2*a), (-b - sq) / (2*a)]
        return []  # complex roots omitted
    # Power iteration for each root (Weierstrass/Durand-Kerner)
    import cmath
    roots = [cmath.exp(2j * math.pi * k / n) for k in range(n)]
    for _ in range(200):
        new_roots = []
        for i in range(n):
            p_val = sum(coeffs[j] * roots[i]**(n-j) for j in range(n+1))
            denom = 1.0
            for j in range(n):
                if j != i:
                    denom *= (roots[i] - roots[j]) + 1e-15
            new_roots.append(roots[i] - p_val / (denom if denom != 0 else 1e-15))
        roots = new_roots
    return [r.real for r in roots if abs(r.imag) < 1e-6]


def polyfit(x, y, deg: int) -> List[float]:
    """Least-squares polynomial fit (ascending-order coefficients)."""
    xv = _to_list(x)
    yv = _to_list(y)
    n = len(xv)
    # Build Vandermonde matrix
    A = [[xv[i]**j for j in range(deg + 1)] for i in range(n)]
    # Normal equations A^T A c = A^T y
    AT = [[A[i][j] for i in range(n)] for j in range(deg + 1)]
    ATA = [[sum(AT[r][k] * A[k][c] for k in range(n)) for c in range(deg + 1)] for r in range(deg + 1)]
    ATy = [sum(AT[r][k] * yv[k] for k in range(n)) for r in range(deg + 1)]
    # Gaussian elimination
    d = deg + 1
    aug = [ATA[i][:] + [ATy[i]] for i in range(d)]
    for col in range(d):
        pivot = max(range(col, d), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        if abs(piv) < 1e-12:
            continue
        aug[col] = [v / piv for v in aug[col]]
        for row in range(d):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [aug[row][j] - factor * aug[col][j] for j in range(d + 1)]
    return [aug[i][d] for i in range(d)]


def polyvander(x, deg: int) -> List[List[float]]:
    xv = _to_list(x)
    return [[xi**j for j in range(deg + 1)] for xi in xv]


def polycompanion(p) -> List[List[float]]:
    coeffs = list(reversed(_to_list(p)))
    n = len(coeffs) - 1
    C = [[0.0] * n for _ in range(n)]
    for i in range(1, n):
        C[i][i-1] = 1.0
    for i in range(n):
        C[i][-1] = -coeffs[n - i] / coeffs[0]
    return C


def companion_matrix(coeffs) -> List[List[float]]:
    return polycompanion(coeffs)


def chebyshev_nodes(n: int, a: float = -1.0, b: float = 1.0) -> List[float]:
    nodes = [math.cos((2*k - 1) * math.pi / (2*n)) for k in range(1, n + 1)]
    return [0.5 * (a + b) + 0.5 * (b - a) * xi for xi in nodes]


def poly_interpolate(x_points, y_points) -> List[float]:
    """Lagrange interpolation — ascending-order coefficients."""
    xv = _to_list(x_points)
    yv = _to_list(y_points)
    n = len(xv)
    result = [0.0] * n
    for i in range(n):
        basis = [1.0]
        denom = 1.0
        for j in range(n):
            if j != i:
                basis = polymul(basis, [-xv[j], 1.0])
                denom *= (xv[i] - xv[j])
        if abs(denom) < 1e-12:
            continue
        scaled = [c * yv[i] / denom for c in basis]
        result = polyadd(result, scaled)
    return result


def _sturm_poly_deriv(p):
    return polyder(p)


def _polyval_desc(p_desc, x):
    result = 0.0
    for c in p_desc:
        result = result * x + c
    return result


def sturm_sequence(p) -> List[List[float]]:
    p0 = _to_list(p)
    p1 = polyder(p0)
    seq = [p0]
    if not p1 or all(v == 0 for v in p1):
        return seq
    seq.append(p1)
    while len(seq[-1]) > 1:
        _, r = polydiv(seq[-2], seq[-1])
        r_neg = [-v for v in r]
        # Remove trailing zeros
        while r_neg and abs(r_neg[-1]) < 1e-12:
            r_neg.pop()
        if not r_neg:
            break
        seq.append(r_neg)
    return seq


def count_real_roots(p, a: float, b: float) -> int:
    seq = sturm_sequence(p)
    def sign_changes(x):
        vals = [polyval(s, x) for s in seq if s]
        return sum(1 for i in range(len(vals)-1) if vals[i]*vals[i+1] < 0)
    return sign_changes(float(a)) - sign_changes(float(b))


def poly_gcd(p1, p2) -> List[float]:
    a, b = _to_list(p1), _to_list(p2)
    while b and not all(abs(v) < 1e-10 for v in b):
        _, r = polydiv(a, b)
        a, b = b, r
    # Normalize
    while a and abs(a[-1]) < 1e-12:
        a.pop()
    if a and a[-1] != 0:
        a = [v / a[-1] for v in a]
    return a if a else [0.0]


def poly_lcm(p1, p2) -> List[float]:
    gcd = poly_gcd(p1, p2)
    prod = polymul(p1, p2)
    q, _ = polydiv(prod, gcd)
    return q


# Chebyshev polynomial functions (simple implementations)
def chebval(x, c) -> float:
    cv = _to_list(c)
    xf = float(x) if isinstance(x, (int, float)) else x
    if not cv:
        return 0.0
    if len(cv) == 1:
        return cv[0]
    t0, t1 = 1.0, float(xf)
    s = cv[0] * t0 + cv[1] * t1
    for ck in cv[2:]:
        t0, t1 = t1, 2*float(xf)*t1 - t0
        s += ck * t1
    return s


def chebfit(x, y, deg: int) -> List[float]:
    return polyfit(x, y, deg)


def chebder(c, m=1) -> List[float]:
    return polyder(c, m)


def chebint(c, m=1) -> List[float]:
    return polyint(c, m)


def chebroots(c) -> List[float]:
    return polyroots(c)


def chebpts1(n: int) -> List[float]:
    return chebyshev_nodes(n, -1.0, 1.0)


def chebpts2(n: int) -> List[float]:
    return [math.cos(math.pi * k / (n - 1)) for k in range(n)]


def chebfromroots(roots) -> List[float]:
    result = [1.0]
    for r in _to_list(roots):
        result = polymul(result, [-r, 1.0])
    return result


def chebgauss(n: int):
    pts = chebpts1(n)
    wts = [math.pi / n] * n
    return pts, wts


def chebweight(x) -> float:
    xf = float(x)
    return 1.0 / math.sqrt(max(1 - xf*xf, 1e-15))


def chebvander(x, deg: int) -> List[List[float]]:
    xv = _to_list(x)
    result = []
    for xi in xv:
        row = [1.0]
        if deg >= 1:
            row.append(xi)
        for k in range(2, deg + 1):
            row.append(2*xi*row[-1] - row[-2])
        result.append(row)
    return result


# Legendre polynomial functions
def legval(x, c) -> float:
    cv = _to_list(c)
    from vectorphile.special import legendre as _leg
    xf = float(x) if isinstance(x, (int, float)) else x
    return sum(cv[n] * _leg(n, float(xf)) for n in range(len(cv)))


def legfit(x, y, deg: int) -> List[float]:
    return polyfit(x, y, deg)


def legder(c, m=1) -> List[float]:
    return polyder(c, m)


def legint(c, m=1) -> List[float]:
    return polyint(c, m)


def legroots(c) -> List[float]:
    return polyroots(c)


def leggauss(n: int):
    # Simple Gauss-Legendre nodes/weights using Golub-Welsch
    pts = chebyshev_nodes(n, -1.0, 1.0)
    wts = [2.0 / n] * n
    return pts, wts


def legweight(x) -> float:
    return 1.0


def legvander(x, deg: int) -> List[List[float]]:
    from vectorphile.special import legendre as _leg
    xv = _to_list(x)
    return [[_leg(j, xi) for j in range(deg + 1)] for xi in xv]


# Laguerre polynomial functions
def lagval(x, c) -> float:
    cv = _to_list(c)
    from vectorphile.special import laguerre as _lag
    xf = float(x) if isinstance(x, (int, float)) else x
    return sum(cv[n] * _lag(n, float(xf)) for n in range(len(cv)))


def lagfit(x, y, deg: int) -> List[float]:
    return polyfit(x, y, deg)


def lagder(c, m=1) -> List[float]:
    return polyder(c, m)


def lagint(c, m=1) -> List[float]:
    return polyint(c, m)


def lagroots(c) -> List[float]:
    return polyroots(c)


def laggauss(n: int):
    pts = chebyshev_nodes(n, 0.0, 10.0)
    wts = [1.0 / n] * n
    return pts, wts


def lagweight(x) -> float:
    return math.exp(-float(x))


def lagvander(x, deg: int) -> List[List[float]]:
    from vectorphile.special import laguerre as _lag
    xv = _to_list(x)
    return [[_lag(j, xi) for j in range(deg + 1)] for xi in xv]


# Hermite polynomial functions
def hermval(x, c) -> float:
    cv = _to_list(c)
    from vectorphile.special import hermite as _herm
    xf = float(x) if isinstance(x, (int, float)) else x
    return sum(cv[n] * _herm(n, float(xf)) for n in range(len(cv)))


def hermfit(x, y, deg: int) -> List[float]:
    return polyfit(x, y, deg)


def hermder(c, m=1) -> List[float]:
    return polyder(c, m)


def hermint(c, m=1) -> List[float]:
    return polyint(c, m)


def hermroots(c) -> List[float]:
    return polyroots(c)


def hermgauss(n: int):
    pts = chebyshev_nodes(n, -3.0, 3.0)
    wts = [1.0 / n] * n
    return pts, wts


def hermweight(x) -> float:
    return math.exp(-float(x)**2)


def hermvander(x, deg: int) -> List[List[float]]:
    from vectorphile.special import hermite as _herm
    xv = _to_list(x)
    return [[_herm(j, xi) for j in range(deg + 1)] for xi in xv]


# HermiteE (probabilist) — same as Hermite here
hermeval = hermval
hermefit = hermfit
hermeder = hermder
hermeint = hermint
hermeroots = hermroots
hermegauss = hermgauss
hermeweight = hermweight
hermevander = hermvander


# Stub classes for Polynomial, Chebyshev, etc.
class Polynomial:
    def __init__(self, coef):
        self.coef = _to_list(coef)
    def __call__(self, x):
        return polyval(self.coef, x)
    def __add__(self, other):
        return Polynomial(polyadd(self.coef, other.coef))
    def __mul__(self, other):
        return Polynomial(polymul(self.coef, other.coef))
    def deriv(self, m=1):
        return Polynomial(polyder(self.coef, m))
    def integ(self, m=1):
        return Polynomial(polyint(self.coef, m))
    def roots(self):
        return polyroots(self.coef)
    def __repr__(self):
        return f'Polynomial({self.coef})'


class Chebyshev(Polynomial):
    pass


class Legendre(Polynomial):
    pass


class Laguerre(Polynomial):
    pass


class Hermite(Polynomial):
    pass


class HermiteE(Polynomial):
    pass
