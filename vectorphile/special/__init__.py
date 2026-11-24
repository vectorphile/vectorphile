"""vectorphile.special — Special mathematical functions.

Pure Python. No external dependencies.
"""
from __future__ import annotations
import math
from typing import Union, List


Number = Union[float, int]


# ---------------------------------------------------------------------------
# Gamma and related
# ---------------------------------------------------------------------------

def gamma(x) -> float:
    if isinstance(x, (list, tuple)):
        return [math.gamma(float(v)) for v in x]
    return math.gamma(float(x))


def lgamma(x) -> float:
    if isinstance(x, (list, tuple)):
        return [math.lgamma(float(v)) for v in x]
    return math.lgamma(float(x))


def _digamma_scalar(x: float) -> float:
    if x <= 0 and x == int(x):
        return float('nan')
    if x < 6:
        result = 0.0
        while x < 6:
            result -= 1.0 / x
            x += 1
        return result + _digamma_scalar(x)
    r = 1 / x
    return (math.log(x) - 0.5 * r -
            r**2 * (1/12 - r**2 * (1/120 - r**2 * (1/252 - r**2 / 240))))


def digamma(x) -> float:
    if isinstance(x, (list, tuple)):
        return [_digamma_scalar(float(v)) for v in x]
    return _digamma_scalar(float(x))


def beta(a, b) -> float:
    return math.exp(math.lgamma(float(a)) + math.lgamma(float(b)) - math.lgamma(float(a) + float(b)))


def lbeta(a, b) -> float:
    return math.lgamma(float(a)) + math.lgamma(float(b)) - math.lgamma(float(a) + float(b))


def factorial(n: int) -> int:
    return math.factorial(int(n))


def comb(n: int, k: int) -> int:
    return math.comb(int(n), int(k))


def perm(n: int, k: int) -> int:
    return math.perm(int(n), int(k))


# ---------------------------------------------------------------------------
# Error functions
# ---------------------------------------------------------------------------

def erf(x) -> float:
    if isinstance(x, (list, tuple)):
        return [math.erf(float(v)) for v in x]
    return math.erf(float(x))


def erfc(x) -> float:
    if isinstance(x, (list, tuple)):
        return [math.erfc(float(v)) for v in x]
    return math.erfc(float(x))


def _erfinv_scalar(x: float) -> float:
    if abs(x) >= 1.0:
        return math.copysign(float('inf'), x)
    w = -math.log((1 - x) * (1 + x) + 1e-15)
    if w < 6.25:
        guess = x * (1.0 + w * (-0.000508781949658329 + w * (1.0 - w * 0.0083894661)))
    else:
        guess = math.copysign(math.sqrt(max(w - math.log(w + 1e-15), 0.1)) * 1.5, x)
    for _ in range(5):
        f = math.erf(guess) - x
        df = 2.0 / math.sqrt(math.pi) * math.exp(-guess ** 2)
        if abs(df) < 1e-15:
            break
        guess -= f / df
    return guess


def erfinv(x) -> float:
    if isinstance(x, (list, tuple)):
        return [_erfinv_scalar(float(v)) for v in x]
    return _erfinv_scalar(float(x))


def ndtr(x) -> float:
    if isinstance(x, (list, tuple)):
        return [0.5 * math.erfc(-float(v) / math.sqrt(2)) for v in x]
    return 0.5 * math.erfc(-float(x) / math.sqrt(2))


def _ndtri_scalar(p: float) -> float:
    if p <= 0:
        return float('-inf')
    if p >= 1:
        return float('inf')
    return math.sqrt(2) * _erfinv_scalar(2 * p - 1)


def ndtri(p) -> float:
    if isinstance(p, (list, tuple)):
        return [_ndtri_scalar(float(v)) for v in p]
    return _ndtri_scalar(float(p))


# ---------------------------------------------------------------------------
# Logistic / logit / expit
# ---------------------------------------------------------------------------

def expit(x) -> float:
    if isinstance(x, (list, tuple)):
        return [1.0 / (1.0 + math.exp(-float(v))) for v in x]
    return 1.0 / (1.0 + math.exp(-float(x)))


def logit(p) -> float:
    if isinstance(p, (list, tuple)):
        return [math.log(float(v) / (1 - float(v))) for v in p]
    p = float(p)
    return math.log(p / (1 - p))


def log1p(x) -> float:
    if isinstance(x, (list, tuple)):
        return [math.log1p(float(v)) for v in x]
    return math.log1p(float(x))


def expm1(x) -> float:
    if isinstance(x, (list, tuple)):
        return [math.expm1(float(v)) for v in x]
    return math.expm1(float(x))


# ---------------------------------------------------------------------------
# Bessel functions
# ---------------------------------------------------------------------------

def _j0_scalar(x: float) -> float:
    if abs(x) < 8:
        result = 0.0
        t = 1.0
        for m in range(1, 30):
            result += t
            t *= -(x / 2) ** 2 / (m * m)
        return result + t
    return math.sqrt(2 / (math.pi * abs(x))) * math.cos(abs(x) - math.pi / 4)


def j0(x) -> float:
    if isinstance(x, (list, tuple)):
        return [_j0_scalar(float(v)) for v in x]
    return _j0_scalar(float(x))


def _j1_scalar(x: float) -> float:
    if x == 0:
        return 0.0
    if abs(x) < 8:
        result = 0.0
        t = x / 2
        for m in range(1, 30):
            result += t
            t *= -(x / 2) ** 2 / (m * (m + 1))
        return result + t
    return math.sqrt(2 / (math.pi * abs(x))) * math.cos(abs(x) - 3 * math.pi / 4) * math.copysign(1, x)


def j1(x) -> float:
    if isinstance(x, (list, tuple)):
        return [_j1_scalar(float(v)) for v in x]
    return _j1_scalar(float(x))


def _i0_scalar(x: float) -> float:
    result = 0.0
    t = 1.0
    for m in range(1, 30):
        result += t
        t *= (x / 2) ** 2 / (m * m)
    return result + t


def i0(x) -> float:
    if isinstance(x, (list, tuple)):
        return [_i0_scalar(float(v)) for v in x]
    return _i0_scalar(float(x))


# ---------------------------------------------------------------------------
# Orthogonal polynomials (scalar only)
# ---------------------------------------------------------------------------

def legendre(n: int, x) -> float:
    xf = float(x)
    if n == 0:
        return 1.0
    if n == 1:
        return xf
    p_prev, p_curr = 1.0, xf
    for k in range(1, n):
        p_next = ((2 * k + 1) * xf * p_curr - k * p_prev) / (k + 1)
        p_prev, p_curr = p_curr, p_next
    return p_curr


def hermite(n: int, x) -> float:
    xf = float(x)
    if n == 0:
        return 1.0
    if n == 1:
        return xf
    h_prev, h_curr = 1.0, xf
    for k in range(1, n):
        h_next = xf * h_curr - k * h_prev
        h_prev, h_curr = h_curr, h_next
    return h_curr


def laguerre(n: int, x) -> float:
    xf = float(x)
    if n == 0:
        return 1.0
    if n == 1:
        return 1.0 - xf
    l_prev, l_curr = 1.0, 1.0 - xf
    for k in range(1, n):
        l_next = ((2 * k + 1 - xf) * l_curr - k * l_prev) / (k + 1)
        l_prev, l_curr = l_curr, l_next
    return l_curr


# ---------------------------------------------------------------------------
# Softmax and log-sum-exp (pure Python)
# ---------------------------------------------------------------------------

def softmax(x, axis: int = -1):
    if isinstance(x, (list, tuple)):
        if x and isinstance(x[0], (list, tuple)):
            return [softmax(row) for row in x]
        xf = [float(v) for v in x]
        mx = max(xf)
        exps = [math.exp(v - mx) for v in xf]
        s = sum(exps)
        return [e / s for e in exps]
    return [float(x)]


def log_softmax(x, axis: int = -1):
    if isinstance(x, (list, tuple)):
        if x and isinstance(x[0], (list, tuple)):
            return [log_softmax(row) for row in x]
        xf = [float(v) for v in x]
        mx = max(xf)
        shifted = [v - mx for v in xf]
        log_sum = math.log(sum(math.exp(v) for v in shifted))
        return [v - log_sum for v in shifted]
    return [0.0]


def logsumexp(x, axis: int = -1, keepdims: bool = False):
    if isinstance(x, (list, tuple)):
        if x and isinstance(x[0], (list, tuple)):
            return [logsumexp(row) for row in x]
        xf = [float(v) for v in x]
        mx = max(xf)
        return mx + math.log(sum(math.exp(v - mx) for v in xf))
    return float(x)


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b) via Lentz continued fraction."""
    if x < 0 or x > 1:
        raise ValueError("x must be in [0, 1]")
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0
    if x > (a + 1) / (a + b + 2):
        return 1.0 - betainc(b, a, 1 - x)
    lbeta_ab = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta_ab)
    # Lentz's algorithm for the continued fraction
    TINY = 1e-30
    f = TINY
    C = TINY
    D = 0.0
    for m in range(200):
        for n_step in range(2):
            if m == 0 and n_step == 0:
                d = 1.0
            elif n_step == 0:
                # Even step: d = m*(b-m)*x / ((a+2m-1)*(a+2m))
                d = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
            else:
                # Odd step: d = -(a+m)*(a+b+m)*x / ((a+2m)*(a+2m+1))
                d = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
            D = 1.0 + d * D
            if abs(D) < TINY:
                D = TINY
            C = 1.0 + d / C
            if abs(C) < TINY:
                C = TINY
            D = 1.0 / D
            delta = C * D
            f *= delta
            if abs(delta - 1.0) < 3e-7:
                return front * f / a
    return front * f / a


def gammainc(a: float, x: float) -> float:
    """Regularized lower incomplete gamma function P(a, x)."""
    if x < 0:
        raise ValueError("x must be >= 0")
    if x == 0:
        return 0.0
    if x < a + 1:
        # Series expansion
        ap = a
        s = 1.0 / a
        delta = s
        for _ in range(300):
            ap += 1
            delta *= x / ap
            s += delta
            if abs(delta) < 1e-10 * abs(s):
                break
        return s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    else:
        # Continued fraction (Legendre's)
        b = x + 1 - a
        c = 1e30
        d = 1.0 / b
        f = d
        for i in range(1, 300):
            an = -i * (i - a)
            b += 2
            d = an * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            del_ = d * c
            f *= del_
            if abs(del_ - 1.0) < 1e-10:
                break
        return 1.0 - math.exp(-x + a * math.log(x) - math.lgamma(a)) * f


def zeta(s: float, a: float = 1.0) -> float:
    """Hurwitz zeta function ζ(s, a); Riemann zeta when a=1."""
    if s <= 1:
        raise ValueError("s must be > 1")
    result = 0.0
    for n in range(300):
        result += 1.0 / (n + a) ** s
    return result


def hyp2f1(a: float, b: float, c: float, z: float) -> float:
    """Gauss hypergeometric function 2F1(a,b;c;z) via series (|z|<1)."""
    if abs(z) >= 1:
        raise ValueError("|z| must be < 1")
    result = 1.0
    term = 1.0
    for n in range(1, 500):
        term *= (a + n - 1) * (b + n - 1) / ((c + n - 1) * n) * z
        result += term
        if abs(term) < 1e-12:
            break
    return result


def entr(x) -> float:
    """Elementwise entropy: -x * log(x) for x > 0, 0 for x==0, -inf for x<0."""
    import math as _m
    if hasattr(x, '_data'):
        flat = [float(v) for v in (x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data))]
        from vectorphile.core.tensor import Tensor as _T
        from vectorphile._core.ndarray import array as _nd
        result = [-v * _m.log(v) if v > 0 else (0.0 if v == 0 else float('-inf')) for v in flat]
        nd = _nd(result)
        if len(x.shape) > 1:
            nd = nd.reshape(x.shape)
        return _T(nd)
    v = float(x)
    if v < 0: return float('-inf')
    if v == 0: return 0.0
    return -v * _m.log(v)


def xlogy(x, y) -> float:
    """x * log(y), 0 when x == 0."""
    import math as _m
    xv, yv = float(x) if not hasattr(x, 'item') else float(x.item()), \
              float(y) if not hasattr(y, 'item') else float(y.item())
    if xv == 0: return 0.0
    return xv * _m.log(yv)


def xlog1py(x, y) -> float:
    """x * log(1 + y), 0 when x == 0."""
    import math as _m
    xv = float(x) if not hasattr(x, 'item') else float(x.item())
    yv = float(y) if not hasattr(y, 'item') else float(y.item())
    if xv == 0: return 0.0
    return xv * _m.log1p(yv)


def multigammaln(a: float, d: int) -> float:
    """Log of multivariate gamma function."""
    import math as _m
    result = d * (d - 1) / 4.0 * _m.log(_m.pi)
    for j in range(1, d + 1):
        result += lgamma(a + (1 - j) / 2.0)
    return result


def polygamma(n: int, x) -> float:
    """Polygamma function (n-th derivative of digamma)."""
    import math as _m
    xv = float(x) if not hasattr(x, 'item') else float(x.item())
    if n == 0:
        return _digamma_scalar(xv)
    # Use recurrence / asymptotic expansion
    sign = (-1) ** (n + 1)
    factorial_n = _m.factorial(n)
    # Asymptotic: sum 1/(x+k)^(n+1)
    result = 0.0
    for k in range(100):
        result += 1.0 / (xv + k) ** (n + 1)
    return sign * factorial_n * result


# Alias
psi = digamma


__all__ = [
    'gamma', 'lgamma', 'digamma', 'beta', 'lbeta', 'factorial', 'comb', 'perm',
    'erf', 'erfc', 'erfinv', 'ndtr', 'ndtri',
    'expit', 'logit', 'log1p', 'expm1',
    'j0', 'j1', 'i0',
    'legendre', 'hermite', 'laguerre',
    'softmax', 'log_softmax', 'logsumexp',
    'betainc', 'gammainc', 'zeta', 'hyp2f1',
    'entr', 'xlogy', 'xlog1py', 'multigammaln', 'polygamma', 'psi',
]
