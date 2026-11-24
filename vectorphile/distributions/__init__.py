"""Probability distributions for vectorphile.

Mirrors torch.distributions API: each distribution has log_prob(), sample(),
entropy(), mean, variance, and rsample() (reparameterized) where applicable.
All computations use vectorphile's own NDArray/Tensor — no numpy/torch.
"""
import math
import math as _math
import random as _random
from vectorphile.core.tensor import Tensor
from vectorphile._core.ndarray import array as _nd


def _flat(t):
    if isinstance(t, Tensor):
        return list(t._data.to_flat_list() if hasattr(t._data, 'to_flat_list') else t._data)
    if hasattr(t, 'to_flat_list'):
        return list(t.to_flat_list())
    try:
        return [float(t)]
    except Exception:
        return list(t)


def _make(vals, shape=None):
    nd = _nd(vals, dtype='float32')
    if shape is not None:
        nd = nd.reshape(shape)
    return Tensor(nd)


# ── Base class ────────────────────────────────────────────────────────────────

class Distribution:
    """Abstract base class for all distributions."""
    has_rsample = False

    def sample(self, sample_shape=()):
        raise NotImplementedError

    def log_prob(self, value):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def variance(self):
        raise NotImplementedError

    def rsample(self, sample_shape=()):
        raise NotImplementedError("This distribution does not support rsample.")


# ── Normal ────────────────────────────────────────────────────────────────────

class Normal(Distribution):
    """Univariate normal (Gaussian) distribution."""
    has_rsample = True

    def __init__(self, loc, scale):
        self.loc = loc if isinstance(loc, Tensor) else _make([float(loc)])
        self.scale = scale if isinstance(scale, Tensor) else _make([float(scale)])

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        s = _flat(self.scale)
        return _make([v * v for v in s])

    def sample(self, sample_shape=()):
        n = 1
        for s in sample_shape:
            n *= s
        loc_v = _flat(self.loc)[0]
        scale_v = _flat(self.scale)[0]
        vals = [_random.gauss(loc_v, scale_v) for _ in range(max(1, n))]
        return _make(vals, sample_shape if sample_shape else None)

    def rsample(self, sample_shape=()):
        return self.sample(sample_shape)

    def log_prob(self, value):
        vals = _flat(value)
        loc_v = _flat(self.loc)[0]
        scale_v = _flat(self.scale)[0]
        log2pi = math.log(2 * math.pi)
        result = [
            -0.5 * (((v - loc_v) / scale_v) ** 2) - math.log(scale_v) - 0.5 * log2pi
            for v in vals
        ]
        return _make(result)

    def entropy(self):
        scale_v = _flat(self.scale)[0]
        return _make([0.5 * math.log(2 * math.pi * math.e * scale_v * scale_v)])

    def kl_divergence(self, other):
        m1, s1 = _flat(self.loc)[0], _flat(self.scale)[0]
        m2, s2 = _flat(other.loc)[0], _flat(other.scale)[0]
        kl = math.log(s2 / s1) + (s1**2 + (m1-m2)**2) / (2*s2**2) - 0.5
        return _make([kl])


# ── Bernoulli ─────────────────────────────────────────────────────────────────

class Bernoulli(Distribution):
    """Bernoulli distribution parameterized by probs or logits."""
    def __init__(self, probs=None, logits=None):
        if probs is None and logits is None:
            raise ValueError("probs or logits required")
        if logits is not None:
            lv = _flat(logits)[0]
            pv = 1.0 / (1.0 + math.exp(-lv))
        else:
            pv = _flat(probs)[0]
        self._p = max(1e-7, min(1 - 1e-7, pv))

    @property
    def mean(self):
        return _make([self._p])

    @property
    def variance(self):
        return _make([self._p * (1 - self._p)])

    def sample(self, sample_shape=()):
        n = 1
        for s in sample_shape:
            n *= s
        vals = [1.0 if _random.random() < self._p else 0.0 for _ in range(max(1, n))]
        return _make(vals, sample_shape if sample_shape else None)

    def log_prob(self, value):
        vals = _flat(value)
        p = self._p
        result = [
            v * math.log(p) + (1 - v) * math.log(1 - p)
            for v in vals
        ]
        return _make(result)

    def entropy(self):
        p = self._p
        return _make([-p * math.log(p) - (1-p) * math.log(1-p)])


# ── Categorical ───────────────────────────────────────────────────────────────

class Categorical(Distribution):
    """Categorical distribution over K classes."""
    def __init__(self, probs=None, logits=None):
        if logits is not None:
            lv = _flat(logits)
            mx = max(lv)
            exps = [math.exp(v - mx) for v in lv]
            s = sum(exps)
            self._probs = [v / s for v in exps]
        else:
            pv = _flat(probs)
            s = sum(pv)
            self._probs = [v / s for v in pv]

    @property
    def mean(self):
        return _make(list(range(len(self._probs))), (len(self._probs),))

    @property
    def variance(self):
        p = self._probs
        mean_v = sum(i * pi for i, pi in enumerate(p))
        v = sum(pi * (i - mean_v)**2 for i, pi in enumerate(p))
        return _make([v])

    def sample(self, sample_shape=()):
        n = 1
        for s in sample_shape:
            n *= s
        K = len(self._probs)
        vals = []
        for _ in range(max(1, n)):
            r = _random.random()
            cumul = 0.0
            chosen = K - 1
            for i, pi in enumerate(self._probs):
                cumul += pi
                if r < cumul:
                    chosen = i
                    break
            vals.append(float(chosen))
        return _make(vals, sample_shape if sample_shape else None)

    def log_prob(self, value):
        vals = _flat(value)
        result = [math.log(self._probs[int(v)]) for v in vals]
        return _make(result)

    def entropy(self):
        h = -sum(p * math.log(p) for p in self._probs if p > 0)
        return _make([h])


# ── Uniform ───────────────────────────────────────────────────────────────────

class Uniform(Distribution):
    """Uniform distribution on [low, high)."""
    has_rsample = True

    def __init__(self, low, high):
        self._low = float(_flat(low)[0])
        self._high = float(_flat(high)[0])

    @property
    def mean(self):
        return _make([(self._low + self._high) / 2])

    @property
    def variance(self):
        r = self._high - self._low
        return _make([r * r / 12])

    def sample(self, sample_shape=()):
        n = max(1, 1 if not sample_shape else eval('*'.join(str(s) for s in sample_shape)) if len(sample_shape) > 1 else sample_shape[0])
        vals = [_random.uniform(self._low, self._high) for _ in range(n)]
        return _make(vals, sample_shape if sample_shape else None)

    def rsample(self, sample_shape=()):
        return self.sample(sample_shape)

    def log_prob(self, value):
        vals = _flat(value)
        log_range = math.log(self._high - self._low)
        result = [-log_range if self._low <= v < self._high else float('-inf') for v in vals]
        return _make(result)

    def entropy(self):
        return _make([math.log(self._high - self._low)])


# ── Exponential ───────────────────────────────────────────────────────────────

class Exponential(Distribution):
    """Exponential distribution with rate parameter."""
    has_rsample = True

    def __init__(self, rate):
        self._rate = float(_flat(rate)[0])

    @property
    def mean(self):
        return _make([1.0 / self._rate])

    @property
    def variance(self):
        return _make([1.0 / (self._rate * self._rate)])

    def sample(self, sample_shape=()):
        n = max(1, sample_shape[0] if sample_shape else 1)
        vals = [_random.expovariate(self._rate) for _ in range(n)]
        return _make(vals, sample_shape if sample_shape else None)

    def rsample(self, sample_shape=()):
        return self.sample(sample_shape)

    def log_prob(self, value):
        vals = _flat(value)
        r = self._rate
        result = [math.log(r) - r * v if v >= 0 else float('-inf') for v in vals]
        return _make(result)

    def entropy(self):
        return _make([1.0 - math.log(self._rate)])


# ── Beta ──────────────────────────────────────────────────────────────────────

class Beta(Distribution):
    """Beta distribution parameterized by concentration1 (alpha) and concentration0 (beta)."""

    def __init__(self, concentration1, concentration0):
        self._alpha = float(_flat(concentration1)[0])
        self._beta = float(_flat(concentration0)[0])

    @property
    def mean(self):
        a, b = self._alpha, self._beta
        return _make([a / (a + b)])

    @property
    def variance(self):
        a, b = self._alpha, self._beta
        return _make([a * b / ((a + b)**2 * (a + b + 1))])

    def _lgamma(self, x):
        return math.lgamma(x)

    def sample(self, sample_shape=()):
        n = max(1, sample_shape[0] if sample_shape else 1)
        vals = [_random.betavariate(self._alpha, self._beta) for _ in range(n)]
        return _make(vals, sample_shape if sample_shape else None)

    def log_prob(self, value):
        vals = _flat(value)
        a, b = self._alpha, self._beta
        log_beta = self._lgamma(a) + self._lgamma(b) - self._lgamma(a + b)
        result = [(a-1)*math.log(max(1e-45, v)) + (b-1)*math.log(max(1e-45, 1-v)) - log_beta
                  for v in vals]
        return _make(result)

    def entropy(self):
        a, b = self._alpha, self._beta
        log_beta = self._lgamma(a) + self._lgamma(b) - self._lgamma(a + b)
        from math import digamma
        h = log_beta - (a-1)*digamma(a) - (b-1)*digamma(b) + (a+b-2)*digamma(a+b)
        return _make([h])


# ── Dirichlet ─────────────────────────────────────────────────────────────────

class Dirichlet(Distribution):
    """Dirichlet distribution over the simplex."""

    def __init__(self, concentration):
        self._alpha = _flat(concentration)

    @property
    def mean(self):
        s = sum(self._alpha)
        return _make([a / s for a in self._alpha])

    def sample(self, sample_shape=()):
        gammas = [_random.gammavariate(a, 1.0) for a in self._alpha]
        s = sum(gammas)
        vals = [g / s for g in gammas]
        return _make(vals)

    def log_prob(self, value):
        vals = _flat(value)
        alpha = self._alpha
        log_beta = sum(math.lgamma(a) for a in alpha) - math.lgamma(sum(alpha))
        lp = -log_beta + sum((a-1)*math.log(max(1e-45, v)) for a, v in zip(alpha, vals))
        return _make([lp])

    def entropy(self):
        alpha = self._alpha
        a0 = sum(alpha)
        K = len(alpha)
        from math import digamma
        log_beta = sum(math.lgamma(a) for a in alpha) - math.lgamma(a0)
        h = log_beta + (a0 - K) * digamma(a0) - sum((a-1)*digamma(a) for a in alpha)
        return _make([h])


class Poisson(Distribution):
    """Poisson distribution."""
    def __init__(self, rate):
        from vectorphile.core.tensor import Tensor as _T
        from vectorphile._core.ndarray import array as _nd
        self.rate = rate if isinstance(rate, _T) else _T(_nd([float(rate)]))

    def sample(self, sample_shape=()):
        import math as _m; import random as _rnd
        from vectorphile.core.tensor import Tensor as _T
        from vectorphile._core.ndarray import array as _nd
        lam = float(self.rate.item() if hasattr(self.rate, 'item') else self.rate._data.to_flat_list()[0])
        # Knuth algorithm
        L = _m.exp(-lam)
        n = int(1) if not sample_shape else int(sample_shape[0])
        vals = []
        for _ in range(n):
            k, p = 0, 1.0
            while p > L:
                k += 1
                p *= _rnd.random()
            vals.append(float(k - 1))
        nd = _nd(vals, dtype='float32')
        return _T(nd)

    def log_prob(self, x):
        import math as _m
        from vectorphile.core.tensor import Tensor as _T
        from vectorphile._core.ndarray import array as _nd
        lam = float(self.rate.item() if hasattr(self.rate, 'item') else self.rate._data.to_flat_list()[0])
        flat_x = x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data)
        lps = [k * _m.log(lam) - lam - _m.lgamma(k + 1) for k in flat_x]
        return _T(_nd(lps))

    @property
    def mean(self): return self.rate

    @property
    def variance(self): return self.rate


class Gamma(Distribution):
    """Gamma distribution."""
    def __init__(self, concentration, rate):
        from vectorphile.core.tensor import Tensor as _T
        from vectorphile._core.ndarray import array as _nd
        self.concentration = concentration if isinstance(concentration, _T) else _T(_nd([float(concentration)]))
        self.rate = rate if isinstance(rate, _T) else _T(_nd([float(rate)]))

    def sample(self, sample_shape=()):
        import random as _rnd
        from vectorphile.core.tensor import Tensor as _T
        from vectorphile._core.ndarray import array as _nd
        alpha = float(self.concentration._data.to_flat_list()[0])
        beta = float(self.rate._data.to_flat_list()[0])
        n = int(1) if not sample_shape else int(sample_shape[0])
        vals = [_rnd.gammavariate(alpha, 1.0 / beta) for _ in range(n)]
        return _T(_nd(vals, dtype='float32'))

    def log_prob(self, x):
        import math as _m
        from vectorphile.core.tensor import Tensor as _T
        from vectorphile._core.ndarray import array as _nd
        alpha = float(self.concentration._data.to_flat_list()[0])
        beta = float(self.rate._data.to_flat_list()[0])
        flat_x = [float(v) for v in (x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data))]
        lps = [(alpha - 1) * _m.log(max(v, 1e-30)) - beta * v + alpha * _m.log(beta) - _m.lgamma(alpha)
               for v in flat_x]
        return _T(_nd(lps))

    @property
    def mean(self):
        from vectorphile.core.tensor import Tensor as _T
        from vectorphile._core.ndarray import array as _nd
        alpha = float(self.concentration._data.to_flat_list()[0])
        beta = float(self.rate._data.to_flat_list()[0])
        return _T(_nd([alpha / beta]))

    @property
    def variance(self):
        from vectorphile.core.tensor import Tensor as _T
        from vectorphile._core.ndarray import array as _nd
        alpha = float(self.concentration._data.to_flat_list()[0])
        beta = float(self.rate._data.to_flat_list()[0])
        return _T(_nd([alpha / (beta ** 2)]))


class MultivariateNormal(Distribution):
    """Multivariate normal distribution."""
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None):
        self.loc = loc

    def sample(self, sample_shape=()):
        from vectorphile.core.tensor import randn, zeros
        n = int(1) if not sample_shape else int(sample_shape[0])
        d = self.loc.shape[-1] if hasattr(self.loc, 'shape') else 1
        return randn(n, d) + self.loc

    def log_prob(self, x):
        import math as _m
        d = self.loc.shape[-1] if hasattr(self.loc, 'shape') else 1
        diff = x - self.loc
        return -(d/2) * _m.log(2 * _m.pi) - 0.5 * (diff * diff).sum(dim=-1)

    @property
    def mean(self): return self.loc


class Laplace(Distribution):
    """Laplace distribution."""
    def __init__(self, loc, scale):
        from vectorphile.core.tensor import Tensor as _T
        from vectorphile._core.ndarray import array as _nd
        self.loc = loc if isinstance(loc, _T) else _T(_nd([float(loc)]))
        self.scale = scale if isinstance(scale, _T) else _T(_nd([float(scale)]))

    def sample(self, sample_shape=()):
        import random as _rnd
        from vectorphile.core.tensor import Tensor as _T
        from vectorphile._core.ndarray import array as _nd
        import math as _m
        mu = float(self.loc._data.to_flat_list()[0])
        b = float(self.scale._data.to_flat_list()[0])
        n = int(1) if not sample_shape else int(sample_shape[0])
        vals = [mu - b * (1 if _rnd.random() > 0.5 else -1) * _m.log(1 - _rnd.random()) for _ in range(n)]
        return _T(_nd(vals, dtype='float32'))

    def log_prob(self, x):
        import math as _m
        from vectorphile.core.tensor import Tensor as _T
        from vectorphile._core.ndarray import array as _nd
        mu = float(self.loc._data.to_flat_list()[0])
        b = float(self.scale._data.to_flat_list()[0])
        flat_x = [float(v) for v in (x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data))]
        lps = [-_m.log(2 * b) - abs(v - mu) / b for v in flat_x]
        return _T(_nd(lps))

    @property
    def mean(self): return self.loc

    @property
    def variance(self):
        return self.scale * self.scale * 2


class LogNormal(Distribution):
    """Log-normal distribution."""
    def __init__(self, loc, scale):
        from vectorphile.core.tensor import Tensor as _T
        from vectorphile._core.ndarray import array as _nd
        self.loc = loc if isinstance(loc, _T) else _T(_nd([float(loc)]))
        self.scale = scale if isinstance(scale, _T) else _T(_nd([float(scale)]))

    def sample(self, sample_shape=()):
        import random as _rnd; import math as _m
        from vectorphile.core.tensor import Tensor as _T
        from vectorphile._core.ndarray import array as _nd
        mu = float(self.loc._data.to_flat_list()[0])
        sigma = float(self.scale._data.to_flat_list()[0])
        n = int(1) if not sample_shape else int(sample_shape[0])
        vals = [_m.exp(_rnd.gauss(mu, sigma)) for _ in range(n)]
        return _T(_nd(vals, dtype='float32'))

    def log_prob(self, x):
        import math as _m
        from vectorphile.core.tensor import Tensor as _T
        from vectorphile._core.ndarray import array as _nd
        mu = float(self.loc._data.to_flat_list()[0])
        sigma = float(self.scale._data.to_flat_list()[0])
        flat_x = [float(v) for v in (x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data))]
        lps = [-(v + _m.log(sigma * _m.sqrt(2*_m.pi))) - ((_m.log(max(v, 1e-30)) - mu)**2)/(2*sigma**2) for v in flat_x]
        return _T(_nd(lps))

    @property
    def mean(self):
        import math as _m
        from vectorphile.core.tensor import Tensor as _T
        from vectorphile._core.ndarray import array as _nd
        mu = float(self.loc._data.to_flat_list()[0])
        sigma = float(self.scale._data.to_flat_list()[0])
        return _T(_nd([_m.exp(mu + sigma**2 / 2)]))


# ── Additional distributions ──────────────────────────────────────────────────

class StudentT(Distribution):
    """Student's t-distribution."""
    has_rsample = True

    def __init__(self, df, loc=0.0, scale=1.0):
        self.df = float(df)
        self.loc = float(loc)
        self.scale = float(scale)

    def sample(self, sample_shape=()):
        import math as _m
        n = 1
        for s in (sample_shape if isinstance(sample_shape, (tuple, list)) else (sample_shape,)):
            n *= int(s)
        # Use ratio of Normal to sqrt(Chi2/df)
        vals = []
        for _ in range(n):
            z = _random.gauss(0, 1)
            chi2 = sum(_random.gauss(0, 1)**2 for _ in range(int(self.df)))
            t = z / _math.sqrt(chi2 / self.df) * self.scale + self.loc
            vals.append(t)
        return _make(vals, sample_shape if isinstance(sample_shape, tuple) else (sample_shape,))

    def log_prob(self, value):
        import math as _m
        df, loc, scale = self.df, self.loc, self.scale
        flat = _flat(value)
        lp = []
        log_norm = _m.lgamma((df + 1) / 2) - _m.lgamma(df / 2) - 0.5 * _m.log(df * _m.pi) - _m.log(scale)
        for v in flat:
            z = (v - loc) / scale
            lp.append(log_norm - (df + 1) / 2 * _m.log(1 + z * z / df))
        return _make(lp, _nd(flat).shape if hasattr(value, '_data') else (len(flat),))

    @property
    def mean(self): return _make([self.loc]) if self.df > 1 else _make([float('nan')])

    @property
    def variance(self):
        if self.df > 2: return _make([self.df / (self.df - 2) * self.scale**2])
        return _make([float('inf') if self.df > 1 else float('nan')])


class Chi2(Distribution):
    """Chi-squared distribution (Gamma with scale=2, concentration=df/2)."""
    has_rsample = True

    def __init__(self, df):
        self.df = float(df)
        self._gamma = Gamma(df / 2, 0.5)

    def sample(self, sample_shape=()):
        return self._gamma.sample(sample_shape)

    def log_prob(self, value): return self._gamma.log_prob(value)

    @property
    def mean(self): return _make([self.df])

    @property
    def variance(self): return _make([2 * self.df])


class Geometric(Distribution):
    """Geometric distribution (number of failures before first success)."""

    def __init__(self, probs=None, logits=None):
        if probs is not None:
            self.probs = float(probs)
        else:
            self.probs = 1.0 / (1.0 + _math.exp(-float(logits)))

    def sample(self, sample_shape=()):
        n = 1
        for s in (sample_shape if isinstance(sample_shape, (tuple, list)) else (sample_shape,)):
            n *= int(s)
        vals = [_math.floor(_math.log(1 - _random.random()) / _math.log(1 - self.probs)) for _ in range(n)]
        return _make(vals, sample_shape if isinstance(sample_shape, tuple) else (sample_shape,))

    def log_prob(self, value):
        p = self.probs
        flat = _flat(value)
        lp = [_math.log(1 - p) * float(v) + _math.log(p) for v in flat]
        return _make(lp)

    @property
    def mean(self): return _make([(1 - self.probs) / self.probs])

    @property
    def variance(self): return _make([(1 - self.probs) / self.probs**2])


class Binomial(Distribution):
    """Binomial distribution."""

    def __init__(self, total_count, probs=None, logits=None):
        self.total_count = int(total_count)
        if probs is not None:
            self.probs = float(probs)
        else:
            self.probs = 1.0 / (1.0 + _math.exp(-float(logits)))

    def sample(self, sample_shape=()):
        n = 1
        for s in (sample_shape if isinstance(sample_shape, (tuple, list)) else (sample_shape,)):
            n *= int(s)
        vals = [float(sum(1 for _ in range(self.total_count) if _random.random() < self.probs)) for _ in range(n)]
        return _make(vals, sample_shape if isinstance(sample_shape, tuple) else (sample_shape,))

    def log_prob(self, value):
        import math as _m
        n, p = self.total_count, self.probs
        flat = _flat(value)
        lp = []
        for v in flat:
            k = int(v)
            log_binom = _m.lgamma(n + 1) - _m.lgamma(k + 1) - _m.lgamma(n - k + 1)
            lp.append(log_binom + k * _m.log(p + 1e-10) + (n - k) * _m.log(1 - p + 1e-10))
        return _make(lp)

    @property
    def mean(self): return _make([self.total_count * self.probs])

    @property
    def variance(self): return _make([self.total_count * self.probs * (1 - self.probs)])


class Cauchy(Distribution):
    """Cauchy distribution."""
    has_rsample = True

    def __init__(self, loc=0.0, scale=1.0):
        self.loc = float(loc)
        self.scale = float(scale)

    def sample(self, sample_shape=()):
        n = 1
        for s in (sample_shape if isinstance(sample_shape, (tuple, list)) else (sample_shape,)):
            n *= int(s)
        vals = [self.loc + self.scale * _math.tan(_math.pi * (_random.random() - 0.5)) for _ in range(n)]
        return _make(vals, sample_shape if isinstance(sample_shape, tuple) else (sample_shape,))

    def log_prob(self, value):
        import math as _m
        loc, scale = self.loc, self.scale
        flat = _flat(value)
        log_norm = -_m.log(_m.pi * scale)
        lp = [log_norm - _m.log(1 + ((v - loc) / scale)**2) for v in flat]
        return _make(lp)

    @property
    def mean(self): return _make([float('nan')])

    @property
    def variance(self): return _make([float('inf')])


class Gumbel(Distribution):
    """Gumbel (extreme value type I) distribution."""
    has_rsample = True

    def __init__(self, loc=0.0, scale=1.0):
        self.loc = float(loc)
        self.scale = float(scale)

    def sample(self, sample_shape=()):
        n = 1
        for s in (sample_shape if isinstance(sample_shape, (tuple, list)) else (sample_shape,)):
            n *= int(s)
        # Inverse CDF: loc - scale * log(-log(U))
        vals = [self.loc - self.scale * _math.log(-_math.log(max(1e-30, _random.random()))) for _ in range(n)]
        return _make(vals, sample_shape if isinstance(sample_shape, tuple) else (sample_shape,))

    def log_prob(self, value):
        import math as _m
        loc, scale = self.loc, self.scale
        flat = _flat(value)
        lp = []
        for v in flat:
            z = (v - loc) / scale
            lp.append(-z - _m.exp(-z) - _m.log(scale))
        return _make(lp)

    @property
    def mean(self): return _make([self.loc + self.scale * 0.5772156649015329])

    @property
    def variance(self): return _make([(_math.pi * self.scale)**2 / 6])


class HalfNormal(Distribution):
    """Half-normal distribution (truncated normal at zero)."""
    has_rsample = True

    def __init__(self, scale=1.0):
        self.scale = float(scale)

    def sample(self, sample_shape=()):
        n = 1
        for s in (sample_shape if isinstance(sample_shape, (tuple, list)) else (sample_shape,)):
            n *= int(s)
        vals = [abs(_random.gauss(0, self.scale)) for _ in range(n)]
        return _make(vals, sample_shape if isinstance(sample_shape, tuple) else (sample_shape,))

    def log_prob(self, value):
        import math as _m
        scale = self.scale
        flat = _flat(value)
        log_norm = _m.log(2) - _m.log(scale) - 0.5 * _m.log(2 * _m.pi)
        lp = [log_norm - 0.5 * (v / scale)**2 if v >= 0 else float('-inf') for v in flat]
        return _make(lp)

    @property
    def mean(self): return _make([self.scale * _math.sqrt(2 / _math.pi)])

    @property
    def variance(self): return _make([self.scale**2 * (1 - 2 / _math.pi)])


class Weibull(Distribution):
    """Weibull distribution."""
    has_rsample = True

    def __init__(self, scale=1.0, concentration=1.0):
        self.scale = float(scale)
        self.concentration = float(concentration)

    def sample(self, sample_shape=()):
        n = 1
        for s in (sample_shape if isinstance(sample_shape, (tuple, list)) else (sample_shape,)):
            n *= int(s)
        # Inverse CDF
        vals = [self.scale * (-_math.log(max(1e-30, _random.random())))**(1 / self.concentration) for _ in range(n)]
        return _make(vals, sample_shape if isinstance(sample_shape, tuple) else (sample_shape,))

    def log_prob(self, value):
        import math as _m
        k, lam = self.concentration, self.scale
        flat = _flat(value)
        lp = []
        for v in flat:
            if v < 0:
                lp.append(float('-inf'))
            else:
                lp.append(_m.log(k) - _m.log(lam) + (k - 1) * _m.log(max(v, 1e-30) / lam) - (v / lam)**k)
        return _make(lp)

    @property
    def mean(self): return _make([self.scale * _math.gamma(1 + 1 / self.concentration)])

    @property
    def variance(self):
        k, lam = self.concentration, self.scale
        return _make([lam**2 * (_math.gamma(1 + 2 / k) - _math.gamma(1 + 1 / k)**2)])


# ── kl_divergence registry ────────────────────────────────────────────────────

def kl_divergence(p, q):
    """KL(p || q) for supported distribution pairs."""
    if isinstance(p, Normal) and isinstance(q, Normal):
        return p.kl_divergence(q)
    raise NotImplementedError(f"kl_divergence not implemented for {type(p)} || {type(q)}")


__all__ = [
    'Distribution',
    'Normal', 'Bernoulli', 'Categorical', 'Uniform',
    'Exponential', 'Beta', 'Dirichlet',
    'Poisson', 'Gamma', 'MultivariateNormal', 'Laplace', 'LogNormal',
    'StudentT', 'Chi2', 'Geometric', 'Binomial', 'Cauchy', 'Gumbel',
    'HalfNormal', 'Weibull',
    'kl_divergence',
]
