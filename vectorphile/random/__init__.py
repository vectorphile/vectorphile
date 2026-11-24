"""Random number generation — pure Python, no external dependencies."""
import random as _random
import math
from vectorphile.core.tensor import Tensor
from vectorphile._core.ndarray import NDArray, zeros, randn_nd, uniform as _nd_uniform, array as _nd_array

_rng = _random.Random()

def seed(s: int):
    global _rng
    _rng = _random.Random(s)


def _box_muller() -> float:
    u1 = _rng.random()
    u2 = _rng.random()
    return math.sqrt(-2 * math.log(max(u1, 1e-10))) * math.cos(2 * math.pi * u2)


def _shape_size(shape) -> int:
    s = 1
    for d in shape:
        s *= d
    return s


def _nd_randn(shape):
    n = _shape_size(shape)
    flat = [_box_muller() for _ in range(n)]
    return _nd_array(flat).reshape(shape) if hasattr(_nd_array(flat), 'reshape') else _nd_array(flat)


def _nd_rand(shape):
    n = _shape_size(shape)
    flat = [_rng.random() for _ in range(n)]
    nd = _nd_array(flat)
    if hasattr(nd, 'reshape'):
        return nd.reshape(shape)
    return nd


def rand(*shape, device='cpu') -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(_nd_rand(shape), device=device)


def randn(*shape, device='cpu') -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return Tensor(_nd_randn(shape), device=device)


def randint(low, high, shape, device='cpu') -> Tensor:
    if isinstance(shape, int):
        shape = (shape,)
    n = _shape_size(shape)
    flat = [float(_rng.randint(low, high - 1)) for _ in range(n)]
    nd = _nd_array(flat)
    if hasattr(nd, 'reshape'):
        nd = nd.reshape(shape)
    return Tensor(nd, device=device)


def choice(a, size=None, replace=True, p=None) -> Tensor:
    if isinstance(a, Tensor):
        vals = a._data.to_flat_list() if hasattr(a._data, 'to_flat_list') else list(a._data)
    elif hasattr(a, 'to_flat_list'):
        vals = a.to_flat_list()
    else:
        vals = list(range(a)) if isinstance(a, int) else list(a)
    n = len(vals)
    if size is None:
        if p is not None:
            r = _rng.random()
            cumsum = 0.0
            for i, prob in enumerate(p):
                cumsum += prob
                if r <= cumsum:
                    return Tensor(_nd_array([float(vals[i])]))
        return Tensor(_nd_array([float(_rng.choice(vals))]))
    total = size if isinstance(size, int) else _shape_size(size)
    if p is not None:
        result = []
        for _ in range(total):
            r = _rng.random()
            cumsum = 0.0
            for i, prob in enumerate(p):
                cumsum += prob
                if r <= cumsum:
                    result.append(float(vals[i]))
                    break
            else:
                result.append(float(vals[-1]))
    elif replace:
        result = [float(_rng.choice(vals)) for _ in range(total)]
    else:
        result = [float(v) for v in _rng.sample(vals, total)]
    nd = _nd_array(result)
    if hasattr(nd, 'reshape') and not isinstance(size, int):
        nd = nd.reshape(size)
    return Tensor(nd)


def shuffle(x: Tensor) -> Tensor:
    vals = x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data.tolist() if hasattr(x._data, 'tolist') else x._data)
    _rng.shuffle(vals)
    nd = _nd_array(vals)
    if hasattr(x._data, 'shape') and hasattr(nd, 'reshape'):
        nd = nd.reshape(x._data.shape)
    return Tensor(nd)


def normal(loc=0.0, scale=1.0, size=None, device='cpu') -> Tensor:
    if size is None:
        return Tensor(_nd_array([loc + scale * _box_muller()]), device=device)
    if isinstance(size, int):
        size = (size,)
    n = _shape_size(size)
    flat = [loc + scale * _box_muller() for _ in range(n)]
    nd = _nd_array(flat)
    if hasattr(nd, 'reshape'):
        nd = nd.reshape(size)
    return Tensor(nd, device=device)


def uniform(low=0.0, high=1.0, size=None, device='cpu') -> Tensor:
    if size is None:
        return Tensor(_nd_array([_rng.uniform(low, high)]), device=device)
    if isinstance(size, int):
        size = (size,)
    n = _shape_size(size)
    flat = [_rng.uniform(low, high) for _ in range(n)]
    nd = _nd_array(flat)
    if hasattr(nd, 'reshape'):
        nd = nd.reshape(size)
    return Tensor(nd, device=device)


def binomial(n, p, size=None) -> Tensor:
    def _binom_sample():
        return float(sum(1 for _ in range(n) if _rng.random() < p))
    if size is None:
        return Tensor(_nd_array([_binom_sample()]))
    if isinstance(size, int):
        size = (size,)
    total = _shape_size(size)
    flat = [_binom_sample() for _ in range(total)]
    nd = _nd_array(flat)
    if hasattr(nd, 'reshape'):
        nd = nd.reshape(size)
    return Tensor(nd)


def poisson(lam=1.0, size=None) -> Tensor:
    def _pois():
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            p *= _rng.random()
            k += 1
        return float(k - 1)
    if size is None:
        return Tensor(_nd_array([_pois()]))
    if isinstance(size, int):
        size = (size,)
    flat = [_pois() for _ in range(_shape_size(size))]
    nd = _nd_array(flat)
    if hasattr(nd, 'reshape'):
        nd = nd.reshape(size)
    return Tensor(nd)


def exponential(scale=1.0, size=None) -> Tensor:
    def _exp():
        return -scale * math.log(max(_rng.random(), 1e-10))
    if size is None:
        return Tensor(_nd_array([_exp()]))
    if isinstance(size, int):
        size = (size,)
    flat = [_exp() for _ in range(_shape_size(size))]
    nd = _nd_array(flat)
    if hasattr(nd, 'reshape'):
        nd = nd.reshape(size)
    return Tensor(nd)


def multinomial(inp: Tensor, num_samples: int, replacement=False) -> Tensor:
    probs_flat = inp._data.to_flat_list() if hasattr(inp._data, 'to_flat_list') else list(inp._data.tolist() if hasattr(inp._data, 'tolist') else inp._data)
    total = sum(probs_flat)
    probs_norm = [v / total for v in probs_flat]
    def _sample():
        r = _rng.random()
        cumsum = 0.0
        for i, p in enumerate(probs_norm):
            cumsum += p
            if r <= cumsum:
                return float(i)
        return float(len(probs_norm) - 1)
    result = [_sample() for _ in range(num_samples)]
    return Tensor(_nd_array(result))


def permutation(x) -> Tensor:
    if isinstance(x, int):
        perm = list(range(x))
        _rng.shuffle(perm)
        return Tensor(_nd_array([float(v) for v in perm]))
    vals = x._data.to_flat_list() if hasattr(x, '_data') and hasattr(x._data, 'to_flat_list') else list(x)
    _rng.shuffle(vals)
    return Tensor(_nd_array([float(v) for v in vals]))


def bernoulli(p: Tensor) -> Tensor:
    vals = p._data.to_flat_list() if hasattr(p._data, 'to_flat_list') else list(p._data.tolist() if hasattr(p._data, 'tolist') else p._data)
    result = [1.0 if _rng.random() < v else 0.0 for v in vals]
    nd = _nd_array(result)
    if hasattr(p._data, 'shape') and hasattr(nd, 'reshape'):
        nd = nd.reshape(p._data.shape)
    return Tensor(nd)


def _gamma_sample(alpha: float, beta_scale: float = 1.0) -> float:
    """Marsaglia-Tsang fast gamma sampler."""
    if alpha < 1.0:
        return _gamma_sample(alpha + 1.0, beta_scale) * (_rng.random() ** (1.0 / alpha))
    d = alpha - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = _rng.gauss(0, 1)
        v = (1.0 + c * x) ** 3
        if v > 0:
            u = _rng.random()
            if u < 1.0 - 0.0331 * (x * x) ** 2:
                return d * v * beta_scale
            if math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
                return d * v * beta_scale


def gamma(shape_param, scale=1.0, size=None) -> Tensor:
    """Sample from Gamma(shape, scale) distribution."""
    n = 1
    out_shape = None
    if size is not None:
        if isinstance(size, int):
            size = (size,)
        for s in size:
            n *= s
        out_shape = tuple(size)
    result = [_gamma_sample(float(shape_param), float(scale)) for _ in range(n)]
    nd = _nd_array(result)
    if out_shape:
        nd = nd.reshape(out_shape)
    return Tensor(nd)


def beta(alpha, beta_param, size=None) -> Tensor:
    """Sample from Beta(alpha, beta) distribution via ratio of gammas."""
    n = 1
    out_shape = None
    if size is not None:
        if isinstance(size, int):
            size = (size,)
        for s in size:
            n *= s
        out_shape = tuple(size)
    result = []
    for _ in range(n):
        x = _gamma_sample(float(alpha))
        y = _gamma_sample(float(beta_param))
        result.append(x / max(x + y, 1e-10))
    nd = _nd_array(result)
    if out_shape:
        nd = nd.reshape(out_shape)
    return Tensor(nd)


def dirichlet(alpha, size=None) -> Tensor:
    """Sample from Dirichlet(alpha) distribution. alpha is a list of concentrations."""
    alpha_list = list(alpha) if hasattr(alpha, '__iter__') else [float(alpha)]
    K = len(alpha_list)
    n = size if isinstance(size, int) else (size[0] if size else 1)
    result = []
    for _ in range(n):
        samples = [_gamma_sample(float(a)) for a in alpha_list]
        s = sum(samples)
        result.extend([v / max(s, 1e-10) for v in samples])
    nd = _nd_array(result)
    nd = nd.reshape((n, K))
    if size is None:
        nd = nd.reshape((K,))
    return Tensor(nd)


def cauchy(loc=0.0, scale=1.0, size=None) -> Tensor:
    """Sample from Cauchy(loc, scale) distribution: loc + scale * tan(pi * (u - 0.5))."""
    n = 1
    out_shape = None
    if size is not None:
        if isinstance(size, int):
            size = (size,)
        for s in size:
            n *= s
        out_shape = tuple(size)
    result = [loc + scale * math.tan(math.pi * (_rng.random() - 0.5)) for _ in range(n)]
    nd = _nd_array(result)
    if out_shape:
        nd = nd.reshape(out_shape)
    return Tensor(nd)


def laplace(loc=0.0, scale=1.0, size=None) -> Tensor:
    """Sample from Laplace(loc, scale) distribution."""
    n = 1
    out_shape = None
    if size is not None:
        if isinstance(size, int):
            size = (size,)
        for s in size:
            n *= s
        out_shape = tuple(size)
    result = []
    for _ in range(n):
        u = _rng.random() - 0.5
        result.append(loc - scale * math.copysign(1, u) * math.log(1 - 2 * abs(u)))
    nd = _nd_array(result)
    if out_shape:
        nd = nd.reshape(out_shape)
    return Tensor(nd)


def gumbel(loc=0.0, scale=1.0, size=None) -> Tensor:
    """Sample from Gumbel(loc, scale) distribution: loc - scale * log(-log(u))."""
    n = 1
    out_shape = None
    if size is not None:
        if isinstance(size, int):
            size = (size,)
        for s in size:
            n *= s
        out_shape = tuple(size)
    result = [loc - scale * math.log(-math.log(max(_rng.random(), 1e-10))) for _ in range(n)]
    nd = _nd_array(result)
    if out_shape:
        nd = nd.reshape(out_shape)
    return Tensor(nd)


def weibull(scale=1.0, concentration=1.0, size=None) -> Tensor:
    """Sample from Weibull(scale, concentration) distribution."""
    n = 1
    out_shape = None
    if size is not None:
        if isinstance(size, int):
            size = (size,)
        for s in size:
            n *= s
        out_shape = tuple(size)
    result = [scale * (-math.log(max(_rng.random(), 1e-10))) ** (1.0 / concentration)
              for _ in range(n)]
    nd = _nd_array(result)
    if out_shape:
        nd = nd.reshape(out_shape)
    return Tensor(nd)


def chi2(df, size=None) -> Tensor:
    """Sample from chi-squared(df) distribution via sum of squared normals."""
    return gamma(df / 2.0, scale=2.0, size=size)


def student_t(df, loc=0.0, scale=1.0, size=None) -> Tensor:
    """Sample from Student's t(df) distribution via normal / sqrt(chi2/df)."""
    n = 1
    out_shape = None
    if size is not None:
        if isinstance(size, int):
            size = (size,)
        for s in size:
            n *= s
        out_shape = tuple(size)
    result = []
    for _ in range(n):
        z = _rng.gauss(0, 1)
        v = _gamma_sample(df / 2.0, 2.0)
        result.append(loc + scale * z / math.sqrt(max(v / df, 1e-10)))
    nd = _nd_array(result)
    if out_shape:
        nd = nd.reshape(out_shape)
    return Tensor(nd)
