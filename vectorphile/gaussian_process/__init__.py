"""vectorphile.gaussian_process — Gaussian Process models.

Provides:
- GaussianProcessRegressor (GPR) with RBF/Matern/Linear/Polynomial kernels
- GaussianProcessClassifier (GPC) one-vs-rest
- Kernels: RBF, Matern, DotProduct, WhiteKernel, ConstantKernel, Sum, Product

All computations use vectorphile's own NDArray and Tensor primitives.
No external dependencies.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple

from vectorphile._core.ndarray import (
    NDArray as _NDA,
    zeros as _nd_zeros, ones as _nd_ones, eye as _nd_eye, full as _nd_full,
    exp as _nd_exp, log as _nd_log, sqrt as _nd_sqrt,
    randn_nd as _nd_randn,
    array as _nd_array,
    _c_strides as _nd_c_strides,
    _DType as _NDType,
    matmul as _nd_matmul,
)
import array as _pyarray
from vectorphile.core.tensor import Tensor
from vectorphile import linalg as _vp_linalg


def _to_nd(X) -> _NDA:
    """Convert input to float32 NDArray."""
    if isinstance(X, _NDA):
        return X.astype('float32')
    if isinstance(X, Tensor):
        return X._data.astype('float32')
    # Python list / nested list
    flat = []
    def _fl(v):
        if isinstance(v, (list, tuple)):
            for x in v: _fl(x)
        else: flat.append(float(v))
    if isinstance(X, (list, tuple)):
        _fl(X)
        # infer shape
        shape = []
        cur = X
        while isinstance(cur, (list, tuple)) and len(cur) > 0:
            shape.append(len(cur))
            cur = cur[0]
        shape = tuple(shape)
        import array as _a
        buf = _a.array('f', flat)
        return _NDA(buf, shape, _nd_c_strides(shape), _NDType('float32'))
    # Fallback: try NDArray constructor via array module
    return _nd_array(X, dtype='float32')


def _nd_to_tensor(arr: _NDA) -> Tensor:
    return Tensor(arr)


def _tensor_to_nd(t: Tensor) -> _NDA:
    return t._data


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

class Kernel:
    """Base kernel class."""

    def __call__(self, X: _NDA, Y: Optional[_NDA] = None) -> _NDA:
        raise NotImplementedError

    def __add__(self, other: 'Kernel') -> 'Sum':
        return Sum(self, other)

    def __mul__(self, other) -> 'Product':
        if isinstance(other, (int, float)):
            return Product(ConstantKernel(float(other)), self)
        return Product(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)


def _sq_dists(X: _NDA, Y: _NDA) -> _NDA:
    """Compute pairwise squared distances ||x-y||²."""
    # ||x-y||² = ||x||² + ||y||² - 2*x·y
    X_sq = (X * X).sum(axis=1, keepdims=True)   # (n, 1)
    Y_sq = (Y * Y).sum(axis=1, keepdims=True)   # (m, 1)
    XY = _nd_matmul(X, Y.transpose())            # (n, m)
    sq = X_sq + Y_sq.transpose() - XY * 2.0
    # clamp negatives due to float rounding
    _zero = _nd_zeros(sq.shape)
    # element-wise max with 0
    flat = sq.to_flat_list()
    clamped = _pyarray.array('f', [max(0.0, v) for v in flat])
    return _NDA(clamped, sq.shape, _nd_c_strides(sq.shape), _NDType('float32'))


class RBF(Kernel):
    """RBF kernel: k(x,y) = exp(-||x-y||² / (2*l²))."""

    def __init__(self, length_scale: float = 1.0):
        self.length_scale = length_scale

    def __call__(self, X: _NDA, Y: Optional[_NDA] = None) -> _NDA:
        X = _to_nd(X)
        Y = X if Y is None else _to_nd(Y)
        sq = _sq_dists(X, Y)
        return _nd_exp(sq * (-1.0 / (2.0 * self.length_scale ** 2)))

    def get_params(self) -> dict:
        return {'length_scale': self.length_scale}


class Matern(Kernel):
    """Matern kernel (nu=0.5, 1.5, 2.5)."""

    def __init__(self, length_scale: float = 1.0, nu: float = 1.5):
        self.length_scale = length_scale
        self.nu = nu

    def __call__(self, X: _NDA, Y: Optional[_NDA] = None) -> _NDA:
        X = _to_nd(X)
        Y = X if Y is None else _to_nd(Y)
        sq = _sq_dists(X, Y)
        # r = sqrt(sq) / length_scale
        r_flat = [math.sqrt(v) / self.length_scale for v in sq.to_flat_list()]
        r_buf = _pyarray.array('f', r_flat)
        r = _NDA(r_buf, sq.shape, _nd_c_strides(sq.shape), _NDType('float32'))
        s3 = math.sqrt(3)
        s5 = math.sqrt(5)
        if self.nu == 0.5:
            return _nd_exp(r * -1.0)
        elif self.nu == 1.5:
            return (_nd_full(r.shape, 1.0) + r * s3) * _nd_exp(r * -s3)
        elif self.nu == 2.5:
            return (_nd_full(r.shape, 1.0) + r * s5 + r * r * (5.0 / 3.0)) * _nd_exp(r * -s5)
        else:
            return _nd_exp(r * r * -0.5)

    def get_params(self) -> dict:
        return {'length_scale': self.length_scale, 'nu': self.nu}


class DotProduct(Kernel):
    """Linear kernel: k(x,y) = sigma_0² + x·y."""

    def __init__(self, sigma_0: float = 1.0):
        self.sigma_0 = sigma_0

    def __call__(self, X: _NDA, Y: Optional[_NDA] = None) -> _NDA:
        X = _to_nd(X)
        Y = X if Y is None else _to_nd(Y)
        return _nd_matmul(X, Y.transpose()) + self.sigma_0 ** 2

    def get_params(self) -> dict:
        return {'sigma_0': self.sigma_0}


class WhiteKernel(Kernel):
    """White noise kernel: k(x,y) = noise * I(x==y)."""

    def __init__(self, noise_level: float = 1.0):
        self.noise_level = noise_level

    def __call__(self, X: _NDA, Y: Optional[_NDA] = None) -> _NDA:
        X = _to_nd(X)
        if Y is None:
            return _nd_eye(len(X)) * self.noise_level
        Y = _to_nd(Y)
        return _nd_zeros((len(X), len(Y)))

    def get_params(self) -> dict:
        return {'noise_level': self.noise_level}


class ConstantKernel(Kernel):
    """Constant kernel: k(x,y) = c."""

    def __init__(self, constant_value: float = 1.0):
        self.constant_value = constant_value

    def __call__(self, X: _NDA, Y: Optional[_NDA] = None) -> _NDA:
        X = _to_nd(X)
        Y = X if Y is None else _to_nd(Y)
        return _nd_full((len(X), len(Y)), self.constant_value)

    def get_params(self) -> dict:
        return {'constant_value': self.constant_value}


class Sum(Kernel):
    def __init__(self, k1: Kernel, k2: Kernel):
        self.k1 = k1; self.k2 = k2
    def __call__(self, X, Y=None) -> _NDA:
        return self.k1(X, Y) + self.k2(X, Y)


class Product(Kernel):
    def __init__(self, k1: Kernel, k2: Kernel):
        self.k1 = k1; self.k2 = k2
    def __call__(self, X, Y=None) -> _NDA:
        return self.k1(X, Y) * self.k2(X, Y)


# ---------------------------------------------------------------------------
# Cholesky and triangular solve (pure Python using NDArray)
# ---------------------------------------------------------------------------

def _cholesky(A: _NDA) -> _NDA:
    """Cholesky decomposition L such that A = L @ L.T (lower triangular)."""
    t = Tensor(A)
    L = _vp_linalg.cholesky(t)
    return _tensor_to_nd(L)


def _tri_solve(L: _NDA, b: _NDA) -> _NDA:
    """Solve L @ x = b for x (L is lower triangular)."""
    tL = Tensor(L)
    tb = Tensor(b)
    # solve L @ x = b using vectorphile.linalg.solve
    x = _vp_linalg.solve(tL, tb)
    return _tensor_to_nd(x)


# ---------------------------------------------------------------------------
# GaussianProcessRegressor
# ---------------------------------------------------------------------------

class GaussianProcessRegressor:
    """Gaussian Process Regression using vectorphile NDArray ops."""

    def __init__(self, kernel: Optional[Kernel] = None,
                 alpha: float = 1e-10,
                 normalize_y: bool = False,
                 random_state: Optional[int] = None):
        self.kernel = kernel or RBF(1.0)
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.random_state = random_state
        self.X_train_: Optional[_NDA] = None
        self.y_train_: Optional[_NDA] = None
        self.alpha_: Optional[_NDA] = None
        self.L_: Optional[_NDA] = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    def fit(self, X, y) -> 'GaussianProcessRegressor':
        self.X_train_ = _to_nd(X)
        y_nd = _to_nd(y).ravel()
        self.y_train_ = y_nd
        if self.normalize_y:
            self._y_mean = float(y_nd.mean())
            flat = y_nd.to_flat_list()
            self._y_std = math.sqrt(sum((v - self._y_mean) ** 2 for v in flat) / len(flat)) + 1e-10
            y_flat = [(v - self._y_mean) / self._y_std for v in flat]
        else:
            self._y_mean = 0.0
            self._y_std = 1.0
            y_flat = y_nd.to_flat_list()
        n = self.X_train_.shape[0]
        K = self.kernel(self.X_train_) + _nd_eye(n) * self.alpha
        try:
            self.L_ = _cholesky(K)
            # alpha_ = L^{-T} L^{-1} y
            y_arr = _pyarray.array('f', y_flat)
            y_col = _NDA(y_arr, (n, 1), _nd_c_strides((n, 1)), _NDType('float32'))
            v = _tri_solve(self.L_, y_col)
            self.alpha_ = _tri_solve(self.L_.transpose(), v)
        except Exception:
            K_reg = K + _nd_eye(n) * 1e-6
            self.L_ = _cholesky(K_reg)
            y_arr = _pyarray.array('f', y_flat)
            y_col = _NDA(y_arr, (n, 1), _nd_c_strides((n, 1)), _NDType('float32'))
            v = _tri_solve(self.L_, y_col)
            self.alpha_ = _tri_solve(self.L_.transpose(), v)
        return self

    def predict(self, X, return_std: bool = False, return_cov: bool = False):
        X = _to_nd(X)
        K_trans = self.kernel(X, self.X_train_)   # (n_test, n_train)
        y_mean = _nd_matmul(K_trans, self.alpha_).ravel()
        y_mean = y_mean * self._y_std + self._y_mean
        if not (return_std or return_cov):
            return y_mean
        # V = L^{-1} K_trans.T
        V = _tri_solve(self.L_, K_trans.transpose())  # (n_train, n_test)
        K_diag_flat = [float(self.kernel(_to_nd([X[i].to_flat_list()]), _to_nd([X[i].to_flat_list()]))[0, 0])
                       if X.ndim > 1 else 0.0
                       for i in range(X.shape[0])]
        K_diag_arr = _pyarray.array('f', K_diag_flat)
        K_diag = _NDA(K_diag_arr, (len(K_diag_flat),), _nd_c_strides((len(K_diag_flat),)), _NDType('float32'))
        v2_flat = [(V * V).sum(axis=0).to_flat_list()[i] if hasattr((V * V).sum(axis=0), 'to_flat_list')
                   else 0.0 for i in range(len(K_diag_flat))]
        y_var_flat = [max(0.0, K_diag_flat[i] - v2_flat[i]) * self._y_std ** 2
                      for i in range(len(K_diag_flat))]
        y_var_arr = _pyarray.array('f', y_var_flat)
        y_var = _NDA(y_var_arr, (len(y_var_flat),), _nd_c_strides((len(y_var_flat),)), _NDType('float32'))
        y_std = _NDA(_pyarray.array('f', [math.sqrt(v) for v in y_var_flat]),
                     y_var.shape, _nd_c_strides(y_var.shape), _NDType('float32'))
        if return_cov:
            K_cov = self.kernel(X) - _nd_matmul(V.transpose(), V)
            return y_mean, K_cov * self._y_std ** 2
        return y_mean, y_std

    def sample_y(self, X, n_samples: int = 1, random_state: Optional[int] = None) -> _NDA:
        # Simple: return mean + noise scaled by std
        y_mean, y_std = self.predict(_to_nd(X), return_std=True)
        samples_list = []
        for _ in range(n_samples):
            noise = _nd_randn(*y_mean.shape)
            samples_list.append((y_mean + noise * y_std).to_flat_list())
        n_test = y_mean.shape[0]
        flat = []
        for s in samples_list:
            flat.extend(s)
        result_arr = _pyarray.array('f', flat)
        # shape: (n_test, n_samples)
        result = _NDA(result_arr, (n_samples, n_test), _nd_c_strides((n_samples, n_test)), _NDType('float32'))
        return result.transpose()

    def log_marginal_likelihood(self) -> float:
        if self.L_ is None or self.alpha_ is None:
            return -1e38
        y_flat = self.y_train_.to_flat_list()
        n = len(y_flat)
        y_arr = _pyarray.array('f', y_flat)
        y_nd = _NDA(y_arr, (n,), _nd_c_strides((n,)), _NDType('float32'))
        alpha_flat = self.alpha_.ravel().to_flat_list()
        data_fit = -0.5 * sum(y_flat[i] * alpha_flat[i] for i in range(n))
        # log det via diagonal of L
        L_diag = self.L_.diagonal().to_flat_list()
        complexity = -sum(math.log(abs(v) + 1e-10) for v in L_diag)
        norm_const = -0.5 * n * math.log(2 * math.pi)
        return data_fit + complexity + norm_const

    def score(self, X, y) -> float:
        preds_nd = self.predict(_to_nd(X))
        preds = preds_nd.to_flat_list()
        y_true = _to_nd(y).ravel().to_flat_list()
        mean_y = sum(y_true) / len(y_true)
        ss_res = sum((y_true[i] - preds[i]) ** 2 for i in range(len(y_true)))
        ss_tot = sum((v - mean_y) ** 2 for v in y_true) + 1e-10
        return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# GaussianProcessClassifier
# ---------------------------------------------------------------------------

class GaussianProcessClassifier:
    """GP Classification via one-vs-rest."""

    def __init__(self, kernel: Optional[Kernel] = None,
                 random_state: Optional[int] = None):
        self.kernel = kernel or RBF(1.0)
        self.random_state = random_state
        self.classes_: Optional[list] = None
        self.gprs_: list = []

    def fit(self, X, y) -> 'GaussianProcessClassifier':
        X_nd = _to_nd(X)
        y_flat = _to_nd(y).ravel().to_flat_list()
        self.classes_ = sorted(set(int(v) for v in y_flat))
        self.gprs_ = []
        for c in self.classes_:
            y_bin = _pyarray.array('f', [1.0 if int(v) == c else 0.0 for v in y_flat])
            y_bin_nd = _NDA(y_bin, (len(y_flat),), _nd_c_strides((len(y_flat),)), _NDType('float32'))
            gpr = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-6)
            gpr.fit(X_nd, y_bin_nd)
            self.gprs_.append(gpr)
        return self

    def predict_proba(self, X) -> _NDA:
        X_nd = _to_nd(X)
        n = X_nd.shape[0]
        K = len(self.classes_)
        preds_flat = []
        for gpr in self.gprs_:
            p = gpr.predict(X_nd)
            preds_flat.extend(p.to_flat_list())
        # arrange as (n, K)
        raw = [[preds_flat[k * n + i] for k in range(K)] for i in range(n)]
        # softmax
        result = []
        for row in raw:
            m = max(row)
            exps = [math.exp(v - m) for v in row]
            s = sum(exps)
            result.extend([max(0.0, min(1.0, e / s)) for e in exps])
        buf = _pyarray.array('f', result)
        return _NDA(buf, (n, K), _nd_c_strides((n, K)), _NDType('float32'))

    def predict(self, X) -> _NDA:
        proba = self.predict_proba(_to_nd(X))
        proba_list = proba.to_flat_list()
        n, K = proba.shape
        preds = []
        for i in range(n):
            row = proba_list[i * K:(i + 1) * K]
            preds.append(float(self.classes_[row.index(max(row))]))
        buf = _pyarray.array('f', preds)
        return _NDA(buf, (n,), _nd_c_strides((n,)), _NDType('float32'))

    def score(self, X, y) -> float:
        preds = self.predict(_to_nd(X)).to_flat_list()
        y_true = _to_nd(y).ravel().to_flat_list()
        return sum(1 for p, t in zip(preds, y_true) if int(p) == int(t)) / max(len(y_true), 1)


# Expose kernels as a submodule-like namespace
class kernels:
    RBF = RBF
    Matern = Matern
    DotProduct = DotProduct
    WhiteKernel = WhiteKernel
    ConstantKernel = ConstantKernel
    Sum = Sum
    Product = Product


__all__ = [
    'GaussianProcessRegressor', 'GaussianProcessClassifier',
    'kernels',
    'RBF', 'Matern', 'DotProduct', 'WhiteKernel', 'ConstantKernel', 'Sum', 'Product',
]
