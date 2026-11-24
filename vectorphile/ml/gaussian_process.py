"""Gaussian Process models — pure Python."""
import math
import random as _random
from typing import Optional


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _dist_sq(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _cholesky(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                L[i][j] = math.sqrt(max(val, 1e-15))
            else:
                L[i][j] = (A[i][j] - s) / (L[j][j] + 1e-15)
    return L


def _solve_lower(L, b):
    n = len(b)
    x = [0.0] * n
    for i in range(n):
        x[i] = (b[i] - sum(L[i][j] * x[j] for j in range(i))) / (L[i][i] + 1e-15)
    return x


def _solve_upper(U, b):
    n = len(b)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / (U[i][i] + 1e-15)
    return x


def _cho_solve(L, b):
    """Solve A x = b where L L^T = A."""
    y = _solve_lower(L, b)
    LT = [[L[j][i] for j in range(len(L))] for i in range(len(L))]
    return _solve_upper(LT, y)


# ─── Kernels ─────────────────────────────────────────────────────────────────

class RBF:
    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale

    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        n1, n2 = len(X1), len(X2)
        K = [[math.exp(-_dist_sq(X1[i], X2[j]) / (2 * self.length_scale ** 2))
               for j in range(n2)] for i in range(n1)]
        return K


class Matern:
    def __init__(self, length_scale=1.0, nu=1.5):
        self.length_scale = length_scale
        self.nu = nu

    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        n1, n2 = len(X1), len(X2)
        K = []
        for i in range(n1):
            row = []
            for j in range(n2):
                d = math.sqrt(_dist_sq(X1[i], X2[j])) / self.length_scale
                if abs(self.nu - 0.5) < 1e-9:
                    row.append(math.exp(-d))
                elif abs(self.nu - 1.5) < 1e-9:
                    row.append((1 + math.sqrt(3) * d) * math.exp(-math.sqrt(3) * d))
                elif abs(self.nu - 2.5) < 1e-9:
                    row.append((1 + math.sqrt(5) * d + 5 / 3 * d ** 2) * math.exp(-math.sqrt(5) * d))
                else:
                    row.append(math.exp(-d))
            K.append(row)
        return K


class DotProduct:
    def __init__(self, sigma_0=1.0):
        self.sigma_0 = sigma_0

    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        return [[self.sigma_0 ** 2 + _dot(X1[i], X2[j]) for j in range(len(X2))]
                for i in range(len(X1))]


class ConstantKernel:
    def __init__(self, constant_value=1.0):
        self.constant_value = constant_value

    def __call__(self, X1, X2=None):
        n1 = len(X1)
        n2 = len(X2) if X2 is not None else n1
        return [[self.constant_value] * n2 for _ in range(n1)]


class WhiteKernel:
    def __init__(self, noise_level=1.0):
        self.noise_level = noise_level

    def __call__(self, X1, X2=None):
        n1 = len(X1)
        if X2 is None:
            return [[self.noise_level if i == j else 0.0 for j in range(n1)] for i in range(n1)]
        n2 = len(X2)
        return [[0.0] * n2 for _ in range(n1)]


# ─── GP Regressor ─────────────────────────────────────────────────────────────

class GaussianProcessRegressor:
    def __init__(self, kernel=None, alpha=1e-10, n_restarts_optimizer=0,
                 normalize_y=False, random_state=None):
        self.kernel = kernel or RBF(1.0)
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.random_state = random_state

    def fit(self, X, y):
        self.X_train_ = [[float(v) for v in row] for row in X]
        y2 = [float(v) for v in y]
        if self.normalize_y:
            self._y_mean = sum(y2) / len(y2)
            self._y_std = math.sqrt(sum((v - self._y_mean) ** 2 for v in y2) / len(y2)) + 1e-10
            self.y_train_ = [(v - self._y_mean) / self._y_std for v in y2]
        else:
            self._y_mean, self._y_std = 0.0, 1.0
            self.y_train_ = y2
        K = self.kernel(self.X_train_)
        n = len(K)
        for i in range(n):
            K[i][i] += self.alpha
        self.L_ = _cholesky(K)
        self.alpha_ = _cho_solve(self.L_, self.y_train_)
        return self

    def predict(self, X, return_std=False):
        X2 = [[float(v) for v in row] for row in X]
        K_trans = self.kernel(X2, self.X_train_)
        mean = [sum(K_trans[i][j] * self.alpha_[j] for j in range(len(self.X_train_)))
                for i in range(len(X2))]
        if self.normalize_y:
            mean = [v * self._y_std + self._y_mean for v in mean]
        if not return_std:
            return mean
        K_xx = self.kernel(X2)
        std = []
        for i in range(len(X2)):
            v = _solve_lower(self.L_, K_trans[i])
            var = K_xx[i][i] - sum(x ** 2 for x in v)
            std.append(math.sqrt(max(var, 0)) * self._y_std)
        return mean, std

    def score(self, X, y):
        preds = self.predict(X)
        y2 = [float(v) for v in y]
        mu = sum(y2) / len(y2)
        ss_res = sum((a - b) ** 2 for a, b in zip(y2, preds))
        ss_tot = sum((v - mu) ** 2 for v in y2)
        return float(1 - ss_res / (ss_tot + 1e-12))


class GaussianProcessClassifier:
    """Binary GP classifier via Laplace approximation."""

    def __init__(self, kernel=None, max_iter_predict=100, n_restarts_optimizer=0,
                 random_state=None):
        self.kernel = kernel or RBF(1.0)
        self.max_iter_predict = max_iter_predict
        self.random_state = random_state

    @staticmethod
    def _sigmoid(z):
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def fit(self, X, y):
        self.X_train_ = [[float(v) for v in row] for row in X]
        y2 = list(y)
        self.classes_ = sorted(set(y2))
        y_bin = [1.0 if v == self.classes_[1] else 0.0 for v in y2]
        n = len(y_bin)
        K = self.kernel(self.X_train_)
        # Laplace approximation: find mode via Newton
        f = [0.0] * n
        for _ in range(self.max_iter_predict):
            pi = [self._sigmoid(fi) for fi in f]
            W = [p * (1 - p) for p in pi]
            grad = [y_bin[i] - pi[i] for i in range(n)]
            # K W K + I f_new ~ K grad + f
            WK = [[W[i] * K[i][j] for j in range(n)] for i in range(n)]
            # Simplified Newton step
            Kf = [sum(K[i][j] * f[j] for j in range(n)) for i in range(n)]
            new_f = [Kf[i] - sum(WK[i][j] * Kf[j] for j in range(n)) + sum(K[i][j] * grad[j] for j in range(n))
                     for i in range(n)]
            if max(abs(new_f[i] - f[i]) for i in range(n)) < 1e-5:
                break
            f = new_f
        self._f_mode = f
        return self

    def predict_proba(self, X):
        X2 = [[float(v) for v in row] for row in X]
        K_trans = self.kernel(X2, self.X_train_)
        result = []
        for i in range(len(X2)):
            f_star = sum(K_trans[i][j] * self._f_mode[j] for j in range(len(self.X_train_)))
            p = self._sigmoid(f_star)
            result.append([1 - p, p])
        return result

    def predict(self, X):
        proba = self.predict_proba(X)
        return [self.classes_[row.index(max(row))] for row in proba]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(a == b for a, b in zip(preds, y)) / len(y)


class BayesianOptimization:
    """Simple Bayesian optimization via GP + Expected Improvement."""

    def __init__(self, f, pbounds, random_state=None, verbose=0):
        self.f = f
        self.pbounds = pbounds
        self.random_state = random_state
        self.verbose = verbose
        self._gp = GaussianProcessRegressor(kernel=RBF(1.0), alpha=1e-6)
        self._X = []
        self._y = []

    def maximize(self, init_points=5, n_iter=25):
        rng = _random.Random(self.random_state)
        keys = list(self.pbounds.keys())
        bounds = [self.pbounds[k] for k in keys]
        # Initial random exploration
        for _ in range(init_points):
            params = {k: rng.uniform(lo, hi) for k, (lo, hi) in zip(keys, bounds)}
            val = self.f(**params)
            self._X.append([params[k] for k in keys])
            self._y.append(float(val))
        # Optimization
        for _ in range(n_iter):
            self._gp.fit(self._X, self._y)
            best_y = max(self._y)
            # Maximize EI over random candidates
            best_ei, best_x = -float('inf'), None
            for _ in range(1000):
                x_cand = [[rng.uniform(lo, hi) for lo, hi in bounds]]
                mu, std = self._gp.predict(x_cand, return_std=True)
                mu, std = mu[0], std[0]
                z = (mu - best_y) / (std + 1e-15)
                # Normal CDF and PDF approximation
                phi = math.exp(-z ** 2 / 2) / math.sqrt(2 * math.pi)
                Phi = 0.5 * (1 + math.erf(z / math.sqrt(2)))
                ei = (mu - best_y) * Phi + std * phi
                if ei > best_ei:
                    best_ei = ei
                    best_x = x_cand[0]
            if best_x is None:
                best_x = [rng.uniform(lo, hi) for lo, hi in bounds]
            params = {k: best_x[i] for i, k in enumerate(keys)}
            val = self.f(**params)
            self._X.append(best_x)
            self._y.append(float(val))
        best_idx = self._y.index(max(self._y))
        self.max = {'target': self._y[best_idx],
                    'params': {k: self._X[best_idx][i] for i, k in enumerate(keys)}}
        return self


# Alias names expected by __init__.py
RBFKernel = RBF
MaternKernel = Matern
LinearKernel = DotProduct


class PeriodicKernel:
    """Periodic (Exp-Sine-Squared) kernel: k(x,y) = exp(-2 sin^2(pi |x-y| / p) / l^2)."""
    def __init__(self, length_scale=1.0, periodicity=1.0):
        self.length_scale = length_scale
        self.periodicity = periodicity

    def __call__(self, X, Y=None):
        import math
        if Y is None:
            Y = X
        n, m = len(X), len(Y)
        K = []
        for i in range(n):
            row = []
            for j in range(m):
                dist = math.sqrt(sum((X[i][k] - Y[j][k]) ** 2 for k in range(len(X[i]))))
                val = math.exp(-2.0 * math.sin(math.pi * dist / self.periodicity) ** 2
                               / self.length_scale ** 2)
                row.append(val)
            K.append(row)
        return K


class PolynomialKernel:
    """Polynomial kernel: k(x,y) = (x·y + c)^d."""
    def __init__(self, degree=3, coef0=1.0, gamma=1.0):
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        n, m = len(X), len(Y)
        K = []
        for i in range(n):
            row = []
            for j in range(m):
                dot = sum(X[i][k] * Y[j][k] for k in range(len(X[i])))
                row.append((self.gamma * dot + self.coef0) ** self.degree)
            K.append(row)
        return K
