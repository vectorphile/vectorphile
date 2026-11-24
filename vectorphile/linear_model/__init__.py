"""vectorphile.linear_model — Linear models for regression and classification.

Re-exports from ml module + adds:
- BayesianRidge
- HuberRegressor (robust regression)
- Perceptron
- SGDClassifier / SGDRegressor

All computations use pure Python and vectorphile's NDArray primitives.
No external dependencies.
"""
from __future__ import annotations
import math
import random
from typing import Any, Optional, List

# Re-export from ml
from vectorphile.ml import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression,
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _to_list2d(X) -> List[List[float]]:
    if hasattr(X, 'to_flat_list'):
        flat = X.to_flat_list()
        if X.ndim == 1:
            return [[float(v)] for v in flat]
        n, d = X.shape[0], X.shape[1]
        return [[float(flat[i * d + j]) for j in range(d)] for i in range(n)]
    if hasattr(X, 'tolist'):
        data = X.tolist()
    else:
        data = list(X)
    if data and not isinstance(data[0], (list, tuple)):
        return [[float(v)] for v in data]
    return [[float(v) for v in row] for row in data]


def _to_list1d(y) -> List[float]:
    if hasattr(y, 'to_flat_list'):
        return [float(v) for v in y.ravel().to_flat_list()]
    if hasattr(y, 'tolist'):
        data = y.tolist()
    else:
        data = list(y)
    if isinstance(data, (int, float)):
        return [float(data)]
    flat = []
    for v in data:
        if isinstance(v, (list, tuple)):
            flat.extend(float(x) for x in v)
        else:
            flat.append(float(v))
    return flat


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _sq_norm(a: List[float]) -> float:
    return sum(x * x for x in a)


def _mat_vec(A: List[List[float]], x: List[float]) -> List[float]:
    """Matrix-vector product A @ x."""
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def _mat_mat(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Matrix-matrix product A @ B."""
    n, m, k = len(A), len(B[0]), len(B)
    return [[sum(A[i][p] * B[p][j] for p in range(k)) for j in range(m)] for i in range(n)]


def _transpose(A: List[List[float]]) -> List[List[float]]:
    n, m = len(A), len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]


def _mat_add_diag(A: List[List[float]], diag_val: float) -> List[List[float]]:
    n = len(A)
    result = [row[:] for row in A]
    for i in range(n):
        result[i][i] += diag_val
    return result


def _inv_sym(A: List[List[float]]) -> List[List[float]]:
    """Invert symmetric positive definite matrix via Cholesky."""
    n = len(A)
    # Cholesky L s.t. A = L L^T
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                L[i][j] = math.sqrt(max(val, 1e-15))
            else:
                L[i][j] = (A[i][j] - s) / (L[j][j] + 1e-15)
    # Invert L
    Li = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Li[i][i] = 1.0 / (L[i][i] + 1e-15)
        for j in range(i):
            Li[i][j] = -sum(L[i][k] * Li[k][j] for k in range(j, i)) / (L[i][i] + 1e-15)
    # A^{-1} = Li^T @ Li
    return _mat_mat(_transpose(Li), Li)


def _solve_sym(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve A x = b for SPD A."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = math.sqrt(max(A[i][i] - s, 1e-15))
            else:
                L[i][j] = (A[i][j] - s) / (L[j][j] + 1e-15)
    # Forward substitution
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / (L[i][i] + 1e-15)
    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L[j][i] * x[j] for j in range(i + 1, n))) / (L[i][i] + 1e-15)
    return x


def _trace(A: List[List[float]]) -> float:
    return sum(A[i][i] for i in range(len(A)))


def _median(vals: List[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


# ---------------------------------------------------------------------------
# BayesianRidge
# ---------------------------------------------------------------------------

class BayesianRidge:
    """Bayesian Ridge Regression via evidence maximization (Type II ML).

    Estimates regularization hyperparameters alpha (noise precision)
    and lambda_ (weight precision) from the data.
    """

    def __init__(self, max_iter: int = 300, tol: float = 1e-3,
                 alpha_1: float = 1e-6, alpha_2: float = 1e-6,
                 lambda_1: float = 1e-6, lambda_2: float = 1e-6,
                 fit_intercept: bool = True):
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.fit_intercept = fit_intercept
        self.coef_: Optional[List[float]] = None
        self.intercept_: float = 0.0
        self.alpha_: float = 1.0
        self.lambda_: float = 1.0
        self.sigma_: Optional[List[List[float]]] = None
        self._X_mean: Optional[List[float]] = None
        self._y_mean: float = 0.0

    def fit(self, X, y) -> 'BayesianRidge':
        X2d = _to_list2d(X)
        y1d = _to_list1d(y)
        n, p = len(X2d), len(X2d[0])
        if self.fit_intercept:
            X_mean = [sum(X2d[i][j] for i in range(n)) / n for j in range(p)]
            y_mean = sum(y1d) / n
            X2d = [[X2d[i][j] - X_mean[j] for j in range(p)] for i in range(n)]
            y1d = [v - y_mean for v in y1d]
            self._X_mean = X_mean
            self._y_mean = y_mean
        # XtX = X^T X
        Xt = _transpose(X2d)
        XtX = _mat_mat(Xt, X2d)
        # Xty = X^T y
        Xty = [sum(Xt[j][i] * y1d[i] for i in range(n)) for j in range(p)]
        alpha = 1.0
        lam = 1.0
        for _ in range(self.max_iter):
            # Posterior covariance: A = lam*XtX + alpha*I
            A = _mat_add_diag([[lam * XtX[i][j] for j in range(p)] for i in range(p)], alpha)
            A_inv = _inv_sym(A)
            # Posterior mean: m_n = alpha * A_inv @ Xty
            m_n = [alpha * sum(A_inv[i][j] * Xty[j] for j in range(p)) for i in range(p)]
            # gamma = p - lam * tr(A_inv)
            gamma = p - lam * _trace(A_inv)
            gamma = max(1e-10, min(gamma, p - 1e-10))
            # Residuals y - X @ m_n
            resid = [y1d[i] - sum(X2d[i][j] * m_n[j] for j in range(p)) for i in range(n)]
            ss_resid = sum(r * r for r in resid)
            ss_coef = sum(c * c for c in m_n)
            alpha_new = (n - gamma + 2 * self.alpha_1 - 2) / (ss_resid + 2 * self.alpha_2)
            lambda_new = (gamma + 2 * self.lambda_1 - 2) / (ss_coef + 2 * self.lambda_2)
            alpha_new = max(1e-10, alpha_new)
            lambda_new = max(1e-10, lambda_new)
            if abs(alpha_new - alpha) < self.tol and abs(lambda_new - lam) < self.tol:
                alpha, lam = alpha_new, lambda_new
                break
            alpha, lam = alpha_new, lambda_new
        self.alpha_ = alpha
        self.lambda_ = lam
        A = _mat_add_diag([[lam * XtX[i][j] for j in range(p)] for i in range(p)], alpha)
        self.sigma_ = _inv_sym(A)
        self.coef_ = [alpha * sum(self.sigma_[i][j] * Xty[j] for j in range(p)) for i in range(p)]
        if self.fit_intercept:
            self.intercept_ = y_mean - sum(X_mean[j] * self.coef_[j] for j in range(p))
        return self

    def predict(self, X, return_std: bool = False):
        X2d = _to_list2d(X)
        n, p = len(X2d), len(X2d[0])
        preds = [sum(X2d[i][j] * self.coef_[j] for j in range(p)) + self.intercept_ for i in range(n)]
        if return_std:
            stds = []
            for i in range(n):
                # var = x^T sigma x + 1/alpha
                xSx = sum(X2d[i][r] * sum(self.sigma_[r][c] * X2d[i][c] for c in range(p)) for r in range(p))
                stds.append(math.sqrt(max(xSx + 1.0 / self.alpha_, 0.0)))
            return preds, stds
        return preds

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        n = len(y_true)
        mean_y = sum(y_true) / n
        ss_res = sum((y_true[i] - preds[i]) ** 2 for i in range(n))
        ss_tot = sum((v - mean_y) ** 2 for v in y_true) + 1e-10
        return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# HuberRegressor (robust, M-estimator via IRLS)
# ---------------------------------------------------------------------------

class HuberRegressor:
    """Robust regression using Huber loss (less sensitive to outliers).

    For residuals |e| <= epsilon*sigma: squared loss.
    For residuals |e| > epsilon*sigma: linear loss.
    """

    def __init__(self, epsilon: float = 1.35, max_iter: int = 100,
                 alpha: float = 0.0001, tol: float = 1e-5,
                 fit_intercept: bool = True):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.alpha = alpha
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.coef_: Optional[List[float]] = None
        self.intercept_: float = 0.0
        self.scale_: float = 1.0

    def fit(self, X, y) -> 'HuberRegressor':
        X2d = _to_list2d(X)
        y1d = _to_list1d(y)
        n, p = len(X2d), len(X2d[0])
        if self.fit_intercept:
            X2d = [row + [1.0] for row in X2d]
            p += 1
        w = [0.0] * p
        eps = self.epsilon
        for _ in range(self.max_iter):
            # Residuals
            resid = [y1d[i] - sum(X2d[i][j] * w[j] for j in range(p)) for i in range(n)]
            abs_resid = [abs(r) for r in resid]
            sigma = _median(abs_resid) / 0.6745 + 1e-10
            # Huber weights
            huber_w = [1.0 if abs_resid[i] / (sigma * eps) <= 1.0 else (sigma * eps) / abs_resid[i]
                       for i in range(n)]
            # Weighted XtWX and XtWy
            XtWX = [[sum(X2d[i][r] * huber_w[i] * X2d[i][c] for i in range(n)) for c in range(p)]
                    for r in range(p)]
            XtWy = [sum(X2d[i][j] * huber_w[i] * y1d[i] for i in range(n)) for j in range(p)]
            # Ridge regularization
            for j in range(p):
                XtWX[j][j] += self.alpha
            w_new = _solve_sym(XtWX, XtWy)
            if max(abs(w_new[j] - w[j]) for j in range(p)) < self.tol:
                w = w_new
                break
            w = w_new
        if self.fit_intercept:
            self.coef_ = w[:-1]
            self.intercept_ = w[-1]
        else:
            self.coef_ = w
        self.scale_ = sigma
        return self

    def predict(self, X) -> List[float]:
        X2d = _to_list2d(X)
        n, p = len(X2d), len(X2d[0])
        return [sum(X2d[i][j] * self.coef_[j] for j in range(p)) + self.intercept_
                for i in range(n)]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        n = len(y_true)
        mean_y = sum(y_true) / n
        ss_res = sum((y_true[i] - preds[i]) ** 2 for i in range(n))
        ss_tot = sum((v - mean_y) ** 2 for v in y_true) + 1e-10
        return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Perceptron
# ---------------------------------------------------------------------------

class Perceptron:
    """Simple linear perceptron (threshold update rule)."""

    def __init__(self, eta0: float = 1.0, max_iter: int = 1000,
                 tol: float = 1e-3, random_state: Optional[int] = None,
                 fit_intercept: bool = True):
        self.eta0 = eta0
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.fit_intercept = fit_intercept
        self.coef_: Optional[List[float]] = None
        self.intercept_: float = 0.0
        self.classes_: Optional[List] = None

    def fit(self, X, y) -> 'Perceptron':
        X2d = _to_list2d(X)
        y1d = _to_list1d(y)
        n, p = len(X2d), len(X2d[0])
        unique = sorted(set(y1d))
        self.classes_ = unique
        pos_class = unique[-1]
        y_bin = [1.0 if v == pos_class else -1.0 for v in y1d]
        rng = random.Random(self.random_state)
        w = [0.0] * p
        b = 0.0
        for _ in range(self.max_iter):
            idx = list(range(n))
            rng.shuffle(idx)
            errors = 0
            for i in idx:
                score = sum(w[j] * X2d[i][j] for j in range(p)) + b
                pred = 1.0 if score >= 0.0 else -1.0
                if pred != y_bin[i]:
                    for j in range(p):
                        w[j] += self.eta0 * y_bin[i] * X2d[i][j]
                    b += self.eta0 * y_bin[i]
                    errors += 1
            if errors == 0:
                break
        self.coef_ = w
        self.intercept_ = b
        return self

    def decision_function(self, X) -> List[float]:
        X2d = _to_list2d(X)
        p = len(X2d[0])
        return [sum(X2d[i][j] * self.coef_[j] for j in range(p)) + self.intercept_
                for i in range(len(X2d))]

    def predict(self, X) -> List:
        scores = self.decision_function(X)
        return [self.classes_[-1] if s >= 0.0 else self.classes_[0] for s in scores]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        return sum(1 for p, t in zip(preds, y_true) if p == t) / max(len(y_true), 1)


# ---------------------------------------------------------------------------
# SGDClassifier
# ---------------------------------------------------------------------------

class SGDClassifier:
    """Linear classifier trained with Stochastic Gradient Descent.

    Supports hinge (SVM), log (logistic regression), and perceptron losses.
    """

    def __init__(self, loss: str = 'hinge', penalty: str = 'l2',
                 alpha: float = 0.0001, max_iter: int = 1000,
                 tol: float = 1e-3, learning_rate: str = 'optimal',
                 eta0: float = 0.01, random_state: Optional[int] = None,
                 fit_intercept: bool = True):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.random_state = random_state
        self.fit_intercept = fit_intercept
        self.coef_: Optional[List[float]] = None
        self.intercept_: float = 0.0
        self.classes_: Optional[List] = None

    def _get_lr(self, t: int) -> float:
        if self.learning_rate == 'constant':
            return self.eta0
        if self.learning_rate == 'optimal':
            return 1.0 / (self.alpha * (t + 1) + 1e-10)
        return self.eta0 / math.sqrt(t + 1)

    def fit(self, X, y) -> 'SGDClassifier':
        X2d = _to_list2d(X)
        y1d = _to_list1d(y)
        n, p = len(X2d), len(X2d[0])
        unique = sorted(set(y1d))
        self.classes_ = unique
        pos_class = unique[-1]
        y_bin = [1.0 if v == pos_class else -1.0 for v in y1d]
        rng = random.Random(self.random_state)
        w = [0.0] * p
        b = 0.0
        for it in range(self.max_iter):
            idx = list(range(n))
            rng.shuffle(idx)
            for t, i in enumerate(idx):
                lr = self._get_lr(it * n + t)
                xi = X2d[i]
                yi = y_bin[i]
                score = sum(w[j] * xi[j] for j in range(p)) + b
                if self.loss == 'hinge':
                    if yi * score < 1.0:
                        for j in range(p):
                            w[j] = (1 - lr * self.alpha) * w[j] + lr * yi * xi[j]
                        b += lr * yi
                    else:
                        for j in range(p):
                            w[j] = (1 - lr * self.alpha) * w[j]
                elif self.loss == 'log':
                    p_pos = 1.0 / (1.0 + math.exp(-score))
                    grad = p_pos - (1.0 if yi > 0 else 0.0)
                    for j in range(p):
                        w[j] = (1 - lr * self.alpha) * w[j] - lr * grad * xi[j]
                    b -= lr * grad
                else:  # perceptron
                    pred = 1.0 if score >= 0 else -1.0
                    if pred != yi:
                        for j in range(p):
                            w[j] += lr * yi * xi[j]
                        b += lr * yi
        self.coef_ = w
        self.intercept_ = b
        return self

    def decision_function(self, X) -> List[float]:
        X2d = _to_list2d(X)
        p = len(X2d[0])
        return [sum(X2d[i][j] * self.coef_[j] for j in range(p)) + self.intercept_
                for i in range(len(X2d))]

    def predict(self, X) -> List:
        scores = self.decision_function(X)
        return [self.classes_[-1] if s >= 0.0 else self.classes_[0] for s in scores]

    def predict_proba(self, X) -> List[List[float]]:
        scores = self.decision_function(X)
        result = []
        for s in scores:
            p_pos = 1.0 / (1.0 + math.exp(-s))
            result.append([1.0 - p_pos, p_pos])
        return result

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        return sum(1 for p, t in zip(preds, y_true) if p == t) / max(len(y_true), 1)


class SGDRegressor:
    """Linear regressor trained with SGD."""

    def __init__(self, loss: str = 'squared_error', penalty: str = 'l2',
                 alpha: float = 0.0001, max_iter: int = 1000,
                 tol: float = 1e-3, eta0: float = 0.01,
                 learning_rate: str = 'invscaling',
                 random_state: Optional[int] = None,
                 fit_intercept: bool = True):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.fit_intercept = fit_intercept
        self.coef_: Optional[List[float]] = None
        self.intercept_: float = 0.0

    def _get_lr(self, t: int) -> float:
        if self.learning_rate == 'constant':
            return self.eta0
        return self.eta0 / math.sqrt(t + 1)

    def fit(self, X, y) -> 'SGDRegressor':
        X2d = _to_list2d(X)
        y1d = _to_list1d(y)
        n, p = len(X2d), len(X2d[0])
        rng = random.Random(self.random_state)
        w = [0.0] * p
        b = 0.0
        for it in range(self.max_iter):
            idx = list(range(n))
            rng.shuffle(idx)
            for t, i in enumerate(idx):
                lr = self._get_lr(it * n + t)
                xi = X2d[i]
                yi = y1d[i]
                pred = sum(w[j] * xi[j] for j in range(p)) + b
                residual = pred - yi
                for j in range(p):
                    w[j] = (1 - lr * self.alpha) * w[j] - lr * residual * xi[j]
                b -= lr * residual
        self.coef_ = w
        self.intercept_ = b
        return self

    def predict(self, X) -> List[float]:
        X2d = _to_list2d(X)
        p = len(X2d[0])
        return [sum(X2d[i][j] * self.coef_[j] for j in range(p)) + self.intercept_
                for i in range(len(X2d))]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        n = len(y_true)
        mean_y = sum(y_true) / n
        ss_res = sum((y_true[i] - preds[i]) ** 2 for i in range(n))
        ss_tot = sum((v - mean_y) ** 2 for v in y_true) + 1e-10
        return 1.0 - ss_res / ss_tot


class PassiveAggressiveClassifier:
    """Online PA classifier (Crammer-Singer variant)."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000, tol: float = 1e-3,
                 random_state: Optional[int] = None):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.coef_: List[List[float]] = []
        self.intercept_: List[float] = []
        self.classes_: List = []

    def fit(self, X, y):
        X2d = _to_list2d(X)
        y_list = _to_list1d(y)
        self.classes_ = sorted(set(y_list))
        n, p = len(X2d), len(X2d[0])
        rng = random.Random(self.random_state)
        self.coef_ = [[0.0] * p for _ in self.classes_]
        self.intercept_ = [0.0] * len(self.classes_)
        indices = list(range(n))
        for _ in range(self.max_iter):
            rng.shuffle(indices)
            for i in indices:
                xi, yi = X2d[i], y_list[i]
                ci = self.classes_.index(yi)
                scores = [_dot(self.coef_[k], xi) + self.intercept_[k]
                          for k in range(len(self.classes_))]
                pred = max(range(len(self.classes_)), key=lambda k: scores[k])
                if pred != ci:
                    loss = 1 - scores[ci] + scores[pred]
                    norm_sq = _sq_norm(xi) + 1e-10
                    tau = min(self.C, loss / (2 * norm_sq))
                    for j in range(p):
                        self.coef_[ci][j] += tau * xi[j]
                        self.coef_[pred][j] -= tau * xi[j]
        return self

    def predict(self, X) -> List:
        X2d = _to_list2d(X)
        return [self.classes_[max(range(len(self.classes_)),
                key=lambda k: _dot(self.coef_[k], row) + self.intercept_[k])]
                for row in X2d]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        return sum(a == b for a, b in zip(y_true, preds)) / len(y_true)


class PassiveAggressiveRegressor:
    """Online PA regressor."""

    def __init__(self, C: float = 1.0, epsilon: float = 0.1, max_iter: int = 1000,
                 random_state: Optional[int] = None):
        self.C = C
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.random_state = random_state
        self.coef_: List[float] = []
        self.intercept_: float = 0.0

    def fit(self, X, y):
        X2d = _to_list2d(X)
        y_list = _to_list1d(y)
        n, p = len(X2d), len(X2d[0])
        rng = random.Random(self.random_state)
        self.coef_ = [0.0] * p
        self.intercept_ = 0.0
        indices = list(range(n))
        for _ in range(self.max_iter):
            rng.shuffle(indices)
            for i in indices:
                xi, yi = X2d[i], float(y_list[i])
                pred = _dot(self.coef_, xi) + self.intercept_
                loss = max(0.0, abs(pred - yi) - self.epsilon)
                if loss > 0:
                    sign = 1.0 if pred < yi else -1.0
                    norm_sq = _sq_norm(xi) + 1e-10
                    tau = min(self.C, loss / norm_sq)
                    self.coef_ = [self.coef_[j] + sign * tau * xi[j] for j in range(p)]
        return self

    def predict(self, X) -> List[float]:
        return [_dot(self.coef_, row) + self.intercept_ for row in _to_list2d(X)]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = [float(v) for v in _to_list1d(y)]
        mean_y = sum(y_true) / len(y_true)
        ss_res = sum((y_true[i] - preds[i]) ** 2 for i in range(len(y_true)))
        ss_tot = sum((v - mean_y) ** 2 for v in y_true) + 1e-10
        return 1.0 - ss_res / ss_tot


class RANSACRegressor:
    """RANSAC robust regression."""

    def __init__(self, estimator=None, min_samples: float = 0.5,
                 residual_threshold: float = 1.0, max_trials: int = 100,
                 random_state: Optional[int] = None):
        self.estimator = estimator
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials
        self.random_state = random_state
        self.estimator_: Any = None
        self.inlier_mask_: List[bool] = []

    def fit(self, X, y):
        import copy
        X2d = _to_list2d(X)
        y_list = [float(v) for v in _to_list1d(y)]
        n = len(X2d)
        n_samples = max(2, int(n * self.min_samples) if isinstance(self.min_samples, float)
                        else self.min_samples)
        rng = random.Random(self.random_state)

        best_inliers: List[int] = []
        best_est = None

        for _ in range(self.max_trials):
            sample = rng.sample(range(n), n_samples)
            X_s = [X2d[i] for i in sample]
            y_s = [y_list[i] for i in sample]
            if self.estimator is not None:
                est = copy.deepcopy(self.estimator)
            else:
                est = _SimpleLinReg()
            est.fit(X_s, y_s)
            preds = est.predict(X2d)
            inliers = [i for i in range(n) if abs(y_list[i] - preds[i]) <= self.residual_threshold]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_est = est

        if best_est is not None and best_inliers:
            X_in = [X2d[i] for i in best_inliers]
            y_in = [y_list[i] for i in best_inliers]
            best_est.fit(X_in, y_in)
            self.estimator_ = best_est
            self.inlier_mask_ = [i in best_inliers for i in range(n)]
        return self

    def predict(self, X) -> List[float]:
        return self.estimator_.predict(X)

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = [float(v) for v in _to_list1d(y)]
        mean_y = sum(y_true) / len(y_true)
        ss_res = sum((y_true[i] - preds[i]) ** 2 for i in range(len(y_true)))
        ss_tot = sum((v - mean_y) ** 2 for v in y_true) + 1e-10
        return 1.0 - ss_res / ss_tot


class _SimpleLinReg:
    """Minimal OLS used by RANSAC."""
    def fit(self, X, y):
        X2d = _to_list2d(X)
        y_list = [float(v) for v in y]
        n, p = len(X2d), len(X2d[0])
        # Augment with bias
        Xa = [row + [1.0] for row in X2d]
        # Normal equations X^T X w = X^T y
        XtX = [[sum(Xa[i][a] * Xa[i][b] for i in range(n)) for b in range(p + 1)]
                for a in range(p + 1)]
        Xty = [sum(Xa[i][a] * y_list[i] for i in range(n)) for a in range(p + 1)]
        # Solve with Gaussian elimination
        m = p + 1
        aug = [XtX[r][:] + [Xty[r]] for r in range(m)]
        for col in range(m):
            pivot = max(range(col, m), key=lambda r: abs(aug[r][col]))
            aug[col], aug[pivot] = aug[pivot], aug[col]
            if abs(aug[col][col]) < 1e-12:
                continue
            for r in range(m):
                if r != col:
                    f = aug[r][col] / aug[col][col]
                    aug[r] = [aug[r][c] - f * aug[col][c] for c in range(m + 1)]
        w = [aug[r][m] / aug[r][r] if abs(aug[r][r]) > 1e-12 else 0.0 for r in range(m)]
        self.coef_ = w[:p]
        self.intercept_ = w[p]
        return self

    def predict(self, X) -> List[float]:
        return [sum(self.coef_[j] * row[j] for j in range(len(self.coef_))) + self.intercept_
                for row in _to_list2d(X)]


class TheilSenRegressor:
    """Theil-Sen robust regression via median of pairwise slopes."""

    def __init__(self, max_subpopulation: int = 10000, random_state: Optional[int] = None):
        self.max_subpopulation = max_subpopulation
        self.random_state = random_state
        self.coef_: List[float] = []
        self.intercept_: float = 0.0

    def fit(self, X, y):
        X2d = _to_list2d(X)
        y_list = [float(v) for v in _to_list1d(y)]
        n, p = len(X2d), len(X2d[0])
        rng = random.Random(self.random_state)
        # For each pair of points, compute the slope vector
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if len(pairs) > self.max_subpopulation:
            pairs = rng.sample(pairs, self.max_subpopulation)
        slopes = [[] for _ in range(p)]
        for i, j in pairs:
            dy = y_list[j] - y_list[i]
            for k in range(p):
                dx = X2d[j][k] - X2d[i][k]
                if abs(dx) > 1e-10:
                    slopes[k].append(dy / dx)
        self.coef_ = []
        for k in range(p):
            s = sorted(slopes[k]) if slopes[k] else [0.0]
            mid = len(s) // 2
            self.coef_.append(s[mid] if len(s) % 2 == 1 else (s[mid - 1] + s[mid]) / 2)
        # Intercept via median of residuals
        residuals = [y_list[i] - sum(self.coef_[k] * X2d[i][k] for k in range(p)) for i in range(n)]
        residuals.sort()
        mid = len(residuals) // 2
        self.intercept_ = (residuals[mid] if len(residuals) % 2 == 1
                           else (residuals[mid - 1] + residuals[mid]) / 2)
        return self

    def predict(self, X) -> List[float]:
        return [sum(self.coef_[k] * row[k] for k in range(len(self.coef_))) + self.intercept_
                for row in _to_list2d(X)]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = [float(v) for v in _to_list1d(y)]
        mean_y = sum(y_true) / len(y_true)
        ss_res = sum((y_true[i] - preds[i]) ** 2 for i in range(len(y_true)))
        ss_tot = sum((v - mean_y) ** 2 for v in y_true) + 1e-10
        return 1.0 - ss_res / ss_tot


__all__ = [
    'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'LogisticRegression',
    'BayesianRidge', 'HuberRegressor',
    'Perceptron', 'SGDClassifier', 'SGDRegressor',
    'PassiveAggressiveClassifier', 'PassiveAggressiveRegressor',
    'RANSACRegressor', 'TheilSenRegressor',
]
