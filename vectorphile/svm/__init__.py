"""vectorphile.svm — Support Vector Machines.

All computations use pure Python. No external dependencies.
"""
from __future__ import annotations
import math
import random
from typing import Optional, Union, List


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


def _to_list1d(y) -> List:
    if hasattr(y, 'to_flat_list'):
        return y.ravel().to_flat_list()
    if hasattr(y, 'tolist'):
        data = y.tolist()
    else:
        data = list(y)
    flat = []
    for v in data:
        if isinstance(v, (list, tuple)):
            flat.extend(v)
        else:
            flat.append(v)
    return flat


def _dot(a: List[float], b: List[float]) -> float:
    return sum(a[i] * b[i] for i in range(len(a)))


def _sq_norm(a: List[float]) -> float:
    return sum(v * v for v in a)


def _linear_kernel_row(xi: List[float], xj: List[float]) -> float:
    return _dot(xi, xj)


def _rbf_kernel_row(xi: List[float], xj: List[float], gamma: float) -> float:
    sq = sum((xi[k] - xj[k]) ** 2 for k in range(len(xi)))
    return math.exp(-gamma * sq)


def _poly_kernel_row(xi: List[float], xj: List[float],
                     degree: int, gamma: float, coef0: float) -> float:
    return (gamma * _dot(xi, xj) + coef0) ** degree


def _sigmoid_kernel_row(xi: List[float], xj: List[float],
                        gamma: float, coef0: float) -> float:
    return math.tanh(gamma * _dot(xi, xj) + coef0)


def _kernel_matrix(X1: List[List[float]], X2: List[List[float]],
                   kernel: str, degree: int, gamma: float, coef0: float) -> List[List[float]]:
    n1, n2 = len(X1), len(X2)
    K = []
    for i in range(n1):
        row = []
        for j in range(n2):
            if kernel == 'linear':
                row.append(_linear_kernel_row(X1[i], X2[j]))
            elif kernel == 'rbf':
                row.append(_rbf_kernel_row(X1[i], X2[j], gamma))
            elif kernel == 'poly':
                row.append(_poly_kernel_row(X1[i], X2[j], degree, gamma, coef0))
            elif kernel == 'sigmoid':
                row.append(_sigmoid_kernel_row(X1[i], X2[j], gamma, coef0))
            else:
                row.append(_rbf_kernel_row(X1[i], X2[j], gamma))
        K.append(row)
    return K


def _get_gamma(X: List[List[float]], gamma) -> float:
    if isinstance(gamma, float):
        return gamma
    n, p = len(X), len(X[0])
    if gamma == 'auto':
        return 1.0 / p
    # 'scale': 1/(p * var(X))
    flat = [X[i][j] for i in range(n) for j in range(p)]
    mean = sum(flat) / len(flat)
    var = sum((v - mean) ** 2 for v in flat) / len(flat)
    return 1.0 / (p * var + 1e-10)


# ---------------------------------------------------------------------------
# LinearSVC (Pegasos SGD)
# ---------------------------------------------------------------------------

class LinearSVC:
    """Linear SVM classifier using SGD (Pegasos algorithm)."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000,
                 tol: float = 1e-4, random_state: Optional[int] = None,
                 multi_class: str = 'ovr'):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.multi_class = multi_class
        self.coef_: Optional[List[List[float]]] = None
        self.intercept_: Optional[List[float]] = None
        self.classes_: Optional[List] = None

    def _fit_binary(self, X: List[List[float]], y_bin: List[float]) -> tuple:
        n, d = len(X), len(X[0])
        rng = random.Random(self.random_state)
        w = [0.0] * d
        b = 0.0
        lr0 = 1.0 / (self.C * n + 1e-10)
        for t in range(1, self.max_iter * n + 1):
            i = rng.randint(0, n - 1)
            xi, yi = X[i], y_bin[i]
            lr = lr0 / math.sqrt(t)
            margin = yi * (sum(w[j] * xi[j] for j in range(d)) + b)
            if margin < 1.0:
                for j in range(d):
                    w[j] = (1 - lr) * w[j] + lr * self.C * yi * xi[j]
                b += lr * self.C * yi
            else:
                for j in range(d):
                    w[j] = (1 - lr) * w[j]
        return w, b

    def fit(self, X, y) -> 'LinearSVC':
        X2d = _to_list2d(X)
        y1d = _to_list1d(y)
        self.classes_ = sorted(set(y1d))
        if len(self.classes_) == 2:
            y_bin = [1.0 if v == self.classes_[1] else -1.0 for v in y1d]
            w, b = self._fit_binary(X2d, y_bin)
            self.coef_ = [w]
            self.intercept_ = [b]
        else:
            coefs, intercepts = [], []
            for cls in self.classes_:
                y_bin = [1.0 if v == cls else -1.0 for v in y1d]
                w, b = self._fit_binary(X2d, y_bin)
                coefs.append(w)
                intercepts.append(b)
            self.coef_ = coefs
            self.intercept_ = intercepts
        return self

    def decision_function(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        p = len(X2d[0])
        return [[sum(X2d[i][j] * self.coef_[k][j] for j in range(p)) + self.intercept_[k]
                 for k in range(len(self.coef_))] for i in range(len(X2d))]

    def predict(self, X) -> List:
        scores = self.decision_function(X)
        if len(self.classes_) == 2:
            return [self.classes_[1] if row[0] >= 0.0 else self.classes_[0] for row in scores]
        return [self.classes_[max(range(len(row)), key=lambda i: row[i])] for row in scores]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        return sum(1 for p, t in zip(preds, y_true) if p == t) / max(len(y_true), 1)


class LinearSVR:
    """Linear SVM regressor using SGD (ε-insensitive loss)."""

    def __init__(self, C: float = 1.0, epsilon: float = 0.1,
                 max_iter: int = 1000, random_state: Optional[int] = None):
        self.C = C
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.random_state = random_state
        self.coef_: Optional[List[float]] = None
        self.intercept_: float = 0.0

    def fit(self, X, y) -> 'LinearSVR':
        X2d = _to_list2d(X)
        y1d = [float(v) for v in _to_list1d(y)]
        n, d = len(X2d), len(X2d[0])
        rng = random.Random(self.random_state)
        w = [0.0] * d
        b = 0.0
        for t in range(1, self.max_iter * n + 1):
            i = rng.randint(0, n - 1)
            xi, yi = X2d[i], y1d[i]
            lr = 1.0 / math.sqrt(t)
            pred = sum(w[j] * xi[j] for j in range(d)) + b
            residual = yi - pred
            if abs(residual) > self.epsilon:
                sign = 1.0 if residual > 0 else -1.0
                for j in range(d):
                    w[j] = (1 - lr / n) * w[j] + lr * self.C * sign * xi[j]
                b += lr * self.C * sign
            else:
                for j in range(d):
                    w[j] = (1 - lr / n) * w[j]
        self.coef_ = w
        self.intercept_ = b
        return self

    def predict(self, X) -> List[float]:
        X2d = _to_list2d(X)
        d = len(X2d[0])
        return [sum(X2d[i][j] * self.coef_[j] for j in range(d)) + self.intercept_
                for i in range(len(X2d))]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = [float(v) for v in _to_list1d(y)]
        n = len(y_true)
        mean_y = sum(y_true) / n
        ss_res = sum((y_true[i] - preds[i]) ** 2 for i in range(n))
        ss_tot = sum((v - mean_y) ** 2 for v in y_true) + 1e-10
        return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# SVC (kernel SVM via simplified SMO)
# ---------------------------------------------------------------------------

class SVC:
    """Kernel Support Vector Classifier (simplified SMO solver)."""

    def __init__(self, C: float = 1.0, kernel: str = 'rbf',
                 degree: int = 3, gamma: Union[str, float] = 'scale',
                 coef0: float = 0.0, max_iter: int = 1000,
                 tol: float = 1e-3, probability: bool = False):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.max_iter = max_iter
        self.tol = tol
        self.probability = probability
        self.classes_: Optional[List] = None
        self._X_fit: Optional[List[List[float]]] = None
        self._gamma_val: float = 1.0

    def _kern_row(self, xi: List[float], xj: List[float]) -> float:
        if self.kernel == 'linear':
            return _linear_kernel_row(xi, xj)
        elif self.kernel == 'rbf':
            return _rbf_kernel_row(xi, xj, self._gamma_val)
        elif self.kernel == 'poly':
            return _poly_kernel_row(xi, xj, self.degree, self._gamma_val, self.coef0)
        elif self.kernel == 'sigmoid':
            return _sigmoid_kernel_row(xi, xj, self._gamma_val, self.coef0)
        return _rbf_kernel_row(xi, xj, self._gamma_val)

    def _fit_binary(self, X: List[List[float]], y_bin: List[float]) -> tuple:
        n = len(X)
        alpha = [0.0] * n
        b = 0.0
        K = [[self._kern_row(X[i], X[j]) for j in range(n)] for i in range(n)]
        for _ in range(self.max_iter):
            updated = 0
            for i in range(n):
                yi = y_bin[i]
                Ei = sum(alpha[k] * y_bin[k] * K[i][k] for k in range(n)) + b - yi
                if (yi * Ei < -self.tol and alpha[i] < self.C) or \
                   (yi * Ei > self.tol and alpha[i] > 0):
                    j = random.randint(0, n - 1)
                    while j == i:
                        j = random.randint(0, n - 1)
                    yj = y_bin[j]
                    Ej = sum(alpha[k] * y_bin[k] * K[j][k] for k in range(n)) + b - yj
                    if yi != yj:
                        L = max(0.0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0.0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])
                    if H <= L:
                        continue
                    eta = 2 * K[i][j] - K[i][i] - K[j][j]
                    if eta >= 0:
                        continue
                    alpha_j_old = alpha[j]
                    alpha[j] -= yj * (Ei - Ej) / eta
                    alpha[j] = max(L, min(H, alpha[j]))
                    if abs(alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    alpha_i_old = alpha[i]
                    alpha[i] += yi * yj * (alpha_j_old - alpha[j])
                    b1 = b - Ei - yi * (alpha[i] - alpha_i_old) * K[i][i] \
                         - yj * (alpha[j] - alpha_j_old) * K[i][j]
                    b2 = b - Ej - yi * (alpha[i] - alpha_i_old) * K[i][j] \
                         - yj * (alpha[j] - alpha_j_old) * K[j][j]
                    b = (b1 + b2) / 2
                    updated += 1
            if updated == 0:
                break
        return alpha, b

    def fit(self, X, y) -> 'SVC':
        X2d = _to_list2d(X)
        y1d = _to_list1d(y)
        self.classes_ = sorted(set(y1d))
        self._gamma_val = _get_gamma(X2d, self.gamma)
        self._X_fit = X2d
        if len(self.classes_) == 2:
            y_bin = [1.0 if v == self.classes_[1] else -1.0 for v in y1d]
            self._alpha, self._b = self._fit_binary(X2d, y_bin)
            self._y_bin = y_bin
        else:
            self._ovo_classifiers = []
            for ci in self.classes_:
                for cj in [c for c in self.classes_ if c > ci]:
                    mask = [i for i, v in enumerate(y1d) if v == ci or v == cj]
                    X_sub = [X2d[i] for i in mask]
                    y_sub = [1.0 if y1d[i] == ci else -1.0 for i in mask]
                    alpha, b = self._fit_binary(X_sub, y_sub)
                    self._ovo_classifiers.append({
                        'ci': ci, 'cj': cj, 'X': X_sub,
                        'alpha': alpha, 'y': y_sub, 'b': b
                    })
        return self

    def predict(self, X) -> List:
        X2d = _to_list2d(X)
        if len(self.classes_) == 2:
            scores = [sum(self._alpha[k] * self._y_bin[k] * self._kern_row(X2d[i], self._X_fit[k])
                         for k in range(len(self._X_fit))) + self._b
                      for i in range(len(X2d))]
            return [self.classes_[1] if s >= 0 else self.classes_[0] for s in scores]
        votes = [[0] * len(self.classes_) for _ in range(len(X2d))]
        ci_to_idx = {c: i for i, c in enumerate(self.classes_)}
        for clf in self._ovo_classifiers:
            X_tr, alpha, y_bin, b = clf['X'], clf['alpha'], clf['y'], clf['b']
            for i in range(len(X2d)):
                s = sum(alpha[k] * y_bin[k] * self._kern_row(X2d[i], X_tr[k])
                        for k in range(len(X_tr))) + b
                if s >= 0:
                    votes[i][ci_to_idx[clf['ci']]] += 1
                else:
                    votes[i][ci_to_idx[clf['cj']]] += 1
        return [self.classes_[max(range(len(self.classes_)), key=lambda k: votes[i][k])]
                for i in range(len(X2d))]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        return sum(1 for p, t in zip(preds, y_true) if p == t) / max(len(y_true), 1)


class SVR:
    """Kernel Support Vector Regressor (delegates to LinearSVR)."""

    def __init__(self, C: float = 1.0, kernel: str = 'rbf',
                 epsilon: float = 0.1, gamma: Union[str, float] = 'scale',
                 degree: int = 3, coef0: float = 0.0, max_iter: int = 1000):
        self.C = C
        self.kernel = kernel
        self.epsilon = epsilon
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.max_iter = max_iter
        self._linear = LinearSVR(C=C, epsilon=epsilon, max_iter=max_iter)

    def fit(self, X, y) -> 'SVR':
        self._linear.fit(X, y)
        return self

    def predict(self, X) -> List[float]:
        return self._linear.predict(X)

    def score(self, X, y) -> float:
        return self._linear.score(X, y)


class OneClassSVM:
    """One-Class SVM for anomaly detection (distance-based)."""

    def __init__(self, nu: float = 0.5, kernel: str = 'rbf',
                 gamma: Union[str, float] = 'scale', max_iter: int = 1000):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        self._threshold: float = 0.0
        self._mean: Optional[List[float]] = None
        self._gamma_val: float = 1.0

    def fit(self, X) -> 'OneClassSVM':
        X2d = _to_list2d(X)
        n, p = len(X2d), len(X2d[0])
        self._gamma_val = _get_gamma(X2d, self.gamma)
        self._mean = [sum(X2d[i][j] for i in range(n)) / n for j in range(p)]
        scores = [sum((X2d[i][j] - self._mean[j]) ** 2 for j in range(p)) for i in range(n)]
        sorted_scores = sorted(scores)
        k = max(0, min(int((1 - self.nu) * n), n - 1))
        self._threshold = sorted_scores[k]
        return self

    def decision_function(self, X) -> List[float]:
        X2d = _to_list2d(X)
        p = len(X2d[0])
        scores = [sum((X2d[i][j] - self._mean[j]) ** 2 for j in range(p)) for i in range(len(X2d))]
        return [self._threshold - s for s in scores]

    def predict(self, X) -> List[int]:
        return [1 if s >= 0 else -1 for s in self.decision_function(X)]

    def score_samples(self, X) -> List[float]:
        return self.decision_function(X)


__all__ = [
    'LinearSVC', 'LinearSVR', 'SVC', 'SVR', 'OneClassSVM',
]
