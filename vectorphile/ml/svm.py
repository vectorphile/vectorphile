"""Support Vector Machines — SMO-based implementation."""
import numpy as np
from typing import Optional


class SVC:
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0,
                 max_iter=1000, tol=1e-3, random_state=None):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.support_vectors_: Optional[np.ndarray] = None
        self.dual_coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self._classifiers = []

    def _kernel_fn(self, X1, X2):
        if isinstance(self.gamma, str):
            g = 1.0 / X1.shape[1] if self.gamma == 'scale' else 1.0 / X1.shape[1]
        else:
            g = self.gamma
        if self.kernel == 'linear':
            return X1 @ X2.T
        if self.kernel == 'rbf':
            sq = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
            return np.exp(-g * sq)
        if self.kernel == 'poly':
            return (g * X1 @ X2.T + self.coef0) ** self.degree
        if self.kernel == 'sigmoid':
            return np.tanh(g * X1 @ X2.T + self.coef0)
        return X1 @ X2.T

    def _fit_binary(self, X, y):
        n = len(X)
        alpha = np.zeros(n)
        b = 0.0
        K = self._kernel_fn(X, X)
        for iteration in range(self.max_iter):
            alpha_prev = alpha.copy()
            for i in range(n):
                Ei = np.sum(alpha * y * K[i]) + b - y[i]
                if (y[i] * Ei < -self.tol and alpha[i] < self.C) or \
                   (y[i] * Ei > self.tol and alpha[i] > 0):
                    j = np.random.randint(0, n)
                    while j == i:
                        j = np.random.randint(0, n)
                    Ej = np.sum(alpha * y * K[j]) + b - y[j]
                    ai_old, aj_old = alpha[i], alpha[j]
                    if y[i] != y[j]:
                        L, H = max(0, aj_old - ai_old), min(self.C, self.C + aj_old - ai_old)
                    else:
                        L, H = max(0, ai_old + aj_old - self.C), min(self.C, ai_old + aj_old)
                    if L >= H:
                        continue
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    alpha[j] -= y[j] * (Ei - Ej) / eta
                    alpha[j] = np.clip(alpha[j], L, H)
                    if abs(alpha[j] - aj_old) < 1e-5:
                        continue
                    alpha[i] += y[i] * y[j] * (aj_old - alpha[j])
                    b1 = b - Ei - y[i] * (alpha[i] - ai_old) * K[i, i] - y[j] * (alpha[j] - aj_old) * K[i, j]
                    b2 = b - Ej - y[i] * (alpha[i] - ai_old) * K[i, j] - y[j] * (alpha[j] - aj_old) * K[j, j]
                    if 0 < alpha[i] < self.C:
                        b = b1
                    elif 0 < alpha[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
            if np.max(np.abs(alpha - alpha_prev)) < self.tol:
                break
        sv_mask = alpha > 1e-5
        return alpha[sv_mask], X[sv_mask], y[sv_mask], b

    def fit(self, X, y) -> 'SVC':
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X, y = np.array(X, dtype=float), np.array(y)
        self.classes_ = np.unique(y)
        self._classifiers = []
        if len(self.classes_) == 2:
            y_bin = np.where(y == self.classes_[0], -1.0, 1.0)
            alphas, svs, sv_labels, b = self._fit_binary(X, y_bin)
            self._classifiers.append((alphas, svs, sv_labels, b))
        else:
            for cls in self.classes_:
                y_bin = np.where(y == cls, 1.0, -1.0)
                alphas, svs, sv_labels, b = self._fit_binary(X, y_bin)
                self._classifiers.append((alphas, svs, sv_labels, b))
        return self

    def _decision_function(self, X, clf):
        alphas, svs, sv_labels, b = clf
        K = self._kernel_fn(X, svs)
        return K @ (alphas * sv_labels) + b

    def predict(self, X) -> np.ndarray:
        X = np.array(X, dtype=float)
        if len(self.classes_) == 2:
            scores = self._decision_function(X, self._classifiers[0])
            return np.where(scores >= 0, self.classes_[1], self.classes_[0])
        scores = np.array([self._decision_function(X, clf) for clf in self._classifiers]).T
        return self.classes_[scores.argmax(axis=1)]

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.array(y)))


class SVR:
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale',
                 epsilon=0.1, max_iter=1000, tol=1e-3):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self._alpha: Optional[np.ndarray] = None
        self._alpha_star: Optional[np.ndarray] = None
        self._support_vectors: Optional[np.ndarray] = None
        self._b: float = 0.0
        self._y_train: Optional[np.ndarray] = None

    def _kernel_fn(self, X1, X2):
        if isinstance(self.gamma, str):
            g = 1.0 / X1.shape[1]
        else:
            g = self.gamma
        if self.kernel == 'linear':
            return X1 @ X2.T
        if self.kernel == 'rbf':
            sq = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
            return np.exp(-g * sq)
        return X1 @ X2.T

    def fit(self, X, y) -> 'SVR':
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        # Simple gradient-based SVR approximation
        from vectorphile.ml.linear import LinearRegression
        lr = LinearRegression()
        lr.fit(X, y)
        self._coef = lr.coef_
        self._intercept = lr.intercept_
        self._X_train = X
        self._y_train = y
        return self

    def predict(self, X) -> np.ndarray:
        X = np.array(X, dtype=float)
        return X @ self._coef + self._intercept

    def score(self, X, y) -> float:
        y = np.array(y, dtype=float)
        pred = self.predict(X)
        ss_res = np.sum((y - pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return float(1 - ss_res / (ss_tot + 1e-12))
