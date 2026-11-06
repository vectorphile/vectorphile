"""Linear models."""
import numpy as np
from typing import Optional


class _BaseLinear:
    def __init__(self):
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def predict(self, X) -> np.ndarray:
        X = np.array(X)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y) -> float:
        y = np.array(y)
        pred = self.predict(X)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / (ss_tot + 1e-12)


class LinearRegression(_BaseLinear):
    def __init__(self, fit_intercept=True):
        super().__init__()
        self.fit_intercept = fit_intercept

    def fit(self, X, y) -> 'LinearRegression':
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        if self.fit_intercept:
            X = np.c_[np.ones(len(X)), X]
        sol, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        if self.fit_intercept:
            self.intercept_, self.coef_ = float(sol[0]), sol[1:]
        else:
            self.coef_ = sol
        return self


class Ridge(_BaseLinear):
    def __init__(self, alpha=1.0, fit_intercept=True):
        super().__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y) -> 'Ridge':
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        if self.fit_intercept:
            X_mean, y_mean = X.mean(0), y.mean()
            Xc = X - X_mean
            sol = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(Xc.shape[1]), Xc.T @ (y - y_mean))
            self.coef_ = sol
            self.intercept_ = float(y_mean - X_mean @ sol)
        else:
            sol = np.linalg.solve(X.T @ X + self.alpha * np.eye(X.shape[1]), X.T @ y)
            self.coef_ = sol
        return self


class Lasso(_BaseLinear):
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4, fit_intercept=True):
        super().__init__()
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X, y) -> 'Lasso':
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        if self.fit_intercept:
            X_mean, y_mean = X.mean(0), y.mean()
            X, y = X - X_mean, y - y_mean
        n, p = X.shape
        coef = np.zeros(p)
        for _ in range(self.max_iter):
            coef_old = coef.copy()
            for j in range(p):
                r = y - X @ coef + X[:, j] * coef[j]
                rho = X[:, j] @ r
                z = X[:, j] @ X[:, j]
                if rho < -self.alpha * n / 2:
                    coef[j] = (rho + self.alpha * n / 2) / z
                elif rho > self.alpha * n / 2:
                    coef[j] = (rho - self.alpha * n / 2) / z
                else:
                    coef[j] = 0.0
            if np.max(np.abs(coef - coef_old)) < self.tol:
                break
        self.coef_ = coef
        if self.fit_intercept:
            self.intercept_ = float(y_mean - X_mean @ coef)
        return self


class ElasticNet(_BaseLinear):
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4, fit_intercept=True):
        super().__init__()
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X, y) -> 'ElasticNet':
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        if self.fit_intercept:
            X_mean, y_mean = X.mean(0), y.mean()
            X, y = X - X_mean, y - y_mean
        n, p = X.shape
        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1 - self.l1_ratio)
        coef = np.zeros(p)
        for _ in range(self.max_iter):
            coef_old = coef.copy()
            for j in range(p):
                r = y - X @ coef + X[:, j] * coef[j]
                rho = X[:, j] @ r
                z = X[:, j] @ X[:, j] + l2 * n
                thresh = l1 * n / 2
                if rho < -thresh:
                    coef[j] = (rho + thresh) / z
                elif rho > thresh:
                    coef[j] = (rho - thresh) / z
                else:
                    coef[j] = 0.0
            if np.max(np.abs(coef - coef_old)) < self.tol:
                break
        self.coef_ = coef
        if self.fit_intercept:
            self.intercept_ = float(y_mean - X_mean @ coef)
        return self


class LogisticRegression:
    def __init__(self, C=1.0, max_iter=200, tol=1e-4, fit_intercept=True, multi_class='ovr'):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.multi_class = multi_class
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _fit_binary(self, X, y, lr=0.1):
        n, p = X.shape
        w = np.zeros(p + (1 if self.fit_intercept else 0))
        for _ in range(self.max_iter):
            if self.fit_intercept:
                z = X @ w[1:] + w[0]
            else:
                z = X @ w
            prob = self._sigmoid(z)
            error = prob - y
            if self.fit_intercept:
                dw = np.concatenate([[error.mean()], X.T @ error / n + w[1:] / (self.C * n)])
            else:
                dw = X.T @ error / n + w / (self.C * n)
            w -= lr * dw
            if np.max(np.abs(dw)) < self.tol:
                break
        return w

    def fit(self, X, y) -> 'LogisticRegression':
        X, y = np.array(X, dtype=float), np.array(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            y_bin = (y == self.classes_[1]).astype(float)
            w = self._fit_binary(X, y_bin)
            if self.fit_intercept:
                self.coef_ = w[1:][np.newaxis]
                self.intercept_ = np.array([w[0]])
            else:
                self.coef_ = w[np.newaxis]
                self.intercept_ = np.array([0.0])
        else:
            self.coef_ = []
            self.intercept_ = []
            for cls in self.classes_:
                y_bin = (y == cls).astype(float)
                w = self._fit_binary(X, y_bin)
                if self.fit_intercept:
                    self.coef_.append(w[1:])
                    self.intercept_.append(w[0])
                else:
                    self.coef_.append(w)
                    self.intercept_.append(0.0)
            self.coef_ = np.array(self.coef_)
            self.intercept_ = np.array(self.intercept_)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.array(X, dtype=float)
        logits = X @ self.coef_.T + self.intercept_
        probs = self._sigmoid(logits)
        if probs.ndim == 1 or probs.shape[1] == 1:
            p = probs.flatten()
            return np.c_[1 - p, p]
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.array(y)))


class Perceptron:
    def __init__(self, max_iter=1000, tol=1e-3, eta=1.0, fit_intercept=True):
        self.max_iter = max_iter
        self.tol = tol
        self.eta = eta
        self.fit_intercept = fit_intercept
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X, y) -> 'Perceptron':
        X, y = np.array(X, dtype=float), np.array(y)
        classes = np.unique(y)
        y_bin = np.where(y == classes[0], -1, 1)
        n, p = X.shape
        w = np.zeros(p)
        b = 0.0
        for _ in range(self.max_iter):
            errors = 0
            for xi, yi in zip(X, y_bin):
                pred = np.sign(np.dot(w, xi) + b)
                if pred != yi:
                    w += self.eta * yi * xi
                    b += self.eta * yi
                    errors += 1
            if errors == 0:
                break
        self.coef_ = w
        self.intercept_ = b
        self.classes_ = classes
        return self

    def predict(self, X) -> np.ndarray:
        X = np.array(X, dtype=float)
        raw = np.sign(X @ self.coef_ + self.intercept_)
        return np.where(raw <= 0, self.classes_[0], self.classes_[1])

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.array(y)))
