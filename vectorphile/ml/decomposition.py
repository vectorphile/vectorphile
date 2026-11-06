"""Dimensionality reduction."""
import numpy as np
from typing import Optional


class PCA:
    def __init__(self, n_components=None, whiten=False, random_state=None):
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None

    def fit(self, X) -> 'PCA':
        X = np.array(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        n = len(X)
        k = self.n_components or Vt.shape[0]
        self.components_ = Vt[:k]
        self.singular_values_ = S[:k]
        self.explained_variance_ = S[:k]**2 / (n - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / (S**2 / (n - 1)).sum()
        return self

    def transform(self, X) -> np.ndarray:
        X = np.array(X, dtype=float)
        Xc = X - self.mean_
        result = Xc @ self.components_.T
        if self.whiten:
            result /= np.sqrt(self.explained_variance_) + 1e-12
        return result

    def fit_transform(self, X) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X) -> np.ndarray:
        X = np.array(X, dtype=float)
        if self.whiten:
            X = X * np.sqrt(self.explained_variance_)
        return X @ self.components_ + self.mean_


class TruncatedSVD:
    def __init__(self, n_components=2, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None

    def fit(self, X) -> 'TruncatedSVD':
        X = np.array(X, dtype=float)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        k = self.n_components
        self.components_ = Vt[:k]
        self.singular_values_ = S[:k]
        total = (S**2).sum()
        self.explained_variance_ = S[:k]**2 / len(X)
        self.explained_variance_ratio_ = S[:k]**2 / (total + 1e-12)
        return self

    def transform(self, X) -> np.ndarray:
        return np.array(X, dtype=float) @ self.components_.T

    def fit_transform(self, X) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class NMF:
    def __init__(self, n_components=2, max_iter=200, tol=1e-4, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.components_: Optional[np.ndarray] = None
        self.reconstruction_err_: float = np.inf

    def fit_transform(self, X) -> np.ndarray:
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = np.array(X, dtype=float)
        X = np.maximum(X, 0)
        n, m = X.shape
        k = self.n_components
        W = np.random.rand(n, k) + 0.1
        H = np.random.rand(k, m) + 0.1
        for _ in range(self.max_iter):
            WH = W @ H + 1e-10
            H *= (W.T @ X) / (W.T @ WH)
            WH = W @ H + 1e-10
            W *= (X @ H.T) / (WH @ H.T)
        self.components_ = H
        self.reconstruction_err_ = float(np.linalg.norm(X - W @ H))
        return W

    def fit(self, X) -> 'NMF':
        self.fit_transform(X)
        return self

    def transform(self, X) -> np.ndarray:
        X = np.maximum(np.array(X, dtype=float), 0)
        H = self.components_
        W, _ = np.linalg.lstsq(H.T, X.T, rcond=None)
        return np.maximum(W.T, 0)


class FastICA:
    def __init__(self, n_components=None, max_iter=200, tol=1e-4, fun='logcosh', random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.fun = fun
        self.random_state = random_state
        self.components_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None

    def _g(self, x):
        if self.fun == 'logcosh':
            return np.tanh(x), (1 - np.tanh(x)**2).mean(axis=-1)
        return x * np.exp(-x**2 / 2), (1 - x**2) * np.exp(-x**2 / 2).mean(axis=-1)

    def fit_transform(self, X) -> np.ndarray:
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = np.array(X, dtype=float)
        n, p = X.shape
        self.mean_ = X.mean(axis=0)
        X = X - self.mean_
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        k = self.n_components or min(n, p)
        X_white = (X @ Vt[:k].T) / (S[:k] / np.sqrt(n) + 1e-10)
        W = np.random.randn(k, k)
        for i in range(self.max_iter):
            WX = W @ X_white.T
            g, g_prime = self._g(WX)
            W_new = (g @ X_white) / n - g_prime[:, None] * W
            # Orthogonalization
            u, _, vt = np.linalg.svd(W_new)
            W_new = u @ vt
            if np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1)) < self.tol:
                W = W_new
                break
            W = W_new
        self.components_ = W @ np.diag(1.0 / (S[:k] / np.sqrt(n) + 1e-10)) @ Vt[:k]
        return X_white @ W.T

    def fit(self, X) -> 'FastICA':
        self.fit_transform(X)
        return self

    def transform(self, X) -> np.ndarray:
        X = np.array(X, dtype=float) - self.mean_
        return X @ self.components_.T
