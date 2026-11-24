"""Dimensionality reduction — pure Python."""
import math
import random as _random
from typing import Optional
from vectorphile.ml._np_compat import to_array as _ta


def _col(X, j):
    return [X[i][j] for i in range(len(X))]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _mat_mul(A, B):
    n, k = len(A), len(A[0])
    m = len(B[0])
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for l in range(k):
            if A[i][l] == 0.0:
                continue
            for j in range(m):
                C[i][j] += A[i][l] * B[l][j]
    return C


def _transpose(A):
    n, m = len(A), len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]


def _norm(v):
    return math.sqrt(sum(x ** 2 for x in v))


def _normalize(v):
    n = _norm(v)
    return [x / (n + 1e-15) for x in v]


def _mean_center(X):
    n, p = len(X), len(X[0])
    mu = [sum(_col(X, j)) / n for j in range(p)]
    return [[X[i][j] - mu[j] for j in range(p)] for i in range(n)], mu


def _power_iter_top_k(C, k, n_iter=200, seed=42):
    """Approximate top-k eigenvectors of symmetric matrix C via deflation + power iteration."""
    p = len(C)
    rng = _random.Random(seed)
    eigenvecs = []
    eigenvals = []
    # Working copy of C (deflated)
    Cd = [row[:] for row in C]
    for _ in range(k):
        # Random init
        v = _normalize([rng.gauss(0, 1) for _ in range(p)])
        for _ in range(n_iter):
            # v <- Cd @ v
            new_v = [sum(Cd[i][j] * v[j] for j in range(p)) for i in range(p)]
            lam = _norm(new_v)
            if lam < 1e-15:
                break
            v = [x / lam for x in new_v]
        lam = sum(sum(Cd[i][j] * v[j] for j in range(p)) * v[i] for i in range(p))
        eigenvecs.append(v)
        eigenvals.append(lam)
        # Deflate: Cd -= lam * v @ v^T
        for i in range(p):
            for j in range(p):
                Cd[i][j] -= lam * v[i] * v[j]
    return eigenvals, eigenvecs


class PCA:
    def __init__(self, n_components=None, whiten=False, random_state=None):
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state

    def fit(self, X):
        X2 = [[float(v) for v in row] for row in X]
        n, p = len(X2), len(X2[0])
        Xc, self.mean_ = _mean_center(X2)
        k = self.n_components or p
        # Covariance matrix
        XT = _transpose(Xc)
        C = _mat_mul(XT, Xc)
        C = [[C[i][j] / (n - 1) for j in range(p)] for i in range(p)]
        eigenvals, eigenvecs = _power_iter_top_k(C, k, seed=self.random_state or 0)
        # Sort by descending eigenvalue
        order = sorted(range(k), key=lambda i: -eigenvals[i])
        self.components_ = [eigenvecs[i] for i in order]
        self.explained_variance_ = [max(eigenvals[i], 0.0) for i in order]
        total_var = sum(max(v, 0) for v in eigenvals) + 1e-15
        self.explained_variance_ratio_ = _ta([v / total_var for v in self.explained_variance_])
        return self

    def transform(self, X):
        X2 = [[float(v) for v in row] for row in X]
        Xc = [[X2[i][j] - self.mean_[j] for j in range(len(self.mean_))] for i in range(len(X2))]
        result = [[_dot(row, comp) for comp in self.components_] for row in Xc]
        if self.whiten:
            result = [[result[i][k] / math.sqrt(max(self.explained_variance_[k], 1e-15))
                       for k in range(len(self.components_))] for i in range(len(result))]
        return _ta(result)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        result = []
        for row in X:
            rec = [self.mean_[j] + sum(row[k] * self.components_[k][j]
                                        for k in range(len(row)))
                   for j in range(len(self.mean_))]
            result.append(rec)
        return result


class TruncatedSVD:
    def __init__(self, n_components=2, n_iter=5, random_state=None):
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X):
        X2 = [[float(v) for v in row] for row in X]
        n, p = len(X2), len(X2[0])
        XT = _transpose(X2)
        # Approximate SVD via power iteration on X^T X
        XTX = _mat_mul(XT, X2)
        k = min(self.n_components, min(n, p))
        eigenvals, eigenvecs = _power_iter_top_k(XTX, k, n_iter=100, seed=self.random_state or 0)
        self.components_ = eigenvecs
        self.singular_values_ = [math.sqrt(max(v, 0)) for v in eigenvals]
        self.explained_variance_ = [max(v, 0) / max(n - 1, 1) for v in eigenvals]
        total = sum(self.explained_variance_) + 1e-15
        self.explained_variance_ratio_ = [v / total for v in self.explained_variance_]
        return self

    def transform(self, X):
        X2 = [[float(v) for v in row] for row in X]
        return _ta([[_dot(row, comp) for comp in self.components_] for row in X2])

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        p = len(self.components_[0]) if self.components_ else 0
        result = []
        for row in X:
            rec = [sum(row[k] * self.components_[k][j]
                       for k in range(len(row))) for j in range(p)]
            result.append(rec)
        return result


class NMF:
    """Non-negative Matrix Factorization via multiplicative updates."""

    def __init__(self, n_components=2, max_iter=200, tol=1e-4, random_state=None,
                 init='random', l1_ratio=0.0, alpha_W=0.0):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit_transform(self, X):
        X2 = [[max(float(v), 0.0) for v in row] for row in X]
        n, p = len(X2), len(X2[0])
        k = self.n_components
        rng = _random.Random(self.random_state)
        eps = 1e-10
        W = [[rng.random() for _ in range(k)] for _ in range(n)]
        H = [[rng.random() for _ in range(p)] for _ in range(k)]
        for _ in range(self.max_iter):
            # Update H: H *= (W^T X) / (W^T W H + eps)
            WTX = _mat_mul(_transpose(W), X2)
            WTW = _mat_mul(_transpose(W), W)
            WTWH = _mat_mul(WTW, H)
            H = [[H[i][j] * WTX[i][j] / (WTWH[i][j] + eps) for j in range(p)] for i in range(k)]
            # Update W: W *= (X H^T) / (W H H^T + eps)
            XHT = _mat_mul(X2, _transpose(H))
            HHT = _mat_mul(H, _transpose(H))
            WHHT = _mat_mul(W, HHT)
            W = [[W[i][j] * XHT[i][j] / (WHHT[i][j] + eps) for j in range(k)] for i in range(n)]
        self.components_ = H
        self.reconstruction_err_ = sum(
            (X2[i][j] - sum(W[i][l] * H[l][j] for l in range(k))) ** 2
            for i in range(n) for j in range(p))
        self._W = W
        return _ta(W)

    def fit(self, X):
        self.fit_transform(X)
        return self

    def transform(self, X):
        X2 = [[max(float(v), 0.0) for v in row] for row in X]
        n, p = len(X2), len(X2[0])
        k = self.n_components
        rng = _random.Random(42)
        eps = 1e-10
        W = [[rng.random() for _ in range(k)] for _ in range(n)]
        HHT = _mat_mul(self.components_, _transpose(self.components_))
        XHT = _mat_mul(X2, _transpose(self.components_))
        for _ in range(100):
            WHHT = _mat_mul(W, HHT)
            W = [[W[i][j] * XHT[i][j] / (WHHT[i][j] + eps) for j in range(k)] for i in range(n)]
        return _ta(W)


class FactorAnalysis:
    def __init__(self, n_components=None, max_iter=1000, tol=1e-2, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        X2 = [[float(v) for v in row] for row in X]
        Xc, self.mean_ = _mean_center(X2)
        p = len(Xc[0])
        k = self.n_components or min(p, 2)
        XT = _transpose(Xc)
        C = _mat_mul(XT, Xc)
        n = len(Xc)
        C = [[C[i][j] / max(n - 1, 1) for j in range(p)] for i in range(p)]
        eigenvals, eigenvecs = _power_iter_top_k(C, k, seed=self.random_state or 0)
        self.components_ = eigenvecs
        return self

    def transform(self, X):
        X2 = [[float(v) for v in row] for row in X]
        Xc = [[X2[i][j] - self.mean_[j] for j in range(len(self.mean_))] for i in range(len(X2))]
        return _ta([[_dot(row, comp) for comp in self.components_] for row in Xc])

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class FastICA:
    """Independent Component Analysis via FastICA algorithm (pure Python)."""

    def __init__(self, n_components=None, max_iter=200, tol=1e-4, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _tanh(self, u):
        try:
            e2 = math.exp(min(2.0 * u, 50.0))
        except OverflowError:
            return 1.0
        return (e2 - 1.0) / (e2 + 1.0)

    def _g(self, u):
        t = self._tanh(u)
        return t, 1.0 - t * t

    def _whiten(self, X):
        n, p = len(X), len(X[0])
        mean = [sum(_col(X, j)) / n for j in range(p)]
        Xc = [[X[i][j] - mean[j] for j in range(p)] for i in range(n)]
        XT = _transpose(Xc)
        C = _mat_mul(XT, Xc)
        C = [[C[i][j] / max(n - 1, 1) for j in range(p)] for i in range(p)]
        k = self.n_components or p
        eigenvals, eigenvecs = _power_iter_top_k(C, k, seed=self.random_state or 0)
        W = []
        for idx in range(k):
            scale = 1.0 / max(math.sqrt(abs(eigenvals[idx])), 1e-10)
            W.append([eigenvecs[idx][j] * scale for j in range(p)])
        X_white = _mat_mul(Xc, _transpose(W))
        return X_white, mean, W

    def fit(self, X):
        X2 = [[float(v) for v in row] for row in X]
        X_white, self.mean_, self._whitening = self._whiten(X2)
        n, k = len(X_white), len(X_white[0])

        rng = _random.Random(self.random_state or 42)
        W = []
        for _ in range(k):
            w = [rng.gauss(0.0, 1.0) for _ in range(k)]
            norm = max(math.sqrt(sum(v * v for v in w)), 1e-10)
            W.append([v / norm for v in w])

        for _ in range(self.max_iter):
            W_new = []
            for c in range(k):
                w = W[c]
                projs = [sum(w[j] * X_white[i][j] for j in range(k)) for i in range(n)]
                g_vals = [self._g(u) for u in projs]
                g_u = [gv[0] for gv in g_vals]
                gp_u = [gv[1] for gv in g_vals]
                e_xg = [sum(X_white[i][j] * g_u[i] for i in range(n)) / n for j in range(k)]
                e_gp = sum(gp_u) / n
                w_new = [e_xg[j] - e_gp * w[j] for j in range(k)]
                W_new.append(w_new)

            # Symmetric decorrelation: W = (W W^T)^{-0.5} W
            WWT = _mat_mul(W_new, _transpose(W_new))
            evals, evecs = _power_iter_top_k(WWT, k, seed=0)
            evecs_T = _transpose(evecs)
            diag_inv_sqrt = [[0.0] * k for _ in range(k)]
            for i in range(k):
                diag_inv_sqrt[i][i] = 1.0 / max(math.sqrt(abs(evals[i])), 1e-10)
            WWT_inv_sqrt = _mat_mul(evecs_T, _mat_mul(diag_inv_sqrt, evecs))
            W_orth = _mat_mul(WWT_inv_sqrt, W_new)

            max_change = 0.0
            for c in range(k):
                dot = abs(sum(W_orth[c][j] * W[c][j] for j in range(k)))
                max_change = max(max_change, abs(1.0 - dot))
            W = W_orth
            if max_change < self.tol:
                break

        self.components_ = W
        return self

    def transform(self, X):
        X2 = [[float(v) for v in row] for row in X]
        p = len(X2[0])
        mean = self.mean_
        Xc = [[X2[i][j] - mean[j] for j in range(p)] for i in range(len(X2))]
        X_white = _mat_mul(Xc, _transpose(self._whitening))
        return _ta(_mat_mul(X_white, _transpose(self.components_)))

    def fit_transform(self, X):
        return self.fit(X).transform(X)
