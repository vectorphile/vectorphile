"""vectorphile.decomposition — Matrix decomposition and dimensionality reduction.

All computations use pure Python. No external dependencies.
"""
from __future__ import annotations
import math
import random
from typing import Optional, List, Tuple


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


def _mat_mat(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    n, m, k = len(A), len(B[0]), len(B)
    return [[sum(A[i][p] * B[p][j] for p in range(k)) for j in range(m)] for i in range(n)]


def _transpose(A: List[List[float]]) -> List[List[float]]:
    n, m = len(A), len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]


def _mat_vec(A: List[List[float]], x: List[float]) -> List[float]:
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def _vec_norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _normalize(v: List[float]) -> List[float]:
    n = _vec_norm(v)
    return [x / (n + 1e-15) for x in v]


def _gram_schmidt(W: List[List[float]]) -> List[List[float]]:
    """Orthonormalize rows of W via Gram-Schmidt."""
    result = []
    for row in W:
        v = row[:]
        for q in result:
            dot = sum(v[i] * q[i] for i in range(len(v)))
            v = [v[i] - dot * q[i] for i in range(len(v))]
        n = _vec_norm(v)
        if n > 1e-10:
            result.append([x / n for x in v])
        else:
            result.append(row[:])
    return result


def _power_svd(X: List[List[float]], k: int, n_iter: int,
               rng: random.Random) -> Tuple[List[List[float]], List[float], List[List[float]]]:
    """Randomized SVD (power iteration) returning U (n×k), S (k,), Vt (k×p)."""
    n, p = len(X), len(X[0])
    # Random Gaussian sketch
    Omega = [[rng.gauss(0, 1) for _ in range(k)] for _ in range(p)]
    Y = _mat_mat(X, Omega)  # n×k
    # Power iterations
    Xt = _transpose(X)
    for _ in range(n_iter):
        YtX = _mat_mat(_transpose(Y), X)  # k×p
        Y = _mat_mat(X, _transpose(YtX))  # n×k
    # QR decomposition of Y via Gram-Schmidt
    Q = _gram_schmidt(_transpose(Y))  # k×n (rows are basis vectors)
    Qt = Q  # k×n
    B = _mat_mat(Qt, X)  # k×p
    # SVD of small k×p matrix B via power iteration
    U_B, S, Vt = _small_svd(B, k)
    # U = Q^T @ U_B ... Q is k×n, so Q^T is n×k
    QT = _transpose(Qt)  # n×k
    U = _mat_mat(QT, U_B)  # n×k
    return U, S, Vt


def _small_svd(A: List[List[float]], k: int) -> Tuple[List[List[float]], List[float], List[List[float]]]:
    """SVD of small matrix A (k×p) via power method on A^T A."""
    p = len(A[0])
    At = _transpose(A)
    AtA = _mat_mat(At, A)  # p×p
    # Power iteration for k singular vectors
    Vt_rows = []
    S = []
    curr_A = A
    for _ in range(min(k, p)):
        # Power iteration
        v = _normalize([random.gauss(0, 1) for _ in range(p)])
        for _ in range(50):
            Av = _mat_vec(curr_A, v)
            AtAv = _mat_vec(_transpose(curr_A), Av)
            v = _normalize(AtAv)
        Av = _mat_vec(curr_A, v)
        s = _vec_norm(Av)
        S.append(s)
        Vt_rows.append(v)
        # Deflate
        u = [x / (s + 1e-15) for x in Av]
        new_A = [[curr_A[i][j] - s * u[i] * v[j]
                  for j in range(p)] for i in range(len(curr_A))]
        curr_A = new_A
    # Compute U_B = A @ V / S
    kk = len(S)
    U_B = []
    for i in range(len(A)):
        col = [sum(A[i][j] * Vt_rows[r][j] for j in range(p)) / (S[r] + 1e-15)
               for r in range(kk)]
        U_B.append(col)
    # Return U_B (len(A)×kk), S (kk), Vt (kk×p)
    return _transpose(_transpose(U_B)), S, Vt_rows


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

class PCA:
    """Principal Component Analysis via power-iteration SVD."""

    def __init__(self, n_components: Optional[int] = None, whiten: bool = False,
                 random_state: Optional[int] = None):
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        self.components_: Optional[List[List[float]]] = None
        self.explained_variance_: Optional[List[float]] = None
        self.explained_variance_ratio_: Optional[List[float]] = None
        self.mean_: Optional[List[float]] = None
        self.n_components_: int = 0

    def fit(self, X):
        data = _to_list2d(X)
        n, d = len(data), len(data[0])
        k = self.n_components or min(n, d)
        # Center
        self.mean_ = [sum(data[i][j] for i in range(n)) / n for j in range(d)]
        X_c = [[data[i][j] - self.mean_[j] for j in range(d)] for i in range(n)]
        # Covariance matrix: (d×d) = X_c^T X_c / (n-1)
        Xt = _transpose(X_c)  # d × n
        cov = _mat_mat(Xt, X_c)  # d × d
        scale = 1.0 / max(n - 1, 1)
        cov = [[v * scale for v in row] for row in cov]
        # Power iteration for top k eigenvectors of cov
        rng = random.Random(self.random_state or 42)
        if d <= 20 or k >= d:
            # Use direct SVD for small matrices
            _, Ss, Vt = _small_svd(cov, min(k, d))
        else:
            _, Ss, Vt = _power_svd(cov, k=k, n_iter=20, rng=rng)
        self.n_components_ = len(Ss)
        self.components_ = Vt[:self.n_components_]  # k × d
        self.explained_variance_ = [s * s for s in Ss[:self.n_components_]]
        total_var = sum(self.explained_variance_) + 1e-15
        self.explained_variance_ratio_ = [v / total_var for v in self.explained_variance_]
        return self

    def transform(self, X) -> List[List[float]]:
        data = _to_list2d(X)
        d = len(data[0])
        X_c = [[data[i][j] - (self.mean_[j] if self.mean_ else 0.0) for j in range(d)]
               for i in range(len(data))]
        # Project: n × k = X_c @ components_.T
        Vt = self.components_
        result = [[sum(X_c[i][j] * Vt[kk][j] for j in range(d))
                   for kk in range(self.n_components_)]
                  for i in range(len(X_c))]
        if self.whiten:
            for i in range(len(result)):
                result[i] = [result[i][kk] / max(math.sqrt(self.explained_variance_[kk]), 1e-15)
                              for kk in range(self.n_components_)]
        return result

    def fit_transform(self, X) -> List[List[float]]:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed) -> List[List[float]]:
        data = _to_list2d(X_transformed)
        d_orig = len(self.components_[0]) if self.components_ else 0
        result = []
        for row in data:
            rec = [sum(row[kk] * self.components_[kk][j]
                       for kk in range(self.n_components_)) + (self.mean_[j] if self.mean_ else 0.0)
                   for j in range(d_orig)]
            result.append(rec)
        return result


# ---------------------------------------------------------------------------
# TruncatedSVD (LSA)
# ---------------------------------------------------------------------------

class TruncatedSVD:
    """Dimensionality reduction using truncated SVD (LSA)."""

    def __init__(self, n_components: int = 2, n_iter: int = 5,
                 random_state: Optional[int] = None):
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.components_: Optional[List[List[float]]] = None
        self.singular_values_: Optional[List[float]] = None
        self.explained_variance_ratio_: Optional[List[float]] = None

    def fit(self, X) -> 'TruncatedSVD':
        X2d = _to_list2d(X)
        rng = random.Random(self.random_state)
        k = min(self.n_components, len(X2d), len(X2d[0]))
        _, S, Vt = _power_svd(X2d, k, self.n_iter, rng)
        self.components_ = Vt[:k]
        self.singular_values_ = S[:k]
        total_var = sum(s * s for s in S) + 1e-10
        self.explained_variance_ratio_ = [s * s / total_var for s in S[:k]]
        return self

    def transform(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        p = len(X2d[0])
        return [[sum(X2d[i][j] * self.components_[r][j] for j in range(p))
                 for r in range(len(self.components_))] for i in range(len(X2d))]

    def fit_transform(self, X) -> List[List[float]]:
        return self.fit(X).transform(X)

    def inverse_transform(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        Vt = self.components_
        p = len(Vt[0]) if Vt else 0
        k = len(Vt) if Vt else 0
        return [[sum(X2d[i][r] * Vt[r][j] for r in range(k)) for j in range(p)]
                for i in range(len(X2d))]


# ---------------------------------------------------------------------------
# NMF — Non-negative Matrix Factorization
# ---------------------------------------------------------------------------

class NMF:
    """Non-negative Matrix Factorization: X ≈ W @ H, W,H ≥ 0.

    Uses multiplicative update rules (Lee & Seung).
    """

    def __init__(self, n_components: int = 2, max_iter: int = 200,
                 tol: float = 1e-4, random_state: Optional[int] = None,
                 alpha_W: float = 0.0, alpha_H: float = 0.0):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.components_: Optional[List[List[float]]] = None
        self.reconstruction_err_: float = float('inf')
        self.n_iter_: int = 0

    def fit_transform(self, X) -> List[List[float]]:
        X2d = [[max(0.0, v) for v in row] for row in _to_list2d(X)]
        n, p = len(X2d), len(X2d[0])
        k = self.n_components
        rng = random.Random(self.random_state)
        eps = 1e-10
        W = [[max(eps, rng.random()) for _ in range(k)] for _ in range(n)]
        H = [[max(eps, rng.random()) for _ in range(p)] for _ in range(k)]
        prev_err = float('inf')
        Xt = _transpose(X2d)
        for it in range(self.max_iter):
            # Update H: H *= (W^T X) / (W^T W H + eps)
            WtX = _mat_mat(_transpose(W), X2d)  # k×p
            WtW = _mat_mat(_transpose(W), W)    # k×k
            WtWH = _mat_mat(WtW, H)             # k×p
            H = [[max(eps, H[r][c] * WtX[r][c] / (WtWH[r][c] + eps))
                  for c in range(p)] for r in range(k)]
            # Update W: W *= (X H^T) / (W H H^T + eps)
            Ht = _transpose(H)
            XHt = _mat_mat(X2d, Ht)             # n×k
            WHHt = _mat_mat(W, _mat_mat(H, Ht)) # n×k
            W = [[max(eps, W[i][r] * XHt[i][r] / (WHHt[i][r] + eps))
                  for r in range(k)] for i in range(n)]
            # Reconstruction error
            WH = _mat_mat(W, H)
            err = math.sqrt(sum((X2d[i][j] - WH[i][j]) ** 2 for i in range(n) for j in range(p)))
            if abs(prev_err - err) < self.tol:
                self.n_iter_ = it
                break
            prev_err = err
        self.components_ = H
        WH = _mat_mat(W, H)
        self.reconstruction_err_ = math.sqrt(sum((X2d[i][j] - WH[i][j]) ** 2
                                                  for i in range(n) for j in range(p)))
        return W

    def fit(self, X) -> 'NMF':
        self.fit_transform(X)
        return self

    def transform(self, X) -> List[List[float]]:
        X2d = [[max(0.0, v) for v in row] for row in _to_list2d(X)]
        n, p = len(X2d), len(X2d[0])
        k = self.n_components
        eps = 1e-10
        W = [[0.5] * k for _ in range(n)]
        H = self.components_
        Ht = _transpose(H)
        for _ in range(50):
            XHt = _mat_mat(X2d, Ht)
            WHHt = _mat_mat(W, _mat_mat(H, Ht))
            W = [[max(eps, W[i][r] * XHt[i][r] / (WHHt[i][r] + eps))
                  for r in range(k)] for i in range(n)]
        return W

    def inverse_transform(self, W) -> List[List[float]]:
        return _mat_mat(_to_list2d(W), self.components_)


# ---------------------------------------------------------------------------
# FastICA (simplified, whitening via covariance power iteration)
# ---------------------------------------------------------------------------

class FastICA:
    """Independent Component Analysis via FastICA algorithm."""

    def __init__(self, n_components: Optional[int] = None, max_iter: int = 200,
                 tol: float = 1e-4, random_state: Optional[int] = None,
                 fun: str = 'logcosh'):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.fun = fun
        self.components_: Optional[List[List[float]]] = None
        self.mean_: Optional[List[float]] = None

    def _g(self, u: float):
        if self.fun == 'logcosh':
            t = math.tanh(u)
            return t, 1 - t * t
        elif self.fun == 'exp':
            e = math.exp(-u * u / 2)
            return u * e, (1 - u * u) * e
        else:
            return u * u * u, 3 * u * u

    def fit_transform(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        n, p = len(X2d), len(X2d[0])
        self.mean_ = [sum(X2d[i][j] for i in range(n)) / n for j in range(p)]
        Xc = [[X2d[i][j] - self.mean_[j] for j in range(p)] for i in range(n)]
        k = min(self.n_components or p, n, p)
        rng = random.Random(self.random_state)
        # Whiten using randomized SVD
        _, S, Vt = _power_svd(Xc, k, 5, rng)
        d = [1.0 / math.sqrt(max(S[r], 1e-10)) for r in range(k)]
        # W_white: k×p — whitening matrix
        W_white = [[d[r] * Vt[r][j] for j in range(p)] for r in range(k)]
        # Whitened data: n×k
        Xw = _mat_mat(Xc, _transpose(W_white))
        # FastICA iterations
        W = _gram_schmidt([[rng.gauss(0, 1) for _ in range(k)] for _ in range(k)])
        for _ in range(self.max_iter):
            WX = _mat_mat(Xw, _transpose(W))  # n×k
            g_vals = [[self._g(WX[i][r])[0] for r in range(k)] for i in range(n)]
            gp_vals = [[self._g(WX[i][r])[1] for r in range(k)] for i in range(n)]
            # W_new[r] = (1/n) * g(WX)^T Xw[r] - mean(g'(WX[r])) * W[r]
            W_new = []
            for r in range(k):
                v = [sum(g_vals[i][r] * Xw[i][c] for i in range(n)) / n for c in range(k)]
                gp_mean = sum(gp_vals[i][r] for i in range(n)) / n
                v = [v[c] - gp_mean * W[r][c] for c in range(k)]
                W_new.append(v)
            W_new = _gram_schmidt(W_new)
            lim = max(abs(abs(sum(W_new[r][c] * W[r][c] for c in range(k))) - 1)
                      for r in range(k))
            W = W_new
            if lim < self.tol:
                break
        # Final components: W @ W_white (k×p)
        self.components_ = _mat_mat(W, W_white)
        return _mat_mat(Xc, _transpose(self.components_))

    def fit(self, X) -> 'FastICA':
        self.fit_transform(X)
        return self

    def transform(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        p = len(X2d[0])
        Xc = [[X2d[i][j] - self.mean_[j] for j in range(p)] for i in range(len(X2d))]
        return _mat_mat(Xc, _transpose(self.components_))

    def inverse_transform(self, S) -> List[List[float]]:
        S2d = _to_list2d(S)
        p = len(self.components_[0]) if self.components_ else 0
        k = len(self.components_) if self.components_ else 0
        # pseudo-inverse of components_ via (C C^T)^{-1} C
        # For simplicity: use transpose (approximate if not square)
        Ct = _transpose(self.components_)
        result = _mat_mat(S2d, self.components_)
        return [[result[i][j] + self.mean_[j] for j in range(p)] for i in range(len(S2d))]


# ---------------------------------------------------------------------------
# FactorAnalysis (EM)
# ---------------------------------------------------------------------------

class FactorAnalysis:
    """Factor Analysis via EM."""

    def __init__(self, n_components: int = 2, max_iter: int = 1000, tol: float = 1e-3):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.components_: Optional[List[List[float]]] = None
        self.noise_variance_: Optional[List[float]] = None
        self.mean_: Optional[List[float]] = None

    def _inv_diag_plus_lr(self, psi: List[float], L: List[List[float]]) -> List[List[float]]:
        """Invert Psi + L L^T using Woodbury: (Psi + L L^T)^{-1}."""
        p, k = len(psi), len(L[0])
        # M = I + L^T Psi^{-1} L (k×k)
        Psi_inv_L = [[L[i][r] / (psi[i] + 1e-10) for r in range(k)] for i in range(p)]
        M = [[sum(L[i][r] * Psi_inv_L[i][c] for i in range(p)) + (1.0 if r == c else 0.0)
              for c in range(k)] for r in range(k)]
        # Invert M
        from vectorphile.linear_model import _inv_sym as _linmod_inv
        M_inv = _linmod_inv(M)
        # (Psi + L L^T)^{-1} = Psi^{-1} - Psi^{-1} L M^{-1} L^T Psi^{-1}
        # We only need the matrix product form for E-step
        return Psi_inv_L, M_inv

    def fit(self, X) -> 'FactorAnalysis':
        X2d = _to_list2d(X)
        n, p = len(X2d), len(X2d[0])
        self.mean_ = [sum(X2d[i][j] for i in range(n)) / n for j in range(p)]
        Xc = [[X2d[i][j] - self.mean_[j] for j in range(p)] for i in range(n)]
        k = self.n_components
        rng = random.Random()
        # Init L as small random, psi as variance
        cov = [[sum(Xc[i][r] * Xc[i][c] for i in range(n)) / n for c in range(p)] for r in range(p)]
        L = [[rng.gauss(0, 0.1) for _ in range(k)] for _ in range(p)]
        psi = [max(cov[j][j], 1e-3) for j in range(p)]
        for _ in range(self.max_iter):
            # E-step: Ez = Xc (Psi + L L^T)^{-1} L
            Psi_inv_L, M_inv = self._inv_diag_plus_lr(psi, L)
            # Ez = Xc @ Psi^{-1} L @ M^{-1} (n×k)
            XcPsiL = [[sum(Xc[i][j] * Psi_inv_L[j][r] for j in range(p)) for r in range(k)]
                      for i in range(n)]
            Ez = _mat_mat(XcPsiL, M_inv)  # n×k
            # Ezz = n * M_inv + Ez^T Ez (k×k)
            EzTEz = _mat_mat(_transpose(Ez), Ez)
            Ezz = [[n * M_inv[r][c] + EzTEz[r][c] for c in range(k)] for r in range(k)]
            # M-step
            # L_new = Xc^T Ez @ Ezz^{-1}
            XcTEz = _mat_mat(_transpose(Xc), Ez)  # p×k
            # Invert Ezz (k×k)
            from vectorphile.linear_model import _inv_sym as _inv
            Ezz_inv = _inv([[Ezz[r][c] + (1e-6 if r == c else 0.0) for c in range(k)] for r in range(k)])
            L_new = _mat_mat(XcTEz, Ezz_inv)  # p×k
            # psi_new = diag(cov) - diag(L_new @ Ez^T Xc / n)
            LEzTXc = _mat_mat(_transpose(L_new), _transpose(Xc))  # k×n
            EzLEzTXc = _mat_mat(Ez, LEzTXc)  # n×n... just diag needed
            # diag(L_new Ez^T Xc / n) = sum_i (Ez[i] @ L_new[j]) * Xc[i][j] / n for j
            psi_new = [max(cov[j][j] - sum(L_new[j][r] * sum(Ez[i][r] * Xc[i][j]
                           for i in range(n)) / n for r in range(k)), 1e-3) for j in range(p)]
            lim = max(abs(L_new[j][r] - L[j][r]) for j in range(p) for r in range(k))
            L = L_new
            psi = psi_new
            if lim < self.tol:
                break
        self.components_ = _transpose(L)  # k×p
        self.noise_variance_ = psi
        return self

    def transform(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        p = len(X2d[0])
        Xc = [[X2d[i][j] - self.mean_[j] for j in range(p)] for i in range(len(X2d))]
        L = _transpose(self.components_)  # p×k
        psi = self.noise_variance_
        k = len(self.components_)
        Psi_inv_L, M_inv = self._inv_diag_plus_lr(psi, L)
        XcPsiL = [[sum(Xc[i][j] * Psi_inv_L[j][r] for j in range(p)) for r in range(k)]
                  for i in range(len(Xc))]
        return _mat_mat(XcPsiL, M_inv)

    def fit_transform(self, X) -> List[List[float]]:
        return self.fit(X).transform(X)



class KernelPCA:
    """Kernel PCA with RBF, polynomial, cosine or linear kernel."""

    def __init__(self, n_components: int = 2, kernel: str = 'rbf', gamma: float = 1.0,
                 degree: int = 3, coef0: float = 1.0):
        self.n_components = n_components; self.kernel = kernel
        self.gamma = gamma; self.degree = degree; self.coef0 = coef0
        self.alphas_: Optional[List[List[float]]] = None
        self.lambdas_: Optional[List[float]] = None
        self.X_fit_: Optional[List[List[float]]] = None
        self.K_fit_rows_mean_: Optional[List[float]] = None
        self.K_fit_mean_: float = 0.0

    def _kernel(self, a: List[float], b: List[float]) -> float:
        if self.kernel == 'rbf':
            sq = sum((x-y)**2 for x,y in zip(a,b))
            return math.exp(-self.gamma * sq)
        elif self.kernel == 'linear':
            return sum(x*y for x,y in zip(a,b))
        elif self.kernel == 'poly':
            return (self.gamma*sum(x*y for x,y in zip(a,b)) + self.coef0)**self.degree
        elif self.kernel == 'cosine':
            na = sum(x**2 for x in a)**0.5; nb = sum(x**2 for x in b)**0.5
            return sum(x*y for x,y in zip(a,b)) / max(na*nb, 1e-10)
        return sum(x*y for x,y in zip(a,b))

    def _build_K(self, X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
        return [[self._kernel(X[i], Y[j]) for j in range(len(Y))] for i in range(len(X))]

    def _center_K(self, K: List[List[float]], row_means=None, total_mean=None):
        n = len(K); m = len(K[0])
        if row_means is None:
            row_means = [sum(K[i]) / m for i in range(n)]
        if total_mean is None:
            total_mean = sum(sum(r) for r in K) / (n * m)
        col_means = [sum(K[i][j] for i in range(n)) / n for j in range(m)]
        Kc = [[K[i][j] - row_means[i] - col_means[j] + total_mean for j in range(m)] for i in range(n)]
        return Kc, row_means, total_mean

    def _power_eig(self, K: List[List[float]], k: int) -> Tuple[List[float], List[List[float]]]:
        n = len(K); lambdas = []; vecs = []
        A = [row[:] for row in K]
        for _ in range(k):
            v = [1.0/n**0.5]*n
            for __ in range(100):
                Av = [sum(A[i][j]*v[j] for j in range(n)) for i in range(n)]
                lam = sum(Av[i]*v[i] for i in range(n))
                nrm = sum(x**2 for x in Av)**0.5
                if nrm < 1e-10: break
                v_new = [x/nrm for x in Av]
                if sum((v_new[i]-v[i])**2 for i in range(n)) < 1e-12: v=v_new; break
                v = v_new
            lam = sum(sum(A[i][j]*v[j] for j in range(n))*v[i] for i in range(n))
            lambdas.append(lam)
            vecs.append(v[:])
            A = [[A[i][j]-lam*v[i]*v[j] for j in range(n)] for i in range(n)]
        return lambdas, vecs

    def fit(self, X) -> 'KernelPCA':
        self.X_fit_ = _to_list2d(X)
        K = self._build_K(self.X_fit_, self.X_fit_)
        Kc, self.K_fit_rows_mean_, self.K_fit_mean_ = self._center_K(K)
        n = len(Kc)
        lambdas, vecs = self._power_eig(Kc, self.n_components)
        self.lambdas_ = lambdas
        self.alphas_ = [[vecs[c][i] / max(abs(lambdas[c]), 1e-10)**0.5 for c in range(len(vecs))]
                        for i in range(n)]
        return self

    def transform(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        K_new = self._build_K(X2d, self.X_fit_)
        n_new, n_fit = len(X2d), len(self.X_fit_)
        col_means = self.K_fit_rows_mean_
        row_means_new = [sum(K_new[i]) / n_fit for i in range(n_new)]
        Kc = [[K_new[i][j] - row_means_new[i] - col_means[j] + self.K_fit_mean_
               for j in range(n_fit)] for i in range(n_new)]
        return _mat_mat(Kc, self.alphas_)

    def fit_transform(self, X) -> List[List[float]]:
        return self.fit(X).transform(X)


class DictionaryLearning:
    """Dictionary Learning via alternating sparse coding + dict update (K-SVD-lite)."""

    def __init__(self, n_components: int = 10, max_iter: int = 100, tol: float = 1e-4,
                 fit_algorithm: str = 'lars', transform_algorithm: str = 'omp',
                 alpha: float = 1.0, random_state=None):
        self.n_components = n_components; self.max_iter = max_iter; self.tol = tol
        self.alpha = alpha
        self.components_: Optional[List[List[float]]] = None

    def _soft_thresh(self, v: List[float], lam: float) -> List[float]:
        return [max(0, x-lam)-max(0,-x-lam) for x in v]

    def _sparse_code(self, X: List[List[float]], D: List[List[float]]) -> List[List[float]]:
        """Lasso-like via ISTA per sample."""
        k, p = len(D), len(D[0]); lam = self.alpha * 0.1; lr = 0.1
        codes = []
        for x in X:
            z = [0.0]*k
            for _ in range(50):
                r = [x[j] - sum(D[c][j]*z[c] for c in range(k)) for j in range(p)]
                grad = [-sum(r[j]*D[c][j] for j in range(p)) for c in range(k)]
                z = self._soft_thresh([z[c] - lr*grad[c] for c in range(k)], lr*lam)
            codes.append(z)
        return codes

    def fit(self, X) -> 'DictionaryLearning':
        import random as _r
        X2d = _to_list2d(X); n, p = len(X2d), len(X2d[0]); k = self.n_components
        rng = _r.Random()
        D = [[rng.gauss(0, 1.0/p**0.5) for _ in range(p)] for _ in range(k)]
        D = [_normalize(row) for row in D]
        for _ in range(self.max_iter):
            codes = self._sparse_code(X2d, D)
            # Update dict via gradient descent
            for c in range(k):
                grad = [0.0]*p
                for i in range(n):
                    r = [X2d[i][j] - sum(D[cc][j]*codes[i][cc] for cc in range(k)) for j in range(p)]
                    for j in range(p): grad[j] -= codes[i][c]*r[j]
                D[c] = _normalize([D[c][j] - 0.01*grad[j]/n for j in range(p)])
        self.components_ = D
        return self

    def transform(self, X) -> List[List[float]]:
        return self._sparse_code(_to_list2d(X), self.components_)

    def fit_transform(self, X) -> List[List[float]]:
        return self.fit(X).transform(X)


__all__ = [
    'PCA', 'TruncatedSVD', 'NMF', 'FastICA', 'FactorAnalysis',
    'KernelPCA', 'DictionaryLearning',
]
