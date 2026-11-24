"""vectorphile.manifold — Manifold learning / nonlinear dimensionality reduction.

All computations use pure Python. No external dependencies.
"""
from __future__ import annotations
import math
import random
from typing import Optional, List


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


def _euclid_sq(a: List[float], b: List[float]) -> float:
    return sum((a[i] - b[i]) ** 2 for i in range(len(a)))


def _pairwise_sq(X: List[List[float]]) -> List[List[float]]:
    n = len(X)
    return [[_euclid_sq(X[i], X[j]) for j in range(n)] for i in range(n)]


def _mds_from_dists(D_sq: List[List[float]], k: int) -> List[List[float]]:
    """Classical MDS from squared distance matrix."""
    n = len(D_sq)
    # Double centering: B = -0.5 H D² H
    row_mean = [sum(D_sq[i][j] for j in range(n)) / n for i in range(n)]
    total_mean = sum(row_mean) / n
    B = [[-0.5 * (D_sq[i][j] - row_mean[i] - row_mean[j] + total_mean)
           for j in range(n)] for i in range(n)]
    # Power iteration for top-k eigenvectors
    eigvecs, eigvals = [], []
    curr_B = [row[:] for row in B]
    for _ in range(k):
        v = [random.gauss(0, 1) for _ in range(n)]
        norm = math.sqrt(sum(x * x for x in v))
        v = [x / norm for x in v]
        # Deflation iterations
        for _it in range(100):
            Bv = [sum(curr_B[i][j] * v[j] for j in range(n)) for i in range(n)]
            lam = sum(v[j] * Bv[j] for j in range(n))
            norm_new = math.sqrt(sum(x * x for x in Bv))
            if norm_new < 1e-10:
                break
            v_new = [x / norm_new for x in Bv]
            if max(abs(v_new[i] - v[i]) for i in range(n)) < 1e-8:
                v = v_new
                break
            v = v_new
        Bv = [sum(curr_B[i][j] * v[j] for j in range(n)) for i in range(n)]
        lam = sum(v[j] * Bv[j] for j in range(n))
        eigvals.append(lam)
        eigvecs.append(v)
        # Deflate
        curr_B = [[curr_B[i][j] - lam * v[i] * v[j] for j in range(n)] for i in range(n)]
    # Y[i][r] = eigvec[r][i] * sqrt(max(eigval[r], 0))
    result = [[eigvecs[r][i] * math.sqrt(max(eigvals[r], 0.0)) for r in range(len(eigvals))]
              for i in range(n)]
    return result


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------

class TSNE:
    """t-distributed Stochastic Neighbor Embedding."""

    def __init__(self, n_components: int = 2, perplexity: float = 30.0,
                 learning_rate: float = 200.0, n_iter: int = 1000,
                 early_exaggeration: float = 12.0,
                 random_state: Optional[int] = None, verbose: int = 0):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.random_state = random_state
        self.verbose = verbose
        self.embedding_: Optional[List[List[float]]] = None
        self.kl_divergence_: float = float('inf')

    def _binary_search_beta(self, Di: List[float], perp: float) -> float:
        beta_min, beta_max = -float('inf'), float('inf')
        beta = 1.0
        target_H = math.log(perp)
        for _ in range(50):
            pi = [math.exp(-Di[j] * beta) for j in range(len(Di))]
            sum_pi = sum(pi) + 1e-300
            pi = [p / sum_pi for p in pi]
            H = -sum(p * math.log(max(p, 1e-300)) for p in pi)
            Hdiff = H - target_H
            if abs(Hdiff) < 1e-5:
                break
            if Hdiff > 0:
                beta_min = beta
                beta = beta * 2 if beta_max == float('inf') else (beta + beta_max) / 2
            else:
                beta_max = beta
                beta = beta / 2 if beta_min == -float('inf') else (beta + beta_min) / 2
        return beta

    def _compute_P(self, X: List[List[float]]) -> List[List[float]]:
        n = len(X)
        D_sq = _pairwise_sq(X)
        P = [[0.0] * n for _ in range(n)]
        for i in range(n):
            Di = [D_sq[i][j] if j != i else float('inf') for j in range(n)]
            beta = self._binary_search_beta(Di, self.perplexity)
            pi = [math.exp(-Di[j] * beta) for j in range(n)]
            pi[i] = 0.0
            s = sum(pi) + 1e-300
            P[i] = [p / s for p in pi]
        # Symmetrize
        result = [[(P[i][j] + P[j][i]) / (2 * n) for j in range(n)] for i in range(n)]
        return [[max(v, 1e-12) for v in row] for row in result]

    def fit_transform(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        n, k = len(X2d), self.n_components
        rng = random.Random(self.random_state)
        Y = [[rng.gauss(0, 1e-4) for _ in range(k)] for _ in range(n)]
        P = self._compute_P(X2d)
        iY = [[0.0] * k for _ in range(n)]
        gains = [[1.0] * k for _ in range(n)]
        lr = self.learning_rate
        for it in range(self.n_iter):
            # Compute Q
            Dsq = _pairwise_sq(Y)
            Q = [[1.0 / (1.0 + Dsq[i][j]) for j in range(n)] for i in range(n)]
            for i in range(n):
                Q[i][i] = 0.0
            q_sum = sum(Q[i][j] for i in range(n) for j in range(n)) + 1e-300
            Q = [[max(Q[i][j] / q_sum, 1e-12) for j in range(n)] for i in range(n)]
            exag = self.early_exaggeration if it < 250 else 1.0
            # Gradient
            dY = [[0.0] * k for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    pq = exag * P[i][j] - Q[i][j]
                    q_ij = 1.0 / (1.0 + Dsq[i][j])
                    for d in range(k):
                        dY[i][d] += 4 * pq * (Y[i][d] - Y[j][d]) * q_ij
            # Update
            momentum = 0.5 if it < 250 else 0.8
            for i in range(n):
                for d in range(k):
                    gains[i][d] = (gains[i][d] + 0.2) if abs(dY[i][d]) > 0 else (gains[i][d] * 0.8)
                    gains[i][d] = max(gains[i][d], 0.01)
                    iY[i][d] = momentum * iY[i][d] - lr * gains[i][d] * dY[i][d]
                    Y[i][d] += iY[i][d]
            # Center
            for d in range(k):
                mean_d = sum(Y[i][d] for i in range(n)) / n
                for i in range(n):
                    Y[i][d] -= mean_d
            if it % 100 == 0 and self.verbose:
                kl = sum(P[i][j] * math.log(max(P[i][j] / Q[i][j], 1e-12))
                         for i in range(n) for j in range(n))
                self.kl_divergence_ = kl
                print(f't-SNE iter {it}, KL={kl:.4f}')
        self.embedding_ = Y
        return Y

    def fit(self, X) -> 'TSNE':
        self.fit_transform(X)
        return self


# ---------------------------------------------------------------------------
# MDS
# ---------------------------------------------------------------------------

class MDS:
    """Classical Multidimensional Scaling."""

    def __init__(self, n_components: int = 2, metric: bool = True,
                 random_state: Optional[int] = None, max_iter: int = 300,
                 n_init: int = 4):
        self.n_components = n_components
        self.metric = metric
        self.random_state = random_state
        self.embedding_: Optional[List[List[float]]] = None
        self.stress_: float = float('inf')

    def fit_transform(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        D_sq = _pairwise_sq(X2d)
        Y = _mds_from_dists(D_sq, self.n_components)
        self.embedding_ = Y
        return Y

    def fit(self, X) -> 'MDS':
        self.fit_transform(X)
        return self


# ---------------------------------------------------------------------------
# Isomap
# ---------------------------------------------------------------------------

class Isomap:
    """Isometric Mapping."""

    def __init__(self, n_components: int = 2, n_neighbors: int = 5):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.embedding_: Optional[List[List[float]]] = None

    def fit_transform(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        n = len(X2d)
        D = [[math.sqrt(_euclid_sq(X2d[i], X2d[j])) for j in range(n)] for i in range(n)]
        INF = float('inf')
        G = [[INF] * n for _ in range(n)]
        for i in range(n):
            G[i][i] = 0.0
        for i in range(n):
            neighbor_idx = sorted(range(n), key=lambda j: D[i][j])[1:self.n_neighbors + 1]
            for j in neighbor_idx:
                G[i][j] = D[i][j]
                G[j][i] = D[i][j]
        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                if G[i][k] == INF:
                    continue
                for j in range(n):
                    new_d = G[i][k] + G[k][j]
                    if new_d < G[i][j]:
                        G[i][j] = new_d
        max_finite = max(G[i][j] for i in range(n) for j in range(n) if G[i][j] != INF)
        D_geo_sq = [[(G[i][j] if G[i][j] != INF else max_finite * 10) ** 2 for j in range(n)]
                    for i in range(n)]
        Y = _mds_from_dists(D_geo_sq, self.n_components)
        self.embedding_ = Y
        return Y

    def fit(self, X) -> 'Isomap':
        self.fit_transform(X)
        return self


# ---------------------------------------------------------------------------
# LocallyLinearEmbedding
# ---------------------------------------------------------------------------

class LocallyLinearEmbedding:
    """Locally Linear Embedding (LLE)."""

    def __init__(self, n_components: int = 2, n_neighbors: int = 5,
                 reg: float = 1e-3):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.reg = reg
        self.embedding_: Optional[List[List[float]]] = None

    def fit_transform(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        n, d = len(X2d), len(X2d[0])
        k = self.n_neighbors
        D_sq = _pairwise_sq(X2d)
        nbr_idx = [sorted(range(n), key=lambda j: D_sq[i][j])[1:k + 1] for i in range(n)]
        # Compute weights W
        W = [[0.0] * n for _ in range(n)]
        for i in range(n):
            Z = [[X2d[nbr_idx[i][a]][f] - X2d[i][f] for f in range(d)] for a in range(k)]
            # C = Z @ Z^T (k×k)
            C = [[sum(Z[a][f] * Z[b][f] for f in range(d)) for b in range(k)] for a in range(k)]
            # Regularize
            tr = sum(C[a][a] for a in range(k))
            reg_val = self.reg * tr
            for a in range(k):
                C[a][a] += reg_val
            # Solve C w = 1
            ones = [1.0] * k
            # Gaussian elimination
            M = [C[a][:] + [ones[a]] for a in range(k)]
            for col in range(k):
                pivot = max(range(col, k), key=lambda r: abs(M[r][col]))
                M[col], M[pivot] = M[pivot], M[col]
                if abs(M[col][col]) < 1e-12:
                    continue
                for r in range(k):
                    if r != col:
                        factor = M[r][col] / M[col][col]
                        M[r] = [M[r][c] - factor * M[col][c] for c in range(k + 1)]
            w_raw = [M[a][k] / (M[a][a] + 1e-15) for a in range(k)]
            w_sum = sum(w_raw) + 1e-10
            for a, j in enumerate(nbr_idx[i]):
                W[i][j] = w_raw[a] / w_sum
        # M = (I-W)^T (I-W)
        IW = [[(-W[i][j] if i != j else 1.0 - W[i][j]) for j in range(n)] for i in range(n)]
        M = [[sum(IW[k_][i] * IW[k_][j] for k_ in range(n)) for j in range(n)] for i in range(n)]
        # Find bottom n_components+1 eigenvectors of M (skip smallest = 0)
        bottom = []
        curr_M = [row[:] for row in M]
        for _ in range(self.n_components + 1):
            v = [random.gauss(0, 1) for _ in range(n)]
            norm = math.sqrt(sum(x * x for x in v))
            v = [x / norm for x in v]
            # Inverse power iteration (shift with large value to get smallest eigvec)
            for _it in range(100):
                Mv = [sum(curr_M[i][j] * v[j] for j in range(n)) for i in range(n)]
                norm = math.sqrt(sum(x * x for x in Mv))
                if norm < 1e-10:
                    break
                v_new = [x / norm for x in Mv]
                if max(abs(abs(v_new[i]) - abs(v[i])) for i in range(n)) < 1e-8:
                    v = v_new
                    break
                v = v_new
            bottom.append(v)
            lam = sum(v[j] * sum(curr_M[i][j] * v[i] for i in range(n)) for j in range(n))
            curr_M = [[curr_M[i][j] - lam * v[i] * v[j] for j in range(n)] for i in range(n)]
        # Return bottom[1:] (skip trivial eigenvector)
        result = [[bottom[r + 1][i] for r in range(self.n_components)] for i in range(n)]
        self.embedding_ = result
        return result

    def fit(self, X) -> 'LocallyLinearEmbedding':
        self.fit_transform(X)
        return self


# ---------------------------------------------------------------------------
# SpectralEmbedding
# ---------------------------------------------------------------------------

class SpectralEmbedding:
    """Spectral Embedding via Laplacian Eigenmaps."""

    def __init__(self, n_components: int = 2, n_neighbors: int = 10,
                 gamma: Optional[float] = None, random_state: Optional[int] = None):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self.random_state = random_state
        self.embedding_: Optional[List[List[float]]] = None

    def fit_transform(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        n = len(X2d)
        D_sq = _pairwise_sq(X2d)
        mean_sq = sum(D_sq[i][j] for i in range(n) for j in range(n)) / (n * n) + 1e-10
        gamma = self.gamma or 1.0 / mean_sq
        knn_idx = [sorted(range(n), key=lambda j: D_sq[i][j])[1:self.n_neighbors + 1]
                   for i in range(n)]
        W = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in knn_idx[i]:
                val = math.exp(-gamma * D_sq[i][j])
                W[i][j] = val
                W[j][i] = val
        deg = [sum(W[i]) for i in range(n)]
        D_inv_sqrt = [1.0 / math.sqrt(max(d, 1e-10)) for d in deg]
        # L_sym = I - D^{-1/2} W D^{-1/2}
        # Eigendecomposition via power iteration on L_sym
        # Use bottom eigenvectors (skip 0)
        L = [[(-W[i][j] * D_inv_sqrt[i] * D_inv_sqrt[j]) + (1.0 if i == j else 0.0)
               for j in range(n)] for i in range(n)]
        # Find bottom n_components+1 eigenvectors of L (by largest of -L)
        rng = random.Random(self.random_state)
        bottom = []
        curr_L = [row[:] for row in L]
        for _ in range(self.n_components + 1):
            v = [rng.gauss(0, 1) for _ in range(n)]
            norm = math.sqrt(sum(x * x for x in v))
            v = [x / norm for x in v]
            for _it in range(200):
                # Power iteration on (2I - L) for bottom eigenvalue
                Lv = [sum(curr_L[i][j] * v[j] for j in range(n)) for i in range(n)]
                # Inverse shift: use shifted matrix
                Av = [2.0 * v[i] - Lv[i] for i in range(n)]
                norm = math.sqrt(sum(x * x for x in Av))
                if norm < 1e-10:
                    break
                v_new = [x / norm for x in Av]
                if max(abs(abs(v_new[i]) - abs(v[i])) for i in range(n)) < 1e-8:
                    v = v_new
                    break
                v = v_new
            bottom.append(v)
            lam = sum(v[j] * sum(curr_L[i][j] * v[i] for i in range(n)) for j in range(n))
            curr_L = [[curr_L[i][j] - lam * v[i] * v[j] for j in range(n)] for i in range(n)]
        result = [[bottom[r + 1][i] for r in range(self.n_components)] for i in range(n)]
        self.embedding_ = result
        return result

    def fit(self, X) -> 'SpectralEmbedding':
        self.fit_transform(X)
        return self


class UMAP:
    """Uniform Manifold Approximation and Projection (McInnes et al., 2018).

    Simplified implementation using fuzzy simplicial set + gradient descent.
    For production use, install the `umap-learn` package.
    """

    def __init__(self, n_components: int = 2, n_neighbors: int = 15,
                 min_dist: float = 0.1, n_epochs: int = 200,
                 learning_rate: float = 1.0, random_state: Optional[int] = None):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.embedding_: Optional[List[List[float]]] = None

    def fit_transform(self, X) -> List[List[float]]:
        import random as _rnd
        rng = _rnd.Random(self.random_state or 42)
        X2d = _to_list2d(X)
        n, d = len(X2d), len(X2d[0])
        D_sq = _pairwise_sq(X2d)

        # Compute fuzzy membership strengths via kNN distances
        knn_dists = []
        for i in range(n):
            dists = sorted((D_sq[i][j], j) for j in range(n) if j != i)
            knn_dists.append([dists[k] for k in range(min(self.n_neighbors, len(dists)))])

        # Build symmetrized fuzzy graph
        W = [[0.0] * n for _ in range(n)]
        for i in range(n):
            rho_i = math.sqrt(knn_dists[i][0][0]) if knn_dists[i] else 0.0
            sigma_i = math.sqrt(knn_dists[i][-1][0]) if knn_dists[i] else 1.0
            sigma_i = max(sigma_i - rho_i, 1e-6)
            for dist_sq, j in knn_dists[i]:
                val = math.exp(-(math.sqrt(dist_sq) - rho_i) / sigma_i)
                W[i][j] = max(W[i][j], val)
                W[j][i] = max(W[j][i], val)

        # Initialize low-dim embedding (spectral or random)
        Y = [[rng.gauss(0, 1e-4) for _ in range(self.n_components)] for _ in range(n)]

        # Stochastic gradient descent
        a = 1.0 / (self.min_dist ** 0.5) if self.min_dist > 0 else 10.0
        b = 1.0

        def _d2_low(yi, yj):
            return sum((yi[k] - yj[k]) ** 2 for k in range(self.n_components))

        def _phi(d2):
            return 1.0 / (1 + a * d2 ** b)

        lr = self.learning_rate
        for epoch in range(self.n_epochs):
            lr_ep = lr * (1.0 - epoch / self.n_epochs)
            for i in range(n):
                for dist_sq, j in knn_dists[i]:
                    w_ij = W[i][j]
                    if w_ij < 1e-5:
                        continue
                    d2 = _d2_low(Y[i], Y[j])
                    phi = _phi(d2)
                    grad_coeff = -2 * a * b * d2 ** (b - 1) * w_ij * phi ** 2
                    for k in range(self.n_components):
                        g = max(-4.0, min(4.0, grad_coeff * (Y[i][k] - Y[j][k])))
                        Y[i][k] -= lr_ep * g
                        Y[j][k] += lr_ep * g
                    # Negative sampling
                    neg_j = rng.randint(0, n - 1)
                    if neg_j != i:
                        d2_neg = _d2_low(Y[i], Y[neg_j])
                        phi_neg = _phi(d2_neg)
                        grad_neg = 2 * b * (1 - w_ij) / max(d2_neg + 0.001, 1e-5) * phi_neg
                        for k in range(self.n_components):
                            g = max(-4.0, min(4.0, grad_neg * (Y[i][k] - Y[neg_j][k])))
                            Y[i][k] -= lr_ep * g

        self.embedding_ = Y
        return Y

    def fit(self, X) -> 'UMAP':
        self.fit_transform(X)
        return self

    def transform(self, X) -> List[List[float]]:
        return self.fit_transform(X)


__all__ = [
    'TSNE', 'MDS', 'Isomap',
    'LocallyLinearEmbedding', 'SpectralEmbedding', 'UMAP',
]
