"""Manifold learning — pure Python."""
import math
import random as _random
from typing import Optional

from vectorphile.ml.decomposition import _power_iter_top_k, _dot, _transpose, _mat_mul, _mean_center


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _pairwise_dists(X):
    n = len(X)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _dist(X[i], X[j])
            D[i][j] = d
            D[j][i] = d
    return D


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


def _solve(A, b):
    n = len(b)
    M = [A[i][:] + [float(b[i])] for i in range(n)]
    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[max_row] = M[max_row], M[col]
        pivot = M[col][col]
        if abs(pivot) < 1e-15:
            continue
        for row in range(col + 1, n):
            factor = M[row][col] / pivot
            M[row] = [M[row][j] - factor * M[col][j] for j in range(n + 1)]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i] if abs(M[i][i]) > 1e-15 else 1e-15
    return x


class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0,
                 n_iter=1000, random_state=None, metric='euclidean',
                 early_exaggeration=12.0):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.early_exaggeration = early_exaggeration

    def _compute_p(self, D, perplexity):
        n = len(D)
        P = [[0.0] * n for _ in range(n)]
        target_H = math.log(perplexity)
        for i in range(n):
            # Binary search for sigma
            lo, hi = 1e-10, 1e5
            for _ in range(50):
                sigma2 = (lo + hi) / 2
                dists = [D[i][j] for j in range(n) if j != i]
                exps = [math.exp(-D[i][j] ** 2 / (2 * sigma2)) for j in range(n) if j != i]
                total = sum(exps) + 1e-15
                probs = [e / total for e in exps]
                H = -sum(p * math.log(p + 1e-15) for p in probs if p > 0)
                if H < target_H:
                    lo = sigma2
                else:
                    hi = sigma2
            sigma2 = (lo + hi) / 2
            exps = [math.exp(-D[i][j] ** 2 / (2 * sigma2)) for j in range(n)]
            exps[i] = 0.0
            total = sum(exps) + 1e-15
            for j in range(n):
                P[i][j] = exps[j] / total
        # Symmetrize
        Psym = [[(P[i][j] + P[j][i]) / (2 * n) for j in range(n)] for i in range(n)]
        return Psym

    def fit_transform(self, X):
        X2 = [[float(v) for v in row] for row in X]
        n = len(X2)
        rng = _random.Random(self.random_state)
        # Sample D for efficiency (use 2D embedding init)
        D = _pairwise_dists(X2)
        P = self._compute_p(D, self.perplexity)
        # Random init for Y
        Y = [[rng.gauss(0, 1e-4) for _ in range(self.n_components)] for _ in range(n)]
        lr = self.learning_rate
        momentum = 0.5
        Y_prev = [row[:] for row in Y]
        for iteration in range(self.n_iter):
            # Compute Q (t-distribution)
            d_sq = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    s = sum((Y[i][k] - Y[j][k]) ** 2 for k in range(self.n_components))
                    d_sq[i][j] = d_sq[j][i] = 1.0 / (1.0 + s)
            # Row-zero diagonal
            for i in range(n):
                d_sq[i][i] = 0.0
            total_q = sum(d_sq[i][j] for i in range(n) for j in range(n)) + 1e-15
            # Gradients
            exagg = self.early_exaggeration if iteration < 250 else 1.0
            dY = [[0.0] * self.n_components for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    q = d_sq[i][j] / total_q
                    pq_diff = (exagg * P[i][j] - q) * d_sq[i][j]
                    for k in range(self.n_components):
                        dY[i][k] += 4 * pq_diff * (Y[i][k] - Y[j][k])
            if iteration == 250:
                momentum = 0.8
            Y_new = [[Y[i][k] - lr * dY[i][k] + momentum * (Y[i][k] - Y_prev[i][k])
                       for k in range(self.n_components)] for i in range(n)]
            Y_prev = Y
            Y = Y_new
        self.embedding_ = Y
        return Y

    def fit(self, X):
        self.fit_transform(X)
        return self


class MDS:
    def __init__(self, n_components=2, metric=True, max_iter=300, eps=1e-3, random_state=None):
        self.n_components = n_components
        self.metric = metric
        self.max_iter = max_iter
        self.random_state = random_state

    def fit_transform(self, X):
        X2 = [[float(v) for v in row] for row in X]
        n = len(X2)
        # Classical MDS
        D = _pairwise_dists(X2)
        D2 = [[D[i][j] ** 2 for j in range(n)] for i in range(n)]
        # Double centering
        row_means = [_mean(D2[i]) for i in range(n)]
        col_means = [_mean([D2[i][j] for i in range(n)]) for j in range(n)]
        grand_mean = _mean(row_means)
        B = [[-0.5 * (D2[i][j] - row_means[i] - col_means[j] + grand_mean)
               for j in range(n)] for i in range(n)]
        eigenvals, eigenvecs = _power_iter_top_k(B, self.n_components, seed=self.random_state or 0)
        result = []
        for i in range(n):
            row = [eigenvecs[k][i] * math.sqrt(max(eigenvals[k], 0))
                   for k in range(self.n_components)]
            result.append(row)
        self.embedding_ = result
        return result

    def fit(self, X):
        self.fit_transform(X)
        return self


class Isomap:
    def __init__(self, n_neighbors=5, n_components=2, path_method='auto'):
        self.n_neighbors = n_neighbors
        self.n_components = n_components

    def fit_transform(self, X):
        X2 = [[float(v) for v in row] for row in X]
        n = len(X2)
        # Build k-NN graph
        INF = float('inf')
        G = [[INF] * n for _ in range(n)]
        for i in range(n):
            G[i][i] = 0.0
            dists = sorted([(j, _dist(X2[i], X2[j])) for j in range(n) if j != i],
                           key=lambda t: t[1])
            for j, d in dists[:self.n_neighbors]:
                G[i][j] = d
                G[j][i] = d
        # Floyd-Warshall
        for k in range(n):
            for i in range(n):
                if G[i][k] == INF:
                    continue
                for j in range(n):
                    if G[i][k] + G[k][j] < G[i][j]:
                        G[i][j] = G[i][k] + G[k][j]
        mds = MDS(n_components=self.n_components)
        # Use G as distance matrix for MDS
        # Temporarily override
        n2 = n
        D2 = [[G[i][j] ** 2 for j in range(n2)] for i in range(n2)]
        row_means = [_mean(D2[i]) for i in range(n2)]
        col_means = [_mean([D2[i][j] for i in range(n2)]) for j in range(n2)]
        grand_mean = _mean(row_means)
        B = [[-0.5 * (D2[i][j] - row_means[i] - col_means[j] + grand_mean)
               for j in range(n2)] for i in range(n2)]
        eigenvals, eigenvecs = _power_iter_top_k(B, self.n_components)
        result = []
        for i in range(n2):
            row = [eigenvecs[k][i] * math.sqrt(max(eigenvals[k], 0))
                   for k in range(self.n_components)]
            result.append(row)
        self.embedding_ = result
        return result

    def fit(self, X):
        self.fit_transform(X)
        return self


class LocallyLinearEmbedding:
    def __init__(self, n_neighbors=5, n_components=2, max_iter=100, random_state=None):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state

    def fit_transform(self, X):
        X2 = [[float(v) for v in row] for row in X]
        n, p = len(X2), len(X2[0])
        K = self.n_neighbors
        # Find KNN for each point
        W = [[0.0] * n for _ in range(n)]
        for i in range(n):
            dists = sorted([(j, _dist(X2[i], X2[j])) for j in range(n) if j != i],
                           key=lambda t: t[1])
            nbr_idxs = [j for j, _ in dists[:K]]
            # Compute reconstruction weights: minimize |xi - sum(w*xj)|^2 s.t. sum(w)=1
            Z = [[X2[i][d] - X2[j][d] for d in range(p)] for j in nbr_idxs]
            # Gram matrix
            G = _mat_mul(Z, _transpose(Z))
            # Regularize
            reg = 1e-3 * sum(G[k][k] for k in range(K)) / K
            for k in range(K):
                G[k][k] += reg
            ones = [1.0] * K
            w = _solve(G, ones)
            total = sum(w)
            w = [v / (total + 1e-15) for v in w]
            for idx, j in enumerate(nbr_idxs):
                W[i][j] = w[idx]
        # Compute M = (I-W)^T (I-W)
        IW = [[(-W[i][j] if i != j else 1.0 - W[i][j]) for j in range(n)] for i in range(n)]
        M = _mat_mul(_transpose(IW), IW)
        # Find bottom n_components+1 eigenvectors of M
        eigenvals, eigenvecs = _power_iter_top_k(
            [[-M[i][j] for j in range(n)] for i in range(n)],
            self.n_components + 1, seed=self.random_state or 0)
        # Sort by ascending eigenvalue of M (descending of -M)
        order = sorted(range(len(eigenvals)), key=lambda k: -eigenvals[k])
        # Skip the smallest (last eigenvec of M = first of -M sort by desc)
        # Actually for LLE we want the eigenvecs of M with smallest nonzero eigenvalues
        # i.e. largest of -M
        result = []
        for i in range(n):
            row = [eigenvecs[order[k]][i] for k in range(1, self.n_components + 1)]
            result.append(row)
        self.embedding_ = result
        return result

    def fit(self, X):
        self.fit_transform(X)
        return self


class SpectralEmbedding:
    def __init__(self, n_components=2, affinity='nearest_neighbors', n_neighbors=None,
                 gamma=None, random_state=None):
        self.n_components = n_components
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.gamma = gamma
        self.random_state = random_state

    def fit_transform(self, X):
        X2 = [[float(v) for v in row] for row in X]
        n, p = len(X2), len(X2[0])
        K = self.n_neighbors or max(2, int(math.log(n)) + 1)
        gamma = self.gamma or 1.0
        # Affinity matrix
        W = [[0.0] * n for _ in range(n)]
        for i in range(n):
            dists = sorted([(j, _dist(X2[i], X2[j])) for j in range(n) if j != i],
                           key=lambda t: t[1])
            for j, d in dists[:K]:
                aff = math.exp(-gamma * d ** 2)
                W[i][j] = aff
                W[j][i] = aff
        # Degree matrix (diagonal)
        D_diag = [sum(W[i]) for i in range(n)]
        # Normalized Laplacian: L = D^{-1/2} W D^{-1/2}
        d_inv_sqrt = [1.0 / math.sqrt(max(d, 1e-15)) for d in D_diag]
        L = [[d_inv_sqrt[i] * W[i][j] * d_inv_sqrt[j] for j in range(n)] for i in range(n)]
        eigenvals, eigenvecs = _power_iter_top_k(L, self.n_components, seed=self.random_state or 0)
        result = []
        for i in range(n):
            row = [eigenvecs[k][i] for k in range(self.n_components)]
            result.append(row)
        self.embedding_ = result
        return result

    def fit(self, X):
        self.fit_transform(X)
        return self


class UMAP:
    """Approximate UMAP via spectral init + force-directed layout."""

    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, n_epochs=200,
                 learning_rate=1.0, random_state=None, metric='euclidean'):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit_transform(self, X):
        X2 = [[float(v) for v in row] for row in X]
        n = len(X2)
        rng = _random.Random(self.random_state)
        K = self.n_neighbors
        # Spectral init
        se = SpectralEmbedding(n_components=self.n_components, n_neighbors=K,
                               random_state=self.random_state)
        Y = se.fit_transform(X2)
        # Force-directed optimization
        a, b = 1.0, 1.0  # simplified UMAP params
        lr = self.learning_rate
        for epoch in range(self.n_epochs):
            # Build k-NN graph
            for i in range(n):
                dists = sorted([(j, _dist(X2[i], X2[j])) for j in range(n) if j != i],
                               key=lambda t: t[1])
                nbrs = [j for j, _ in dists[:K]]
                # Attractive forces
                for j in nbrs:
                    d = max(_dist(Y[i], Y[j]), 1e-5)
                    grad_coeff = -2 * a * b * d ** (2 * b - 2) / (1 + a * d ** (2 * b))
                    for k in range(self.n_components):
                        g = grad_coeff * (Y[i][k] - Y[j][k])
                        Y[i][k] -= lr * g
                        Y[j][k] += lr * g
                # Repulsive force (random negative sample)
                j = rng.randint(0, n - 1)
                d = max(_dist(Y[i], Y[j]), 1e-5)
                grad_coeff = 2 * b / (d ** 2 * (1 + a * d ** (2 * b)))
                for k in range(self.n_components):
                    Y[i][k] += lr * grad_coeff * (Y[i][k] - Y[j][k])
        self.embedding_ = Y
        return Y

    def fit(self, X):
        self.fit_transform(X)
        return self
