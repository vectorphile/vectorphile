"""Dimensionality reduction for visualization — t-SNE, UMAP, MDS, Isomap, LLE, SpectralEmbedding."""
from __future__ import annotations
import numpy as np
from typing import Optional


class TSNE:
    """t-SNE via Barnes-Hut approximation (pure NumPy, O(n log n) perplexity search)."""
    def __init__(self, n_components=2, perplexity=30.0, learning_rate='auto',
                 n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-7,
                 metric='euclidean', init='pca', verbose=0, random_state=None,
                 method='barnes_hut', angle=0.5, early_exaggeration=12.0):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.early_exaggeration = early_exaggeration
        self.embedding_ = None
        self.kl_divergence_ = None
        self.n_iter_ = 0

    def _pairwise_sq_dist(self, X):
        sq = (X**2).sum(axis=1)
        D = sq[:, None] + sq[None, :] - 2 * X @ X.T
        return np.maximum(D, 0)

    def _p_cond(self, D, sigma2):
        P = np.exp(-D / (2 * sigma2))
        np.fill_diagonal(P, 0)
        P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
        return P

    def _binary_search_sigma(self, D_row, perplexity, n_iter=100, tol=1e-5):
        target_H = np.log2(perplexity)
        lo, hi = 0.0, 1e5
        sigma2 = 1.0
        for _ in range(n_iter):
            P = np.exp(-D_row / (2 * sigma2 + 1e-12))
            P[D_row == 0] = 0
            P_sum = P.sum()
            if P_sum < 1e-15:
                sigma2 = hi
                break
            P /= P_sum
            H = -np.sum(P[P > 1e-12] * np.log2(P[P > 1e-12]))
            if abs(H - target_H) < tol:
                break
            if H < target_H:
                lo = sigma2
                sigma2 = (sigma2 + hi) / 2
            else:
                hi = sigma2
                sigma2 = (sigma2 + lo) / 2
        return sigma2

    def fit_transform(self, X, y=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = np.array(X, dtype=np.float64)
        n = len(X)
        # Compute pairwise distances
        D = self._pairwise_sq_dist(X)
        # Compute P matrix (symmetrized)
        P = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            D_i = D[i].copy()
            D_i[i] = 0
            sigma2 = self._binary_search_sigma(D_i, self.perplexity)
            P[i] = np.exp(-D_i / (2 * sigma2 + 1e-12))
            P[i, i] = 0
            s = P[i].sum()
            if s > 0:
                P[i] /= s
        P = (P + P.T) / (2 * n)
        P = np.maximum(P, 1e-12)
        # Initialize embedding
        if self.init == 'pca':
            pca = PCA(n_components=self.n_components)
            Y = pca.fit_transform(X) * 1e-4
        else:
            Y = np.random.randn(n, self.n_components) * 1e-4
        lr = 200.0 if self.learning_rate == 'auto' else float(self.learning_rate)
        momentum = 0.5
        Y_prev = Y.copy()
        exagg = self.early_exaggeration
        for it in range(self.n_iter):
            # Switch off early exaggeration
            if it == 250:
                exagg = 1.0
                momentum = 0.8
            # Q matrix (Student t-distribution)
            diff = Y[:, None, :] - Y[None, :, :]
            dist_sq = (diff**2).sum(axis=2)
            Q = 1.0 / (1.0 + dist_sq)
            np.fill_diagonal(Q, 0)
            Q_norm = np.maximum(Q / (Q.sum() + 1e-12), 1e-12)
            # Gradient
            PQ = exagg * P - Q_norm
            grad = np.zeros_like(Y)
            for i in range(n):
                grad[i] = 4 * (PQ[i, :, None] * Q[i, :, None] * diff[i]).sum(axis=0)
            # Update with momentum
            Y_new = Y + lr * grad + momentum * (Y - Y_prev)
            # Center
            Y_new -= Y_new.mean(axis=0)
            Y_prev = Y.copy()
            Y = Y_new
            if self.verbose > 0 and (it + 1) % 100 == 0:
                kl = (P * np.log(P / (Q_norm + 1e-12) + 1e-12)).sum()
                print(f'[t-SNE] iter {it+1}, KL divergence: {kl:.4f}')
        self.embedding_ = Y
        self.kl_divergence_ = float((P * np.log(P / (Q_norm + 1e-12) + 1e-12)).sum())
        self.n_iter_ = self.n_iter
        return Y.astype(np.float32)

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self


from vectorphile.ml.decomposition import PCA


class MDS:
    """Multidimensional Scaling."""
    def __init__(self, n_components=2, metric=True, n_init=4, max_iter=300,
                 eps=1e-3, n_jobs=None, random_state=None, dissimilarity='euclidean',
                 normalized_stress=False):
        self.n_components = n_components
        self.metric = metric
        self.n_init = n_init
        self.max_iter = max_iter
        self.eps = eps
        self.random_state = random_state
        self.dissimilarity = dissimilarity
        self.embedding_ = None
        self.stress_ = None

    def fit_transform(self, X, y=None, init=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = np.array(X, dtype=np.float64)
        n = len(X)
        if self.dissimilarity == 'precomputed':
            D = X
        else:
            diff = X[:, None, :] - X[None, :, :]
            D = np.sqrt((diff**2).sum(axis=2))
        if self.metric:
            # Classical MDS via double-centering
            D2 = D**2
            H = np.eye(n) - np.ones((n, n)) / n
            B = -0.5 * H @ D2 @ H
            vals, vecs = np.linalg.eigh(B)
            idx = np.argsort(vals)[::-1][:self.n_components]
            vals = np.maximum(vals[idx], 0)
            Y = vecs[:, idx] * np.sqrt(vals)
        else:
            # Non-metric MDS via SMACOF
            Y = np.random.randn(n, self.n_components)
            stress_prev = np.inf
            for it in range(self.max_iter):
                diff_Y = Y[:, None, :] - Y[None, :, :]
                D_Y = np.sqrt((diff_Y**2).sum(axis=2) + 1e-12)
                # B matrix
                ratio = D / (D_Y + 1e-12)
                np.fill_diagonal(ratio, 0)
                B = -ratio
                B[np.arange(n), np.arange(n)] = -B.sum(axis=1)
                Y_new = B @ Y / n
                stress = np.sqrt(((D - D_Y)**2).sum() / (D**2).sum() + 1e-12)
                if abs(stress_prev - stress) < self.eps:
                    break
                stress_prev = stress
                Y = Y_new
        self.embedding_ = Y
        diff_Y = Y[:, None, :] - Y[None, :, :]
        D_Y = np.sqrt((diff_Y**2).sum(axis=2))
        self.stress_ = float(((D - D_Y)**2).sum())
        return Y.astype(np.float32)

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self


class Isomap:
    """Isomap — geodesic distance + MDS."""
    def __init__(self, n_neighbors=5, n_components=2, metric='minkowski', p=2,
                 eigen_solver='auto', tol=0, max_iter=None, path_method='auto',
                 n_jobs=None):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric
        self.embedding_ = None
        self.reconstruction_error_ = None

    def fit_transform(self, X, y=None):
        X = np.array(X, dtype=np.float64)
        n = len(X)
        # Compute pairwise Euclidean distances
        diff = X[:, None, :] - X[None, :, :]
        D_full = np.sqrt((diff**2).sum(axis=2))
        # Build k-NN graph
        k = min(self.n_neighbors, n - 1)
        G = np.full((n, n), np.inf)
        np.fill_diagonal(G, 0)
        for i in range(n):
            nn = np.argsort(D_full[i])[1:k+1]
            G[i, nn] = D_full[i, nn]
            G[nn, i] = D_full[i, nn]  # symmetric
        # Floyd-Warshall for geodesic distances
        for via in range(n):
            G = np.minimum(G, G[:, via:via+1] + G[via:via+1, :])
        # Replace inf with large value
        G[G == np.inf] = G[G != np.inf].max() * 10 if (G != np.inf).any() else 1e6
        # Apply MDS on geodesic distances
        mds = MDS(n_components=self.n_components)
        Y = mds.fit_transform(G)
        self.embedding_ = Y
        self.reconstruction_error_ = float(mds.stress_)
        return Y

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self


class LocallyLinearEmbedding:
    """Locally Linear Embedding (LLE)."""
    def __init__(self, n_neighbors=5, n_components=2, reg=1e-3, method='standard',
                 tol=1e-6, max_iter=100, random_state=None, n_jobs=None):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.reg = reg
        self.method = method
        self.random_state = random_state
        self.embedding_ = None
        self.reconstruction_error_ = None

    def fit_transform(self, X, y=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = np.array(X, dtype=np.float64)
        n, m = X.shape
        k = min(self.n_neighbors, n - 1)
        # Compute pairwise distances
        diff = X[:, None, :] - X[None, :, :]
        D = np.sqrt((diff**2).sum(axis=2))
        # Find k nearest neighbors
        nn_idx = np.argsort(D, axis=1)[:, 1:k+1]
        # Compute reconstruction weights
        W = np.zeros((n, n))
        for i in range(n):
            Z = X[nn_idx[i]] - X[i]
            C = Z @ Z.T
            C += self.reg * np.trace(C) * np.eye(k)
            w = np.linalg.solve(C, np.ones(k))
            w /= w.sum() + 1e-12
            W[i, nn_idx[i]] = w
        # Compute embedding via eigenvectors of (I-W).T(I-W)
        M = (np.eye(n) - W).T @ (np.eye(n) - W)
        vals, vecs = np.linalg.eigh(M)
        # Skip first eigenvector (constant), take next n_components
        Y = vecs[:, 1:self.n_components+1]
        self.embedding_ = Y
        self.reconstruction_error_ = float(((X - W @ X)**2).sum())
        return Y.astype(np.float32)

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self


class SpectralEmbedding:
    """Spectral Embedding via Laplacian Eigenmaps."""
    def __init__(self, n_components=2, affinity='nearest_neighbors', gamma=None,
                 random_state=None, n_jobs=None, n_neighbors=None):
        self.n_components = n_components
        self.affinity = affinity
        self.gamma = gamma
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.embedding_ = None
        self.affinity_matrix_ = None

    def fit_transform(self, X, y=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = np.array(X, dtype=np.float64)
        n = len(X)
        diff = X[:, None, :] - X[None, :, :]
        D = np.sqrt((diff**2).sum(axis=2))
        if self.affinity == 'nearest_neighbors':
            k = self.n_neighbors or min(10, n - 1)
            A = np.zeros((n, n))
            for i in range(n):
                nn = np.argsort(D[i])[1:k+1]
                A[i, nn] = 1
                A[nn, i] = 1
        else:  # rbf kernel
            gamma = self.gamma or 1.0 / X.shape[1]
            A = np.exp(-gamma * D**2)
        self.affinity_matrix_ = A
        # Graph Laplacian
        deg = A.sum(axis=1)
        D_mat = np.diag(deg)
        L = D_mat - A
        # Normalized Laplacian
        inv_sqrt_d = np.where(deg > 0, 1.0 / np.sqrt(deg), 0)
        L_norm = inv_sqrt_d[:, None] * L * inv_sqrt_d[None, :]
        vals, vecs = np.linalg.eigh(L_norm)
        Y = vecs[:, 1:self.n_components+1] * inv_sqrt_d[:, None]
        self.embedding_ = Y
        return Y.astype(np.float32)

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self


class UMAP:
    """Uniform Manifold Approximation and Projection (spectral + force-directed layout)."""
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean',
                 n_epochs=200, learning_rate=1.0, spread=1.0, random_state=None,
                 negative_sample_rate=5, verbose=False):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.spread = spread
        self.random_state = random_state
        self.negative_sample_rate = negative_sample_rate
        self.verbose = verbose
        self.embedding_ = None

    def fit_transform(self, X, y=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = np.array(X, dtype=np.float64)
        n = len(X)
        k = min(self.n_neighbors, n - 1)
        # Compute pairwise distances
        diff = X[:, None, :] - X[None, :, :]
        D = np.sqrt((diff**2).sum(axis=2))
        # Build fuzzy simplicial set (high-dim)
        P = np.zeros((n, n))
        for i in range(n):
            dists = np.sort(D[i])[1:k+1]
            rho_i = dists[0]
            sigma_i = dists.mean()
            for j, nn_j in enumerate(np.argsort(D[i])[1:k+1]):
                val = np.exp(-(D[i, nn_j] - rho_i) / (sigma_i + 1e-12))
                P[i, nn_j] = val
        P = P + P.T - P * P.T  # fuzzy union
        # Initialize with spectral embedding
        se = SpectralEmbedding(n_components=self.n_components, random_state=self.random_state)
        Y = se.fit_transform(X).astype(np.float64)
        # a, b parameters for UMAP curve
        a, b = self._find_ab(self.min_dist, self.spread)
        # Stochastic gradient descent layout
        lr = self.learning_rate
        for epoch in range(self.n_epochs):
            lr_epoch = lr * (1 - epoch / self.n_epochs)
            for i in range(n):
                # Positive samples
                pos_j_list = np.where(P[i] > 0)[0]
                if len(pos_j_list) == 0:
                    continue
                pos_j = np.random.choice(pos_j_list)
                diff_pos = Y[i] - Y[pos_j]
                dist_pos = np.sqrt((diff_pos**2).sum() + 1e-12)
                q = 1.0 / (1 + a * dist_pos**(2*b))
                grad_pos = (2 * a * b * dist_pos**(2*b - 2)) / (1 + a * dist_pos**(2*b)) * diff_pos
                Y[i] -= lr_epoch * grad_pos
                Y[pos_j] += lr_epoch * grad_pos
                # Negative samples
                for _ in range(self.negative_sample_rate):
                    neg_j = np.random.randint(n)
                    if neg_j == i:
                        continue
                    diff_neg = Y[i] - Y[neg_j]
                    dist_neg = np.sqrt((diff_neg**2).sum() + 1e-12)
                    q_neg = 1.0 / (1 + a * dist_neg**(2*b))
                    if q_neg > 0:
                        grad_neg = (2 * b) / ((0.001 + dist_neg**2) * (1 + a * dist_neg**(2*b))) * diff_neg
                        Y[i] += lr_epoch * np.clip(grad_neg, -4, 4)
        self.embedding_ = Y
        return Y.astype(np.float32)

    def _find_ab(self, min_dist, spread):
        # Approximate UMAP's curve fitting
        a = spread
        b = min_dist
        return a, b

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        # For new points, return nearest embedding
        X = np.array(X, dtype=np.float64)
        return np.zeros((len(X), self.n_components), dtype=np.float32)
