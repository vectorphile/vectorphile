"""vectorphile.cluster — Clustering algorithms.

All computations use pure Python. No external dependencies.
"""
from __future__ import annotations
import math
import random
from typing import List, Optional, Tuple


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


def _euclid(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def _centroid(rows: List[List[float]]) -> List[float]:
    n, d = len(rows), len(rows[0])
    return [sum(rows[i][j] for i in range(n)) / n for j in range(d)]


# ---------------------------------------------------------------------------
# AgglomerativeCluster
# ---------------------------------------------------------------------------

class AgglomerativeCluster:
    """Agglomerative (bottom-up) hierarchical clustering."""

    def __init__(self, n_clusters: int = 2, linkage: str = 'ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_: Optional[List[int]] = None

    def _linkage_distance(self, X: List[List[float]],
                          group_a: List[int], group_b: List[int]) -> float:
        a_rows = [X[i] for i in group_a]
        b_rows = [X[i] for i in group_b]
        if self.linkage == 'single':
            return min(_euclid(a, b) for a in a_rows for b in b_rows)
        elif self.linkage == 'complete':
            return max(_euclid(a, b) for a in a_rows for b in b_rows)
        else:  # ward / average: centroid distance
            ca = _centroid(a_rows)
            cb = _centroid(b_rows)
            return _euclid(ca, cb)

    def fit(self, X) -> 'AgglomerativeCluster':
        X2d = _to_list2d(X)
        n = len(X2d)
        clusters = {i: [i] for i in range(n)}
        active = set(range(n))
        while len(active) > self.n_clusters:
            best_pair = None
            best_dist = float('inf')
            active_list = sorted(active)
            for idx_i, ci in enumerate(active_list):
                for cj in active_list[idx_i + 1:]:
                    d = self._linkage_distance(X2d, clusters[ci], clusters[cj])
                    if d < best_dist:
                        best_dist = d
                        best_pair = (ci, cj)
            if best_pair is None:
                break
            ci, cj = best_pair
            clusters[ci] = clusters[ci] + clusters[cj]
            del clusters[cj]
            active.discard(cj)
        labels = [0] * n
        for label_idx, (_, members) in enumerate(clusters.items()):
            for m in members:
                labels[m] = label_idx
        self.labels_ = labels
        return self

    def fit_predict(self, X) -> List[int]:
        self.fit(X)
        return self.labels_


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------

class DBSCAN:
    """Density-Based Spatial Clustering of Applications with Noise."""

    def __init__(self, eps: float = 0.5, min_samples: int = 5,
                 metric: str = 'euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_: Optional[List[int]] = None
        self.core_sample_indices_: Optional[List[int]] = None

    def _distance(self, a: List[float], b: List[float]) -> float:
        if self.metric == 'manhattan':
            return sum(abs(a[i] - b[i]) for i in range(len(a)))
        return _euclid(a, b)

    def _region_query(self, X: List[List[float]], point_idx: int) -> List[int]:
        p = X[point_idx]
        return [i for i, x in enumerate(X) if self._distance(p, x) <= self.eps]

    def fit(self, X) -> 'DBSCAN':
        X2d = _to_list2d(X)
        n = len(X2d)
        labels = [-1] * n
        visited = [False] * n
        cluster_id = 0
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            neighbors = self._region_query(X2d, i)
            if len(neighbors) < self.min_samples:
                labels[i] = -1
            else:
                labels[i] = cluster_id
                queue = [q for q in neighbors if q != i]
                qi = 0
                while qi < len(queue):
                    q = queue[qi]
                    if not visited[q]:
                        visited[q] = True
                        q_neighbors = self._region_query(X2d, q)
                        if len(q_neighbors) >= self.min_samples:
                            queue.extend([x for x in q_neighbors if x not in set(queue)])
                    if labels[q] == -1:
                        labels[q] = cluster_id
                    qi += 1
                cluster_id += 1
        self.labels_ = labels
        self.core_sample_indices_ = [i for i, lb in enumerate(labels) if lb >= 0]
        return self

    def fit_predict(self, X) -> List[int]:
        self.fit(X)
        return self.labels_


# ---------------------------------------------------------------------------
# GaussianMixture (EM)
# ---------------------------------------------------------------------------

class GaussianMixture:
    """Gaussian Mixture Model fit via Expectation-Maximization."""

    def __init__(self, n_components: int = 1, max_iter: int = 100,
                 tol: float = 1e-4, random_state: Optional[int] = None,
                 covariance_type: str = 'full'):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.weights_: Optional[List[float]] = None
        self.means_: Optional[List[List[float]]] = None
        self.covariances_: Optional[List[List[List[float]]]] = None

    def _cholesky_det_inv(self, cov: List[List[float]]) -> Tuple[float, List[List[float]]]:
        d = len(cov)
        reg = [[cov[i][j] + (1e-6 if i == j else 0.0) for j in range(d)] for i in range(d)]
        # Cholesky L s.t. reg = L L^T
        L = [[0.0] * d for _ in range(d)]
        for i in range(d):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    L[i][j] = math.sqrt(max(reg[i][i] - s, 1e-15))
                else:
                    L[i][j] = (reg[i][j] - s) / (L[j][j] + 1e-15)
        # det = prod(L[i][i])^2
        det = 1.0
        for i in range(d):
            det *= L[i][i] ** 2
        det = max(det, 1e-300)
        # inv via Li^T Li
        Li = [[0.0] * d for _ in range(d)]
        for i in range(d):
            Li[i][i] = 1.0 / (L[i][i] + 1e-15)
            for j in range(i):
                Li[i][j] = -sum(L[i][k] * Li[k][j] for k in range(j, i)) / (L[i][i] + 1e-15)
        # inv = Li^T @ Li
        inv = [[sum(Li[k][i] * Li[k][j] for k in range(d)) for j in range(d)] for i in range(d)]
        return det, inv

    def _gaussian_pdf_row(self, x: List[float], mean: List[float],
                          cov_inv: List[List[float]], det: float) -> float:
        d = len(x)
        diff = [x[j] - mean[j] for j in range(d)]
        exponent = -0.5 * sum(diff[i] * sum(cov_inv[i][j] * diff[j] for j in range(d))
                               for i in range(d))
        norm = 1.0 / (math.sqrt((2 * math.pi) ** d * det) + 1e-300)
        return norm * math.exp(max(exponent, -500))

    def fit(self, X) -> 'GaussianMixture':
        X2d = _to_list2d(X)
        n, d = len(X2d), len(X2d[0])
        rng = random.Random(self.random_state)
        indices = rng.sample(range(n), min(self.n_components, n))
        K = self.n_components
        self.means_ = [X2d[i][:] for i in indices]
        self.weights_ = [1.0 / K] * K
        self.covariances_ = [[[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]
                              for _ in range(K)]
        prev_ll = -float('inf')
        for _ in range(self.max_iter):
            # E-step
            resp = [[0.0] * K for _ in range(n)]
            det_inv_cache = []
            for k in range(K):
                det, cov_inv = self._cholesky_det_inv(self.covariances_[k])
                det_inv_cache.append((det, cov_inv))
            for i in range(n):
                row_sum = 0.0
                for k in range(K):
                    det, cov_inv = det_inv_cache[k]
                    p = self.weights_[k] * self._gaussian_pdf_row(X2d[i], self.means_[k], cov_inv, det)
                    resp[i][k] = p
                    row_sum += p
                for k in range(K):
                    resp[i][k] /= (row_sum + 1e-300)
            # M-step
            nk = [sum(resp[i][k] for i in range(n)) + 1e-10 for k in range(K)]
            self.weights_ = [nk[k] / n for k in range(K)]
            self.means_ = [[sum(resp[i][k] * X2d[i][j] for i in range(n)) / nk[k]
                            for j in range(d)] for k in range(K)]
            for k in range(K):
                cov = [[0.0] * d for _ in range(d)]
                for i in range(n):
                    diff = [X2d[i][j] - self.means_[k][j] for j in range(d)]
                    for r in range(d):
                        for c in range(d):
                            cov[r][c] += resp[i][k] * diff[r] * diff[c]
                for r in range(d):
                    for c in range(d):
                        cov[r][c] /= nk[k]
                    cov[r][r] += 1e-6
                self.covariances_[k] = cov
            # Log-likelihood
            ll = 0.0
            det_inv_cache2 = [self._cholesky_det_inv(self.covariances_[k]) for k in range(K)]
            for i in range(n):
                p = sum(self.weights_[k] * self._gaussian_pdf_row(X2d[i], self.means_[k],
                        det_inv_cache2[k][1], det_inv_cache2[k][0]) for k in range(K))
                ll += math.log(p + 1e-300)
            ll /= n
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        return self

    def predict_proba(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        K = self.n_components
        det_inv_cache = [self._cholesky_det_inv(self.covariances_[k]) for k in range(K)]
        result = []
        for x in X2d:
            row = []
            for k in range(K):
                det, cov_inv = det_inv_cache[k]
                row.append(self.weights_[k] * self._gaussian_pdf_row(x, self.means_[k], cov_inv, det))
            s = sum(row) + 1e-300
            result.append([v / s for v in row])
        return result

    def predict(self, X) -> List[int]:
        proba = self.predict_proba(X)
        return [max(range(len(row)), key=lambda k: row[k]) for row in proba]

    def score(self, X) -> float:
        X2d = _to_list2d(X)
        K = self.n_components
        det_inv_cache = [self._cholesky_det_inv(self.covariances_[k]) for k in range(K)]
        ll = 0.0
        for x in X2d:
            p = sum(self.weights_[k] * self._gaussian_pdf_row(x, self.means_[k],
                    det_inv_cache[k][1], det_inv_cache[k][0]) for k in range(K))
            ll += math.log(p + 1e-300)
        return ll / len(X2d)

    def sample(self, n_samples: int = 1) -> Tuple[List[List[float]], List[int]]:
        rng = random.Random()
        cum = []
        s = 0.0
        for w in self.weights_:
            s += w; cum.append(s)
        X_out = []
        labels_out = []
        d = len(self.means_[0])
        for _ in range(n_samples):
            r = rng.random()
            k = next(i for i, c in enumerate(cum) if r <= c)
            # Sample from multivariate Gaussian via Cholesky
            L = [[0.0] * d for _ in range(d)]
            cov = self.covariances_[k]
            for i in range(d):
                for j in range(i + 1):
                    sv = sum(L[i][m] * L[j][m] for m in range(j))
                    if i == j:
                        L[i][j] = math.sqrt(max(cov[i][i] - sv, 1e-15))
                    else:
                        L[i][j] = (cov[i][j] - sv) / (L[j][j] + 1e-15)
            import statistics
            z = [random.gauss(0, 1) for _ in range(d)]
            x = [self.means_[k][j] + sum(L[j][m] * z[m] for m in range(j + 1)) for j in range(d)]
            X_out.append(x)
            labels_out.append(k)
        return X_out, labels_out


# ---------------------------------------------------------------------------
# MeanShift
# ---------------------------------------------------------------------------

class MeanShift:
    """Mean Shift clustering (bandwidth-based mode finding)."""

    def __init__(self, bandwidth: float = 1.0, max_iter: int = 300,
                 tol: float = 1e-3):
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_: Optional[List[List[float]]] = None
        self.labels_: Optional[List[int]] = None

    def fit(self, X) -> 'MeanShift':
        X2d = _to_list2d(X)
        n, d = len(X2d), len(X2d[0])
        seeds = [row[:] for row in X2d]
        for _ in range(self.max_iter):
            new_seeds = []
            max_shift = 0.0
            for s_idx, seed in enumerate(seeds):
                neighbors = [X2d[i] for i in range(n) if _euclid(seed, X2d[i]) < self.bandwidth]
                if not neighbors:
                    new_seeds.append(seed)
                    continue
                new_seed = _centroid(neighbors)
                shift = _euclid(new_seed, seed)
                if shift > max_shift:
                    max_shift = shift
                new_seeds.append(new_seed)
            seeds = new_seeds
            if max_shift < self.tol:
                break
        centers: List[List[float]] = []
        for seed in seeds:
            if not any(_euclid(seed, c) < self.bandwidth / 2 for c in centers):
                centers.append(seed)
        self.cluster_centers_ = centers
        labels = []
        for x in X2d:
            dists = [_euclid(x, c) for c in centers]
            labels.append(dists.index(min(dists)))
        self.labels_ = labels
        return self

    def fit_predict(self, X) -> List[int]:
        self.fit(X)
        return self.labels_


class SpectralClustering:
    """Spectral clustering via graph Laplacian eigenvectors + k-means.

    Args:
        n_clusters: number of clusters.
        affinity: 'rbf' (Gaussian kernel) or 'nearest_neighbors'.
        gamma: bandwidth for RBF affinity (1 / (2 * sigma^2)).
        n_neighbors: used when affinity='nearest_neighbors'.
        n_init: number of k-means initializations.
    """

    def __init__(self, n_clusters: int = 2, affinity: str = 'rbf',
                 gamma: float = 1.0, n_neighbors: int = 10, n_init: int = 10):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.n_init = n_init
        self.labels_: List[int] = []

    def _build_affinity(self, X: List[List[float]]) -> List[List[float]]:
        n = len(X)
        W = [[0.0] * n for _ in range(n)]
        if self.affinity == 'rbf':
            for i in range(n):
                for j in range(i + 1, n):
                    d2 = sum((X[i][k] - X[j][k]) ** 2 for k in range(len(X[i])))
                    w = math.exp(-self.gamma * d2)
                    W[i][j] = W[j][i] = w
        else:
            for i in range(n):
                dists = [(sum((X[i][k] - X[j][k]) ** 2 for k in range(len(X[i]))), j)
                         for j in range(n) if j != i]
                dists.sort()
                for _, j in dists[:self.n_neighbors]:
                    W[i][j] = W[j][i] = 1.0
        return W

    def fit(self, X):
        X2d = [[float(v) for v in row] for row in X] if not (
            X and isinstance(X[0], list)) else [[float(v) for v in row] for row in X]
        if hasattr(X, 'to_flat_list') and hasattr(X, 'shape'):
            n, d = X.shape[0], X.shape[1]
            flat = X.to_flat_list()
            X2d = [[float(flat[i * d + j]) for j in range(d)] for i in range(n)]
        n = len(X2d)
        W = self._build_affinity(X2d)
        # Degree matrix diagonal
        D = [sum(W[i]) for i in range(n)]
        # Normalized Laplacian: L = D^-1/2 (D - W) D^-1/2
        # Equivalent: L_sym = I - D^-1/2 W D^-1/2
        d_inv_sqrt = [1.0 / max(math.sqrt(D[i]), 1e-10) for i in range(n)]
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                L[i][j] = -d_inv_sqrt[i] * W[i][j] * d_inv_sqrt[j]
            L[i][i] += 1.0
        # Power iteration to get k smallest eigenvectors of L
        # Equivalent: largest eigenvectors of (2I - L)
        k = self.n_clusters
        V = [[1.0 / math.sqrt(n)] * n for _ in range(k)]
        import random as _rand
        for ki in range(k):
            # Random init
            v = [_rand.gauss(0, 1) for _ in range(n)]
            # Orthogonalize against previous
            for prev in V[:ki]:
                dot = sum(v[j] * prev[j] for j in range(n))
                v = [v[j] - dot * prev[j] for j in range(n)]
            norm = math.sqrt(sum(x * x for x in v))
            v = [x / max(norm, 1e-10) for x in v]
            # Power iteration on (2I - L)
            for _ in range(100):
                # Lv = L @ v
                Lv = [sum(L[i][j] * v[j] for j in range(n)) for i in range(n)]
                # (2I - L)v = 2v - Lv
                new_v = [2 * v[i] - Lv[i] for i in range(n)]
                # Orthogonalize
                for prev in V[:ki]:
                    dot = sum(new_v[j] * prev[j] for j in range(n))
                    new_v = [new_v[j] - dot * prev[j] for j in range(n)]
                norm = math.sqrt(sum(x * x for x in new_v))
                v = [x / max(norm, 1e-10) for x in new_v]
            V[ki] = v
        # Rows of V^T give the spectral embedding, one column per eigenvector
        embedding = [[V[ki][i] for ki in range(k)] for i in range(n)]
        # K-means on embedding
        from vectorphile.ml.cluster import KMeans as _KMeans
        try:
            km = _KMeans(n_clusters=k, n_init=self.n_init)
            km.fit(embedding)
            self.labels_ = km.labels_
        except Exception:
            # Fallback: greedy assignment
            centers = [embedding[i] for i in range(k)]
            for _ in range(100):
                labels = []
                for pt in embedding:
                    dists = [sum((pt[j] - c[j]) ** 2 for j in range(k)) for c in centers]
                    labels.append(dists.index(min(dists)))
                new_centers = [[0.0] * k for _ in range(k)]
                counts = [0] * k
                for i, pt in enumerate(embedding):
                    for j in range(k):
                        new_centers[labels[i]][j] += pt[j]
                    counts[labels[i]] += 1
                for c in range(k):
                    if counts[c] > 0:
                        centers[c] = [new_centers[c][j] / counts[c] for j in range(k)]
            self.labels_ = labels
        return self

    def fit_predict(self, X) -> List[int]:
        self.fit(X)
        return self.labels_


__all__ = [
    'AgglomerativeCluster', 'DBSCAN', 'GaussianMixture', 'MeanShift', 'SpectralClustering',
]
