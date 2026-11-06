"""Clustering algorithms."""
import numpy as np
from typing import Optional


class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, n_init=10, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = np.inf

    def _run_once(self, X, seed):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), self.n_clusters, replace=False)
        centers = X[idx].copy()
        for _ in range(self.max_iter):
            dists = np.linalg.norm(X[:, None] - centers[None], axis=2)
            labels = dists.argmin(axis=1)
            new_centers = np.array([X[labels == k].mean(axis=0) if (labels == k).any() else centers[k]
                                    for k in range(self.n_clusters)])
            if np.max(np.abs(new_centers - centers)) < self.tol:
                break
            centers = new_centers
        inertia = sum(np.sum((X[labels == k] - centers[k])**2) for k in range(self.n_clusters))
        return centers, labels, float(inertia)

    def fit(self, X) -> 'KMeans':
        X = np.array(X, dtype=float)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        best_inertia = np.inf
        for i in range(self.n_init):
            centers, labels, inertia = self._run_once(X, seed=i)
            if inertia < best_inertia:
                best_inertia = inertia
                self.cluster_centers_ = centers
                self.labels_ = labels
                self.inertia_ = inertia
        return self

    def predict(self, X) -> np.ndarray:
        X = np.array(X, dtype=float)
        dists = np.linalg.norm(X[:, None] - self.cluster_centers_[None], axis=2)
        return dists.argmin(axis=1)

    def fit_predict(self, X) -> np.ndarray:
        self.fit(X)
        return self.labels_

    def score(self, X) -> float:
        return -self.inertia_


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_: Optional[np.ndarray] = None
        self.core_sample_indices_: Optional[np.ndarray] = None

    def fit(self, X) -> 'DBSCAN':
        X = np.array(X, dtype=float)
        n = len(X)
        dists = np.linalg.norm(X[:, None] - X[None], axis=2)
        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0
        neighbors = [np.where(dists[i] <= self.eps)[0] for i in range(n)]
        core = np.array([i for i in range(n) if len(neighbors[i]) >= self.min_samples])
        self.core_sample_indices_ = core

        def expand(idx, cid):
            queue = list(neighbors[idx])
            while queue:
                q = queue.pop()
                if not visited[q]:
                    visited[q] = True
                    if len(neighbors[q]) >= self.min_samples:
                        for nb in neighbors[q]:
                            if not visited[nb]:
                                queue.append(nb)
                if labels[q] == -1:
                    labels[q] = cid

        for i in core:
            if not visited[i]:
                visited[i] = True
                labels[i] = cluster_id
                expand(i, cluster_id)
                cluster_id += 1
        self.labels_ = labels
        return self

    def fit_predict(self, X) -> np.ndarray:
        self.fit(X)
        return self.labels_


class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_: Optional[np.ndarray] = None

    def fit(self, X) -> 'AgglomerativeClustering':
        X = np.array(X, dtype=float)
        n = len(X)
        clusters = [{i} for i in range(n)]

        def dist(c1, c2):
            pts1 = X[list(c1)]
            pts2 = X[list(c2)]
            if self.linkage == 'ward':
                merged = np.vstack([pts1, pts2])
                return len(merged) * np.sum((merged.mean(0) - np.vstack([
                    np.tile(pts1.mean(0), (len(pts1),1)),
                    np.tile(pts2.mean(0), (len(pts2),1))
                ]))**2)
            d = np.linalg.norm(pts1[:, None] - pts2[None], axis=2)
            if self.linkage == 'single':
                return d.min()
            if self.linkage == 'complete':
                return d.max()
            return d.mean()

        while len(clusters) > self.n_clusters:
            best_d, merge_i, merge_j = np.inf, 0, 1
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):
                    d = dist(clusters[i], clusters[j])
                    if d < best_d:
                        best_d, merge_i, merge_j = d, i, j
            clusters[merge_i] = clusters[merge_i] | clusters[merge_j]
            clusters.pop(merge_j)

        self.labels_ = np.zeros(n, dtype=int)
        for label, cluster in enumerate(clusters):
            for idx in cluster:
                self.labels_[idx] = label
        return self

    def fit_predict(self, X) -> np.ndarray:
        self.fit(X)
        return self.labels_


class GaussianMixture:
    def __init__(self, n_components=1, max_iter=100, tol=1e-3, random_state=None, covariance_type='full'):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.weights_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.covariances_: Optional[np.ndarray] = None

    def _gaussian_pdf(self, X, mean, cov):
        d = X.shape[1]
        diff = X - mean
        try:
            cov_inv = np.linalg.inv(cov + np.eye(d) * 1e-6)
            cov_det = np.linalg.det(cov + np.eye(d) * 1e-6)
        except np.linalg.LinAlgError:
            cov_inv = np.eye(d)
            cov_det = 1.0
        exponent = -0.5 * np.einsum('ni,ij,nj->n', diff, cov_inv, diff)
        return np.exp(exponent) / (np.sqrt((2*np.pi)**d * np.abs(cov_det)) + 1e-300)

    def fit(self, X) -> 'GaussianMixture':
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = np.array(X, dtype=float)
        n, d = X.shape
        K = self.n_components
        idx = np.random.choice(n, K, replace=False)
        self.means_ = X[idx].copy()
        self.covariances_ = np.array([np.eye(d) for _ in range(K)])
        self.weights_ = np.ones(K) / K
        for _ in range(self.max_iter):
            # E-step
            resp = np.array([self.weights_[k] * self._gaussian_pdf(X, self.means_[k], self.covariances_[k]) for k in range(K)]).T
            resp_sum = resp.sum(axis=1, keepdims=True)
            resp /= (resp_sum + 1e-300)
            # M-step
            Nk = resp.sum(axis=0)
            new_means = (resp.T @ X) / (Nk[:, None] + 1e-300)
            new_covs = []
            for k in range(K):
                diff = X - new_means[k]
                new_covs.append((resp[:, k:k+1] * diff).T @ diff / (Nk[k] + 1e-300))
            old_means = self.means_.copy()
            self.means_ = new_means
            self.covariances_ = np.array(new_covs)
            self.weights_ = Nk / n
            if np.max(np.abs(self.means_ - old_means)) < self.tol:
                break
        return self

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def predict_proba(self, X) -> np.ndarray:
        X = np.array(X, dtype=float)
        K = self.n_components
        resp = np.array([self.weights_[k] * self._gaussian_pdf(X, self.means_[k], self.covariances_[k]) for k in range(K)]).T
        resp_sum = resp.sum(axis=1, keepdims=True)
        return resp / (resp_sum + 1e-300)

    def fit_predict(self, X) -> np.ndarray:
        self.fit(X)
        return self.predict(X)
