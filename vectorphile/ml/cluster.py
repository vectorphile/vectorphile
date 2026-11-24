"""Clustering algorithms — pure Python."""
import math
import random as _random
from typing import Optional
from vectorphile.ml._np_compat import to_array as _ta


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _mean_col(rows, j):
    return sum(row[j] for row in rows) / len(rows)


class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None,
                 init='k-means++', n_init=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.n_init = n_init

    def _init_centers(self, X, rng):
        n, p = len(X), len(X[0])
        if self.init == 'k-means++':
            centers = [X[rng.randint(0, n - 1)][:]]
            for _ in range(1, self.n_clusters):
                dists = [min(_dist(x, c) ** 2 for c in centers) for x in X]
                total = sum(dists)
                r = rng.random() * total
                cum = 0.0
                for i, d in enumerate(dists):
                    cum += d
                    if cum >= r:
                        centers.append(X[i][:])
                        break
                else:
                    centers.append(X[rng.randint(0, n - 1)][:])
        else:
            centers = [X[i][:] for i in rng.sample(range(n), self.n_clusters)]
        return centers

    def _run_once(self, X, rng):
        n, p = len(X), len(X[0])
        centers = self._init_centers(X, rng)
        labels = [-1] * n
        for _ in range(self.max_iter):
            new_labels = [min(range(self.n_clusters), key=lambda k: _dist(X[i], centers[k]))
                          for i in range(n)]
            # Update centers
            new_centers = []
            for k in range(self.n_clusters):
                cluster = [X[i] for i in range(n) if new_labels[i] == k]
                if cluster:
                    new_centers.append([_mean_col(cluster, j) for j in range(p)])
                else:
                    new_centers.append(centers[k][:])
            # Check convergence
            shift = max(_dist(centers[k], new_centers[k]) for k in range(self.n_clusters))
            centers, labels = new_centers, new_labels
            if shift < self.tol:
                break
        inertia = sum(_dist(X[i], centers[labels[i]]) ** 2 for i in range(n))
        return labels, centers, inertia

    def fit(self, X):
        X2 = [[float(v) for v in row] for row in X]
        rng = _random.Random(self.random_state)
        best_labels, best_centers, best_inertia = None, None, float('inf')
        for _ in range(self.n_init):
            labels, centers, inertia = self._run_once(X2, rng)
            if inertia < best_inertia:
                best_labels, best_centers, best_inertia = labels, centers, inertia
        self.labels_ = best_labels
        self.cluster_centers_ = best_centers
        self.inertia_ = best_inertia
        return self

    def predict(self, X):
        X2 = [[float(v) for v in row] for row in X]
        return [min(range(self.n_clusters), key=lambda k: _dist(x, self.cluster_centers_[k]))
                for x in X2]

    def fit_predict(self, X):
        return self.fit(X).labels_


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def fit(self, X):
        X2 = [[float(v) for v in row] for row in X]
        n = len(X2)
        labels = [-1] * n
        visited = [False] * n
        cluster_id = 0

        def neighbors(i):
            return [j for j in range(n) if j != i and _dist(X2[i], X2[j]) <= self.eps]

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            nbrs = neighbors(i)
            if len(nbrs) < self.min_samples:
                labels[i] = -1  # noise
                continue
            labels[i] = cluster_id
            queue = list(nbrs)
            qi = 0
            while qi < len(queue):
                j = queue[qi]
                qi += 1
                if not visited[j]:
                    visited[j] = True
                    nbrs2 = neighbors(j)
                    if len(nbrs2) >= self.min_samples:
                        queue.extend(nbrs2)
                if labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1

        self.labels_ = _ta(labels)
        self.core_sample_indices_ = [i for i in range(n)
                                     if labels[i] >= 0 and
                                     len([j for j in range(n) if j != i and
                                          _dist(X2[i], X2[j]) <= self.eps]) >= self.min_samples]
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_


class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage='ward', affinity='euclidean'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.affinity = affinity

    def fit(self, X):
        X2 = [[float(v) for v in row] for row in X]
        n = len(X2)
        # Initialize each point as its own cluster
        clusters = {i: [i] for i in range(n)}
        active = set(range(n))

        def cluster_dist(a_idx, b_idx):
            pts_a = clusters[a_idx]
            pts_b = clusters[b_idx]
            dists = [_dist(X2[i], X2[j]) for i in pts_a for j in pts_b]
            if self.linkage == 'single':
                return min(dists)
            if self.linkage == 'complete':
                return max(dists)
            return sum(dists) / len(dists)  # average / ward approx

        next_id = n
        while len(active) > self.n_clusters:
            best_d, best_a, best_b = float('inf'), None, None
            active_list = sorted(active)
            for i in range(len(active_list)):
                for j in range(i + 1, len(active_list)):
                    d = cluster_dist(active_list[i], active_list[j])
                    if d < best_d:
                        best_d, best_a, best_b = d, active_list[i], active_list[j]
            new_cluster = clusters[best_a] + clusters[best_b]
            clusters[next_id] = new_cluster
            active.discard(best_a)
            active.discard(best_b)
            active.add(next_id)
            next_id += 1

        labels = [0] * n
        for label_id, cluster_id in enumerate(sorted(active)):
            for pt in clusters[cluster_id]:
                labels[pt] = label_id
        self.labels_ = labels
        return self

    def fit_predict(self, X):
        return self.fit(X).labels_


class GaussianMixture:
    """Gaussian Mixture Model with EM — pure Python."""

    def __init__(self, n_components=1, max_iter=100, tol=1e-3, random_state=None,
                 covariance_type='full'):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.covariance_type = covariance_type

    def _log_gaussian(self, x, mean, cov_diag):
        p = len(x)
        log_det = sum(math.log(max(v, 1e-15)) for v in cov_diag)
        z = sum((x[j] - mean[j]) ** 2 / max(cov_diag[j], 1e-15) for j in range(p))
        return -0.5 * (p * math.log(2 * math.pi) + log_det + z)

    def fit(self, X):
        X2 = [[float(v) for v in row] for row in X]
        n, p = len(X2), len(X2[0])
        rng = _random.Random(self.random_state)
        K = self.n_components
        # Initialize
        self.weights_ = [1.0 / K] * K
        idxs = rng.sample(range(n), K)
        self.means_ = [X2[i][:] for i in idxs]
        self.covariances_ = [[1.0] * p for _ in range(K)]

        for _ in range(self.max_iter):
            # E-step
            gamma = []
            for i in range(n):
                log_probs = [math.log(max(self.weights_[k], 1e-300)) +
                             self._log_gaussian(X2[i], self.means_[k], self.covariances_[k])
                             for k in range(K)]
                mx = max(log_probs)
                probs = [math.exp(lp - mx) for lp in log_probs]
                total = sum(probs)
                gamma.append([pr / total for pr in probs])

            # M-step
            Nk = [sum(gamma[i][k] for i in range(n)) for k in range(K)]
            old_means = [m[:] for m in self.means_]
            for k in range(K):
                nk = max(Nk[k], 1e-10)
                self.means_[k] = [sum(gamma[i][k] * X2[i][j] for i in range(n)) / nk
                                   for j in range(p)]
                self.covariances_[k] = [
                    sum(gamma[i][k] * (X2[i][j] - self.means_[k][j]) ** 2 for i in range(n)) / nk + 1e-6
                    for j in range(p)]
                self.weights_[k] = Nk[k] / n

            shift = max(_dist(self.means_[k], old_means[k]) for k in range(K))
            if shift < self.tol:
                break

        return self

    def predict(self, X):
        X2 = [[float(v) for v in row] for row in X]
        K = self.n_components
        result = []
        for x in X2:
            scores = [math.log(max(self.weights_[k], 1e-300)) +
                      self._log_gaussian(x, self.means_[k], self.covariances_[k])
                      for k in range(K)]
            result.append(scores.index(max(scores)))
        return result

    def predict_proba(self, X):
        X2 = [[float(v) for v in row] for row in X]
        K = self.n_components
        result = []
        for x in X2:
            log_probs = [math.log(max(self.weights_[k], 1e-300)) +
                         self._log_gaussian(x, self.means_[k], self.covariances_[k])
                         for k in range(K)]
            mx = max(log_probs)
            probs = [math.exp(lp - mx) for lp in log_probs]
            total = sum(probs)
            result.append([pr / total for pr in probs])
        return result

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    def score(self, X):
        X2 = [[float(v) for v in row] for row in X]
        K = self.n_components
        total = 0.0
        for x in X2:
            log_probs = [math.log(max(self.weights_[k], 1e-300)) +
                         self._log_gaussian(x, self.means_[k], self.covariances_[k])
                         for k in range(K)]
            mx = max(log_probs)
            total += mx + math.log(sum(math.exp(lp - mx) for lp in log_probs))
        return total / len(X2)
