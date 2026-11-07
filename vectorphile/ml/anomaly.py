"""Anomaly detection algorithms."""
from __future__ import annotations
import numpy as np
from typing import Optional


class IsolationForest:
    """Isolation Forest anomaly detector."""
    def __init__(self, n_estimators=100, max_samples='auto', contamination=0.1,
                 max_features=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []
        self.threshold_ = None

    def _c(self, n):
        if n <= 1: return 0.0
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

    def _build_tree(self, X, depth_limit):
        def _split(data, depth):
            n = len(data)
            if n <= 1 or depth >= depth_limit:
                return {'type': 'leaf', 'size': n}
            feat = np.random.randint(data.shape[1])
            lo, hi = data[:, feat].min(), data[:, feat].max()
            if lo >= hi:
                return {'type': 'leaf', 'size': n}
            threshold = np.random.uniform(lo, hi)
            left_mask = data[:, feat] < threshold
            return {
                'type': 'node', 'feat': feat, 'threshold': threshold,
                'left': _split(data[left_mask], depth + 1),
                'right': _split(data[~left_mask], depth + 1),
            }
        return _split(X, 0)

    def _path_length(self, node, x, depth=0):
        if node['type'] == 'leaf':
            return depth + self._c(node['size'])
        if x[node['feat']] < node['threshold']:
            return self._path_length(node['left'], x, depth + 1)
        return self._path_length(node['right'], x, depth + 1)

    def fit(self, X, y=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = np.array(X, dtype=np.float64)
        n = len(X)
        if self.max_samples == 'auto':
            sample_size = min(256, n)
        elif isinstance(self.max_samples, float):
            sample_size = max(1, int(self.max_samples * n))
        else:
            sample_size = min(int(self.max_samples), n)
        depth_limit = int(np.ceil(np.log2(sample_size)))
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = np.random.choice(n, sample_size, replace=False)
            tree = self._build_tree(X[idx], depth_limit)
            self.trees_.append(tree)
        scores = self.score_samples(X)
        self.threshold_ = np.percentile(scores, 100 * self.contamination)
        return self

    def score_samples(self, X):
        X = np.array(X, dtype=np.float64)
        avg_paths = np.array([
            np.mean([self._path_length(t, x) for t in self.trees_])
            for x in X
        ])
        c = self._c(min(256, len(X)))
        return -np.power(2, -avg_paths / (c + 1e-12))

    def decision_function(self, X):
        return self.score_samples(X) - self.threshold_

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores < 0, -1, 1)

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)

    def __repr__(self):
        return f'IsolationForest(n_estimators={self.n_estimators}, contamination={self.contamination})'


class LocalOutlierFactor:
    """Local Outlier Factor — density-based anomaly detection."""
    def __init__(self, n_neighbors=20, contamination=0.1, metric='euclidean', novelty=False):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.novelty = novelty
        self.X_train_ = None
        self.lof_scores_ = None
        self.threshold_ = None

    def _distances(self, X1, X2):
        if self.metric == 'euclidean':
            diff = X1[:, None, :] - X2[None, :, :]
            return np.sqrt((diff**2).sum(axis=2))
        elif self.metric == 'manhattan':
            diff = X1[:, None, :] - X2[None, :, :]
            return np.abs(diff).sum(axis=2)
        else:
            diff = X1[:, None, :] - X2[None, :, :]
            return np.sqrt((diff**2).sum(axis=2))

    def _lrd(self, X_query, X_ref, k_dists_ref, nn_ref):
        D = self._distances(X_query, X_ref)
        # k nearest neighbors in ref for each query point
        k = min(self.n_neighbors, X_ref.shape[0] - 1)
        nn_query_idx = np.argsort(D, axis=1)[:, 1:k+1]
        reach_dists = np.array([
            [max(D[i, j], k_dists_ref[j]) for j in nn_query_idx[i]]
            for i in range(len(X_query))
        ])
        lrd = k / (reach_dists.sum(axis=1) + 1e-12)
        return lrd, nn_query_idx

    def fit(self, X, y=None):
        X = np.array(X, dtype=np.float64)
        self.X_train_ = X
        k = min(self.n_neighbors, len(X) - 1)
        D = self._distances(X, X)
        nn_idx = np.argsort(D, axis=1)[:, 1:k+1]
        k_dists = D[np.arange(len(X)), nn_idx[:, -1]]  # k-th neighbor distance
        reach_dists = np.array([
            [max(D[i, j], k_dists[j]) for j in nn_idx[i]]
            for i in range(len(X))
        ])
        lrd_train = k / (reach_dists.sum(axis=1) + 1e-12)
        # LOF = mean(lrd of neighbors) / lrd of point
        lof = np.array([
            lrd_train[nn_idx[i]].mean() / (lrd_train[i] + 1e-12)
            for i in range(len(X))
        ])
        self._lrd_train = lrd_train
        self._nn_idx = nn_idx
        self._k_dists = k_dists
        self.lof_scores_ = lof
        self.negative_outlier_factor_ = -lof
        self.threshold_ = np.percentile(lof, 100 * (1 - self.contamination))
        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        return np.where(self.lof_scores_ > self.threshold_, -1, 1)

    def predict(self, X):
        if not self.novelty:
            raise AttributeError("predict only available with novelty=True")
        X = np.array(X, dtype=np.float64)
        k = min(self.n_neighbors, len(self.X_train_))
        D = self._distances(X, self.X_train_)
        nn_idx = np.argsort(D, axis=1)[:, :k]
        reach_dists = np.array([
            [max(D[i, j], self._k_dists[j]) for j in nn_idx[i]]
            for i in range(len(X))
        ])
        lrd = k / (reach_dists.sum(axis=1) + 1e-12)
        lof = np.array([
            self._lrd_train[nn_idx[i]].mean() / (lrd[i] + 1e-12)
            for i in range(len(X))
        ])
        return np.where(lof > self.threshold_, -1, 1)

    def __repr__(self):
        return f'LocalOutlierFactor(n_neighbors={self.n_neighbors}, contamination={self.contamination})'


class EllipticEnvelope:
    """Fit a Gaussian distribution and flag outliers by Mahalanobis distance."""
    def __init__(self, contamination=0.1, support_fraction=None, random_state=None):
        self.contamination = contamination
        self.support_fraction = support_fraction
        self.random_state = random_state
        self.location_ = None
        self.covariance_ = None
        self.precision_ = None
        self.threshold_ = None

    def fit(self, X, y=None):
        X = np.array(X, dtype=np.float64)
        n, m = X.shape
        if self.support_fraction is not None:
            # Minimum Covariance Determinant (approximation)
            h = max(int(self.support_fraction * n), m + 1)
            best_det = np.inf
            best_subset = None
            for _ in range(10):  # 10 random starts
                idx = np.random.choice(n, h, replace=False)
                mu = X[idx].mean(axis=0)
                cov = np.cov(X[idx].T) + 1e-6 * np.eye(m)
                det = np.linalg.det(cov)
                if det < best_det:
                    best_det = det
                    best_subset = idx
            self.location_ = X[best_subset].mean(axis=0)
            self.covariance_ = np.cov(X[best_subset].T) + 1e-6 * np.eye(m)
        else:
            self.location_ = X.mean(axis=0)
            self.covariance_ = np.cov(X.T) + 1e-6 * np.eye(m)
        self.precision_ = np.linalg.inv(self.covariance_)
        mahal = self._mahalanobis(X)
        self.threshold_ = np.percentile(mahal, 100 * (1 - self.contamination))
        return self

    def _mahalanobis(self, X):
        X = np.array(X, dtype=np.float64)
        diff = X - self.location_
        return np.einsum('ni,ij,nj->n', diff, self.precision_, diff)

    def decision_function(self, X):
        return self.threshold_ - self._mahalanobis(X)

    def predict(self, X):
        mah = self._mahalanobis(X)
        return np.where(mah > self.threshold_, -1, 1)

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)

    def score_samples(self, X):
        return -self._mahalanobis(X)

    def __repr__(self):
        return f'EllipticEnvelope(contamination={self.contamination})'


class OneClassSVM:
    """One-Class SVM for novelty detection using RBF kernel approximation."""
    def __init__(self, nu=0.5, kernel='rbf', gamma='scale', tol=1e-3,
                 max_iter=1000, random_state=None):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.support_vectors_ = None
        self._rho = 0.0

    def _rbf_kernel(self, X, Y, gamma):
        diff = X[:, None, :] - Y[None, :, :]
        return np.exp(-gamma * (diff**2).sum(axis=2))

    def fit(self, X, y=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = np.array(X, dtype=np.float64)
        n, m = X.shape
        if self.gamma == 'scale':
            g = 1.0 / (m * X.var())
        elif self.gamma == 'auto':
            g = 1.0 / m
        else:
            g = float(self.gamma)
        self._gamma = g
        # Random Fourier Features approximation for scalability
        D = min(500, n)
        omega = np.random.randn(m, D) * np.sqrt(2 * g)
        b = np.random.uniform(0, 2 * np.pi, D)
        Z = np.sqrt(2.0 / D) * np.cos(X @ omega + b)
        # Optimize w via SGD on hinge loss
        w = np.zeros(D)
        lr = 1.0 / (n * self.nu)
        for it in range(self.max_iter):
            idx = np.random.permutation(n)
            for i in idx:
                score = Z[i] @ w
                if score < 1:
                    w = w * (1 - lr) + lr * Z[i]
                else:
                    w = w * (1 - lr)
        self._w = w
        self._omega = omega
        self._b = b
        self._rho = np.percentile(Z @ w, 100 * self.nu)
        self.support_vectors_ = X[Z @ w <= self._rho]
        return self

    def decision_function(self, X):
        X = np.array(X, dtype=np.float64)
        Z = np.sqrt(2.0 / len(self._b)) * np.cos(X @ self._omega + self._b)
        return Z @ self._w - self._rho

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def score_samples(self, X):
        return self.decision_function(X)

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)

    def __repr__(self):
        return f'OneClassSVM(nu={self.nu}, kernel={self.kernel})'
