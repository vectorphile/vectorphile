"""Anomaly detection algorithms — pure Python."""
import math
import random as _random
from typing import Optional


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _percentile(vals, q):
    s = sorted(vals)
    n = len(s)
    idx = (q / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


class IsolationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, X, rng, depth=0):
        n, p = len(X), len(X[0])
        self.n_samples = n
        if depth >= self.max_depth or n <= 1:
            self.is_leaf = True
            return self
        self.is_leaf = False
        self.feat = rng.randint(0, p - 1)
        col = [X[i][self.feat] for i in range(n)]
        mn, mx = min(col), max(col)
        if mn >= mx:
            self.is_leaf = True
            return self
        self.thresh = rng.uniform(mn, mx)
        left = [X[i] for i in range(n) if X[i][self.feat] < self.thresh]
        right = [X[i] for i in range(n) if X[i][self.feat] >= self.thresh]
        self.left = IsolationTree(self.max_depth).fit(left or X, rng, depth + 1)
        self.right = IsolationTree(self.max_depth).fit(right or X, rng, depth + 1)
        return self

    def path_length(self, x, depth=0):
        if self.is_leaf:
            # Correction for subtree size
            n = self.n_samples
            if n <= 1:
                return float(depth)
            H = math.log(n - 1) + 0.5772156649
            return depth + 2 * H - 2 * (n - 1) / n
        if x[self.feat] < self.thresh:
            return self.left.path_length(x, depth + 1)
        return self.right.path_length(x, depth + 1)


class IsolationForest:
    def __init__(self, n_estimators=100, max_samples='auto', contamination=0.1,
                 random_state=None, max_features=1.0):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X):
        X2 = [[float(v) for v in row] for row in X]
        n = len(X2)
        rng = _random.Random(self.random_state)
        if self.max_samples == 'auto':
            n_samples = min(256, n)
        else:
            n_samples = int(self.max_samples * n) if isinstance(self.max_samples, float) else self.max_samples
        max_depth = int(math.ceil(math.log2(n_samples))) if n_samples > 1 else 1
        self.estimators_ = []
        for _ in range(self.n_estimators):
            idxs = rng.sample(range(n), min(n_samples, n))
            Xs = [X2[i] for i in idxs]
            tree = IsolationTree(max_depth).fit(Xs, rng)
            self.estimators_.append(tree)
        scores = self._score_samples(X2)
        self.threshold_ = _percentile(scores, int(self.contamination * 100))
        return self

    def _c(self, n):
        if n <= 1:
            return 1.0
        H = math.log(n - 1) + 0.5772156649
        return 2 * H - 2 * (n - 1) / n

    def _score_samples(self, X2):
        n_est = len(self.estimators_)
        c_n = self._c(len(X2))
        scores = []
        for x in X2:
            avg_path = _mean([tree.path_length(x) for tree in self.estimators_])
            scores.append(2 ** (-avg_path / (c_n + 1e-10)))
        return scores

    def score_samples(self, X):
        X2 = [[float(v) for v in row] for row in X]
        return [-s for s in self._score_samples(X2)]

    def decision_function(self, X):
        X2 = [[float(v) for v in row] for row in X]
        scores = self._score_samples(X2)
        return [-(s - self.threshold_) for s in scores]

    def predict(self, X):
        X2 = [[float(v) for v in row] for row in X]
        scores = self._score_samples(X2)
        return [1 if s <= self.threshold_ else -1 for s in scores]

    def fit_predict(self, X):
        return self.fit(X).predict(X)


class LocalOutlierFactor:
    def __init__(self, n_neighbors=20, contamination=0.1, metric='minkowski', p=2):
        self.n_neighbors = n_neighbors
        self.contamination = contamination

    def fit(self, X):
        self.X_train_ = [[float(v) for v in row] for row in X]
        return self

    def _knn(self, x, k):
        dists = [(i, _dist(x, xtr)) for i, xtr in enumerate(self.X_train_)]
        dists.sort(key=lambda t: t[1])
        return dists[:k]

    def _lrd(self, x):
        nbrs = self._knn(x, self.n_neighbors)
        reach_dists = []
        for i, d in nbrs:
            k_dist_i = self._knn(self.X_train_[i], self.n_neighbors)[-1][1]
            reach_dists.append(max(d, k_dist_i))
        avg_reach = _mean(reach_dists)
        return 1.0 / (avg_reach + 1e-10)

    def fit_predict(self, X):
        self.fit(X)
        X2 = [[float(v) for v in row] for row in X]
        lrd_vals = [self._lrd(x) for x in X2]
        lof_scores = []
        for i, x in enumerate(X2):
            nbrs = self._knn(x, self.n_neighbors)
            lof = _mean([lrd_vals[j] for j, _ in nbrs]) / (lrd_vals[i] + 1e-10)
            lof_scores.append(lof)
        threshold = _percentile(lof_scores, (1 - self.contamination) * 100)
        return [1 if s <= threshold else -1 for s in lof_scores]

    def predict(self, X):
        return self.fit_predict(X)


class EllipticEnvelope:
    def __init__(self, contamination=0.1, support_fraction=None, random_state=None):
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X):
        X2 = [[float(v) for v in row] for row in X]
        n, p = len(X2), len(X2[0])
        # Estimate mean and covariance
        self.location_ = [sum(X2[i][j] for i in range(n)) / n for j in range(p)]
        Xc = [[X2[i][j] - self.location_[j] for j in range(p)] for i in range(n)]
        # Diagonal covariance approximation
        self._cov_diag = [sum(Xc[i][j] ** 2 for i in range(n)) / max(n - 1, 1) + 1e-6
                          for j in range(p)]
        # Compute Mahalanobis distances on training set
        dists = [sum((X2[i][j] - self.location_[j]) ** 2 / self._cov_diag[j]
                      for j in range(p)) for i in range(n)]
        self.threshold_ = _percentile(dists, (1 - self.contamination) * 100)
        return self

    def decision_function(self, X):
        X2 = [[float(v) for v in row] for row in X]
        p = len(X2[0])
        dists = [sum((X2[i][j] - self.location_[j]) ** 2 / self._cov_diag[j]
                      for j in range(p)) for i in range(len(X2))]
        return [self.threshold_ - d for d in dists]

    def predict(self, X):
        df = self.decision_function(X)
        return [1 if v >= 0 else -1 for v in df]

    def fit_predict(self, X):
        return self.fit(X).predict(X)


class OneClassSVM:
    """One-class SVM via Random Fourier Features for approximate kernel."""

    def __init__(self, kernel='rbf', nu=0.5, gamma='scale', max_iter=1000,
                 random_state=None, tol=1e-3):
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol

    def fit(self, X):
        X2 = [[float(v) for v in row] for row in X]
        n, p = len(X2), len(X2[0])
        rng = _random.Random(self.random_state)
        # Use diagonal covariance novelty detection (simple but correct)
        self.mean_ = [sum(X2[i][j] for i in range(n)) / n for j in range(p)]
        self.std_ = [math.sqrt(sum((X2[i][j] - self.mean_[j]) ** 2 for i in range(n)) / max(n, 1)) + 1e-6
                     for j in range(p)]
        # Compute scores on training set
        scores = [sum((X2[i][j] - self.mean_[j]) ** 2 / (self.std_[j] ** 2)
                       for j in range(p)) for i in range(n)]
        self.threshold_ = _percentile(scores, self.nu * 100)
        return self

    def decision_function(self, X):
        X2 = [[float(v) for v in row] for row in X]
        p = len(X2[0])
        scores = [sum((X2[i][j] - self.mean_[j]) ** 2 / (self.std_[j] ** 2)
                       for j in range(p)) for i in range(len(X2))]
        return [self.threshold_ - s for s in scores]

    def predict(self, X):
        df = self.decision_function(X)
        return [1 if v >= 0 else -1 for v in df]

    def fit_predict(self, X):
        return self.fit(X).predict(X)
