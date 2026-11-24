"""K-Nearest Neighbors — pure Python."""
import math
from typing import Optional


def _uniq_sorted(seq):
    return sorted(set(seq))


def _euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _minkowski(a, b, p):
    return sum(abs(x - y) ** p for x, y in zip(a, b)) ** (1 / p)


class _BaseKNN:
    def __init__(self, n_neighbors=5, metric='minkowski', p=2, weights='uniform',
                 algorithm='auto', n_jobs=None):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.weights = weights

    def _dist(self, a, b):
        if self.metric == 'euclidean' or (self.metric == 'minkowski' and self.p == 2):
            return _euclidean(a, b)
        if self.metric == 'manhattan' or (self.metric == 'minkowski' and self.p == 1):
            return sum(abs(x - y) for x, y in zip(a, b))
        if self.metric == 'minkowski':
            return _minkowski(a, b, self.p)
        return _euclidean(a, b)

    def fit(self, X, y):
        self.X_train_ = [[float(v) for v in row] for row in X]
        self.y_train_ = list(y)
        return self

    def _get_neighbors(self, x):
        x = [float(v) for v in x]
        dists = [self._dist(x, xtr) for xtr in self.X_train_]
        order = sorted(range(len(dists)), key=lambda i: dists[i])
        nn_idx = order[:self.n_neighbors]
        nn_dist = [dists[i] for i in nn_idx]
        nn_y = [self.y_train_[i] for i in nn_idx]
        return nn_idx, nn_dist, nn_y


class KNeighborsClassifier(_BaseKNN):
    def fit(self, X, y):
        super().fit(X, y)
        self.classes_ = _uniq_sorted(self.y_train_)
        return self

    def predict_proba(self, X):
        K = len(self.classes_)
        result = []
        for row in X:
            nn_idx, nn_dist, nn_y = self._get_neighbors(row)
            if self.weights == 'uniform':
                weights = [1.0] * len(nn_y)
            else:
                weights = [1.0 / (d + 1e-10) for d in nn_dist]
            total_w = sum(weights)
            proba = [0.0] * K
            for label, w in zip(nn_y, weights):
                ci = self.classes_.index(label)
                proba[ci] += w / total_w
            result.append(proba)
        return result

    def predict(self, X):
        proba = self.predict_proba(X)
        return [self.classes_[row.index(max(row))] for row in proba]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(a == b for a, b in zip(preds, y)) / len(y)

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        k = n_neighbors or self.n_neighbors
        all_dists, all_idx = [], []
        for row in X:
            nn_idx, nn_dist, _ = self._get_neighbors(row)
            all_idx.append(nn_idx[:k])
            all_dists.append(nn_dist[:k])
        if return_distance:
            return all_dists, all_idx
        return all_idx


class KNeighborsRegressor(_BaseKNN):
    def predict(self, X):
        result = []
        for row in X:
            nn_idx, nn_dist, nn_y = self._get_neighbors(row)
            nn_y_f = [float(v) for v in nn_y]
            if self.weights == 'uniform':
                result.append(sum(nn_y_f) / len(nn_y_f))
            else:
                weights = [1.0 / (d + 1e-10) for d in nn_dist]
                total_w = sum(weights)
                result.append(sum(v * w for v, w in zip(nn_y_f, weights)) / total_w)
        return result

    def score(self, X, y):
        preds = self.predict(X)
        y2 = [float(v) for v in y]
        mu = sum(y2) / len(y2)
        ss_res = sum((a - b) ** 2 for a, b in zip(y2, preds))
        ss_tot = sum((v - mu) ** 2 for v in y2)
        return float(1 - ss_res / (ss_tot + 1e-12))


class RadiusNeighborsClassifier:
    def __init__(self, radius=1.0, weights='uniform', metric='minkowski', p=2):
        self.radius = radius
        self.weights = weights
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        self.X_train_ = [[float(v) for v in row] for row in X]
        self.y_train_ = list(y)
        self.classes_ = _uniq_sorted(self.y_train_)
        return self

    def _dist(self, a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def predict(self, X):
        result = []
        for row in X:
            x = [float(v) for v in row]
            candidates = [(self._dist(x, xtr), self.y_train_[i])
                          for i, xtr in enumerate(self.X_train_)
                          if self._dist(x, xtr) <= self.radius]
            if not candidates:
                result.append(self.classes_[0])
                continue
            counts = {}
            for _, label in candidates:
                counts[label] = counts.get(label, 0) + 1
            result.append(max(counts, key=counts.get))
        return result

    def score(self, X, y):
        preds = self.predict(X)
        return sum(a == b for a, b in zip(preds, y)) / len(y)
