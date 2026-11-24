"""vectorphile.neighbors — K-Nearest Neighbors algorithms.

All computations use pure Python. No external dependencies.
"""
from __future__ import annotations
import math
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


def _to_list1d(y) -> List:
    if hasattr(y, 'to_flat_list'):
        return y.ravel().to_flat_list()
    if hasattr(y, 'tolist'):
        data = y.tolist()
    else:
        data = list(y)
    if isinstance(data, (int, float, str)):
        return [data]
    flat = []
    for v in data:
        if isinstance(v, (list, tuple)):
            flat.extend(v)
        else:
            flat.append(v)
    return flat


def _euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


def _manhattan(a: List[float], b: List[float]) -> float:
    return sum(abs(a[i] - b[i]) for i in range(len(a)))


def _chebyshev(a: List[float], b: List[float]) -> float:
    return max(abs(a[i] - b[i]) for i in range(len(a)))


def _dist_fn(metric: str):
    if metric in ('manhattan', 'l1'):
        return _manhattan
    if metric == 'chebyshev':
        return _chebyshev
    return _euclidean


def _pairwise_dists(X1: List[List[float]], X2: List[List[float]],
                    fn) -> List[List[float]]:
    return [[fn(X1[i], X2[j]) for j in range(len(X2))] for i in range(len(X1))]


# ---------------------------------------------------------------------------
# NearestNeighbors (brute-force)
# ---------------------------------------------------------------------------

class NearestNeighbors:
    """Unsupervised k-nearest neighbor finder."""

    def __init__(self, n_neighbors: int = 5, metric: str = 'euclidean',
                 algorithm: str = 'auto'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.algorithm = algorithm
        self._X: Optional[List[List[float]]] = None

    def fit(self, X) -> 'NearestNeighbors':
        self._X = _to_list2d(X)
        return self

    def kneighbors(self, X, n_neighbors: Optional[int] = None,
                   return_distance: bool = True):
        X2d = _to_list2d(X)
        k = n_neighbors or self.n_neighbors
        fn = _dist_fn(self.metric)
        D = _pairwise_dists(X2d, self._X, fn)
        indices = []
        dists = []
        for row in D:
            sorted_idx = sorted(range(len(row)), key=lambda i: row[i])[:k]
            indices.append(sorted_idx)
            dists.append([row[i] for i in sorted_idx])
        if return_distance:
            return dists, indices
        return indices

    def radius_neighbors(self, X, radius: float):
        X2d = _to_list2d(X)
        fn = _dist_fn(self.metric)
        D = _pairwise_dists(X2d, self._X, fn)
        all_indices = []
        all_dists = []
        for row in D:
            idx = [j for j, d in enumerate(row) if d <= radius]
            all_indices.append(idx)
            all_dists.append([row[j] for j in idx])
        return all_dists, all_indices


# ---------------------------------------------------------------------------
# KNeighborsClassifier
# ---------------------------------------------------------------------------

class KNeighborsClassifier:
    """K-Nearest Neighbors classifier."""

    def __init__(self, n_neighbors: int = 5, metric: str = 'euclidean',
                 weights: str = 'uniform'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self._nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        self._y: Optional[List] = None
        self.classes_: Optional[List] = None

    def fit(self, X, y) -> 'KNeighborsClassifier':
        self._nn.fit(X)
        self._y = _to_list1d(y)
        self.classes_ = sorted(set(self._y))
        return self

    def predict(self, X) -> List:
        dists, indices = self._nn.kneighbors(X)
        preds = []
        for i in range(len(dists)):
            neighbor_labels = [self._y[j] for j in indices[i]]
            if self.weights == 'distance':
                w = [1.0 / (d + 1e-10) for d in dists[i]]
            else:
                w = [1.0] * len(neighbor_labels)
            class_weights: dict = {}
            for label, weight in zip(neighbor_labels, w):
                class_weights[label] = class_weights.get(label, 0.0) + weight
            preds.append(max(class_weights, key=class_weights.get))
        return preds

    def predict_proba(self, X) -> List[List[float]]:
        dists, indices = self._nn.kneighbors(X)
        K = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        result = []
        for i in range(len(dists)):
            neighbor_labels = [self._y[j] for j in indices[i]]
            if self.weights == 'distance':
                w = [1.0 / (d + 1e-10) for d in dists[i]]
            else:
                w = [1.0] * len(neighbor_labels)
            total_w = sum(w)
            row = [0.0] * K
            for label, weight in zip(neighbor_labels, w):
                row[class_to_idx[label]] += weight / total_w
            result.append(row)
        return result

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        return sum(1 for p, t in zip(preds, y_true) if p == t) / max(len(y_true), 1)

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        return self._nn.kneighbors(X, n_neighbors, return_distance)


# ---------------------------------------------------------------------------
# KNeighborsRegressor
# ---------------------------------------------------------------------------

class KNeighborsRegressor:
    """K-Nearest Neighbors regressor."""

    def __init__(self, n_neighbors: int = 5, metric: str = 'euclidean',
                 weights: str = 'uniform'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self._nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        self._y: Optional[List[float]] = None

    def fit(self, X, y) -> 'KNeighborsRegressor':
        self._nn.fit(X)
        self._y = [float(v) for v in _to_list1d(y)]
        return self

    def predict(self, X) -> List[float]:
        dists, indices = self._nn.kneighbors(X)
        preds = []
        for i in range(len(dists)):
            neighbor_vals = [self._y[j] for j in indices[i]]
            if self.weights == 'distance':
                w = [1.0 / (d + 1e-10) for d in dists[i]]
                total_w = sum(w)
                preds.append(sum(w[k] * neighbor_vals[k] for k in range(len(w))) / total_w)
            else:
                preds.append(sum(neighbor_vals) / len(neighbor_vals))
        return preds

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = [float(v) for v in _to_list1d(y)]
        n = len(y_true)
        mean_y = sum(y_true) / n
        ss_res = sum((y_true[i] - preds[i]) ** 2 for i in range(n))
        ss_tot = sum((v - mean_y) ** 2 for v in y_true) + 1e-10
        return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# RadiusNeighborsClassifier
# ---------------------------------------------------------------------------

class RadiusNeighborsClassifier:
    """Classify based on neighbors within a fixed radius."""

    def __init__(self, radius: float = 1.0, metric: str = 'euclidean',
                 weights: str = 'uniform', outlier_label: int = -1):
        self.radius = radius
        self.metric = metric
        self.weights = weights
        self.outlier_label = outlier_label
        self._nn = NearestNeighbors(metric=metric)
        self._y: Optional[List] = None
        self.classes_: Optional[List] = None

    def fit(self, X, y) -> 'RadiusNeighborsClassifier':
        self._nn.fit(X)
        self._y = _to_list1d(y)
        self.classes_ = sorted(set(self._y))
        return self

    def predict(self, X) -> List:
        dists_list, indices_list = self._nn.radius_neighbors(X, self.radius)
        preds = []
        for i in range(len(dists_list)):
            idx = indices_list[i]
            if not idx:
                preds.append(self.outlier_label)
                continue
            counts: dict = {}
            for j in idx:
                label = self._y[j]
                counts[label] = counts.get(label, 0) + 1
            preds.append(max(counts, key=counts.get))
        return preds

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        return sum(1 for p, t in zip(preds, y_true) if p == t) / max(len(y_true), 1)


class RadiusNeighborsRegressor:
    """Regression based on neighbors within a fixed radius."""

    def __init__(self, radius: float = 1.0, weights: str = 'uniform', metric: str = 'euclidean'):
        self.radius = radius
        self.weights = weights
        self.metric = metric
        self._X: List[List[float]] = []
        self._y: List[float] = []

    def fit(self, X, y):
        self._X = _to_list2d(X)
        self._y = [float(v) for v in _to_list1d(y)]
        return self

    def predict(self, X) -> List[float]:
        X2d = _to_list2d(X)
        result = []
        for row in X2d:
            dists = [(_euclidean(row, x), i) for i, x in enumerate(self._X)]
            nbrs = [(d, i) for d, i in dists if d <= self.radius]
            if not nbrs:
                result.append(sum(self._y) / len(self._y) if self._y else 0.0)
            elif self.weights == 'distance':
                total_w = sum(1.0 / (d + 1e-10) for d, _ in nbrs)
                result.append(sum((1.0 / (d + 1e-10)) * self._y[i] for d, i in nbrs) / total_w)
            else:
                result.append(sum(self._y[i] for _, i in nbrs) / len(nbrs))
        return result

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = [float(v) for v in _to_list1d(y)]
        mean_y = sum(y_true) / len(y_true)
        ss_res = sum((y_true[i] - preds[i]) ** 2 for i in range(len(y_true)))
        ss_tot = sum((v - mean_y) ** 2 for v in y_true) + 1e-10
        return 1.0 - ss_res / ss_tot


class NearestCentroid:
    """Nearest centroid (Rocchio) classifier."""

    def __init__(self, metric: str = 'euclidean', shrink_threshold: Optional[float] = None):
        self.metric = metric
        self.shrink_threshold = shrink_threshold
        self.centroids_: List[List[float]] = []
        self.classes_: List = []

    def fit(self, X, y):
        X2d = _to_list2d(X)
        y_list = _to_list1d(y)
        self.classes_ = sorted(set(y_list))
        p = len(X2d[0]) if X2d else 0
        self.centroids_ = []
        for cls in self.classes_:
            idxs = [i for i, lbl in enumerate(y_list) if lbl == cls]
            centroid = [sum(X2d[i][j] for i in idxs) / len(idxs) for j in range(p)]
            self.centroids_.append(centroid)
        return self

    def predict(self, X) -> List:
        X2d = _to_list2d(X)
        return [self.classes_[min(range(len(self.centroids_)),
                key=lambda k: _euclidean(row, self.centroids_[k]))]
                for row in X2d]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        return sum(1 for p, t in zip(preds, y_true) if p == t) / max(len(y_true), 1)


__all__ = [
    'NearestNeighbors',
    'KNeighborsClassifier', 'KNeighborsRegressor',
    'RadiusNeighborsClassifier', 'RadiusNeighborsRegressor',
    'NearestCentroid',
]
