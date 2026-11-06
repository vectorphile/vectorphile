"""K-Nearest Neighbors."""
import numpy as np
from typing import Optional


class _BaseKNN:
    def __init__(self, n_neighbors=5, metric='euclidean', weights='uniform'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None

    def fit(self, X, y) -> '_BaseKNN':
        self.X_train_ = np.array(X, dtype=float)
        self.y_train_ = np.array(y)
        return self

    def _distances(self, X) -> np.ndarray:
        X = np.array(X, dtype=float)
        if self.metric == 'euclidean':
            return np.linalg.norm(X[:, None] - self.X_train_[None], axis=2)
        if self.metric == 'manhattan':
            return np.abs(X[:, None] - self.X_train_[None]).sum(axis=2)
        return np.linalg.norm(X[:, None] - self.X_train_[None], axis=2)

    def _get_neighbors(self, X):
        dists = self._distances(X)
        nn_idx = np.argsort(dists, axis=1)[:, :self.n_neighbors]
        nn_dists = np.take_along_axis(dists, nn_idx, axis=1)
        return nn_idx, nn_dists


class KNeighborsClassifier(_BaseKNN):
    def __init__(self, n_neighbors=5, metric='euclidean', weights='uniform'):
        super().__init__(n_neighbors, metric, weights)
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X, y) -> 'KNeighborsClassifier':
        self.classes_ = np.unique(y)
        return super().fit(X, y)

    def predict(self, X) -> np.ndarray:
        nn_idx, nn_dists = self._get_neighbors(X)
        nn_labels = self.y_train_[nn_idx]
        result = []
        for i, (labels, dists) in enumerate(zip(nn_labels, nn_dists)):
            if self.weights == 'distance':
                dists = np.where(dists == 0, 1e-10, dists)
                w = 1.0 / dists
            else:
                w = np.ones_like(dists)
            counts = {}
            for label, weight in zip(labels, w):
                counts[label] = counts.get(label, 0) + weight
            result.append(max(counts, key=counts.get))
        return np.array(result)

    def predict_proba(self, X) -> np.ndarray:
        nn_idx, nn_dists = self._get_neighbors(X)
        nn_labels = self.y_train_[nn_idx]
        result = np.zeros((len(X), len(self.classes_)))
        for i, (labels, dists) in enumerate(zip(nn_labels, nn_dists)):
            if self.weights == 'distance':
                dists = np.where(dists == 0, 1e-10, dists)
                w = 1.0 / dists
            else:
                w = np.ones_like(dists)
            w /= w.sum()
            for label, weight in zip(labels, w):
                class_idx = np.where(self.classes_ == label)[0][0]
                result[i, class_idx] += weight
        return result

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.array(y)))


class KNeighborsRegressor(_BaseKNN):
    def predict(self, X) -> np.ndarray:
        nn_idx, nn_dists = self._get_neighbors(X)
        nn_vals = self.y_train_[nn_idx]
        if self.weights == 'distance':
            nn_dists = np.where(nn_dists == 0, 1e-10, nn_dists)
            w = 1.0 / nn_dists
            w /= w.sum(axis=1, keepdims=True)
            return (nn_vals * w).sum(axis=1)
        return nn_vals.mean(axis=1)

    def score(self, X, y) -> float:
        y = np.array(y, dtype=float)
        pred = self.predict(X)
        ss_res = np.sum((y - pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return float(1 - ss_res / (ss_tot + 1e-12))
