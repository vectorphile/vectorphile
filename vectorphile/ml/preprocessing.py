"""Preprocessing utilities."""
import numpy as np
from typing import Optional, List


class StandardScaler:
    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None

    def fit(self, X) -> 'StandardScaler':
        X = np.array(X, dtype=float)
        self.mean_ = X.mean(axis=0) if self.with_mean else np.zeros(X.shape[1])
        self.var_ = X.var(axis=0) if self.with_std else np.ones(X.shape[1])
        self.scale_ = np.sqrt(self.var_)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self

    def transform(self, X) -> np.ndarray:
        X = np.array(X, dtype=float)
        if self.copy:
            X = X.copy()
        if self.with_mean:
            X -= self.mean_
        if self.with_std:
            X /= self.scale_
        return X

    def fit_transform(self, X) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X) -> np.ndarray:
        X = np.array(X, dtype=float).copy()
        X *= self.scale_
        X += self.mean_
        return X


class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None

    def fit(self, X) -> 'MinMaxScaler':
        X = np.array(X, dtype=float)
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        data_range = self.data_max_ - self.data_min_
        data_range = np.where(data_range == 0, 1.0, data_range)
        fmin, fmax = self.feature_range
        self.scale_ = (fmax - fmin) / data_range
        self.min_ = fmin - self.data_min_ * self.scale_
        return self

    def transform(self, X) -> np.ndarray:
        return np.array(X, dtype=float) * self.scale_ + self.min_

    def fit_transform(self, X) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X) -> np.ndarray:
        return (np.array(X, dtype=float) - self.min_) / self.scale_


class RobustScaler:
    def __init__(self, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X) -> 'RobustScaler':
        X = np.array(X, dtype=float)
        if self.with_centering:
            self.center_ = np.median(X, axis=0)
        q_low, q_high = self.quantile_range
        if self.with_scaling:
            q = np.percentile(X, [q_low, q_high], axis=0)
            iqr = q[1] - q[0]
            self.scale_ = np.where(iqr == 0, 1.0, iqr)
        return self

    def transform(self, X) -> np.ndarray:
        X = np.array(X, dtype=float).copy()
        if self.with_centering:
            X -= self.center_
        if self.with_scaling:
            X /= self.scale_
        return X

    def fit_transform(self, X) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class LabelEncoder:
    def __init__(self):
        self.classes_: Optional[np.ndarray] = None

    def fit(self, y) -> 'LabelEncoder':
        self.classes_ = np.unique(y)
        return self

    def transform(self, y) -> np.ndarray:
        y = np.array(y)
        idx = np.searchsorted(self.classes_, y)
        return idx

    def fit_transform(self, y) -> np.ndarray:
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y) -> np.ndarray:
        return self.classes_[np.array(y)]


class OneHotEncoder:
    def __init__(self, sparse=False, drop=None, handle_unknown='error'):
        self.sparse = sparse
        self.drop = drop
        self.handle_unknown = handle_unknown
        self.categories_: Optional[List[np.ndarray]] = None

    def fit(self, X) -> 'OneHotEncoder':
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self

    def transform(self, X) -> np.ndarray:
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        parts = []
        for i, cats in enumerate(self.categories_):
            col = X[:, i]
            ohe = np.zeros((len(col), len(cats)))
            for j, c in enumerate(cats):
                ohe[:, j] = (col == c)
            parts.append(ohe)
        return np.concatenate(parts, axis=1)

    def fit_transform(self, X) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X) -> np.ndarray:
        X = np.array(X)
        result = []
        offset = 0
        for cats in self.categories_:
            k = len(cats)
            chunk = X[:, offset:offset+k]
            result.append(cats[chunk.argmax(axis=1)])
            offset += k
        return np.array(result).T


class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True, interaction_only=False):
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.n_features_in_: int = 0
        self.n_output_features_: int = 0

    def fit(self, X) -> 'PolynomialFeatures':
        X = np.array(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X) -> np.ndarray:
        from itertools import combinations_with_replacement, combinations
        X = np.array(X, dtype=float)
        n, p = X.shape
        features = [np.ones((n, 1))] if self.include_bias else []
        for d in range(1, self.degree + 1):
            if self.interaction_only and d > 1:
                combos = combinations(range(p), d)
            else:
                combos = combinations_with_replacement(range(p), d)
            for combo in combos:
                feat = np.prod(X[:, list(combo)], axis=1, keepdims=True)
                features.append(feat)
        result = np.concatenate(features, axis=1)
        self.n_output_features_ = result.shape[1]
        return result

    def fit_transform(self, X) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
