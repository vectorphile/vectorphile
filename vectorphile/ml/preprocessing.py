"""Preprocessing transformers — pure Python."""
import math
import random as _random
from typing import List, Optional
from vectorphile.ml._np_compat import to_array as _ta


def _col(X, j):
    return [X[i][j] for i in range(len(X))]


def _to_float_matrix(X):
    return [[float(v) for v in row] for row in X]


class StandardScaler:
    def fit(self, X):
        X2 = _to_float_matrix(X)
        n, p = len(X2), len(X2[0])
        self.mean_ = [sum(_col(X2, j)) / n for j in range(p)]
        self.scale_ = []
        for j in range(p):
            col = _col(X2, j)
            var = sum((v - self.mean_[j])**2 for v in col) / n
            self.scale_.append(math.sqrt(var) if var > 0 else 1.0)
        return self

    def transform(self, X):
        X2 = _to_float_matrix(X)
        return _ta([[(X2[i][j] - self.mean_[j]) / self.scale_[j]
                 for j in range(len(X2[0]))] for i in range(len(X2))])

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X2 = _to_float_matrix(X)
        return [[X2[i][j] * self.scale_[j] + self.mean_[j]
                 for j in range(len(X2[0]))] for i in range(len(X2))]


class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

    def fit(self, X):
        X2 = _to_float_matrix(X)
        p = len(X2[0])
        self.data_min_ = [min(_col(X2, j)) for j in range(p)]
        self.data_max_ = [max(_col(X2, j)) for j in range(p)]
        self.scale_ = [(self.feature_range[1] - self.feature_range[0]) /
                       max(self.data_max_[j] - self.data_min_[j], 1e-10)
                       for j in range(p)]
        return self

    def transform(self, X):
        X2 = _to_float_matrix(X)
        lo, hi = self.feature_range
        return _ta([[(X2[i][j] - self.data_min_[j]) * self.scale_[j] + lo
                 for j in range(len(X2[0]))] for i in range(len(X2))])

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X2 = _to_float_matrix(X)
        lo = self.feature_range[0]
        return [[(X2[i][j] - lo) / self.scale_[j] + self.data_min_[j]
                 for j in range(len(X2[0]))] for i in range(len(X2))]


class RobustScaler:
    def fit(self, X):
        X2 = _to_float_matrix(X)
        p = len(X2[0])
        self.center_ = []
        self.scale_ = []
        for j in range(p):
            col = sorted(_col(X2, j))
            n = len(col)
            med = col[n // 2] if n % 2 else (col[n//2-1] + col[n//2]) / 2
            q1 = col[n // 4]
            q3 = col[3 * n // 4]
            self.center_.append(med)
            self.scale_.append(max(q3 - q1, 1e-10))
        return self

    def transform(self, X):
        X2 = _to_float_matrix(X)
        return [[(X2[i][j] - self.center_[j]) / self.scale_[j]
                 for j in range(len(X2[0]))] for i in range(len(X2))]

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LabelEncoder:
    def fit(self, y):
        self.classes_ = sorted(set(y))
        self._map = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, y):
        return [self._map[v] for v in y]

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        return [self.classes_[int(v)] for v in y]


class OneHotEncoder:
    def __init__(self, sparse=False, drop=None):
        self.sparse = sparse
        self.drop = drop

    def fit(self, X):
        X2 = [list(row) for row in X]
        p = len(X2[0]) if X2 else 0
        self.categories_ = [sorted(set(_col(X2, j))) for j in range(p)]
        return self

    def transform(self, X):
        X2 = [list(row) for row in X]
        result = []
        for row in X2:
            encoded = []
            for j, val in enumerate(row):
                cats = self.categories_[j]
                encoded.extend([1.0 if val == c else 0.0 for c in cats])
            result.append(encoded)
        return _ta(result)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True, interaction_only=False):
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only

    def fit(self, X):
        p = len(X[0]) if X else 0
        self._p = p
        self._combos = []
        self._build_combos(p, self.degree)
        return self

    def _build_combos(self, p, degree):
        from itertools import combinations_with_replacement, combinations
        indices = list(range(p))
        if self.include_bias:
            self._combos.append(())
        for d in range(1, degree + 1):
            if self.interaction_only and d > 1:
                for combo in combinations(indices, d):
                    self._combos.append(combo)
            else:
                for combo in combinations_with_replacement(indices, d):
                    self._combos.append(combo)

    def transform(self, X):
        result = []
        for row in X:
            new_row = []
            for combo in self._combos:
                val = 1.0
                for idx in combo:
                    val *= float(row[idx])
                new_row.append(val)
            result.append(new_row)
        return result

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class Normalizer:
    def __init__(self, norm='l2'):
        self.norm = norm

    def fit(self, X):
        return self

    def transform(self, X):
        result = []
        for row in X:
            row = [float(v) for v in row]
            if self.norm == 'l2':
                n = math.sqrt(sum(v**2 for v in row)) or 1.0
            elif self.norm == 'l1':
                n = sum(abs(v) for v in row) or 1.0
            else:
                n = max(abs(v) for v in row) or 1.0
            result.append([v / n for v in row])
        return result

    def fit_transform(self, X):
        return self.transform(X)


class Binarizer:
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def transform(self, X):
        return [[1.0 if v > self.threshold else 0.0 for v in row] for row in X]

    def fit_transform(self, X):
        return self.transform(X)
