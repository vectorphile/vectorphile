"""vectorphile.preprocessing — Data preprocessing utilities.

All computations use pure Python. No external dependencies.
"""
from __future__ import annotations
import math
import itertools
from typing import Any, Dict, List, Optional, Union


def _to_list2d(X) -> List[List]:
    if hasattr(X, 'to_flat_list'):
        flat = X.to_flat_list()
        if X.ndim == 1:
            return [[v] for v in flat]
        n, d = X.shape[0], X.shape[1]
        return [[flat[i * d + j] for j in range(d)] for i in range(n)]
    if hasattr(X, 'tolist'):
        data = X.tolist()
    else:
        data = list(X)
    if data and not isinstance(data[0], (list, tuple)):
        return [[v] for v in data]
    return [list(row) for row in data]


def _to_list1d(y) -> List:
    if hasattr(y, 'to_flat_list'):
        return y.ravel().to_flat_list()
    if hasattr(y, 'tolist'):
        data = y.tolist()
    else:
        data = list(y)
    flat = []
    for v in data:
        if isinstance(v, (list, tuple)):
            flat.extend(v)
        else:
            flat.append(v)
    return flat


def _is_nan(x) -> bool:
    try:
        return math.isnan(x)
    except (TypeError, ValueError):
        return x is None


def _median(vals: List[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0


# ---------------------------------------------------------------------------
# Scalers
# ---------------------------------------------------------------------------

class StandardScaler:
    """Zero-mean unit-variance scaling per feature."""

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: Optional[List[float]] = None
        self.scale_: Optional[List[float]] = None

    def fit(self, X):
        data = _to_list2d(X)
        n, d = len(data), len(data[0])
        self.mean_ = [sum(data[i][j] for i in range(n)) / n for j in range(d)]
        self.scale_ = []
        for j in range(d):
            var = sum((data[i][j] - self.mean_[j]) ** 2 for i in range(n)) / max(n - 1, 1)
            self.scale_.append(math.sqrt(var) or 1.0)
        return self

    def transform(self, X) -> List[List[float]]:
        data = _to_list2d(X)
        result = []
        for row in data:
            r = []
            for j, v in enumerate(row):
                val = v - (self.mean_[j] if self.with_mean else 0.0)
                val = val / (self.scale_[j] if self.with_std else 1.0)
                r.append(val)
            result.append(r)
        return result

    def fit_transform(self, X) -> List[List[float]]:
        return self.fit(X).transform(X)

    def inverse_transform(self, X) -> List[List[float]]:
        data = _to_list2d(X)
        return [[data[i][j] * (self.scale_[j] if self.with_std else 1.0) +
                 (self.mean_[j] if self.with_mean else 0.0)
                 for j in range(len(data[i]))] for i in range(len(data))]


class MinMaxScaler:
    """Scale features to [feature_range[0], feature_range[1]]."""

    def __init__(self, feature_range=(0.0, 1.0)):
        self.feature_range = feature_range
        self.min_: Optional[List[float]] = None
        self.max_: Optional[List[float]] = None

    def fit(self, X):
        data = _to_list2d(X)
        n, d = len(data), len(data[0])
        self.min_ = [min(data[i][j] for i in range(n)) for j in range(d)]
        self.max_ = [max(data[i][j] for i in range(n)) for j in range(d)]
        return self

    def transform(self, X) -> List[List[float]]:
        lo, hi = self.feature_range
        data = _to_list2d(X)
        result = []
        for row in data:
            r = []
            for j, v in enumerate(row):
                denom = self.max_[j] - self.min_[j] or 1.0
                r.append(lo + (v - self.min_[j]) / denom * (hi - lo))
            result.append(r)
        return result

    def fit_transform(self, X) -> List[List[float]]:
        return self.fit(X).transform(X)

    def inverse_transform(self, X) -> List[List[float]]:
        lo, hi = self.feature_range
        data = _to_list2d(X)
        return [[self.min_[j] + (data[i][j] - lo) / max(hi - lo, 1e-15) *
                 (self.max_[j] - self.min_[j])
                 for j in range(len(data[i]))] for i in range(len(data))]


class RobustScaler:
    """Scale features using median and IQR (robust to outliers)."""

    def __init__(self, with_centering: bool = True, with_scaling: bool = True,
                 quantile_range=(25.0, 75.0)):
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.center_: Optional[List[float]] = None
        self.scale_: Optional[List[float]] = None

    def fit(self, X):
        data = _to_list2d(X)
        n, d = len(data), len(data[0])
        self.center_, self.scale_ = [], []
        for j in range(d):
            col = sorted(data[i][j] for i in range(n))
            self.center_.append(_median(col))
            q1 = _percentile_sorted(col, self.quantile_range[0])
            q3 = _percentile_sorted(col, self.quantile_range[1])
            self.scale_.append(q3 - q1 or 1.0)
        return self

    def transform(self, X) -> List[List[float]]:
        data = _to_list2d(X)
        return [[((v - (self.center_[j] if self.with_centering else 0.0)) /
                  (self.scale_[j] if self.with_scaling else 1.0))
                 for j, v in enumerate(row)] for row in data]

    def fit_transform(self, X) -> List[List[float]]:
        return self.fit(X).transform(X)


class Normalizer:
    """Normalize samples to unit norm (L1, L2, or max)."""

    def __init__(self, norm: str = 'l2'):
        self.norm = norm

    def fit(self, X):
        return self  # stateless

    def transform(self, X) -> List[List[float]]:
        data = _to_list2d(X)
        result = []
        for row in data:
            if self.norm == 'l1':
                n = sum(abs(v) for v in row) or 1.0
            elif self.norm == 'max':
                n = max(abs(v) for v in row) or 1.0
            else:  # l2
                n = math.sqrt(sum(v * v for v in row)) or 1.0
            result.append([v / n for v in row])
        return result

    def fit_transform(self, X) -> List[List[float]]:
        return self.transform(X)


def _percentile_sorted(sorted_vals: List[float], q: float) -> float:
    n = len(sorted_vals)
    idx = q / 100.0 * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


# ---------------------------------------------------------------------------
# LabelEncoder
# ---------------------------------------------------------------------------

class LabelEncoder:
    """Encode labels to integers 0..n_classes-1."""

    def __init__(self):
        self.classes_: Optional[List] = None
        self._class_to_int: Dict = {}

    def fit(self, y) -> 'LabelEncoder':
        y_flat = _to_list1d(y)
        self.classes_ = sorted(set(y_flat))
        self._class_to_int = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, y) -> List[int]:
        return [self._class_to_int[c] for c in _to_list1d(y)]

    def fit_transform(self, y) -> List[int]:
        return self.fit(y).transform(y)

    def inverse_transform(self, y) -> List:
        return [self.classes_[int(i)] for i in _to_list1d(y)]

    @property
    def n_classes_(self) -> int:
        return len(self.classes_) if self.classes_ is not None else 0


# ---------------------------------------------------------------------------
# OrdinalEncoder
# ---------------------------------------------------------------------------

class OrdinalEncoder:
    """Encode categorical features as ordinal integers."""

    def __init__(self, handle_unknown: str = 'error'):
        self.handle_unknown = handle_unknown
        self.categories_: Optional[List[List]] = None

    def fit(self, X) -> 'OrdinalEncoder':
        X2d = _to_list2d(X)
        p = len(X2d[0]) if X2d else 0
        self.categories_ = [sorted(set(row[j] for row in X2d)) for j in range(p)]
        return self

    def transform(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        p = len(X2d[0])
        result = []
        for row in X2d:
            encoded = []
            for j, cats in enumerate(self.categories_):
                val = row[j]
                if val in cats:
                    encoded.append(float(cats.index(val)))
                elif self.handle_unknown == 'use_encoded_value':
                    encoded.append(-1.0)
                else:
                    raise ValueError(f'Unknown category {val!r} in column {j}')
            result.append(encoded)
        return result

    def fit_transform(self, X) -> List[List[float]]:
        return self.fit(X).transform(X)

    def inverse_transform(self, X) -> List[List]:
        X2d = _to_list2d(X)
        result = []
        for row in X2d:
            decoded = []
            for j, cats in enumerate(self.categories_):
                i = int(row[j])
                decoded.append(cats[i] if 0 <= i < len(cats) else None)
            result.append(decoded)
        return result


# ---------------------------------------------------------------------------
# OneHotEncoder
# ---------------------------------------------------------------------------

class OneHotEncoder:
    """Encode categorical features as one-hot binary arrays."""

    def __init__(self, sparse: bool = False, drop: Optional[str] = None,
                 handle_unknown: str = 'error'):
        self.sparse = sparse
        self.drop = drop
        self.handle_unknown = handle_unknown
        self.categories_: Optional[List[List]] = None
        self._n_values: List[int] = []

    def fit(self, X) -> 'OneHotEncoder':
        X2d = _to_list2d(X)
        p = len(X2d[0]) if X2d else 0
        self.categories_ = [sorted(set(row[j] for row in X2d)) for j in range(p)]
        self._n_values = [len(cats) for cats in self.categories_]
        return self

    def transform(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        total_cols = sum(self._n_values)
        result = []
        for row in X2d:
            encoded = [0.0] * total_cols
            col_offset = 0
            for j, cats in enumerate(self.categories_):
                val = row[j]
                if val in cats:
                    encoded[col_offset + cats.index(val)] = 1.0
                elif self.handle_unknown != 'ignore':
                    raise ValueError(f'Unknown category {val!r}')
                col_offset += self._n_values[j]
            result.append(encoded)
        return result

    def fit_transform(self, X) -> List[List[float]]:
        return self.fit(X).transform(X)

    def get_feature_names_out(self) -> List[str]:
        return [f'x{j}_{c}' for j, cats in enumerate(self.categories_) for c in cats]


# ---------------------------------------------------------------------------
# LabelBinarizer
# ---------------------------------------------------------------------------

class LabelBinarizer:
    """Binarize labels in a one-vs-all fashion."""

    def __init__(self):
        self.classes_: Optional[List] = None

    def fit(self, y) -> 'LabelBinarizer':
        self.classes_ = sorted(set(_to_list1d(y)))
        return self

    def transform(self, y) -> List[List[float]]:
        y_flat = _to_list1d(y)
        n, K = len(y_flat), len(self.classes_)
        if K == 2:
            c = self.classes_[1]
            return [[1.0 if v == c else 0.0] for v in y_flat]
        cat_map = {c: i for i, c in enumerate(self.classes_)}
        result = [[0.0] * K for _ in range(n)]
        for i, v in enumerate(y_flat):
            if v in cat_map:
                result[i][cat_map[v]] = 1.0
        return result

    def fit_transform(self, y) -> List[List[float]]:
        return self.fit(y).transform(y)

    def inverse_transform(self, Y) -> List:
        Y2d = _to_list2d(Y)
        if len(Y2d[0]) == 1:
            return [self.classes_[1] if row[0] >= 0.5 else self.classes_[0] for row in Y2d]
        return [self.classes_[max(range(len(row)), key=lambda i: row[i])] for row in Y2d]


# ---------------------------------------------------------------------------
# SimpleImputer
# ---------------------------------------------------------------------------

class SimpleImputer:
    """Impute missing values with mean/median/most_frequent/constant."""

    def __init__(self, strategy: str = 'mean', fill_value: Any = None,
                 missing_values: Any = float('nan')):
        self.strategy = strategy
        self.fill_value = fill_value
        self.missing_values = missing_values
        self.statistics_: Optional[List[float]] = None

    def _is_missing(self, x) -> bool:
        try:
            if math.isnan(self.missing_values):
                return _is_nan(x)
        except (TypeError, ValueError):
            pass
        return x == self.missing_values

    def fit(self, X) -> 'SimpleImputer':
        X2d = _to_list2d(X)
        p = len(X2d[0]) if X2d else 0
        stats = []
        for j in range(p):
            col = [float(row[j]) for row in X2d if not self._is_missing(row[j])]
            if not col:
                stats.append(0.0)
            elif self.strategy == 'mean':
                stats.append(sum(col) / len(col))
            elif self.strategy == 'median':
                stats.append(_median(col))
            elif self.strategy == 'most_frequent':
                counts: Dict = {}
                for v in col:
                    counts[v] = counts.get(v, 0) + 1
                stats.append(max(counts, key=counts.get))
            else:  # constant
                stats.append(float(self.fill_value) if self.fill_value is not None else 0.0)
        self.statistics_ = stats
        return self

    def transform(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        p = len(X2d[0])
        result = []
        for row in X2d:
            new_row = []
            for j in range(p):
                new_row.append(self.statistics_[j] if self._is_missing(row[j]) else float(row[j]))
            result.append(new_row)
        return result

    def fit_transform(self, X) -> List[List[float]]:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# PolynomialFeatures
# ---------------------------------------------------------------------------

class PolynomialFeatures:
    """Generate polynomial and interaction features."""

    def __init__(self, degree: int = 2, interaction_only: bool = False,
                 include_bias: bool = True):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self._combos: Optional[List[tuple]] = None

    def fit(self, X) -> 'PolynomialFeatures':
        X2d = _to_list2d(X)
        n_features = len(X2d[0]) if X2d else 0
        self._combos = []
        if self.include_bias:
            self._combos.append(())
        for d in range(1, self.degree + 1):
            if self.interaction_only:
                for combo in itertools.combinations(range(n_features), d):
                    self._combos.append(combo)
            else:
                for combo in itertools.combinations_with_replacement(range(n_features), d):
                    self._combos.append(combo)
        return self

    def transform(self, X) -> List[List[float]]:
        X2d = [[float(v) for v in row] for row in _to_list2d(X)]
        result = []
        for row in X2d:
            new_row = []
            for combo in self._combos:
                if not combo:
                    new_row.append(1.0)
                else:
                    val = 1.0
                    for idx in combo:
                        val *= row[idx]
                    new_row.append(val)
            result.append(new_row)
        return result

    def fit_transform(self, X) -> List[List[float]]:
        return self.fit(X).transform(X)

    @property
    def n_output_features_(self) -> int:
        return len(self._combos) if self._combos else 0


# ---------------------------------------------------------------------------
# Binarizer
# ---------------------------------------------------------------------------

class Binarizer:
    """Binarize data at a threshold."""

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def fit(self, X) -> 'Binarizer':
        return self

    def transform(self, X) -> List[List[float]]:
        return [[1.0 if float(v) > self.threshold else 0.0 for v in row]
                for row in _to_list2d(X)]

    def fit_transform(self, X) -> List[List[float]]:
        return self.transform(X)


# ---------------------------------------------------------------------------
# FunctionTransformer
# ---------------------------------------------------------------------------

class FunctionTransformer:
    """Apply a function as a transform."""

    def __init__(self, func=None, inverse_func=None, validate: bool = True):
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate

    def fit(self, X) -> 'FunctionTransformer':
        return self

    def transform(self, X):
        if self.func is None:
            return X
        return self.func(X)

    def inverse_transform(self, X):
        if self.inverse_func is None:
            return X
        return self.inverse_func(X)

    def fit_transform(self, X):
        return self.transform(X)


# ---------------------------------------------------------------------------
# KBinsDiscretizer
# ---------------------------------------------------------------------------

class KBinsDiscretizer:
    """Discretize continuous features into k bins."""

    def __init__(self, n_bins: int = 5, strategy: str = 'uniform',
                 encode: str = 'ordinal'):
        self.n_bins = n_bins
        self.strategy = strategy
        self.encode = encode
        self.bin_edges_: Optional[List[List[float]]] = None

    def fit(self, X) -> 'KBinsDiscretizer':
        X2d = [[float(v) for v in row] for row in _to_list2d(X)]
        n, p = len(X2d), len(X2d[0]) if X2d else 0
        self.bin_edges_ = []
        for j in range(p):
            col = sorted(X2d[i][j] for i in range(n))
            mn, mx = col[0], col[-1]
            if self.strategy == 'quantile':
                edges = [col[int(q * (n - 1) / self.n_bins)] for q in range(self.n_bins + 1)]
                edges[0] = mn; edges[-1] = mx
            else:  # uniform
                step = (mx - mn) / self.n_bins
                edges = [mn + step * i for i in range(self.n_bins + 1)]
            self.bin_edges_.append(edges)
        return self

    def transform(self, X) -> List[List[float]]:
        X2d = [[float(v) for v in row] for row in _to_list2d(X)]
        p = len(X2d[0]) if X2d else 0
        result = []
        for row in X2d:
            new_row = []
            for j, edges in enumerate(self.bin_edges_):
                v = row[j]
                # digitize: find bin index
                idx = 0
                for e in edges[1:-1]:
                    if v > e:
                        idx += 1
                new_row.append(float(idx))
            result.append(new_row)
        return result

    def fit_transform(self, X) -> List[List[float]]:
        return self.fit(X).transform(X)


class PowerTransformer:
    """Yeo-Johnson and Box-Cox power transforms."""

    def __init__(self, method: str = 'yeo-johnson', standardize: bool = True):
        self.method = method
        self.standardize = standardize
        self.lambdas_: List[float] = []
        self._means: List[float] = []
        self._stds: List[float] = []

    @staticmethod
    def _yeo_johnson(x: float, lam: float) -> float:
        if x >= 0:
            if abs(lam) < 1e-10:
                return math.log1p(x)
            return ((x + 1) ** lam - 1) / lam
        else:
            if abs(lam - 2) < 1e-10:
                return -math.log1p(-x)
            return -((-x + 1) ** (2 - lam) - 1) / (2 - lam)

    @staticmethod
    def _box_cox(x: float, lam: float) -> float:
        if x <= 0:
            raise ValueError("Box-Cox requires positive values.")
        if abs(lam) < 1e-10:
            return math.log(x)
        return (x ** lam - 1) / lam

    def _opt_lambda(self, col: List[float]) -> float:
        best_lam, best_ll = 0.0, -1e18
        n = len(col)
        for lam_int in range(-20, 21):
            lam = lam_int / 10.0
            try:
                transformed = [self._yeo_johnson(v, lam) if self.method == 'yeo-johnson'
                                else self._box_cox(v, lam) for v in col]
            except (ValueError, OverflowError):
                continue
            mean_t = sum(transformed) / n
            var_t = sum((t - mean_t) ** 2 for t in transformed) / n
            if var_t <= 0:
                continue
            ll = -n / 2 * math.log(var_t)
            if self.method == 'yeo-johnson':
                ll += (lam - 1) * sum(math.log(abs(v) + 1) * (1 if v >= 0 else -1) for v in col)
            else:
                ll += (lam - 1) * sum(math.log(v) for v in col)
            if ll > best_ll:
                best_ll = ll
                best_lam = lam
        return best_lam

    def fit(self, X):
        X2d = _to_list2d(X)
        n = len(X2d)
        p = len(X2d[0]) if X2d else 0
        cols = [[X2d[i][j] for i in range(n)] for j in range(p)]
        self.lambdas_ = [self._opt_lambda(c) for c in cols]
        if self.standardize:
            self._means = []
            self._stds = []
            for j, col in enumerate(cols):
                lam = self.lambdas_[j]
                t = [self._yeo_johnson(v, lam) if self.method == 'yeo-johnson'
                     else self._box_cox(v, lam) for v in col]
                mn = sum(t) / len(t)
                sd = math.sqrt(sum((x - mn) ** 2 for x in t) / len(t)) or 1.0
                self._means.append(mn)
                self._stds.append(sd)
        return self

    def transform(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        result = []
        for row in X2d:
            new_row = []
            for j, v in enumerate(row):
                lam = self.lambdas_[j]
                t = (self._yeo_johnson(float(v), lam) if self.method == 'yeo-johnson'
                     else self._box_cox(float(v), lam))
                if self.standardize:
                    t = (t - self._means[j]) / self._stds[j]
                new_row.append(t)
            result.append(new_row)
        return result

    def fit_transform(self, X) -> List[List[float]]:
        return self.fit(X).transform(X)

    def inverse_transform(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        result = []
        for row in X2d:
            new_row = []
            for j, v in enumerate(row):
                t = float(v)
                if self.standardize:
                    t = t * self._stds[j] + self._means[j]
                lam = self.lambdas_[j]
                if self.method == 'yeo-johnson':
                    if t >= 0:
                        new_row.append((t * lam + 1) ** (1 / lam) - 1 if abs(lam) > 1e-10
                                       else math.expm1(t))
                    else:
                        new_row.append(1 - (-(2 - lam) * t + 1) ** (1 / (2 - lam))
                                       if abs(lam - 2) > 1e-10 else -math.expm1(-t))
                else:
                    new_row.append(math.exp(t) if abs(lam) < 1e-10
                                   else (t * lam + 1) ** (1 / lam))
            result.append(new_row)
        return result


__all__ = [
    'LabelEncoder', 'OrdinalEncoder', 'OneHotEncoder', 'LabelBinarizer',
    'SimpleImputer',
    'PolynomialFeatures',
    'Binarizer', 'FunctionTransformer',
    'KBinsDiscretizer', 'PowerTransformer',
]
