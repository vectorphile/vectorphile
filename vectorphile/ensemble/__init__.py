"""vectorphile.ensemble — Ensemble methods.

All computations use pure Python and vectorphile's NDArray primitives.
No external dependencies.
"""
from __future__ import annotations
import math
import random
import copy
from typing import List, Optional, Any

from vectorphile._core.ndarray import (
    NDArray as _NDA,
    _c_strides as _nd_c_strides,
    _DType as _NDType,
)
import array as _pyarray


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _to_list2d(X) -> List[List[float]]:
    """Convert X (array-like) to list of float rows."""
    if isinstance(X, _NDA):
        flat = X.to_flat_list()
        if X.ndim == 1:
            return [[float(v)] for v in flat]
        n, d = X.shape[0], X.shape[1]
        return [[float(flat[i * d + j]) for j in range(d)] for i in range(n)]
    if hasattr(X, 'tolist'):
        data = X.tolist()
    elif isinstance(X, (list, tuple)):
        data = list(X)
    else:
        try:
            import array as _a
            data = list(X)
        except Exception:
            data = [X]
    if data and not isinstance(data[0], (list, tuple)):
        return [[float(v)] for v in data]
    return [[float(v) for v in row] for row in data]


def _to_list1d(y) -> List[float]:
    """Convert y (array-like) to flat float list."""
    if isinstance(y, _NDA):
        return [float(v) for v in y.ravel().to_flat_list()]
    if hasattr(y, 'tolist'):
        return [float(v) for v in y.tolist()]
    if isinstance(y, (list, tuple)):
        return [float(v) for v in y]
    return [float(y)]


def _unique_sorted(vals: List) -> List:
    seen = set()
    result = []
    for v in vals:
        if v not in seen:
            seen.add(v); result.append(v)
    return sorted(result)


def _argmax(row: List[float]) -> int:
    best, best_i = row[0], 0
    for i, v in enumerate(row[1:], 1):
        if v > best: best, best_i = v, i
    return best_i


def _nd_from_list(flat: List[float], shape: tuple) -> _NDA:
    buf = _pyarray.array('f', [float(v) for v in flat])
    return _NDA(buf, shape, _nd_c_strides(shape), _NDType('float32'))


def _bootstrap(X: List[List[float]], y: List[float], rng: random.Random,
               n_samples: int, feature_indices: List[int]):
    idx = [rng.randint(0, len(X) - 1) for _ in range(n_samples)]
    X_b = [[X[i][j] for j in feature_indices] for i in idx]
    y_b = [y[i] for i in idx]
    return X_b, y_b


def _select_features(X: List[List[float]], feature_indices: List[int]) -> List[List[float]]:
    return [[row[j] for j in feature_indices] for row in X]


# ---------------------------------------------------------------------------
# VotingClassifier
# ---------------------------------------------------------------------------

class VotingClassifier:
    """Voting ensemble for classification."""

    def __init__(self, estimators: List, voting: str = 'hard',
                 weights: Optional[List[float]] = None):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.estimators_: List = []
        self.classes_: Optional[List] = None

    def fit(self, X, y) -> 'VotingClassifier':
        X2d = _to_list2d(X)
        y1d = _to_list1d(y)
        self.classes_ = _unique_sorted(y1d)
        self.estimators_ = []
        for name, est in self.estimators:
            cloned = copy.deepcopy(est)
            cloned.fit(X2d, y1d)
            self.estimators_.append((name, cloned))
        return self

    def predict_proba(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        K = len(self.classes_)
        n = len(X2d)
        weights = self.weights or [1.0] * len(self.estimators_)
        total_w = sum(weights)
        result = [[0.0] * K for _ in range(n)]
        for (name, est), w in zip(self.estimators_, weights):
            if hasattr(est, 'predict_proba'):
                proba = est.predict_proba(X2d)
                proba_flat = _to_list2d(proba) if not isinstance(proba, list) else proba
                for i in range(n):
                    for k in range(K):
                        result[i][k] += w * (proba_flat[i][k] if k < len(proba_flat[i]) else 0.0)
        for i in range(n):
            for k in range(K):
                result[i][k] /= total_w
        return result

    def predict(self, X) -> List:
        X2d = _to_list2d(X)
        n = len(X2d)
        if self.voting == 'soft':
            proba = self.predict_proba(X2d)
            return [self.classes_[_argmax(row)] for row in proba]
        # Hard voting
        all_preds = []
        for _, est in self.estimators_:
            p = est.predict(X2d)
            p_list = _to_list1d(p) if not isinstance(p, list) else p
            all_preds.append(p_list)
        result = []
        for i in range(n):
            votes = [all_preds[e][i] for e in range(len(all_preds))]
            counts = {}
            for v in votes:
                counts[v] = counts.get(v, 0) + 1
            result.append(max(counts, key=counts.get))
        return result

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        return sum(1 for p, t in zip(preds, y_true) if p == t) / max(len(y_true), 1)


class VotingRegressor:
    """Voting ensemble for regression (weighted average)."""

    def __init__(self, estimators: List, weights: Optional[List[float]] = None):
        self.estimators = estimators
        self.weights = weights
        self.estimators_: List = []

    def fit(self, X, y) -> 'VotingRegressor':
        X2d = _to_list2d(X)
        y1d = _to_list1d(y)
        self.estimators_ = []
        for name, est in self.estimators:
            cloned = copy.deepcopy(est)
            cloned.fit(X2d, y1d)
            self.estimators_.append((name, cloned))
        return self

    def predict(self, X) -> List[float]:
        X2d = _to_list2d(X)
        weights = self.weights or [1.0] * len(self.estimators_)
        total_w = sum(weights)
        n = len(X2d)
        result = [0.0] * n
        for (_, est), w in zip(self.estimators_, weights):
            p = est.predict(X2d)
            p_list = _to_list1d(p) if not isinstance(p, list) else p
            for i in range(n):
                result[i] += w * p_list[i] / total_w
        return result

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        mean_y = sum(y_true) / len(y_true)
        ss_res = sum((y_true[i] - preds[i]) ** 2 for i in range(len(y_true)))
        ss_tot = sum((v - mean_y) ** 2 for v in y_true) + 1e-10
        return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# BaggingClassifier / BaggingRegressor
# ---------------------------------------------------------------------------

class BaggingClassifier:
    """Bootstrap Aggregating (Bagging) for classification."""

    def __init__(self, base_estimator=None, n_estimators: int = 10,
                 max_samples: float = 1.0, max_features: float = 1.0,
                 bootstrap: bool = True, random_state: Optional[int] = None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.estimators_: List = []
        self.estimators_features_: List = []
        self.classes_: Optional[List] = None

    def fit(self, X, y) -> 'BaggingClassifier':
        X2d = _to_list2d(X)
        y1d = _to_list1d(y)
        self.classes_ = _unique_sorted(y1d)
        n, d = len(X2d), len(X2d[0]) if X2d else 1
        n_samples = max(1, int(n * self.max_samples))
        n_features = max(1, int(d * self.max_features))
        rng = random.Random(self.random_state)
        self.estimators_ = []
        self.estimators_features_ = []
        for _ in range(self.n_estimators):
            feat_idx = sorted(rng.sample(range(d), min(n_features, d)))
            if self.bootstrap:
                X_b, y_b = _bootstrap(X2d, y1d, rng, n_samples, feat_idx)
            else:
                indices = rng.sample(range(n), min(n_samples, n))
                X_b = [[X2d[i][j] for j in feat_idx] for i in indices]
                y_b = [y1d[i] for i in indices]
            est = copy.deepcopy(self.base_estimator)
            est.fit(X_b, y_b)
            self.estimators_.append(est)
            self.estimators_features_.append(feat_idx)
        return self

    def predict_proba(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        K = len(self.classes_)
        n = len(X2d)
        result = [[0.0] * K for _ in range(n)]
        for est, feat_idx in zip(self.estimators_, self.estimators_features_):
            X_sub = [[row[j] for j in feat_idx] for row in X2d]
            if hasattr(est, 'predict_proba'):
                proba = est.predict_proba(X_sub)
                p_flat = _to_list2d(proba) if not isinstance(proba, list) else proba
                for i in range(n):
                    for k in range(K):
                        result[i][k] += p_flat[i][k] if k < len(p_flat[i]) else 0.0
            else:
                preds = est.predict(X_sub)
                p_list = _to_list1d(preds) if not isinstance(preds, list) else preds
                for i in range(n):
                    if int(p_list[i]) in self.classes_:
                        ci = self.classes_.index(int(p_list[i]))
                        result[i][ci] += 1.0
        m = self.n_estimators
        for i in range(n):
            for k in range(K):
                result[i][k] /= m
        return result

    def predict(self, X) -> List:
        proba = self.predict_proba(X)
        return [self.classes_[_argmax(row)] for row in proba]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        return sum(1 for p, t in zip(preds, y_true) if p == t) / max(len(y_true), 1)


class BaggingRegressor:
    """Bootstrap Aggregating (Bagging) for regression."""

    def __init__(self, base_estimator=None, n_estimators: int = 10,
                 max_samples: float = 1.0, max_features: float = 1.0,
                 bootstrap: bool = True, random_state: Optional[int] = None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.estimators_: List = []
        self.estimators_features_: List = []

    def fit(self, X, y) -> 'BaggingRegressor':
        X2d = _to_list2d(X)
        y1d = _to_list1d(y)
        n, d = len(X2d), len(X2d[0]) if X2d else 1
        n_samples = max(1, int(n * self.max_samples))
        n_features = max(1, int(d * self.max_features))
        rng = random.Random(self.random_state)
        self.estimators_ = []
        self.estimators_features_ = []
        for _ in range(self.n_estimators):
            feat_idx = sorted(rng.sample(range(d), min(n_features, d)))
            if self.bootstrap:
                X_b, y_b = _bootstrap(X2d, y1d, rng, n_samples, feat_idx)
            else:
                indices = rng.sample(range(n), min(n_samples, n))
                X_b = [[X2d[i][j] for j in feat_idx] for i in indices]
                y_b = [y1d[i] for i in indices]
            est = copy.deepcopy(self.base_estimator)
            est.fit(X_b, y_b)
            self.estimators_.append(est)
            self.estimators_features_.append(feat_idx)
        return self

    def predict(self, X) -> List[float]:
        X2d = _to_list2d(X)
        n = len(X2d)
        result = [0.0] * n
        for est, feat_idx in zip(self.estimators_, self.estimators_features_):
            X_sub = [[row[j] for j in feat_idx] for row in X2d]
            p = est.predict(X_sub)
            p_list = _to_list1d(p) if not isinstance(p, list) else p
            for i in range(n):
                result[i] += p_list[i] / self.n_estimators
        return result

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        mean_y = sum(y_true) / len(y_true)
        ss_res = sum((y_true[i] - preds[i]) ** 2 for i in range(len(y_true)))
        ss_tot = sum((v - mean_y) ** 2 for v in y_true) + 1e-10
        return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# AdaBoostClassifier (SAMME)
# ---------------------------------------------------------------------------

class AdaBoostClassifier:
    """Adaptive Boosting (SAMME) for classification."""

    def __init__(self, base_estimator=None, n_estimators: int = 50,
                 learning_rate: float = 1.0, random_state: Optional[int] = None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.estimators_: List = []
        self.estimator_weights_: List[float] = []
        self.classes_: Optional[List] = None

    def fit(self, X, y) -> 'AdaBoostClassifier':
        X2d = _to_list2d(X)
        y1d = _to_list1d(y)
        n = len(y1d)
        self.classes_ = _unique_sorted(y1d)
        K = len(self.classes_)
        weights = [1.0 / n] * n
        self.estimators_ = []
        self.estimator_weights_ = []
        for _ in range(self.n_estimators):
            # Weighted sample (bootstrap with weights)
            rng = random.Random(self.random_state)
            cum = []
            s = 0.0
            for w in weights:
                s += w; cum.append(s)
            indices = []
            for _ in range(n):
                r = rng.random()
                for idx, c in enumerate(cum):
                    if r <= c:
                        indices.append(idx); break
                else:
                    indices.append(n - 1)
            X_w = [X2d[i] for i in indices]
            y_w = [y1d[i] for i in indices]
            est = copy.deepcopy(self.base_estimator)
            est.fit(X_w, y_w)
            preds = est.predict(X2d)
            p_list = _to_list1d(preds) if not isinstance(preds, list) else preds
            # Weighted error
            err = sum(weights[i] for i in range(n) if p_list[i] != y1d[i])
            err = max(1e-10, min(err, 1 - 1e-10))
            # SAMME alpha
            alpha = self.learning_rate * (math.log((1 - err) / err) + math.log(K - 1))
            # Update weights
            new_weights = []
            for i in range(n):
                if p_list[i] != y1d[i]:
                    new_weights.append(weights[i] * math.exp(alpha))
                else:
                    new_weights.append(weights[i])
            total = sum(new_weights)
            weights = [w / total for w in new_weights]
            self.estimators_.append(est)
            self.estimator_weights_.append(alpha)
        return self

    def predict(self, X) -> List:
        X2d = _to_list2d(X)
        n = len(X2d)
        K = len(self.classes_)
        scores = [[0.0] * K for _ in range(n)]
        for est, alpha in zip(self.estimators_, self.estimator_weights_):
            preds = est.predict(X2d)
            p_list = _to_list1d(preds) if not isinstance(preds, list) else preds
            for i in range(n):
                try:
                    ci = self.classes_.index(int(p_list[i]) if isinstance(p_list[i], float) else p_list[i])
                    scores[i][ci] += alpha
                except ValueError:
                    pass
        return [self.classes_[_argmax(row)] for row in scores]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        return sum(1 for p, t in zip(preds, y_true) if p == t) / max(len(y_true), 1)


class AdaBoostRegressor:
    """AdaBoost R2 for regression."""

    def __init__(self, base_estimator=None, n_estimators: int = 50,
                 learning_rate: float = 1.0, random_state: Optional[int] = None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.estimators_: List = []
        self.estimator_weights_: List[float] = []

    def fit(self, X, y) -> 'AdaBoostRegressor':
        X2d = _to_list2d(X)
        y1d = _to_list1d(y)
        n = len(y1d)
        weights = [1.0 / n] * n
        self.estimators_ = []
        self.estimator_weights_ = []
        rng = random.Random(self.random_state)
        for _ in range(self.n_estimators):
            # Weighted sample
            cum = []
            s = 0.0
            for w in weights:
                s += w; cum.append(s)
            indices = []
            for _ in range(n):
                r = rng.random()
                for idx, c in enumerate(cum):
                    if r <= c:
                        indices.append(idx); break
                else:
                    indices.append(n - 1)
            X_w = [X2d[i] for i in indices]
            y_w = [y1d[i] for i in indices]
            est = copy.deepcopy(self.base_estimator)
            est.fit(X_w, y_w)
            preds = est.predict(X2d)
            p_list = _to_list1d(preds) if not isinstance(preds, list) else preds
            max_err = max(abs(y1d[i] - p_list[i]) for i in range(n)) + 1e-10
            err_i = [abs(y1d[i] - p_list[i]) / max_err for i in range(n)]
            # Weighted loss
            loss = sum(weights[i] * err_i[i] for i in range(n))
            loss = max(1e-10, min(loss, 0.5 - 1e-10))
            beta = loss / (1 - loss)
            new_weights = [weights[i] * beta ** (1 - err_i[i]) for i in range(n)]
            total = sum(new_weights)
            weights = [w / total for w in new_weights]
            self.estimators_.append(est)
            self.estimator_weights_.append(math.log(1.0 / beta))
        return self

    def predict(self, X) -> List[float]:
        X2d = _to_list2d(X)
        n = len(X2d)
        all_preds = []
        for est in self.estimators_:
            p = est.predict(X2d)
            all_preds.append(_to_list1d(p) if not isinstance(p, list) else p)
        total_w = sum(self.estimator_weights_)
        result = []
        for i in range(n):
            # Weighted median
            vals = [(all_preds[e][i], self.estimator_weights_[e]) for e in range(len(self.estimators_))]
            vals.sort(key=lambda x: x[0])
            cum = 0.0
            median_val = vals[0][0]
            for v, w in vals:
                cum += w
                if cum >= total_w / 2:
                    median_val = v
                    break
            result.append(median_val)
        return result

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        mean_y = sum(y_true) / len(y_true)
        ss_res = sum((y_true[i] - preds[i]) ** 2 for i in range(len(y_true)))
        ss_tot = sum((v - mean_y) ** 2 for v in y_true) + 1e-10
        return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# GradientBoostingRegressor
# ---------------------------------------------------------------------------

class _MeanLeaf:
    """Single-level mean-leaf tree for GBDT residuals."""

    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self._tree = None

    def _fit_node(self, X, r, depth):
        n = len(r)
        if depth == 0 or n <= 2:
            return {'val': sum(r) / n}
        # Find best split
        best_j, best_t, best_loss = 0, 0.0, float('inf')
        d = len(X[0])
        for j in range(d):
            vals = sorted(set(X[i][j] for i in range(n)))
            for t in vals[:-1]:
                L = [r[i] for i in range(n) if X[i][j] <= t]
                R = [r[i] for i in range(n) if X[i][j] > t]
                if not L or not R:
                    continue
                ml = sum(L) / len(L); mr = sum(R) / len(R)
                loss = sum((v - ml) ** 2 for v in L) + sum((v - mr) ** 2 for v in R)
                if loss < best_loss:
                    best_loss = loss; best_j = j; best_t = t
        L_idx = [i for i in range(n) if X[i][best_j] <= best_t]
        R_idx = [i for i in range(n) if X[i][best_j] > best_t]
        if not L_idx or not R_idx:
            return {'val': sum(r) / n}
        return {
            'feat': best_j, 'thresh': best_t,
            'left': self._fit_node([X[i] for i in L_idx], [r[i] for i in L_idx], depth - 1),
            'right': self._fit_node([X[i] for i in R_idx], [r[i] for i in R_idx], depth - 1),
        }

    def fit(self, X, r):
        self._tree = self._fit_node(X, r, self.max_depth)
        return self

    def _predict_one(self, x, node):
        if 'val' in node:
            return node['val']
        if x[node['feat']] <= node['thresh']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        return [self._predict_one(x, self._tree) for x in X]


class GradientBoostingRegressor:
    """Gradient Boosting for regression (squared loss)."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, subsample: float = 1.0,
                 random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        self.estimators_: List = []
        self.init_pred_: float = 0.0

    def fit(self, X, y) -> 'GradientBoostingRegressor':
        X2d = _to_list2d(X)
        y1d = _to_list1d(y)
        n = len(y1d)
        self.init_pred_ = sum(y1d) / n
        F = [self.init_pred_] * n
        rng = random.Random(self.random_state)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            resid = [y1d[i] - F[i] for i in range(n)]
            # Subsample
            if self.subsample < 1.0:
                n_sub = max(1, int(n * self.subsample))
                idx = rng.sample(range(n), n_sub)
                X_sub = [X2d[i] for i in idx]
                r_sub = [resid[i] for i in idx]
            else:
                X_sub, r_sub = X2d, resid
            tree = _MeanLeaf(self.max_depth)
            tree.fit(X_sub, r_sub)
            upd = tree.predict(X2d)
            F = [F[i] + self.learning_rate * upd[i] for i in range(n)]
            self.estimators_.append(tree)
        return self

    def predict(self, X) -> List[float]:
        X2d = _to_list2d(X)
        n = len(X2d)
        F = [self.init_pred_] * n
        for tree in self.estimators_:
            upd = tree.predict(X2d)
            F = [F[i] + self.learning_rate * upd[i] for i in range(n)]
        return F

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        mean_y = sum(y_true) / len(y_true)
        ss_res = sum((y_true[i] - preds[i]) ** 2 for i in range(len(y_true)))
        ss_tot = sum((v - mean_y) ** 2 for v in y_true) + 1e-10
        return 1.0 - ss_res / ss_tot


class GradientBoostingClassifier:
    """Gradient Boosting for classification (one-vs-rest log-loss)."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, subsample: float = 1.0,
                 random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        self.regressors_: List = []  # one per class
        self.classes_: Optional[List] = None
        self.init_preds_: List[float] = []

    def fit(self, X, y) -> 'GradientBoostingClassifier':
        X2d = _to_list2d(X)
        y1d = _to_list1d(y)
        self.classes_ = _unique_sorted(y1d)
        K = len(self.classes_)
        n = len(y1d)
        # one-vs-rest
        self.regressors_ = []
        self.init_preds_ = []
        for c in self.classes_:
            y_bin = [1.0 if v == c else 0.0 for v in y1d]
            mean_y = sum(y_bin) / n
            self.init_preds_.append(mean_y)
            F = [mean_y] * n
            trees = []
            rng = random.Random(self.random_state)
            for _ in range(self.n_estimators):
                # sigmoid derivative = p*(1-p), residual = y - p
                p = [1.0 / (1.0 + math.exp(-f)) for f in F]
                resid = [y_bin[i] - p[i] for i in range(n)]
                if self.subsample < 1.0:
                    n_sub = max(1, int(n * self.subsample))
                    idx = rng.sample(range(n), n_sub)
                    X_sub = [X2d[i] for i in idx]
                    r_sub = [resid[i] for i in idx]
                else:
                    X_sub, r_sub = X2d, resid
                tree = _MeanLeaf(self.max_depth)
                tree.fit(X_sub, r_sub)
                upd = tree.predict(X2d)
                F = [F[i] + self.learning_rate * upd[i] for i in range(n)]
                trees.append(tree)
            self.regressors_.append(trees)
        return self

    def predict_proba(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        n = len(X2d)
        K = len(self.classes_)
        scores = [[self.init_preds_[k]] * n for k in range(K)]
        for k, trees in enumerate(self.regressors_):
            for tree in trees:
                upd = tree.predict(X2d)
                for i in range(n):
                    scores[k][i] += self.learning_rate * upd[i]
        # Softmax
        result = []
        for i in range(n):
            row = [scores[k][i] for k in range(K)]
            m = max(row)
            exps = [math.exp(v - m) for v in row]
            s = sum(exps)
            result.append([e / s for e in exps])
        return result

    def predict(self, X) -> List:
        proba = self.predict_proba(X)
        return [self.classes_[_argmax(row)] for row in proba]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        return sum(1 for p, t in zip(preds, y_true) if p == t) / max(len(y_true), 1)


# ---------------------------------------------------------------------------
# StackingClassifier / StackingRegressor
# ---------------------------------------------------------------------------

class StackingClassifier:
    """Stacked generalization for classification."""

    def __init__(self, estimators: List, final_estimator=None,
                 cv: int = 5, passthrough: bool = False):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.passthrough = passthrough
        self.estimators_: List = []
        self.final_estimator_: Any = None
        self.classes_: Optional[List] = None

    def fit(self, X, y) -> 'StackingClassifier':
        X2d = _to_list2d(X)
        y1d = _to_list1d(y)
        n = len(y1d)
        self.classes_ = _unique_sorted(y1d)
        K = len(self.classes_)
        # Build OOF meta-features
        n_base = len(self.estimators)
        meta_X = [[0.0] * (n_base * K) for _ in range(n)]
        fold_size = n // self.cv
        for fold_i in range(self.cv):
            val_start = fold_i * fold_size
            val_end = val_start + fold_size if fold_i < self.cv - 1 else n
            val_idx = list(range(val_start, val_end))
            tr_idx = [i for i in range(n) if i not in set(val_idx)]
            X_tr = [X2d[i] for i in tr_idx]
            y_tr = [y1d[i] for i in tr_idx]
            X_val = [X2d[i] for i in val_idx]
            for e_idx, (name, est) in enumerate(self.estimators):
                cloned = copy.deepcopy(est)
                cloned.fit(X_tr, y_tr)
                if hasattr(cloned, 'predict_proba'):
                    proba = cloned.predict_proba(X_val)
                    p_flat = _to_list2d(proba) if not isinstance(proba, list) else proba
                    for i_v, orig_i in enumerate(val_idx):
                        for k in range(K):
                            meta_X[orig_i][e_idx * K + k] = p_flat[i_v][k] if k < len(p_flat[i_v]) else 0.0
                else:
                    preds = cloned.predict(X_val)
                    p_list = _to_list1d(preds) if not isinstance(preds, list) else preds
                    for i_v, orig_i in enumerate(val_idx):
                        ci = self.classes_.index(int(p_list[i_v])) if int(p_list[i_v]) in self.classes_ else 0
                        meta_X[orig_i][e_idx * K + ci] += 1.0
        if self.passthrough:
            meta_X = [meta_X[i] + X2d[i] for i in range(n)]
        # Train final estimator on meta features
        from vectorphile.ml import LogisticRegression as _LR
        self.final_estimator_ = copy.deepcopy(self.final_estimator or _LR())
        self.final_estimator_.fit(meta_X, y1d)
        # Refit base estimators on full data
        self.estimators_ = []
        for name, est in self.estimators:
            cloned = copy.deepcopy(est)
            cloned.fit(X2d, y1d)
            self.estimators_.append((name, cloned))
        return self

    def _make_meta(self, X2d: List[List[float]]) -> List[List[float]]:
        n = len(X2d)
        K = len(self.classes_)
        meta = [[0.0] * (len(self.estimators_) * K) for _ in range(n)]
        for e_idx, (name, est) in enumerate(self.estimators_):
            if hasattr(est, 'predict_proba'):
                proba = est.predict_proba(X2d)
                p_flat = _to_list2d(proba) if not isinstance(proba, list) else proba
                for i in range(n):
                    for k in range(K):
                        meta[i][e_idx * K + k] = p_flat[i][k] if k < len(p_flat[i]) else 0.0
            else:
                preds = est.predict(X2d)
                p_list = _to_list1d(preds) if not isinstance(preds, list) else preds
                for i in range(n):
                    ci = self.classes_.index(int(p_list[i])) if int(p_list[i]) in self.classes_ else 0
                    meta[i][e_idx * K + ci] += 1.0
        if self.passthrough:
            meta = [meta[i] + X2d[i] for i in range(n)]
        return meta

    def predict(self, X) -> List:
        return self.final_estimator_.predict(self._make_meta(_to_list2d(X)))

    def predict_proba(self, X) -> List[List[float]]:
        meta = self._make_meta(_to_list2d(X))
        if hasattr(self.final_estimator_, 'predict_proba'):
            return self.final_estimator_.predict_proba(meta)
        preds = self.final_estimator_.predict(meta)
        p_list = _to_list1d(preds) if not isinstance(preds, list) else preds
        K = len(self.classes_)
        result = [[0.0] * K for _ in range(len(p_list))]
        for i, v in enumerate(p_list):
            if int(v) in self.classes_:
                result[i][self.classes_.index(int(v))] = 1.0
        return result

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        return sum(1 for p, t in zip(preds, y_true) if p == t) / max(len(y_true), 1)


class StackingRegressor:
    """Stacked generalization for regression."""

    def __init__(self, estimators: List, final_estimator=None,
                 cv: int = 5, passthrough: bool = False):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.passthrough = passthrough
        self.estimators_: List = []
        self.final_estimator_: Any = None

    def fit(self, X, y) -> 'StackingRegressor':
        X2d = _to_list2d(X)
        y1d = _to_list1d(y)
        n = len(y1d)
        n_base = len(self.estimators)
        meta_X = [[0.0] * n_base for _ in range(n)]
        fold_size = n // self.cv
        for fold_i in range(self.cv):
            val_start = fold_i * fold_size
            val_end = val_start + fold_size if fold_i < self.cv - 1 else n
            val_idx = list(range(val_start, val_end))
            tr_idx = [i for i in range(n) if i not in set(val_idx)]
            X_tr = [X2d[i] for i in tr_idx]
            y_tr = [y1d[i] for i in tr_idx]
            X_val = [X2d[i] for i in val_idx]
            for e_idx, (name, est) in enumerate(self.estimators):
                cloned = copy.deepcopy(est)
                cloned.fit(X_tr, y_tr)
                preds = cloned.predict(X_val)
                p_list = _to_list1d(preds) if not isinstance(preds, list) else preds
                for i_v, orig_i in enumerate(val_idx):
                    meta_X[orig_i][e_idx] = p_list[i_v]
        if self.passthrough:
            meta_X = [meta_X[i] + X2d[i] for i in range(n)]
        from vectorphile.ml import LinearRegression as _LReg
        self.final_estimator_ = copy.deepcopy(self.final_estimator or _LReg())
        self.final_estimator_.fit(meta_X, y1d)
        self.estimators_ = []
        for name, est in self.estimators:
            cloned = copy.deepcopy(est)
            cloned.fit(X2d, y1d)
            self.estimators_.append((name, cloned))
        return self

    def predict(self, X) -> List[float]:
        X2d = _to_list2d(X)
        n = len(X2d)
        n_base = len(self.estimators_)
        meta = [[0.0] * n_base for _ in range(n)]
        for e_idx, (name, est) in enumerate(self.estimators_):
            preds = est.predict(X2d)
            p_list = _to_list1d(preds) if not isinstance(preds, list) else preds
            for i in range(n):
                meta[i][e_idx] = p_list[i]
        if self.passthrough:
            meta = [meta[i] + X2d[i] for i in range(n)]
        result = self.final_estimator_.predict(meta)
        return _to_list1d(result) if not isinstance(result, list) else result

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        mean_y = sum(y_true) / len(y_true)
        ss_res = sum((y_true[i] - preds[i]) ** 2 for i in range(len(y_true)))
        ss_tot = sum((v - mean_y) ** 2 for v in y_true) + 1e-10
        return 1.0 - ss_res / ss_tot


class _ExtraTreeClassifier:
    """Single extra-randomized tree for classification."""

    def __init__(self, max_depth=None, min_samples_split=2, max_features='sqrt',
                 random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self._tree = None
        self._classes = []
        self._rng = random.Random(random_state)

    def fit(self, X, y):
        X2d = _to_list2d(X)
        y_list = _to_list1d(y)
        self._classes = sorted(set(y_list))
        p = len(X2d[0]) if X2d else 0
        if self.max_features == 'sqrt':
            n_feat = max(1, int(p ** 0.5))
        elif self.max_features == 'log2':
            n_feat = max(1, int(math.log2(p + 1)))
        else:
            n_feat = p
        self._n_feat = n_feat
        self._tree = self._build(X2d, y_list, 0)
        return self

    def _gini(self, y_list):
        n = len(y_list)
        if n == 0:
            return 0.0
        counts: dict = {}
        for lbl in y_list:
            counts[lbl] = counts.get(lbl, 0) + 1
        return 1.0 - sum((c / n) ** 2 for c in counts.values())

    def _build(self, X, y, depth):
        if (not X or len(set(y)) == 1 or
                (self.max_depth is not None and depth >= self.max_depth) or
                len(X) < self.min_samples_split):
            counts: dict = {}
            for lbl in y:
                counts[lbl] = counts.get(lbl, 0) + 1
            return max(counts, key=counts.get)
        p = len(X[0])
        feat_idxs = self._rng.sample(range(p), min(self._n_feat, p))
        best_feat, best_thr, best_score = 0, 0.0, float('inf')
        for fi in feat_idxs:
            vals = [X[i][fi] for i in range(len(X))]
            mn, mx = min(vals), max(vals)
            if mn == mx:
                continue
            thr = self._rng.uniform(mn, mx)
            left_y = [y[i] for i in range(len(X)) if X[i][fi] <= thr]
            right_y = [y[i] for i in range(len(X)) if X[i][fi] > thr]
            if not left_y or not right_y:
                continue
            score = (len(left_y) * self._gini(left_y) + len(right_y) * self._gini(right_y)) / len(y)
            if score < best_score:
                best_score, best_feat, best_thr = score, fi, thr
        left_X = [X[i] for i in range(len(X)) if X[i][best_feat] <= best_thr]
        left_y = [y[i] for i in range(len(X)) if X[i][best_feat] <= best_thr]
        right_X = [X[i] for i in range(len(X)) if X[i][best_feat] > best_thr]
        right_y = [y[i] for i in range(len(X)) if X[i][best_feat] > best_thr]
        if not left_X or not right_X:
            counts: dict = {}
            for lbl in y:
                counts[lbl] = counts.get(lbl, 0) + 1
            return max(counts, key=counts.get)
        return (best_feat, best_thr,
                self._build(left_X, left_y, depth + 1),
                self._build(right_X, right_y, depth + 1))

    def _predict_one(self, node, row):
        if not isinstance(node, tuple):
            return node
        fi, thr, left, right = node
        return self._predict_one(left if row[fi] <= thr else right, row)

    def predict(self, X) -> List:
        return [self._predict_one(self._tree, row) for row in _to_list2d(X)]

    def predict_proba(self, X) -> List[List[float]]:
        preds = self.predict(X)
        result = []
        for p in preds:
            row = [1.0 if c == p else 0.0 for c in self._classes]
            result.append(row)
        return result

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        return sum(1 for a, b in zip(preds, y_true) if a == b) / max(len(y_true), 1)


class _ExtraTreeRegressor:
    """Single extra-randomized tree for regression."""

    def __init__(self, max_depth=None, min_samples_split=2, max_features='auto',
                 random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self._tree = None
        self._rng = random.Random(random_state)

    def fit(self, X, y):
        X2d = _to_list2d(X)
        y_list = [float(v) for v in _to_list1d(y)]
        p = len(X2d[0]) if X2d else 0
        if self.max_features in ('auto', 'sqrt'):
            n_feat = max(1, int(p ** 0.5))
        elif self.max_features == 'log2':
            n_feat = max(1, int(math.log2(p + 1)))
        else:
            n_feat = p
        self._n_feat = n_feat
        self._tree = self._build(X2d, y_list, 0)
        return self

    def _mse(self, y_list):
        if not y_list:
            return 0.0
        mean = sum(y_list) / len(y_list)
        return sum((v - mean) ** 2 for v in y_list) / len(y_list)

    def _build(self, X, y, depth):
        if (not X or len(set(y)) == 0 or
                (self.max_depth is not None and depth >= self.max_depth) or
                len(X) < self.min_samples_split):
            return sum(y) / len(y) if y else 0.0
        p = len(X[0])
        feat_idxs = self._rng.sample(range(p), min(self._n_feat, p))
        best_feat, best_thr, best_score = 0, 0.0, float('inf')
        for fi in feat_idxs:
            vals = [X[i][fi] for i in range(len(X))]
            mn, mx = min(vals), max(vals)
            if mn == mx:
                continue
            thr = self._rng.uniform(mn, mx)
            left_y = [y[i] for i in range(len(X)) if X[i][fi] <= thr]
            right_y = [y[i] for i in range(len(X)) if X[i][fi] > thr]
            if not left_y or not right_y:
                continue
            score = (len(left_y) * self._mse(left_y) + len(right_y) * self._mse(right_y)) / len(y)
            if score < best_score:
                best_score, best_feat, best_thr = score, fi, thr
        left_X = [X[i] for i in range(len(X)) if X[i][best_feat] <= best_thr]
        left_y = [y[i] for i in range(len(X)) if X[i][best_feat] <= best_thr]
        right_X = [X[i] for i in range(len(X)) if X[i][best_feat] > best_thr]
        right_y = [y[i] for i in range(len(X)) if X[i][best_feat] > best_thr]
        if not left_X or not right_X:
            return sum(y) / len(y)
        return (best_feat, best_thr,
                self._build(left_X, left_y, depth + 1),
                self._build(right_X, right_y, depth + 1))

    def _predict_one(self, node, row):
        if not isinstance(node, tuple):
            return node
        fi, thr, left, right = node
        return self._predict_one(left if row[fi] <= thr else right, row)

    def predict(self, X) -> List[float]:
        return [self._predict_one(self._tree, row) for row in _to_list2d(X)]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = [float(v) for v in _to_list1d(y)]
        mean_y = sum(y_true) / len(y_true)
        ss_res = sum((y_true[i] - preds[i]) ** 2 for i in range(len(y_true)))
        ss_tot = sum((v - mean_y) ** 2 for v in y_true) + 1e-10
        return 1.0 - ss_res / ss_tot


class ExtraTreesClassifier:
    """Extremely randomized trees for classification."""

    def __init__(self, n_estimators: int = 100, max_depth=None, min_samples_split: int = 2,
                 max_features='sqrt', bootstrap: bool = False, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.estimators_: List[_ExtraTreeClassifier] = []
        self.classes_: List = []

    def fit(self, X, y):
        X2d = _to_list2d(X)
        y_list = _to_list1d(y)
        self.classes_ = sorted(set(y_list))
        n = len(X2d)
        rng = random.Random(self.random_state)
        self.estimators_ = []
        for i in range(self.n_estimators):
            tree = _ExtraTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=rng.randint(0, 2**31))
            if self.bootstrap:
                idxs = [rng.randint(0, n - 1) for _ in range(n)]
                tree.fit([X2d[j] for j in idxs], [y_list[j] for j in idxs])
            else:
                tree.fit(X2d, y_list)
            self.estimators_.append(tree)
        return self

    def predict(self, X) -> List:
        X2d = _to_list2d(X)
        votes = [est.predict(X2d) for est in self.estimators_]
        result = []
        for i in range(len(X2d)):
            counts: dict = {}
            for v in votes:
                lbl = v[i]
                counts[lbl] = counts.get(lbl, 0) + 1
            result.append(max(counts, key=counts.get))
        return result

    def predict_proba(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        probs = [[0.0] * len(self.classes_) for _ in range(len(X2d))]
        for est in self.estimators_:
            p = est.predict(X2d)
            for i, lbl in enumerate(p):
                if lbl in self.classes_:
                    probs[i][self.classes_.index(lbl)] += 1.0
        n_est = len(self.estimators_)
        return [[v / n_est for v in row] for row in probs]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        return sum(1 for a, b in zip(preds, y_true) if a == b) / max(len(y_true), 1)


class ExtraTreesRegressor:
    """Extremely randomized trees for regression."""

    def __init__(self, n_estimators: int = 100, max_depth=None, min_samples_split: int = 2,
                 max_features='auto', bootstrap: bool = False, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.estimators_: List[_ExtraTreeRegressor] = []

    def fit(self, X, y):
        X2d = _to_list2d(X)
        y_list = [float(v) for v in _to_list1d(y)]
        n = len(X2d)
        rng = random.Random(self.random_state)
        self.estimators_ = []
        for i in range(self.n_estimators):
            tree = _ExtraTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=rng.randint(0, 2**31))
            if self.bootstrap:
                idxs = [rng.randint(0, n - 1) for _ in range(n)]
                tree.fit([X2d[j] for j in idxs], [y_list[j] for j in idxs])
            else:
                tree.fit(X2d, y_list)
            self.estimators_.append(tree)
        return self

    def predict(self, X) -> List[float]:
        X2d = _to_list2d(X)
        preds = [est.predict(X2d) for est in self.estimators_]
        return [sum(preds[k][i] for k in range(len(preds))) / len(preds)
                for i in range(len(X2d))]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = [float(v) for v in _to_list1d(y)]
        mean_y = sum(y_true) / len(y_true)
        ss_res = sum((y_true[i] - preds[i]) ** 2 for i in range(len(y_true)))
        ss_tot = sum((v - mean_y) ** 2 for v in y_true) + 1e-10
        return 1.0 - ss_res / ss_tot


__all__ = [
    'VotingClassifier', 'VotingRegressor',
    'BaggingClassifier', 'BaggingRegressor',
    'AdaBoostClassifier', 'AdaBoostRegressor',
    'GradientBoostingClassifier', 'GradientBoostingRegressor',
    'StackingClassifier', 'StackingRegressor',
    'ExtraTreesClassifier', 'ExtraTreesRegressor',
]
