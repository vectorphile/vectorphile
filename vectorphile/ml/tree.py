"""Decision tree classifiers and regressors — pure Python."""
import math
import random as _random
from typing import Optional


def _col(X, j):
    return [X[i][j] for i in range(len(X))]


def _uniq_sorted(seq):
    return sorted(set(seq))


class _Node:
    __slots__ = ('feat', 'thresh', 'left', 'right', 'value', 'proba')

    def __init__(self):
        self.feat = None
        self.thresh = None
        self.left = None
        self.right = None
        self.value = None
        self.proba = None


class _BaseTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def _n_features(self, p):
        if self.max_features is None:
            return p
        if self.max_features == 'sqrt':
            return max(1, int(math.sqrt(p)))
        if self.max_features == 'log2':
            return max(1, int(math.log2(p)))
        if isinstance(self.max_features, int):
            return min(self.max_features, p)
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * p))
        return p

    def _best_split(self, X, y, feat_idxs):
        best_feat, best_thresh, best_score = None, None, float('inf')
        for feat in feat_idxs:
            vals = sorted(set(_col(X, feat)))
            thresholds = [(vals[i] + vals[i+1]) / 2 for i in range(len(vals) - 1)]
            for thresh in thresholds:
                left_mask = [X[i][feat] <= thresh for i in range(len(X))]
                right_mask = [not m for m in left_mask]
                n_left = sum(left_mask)
                n_right = sum(right_mask)
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                y_left = [y[i] for i in range(len(y)) if left_mask[i]]
                y_right = [y[i] for i in range(len(y)) if right_mask[i]]
                score = (len(y_left) * self._impurity(y_left) +
                         len(y_right) * self._impurity(y_right)) / len(y)
                if score < best_score:
                    best_score, best_feat, best_thresh = score, feat, thresh
        return best_feat, best_thresh

    def _build(self, X, y, depth):
        node = _Node()
        n, p = len(X), len(X[0])
        if (len(y) < self.min_samples_split or
                (self.max_depth is not None and depth >= self.max_depth) or
                len(_uniq_sorted(y)) == 1):
            node.value = self._leaf_value(y)
            node.proba = self._leaf_proba(y)
            return node
        nf = self._n_features(p)
        rng = _random.Random(self.random_state)
        feat_idxs = rng.sample(range(p), nf) if nf < p else list(range(p))
        feat, thresh = self._best_split(X, y, feat_idxs)
        if feat is None:
            node.value = self._leaf_value(y)
            node.proba = self._leaf_proba(y)
            return node
        node.feat, node.thresh = feat, thresh
        left_mask = [X[i][feat] <= thresh for i in range(n)]
        X_left = [X[i] for i in range(n) if left_mask[i]]
        y_left = [y[i] for i in range(n) if left_mask[i]]
        X_right = [X[i] for i in range(n) if not left_mask[i]]
        y_right = [y[i] for i in range(n) if not left_mask[i]]
        node.left = self._build(X_left, y_left, depth + 1)
        node.right = self._build(X_right, y_right, depth + 1)
        return node

    def fit(self, X, y):
        X2 = [[float(v) for v in row] for row in X]
        y2 = list(y)
        self.n_features_in_ = len(X2[0]) if X2 else 0
        self.root_ = self._build(X2, y2, 0)
        return self

    def _predict_one(self, x, node):
        if node.feat is None:
            return node.value
        if x[node.feat] <= node.thresh:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def _predict_proba_one(self, x, node):
        if node.feat is None:
            return node.proba
        if x[node.feat] <= node.thresh:
            return self._predict_proba_one(x, node.left)
        return self._predict_proba_one(x, node.right)

    def predict(self, X):
        X2 = [[float(v) for v in row] for row in X]
        return [self._predict_one(row, self.root_) for row in X2]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(a == b for a, b in zip(preds, y)) / len(y)


class DecisionTreeClassifier(_BaseTree):
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, random_state=None):
        super().__init__(max_depth, min_samples_split, min_samples_leaf,
                         max_features, random_state)
        self.criterion = criterion

    def _impurity(self, y):
        n = len(y)
        if n == 0:
            return 0.0
        counts = {}
        for v in y:
            counts[v] = counts.get(v, 0) + 1
        if self.criterion == 'gini':
            return 1.0 - sum((c / n) ** 2 for c in counts.values())
        # entropy
        return -sum((c / n) * math.log(c / n + 1e-15) for c in counts.values())

    def _leaf_value(self, y):
        counts = {}
        for v in y:
            counts[v] = counts.get(v, 0) + 1
        return max(counts, key=counts.get)

    def _leaf_proba(self, y):
        n = len(y)
        counts = {}
        for v in y:
            counts[v] = counts.get(v, 0) + 1
        return {c: counts.get(c, 0) / n for c in self.classes_}

    def fit(self, X, y):
        self.classes_ = _uniq_sorted(y)
        return super().fit(X, y)

    def predict_proba(self, X):
        X2 = [[float(v) for v in row] for row in X]
        result = []
        for row in X2:
            proba_dict = self._predict_proba_one(row, self.root_)
            result.append([proba_dict[c] for c in self.classes_])
        return result

    def score(self, X, y):
        preds = self.predict(X)
        return sum(a == b for a, b in zip(preds, y)) / len(y)


class DecisionTreeRegressor(_BaseTree):
    def __init__(self, criterion='mse', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, random_state=None):
        super().__init__(max_depth, min_samples_split, min_samples_leaf,
                         max_features, random_state)
        self.criterion = criterion

    def _impurity(self, y):
        n = len(y)
        if n == 0:
            return 0.0
        mu = sum(y) / n
        return sum((v - mu) ** 2 for v in y) / n

    def _leaf_value(self, y):
        return sum(y) / len(y) if y else 0.0

    def _leaf_proba(self, y):
        return self._leaf_value(y)

    def score(self, X, y):
        preds = self.predict(X)
        y2 = [float(v) for v in y]
        mu = sum(y2) / len(y2)
        ss_res = sum((a - b) ** 2 for a, b in zip(y2, preds))
        ss_tot = sum((v - mu) ** 2 for v in y2)
        return float(1 - ss_res / (ss_tot + 1e-12))
