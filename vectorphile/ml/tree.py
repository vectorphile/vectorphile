"""Decision Tree classifiers and regressors."""
import numpy as np
from typing import Optional


class _Node:
    __slots__ = ['feature', 'threshold', 'left', 'right', 'value', 'impurity', 'n_samples']
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.impurity = 0.0
        self.n_samples = 0


class _BaseTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.root_: Optional[_Node] = None
        self.n_features_: int = 0

    def _impurity(self, y):
        raise NotImplementedError

    def _leaf_value(self, y):
        raise NotImplementedError

    def _best_split(self, X, y):
        n, p = X.shape
        best_gain, best_feat, best_thresh = -np.inf, None, None
        parent_imp = self._impurity(y)
        n_feats = p if self.max_features is None else min(p, int(np.ceil(
            self.max_features * p if isinstance(self.max_features, float)
            else (np.sqrt(p) if self.max_features == 'sqrt' else np.log2(p)))))
        feature_idxs = np.random.choice(p, n_feats, replace=False)
        for feat in feature_idxs:
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds[:-1]:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue
                gain = parent_imp - (left_mask.sum()/n * self._impurity(y[left_mask]) +
                                     right_mask.sum()/n * self._impurity(y[right_mask]))
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh
        return best_feat, best_thresh

    def _build(self, X, y, depth):
        node = _Node()
        node.n_samples = len(y)
        node.value = self._leaf_value(y)
        node.impurity = self._impurity(y)
        if (self.max_depth is not None and depth >= self.max_depth) or len(y) < self.min_samples_split:
            return node
        feat, thresh = self._best_split(X, y)
        if feat is None:
            return node
        node.feature = feat
        node.threshold = thresh
        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)
        return node

    def fit(self, X, y) -> '_BaseTree':
        X, y = np.array(X, dtype=float), np.array(y)
        self.n_features_ = X.shape[1]
        self.root_ = self._build(X, y, 0)
        return self

    def _predict_one(self, x, node):
        if node.left is None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X) -> np.ndarray:
        X = np.array(X, dtype=float)
        return np.array([self._predict_one(x, self.root_) for x in X])

    def score(self, X, y) -> float:
        raise NotImplementedError


class DecisionTreeClassifier(_BaseTree):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, criterion='gini'):
        super().__init__(max_depth, min_samples_split, min_samples_leaf, max_features, criterion)
        self.classes_: Optional[np.ndarray] = None

    def _impurity(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        if self.criterion == 'gini':
            return 1.0 - np.sum(probs**2)
        return -np.sum(probs * np.log2(probs + 1e-12))

    def _leaf_value(self, y):
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

    def fit(self, X, y) -> 'DecisionTreeClassifier':
        self.classes_ = np.unique(y)
        super().fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        def _proba_one(x, node):
            if node.left is None:
                vals = np.zeros(len(self.classes_))
                for i, c in enumerate(self.classes_):
                    if c == node.value:
                        vals[i] = 1.0
                return vals
            if x[node.feature] <= node.threshold:
                return _proba_one(x, node.left)
            return _proba_one(x, node.right)
        X = np.array(X, dtype=float)
        return np.array([_proba_one(x, self.root_) for x in X])

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.array(y)))


class DecisionTreeRegressor(_BaseTree):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, criterion='mse'):
        super().__init__(max_depth, min_samples_split, min_samples_leaf, max_features, criterion)

    def _impurity(self, y):
        if len(y) == 0:
            return 0.0
        return float(np.var(y))

    def _leaf_value(self, y):
        return float(np.mean(y))

    def score(self, X, y) -> float:
        y = np.array(y, dtype=float)
        pred = self.predict(X)
        ss_res = np.sum((y - pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return float(1 - ss_res / (ss_tot + 1e-12))
