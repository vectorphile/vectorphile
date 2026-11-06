"""Ensemble methods: Random Forest, Gradient Boosting."""
import numpy as np
from typing import Optional
from vectorphile.ml.tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.estimators_ = []
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X, y) -> 'RandomForestClassifier':
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X, y = np.array(X, dtype=float), np.array(y)
        self.classes_ = np.unique(y)
        n = len(X)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                idx = np.random.choice(n, n, replace=True)
                Xb, yb = X[idx], y[idx]
            else:
                Xb, yb = X, y
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
            )
            tree.fit(Xb, yb)
            self.estimators_.append(tree)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.array(X, dtype=float)
        all_proba = np.array([t.predict_proba(X) for t in self.estimators_])
        return all_proba.mean(axis=0)

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.array(y)))


class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X, y) -> 'RandomForestRegressor':
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        n = len(X)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                idx = np.random.choice(n, n, replace=True)
                Xb, yb = X[idx], y[idx]
            else:
                Xb, yb = X, y
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
            )
            tree.fit(Xb, yb)
            self.estimators_.append(tree)
        return self

    def predict(self, X) -> np.ndarray:
        X = np.array(X, dtype=float)
        preds = np.array([t.predict(X) for t in self.estimators_])
        return preds.mean(axis=0)

    def score(self, X, y) -> float:
        y = np.array(y, dtype=float)
        pred = self.predict(X)
        ss_res = np.sum((y - pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return float(1 - ss_res / (ss_tot + 1e-12))


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, subsample=1.0, loss='mse', random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.loss = loss
        self.random_state = random_state
        self.estimators_ = []
        self.init_pred_: float = 0.0

    def fit(self, X, y) -> 'GradientBoostingRegressor':
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        self.init_pred_ = float(y.mean())
        F = np.full(len(y), self.init_pred_)
        self.estimators_ = []
        n = len(X)
        for _ in range(self.n_estimators):
            residuals = y - F
            if self.subsample < 1.0:
                idx = np.random.choice(n, int(n * self.subsample), replace=False)
                Xs, rs = X[idx], residuals[idx]
            else:
                Xs, rs = X, residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(Xs, rs)
            update = tree.predict(X)
            F += self.learning_rate * update
            self.estimators_.append(tree)
        return self

    def predict(self, X) -> np.ndarray:
        X = np.array(X, dtype=float)
        F = np.full(len(X), self.init_pred_)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
        return F

    def score(self, X, y) -> float:
        y = np.array(y, dtype=float)
        pred = self.predict(X)
        ss_res = np.sum((y - pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return float(1 - ss_res / (ss_tot + 1e-12))


class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, subsample=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state
        self.estimators_ = []
        self.classes_: Optional[np.ndarray] = None
        self.init_pred_: Optional[np.ndarray] = None

    def _softmax(self, z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X, y) -> 'GradientBoostingClassifier':
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X, y = np.array(X, dtype=float), np.array(y)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        n = len(X)
        y_ohe = np.zeros((n, K))
        for i, c in enumerate(self.classes_):
            y_ohe[:, i] = (y == c).astype(float)
        F = np.zeros((n, K))
        self.init_pred_ = np.log(y_ohe.mean(axis=0) + 1e-12)
        F += self.init_pred_
        self.estimators_ = []
        for _ in range(self.n_estimators):
            probs = self._softmax(F)
            self.estimators_.append([])
            for k in range(K):
                residuals = y_ohe[:, k] - probs[:, k]
                tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
                tree.fit(X, residuals)
                F[:, k] += self.learning_rate * tree.predict(X)
                self.estimators_[-1].append(tree)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.array(X, dtype=float)
        F = np.tile(self.init_pred_, (len(X), 1)).copy()
        for group in self.estimators_:
            for k, tree in enumerate(group):
                F[:, k] += self.learning_rate * tree.predict(X)
        return self._softmax(F)

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.array(y)))
