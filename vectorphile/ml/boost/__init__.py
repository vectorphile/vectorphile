"""XGBoost-style gradient boosting (pure NumPy)."""
import numpy as np
from typing import Optional
from vectorphile.ml.tree import DecisionTreeRegressor


class XGBRegressor:
    """Extreme Gradient Boosting Regressor with second-order statistics."""
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6,
                 min_child_weight=1, gamma=0.0, reg_alpha=0.0, reg_lambda=1.0,
                 subsample=1.0, colsample_bytree=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.estimators_ = []
        self.base_score_: float = 0.0

    def fit(self, X, y, eval_set=None, verbose=False) -> 'XGBRegressor':
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X, y = np.array(X, dtype=float), np.array(y, dtype=float)
        self.base_score_ = y.mean()
        F = np.full(len(y), self.base_score_)
        n, p = X.shape
        for i in range(self.n_estimators):
            # MSE gradient = -2*(y-F), hessian = 2 (constant for MSE)
            g = -(y - F)  # negative gradient
            h = np.ones_like(g)
            # Subsample rows
            if self.subsample < 1.0:
                row_idx = np.random.choice(n, int(n * self.subsample), replace=False)
                Xs, gs, hs = X[row_idx], g[row_idx], h[row_idx]
            else:
                Xs, gs, hs = X, g, h
            # Subsample cols
            if self.colsample_bytree < 1.0:
                col_idx = np.random.choice(p, int(p * self.colsample_bytree), replace=False)
                Xs = Xs[:, col_idx]
            # Weight: -g/h for leaf values
            w = -gs / (hs + self.reg_lambda)
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(Xs, w)
            update = tree.predict(X[:, col_idx] if self.colsample_bytree < 1.0 else X)
            F += self.learning_rate * update
            self.estimators_.append((tree, col_idx if self.colsample_bytree < 1.0 else None))
            if verbose and (i + 1) % 10 == 0:
                loss = np.mean((y - F)**2)
                print(f'[{i+1}] train-rmse={np.sqrt(loss):.4f}')
        return self

    def predict(self, X) -> np.ndarray:
        X = np.array(X, dtype=float)
        F = np.full(len(X), self.base_score_)
        for tree, col_idx in self.estimators_:
            Xs = X[:, col_idx] if col_idx is not None else X
            F += self.learning_rate * tree.predict(Xs)
        return F

    def score(self, X, y) -> float:
        y = np.array(y, dtype=float)
        pred = self.predict(X)
        ss_res = np.sum((y - pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return float(1 - ss_res / (ss_tot + 1e-12))


class XGBClassifier:
    """Extreme Gradient Boosting Classifier."""
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6,
                 reg_lambda=1.0, subsample=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.random_state = random_state
        self.classes_: Optional[np.ndarray] = None
        self.estimators_ = []
        self.base_score_: Optional[np.ndarray] = None

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def fit(self, X, y) -> 'XGBClassifier':
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X, y = np.array(X, dtype=float), np.array(y)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        n = len(X)
        y_ohe = np.zeros((n, K))
        for i, c in enumerate(self.classes_):
            y_ohe[:, i] = (y == c).astype(float)
        self.base_score_ = np.log(y_ohe.mean(axis=0) + 1e-12)
        F = np.tile(self.base_score_, (n, 1)).copy()
        self.estimators_ = []
        for _ in range(self.n_estimators):
            probs = self._sigmoid(F)
            probs /= probs.sum(axis=1, keepdims=True)
            group = []
            for k in range(K):
                g = -(y_ohe[:, k] - probs[:, k])
                h = probs[:, k] * (1 - probs[:, k]) + 1e-6
                w = -g / (h + self.reg_lambda)
                tree = DecisionTreeRegressor(max_depth=self.max_depth)
                tree.fit(X, w)
                F[:, k] += self.learning_rate * tree.predict(X)
                group.append(tree)
            self.estimators_.append(group)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.array(X, dtype=float)
        F = np.tile(self.base_score_, (len(X), 1)).copy()
        for group in self.estimators_:
            for k, tree in enumerate(group):
                F[:, k] += self.learning_rate * tree.predict(X)
        probs = self._sigmoid(F)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.array(y)))
