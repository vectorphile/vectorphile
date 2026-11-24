"""Random Forest and Gradient Boosting — pure Python."""
import math
import random as _random
from typing import Optional

from vectorphile.ml.tree import DecisionTreeClassifier, DecisionTreeRegressor


def _uniq_sorted(seq):
    return sorted(set(seq))


class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', random_state=None,
                 bootstrap=True, n_jobs=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.bootstrap = bootstrap

    def fit(self, X, y):
        X2 = [[float(v) for v in row] for row in X]
        y2 = list(y)
        self.classes_ = _uniq_sorted(y2)
        n = len(X2)
        rng = _random.Random(self.random_state)
        self.estimators_ = []
        for i in range(self.n_estimators):
            seed = rng.randint(0, 2**31)
            if self.bootstrap:
                idxs = [rng.randint(0, n - 1) for _ in range(n)]
            else:
                idxs = list(range(n))
            Xb = [X2[j] for j in idxs]
            yb = [y2[j] for j in idxs]
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=seed,
            )
            tree.fit(Xb, yb)
            # Align classes_
            tree.classes_ = self.classes_
            self.estimators_.append(tree)
        return self

    def predict_proba(self, X):
        X2 = [[float(v) for v in row] for row in X]
        K = len(self.classes_)
        avg = [[0.0] * K for _ in range(len(X2))]
        for tree in self.estimators_:
            probas = tree.predict_proba(X2)
            for i, row in enumerate(probas):
                for k in range(K):
                    avg[i][k] += row[k]
        T = len(self.estimators_)
        return [[v / T for v in row] for row in avg]

    def predict(self, X):
        proba = self.predict_proba(X)
        return [self.classes_[row.index(max(row))] for row in proba]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(a == b for a, b in zip(preds, y)) / len(y)


class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', random_state=None,
                 bootstrap=True, n_jobs=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.bootstrap = bootstrap

    def fit(self, X, y):
        X2 = [[float(v) for v in row] for row in X]
        y2 = [float(v) for v in y]
        n = len(X2)
        rng = _random.Random(self.random_state)
        self.estimators_ = []
        for i in range(self.n_estimators):
            seed = rng.randint(0, 2**31)
            if self.bootstrap:
                idxs = [rng.randint(0, n - 1) for _ in range(n)]
            else:
                idxs = list(range(n))
            Xb = [X2[j] for j in idxs]
            yb = [y2[j] for j in idxs]
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=seed,
            )
            tree.fit(Xb, yb)
            self.estimators_.append(tree)
        return self

    def predict(self, X):
        X2 = [[float(v) for v in row] for row in X]
        all_preds = [tree.predict(X2) for tree in self.estimators_]
        n = len(X2)
        return [sum(all_preds[t][i] for t in range(len(self.estimators_))) / len(self.estimators_)
                for i in range(n)]

    def score(self, X, y):
        preds = self.predict(X)
        y2 = [float(v) for v in y]
        mu = sum(y2) / len(y2)
        ss_res = sum((a - b) ** 2 for a, b in zip(y2, preds))
        ss_tot = sum((v - mu) ** 2 for v in y2)
        return float(1 - ss_res / (ss_tot + 1e-12))


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 subsample=1.0, random_state=None, min_samples_split=2,
                 min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X2 = [[float(v) for v in row] for row in X]
        y2 = [float(v) for v in y]
        n = len(y2)
        self.init_pred_ = sum(y2) / n
        F = [self.init_pred_] * n
        rng = _random.Random(self.random_state)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            residuals = [y2[i] - F[i] for i in range(n)]
            if self.subsample < 1.0:
                n_sub = max(1, int(n * self.subsample))
                idxs = rng.sample(range(n), n_sub)
            else:
                idxs = list(range(n))
            Xs = [X2[i] for i in idxs]
            rs = [residuals[i] for i in idxs]
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(Xs, rs)
            preds = tree.predict(X2)
            for i in range(n):
                F[i] += self.learning_rate * preds[i]
            self.estimators_.append(tree)
        return self

    def predict(self, X):
        X2 = [[float(v) for v in row] for row in X]
        n = len(X2)
        F = [self.init_pred_] * n
        for tree in self.estimators_:
            preds = tree.predict(X2)
            for i in range(n):
                F[i] += self.learning_rate * preds[i]
        return F

    def score(self, X, y):
        preds = self.predict(X)
        y2 = [float(v) for v in y]
        mu = sum(y2) / len(y2)
        ss_res = sum((a - b) ** 2 for a, b in zip(y2, preds))
        ss_tot = sum((v - mu) ** 2 for v in y2)
        return float(1 - ss_res / (ss_tot + 1e-12))


class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 subsample=1.0, random_state=None, min_samples_split=2,
                 min_samples_leaf=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    @staticmethod
    def _softmax(scores):
        mx = max(scores)
        exps = [math.exp(s - mx) for s in scores]
        total = sum(exps)
        return [e / total for e in exps]

    def fit(self, X, y):
        X2 = [[float(v) for v in row] for row in X]
        y2 = list(y)
        self.classes_ = sorted(set(y2))
        K = len(self.classes_)
        n = len(X2)
        rng = _random.Random(self.random_state)
        # F[i][k] = raw score for class k at sample i
        F = [[0.0] * K for _ in range(n)]
        self.estimators_ = [[] for _ in range(K)]
        for _ in range(self.n_estimators):
            # Compute probabilities
            probs = [self._softmax(F[i]) for i in range(n)]
            for k in range(K):
                y_k = [1.0 if y2[i] == self.classes_[k] else 0.0 for i in range(n)]
                residuals = [y_k[i] - probs[i][k] for i in range(n)]
                if self.subsample < 1.0:
                    n_sub = max(1, int(n * self.subsample))
                    idxs = rng.sample(range(n), n_sub)
                else:
                    idxs = list(range(n))
                Xs = [X2[i] for i in idxs]
                rs = [residuals[i] for i in idxs]
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                )
                tree.fit(Xs, rs)
                preds = tree.predict(X2)
                for i in range(n):
                    F[i][k] += self.learning_rate * preds[i]
                self.estimators_[k].append(tree)
        return self

    def predict_proba(self, X):
        X2 = [[float(v) for v in row] for row in X]
        n = len(X2)
        K = len(self.classes_)
        F = [[0.0] * K for _ in range(n)]
        for k in range(K):
            for tree in self.estimators_[k]:
                preds = tree.predict(X2)
                for i in range(n):
                    F[i][k] += self.learning_rate * preds[i]
        return [self._softmax(F[i]) for i in range(n)]

    def predict(self, X):
        proba = self.predict_proba(X)
        return [self.classes_[row.index(max(row))] for row in proba]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(a == b for a, b in zip(preds, y)) / len(y)
