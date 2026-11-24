"""LightGBM-style gradient boosting — pure Python."""
import math
import random as _random
from typing import Optional


def _col(X, j):
    return [X[i][j] for i in range(len(X))]


def _uniq_sorted(seq):
    return sorted(set(seq))


class _LeafTree:
    """Histogram-based leaf-wise tree."""

    def __init__(self, num_leaves=31, min_data_in_leaf=20, max_bin=255):
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.max_bin = max_bin
        self.tree = None

    def fit(self, X, grad, hess):
        n, p = len(X), len(X[0])
        # Simple depth-limited tree via best-split, leaf limited by num_leaves
        self.tree = self._build(list(range(n)), X, grad, hess, 0)
        return self

    def _gain(self, g_L, h_L, g_R, h_R, lam=1e-3):
        def val(g, h):
            return (g ** 2) / (h + lam) if (h + lam) > 0 else 0.0
        return 0.5 * (val(g_L, h_L) + val(g_R, h_R) - val(g_L + g_R, h_L + h_R))

    def _build(self, idxs, X, grad, hess, depth):
        if len(idxs) <= self.min_data_in_leaf or depth >= int(math.log2(max(self.num_leaves, 1)) + 1):
            g = sum(grad[i] for i in idxs)
            h = sum(hess[i] for i in idxs)
            return {'leaf': True, 'value': -g / (h + 1e-3)}
        n, p = len(X), len(X[0])
        best_gain, best_feat, best_thresh = -float('inf'), None, None
        for feat in range(p):
            vals = sorted(set(_col([X[i] for i in idxs], feat)))
            for t in range(len(vals) - 1):
                thresh = (vals[t] + vals[t + 1]) / 2
                left = [i for i in idxs if X[i][feat] <= thresh]
                right = [i for i in idxs if X[i][feat] > thresh]
                if len(left) < self.min_data_in_leaf or len(right) < self.min_data_in_leaf:
                    continue
                g_L = sum(grad[i] for i in left)
                h_L = sum(hess[i] for i in left)
                g_R = sum(grad[i] for i in right)
                h_R = sum(hess[i] for i in right)
                gain = self._gain(g_L, h_L, g_R, h_R)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh
        if best_feat is None:
            g = sum(grad[i] for i in idxs)
            h = sum(hess[i] for i in idxs)
            return {'leaf': True, 'value': -g / (h + 1e-3)}
        left = [i for i in idxs if X[i][best_feat] <= best_thresh]
        right = [i for i in idxs if X[i][best_feat] > best_thresh]
        return {
            'leaf': False,
            'feat': best_feat,
            'thresh': best_thresh,
            'left': self._build(left, X, grad, hess, depth + 1),
            'right': self._build(right, X, grad, hess, depth + 1),
        }

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['value']
        if x[node['feat']] <= node['thresh']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]


class LGBMRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, num_leaves=31,
                 min_data_in_leaf=20, subsample=1.0, colsample_bytree=1.0,
                 reg_lambda=1.0, reg_alpha=0.0, random_state=None, n_jobs=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state

    def fit(self, X, y, eval_set=None):
        X2 = [[float(v) for v in row] for row in X]
        y2 = [float(v) for v in y]
        n = len(y2)
        self.init_score_ = sum(y2) / n
        F = [self.init_score_] * n
        rng = _random.Random(self.random_state)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            grad = [F[i] - y2[i] for i in range(n)]
            hess = [1.0] * n
            if self.subsample < 1.0:
                n_sub = max(1, int(n * self.subsample))
                idxs = rng.sample(range(n), n_sub)
                Xs = [X2[i] for i in idxs]
                gs = [grad[i] for i in idxs]
                hs = [hess[i] for i in idxs]
            else:
                Xs, gs, hs = X2, grad, hess
            tree = _LeafTree(num_leaves=self.num_leaves, min_data_in_leaf=self.min_data_in_leaf)
            tree.fit(Xs, gs, hs)
            preds = tree.predict(X2)
            for i in range(n):
                F[i] += self.learning_rate * preds[i]
            self.estimators_.append(tree)
        return self

    def predict(self, X):
        X2 = [[float(v) for v in row] for row in X]
        n = len(X2)
        F = [self.init_score_] * n
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


class LGBMClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, num_leaves=31,
                 min_data_in_leaf=20, subsample=1.0, random_state=None, n_jobs=None,
                 num_class=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.subsample = subsample
        self.random_state = random_state

    @staticmethod
    def _softmax(scores):
        mx = max(scores)
        exps = [math.exp(s - mx) for s in scores]
        total = sum(exps)
        return [e / total for e in exps]

    @staticmethod
    def _sigmoid(z):
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def fit(self, X, y, eval_set=None):
        X2 = [[float(v) for v in row] for row in X]
        y2 = list(y)
        self.classes_ = _uniq_sorted(y2)
        K = len(self.classes_)
        n = len(y2)
        rng = _random.Random(self.random_state)
        # F[k][i] = raw score for class k at sample i
        F = [[0.0] * n for _ in range(K)]
        self.estimators_ = [[] for _ in range(K)]
        for _ in range(self.n_estimators):
            probs = [self._softmax([F[k][i] for k in range(K)]) for i in range(n)]
            for k in range(K):
                y_k = [1.0 if y2[i] == self.classes_[k] else 0.0 for i in range(n)]
                grad = [probs[i][k] - y_k[i] for i in range(n)]
                hess = [max(probs[i][k] * (1 - probs[i][k]), 1e-6) for i in range(n)]
                if self.subsample < 1.0:
                    n_sub = max(1, int(n * self.subsample))
                    idxs = rng.sample(range(n), n_sub)
                    Xs = [X2[i] for i in idxs]
                    gs = [grad[i] for i in idxs]
                    hs = [hess[i] for i in idxs]
                else:
                    Xs, gs, hs = X2, grad, hess
                tree = _LeafTree(num_leaves=self.num_leaves, min_data_in_leaf=self.min_data_in_leaf)
                tree.fit(Xs, gs, hs)
                preds = tree.predict(X2)
                for i in range(n):
                    F[k][i] += self.learning_rate * preds[i]
                self.estimators_[k].append(tree)
        return self

    def predict_proba(self, X):
        X2 = [[float(v) for v in row] for row in X]
        n = len(X2)
        K = len(self.classes_)
        F = [[0.0] * n for _ in range(K)]
        for k in range(K):
            for tree in self.estimators_[k]:
                preds = tree.predict(X2)
                for i in range(n):
                    F[k][i] += self.learning_rate * preds[i]
        return [self._softmax([F[k][i] for k in range(K)]) for i in range(n)]

    def predict(self, X):
        proba = self.predict_proba(X)
        return [self.classes_[row.index(max(row))] for row in proba]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(a == b for a, b in zip(preds, y)) / len(y)


class LGBMRanker:
    def __init__(self, n_estimators=100, learning_rate=0.1, num_leaves=31,
                 min_data_in_leaf=20, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.random_state = random_state

    def fit(self, X, y, group=None, eval_set=None):
        X2 = [[float(v) for v in row] for row in X]
        y2 = [float(v) for v in y]
        n = len(y2)
        rng = _random.Random(self.random_state)
        self.init_score_ = 0.0
        F = [0.0] * n
        self.estimators_ = []
        for _ in range(self.n_estimators):
            # Pairwise gradient (LambdaRank approximation)
            grad = [0.0] * n
            hess = [1.0] * n
            for i in range(n):
                for j in range(n):
                    if y2[i] > y2[j]:
                        s = F[i] - F[j]
                        p = 1.0 / (1.0 + math.exp(-s)) if s > -100 else 0.0
                        g = p - 1.0
                        grad[i] += g
                        grad[j] -= g
                        hess[i] += max(p * (1 - p), 1e-6)
                        hess[j] += max(p * (1 - p), 1e-6)
            tree = _LeafTree(num_leaves=self.num_leaves, min_data_in_leaf=self.min_data_in_leaf)
            tree.fit(X2, grad, hess)
            preds = tree.predict(X2)
            for i in range(n):
                F[i] += self.learning_rate * preds[i]
            self.estimators_.append(tree)
        return self

    def predict(self, X):
        X2 = [[float(v) for v in row] for row in X]
        n = len(X2)
        F = [0.0] * n
        for tree in self.estimators_:
            preds = tree.predict(X2)
            for i in range(n):
                F[i] += self.learning_rate * preds[i]
        return F
