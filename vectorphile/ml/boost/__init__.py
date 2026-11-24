"""XGBoost-style gradient boosting — pure Python."""
import math
import random as _random
from typing import Optional


def _col(X, j):
    return [X[i][j] for i in range(len(X))]


def _uniq_sorted(seq):
    return sorted(set(seq))


class _XGBTree:
    def __init__(self, max_depth=6, min_child_weight=1.0, reg_lambda=1.0,
                 reg_alpha=0.0, subsample_feats=1.0, random_state=None):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.subsample_feats = subsample_feats
        self.random_state = random_state

    def _gain(self, g_L, h_L, g_R, h_R):
        lam = self.reg_lambda

        def val(g, h):
            return g ** 2 / (h + lam)

        return 0.5 * (val(g_L, h_L) + val(g_R, h_R) - val(g_L + g_R, h_L + h_R))

    def _leaf_weight(self, g, h):
        return -g / (h + self.reg_lambda)

    def fit(self, X, grad, hess):
        n, p = len(X), len(X[0])
        self._p = p
        rng = _random.Random(self.random_state)
        if self.subsample_feats < 1.0:
            nf = max(1, int(p * self.subsample_feats))
            self._feat_idxs = rng.sample(range(p), nf)
        else:
            self._feat_idxs = list(range(p))
        self.tree = self._build(list(range(n)), X, grad, hess, 0)
        return self

    def _build(self, idxs, X, grad, hess, depth):
        g = sum(grad[i] for i in idxs)
        h = sum(hess[i] for i in idxs)
        if depth >= self.max_depth or h < self.min_child_weight or len(idxs) <= 1:
            return {'leaf': True, 'value': self._leaf_weight(g, h)}
        best_gain, best_feat, best_thresh = -float('inf'), None, None
        for feat in self._feat_idxs:
            vals = sorted(set(float(X[i][feat]) for i in idxs))
            for t in range(len(vals) - 1):
                thresh = (vals[t] + vals[t + 1]) / 2
                left = [i for i in idxs if float(X[i][feat]) <= thresh]
                right = [i for i in idxs if float(X[i][feat]) > thresh]
                h_L = sum(hess[i] for i in left)
                h_R = sum(hess[i] for i in right)
                if h_L < self.min_child_weight or h_R < self.min_child_weight:
                    continue
                g_L = sum(grad[i] for i in left)
                g_R = sum(grad[i] for i in right)
                gain = self._gain(g_L, h_L, g_R, h_R)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh
        if best_feat is None:
            return {'leaf': True, 'value': self._leaf_weight(g, h)}
        left = [i for i in idxs if float(X[i][best_feat]) <= best_thresh]
        right = [i for i in idxs if float(X[i][best_feat]) > best_thresh]
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
        if float(x[node['feat']]) <= node['thresh']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]


class XGBRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.3, max_depth=6,
                 min_child_weight=1.0, subsample=1.0, colsample_bytree=1.0,
                 reg_lambda=1.0, reg_alpha=0.0, random_state=None, n_jobs=None,
                 gamma=0.0, eval_metric=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.random_state = random_state

    def fit(self, X, y, eval_set=None, verbose=False):
        X2 = [[float(v) for v in row] for row in X]
        y2 = [float(v) for v in y]
        n = len(y2)
        self.base_score_ = sum(y2) / n
        F = [self.base_score_] * n
        rng = _random.Random(self.random_state)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            # MSE gradient / hessian
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
            tree = _XGBTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                reg_alpha=self.reg_alpha,
                subsample_feats=self.colsample_bytree,
                random_state=rng.randint(0, 2**31),
            )
            tree.fit(Xs, gs, hs)
            preds = tree.predict(X2)
            for i in range(n):
                F[i] += self.learning_rate * preds[i]
            self.estimators_.append(tree)
        return self

    def predict(self, X):
        X2 = [[float(v) for v in row] for row in X]
        n = len(X2)
        F = [self.base_score_] * n
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


class XGBClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.3, max_depth=6,
                 min_child_weight=1.0, subsample=1.0, colsample_bytree=1.0,
                 reg_lambda=1.0, reg_alpha=0.0, random_state=None, n_jobs=None,
                 use_label_encoder=False, eval_metric=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.random_state = random_state

    @staticmethod
    def _softmax(scores):
        mx = max(scores)
        exps = [math.exp(s - mx) for s in scores]
        total = sum(exps)
        return [e / total for e in exps]

    def fit(self, X, y, eval_set=None, verbose=False):
        X2 = [[float(v) for v in row] for row in X]
        y2 = list(y)
        self.classes_ = _uniq_sorted(y2)
        K = len(self.classes_)
        n = len(y2)
        rng = _random.Random(self.random_state)
        F = [[0.0] * n for _ in range(K)]
        self.estimators_ = [[] for _ in range(K)]
        for _ in range(self.n_estimators):
            probs = [self._softmax([F[k][i] for k in range(K)]) for i in range(n)]
            for k in range(K):
                y_k = [1.0 if y2[i] == self.classes_[k] else 0.0 for i in range(n)]
                grad = [probs[i][k] - y_k[i] for i in range(n)]
                hess = [max(2 * probs[i][k] * (1 - probs[i][k]), 1e-6) for i in range(n)]
                if self.subsample < 1.0:
                    n_sub = max(1, int(n * self.subsample))
                    idxs = rng.sample(range(n), n_sub)
                    Xs = [X2[i] for i in idxs]
                    gs = [grad[i] for i in idxs]
                    hs = [hess[i] for i in idxs]
                else:
                    Xs, gs, hs = X2, grad, hess
                tree = _XGBTree(
                    max_depth=self.max_depth,
                    min_child_weight=self.min_child_weight,
                    reg_lambda=self.reg_lambda,
                    subsample_feats=self.colsample_bytree,
                    random_state=rng.randint(0, 2**31),
                )
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
