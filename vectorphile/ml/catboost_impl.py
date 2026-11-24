"""CatBoost-style gradient boosting with categorical support — pure Python."""
import math
import random as _random
from typing import Optional, List


def _col(X, j):
    return [X[i][j] for i in range(len(X))]


def _uniq_sorted(seq):
    return sorted(set(seq))


def _target_encode(col, y, prior_mean, alpha=10.0):
    """Target-encode a categorical column."""
    n = len(col)
    # Compute per-category statistics
    cat_sum = {}
    cat_count = {}
    for v, t in zip(col, y):
        cat_sum[v] = cat_sum.get(v, 0.0) + float(t)
        cat_count[v] = cat_count.get(v, 0) + 1
    result = []
    for v in col:
        cnt = cat_count[v]
        s = cat_sum[v]
        # Smoothed target encoding
        encoded = (s + prior_mean * alpha) / (cnt + alpha)
        result.append(encoded)
    return result


class _CatTree:
    def __init__(self, max_depth=6, min_data_in_leaf=1):
        self.max_depth = max_depth
        self.min_data_in_leaf = min_data_in_leaf

    def fit(self, X, grad, hess, cat_features=None):
        n, p = len(X), len(X[0])
        self.cat_features = cat_features or []
        self.tree = self._build(list(range(n)), X, grad, hess, 0)
        return self

    def _gain(self, g_L, h_L, g_R, h_R, lam=1.0):
        def v(g, h):
            return g ** 2 / (h + lam) if (h + lam) > 0 else 0.0
        return 0.5 * (v(g_L, h_L) + v(g_R, h_R) - v(g_L + g_R, h_L + h_R))

    def _build(self, idxs, X, grad, hess, depth):
        n, p = len(X), len(X[0])
        if len(idxs) <= self.min_data_in_leaf or depth >= self.max_depth:
            g = sum(grad[i] for i in idxs)
            h = sum(hess[i] for i in idxs)
            return {'leaf': True, 'value': -g / (h + 1.0)}
        best_gain, best_feat, best_thresh, best_cats = -float('inf'), None, None, None
        for feat in range(p):
            feat_vals = [X[i][feat] for i in idxs]
            if feat in self.cat_features:
                cats = _uniq_sorted(feat_vals)
                # Try each category vs rest split
                for cat in cats:
                    left = [i for i in idxs if X[i][feat] == cat]
                    right = [i for i in idxs if X[i][feat] != cat]
                    if len(left) < self.min_data_in_leaf or len(right) < self.min_data_in_leaf:
                        continue
                    g_L = sum(grad[i] for i in left)
                    h_L = sum(hess[i] for i in left)
                    g_R = sum(grad[i] for i in right)
                    h_R = sum(hess[i] for i in right)
                    gain = self._gain(g_L, h_L, g_R, h_R)
                    if gain > best_gain:
                        best_gain, best_feat, best_cats = gain, feat, {cat}
                        best_thresh = None
            else:
                vals = sorted(set(float(v) for v in feat_vals))
                for t in range(len(vals) - 1):
                    thresh = (vals[t] + vals[t + 1]) / 2
                    left = [i for i in idxs if float(X[i][feat]) <= thresh]
                    right = [i for i in idxs if float(X[i][feat]) > thresh]
                    if len(left) < self.min_data_in_leaf or len(right) < self.min_data_in_leaf:
                        continue
                    g_L = sum(grad[i] for i in left)
                    h_L = sum(hess[i] for i in left)
                    g_R = sum(grad[i] for i in right)
                    h_R = sum(hess[i] for i in right)
                    gain = self._gain(g_L, h_L, g_R, h_R)
                    if gain > best_gain:
                        best_gain, best_feat, best_thresh = gain, feat, thresh
                        best_cats = None
        if best_feat is None:
            g = sum(grad[i] for i in idxs)
            h = sum(hess[i] for i in idxs)
            return {'leaf': True, 'value': -g / (h + 1.0)}
        if best_cats is not None:
            left = [i for i in idxs if X[i][best_feat] in best_cats]
            right = [i for i in idxs if X[i][best_feat] not in best_cats]
        else:
            left = [i for i in idxs if float(X[i][best_feat]) <= best_thresh]
            right = [i for i in idxs if float(X[i][best_feat]) > best_thresh]
        return {
            'leaf': False,
            'feat': best_feat,
            'thresh': best_thresh,
            'cats': best_cats,
            'left': self._build(left, X, grad, hess, depth + 1),
            'right': self._build(right, X, grad, hess, depth + 1),
        }

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['value']
        feat = node['feat']
        if node['cats'] is not None:
            if x[feat] in node['cats']:
                return self._predict_one(x, node['left'])
            return self._predict_one(x, node['right'])
        if float(x[feat]) <= node['thresh']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]


def _prepare_X(X, cat_features=None, y=None):
    """Convert raw feature matrix to float, encoding categorical columns."""
    cat_features = cat_features or []
    result = []
    if y is not None:
        prior = sum(float(v) for v in y) / len(y) if y else 0.0
    else:
        prior = 0.0
    for j in range(len(X[0])):
        col = [X[i][j] for i in range(len(X))]
        if j in cat_features:
            if y is not None:
                col = _target_encode(col, y, prior)
            else:
                # Label encode
                cats = _uniq_sorted(col)
                cat_map = {c: float(i) for i, c in enumerate(cats)}
                col = [cat_map.get(v, 0.0) for v in col]
        else:
            col = [float(v) for v in col]
        for i in range(len(X)):
            if len(result) <= i:
                result.append([])
            result[i].append(col[i])
    return result


class CatBoostRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, depth=6,
                 l2_leaf_reg=3.0, subsample=1.0, random_state=None,
                 cat_features=None, verbose=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.subsample = subsample
        self.random_state = random_state
        self.cat_features = cat_features or []
        self.verbose = verbose

    def fit(self, X, y, eval_set=None):
        X2 = _prepare_X(X, self.cat_features, y)
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
                rng.shuffle(idxs)
                # Ordered boosting approximation: use first half for target encoding, second for split
                Xs = [X2[i] for i in idxs]
                gs = [grad[i] for i in idxs]
                hs = [hess[i] for i in idxs]
            else:
                Xs, gs, hs = X2, grad, hess
            tree = _CatTree(max_depth=self.depth)
            tree.fit(Xs, gs, hs)
            preds = tree.predict(X2)
            for i in range(n):
                F[i] += self.learning_rate * preds[i]
            self.estimators_.append(tree)
        return self

    def predict(self, X):
        X2 = _prepare_X(X, self.cat_features)
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


class CatBoostClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, depth=6,
                 l2_leaf_reg=3.0, subsample=1.0, random_state=None,
                 cat_features=None, verbose=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.subsample = subsample
        self.random_state = random_state
        self.cat_features = cat_features or []
        self.verbose = verbose

    @staticmethod
    def _softmax(scores):
        mx = max(scores)
        exps = [math.exp(s - mx) for s in scores]
        total = sum(exps)
        return [e / total for e in exps]

    def fit(self, X, y, eval_set=None):
        X2 = _prepare_X(X, self.cat_features, y)
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
                hess = [max(probs[i][k] * (1 - probs[i][k]), 1e-6) for i in range(n)]
                if self.subsample < 1.0:
                    n_sub = max(1, int(n * self.subsample))
                    idxs = rng.sample(range(n), n_sub)
                    Xs = [X2[i] for i in idxs]
                    gs = [grad[i] for i in idxs]
                    hs = [hess[i] for i in idxs]
                else:
                    Xs, gs, hs = X2, grad, hess
                tree = _CatTree(max_depth=self.depth)
                tree.fit(Xs, gs, hs)
                preds = tree.predict(X2)
                for i in range(n):
                    F[k][i] += self.learning_rate * preds[i]
                self.estimators_[k].append(tree)
        return self

    def predict_proba(self, X):
        X2 = _prepare_X(X, self.cat_features)
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


class CatBoostRanker:
    def __init__(self, n_estimators=100, learning_rate=0.1, depth=6,
                 random_state=None, cat_features=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_state = random_state
        self.cat_features = cat_features or []

    def fit(self, X, y, group=None):
        X2 = _prepare_X(X, self.cat_features)
        y2 = [float(v) for v in y]
        n = len(y2)
        F = [0.0] * n
        self.estimators_ = []
        for _ in range(self.n_estimators):
            grad = [0.0] * n
            hess = [1.0] * n
            for i in range(n):
                for j in range(n):
                    if y2[i] > y2[j]:
                        s = F[i] - F[j]
                        p = 1.0 / (1.0 + math.exp(-max(-100, min(100, s))))
                        g = p - 1.0
                        grad[i] += g
                        grad[j] -= g
                        h = max(p * (1 - p), 1e-6)
                        hess[i] += h
                        hess[j] += h
            tree = _CatTree(max_depth=self.depth)
            tree.fit(X2, grad, hess)
            preds = tree.predict(X2)
            for i in range(n):
                F[i] += self.learning_rate * preds[i]
            self.estimators_.append(tree)
        return self

    def predict(self, X):
        X2 = _prepare_X(X, self.cat_features)
        n = len(X2)
        F = [0.0] * n
        for tree in self.estimators_:
            preds = tree.predict(X2)
            for i in range(n):
                F[i] += self.learning_rate * preds[i]
        return F
