"""XGBoost-style gradient boosting — pure Python."""
import math
import random as _random


def _mse_grad_hess(y_true, y_pred):
    grad = [p - t for t, p in zip(y_true, y_pred)]
    hess = [1.0] * len(y_true)
    return grad, hess


def _log_loss_grad_hess(y_true, y_pred):
    def _sigmoid(x):
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        e = math.exp(x)
        return e / (1.0 + e)
    prob = [_sigmoid(p) for p in y_pred]
    grad = [p - t for t, p in zip(y_true, prob)]
    hess = [max(p * (1.0 - p), 1e-6) for p in prob]
    return grad, hess


class _RegTree:
    """Single regression tree using second-order gradient stats (XGBoost style)."""

    def __init__(self, max_depth=6, min_child_weight=1.0, reg_lambda=1.0,
                 gamma=0.0, subsample=1.0, colsample=1.0, rng=None):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.subsample = subsample
        self.colsample = colsample
        self.rng = rng or _random.Random(42)
        self.tree = None

    def _score(self, G, H):
        return -0.5 * G * G / (H + self.reg_lambda)

    def _leaf_val(self, G, H):
        return -G / (H + self.reg_lambda)

    def _best_split(self, X, grad, hess, features):
        G_tot = sum(grad)
        H_tot = sum(hess)
        best_gain = self.gamma
        best_feat, best_thresh = -1, 0.0
        for f in features:
            vals = sorted(set(X[i][f] for i in range(len(X))))
            G_L, H_L = 0.0, 0.0
            for thresh in vals[:-1]:
                for i in range(len(X)):
                    if X[i][f] <= thresh:
                        G_L += grad[i]
                        H_L += hess[i]
                G_R = G_tot - G_L
                H_R = H_tot - H_L
                if H_L < self.min_child_weight or H_R < self.min_child_weight:
                    continue
                gain = -(self._score(G_L, H_L) + self._score(G_R, H_R) - self._score(G_tot, H_tot))
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = thresh
                G_L, H_L = 0.0, 0.0
        return best_feat, best_thresh

    def _build(self, X, grad, hess, depth, features):
        G = sum(grad)
        H = sum(hess)
        leaf_val = self._leaf_val(G, H)
        if depth >= self.max_depth or len(X) <= 1:
            return {'leaf': leaf_val}
        f, thresh = self._best_split(X, grad, hess, features)
        if f < 0:
            return {'leaf': leaf_val}
        left_idx = [i for i in range(len(X)) if X[i][f] <= thresh]
        right_idx = [i for i in range(len(X)) if X[i][f] > thresh]
        if not left_idx or not right_idx:
            return {'leaf': leaf_val}
        return {
            'feat': f, 'thresh': thresh,
            'left': self._build([X[i] for i in left_idx], [grad[i] for i in left_idx],
                                 [hess[i] for i in left_idx], depth + 1, features),
            'right': self._build([X[i] for i in right_idx], [grad[i] for i in right_idx],
                                  [hess[i] for i in right_idx], depth + 1, features),
        }

    def fit(self, X, grad, hess):
        n, p = len(X), len(X[0])
        # Subsample rows
        if self.subsample < 1.0:
            k = max(1, int(n * self.subsample))
            idxs = self.rng.sample(range(n), k)
            X = [X[i] for i in idxs]
            grad = [grad[i] for i in idxs]
            hess = [hess[i] for i in idxs]
        # Subsample cols
        if self.colsample < 1.0:
            k = max(1, int(p * self.colsample))
            features = self.rng.sample(range(p), k)
        else:
            features = list(range(p))
        self.tree = self._build(X, grad, hess, 0, features)
        return self

    def _predict_one(self, x, node):
        if 'leaf' in node:
            return node['leaf']
        if x[node['feat']] <= node['thresh']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]


class XGBClassifier:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.3,
                 min_child_weight=1.0, reg_lambda=1.0, gamma=0.0,
                 subsample=1.0, colsample_bytree=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

    def fit(self, X, y):
        X2 = [[float(v) for v in row] for row in X]
        y2 = [float(v) for v in y]
        self.classes_ = sorted(set(y2))
        rng = _random.Random(self.random_state)
        # F0 = log(mean(y) / (1 - mean(y) + 1e-9))
        mean_y = sum(y2) / len(y2)
        f0 = math.log((mean_y + 1e-9) / (1.0 - mean_y + 1e-9))
        F = [f0] * len(X2)
        self.trees_ = []
        self.f0_ = f0
        for _ in range(self.n_estimators):
            g, h = _log_loss_grad_hess(y2, F)
            tree = _RegTree(max_depth=self.max_depth, min_child_weight=self.min_child_weight,
                            reg_lambda=self.reg_lambda, gamma=self.gamma,
                            subsample=self.subsample, colsample=self.colsample_bytree, rng=rng)
            tree.fit(X2, g, h)
            preds = tree.predict(X2)
            F = [F[i] + self.learning_rate * preds[i] for i in range(len(X2))]
            self.trees_.append(tree)
        return self

    def _sigmoid(self, x):
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        e = math.exp(x)
        return e / (1.0 + e)

    def predict_proba(self, X):
        X2 = [[float(v) for v in row] for row in X]
        F = [self.f0_] * len(X2)
        for tree in self.trees_:
            preds = tree.predict(X2)
            F = [F[i] + self.learning_rate * preds[i] for i in range(len(X2))]
        probs = [self._sigmoid(f) for f in F]
        return [[1.0 - p, p] for p in probs]

    def predict(self, X):
        proba = self.predict_proba(X)
        return [self.classes_[1] if row[1] >= 0.5 else self.classes_[0] for row in proba]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(a == b for a, b in zip(preds, y)) / len(y)


class XGBRegressor:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.3,
                 min_child_weight=1.0, reg_lambda=1.0, gamma=0.0,
                 subsample=1.0, colsample_bytree=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

    def fit(self, X, y):
        X2 = [[float(v) for v in row] for row in X]
        y2 = [float(v) for v in y]
        rng = _random.Random(self.random_state)
        self.f0_ = sum(y2) / len(y2)
        F = [self.f0_] * len(X2)
        self.trees_ = []
        for _ in range(self.n_estimators):
            g, h = _mse_grad_hess(y2, F)
            tree = _RegTree(max_depth=self.max_depth, min_child_weight=self.min_child_weight,
                            reg_lambda=self.reg_lambda, gamma=self.gamma,
                            subsample=self.subsample, colsample=self.colsample_bytree, rng=rng)
            tree.fit(X2, g, h)
            preds = tree.predict(X2)
            F = [F[i] + self.learning_rate * preds[i] for i in range(len(X2))]
            self.trees_.append(tree)
        return self

    def predict(self, X):
        X2 = [[float(v) for v in row] for row in X]
        F = [self.f0_] * len(X2)
        for tree in self.trees_:
            preds = tree.predict(X2)
            F = [F[i] + self.learning_rate * preds[i] for i in range(len(X2))]
        return F

    def score(self, X, y):
        preds = self.predict(X)
        y2 = [float(v) for v in y]
        mu = sum(y2) / len(y2)
        ss_res = sum((a - b) ** 2 for a, b in zip(y2, preds))
        ss_tot = sum((v - mu) ** 2 for v in y2)
        return float(1 - ss_res / (ss_tot + 1e-12))
