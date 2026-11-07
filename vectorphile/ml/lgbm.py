"""LightGBM-inspired gradient boosting with leaf-wise tree growth."""
from __future__ import annotations
import numpy as np
from typing import Optional, List


class _LeafTree:
    """Leaf-wise tree with histogram-based splits."""
    def __init__(self, max_leaves=31, max_depth=-1, min_child_samples=20,
                 min_child_weight=1e-3, reg_lambda=0.1, reg_alpha=0.0,
                 subsample_feature=1.0):
        self.max_leaves = max_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.subsample_feature = subsample_feature
        self.leaves = {}      # leaf_id -> prediction value
        self.splits = {}      # node_id -> (feature_idx, threshold, left_child, right_child)
        self.root = 0
        self._next_id = 0

    def _leaf_weight(self, grad, hess):
        g, h = grad.sum(), hess.sum()
        return -g / (h + self.reg_lambda)

    def _split_gain(self, grad_l, hess_l, grad_r, hess_r):
        gl, hl = grad_l.sum(), hess_l.sum()
        gr, hr = grad_r.sum(), hess_r.sum()
        g, h = gl + gr, hl + hr
        def score(g, h): return g**2 / (h + self.reg_lambda)
        return 0.5 * (score(gl, hl) + score(gr, hr) - score(g, h)) - self.reg_alpha

    def fit(self, X, grad, hess):
        n, m = X.shape
        # Start with one leaf covering all samples
        self._next_id = 0
        all_idx = np.arange(n)
        # Beam: list of (node_id, indices, depth)
        queue = [(self._new_node(), all_idx, 0)]
        self.leaves = {}
        self.splits = {}

        # Grow leaves one at a time, always picking best gain
        active_leaves = []
        for node_id, idx, depth in queue:
            active_leaves.append((node_id, idx, depth))
        self.leaves = {node_id: self._leaf_weight(grad[idx], hess[idx]) for node_id, idx, depth in active_leaves}

        for _ in range(self.max_leaves - 1):
            best_gain = -1e9
            best_leaf = None
            best_split = None
            for leaf_id, idx, depth in active_leaves:
                if (self.max_depth > 0 and depth >= self.max_depth) or len(idx) < 2 * self.min_child_samples:
                    continue
                feat_indices = np.arange(m)
                if self.subsample_feature < 1.0:
                    k = max(1, int(m * self.subsample_feature))
                    feat_indices = np.random.choice(m, k, replace=False)
                for f in feat_indices:
                    thresholds = np.unique(X[idx, f])
                    if len(thresholds) < 2:
                        continue
                    thresholds = (thresholds[:-1] + thresholds[1:]) / 2
                    for t in thresholds:
                        left_mask = X[idx, f] <= t
                        right_mask = ~left_mask
                        if left_mask.sum() < self.min_child_samples or right_mask.sum() < self.min_child_samples:
                            continue
                        gain = self._split_gain(grad[idx[left_mask]], hess[idx[left_mask]],
                                                grad[idx[right_mask]], hess[idx[right_mask]])
                        if gain > best_gain:
                            best_gain = gain
                            best_leaf = (leaf_id, idx, depth)
                            best_split = (f, t, left_mask)

            if best_leaf is None or best_gain <= 0:
                break

            leaf_id, idx, depth = best_leaf
            f, t, left_mask = best_split
            left_node = self._new_node()
            right_node = self._new_node()
            self.splits[leaf_id] = (f, t, left_node, right_node)
            del self.leaves[leaf_id]
            left_idx = idx[left_mask]
            right_idx = idx[~left_mask]
            self.leaves[left_node] = self._leaf_weight(grad[left_idx], hess[left_idx])
            self.leaves[right_node] = self._leaf_weight(grad[right_idx], hess[right_idx])
            active_leaves = [(lid, ix, d) if lid != leaf_id else None for lid, ix, d in active_leaves]
            active_leaves = [x for x in active_leaves if x is not None]
            active_leaves.append((left_node, left_idx, depth + 1))
            active_leaves.append((right_node, right_idx, depth + 1))

    def _new_node(self):
        nid = self._next_id
        self._next_id += 1
        return nid

    def predict(self, X):
        n = X.shape[0]
        preds = np.zeros(n, dtype=np.float64)
        for i in range(n):
            node = self.root
            while node in self.splits:
                f, t, left, right = self.splits[node]
                node = left if X[i, f] <= t else right
            preds[i] = self.leaves.get(node, 0.0)
        return preds


class LGBMClassifier:
    """LightGBM-style gradient boosting classifier with leaf-wise growth."""
    def __init__(self, n_estimators=100, learning_rate=0.1, num_leaves=31,
                 max_depth=-1, min_child_samples=20, subsample=1.0,
                 colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.1,
                 random_state=None, early_stopping_rounds=None, verbose=-1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.trees_: List = []
        self.classes_: np.ndarray = None
        self.feature_importances_: np.ndarray = None
        self._init_score = 0.0

    def _sigmoid(self, x): return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X, y, eval_set=None, callbacks=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = np.array(X, dtype=np.float64)
        y = np.array(y)
        self.classes_ = np.unique(y)
        y_bin = (y == self.classes_[-1]).astype(np.float64)
        p = y_bin.mean()
        self._init_score = np.log(p / (1 - p + 1e-12))
        F = np.full(len(y), self._init_score, dtype=np.float64)
        self.trees_ = []
        self.feature_importances_ = np.zeros(X.shape[1])
        best_val_loss = np.inf
        rounds_no_improve = 0
        for t in range(self.n_estimators):
            if self.subsample < 1.0:
                idx = np.random.choice(len(y), int(len(y)*self.subsample), replace=False)
                Xs, Fs, ys = X[idx], F[idx], y_bin[idx]
            else:
                Xs, Fs, ys = X, F, y_bin
            probs = self._sigmoid(Fs)
            grad = probs - ys
            hess = probs * (1 - probs) + 1e-6
            tree = _LeafTree(self.num_leaves, self.max_depth, self.min_child_samples,
                             reg_lambda=self.reg_lambda, reg_alpha=self.reg_alpha,
                             subsample_feature=self.colsample_bytree)
            tree.fit(Xs, grad, hess)
            preds = tree.predict(X)
            F += self.learning_rate * preds
            self.trees_.append(tree)
            # Feature importance (split counts)
            for node_id, (f, t, l, r) in tree.splits.items():
                self.feature_importances_[f] += 1
            if eval_set and self.early_stopping_rounds:
                Xv, yv = eval_set[0]
                yv_bin = (np.array(yv) == self.classes_[-1]).astype(np.float64)
                probs_v = self._sigmoid(self.predict_proba(np.array(Xv, dtype=np.float64))[:, 1])
                val_loss = -np.mean(yv_bin * np.log(probs_v + 1e-12) + (1-yv_bin) * np.log(1-probs_v+1e-12))
                if val_loss < best_val_loss - 1e-5:
                    best_val_loss = val_loss
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1
                if rounds_no_improve >= self.early_stopping_rounds:
                    if self.verbose >= 0:
                        print(f'Early stopping at round {t+1}')
                    break
        return self

    def _raw_predict(self, X):
        F = np.full(len(X), self._init_score, dtype=np.float64)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        F = self._raw_predict(X)
        p = self._sigmoid(F)
        return np.column_stack([1-p, p])

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.array(y)))

    def __repr__(self):
        return f'LGBMClassifier(n_estimators={self.n_estimators}, learning_rate={self.learning_rate}, num_leaves={self.num_leaves})'


class LGBMRegressor:
    """LightGBM-style gradient boosting regressor."""
    def __init__(self, n_estimators=100, learning_rate=0.1, num_leaves=31,
                 max_depth=-1, min_child_samples=20, subsample=1.0,
                 colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.1,
                 random_state=None, early_stopping_rounds=None, verbose=-1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.trees_: List = []
        self.feature_importances_: np.ndarray = None
        self._init_score = 0.0

    def fit(self, X, y, eval_set=None, callbacks=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        self._init_score = y.mean()
        F = np.full(len(y), self._init_score, dtype=np.float64)
        self.trees_ = []
        self.feature_importances_ = np.zeros(X.shape[1])
        best_val_loss = np.inf
        rounds_no_improve = 0
        for t in range(self.n_estimators):
            if self.subsample < 1.0:
                idx = np.random.choice(len(y), int(len(y)*self.subsample), replace=False)
                Xs, Fs, ys = X[idx], F[idx], y[idx]
            else:
                Xs, Fs, ys = X, F, y
            grad = Fs - ys
            hess = np.ones_like(grad)
            tree = _LeafTree(self.num_leaves, self.max_depth, self.min_child_samples,
                             reg_lambda=self.reg_lambda, reg_alpha=self.reg_alpha,
                             subsample_feature=self.colsample_bytree)
            tree.fit(Xs, grad, hess)
            preds = tree.predict(X)
            F += self.learning_rate * preds
            self.trees_.append(tree)
            for node_id, (f, t2, l, r) in tree.splits.items():
                self.feature_importances_[f] += 1
            if eval_set and self.early_stopping_rounds:
                Xv, yv = eval_set[0]
                val_preds = self.predict(np.array(Xv, dtype=np.float64))
                val_loss = np.mean((val_preds - np.array(yv))**2)
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1
                if rounds_no_improve >= self.early_stopping_rounds:
                    if self.verbose >= 0:
                        print(f'Early stopping at round {t+1}')
                    break
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        F = np.full(len(X), self._init_score, dtype=np.float64)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        return F.astype(np.float32)

    def score(self, X, y):
        y = np.array(y, dtype=np.float64)
        preds = self.predict(X).astype(np.float64)
        ss_res = np.sum((y - preds)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return float(1 - ss_res / (ss_tot + 1e-12))

    def __repr__(self):
        return f'LGBMRegressor(n_estimators={self.n_estimators}, learning_rate={self.learning_rate}, num_leaves={self.num_leaves})'


class LGBMRanker:
    """LightGBM-style LambdaRank for learning-to-rank tasks."""
    def __init__(self, n_estimators=100, learning_rate=0.1, num_leaves=31,
                 max_depth=-1, reg_lambda=0.1, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.trees_ = []
        self._init_score = 0.0

    def fit(self, X, y, group=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        self._init_score = 0.0
        F = np.zeros(len(y), dtype=np.float64)
        for t in range(self.n_estimators):
            # Pairwise gradient via sigmoid(score_i - score_j) for misranked pairs
            grad = np.zeros(len(y))
            hess = np.zeros(len(y))
            if group is None:
                pairs = [(i, j) for i in range(len(y)) for j in range(i+1, len(y)) if y[i] != y[j]]
            else:
                pairs = []
                start = 0
                for g in group:
                    for i in range(start, start+g):
                        for j in range(i+1, start+g):
                            if y[i] != y[j]:
                                pairs.append((i, j))
                    start += g
            for i, j in pairs:
                s = F[i] - F[j] if y[i] > y[j] else F[j] - F[i]
                sig = 1.0 / (1.0 + np.exp(s))
                if y[i] > y[j]:
                    grad[i] -= sig; grad[j] += sig
                else:
                    grad[j] -= sig; grad[i] += sig
                hess[i] += sig * (1 - sig)
                hess[j] += sig * (1 - sig)
            hess = np.maximum(hess, 1e-6)
            tree = _LeafTree(self.num_leaves, self.max_depth, reg_lambda=self.reg_lambda)
            tree.fit(X, grad, hess)
            F += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        F = np.zeros(len(X), dtype=np.float64)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        return F.astype(np.float32)
