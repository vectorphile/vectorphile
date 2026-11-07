"""CatBoost-inspired gradient boosting with ordered boosting and categorical features."""
from __future__ import annotations
import numpy as np
from typing import Optional, List, Union


class _CatTree:
    """Symmetric depth-wise tree (CatBoost style)."""
    def __init__(self, depth=6, min_leaf_samples=1, l2_reg=3.0):
        self.depth = depth
        self.min_leaf_samples = min_leaf_samples
        self.l2_reg = l2_reg
        self.splits = []     # list of (feature_idx, threshold) per level
        self.leaf_values = []  # 2^depth leaves

    def fit(self, X, grad, hess, cat_features=None):
        n, m = X.shape
        indices = np.arange(n)
        # Symmetric splits: same feature+threshold applied across all nodes at each level
        node_indices = [indices]  # level 0: all samples in one node
        for level in range(self.depth):
            if not node_indices:
                break
            # Find best split for this level across all nodes combined
            best_gain = -1e9
            best_f, best_t = None, None
            all_idx = np.concatenate(node_indices) if len(node_indices) > 1 else node_indices[0]
            if len(all_idx) < 2 * self.min_leaf_samples:
                break
            for f in range(m):
                vals = np.unique(X[all_idx, f])
                if len(vals) < 2:
                    continue
                thresholds = (vals[:-1] + vals[1:]) / 2
                # Sample a subset for speed
                if len(thresholds) > 50:
                    thresholds = thresholds[np.linspace(0, len(thresholds)-1, 50, dtype=int)]
                for t in thresholds:
                    total_gain = 0.0
                    for node_idx in node_indices:
                        lm = X[node_idx, f] <= t
                        rm = ~lm
                        if lm.sum() < self.min_leaf_samples or rm.sum() < self.min_leaf_samples:
                            total_gain -= 1e9
                            break
                        gl, hl = grad[node_idx[lm]].sum(), hess[node_idx[lm]].sum()
                        gr, hr = grad[node_idx[rm]].sum(), hess[node_idx[rm]].sum()
                        g, h = gl+gr, hl+hr
                        total_gain += 0.5*(gl**2/(hl+self.l2_reg) + gr**2/(hr+self.l2_reg) - g**2/(h+self.l2_reg))
                    if total_gain > best_gain:
                        best_gain = total_gain
                        best_f, best_t = f, t
            if best_f is None:
                break
            self.splits.append((best_f, best_t))
            new_nodes = []
            for node_idx in node_indices:
                lm = X[node_idx, best_f] <= best_t
                new_nodes.append(node_idx[lm])
                new_nodes.append(node_idx[~lm])
            node_indices = [n for n in new_nodes if len(n) >= 1]
        # Compute leaf values (all nodes at final level)
        self.leaf_values = []
        for node_idx in node_indices:
            if len(node_idx) == 0:
                self.leaf_values.append(0.0)
            else:
                g, h = grad[node_idx].sum(), hess[node_idx].sum()
                self.leaf_values.append(-g / (h + self.l2_reg))
        self._final_nodes = node_indices

    def predict(self, X):
        n = X.shape[0]
        preds = np.zeros(n, dtype=np.float64)
        # Route each sample to leaf
        node_assign = np.zeros(n, dtype=int)  # leaf index
        for level, (f, t) in enumerate(self.splits):
            # At level l, bit l of leaf index encodes left(0)/right(1)
            right_mask = X[:, f] > t
            node_assign = node_assign * 2 + right_mask.astype(int)
        for i in range(n):
            leaf = node_assign[i]
            if leaf < len(self.leaf_values):
                preds[i] = self.leaf_values[leaf]
        return preds


def _target_encode(X_col, y, fold_indices, global_mean, l2=1.0):
    """Ordered target encoding — no leakage."""
    encoded = np.zeros(len(X_col), dtype=np.float64)
    for idx in range(len(X_col)):
        # Use only samples from prior folds
        prior_mask = fold_indices < fold_indices[idx]
        if prior_mask.sum() == 0:
            encoded[idx] = global_mean
        else:
            same_cat = X_col[prior_mask] == X_col[idx]
            n_same = same_cat.sum()
            mean_same = y[prior_mask][same_cat].mean() if n_same > 0 else global_mean
            encoded[idx] = (n_same * mean_same + l2 * global_mean) / (n_same + l2)
    return encoded


class CatBoostClassifier:
    """CatBoost-style gradient boosting with ordered boosting and native categorical support."""
    def __init__(self, iterations=100, learning_rate=0.1, depth=6,
                 l2_leaf_reg=3.0, random_seed=None, cat_features=None,
                 early_stopping_rounds=None, verbose=0):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.trees_: List[_CatTree] = []
        self.classes_: np.ndarray = None
        self.feature_importances_: np.ndarray = None
        self._init_score = 0.0

    def _sigmoid(self, x): return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _encode_cats(self, X, y=None, is_train=False):
        X = np.array(X, dtype=object)
        X_enc = X.astype(np.float64, errors='ignore')
        # For each categorical feature, apply target encoding
        if y is not None and is_train:
            self._cat_stats = {}
            global_mean = float(np.array(y, dtype=np.float64).mean())
            for f in self.cat_features:
                col = X[:, f]
                fold_order = np.arange(len(col))
                encoded = _target_encode(col, np.array(y, dtype=np.float64), fold_order, global_mean, self.l2_leaf_reg)
                X_enc[:, f] = encoded
                # Store category means for inference
                cat_means = {}
                for cat_val in np.unique(col):
                    mask = col == cat_val
                    cat_means[cat_val] = float(np.array(y, dtype=np.float64)[mask].mean())
                self._cat_stats[f] = (cat_means, global_mean)
        elif hasattr(self, '_cat_stats'):
            for f in self.cat_features:
                col = X[:, f]
                cat_means, global_mean = self._cat_stats[f]
                X_enc[:, f] = np.array([cat_means.get(v, global_mean) for v in col], dtype=np.float64)
        return X_enc.astype(np.float64)

    def fit(self, X, y, eval_set=None):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        y = np.array(y)
        self.classes_ = np.unique(y)
        y_bin = (y == self.classes_[-1]).astype(np.float64)
        X_enc = self._encode_cats(X, y_bin, is_train=True)
        p = y_bin.mean().clip(1e-6, 1-1e-6)
        self._init_score = np.log(p / (1 - p))
        F = np.full(len(y), self._init_score, dtype=np.float64)
        self.trees_ = []
        self.feature_importances_ = np.zeros(X_enc.shape[1])
        # Ordered boosting: shuffle data and use random permutations
        perm = np.random.permutation(len(y))
        X_perm, y_perm, F_perm = X_enc[perm], y_bin[perm], F[perm]
        best_val = np.inf
        rounds_no_improve = 0
        for t in range(self.iterations):
            probs = self._sigmoid(F_perm)
            grad = probs - y_perm
            hess = probs * (1 - probs) + 1e-6
            tree = _CatTree(self.depth, l2_reg=self.l2_leaf_reg)
            tree.fit(X_perm, grad, hess)
            preds = tree.predict(X_enc)  # predict on original order
            F += self.learning_rate * preds
            F_perm = F[perm]
            self.trees_.append(tree)
            for feat_idx, _ in tree.splits:
                self.feature_importances_[feat_idx] += 1
            if self.verbose > 0 and (t + 1) % max(1, self.iterations // 10) == 0:
                loss = -np.mean(y_bin * np.log(self._sigmoid(F) + 1e-12) + (1-y_bin) * np.log(1-self._sigmoid(F)+1e-12))
                print(f'[{t+1}] train logloss: {loss:.6f}')
            if eval_set and self.early_stopping_rounds:
                Xv, yv = eval_set
                yv_bin = (np.array(yv) == self.classes_[-1]).astype(np.float64)
                probs_v = self.predict_proba(Xv)[:, 1]
                val_loss = -np.mean(yv_bin * np.log(probs_v+1e-12) + (1-yv_bin)*np.log(1-probs_v+1e-12))
                if val_loss < best_val - 1e-5:
                    best_val = val_loss
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1
                if rounds_no_improve >= self.early_stopping_rounds:
                    break
        return self

    def predict_proba(self, X):
        X_enc = self._encode_cats(X)
        F = np.full(len(X_enc), self._init_score, dtype=np.float64)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X_enc)
        p = self._sigmoid(F)
        return np.column_stack([1-p, p])

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.array(y)))

    def __repr__(self):
        return f'CatBoostClassifier(iterations={self.iterations}, depth={self.depth})'


class CatBoostRegressor:
    """CatBoost-style gradient boosting regressor."""
    def __init__(self, iterations=100, learning_rate=0.1, depth=6,
                 l2_leaf_reg=3.0, random_seed=None, cat_features=None,
                 early_stopping_rounds=None, verbose=0):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.cat_features = cat_features or []
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.trees_: List[_CatTree] = []
        self._init_score = 0.0
        self._cat_stats = {}

    def _encode_cats(self, X, y=None, is_train=False):
        X = np.array(X, dtype=object)
        X_enc = X.copy().astype(np.float64, errors='ignore')
        if y is not None and is_train:
            global_mean = float(np.array(y, dtype=np.float64).mean())
            for f in self.cat_features:
                col = X[:, f]
                fold_order = np.arange(len(col))
                encoded = _target_encode(col, np.array(y, dtype=np.float64), fold_order, global_mean, self.l2_leaf_reg)
                X_enc[:, f] = encoded
                cat_means = {}
                for cv in np.unique(col):
                    mask = col == cv
                    cat_means[cv] = float(np.array(y, dtype=np.float64)[mask].mean())
                self._cat_stats[f] = (cat_means, global_mean)
        elif self._cat_stats:
            for f in self.cat_features:
                col = X[:, f]
                cat_means, global_mean = self._cat_stats[f]
                X_enc[:, f] = [cat_means.get(v, global_mean) for v in col]
        return X_enc.astype(np.float64)

    def fit(self, X, y, eval_set=None):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        y = np.array(y, dtype=np.float64)
        X_enc = self._encode_cats(X, y, is_train=True)
        self._init_score = y.mean()
        F = np.full(len(y), self._init_score, dtype=np.float64)
        self.trees_ = []
        perm = np.random.permutation(len(y))
        X_perm, y_perm, F_perm = X_enc[perm], y[perm], F[perm]
        best_val = np.inf
        rounds_no_improve = 0
        for t in range(self.iterations):
            grad = F_perm - y_perm
            hess = np.ones_like(grad)
            tree = _CatTree(self.depth, l2_reg=self.l2_leaf_reg)
            tree.fit(X_perm, grad, hess)
            preds = tree.predict(X_enc)
            F += self.learning_rate * preds
            F_perm = F[perm]
            self.trees_.append(tree)
            if eval_set and self.early_stopping_rounds:
                Xv, yv = eval_set
                val_preds = self.predict(Xv)
                val_loss = float(np.mean((val_preds - np.array(yv))**2))
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1
                if rounds_no_improve >= self.early_stopping_rounds:
                    break
        return self

    def predict(self, X):
        X_enc = self._encode_cats(X)
        F = np.full(len(X_enc), self._init_score, dtype=np.float64)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X_enc)
        return F.astype(np.float32)

    def score(self, X, y):
        y = np.array(y, dtype=np.float64)
        preds = self.predict(X).astype(np.float64)
        ss_res = np.sum((y - preds)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return float(1 - ss_res / (ss_tot + 1e-12))

    def __repr__(self):
        return f'CatBoostRegressor(iterations={self.iterations}, depth={self.depth})'


class CatBoostRanker:
    """CatBoost-style ranker."""
    def __init__(self, iterations=100, learning_rate=0.1, depth=6, l2_leaf_reg=3.0, random_seed=None):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.trees_ = []
        self._init_score = 0.0

    def fit(self, X, y, group=None):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        self._init_score = 0.0
        F = np.zeros(len(y), dtype=np.float64)
        self.trees_ = []
        for t in range(self.iterations):
            grad = np.zeros(len(y))
            hess = np.ones(len(y)) * 1e-6
            if group is not None:
                start = 0
                for g in group:
                    for i in range(start, start + g):
                        for j in range(i + 1, start + g):
                            diff = F[i] - F[j]
                            sig = 1.0 / (1.0 + np.exp(diff if y[i] > y[j] else -diff))
                            d = sig if y[i] > y[j] else -sig
                            grad[i] -= d; grad[j] += d
                            hess[i] += sig*(1-sig); hess[j] += sig*(1-sig)
                    start += g
            tree = _CatTree(self.depth, l2_reg=self.l2_leaf_reg)
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
