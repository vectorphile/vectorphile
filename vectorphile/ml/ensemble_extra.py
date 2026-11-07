"""Extended ensemble methods — Bagging, AdaBoost, ExtraTrees, Voting, Stacking."""
from __future__ import annotations
import numpy as np
from typing import List, Optional, Union
from vectorphile.ml.tree import DecisionTreeClassifier, DecisionTreeRegressor


class BaggingClassifier:
    def __init__(self, estimator=None, n_estimators=10, max_samples=1.0,
                 max_features=1.0, bootstrap=True, bootstrap_features=False,
                 random_state=None, n_jobs=1):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.random_state = random_state
        self.estimators_ = []
        self.feature_indices_ = []
        self.classes_ = None

    def _make_estimator(self):
        import copy
        return copy.deepcopy(self.estimator) if self.estimator else DecisionTreeClassifier()

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X, y = np.array(X), np.array(y)
        n, m = X.shape
        self.classes_ = np.unique(y)
        n_samples = int(n * self.max_samples) if isinstance(self.max_samples, float) else self.max_samples
        n_features = int(m * self.max_features) if isinstance(self.max_features, float) else self.max_features
        self.estimators_, self.feature_indices_ = [], []
        for _ in range(self.n_estimators):
            sample_idx = np.random.choice(n, n_samples, replace=self.bootstrap)
            feat_idx = np.random.choice(m, n_features, replace=self.bootstrap_features)
            est = self._make_estimator()
            est.fit(X[sample_idx][:, feat_idx], y[sample_idx])
            self.estimators_.append(est)
            self.feature_indices_.append(feat_idx)
        return self

    def predict_proba(self, X):
        X = np.array(X)
        proba = np.zeros((len(X), len(self.classes_)))
        for est, feat_idx in zip(self.estimators_, self.feature_indices_):
            if hasattr(est, 'predict_proba'):
                p = est.predict_proba(X[:, feat_idx])
            else:
                pred = est.predict(X[:, feat_idx])
                p = np.zeros((len(X), len(self.classes_)))
                for i, cls in enumerate(self.classes_):
                    p[:, i] = (pred == cls).astype(float)
            proba += p
        return proba / self.n_estimators

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.array(y)))


class BaggingRegressor:
    def __init__(self, estimator=None, n_estimators=10, max_samples=1.0,
                 max_features=1.0, bootstrap=True, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.estimators_ = []
        self.feature_indices_ = []

    def _make_estimator(self):
        import copy
        return copy.deepcopy(self.estimator) if self.estimator else DecisionTreeRegressor()

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X, y = np.array(X), np.array(y, dtype=np.float64)
        n, m = X.shape
        n_samples = int(n * self.max_samples) if isinstance(self.max_samples, float) else self.max_samples
        n_features = int(m * self.max_features) if isinstance(self.max_features, float) else self.max_features
        self.estimators_, self.feature_indices_ = [], []
        for _ in range(self.n_estimators):
            sample_idx = np.random.choice(n, n_samples, replace=self.bootstrap)
            feat_idx = np.random.choice(m, n_features, replace=False)
            est = self._make_estimator()
            est.fit(X[sample_idx][:, feat_idx], y[sample_idx])
            self.estimators_.append(est)
            self.feature_indices_.append(feat_idx)
        return self

    def predict(self, X):
        X = np.array(X)
        return np.mean([est.predict(X[:, fi]) for est, fi in zip(self.estimators_, self.feature_indices_)], axis=0).astype(np.float32)

    def score(self, X, y):
        y = np.array(y, dtype=np.float64)
        preds = self.predict(X).astype(np.float64)
        ss_res = np.sum((y - preds)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return float(1 - ss_res / (ss_tot + 1e-12))


class AdaBoostClassifier:
    def __init__(self, estimator=None, n_estimators=50, learning_rate=1.0,
                 algorithm='SAMME', random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        self.classes_ = None

    def _make_estimator(self):
        import copy
        return copy.deepcopy(self.estimator) if self.estimator else DecisionTreeClassifier(max_depth=1)

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X, y = np.array(X), np.array(y)
        n = len(y)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        weights = np.full(n, 1.0 / n)
        self.estimators_, self.estimator_weights_, self.estimator_errors_ = [], [], []
        for t in range(self.n_estimators):
            idx = np.random.choice(n, n, replace=True, p=weights)
            est = self._make_estimator()
            est.fit(X[idx], y[idx])
            pred = est.predict(X)
            err = np.sum(weights * (pred != y))
            if err <= 0:
                self.estimators_.append(est)
                self.estimator_weights_.append(1.0)
                self.estimator_errors_.append(0.0)
                break
            if err >= 1 - 1/K:
                break
            alpha = self.learning_rate * (np.log((1 - err) / (err + 1e-12)) + np.log(K - 1))
            weights = weights * np.exp(alpha * (pred != y).astype(float))
            weights /= weights.sum()
            self.estimators_.append(est)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(err)
        return self

    def predict(self, X):
        X = np.array(X)
        class_scores = np.zeros((len(X), len(self.classes_)))
        for est, alpha in zip(self.estimators_, self.estimator_weights_):
            pred = est.predict(X)
            for i, cls in enumerate(self.classes_):
                class_scores[:, i] += alpha * (pred == cls).astype(float)
        return self.classes_[np.argmax(class_scores, axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.array(y)))


class AdaBoostRegressor:
    def __init__(self, estimator=None, n_estimators=50, learning_rate=1.0,
                 loss='linear', random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_weights_ = []

    def _make_estimator(self):
        import copy
        return copy.deepcopy(self.estimator) if self.estimator else DecisionTreeRegressor(max_depth=3)

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X, y = np.array(X), np.array(y, dtype=np.float64)
        n = len(y)
        weights = np.full(n, 1.0 / n)
        self.estimators_, self.estimator_weights_ = [], []
        for t in range(self.n_estimators):
            idx = np.random.choice(n, n, replace=True, p=weights)
            est = self._make_estimator()
            est.fit(X[idx], y[idx])
            pred = est.predict(X)
            err_arr = np.abs(pred - y)
            max_err = err_arr.max()
            if max_err == 0:
                self.estimators_.append(est)
                self.estimator_weights_.append(1.0)
                break
            if self.loss == 'linear':
                loss_arr = err_arr / max_err
            elif self.loss == 'square':
                loss_arr = (err_arr / max_err) ** 2
            else:
                loss_arr = 1 - np.exp(-err_arr / max_err)
            err = (weights * loss_arr).sum()
            if err >= 0.5:
                break
            beta = err / (1 - err + 1e-12)
            alpha = self.learning_rate * np.log(1.0 / (beta + 1e-12))
            weights = weights * np.power(beta, (1 - loss_arr))
            weights /= weights.sum()
            self.estimators_.append(est)
            self.estimator_weights_.append(alpha)
        return self

    def predict(self, X):
        X = np.array(X)
        if not self.estimators_:
            return np.zeros(len(X))
        preds = np.array([est.predict(X) for est in self.estimators_])
        weights = np.array(self.estimator_weights_)
        weights = weights / (weights.sum() + 1e-12)
        return (preds * weights[:, None]).sum(axis=0).astype(np.float32)

    def score(self, X, y):
        y = np.array(y, dtype=np.float64)
        preds = self.predict(X).astype(np.float64)
        ss_res = np.sum((y - preds)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return float(1 - ss_res / (ss_tot + 1e-12))


class ExtraTreesClassifier:
    """Extremely Randomized Trees — random thresholds reduce variance."""
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 max_features='sqrt', random_state=None, n_jobs=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.estimators_ = []
        self.classes_ = None

    def fit(self, X, y):
        from vectorphile.ml.tree import _TreeNode, _build_extra_tree
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X, y = np.array(X), np.array(y)
        self.classes_ = np.unique(y)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split,
                                          max_features=self.max_features,
                                          splitter='random')
            idx = np.random.choice(len(y), len(y), replace=True)
            tree.fit(X[idx], y[idx])
            self.estimators_.append(tree)
        return self

    def predict_proba(self, X):
        X = np.array(X)
        proba = np.zeros((len(X), len(self.classes_)))
        for est in self.estimators_:
            if hasattr(est, 'predict_proba'):
                proba += est.predict_proba(X)
            else:
                pred = est.predict(X)
                for i, cls in enumerate(self.classes_):
                    proba[:, i] += (pred == cls).astype(float)
        return proba / self.n_estimators

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.array(y)))

    @property
    def feature_importances_(self):
        if not self.estimators_:
            return None
        importances = [getattr(est, 'feature_importances_', None) for est in self.estimators_]
        importances = [imp for imp in importances if imp is not None]
        if not importances:
            return None
        return np.mean(importances, axis=0)


class ExtraTreesRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X, y = np.array(X), np.array(y, dtype=np.float64)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         min_samples_split=self.min_samples_split,
                                         max_features=self.max_features,
                                         splitter='random')
            idx = np.random.choice(len(y), len(y), replace=True)
            tree.fit(X[idx], y[idx])
            self.estimators_.append(tree)
        return self

    def predict(self, X):
        X = np.array(X)
        return np.mean([est.predict(X) for est in self.estimators_], axis=0).astype(np.float32)

    def score(self, X, y):
        y = np.array(y, dtype=np.float64)
        preds = self.predict(X).astype(np.float64)
        ss_res = np.sum((y - preds)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return float(1 - ss_res / (ss_tot + 1e-12))


class VotingClassifier:
    def __init__(self, estimators: List, voting='hard', weights=None, flatten_transform=True):
        self.estimators = estimators  # list of (name, estimator)
        self.voting = voting
        self.weights = weights
        self.flatten_transform = flatten_transform
        self.classes_ = None
        self.estimators_ = []

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.classes_ = np.unique(y)
        self.estimators_ = []
        for name, est in self.estimators:
            import copy
            cloned = copy.deepcopy(est)
            cloned.fit(X, y)
            self.estimators_.append(cloned)
        return self

    def predict(self, X):
        X = np.array(X)
        if self.voting == 'soft':
            avg = np.zeros((len(X), len(self.classes_)))
            for i, est in enumerate(self.estimators_):
                w = self.weights[i] if self.weights else 1.0
                p = est.predict_proba(X) if hasattr(est, 'predict_proba') else \
                    np.eye(len(self.classes_))[(np.searchsorted(self.classes_, est.predict(X)))]
                avg += w * p
            return self.classes_[np.argmax(avg, axis=1)]
        else:
            preds = np.array([est.predict(X) for est in self.estimators_])
            result = np.apply_along_axis(
                lambda col: np.bincount([np.searchsorted(self.classes_, v) for v in col]).argmax(),
                axis=0, arr=preds
            )
            return self.classes_[result]

    def predict_proba(self, X):
        X = np.array(X)
        avg = np.zeros((len(X), len(self.classes_)))
        for i, est in enumerate(self.estimators_):
            w = self.weights[i] if self.weights else 1.0
            p = est.predict_proba(X) if hasattr(est, 'predict_proba') else \
                np.eye(len(self.classes_))[(np.searchsorted(self.classes_, est.predict(X)))]
            avg += w * p
        total_w = sum(self.weights) if self.weights else len(self.estimators_)
        return avg / total_w

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.array(y)))


class VotingRegressor:
    def __init__(self, estimators: List, weights=None):
        self.estimators = estimators
        self.weights = weights
        self.estimators_ = []

    def fit(self, X, y):
        import copy
        X, y = np.array(X), np.array(y, dtype=np.float64)
        self.estimators_ = []
        for _, est in self.estimators:
            cloned = copy.deepcopy(est)
            cloned.fit(X, y)
            self.estimators_.append(cloned)
        return self

    def predict(self, X):
        X = np.array(X)
        preds = np.array([est.predict(X) for est in self.estimators_], dtype=np.float64)
        if self.weights:
            w = np.array(self.weights)
            return (preds * w[:, None]).sum(axis=0) / w.sum()
        return preds.mean(axis=0).astype(np.float32)

    def score(self, X, y):
        y = np.array(y, dtype=np.float64)
        preds = self.predict(X).astype(np.float64)
        ss_res = np.sum((y - preds)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return float(1 - ss_res / (ss_tot + 1e-12))


class StackingClassifier:
    def __init__(self, estimators: List, final_estimator=None, cv=5,
                 passthrough=False, stack_method='auto'):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.passthrough = passthrough
        self.stack_method = stack_method
        self.estimators_ = []
        self.final_estimator_ = None
        self.classes_ = None

    def fit(self, X, y):
        import copy
        from vectorphile.ml.linear import LogisticRegression
        X, y = np.array(X), np.array(y)
        n = len(y)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        # First-level predictions via cross-validation
        meta_X = np.zeros((n, len(self.estimators) * K))
        self.estimators_ = []
        for j, (name, est) in enumerate(self.estimators):
            cloned_full = copy.deepcopy(est)
            cloned_full.fit(X, y)
            self.estimators_.append(cloned_full)
            # OOF predictions
            fold_size = n // self.cv
            for fold in range(self.cv):
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < self.cv - 1 else n
                train_idx = np.concatenate([np.arange(val_start), np.arange(val_end, n)])
                val_idx = np.arange(val_start, val_end)
                cloned_fold = copy.deepcopy(est)
                cloned_fold.fit(X[train_idx], y[train_idx])
                if hasattr(cloned_fold, 'predict_proba'):
                    p = cloned_fold.predict_proba(X[val_idx])
                else:
                    pred = cloned_fold.predict(X[val_idx])
                    p = np.eye(K)[(np.searchsorted(self.classes_, pred))]
                meta_X[val_idx, j*K:(j+1)*K] = p
        if self.passthrough:
            meta_X = np.hstack([meta_X, X])
        self.final_estimator_ = copy.deepcopy(self.final_estimator) if self.final_estimator else LogisticRegression()
        self.final_estimator_.fit(meta_X, y)
        return self

    def _make_meta_features(self, X):
        K = len(self.classes_)
        meta = np.zeros((len(X), len(self.estimators_) * K))
        for j, est in enumerate(self.estimators_):
            if hasattr(est, 'predict_proba'):
                p = est.predict_proba(X)
            else:
                pred = est.predict(X)
                p = np.eye(K)[(np.searchsorted(self.classes_, pred))]
            meta[:, j*K:(j+1)*K] = p
        if self.passthrough:
            meta = np.hstack([meta, X])
        return meta

    def predict(self, X):
        return self.final_estimator_.predict(self._make_meta_features(np.array(X)))

    def predict_proba(self, X):
        return self.final_estimator_.predict_proba(self._make_meta_features(np.array(X)))

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.array(y)))


class StackingRegressor:
    def __init__(self, estimators: List, final_estimator=None, cv=5, passthrough=False):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.passthrough = passthrough
        self.estimators_ = []
        self.final_estimator_ = None

    def fit(self, X, y):
        import copy
        from vectorphile.ml.linear import LinearRegression
        X, y = np.array(X), np.array(y, dtype=np.float64)
        n = len(y)
        meta_X = np.zeros((n, len(self.estimators)))
        self.estimators_ = []
        for j, (name, est) in enumerate(self.estimators):
            cloned_full = copy.deepcopy(est)
            cloned_full.fit(X, y)
            self.estimators_.append(cloned_full)
            fold_size = n // self.cv
            for fold in range(self.cv):
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < self.cv - 1 else n
                train_idx = np.concatenate([np.arange(val_start), np.arange(val_end, n)])
                val_idx = np.arange(val_start, val_end)
                cloned_fold = copy.deepcopy(est)
                cloned_fold.fit(X[train_idx], y[train_idx])
                meta_X[val_idx, j] = cloned_fold.predict(X[val_idx])
        if self.passthrough:
            meta_X = np.hstack([meta_X, X])
        self.final_estimator_ = copy.deepcopy(self.final_estimator) if self.final_estimator else LinearRegression()
        self.final_estimator_.fit(meta_X, y)
        return self

    def predict(self, X):
        X = np.array(X)
        meta = np.column_stack([est.predict(X) for est in self.estimators_])
        if self.passthrough:
            meta = np.hstack([meta, X])
        return self.final_estimator_.predict(meta)

    def score(self, X, y):
        y = np.array(y, dtype=np.float64)
        preds = self.predict(X).astype(np.float64)
        ss_res = np.sum((y - preds)**2)
        ss_tot = np.sum((y - y.mean())**2)
        return float(1 - ss_res / (ss_tot + 1e-12))
