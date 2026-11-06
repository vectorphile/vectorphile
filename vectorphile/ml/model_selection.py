"""Model selection and validation utilities."""
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


def train_test_split(*arrays, test_size=0.2, random_state=None, shuffle=True, stratify=None):
    if random_state is not None:
        np.random.seed(random_state)
    n = len(arrays[0])
    n_test = int(n * test_size)
    if shuffle:
        idx = np.random.permutation(n)
    else:
        idx = np.arange(n)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    result = []
    for arr in arrays:
        arr = np.array(arr)
        result.append(arr[train_idx])
        result.append(arr[test_idx])
    return result


class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        X = np.array(X)
        n = len(X)
        idx = np.arange(n)
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(idx)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[:n % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = idx[start:stop]
            train_idx = np.concatenate([idx[:start], idx[stop:]])
            yield train_idx, test_idx
            current = stop

    def get_n_splits(self):
        return self.n_splits


class StratifiedKFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        X, y = np.array(X), np.array(y)
        classes, y_indices = np.unique(y, return_inverse=True)
        rng = np.random.RandomState(self.random_state) if self.random_state else np.random.RandomState()
        folds = [[] for _ in range(self.n_splits)]
        for cls_idx in range(len(classes)):
            class_indices = np.where(y_indices == cls_idx)[0]
            if self.shuffle:
                rng.shuffle(class_indices)
            for i, idx in enumerate(class_indices):
                folds[i % self.n_splits].append(idx)
        for fold_i in range(self.n_splits):
            test_idx = np.array(folds[fold_i])
            train_idx = np.concatenate([np.array(folds[j]) for j in range(self.n_splits) if j != fold_i])
            yield train_idx, test_idx


def cross_val_score(estimator, X, y, cv=5, scoring='accuracy') -> np.ndarray:
    X, y = np.array(X), np.array(y)
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        estimator.fit(X_train, y_train)
        if scoring == 'accuracy' or scoring == 'r2':
            scores.append(estimator.score(X_test, y_test))
        else:
            scores.append(estimator.score(X_test, y_test))
    return np.array(scores)


class GridSearchCV:
    def __init__(self, estimator, param_grid: Dict, cv=5, scoring='accuracy', refit=True, n_jobs=1):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.n_jobs = n_jobs
        self.best_params_: Optional[Dict] = None
        self.best_score_: float = -np.inf
        self.best_estimator_ = None
        self.cv_results_: Dict = {}

    def _param_combinations(self):
        from itertools import product
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        for combo in product(*values):
            yield dict(zip(keys, combo))

    def fit(self, X, y) -> 'GridSearchCV':
        X, y = np.array(X), np.array(y)
        results = []
        for params in self._param_combinations():
            est = type(self.estimator)(**{**self.estimator.__dict__, **params})
            scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            mean_score = scores.mean()
            results.append({'params': params, 'mean_test_score': mean_score, 'std_test_score': scores.std()})
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params
        self.cv_results_ = {k: [r[k] for r in results] for k in results[0]}
        if self.refit:
            self.best_estimator_ = type(self.estimator)(**{**self.estimator.__dict__, **self.best_params_})
            self.best_estimator_.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.best_estimator_.predict(X)

    def score(self, X, y) -> float:
        return self.best_estimator_.score(X, y)


class RandomizedSearchCV:
    def __init__(self, estimator, param_distributions: Dict, n_iter=10, cv=5,
                 scoring='accuracy', refit=True, random_state=None):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.random_state = random_state
        self.best_params_: Optional[Dict] = None
        self.best_score_: float = -np.inf
        self.best_estimator_ = None

    def fit(self, X, y) -> 'RandomizedSearchCV':
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X, y = np.array(X), np.array(y)
        for _ in range(self.n_iter):
            params = {k: np.random.choice(v) if isinstance(v, (list, np.ndarray))
                      else float(np.random.uniform(*v)) for k, v in self.param_distributions.items()}
            try:
                est = type(self.estimator)(**{**self.estimator.__dict__, **params})
                scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
                mean_score = scores.mean()
                if mean_score > self.best_score_:
                    self.best_score_ = mean_score
                    self.best_params_ = params
            except Exception:
                continue
        if self.refit and self.best_params_ is not None:
            self.best_estimator_ = type(self.estimator)(**{**self.estimator.__dict__, **self.best_params_})
            self.best_estimator_.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.best_estimator_.predict(X)
