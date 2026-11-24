"""Model selection utilities — pure Python."""
import math
import random as _random
from typing import List, Optional, Iterator, Tuple


def train_test_split(*arrays, test_size=0.2, random_state=None, shuffle=True, stratify=None):
    if random_state is not None:
        _random.seed(random_state)
    n = len(arrays[0])
    indices = list(range(n))
    if shuffle:
        _random.shuffle(indices)
    if stratify is not None:
        # stratified split
        strat = list(stratify)
        classes = sorted(set(strat))
        train_idx, test_idx = [], []
        for c in classes:
            c_idx = [i for i in indices if strat[i] == c]
            n_test = max(1, round(len(c_idx) * test_size))
            test_idx.extend(c_idx[:n_test])
            train_idx.extend(c_idx[n_test:])
    else:
        n_test = max(1, round(n * test_size))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
    result = []
    for arr in arrays:
        lst = list(arr)
        result.append([lst[i] for i in train_idx])
        result.append([lst[i] for i in test_idx])
    return result


class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        n = len(X)
        indices = list(range(n))
        if self.shuffle:
            rng = _random.Random(self.random_state)
            rng.shuffle(indices)
        fold_sizes = [n // self.n_splits + (1 if i < n % self.n_splits else 0)
                      for i in range(self.n_splits)]
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_idx = indices[:start] + indices[stop:]
            yield train_idx, test_idx
            current = stop


class StratifiedKFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        y = list(y)
        classes = sorted(set(y))
        rng = _random.Random(self.random_state)
        # group indices by class
        class_indices = {c: [i for i, v in enumerate(y) if v == c] for c in classes}
        if self.shuffle:
            for c in classes:
                rng.shuffle(class_indices[c])
        # assign each index to a fold
        fold_indices = [[] for _ in range(self.n_splits)]
        for c in classes:
            idxs = class_indices[c]
            for i, idx in enumerate(idxs):
                fold_indices[i % self.n_splits].append(idx)
        for fold in range(self.n_splits):
            test_idx = fold_indices[fold]
            train_idx = [i for f in range(self.n_splits) if f != fold for i in fold_indices[f]]
            yield train_idx, test_idx


def cross_val_score(estimator, X, y, cv=5, scoring=None):
    X, y = list(X), list(y)
    kf = KFold(n_splits=cv)
    scores = []
    for train_idx, test_idx in kf.split(X):
        X_tr = [X[i] for i in train_idx]
        y_tr = [y[i] for i in train_idx]
        X_te = [X[i] for i in test_idx]
        y_te = [y[i] for i in test_idx]
        est = estimator.__class__(**{k: v for k, v in estimator.__dict__.items()
                                     if not k.endswith('_')})
        est.fit(X_tr, y_tr)
        if scoring is None:
            score = est.score(X_te, y_te)
        elif callable(scoring):
            score = scoring(est, X_te, y_te)
        else:
            score = est.score(X_te, y_te)
        scores.append(score)
    return scores


class GridSearchCV:
    def __init__(self, estimator, param_grid, cv=5, scoring=None, refit=True):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.refit = refit

    def _param_combinations(self):
        from itertools import product
        keys = list(self.param_grid.keys())
        vals = list(self.param_grid.values())
        for combo in product(*vals):
            yield dict(zip(keys, combo))

    def fit(self, X, y):
        self.best_score_ = float('-inf')
        self.best_params_ = None
        X, y = list(X), list(y)
        for params in self._param_combinations():
            scores = []
            kf = KFold(n_splits=self.cv)
            for train_idx, test_idx in kf.split(X):
                X_tr = [X[i] for i in train_idx]
                y_tr = [y[i] for i in train_idx]
                X_te = [X[i] for i in test_idx]
                y_te = [y[i] for i in test_idx]
                cls = self.estimator.__class__
                est = cls(**params)
                est.fit(X_tr, y_tr)
                score = est.score(X_te, y_te)
                scores.append(score)
            mean_score = sum(scores) / len(scores)
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params
        if self.refit and self.best_params_:
            self.best_estimator_ = self.estimator.__class__(**self.best_params_)
            self.best_estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        return self.best_estimator_.score(X, y)


class RandomizedSearchCV:
    def __init__(self, estimator, param_distributions, n_iter=10, cv=5,
                 scoring=None, random_state=None, refit=True):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.refit = refit

    def _sample_params(self, rng):
        params = {}
        for k, v in self.param_distributions.items():
            if hasattr(v, 'rvs'):
                params[k] = v.rvs()
            elif isinstance(v, list):
                params[k] = rng.choice(v)
            else:
                params[k] = v
        return params

    def fit(self, X, y):
        rng = _random.Random(self.random_state)
        self.best_score_ = float('-inf')
        self.best_params_ = None
        X, y = list(X), list(y)
        for _ in range(self.n_iter):
            params = self._sample_params(rng)
            scores = []
            kf = KFold(n_splits=self.cv)
            for train_idx, test_idx in kf.split(X):
                X_tr = [X[i] for i in train_idx]
                y_tr = [y[i] for i in train_idx]
                X_te = [X[i] for i in test_idx]
                y_te = [y[i] for i in test_idx]
                est = self.estimator.__class__(**params)
                est.fit(X_tr, y_tr)
                score = est.score(X_te, y_te)
                scores.append(score)
            mean_score = sum(scores) / len(scores)
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params
        if self.refit and self.best_params_:
            self.best_estimator_ = self.estimator.__class__(**self.best_params_)
            self.best_estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)
