"""vectorphile.model_selection — Model selection and evaluation tools.

All computations use pure Python. No external dependencies.
"""
from __future__ import annotations
import copy
import math
import random
import time
from typing import Any, Iterator, List, Optional, Tuple

# Re-export from ml
from vectorphile.ml import train_test_split


def _to_list(X) -> list:
    if hasattr(X, 'to_flat_list'):
        return X.to_flat_list()
    if hasattr(X, 'tolist'):
        return X.tolist()
    return list(X)


def _subset(X, indices: List[int]):
    """Subset X (list-of-rows or similar) by indices."""
    if isinstance(X, list):
        return [X[i] for i in indices]
    if hasattr(X, '__getitem__'):
        try:
            return X[indices]
        except Exception:
            pass
    return [X[i] for i in indices]


# ---------------------------------------------------------------------------
# KFold
# ---------------------------------------------------------------------------

class KFold:
    """K-Folds cross-validator."""

    def __init__(self, n_splits: int = 5, shuffle: bool = False,
                 random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        n = len(X)
        indices = list(range(n))
        if self.shuffle:
            rng = random.Random(self.random_state)
            rng.shuffle(indices)
        fold_size = n // self.n_splits
        for i in range(self.n_splits):
            start = i * fold_size
            end = start + fold_size if i < self.n_splits - 1 else n
            val_idx = indices[start:end]
            train_idx = indices[:start] + indices[end:]
            yield train_idx, val_idx

    def get_n_splits(self, X=None) -> int:
        return self.n_splits


class StratifiedKFold:
    """Stratified K-Fold (preserves class distribution)."""

    def __init__(self, n_splits: int = 5, shuffle: bool = False,
                 random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        y_flat = list(y) if not isinstance(y, list) else y
        if hasattr(y_flat[0], '__iter__') and not isinstance(y_flat[0], str):
            y_flat = [v for row in y_flat for v in row]
        classes = sorted(set(y_flat))
        class_indices = {c: [i for i, v in enumerate(y_flat) if v == c] for c in classes}
        rng = random.Random(self.random_state)
        if self.shuffle:
            for c in classes:
                rng.shuffle(class_indices[c])
        fold_assignments = [0] * len(y_flat)
        for c in classes:
            for i, sample_idx in enumerate(class_indices[c]):
                fold_assignments[sample_idx] = i % self.n_splits
        for fold in range(self.n_splits):
            val_idx = [i for i, f in enumerate(fold_assignments) if f == fold]
            train_idx = [i for i, f in enumerate(fold_assignments) if f != fold]
            yield train_idx, val_idx

    def get_n_splits(self, X=None) -> int:
        return self.n_splits


class LeaveOneOut:
    """Leave-one-out cross-validator."""

    def split(self, X, y=None):
        n = len(X)
        for i in range(n):
            yield list(range(0, i)) + list(range(i + 1, n)), [i]

    def get_n_splits(self, X) -> int:
        return len(X)


class RepeatedKFold:
    """Repeated K-Fold."""

    def __init__(self, n_splits: int = 5, n_repeats: int = 10,
                 random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X, y=None):
        rng = random.Random(self.random_state)
        for _ in range(self.n_repeats):
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=rng.randint(0, 2**31))
            yield from kf.split(X, y)

    def get_n_splits(self, X=None) -> int:
        return self.n_splits * self.n_repeats


# ---------------------------------------------------------------------------
# cross_val_score / cross_validate
# ---------------------------------------------------------------------------

def _score_estimator(est, X, y, train_idx, val_idx, scoring: str) -> float:
    X_tr = _subset(X, train_idx)
    X_v = _subset(X, val_idx)
    y_tr = _subset(y, train_idx) if not isinstance(y, list) else [y[i] for i in train_idx]
    y_v = _subset(y, val_idx) if not isinstance(y, list) else [y[i] for i in val_idx]
    e = copy.deepcopy(est)
    e.fit(X_tr, y_tr)
    preds = e.predict(X_v)
    if hasattr(preds, 'tolist'):
        preds = preds.tolist()
    if hasattr(preds, 'to_flat_list'):
        preds = preds.to_flat_list()
    preds = list(preds) if not isinstance(preds, list) else preds
    y_v_list = list(y_v) if not isinstance(y_v, list) else y_v
    if scoring == 'accuracy':
        return sum(1 for p, t in zip(preds, y_v_list) if p == t) / max(len(y_v_list), 1)
    elif scoring in ('r2',):
        preds_f = [float(v) for v in preds]
        y_vf = [float(v) for v in y_v_list]
        mean_y = sum(y_vf) / len(y_vf)
        ss_res = sum((y_vf[i] - preds_f[i]) ** 2 for i in range(len(y_vf)))
        ss_tot = sum((v - mean_y) ** 2 for v in y_vf) + 1e-10
        return 1.0 - ss_res / ss_tot
    elif scoring in ('neg_mse', 'neg_mean_squared_error'):
        preds_f = [float(v) for v in preds]
        y_vf = [float(v) for v in y_v_list]
        return -sum((preds_f[i] - y_vf[i]) ** 2 for i in range(len(y_vf))) / len(y_vf)
    elif scoring == 'f1':
        from vectorphile.metrics import f1_score
        return f1_score(y_v_list, preds, average='macro')
    else:
        if hasattr(e, 'score'):
            return float(e.score(X_v, y_v))
        return 0.0


def cross_val_score(estimator, X, y, cv=5, scoring: str = 'accuracy') -> List[float]:
    y_list = list(y) if not isinstance(y, list) else y
    if isinstance(cv, int):
        cv = KFold(n_splits=cv)
    if isinstance(cv, list):
        splits = cv
    else:
        splits = list(cv.split(X, y_list))
    scores = []
    for train_idx, val_idx in splits:
        scores.append(_score_estimator(estimator, X, y_list, train_idx, val_idx, scoring))
    return scores


def cross_validate(estimator, X, y, cv=5, scoring=('accuracy',),
                   return_train_score: bool = False) -> dict:
    y_list = list(y) if not isinstance(y, list) else y
    if isinstance(cv, int):
        cv = KFold(n_splits=cv)
    if isinstance(scoring, str):
        scoring = (scoring,)
    splits = list(cv.split(X, y_list))
    results: dict = {f'test_{s}': [] for s in scoring}
    results['fit_time'] = []
    results['score_time'] = []
    if return_train_score:
        for s in scoring:
            results[f'train_{s}'] = []
    for train_idx, val_idx in splits:
        X_tr = _subset(X, train_idx)
        y_tr = [y_list[i] for i in train_idx]
        est = copy.deepcopy(estimator)
        t0 = time.perf_counter()
        est.fit(X_tr, y_tr)
        results['fit_time'].append(time.perf_counter() - t0)
        t1 = time.perf_counter()
        for s in scoring:
            results[f'test_{s}'].append(_score_estimator(est, X, y_list, train_idx, val_idx, s))
        results['score_time'].append(time.perf_counter() - t1)
        if return_train_score:
            for s in scoring:
                results[f'train_{s}'].append(_score_estimator(est, X, y_list, train_idx, train_idx, s))
    return results


# ---------------------------------------------------------------------------
# learning_curve / validation_curve
# ---------------------------------------------------------------------------

def learning_curve(estimator, X, y, train_sizes=None,
                   cv: int = 5, scoring: str = 'accuracy') -> Tuple:
    n = len(X)
    y_list = list(y) if not isinstance(y, list) else y
    if train_sizes is None:
        train_sizes = [0.1, 0.325, 0.55, 0.775, 1.0]
    abs_sizes = [max(2, int(s * n)) for s in train_sizes]
    kf = KFold(n_splits=cv)
    splits = list(kf.split(X, y_list))
    train_scores_all = []
    val_scores_all = []
    for size in abs_sizes:
        t_scores, v_scores = [], []
        for train_idx, val_idx in splits:
            sub_idx = train_idx[:size]
            t_scores.append(_score_estimator(estimator, X, y_list, sub_idx, sub_idx, scoring))
            v_scores.append(_score_estimator(estimator, X, y_list, sub_idx, val_idx, scoring))
        train_scores_all.append(t_scores)
        val_scores_all.append(v_scores)
    return abs_sizes, train_scores_all, val_scores_all


class ShuffleSplit:
    """Random permutation cross-validator."""

    def __init__(self, n_splits: int = 10, test_size: float = 0.1,
                 random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y=None):
        X_list = _to_list(X)
        n = len(X_list)
        n_test = max(1, int(n * self.test_size))
        rng = random.Random(self.random_state)
        for _ in range(self.n_splits):
            indices = list(range(n))
            rng.shuffle(indices)
            test_idx = sorted(indices[:n_test])
            train_idx = sorted(indices[n_test:])
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class TimeSeriesSplit:
    """Time-series cross-validator (expanding window)."""

    def __init__(self, n_splits: int = 5, gap: int = 0,
                 max_train_size: Optional[int] = None, test_size: Optional[int] = None):
        self.n_splits = n_splits
        self.gap = gap
        self.max_train_size = max_train_size
        self.test_size = test_size

    def split(self, X, y=None):
        X_list = _to_list(X)
        n = len(X_list)
        n_test = self.test_size or max(1, n // (self.n_splits + 1))
        for i in range(self.n_splits):
            test_end = n - (self.n_splits - 1 - i) * n_test
            test_start = test_end - n_test
            train_end = test_start - self.gap
            if train_end <= 0:
                continue
            train_start = 0
            if self.max_train_size:
                train_start = max(0, train_end - self.max_train_size)
            yield list(range(train_start, train_end)), list(range(test_start, test_end))

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def validation_curve(estimator, X, y, param_name: str,
                     param_range: list, cv: int = 5,
                     scoring: str = 'accuracy') -> Tuple:
    y_list = list(y) if not isinstance(y, list) else y
    kf = KFold(n_splits=cv)
    splits = list(kf.split(X, y_list))
    train_scores_all = []
    val_scores_all = []
    for val in param_range:
        est = copy.deepcopy(estimator)
        setattr(est, param_name, val)
        t_scores, v_scores = [], []
        for train_idx, val_idx in splits:
            t_scores.append(_score_estimator(est, X, y_list, train_idx, train_idx, scoring))
            v_scores.append(_score_estimator(est, X, y_list, train_idx, val_idx, scoring))
        train_scores_all.append(t_scores)
        val_scores_all.append(v_scores)
    return train_scores_all, val_scores_all


__all__ = [
    'train_test_split',
    'KFold', 'StratifiedKFold', 'LeaveOneOut', 'RepeatedKFold',
    'cross_val_score', 'cross_validate',
    'learning_curve', 'validation_curve',
    'ShuffleSplit', 'TimeSeriesSplit',
]
