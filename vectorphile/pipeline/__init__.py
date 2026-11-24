"""vectorphile.pipeline — sklearn-compatible ML Pipeline.

All computations use pure Python. No external dependencies.
"""
from __future__ import annotations
import copy
import itertools
import random
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union


def _to_list(X):
    if hasattr(X, 'to_flat_list'):
        return X.to_flat_list()
    if hasattr(X, 'tolist'):
        return X.tolist()
    return list(X) if not isinstance(X, list) else X


def _subset(X, indices: List[int]):
    if isinstance(X, list):
        return [X[i] for i in indices]
    if hasattr(X, '__getitem__'):
        try:
            return X[indices]
        except Exception:
            pass
    return [X[i] for i in indices]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """Chain of transformers ending in an estimator."""

    def __init__(self, steps: List[Tuple[str, Any]]):
        self.steps = steps
        self._validate_steps()

    def _validate_steps(self) -> None:
        names = [n for n, _ in self.steps]
        if len(set(names)) != len(names):
            raise ValueError(f'Duplicate step names: {names}')

    @property
    def named_steps(self) -> Dict[str, Any]:
        return dict(self.steps)

    def _transformers(self):
        return self.steps[:-1]

    def _final_estimator(self):
        return self.steps[-1][1] if self.steps else None

    def fit(self, X, y=None, **fit_params) -> 'Pipeline':
        X_t = X
        for name, step in self._transformers():
            if hasattr(step, 'fit_transform'):
                X_t = step.fit_transform(X_t, y)
            else:
                step.fit(X_t, y)
                X_t = step.transform(X_t)
        final = self._final_estimator()
        if final is not None:
            if y is not None:
                final.fit(X_t, y, **fit_params) if fit_params else final.fit(X_t, y)
            else:
                final.fit(X_t, **fit_params) if fit_params else final.fit(X_t)
        return self

    def transform(self, X):
        X_t = X
        for name, step in self._transformers():
            X_t = step.transform(X_t)
        return X_t

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        return self._final_estimator().predict(self.transform(X))

    def predict_proba(self, X):
        return self._final_estimator().predict_proba(self.transform(X))

    def predict_log_proba(self, X):
        proba = self.predict_proba(X)
        if isinstance(proba, list):
            return [[math.log(max(v, 1e-10)) for v in row] for row in proba]
        return proba

    def score(self, X, y) -> float:
        X_t = self.transform(X)
        final = self._final_estimator()
        if hasattr(final, 'score'):
            return final.score(X_t, y)
        preds = final.predict(X_t)
        preds_list = _to_list(preds) if not isinstance(preds, list) else preds
        y_list = _to_list(y) if not isinstance(y, list) else y
        return sum(1 for p, t in zip(preds_list, y_list) if p == t) / max(len(y_list), 1)

    def get_params(self, deep: bool = True) -> Dict:
        params = {}
        for name, step in self.steps:
            params[name] = step
            if deep and hasattr(step, 'get_params'):
                for k, v in step.get_params().items():
                    params[f'{name}__{k}'] = v
        return params

    def set_params(self, **params) -> 'Pipeline':
        for key, val in params.items():
            if '__' in key:
                step_name, param = key.split('__', 1)
                step = self.named_steps.get(step_name)
                if step and hasattr(step, 'set_params'):
                    step.set_params(**{param: val})
            else:
                for i, (n, _) in enumerate(self.steps):
                    if n == key:
                        self.steps[i] = (key, val)
        return self

    def __repr__(self) -> str:
        step_reprs = ', '.join(f'{n}: {type(s).__name__}' for n, s in self.steps)
        return f'Pipeline([{step_reprs}])'


def make_pipeline(*steps) -> Pipeline:
    counts: Dict[str, int] = {}
    result = []
    for s in steps:
        name = type(s).__name__.lower()
        c = counts.get(name, 0)
        counts[name] = c + 1
        result.append((f'{name}{c}' if c else name, s))
    return Pipeline(result)


# ---------------------------------------------------------------------------
# FeatureUnion
# ---------------------------------------------------------------------------

class FeatureUnion:
    """Concatenate feature matrices from multiple transformers."""

    def __init__(self, transformer_list: List[Tuple[str, Any]], n_jobs: int = 1):
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs

    def fit(self, X, y=None) -> 'FeatureUnion':
        for name, t in self.transformer_list:
            t.fit(X, y)
        return self

    def transform(self, X) -> List[List[float]]:
        parts = []
        for name, t in self.transformer_list:
            result = t.transform(X)
            if isinstance(result, list):
                parts.append(result)
            elif hasattr(result, 'to_flat_list'):
                n = result.shape[0] if hasattr(result, 'shape') else len(result)
                d = result.shape[1] if hasattr(result, 'shape') and len(result.shape) > 1 else 1
                flat = result.to_flat_list()
                parts.append([[flat[i * d + j] for j in range(d)] for i in range(n)])
            else:
                lst = list(result)
                parts.append([[float(v) for v in row] if isinstance(row, (list, tuple)) else [float(row)]
                               for row in lst])
        # Concatenate column-wise
        if not parts:
            return []
        n = len(parts[0])
        return [[v for part in parts for v in (part[i] if isinstance(part[i], list) else [part[i]])]
                for i in range(n)]

    def fit_transform(self, X, y=None) -> List[List[float]]:
        self.fit(X, y)
        return self.transform(X)

    def get_params(self, deep: bool = True) -> Dict:
        return {n: t for n, t in self.transformer_list}


def make_union(*transformers) -> FeatureUnion:
    named = [(type(t).__name__.lower(), t) for t in transformers]
    return FeatureUnion(named)


# ---------------------------------------------------------------------------
# ColumnTransformer
# ---------------------------------------------------------------------------

class ColumnTransformer:
    """Apply different transformers to different columns."""

    def __init__(self, transformers: List[Tuple[str, Any, Any]],
                 remainder: str = 'drop'):
        self.transformers = transformers
        self.remainder = remainder

    def _get_cols(self, X: List[List], cols) -> List[List]:
        if isinstance(cols, (list, tuple)):
            return [[row[j] for j in cols] for row in X]
        elif isinstance(cols, slice):
            return [row[cols] for row in X]
        return [[row[cols]] for row in X]

    def fit(self, X, y=None) -> 'ColumnTransformer':
        X2d = X if isinstance(X, list) else list(X)
        for name, t, cols in self.transformers:
            X_sub = self._get_cols(X2d, cols)
            t.fit(X_sub, y)
        return self

    def transform(self, X) -> List[List[float]]:
        X2d = X if isinstance(X, list) else list(X)
        p = len(X2d[0]) if X2d else 0
        parts = []
        used_cols: set = set()
        for name, t, cols in self.transformers:
            if isinstance(cols, (list, tuple)):
                used_cols.update(cols)
            X_sub = self._get_cols(X2d, cols)
            result = t.transform(X_sub)
            if not isinstance(result, list):
                result = list(result)
            parts.append(result)
        if self.remainder == 'passthrough':
            remain = [j for j in range(p) if j not in used_cols]
            if remain:
                parts.append([[row[j] for j in remain] for row in X2d])
        if not parts:
            return []
        n = len(parts[0])
        return [[v for part in parts for v in (part[i] if isinstance(part[i], list) else [part[i]])]
                for i in range(n)]

    def fit_transform(self, X, y=None) -> List[List[float]]:
        self.fit(X, y)
        return self.transform(X)


# ---------------------------------------------------------------------------
# cross_val_score (local version for GridSearchCV)
# ---------------------------------------------------------------------------

def _kfold_split(n: int, k: int, shuffle: bool = True,
                 random_state: Optional[int] = None) -> List[Tuple[List[int], List[int]]]:
    indices = list(range(n))
    if shuffle:
        rng = random.Random(random_state)
        rng.shuffle(indices)
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        val_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        folds.append((train_idx, val_idx))
    return folds


def cross_val_score(estimator, X, y, cv: int = 5,
                    scoring: str = 'accuracy') -> List[float]:
    y_list = list(y) if not isinstance(y, list) else y
    folds = _kfold_split(len(y_list), cv)
    scores = []
    for train_idx, val_idx in folds:
        est = copy.deepcopy(estimator)
        X_tr = _subset(X, train_idx)
        X_v = _subset(X, val_idx)
        y_tr = [y_list[i] for i in train_idx]
        y_v = [y_list[i] for i in val_idx]
        est.fit(X_tr, y_tr)
        preds = est.predict(X_v)
        preds = _to_list(preds) if not isinstance(preds, list) else preds
        if scoring == 'accuracy':
            score = sum(1 for p, t in zip(preds, y_v) if p == t) / max(len(y_v), 1)
        elif scoring == 'r2':
            pf = [float(p) for p in preds]
            yf = [float(t) for t in y_v]
            mean_y = sum(yf) / len(yf)
            ss_res = sum((yf[i] - pf[i]) ** 2 for i in range(len(yf)))
            ss_tot = sum((v - mean_y) ** 2 for v in yf) + 1e-10
            score = 1.0 - ss_res / ss_tot
        elif scoring in ('neg_mse', 'neg_mean_squared_error'):
            pf = [float(p) for p in preds]
            yf = [float(t) for t in y_v]
            score = -sum((pf[i] - yf[i]) ** 2 for i in range(len(yf))) / len(yf)
        else:
            score = est.score(X_v, y_v) if hasattr(est, 'score') else 0.0
        scores.append(score)
    return scores


# ---------------------------------------------------------------------------
# GridSearchCV
# ---------------------------------------------------------------------------

class GridSearchCV:
    """Exhaustive search over a hyperparameter grid."""

    def __init__(self, estimator, param_grid: Dict[str, List],
                 cv: int = 5, scoring: str = 'accuracy', verbose: int = 0):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.best_params_: Optional[Dict] = None
        self.best_score_: float = -float('inf')
        self.best_estimator_ = None
        self.cv_results_: List[Dict] = []

    def _param_combinations(self) -> List[Dict]:
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        return [dict(zip(keys, c)) for c in itertools.product(*values)]

    def fit(self, X, y) -> 'GridSearchCV':
        y_list = list(y) if not isinstance(y, list) else y
        for params in self._param_combinations():
            est = copy.deepcopy(self.estimator)
            if hasattr(est, 'set_params'):
                est.set_params(**params)
            else:
                for k, v in params.items():
                    setattr(est, k, v)
            scores = cross_val_score(est, X, y_list, cv=self.cv, scoring=self.scoring)
            mean_score = sum(scores) / len(scores)
            std_score = math.sqrt(sum((s - mean_score) ** 2 for s in scores) / len(scores))
            self.cv_results_.append({'params': params, 'mean_score': mean_score, 'std_score': std_score})
            if self.verbose:
                print(f'  {params} → {mean_score:.4f}')
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params
                self.best_estimator_ = copy.deepcopy(est)
        if self.best_estimator_ is not None:
            self.best_estimator_.fit(X, y_list)
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def score(self, X, y) -> float:
        if hasattr(self.best_estimator_, 'score'):
            return self.best_estimator_.score(X, y)
        preds = _to_list(self.predict(X))
        y_list = list(y) if not isinstance(y, list) else y
        return sum(1 for p, t in zip(preds, y_list) if p == t) / max(len(y_list), 1)


# ---------------------------------------------------------------------------
# RandomizedSearchCV
# ---------------------------------------------------------------------------

class RandomizedSearchCV:
    """Random search over hyperparameter distributions."""

    def __init__(self, estimator, param_distributions: Dict,
                 n_iter: int = 10, cv: int = 5,
                 scoring: str = 'accuracy', random_state: Optional[int] = None,
                 verbose: int = 0):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.verbose = verbose
        self.best_params_: Optional[Dict] = None
        self.best_score_: float = -float('inf')
        self.best_estimator_ = None
        self.cv_results_: List[Dict] = []

    def _sample_params(self, rng: random.Random) -> Dict:
        params = {}
        for k, dist in self.param_distributions.items():
            if hasattr(dist, 'rvs'):
                params[k] = dist.rvs()
            elif isinstance(dist, (list, tuple)):
                params[k] = rng.choice(dist)
            else:
                params[k] = dist
        return params

    def fit(self, X, y) -> 'RandomizedSearchCV':
        y_list = list(y) if not isinstance(y, list) else y
        rng = random.Random(self.random_state)
        for _ in range(self.n_iter):
            params = self._sample_params(rng)
            est = copy.deepcopy(self.estimator)
            if hasattr(est, 'set_params'):
                est.set_params(**params)
            else:
                for k, v in params.items():
                    setattr(est, k, v)
            scores = cross_val_score(est, X, y_list, cv=self.cv, scoring=self.scoring)
            mean_score = sum(scores) / len(scores)
            std_score = math.sqrt(sum((s - mean_score) ** 2 for s in scores) / len(scores))
            self.cv_results_.append({'params': params, 'mean_score': mean_score, 'std_score': std_score})
            if self.verbose:
                print(f'  {params} → {mean_score:.4f}')
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = params
                self.best_estimator_ = copy.deepcopy(est)
        if self.best_estimator_ is not None:
            self.best_estimator_.fit(X, y_list)
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)


__all__ = [
    'Pipeline', 'make_pipeline',
    'FeatureUnion', 'make_union',
    'ColumnTransformer',
    'cross_val_score',
    'GridSearchCV', 'RandomizedSearchCV',
]
