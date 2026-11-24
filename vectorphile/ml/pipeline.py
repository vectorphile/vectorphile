"""Pipeline and ColumnTransformer — pure Python."""
import math
from typing import List, Tuple, Optional


def _hstack(arrays):
    if not arrays:
        return []
    n = len(arrays[0])
    result = []
    for i in range(n):
        row = []
        for arr in arrays:
            v = arr[i]
            if isinstance(v, (list, tuple)):
                row.extend(v)
            else:
                row.append(v)
        result.append(row)
    return result


class Pipeline:
    def __init__(self, steps: List[Tuple], memory=None):
        self.steps = steps
        self.memory = memory

    @property
    def named_steps(self):
        return dict(self.steps)

    def fit(self, X, y=None, **fit_params):
        X_t = X
        for name, step in self.steps[:-1]:
            if hasattr(step, 'fit_transform'):
                X_t = step.fit_transform(X_t, y)
            else:
                step.fit(X_t, y)
                X_t = step.transform(X_t)
        final_name, final_step = self.steps[-1]
        final_step.fit(X_t, y, **fit_params)
        return self

    def predict(self, X):
        X_t = X
        for name, step in self.steps[:-1]:
            X_t = step.transform(X_t)
        return self.steps[-1][1].predict(X_t)

    def predict_proba(self, X):
        X_t = X
        for name, step in self.steps[:-1]:
            X_t = step.transform(X_t)
        return self.steps[-1][1].predict_proba(X_t)

    def predict_log_proba(self, X):
        proba = self.predict_proba(X)
        if isinstance(proba[0], (list, tuple)):
            return [[math.log(max(p, 1e-15)) for p in row] for row in proba]
        return [math.log(max(p, 1e-15)) for p in proba]

    def score(self, X, y):
        X_t = X
        for name, step in self.steps[:-1]:
            X_t = step.transform(X_t)
        return self.steps[-1][1].score(X_t, y)

    def transform(self, X):
        X_t = X
        for name, step in self.steps:
            X_t = step.transform(X_t)
        return X_t

    def fit_transform(self, X, y=None):
        X_t = X
        for name, step in self.steps[:-1]:
            if hasattr(step, 'fit_transform'):
                X_t = step.fit_transform(X_t, y)
            else:
                step.fit(X_t, y)
                X_t = step.transform(X_t)
        final_name, final_step = self.steps[-1]
        if hasattr(final_step, 'fit_transform'):
            return final_step.fit_transform(X_t, y)
        final_step.fit(X_t, y)
        return final_step.transform(X_t)

    def __getitem__(self, key):
        return dict(self.steps)[key]

    def set_params(self, **params):
        for key, val in params.items():
            if '__' in key:
                step_name, param = key.split('__', 1)
                step = dict(self.steps)[step_name]
                setattr(step, param, val)
            else:
                setattr(self, key, val)
        return self

    def get_params(self, deep=True):
        params = {}
        for name, step in self.steps:
            if deep and hasattr(step, 'get_params'):
                for k, v in step.get_params(deep=True).items():
                    params[f'{name}__{k}'] = v
            params[name] = step
        return params


class _Passthrough:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [[float(v) for v in row] for row in X]

    def fit_transform(self, X, y=None):
        return self.transform(X)


class ColumnTransformer:
    def __init__(self, transformers, remainder='drop', sparse_threshold=0.3, n_jobs=None):
        self.transformers = transformers
        self.remainder = remainder
        self.sparse_threshold = sparse_threshold

    def fit(self, X, y=None):
        X2 = [list(row) for row in X]
        p = len(X2[0]) if X2 else 0
        self._used_cols = set()
        for name, trans, cols in self.transformers:
            if isinstance(cols, slice):
                cols = list(range(*cols.indices(p)))
            elif isinstance(cols, int):
                cols = [cols]
            self._used_cols.update(cols)
            if trans == 'passthrough':
                trans = _Passthrough()
                # update reference in list
            sub = [[X2[i][j] for j in cols] for i in range(len(X2))]
            if trans != 'drop':
                trans.fit(sub, y)
        if self.remainder == 'passthrough':
            self._remainder_cols = [j for j in range(p) if j not in self._used_cols]
        return self

    def transform(self, X):
        X2 = [list(row) for row in X]
        parts = []
        for name, trans, cols in self.transformers:
            p = len(X2[0]) if X2 else 0
            if isinstance(cols, slice):
                cols = list(range(*cols.indices(p)))
            elif isinstance(cols, int):
                cols = [cols]
            if trans == 'drop':
                continue
            sub = [[X2[i][j] for j in cols] for i in range(len(X2))]
            if trans == 'passthrough':
                parts.append([[float(v) for v in row] for row in sub])
            else:
                parts.append(trans.transform(sub))
        if self.remainder == 'passthrough':
            rem_cols = getattr(self, '_remainder_cols', [])
            if rem_cols:
                parts.append([[float(X2[i][j]) for j in rem_cols] for i in range(len(X2))])
        return _hstack(parts) if parts else [[] for _ in X2]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class FeatureUnion:
    def __init__(self, transformer_list, n_jobs=None, transformer_weights=None):
        self.transformer_list = transformer_list
        self.transformer_weights = transformer_weights

    def fit(self, X, y=None):
        for name, trans in self.transformer_list:
            trans.fit(X, y)
        return self

    def transform(self, X):
        parts = []
        for name, trans in self.transformer_list:
            part = trans.transform(X)
            w = (self.transformer_weights or {}).get(name, 1.0)
            if w != 1.0:
                part = [[v * w for v in row] for row in part]
            parts.append(part)
        return _hstack(parts)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def make_pipeline(*steps):
    """Create Pipeline from estimators without naming them."""
    named = [(type(s).__name__.lower() + str(i), s) for i, s in enumerate(steps)]
    return Pipeline(named)


def make_union(*transformers, n_jobs=None, verbose=False):
    """Create FeatureUnion from transformers without naming them."""
    named = [(type(t).__name__.lower() + str(i), t) for i, t in enumerate(transformers)]
    return FeatureUnion(named)
