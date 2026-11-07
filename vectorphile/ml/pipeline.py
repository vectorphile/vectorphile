"""Pipeline, ColumnTransformer, FeatureUnion."""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Union


class Pipeline:
    """Chain estimators and transformers into a single object."""
    def __init__(self, steps: List[Tuple[str, object]], memory=None, verbose=False):
        self.steps = steps
        self.memory = memory
        self.verbose = verbose
        self._validate_steps()

    def _validate_steps(self):
        for name, est in self.steps[:-1]:
            if not (hasattr(est, 'fit') and hasattr(est, 'transform')):
                raise ValueError(f"All intermediate steps must have fit and transform. '{name}' does not.")

    @property
    def named_steps(self):
        return dict(self.steps)

    def fit(self, X, y=None, **fit_params):
        X_transformed = X
        for name, step in self.steps[:-1]:
            if self.verbose:
                print(f'[Pipeline] Fitting step: {name}')
            step.fit(X_transformed, y)
            X_transformed = step.transform(X_transformed)
        final_name, final_step = self.steps[-1]
        if self.verbose:
            print(f'[Pipeline] Fitting final step: {final_name}')
        final_step.fit(X_transformed, y, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        X_transformed = X
        for name, step in self.steps[:-1]:
            if hasattr(step, 'fit_transform'):
                X_transformed = step.fit_transform(X_transformed, y)
            else:
                step.fit(X_transformed, y)
                X_transformed = step.transform(X_transformed)
        _, final_step = self.steps[-1]
        if hasattr(final_step, 'fit_transform'):
            return final_step.fit_transform(X_transformed, y, **fit_params)
        final_step.fit(X_transformed, y, **fit_params)
        return final_step.transform(X_transformed)

    def predict(self, X):
        X_t = self._transform_all_but_last(X)
        return self.steps[-1][1].predict(X_t)

    def predict_proba(self, X):
        X_t = self._transform_all_but_last(X)
        return self.steps[-1][1].predict_proba(X_t)

    def predict_log_proba(self, X):
        X_t = self._transform_all_but_last(X)
        return np.log(self.steps[-1][1].predict_proba(X_t) + 1e-12)

    def transform(self, X):
        X_t = self._transform_all_but_last(X)
        return self.steps[-1][1].transform(X_t)

    def _transform_all_but_last(self, X):
        X_t = X
        for name, step in self.steps[:-1]:
            X_t = step.transform(X_t)
        return X_t

    def score(self, X, y):
        X_t = self._transform_all_but_last(X)
        return self.steps[-1][1].score(X_t, y)

    def set_params(self, **params):
        for key, val in params.items():
            if '__' in key:
                step_name, param_name = key.split('__', 1)
                step = self.named_steps[step_name]
                step.set_params(**{param_name: val})
            else:
                setattr(self, key, val)
        return self

    def get_params(self, deep=True):
        params = {'steps': self.steps}
        if deep:
            for name, step in self.steps:
                if hasattr(step, 'get_params'):
                    for k, v in step.get_params(deep).items():
                        params[f'{name}__{k}'] = v
        return params

    def __repr__(self):
        step_str = '\n  '.join(f'({n}): {e}' for n, e in self.steps)
        return f'Pipeline(\n  {step_str}\n)'

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.named_steps[key]
        return self.steps[key]

    def __len__(self):
        return len(self.steps)


class ColumnTransformer:
    """Apply different transformers to different columns."""
    def __init__(self, transformers: List[Tuple], remainder='drop', sparse_threshold=0.3,
                 n_jobs=None, transformer_weights=None, verbose=False):
        # transformers: list of (name, transformer, columns)
        # columns: list of int indices or str column names or slice
        self.transformers = transformers
        self.remainder = remainder
        self.sparse_threshold = sparse_threshold
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose
        self.transformers_ = []
        self._n_features = None
        self._all_cols = None

    def _get_col_indices(self, X, columns):
        if isinstance(columns, slice):
            return list(range(*columns.indices(X.shape[1])))
        elif isinstance(columns, (list, np.ndarray)):
            if len(columns) == 0:
                return []
            if isinstance(columns[0], str):
                raise ValueError("String column names require pandas DataFrame input")
            return list(columns)
        else:
            return [int(columns)]

    def fit(self, X, y=None):
        X = np.array(X, dtype=np.float64)
        self._n_features = X.shape[1]
        self.transformers_ = []
        self._all_cols = set()
        for name, transformer, columns in self.transformers:
            col_idx = self._get_col_indices(X, columns)
            self._all_cols.update(col_idx)
            import copy
            t = copy.deepcopy(transformer)
            if transformer == 'passthrough':
                t = _Passthrough()
            elif transformer == 'drop':
                t = None
            if t is not None:
                t.fit(X[:, col_idx], y)
            self.transformers_.append((name, t, col_idx))
        # Remainder columns
        all_col_set = self._all_cols
        if self.remainder == 'passthrough':
            rem_cols = [i for i in range(self._n_features) if i not in all_col_set]
            if rem_cols:
                self.transformers_.append(('remainder', _Passthrough(), rem_cols))
        return self

    def transform(self, X):
        X = np.array(X, dtype=np.float64)
        parts = []
        for name, t, col_idx in self.transformers_:
            if t is None:
                continue
            part = t.transform(X[:, col_idx])
            parts.append(part)
        return np.hstack(parts) if parts else np.empty((len(X), 0))

    def fit_transform(self, X, y=None):
        X = np.array(X, dtype=np.float64)
        self._n_features = X.shape[1]
        self.transformers_ = []
        self._all_cols = set()
        parts = []
        for name, transformer, columns in self.transformers:
            col_idx = self._get_col_indices(X, columns)
            self._all_cols.update(col_idx)
            import copy
            t = copy.deepcopy(transformer)
            if transformer == 'passthrough':
                t = _Passthrough()
            elif transformer == 'drop':
                t = None
            if t is not None:
                if hasattr(t, 'fit_transform'):
                    part = t.fit_transform(X[:, col_idx], y)
                else:
                    t.fit(X[:, col_idx], y)
                    part = t.transform(X[:, col_idx])
                parts.append(part)
                self.transformers_.append((name, t, col_idx))
        if self.remainder == 'passthrough':
            rem_cols = [i for i in range(self._n_features) if i not in self._all_cols]
            if rem_cols:
                parts.append(X[:, rem_cols])
                self.transformers_.append(('remainder', _Passthrough(), rem_cols))
        return np.hstack(parts) if parts else np.empty((len(X), 0))

    def __repr__(self):
        return f'ColumnTransformer(transformers={[n for n,_,_ in self.transformers]})'


class _Passthrough:
    def fit(self, X, y=None): return self
    def transform(self, X): return np.array(X, dtype=np.float64)
    def fit_transform(self, X, y=None): return self.transform(X)


class FeatureUnion:
    """Concatenate results of multiple transformers."""
    def __init__(self, transformer_list: List[Tuple], n_jobs=None,
                 transformer_weights=None, verbose=False):
        self.transformer_list = transformer_list
        self.transformer_weights = transformer_weights
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.transformers_ = []

    def fit(self, X, y=None):
        import copy
        self.transformers_ = []
        for name, t in self.transformer_list:
            cloned = copy.deepcopy(t)
            cloned.fit(X, y)
            self.transformers_.append((name, cloned))
        return self

    def transform(self, X):
        parts = []
        for i, (name, t) in enumerate(self.transformers_):
            part = t.transform(X)
            w = self.transformer_weights[name] if self.transformer_weights else 1.0
            parts.append(np.array(part, dtype=np.float64) * w)
        return np.hstack(parts)

    def fit_transform(self, X, y=None):
        import copy
        self.transformers_ = []
        parts = []
        for name, t in self.transformer_list:
            cloned = copy.deepcopy(t)
            if hasattr(cloned, 'fit_transform'):
                part = cloned.fit_transform(X, y)
            else:
                cloned.fit(X, y)
                part = cloned.transform(X)
            w = self.transformer_weights[name] if self.transformer_weights else 1.0
            parts.append(np.array(part, dtype=np.float64) * w)
            self.transformers_.append((name, cloned))
        return np.hstack(parts)

    def get_feature_names_out(self):
        names = []
        for name, t in self.transformers_:
            if hasattr(t, 'get_feature_names_out'):
                for feat in t.get_feature_names_out():
                    names.append(f'{name}__{feat}')
            else:
                names.append(name)
        return np.array(names)

    def __repr__(self):
        return f'FeatureUnion([{", ".join(n for n,_ in self.transformer_list)}])'


def make_pipeline(*steps, memory=None, verbose=False) -> Pipeline:
    names = [type(s).__name__.lower() for s in steps]
    return Pipeline(list(zip(names, steps)), memory=memory, verbose=verbose)


def make_union(*transformers, n_jobs=None, verbose=False) -> FeatureUnion:
    names = [type(t).__name__.lower() for t in transformers]
    return FeatureUnion(list(zip(names, transformers)), n_jobs=n_jobs, verbose=verbose)
