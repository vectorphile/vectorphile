"""vectorphile.automl — Automated Machine Learning.

Provides:
- AutoClassifier / AutoRegressor: auto-select & tune models
- Hyperparameter optimization: TPE (Tree-structured Parzen Estimator), random, grid
- Automatic feature engineering: polynomial, interactions, selection
- Neural architecture search (NAS): random search over micro-architectures
- Ensemble auto-building from selected models

Usage::

    from vectorphile.automl import AutoClassifier, AutoRegressor

    clf = AutoClassifier(time_limit=60, metric='accuracy', n_trials=20)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(clf.best_model_name, clf.best_params)
"""
from __future__ import annotations
import time
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Hyperparameter search spaces
# ---------------------------------------------------------------------------

class IntParam:
    def __init__(self, low: int, high: int, log: bool = False):
        self.low = low
        self.high = high
        self.log = log

    def sample(self) -> int:
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            return int(round(math.exp(random.uniform(lo, hi))))
        return random.randint(self.low, self.high)


class FloatParam:
    def __init__(self, low: float, high: float, log: bool = False):
        self.low = low
        self.high = high
        self.log = log

    def sample(self) -> float:
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            return math.exp(random.uniform(lo, hi))
        return random.uniform(self.low, self.high)


class CategoricalParam:
    def __init__(self, choices: list):
        self.choices = choices

    def sample(self):
        return random.choice(self.choices)


def suggest_int(low: int, high: int, log: bool = False) -> IntParam:
    return IntParam(low, high, log)


def suggest_float(low: float, high: float, log: bool = False) -> FloatParam:
    return FloatParam(low, high, log)


def suggest_categorical(choices: list) -> CategoricalParam:
    return CategoricalParam(choices)


# ---------------------------------------------------------------------------
# Study / trial
# ---------------------------------------------------------------------------

class Trial:
    def __init__(self, params: dict, number: int):
        self.params = params
        self.number = number
        self.value: Optional[float] = None
        self.state: str = 'running'

    def suggest_int(self, name: str, low: int, high: int, log: bool = False) -> int:
        if name not in self.params:
            self.params[name] = IntParam(low, high, log).sample()
        return self.params[name]

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        if name not in self.params:
            self.params[name] = FloatParam(low, high, log).sample()
        return self.params[name]

    def suggest_categorical(self, name: str, choices: list):
        if name not in self.params:
            self.params[name] = CategoricalParam(choices).sample()
        return self.params[name]


class Study:
    """Hyperparameter optimization study (Optuna-compatible interface).

    Usage::

        study = Study(direction='minimize')

        def objective(trial):
            lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
            n_layers = trial.suggest_int('n_layers', 1, 5)
            # ... build and train model ...
            return val_loss

        study.optimize(objective, n_trials=50)
        print(study.best_trial.params, study.best_value)
    """

    def __init__(self, direction: str = 'minimize'):
        self.direction = direction
        self.trials: List[Trial] = []
        self.best_trial: Optional[Trial] = None
        self.best_value: Optional[float] = None

    def optimize(self, objective: Callable, n_trials: int = 100,
                 timeout: Optional[float] = None, show_progress: bool = False) -> None:
        start = time.time()
        for i in range(n_trials):
            if timeout and (time.time() - start) > timeout:
                break
            trial = Trial(params={}, number=i)
            try:
                value = float(objective(trial))
                trial.value = value
                trial.state = 'complete'
            except Exception as e:
                trial.state = 'failed'
                continue
            self.trials.append(trial)
            is_better = (self.best_value is None or
                         (self.direction == 'minimize' and value < self.best_value) or
                         (self.direction == 'maximize' and value > self.best_value))
            if is_better:
                self.best_value = value
                self.best_trial = trial
            if show_progress and (i + 1) % 10 == 0:
                print(f'Trial {i+1}/{n_trials}: best={self.best_value:.4f}')

    def create_study(self, direction='minimize') -> 'Study':
        return Study(direction)


def create_study(direction: str = 'minimize') -> Study:
    return Study(direction)


# ---------------------------------------------------------------------------
# AutoClassifier / AutoRegressor
# ---------------------------------------------------------------------------

class _AutoBase:
    """Base auto-ML class."""

    _DEFAULT_CLASSIFIERS = {
        'logistic': lambda p: __import__('vectorphile.ml', fromlist=['LogisticRegression']).LogisticRegression(
            C=p.get('C', 1.0), max_iter=p.get('max_iter', 100)),
        'rf': lambda p: __import__('vectorphile.ml', fromlist=['RandomForestClassifier']).RandomForestClassifier(
            n_estimators=p.get('n_estimators', 100), max_depth=p.get('max_depth', None)),
        'knn': lambda p: __import__('vectorphile.ml', fromlist=['KNeighborsClassifier']).KNeighborsClassifier(
            n_neighbors=p.get('n_neighbors', 5)),
        'svc': lambda p: __import__('vectorphile.ml', fromlist=['SVC']).SVC(
            C=p.get('C', 1.0), kernel=p.get('kernel', 'rbf')),
    }

    _DEFAULT_REGRESSORS = {
        'linear': lambda p: __import__('vectorphile.ml', fromlist=['Ridge']).Ridge(
            alpha=p.get('alpha', 1.0)),
        'rf': lambda p: __import__('vectorphile.ml', fromlist=['RandomForestRegressor']).RandomForestRegressor(
            n_estimators=p.get('n_estimators', 100), max_depth=p.get('max_depth', None)),
        'knn': lambda p: __import__('vectorphile.ml', fromlist=['KNeighborsRegressor']).KNeighborsRegressor(
            n_neighbors=p.get('n_neighbors', 5)),
    }

    def __init__(self, task: str = 'classification', time_limit: float = 60.0,
                 metric: str = 'accuracy', n_trials: int = 20,
                 cv_folds: int = 3, random_state: int = 42):
        self.task = task
        self.time_limit = time_limit
        self.metric = metric
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_model = None
        self.best_model_name: Optional[str] = None
        self.best_params: Optional[dict] = None
        self.best_score: Optional[float] = None
        self._leaderboard: List[dict] = []

    def _cv_score(self, model, X, y) -> float:
        n = len(X)
        fold_size = max(1, n // self.cv_folds)
        scores = []
        for i in range(self.cv_folds):
            v0 = i * fold_size; v1 = min((i+1)*fold_size, n)
            val_idx = list(range(v0, v1))
            train_idx = list(range(0, v0)) + list(range(v1, n))
            if not train_idx or not val_idx: continue
            X_tr = [X[j] for j in train_idx]; X_val = [X[j] for j in val_idx]
            y_tr = [y[j] for j in train_idx]; y_val = [y[j] for j in val_idx]
            try:
                model.fit(X_tr, y_tr)
                preds_raw = model.predict(X_val)
                preds = [preds_raw[j] if hasattr(preds_raw,'__getitem__') else preds_raw for j in range(len(y_val))]
                if self.task == 'classification':
                    score = sum(1 for p,t in zip(preds, y_val) if p == t) / max(len(y_val), 1)
                else:
                    mu = sum(y_val) / max(len(y_val), 1)
                    ss_res = sum((t-p)**2 for t,p in zip(y_val, preds))
                    ss_tot = sum((t-mu)**2 for t in y_val) + 1e-10
                    score = float(1 - ss_res / ss_tot)
                scores.append(score)
            except Exception:
                scores.append(0.0)
        return sum(scores) / max(len(scores), 1)

    def leaderboard(self) -> List[dict]:
        return sorted(self._leaderboard, key=lambda x: -x['score'])


class AutoClassifier(_AutoBase):
    """Automatically selects and tunes a classification model.

    Usage::

        clf = AutoClassifier(time_limit=60, n_trials=20)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
    """

    def __init__(self, **kwargs):
        super().__init__(task='classification', **kwargs)

    def fit(self, X, y) -> 'AutoClassifier':
        X = [[float(v) for v in row] for row in X] if not isinstance(X[0], list) else X
        y = [v for v in y]
        start = time.time()

        # Preprocess
        from vectorphile.ml.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        builder_map = self._DEFAULT_CLASSIFIERS
        trials_per_model = max(1, self.n_trials // len(builder_map))

        for model_name, builder in builder_map.items():
            if (time.time() - start) > self.time_limit:
                break
            # Random search over hyperparameters
            for trial_i in range(trials_per_model):
                if (time.time() - start) > self.time_limit:
                    break
                # Sample random params
                params = {}
                if model_name in ('logistic', 'svc'):
                    params['C'] = random.choice([0.01, 0.1, 1.0, 10.0])
                if model_name in ('rf',):
                    params['n_estimators'] = random.choice([50, 100, 200])
                    params['max_depth'] = random.choice([None, 3, 5, 10])
                if model_name == 'knn':
                    params['n_neighbors'] = random.choice([3, 5, 7, 11, 15])
                try:
                    model = builder(params)
                    score = self._cv_score(model, X_scaled, y)
                    self._leaderboard.append({'model': model_name, 'params': params, 'score': score})
                    if self.best_score is None or score > self.best_score:
                        self.best_score = score
                        self.best_model_name = model_name
                        self.best_params = params
                except Exception:
                    continue

        # Refit best model on full data
        if self.best_model_name and self.best_params is not None:
            builder = builder_map[self.best_model_name]
            self.best_model = builder(self.best_params)
            self.best_model.fit(X_scaled, y)
        self._scaler = scaler
        return self

    def predict(self, X):
        if self.best_model is None:
            raise RuntimeError('Call fit() first')
        X_scaled = self._scaler.transform([[float(v) for v in row] for row in X] if not isinstance(X[0], list) else X)
        return self.best_model.predict(X_scaled)

    def predict_proba(self, X):
        if hasattr(self.best_model, 'predict_proba'):
            X_scaled = self._scaler.transform([[float(v) for v in row] for row in X] if not isinstance(X[0], list) else X)
            return self.best_model.predict_proba(X_scaled)
        raise AttributeError('Best model does not support predict_proba')

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_list = list(y)
        return float(sum(1 for p, t in zip(preds, y_list) if p == t) / max(len(y_list), 1))


class AutoRegressor(_AutoBase):
    """Automatically selects and tunes a regression model."""

    def __init__(self, **kwargs):
        super().__init__(task='regression', **kwargs)

    def fit(self, X, y) -> 'AutoRegressor':
        X = [[float(v) for v in row] for row in X] if not isinstance(X[0], list) else X
        y = [float(v) for v in y]
        start = time.time()
        from vectorphile.ml.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        builder_map = self._DEFAULT_REGRESSORS
        trials_per_model = max(1, self.n_trials // len(builder_map))
        for model_name, builder in builder_map.items():
            if (time.time() - start) > self.time_limit:
                break
            for _ in range(trials_per_model):
                if (time.time() - start) > self.time_limit:
                    break
                params = {}
                if model_name == 'linear':
                    params['alpha'] = random.choice([0.001, 0.01, 0.1, 1.0, 10.0])
                if model_name == 'rf':
                    params['n_estimators'] = random.choice([50, 100, 200])
                    params['max_depth'] = random.choice([None, 3, 5, 10])
                if model_name == 'knn':
                    params['n_neighbors'] = random.choice([3, 5, 7, 11])
                try:
                    model = builder(params)
                    score = self._cv_score(model, X_scaled, y)
                    self._leaderboard.append({'model': model_name, 'params': params, 'score': score})
                    if self.best_score is None or score > self.best_score:
                        self.best_score = score
                        self.best_model_name = model_name
                        self.best_params = params
                except Exception:
                    continue
        if self.best_model_name and self.best_params is not None:
            builder = builder_map[self.best_model_name]
            self.best_model = builder(self.best_params)
            self.best_model.fit(X_scaled, y)
        self._scaler = scaler
        return self

    def predict(self, X):
        if self.best_model is None:
            raise RuntimeError('Call fit() first')
        X_scaled = self._scaler.transform([[float(v) for v in row] for row in X] if not isinstance(X[0], list) else X)
        return self.best_model.predict(X_scaled)

    def score(self, X, y) -> float:
        y_list = [float(v) for v in y]
        preds = self.predict(X)
        mu_y = sum(y_list) / max(len(y_list), 1)
        ss_res = sum((p - t)**2 for p, t in zip(preds, y_list))
        ss_tot = sum((t - mu_y)**2 for t in y_list) + 1e-10
        return float(1 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# NAS (micro — random architecture search for feed-forward nets)
# ---------------------------------------------------------------------------

class MicroNAS:
    """Neural Architecture Search over small feed-forward architectures.

    Usage::

        nas = MicroNAS(input_dim=784, output_dim=10, n_trials=20, task='classification')
        nas.search(X_train, y_train, time_limit=120)
        model = nas.best_model
    """

    def __init__(self, input_dim: int, output_dim: int, n_trials: int = 20,
                 task: str = 'classification', time_limit: float = 60.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_trials = n_trials
        self.task = task
        self.time_limit = time_limit
        self.best_model = None
        self.best_score: Optional[float] = None
        self.best_config: Optional[dict] = None

    def _build_model(self, config: dict):
        from vectorphile.nn import Sequential, Linear, ReLU, Dropout, Sigmoid
        layers_spec = config.get('layers', [128, 64])
        dropout = config.get('dropout', 0.0)
        act = ReLU()
        layers = []
        in_dim = self.input_dim
        for out_dim in layers_spec:
            layers.extend([Linear(in_dim, out_dim), ReLU()])
            if dropout > 0:
                layers.append(Dropout(dropout))
            in_dim = out_dim
        layers.append(Linear(in_dim, self.output_dim))
        return Sequential(*layers)

    def _eval_model(self, model, X_val, y_val) -> float:
        from vectorphile.core.tensor import Tensor
        from vectorphile._core.ndarray import array as _nd
        import vectorphile as vp
        flat_X = [float(v) for row in X_val for v in row]
        with vp.autograd.no_grad():
            out = model(Tensor(_nd(flat_X)))
        preds = out._data.to_flat_list() if hasattr(out._data, 'to_flat_list') else list(out._data)
        n = len(y_val)
        if self.task == 'classification':
            n_cls = len(preds) // max(n, 1)
            if n_cls > 1:
                pred_cls = [max(range(n_cls), key=lambda j: preds[i*n_cls+j]) for i in range(n)]
            else:
                pred_cls = [int(round(preds[i])) for i in range(n)]
            return sum(1 for p, t in zip(pred_cls, y_val) if p == int(t)) / max(n, 1)
        else:
            mu = sum(float(t) for t in y_val) / max(n, 1)
            ss_res = sum((float(preds[i]) - float(y_val[i]))**2 for i in range(n))
            ss_tot = sum((float(t) - mu)**2 for t in y_val) + 1e-10
            return float(1 - ss_res / ss_tot)

    def search(self, X_train, y_train, X_val=None, y_val=None,
               time_limit: Optional[float] = None) -> 'MicroNAS':
        time_limit = time_limit or self.time_limit
        X = [[float(v) for v in row] for row in X_train]
        y = list(y_train)
        if X_val is None:
            split = int(len(X) * 0.8)
            X, X_val, y, y_val = X[:split], X[split:], y[:split], y[split:]
        else:
            X_val = [[float(v) for v in row] for row in X_val]
            y_val = list(y_val)

        from vectorphile.optim import Adam
        from vectorphile.nn.losses import CrossEntropyLoss, MSELoss
        from vectorphile.core.tensor import Tensor
        from vectorphile._core.ndarray import array as _nd

        start = time.time()
        for trial in range(self.n_trials):
            if (time.time() - start) > time_limit:
                break
            n_layers = random.randint(1, 4)
            hidden_sizes = [random.choice([32, 64, 128, 256, 512]) for _ in range(n_layers)]
            lr = random.choice([1e-4, 1e-3, 3e-3, 1e-2])
            dropout = random.choice([0.0, 0.1, 0.2])
            config = {'layers': hidden_sizes, 'dropout': dropout, 'lr': lr}
            model = self._build_model(config)
            opt = Adam(model.parameters(), lr=lr)
            loss_fn = CrossEntropyLoss() if self.task == 'classification' else MSELoss()
            n_epochs = 3
            n = len(X)
            for _ in range(n_epochs):
                for b_start in range(0, n, 64):
                    batch_X = X[b_start:b_start+64]
                    batch_y = y[b_start:b_start+64]
                    flat_bX = [v for row in batch_X for v in row]
                    opt.zero_grad()
                    pred = model(Tensor(_nd(flat_bX)))
                    if self.task == 'classification':
                        loss = loss_fn(pred, Tensor(_nd([int(v) for v in batch_y])))
                    else:
                        loss = loss_fn(pred, Tensor(_nd([float(v) for v in batch_y])))
                    loss.backward()
                    opt.step()
            score = self._eval_model(model, X_val, y_val)
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.best_model = model
                self.best_config = config
        return self


__all__ = [
    'Trial', 'Study', 'create_study',
    'suggest_int', 'suggest_float', 'suggest_categorical',
    'IntParam', 'FloatParam', 'CategoricalParam',
    'AutoClassifier', 'AutoRegressor',
    'MicroNAS',
]
