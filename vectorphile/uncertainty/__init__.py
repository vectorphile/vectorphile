"""vectorphile.uncertainty — Uncertainty Quantification — pure Python."""
from __future__ import annotations
import math
from typing import Any, List, Optional, Tuple


def _flat(x) -> List[float]:
    if hasattr(x, 'to_flat_list'):
        return [float(v) for v in x.to_flat_list()]
    if hasattr(x, 'tolist'):
        v = x.tolist()
        if isinstance(v, list) and v and isinstance(v[0], list):
            return [float(item) for row in v for item in row]
        return [float(item) for item in v]
    if isinstance(x, list):
        if x and isinstance(x[0], list):
            return [float(item) for row in x for item in row]
        return [float(item) for item in x]
    return [float(x)]

def _mean_list(x: List[float]) -> float:
    return sum(x) / max(len(x), 1)

def _var_list(x: List[float]) -> float:
    mu = _mean_list(x)
    return sum((v-mu)**2 for v in x) / max(len(x), 1)

def _softmax(x: List[float]) -> List[float]:
    mx = max(x)
    exps = [math.exp(v - mx) for v in x]
    s = sum(exps)
    return [e/s for e in exps]


# ---------------------------------------------------------------------------
# Monte Carlo Dropout
# ---------------------------------------------------------------------------

class MCDropout:
    """Epistemic uncertainty estimation via Monte Carlo Dropout."""

    def __init__(self, model, n_samples: int = 100, dropout_p: float = 0.1):
        self.model = model
        self.n_samples = n_samples
        self.dropout_p = dropout_p

    def predict(self, x, return_samples: bool = False):
        from vectorphile.core.tensor import Tensor
        import vectorphile as vp
        samples = []
        self.model.train()
        for _ in range(self.n_samples):
            with vp.autograd.no_grad():
                pred = self.model(x if isinstance(x, Tensor) else Tensor(x))
            samples.append(_flat(pred._data if hasattr(pred, '_data') else pred))
        self.model.eval()
        # Compute mean and var per output dimension
        n_out = len(samples[0]) if samples else 0
        mean = [_mean_list([s[i] for s in samples]) for i in range(n_out)]
        var = [_var_list([s[i] for s in samples]) for i in range(n_out)]
        if return_samples:
            return mean, var, samples
        return mean, var


# ---------------------------------------------------------------------------
# Deep Ensembles
# ---------------------------------------------------------------------------

class DeepEnsemble:
    """Uncertainty via ensemble of independently trained models."""

    def __init__(self, models: List):
        self.models = models

    def predict(self, x):
        from vectorphile.core.tensor import Tensor
        import vectorphile as vp
        preds = []
        for model in self.models:
            with vp.autograd.no_grad():
                p = model(x if isinstance(x, Tensor) else Tensor(x))
            preds.append(_flat(p._data if hasattr(p, '_data') else p))
        n_out = len(preds[0]) if preds else 0
        mean = [_mean_list([s[i] for s in preds]) for i in range(n_out)]
        var = [_var_list([s[i] for s in preds]) for i in range(n_out)]
        return mean, var

    def predict_proba(self, x):
        mean, _ = self.predict(x)
        return _softmax(mean)


# ---------------------------------------------------------------------------
# Conformal Prediction
# ---------------------------------------------------------------------------

class SplitConformalPredictor:
    """Split Conformal Prediction."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._q_hat = None

    def calibrate(self, scores, y_cal=None) -> float:
        sc = sorted(_flat(scores))
        n = len(sc)
        level = min(1.0, math.ceil((n + 1) * (1 - self.alpha)) / n)
        idx = min(int(level * n), n - 1)
        self._q_hat = sc[idx]
        return self._q_hat

    def predict_sets(self, scores) -> List[bool]:
        return [v <= self._q_hat for v in _flat(scores)]

    def predict_intervals(self, predictions) -> List[Tuple[float, float]]:
        preds = _flat(predictions)
        return [(v - self._q_hat, v + self._q_hat) for v in preds]

    def coverage(self, scores_test) -> float:
        sc = _flat(scores_test)
        return sum(1 for v in sc if v <= self._q_hat) / max(len(sc), 1)


class RegressionConformalPredictor:
    """Conformal prediction for regression."""

    def __init__(self, model, alpha: float = 0.1):
        self.model = model
        self.alpha = alpha
        self._q_hat = None

    def _get_residuals(self, X, y) -> List[float]:
        from vectorphile.core.tensor import Tensor
        import vectorphile as vp
        with vp.autograd.no_grad():
            preds = self.model(X if isinstance(X, Tensor) else Tensor(X))
        preds_flat = _flat(preds._data if hasattr(preds, '_data') else preds)
        y_flat = _flat(y)
        return [abs(p - t) for p, t in zip(preds_flat, y_flat)]

    def calibrate(self, X_cal, y_cal) -> float:
        residuals = self._get_residuals(X_cal, y_cal)
        cp = SplitConformalPredictor(self.alpha)
        self._q_hat = cp.calibrate(residuals)
        return self._q_hat

    def predict(self, X) -> List[Tuple[float, float]]:
        from vectorphile.core.tensor import Tensor
        import vectorphile as vp
        with vp.autograd.no_grad():
            preds = self.model(X if isinstance(X, Tensor) else Tensor(X))
        preds_flat = _flat(preds._data if hasattr(preds, '_data') else preds)
        return [(v - self._q_hat, v + self._q_hat) for v in preds_flat]


# ---------------------------------------------------------------------------
# Temperature Scaling (calibration)
# ---------------------------------------------------------------------------

class TemperatureScaling:
    """Post-hoc calibration via temperature scaling."""

    def __init__(self):
        self.temperature = 1.0

    def _nll(self, logits: List[List[float]], labels: List[int], T: float) -> float:
        nll = 0.0
        n = len(labels)
        for i, label in enumerate(labels):
            row = [v / T for v in logits[i]]
            mx = max(row)
            lse = math.log(sum(math.exp(v - mx) for v in row)) + mx
            nll -= (row[int(label)] - lse)
        return nll / max(n, 1)

    def fit(self, logits, y, lr: float = 0.01, max_iter: int = 1000) -> 'TemperatureScaling':
        if isinstance(logits[0], (int, float)):
            logits_2d = [[v] for v in logits]
        else:
            logits_2d = [[float(v) for v in row] for row in logits]
        labels = [int(v) for v in _flat(y)]
        T = 1.0
        # Grid search
        best_T, best_nll = 1.0, float('inf')
        for k in range(50):
            T_cand = 0.1 + k * (5.0 - 0.1) / 49
            nll = self._nll(logits_2d, labels, T_cand)
            if nll < best_nll:
                best_nll = nll; best_T = T_cand
        T = best_T
        for _ in range(max_iter):
            eps = 1e-5
            d_nll = (self._nll(logits_2d, labels, T+eps) - self._nll(logits_2d, labels, T-eps)) / (2*eps)
            T -= lr * d_nll
            T = max(0.01, T)
        self.temperature = T
        return self

    def predict_proba(self, logits) -> List[List[float]]:
        if isinstance(logits[0], (int, float)):
            logits_2d = [[v / self.temperature] for v in logits]
        else:
            logits_2d = [[v / self.temperature for v in row] for row in logits]
        return [_softmax(row) for row in logits_2d]


# ---------------------------------------------------------------------------
# Platt Scaling
# ---------------------------------------------------------------------------

class PlattScaling:
    """Platt scaling for binary calibration."""

    def __init__(self):
        self.A = -1.0
        self.B = 0.0

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def fit(self, scores, y, max_iter: int = 1000, lr: float = 0.01) -> 'PlattScaling':
        sc = _flat(scores); yt = _flat(y)
        A, B = -1.0, 0.0
        n = len(sc)
        for _ in range(max_iter):
            p = [self._sigmoid(A*s + B) for s in sc]
            grad_A = sum((p[i]-yt[i])*sc[i] for i in range(n)) / n
            grad_B = sum(p[i]-yt[i] for i in range(n)) / n
            A -= lr * grad_A; B -= lr * grad_B
        self.A, self.B = A, B
        return self

    def predict_proba(self, scores) -> List[float]:
        sc = _flat(scores)
        return [self._sigmoid(self.A*s + self.B) for s in sc]


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------

def expected_calibration_error(y_true, y_proba, n_bins: int = 10) -> float:
    from vectorphile.metrics import expected_calibration_error as _ece
    return _ece(y_true, y_proba, n_bins)


def reliability_diagram(y_true, y_proba, n_bins: int = 10) -> dict:
    from vectorphile.metrics import reliability_diagram as _rd
    return _rd(y_true, y_proba, n_bins)


__all__ = [
    'MCDropout', 'DeepEnsemble',
    'SplitConformalPredictor', 'RegressionConformalPredictor',
    'TemperatureScaling', 'PlattScaling',
    'expected_calibration_error', 'reliability_diagram',
]
