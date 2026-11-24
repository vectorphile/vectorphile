"""vectorphile.semi_supervised — Semi-supervised learning algorithms.

All computations use vectorphile's own NDArray primitives. No external deps.
"""
from __future__ import annotations
import math
import copy
from typing import Optional

from vectorphile._core.ndarray import (
    NDArray as _NDA,
    zeros as _nd_zeros, ones as _nd_ones, eye as _nd_eye,
    exp as _nd_exp, log as _nd_log,
    _c_strides as _nd_c_strides,
    _DType as _NDType,
    matmul as _nd_matmul,
)
import array as _pyarray


def _to_nd_f32(X) -> _NDA:
    if isinstance(X, _NDA):
        return X.astype('float32')
    flat = []
    def _fl(v):
        if isinstance(v, (list, tuple)): [_fl(x) for x in v]
        else: flat.append(float(v))
    _fl(X)
    cur = X
    shape = []
    while isinstance(cur, (list, tuple)) and len(cur) > 0:
        shape.append(len(cur)); cur = cur[0]
    shape = tuple(shape) if shape else (len(flat),)
    buf = _pyarray.array('f', flat)
    return _NDA(buf, shape, _nd_c_strides(shape), _NDType('float32'))


def _nd_argmax_1d(row_flat: list) -> int:
    return row_flat.index(max(row_flat))


# ---------------------------------------------------------------------------
# LabelPropagation
# ---------------------------------------------------------------------------

class LabelPropagation:
    """Label Propagation algorithm for semi-supervised classification."""

    def __init__(self, kernel: str = 'rbf', gamma: float = 20.0,
                 n_neighbors: int = 7, max_iter: int = 1000,
                 tol: float = 1e-3):
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.tol = tol
        self.X_: Optional[_NDA] = None
        self.classes_: Optional[list] = None
        self.label_distributions_: Optional[_NDA] = None

    def _build_graph(self, X: _NDA) -> _NDA:
        n = X.shape[0]
        X_flat = X.to_flat_list()
        d = X.shape[1] if X.ndim > 1 else 1
        # compute pairwise RBF weights
        W_flat = []
        for i in range(n):
            xi = X_flat[i * d:(i + 1) * d]
            row = []
            for j in range(n):
                if i == j:
                    row.append(0.0)
                    continue
                xj = X_flat[j * d:(j + 1) * d]
                sq = sum((xi[k] - xj[k]) ** 2 for k in range(d))
                row.append(math.exp(-self.gamma * sq))
            W_flat.extend(row)
        # row-normalize
        result = []
        for i in range(n):
            row = W_flat[i * n:(i + 1) * n]
            s = sum(row) + 1e-10
            result.extend([v / s for v in row])
        buf = _pyarray.array('f', result)
        return _NDA(buf, (n, n), _nd_c_strides((n, n)), _NDType('float32'))

    def fit(self, X, y) -> 'LabelPropagation':
        X_nd = _to_nd_f32(X)
        y_flat = [int(v) for v in _to_nd_f32(y).ravel().to_flat_list()]
        self.X_ = X_nd
        labeled = [i for i, v in enumerate(y_flat) if v >= 0]
        self.classes_ = sorted(set(y_flat[i] for i in labeled))
        n, K = X_nd.shape[0], len(self.classes_)
        # init F
        F = [[0.0] * K for _ in range(n)]
        for i in range(n):
            if y_flat[i] >= 0 and y_flat[i] in self.classes_:
                ci = self.classes_.index(y_flat[i])
                F[i][ci] = 1.0
        Y = [row[:] for row in F]
        T = self._build_graph(X_nd).to_flat_list()
        # propagate
        for _ in range(self.max_iter):
            F_new = [[0.0] * K for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    w = T[i * n + j]
                    for k in range(K):
                        F_new[i][k] += w * F[j][k]
            # clamp labeled
            for i in labeled:
                F_new[i] = Y[i][:]
            # convergence check
            diff = max(abs(F_new[i][k] - F[i][k]) for i in range(n) for k in range(K))
            F = F_new
            if diff < self.tol:
                break
        flat = [F[i][k] for i in range(n) for k in range(K)]
        buf = _pyarray.array('f', flat)
        self.label_distributions_ = _NDA(buf, (n, K), _nd_c_strides((n, K)), _NDType('float32'))
        return self

    def predict_proba(self) -> _NDA:
        F = self.label_distributions_
        n, K = F.shape
        F_flat = F.to_flat_list()
        result = []
        for i in range(n):
            row = F_flat[i * K:(i + 1) * K]
            s = sum(row) + 1e-10
            result.extend([v / s for v in row])
        buf = _pyarray.array('f', result)
        return _NDA(buf, (n, K), _nd_c_strides((n, K)), _NDType('float32'))

    def predict(self) -> _NDA:
        proba = self.predict_proba()
        n, K = proba.shape
        flat = proba.to_flat_list()
        preds = [float(self.classes_[_nd_argmax_1d(flat[i * K:(i + 1) * K])]) for i in range(n)]
        buf = _pyarray.array('f', preds)
        return _NDA(buf, (n,), _nd_c_strides((n,)), _NDType('float32'))

    def score(self, X, y_true) -> float:
        y = [int(v) for v in _to_nd_f32(y_true).ravel().to_flat_list()]
        preds = [int(v) for v in self.predict().to_flat_list()]
        labeled = [i for i, v in enumerate(y) if v >= 0]
        return sum(1 for i in labeled if preds[i] == y[i]) / max(len(labeled), 1)


# ---------------------------------------------------------------------------
# LabelSpreading
# ---------------------------------------------------------------------------

class LabelSpreading:
    """Label Spreading with clamping factor alpha."""

    def __init__(self, kernel: str = 'rbf', gamma: float = 20.0,
                 n_neighbors: int = 7, alpha: float = 0.2,
                 max_iter: int = 30, tol: float = 1e-3):
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.classes_: Optional[list] = None
        self.label_distributions_: Optional[_NDA] = None

    def _build_graph(self, X: _NDA) -> _NDA:
        n = X.shape[0]
        X_flat = X.to_flat_list()
        d = X.shape[1] if X.ndim > 1 else 1
        W_flat = []
        for i in range(n):
            xi = X_flat[i * d:(i + 1) * d]
            row = []
            for j in range(n):
                if i == j:
                    row.append(0.0)
                    continue
                xj = X_flat[j * d:(j + 1) * d]
                sq = sum((xi[k] - xj[k]) ** 2 for k in range(d))
                row.append(math.exp(-self.gamma * sq))
            W_flat.extend(row)
        # Normalize: D^{-1/2} W D^{-1/2}
        D = [sum(W_flat[i * n:(i + 1) * n]) for i in range(n)]
        D_inv_sqrt = [1.0 / math.sqrt(d + 1e-10) if d > 0 else 0.0 for d in D]
        result = [D_inv_sqrt[i] * W_flat[i * n + j] * D_inv_sqrt[j]
                  for i in range(n) for j in range(n)]
        buf = _pyarray.array('f', result)
        return _NDA(buf, (n, n), _nd_c_strides((n, n)), _NDType('float32'))

    def fit(self, X, y) -> 'LabelSpreading':
        X_nd = _to_nd_f32(X)
        y_flat = [int(v) for v in _to_nd_f32(y).ravel().to_flat_list()]
        labeled = [i for i, v in enumerate(y_flat) if v >= 0]
        self.classes_ = sorted(set(y_flat[i] for i in labeled))
        n, K = X_nd.shape[0], len(self.classes_)
        Y = [[0.0] * K for _ in range(n)]
        for i in labeled:
            if y_flat[i] in self.classes_:
                Y[i][self.classes_.index(y_flat[i])] = 1.0
        S = self._build_graph(X_nd).to_flat_list()
        F = [row[:] for row in Y]
        a, b = self.alpha, 1.0 - self.alpha
        for _ in range(self.max_iter):
            F_new = [[0.0] * K for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    w = S[i * n + j]
                    for k in range(K):
                        F_new[i][k] += a * w * F[j][k]
                for k in range(K):
                    F_new[i][k] += b * Y[i][k]
            diff = max(abs(F_new[i][k] - F[i][k]) for i in range(n) for k in range(K))
            F = F_new
            if diff < self.tol:
                break
        flat = [F[i][k] for i in range(n) for k in range(K)]
        buf = _pyarray.array('f', flat)
        self.label_distributions_ = _NDA(buf, (n, K), _nd_c_strides((n, K)), _NDType('float32'))
        return self

    def predict_proba(self) -> _NDA:
        F = self.label_distributions_
        n, K = F.shape
        flat = F.to_flat_list()
        result = []
        for i in range(n):
            row = flat[i * K:(i + 1) * K]
            s = sum(row) + 1e-10
            result.extend([v / s for v in row])
        buf = _pyarray.array('f', result)
        return _NDA(buf, (n, K), _nd_c_strides((n, K)), _NDType('float32'))

    def predict(self) -> _NDA:
        proba = self.predict_proba()
        n, K = proba.shape
        flat = proba.to_flat_list()
        preds = [float(self.classes_[_nd_argmax_1d(flat[i * K:(i + 1) * K])]) for i in range(n)]
        buf = _pyarray.array('f', preds)
        return _NDA(buf, (n,), _nd_c_strides((n,)), _NDType('float32'))

    def score(self, X, y_true) -> float:
        y = [int(v) for v in _to_nd_f32(y_true).ravel().to_flat_list()]
        preds = [int(v) for v in self.predict().to_flat_list()]
        labeled = [i for i, v in enumerate(y) if v >= 0]
        return sum(1 for i in labeled if preds[i] == y[i]) / max(len(labeled), 1)


# ---------------------------------------------------------------------------
# SelfTrainingClassifier
# ---------------------------------------------------------------------------

class SelfTrainingClassifier:
    """Self-training meta-classifier with pseudo-labels."""

    def __init__(self, base_estimator, threshold: float = 0.75,
                 criterion: str = 'threshold', k_best: int = 10,
                 max_iter: int = 10, verbose: bool = False):
        self.base_estimator = base_estimator
        self.threshold = threshold
        self.criterion = criterion
        self.k_best = k_best
        self.max_iter = max_iter
        self.verbose = verbose
        self.base_estimator_: object = None
        self.transduction_: Optional[list] = None
        self.n_iter_: int = 0

    def fit(self, X, y) -> 'SelfTrainingClassifier':
        X_nd = _to_nd_f32(X)
        y_flat = [int(v) for v in _to_nd_f32(y).ravel().to_flat_list()]
        transduction = y_flat[:]
        est = None
        for it in range(1, self.max_iter + 1):
            est = copy.deepcopy(self.base_estimator)
            lm = [i for i, v in enumerate(transduction) if v != -1]
            X_lm_flat = []
            d = X_nd.shape[1] if X_nd.ndim > 1 else 1
            for i in lm:
                if X_nd.ndim > 1:
                    X_lm_flat.extend(X_nd[i].to_flat_list())
                else:
                    X_lm_flat.append(float(X_nd[i]))
            if not lm:
                break
            X_lm_buf = _pyarray.array('f', X_lm_flat)
            X_lm = _NDA(X_lm_buf, (len(lm), d) if X_nd.ndim > 1 else (len(lm),),
                        _nd_c_strides((len(lm), d) if X_nd.ndim > 1 else (len(lm),)), _NDType('float32'))
            y_lm_buf = _pyarray.array('f', [float(transduction[i]) for i in lm])
            y_lm = _NDA(y_lm_buf, (len(lm),), _nd_c_strides((len(lm),)), _NDType('float32'))
            est.fit(X_lm, y_lm)
            unlabeled = [i for i, v in enumerate(transduction) if v == -1]
            if not unlabeled:
                break
            X_ul_flat = []
            for i in unlabeled:
                if X_nd.ndim > 1:
                    X_ul_flat.extend(X_nd[i].to_flat_list())
                else:
                    X_ul_flat.append(float(X_nd[i]))
            X_ul_buf = _pyarray.array('f', X_ul_flat)
            X_ul = _NDA(X_ul_buf, (len(unlabeled), d) if X_nd.ndim > 1 else (len(unlabeled),),
                        _nd_c_strides((len(unlabeled), d) if X_nd.ndim > 1 else (len(unlabeled),)), _NDType('float32'))
            if hasattr(est, 'predict_proba'):
                proba_nd = est.predict_proba(X_ul)
                proba_flat = proba_nd.to_flat_list() if isinstance(proba_nd, _NDA) else list(proba_nd)
                K = proba_nd.shape[1] if isinstance(proba_nd, _NDA) else 2
                added = 0
                for idx_ul, i in enumerate(unlabeled):
                    row = proba_flat[idx_ul * K:(idx_ul + 1) * K]
                    conf = max(row)
                    pred = row.index(conf)
                    if conf >= self.threshold:
                        transduction[i] = pred
                        added += 1
                if added == 0:
                    break
            else:
                preds = est.predict(X_ul)
                pred_list = preds.to_flat_list() if isinstance(preds, _NDA) else list(preds)
                for idx_ul, i in enumerate(unlabeled):
                    transduction[i] = int(pred_list[idx_ul])
            self.n_iter_ = it
        self.base_estimator_ = est
        self.transduction_ = transduction
        return self

    def predict(self, X) -> _NDA:
        X_nd = _to_nd_f32(X)
        result = self.base_estimator_.predict(X_nd)
        if isinstance(result, _NDA):
            return result
        flat = list(result) if hasattr(result, '__iter__') else [result]
        buf = _pyarray.array('f', [float(v) for v in flat])
        return _NDA(buf, (len(flat),), _nd_c_strides((len(flat),)), _NDType('float32'))

    def score(self, X, y) -> float:
        preds = [int(v) for v in self.predict(_to_nd_f32(X)).to_flat_list()]
        y_true = [int(v) for v in _to_nd_f32(y).ravel().to_flat_list()]
        return sum(1 for p, t in zip(preds, y_true) if p == t) / max(len(y_true), 1)


__all__ = ['LabelPropagation', 'LabelSpreading', 'SelfTrainingClassifier']
