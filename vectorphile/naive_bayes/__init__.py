"""vectorphile.naive_bayes — Naive Bayes classifiers.

All computations use pure Python. No external dependencies.
"""
from __future__ import annotations
import math
from typing import Optional, List


def _to_list2d(X) -> List[List[float]]:
    if hasattr(X, 'to_flat_list'):
        flat = X.to_flat_list()
        if X.ndim == 1:
            return [[float(v)] for v in flat]
        n, d = X.shape[0], X.shape[1]
        return [[float(flat[i * d + j]) for j in range(d)] for i in range(n)]
    if hasattr(X, 'tolist'):
        data = X.tolist()
    else:
        data = list(X)
    if data and not isinstance(data[0], (list, tuple)):
        return [[float(v)] for v in data]
    return [[float(v) for v in row] for row in data]


def _to_list1d(y) -> List:
    if hasattr(y, 'to_flat_list'):
        return y.ravel().to_flat_list()
    if hasattr(y, 'tolist'):
        data = y.tolist()
    else:
        data = list(y)
    if isinstance(data, (int, float, str)):
        return [data]
    flat = []
    for v in data:
        if isinstance(v, (list, tuple)):
            flat.extend(v)
        else:
            flat.append(v)
    return flat


def _argmax(row: List[float]) -> int:
    best, best_i = row[0], 0
    for i, v in enumerate(row[1:], 1):
        if v > best:
            best, best_i = v, i
    return best_i


class GaussianNB:
    """Gaussian Naive Bayes classifier."""

    def __init__(self, priors=None, var_smoothing: float = 1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing
        self.classes_: Optional[List] = None
        self.theta_: Optional[List[List[float]]] = None
        self.var_: Optional[List[List[float]]] = None
        self.class_prior_: Optional[List[float]] = None

    def fit(self, X, y) -> 'GaussianNB':
        X2d = _to_list2d(X)
        y1d = _to_list1d(y)
        n, p = len(X2d), len(X2d[0])
        self.classes_ = sorted(set(y1d))
        K = len(self.classes_)
        self.theta_ = [[0.0] * p for _ in range(K)]
        self.var_ = [[0.0] * p for _ in range(K)]
        counts = [0] * K
        for i, c in enumerate(self.classes_):
            Xc = [X2d[j] for j in range(n) if y1d[j] == c]
            nc = len(Xc)
            counts[i] = nc
            if nc == 0:
                continue
            for f in range(p):
                vals = [Xc[r][f] for r in range(nc)]
                mu = sum(vals) / nc
                var = sum((v - mu) ** 2 for v in vals) / nc
                self.theta_[i][f] = mu
                self.var_[i][f] = var + self.var_smoothing
        total = sum(counts)
        if self.priors is not None:
            pr = _to_list1d(self.priors) if not isinstance(self.priors, list) else self.priors
            self.class_prior_ = [float(v) for v in pr]
        else:
            self.class_prior_ = [counts[i] / total for i in range(K)]
        return self

    def _log_likelihood(self, X2d: List[List[float]]) -> List[List[float]]:
        n, K, p = len(X2d), len(self.classes_), len(X2d[0])
        log_lk = [[0.0] * K for _ in range(n)]
        for i in range(K):
            for r in range(n):
                s = 0.0
                for f in range(p):
                    v = self.var_[i][f]
                    s += -0.5 * math.log(2 * math.pi * v + 1e-300)
                    diff = X2d[r][f] - self.theta_[i][f]
                    s += -0.5 * diff * diff / v
                log_lk[r][i] = s
        return log_lk

    def predict_log_proba(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        log_lk = self._log_likelihood(X2d)
        log_prior = [math.log(p + 1e-300) for p in self.class_prior_]
        return [[log_lk[r][i] + log_prior[i] for i in range(len(self.classes_))] for r in range(len(X2d))]

    def predict_proba(self, X) -> List[List[float]]:
        log_p = self.predict_log_proba(X)
        result = []
        for row in log_p:
            m = max(row)
            exps = [math.exp(v - m) for v in row]
            s = sum(exps)
            result.append([e / s for e in exps])
        return result

    def predict(self, X) -> List:
        log_p = self.predict_log_proba(X)
        return [self.classes_[_argmax(row)] for row in log_p]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        return sum(1 for p, t in zip(preds, y_true) if p == t) / max(len(y_true), 1)

    def partial_fit(self, X, y, classes=None) -> 'GaussianNB':
        if self.classes_ is None:
            return self.fit(X, y)
        X2d = _to_list2d(X)
        y1d = _to_list1d(y)
        p = len(X2d[0]) if X2d else 0
        for i, c in enumerate(self.classes_):
            Xc = [X2d[j] for j in range(len(X2d)) if y1d[j] == c]
            if Xc:
                nc = len(Xc)
                for f in range(p):
                    vals = [Xc[r][f] for r in range(nc)]
                    mu = sum(vals) / nc
                    var = sum((v - mu) ** 2 for v in vals) / nc
                    self.theta_[i][f] = (self.theta_[i][f] + mu) / 2
                    self.var_[i][f] = (self.var_[i][f] + var) / 2 + self.var_smoothing
        return self


class MultinomialNB:
    """Multinomial Naive Bayes for text classification (count features)."""

    def __init__(self, alpha: float = 1.0, fit_prior: bool = True,
                 class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.classes_: Optional[List] = None
        self.feature_log_prob_: Optional[List[List[float]]] = None
        self.class_log_prior_: Optional[List[float]] = None

    def fit(self, X, y) -> 'MultinomialNB':
        X2d = _to_list2d(X)
        y1d = _to_list1d(y)
        n, p = len(X2d), len(X2d[0])
        self.classes_ = sorted(set(y1d))
        K = len(self.classes_)
        class_count = [0] * K
        feature_count = [[0.0] * p for _ in range(K)]
        for r in range(n):
            c = y1d[r]
            if c in self.classes_:
                i = self.classes_.index(c)
                class_count[i] += 1
                for f in range(p):
                    feature_count[i][f] += X2d[r][f]
        # Apply Laplace smoothing + log
        self.feature_log_prob_ = []
        for i in range(K):
            fc = [feature_count[i][f] + self.alpha for f in range(p)]
            total = sum(fc)
            self.feature_log_prob_.append([math.log(v / total) for v in fc])
        total_count = sum(class_count)
        if self.class_prior is not None:
            pr = _to_list1d(self.class_prior) if not isinstance(self.class_prior, list) else self.class_prior
            self.class_log_prior_ = [math.log(float(v) + 1e-300) for v in pr]
        elif self.fit_prior:
            self.class_log_prior_ = [math.log(class_count[i] / total_count + 1e-300) for i in range(K)]
        else:
            self.class_log_prior_ = [-math.log(K)] * K
        return self

    def predict_log_proba(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        K, p = len(self.classes_), len(X2d[0])
        result = []
        for row in X2d:
            scores = []
            for i in range(K):
                s = self.class_log_prior_[i] + sum(row[f] * self.feature_log_prob_[i][f] for f in range(p))
                scores.append(s)
            result.append(scores)
        return result

    def predict_proba(self, X) -> List[List[float]]:
        log_p = self.predict_log_proba(X)
        result = []
        for row in log_p:
            m = max(row)
            exps = [math.exp(v - m) for v in row]
            s = sum(exps)
            result.append([e / s for e in exps])
        return result

    def predict(self, X) -> List:
        log_p = self.predict_log_proba(X)
        return [self.classes_[_argmax(row)] for row in log_p]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        return sum(1 for p, t in zip(preds, y_true) if p == t) / max(len(y_true), 1)


class BernoulliNB:
    """Bernoulli Naive Bayes for binary/boolean features."""

    def __init__(self, alpha: float = 1.0, binarize: Optional[float] = 0.0,
                 fit_prior: bool = True):
        self.alpha = alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.classes_: Optional[List] = None
        self.feature_log_prob_: Optional[List[List[float]]] = None
        self.log1m_fp_: Optional[List[List[float]]] = None
        self.class_log_prior_: Optional[List[float]] = None

    def _bin(self, X2d: List[List[float]]) -> List[List[float]]:
        if self.binarize is None:
            return X2d
        t = self.binarize
        return [[1.0 if v > t else 0.0 for v in row] for row in X2d]

    def fit(self, X, y) -> 'BernoulliNB':
        X2d = self._bin(_to_list2d(X))
        y1d = _to_list1d(y)
        n, p = len(X2d), len(X2d[0])
        self.classes_ = sorted(set(y1d))
        K = len(self.classes_)
        class_count = [0] * K
        fc = [[0.0] * p for _ in range(K)]
        for r in range(n):
            c = y1d[r]
            if c in self.classes_:
                i = self.classes_.index(c)
                class_count[i] += 1
                for f in range(p):
                    fc[i][f] += X2d[r][f]
        self.feature_log_prob_ = []
        self.log1m_fp_ = []
        for i in range(K):
            ni = class_count[i]
            fp = [(fc[i][f] + self.alpha) / (ni + 2 * self.alpha) for f in range(p)]
            self.feature_log_prob_.append([math.log(v) for v in fp])
            self.log1m_fp_.append([math.log(1 - v + 1e-300) for v in fp])
        total = sum(class_count)
        self.class_log_prior_ = [math.log(class_count[i] / total + 1e-300) for i in range(K)]
        return self

    def predict_log_proba(self, X) -> List[List[float]]:
        X2d = self._bin(_to_list2d(X))
        K, p = len(self.classes_), len(X2d[0])
        result = []
        for row in X2d:
            scores = []
            for i in range(K):
                s = self.class_log_prior_[i]
                for f in range(p):
                    if row[f] > 0:
                        s += self.feature_log_prob_[i][f]
                    else:
                        s += self.log1m_fp_[i][f]
                scores.append(s)
            result.append(scores)
        return result

    def predict_proba(self, X) -> List[List[float]]:
        log_p = self.predict_log_proba(X)
        result = []
        for row in log_p:
            m = max(row)
            exps = [math.exp(v - m) for v in row]
            s = sum(exps)
            result.append([e / s for e in exps])
        return result

    def predict(self, X) -> List:
        log_p = self.predict_log_proba(X)
        return [self.classes_[_argmax(row)] for row in log_p]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        return sum(1 for p, t in zip(preds, y_true) if p == t) / max(len(y_true), 1)


class ComplementNB:
    """Complement Naive Bayes (better for imbalanced text data)."""

    def __init__(self, alpha: float = 1.0, norm: bool = False):
        self.alpha = alpha
        self.norm = norm
        self.classes_: List = []
        self._log_weights: List[List[float]] = []

    def fit(self, X, y):
        X2d = _to_list2d(X)
        y_list = _to_list1d(y)
        self.classes_ = sorted(set(y_list))
        n, p = len(X2d), len(X2d[0])
        # Compute complement class feature sums
        self._log_weights = []
        total_counts = [sum(X2d[i][j] for i in range(n)) for j in range(p)]
        for cls in self.classes_:
            idxs = [i for i, lbl in enumerate(y_list) if lbl != cls]
            class_counts = [sum(X2d[i][j] for i in idxs) + self.alpha for j in range(p)]
            total = sum(class_counts)
            log_w = [math.log(c / total) for c in class_counts]
            if self.norm:
                norm_sum = sum(abs(w) for w in log_w)
                log_w = [w / (norm_sum + 1e-10) for w in log_w]
            self._log_weights.append(log_w)
        return self

    def predict_log_proba(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        result = []
        for row in X2d:
            scores = [-sum(row[j] * self._log_weights[k][j] for j in range(len(row)))
                      for k in range(len(self.classes_))]
            result.append(scores)
        return result

    def predict(self, X) -> List:
        return [self.classes_[_argmax(row)] for row in self.predict_log_proba(X)]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        return sum(1 for p, t in zip(preds, y_true) if p == t) / max(len(y_true), 1)


class CategoricalNB:
    """Naive Bayes for categorical features."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_: List = []
        self._log_priors: List[float] = []
        self._log_cond: List[List[dict]] = []  # [class][feature] -> {value: log_prob}

    def fit(self, X, y):
        X2d = _to_list2d(X)
        y_list = _to_list1d(y)
        self.classes_ = sorted(set(y_list))
        n, p = len(X2d), len(X2d[0])
        class_counts = {cls: 0 for cls in self.classes_}
        for lbl in y_list:
            class_counts[lbl] += 1
        self._log_priors = [math.log(class_counts[cls] / n) for cls in self.classes_]
        # For each class and feature, count value occurrences
        self._log_cond = []
        for k, cls in enumerate(self.classes_):
            idxs = [i for i, lbl in enumerate(y_list) if lbl == cls]
            feat_probs = []
            for j in range(p):
                vals = [X2d[i][j] for i in idxs]
                unique_vals = set(X2d[i][j] for i in range(n))
                n_cats = len(unique_vals)
                val_counts: dict = {}
                for v in vals:
                    val_counts[v] = val_counts.get(v, 0) + 1
                total = len(idxs) + self.alpha * n_cats
                log_probs = {v: math.log((val_counts.get(v, 0) + self.alpha) / total)
                             for v in unique_vals}
                feat_probs.append(log_probs)
            self._log_cond.append(feat_probs)
        return self

    def predict_log_proba(self, X) -> List[List[float]]:
        X2d = _to_list2d(X)
        result = []
        for row in X2d:
            scores = []
            for k in range(len(self.classes_)):
                log_p = self._log_priors[k]
                for j, v in enumerate(row):
                    log_p += self._log_cond[k][j].get(v, math.log(self.alpha / (self.alpha + 1)))
                scores.append(log_p)
            result.append(scores)
        return result

    def predict(self, X) -> List:
        return [self.classes_[_argmax(row)] for row in self.predict_log_proba(X)]

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y_true = _to_list1d(y)
        return sum(1 for p, t in zip(preds, y_true) if p == t) / max(len(y_true), 1)


__all__ = ['GaussianNB', 'MultinomialNB', 'BernoulliNB', 'ComplementNB', 'CategoricalNB']
