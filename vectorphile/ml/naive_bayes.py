"""Naive Bayes classifiers."""
import numpy as np
from typing import Optional


class GaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_: Optional[np.ndarray] = None
        self.theta_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None
        self.class_prior_: Optional[np.ndarray] = None

    def fit(self, X, y) -> 'GaussianNB':
        X, y = np.array(X, dtype=float), np.array(y)
        self.classes_ = np.unique(y)
        self.theta_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        self.var_ = np.array([X[y == c].var(axis=0) for c in self.classes_])
        max_var = self.var_.max()
        self.var_ += self.var_smoothing * max_var
        counts = np.array([(y == c).sum() for c in self.classes_])
        self.class_prior_ = counts / len(y)
        return self

    def _log_likelihood(self, X):
        log_probs = []
        for k in range(len(self.classes_)):
            log_prior = np.log(self.class_prior_[k])
            diff = X - self.theta_[k]
            log_lk = -0.5 * np.sum(np.log(2 * np.pi * self.var_[k]) + diff**2 / self.var_[k], axis=1)
            log_probs.append(log_prior + log_lk)
        return np.array(log_probs).T

    def predict(self, X) -> np.ndarray:
        return self.classes_[self._log_likelihood(np.array(X, dtype=float)).argmax(axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        log_probs = self._log_likelihood(np.array(X, dtype=float))
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        return probs / probs.sum(axis=1, keepdims=True)

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.array(y)))


class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_: Optional[np.ndarray] = None
        self.feature_log_prob_: Optional[np.ndarray] = None
        self.class_log_prior_: Optional[np.ndarray] = None

    def fit(self, X, y) -> 'MultinomialNB':
        X, y = np.array(X, dtype=float), np.array(y)
        self.classes_ = np.unique(y)
        K, p = len(self.classes_), X.shape[1]
        smoothed = np.array([X[y == c].sum(axis=0) + self.alpha for c in self.classes_])
        self.feature_log_prob_ = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
        counts = np.array([(y == c).sum() for c in self.classes_])
        self.class_log_prior_ = np.log(counts / len(y))
        return self

    def predict(self, X) -> np.ndarray:
        X = np.array(X, dtype=float)
        log_probs = X @ self.feature_log_prob_.T + self.class_log_prior_
        return self.classes_[log_probs.argmax(axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        X = np.array(X, dtype=float)
        log_probs = X @ self.feature_log_prob_.T + self.class_log_prior_
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        return probs / probs.sum(axis=1, keepdims=True)

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.array(y)))


class BernoulliNB:
    def __init__(self, alpha=1.0, binarize=0.0):
        self.alpha = alpha
        self.binarize = binarize
        self.classes_: Optional[np.ndarray] = None
        self.feature_log_prob_: Optional[np.ndarray] = None
        self.class_log_prior_: Optional[np.ndarray] = None

    def fit(self, X, y) -> 'BernoulliNB':
        X = np.array(X, dtype=float)
        if self.binarize is not None:
            X = (X > self.binarize).astype(float)
        y = np.array(y)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        smoothed_pos = np.array([X[y == c].sum(axis=0) + self.alpha for c in self.classes_])
        smoothed_tot = np.array([(y == c).sum() + 2 * self.alpha for c in self.classes_])
        prob = smoothed_pos / smoothed_tot[:, None]
        self.feature_log_prob_ = np.log(prob)
        self._feature_log_prob_neg = np.log(1 - prob)
        counts = np.array([(y == c).sum() for c in self.classes_])
        self.class_log_prior_ = np.log(counts / len(y))
        return self

    def predict(self, X) -> np.ndarray:
        X = np.array(X, dtype=float)
        if self.binarize is not None:
            X = (X > self.binarize).astype(float)
        log_probs = (X @ self.feature_log_prob_.T +
                     (1 - X) @ self._feature_log_prob_neg.T + self.class_log_prior_)
        return self.classes_[log_probs.argmax(axis=1)]

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.array(y)))
