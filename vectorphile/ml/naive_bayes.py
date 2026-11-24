"""Naive Bayes classifiers — pure Python."""
import math
from typing import List


def _col(X, j):
    return [X[i][j] for i in range(len(X))]


def _uniq_sorted(seq):
    return sorted(set(seq))


class GaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X2 = [[float(v) for v in row] for row in X]
        y2 = list(y)
        n, p = len(X2), len(X2[0])
        self.classes_ = _uniq_sorted(y2)
        self.class_prior_ = []
        self.theta_ = []  # means per class
        self.var_ = []    # variances per class
        for c in self.classes_:
            idxs = [i for i, v in enumerate(y2) if v == c]
            Xc = [X2[i] for i in idxs]
            nc = len(idxs)
            self.class_prior_.append(nc / n)
            means = [sum(_col(Xc, j)) / nc for j in range(p)]
            vars_ = [sum((Xc[i][j] - means[j])**2 for i in range(nc)) / nc + self.var_smoothing
                     for j in range(p)]
            self.theta_.append(means)
            self.var_.append(vars_)
        return self

    def _log_likelihood(self, x, ci):
        ll = 0.0
        for j in range(len(x)):
            mu = self.theta_[ci][j]
            var = self.var_[ci][j]
            ll += -0.5 * math.log(2 * math.pi * var) - 0.5 * (x[j] - mu)**2 / var
        return ll

    def predict_log_proba(self, X):
        X2 = [[float(v) for v in row] for row in X]
        result = []
        for x in X2:
            lps = [math.log(self.class_prior_[ci]) + self._log_likelihood(x, ci)
                   for ci in range(len(self.classes_))]
            mx = max(lps)
            log_sum = mx + math.log(sum(math.exp(lp - mx) for lp in lps))
            result.append([lp - log_sum for lp in lps])
        return result

    def predict_proba(self, X):
        lps = self.predict_log_proba(X)
        return [[math.exp(v) for v in row] for row in lps]

    def predict(self, X):
        proba = self.predict_log_proba(X)
        return [self.classes_[row.index(max(row))] for row in proba]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(a == b for a, b in zip(preds, y)) / len(y)


class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X2 = [[float(v) for v in row] for row in X]
        y2 = list(y)
        n, p = len(X2), len(X2[0])
        self.classes_ = _uniq_sorted(y2)
        self.class_log_prior_ = []
        self.feature_log_prob_ = []
        for c in self.classes_:
            idxs = [i for i, v in enumerate(y2) if v == c]
            Xc = [X2[i] for i in idxs]
            nc = len(idxs)
            self.class_log_prior_.append(math.log(nc / n))
            col_sums = [sum(_col(Xc, j)) + self.alpha for j in range(p)]
            total = sum(col_sums)
            self.feature_log_prob_.append([math.log(s / total) for s in col_sums])
        return self

    def predict_log_proba(self, X):
        X2 = [[float(v) for v in row] for row in X]
        result = []
        for x in X2:
            lps = [self.class_log_prior_[ci] + sum(x[j] * self.feature_log_prob_[ci][j]
                                                    for j in range(len(x)))
                   for ci in range(len(self.classes_))]
            mx = max(lps)
            log_sum = mx + math.log(sum(math.exp(lp - mx) for lp in lps))
            result.append([lp - log_sum for lp in lps])
        return result

    def predict_proba(self, X):
        return [[math.exp(v) for v in row] for row in self.predict_log_proba(X)]

    def predict(self, X):
        lps = self.predict_log_proba(X)
        return [self.classes_[row.index(max(row))] for row in lps]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(a == b for a, b in zip(preds, y)) / len(y)


class BernoulliNB:
    def __init__(self, alpha=1.0, binarize=0.0):
        self.alpha = alpha
        self.binarize = binarize

    def fit(self, X, y):
        X2 = [[1.0 if v > self.binarize else 0.0 for v in row] for row in X]
        y2 = list(y)
        n, p = len(X2), len(X2[0])
        self.classes_ = _uniq_sorted(y2)
        self.class_log_prior_ = []
        self.feature_log_prob_ = []
        self.feature_log_prob_neg_ = []
        for c in self.classes_:
            idxs = [i for i, v in enumerate(y2) if v == c]
            Xc = [X2[i] for i in idxs]
            nc = len(idxs)
            self.class_log_prior_.append(math.log(nc / n))
            probs = [(sum(_col(Xc, j)) + self.alpha) / (nc + 2 * self.alpha)
                     for j in range(p)]
            self.feature_log_prob_.append([math.log(pr) for pr in probs])
            self.feature_log_prob_neg_.append([math.log(1 - pr) for pr in probs])
        return self

    def predict_log_proba(self, X):
        X2 = [[1.0 if v > self.binarize else 0.0 for v in row] for row in X]
        result = []
        for x in X2:
            lps = []
            for ci in range(len(self.classes_)):
                lp = self.class_log_prior_[ci]
                for j in range(len(x)):
                    lp += (self.feature_log_prob_[ci][j] if x[j] else
                            self.feature_log_prob_neg_[ci][j])
                lps.append(lp)
            mx = max(lps)
            log_sum = mx + math.log(sum(math.exp(lp - mx) for lp in lps))
            result.append([lp - log_sum for lp in lps])
        return result

    def predict_proba(self, X):
        return [[math.exp(v) for v in row] for row in self.predict_log_proba(X)]

    def predict(self, X):
        lps = self.predict_log_proba(X)
        return [self.classes_[row.index(max(row))] for row in lps]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(a == b for a, b in zip(preds, y)) / len(y)
