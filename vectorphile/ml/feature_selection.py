"""Feature selection — pure Python."""
import math
import random as _random
from typing import Optional, List


def _col(X, j):
    return [X[i][j] for i in range(len(X))]


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _var(vals):
    if len(vals) < 2:
        return 0.0
    mu = _mean(vals)
    return sum((v - mu) ** 2 for v in vals) / len(vals)


def _std(vals):
    return math.sqrt(_var(vals))


def _uniq_sorted(seq):
    return sorted(set(seq))


def _argsort(vals, reverse=False):
    return sorted(range(len(vals)), key=lambda i: vals[i], reverse=reverse)


def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


class VarianceThreshold:
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X):
        X2 = [[float(v) for v in row] for row in X]
        p = len(X2[0]) if X2 else 0
        self.variances_ = [_var(_col(X2, j)) for j in range(p)]
        self._mask = [v > self.threshold for v in self.variances_]
        self.n_features_in_ = p
        return self

    def transform(self, X):
        X2 = [[float(v) for v in row] for row in X]
        return [[X2[i][j] for j in range(len(X2[0])) if self._mask[j]]
                for i in range(len(X2))]

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def get_support(self, indices=False):
        if indices:
            return [j for j, m in enumerate(self._mask) if m]
        return self._mask[:]


def _chi2_stat(X, y):
    """Compute chi2 statistic between each feature and y."""
    X2 = [[float(v) for v in row] for row in X]
    classes = _uniq_sorted(y)
    n, p = len(X2), len(X2[0])
    stats = []
    for j in range(p):
        col = _col(X2, j)
        # Discretize into bins
        n_bins = min(10, len(set(col)))
        mn, mx = min(col), max(col)
        bw = (mx - mn) / n_bins if mx > mn else 1.0
        bin_col = [min(int((v - mn) / bw), n_bins - 1) for v in col]
        bins = list(range(n_bins))
        chi2 = 0.0
        for c in classes:
            n_c = sum(1 for v in y if v == c)
            for b in bins:
                observed = sum(1 for i in range(n) if bin_col[i] == b and y[i] == c)
                expected = (n_c / n) * sum(1 for bc in bin_col if bc == b)
                if expected > 0:
                    chi2 += (observed - expected) ** 2 / expected
        stats.append(chi2)
    return stats


def chi2(X, y):
    stats = _chi2_stat(X, y)
    # Return (stats, pvalues) — pvalues approximated as 1
    return stats, [1.0] * len(stats)


def _f_classif_stats(X, y):
    X2 = [[float(v) for v in row] for row in X]
    classes = _uniq_sorted(y)
    n, p = len(X2), len(X2[0])
    stats = []
    for j in range(p):
        col = _col(X2, j)
        overall_mean = _mean(col)
        K = len(classes)
        group_means = {c: _mean([col[i] for i in range(n) if y[i] == c]) for c in classes}
        group_sizes = {c: sum(1 for v in y if v == c) for c in classes}
        SS_between = sum(group_sizes[c] * (group_means[c] - overall_mean) ** 2 for c in classes)
        SS_within = sum(sum((col[i] - group_means[y[i]]) ** 2 for i in range(n) if y[i] == c) for c in classes)
        df_between = K - 1
        df_within = n - K
        if df_between == 0 or df_within <= 0 or SS_within == 0:
            stats.append(0.0)
        else:
            F = (SS_between / df_between) / (SS_within / df_within)
            stats.append(F)
    return stats


def f_classif(X, y):
    stats = _f_classif_stats(X, y)
    return stats, [1.0] * len(stats)


def f_regression(X, y):
    X2 = [[float(v) for v in row] for row in X]
    y2 = [float(v) for v in y]
    n, p = len(X2), len(X2[0])
    y_mean = _mean(y2)
    stats = []
    for j in range(p):
        col = _col(X2, j)
        x_mean = _mean(col)
        cov = sum((col[i] - x_mean) * (y2[i] - y_mean) for i in range(n))
        ss_x = sum((v - x_mean) ** 2 for v in col)
        ss_y = sum((v - y_mean) ** 2 for v in y2)
        if ss_x < 1e-15 or ss_y < 1e-15:
            stats.append(0.0)
            continue
        r = cov / math.sqrt(ss_x * ss_y)
        r = max(-1.0, min(1.0, r))
        F = r ** 2 * (n - 2) / (1 - r ** 2 + 1e-15)
        stats.append(F)
    return stats, [1.0] * len(stats)


def _mutual_info(x_vals, y_vals, n_bins=10):
    n = len(x_vals)
    mn, mx = min(x_vals), max(x_vals)
    bw = (mx - mn) / n_bins if mx > mn else 1.0
    x_bins = [min(int((v - mn) / bw), n_bins - 1) for v in x_vals]
    y_classes = _uniq_sorted(y_vals)
    p_x = {}
    for b in x_bins:
        p_x[b] = p_x.get(b, 0) + 1.0 / n
    p_y = {c: sum(1 for v in y_vals if v == c) / n for c in y_classes}
    mi = 0.0
    for b in set(x_bins):
        for c in y_classes:
            joint = sum(1 for i in range(n) if x_bins[i] == b and y_vals[i] == c) / n
            if joint > 0:
                mi += joint * math.log(joint / (p_x.get(b, 1e-10) * p_y.get(c, 1e-10)))
    return max(mi, 0.0)


def mutual_info_classif(X, y, n_neighbors=3, random_state=None):
    X2 = [[float(v) for v in row] for row in X]
    p = len(X2[0]) if X2 else 0
    return [_mutual_info(_col(X2, j), y) for j in range(p)]


def mutual_info_regression(X, y, n_neighbors=3, random_state=None):
    X2 = [[float(v) for v in row] for row in X]
    y2 = [float(v) for v in y]
    p = len(X2[0]) if X2 else 0
    n_bins = 10
    n = len(y2)
    mn_y, mx_y = min(y2), max(y2)
    bw_y = (mx_y - mn_y) / n_bins if mx_y > mn_y else 1.0
    y_bins = [min(int((v - mn_y) / bw_y), n_bins - 1) for v in y2]
    return [_mutual_info(_col(X2, j), y_bins) for j in range(p)]


class SelectKBest:
    def __init__(self, score_func=f_classif, k=10):
        self.score_func = score_func
        self.k = k

    def fit(self, X, y):
        result = self.score_func(X, y)
        self.scores_ = result[0] if isinstance(result, tuple) else result
        self.pvalues_ = result[1] if isinstance(result, tuple) else [1.0] * len(self.scores_)
        k = self.k if self.k != 'all' else len(self.scores_)
        order = _argsort(self.scores_, reverse=True)
        top = set(order[:k])
        self._mask = [j in top for j in range(len(self.scores_))]
        return self

    def transform(self, X):
        X2 = [[float(v) for v in row] for row in X]
        return [[X2[i][j] for j in range(len(X2[0])) if self._mask[j]]
                for i in range(len(X2))]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_support(self, indices=False):
        if indices:
            return [j for j, m in enumerate(self._mask) if m]
        return self._mask[:]


class SelectFromModel:
    def __init__(self, estimator, threshold=None, max_features=None, prefit=False):
        self.estimator = estimator
        self.threshold = threshold
        self.max_features = max_features
        self.prefit = prefit

    def fit(self, X, y):
        if not self.prefit:
            self.estimator_ = self.estimator
            self.estimator_.fit(X, y)
        else:
            self.estimator_ = self.estimator
        coef = getattr(self.estimator_, 'coef_', None) or getattr(self.estimator_, 'feature_importances_', None)
        if coef is None:
            self._mask = [True] * len(X[0])
            return self
        if isinstance(coef[0], list):
            importances = [math.sqrt(sum(coef[k][j] ** 2 for k in range(len(coef)))) for j in range(len(coef[0]))]
        else:
            importances = [abs(v) for v in coef]
        thr = self.threshold
        if thr is None or thr == 'mean':
            thr = _mean(importances)
        elif thr == 'median':
            thr = _median(importances)
        else:
            thr = float(thr)
        self._mask = [imp >= thr for imp in importances]
        return self

    def transform(self, X):
        X2 = [[float(v) for v in row] for row in X]
        return [[X2[i][j] for j in range(len(X2[0])) if self._mask[j]]
                for i in range(len(X2))]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class RFE:
    def __init__(self, estimator, n_features_to_select=None, step=1):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step

    def fit(self, X, y):
        import copy
        X2 = [[float(v) for v in row] for row in X]
        p = len(X2[0]) if X2 else 0
        n_to_select = self.n_features_to_select or max(1, p // 2)
        support = list(range(p))
        ranking = [1] * p
        while len(support) > n_to_select:
            Xs = [[X2[i][j] for j in support] for i in range(len(X2))]
            est = copy.deepcopy(self.estimator)
            est.fit(Xs, y)
            coef = getattr(est, 'coef_', None) or getattr(est, 'feature_importances_', None)
            if coef is None:
                break
            if isinstance(coef[0], list):
                importances = [math.sqrt(sum(coef[k][j] ** 2 for k in range(len(coef)))) for j in range(len(support))]
            else:
                importances = [abs(v) for v in coef]
            n_remove = min(self.step, len(support) - n_to_select)
            worst = _argsort(importances)[:n_remove]
            worst_set = set(worst)
            for rank_idx in worst_set:
                ranking[support[rank_idx]] = len(support) - n_to_select + 1
            support = [support[j] for j in range(len(support)) if j not in worst_set]
        self.support_ = [j in support for j in range(p)]
        self.ranking_ = ranking
        self.n_features_ = len(support)
        # Final fit
        Xs = [[X2[i][j] for j in support] for i in range(len(X2))]
        self.estimator_ = copy.deepcopy(self.estimator)
        self.estimator_.fit(Xs, y)
        return self

    def transform(self, X):
        X2 = [[float(v) for v in row] for row in X]
        return [[X2[i][j] for j in range(len(X2[0])) if self.support_[j]]
                for i in range(len(X2))]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class SelectPercentile:
    def __init__(self, score_func=None, percentile=10):
        self.score_func = score_func or f_classif
        self.percentile = percentile

    def fit(self, X, y):
        scores, _ = self.score_func(X, y)
        k = max(1, int(len(scores) * self.percentile / 100))
        threshold = sorted(scores, reverse=True)[k - 1]
        self.support_ = [s >= threshold for s in scores]
        self.scores_ = scores
        return self

    def transform(self, X):
        X2 = [[float(v) for v in row] for row in X]
        return [[X2[i][j] for j in range(len(X2[0])) if self.support_[j]]
                for i in range(len(X2))]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_support(self):
        return list(self.support_)


class GenericUnivariateSelect:
    def __init__(self, score_func=None, mode='percentile', param=1e-5):
        self.score_func = score_func or f_classif
        self.mode = mode
        self.param = param

    def fit(self, X, y):
        scores, pvalues = self.score_func(X, y)
        p = len(scores)
        if self.mode == 'percentile':
            k = max(1, int(p * self.param / 100))
            threshold = sorted(scores, reverse=True)[k - 1]
            self.support_ = [s >= threshold for s in scores]
        elif self.mode == 'k_best':
            k = max(1, int(self.param))
            threshold = sorted(scores, reverse=True)[min(k - 1, p - 1)]
            self.support_ = [s >= threshold for s in scores]
        elif self.mode == 'fpr':
            self.support_ = [pv <= self.param for pv in pvalues]
        elif self.mode == 'fdr':
            m = sum(1 for pv in pvalues if pv <= self.param)
            self.support_ = [pv <= self.param * (i + 1) / p
                              for i, pv in enumerate(sorted(range(p), key=lambda x: pvalues[x]),)]
        elif self.mode == 'fwe':
            self.support_ = [pv <= self.param / p for pv in pvalues]
        else:
            self.support_ = [True] * p
        self.scores_ = scores
        return self

    def transform(self, X):
        X2 = [[float(v) for v in row] for row in X]
        return [[X2[i][j] for j in range(len(X2[0])) if self.support_[j]]
                for i in range(len(X2))]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_support(self):
        return list(self.support_)


class SelectFpr(SelectPercentile):
    """Select features with p-value below alpha (False Positive Rate)."""
    def __init__(self, score_func=None, alpha=5e-2):
        self.score_func = score_func or f_classif
        self.alpha = alpha

    def fit(self, X, y):
        scores, pvalues = self.score_func(X, y)
        self.support_ = [pv <= self.alpha for pv in pvalues]
        self.scores_ = scores
        return self


class SelectFdr(SelectPercentile):
    """Select features with FDR-corrected p-value below alpha."""
    def __init__(self, score_func=None, alpha=5e-2):
        self.score_func = score_func or f_classif
        self.alpha = alpha

    def fit(self, X, y):
        scores, pvalues = self.score_func(X, y)
        p = len(pvalues)
        order = sorted(range(p), key=lambda i: pvalues[i])
        self.support_ = [False] * p
        for rank, idx in enumerate(order):
            if pvalues[idx] <= self.alpha * (rank + 1) / p:
                self.support_[idx] = True
        self.scores_ = scores
        return self


class SelectFwe(SelectPercentile):
    """Select features with Bonferroni-corrected p-value below alpha."""
    def __init__(self, score_func=None, alpha=5e-2):
        self.score_func = score_func or f_classif
        self.alpha = alpha

    def fit(self, X, y):
        scores, pvalues = self.score_func(X, y)
        p = len(pvalues)
        self.support_ = [pv <= self.alpha / p for pv in pvalues]
        self.scores_ = scores
        return self


class RFECV:
    """Recursive Feature Elimination with Cross-Validation."""
    def __init__(self, estimator, step=1, cv=5, scoring=None, min_features_to_select=1):
        self.estimator = estimator
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.min_features_to_select = min_features_to_select

    def fit(self, X, y):
        import copy
        from vectorphile.ml.model_selection import KFold
        X2 = [[float(v) for v in row] for row in X]
        y2 = list(y)
        p = len(X2[0])
        feature_mask = list(range(p))
        best_score = -1e18
        best_mask = list(feature_mask)
        while len(feature_mask) >= self.min_features_to_select:
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
            scores = []
            for train_idx, test_idx in kf.split(X2):
                Xtr = [[X2[i][j] for j in feature_mask] for i in train_idx]
                Xte = [[X2[i][j] for j in feature_mask] for i in test_idx]
                ytr = [y2[i] for i in train_idx]
                yte = [y2[i] for i in test_idx]
                est = copy.deepcopy(self.estimator)
                est.fit(Xtr, ytr)
                scores.append(est.score(Xte, yte))
            mean_score = sum(scores) / len(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_mask = list(feature_mask)
            # Remove least important features
            est = copy.deepcopy(self.estimator)
            Xf = [[X2[i][j] for j in feature_mask] for i in range(len(X2))]
            est.fit(Xf, y2)
            importances = getattr(est, 'feature_importances_', None)
            if importances is None:
                coef = getattr(est, 'coef_', None)
                importances = [abs(coef[j]) for j in range(len(feature_mask))] if coef else [1.0] * len(feature_mask)
            remove_n = max(1, min(self.step, len(feature_mask) - self.min_features_to_select))
            order = sorted(range(len(feature_mask)), key=lambda i: importances[i])
            remove_set = set(order[:remove_n])
            feature_mask = [feature_mask[i] for i in range(len(feature_mask)) if i not in remove_set]
        self.support_ = [i in best_mask for i in range(p)]
        self.n_features_ = len(best_mask)
        return self

    def transform(self, X):
        X2 = [[float(v) for v in row] for row in X]
        return [[X2[i][j] for j in range(len(X2[0])) if self.support_[j]]
                for i in range(len(X2))]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_support(self):
        return list(self.support_)
