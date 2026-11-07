"""Feature selection — VarianceThreshold, SelectKBest, RFE, SelectFromModel, etc."""
from __future__ import annotations
import numpy as np
from typing import Optional, Union, Callable


class VarianceThreshold:
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.variances_ = None
        self._mask = None

    def fit(self, X, y=None):
        X = np.array(X, dtype=np.float64)
        self.variances_ = X.var(axis=0)
        self._mask = self.variances_ > self.threshold
        return self

    def transform(self, X):
        return np.array(X, dtype=np.float64)[:, self._mask]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_support(self, indices=False):
        if indices:
            return np.where(self._mask)[0]
        return self._mask

    def inverse_transform(self, X):
        n_features = len(self._mask)
        result = np.zeros((X.shape[0], n_features))
        result[:, self._mask] = X
        return result


def chi2(X, y):
    """Chi-squared statistic for classification feature selection."""
    X = np.array(X, dtype=np.float64)
    classes = np.unique(y)
    n = len(y)
    scores = np.zeros(X.shape[1])
    pvals = np.zeros(X.shape[1])
    for f in range(X.shape[1]):
        observed = np.array([[((y == c) & (X[:, f] > np.median(X[:, f]))).sum(),
                               ((y == c) & (X[:, f] <= np.median(X[:, f]))).sum()]
                              for c in classes], dtype=np.float64)
        expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
        chi_sq = ((observed - expected)**2 / (expected + 1e-12)).sum()
        scores[f] = chi_sq
        # Approximate p-value via chi2 CDF
        df = (len(classes) - 1) * 1
        try:
            from scipy.stats import chi2 as sp_chi2
            pvals[f] = 1 - sp_chi2.cdf(chi_sq, df)
        except ImportError:
            pvals[f] = np.exp(-chi_sq / 2)
    return scores, pvals


def f_classif(X, y):
    """ANOVA F-statistic for classification."""
    X = np.array(X, dtype=np.float64)
    classes = np.unique(y)
    n = len(y)
    scores = np.zeros(X.shape[1])
    pvals = np.zeros(X.shape[1])
    for f in range(X.shape[1]):
        grand_mean = X[:, f].mean()
        ss_between = sum(((X[y==c, f].mean() - grand_mean)**2 * (y==c).sum()) for c in classes)
        ss_within = sum(((X[y==c, f] - X[y==c, f].mean())**2).sum() for c in classes)
        df_between = len(classes) - 1
        df_within = n - len(classes)
        f_stat = (ss_between / (df_between + 1e-12)) / (ss_within / (df_within + 1e-12))
        scores[f] = f_stat
        try:
            from scipy.stats import f as sp_f
            pvals[f] = 1 - sp_f.cdf(f_stat, df_between, df_within)
        except ImportError:
            pvals[f] = np.exp(-f_stat / 2)
    return scores, pvals


def f_regression(X, y):
    """F-statistic for regression feature selection."""
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    scores = np.zeros(X.shape[1])
    pvals = np.zeros(X.shape[1])
    n = len(y)
    for f in range(X.shape[1]):
        x = X[:, f]
        corr = np.corrcoef(x, y)[0, 1]
        f_stat = corr**2 * (n - 2) / (1 - corr**2 + 1e-12)
        scores[f] = f_stat
        try:
            from scipy.stats import f as sp_f
            pvals[f] = 1 - sp_f.cdf(f_stat, 1, n - 2)
        except ImportError:
            pvals[f] = np.exp(-f_stat / 2)
    return scores, pvals


def mutual_info_classif(X, y, discrete_features=False, n_neighbors=3, random_state=None):
    """Mutual information for classification."""
    X = np.array(X, dtype=np.float64)
    y = np.array(y)
    n, m = X.shape
    mi = np.zeros(m)
    classes = np.unique(y)
    for f in range(m):
        x = X[:, f]
        h_x = -np.sum([p * np.log2(p + 1e-12) for p in
                        (c / n for c in np.histogram(x, bins=min(n//5, 50))[0] if c > 0)])
        h_xy = sum(
            (y==c).mean() * (-np.sum([p * np.log2(p + 1e-12) for p in
                               (cnt / (y==c).sum() for cnt in np.histogram(x[y==c], bins=min((y==c).sum()//2+1, 20))[0] if cnt > 0)]))
            for c in classes
        )
        mi[f] = max(0, h_x - h_xy)
    return mi


def mutual_info_regression(X, y, discrete_features=False, n_neighbors=3, random_state=None):
    """Mutual information for regression via binning."""
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    n, m = X.shape
    bins = min(n // 5, 50)
    mi = np.zeros(m)
    y_hist, _ = np.histogram(y, bins=bins)
    p_y = y_hist / n
    h_y = -np.sum(p_y[p_y > 0] * np.log2(p_y[p_y > 0]))
    for f in range(m):
        x = X[:, f]
        joint_hist, _, _ = np.histogram2d(x, y, bins=bins)
        p_joint = joint_hist / n
        p_x = p_joint.sum(axis=1)
        h_x = -np.sum(p_x[p_x > 0] * np.log2(p_x[p_x > 0]))
        p_xy_nonzero = p_joint[p_joint > 0]
        p_x_rep = (p_joint / (p_x[:, None] + 1e-12))[p_joint > 0]
        h_x_given_y = -np.sum(p_xy_nonzero * np.log2(p_x_rep + 1e-12))
        mi[f] = max(0, h_x - h_x_given_y)
    return mi


class SelectKBest:
    def __init__(self, score_func=f_classif, k=10):
        self.score_func = score_func
        self.k = k
        self.scores_ = None
        self.pvalues_ = None
        self._mask = None

    def fit(self, X, y):
        result = self.score_func(X, y)
        if isinstance(result, tuple):
            self.scores_, self.pvalues_ = result
        else:
            self.scores_ = result
            self.pvalues_ = None
        k = min(self.k, len(self.scores_)) if self.k != 'all' else len(self.scores_)
        top_k = np.argsort(self.scores_)[-k:]
        self._mask = np.zeros(len(self.scores_), dtype=bool)
        self._mask[top_k] = True
        return self

    def transform(self, X):
        return np.array(X)[:, self._mask]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_support(self, indices=False):
        if indices:
            return np.where(self._mask)[0]
        return self._mask

    def inverse_transform(self, X):
        result = np.zeros((X.shape[0], len(self._mask)))
        result[:, self._mask] = X
        return result


class SelectPercentile:
    def __init__(self, score_func=f_classif, percentile=10):
        self.score_func = score_func
        self.percentile = percentile
        self.scores_ = None
        self._mask = None

    def fit(self, X, y):
        result = self.score_func(X, y)
        self.scores_ = result[0] if isinstance(result, tuple) else result
        threshold = np.percentile(self.scores_, 100 - self.percentile)
        self._mask = self.scores_ >= threshold
        return self

    def transform(self, X):
        return np.array(X)[:, self._mask]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_support(self, indices=False):
        return np.where(self._mask)[0] if indices else self._mask


class GenericUnivariateSelect:
    def __init__(self, score_func=f_classif, mode='percentile', param=1e-5):
        self.score_func = score_func
        self.mode = mode
        self.param = param
        self._selector = None

    def fit(self, X, y):
        if self.mode == 'percentile':
            self._selector = SelectPercentile(self.score_func, int(self.param))
        elif self.mode == 'k_best':
            self._selector = SelectKBest(self.score_func, int(self.param))
        elif self.mode == 'fpr':
            self._selector = SelectFpr(self.score_func, self.param)
        else:
            self._selector = SelectKBest(self.score_func, 10)
        self._selector.fit(X, y)
        return self

    def transform(self, X):
        return self._selector.transform(X)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_support(self, indices=False):
        return self._selector.get_support(indices)


class SelectFpr:
    def __init__(self, score_func=f_classif, alpha=5e-2):
        self.score_func = score_func
        self.alpha = alpha
        self._mask = None

    def fit(self, X, y):
        result = self.score_func(X, y)
        pvalues = result[1] if isinstance(result, tuple) else np.ones(X.shape[1])
        self._mask = pvalues < self.alpha
        return self

    def transform(self, X):
        return np.array(X)[:, self._mask]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_support(self, indices=False):
        return np.where(self._mask)[0] if indices else self._mask


class SelectFdr(SelectFpr):
    """Benjamini-Hochberg FDR control."""
    def fit(self, X, y):
        result = self.score_func(X, y)
        pvalues = result[1] if isinstance(result, tuple) else np.ones(X.shape[1])
        m = len(pvalues)
        sorted_p = np.sort(pvalues)
        bh_threshold = sorted_p[np.arange(1, m+1) * self.alpha / m <= sorted_p].min() if any(np.arange(1, m+1) * self.alpha / m >= sorted_p) else 0
        self._mask = pvalues <= bh_threshold
        return self


class SelectFwe(SelectFpr):
    """Bonferroni FWER control."""
    def fit(self, X, y):
        result = self.score_func(X, y)
        pvalues = result[1] if isinstance(result, tuple) else np.ones(X.shape[1])
        self._mask = pvalues < self.alpha / len(pvalues)
        return self


class SelectFromModel:
    def __init__(self, estimator, threshold=None, prefit=False, norm_order=1,
                 max_features=None):
        self.estimator = estimator
        self.threshold = threshold
        self.prefit = prefit
        self.norm_order = norm_order
        self.max_features = max_features
        self._mask = None

    def fit(self, X, y=None):
        import copy
        X = np.array(X, dtype=np.float64)
        if not self.prefit:
            self.estimator_ = copy.deepcopy(self.estimator)
            self.estimator_.fit(X, y)
        else:
            self.estimator_ = self.estimator
        importances = self._get_importances()
        threshold = self.threshold
        if threshold is None:
            threshold = 'mean'
        if isinstance(threshold, str):
            if threshold == 'mean':
                threshold = importances.mean()
            elif threshold == 'median':
                threshold = np.median(importances)
            elif threshold.startswith('1.'):
                threshold = float(threshold) * importances.mean()
        self._mask = importances >= threshold
        if self.max_features is not None:
            top_k = np.argsort(importances)[-self.max_features:]
            mask2 = np.zeros(len(importances), dtype=bool)
            mask2[top_k] = True
            self._mask = self._mask & mask2
        return self

    def _get_importances(self):
        est = self.estimator_
        if hasattr(est, 'feature_importances_') and est.feature_importances_ is not None:
            return est.feature_importances_
        if hasattr(est, 'coef_'):
            coef = est.coef_
            if coef.ndim > 1:
                return np.linalg.norm(coef, ord=self.norm_order, axis=0)
            return np.abs(coef)
        raise AttributeError("estimator has neither feature_importances_ nor coef_")

    def transform(self, X):
        return np.array(X, dtype=np.float64)[:, self._mask]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_support(self, indices=False):
        return np.where(self._mask)[0] if indices else self._mask

    def inverse_transform(self, X):
        result = np.zeros((X.shape[0], len(self._mask)))
        result[:, self._mask] = X
        return result


class RFE:
    """Recursive Feature Elimination."""
    def __init__(self, estimator, n_features_to_select=None, step=1, verbose=0):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.verbose = verbose
        self.support_ = None
        self.ranking_ = None

    def fit(self, X, y):
        import copy
        X = np.array(X, dtype=np.float64)
        n_features = X.shape[1]
        n_to_select = self.n_features_to_select or (n_features // 2)
        step = int(max(1, min(self.step if isinstance(self.step, int) else int(self.step * n_features), n_features - n_to_select)))
        support = np.ones(n_features, dtype=bool)
        ranking = np.ones(n_features, dtype=int)
        while support.sum() > n_to_select:
            feat_idx = np.where(support)[0]
            est = copy.deepcopy(self.estimator)
            est.fit(X[:, feat_idx], y)
            if hasattr(est, 'feature_importances_') and est.feature_importances_ is not None:
                importances = est.feature_importances_
            elif hasattr(est, 'coef_'):
                coef = est.coef_
                importances = np.linalg.norm(coef.reshape(-1, len(feat_idx)), axis=0) if coef.ndim > 1 else np.abs(coef)
            else:
                break
            # Remove lowest importance features
            n_remove = min(step, support.sum() - n_to_select)
            worst = np.argsort(importances)[:n_remove]
            remove_idx = feat_idx[worst]
            support[remove_idx] = False
            ranking[remove_idx] = support.sum() + 1
            if self.verbose > 0:
                print(f'[RFE] Features remaining: {support.sum()}')
        self.support_ = support
        self.ranking_ = ranking
        self.estimator_ = copy.deepcopy(self.estimator)
        self.estimator_.fit(X[:, support], y)
        return self

    def transform(self, X):
        return np.array(X, dtype=np.float64)[:, self.support_]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def predict(self, X):
        return self.estimator_.predict(self.transform(X))

    def score(self, X, y):
        return self.estimator_.score(self.transform(X), y)

    def get_support(self, indices=False):
        return np.where(self.support_)[0] if indices else self.support_


class RFECV(RFE):
    """RFE with cross-validation to select optimal number of features."""
    def __init__(self, estimator, step=1, cv=5, scoring=None, min_features_to_select=1, verbose=0):
        super().__init__(estimator, None, step, verbose)
        self.cv = cv
        self.scoring = scoring
        self.min_features_to_select = min_features_to_select
        self.cv_results_ = {}
        self.n_features_ = None

    def fit(self, X, y):
        import copy
        from vectorphile.ml.model_selection import KFold
        X = np.array(X, dtype=np.float64)
        n_features = X.shape[1]
        step = max(1, int(self.step * n_features) if isinstance(self.step, float) else self.step)
        # Try each subset size and pick best via CV
        kf = KFold(n_splits=self.cv)
        sizes = list(range(self.min_features_to_select, n_features + 1, step))
        if n_features not in sizes:
            sizes.append(n_features)
        sizes = sorted(set(sizes))
        mean_scores = []
        for n_sel in sizes:
            self.n_features_to_select = n_sel
            fold_scores = []
            for train_idx, val_idx in kf.split(X):
                self.fit.__func__(self, X[train_idx], y[train_idx]) if False else None
                # Simplified: fit RFE with n_sel on train, score on val
                rfe = RFE(copy.deepcopy(self.estimator), n_sel, step, 0)
                rfe.fit(X[train_idx], y[train_idx])
                fold_scores.append(rfe.score(X[val_idx], y[val_idx]))
            mean_scores.append(np.mean(fold_scores))
        best_idx = np.argmax(mean_scores)
        self.n_features_ = sizes[best_idx]
        self.cv_results_ = {'mean_test_score': mean_scores, 'n_features': sizes}
        self.n_features_to_select = self.n_features_
        super().fit(X, y)
        return self
