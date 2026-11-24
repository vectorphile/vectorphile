"""Support Vector Machines — pure Python."""
import math
import random as _random
from typing import Optional

from vectorphile.ml.linear import LinearRegression, _dot


def _uniq_sorted(seq):
    return sorted(set(seq))


class SVC:
    """Binary/multiclass SVM via Sequential Minimal Optimization (SMO)."""

    def __init__(self, C=1.0, kernel='rbf', gamma='scale', degree=3, coef0=0.0,
                 max_iter=1000, tol=1e-3, random_state=None, decision_function_shape='ovr'):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.decision_function_shape = decision_function_shape

    def _get_gamma(self, p):
        if self.gamma == 'scale':
            return 1.0 / p
        if self.gamma == 'auto':
            return 1.0 / p
        return float(self.gamma)

    def _kernel_fn(self, a, b, gamma):
        if self.kernel == 'linear':
            return _dot(a, b)
        if self.kernel == 'rbf':
            d2 = sum((x - y) ** 2 for x, y in zip(a, b))
            return math.exp(-gamma * d2)
        if self.kernel == 'poly':
            return (gamma * _dot(a, b) + self.coef0) ** self.degree
        if self.kernel == 'sigmoid':
            return math.tanh(gamma * _dot(a, b) + self.coef0)
        return _dot(a, b)

    def _smo(self, X, y, rng):
        n, p = len(X), len(X[0])
        gamma = self._get_gamma(p)
        # Build kernel matrix
        K = [[self._kernel_fn(X[i], X[j], gamma) for j in range(n)] for i in range(n)]
        alpha = [0.0] * n
        b = 0.0
        for _ in range(self.max_iter):
            num_changed = 0
            for i in range(n):
                Ei = sum(alpha[j] * y[j] * K[i][j] for j in range(n)) + b - y[i]
                if (y[i] * Ei < -self.tol and alpha[i] < self.C) or \
                   (y[i] * Ei > self.tol and alpha[i] > 0):
                    # Choose j randomly != i
                    j = rng.randint(0, n - 2)
                    if j >= i:
                        j += 1
                    Ej = sum(alpha[k] * y[k] * K[j][k] for k in range(n)) + b - y[j]
                    ai_old, aj_old = alpha[i], alpha[j]
                    if y[i] != y[j]:
                        L = max(0.0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0.0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])
                    if L >= H:
                        continue
                    eta = 2 * K[i][j] - K[i][i] - K[j][j]
                    if eta >= 0:
                        continue
                    alpha[j] -= y[j] * (Ei - Ej) / eta
                    alpha[j] = max(L, min(H, alpha[j]))
                    if abs(alpha[j] - aj_old) < 1e-5:
                        continue
                    alpha[i] += y[i] * y[j] * (aj_old - alpha[j])
                    b1 = b - Ei - y[i] * (alpha[i] - ai_old) * K[i][i] - \
                         y[j] * (alpha[j] - aj_old) * K[i][j]
                    b2 = b - Ej - y[i] * (alpha[i] - ai_old) * K[i][j] - \
                         y[j] * (alpha[j] - aj_old) * K[j][j]
                    b = (b1 + b2) / 2
                    num_changed += 1
            if num_changed == 0:
                break
        return alpha, b, gamma

    def fit(self, X, y):
        X2 = [[float(v) for v in row] for row in X]
        y2 = list(y)
        self.classes_ = _uniq_sorted(y2)
        rng = _random.Random(self.random_state)
        K = len(self.classes_)
        if K == 2:
            y_bin = [1.0 if v == self.classes_[1] else -1.0 for v in y2]
            self.alpha_, self.b_, self.gamma_ = self._smo(X2, y_bin, rng)
            self.support_vectors_ = [X2[i] for i in range(len(X2)) if self.alpha_[i] > 1e-5]
            self.support_y_ = [y_bin[i] for i in range(len(X2)) if self.alpha_[i] > 1e-5]
            self.support_alpha_ = [self.alpha_[i] for i in range(len(X2)) if self.alpha_[i] > 1e-5]
            self._X_train = X2
            self._y_bin = y_bin
        else:
            # one-vs-rest
            self._ovr_models = []
            for c in self.classes_:
                y_bin = [1.0 if v == c else -1.0 for v in y2]
                alpha, b, gamma = self._smo(X2, y_bin, rng)
                self._ovr_models.append((alpha, b, gamma, X2, y_bin))
        return self

    def decision_function(self, X):
        X2 = [[float(v) for v in row] for row in X]
        if len(self.classes_) == 2:
            return [sum(self.alpha_[j] * self._y_bin[j] *
                        self._kernel_fn(x, self._X_train[j], self.gamma_)
                        for j in range(len(self._X_train))) + self.b_
                    for x in X2]
        # OVR
        scores = []
        for x in X2:
            row_scores = []
            for alpha, b, gamma, X_tr, y_bin in self._ovr_models:
                s = sum(alpha[j] * y_bin[j] * self._kernel_fn(x, X_tr[j], gamma)
                        for j in range(len(X_tr))) + b
                row_scores.append(s)
            scores.append(row_scores)
        return scores

    def predict(self, X):
        df = self.decision_function(X)
        if len(self.classes_) == 2:
            return [self.classes_[1] if s >= 0 else self.classes_[0] for s in df]
        return [self.classes_[row.index(max(row))] for row in df]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(a == b for a, b in zip(preds, y)) / len(y)


class SVR:
    """Support Vector Regression via linear approximation with regularization."""

    def __init__(self, C=1.0, epsilon=0.1, kernel='rbf', gamma='scale', max_iter=1000,
                 tol=1e-3, random_state=None):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y):
        # Use gradient descent for linear SVR (simplest correct implementation)
        X2 = [[float(v) for v in row] for row in X]
        y2 = [float(v) for v in y]
        n, p = len(X2), len(X2[0])
        self.coef_ = [0.0] * p
        self.intercept_ = 0.0
        lr = 0.01
        reg = 1.0 / (self.C * n)
        for _ in range(self.max_iter):
            dw = [reg * self.coef_[j] for j in range(p)]
            db = 0.0
            for i in range(n):
                pred = _dot(X2[i], self.coef_) + self.intercept_
                residual = pred - y2[i]
                if residual > self.epsilon:
                    for j in range(p):
                        dw[j] += X2[i][j] / n
                    db += 1.0 / n
                elif residual < -self.epsilon:
                    for j in range(p):
                        dw[j] -= X2[i][j] / n
                    db -= 1.0 / n
            for j in range(p):
                self.coef_[j] -= lr * dw[j]
            self.intercept_ -= lr * db
        return self

    def predict(self, X):
        X2 = [[float(v) for v in row] for row in X]
        return [_dot(row, self.coef_) + self.intercept_ for row in X2]

    def score(self, X, y):
        preds = self.predict(X)
        y2 = [float(v) for v in y]
        mu = sum(y2) / len(y2)
        ss_res = sum((a - b) ** 2 for a, b in zip(y2, preds))
        ss_tot = sum((v - mu) ** 2 for v in y2)
        return float(1 - ss_res / (ss_tot + 1e-12))
