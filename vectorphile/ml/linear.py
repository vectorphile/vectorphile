"""Linear models — pure Python."""
import math
import random as _random
from typing import Optional


def _col(X, j):
    return [X[i][j] for i in range(len(X))]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _mat_vec(A, v):
    return [_dot(row, v) for row in A]


def _transpose(A):
    n, m = len(A), len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]


def _mat_mul(A, B):
    n, k = len(A), len(A[0])
    m = len(B[0])
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for l in range(k):
            if A[i][l] == 0.0:
                continue
            for j in range(m):
                C[i][j] += A[i][l] * B[l][j]
    return C


def _solve(A, b):
    """Solve Ax=b via Gaussian elimination with partial pivoting."""
    n = len(b)
    M = [A[i][:] + [float(b[i])] for i in range(n)]
    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[max_row] = M[max_row], M[col]
        pivot = M[col][col]
        if abs(pivot) < 1e-15:
            continue
        for row in range(col + 1, n):
            factor = M[row][col] / pivot
            M[row] = [M[row][j] - factor * M[col][j] for j in range(n + 1)]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i] if abs(M[i][i]) > 1e-15 else 1e-15
    return x


def _lstsq(A, b):
    """Solve least-squares via normal equations (A^T A) x = A^T b."""
    AT = _transpose(A)
    ATA = _mat_mul(AT, A)
    ATb = _mat_vec(AT, b)
    return _solve(ATA, ATb)


class _BaseLinear:
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


class LinearRegression(_BaseLinear):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X2 = [[float(v) for v in row] for row in X]
        y2 = [float(v) for v in y]
        n, p = len(X2), len(X2[0])
        if self.fit_intercept:
            Xa = [row + [1.0] for row in X2]
            coeffs = _lstsq(Xa, y2)
            self.coef_ = coeffs[:-1]
            self.intercept_ = coeffs[-1]
        else:
            self.coef_ = _lstsq(X2, y2)
            self.intercept_ = 0.0
        return self


class Ridge(_BaseLinear):
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X2 = [[float(v) for v in row] for row in X]
        y2 = [float(v) for v in y]
        n, p = len(X2), len(X2[0])
        if self.fit_intercept:
            x_mean = [sum(_col(X2, j)) / n for j in range(p)]
            y_mean = sum(y2) / n
            Xc = [[X2[i][j] - x_mean[j] for j in range(p)] for i in range(n)]
            yc = [v - y_mean for v in y2]
        else:
            Xc, yc, x_mean, y_mean = X2, y2, [0.0] * p, 0.0
        AT = _transpose(Xc)
        ATA = _mat_mul(AT, Xc)
        for i in range(p):
            ATA[i][i] += self.alpha
        ATb = _mat_vec(AT, yc)
        self.coef_ = _solve(ATA, ATb)
        if self.fit_intercept:
            self.intercept_ = y_mean - _dot(self.coef_, x_mean)
        else:
            self.intercept_ = 0.0
        return self


class Lasso(_BaseLinear):
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4, fit_intercept=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X2 = [[float(v) for v in row] for row in X]
        y2 = [float(v) for v in y]
        n, p = len(X2), len(X2[0])
        if self.fit_intercept:
            x_mean = [sum(_col(X2, j)) / n for j in range(p)]
            y_mean = sum(y2) / n
            Xc = [[X2[i][j] - x_mean[j] for j in range(p)] for i in range(n)]
            yc = [v - y_mean for v in y2]
        else:
            Xc, yc, x_mean, y_mean = X2, y2, [0.0] * p, 0.0

        # Precompute column norms squared
        col_norm2 = [sum(Xc[i][j] ** 2 for i in range(n)) for j in range(p)]
        w = [0.0] * p
        for _ in range(self.max_iter):
            w_old = w[:]
            r = [yc[i] - _dot(Xc[i], w) for i in range(n)]
            for j in range(p):
                # Add back j's contribution to residual
                rj = [r[i] + Xc[i][j] * w[j] for i in range(n)]
                rho = sum(Xc[i][j] * rj[i] for i in range(n))
                cn2 = col_norm2[j]
                if cn2 < 1e-15:
                    w[j] = 0.0
                else:
                    thr = self.alpha * n
                    if rho > thr:
                        w[j] = (rho - thr) / cn2
                    elif rho < -thr:
                        w[j] = (rho + thr) / cn2
                    else:
                        w[j] = 0.0
                r = [rj[i] - Xc[i][j] * w[j] for i in range(n)]
            if max(abs(w[j] - w_old[j]) for j in range(p)) < self.tol:
                break
        self.coef_ = w
        if self.fit_intercept:
            self.intercept_ = y_mean - _dot(w, x_mean)
        else:
            self.intercept_ = 0.0
        return self


class ElasticNet(_BaseLinear):
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4, fit_intercept=True):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X2 = [[float(v) for v in row] for row in X]
        y2 = [float(v) for v in y]
        n, p = len(X2), len(X2[0])
        if self.fit_intercept:
            x_mean = [sum(_col(X2, j)) / n for j in range(p)]
            y_mean = sum(y2) / n
            Xc = [[X2[i][j] - x_mean[j] for j in range(p)] for i in range(n)]
            yc = [v - y_mean for v in y2]
        else:
            Xc, yc, x_mean, y_mean = X2, y2, [0.0] * p, 0.0

        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1 - self.l1_ratio)
        col_norm2 = [sum(Xc[i][j] ** 2 for i in range(n)) + l2 * n for j in range(p)]
        w = [0.0] * p
        for _ in range(self.max_iter):
            w_old = w[:]
            r = [yc[i] - _dot(Xc[i], w) for i in range(n)]
            for j in range(p):
                rj = [r[i] + Xc[i][j] * w[j] for i in range(n)]
                rho = sum(Xc[i][j] * rj[i] for i in range(n))
                cn2 = col_norm2[j]
                thr = l1 * n
                if rho > thr:
                    w[j] = (rho - thr) / cn2
                elif rho < -thr:
                    w[j] = (rho + thr) / cn2
                else:
                    w[j] = 0.0
                r = [rj[i] - Xc[i][j] * w[j] for i in range(n)]
            if max(abs(w[j] - w_old[j]) for j in range(p)) < self.tol:
                break
        self.coef_ = w
        if self.fit_intercept:
            self.intercept_ = y_mean - _dot(w, x_mean)
        else:
            self.intercept_ = 0.0
        return self


class LogisticRegression:
    def __init__(self, C=1.0, max_iter=100, tol=1e-4, fit_intercept=True,
                 multi_class='auto', solver='lbfgs'):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    @staticmethod
    def _sigmoid(z):
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def fit(self, X, y):
        X2 = [[float(v) for v in row] for row in X]
        y2 = list(y)
        self.classes_ = sorted(set(y2))
        K = len(self.classes_)
        n, p = len(X2), len(X2[0])
        lr = 0.1 / n
        reg = 1.0 / (self.C * n)
        if K == 2:
            y_bin = [1.0 if v == self.classes_[1] else 0.0 for v in y2]
            w = [0.0] * p
            b = 0.0
            for _ in range(self.max_iter):
                grad_w = [0.0] * p
                grad_b = 0.0
                for i in range(n):
                    z = _dot(X2[i], w) + b
                    pred = self._sigmoid(z)
                    err = pred - y_bin[i]
                    for j in range(p):
                        grad_w[j] += err * X2[i][j]
                    grad_b += err
                for j in range(p):
                    grad_w[j] = grad_w[j] / n + reg * w[j]
                    w[j] -= lr * grad_w[j]
                b -= lr * grad_b / n
            self.coef_ = [w]
            self.intercept_ = [b]
        else:
            # one-vs-rest
            self.coef_ = []
            self.intercept_ = []
            for c in self.classes_:
                y_bin = [1.0 if v == c else 0.0 for v in y2]
                w = [0.0] * p
                b = 0.0
                for _ in range(self.max_iter):
                    grad_w = [0.0] * p
                    grad_b = 0.0
                    for i in range(n):
                        z = _dot(X2[i], w) + b
                        pred = self._sigmoid(z)
                        err = pred - y_bin[i]
                        for j in range(p):
                            grad_w[j] += err * X2[i][j]
                        grad_b += err
                    for j in range(p):
                        grad_w[j] = grad_w[j] / n + reg * w[j]
                        w[j] -= lr * grad_w[j]
                    b -= lr * grad_b / n
                self.coef_.append(w)
                self.intercept_.append(b)
        return self

    def decision_function(self, X):
        X2 = [[float(v) for v in row] for row in X]
        scores = [[_dot(X2[i], self.coef_[k]) + self.intercept_[k]
                   for k in range(len(self.classes_))] for i in range(len(X2))]
        if len(self.classes_) == 2:
            return [row[1] for row in scores]
        return scores

    def predict_proba(self, X):
        X2 = [[float(v) for v in row] for row in X]
        K = len(self.classes_)
        result = []
        for i in range(len(X2)):
            if K == 2 and len(self.coef_) == 1:
                # Binary: one weight vector, return [1-p, p]
                p1 = self._sigmoid(_dot(X2[i], self.coef_[0]) + self.intercept_[0])
                result.append([1.0 - p1, p1])
            else:
                scores = [_dot(X2[i], self.coef_[k]) + self.intercept_[k] for k in range(K)]
                probs = [self._sigmoid(s) for s in scores]
                total = sum(probs) or 1.0
                probs = [p / total for p in probs]
                result.append(probs)
        return result

    def predict(self, X):
        proba = self.predict_proba(X)
        return [self.classes_[row.index(max(row))] for row in proba]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(a == b for a, b in zip(preds, y)) / len(y)


class Perceptron:
    def __init__(self, max_iter=1000, tol=1e-3, eta0=1.0, random_state=None):
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.random_state = random_state

    def fit(self, X, y):
        X2 = [[float(v) for v in row] for row in X]
        y2 = [1 if v > 0 else -1 for v in y]
        n, p = len(X2), len(X2[0])
        self.coef_ = [0.0] * p
        self.intercept_ = 0.0
        for _ in range(self.max_iter):
            errors = 0
            for i in range(n):
                z = _dot(X2[i], self.coef_) + self.intercept_
                pred = 1 if z >= 0 else -1
                if pred != y2[i]:
                    for j in range(p):
                        self.coef_[j] += self.eta0 * y2[i] * X2[i][j]
                    self.intercept_ += self.eta0 * y2[i]
                    errors += 1
            if errors == 0:
                break
        return self

    def predict(self, X):
        X2 = [[float(v) for v in row] for row in X]
        return [1 if _dot(row, self.coef_) + self.intercept_ >= 0 else -1 for row in X2]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(a == b for a, b in zip(preds, y)) / len(y)
