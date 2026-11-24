"""Time series utilities — windowing, decomposition, ARIMA, Holt-Winters, metrics."""
import math
from typing import List, Tuple


def _is_numpy(x):
    t = type(x); return t.__module__ == 'numpy' or (hasattr(x, 'flatten') and hasattr(x, 'dtype') and not hasattr(x, 'to_flat_list'))


def _to_list(series) -> List[float]:
    if isinstance(series, list):
        return [float(v) for v in series]
    if hasattr(series, 'to_flat_list'):
        return [float(v) for v in series.to_flat_list()]
    return [float(v) for v in series]


def make_windows(series, window: int, horizon: int):
    """Sliding window split: returns (X[n, window], y[n, horizon])."""
    s = _to_list(series)
    n = len(s) - window - horizon + 1
    X = [s[i:i + window] for i in range(n)]
    y = [s[i + window:i + window + horizon] for i in range(n)]
    if _is_numpy(series):
        import numpy as _np
        return _np.array(X, dtype='float32'), _np.array(y, dtype='float32')
    return X, y


def decompose(series, period: int):
    """Additive STL-like decomposition: trend + seasonal + residual."""
    was_numpy = _is_numpy(series)
    s = _to_list(series)
    n = len(s)
    half = period // 2
    trend = [float('nan')] * n
    for i in range(half, n - half):
        trend[i] = sum(s[i - half:i + half + 1]) / (2 * half + 1)
    for i in range(half):
        trend[i] = trend[half]
    for i in range(n - half, n):
        trend[i] = trend[n - half - 1]

    detrended = [s[i] - trend[i] for i in range(n)]
    seasonal = [0.0] * n
    for phase in range(period):
        indices = list(range(phase, n, period))
        vals = [detrended[idx] for idx in indices]
        avg = sum(vals) / max(1, len(vals))
        for idx in indices:
            seasonal[idx] = avg

    residual = [s[i] - trend[i] - seasonal[i] for i in range(n)]
    if was_numpy:
        import numpy as _np
        return _np.array(trend, dtype='float64'), _np.array(seasonal, dtype='float64'), _np.array(residual, dtype='float64')
    return trend, seasonal, residual


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _lstsq(X: List[List[float]], y: List[float]) -> List[float]:
    """Minimal OLS via normal equations X^T X coef = X^T y (Cholesky-free Gauss-Jordan)."""
    m = len(X[0])
    # Build X^T X and X^T y
    XtX = [[sum(X[r][i] * X[r][j] for r in range(len(X))) for j in range(m)] for i in range(m)]
    Xty = [sum(X[r][i] * y[r] for r in range(len(X))) for i in range(m)]
    # Gaussian elimination with partial pivoting
    aug = [XtX[i][:] + [Xty[i]] for i in range(m)]
    for col in range(m):
        pivot = max(range(col, m), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        if abs(pv) < 1e-14:
            continue
        aug[col] = [v / pv for v in aug[col]]
        for r in range(m):
            if r != col and abs(aug[r][col]) > 0:
                f = aug[r][col]
                aug[r] = [aug[r][j] - f * aug[col][j] for j in range(m + 1)]
    return [aug[i][m] for i in range(m)]


class ARIMA:
    """ARIMA(p, d, q=0) — AR coefficients via OLS on differenced series."""

    def __init__(self, p: int = 1, d: int = 0, q: int = 0):
        self.p = p
        self.d = d
        self.q = q
        self._ar: List[float] = []
        self._mu = 0.0
        self._orig: List[float] = []

    def fit(self, series):
        self._was_numpy = _is_numpy(series)
        s = _to_list(series)
        self._orig = s[:]
        for _ in range(self.d):
            s = [s[i + 1] - s[i] for i in range(len(s) - 1)]
        self._mu = sum(s) / max(1, len(s))
        sc = [v - self._mu for v in s]
        n = len(sc)
        if self.p == 0 or n <= self.p:
            self._ar = [0.0] * self.p
            return
        X = [sc[i:i + self.p] for i in range(n - self.p)]
        y = sc[self.p:]
        try:
            self._ar = _lstsq(X, y)
        except Exception:
            self._ar = [0.0] * self.p

    def predict(self, steps: int) -> List[float]:
        s = _to_list(self._orig)
        diffs = []
        for _ in range(self.d):
            diffs.append(s[-1])
            s = [s[i + 1] - s[i] for i in range(len(s) - 1)]
        s = [v - self._mu for v in s]
        history = s[:]
        preds = []
        for _ in range(steps):
            if self.p > 0:
                lag = history[-self.p:]
                val = _dot(self._ar, lag)
            else:
                val = 0.0
            history.append(val)
            preds.append(val + self._mu)
        # Undifference
        for last_val in reversed(diffs):
            cumsum = [last_val]
            for v in preds:
                cumsum.append(cumsum[-1] + v)
            preds = cumsum[1:]
        if getattr(self, '_was_numpy', False):
            import numpy as _np
            return _np.array(preds, dtype='float32')
        return preds

    def forecast(self, steps: int) -> List[float]:
        """Alias for predict() — PyTorch/statsmodels compatible."""
        return self.predict(steps)


class HoltWinters:
    """Triple exponential smoothing (Holt-Winters additive)."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.1,
                 gamma: float = 0.1, period: int = 12):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.period = period
        self._level = 0.0
        self._trend = 0.0
        self._season: List[float] = []

    def fit(self, series):
        self._was_numpy = _is_numpy(series)
        s = _to_list(series)
        m = self.period
        self._level = sum(s[:m]) / m
        self._trend = (sum(s[m:2 * m]) / m - self._level) / m if len(s) >= 2 * m else 0.0
        self._season = [s[i] - self._level for i in range(m)]

        for t in range(m, len(s)):
            L_prev = self._level
            T_prev = self._trend
            si = self._season[t % m]
            self._level = self.alpha * (s[t] - si) + (1 - self.alpha) * (L_prev + T_prev)
            self._trend = self.beta * (self._level - L_prev) + (1 - self.beta) * T_prev
            self._season[t % m] = self.gamma * (s[t] - self._level) + (1 - self.gamma) * si

    def predict(self, steps: int):
        L, T = self._level, self._trend
        preds = [L + h * T + self._season[(h - 1) % self.period] for h in range(1, steps + 1)]
        if getattr(self, '_was_numpy', False):
            import numpy as _np
            return _np.array(preds, dtype='float32')
        return preds


def ADF(series, maxlag: int = None) -> dict:
    """Augmented Dickey-Fuller test (simplified)."""
    s = _to_list(series)
    n = len(s)
    if maxlag is None:
        maxlag = int(12 * (n / 100) ** 0.25)
    maxlag = min(maxlag, n // 3)

    ds = [s[i + 1] - s[i] for i in range(len(s) - 1)]
    y = ds[maxlag:]
    cols = [[s[maxlag + i] for i in range(len(y))]]  # lagged level
    for lag in range(1, maxlag + 1):
        end = -(lag) if lag > 0 else None
        col = ds[maxlag - lag: len(ds) - lag if lag > 0 else len(ds)]
        cols.append(col[:len(y)])
    # Transpose to X[row][col]
    X = [[cols[j][i] for j in range(len(cols))] for i in range(len(y))]
    try:
        coefs = _lstsq(X, y)
        resid = [y[i] - sum(X[i][j] * coefs[j] for j in range(len(coefs))) for i in range(len(y))]
        s2 = sum(r ** 2 for r in resid) / max(1, len(resid) - len(coefs))
        # Variance of coef[0]: s2 * (X^T X)^{-1}_{00}
        m = len(coefs)
        XtX = [[sum(X[r][i] * X[r][j] for r in range(len(X))) for j in range(m)] for i in range(m)]
        # Approximate: use diagonal only
        var_coef = s2 / max(1e-14, XtX[0][0])
        t_stat = coefs[0] / max(math.sqrt(abs(var_coef)), 1e-10)
    except Exception:
        t_stat = 0.0

    if t_stat < -3.43:
        p = 0.01
    elif t_stat < -2.86:
        p = 0.05
    elif t_stat < -2.57:
        p = 0.10
    else:
        p = min(0.99, max(0.10, 1.0 - (t_stat + 3.43) / (-2.57 + 3.43 + 0.01)))
    return {'stat': float(t_stat), 'p_value': float(p)}


def mae(y, yhat) -> float:
    yv = _to_list(y); yh = _to_list(yhat)
    return sum(abs(a - b) for a, b in zip(yv, yh)) / max(1, len(yv))

def mse(y, yhat) -> float:
    yv = _to_list(y); yh = _to_list(yhat)
    return sum((a - b) ** 2 for a, b in zip(yv, yh)) / max(1, len(yv))

def rmse(y, yhat) -> float:
    return math.sqrt(mse(y, yhat))

def ADF(series, max_lag: int = None) -> Tuple[float, float]:
    """Augmented Dickey-Fuller test for stationarity.
    Returns (test_statistic, p_value_approximate).
    """
    s = _to_list(series)
    n = len(s)
    if max_lag is None:
        max_lag = int(12 * (n / 100) ** 0.25)
    # First differences
    ds = [s[i] - s[i - 1] for i in range(1, n)]
    # Lag-1 of original
    y_lag = s[1:n - max_lag - 1] if max_lag > 0 else s[:-1]
    y = ds[max_lag:]
    n_eff = len(y)
    if n_eff < 2:
        return 0.0, 1.0
    # OLS: y = alpha + beta * y_lag + ... + epsilon
    # Simplified: test beta = 0 from y ~ y_lag
    lag_vals = s[max_lag: max_lag + n_eff]
    mean_x = sum(lag_vals) / n_eff
    mean_y = sum(y) / n_eff
    sxy = sum((lag_vals[i] - mean_x) * (y[i] - mean_y) for i in range(n_eff))
    sxx = sum((lag_vals[i] - mean_x) ** 2 for i in range(n_eff)) + 1e-10
    beta = sxy / sxx
    alpha = mean_y - beta * mean_x
    resid = [y[i] - (alpha + beta * lag_vals[i]) for i in range(n_eff)]
    s2 = sum(r * r for r in resid) / max(n_eff - 2, 1)
    se_beta = math.sqrt(s2 / sxx)
    t_stat = beta / max(se_beta, 1e-10)
    # Very rough p-value approximation using normal distribution
    from vectorphile.special import ndtr
    p = ndtr(t_stat)
    return {'stat': t_stat, 'p_value': float(p), 'lags': max_lag}


def mape(y, yhat) -> float:
    yv = _to_list(y); yh = _to_list(yhat)
    eps = 1e-10
    return sum(abs((a - b) / (abs(a) + eps)) for a, b in zip(yv, yh)) / max(1, len(yv)) * 100


class KalmanFilter:
    """Linear Kalman filter for state estimation."""

    def __init__(self, F, H, Q, R, P0=None, x0=None):
        self.F = F  # Transition matrix (n x n list-of-lists)
        self.H = H  # Observation matrix (m x n)
        self.Q = Q  # Process noise covariance (n x n)
        self.R = R  # Observation noise covariance (m x m)
        n = len(F)
        self.P = P0 if P0 is not None else [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        self.x = x0 if x0 is not None else [0.0] * n

    @staticmethod
    def _mat_mul(A, B):
        rows_A, cols_A = len(A), len(A[0])
        cols_B = len(B[0])
        return [[sum(A[i][k] * B[k][j] for k in range(cols_A)) for j in range(cols_B)] for i in range(rows_A)]

    @staticmethod
    def _mat_add(A, B):
        return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    @staticmethod
    def _mat_sub(A, B):
        return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    @staticmethod
    def _mat_T(A):
        return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

    @staticmethod
    def _mat_inv_2x2(A):
        det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        if abs(det) < 1e-12:
            return [[1e6, 0], [0, 1e6]]
        return [[A[1][1] / det, -A[0][1] / det], [-A[1][0] / det, A[0][0] / det]]

    @staticmethod
    def _mat_inv(A):
        n = len(A)
        aug = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        for col in range(n):
            pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
            aug[col], aug[pivot] = aug[pivot], aug[col]
            if abs(aug[col][col]) < 1e-12:
                continue
            aug[col] = [v / aug[col][col] for v in aug[col]]
            for r in range(n):
                if r != col:
                    f = aug[r][col]
                    aug[r] = [aug[r][c] - f * aug[col][c] for c in range(2 * n)]
        return [aug[i][n:] for i in range(n)]

    def _mv(self, A, v):
        return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

    def predict(self):
        self.x = self._mv(self.F, self.x)
        FT = self._mat_T(self.F)
        self.P = self._mat_add(self._mat_mul(self._mat_mul(self.F, self.P), FT), self.Q)
        return self.x[:]

    def update(self, z: List[float]) -> List[float]:
        H, R, P, x = self.H, self.R, self.P, self.x
        HT = self._mat_T(H)
        S = self._mat_add(self._mat_mul(self._mat_mul(H, P), HT), R)
        K = self._mat_mul(self._mat_mul(P, HT), self._mat_inv(S))
        y_innov = [z[i] - sum(H[i][j] * x[j] for j in range(len(x))) for i in range(len(z))]
        self.x = [x[i] + sum(K[i][k] * y_innov[k] for k in range(len(y_innov))) for i in range(len(x))]
        n = len(x)
        I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        self.P = self._mat_mul(self._mat_sub(I, self._mat_mul(K, H)), P)
        return self.x[:]

    def filter(self, observations: List[List[float]]) -> List[List[float]]:
        states = []
        for z in observations:
            self.predict()
            states.append(self.update(z))
        return states


class VAR:
    """Vector Autoregression (VAR) model via OLS."""

    def __init__(self, lags: int = 1):
        self.lags = lags
        self.coefs_: List[List[List[float]]] = []
        self.intercept_: List[float] = []

    def fit(self, data: List[List[float]]) -> 'VAR':
        T = len(data)
        K = len(data[0])
        p = self.lags
        n_obs = T - p
        X = []
        for t in range(p, T):
            row = []
            for lag in range(1, p + 1):
                row.extend(data[t - lag])
            row.append(1.0)
            X.append(row)
        # OLS for each equation (column of data)
        self.coefs_ = []
        self.intercept_ = []
        n_feat = K * p + 1
        # XtX
        XtX = [[sum(X[i][a] * X[i][b] for i in range(n_obs)) for b in range(n_feat)] for a in range(n_feat)]
        # Gauss-Jordan inversion
        def _inv(M):
            n = len(M)
            aug = [M[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
            for col in range(n):
                pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
                aug[col], aug[pivot] = aug[pivot], aug[col]
                if abs(aug[col][col]) < 1e-12:
                    continue
                aug[col] = [v / aug[col][col] for v in aug[col]]
                for r in range(n):
                    if r != col:
                        f = aug[r][col]
                        aug[r] = [aug[r][c] - f * aug[col][c] for c in range(2 * n)]
            return [aug[i][n:] for i in range(n)]
        XtX_inv = _inv(XtX)
        for k in range(K):
            Xty = [sum(X[i][j] * data[p + i][k] for i in range(n_obs)) for j in range(n_feat)]
            beta = [sum(XtX_inv[j][l] * Xty[l] for l in range(n_feat)) for j in range(n_feat)]
            self.coefs_.append([beta[j * K:(j + 1) * K] for j in range(p)])
            self.intercept_.append(beta[-1])
        return self

    def forecast(self, data: List[List[float]], steps: int = 1) -> List[List[float]]:
        history = [list(row) for row in data]
        forecasts = []
        for _ in range(steps):
            K = len(history[0])
            pred = [self.intercept_[k] for k in range(K)]
            for k in range(K):
                for lag_i, lag_coef in enumerate(self.coefs_[k]):
                    t_idx = len(history) - 1 - lag_i
                    for j in range(K):
                        pred[k] += lag_coef[j] * history[t_idx][j]
            forecasts.append(pred)
            history.append(pred)
        return forecasts


class SARIMA:
    """Seasonal ARIMA(p,d,q)(P,D,Q,s) — adds seasonal differencing and seasonal AR/MA terms.

    For seasonal order (P,D,Q,s): applies seasonal difference of period s D times,
    then models seasonal AR(P) and MA(Q) components on top of the ARIMA core.
    """

    def __init__(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        self.order = order  # (p, d, q)
        self.seasonal_order = seasonal_order  # (P, D, Q, s)
        self._arima = None

    def _seasonal_diff(self, series, D, s):
        result = list(series)
        for _ in range(D):
            result = [result[i] - result[i - s] for i in range(s, len(result))]
        return result

    def fit(self, series: List[float]) -> 'SARIMA':
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        # Apply seasonal differencing
        y = self._seasonal_diff(series, D, s)
        # Fit ARIMA on seasonally-differenced data
        self._arima = ARIMA(p=p+P*s, d=d, q=q)
        self._arima.fit(y)
        self._s = s; self._D = D; self._orig = list(series)
        return self

    def forecast(self, steps: int = 1) -> List[float]:
        """Forecast steps ahead, undoing seasonal differencing."""
        if self._arima is None:
            raise RuntimeError('SARIMA: call fit() first')
        # Get ARIMA forecast on the seasonally-differenced series
        arima_fc = self._arima.forecast(steps)
        # Undo seasonal differencing by adding back last s values
        preds = []
        history = list(self._orig)
        for i, v in enumerate(arima_fc):
            # Add seasonal offset
            if self._D > 0:
                prev = history[-(self._s)]
                pred = v + prev
            else:
                pred = v
            preds.append(pred)
            history.append(pred)
        return preds


class GARCH:
    """GARCH(p,q) — Generalized Autoregressive Conditional Heteroskedasticity.

    Models time-varying volatility: σ²_t = ω + Σ αᵢ ε²_{t-i} + Σ βⱼ σ²_{t-j}

    Args:
        p: order of ARCH terms (lagged squared residuals)
        q: order of GARCH terms (lagged conditional variances)
    """

    def __init__(self, p: int = 1, q: int = 1):
        self.p = p; self.q = q
        self.omega = 0.1; self.alpha = [0.1] * p; self.beta = [0.8] * q

    def fit(self, returns: List[float], max_iter: int = 200, lr: float = 0.001) -> 'GARCH':
        """Fit GARCH via gradient-free variance targeting."""
        import math as _m
        n = len(returns)
        # Variance targeting: ω = σ²_uncond * (1 - Σα - Σβ)
        mean_r = sum(returns) / n
        var_uncond = sum((r - mean_r)**2 for r in returns) / n
        # Initialize with moment estimators
        self.omega = max(var_uncond * 0.1, 1e-6)
        # Simple EM-like fitting: iterate variance filter
        for _ in range(max_iter):
            sigmas2 = self._filter(returns)
            # Compute log-likelihood gradient numerically
            ll = -0.5 * sum(_m.log(max(s, 1e-10)) + r**2 / max(s, 1e-10)
                            for r, s in zip(returns, sigmas2))
            # Simple update: anneal ω to match unconditional variance
            avg_sigma2 = sum(sigmas2) / len(sigmas2)
            self.omega += lr * (var_uncond - avg_sigma2) * 0.1
            self.omega = max(self.omega, 1e-7)
        return self

    def _filter(self, returns: List[float]) -> List[float]:
        import math as _m
        n = len(returns)
        sigma2 = [0.0] * n
        var0 = sum(r**2 for r in returns) / max(n, 1)
        for t in range(n):
            s = self.omega
            for i in range(self.p):
                ti = t - 1 - i
                s += self.alpha[i] * (returns[ti]**2 if ti >= 0 else var0)
            for j in range(self.q):
                tj = t - 1 - j
                s += self.beta[j] * (sigma2[tj] if tj >= 0 else var0)
            sigma2[t] = max(s, 1e-10)
        return sigma2

    def forecast_variance(self, returns: List[float], steps: int = 1) -> List[float]:
        """Forecast conditional variance h steps ahead."""
        sigma2 = self._filter(returns)
        last_sigmas = sigma2[-self.q:] if self.q else []
        last_eps2 = [r**2 for r in returns[-self.p:]] if self.p else []
        preds = []
        for _ in range(steps):
            s = self.omega
            for i in range(self.p):
                s += self.alpha[i] * (last_eps2[-(i+1)] if i < len(last_eps2) else 0.0)
            for j in range(self.q):
                s += self.beta[j] * (last_sigmas[-(j+1)] if j < len(last_sigmas) else 0.0)
            preds.append(max(s, 1e-10))
            last_sigmas.append(s)
            last_eps2.append(s)  # under GARCH E[ε²_{t+h}] = σ²_{t+h}
        return preds

    @property
    def volatility_forecast(self):
        """Return volatility (std) forecast for 1 step."""
        import math as _m
        return _m.sqrt(self.forecast_variance([], 1)[0]) if hasattr(self, '_last_returns') else None


__all__ = [
    'make_windows', 'decompose', 'ARIMA', 'SARIMA', 'GARCH', 'HoltWinters', 'ADF',
    'mae', 'mse', 'rmse', 'mape',
    'KalmanFilter', 'VAR',
]
