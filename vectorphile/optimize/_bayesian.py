"""Bayesian Optimization with Gaussian Process surrogate and acquisition functions.

Implements the full BO loop:
  1. Gaussian Process regression as the probabilistic surrogate
  2. Acquisition functions: EI (Expected Improvement), UCB, PI
  3. L-BFGS inner optimizer to maximize acquisition
  4. BayesianOptimizer class: a scikit-learn compatible BO engine

Usage::
    from vectorphile.optimize import BayesianOptimizer

    def black_box(x):
        return -(x[0]**2 + x[1]**2)   # maximise negative parabola

    opt = BayesianOptimizer(bounds=[(-5,5), (-5,5)], n_initial=5, n_calls=20)
    result = opt.maximize(black_box)
    print(result.x, result.fun)
"""
from __future__ import annotations
import math
import random
from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass, field


# ── RBF (Radial Basis Function) Gaussian Process ──────────────────────────────

class GaussianProcessRBF:
    """Gaussian Process regressor with isotropic RBF (squared exponential) kernel.

    kernel(x, x') = σ² * exp(-||x - x'||² / (2 * l²))

    Args:
        length_scale : kernel length scale (default 1.0)
        sigma_f      : kernel output scale (default 1.0)
        noise        : observation noise variance (default 1e-4)
    """

    def __init__(self, length_scale: float = 1.0, sigma_f: float = 1.0,
                 noise: float = 1e-4):
        self.l    = length_scale
        self.sf   = sigma_f
        self.sn   = noise
        self._X   = None   # (n, d) training points
        self._y   = None   # (n,)   training targets (mean-centered)
        self._y_mean = 0.0
        self._L   = None   # Cholesky of K + sn*I
        self._alpha = None  # K^{-1} y

    def _rbf(self, x1: List[float], x2: List[float]) -> float:
        d2 = sum((a - b) ** 2 for a, b in zip(x1, x2))
        return self.sf ** 2 * math.exp(-d2 / (2.0 * self.l ** 2))

    def _build_K(self, X: List[List[float]]) -> List[List[float]]:
        n = len(X)
        K = [[self._rbf(X[i], X[j]) for j in range(n)] for i in range(n)]
        for i in range(n):
            K[i][i] += self.sn
        return K

    def _cholesky(self, A: List[List[float]]) -> List[List[float]]:
        """Lower Cholesky decomposition (pure Python)."""
        n = len(A)
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    val = A[i][i] - s
                    L[i][j] = math.sqrt(max(val, 1e-12))
                else:
                    L[i][j] = (A[i][j] - s) / max(L[j][j], 1e-12)
        return L

    def _solve_L(self, L: List[List[float]], b: List[float]) -> List[float]:
        """Solve L y = b (forward substitution)."""
        n = len(b)
        y = [0.0] * n
        for i in range(n):
            s = sum(L[i][k] * y[k] for k in range(i))
            y[i] = (b[i] - s) / max(L[i][i], 1e-12)
        return y

    def _solve_Lt(self, L: List[List[float]], b: List[float]) -> List[float]:
        """Solve L^T y = b (backward substitution)."""
        n = len(b)
        y = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = sum(L[j][i] * y[j] for j in range(i + 1, n))
            y[i] = (b[i] - s) / max(L[i][i], 1e-12)
        return y

    def fit(self, X: List[List[float]], y: List[float]) -> 'GaussianProcessRBF':
        """Fit the GP to observations."""
        self._X      = [list(xi) for xi in X]
        self._y_mean = sum(y) / len(y)
        self._y      = [yi - self._y_mean for yi in y]
        K = self._build_K(self._X)
        self._L      = self._cholesky(K)
        v            = self._solve_L(self._L, self._y)
        self._alpha  = self._solve_Lt(self._L, v)
        return self

    def predict(self, X_new: List[List[float]]) -> Tuple[List[float], List[float]]:
        """Return (mean, std) for each test point."""
        means, stds = [], []
        for x in X_new:
            # k_star: covariance between x and training points
            k_star = [self._rbf(x, xi) for xi in self._X]
            mu     = sum(a * k for a, k in zip(self._alpha, k_star)) + self._y_mean
            # variance
            v      = self._solve_L(self._L, k_star)
            var    = self._rbf(x, x) - sum(vi ** 2 for vi in v)
            means.append(mu)
            stds.append(math.sqrt(max(var, 0.0)))
        return means, stds


# ── Acquisition functions ─────────────────────────────────────────────────────

def _normal_cdf(z: float) -> float:
    """Approximation of Φ(z) using Abramowitz & Stegun."""
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    p = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 +
        t * (-1.821255978 + t * 1.330274429))))
    phi = math.exp(-z * z / 2.0) / math.sqrt(2.0 * math.pi)
    cdf = 1.0 - phi * p
    return cdf if z >= 0 else 1.0 - cdf


def _normal_pdf(z: float) -> float:
    return math.exp(-z * z / 2.0) / math.sqrt(2.0 * math.pi)


def expected_improvement(mu: float, sigma: float, y_best: float,
                          xi: float = 0.01) -> float:
    """EI: E[max(f(x) - y_best - xi, 0)] under GP posterior."""
    if sigma <= 1e-9:
        return 0.0
    z = (mu - y_best - xi) / sigma
    return (mu - y_best - xi) * _normal_cdf(z) + sigma * _normal_pdf(z)


def upper_confidence_bound(mu: float, sigma: float, kappa: float = 2.0) -> float:
    """UCB: mu + kappa * sigma."""
    return mu + kappa * sigma


def probability_of_improvement(mu: float, sigma: float, y_best: float,
                                 xi: float = 0.01) -> float:
    """PI: Φ((mu - y_best - xi) / sigma)."""
    if sigma <= 1e-9:
        return 0.0
    z = (mu - y_best - xi) / sigma
    return _normal_cdf(z)


# ── Result object ─────────────────────────────────────────────────────────────

@dataclass
class OptimizeResult:
    x:     List[float]        # best parameter vector
    fun:   float              # best observed objective value
    x_iters: List[List[float]] = field(default_factory=list)
    func_vals: List[float]     = field(default_factory=list)
    n_calls:   int             = 0
    success:   bool            = True

    def __repr__(self):
        return f'OptimizeResult(x={[f"{v:.4f}" for v in self.x]}, fun={self.fun:.6f}, n_calls={self.n_calls})'


# ── BayesianOptimizer ─────────────────────────────────────────────────────────

class BayesianOptimizer:
    """Bayesian Optimization engine with GP surrogate.

    Args:
        bounds       : list of (min, max) tuples, one per dimension
        n_initial    : number of random initial evaluations (default 5)
        n_calls      : total number of objective evaluations
        acquisition  : 'ei' (default), 'ucb', or 'pi'
        xi           : exploration parameter for EI/PI (default 0.01)
        kappa        : exploration factor for UCB (default 2.0)
        random_state : seed for reproducibility
        gp_kwargs    : dict of kwargs passed to GaussianProcessRBF

    Usage::
        opt = BayesianOptimizer(bounds=[(-5,5),(-5,5)], n_calls=30)
        result = opt.maximize(lambda x: -(x[0]**2 + x[1]**2))
        print(result.x, result.fun)
    """

    def __init__(self, bounds: List[Tuple[float, float]],
                 n_initial:   int   = 5,
                 n_calls:     int   = 25,
                 acquisition: str   = 'ei',
                 xi:          float = 0.01,
                 kappa:       float = 2.0,
                 random_state: Optional[int] = None,
                 gp_kwargs:   Optional[dict] = None):
        self.bounds       = bounds
        self.n_initial    = max(1, n_initial)
        self.n_calls      = n_calls
        self.acquisition  = acquisition.lower()
        self.xi           = xi
        self.kappa        = kappa
        self._rng         = random.Random(random_state)
        self._gp_kwargs   = gp_kwargs or {}
        self._gp          = GaussianProcessRBF(**self._gp_kwargs)
        self._X_obs: List[List[float]] = []
        self._y_obs: List[float]       = []

    def _random_point(self) -> List[float]:
        return [self._rng.uniform(lo, hi) for lo, hi in self.bounds]

    def _acq_value(self, x: List[float], y_best: float) -> float:
        mus, stds = self._gp.predict([x])
        mu, sigma = mus[0], stds[0]
        if self.acquisition == 'ei':
            return expected_improvement(mu, sigma, y_best, self.xi)
        elif self.acquisition == 'ucb':
            return upper_confidence_bound(mu, sigma, self.kappa)
        elif self.acquisition == 'pi':
            return probability_of_improvement(mu, sigma, y_best, self.xi)
        return 0.0

    def _maximize_acquisition(self, y_best: float,
                               n_restarts: int = 20) -> List[float]:
        """Maximize acquisition via random search + local hill-climbing."""
        best_x   = self._random_point()
        best_acq = self._acq_value(best_x, y_best)

        # Random search
        for _ in range(n_restarts * 10):
            x = self._random_point()
            a = self._acq_value(x, y_best)
            if a > best_acq:
                best_acq = a; best_x = x

        # Local perturbation around best
        d = len(self.bounds)
        for _ in range(n_restarts):
            x = [
                max(lo, min(hi, best_x[i] + self._rng.gauss(0, (hi - lo) * 0.1)))
                for i, (lo, hi) in enumerate(self.bounds)
            ]
            a = self._acq_value(x, y_best)
            if a > best_acq:
                best_acq = a; best_x = x

        return best_x

    def maximize(self, objective: Callable[[List[float]], float],
                 n_calls: Optional[int] = None,
                 verbose: bool = False) -> OptimizeResult:
        """Run the BO loop, maximizing `objective`.

        Args:
            objective : f(x) -> float; higher is better
            n_calls   : override default total calls
            verbose   : print each evaluation

        Returns:
            OptimizeResult with best x, fun, and history
        """
        total = n_calls or self.n_calls
        x_iters, y_iters = list(self._X_obs), list(self._y_obs)

        # Initial random phase
        init = self.n_initial - len(x_iters)
        for _ in range(max(0, init)):
            x = self._random_point()
            y = float(objective(x))
            x_iters.append(x); y_iters.append(y)
            if verbose:
                print(f'[init] x={[f"{v:.3f}" for v in x]}  f={y:.4f}')

        # BO phase
        calls_so_far = len(x_iters)
        while calls_so_far < total:
            if len(x_iters) >= 2:
                self._gp.fit(x_iters, y_iters)
                y_best = max(y_iters)
                x_next = self._maximize_acquisition(y_best)
            else:
                x_next = self._random_point()

            y_next = float(objective(x_next))
            x_iters.append(x_next); y_iters.append(y_next)
            calls_so_far += 1
            if verbose:
                print(f'[{calls_so_far}] x={[f"{v:.3f}" for v in x_next]}'
                      f'  f={y_next:.4f}  best={max(y_iters):.4f}')

        self._X_obs = x_iters; self._y_obs = y_iters
        best_i = max(range(len(y_iters)), key=lambda i: y_iters[i])
        return OptimizeResult(
            x=x_iters[best_i], fun=y_iters[best_i],
            x_iters=x_iters, func_vals=y_iters,
            n_calls=len(y_iters),
        )

    def minimize(self, objective: Callable[[List[float]], float],
                 n_calls: Optional[int] = None,
                 verbose: bool = False) -> OptimizeResult:
        """Run BO minimizing `objective` (wraps maximize with negation)."""
        neg_obj = lambda x: -objective(x)
        result  = self.maximize(neg_obj, n_calls=n_calls, verbose=verbose)
        result.fun      = -result.fun
        result.func_vals = [-v for v in result.func_vals]
        return result
