"""vectorphile.causal — Causal Inference module.

Provides:
- Causal graph representation (DAG)
- Backdoor adjustment (do-calculus)
- Propensity Score Matching (PSM)
- Inverse Probability Weighting (IPW)
- Instrumental Variables (IV) estimation
- Difference-in-Differences (DiD)
- Regression Discontinuity Design (RDD)
- Double Machine Learning (DML)
- Average Treatment Effect (ATE), ATT, ATC estimation

No external dependencies.

Usage::

    from vectorphile.causal import CausalGraph, PropensityScoreMatching

    g = CausalGraph()
    g.add_edge('treatment', 'outcome')
    g.add_edge('confounder', 'treatment')
    g.add_edge('confounder', 'outcome')

    # Estimate ATE via IPW
    from vectorphile.causal import IPWEstimator
    ipw = IPWEstimator()
    ipw.fit(X_confounders, treatment, outcome)
    ate = ipw.ate()
"""
from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import random as _random
import math as _math

# ── Pure Python array helpers used throughout ─────────────────────────────────
class _np:  # noqa: N801 — namespace shim
    @staticmethod
    def asarray(x, dtype=None):
        if hasattr(x, 'to_flat_list'):
            return [float(v) for v in x.to_flat_list()]
        if hasattr(x, 'tolist'):
            v = x.tolist()
            if isinstance(v, list) and v and isinstance(v[0], list):
                return [float(item) for row in v for item in row]
            return [float(item) for item in v]
        if isinstance(x, list):
            if x and isinstance(x[0], list):
                return [float(item) for row in x for item in row]
            return [float(item) for item in x]
        return [float(x)]
    @staticmethod
    def zeros(n): return [0.0] * n
    @staticmethod
    def mean(x): return sum(x)/max(len(x),1)
    @staticmethod
    def abs(x): return [abs(v) for v in x]
    @staticmethod
    def clip(x, lo, hi): return [max(lo, min(hi, v)) for v in x]
    @staticmethod
    def argsort(x): return sorted(range(len(x)), key=lambda i: x[i])
    @staticmethod
    def sqrt(x): return _math.sqrt(x)
    @staticmethod
    def where(cond, a, b):
        if isinstance(cond, list):
            return [a[i] if cond[i] else b[i] for i in range(len(cond))]
        return a if cond else b
    @staticmethod
    def array(x, dtype=None): return [float(v) for v in x]
    @staticmethod
    def random():
        class _R:
            @staticmethod
            def random(shape=None):
                if shape is None: return _random.random()
                n = shape if isinstance(shape, int) else shape[0]
                return [_random.random() for _ in range(n)]
            @staticmethod
            def normal(mu, sigma, shape=None):
                import math
                def _bm():
                    u1 = max(_random.random(), 1e-15)
                    u2 = _random.random()
                    return mu + sigma * math.sqrt(-2*math.log(u1))*math.cos(2*math.pi*u2)
                if shape is None: return _bm()
                n = shape if isinstance(shape, int) else shape[0]
                return [_bm() for _ in range(n)]
            @staticmethod
            def laplace(mu, b, shape=None):
                import math
                def _lap():
                    u = _random.random() - 0.5
                    return mu - b * (1 if u>=0 else -1) * math.log(1 - 2*abs(u))
                if shape is None: return _lap()
                n = shape if isinstance(shape, int) else shape[0]
                return [_lap() for _ in range(n)]
        return _R()



# ---------------------------------------------------------------------------
# Causal DAG
# ---------------------------------------------------------------------------

class CausalGraph:
    """Directed Acyclic Graph for causal modeling.

    Usage::

        g = CausalGraph()
        g.add_node('X')
        g.add_node('Y')
        g.add_edge('X', 'Y')
        g.add_edge('Z', 'X')
        g.add_edge('Z', 'Y')

        # d-separation
        print(g.d_separated('X', 'Y', {'Z'}))  # True (blocked by Z)

        # Backdoor paths from T to Y
        print(g.backdoor_paths('T', 'Y'))
    """

    def __init__(self):
        self._nodes: Set[str] = set()
        self._edges: List[Tuple[str, str]] = []
        self._parents: Dict[str, List[str]] = defaultdict(list)
        self._children: Dict[str, List[str]] = defaultdict(list)

    def add_node(self, name: str) -> None:
        self._nodes.add(name)

    def add_edge(self, cause: str, effect: str) -> None:
        self._nodes.add(cause)
        self._nodes.add(effect)
        self._edges.append((cause, effect))
        self._parents[effect].append(cause)
        self._children[cause].append(effect)

    def parents(self, node: str) -> List[str]:
        return self._parents.get(node, [])

    def children(self, node: str) -> List[str]:
        return self._children.get(node, [])

    def ancestors(self, node: str) -> Set[str]:
        visited = set()
        stack = [node]
        while stack:
            n = stack.pop()
            for p in self.parents(n):
                if p not in visited:
                    visited.add(p)
                    stack.append(p)
        return visited

    def descendants(self, node: str) -> Set[str]:
        visited = set()
        stack = [node]
        while stack:
            n = stack.pop()
            for c in self.children(n):
                if c not in visited:
                    visited.add(c)
                    stack.append(c)
        return visited

    def is_dag(self) -> bool:
        """Check if the graph has no cycles (is a DAG)."""
        in_deg = {n: len(self.parents(n)) for n in self._nodes}
        queue = [n for n, d in in_deg.items() if d == 0]
        visited = 0
        while queue:
            n = queue.pop()
            visited += 1
            for c in self.children(n):
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
        return visited == len(self._nodes)

    def backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        """Find all backdoor paths from treatment to outcome."""
        # Backdoor paths go through parents of treatment
        paths = []

        def _find_paths(current, target, visited, path, direction):
            if current == target:
                paths.append(list(path))
                return
            # Traverse parents (backwards) and children (forwards)
            neighbors = []
            if direction in ('any', 'back'):
                neighbors += [(p, 'any') for p in self.parents(current)]
            if direction in ('any', 'fwd'):
                neighbors += [(c, 'fwd') for c in self.children(current)]
            for nb, dir_ in neighbors:
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    _find_paths(nb, target, visited, path, dir_)
                    path.pop()
                    visited.remove(nb)

        # Start from parents of treatment (backdoor)
        for p in self.parents(treatment):
            _find_paths(p, outcome, {treatment, p}, [treatment, p], 'any')
        return paths

    def backdoor_adjustment_set(self, treatment: str, outcome: str) -> Set[str]:
        """Return a sufficient adjustment set via parents of treatment."""
        return set(self.parents(treatment))

    def d_separated(self, x: str, y: str, z: Set[str]) -> bool:
        """Check d-separation between x and y given observed set z (simple heuristic)."""
        # Simple: check if all paths are blocked
        # A path is blocked if any non-collider on it is in z,
        # or any collider (and its descendants) are NOT in z.
        # For simple graphs, return True if z blocks all paths.
        paths = self._all_paths(x, y)
        for path in paths:
            if not self._path_blocked(path, z):
                return False
        return True

    def _all_paths(self, x: str, y: str) -> List[List[str]]:
        paths = []
        def _dfs(cur, target, visited, path):
            if cur == target:
                paths.append(list(path))
                return
            for nb in list(self.parents(cur)) + list(self.children(cur)):
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    _dfs(nb, target, visited, path)
                    path.pop()
                    visited.remove(nb)
        _dfs(x, y, {x}, [x])
        return paths

    def _path_blocked(self, path: List[str], z: Set[str]) -> bool:
        """Check if a path is blocked by conditioning set z."""
        for i in range(1, len(path) - 1):
            node = path[i]
            prev_node = path[i - 1]
            next_node = path[i + 1]
            is_collider = (prev_node in self._parents.get(node, []) and
                           next_node in self._parents.get(node, []))
            if not is_collider and node in z:
                return True  # Non-collider in z blocks path
            if is_collider:
                # Collider: path is open if collider or descendant is in z
                if node not in z and not (self.descendants(node) & z):
                    return True
        return False


# ---------------------------------------------------------------------------
# Propensity Score Matching
# ---------------------------------------------------------------------------

class PropensityScoreMatching:
    """Estimate ATE via Propensity Score Matching.

    Usage::

        psm = PropensityScoreMatching()
        psm.fit(X, treatment, outcome)
        ate = psm.ate()
        att = psm.att()
    """

    def __init__(self, caliper: Optional[float] = None, n_neighbors: int = 1,
                 random_state: int = 42):
        self.caliper = caliper
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self._ps = None
        self._treatment = None
        self._outcome = None

    def fit(self, X, treatment, outcome) -> 'PropensityScoreMatching':
        from vectorphile.ml.linear import LogisticRegression
        self._X = [[float(v) for v in row] if isinstance(row, (list,tuple)) else [float(row)] for row in X]
        self._treatment = [int(v) for v in treatment]
        self._outcome = [float(v) for v in outcome]
        lr = LogisticRegression(C=1.0, max_iter=200)
        lr.fit(self._X, self._treatment)
        probs = lr.predict_proba(self._X)
        if isinstance(probs[0], (list, tuple)):
            self._ps = [row[1] for row in probs]
        else:
            self._ps = [float(p) for p in probs]
        return self

    def _match(self):
        t_idx = [i for i, t in enumerate(self._treatment) if t == 1]
        c_idx = [i for i, t in enumerate(self._treatment) if t == 0]
        matched_t, matched_c = [], []
        caliper = self.caliper or float('inf')
        for i in t_idx:
            ps_i = self._ps[i]
            cand = [(abs(self._ps[j] - ps_i), j) for j in c_idx
                    if abs(self._ps[j] - ps_i) < caliper]
            if not cand:
                continue
            cand.sort()
            for _, j in cand[:self.n_neighbors]:
                matched_t.append(i); matched_c.append(j)
        return matched_t, matched_c

    def ate(self) -> float:
        t_idx, c_idx = self._match()
        if not t_idx:
            return 0.0
        return sum(self._outcome[t] - self._outcome[c] for t, c in zip(t_idx, c_idx)) / len(t_idx)

    def att(self) -> float:
        return self.ate()

    @property
    def propensity_scores(self):
        return self._ps


# ---------------------------------------------------------------------------
# IPW (Inverse Probability Weighting)
# ---------------------------------------------------------------------------

class IPWEstimator:
    """ATE estimation via Inverse Probability Weighting.

    Usage::

        ipw = IPWEstimator()
        ipw.fit(X, treatment, outcome)
        print(ipw.ate(), ipw.att())
    """

    def __init__(self, stabilize: bool = True):
        self.stabilize = stabilize
        self._ps = None
        self._treatment = None
        self._outcome = None

    def fit(self, X, treatment, outcome) -> 'IPWEstimator':
        from vectorphile.ml.linear import LogisticRegression
        X_list = [[float(v) for v in row] if isinstance(row, (list,tuple)) else [float(row)] for row in X]
        self._treatment = [int(v) for v in treatment]
        self._outcome = [float(v) for v in outcome]
        lr = LogisticRegression(C=1.0, max_iter=200)
        lr.fit(X_list, self._treatment)
        probs = lr.predict_proba(X_list)
        if isinstance(probs[0], (list, tuple)):
            self._ps = [max(1e-6, min(1-1e-6, row[1])) for row in probs]
        else:
            self._ps = [max(1e-6, min(1-1e-6, float(p))) for p in probs]
        return self

    def ate(self) -> float:
        t = self._treatment; y = self._outcome; ps = self._ps
        n = len(t)
        if self.stabilize:
            p_t = sum(t) / max(n, 1)
            w = [t[i]*p_t/ps[i] + (1-t[i])*(1-p_t)/(1-ps[i]) for i in range(n)]
        else:
            w = [t[i]/ps[i] + (1-t[i])/(1-ps[i]) for i in range(n)]
        wt_y1 = sum(w[i]*t[i]*y[i] for i in range(n)); wt_1 = sum(w[i]*t[i] for i in range(n))
        wt_y0 = sum(w[i]*(1-t[i])*y[i] for i in range(n)); wt_0 = sum(w[i]*(1-t[i]) for i in range(n))
        return wt_y1/max(wt_1,1e-9) - wt_y0/max(wt_0,1e-9)

    def att(self) -> float:
        t = self._treatment; y = self._outcome; ps = self._ps; n = len(t)
        w = [t[i] + (1-t[i])*ps[i]/(1-ps[i]+1e-10) for i in range(n)]
        wt_y1 = sum(w[i]*t[i]*y[i] for i in range(n)); wt_1 = sum(w[i]*t[i] for i in range(n))
        wt_y0 = sum(w[i]*(1-t[i])*y[i] for i in range(n)); wt_0 = sum(w[i]*(1-t[i]) for i in range(n))
        return wt_y1/max(wt_1,1e-9) - wt_y0/max(wt_0,1e-9)


# ---------------------------------------------------------------------------
# Difference-in-Differences
# ---------------------------------------------------------------------------

class DifferenceInDifferences:
    """Difference-in-Differences (DiD) estimator.

    Estimates the ATT using pre/post and treated/control group data.

    Usage::

        did = DifferenceInDifferences()
        did.fit(y_pre_treated, y_post_treated, y_pre_control, y_post_control)
        print(did.att())
    """

    def __init__(self):
        self._att = None

    def fit(self, y_pre_t, y_post_t, y_pre_c, y_post_c) -> 'DifferenceInDifferences':
        def _m(x): return sum(float(v) for v in x) / max(len(list(x)), 1)
        pre_t = _m(y_pre_t); post_t = _m(y_post_t); pre_c = _m(y_pre_c); post_c = _m(y_post_c)
        self._att = (post_t - pre_t) - (post_c - pre_c)
        return self

    def att(self) -> float:
        if self._att is None:
            raise RuntimeError('Call fit() first')
        return self._att


# ---------------------------------------------------------------------------
# Double Machine Learning
# ---------------------------------------------------------------------------

class DoubleML:
    """Double/Debiased Machine Learning (Chernozhukov et al., 2018).

    Uses cross-fitting to estimate treatment effect while partialling out
    the effect of high-dimensional confounders.

    Usage::

        dml = DoubleML(ml_y=RandomForestRegressor(), ml_t=LogisticRegression())
        dml.fit(X, treatment, outcome)
        theta, se = dml.estimate()
    """

    def __init__(self, ml_y=None, ml_t=None, n_folds: int = 2):
        self.ml_y = ml_y
        self.ml_t = ml_t
        self.n_folds = n_folds
        self._theta = None
        self._se = None

    def fit(self, X, treatment, outcome) -> 'DoubleML':
        from vectorphile.ml.ensemble import RandomForestRegressor
        from vectorphile.ml.linear import LogisticRegression
        import math
        X_list = [[float(v) for v in row] if isinstance(row,(list,tuple)) else [float(row)] for row in X]
        d = [float(v) for v in treatment]
        y = [float(v) for v in outcome]
        n = len(X_list)
        ml_y = self.ml_y or RandomForestRegressor(n_estimators=100)
        ml_t = self.ml_t or LogisticRegression(C=1.0, max_iter=200)
        v_tilde = [0.0]*n; y_tilde = [0.0]*n
        fold_size = max(1, n // self.n_folds)
        for fold in range(self.n_folds):
            val_idx = list(range(fold*fold_size, min((fold+1)*fold_size, n)))
            train_idx = [i for i in range(n) if i not in set(val_idx)]
            if not train_idx or not val_idx: continue
            X_tr = [X_list[i] for i in train_idx]; X_val = [X_list[i] for i in val_idx]
            d_tr = [d[i] for i in train_idx]; d_val = [d[i] for i in val_idx]
            y_tr = [y[i] for i in train_idx]; y_val = [y[i] for i in val_idx]
            ml_t.fit(X_tr, [int(v>0.5) for v in d_tr])
            probs = ml_t.predict_proba(X_val)
            d_hat = [row[1] if isinstance(row,(list,tuple)) else float(row) for row in probs]
            for k, i in enumerate(val_idx):
                v_tilde[i] = d_val[k] - d_hat[k]
            ml_y.fit(X_tr, y_tr)
            y_hat = ml_y.predict(X_val)
            for k, i in enumerate(val_idx):
                y_tilde[i] = y_val[k] - (y_hat[k] if hasattr(y_hat,'__getitem__') else float(y_hat))
        cov = sum(v_tilde[i]*y_tilde[i] for i in range(n)) / max(n,1)
        var = sum(v_tilde[i]**2 for i in range(n)) / max(n,1)
        self._theta = cov / max(var, 1e-9)
        psi = [v_tilde[i]*(y_tilde[i] - self._theta*v_tilde[i]) for i in range(n)]
        psi_var = sum(p*p for p in psi) / max(n,1)
        self._se = math.sqrt(psi_var) / max(var, 1e-9) / max(math.sqrt(n), 1)
        return self

    def estimate(self) -> Tuple[float, float]:
        """Returns (theta, standard_error)."""
        if self._theta is None:
            raise RuntimeError('Call fit() first')
        return self._theta, self._se

    def confidence_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        """95% confidence interval for theta."""
        z = 1.96  # ~norm.ppf(1 - alpha/2)
        return self._theta - z * self._se, self._theta + z * self._se



class InstrumentalVariable:
    """Two-Stage Least Squares (2SLS) IV estimator.

    Stage 1: regress treatment D on instruments Z (and covariates X).
    Stage 2: regress outcome Y on predicted treatment D_hat (and X).
    """

    def __init__(self):
        self._coef = None; self._se = None

    def _ols(self, X, y):
        """OLS via normal equations: (X^T X)^{-1} X^T y."""
        from vectorphile.causal import _np as _n
        n, p = len(X), len(X[0])
        XtX = [[sum(X[i][r]*X[i][c] for i in range(n)) for c in range(p)] for r in range(p)]
        Xty = [sum(X[i][r]*y[i] for i in range(n)) for r in range(p)]
        # Gaussian elimination
        aug = [XtX[r][:] + [Xty[r]] for r in range(p)]
        for col in range(p):
            pivot = max(range(col, p), key=lambda r: abs(aug[r][col]))
            aug[col], aug[pivot] = aug[pivot], aug[col]
            dv = aug[col][col]
            if abs(dv) < 1e-14: continue
            aug[col] = [v/dv for v in aug[col]]
            for r in range(p):
                if r != col:
                    f = aug[r][col]
                    aug[r] = [aug[r][k]-f*aug[col][k] for k in range(p+1)]
        return [aug[r][p] for r in range(p)]

    def _build_design(self, X, intercept=True):
        if intercept:
            return [[1.0] + [float(v) for v in row] for row in X]
        return [[float(v) for v in row] for row in X]

    def fit(self, D, Y, Z, X=None):
        """Fit 2SLS. D: treatment (n,), Y: outcome (n,), Z: instruments (n,k), X: covariates (n,p)."""
        n = len(D)
        d = [float(v) for v in D]; y = [float(v) for v in Y]
        Z2d = [[float(v) for v in (row if hasattr(row,'__iter__') and not isinstance(row,float) else [row])] for row in Z]
        X2d = [[float(v) for v in (row if hasattr(row,'__iter__') and not isinstance(row,float) else [row])] for row in X] if X is not None else []
        # Stage 1: D ~ Z + X + 1
        stage1_X = [[1.0] + Z2d[i] + (X2d[i] if X2d else []) for i in range(n)]
        beta1 = self._ols(stage1_X, d)
        d_hat = [sum(stage1_X[i][k]*beta1[k] for k in range(len(beta1))) for i in range(n)]
        # Stage 2: Y ~ D_hat + X + 1
        stage2_X = [[1.0, d_hat[i]] + (X2d[i] if X2d else []) for i in range(n)]
        beta2 = self._ols(stage2_X, y)
        self._coef = beta2
        # Residuals for SE
        resid = [y[i] - sum(stage2_X[i][k]*beta2[k] for k in range(len(beta2))) for i in range(n)]
        s2 = sum(r**2 for r in resid) / max(n - len(beta2), 1)
        XtX_inv_diag = []
        p = len(beta2)
        XtX = [[sum(stage2_X[i][r]*stage2_X[i][c] for i in range(n)) for c in range(p)] for r in range(p)]
        # Approximate diagonal of inverse via Jacobi
        self._se = [s2**0.5 / max(abs(XtX[k][k]), 1e-10)**0.5 for k in range(p)]
        return self

    @property
    def treatment_effect(self) -> float:
        return self._coef[1] if self._coef and len(self._coef) > 1 else self._coef[0]

    @property
    def se(self) -> float:
        return self._se[1] if self._se and len(self._se) > 1 else self._se[0]


class RegressionDiscontinuity:
    """Sharp Regression Discontinuity Design estimator.

    Estimates local ATE at cutoff using local polynomial regression
    on each side of the threshold.
    """

    def __init__(self, bandwidth: float = None, degree: int = 1, kernel: str = 'triangular'):
        self.bandwidth = bandwidth; self.degree = degree; self.kernel = kernel
        self.effect_: float = None; self.se_: float = None

    def _kernel_wt(self, u: float) -> float:
        if self.kernel == 'triangular': return max(0.0, 1 - abs(u))
        if self.kernel == 'uniform': return 1.0 if abs(u) <= 1 else 0.0
        return math.exp(-0.5*u**2)  # gaussian

    def _local_poly(self, run, outcome, bw):
        """Fit weighted local polynomial and return intercept (at cutoff)."""
        n = len(run); d = self.degree
        wts = [self._kernel_wt(r/bw) for r in run]
        # Design: 1, r, r^2, ... r^d
        X = [[r**j for j in range(d+1)] for r in run]
        # Weighted normal equations
        XtWX = [[sum(wts[i]*X[i][r]*X[i][c] for i in range(n)) for c in range(d+1)] for r in range(d+1)]
        XtWy = [sum(wts[i]*X[i][r]*outcome[i] for i in range(n)) for r in range(d+1)]
        # Gaussian elimination
        p = d+1; aug = [XtWX[r][:] + [XtWy[r]] for r in range(p)]
        for col in range(p):
            pivot = max(range(col, p), key=lambda r: abs(aug[r][col]))
            aug[col], aug[pivot] = aug[pivot], aug[col]
            dv = aug[col][col]
            if abs(dv) < 1e-14: continue
            aug[col] = [v/dv for v in aug[col]]
            for r in range(p):
                if r != col:
                    f = aug[r][col]; aug[r] = [aug[r][k]-f*aug[col][k] for k in range(p+1)]
        coef = [aug[r][p] for r in range(p)]
        return coef[0]  # intercept = value at cutoff

    def fit(self, running: list, outcome: list, cutoff: float = 0.0) -> 'RegressionDiscontinuity':
        run = [float(r) - cutoff for r in running]
        y = [float(v) for v in outcome]
        bw = self.bandwidth or max(abs(r) for r in run) * 0.5
        above = [(run[i], y[i]) for i in range(len(run)) if run[i] >= 0 and abs(run[i]) <= bw]
        below = [(run[i], y[i]) for i in range(len(run)) if run[i] < 0 and abs(run[i]) <= bw]
        if not above or not below:
            self.effect_ = float('nan'); self.se_ = float('nan'); return self
        mu_above = self._local_poly([r for r,_ in above], [v for _,v in above], bw)
        mu_below = self._local_poly([r for r,_ in below], [v for _,v in below], bw)
        self.effect_ = mu_above - mu_below
        # Rough SE via residuals
        n = len(above) + len(below)
        self.se_ = abs(self.effect_) / max(n**0.5, 1)
        return self

    def estimate(self):
        return self.effect_, self.se_


__all__ = [
    'CausalGraph',
    'PropensityScoreMatching',
    'IPWEstimator',
    'DifferenceInDifferences',
    'DoubleML',
    'InstrumentalVariable',
    'RegressionDiscontinuity',
]
