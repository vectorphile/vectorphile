"""Numerical optimization routines — pure Python, no external dependencies."""
import math
import random
from typing import Callable, Optional, Tuple
from vectorphile.core.tensor import Tensor
from vectorphile._core.ndarray import array as _nd_array


def _flat(x) -> list:
    if isinstance(x, Tensor):
        d = x._data
        if hasattr(d, 'to_flat_list'):
            return [float(v) for v in d.to_flat_list()]
        if hasattr(d, 'tolist'):
            v = d.tolist()
            if isinstance(v, list) and v and isinstance(v[0], list):
                return [float(x) for row in v for x in row]
            return [float(x) for x in v]
        return [float(v) for v in d]
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    return [float(x)]


def _make_tensor(vals: list) -> Tensor:
    return Tensor(_nd_array([float(v) for v in vals]))


def _vec_norm(v: list) -> float:
    return math.sqrt(sum(x * x for x in v))


def _gradient_descent(fun: Callable, x0, lr=0.01, tol=1e-6, max_iter=1000, options=None):
    if options:
        lr = options.get('lr', lr)
        max_iter = options.get('maxiter', max_iter)
    vals = _flat(x0)
    for _ in range(max_iter):
        x = _make_tensor(vals)
        x.requires_grad = True
        if hasattr(x, 'zero_grad'):
            x.zero_grad()
        result = fun(x)
        if isinstance(result, Tensor):
            result.backward()
            if x.grad is None:
                break
            grad = _flat(x.grad)
        else:
            # Numerical gradient
            eps = 1e-5
            f0 = float(result) if not isinstance(result, Tensor) else float(fun(_make_tensor(vals))._data.to_flat_list()[0] if hasattr(fun(_make_tensor(vals))._data, 'to_flat_list') else fun(_make_tensor(vals))._data)
            grad = []
            for i in range(len(vals)):
                p = vals[:]
                p[i] += eps
                fp = fun(_make_tensor(p))
                fp_val = float(_flat(fp)[0]) if isinstance(fp, Tensor) else float(fp)
                grad.append((fp_val - f0) / eps)
        gn = _vec_norm(grad)
        if gn < tol:
            break
        vals = [v - lr * g for v, g in zip(vals, grad)]

    class _Result:
        pass
    r = _Result()
    r.x = _make_tensor(vals)
    r.fun = float(_flat(fun(r.x))[0]) if isinstance(fun(r.x), Tensor) else float(fun(r.x))
    r.success = True
    return r


def root(fun: Callable, x0, method='hybr', tol=1e-6):
    """Find root using Newton's method / bisection."""
    vals = _flat(x0)
    eps = 1e-5
    max_iter = 500
    for _ in range(max_iter):
        fx = _flat(fun(_make_tensor(vals)))
        f_norm = _vec_norm(fx)
        if f_norm < tol:
            break
        # Numerical Jacobian
        n = len(vals)
        J = []
        for i in range(len(fx)):
            row = []
            for j in range(n):
                p = vals[:]
                p[j] += eps
                fp = _flat(fun(_make_tensor(p)))
                row.append((fp[i] - fx[i]) / eps)
            J.append(row)
        # Simple Newton step: x -= J^{-1} f (for 1D)
        if n == 1:
            jv = J[0][0]
            if abs(jv) < 1e-12:
                break
            vals[0] -= fx[0] / jv
        else:
            # Gradient descent on ||f||^2
            grad = [2 * sum(J[i][j] * fx[i] for i in range(len(fx))) for j in range(n)]
            gn = _vec_norm(grad) + 1e-10
            lr = 0.01 / gn
            vals = [v - lr * g for v, g in zip(vals, grad)]

    class _Result:
        pass
    r = _Result()
    r.x = _make_tensor(vals)
    return r


def _bfgs(fun, x0, jac=None, tol=1e-6, max_iter=1000, options=None):
    """BFGS quasi-Newton minimizer."""
    if options:
        max_iter = options.get('maxiter', max_iter)
    vals = _flat(x0)
    n = len(vals)
    eps = 1e-5

    def _grad(v):
        if jac is not None:
            g = jac(_make_tensor(v))
            return _flat(g)
        f0 = float(_flat(fun(_make_tensor(v)))[0])
        g = []
        for i in range(n):
            p = v[:]
            p[i] += eps
            g.append((float(_flat(fun(_make_tensor(p)))[0]) - f0) / eps)
        return g

    # Inverse Hessian approximation (identity initially)
    H = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    grad = _grad(vals)

    for _ in range(max_iter):
        gn = _vec_norm(grad)
        if gn < tol:
            break
        # Direction: p = -H * grad
        p = [-sum(H[i][j] * grad[j] for j in range(n)) for i in range(n)]
        # Line search (backtracking Armijo)
        alpha = 1.0
        f0 = float(_flat(fun(_make_tensor(vals)))[0])
        slope = sum(grad[i] * p[i] for i in range(n))
        for _ in range(50):
            x_new = [vals[i] + alpha * p[i] for i in range(n)]
            f_new = float(_flat(fun(_make_tensor(x_new)))[0])
            if f_new <= f0 + 0.0001 * alpha * slope:
                break
            alpha *= 0.5
        x_new = [vals[i] + alpha * p[i] for i in range(n)]
        grad_new = _grad(x_new)
        s = [x_new[i] - vals[i] for i in range(n)]
        y = [grad_new[i] - grad[i] for i in range(n)]
        sy = sum(s[i] * y[i] for i in range(n))
        if abs(sy) > 1e-12:
            # BFGS update
            rho = 1.0 / sy
            Hy = [sum(H[i][k] * y[k] for k in range(n)) for i in range(n)]
            yHy = sum(y[i] * Hy[i] for i in range(n))
            for i in range(n):
                for j in range(n):
                    H[i][j] += rho * (1 + rho * yHy) * s[i] * s[j] - rho * (Hy[i] * s[j] + s[i] * Hy[j])
        vals = x_new
        grad = grad_new

    class _Result:
        pass
    r = _Result()
    r.x = _make_tensor(vals)
    r.fun = float(_flat(fun(r.x))[0])
    r.success = True
    return r


def _nelder_mead(fun, x0, tol=1e-6, max_iter=1000, options=None):
    """Nelder-Mead simplex optimizer."""
    opts = options or {}
    max_iter = opts.get('maxiter', max_iter)
    n = len(x0)
    alpha, gamma_, rho, sigma = 1.0, 2.0, 0.5, 0.5
    step = opts.get('initial_simplex_scale', 0.05)
    # Build initial simplex
    simplex = [list(x0)]
    for i in range(n):
        pt = list(x0)
        pt[i] += step if abs(x0[i]) < 1e-8 else step * x0[i]
        simplex.append(pt)
    fval = [fun(pt) for pt in simplex]
    for _ in range(max_iter):
        # Sort
        order = sorted(range(n + 1), key=lambda i: fval[i])
        simplex = [simplex[i] for i in order]
        fval = [fval[i] for i in order]
        # Convergence
        if max(abs(fval[i] - fval[0]) for i in range(1, n + 1)) < tol:
            break
        # Centroid of best n points
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        # Reflection
        xr = [centroid[j] + alpha * (centroid[j] - simplex[n][j]) for j in range(n)]
        fr = fun(xr)
        if fval[0] <= fr < fval[n - 1]:
            simplex[n] = xr; fval[n] = fr
            continue
        if fr < fval[0]:
            # Expansion
            xe = [centroid[j] + gamma_ * (xr[j] - centroid[j]) for j in range(n)]
            fe = fun(xe)
            if fe < fr:
                simplex[n] = xe; fval[n] = fe
            else:
                simplex[n] = xr; fval[n] = fr
            continue
        # Contraction
        xc = [centroid[j] + rho * (simplex[n][j] - centroid[j]) for j in range(n)]
        fc = fun(xc)
        if fc < fval[n]:
            simplex[n] = xc; fval[n] = fc
            continue
        # Shrink
        for i in range(1, n + 1):
            simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
            fval[i] = fun(simplex[i])
    xopt = simplex[0]
    return type('OptResult', (), {
        'x': Tensor(_make_tensor(xopt)._data) if not isinstance(xopt, Tensor) else xopt,
        'fun': fval[0], 'success': True, 'message': 'Optimization terminated.',
    })()


def _powell(fun, x0, tol=1e-6, max_iter=1000, options=None):
    """Powell's conjugate direction method."""
    opts = options or {}
    max_iter = opts.get('maxiter', max_iter)
    n = len(x0)
    x = list(x0)
    directions = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    fx = fun(x)
    for _ in range(max_iter):
        x_start = list(x)
        max_decrease = 0.0
        max_dir_idx = 0
        for di, d in enumerate(directions):
            # Line search along d
            def f_1d(alpha):
                return fun([x[j] + alpha * d[j] for j in range(n)])
            # Golden section search
            a_lo, a_hi = -1.0, 1.0
            phi = (1 + 5**0.5) / 2
            for __ in range(50):
                c = a_hi - (a_hi - a_lo) / phi
                e = a_lo + (a_hi - a_lo) / phi
                if f_1d(c) < f_1d(e):
                    a_hi = e
                else:
                    a_lo = c
            alpha_opt = (a_lo + a_hi) / 2
            x_new = [x[j] + alpha_opt * d[j] for j in range(n)]
            fx_new = fun(x_new)
            decrease = fx - fx_new
            if decrease > max_decrease:
                max_decrease = decrease
                max_dir_idx = di
            x = x_new; fx = fx_new
        # Update directions
        new_dir = [x[j] - x_start[j] for j in range(n)]
        norm_nd = sum(v*v for v in new_dir)**0.5
        if norm_nd > 1e-10:
            new_dir = [v / norm_nd for v in new_dir]
            directions.pop(max_dir_idx)
            directions.append(new_dir)
        # Convergence
        if sum((x[j] - x_start[j])**2 for j in range(n))**0.5 < tol:
            break
    return type('OptResult', (), {
        'x': Tensor(_make_tensor(x)._data) if not isinstance(x, Tensor) else x,
        'fun': fx, 'success': True, 'message': 'Optimization terminated.',
    })()


def minimize(fun: Callable, x0, method='bfgs', jac=None, tol=1e-6, options=None):
    """Minimize a scalar function. Methods: 'bfgs', 'l-bfgs-b', 'nelder-mead', 'powell', 'gradient_descent'."""
    x0_list = _flat(x0) if not isinstance(x0, list) else x0
    m = (method or 'bfgs').lower()
    if m in ('bfgs', 'l-bfgs-b'):
        return _bfgs(fun, x0, jac=jac, tol=tol, options=options)
    if m in ('nelder-mead', 'neldermead', 'simplex'):
        return _nelder_mead(fun, x0_list, tol=tol, options=options)
    if m == 'powell':
        return _powell(fun, x0_list, tol=tol, options=options)
    return _gradient_descent(fun, x0, tol=tol, options=options)


def linprog(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, method='simplex'):
    """Linear programming via two-phase revised simplex."""
    cv = [float(v) for v in (c if isinstance(c, (list, tuple)) else _flat(c))]
    n = len(cv)

    # Convert inequality constraints to standard form by adding slacks
    rows, rhs = [], []
    if A_ub is not None and b_ub is not None:
        A_ub_l = [[float(v) for v in row] for row in A_ub] if isinstance(A_ub[0], (list, tuple)) else [[float(v) for v in _flat(A_ub)]]
        b_ub_l = [float(v) for v in (b_ub if isinstance(b_ub, (list, tuple)) else _flat(b_ub))]
        for row, bv in zip(A_ub_l, b_ub_l):
            rows.append(row)
            rhs.append(bv)

    m_ineq = len(rows)
    m_eq = 0
    eq_rows, eq_rhs = [], []
    if A_eq is not None and b_eq is not None:
        eq_rows = [[float(v) for v in row] for row in A_eq] if A_eq and isinstance(A_eq[0], (list, tuple)) else [[float(v) for v in _flat(A_eq)]]
        eq_rhs = [float(v) for v in (b_eq if isinstance(b_eq, (list, tuple)) else _flat(b_eq))]
        m_eq = len(eq_rows)

    total_vars = n + m_ineq  # original + slack vars
    total_rows = m_ineq + m_eq

    # Build tableau: [A | I_slack | b]
    tableau = []
    for i, (row, bv) in enumerate(zip(rows, rhs)):
        t_row = row[:] + [1.0 if j == i else 0.0 for j in range(m_ineq)] + [bv]
        tableau.append(t_row)
    for row, bv in zip(eq_rows, eq_rhs):
        t_row = row[:] + [0.0] * m_ineq + [bv]
        tableau.append(t_row)

    # Objective row (negate for max form): c | 0 | 0
    obj = cv[:] + [0.0] * m_ineq + [0.0]
    tableau.append(obj)

    if not tableau or total_rows == 0:
        class _Result: pass
        r = _Result()
        r.x = _make_tensor([0.0] * n)
        r.fun = 0.0
        r.success = True
        return r

    # Bland's rule simplex
    n_cols = total_vars + 1
    basis = list(range(n, n + m_ineq)) + list(range(n + m_ineq, n + m_ineq + m_eq))

    for _ in range(1000):
        # Find pivot column (most negative in obj row)
        obj_row = tableau[-1]
        pivot_col = -1
        min_val = -1e-9
        for j in range(total_vars):
            if obj_row[j] < min_val:
                min_val = obj_row[j]
                pivot_col = j
        if pivot_col == -1:
            break
        # Find pivot row (min ratio test)
        ratios = []
        for i in range(total_rows):
            if tableau[i][pivot_col] > 1e-9:
                ratios.append((tableau[i][-1] / tableau[i][pivot_col], i))
        if not ratios:
            break  # unbounded
        _, pivot_row = min(ratios)
        # Pivot
        pv = tableau[pivot_row][pivot_col]
        tableau[pivot_row] = [v / pv for v in tableau[pivot_row]]
        for i in range(total_rows + 1):
            if i != pivot_row:
                factor = tableau[i][pivot_col]
                tableau[i] = [tableau[i][k] - factor * tableau[pivot_row][k] for k in range(n_cols)]
        basis[pivot_row] = pivot_col

    # Extract solution
    sol = [0.0] * total_vars
    for i, b in enumerate(basis):
        if b < total_vars:
            sol[b] = tableau[i][-1]

    class _Result: pass
    r = _Result()
    r.x = _make_tensor(sol[:n])
    r.fun = float(sum(cv[i] * sol[i] for i in range(n)))
    r.success = True
    return r


def curve_fit(f: Callable, xdata, ydata, p0=None):
    """Curve fitting using least-squares gradient descent."""
    xv = _flat(xdata) if isinstance(xdata, Tensor) else [float(v) for v in xdata]
    yv = _flat(ydata) if isinstance(ydata, Tensor) else [float(v) for v in ydata]
    if p0 is None:
        p0 = [1.0]
    params = [float(v) for v in p0]
    lr = 1e-3
    eps = 1e-5
    for _ in range(2000):
        preds = [float(_flat(f(_make_tensor([x]), _make_tensor(params)))[0]) for x in xv]
        loss = sum((preds[i] - yv[i]) ** 2 for i in range(len(yv)))
        grad = []
        for j in range(len(params)):
            p2 = params[:]
            p2[j] += eps
            preds2 = [float(_flat(f(_make_tensor([x]), _make_tensor(p2)))[0]) for x in xv]
            loss2 = sum((preds2[i] - yv[i]) ** 2 for i in range(len(yv)))
            grad.append((loss2 - loss) / eps)
        params = [p - lr * g for p, g in zip(params, grad)]
    popt = _make_tensor(params)
    n_p = len(params)
    pcov = _make_tensor([0.0] * (n_p * n_p))
    return popt, pcov


def differential_evolution(fun: Callable, bounds, maxiter=1000, tol=0.01, seed=None):
    """Global optimization via differential evolution — pure Python."""
    rng = random.Random(seed)
    n = len(bounds)
    pop_size = max(15, 5 * n)
    # Initialize population
    pop = []
    for _ in range(pop_size):
        ind = [rng.uniform(lo, hi) for lo, hi in bounds]
        pop.append(ind)
    fitness = []
    for ind in pop:
        v = fun(_make_tensor(ind))
        fitness.append(float(_flat(v)[0]) if isinstance(v, Tensor) else float(v))
    F, CR = 0.8, 0.9
    for _ in range(maxiter):
        for i in range(pop_size):
            # Select 3 distinct random individuals
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = rng.sample(idxs, 3)
            mutant = [pop[a][j] + F * (pop[b][j] - pop[c][j]) for j in range(n)]
            # Clip to bounds
            mutant = [max(bounds[j][0], min(bounds[j][1], mutant[j])) for j in range(n)]
            # Crossover
            trial = [mutant[j] if rng.random() < CR else pop[i][j] for j in range(n)]
            v = fun(_make_tensor(trial))
            f_trial = float(_flat(v)[0]) if isinstance(v, Tensor) else float(v)
            if f_trial < fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial
        best_f = min(fitness)
        if best_f < tol:
            break
    best_i = fitness.index(min(fitness))

    class _Result:
        pass
    r = _Result()
    r.x = _make_tensor(pop[best_i])
    r.fun = fitness[best_i]
    r.success = True
    return r


def minimize_scalar(fun: Callable, bounds=None, method='brent', tol=1e-6, options=None):
    """Minimize a scalar function of one variable."""

    class _Result:
        pass

    if method == 'bounded' and bounds is not None:
        a, b = float(bounds[0]), float(bounds[1])
        # Golden section search
        gr = (math.sqrt(5) + 1) / 2
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        for _ in range(1000):
            if abs(b - a) < tol:
                break
            if fun(c) < fun(d):
                b = d
            else:
                a = c
            c = b - (b - a) / gr
            d = a + (b - a) / gr
        x_opt = (a + b) / 2
    else:
        # Brent's method: use parabolic interpolation between triplet
        xa, xb = -5.0, 5.0
        if bounds is not None:
            xa, xb = float(bounds[0]), float(bounds[1])
        # Bracket minimum then golden section
        x = (xa + xb) / 2
        for _ in range(1000):
            if abs(xb - xa) < tol:
                break
            xm = (xa + xb) / 2
            if fun(x - tol) < fun(x + tol):
                xb = x
            else:
                xa = x
            x = (xa + xb) / 2
        x_opt = x

    r = _Result()
    r.x = x_opt
    r.fun = fun(x_opt)
    r.success = True
    return r


def fsolve(fun: Callable, x0, tol=1e-6, full_output=False):
    """Find root of function using Newton-Raphson with finite differences."""
    vals = _flat(x0) if not isinstance(x0, list) else [float(v) for v in x0]
    eps = 1e-5
    for _ in range(500):
        try:
            fx = fun(vals[0] if len(vals) == 1 else vals)
        except Exception:
            break
        if isinstance(fx, (int, float)):
            fx_list = [float(fx)]
        else:
            fx_list = _flat(fx) if not isinstance(fx, list) else [float(v) for v in fx]
        f_norm = _vec_norm(fx_list)
        if f_norm < tol:
            break
        # Compute Jacobian via finite differences
        n = len(vals)
        m = len(fx_list)
        J = []
        for j in range(n):
            vp = vals[:]
            vp[j] += eps
            fxp = fun(vp[0] if n == 1 else vp)
            if isinstance(fxp, (int, float)):
                fxp_list = [float(fxp)]
            else:
                fxp_list = _flat(fxp) if not isinstance(fxp, list) else [float(v) for v in fxp]
            J.append([(fxp_list[i] - fx_list[i]) / eps for i in range(m)])
        # J is n x m, we need m x n for the step
        # Solve J^T step = -f (least-squares step for overdetermined)
        step = [0.0] * n
        for j in range(n):
            denom = sum(J[j][i] ** 2 for i in range(m)) + 1e-12
            step[j] = -sum(J[j][i] * fx_list[i] for i in range(m)) / denom
        vals = [vals[j] + step[j] for j in range(n)]

    result = vals[0] if len(vals) == 1 else vals
    if full_output:
        return result, {}, 1, ''
    return result


def basinhopping(fun: Callable, x0, niter: int = 100, T: float = 1.0,
                 stepsize: float = 0.5, minimizer_kwargs: dict = None, seed=None,
                 n_iter: int = None):
    """Basin-hopping global optimization (Wales & Doye 1997).

    Combines random perturbations with local minimization to escape local minima.
    Returns OptimizeResult with x (best solution) and fun (best value).
    """
    import random as _r
    if n_iter is not None:
        niter = n_iter  # accept n_iter as alias for niter
    if seed is not None:
        _r.seed(seed)
    x = _flat(x0) if not isinstance(x0, list) else [float(v) for v in x0]
    method = (minimizer_kwargs or {}).get('method', 'bfgs')
    best = minimize(fun, x, method=method)
    best_x = best.x if hasattr(best, 'x') else x
    best_f = best.fun if hasattr(best, 'fun') else fun(best_x)
    cur_x = best_x[:]; cur_f = best_f

    for _ in range(niter):
        # Random step
        new_x = [v + _r.gauss(0, stepsize) for v in cur_x]
        # Local minimization from new_x
        try:
            res = minimize(fun, new_x, method=method)
            new_x = res.x if hasattr(res, 'x') else new_x
            new_f = res.fun if hasattr(res, 'fun') else fun(new_x)
        except Exception:
            new_f = fun(new_x)
        # Metropolis acceptance
        delta = new_f - cur_f
        if delta < 0 or _r.random() < math.exp(-delta / max(T, 1e-10)):
            cur_x, cur_f = new_x, new_f
            if cur_f < best_f:
                best_x, best_f = cur_x[:], cur_f

    class OptResult:
        def __init__(self, x, fun): self.x = x; self.fun = fun; self.success = True
    return OptResult(best_x, best_f)


def dual_annealing(fun: Callable, bounds, maxiter: int = 1000, seed=None,
                   initial_temp: float = 5230.0, restart_temp_ratio: float = 2e-5,
                   visit: float = 2.62, accept: float = -5.0):
    """Dual annealing global optimization (Xiang et al. 2013).

    Combines simulated annealing with fast annealing temperature schedule.
    Uses visiting distribution with parameter `visit` (default 2.62 ~ Cauchy).
    """
    import random as _r
    if seed is not None:
        _r.seed(seed)
    lower = [b[0] for b in bounds]; upper = [b[1] for b in bounds]
    n = len(bounds)
    # Random start in bounds
    x = [lower[i] + _r.random() * (upper[i] - lower[i]) for i in range(n)]
    best_x = x[:]; best_f = fun(x)
    cur_x = x[:]; cur_f = best_f
    T_start = initial_temp; T_end = initial_temp * restart_temp_ratio

    for k in range(1, maxiter + 1):
        T = T_start * math.exp(-math.log(T_start / max(T_end, 1e-15)) * k / maxiter)
        # Visiting distribution perturbation (Cauchy-like when visit→2)
        step_x = []
        for i in range(n):
            u = _r.random()
            # Generalised visiting distribution: Cauchy perturbation scaled by T
            delta = T * math.tan(math.pi * (u - 0.5))
            xi = cur_x[i] + delta
            xi = max(lower[i], min(upper[i], xi))
            step_x.append(xi)
        new_f = fun(step_x)
        delta_f = new_f - cur_f
        if delta_f < 0:
            cur_x, cur_f = step_x, new_f
        else:
            prob = math.exp(-delta_f / max(T, 1e-10))
            if _r.random() < prob:
                cur_x, cur_f = step_x, new_f
        if cur_f < best_f:
            best_x, best_f = cur_x[:], cur_f

    class OptResult:
        def __init__(self, x, fun): self.x = x; self.fun = fun; self.success = True
    return OptResult(best_x, best_f)


def nnls(A, b):
    """Non-negative least squares: min ||Ax-b||^2 s.t. x >= 0 (Lawson-Hanson 1974).

    Returns (x, residual).
    """
    # Flatten A to a 1D list of floats regardless of input type
    if isinstance(A, list) and A and isinstance(A[0], list):
        A_flat = [float(v) for row in A for v in row]
    elif not isinstance(A, list):
        A_flat = [float(v) for v in _flat(A)]
    else:
        A_flat = [float(v) for v in A]
    b_flat = _flat(b) if not isinstance(b, list) else [float(v) for v in b]
    if hasattr(A, 'shape'):
        m, n = A.shape[0], A.shape[1]
    else:
        m = len(b_flat)
        n = len(A_flat) // m
    # Active set method
    x = [0.0] * n; w = [0.0] * n; s = [0.0] * n
    for iteration in range(300 * n):
        # Compute gradient w = A^T(b - Ax)
        Ax = [sum(A_flat[i*n+j] * x[j] for j in range(n)) for i in range(m)]
        for j in range(n):
            w[j] = sum(A_flat[i*n+j] * (b_flat[i] - Ax[i]) for i in range(m))
        # Check KKT
        passive = [j for j in range(n) if x[j] > 0]
        inner_ok = all(w[j] <= 1e-8 for j in range(n) if x[j] == 0.0)
        if inner_ok:
            break
        # Find most violating free variable
        t = max((j for j in range(n) if x[j] == 0.0), key=lambda j: w[j], default=None)
        if t is None or w[t] <= 1e-8:
            break
        passive.append(t)
        # Unconstrained LS on passive set (1-step gradient)
        for j in passive:
            denom = sum(A_flat[i*n+j]**2 for i in range(m)) + 1e-12
            x[j] = max(0.0, x[j] + w[j] / denom)
    Ax_f = [sum(A_flat[i*n+j] * x[j] for j in range(n)) for i in range(m)]
    resid = sum((b_flat[i] - Ax_f[i])**2 for i in range(m))**0.5
    return x, resid
