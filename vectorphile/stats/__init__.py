"""Statistical functions — pure Python, no external dependencies."""
import math
from typing import Optional, Tuple, List
from vectorphile.core.tensor import Tensor


def _flat(x: Tensor) -> List[float]:
    if hasattr(x, '_data'):
        d = x._data
        if hasattr(d, 'to_flat_list'):
            return [float(v) for v in d.to_flat_list()]
        if hasattr(d, 'flatten'):
            return [float(v) for v in d.flatten().tolist()]
        if hasattr(d, 'tolist'):
            vals = d.tolist()
            out = []
            def _r(v):
                if isinstance(v, (list, tuple)):
                    for x in v: _r(x)
                else:
                    out.append(float(v))
            _r(vals)
            return out
    if hasattr(x, 'to_flat_list'):
        return [float(v) for v in x.to_flat_list()]
    return [float(v) for v in x]


def _percentile_val(vals: List[float], q: float) -> float:
    s = sorted(vals)
    n = len(s)
    idx = q / 100.0 * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _mean_val(vals: List[float]) -> float:
    return sum(vals) / max(len(vals), 1)


def _std_val(vals: List[float], ddof: int = 1) -> float:
    n = len(vals)
    if n <= ddof:
        return 0.0
    mu = _mean_val(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - ddof))


def _var_val(vals: List[float], ddof: int = 1) -> float:
    n = len(vals)
    if n <= ddof:
        return 0.0
    mu = _mean_val(vals)
    return sum((v - mu) ** 2 for v in vals) / (n - ddof)


def _from_list(vals) -> Tensor:
    from vectorphile._core.ndarray import array as nd_array
    return Tensor(nd_array(vals))


def mean(x: Tensor, dim=None, keepdim=False) -> Tensor:
    vals = _flat(x)
    if dim is None:
        return _from_list([_mean_val(vals)])
    # Fallback: delegate to data if possible
    if hasattr(x, '_data') and hasattr(x._data, 'mean'):
        return Tensor(x._data.mean(axis=dim, keepdims=keepdim))
    return _from_list([_mean_val(vals)])


def median(x: Tensor, dim=None) -> Tensor:
    vals = sorted(_flat(x))
    n = len(vals)
    if n == 0:
        return _from_list([0.0])
    m = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
    return _from_list([m])


def std(x: Tensor, dim=None, keepdim=False, ddof=1) -> Tensor:
    vals = _flat(x)
    if dim is None:
        return _from_list([_std_val(vals, ddof)])
    if hasattr(x, '_data') and hasattr(x._data, 'std'):
        return Tensor(x._data.std(axis=dim, keepdims=keepdim, ddof=ddof))
    return _from_list([_std_val(vals, ddof)])


def var(x: Tensor, dim=None, keepdim=False, ddof=1) -> Tensor:
    vals = _flat(x)
    if dim is None:
        return _from_list([_var_val(vals, ddof)])
    if hasattr(x, '_data') and hasattr(x._data, 'var'):
        return Tensor(x._data.var(axis=dim, keepdims=keepdim, ddof=ddof))
    return _from_list([_var_val(vals, ddof)])


def corr(x: Tensor, y: Tensor) -> float:
    xv = _flat(x)
    yv = _flat(y)
    n = min(len(xv), len(yv))
    if n == 0:
        return 0.0
    xv, yv = xv[:n], yv[:n]
    mx, my = _mean_val(xv), _mean_val(yv)
    cov = sum((xv[i] - mx) * (yv[i] - my) for i in range(n)) / max(n - 1, 1)
    sx = _std_val(xv)
    sy = _std_val(yv)
    return float(cov / (sx * sy)) if sx * sy > 0 else 0.0


def cov(x: Tensor, rowvar=True, ddof=1) -> Tensor:
    vals = _flat(x)
    if hasattr(x, '_data') and hasattr(x._data, 'shape') and len(x._data.shape) == 2:
        n, p = x._data.shape
        rows = [[_flat(x)[i * p + j] for j in range(p)] for i in range(n)]
        if not rowvar:
            rows = [[rows[i][j] for i in range(n)] for j in range(p)]
        k = len(rows)
        means = [_mean_val(r) for r in rows]
        cm = []
        for i in range(k):
            row = []
            for j in range(k):
                n2 = min(len(rows[i]), len(rows[j]))
                c = sum((rows[i][t] - means[i]) * (rows[j][t] - means[j]) for t in range(n2)) / max(n2 - ddof, 1)
                row.append(c)
            cm.append(row)
        from vectorphile._core.ndarray import array as nd_array
        return Tensor(nd_array(cm))
    v = _var_val(vals, ddof)
    return _from_list([v])


def zscore(x: Tensor, dim=None) -> Tensor:
    vals = _flat(x)
    mu = _mean_val(vals)
    sigma = _std_val(vals, ddof=0) + 1e-12
    return _from_list([(v - mu) / sigma for v in vals])


def percentile(x: Tensor, q, dim=None) -> Tensor:
    vals = _flat(x)
    if isinstance(q, (list, tuple)):
        return _from_list([_percentile_val(vals, qi) for qi in q])
    return _from_list([_percentile_val(vals, float(q))])


def kurtosis(x: Tensor) -> float:
    vals = _flat(x)
    n = len(vals)
    if n < 4:
        return 0.0
    mu = _mean_val(vals)
    s = _std_val(vals, ddof=0)
    if s == 0:
        return 0.0
    m4 = sum((v - mu) ** 4 for v in vals) / n
    return float(m4 / s ** 4 - 3.0)


def skewness(x: Tensor) -> float:
    vals = _flat(x)
    n = len(vals)
    if n < 3:
        return 0.0
    mu = _mean_val(vals)
    s = _std_val(vals, ddof=0)
    if s == 0:
        return 0.0
    m3 = sum((v - mu) ** 3 for v in vals) / n
    return float(m3 / s ** 3)


def pearsonr(x: Tensor, y: Tensor) -> Tuple[float, float]:
    r = corr(x, y)
    n = len(_flat(x))
    if abs(r) >= 1.0 or n < 3:
        return r, 0.0
    t = r * math.sqrt(n - 2) / math.sqrt(1 - r ** 2 + 1e-12)
    # Approximate two-tailed p-value using normal approximation
    z = abs(t) / math.sqrt(1 + t ** 2 / (n - 2))
    p = 2 * (1 - _normal_cdf(abs(t)))
    return r, p


def _normal_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _rank(vals: List[float]) -> List[float]:
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and vals[order[j]] == vals[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def spearmanr(x: Tensor, y: Tensor) -> Tuple[float, float]:
    xv = _flat(x)
    yv = _flat(y)
    xr = _rank(xv)
    yr = _rank(yv)
    # Compute Pearson on ranks
    n = min(len(xr), len(yr))
    if n == 0:
        return 0.0, 1.0
    xr, yr = xr[:n], yr[:n]
    mx, my = _mean_val(xr), _mean_val(yr)
    num = sum((xr[i] - mx) * (yr[i] - my) for i in range(n))
    den = math.sqrt(sum((v - mx) ** 2 for v in xr) * sum((v - my) ** 2 for v in yr))
    r = float(num / den) if den > 0 else 0.0
    p = 2 * (1 - _normal_cdf(abs(r * math.sqrt(n - 2) / math.sqrt(1 - r ** 2 + 1e-12))))
    return r, p


def t_test(x: Tensor, y: Optional[Tensor] = None, popmean: float = 0.0):
    xv = _flat(x)
    n = len(xv)
    mu = _mean_val(xv)
    s = _std_val(xv)
    if y is None:
        # one-sample
        t = (mu - popmean) / (s / math.sqrt(n) + 1e-12)
        p = 2 * (1 - _normal_cdf(abs(t)))
        return t, p
    yv = _flat(y)
    ny = len(yv)
    muy = _mean_val(yv)
    sy = _std_val(yv)
    se = math.sqrt(s**2 / n + sy**2 / ny + 1e-12)
    t = (mu - muy) / se
    p = 2 * (1 - _normal_cdf(abs(t)))
    return t, p


def chi2_test(observed: Tensor, expected: Optional[Tensor] = None):
    obs = _flat(observed)
    if expected is not None:
        exp = _flat(expected)
    else:
        mean_obs = _mean_val(obs)
        exp = [mean_obs] * len(obs)
    chi2 = sum((o - e) ** 2 / max(e, 1e-10) for o, e in zip(obs, exp))
    df = len(obs) - 1
    # Approximate p-value using chi2 CDF (simple survival function)
    # For now return statistic and df
    return chi2, df


def anova(*groups: Tensor):
    group_vals = [_flat(g) for g in groups]
    k = len(group_vals)
    all_vals = [v for g in group_vals for v in g]
    n = len(all_vals)
    grand_mean = _mean_val(all_vals)
    # Between-group sum of squares
    ss_between = sum(len(g) * (_mean_val(g) - grand_mean) ** 2 for g in group_vals)
    ss_within = sum(sum((v - _mean_val(g)) ** 2 for v in g) for g in group_vals)
    df_between = k - 1
    df_within = n - k
    ms_between = ss_between / max(df_between, 1)
    ms_within = ss_within / max(df_within, 1)
    f = ms_between / max(ms_within, 1e-10)
    # Approximate p-value
    p = 1.0 - _normal_cdf(f)
    return f, p


def histogram(x: Tensor, bins=10, range=None) -> Tuple[Tensor, Tensor]:
    vals = _flat(x)
    if range is not None:
        lo, hi = range
    else:
        lo, hi = min(vals), max(vals)
    if lo == hi:
        hi = lo + 1
    bin_width = (hi - lo) / bins
    counts = [0.0] * bins
    for v in vals:
        idx = int((v - lo) / bin_width)
        idx = min(idx, bins - 1)
        if lo <= v <= hi:
            counts[idx] += 1.0
    edges = [lo + i * bin_width for i in range(bins + 1)]
    return _from_list(counts), _from_list(edges)


def cumulative_distribution(x: Tensor) -> Tuple[Tensor, Tensor]:
    vals = sorted(_flat(x))
    n = len(vals)
    cdf = [(i + 1) / n for i in range(n)]
    return _from_list(vals), _from_list(cdf)


def mode(x: Tensor) -> Tensor:
    vals = _flat(x)
    counts: dict = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts, key=counts.get)
    return _from_list([best])


def describe(x: Tensor) -> dict:
    vals = _flat(x)
    return {
        'count': len(vals),
        'mean': _mean_val(vals),
        'std': _std_val(vals, ddof=1),
        'min': min(vals) if vals else 0.0,
        '25%': _percentile_val(vals, 25),
        '50%': _percentile_val(vals, 50),
        '75%': _percentile_val(vals, 75),
        'max': max(vals) if vals else 0.0,
    }


def mannwhitneyu(x: Tensor, y: Tensor, alternative: str = 'two-sided'):
    """Mann-Whitney U test (non-parametric, two-sample)."""
    xv = sorted(_flat(x))
    yv = sorted(_flat(y))
    n1, n2 = len(xv), len(yv)
    combined = sorted([(v, 0) for v in xv] + [(v, 1) for v in yv])
    # Rank with midpoint tie handling
    ranks = [0.0] * (n1 + n2)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    r1 = sum(ranks[k] for k, (_, g) in enumerate(combined) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u_stat = min(u1, u2)
    # Normal approximation for large samples
    mu_u = n1 * n2 / 2.0
    sigma_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0) + 1e-10
    z = (u_stat - mu_u) / sigma_u
    from vectorphile.special import ndtr
    p_one = ndtr(z)
    if alternative == 'two-sided':
        p = 2.0 * min(p_one, 1 - p_one)
    elif alternative == 'less':
        p = p_one
    else:
        p = 1.0 - p_one
    return u_stat, float(p)


def ks_test(x: Tensor, y=None, cdf=None):
    """Kolmogorov-Smirnov test (one-sample vs CDF, or two-sample)."""
    xv = sorted(_flat(x))
    n = len(xv)
    if y is None and cdf is None:
        # one-sample vs standard normal
        from vectorphile.special import ndtr
        cdf_fn = lambda v: ndtr(v)
    elif cdf is not None:
        cdf_fn = cdf
    else:
        # two-sample KS
        yv = sorted(_flat(y))
        n2 = len(yv)
        combined = sorted(xv + yv)
        d = 0.0
        for v in combined:
            ecdf1 = sum(1 for xi in xv if xi <= v) / n
            ecdf2 = sum(1 for yi in yv if yi <= v) / n2
            d = max(d, abs(ecdf1 - ecdf2))
        en = math.sqrt(n * n2 / (n + n2))
        # Approximate p-value
        p = math.exp(-2 * (en * d) ** 2)
        return d, float(p)
    d = 0.0
    for i, v in enumerate(xv):
        ecdf = (i + 1) / n
        d = max(d, abs(ecdf - cdf_fn(v)))
    en = math.sqrt(n)
    p = math.exp(-2 * (en * d) ** 2)
    return d, float(p)


def f_test(x: Tensor, y: Tensor):
    """F-test for equality of variances (Levene approximation)."""
    xv, yv = _flat(x), _flat(y)
    n1, n2 = len(xv), len(yv)
    v1 = _var_val(xv, ddof=1)
    v2 = _var_val(yv, ddof=1)
    if v2 < 1e-15:
        return float('inf'), 0.0
    f = v1 / v2
    # Approximate p-value via log-normal or F-dist (normal approx for large n)
    df1, df2 = n1 - 1, n2 - 1
    # Simple two-sided approximation
    log_f = math.log(max(f, 1e-10))
    p = min(1.0, 2.0 * math.exp(-0.5 * log_f ** 2 * min(df1, df2) / 4))
    return float(f), float(p)


def wilcoxon(x: Tensor, y: Optional[Tensor] = None, alternative: str = 'two-sided'):
    """Wilcoxon signed-rank test.

    Parameters
    ----------
    x : Tensor
        First sample (or differences if y is None).
    y : Tensor, optional
        Second sample. If provided, computes x - y.
    alternative : str
        'two-sided', 'greater', or 'less'.

    Returns
    -------
    statistic, p-value
    """
    xv = _flat(x)
    if y is not None:
        yv = _flat(y)
        diffs = [a - b for a, b in zip(xv, yv)]
    else:
        diffs = xv
    # Remove zeros
    diffs = [d for d in diffs if d != 0.0]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0
    # Rank by absolute value
    abs_diffs = [abs(d) for d in diffs]
    order = sorted(range(n), key=lambda i: abs_diffs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs_diffs[order[j]] == abs_diffs[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    # W+ = sum of positive ranks, W- = sum of negative ranks
    w_plus = sum(ranks[i] for i in range(n) if diffs[i] > 0)
    w_minus = sum(ranks[i] for i in range(n) if diffs[i] < 0)
    w = w_plus if alternative != 'less' else w_minus
    # Normal approximation
    mu_w = n * (n + 1) / 4.0
    sigma_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma_w < 1e-10:
        return float(min(w_plus, w_minus)), 1.0
    z = (w - mu_w) / sigma_w
    p = 1.0 - _normal_cdf(abs(z))
    if alternative == 'two-sided':
        p = min(1.0, 2.0 * p)
    return float(min(w_plus, w_minus)), float(p)


def binomtest(k: int, n: int, p: float = 0.5, alternative: str = 'two-sided'):
    """Binomial test.

    Parameters
    ----------
    k : int
        Number of successes.
    n : int
        Number of trials.
    p : float
        Hypothesized probability of success.
    alternative : str
        'two-sided', 'greater', or 'less'.

    Returns
    -------
    p-value (float)
    """
    # Exact binomial using log-factorial for numerical stability
    def _log_binom(n_, k_):
        return math.lgamma(n_ + 1) - math.lgamma(k_ + 1) - math.lgamma(n_ - k_ + 1)
    def _binom_pmf(n_, k_, p_):
        if k_ < 0 or k_ > n_:
            return 0.0
        log_p = _log_binom(n_, k_) + k_ * math.log(max(p_, 1e-300)) + (n_ - k_) * math.log(max(1 - p_, 1e-300))
        return math.exp(log_p)
    p_obs = _binom_pmf(n, k, p)
    if alternative == 'less':
        return float(sum(_binom_pmf(n, j, p) for j in range(k + 1)))
    if alternative == 'greater':
        return float(sum(_binom_pmf(n, j, p) for j in range(k, n + 1)))
    # Two-sided: sum all pmf values <= p_obs
    return float(min(1.0, sum(_binom_pmf(n, j, p) for j in range(n + 1) if _binom_pmf(n, j, p) <= p_obs + 1e-10)))


def levene(*groups, center: str = 'median') -> Tuple[float, float]:
    """Levene's test for equality of variances."""
    k = len(groups)
    all_flat = [_flat(g) for g in groups]
    ns = [len(g) for g in all_flat]
    N = sum(ns)
    if center == 'median':
        centrals = [sorted(g)[len(g) // 2] for g in all_flat]
    elif center == 'trimmed':
        centrals = [sum(sorted(g)[len(g)//4: 3*len(g)//4]) / max(1, len(g)//2) for g in all_flat]
    else:
        centrals = [sum(g) / len(g) for g in all_flat]
    Z = [[abs(x - c) for x in g] for g, c in zip(all_flat, centrals)]
    grand_mean = sum(sum(z) for z in Z) / N
    group_means = [sum(z) / len(z) for z in Z]
    ss_between = sum(ns[i] * (group_means[i] - grand_mean) ** 2 for i in range(k))
    ss_within = sum(sum((z - group_means[i]) ** 2 for z in Z[i]) for i in range(k))
    df1, df2 = k - 1, N - k
    if ss_within == 0:
        return float('inf'), 0.0
    W = (ss_between / df1) / (ss_within / df2)
    # F-distribution p-value approximation using beta function
    try:
        x = df1 * W / (df1 * W + df2)
        from vectorphile.special import betainc
        p = 1.0 - betainc(df1 / 2, df2 / 2, x)
    except Exception:
        p = float('nan')
    return float(W), float(p)


def bartlett(*groups) -> Tuple[float, float]:
    """Bartlett's test for homogeneity of variances (assumes normality)."""
    k = len(groups)
    all_flat = [_flat(g) for g in groups]
    ns = [len(g) for g in all_flat]
    N = sum(ns)
    vars_ = []
    for g in all_flat:
        m = sum(g) / len(g)
        vars_.append(sum((x - m) ** 2 for x in g) / (len(g) - 1) if len(g) > 1 else 0.0)
    sp2 = sum((ns[i] - 1) * vars_[i] for i in range(k)) / (N - k)
    if sp2 == 0:
        return float('inf'), 0.0
    ln_sp2 = math.log(sp2)
    numerator = (N - k) * ln_sp2 - sum((ns[i] - 1) * math.log(max(vars_[i], 1e-300)) for i in range(k))
    denom = 1 + (sum(1 / (ns[i] - 1) for i in range(k)) - 1 / (N - k)) / (3 * (k - 1))
    T = numerator / denom
    # Chi-squared approximation with df = k-1
    df = k - 1
    try:
        from vectorphile.special import gammainc
        p = 1.0 - gammainc(df / 2, T / 2)
    except Exception:
        p = float('nan')
    return float(T), float(p)


def kruskal(*groups) -> Tuple[float, float]:
    """Kruskal-Wallis H-test (non-parametric one-way ANOVA)."""
    k = len(groups)
    all_flat = [_flat(g) for g in groups]
    ns = [len(g) for g in all_flat]
    N = sum(ns)
    all_vals = [(v, i) for i, g in enumerate(all_flat) for v in g]
    all_vals.sort(key=lambda x: x[0])
    # Assign ranks with midpoint tie-breaking
    ranks = [0.0] * len(all_vals)
    i = 0
    while i < len(all_vals):
        j = i
        while j < len(all_vals) - 1 and all_vals[j + 1][0] == all_vals[j][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for idx in range(i, j + 1):
            ranks[idx] = avg_rank
        i = j + 1
    group_rank_sums = [0.0] * k
    for idx, (v, gi) in enumerate(all_vals):
        group_rank_sums[gi] += ranks[idx]
    H = 12 / (N * (N + 1)) * sum(group_rank_sums[i] ** 2 / ns[i] for i in range(k)) - 3 * (N + 1)
    df = k - 1
    try:
        from vectorphile.special import gammainc
        p = 1.0 - gammainc(df / 2, H / 2)
    except Exception:
        p = float('nan')
    return float(H), float(p)


def friedmanchisquare(*groups) -> Tuple[float, float]:
    """Friedman test: non-parametric repeated measures ANOVA."""
    k = len(groups)
    all_flat = [_flat(g) for g in groups]
    n = len(all_flat[0])
    # Rank within each row (block)
    chi2 = 0.0
    rank_sums = [0.0] * k
    for i in range(n):
        row = [(all_flat[j][i], j) for j in range(k)]
        row.sort(key=lambda x: x[0])
        # Assign ranks with ties
        ri = 0
        while ri < k:
            rj = ri
            while rj < k - 1 and row[rj + 1][0] == row[rj][0]:
                rj += 1
            avg_rank = (ri + rj) / 2 + 1
            for idx in range(ri, rj + 1):
                rank_sums[row[idx][1]] += avg_rank
            ri = rj + 1
    grand_rank_mean = (k + 1) / 2
    SS = sum((rank_sums[j] / n - grand_rank_mean) ** 2 for j in range(k))
    chi2 = 12 * n / (k * (k + 1)) * SS
    df = k - 1
    try:
        from vectorphile.special import gammainc
        p = 1.0 - gammainc(df / 2, chi2 / 2)
    except Exception:
        p = float('nan')
    return float(chi2), float(p)
