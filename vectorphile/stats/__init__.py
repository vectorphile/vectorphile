"""Statistical functions."""
import numpy as np
from typing import Optional, Tuple
from vectorphile.core.tensor import Tensor


def mean(x: Tensor, dim=None, keepdim=False) -> Tensor:
    return Tensor(x._data.mean(axis=dim, keepdims=keepdim))

def median(x: Tensor, dim=None) -> Tensor:
    return Tensor(np.median(x._data, axis=dim))

def std(x: Tensor, dim=None, keepdim=False, ddof=1) -> Tensor:
    return Tensor(x._data.std(axis=dim, keepdims=keepdim, ddof=ddof))

def var(x: Tensor, dim=None, keepdim=False, ddof=1) -> Tensor:
    return Tensor(x._data.var(axis=dim, keepdims=keepdim, ddof=ddof))

def corr(x: Tensor, y: Tensor) -> float:
    return float(np.corrcoef(x._data.flatten(), y._data.flatten())[0, 1])

def cov(x: Tensor, rowvar=True, ddof=1) -> Tensor:
    return Tensor(np.cov(x._data, rowvar=rowvar, ddof=ddof))

def zscore(x: Tensor, dim=None) -> Tensor:
    mu = x._data.mean(axis=dim, keepdims=True)
    sigma = x._data.std(axis=dim, keepdims=True, ddof=0) + 1e-12
    return Tensor((x._data - mu) / sigma)

def percentile(x: Tensor, q, dim=None) -> Tensor:
    return Tensor(np.percentile(x._data, q, axis=dim))

def kurtosis(x: Tensor) -> float:
    from scipy.stats import kurtosis as sp_kurtosis
    return float(sp_kurtosis(x._data.flatten()))

def skewness(x: Tensor) -> float:
    from scipy.stats import skew
    return float(skew(x._data.flatten()))

def pearsonr(x: Tensor, y: Tensor) -> Tuple[float, float]:
    from scipy.stats import pearsonr as sp_pearsonr
    return sp_pearsonr(x._data.flatten(), y._data.flatten())

def spearmanr(x: Tensor, y: Tensor) -> Tuple[float, float]:
    from scipy.stats import spearmanr as sp_spearmanr
    res = sp_spearmanr(x._data.flatten(), y._data.flatten())
    return res.statistic, res.pvalue

def t_test(x: Tensor, y: Optional[Tensor] = None, popmean: float = 0.0):
    from scipy.stats import ttest_1samp, ttest_ind
    if y is None:
        return ttest_1samp(x._data.flatten(), popmean)
    return ttest_ind(x._data.flatten(), y._data.flatten())

def chi2_test(observed: Tensor, expected: Optional[Tensor] = None):
    from scipy.stats import chisquare
    if expected is not None:
        return chisquare(observed._data.flatten(), expected._data.flatten())
    return chisquare(observed._data.flatten())

def anova(*groups: Tensor):
    from scipy.stats import f_oneway
    return f_oneway(*[g._data.flatten() for g in groups])

def histogram(x: Tensor, bins=10, range=None) -> Tuple[Tensor, Tensor]:
    counts, edges = np.histogram(x._data, bins=bins, range=range)
    return Tensor(counts.astype(np.float32)), Tensor(edges.astype(np.float32))

def cumulative_distribution(x: Tensor) -> Tuple[Tensor, Tensor]:
    sorted_data = np.sort(x._data.flatten())
    cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    return Tensor(sorted_data.astype(np.float32)), Tensor(cdf.astype(np.float32))

def mode(x: Tensor):
    vals, counts = np.unique(x._data, return_counts=True)
    return Tensor(vals[np.argmax(counts)])

def describe(x: Tensor) -> dict:
    flat = x._data.flatten()
    return {
        'count': len(flat),
        'mean': float(flat.mean()),
        'std': float(flat.std(ddof=1)),
        'min': float(flat.min()),
        '25%': float(np.percentile(flat, 25)),
        '50%': float(np.percentile(flat, 50)),
        '75%': float(np.percentile(flat, 75)),
        'max': float(flat.max()),
    }
