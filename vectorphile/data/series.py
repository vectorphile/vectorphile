"""Series — 1D labeled array, pandas-compatible API."""
from __future__ import annotations
import numpy as np
from typing import Any, Callable, Iterable, Optional, Union


class Series:
    def __init__(self, data=None, index=None, name=None, dtype=None):
        if isinstance(data, Series):
            self._data = data._data.copy()
            self._index = data._index.copy() if index is None else list(index)
            self.name = data.name if name is None else name
        elif isinstance(data, dict):
            self._index = list(data.keys())
            self._data = np.array(list(data.values()), dtype=dtype)
            self.name = name
        elif data is None:
            self._data = np.array([], dtype=dtype or np.float64)
            self._index = []
            self.name = name
        else:
            self._data = np.array(data, dtype=dtype)
            self._index = list(index) if index is not None else list(range(len(self._data)))
            self.name = name

    @property
    def values(self) -> np.ndarray:
        return self._data

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, val):
        self._index = list(val)

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self):
        return self._data.shape

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        if isinstance(key, str) or (not isinstance(key, (int, slice, np.ndarray, list))):
            if key in self._index:
                return self._data[self._index.index(key)]
            raise KeyError(key)
        if isinstance(key, slice):
            return Series(self._data[key], index=self._index[key], name=self.name)
        if isinstance(key, (list, np.ndarray)):
            if isinstance(key, list) and len(key) > 0 and isinstance(key[0], bool):
                key = np.array(key)
            if isinstance(key, np.ndarray) and key.dtype == bool:
                idx = [self._index[i] for i in np.where(key)[0]]
                return Series(self._data[key], index=idx, name=self.name)
            idxs = [self._index.index(k) for k in key]
            return Series(self._data[idxs], index=key, name=self.name)
        return self._data[key]

    def __setitem__(self, key, value):
        if isinstance(key, str) and key in self._index:
            self._data[self._index.index(key)] = value
        else:
            self._data[key] = value

    def _op(self, other, fn):
        if isinstance(other, Series):
            return Series(fn(self._data, other._data), index=self._index, name=self.name)
        return Series(fn(self._data, other), index=self._index, name=self.name)

    def __add__(self, other): return self._op(other, np.add)
    def __radd__(self, other): return self._op(other, lambda a, b: np.add(b, a))
    def __sub__(self, other): return self._op(other, np.subtract)
    def __rsub__(self, other): return self._op(other, lambda a, b: np.subtract(b, a))
    def __mul__(self, other): return self._op(other, np.multiply)
    def __rmul__(self, other): return self._op(other, lambda a, b: np.multiply(b, a))
    def __truediv__(self, other): return self._op(other, np.true_divide)
    def __floordiv__(self, other): return self._op(other, np.floor_divide)
    def __pow__(self, other): return self._op(other, np.power)
    def __neg__(self): return Series(-self._data, index=self._index, name=self.name)
    def __eq__(self, other): return self._op(other, np.equal)
    def __ne__(self, other): return self._op(other, np.not_equal)
    def __lt__(self, other): return self._op(other, np.less)
    def __le__(self, other): return self._op(other, np.less_equal)
    def __gt__(self, other): return self._op(other, np.greater)
    def __ge__(self, other): return self._op(other, np.greater_equal)

    def apply(self, func: Callable) -> 'Series':
        return Series([func(v) for v in self._data], index=self._index, name=self.name)

    def map(self, func_or_dict) -> 'Series':
        if callable(func_or_dict):
            return self.apply(func_or_dict)
        return Series([func_or_dict.get(v, np.nan) for v in self._data], index=self._index, name=self.name)

    def isnull(self) -> 'Series':
        try:
            return Series(np.isnan(self._data.astype(float)), index=self._index)
        except (ValueError, TypeError):
            return Series([v is None for v in self._data], index=self._index)

    def notnull(self) -> 'Series':
        null = self.isnull()
        return Series(~null._data, index=self._index)

    def dropna(self) -> 'Series':
        mask = self.notnull()._data
        new_idx = [self._index[i] for i in range(len(mask)) if mask[i]]
        return Series(self._data[mask], index=new_idx, name=self.name)

    def fillna(self, value) -> 'Series':
        result = self._data.copy().astype(object)
        try:
            null_mask = np.isnan(self._data.astype(float))
            result = self._data.copy().astype(float)
            result[null_mask] = value
        except (ValueError, TypeError):
            result = np.array([value if v is None else v for v in self._data])
        return Series(result, index=self._index, name=self.name)

    def sort_values(self, ascending=True) -> 'Series':
        order = np.argsort(self._data)
        if not ascending:
            order = order[::-1]
        new_idx = [self._index[i] for i in order]
        return Series(self._data[order], index=new_idx, name=self.name)

    def sort_index(self, ascending=True) -> 'Series':
        order = np.argsort(self._index)
        if not ascending:
            order = order[::-1]
        new_idx = [self._index[i] for i in order]
        return Series(self._data[order], index=new_idx, name=self.name)

    def unique(self) -> np.ndarray:
        return np.unique(self._data)

    def nunique(self) -> int:
        return len(np.unique(self._data))

    def value_counts(self, normalize=False) -> 'Series':
        vals, counts = np.unique(self._data, return_counts=True)
        order = np.argsort(-counts)
        vals, counts = vals[order], counts[order]
        if normalize:
            counts = counts / counts.sum()
        return Series(counts, index=list(vals), name=self.name)

    def mean(self): return float(np.nanmean(self._data.astype(float)))
    def median(self): return float(np.nanmedian(self._data.astype(float)))
    def std(self, ddof=1): return float(np.nanstd(self._data.astype(float), ddof=ddof))
    def var(self, ddof=1): return float(np.nanvar(self._data.astype(float), ddof=ddof))
    def min(self): return self._data.min()
    def max(self): return self._data.max()
    def sum(self): return self._data.sum()
    def prod(self): return self._data.prod()
    def count(self): return int(self.notnull()._data.sum())
    def abs(self): return Series(np.abs(self._data), index=self._index, name=self.name)
    def cumsum(self): return Series(np.cumsum(self._data), index=self._index, name=self.name)
    def cumprod(self): return Series(np.cumprod(self._data), index=self._index, name=self.name)

    def head(self, n=5) -> 'Series':
        return Series(self._data[:n], index=self._index[:n], name=self.name)

    def tail(self, n=5) -> 'Series':
        return Series(self._data[-n:], index=self._index[-n:], name=self.name)

    def reset_index(self, drop=False) -> 'Series':
        if drop:
            return Series(self._data, index=list(range(len(self._data))), name=self.name)
        # Return as DataFrame when drop=False
        from vectorphile.data.frame import DataFrame
        return DataFrame({'index': self._index, self.name or 0: self._data})

    def to_list(self):
        return self._data.tolist()

    def to_numpy(self) -> np.ndarray:
        return self._data.copy()

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        lines = []
        for i, v in zip(self._index, self._data):
            lines.append(f'{i}    {v}')
        if self.name:
            lines.append(f'Name: {self.name}, dtype: {self.dtype}')
        return '\n'.join(lines)

    def __str__(self):
        return self.__repr__()

    def astype(self, dtype) -> 'Series':
        return Series(self._data.astype(dtype), index=self._index, name=self.name)

    def clip(self, lower=None, upper=None) -> 'Series':
        return Series(np.clip(self._data, lower, upper), index=self._index, name=self.name)

    def round(self, decimals=0) -> 'Series':
        return Series(np.round(self._data.astype(float), decimals), index=self._index, name=self.name)

    def quantile(self, q=0.5):
        return float(np.nanpercentile(self._data.astype(float), q * 100))

    def corr(self, other: 'Series') -> float:
        return float(np.corrcoef(self._data.astype(float), other._data.astype(float))[0, 1])

    def describe(self) -> 'Series':
        d = self._data.astype(float)
        stats = {
            'count': float(len(d)),
            'mean': float(np.nanmean(d)),
            'std': float(np.nanstd(d, ddof=1)),
            'min': float(np.nanmin(d)),
            '25%': float(np.nanpercentile(d, 25)),
            '50%': float(np.nanpercentile(d, 50)),
            '75%': float(np.nanpercentile(d, 75)),
            'max': float(np.nanmax(d)),
        }
        return Series(list(stats.values()), index=list(stats.keys()), name=self.name)
