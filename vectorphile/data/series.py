"""Series — 1D labeled array, pandas-compatible API — pure Python."""
from __future__ import annotations
import math
from typing import Any, Callable, Iterable, Optional, Union

_NAN = float('nan')


def _is_nan(v) -> bool:
    try: return math.isnan(float(v))
    except (TypeError, ValueError): return v is None


def _coerce(v):
    try: return float(v)
    except (TypeError, ValueError): return v


class Series:
    def __init__(self, data=None, index=None, name=None, dtype=None):
        if isinstance(data, Series):
            self._data = data._data[:]
            self._index = data._index[:] if index is None else list(index)
            self.name = data.name if name is None else name
        elif isinstance(data, dict):
            self._index = list(data.keys())
            self._data = [_coerce(v) for v in data.values()]
            self.name = name
        elif data is None:
            self._data = []
            self._index = []
            self.name = name
        else:
            self._data = [_coerce(v) for v in data]
            self._index = list(index) if index is not None else list(range(len(self._data)))
            self.name = name
        if dtype is not None:
            try:
                self._data = [dtype(v) for v in self._data]
            except (TypeError, ValueError):
                pass

    @property
    def values(self) -> list:
        return self._data

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, val):
        self._index = list(val)

    @property
    def dtype(self):
        return type(self._data[0]).__name__ if self._data else 'float64'

    @property
    def shape(self):
        return (len(self._data),)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Series(self._data[key], index=self._index[key], name=self.name)
        if isinstance(key, list):
            if key and isinstance(key[0], bool):
                idxs = [i for i, b in enumerate(key) if b]
                return Series([self._data[i] for i in idxs],
                               index=[self._index[i] for i in idxs], name=self.name)
            idxs = [self._index.index(k) for k in key]
            return Series([self._data[i] for i in idxs], index=key, name=self.name)
        if isinstance(key, int) and not isinstance(key, bool):
            return self._data[key]
        if key in self._index:
            return self._data[self._index.index(key)]
        raise KeyError(key)

    def __setitem__(self, key, value):
        if isinstance(key, str) and key in self._index:
            self._data[self._index.index(key)] = _coerce(value)
        elif isinstance(key, int):
            self._data[key] = _coerce(value)

    def _op(self, other, fn) -> 'Series':
        if isinstance(other, Series):
            return Series([fn(a, b) for a, b in zip(self._data, other._data)],
                          index=self._index, name=self.name)
        return Series([fn(v, other) for v in self._data], index=self._index, name=self.name)

    def __add__(self, other): return self._op(other, lambda a, b: a + b)
    def __radd__(self, other): return self._op(other, lambda a, b: b + a)
    def __sub__(self, other): return self._op(other, lambda a, b: a - b)
    def __rsub__(self, other): return self._op(other, lambda a, b: b - a)
    def __mul__(self, other): return self._op(other, lambda a, b: a * b)
    def __rmul__(self, other): return self._op(other, lambda a, b: b * a)
    def __truediv__(self, other): return self._op(other, lambda a, b: a / b)
    def __floordiv__(self, other): return self._op(other, lambda a, b: a // b)
    def __pow__(self, other): return self._op(other, lambda a, b: a ** b)
    def __neg__(self): return Series([-v for v in self._data], index=self._index, name=self.name)
    def __eq__(self, other): return self._op(other, lambda a, b: a == b)
    def __ne__(self, other): return self._op(other, lambda a, b: a != b)
    def __lt__(self, other): return self._op(other, lambda a, b: a < b)
    def __le__(self, other): return self._op(other, lambda a, b: a <= b)
    def __gt__(self, other): return self._op(other, lambda a, b: a > b)
    def __ge__(self, other): return self._op(other, lambda a, b: a >= b)
    def __invert__(self): return Series([not v for v in self._data], index=self._index, name=self.name)

    def apply(self, func: Callable) -> 'Series':
        return Series([func(v) for v in self._data], index=self._index, name=self.name)

    def map(self, func_or_dict) -> 'Series':
        if callable(func_or_dict):
            return self.apply(func_or_dict)
        return Series([func_or_dict.get(v, _NAN) for v in self._data], index=self._index, name=self.name)

    def isnull(self) -> 'Series':
        return Series([_is_nan(v) for v in self._data], index=self._index)

    def notnull(self) -> 'Series':
        return Series([not _is_nan(v) for v in self._data], index=self._index)

    def dropna(self) -> 'Series':
        pairs = [(i, v) for i, v in zip(self._index, self._data) if not _is_nan(v)]
        if not pairs: return Series([], name=self.name)
        idx, vals = zip(*pairs)
        return Series(list(vals), index=list(idx), name=self.name)

    def fillna(self, value) -> 'Series':
        return Series([value if _is_nan(v) else v for v in self._data],
                      index=self._index, name=self.name)

    def sort_values(self, ascending=True) -> 'Series':
        pairs = sorted(zip(self._index, self._data), key=lambda x: (x[1] is None, x[1]),
                       reverse=not ascending)
        idx = [p[0] for p in pairs]; vals = [p[1] for p in pairs]
        return Series(vals, index=idx, name=self.name)

    def sort_index(self, ascending=True) -> 'Series':
        pairs = sorted(zip(self._index, self._data), key=lambda x: x[0],
                       reverse=not ascending)
        idx = [p[0] for p in pairs]; vals = [p[1] for p in pairs]
        return Series(vals, index=idx, name=self.name)

    def unique(self) -> list:
        seen = {}
        return [seen.setdefault(v, v) for v in self._data if v not in seen]

    def nunique(self) -> int:
        return len(set(v for v in self._data if not _is_nan(v)))

    def value_counts(self, normalize=False) -> 'Series':
        counts = {}
        for v in self._data:
            counts[v] = counts.get(v, 0) + 1
        order = sorted(counts.items(), key=lambda x: -x[1])
        vals = [k for k, _ in order]; cnts = [c for _, c in order]
        if normalize:
            total = sum(cnts)
            cnts = [c / total for c in cnts]
        return Series(cnts, index=vals, name=self.name)

    def _float_data(self) -> list:
        return [float(v) for v in self._data if not _is_nan(v)]

    def mean(self) -> float:
        d = self._float_data(); return sum(d) / len(d) if d else _NAN
    def median(self) -> float:
        d = sorted(self._float_data()); n = len(d)
        if not n: return _NAN
        return (d[n//2] + d[(n-1)//2]) / 2
    def std(self, ddof=1) -> float:
        d = self._float_data(); n = len(d)
        if n <= ddof: return _NAN
        mu = sum(d) / n
        return math.sqrt(sum((v - mu)**2 for v in d) / (n - ddof))
    def var(self, ddof=1) -> float:
        d = self._float_data(); n = len(d)
        if n <= ddof: return _NAN
        mu = sum(d) / n
        return sum((v - mu)**2 for v in d) / (n - ddof)
    def min(self): return min(self._data)
    def max(self): return max(self._data)
    def sum(self): return sum(v for v in self._data if not _is_nan(v))
    def prod(self):
        r = 1
        for v in self._data:
            if not _is_nan(v): r *= v
        return r
    def count(self) -> int: return sum(1 for v in self._data if not _is_nan(v))
    def abs(self): return Series([abs(v) for v in self._data], index=self._index, name=self.name)
    def cumsum(self):
        r, cs = 0, []
        for v in self._data: r += v; cs.append(r)
        return Series(cs, index=self._index, name=self.name)
    def cumprod(self):
        r, cp = 1, []
        for v in self._data: r *= v; cp.append(r)
        return Series(cp, index=self._index, name=self.name)

    def head(self, n=5) -> 'Series':
        return Series(self._data[:n], index=self._index[:n], name=self.name)

    def tail(self, n=5) -> 'Series':
        return Series(self._data[-n:], index=self._index[-n:], name=self.name)

    def reset_index(self, drop=False) -> 'Series':
        if drop:
            return Series(self._data, index=list(range(len(self._data))), name=self.name)
        from vectorphile.data.frame import DataFrame
        return DataFrame({'index': self._index, self.name or 0: self._data})

    def to_list(self) -> list:
        return list(self._data)

    def to_numpy(self):
        try:
            import array as _a
            return _a.array('d', [float(v) for v in self._data])
        except Exception:
            return self._data[:]

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        lines = [f'{i}    {v}' for i, v in zip(self._index, self._data)]
        if self.name:
            lines.append(f'Name: {self.name}, dtype: {self.dtype}')
        return '\n'.join(lines)

    def __str__(self): return self.__repr__()

    def astype(self, dtype) -> 'Series':
        try:
            return Series([dtype(v) for v in self._data], index=self._index, name=self.name)
        except Exception:
            return Series(self._data[:], index=self._index, name=self.name)

    def clip(self, lower=None, upper=None) -> 'Series':
        def _cl(v):
            if lower is not None: v = max(lower, v)
            if upper is not None: v = min(upper, v)
            return v
        return Series([_cl(v) for v in self._data], index=self._index, name=self.name)

    def round(self, decimals=0) -> 'Series':
        return Series([round(float(v), decimals) for v in self._data],
                      index=self._index, name=self.name)

    def quantile(self, q=0.5) -> float:
        d = sorted(self._float_data()); n = len(d)
        if not n: return _NAN
        pos = q * (n - 1); lo = int(pos); hi = min(lo + 1, n - 1)
        return d[lo] + (pos - lo) * (d[hi] - d[lo])

    def corr(self, other: 'Series') -> float:
        a = self._float_data(); b = other._float_data()
        n = min(len(a), len(b))
        if n < 2: return _NAN
        ma = sum(a[:n]) / n; mb = sum(b[:n]) / n
        num = sum((a[i]-ma)*(b[i]-mb) for i in range(n))
        da = math.sqrt(sum((v-ma)**2 for v in a[:n]))
        db = math.sqrt(sum((v-mb)**2 for v in b[:n]))
        return num / (da * db + 1e-10)

    def describe(self) -> 'Series':
        d = self._float_data()
        n = len(d)
        mu = sum(d) / n if n else _NAN
        std = math.sqrt(sum((v-mu)**2 for v in d) / max(n-1, 1)) if n > 1 else _NAN
        def _perc(q): return self.quantile(q)
        stats = {
            'count': float(n),
            'mean': mu,
            'std': std,
            'min': min(d) if d else _NAN,
            '25%': _perc(0.25),
            '50%': _perc(0.50),
            '75%': _perc(0.75),
            'max': max(d) if d else _NAN,
        }
        return Series(list(stats.values()), index=list(stats.keys()), name=self.name)

    def shift(self, periods: int = 1, fill_value=None) -> 'Series':
        """Shift values by periods, filling edge with fill_value or NaN."""
        import math as _m
        fv = fill_value if fill_value is not None else _m.nan
        n = len(self._data)
        if periods >= 0:
            vals = [fv] * min(periods, n) + self._data[:max(0, n - periods)]
        else:
            p = -periods
            vals = self._data[p:] + [fv] * min(p, n)
        return Series(vals, index=list(self._index), name=self.name)

    def diff(self, periods: int = 1) -> 'Series':
        """First discrete difference."""
        import math as _m
        d = self._data
        n = len(d)
        vals = [_m.nan] * min(periods, n)
        for i in range(periods, n):
            try:
                vals.append(float(d[i]) - float(d[i - periods]))
            except (TypeError, ValueError):
                vals.append(_m.nan)
        return Series(vals, index=list(self._index), name=self.name)

    def pct_change(self, periods: int = 1) -> 'Series':
        """Percentage change between current and prior element."""
        import math as _m
        d = self._data
        n = len(d)
        vals = [_m.nan] * min(periods, n)
        for i in range(periods, n):
            try:
                prev = float(d[i - periods])
                if prev == 0:
                    vals.append(_m.nan)
                else:
                    vals.append((float(d[i]) - prev) / abs(prev))
            except (TypeError, ValueError):
                vals.append(_m.nan)
        return Series(vals, index=list(self._index), name=self.name)

    def rank(self, ascending: bool = True) -> 'Series':
        """Rank values (average method for ties)."""
        d = self._float_data()
        n = len(d)
        if not n:
            return Series([], index=list(self._index), name=self.name)
        sorted_vals = sorted(range(n), key=lambda i: d[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and d[sorted_vals[j]] == d[sorted_vals[j + 1]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[sorted_vals[k]] = avg_rank
            i = j + 1
        if not ascending:
            ranks = [n + 1 - r for r in ranks]
        return Series(ranks, index=list(self._index), name=self.name)

    def skew(self) -> float:
        """Sample skewness."""
        import math as _m
        d = self._float_data()
        n = len(d)
        if n < 3:
            return _m.nan
        mu = sum(d) / n
        s = _m.sqrt(sum((v - mu)**2 for v in d) / (n - 1))
        if s == 0:
            return 0.0
        return (sum((v - mu)**3 for v in d) / n) / (s**3)

    def kurt(self) -> float:
        """Sample excess kurtosis."""
        import math as _m
        d = self._float_data()
        n = len(d)
        if n < 4:
            return _m.nan
        mu = sum(d) / n
        s = _m.sqrt(sum((v - mu)**2 for v in d) / (n - 1))
        if s == 0:
            return 0.0
        return (sum((v - mu)**4 for v in d) / n) / (s**4) - 3.0

    def isin(self, values) -> 'Series':
        """Boolean mask: True where series value is in values."""
        val_set = set(values)
        return Series([v in val_set for v in self._data], index=list(self._index), name=self.name)

    def between(self, left, right, inclusive: str = 'both') -> 'Series':
        """Boolean mask: True where left <= value <= right."""
        def _check(v):
            try:
                fv = float(v)
                if inclusive == 'both': return left <= fv <= right
                if inclusive == 'left': return left <= fv < right
                if inclusive == 'right': return left < fv <= right
                return left < fv < right  # neither
            except (TypeError, ValueError):
                return False
        return Series([_check(v) for v in self._data], index=list(self._index), name=self.name)

    def autocorr(self, lag: int = 1) -> float:
        """Lag-1 autocorrelation by default."""
        return self.corr(self.shift(lag).dropna())

    def rolling(self, window: int):
        """Return rolling window object."""
        return _RollingSeries(self, window)


class _RollingSeries:
    def __init__(self, series, window: int):
        self._s = series
        self._w = window

    def _apply(self, func):
        import math as _m
        d = self._s._data; n = len(d); w = self._w
        vals = [_m.nan] * (w - 1)
        for i in range(w - 1, n):
            window_data = [float(v) for v in d[i - w + 1: i + 1] if v == v]  # skip NaN
            vals.append(func(window_data) if window_data else _m.nan)
        return Series(vals, index=list(self._s._index), name=self._s.name)

    def mean(self): return self._apply(lambda w: sum(w) / len(w))
    def sum(self): return self._apply(lambda w: sum(w))
    def min(self): return self._apply(lambda w: min(w))
    def max(self): return self._apply(lambda w: max(w))
    def std(self):
        import math as _m
        def _std(w):
            n = len(w)
            if n < 2: return _m.nan
            mu = sum(w) / n
            return _m.sqrt(sum((v - mu)**2 for v in w) / (n - 1))
        return self._apply(_std)
