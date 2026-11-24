"""GroupBy operations for DataFrame — pure Python."""
from __future__ import annotations
import math
from typing import TYPE_CHECKING, Callable, List, Union

if TYPE_CHECKING:
    from vectorphile.data.frame import DataFrame
    from vectorphile.data.series import Series

_NAN = float('nan')


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))


class GroupBy:
    def __init__(self, df: 'DataFrame', by: Union[str, List[str]]):
        self._df = df
        self.by = [by] if isinstance(by, str) else by
        self._groups = self._build_groups()

    def _build_groups(self):
        n = len(self._df)
        keys = list(zip(*[self._df[col].values for col in self.by]))
        unique_keys = list(dict.fromkeys(keys))
        groups = {}
        for key in unique_keys:
            indices = [i for i in range(n)
                       if all(self._df[col].values[i] == val
                              for col, val in zip(self.by, key))]
            groups[key[0] if len(key) == 1 else key] = indices
        return groups

    def _numeric_cols(self):
        return [c for c in self._df.columns if c not in self.by
                and any(_is_num(self._df[c].values[i]) for i in range(len(self._df)))]

    def _apply_agg(self, fn):
        from vectorphile.data.frame import DataFrame
        num_cols = self._numeric_cols()
        result = {col: [] for col in num_cols}
        keys = []
        for key, indices in self._groups.items():
            keys.append(key)
            for col in num_cols:
                vals = [float(self._df[col].values[i]) for i in indices
                        if _is_num(self._df[col].values[i])]
                result[col].append(fn(vals) if vals else _NAN)
        df = DataFrame(result)
        for i, col in enumerate(self.by):
            df.insert(i, col, keys if len(self.by) == 1 else [k[i] for k in keys])
        return df

    def sum(self): return self._apply_agg(sum)
    def mean(self): return self._apply_agg(lambda v: sum(v) / len(v))
    def max(self): return self._apply_agg(max)
    def min(self): return self._apply_agg(min)
    def std(self):
        def _std(v):
            n = len(v)
            if n < 2: return _NAN
            mu = sum(v) / n
            return math.sqrt(sum((x - mu)**2 for x in v) / (n - 1))
        return self._apply_agg(_std)
    def var(self):
        def _var(v):
            n = len(v)
            if n < 2: return _NAN
            mu = sum(v) / n
            return sum((x - mu)**2 for x in v) / (n - 1)
        return self._apply_agg(_var)

    def count(self):
        from vectorphile.data.frame import DataFrame
        keys = []; counts = []
        for key, indices in self._groups.items():
            keys.append(key)
            counts.append(len(indices))
        return DataFrame({self.by[0]: keys, 'count': counts})

    def agg(self, func):
        from vectorphile.data.frame import DataFrame
        from vectorphile.data.series import Series
        num_cols = [c for c in self._df.columns if c not in self.by]
        result = {col: [] for col in num_cols}
        keys = []
        for key, indices in self._groups.items():
            keys.append(key)
            for col in num_cols:
                vals = [self._df[col].values[i] for i in indices]
                s = Series(vals, name=col)
                if callable(func):
                    result[col].append(func(s))
                elif isinstance(func, str):
                    result[col].append(getattr(s, func)())
        df = DataFrame(result)
        for i, col in enumerate(self.by):
            df.insert(i, col, keys if len(self.by) == 1 else [k[i] for k in keys])
        return df

    def apply(self, func: Callable) -> 'DataFrame':
        from vectorphile.data.frame import DataFrame
        parts = []
        for key, indices in self._groups.items():
            subset = self._df.iloc[indices]
            result = func(subset)
            if isinstance(result, DataFrame):
                parts.append(result)
        return DataFrame.concat(parts, ignore_index=True) if parts else self._df.__class__({})

    def __iter__(self):
        for key, indices in self._groups.items():
            yield key, self._df.iloc[indices]
