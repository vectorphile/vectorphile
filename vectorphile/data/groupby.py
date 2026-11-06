"""GroupBy operations for DataFrame."""
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Callable, List, Union

if TYPE_CHECKING:
    from vectorphile.data.frame import DataFrame
    from vectorphile.data.series import Series


class GroupBy:
    def __init__(self, df: 'DataFrame', by: Union[str, List[str]]):
        self._df = df
        self.by = [by] if isinstance(by, str) else by
        self._groups = self._build_groups()

    def _build_groups(self):
        keys = list(zip(*[self._df[col].values for col in self.by]))
        unique_keys = list(dict.fromkeys(keys))
        groups = {}
        for key in unique_keys:
            mask = np.ones(len(self._df), dtype=bool)
            for col, val in zip(self.by, key):
                mask &= (self._df[col].values == val)
            groups[key[0] if len(key) == 1 else key] = mask
        return groups

    def _apply_agg(self, func_name: str, *args, **kwargs):
        from vectorphile.data.frame import DataFrame
        from vectorphile.data.series import Series
        result_rows = []
        keys = []
        numeric_cols = [c for c in self._df.columns if c not in self.by and
                        np.issubdtype(self._df[c].values.dtype, np.number)]
        for key, mask in self._groups.items():
            row = {}
            for col in numeric_cols:
                vals = self._df[col].values[mask]
                fn = getattr(np, func_name, None)
                if fn:
                    row[col] = float(fn(vals, *args, **kwargs))
            result_rows.append(row)
            keys.append(key)
        result = DataFrame({col: [r.get(col, np.nan) for r in result_rows] for col in numeric_cols})
        # Add group key column
        for i, col in enumerate(self.by):
            result.insert(i, col, keys if len(self.by) == 1 else [k[i] for k in keys])
        return result

    def sum(self): return self._apply_agg('sum')
    def mean(self): return self._apply_agg('mean')
    def max(self): return self._apply_agg('max')
    def min(self): return self._apply_agg('min')
    def std(self): return self._apply_agg('std')
    def var(self): return self._apply_agg('var')
    def count(self):
        from vectorphile.data.frame import DataFrame
        keys, counts = [], []
        for key, mask in self._groups.items():
            keys.append(key)
            counts.append(int(mask.sum()))
        return DataFrame({self.by[0]: keys, 'count': counts})

    def agg(self, func):
        from vectorphile.data.frame import DataFrame
        from vectorphile.data.series import Series
        result_rows = []
        keys = []
        numeric_cols = [c for c in self._df.columns if c not in self.by]
        for key, mask in self._groups.items():
            row = {}
            for col in numeric_cols:
                vals = self._df[col].values[mask]
                s = Series(vals, name=col)
                if callable(func):
                    row[col] = func(s)
                elif isinstance(func, str):
                    row[col] = getattr(s, func)()
            result_rows.append(row)
            keys.append(key)
        result = DataFrame({col: [r.get(col, np.nan) for r in result_rows] for col in numeric_cols})
        for i, col in enumerate(self.by):
            result.insert(i, col, keys if len(self.by) == 1 else [k[i] for k in keys])
        return result

    def apply(self, func: Callable) -> 'DataFrame':
        from vectorphile.data.frame import DataFrame
        parts = []
        for key, mask in self._groups.items():
            subset = self._df[mask]
            result = func(subset)
            if isinstance(result, DataFrame):
                parts.append(result)
        return DataFrame.concat(parts, ignore_index=True) if parts else self._df.__class__({})

    def __iter__(self):
        for key, mask in self._groups.items():
            yield key, self._df[mask]
