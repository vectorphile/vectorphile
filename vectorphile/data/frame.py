"""DataFrame — 2D tabular data, pandas-compatible API — pure Python."""
from __future__ import annotations
import csv
import io
import json
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from vectorphile.data.series import Series

_NAN = float('nan')


def _is_nan(v) -> bool:
    try: return math.isnan(float(v))
    except (TypeError, ValueError): return v is None


def _coerce(v):
    if v is None: return _NAN
    try: return float(v)
    except (TypeError, ValueError): return v


class DataFrame:
    def __init__(self, data=None, index=None, columns=None):
        if data is None:
            self._cols: Dict[str, list] = {}
            self._index: List = []
            self._columns: List[str] = []
        elif isinstance(data, dict):
            self._columns = list(columns) if columns is not None else list(data.keys())
            self._cols = {}
            first_len = None
            for col in self._columns:
                val = data.get(col, [])
                if isinstance(val, Series):
                    lst = list(val.values)
                elif isinstance(val, list):
                    lst = val
                else:
                    lst = list(val)
                self._cols[col] = lst
                if first_len is None:
                    first_len = len(lst)
            n = first_len or 0
            self._index = list(index) if index is not None else list(range(n))
        elif isinstance(data, list):
            if len(data) == 0:
                self._cols, self._columns, self._index = {}, [], []
            elif isinstance(data[0], dict):
                self._columns = list(columns) if columns is not None else list(data[0].keys())
                self._cols = {col: [row.get(col, _NAN) for row in data] for col in self._columns}
                self._index = list(index) if index is not None else list(range(len(data)))
            else:
                n_cols = len(data[0]) if hasattr(data[0], '__len__') else 1
                self._columns = list(columns) if columns is not None else [str(i) for i in range(n_cols)]
                self._cols = {col: [row[i] if hasattr(row, '__getitem__') else row
                                     for row in data] for i, col in enumerate(self._columns)}
                self._index = list(index) if index is not None else list(range(len(data)))
        else:
            raise TypeError(f'Unsupported data type: {type(data)}')

    @property
    def columns(self) -> List[str]:
        return self._columns

    @columns.setter
    def columns(self, val):
        new_cols = list(val)
        self._cols = {new: self._cols[old] for old, new in zip(self._columns, new_cols)}
        self._columns = new_cols

    @property
    def index(self) -> List:
        return self._index

    @index.setter
    def index(self, val):
        self._index = list(val)

    @property
    def shape(self):
        return (len(self._index), len(self._columns))

    @property
    def dtypes(self) -> Series:
        return Series({col: type(self._cols[col][0]).__name__ if self._cols[col] else 'object'
                       for col in self._columns})

    def __len__(self): return len(self._index)

    def __getitem__(self, key):
        if isinstance(key, str):
            return Series(self._cols[key], index=self._index, name=key)
        if isinstance(key, list):
            return DataFrame({k: self._cols[k] for k in key}, index=self._index)
        if isinstance(key, Series):
            mask = key.values
            idxs = [i for i, b in enumerate(mask) if b]
            new_idx = [self._index[i] for i in idxs]
            return DataFrame({col: [self._cols[col][i] for i in idxs] for col in self._columns}, index=new_idx)
        if isinstance(key, int):
            return DataFrame({col: [self._cols[col][key]] for col in self._columns}, index=[self._index[key]])
        if isinstance(key, slice):
            idxs = range(len(self._index))[key]
            new_idx = [self._index[i] for i in idxs]
            return DataFrame({col: [self._cols[col][i] for i in idxs] for col in self._columns}, index=new_idx)
        raise KeyError(key)

    def __setitem__(self, key, value):
        if isinstance(value, Series):
            value = list(value.values)
        elif not isinstance(value, list):
            value = list(value)
        if key not in self._columns:
            self._columns.append(key)
        self._cols[key] = value

    def insert(self, loc: int, column: str, value):
        if isinstance(value, Series):
            value = list(value.values)
        elif not isinstance(value, list):
            value = list(value)
        self._cols[column] = value
        if column not in self._columns:
            self._columns.insert(loc, column)

    @property
    def loc(self): return _LocIndexer(self)

    @property
    def iloc(self): return _ILocIndexer(self)

    def head(self, n=5) -> 'DataFrame': return self[:n]
    def tail(self, n=5) -> 'DataFrame': return self[-n:]

    def copy(self) -> 'DataFrame':
        return DataFrame({col: self._cols[col][:] for col in self._columns}, index=self._index[:])

    def isnull(self) -> 'DataFrame':
        return DataFrame({col: [_is_nan(v) for v in self._cols[col]] for col in self._columns},
                         index=self._index)

    def notnull(self) -> 'DataFrame':
        return DataFrame({col: [not _is_nan(v) for v in self._cols[col]] for col in self._columns},
                         index=self._index)

    def dropna(self, axis=0, how='any', subset=None) -> 'DataFrame':
        if axis == 0:
            cols = subset if subset else self._columns
            n = len(self._index)
            if how == 'any':
                keep = [i for i in range(n) if not any(_is_nan(self._cols[c][i]) for c in cols)]
            else:
                keep = [i for i in range(n) if not all(_is_nan(self._cols[c][i]) for c in cols)]
            new_idx = [self._index[i] for i in keep]
            return DataFrame({col: [self._cols[col][i] for i in keep] for col in self._columns}, index=new_idx)
        result = {}
        for col in self._columns:
            has_null = any(_is_nan(v) for v in self._cols[col])
            if (how == 'any' and not has_null) or (how == 'all' and not all(_is_nan(v) for v in self._cols[col])):
                result[col] = self._cols[col]
        return DataFrame(result, index=self._index)

    def fillna(self, value) -> 'DataFrame':
        return DataFrame({col: [value if _is_nan(v) else v for v in self._cols[col]]
                          for col in self._columns}, index=self._index)

    def astype(self, dtype) -> 'DataFrame':
        if isinstance(dtype, dict):
            result = {col: ([dtype[col](v) for v in self._cols[col]] if col in dtype else self._cols[col][:])
                      for col in self._columns}
        else:
            result = {col: [dtype(v) for v in self._cols[col]] for col in self._columns}
        return DataFrame(result, index=self._index)

    def rename(self, columns=None, index=None) -> 'DataFrame':
        if columns:
            new_cols = {columns.get(col, col): self._cols[col] for col in self._columns}
            new_col_list = [columns.get(col, col) for col in self._columns]
            df = DataFrame(new_cols, index=self._index)
            df._columns = new_col_list
            return df
        if index:
            new_idx = [index.get(i, i) for i in self._index]
            return DataFrame({col: self._cols[col][:] for col in self._columns}, index=new_idx)
        return self.copy()

    def drop(self, labels=None, axis=0, columns=None) -> 'DataFrame':
        if columns is not None or axis == 1:
            drop_cols = set(columns or (labels if isinstance(labels, list) else [labels]))
            keep = [c for c in self._columns if c not in drop_cols]
            return DataFrame({c: self._cols[c] for c in keep}, index=self._index)
        if axis == 0:
            if not isinstance(labels, list): labels = [labels]
            label_set = set(labels)
            keep = [i for i in range(len(self._index)) if self._index[i] not in label_set]
            new_idx = [self._index[i] for i in keep]
            return DataFrame({col: [self._cols[col][i] for i in keep] for col in self._columns}, index=new_idx)
        return self.copy()

    def sort_values(self, by, ascending=True) -> 'DataFrame':
        if isinstance(by, str): by = [by]
        n = len(self._index)
        order = sorted(range(n), key=lambda i: tuple(self._cols[c][i] for c in by),
                       reverse=not ascending)
        new_idx = [self._index[i] for i in order]
        return DataFrame({col: [self._cols[col][i] for i in order] for col in self._columns}, index=new_idx)

    def reset_index(self, drop=False) -> 'DataFrame':
        df = DataFrame({col: self._cols[col][:] for col in self._columns},
                       index=list(range(len(self._index))))
        if not drop:
            df.insert(0, 'index', self._index[:])
        return df

    def set_index(self, col: str) -> 'DataFrame':
        new_idx = self._cols[col][:]
        keep = [c for c in self._columns if c != col]
        return DataFrame({c: self._cols[c] for c in keep}, index=new_idx)

    def groupby(self, by):
        from vectorphile.data.groupby import GroupBy
        return GroupBy(self, by)

    def merge(self, right: 'DataFrame', how='inner', on=None, left_on=None, right_on=None) -> 'DataFrame':
        lkey = on if on else left_on
        rkey = on if on else right_on
        if isinstance(lkey, str): lkey, rkey = [lkey], [rkey or lkey]
        lvals = list(zip(*[self._cols[k] for k in lkey]))
        rvals = list(zip(*[right._cols[k] for k in rkey]))
        rmap = {}
        for i, rv in enumerate(rvals): rmap.setdefault(rv, []).append(i)
        rows_l, rows_r = [], []
        for i, lv in enumerate(lvals):
            if lv in rmap:
                for j in rmap[lv]: rows_l.append(i); rows_r.append(j)
            elif how in ('left', 'outer'):
                rows_l.append(i); rows_r.append(None)
        if how in ('right', 'outer'):
            lset = set(lvals)
            for j, rv in enumerate(rvals):
                if rv not in lset: rows_l.append(None); rows_r.append(j)
        result = {}
        for col in self._columns:
            result[col] = [self._cols[col][i] if i is not None else _NAN for i in rows_l]
        for col in right._columns:
            if col not in lkey:
                result[col] = [right._cols[col][j] if j is not None else _NAN for j in rows_r]
        return DataFrame(result)

    def apply(self, func: Callable, axis=0):
        if axis == 1:
            results = []
            for i in range(len(self)):
                row = Series({col: self._cols[col][i] for col in self._columns})
                results.append(func(row))
            return Series(results, index=self._index)
        result = {}
        for col in self._columns:
            result[col] = func(Series(self._cols[col], index=self._index, name=col))
        return Series(list(result.values()), index=list(result.keys()))

    def pivot(self, index, columns, values) -> 'DataFrame':
        idx_vals = sorted(set(self._cols[index]))
        col_vals = sorted(set(self._cols[columns]))
        result = {str(cv): [] for cv in col_vals}
        new_index = []
        n = len(self._index)
        for iv in idx_vals:
            new_index.append(iv)
            for cv in col_vals:
                found = [self._cols[values][i] for i in range(n)
                         if self._cols[index][i] == iv and self._cols[columns][i] == cv]
                result[str(cv)].append(float(found[0]) if found else _NAN)
        return DataFrame(result, index=new_index)

    def melt(self, id_vars=None, value_vars=None, var_name='variable', value_name='value') -> 'DataFrame':
        id_vars = id_vars or []
        value_vars = value_vars or [c for c in self._columns if c not in id_vars]
        rows = []
        for i in range(len(self)):
            for col in value_vars:
                row = {id_col: self._cols[id_col][i] for id_col in id_vars}
                row[var_name] = col; row[value_name] = self._cols[col][i]
                rows.append(row)
        return DataFrame(rows)

    def describe(self) -> 'DataFrame':
        def _is_num_col(col):
            return any(isinstance(v, (int, float)) and not _is_nan(v) for v in self._cols[col])
        numeric_cols = [c for c in self._columns if _is_num_col(c)]
        stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        result = {}
        for col in numeric_cols:
            d = sorted([float(v) for v in self._cols[col] if not _is_nan(v)])
            n = len(d)
            mu = sum(d) / n if n else _NAN
            std = math.sqrt(sum((v-mu)**2 for v in d) / max(n-1, 1)) if n > 1 else _NAN
            def _perc(q): pos = q*(n-1); lo = int(pos); hi = min(lo+1, n-1); return d[lo]+(pos-lo)*(d[hi]-d[lo]) if d else _NAN
            result[col] = [float(n), mu, std, d[0] if d else _NAN,
                           _perc(0.25), _perc(0.5), _perc(0.75), d[-1] if d else _NAN]
        return DataFrame(result, index=stats)

    def shift(self, periods: int = 1, fill_value=None) -> 'DataFrame':
        """Shift values by `periods` rows (like pandas DataFrame.shift)."""
        result = {}
        for col in self._columns:
            vals = self._cols[col]
            n = len(vals)
            nan = float('nan')
            fill = nan if fill_value is None else fill_value
            if periods > 0:
                result[col] = [fill] * min(periods, n) + list(vals[:max(0, n - periods)])
            elif periods < 0:
                p = -periods
                result[col] = list(vals[p:]) + [fill] * min(p, n)
            else:
                result[col] = list(vals)
        return DataFrame(result, index=list(self._index) if self._index else None)

    def diff(self, periods: int = 1) -> 'DataFrame':
        """First discrete difference of elements (like pandas DataFrame.diff)."""
        nan = float('nan')
        result = {}
        for col in self._columns:
            vals = self._cols[col]
            diffs = [nan] * periods
            for i in range(periods, len(vals)):
                try:
                    diffs.append(float(vals[i]) - float(vals[i - periods]))
                except (TypeError, ValueError):
                    diffs.append(nan)
            result[col] = diffs
        return DataFrame(result, index=list(self._index) if self._index else None)

    def rolling(self, window: int):
        """Return a RollingWindow object for rolling aggregations."""
        return _RollingWindow(self, window)

    def resample(self, rule: str):
        """Stub for time-series resampling; returns self (requires DatetimeIndex)."""
        return _ResampleProxy(self, rule)

    def corr(self) -> 'DataFrame':
        def _is_num_col(col):
            return any(isinstance(v, (int, float)) and not _is_nan(v) for v in self._cols[col])
        numeric_cols = [c for c in self._columns if _is_num_col(c)]
        n = len(self._index)
        mat = [[float(self._cols[c][i]) for c in numeric_cols] for i in range(n)]
        means = [sum(mat[i][j] for i in range(n)) / n for j in range(len(numeric_cols))]
        corr_mat = {}
        for j1, c1 in enumerate(numeric_cols):
            corr_mat[c1] = []
            for j2, c2 in enumerate(numeric_cols):
                num = sum((mat[i][j1]-means[j1])*(mat[i][j2]-means[j2]) for i in range(n))
                d1 = math.sqrt(sum((mat[i][j1]-means[j1])**2 for i in range(n)))
                d2 = math.sqrt(sum((mat[i][j2]-means[j2])**2 for i in range(n)))
                corr_mat[c1].append(num / (d1*d2 + 1e-10))
        return DataFrame(corr_mat, index=numeric_cols)

    def to_numpy(self):
        n = len(self._index)
        result = [[self._cols[c][i] for c in self._columns] for i in range(n)]
        try:
            import numpy as _np_df
            return _np_df.array(result)
        except ImportError:
            return result

    def to_dict(self, orient='dict'):
        if orient in ('dict', 'list'):
            return {col: self._cols[col][:] for col in self._columns}
        if orient == 'records':
            return [{col: self._cols[col][i] for col in self._columns} for i in range(len(self))]
        return {col: self._cols[col][:] for col in self._columns}

    def to_csv(self, path=None, index=True, sep=',') -> Optional[str]:
        buf = io.StringIO()
        writer = csv.writer(buf, delimiter=sep)
        header = (['index'] if index else []) + self._columns
        writer.writerow(header)
        for i, row_idx in enumerate(self._index):
            row = ([row_idx] if index else []) + [self._cols[col][i] for col in self._columns]
            writer.writerow(row)
        result = buf.getvalue()
        if path:
            with open(path, 'w', newline='') as f: f.write(result)
            return None
        return result

    def to_json(self, path=None, orient='records'):
        data = self.to_dict(orient='records')
        result = json.dumps(data, default=str)
        if path:
            with open(path, 'w') as f: f.write(result)
            return None
        return result

    @staticmethod
    def concat(dfs, axis=0, ignore_index=False) -> 'DataFrame':
        if not dfs: return DataFrame()
        if axis == 0:
            all_cols = []
            for df in dfs:
                for c in df._columns:
                    if c not in all_cols: all_cols.append(c)
            result = {}
            for col in all_cols:
                parts = []
                for df in dfs:
                    if col in df._columns:
                        parts.extend(df._cols[col])
                    else:
                        parts.extend([_NAN] * len(df))
                result[col] = parts
            if ignore_index:
                new_idx = list(range(sum(len(d) for d in dfs)))
            else:
                new_idx = []
                for df in dfs: new_idx.extend(df._index)
            return DataFrame(result, index=new_idx)
        new_cols = {}
        for df in dfs:
            for col in df._columns: new_cols[col] = df._cols[col]
        idx = dfs[0]._index if dfs else []
        return DataFrame(new_cols, index=idx)

    def value_counts(self, col: str, normalize: bool = False) -> 'Series':
        from collections import Counter
        from vectorphile.data.series import Series
        counts = Counter(self._cols[col])
        total = len(self._cols[col]) if normalize else 1
        idx = sorted(counts, key=lambda k: -counts[k])
        vals = [counts[k] / total for k in idx]
        return Series(vals, index=idx, name=col)

    def nunique(self) -> 'Series':
        from vectorphile.data.series import Series
        return Series([len(set(self._cols[c])) for c in self._columns], index=self._columns, name='nunique')

    def duplicated(self, subset=None, keep='first') -> 'Series':
        from vectorphile.data.series import Series
        cols = subset if subset is not None else self._columns
        seen = set()
        result = []
        for i in range(len(self._index)):
            key = tuple(self._cols[c][i] for c in cols)
            if keep == 'first':
                result.append(key in seen)
                seen.add(key)
            else:
                result.append(False)  # simplified
        return Series(result, index=list(self._index))

    def drop_duplicates(self, subset=None, keep='first') -> 'DataFrame':
        cols = subset if subset is not None else self._columns
        seen = set()
        keep_rows = []
        for i in range(len(self._index)):
            key = tuple(self._cols[c][i] for c in cols)
            if key not in seen:
                keep_rows.append(i)
                seen.add(key)
        new_cols = {c: [self._cols[c][i] for i in keep_rows] for c in self._columns}
        return DataFrame(new_cols, index=[self._index[i] for i in keep_rows])

    def clip(self, lower=None, upper=None) -> 'DataFrame':
        def _clip_val(v):
            if lower is not None and v < lower: return lower
            if upper is not None and v > upper: return upper
            return v
        new_cols = {}
        for c in self._columns:
            try:
                new_cols[c] = [_clip_val(float(v)) for v in self._cols[c]]
            except (TypeError, ValueError):
                new_cols[c] = list(self._cols[c])
        return DataFrame(new_cols, index=list(self._index))

    def abs(self) -> 'DataFrame':
        new_cols = {}
        for c in self._columns:
            try:
                new_cols[c] = [abs(float(v)) for v in self._cols[c]]
            except (TypeError, ValueError):
                new_cols[c] = list(self._cols[c])
        return DataFrame(new_cols, index=list(self._index))

    @property
    def T(self) -> 'DataFrame':
        return self.transpose()

    def transpose(self) -> 'DataFrame':
        rows = [[self._cols[c][i] for c in self._columns] for i in range(len(self._index))]
        new_cols = {str(idx): rows[i] for i, idx in enumerate(self._index)}
        return DataFrame(new_cols, index=list(self._columns))

    def nlargest(self, n: int, columns) -> 'DataFrame':
        col = columns if isinstance(columns, str) else columns[0]
        idx_sorted = sorted(range(len(self._index)), key=lambda i: float(self._cols[col][i]), reverse=True)[:n]
        new_cols = {c: [self._cols[c][i] for i in idx_sorted] for c in self._columns}
        return DataFrame(new_cols, index=[self._index[i] for i in idx_sorted])

    def nsmallest(self, n: int, columns) -> 'DataFrame':
        col = columns if isinstance(columns, str) else columns[0]
        idx_sorted = sorted(range(len(self._index)), key=lambda i: float(self._cols[col][i]))[:n]
        new_cols = {c: [self._cols[c][i] for i in idx_sorted] for c in self._columns}
        return DataFrame(new_cols, index=[self._index[i] for i in idx_sorted])

    def __repr__(self):
        max_rows = 10
        lines = []
        col_widths = {col: max(len(str(col)), max((len(str(v)) for v in self._cols[col][:max_rows]), default=0))
                      for col in self._columns}
        idx_width = max((len(str(i)) for i in self._index[:max_rows]), default=3)
        header = ' ' * idx_width + '  ' + '  '.join(str(c).ljust(col_widths[c]) for c in self._columns)
        lines.append(header)
        for i, row_idx in enumerate(self._index[:max_rows]):
            row = str(row_idx).ljust(idx_width) + '  ' + '  '.join(
                str(self._cols[col][i]).ljust(col_widths[col]) for col in self._columns)
            lines.append(row)
        if len(self._index) > max_rows:
            lines.append(f'... ({len(self._index)} rows × {len(self._columns)} columns)')
        return '\n'.join(lines)

    def __str__(self): return self.__repr__()
    def __contains__(self, key): return key in self._columns
    def __iter__(self): return iter(self._columns)


class _LocIndexer:
    def __init__(self, df: DataFrame):
        self._df = df

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key, col_key = key, slice(None)
        idx = self._df._index
        n = len(idx)
        if isinstance(row_key, slice):
            if row_key == slice(None):
                row_idxs = list(range(n))
            else:
                start = idx.index(row_key.start) if row_key.start is not None else 0
                stop = idx.index(row_key.stop) + 1 if row_key.stop is not None else n
                row_idxs = list(range(start, stop))
        elif isinstance(row_key, list):
            row_idxs = [idx.index(k) for k in row_key]
        else:
            pos = idx.index(row_key)
            row_idxs = [pos]
        new_idx = [idx[i] for i in row_idxs]
        if isinstance(col_key, slice) and col_key == slice(None):
            cols = self._df._columns
        elif isinstance(col_key, list):
            cols = col_key
        elif isinstance(col_key, str):
            return Series([self._df._cols[col_key][i] for i in row_idxs], index=new_idx, name=col_key)
        else:
            cols = self._df._columns
        return DataFrame({c: [self._df._cols[c][i] for i in row_idxs] for c in cols}, index=new_idx)


class _ILocIndexer:
    def __init__(self, df: DataFrame):
        self._df = df

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key, col_key = key, slice(None)
        n = len(self._df)
        if isinstance(row_key, slice):
            rows = list(range(n))[row_key]
        elif isinstance(row_key, int):
            rows = [row_key]
        elif isinstance(row_key, list):
            rows = row_key
        else:
            rows = [int(row_key)]
        if isinstance(col_key, slice):
            cols = self._df._columns[col_key]
        elif isinstance(col_key, int):
            col = self._df._columns[col_key]
            new_idx = [self._df._index[i] for i in rows]
            return Series([self._df._cols[col][i] for i in rows], index=new_idx, name=col)
        elif isinstance(col_key, list):
            cols = [self._df._columns[i] for i in col_key]
        else:
            cols = self._df._columns
        new_idx = [self._df._index[i] for i in rows]
        return DataFrame({c: [self._df._cols[c][i] for i in rows] for c in cols}, index=new_idx)


class _RollingWindow:
    """Rolling window aggregator returned by DataFrame.rolling(window)."""
    def __init__(self, df: DataFrame, window: int):
        self._df = df
        self._window = window

    def _apply(self, func):
        nan = float('nan')
        result = {}
        for col in self._df._columns:
            vals = self._df._cols[col]
            out = []
            for i in range(len(vals)):
                start = max(0, i - self._window + 1)
                win = []
                for v in vals[start:i + 1]:
                    try: win.append(float(v))
                    except (TypeError, ValueError): pass
                if len(win) < self._window:
                    out.append(nan)
                else:
                    out.append(func(win))
            result[col] = out
        return DataFrame(result, index=list(self._df._index) if self._df._index else None)

    def mean(self) -> DataFrame:
        return self._apply(lambda w: sum(w) / len(w))

    def sum(self) -> DataFrame:
        return self._apply(lambda w: sum(w))

    def min(self) -> DataFrame:
        return self._apply(lambda w: min(w))

    def max(self) -> DataFrame:
        return self._apply(lambda w: max(w))

    def std(self) -> DataFrame:
        import math as _m
        def _std(w):
            mu = sum(w) / len(w)
            return _m.sqrt(sum((v - mu) ** 2 for v in w) / max(len(w) - 1, 1))
        return self._apply(_std)


class _ResampleProxy:
    """Minimal time-series resampling stub — returns aggregated DataFrame."""
    def __init__(self, df: DataFrame, rule: str):
        self._df = df
        self._rule = rule

    def mean(self) -> DataFrame:
        return self._df  # stub: return as-is

    def sum(self) -> DataFrame:
        return self._df


def read_csv(path_or_buf, sep=',', header=0, index_col=None, dtype=None, na_values=None) -> DataFrame:
    if hasattr(path_or_buf, 'read'):
        content = path_or_buf.read()
    else:
        with open(path_or_buf, 'r', newline='', encoding='utf-8') as f:
            content = f.read()
    reader = csv.reader(io.StringIO(content), delimiter=sep)
    rows = list(reader)
    if not rows: return DataFrame()
    col_names = rows[0] if header == 0 else [str(i) for i in range(len(rows[0]))]
    data_rows = rows[1:] if header == 0 else rows
    na_set = set(na_values or ['', 'NA', 'NaN', 'null', 'None', 'nan'])
    result = {col: [] for col in col_names}
    for row in data_rows:
        for i, col in enumerate(col_names):
            val = row[i] if i < len(row) else ''
            if val in na_set:
                result[col].append(_NAN)
            else:
                try: result[col].append(int(val))
                except ValueError:
                    try: result[col].append(float(val))
                    except ValueError: result[col].append(val)
    df = DataFrame(result)
    if dtype: df = df.astype(dtype)
    if index_col is not None:
        idx_col = col_names[index_col] if isinstance(index_col, int) else index_col
        df = df.set_index(idx_col)
    return df


def read_json(path_or_buf) -> DataFrame:
    if hasattr(path_or_buf, 'read'):
        content = path_or_buf.read()
    else:
        with open(path_or_buf, 'r') as f:
            content = f.read()
    data = json.loads(content)
    if isinstance(data, list):
        return DataFrame(data)
    return DataFrame(data)
