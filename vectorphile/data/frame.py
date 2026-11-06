"""DataFrame — 2D tabular data, pandas-compatible API."""
from __future__ import annotations
import numpy as np
import csv
import io
import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from vectorphile.data.series import Series


class DataFrame:
    def __init__(self, data=None, index=None, columns=None):
        if data is None:
            self._cols: Dict[str, np.ndarray] = {}
            self._index: List = []
            self._columns: List[str] = []
        elif isinstance(data, dict):
            self._cols = {}
            self._columns = list(columns) if columns is not None else list(data.keys())
            first_len = None
            for col in self._columns:
                val = data.get(col, [])
                arr = val.values if isinstance(val, Series) else np.array(val)
                self._cols[col] = arr
                if first_len is None:
                    first_len = len(arr)
            n = first_len or 0
            self._index = list(index) if index is not None else list(range(n))
        elif isinstance(data, list):
            if len(data) == 0:
                self._cols, self._columns, self._index = {}, [], []
            elif isinstance(data[0], dict):
                self._columns = list(columns) if columns is not None else list(data[0].keys())
                self._cols = {col: np.array([row.get(col, np.nan) for row in data]) for col in self._columns}
                self._index = list(index) if index is not None else list(range(len(data)))
            else:
                self._columns = list(columns) if columns is not None else [str(i) for i in range(len(data[0]) if hasattr(data[0], '__len__') else 1)]
                self._cols = {col: np.array([row[i] for row in data]) for i, col in enumerate(self._columns)}
                self._index = list(index) if index is not None else list(range(len(data)))
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            self._columns = list(columns) if columns is not None else [str(i) for i in range(data.shape[1])]
            self._cols = {col: data[:, i] for i, col in enumerate(self._columns)}
            self._index = list(index) if index is not None else list(range(data.shape[0]))
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
        return Series({col: str(arr.dtype) for col, arr in self._cols.items()})

    def __len__(self):
        return len(self._index)

    def __getitem__(self, key):
        if isinstance(key, str):
            return Series(self._cols[key], index=self._index, name=key)
        if isinstance(key, list):
            return DataFrame({k: self._cols[k] for k in key}, index=self._index)
        if isinstance(key, Series):
            mask = key.values
            new_idx = [self._index[i] for i in np.where(mask)[0]]
            return DataFrame({col: self._cols[col][mask] for col in self._columns}, index=new_idx)
        if isinstance(key, np.ndarray) and key.dtype == bool:
            new_idx = [self._index[i] for i in np.where(key)[0]]
            return DataFrame({col: self._cols[col][key] for col in self._columns}, index=new_idx)
        if isinstance(key, (int, slice)):
            if isinstance(key, int):
                rows = {col: np.array([self._cols[col][key]]) for col in self._columns}
                return DataFrame(rows, index=[self._index[key]])
            rows = {col: self._cols[col][key] for col in self._columns}
            return DataFrame(rows, index=self._index[key])
        raise KeyError(key)

    def __setitem__(self, key, value):
        if isinstance(value, Series):
            value = value.values
        elif not isinstance(value, np.ndarray):
            value = np.array(value)
        if key not in self._columns:
            self._columns.append(key)
        self._cols[key] = value

    def insert(self, loc: int, column: str, value):
        if isinstance(value, Series):
            value = value.values
        elif not isinstance(value, np.ndarray):
            value = np.array(value)
        self._cols[column] = value
        if column not in self._columns:
            self._columns.insert(loc, column)

    @property
    def loc(self):
        return _LocIndexer(self)

    @property
    def iloc(self):
        return _ILocIndexer(self)

    def head(self, n=5) -> 'DataFrame':
        return self[:n]

    def tail(self, n=5) -> 'DataFrame':
        return self[-n:]

    def copy(self) -> 'DataFrame':
        return DataFrame({col: self._cols[col].copy() for col in self._columns}, index=self._index.copy())

    def isnull(self) -> 'DataFrame':
        result = {}
        for col in self._columns:
            try:
                result[col] = np.isnan(self._cols[col].astype(float))
            except (ValueError, TypeError):
                result[col] = np.array([v is None for v in self._cols[col]])
        return DataFrame(result, index=self._index)

    def notnull(self) -> 'DataFrame':
        null = self.isnull()
        return DataFrame({col: ~null._cols[col] for col in self._columns}, index=self._index)

    def dropna(self, axis=0, how='any', subset=None) -> 'DataFrame':
        if axis == 0:
            cols = subset if subset else self._columns
            null_df = self.isnull()
            if how == 'any':
                mask = ~np.any(np.stack([null_df._cols[c] for c in cols], axis=1), axis=1)
            else:
                mask = ~np.all(np.stack([null_df._cols[c] for c in cols], axis=1), axis=1)
            new_idx = [self._index[i] for i in np.where(mask)[0]]
            return DataFrame({col: self._cols[col][mask] for col in self._columns}, index=new_idx)
        result = {}
        for col in self._columns:
            null = self.isnull()[col]
            if (how == 'any' and not null.values.any()) or (how == 'all' and not null.values.all()):
                result[col] = self._cols[col]
        return DataFrame(result, index=self._index)

    def fillna(self, value) -> 'DataFrame':
        result = {}
        for col in self._columns:
            try:
                arr = self._cols[col].astype(float)
                mask = np.isnan(arr)
                arr[mask] = value
                result[col] = arr
            except (ValueError, TypeError):
                result[col] = np.array([value if v is None else v for v in self._cols[col]])
        return DataFrame(result, index=self._index)

    def astype(self, dtype) -> 'DataFrame':
        if isinstance(dtype, dict):
            result = {col: self._cols[col].astype(dtype[col]) if col in dtype else self._cols[col] for col in self._columns}
        else:
            result = {col: self._cols[col].astype(dtype) for col in self._columns}
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
            return DataFrame({col: self._cols[col] for col in self._columns}, index=new_idx)
        return self.copy()

    def drop(self, labels=None, axis=0, columns=None) -> 'DataFrame':
        if columns is not None or axis == 1:
            drop_cols = set(columns or (labels if isinstance(labels, list) else [labels]))
            keep = [c for c in self._columns if c not in drop_cols]
            return DataFrame({c: self._cols[c] for c in keep}, index=self._index)
        if axis == 0:
            if not isinstance(labels, list):
                labels = [labels]
            mask = np.array([i not in labels for i in self._index])
            new_idx = [self._index[i] for i in np.where(mask)[0]]
            return DataFrame({col: self._cols[col][mask] for col in self._columns}, index=new_idx)
        return self.copy()

    def sort_values(self, by, ascending=True) -> 'DataFrame':
        if isinstance(by, str):
            by = [by]
        keys = tuple(self._cols[col] for col in reversed(by))
        order = np.lexsort(keys)
        if not ascending:
            order = order[::-1]
        new_idx = [self._index[i] for i in order]
        return DataFrame({col: self._cols[col][order] for col in self._columns}, index=new_idx)

    def reset_index(self, drop=False) -> 'DataFrame':
        df = DataFrame({col: self._cols[col].copy() for col in self._columns},
                       index=list(range(len(self._index))))
        if not drop:
            df.insert(0, 'index', np.array(self._index))
        return df

    def set_index(self, col: str) -> 'DataFrame':
        new_idx = self._cols[col].tolist()
        keep = [c for c in self._columns if c != col]
        return DataFrame({c: self._cols[c] for c in keep}, index=new_idx)

    def groupby(self, by):
        from vectorphile.data.groupby import GroupBy
        return GroupBy(self, by)

    def merge(self, right: 'DataFrame', how='inner', on=None, left_on=None, right_on=None) -> 'DataFrame':
        lkey = on if on else left_on
        rkey = on if on else right_on
        if isinstance(lkey, str):
            lkey, rkey = [lkey], [rkey or lkey]
        lvals = list(zip(*[self._cols[k] for k in lkey]))
        rvals = list(zip(*[right._cols[k] for k in rkey]))
        rmap = {}
        for i, rv in enumerate(rvals):
            rmap.setdefault(rv, []).append(i)
        rows_l, rows_r = [], []
        for i, lv in enumerate(lvals):
            if lv in rmap:
                for j in rmap[lv]:
                    rows_l.append(i)
                    rows_r.append(j)
            elif how in ('left', 'outer'):
                rows_l.append(i)
                rows_r.append(None)
        if how in ('right', 'outer'):
            lset = set(map(tuple, lvals))
            for j, rv in enumerate(rvals):
                if rv not in lset:
                    rows_l.append(None)
                    rows_r.append(j)
        result = {}
        for col in self._columns:
            result[col] = np.array([self._cols[col][i] if i is not None else np.nan for i in rows_l])
        for col in right._columns:
            if col not in lkey:
                result[col] = np.array([right._cols[col][j] if j is not None else np.nan for j in rows_r])
        return DataFrame(result)

    def apply(self, func: Callable, axis=0) -> 'DataFrame':
        if axis == 1:
            results = []
            for i in range(len(self)):
                row = {col: self._cols[col][i] for col in self._columns}
                results.append(func(Series(list(row.values()), index=list(row.keys()))))
            return Series(results, index=self._index)
        result = {}
        for col in self._columns:
            result[col] = func(Series(self._cols[col], index=self._index, name=col))
        return Series(list(result.values()), index=list(result.keys()))

    def pivot(self, index, columns, values) -> 'DataFrame':
        idx_vals = np.unique(self._cols[index])
        col_vals = np.unique(self._cols[columns])
        result = {str(cv): [] for cv in col_vals}
        new_index = []
        for iv in idx_vals:
            new_index.append(iv)
            row_mask = self._cols[index] == iv
            for cv in col_vals:
                col_mask = self._cols[columns] == cv
                mask = row_mask & col_mask
                vals = self._cols[values][mask]
                result[str(cv)].append(float(vals[0]) if len(vals) > 0 else np.nan)
        return DataFrame({k: np.array(v) for k, v in result.items()}, index=list(new_index))

    def melt(self, id_vars=None, value_vars=None, var_name='variable', value_name='value') -> 'DataFrame':
        id_vars = id_vars or []
        value_vars = value_vars or [c for c in self._columns if c not in id_vars]
        rows = []
        for i in range(len(self)):
            for col in value_vars:
                row = {id_col: self._cols[id_col][i] for id_col in id_vars}
                row[var_name] = col
                row[value_name] = self._cols[col][i]
                rows.append(row)
        return DataFrame(rows)

    def describe(self) -> 'DataFrame':
        numeric_cols = [c for c in self._columns if np.issubdtype(self._cols[c].dtype, np.number)]
        stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        result = {}
        for col in numeric_cols:
            d = self._cols[col].astype(float)
            d_clean = d[~np.isnan(d)]
            result[col] = [
                float(len(d_clean)),
                float(np.mean(d_clean)),
                float(np.std(d_clean, ddof=1)) if len(d_clean) > 1 else np.nan,
                float(np.min(d_clean)),
                float(np.percentile(d_clean, 25)),
                float(np.percentile(d_clean, 50)),
                float(np.percentile(d_clean, 75)),
                float(np.max(d_clean)),
            ]
        return DataFrame(result, index=stats)

    def corr(self) -> 'DataFrame':
        numeric_cols = [c for c in self._columns if np.issubdtype(self._cols[c].dtype, np.number)]
        mat = np.stack([self._cols[c].astype(float) for c in numeric_cols], axis=1)
        corr_mat = np.corrcoef(mat.T)
        return DataFrame({c: corr_mat[i] for i, c in enumerate(numeric_cols)}, index=numeric_cols)

    def cov(self) -> 'DataFrame':
        numeric_cols = [c for c in self._columns if np.issubdtype(self._cols[c].dtype, np.number)]
        mat = np.stack([self._cols[c].astype(float) for c in numeric_cols], axis=1)
        cov_mat = np.cov(mat.T)
        return DataFrame({c: cov_mat[i] for i, c in enumerate(numeric_cols)}, index=numeric_cols)

    def to_numpy(self) -> np.ndarray:
        return np.stack([self._cols[c] for c in self._columns], axis=1)

    def to_dict(self, orient='dict'):
        if orient == 'dict':
            return {col: self._cols[col].tolist() for col in self._columns}
        if orient == 'records':
            return [{col: self._cols[col][i] for col in self._columns} for i in range(len(self))]
        if orient == 'list':
            return {col: self._cols[col].tolist() for col in self._columns}
        return {col: self._cols[col].tolist() for col in self._columns}

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
            with open(path, 'w', newline='') as f:
                f.write(result)
            return None
        return result

    def to_json(self, path=None, orient='records'):
        data = self.to_dict(orient='records')
        result = json.dumps(data, default=str)
        if path:
            with open(path, 'w') as f:
                f.write(result)
            return None
        return result

    @staticmethod
    def concat(dfs, axis=0, ignore_index=False) -> 'DataFrame':
        if axis == 0:
            all_cols = []
            for df in dfs:
                for c in df._columns:
                    if c not in all_cols:
                        all_cols.append(c)
            result = {}
            for col in all_cols:
                parts = []
                for df in dfs:
                    if col in df._columns:
                        parts.append(df._cols[col])
                    else:
                        parts.append(np.full(len(df), np.nan))
                result[col] = np.concatenate(parts)
            if ignore_index:
                new_idx = list(range(sum(len(d) for d in dfs)))
            else:
                new_idx = []
                for df in dfs:
                    new_idx.extend(df._index)
            return DataFrame(result, index=new_idx)
        # axis=1
        new_cols = {}
        for df in dfs:
            for col in df._columns:
                new_cols[col] = df._cols[col]
        idx = dfs[0]._index if dfs else []
        return DataFrame(new_cols, index=idx)

    def __repr__(self):
        max_rows = 10
        lines = []
        col_widths = {col: max(len(str(col)), max((len(str(v)) for v in self._cols[col][:max_rows]), default=0)) for col in self._columns}
        idx_width = max((len(str(i)) for i in self._index[:max_rows]), default=3)
        header = ' ' * idx_width + '  ' + '  '.join(str(c).ljust(col_widths[c]) for c in self._columns)
        lines.append(header)
        for i, row_idx in enumerate(self._index[:max_rows]):
            row = str(row_idx).ljust(idx_width) + '  ' + '  '.join(str(self._cols[col][i]).ljust(col_widths[col]) for col in self._columns)
            lines.append(row)
        if len(self._index) > max_rows:
            lines.append(f'... ({len(self._index)} rows × {len(self._columns)} columns)')
        return '\n'.join(lines)

    def __str__(self):
        return self.__repr__()

    def __contains__(self, key):
        return key in self._columns

    def __iter__(self):
        return iter(self._columns)


class _LocIndexer:
    def __init__(self, df: DataFrame):
        self._df = df

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key, col_key = key, slice(None)
        # row selection
        idx = self._df._index
        if isinstance(row_key, slice):
            if row_key == slice(None):
                row_mask = np.ones(len(idx), dtype=bool)
            else:
                start = idx.index(row_key.start) if row_key.start is not None else 0
                stop = idx.index(row_key.stop) + 1 if row_key.stop is not None else len(idx)
                row_mask = np.zeros(len(idx), dtype=bool)
                row_mask[start:stop] = True
        elif isinstance(row_key, list):
            positions = [idx.index(k) for k in row_key]
            row_mask = np.zeros(len(idx), dtype=bool)
            row_mask[positions] = True
        else:
            pos = idx.index(row_key)
            row_mask = np.zeros(len(idx), dtype=bool)
            row_mask[pos] = True
        # col selection
        if isinstance(col_key, slice) and col_key == slice(None):
            cols = self._df._columns
        elif isinstance(col_key, list):
            cols = col_key
        elif isinstance(col_key, str):
            return Series(self._df._cols[col_key][row_mask], index=[idx[i] for i in np.where(row_mask)[0]], name=col_key)
        else:
            cols = self._df._columns
        new_idx = [idx[i] for i in np.where(row_mask)[0]]
        return DataFrame({c: self._df._cols[c][row_mask] for c in cols}, index=new_idx)


class _ILocIndexer:
    def __init__(self, df: DataFrame):
        self._df = df

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key
        else:
            row_key, col_key = key, slice(None)
        rows = np.arange(len(self._df))[row_key] if isinstance(row_key, slice) else (np.array([row_key]) if isinstance(row_key, int) else np.array(row_key))
        if isinstance(col_key, slice):
            cols = self._df._columns[col_key]
        elif isinstance(col_key, int):
            col = self._df._columns[col_key]
            new_idx = [self._df._index[i] for i in rows]
            return Series(self._df._cols[col][rows], index=new_idx, name=col)
        elif isinstance(col_key, list):
            cols = [self._df._columns[i] for i in col_key]
        else:
            cols = self._df._columns
        new_idx = [self._df._index[i] for i in rows]
        return DataFrame({c: self._df._cols[c][rows] for c in cols}, index=new_idx)


def read_csv(path_or_buf, sep=',', header=0, index_col=None, dtype=None, na_values=None) -> DataFrame:
    if hasattr(path_or_buf, 'read'):
        content = path_or_buf.read()
    else:
        with open(path_or_buf, 'r', newline='', encoding='utf-8') as f:
            content = f.read()
    reader = csv.reader(io.StringIO(content), delimiter=sep)
    rows = list(reader)
    if not rows:
        return DataFrame()
    col_names = rows[0] if header == 0 else [str(i) for i in range(len(rows[0]))]
    data_rows = rows[1:] if header == 0 else rows
    na_set = set(na_values or ['', 'NA', 'NaN', 'null', 'None', 'nan'])
    result = {col: [] for col in col_names}
    for row in data_rows:
        for i, col in enumerate(col_names):
            val = row[i] if i < len(row) else ''
            if val in na_set:
                result[col].append(np.nan)
            else:
                try:
                    result[col].append(int(val))
                except ValueError:
                    try:
                        result[col].append(float(val))
                    except ValueError:
                        result[col].append(val)
    df = DataFrame({col: np.array(vals) for col, vals in result.items()})
    if dtype:
        df = df.astype(dtype)
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
