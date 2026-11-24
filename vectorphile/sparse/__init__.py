"""Sparse matrix support — pure Python."""
from __future__ import annotations
import math
from typing import List, Optional, Tuple
from vectorphile.core.tensor import Tensor


def _flat2d(x) -> Tuple[List[float], Tuple[int,int]]:
    if hasattr(x, 'shape') and len(x.shape) == 2:
        sh = x.shape
    else:
        sh = None
    if hasattr(x, 'to_flat_list'):
        f = [float(v) for v in x.to_flat_list()]
    elif hasattr(x, 'tolist'):
        v = x.tolist()
        if isinstance(v, list) and v and isinstance(v[0], list):
            sh = sh or (len(v), len(v[0]))
            f = [float(item) for row in v for item in row]
        else:
            f = [float(item) for item in v]
    elif isinstance(x, list):
        if x and isinstance(x[0], list):
            sh = sh or (len(x), len(x[0]))
            f = [float(item) for row in x for item in row]
        else:
            f = [float(item) for item in x]
    else:
        f = [float(x)]
    if sh is None:
        n = len(f)
        s = int(math.sqrt(n))
        sh = (s, s)
    return f, sh


class csr_matrix:
    """Compressed Sparse Row matrix."""

    def __init__(self, data, shape=None):
        if isinstance(data, Tensor):
            self._from_flat(*_flat2d(data._data))
        elif isinstance(data, list):
            self._from_flat(*_flat2d(data))
        elif isinstance(data, tuple) and len(data) == 2:
            (vals, (row, col)), shape = data, shape
            self.data = [float(v) for v in vals]
            self.indices = [int(c) for c in col]
            self.shape = shape
            self.indptr = [0] * (shape[0] + 1)
            for r in row:
                self.indptr[r + 1] += 1
            for i in range(1, len(self.indptr)):
                self.indptr[i] += self.indptr[i-1]
        elif hasattr(data, 'shape'):
            self._from_flat(*_flat2d(data))
        else:
            raise ValueError("Unsupported input format")

    def _from_flat(self, flat: List[float], sh: Tuple[int,int]):
        self.shape = sh
        rows_found = []; cols_found = []; vals = []
        for r in range(sh[0]):
            for c in range(sh[1]):
                v = flat[r*sh[1]+c]
                if v != 0.0:
                    rows_found.append(r)
                    cols_found.append(c)
                    vals.append(v)
        self.data = vals
        self.indices = cols_found
        self.indptr = [0] * (sh[0] + 1)
        for r in rows_found:
            self.indptr[r + 1] += 1
        for i in range(1, len(self.indptr)):
            self.indptr[i] += self.indptr[i-1]

    def tolist(self) -> List[List[float]]:
        result = [[0.0]*self.shape[1] for _ in range(self.shape[0])]
        for i in range(self.shape[0]):
            start, end = self.indptr[i], self.indptr[i+1]
            for k in range(start, end):
                result[i][self.indices[k]] = self.data[k]
        return result

    def to_tensor(self) -> Tensor:
        from vectorphile._core.ndarray import array as _nd
        flat = [v for row in self.tolist() for v in row]
        nd = _nd(flat)
        if hasattr(nd, 'reshape'):
            nd = nd.reshape(self.shape)
        return Tensor(nd)

    def __matmul__(self, other):
        a = self.tolist()
        if isinstance(other, csr_matrix):
            b = other.tolist()
        elif isinstance(other, Tensor):
            b, _ = _flat2d(other._data)
            nb_rows = other.shape[0] if hasattr(other, 'shape') else self.shape[1]
            nb_cols = other.shape[1] if hasattr(other, 'shape') and len(other.shape) > 1 else 1
            b = [[b[r*nb_cols+c] for c in range(nb_cols)] for r in range(nb_rows)]
        else:
            b, bs = _flat2d(other)
            b = [[b[r*bs[1]+c] for c in range(bs[1])] for r in range(bs[0])]
        m = len(a); k = len(a[0]); n = len(b[0]) if b else 0
        out = [[sum(a[i][t]*b[t][j] for t in range(k)) for j in range(n)] for i in range(m)]
        return csr_matrix(out)

    @property
    def nnz(self) -> int:
        return len(self.data)

    def __repr__(self):
        return f'csr_matrix(shape={self.shape}, nnz={self.nnz})'


class csc_matrix:
    """Compressed Sparse Column matrix."""

    def __init__(self, data):
        if isinstance(data, Tensor):
            flat, sh = _flat2d(data._data)
        elif isinstance(data, list):
            flat, sh = _flat2d(data)
        elif hasattr(data, 'shape'):
            flat, sh = _flat2d(data)
        else:
            flat, sh = _flat2d(data)
        self.shape = sh
        cols_found = []; rows_found = []; vals = []
        for c in range(sh[1]):
            for r in range(sh[0]):
                v = flat[r*sh[1]+c]
                if v != 0.0:
                    cols_found.append(c)
                    rows_found.append(r)
                    vals.append(v)
        self.data = vals
        self.indices = rows_found
        self.indptr = [0] * (sh[1] + 1)
        for c in cols_found:
            self.indptr[c + 1] += 1
        for i in range(1, len(self.indptr)):
            self.indptr[i] += self.indptr[i-1]

    def tolist(self) -> List[List[float]]:
        result = [[0.0]*self.shape[1] for _ in range(self.shape[0])]
        for j in range(self.shape[1]):
            start, end = self.indptr[j], self.indptr[j+1]
            for k in range(start, end):
                result[self.indices[k]][j] = self.data[k]
        return result

    def to_tensor(self) -> Tensor:
        from vectorphile._core.ndarray import array as _nd
        flat = [v for row in self.tolist() for v in row]
        nd = _nd(flat)
        if hasattr(nd, 'reshape'):
            nd = nd.reshape(self.shape)
        return Tensor(nd)

    @property
    def nnz(self) -> int:
        return len(self.data)


class lil_matrix:
    """List of Lists (LIL) sparse matrix — efficient for incremental construction."""

    def __init__(self, shape_or_data, shape=None, dtype=float):
        if isinstance(shape_or_data, (list, tuple)) and len(shape_or_data) == 2 and isinstance(shape_or_data[0], int):
            self.shape = tuple(shape_or_data)
            n, m = self.shape
            self.rows: List[List[int]] = [[] for _ in range(n)]
            self.data_: List[List[float]] = [[] for _ in range(n)]
        elif shape is not None:
            self.shape = tuple(shape)
            n, m = self.shape
            self.rows = [[] for _ in range(n)]
            self.data_ = [[] for _ in range(n)]
        else:
            flat2, sh2 = _flat2d(shape_or_data)
            self.shape = sh2
            n, m = sh2
            self.rows = [[] for _ in range(n)]
            self.data_ = [[] for _ in range(n)]
            for r in range(n):
                for c in range(m):
                    v = flat2[r * m + c]
                    if v != 0.0:
                        self.rows[r].append(c)
                        self.data_[r].append(float(v))

    def __setitem__(self, key, val):
        r, c = key
        if c in self.rows[r]:
            idx = self.rows[r].index(c)
            if val == 0.0:
                self.rows[r].pop(idx)
                self.data_[r].pop(idx)
            else:
                self.data_[r][idx] = float(val)
        elif val != 0.0:
            self.rows[r].append(c)
            self.data_[r].append(float(val))

    def __getitem__(self, key):
        r, c = key
        if c in self.rows[r]:
            return self.data_[r][self.rows[r].index(c)]
        return 0.0

    def tolist(self) -> List[List[float]]:
        n, m = self.shape
        result = [[0.0] * m for _ in range(n)]
        for r in range(n):
            for c, v in zip(self.rows[r], self.data_[r]):
                result[r][c] = v
        return result

    def tocsr(self) -> 'csr_matrix':
        return csr_matrix(self.tolist())

    @property
    def nnz(self) -> int:
        return sum(len(row) for row in self.rows)


class dok_matrix:
    """Dictionary of Keys (DOK) sparse matrix."""

    def __init__(self, shape_or_data, shape=None, dtype=float):
        if isinstance(shape_or_data, (list, tuple)) and len(shape_or_data) == 2 and isinstance(shape_or_data[0], int):
            self.shape = tuple(shape_or_data)
            self._dict: dict = {}
        elif shape is not None:
            self.shape = tuple(shape)
            self._dict = {}
        else:
            flat, sh = _flat2d(shape_or_data)
            self.shape = sh
            self._dict = {}
            for r in range(sh[0]):
                for c in range(sh[1]):
                    v = flat[r * sh[1] + c]
                    if v != 0.0:
                        self._dict[(r, c)] = float(v)

    def __setitem__(self, key, val):
        if val == 0.0:
            self._dict.pop(key, None)
        else:
            self._dict[key] = float(val)

    def __getitem__(self, key):
        return self._dict.get(key, 0.0)

    def tolist(self) -> List[List[float]]:
        n, m = self.shape
        result = [[0.0] * m for _ in range(n)]
        for (r, c), v in self._dict.items():
            result[r][c] = v
        return result

    def tocsr(self) -> 'csr_matrix':
        return csr_matrix(self.tolist())

    @property
    def nnz(self) -> int:
        return len(self._dict)


class coo_matrix:
    """Coordinate (COO) sparse matrix."""

    def __init__(self, data, shape=None):
        if isinstance(data, tuple) and len(data) == 2:
            (vals, (rows, cols)), shape = data, shape
            self.data = [float(v) for v in vals]
            self.row = [int(r) for r in rows]
            self.col = [int(c) for c in cols]
            self.shape = shape
        else:
            flat, sh = _flat2d(data)
            self.shape = sh
            self.row = []; self.col = []; self.data = []
            for r in range(sh[0]):
                for c in range(sh[1]):
                    v = flat[r*sh[1]+c]
                    if v != 0.0:
                        self.row.append(r)
                        self.col.append(c)
                        self.data.append(v)

    def tolist(self) -> List[List[float]]:
        result = [[0.0]*self.shape[1] for _ in range(self.shape[0])]
        for r, c, v in zip(self.row, self.col, self.data):
            result[r][c] = v
        return result

    def tocsr(self) -> csr_matrix:
        return csr_matrix(self.tolist())

    def to_tensor(self) -> Tensor:
        from vectorphile._core.ndarray import array as _nd
        flat = [v for row in self.tolist() for v in row]
        nd = _nd(flat)
        if hasattr(nd, 'reshape'):
            nd = nd.reshape(self.shape)
        return Tensor(nd)

    @property
    def nnz(self) -> int:
        return len(self.data)


def coo_tensor(indices, values, size, dtype=None):
    """Create a sparse COO tensor (like torch.sparse_coo_tensor)."""
    return SparseCOOTensor(indices, values, size, dtype=dtype)


class SparseCOOTensor:
    """N-dimensional sparse COO tensor."""

    def __init__(self, indices, values, size, dtype=None):
        if hasattr(indices, "tolist"):
            idx = indices.tolist()
        elif hasattr(indices, "_data"):
            idx = indices._data.tolist()
        else:
            idx = [list(r) for r in indices]
        if hasattr(values, "tolist"):
            vals = values.tolist()
        elif hasattr(values, "_data"):
            vals = values._data.tolist()
        else:
            vals = list(values)
        self.indices_data = idx
        self.values_data = vals
        self.size = tuple(size)
        self.dtype = dtype or "float32"
        self.is_sparse = True

    def to_dense(self):
        from vectorphile._core.ndarray import array as _nd
        total = 1
        for s in self.size:
            total *= s
        flat = [0.0] * total
        ndim = len(self.indices_data)
        nnz = len(self.values_data)
        for i in range(nnz):
            coords = [self.indices_data[d][i] for d in range(ndim)]
            lin = 0
            for d, c in enumerate(coords):
                stride = 1
                for d2 in range(d + 1, ndim):
                    stride *= self.size[d2]
                lin += c * stride
            flat[lin] = float(self.values_data[i])
        nd = _nd(flat, dtype=self.dtype)
        if len(self.size) > 1:
            nd = nd.reshape(self.size)
        from vectorphile.core.tensor import Tensor
        return Tensor(nd)

    def coalesce(self):
        from collections import defaultdict
        ndim = len(self.indices_data)
        nnz = len(self.values_data)
        accum = defaultdict(float)
        for i in range(nnz):
            key = tuple(self.indices_data[d][i] for d in range(ndim))
            accum[key] += float(self.values_data[i])
        new_idx = [[] for _ in range(ndim)]
        new_vals = []
        for key, val in accum.items():
            for d, c in enumerate(key):
                new_idx[d].append(c)
            new_vals.append(val)
        return SparseCOOTensor(new_idx, new_vals, self.size, dtype=self.dtype)

    @property
    def nnz(self):
        return len(self.values_data)

    @property
    def shape(self):
        return self.size

    def __repr__(self):
        return f"SparseCOOTensor(size={self.size}, nnz={self.nnz}, dtype={self.dtype})"
