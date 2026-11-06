"""Sparse matrix support."""
import numpy as np
from vectorphile.core.tensor import Tensor


class csr_matrix:
    """Compressed Sparse Row matrix."""
    def __init__(self, data, shape=None):
        if isinstance(data, np.ndarray):
            self._from_dense(data)
        elif isinstance(data, Tensor):
            self._from_dense(data._data)
        elif isinstance(data, tuple) and len(data) == 3:
            (vals, (row, col)), shape = data, shape
            self.data = np.array(vals)
            self.indices = np.array(col, dtype=int)
            self.indptr = np.zeros(shape[0] + 1, dtype=int)
            for r in row:
                self.indptr[r + 1:] += 1
            self.shape = shape
        else:
            raise ValueError("Unsupported input format")

    def _from_dense(self, arr):
        self.shape = arr.shape
        rows, cols = np.nonzero(arr)
        self.data = arr[rows, cols]
        self.indices = cols
        self.indptr = np.zeros(arr.shape[0] + 1, dtype=int)
        for r in rows:
            self.indptr[r + 1] += 1
        self.indptr = np.cumsum(self.indptr)

    def toarray(self) -> np.ndarray:
        result = np.zeros(self.shape)
        for i in range(self.shape[0]):
            start, end = self.indptr[i], self.indptr[i+1]
            result[i, self.indices[start:end]] = self.data[start:end]
        return result

    def to_tensor(self) -> Tensor:
        return Tensor(self.toarray().astype(np.float32))

    def __matmul__(self, other):
        a = self.toarray()
        if isinstance(other, csr_matrix):
            return csr_matrix(a @ other.toarray())
        if isinstance(other, np.ndarray):
            return a @ other
        if isinstance(other, Tensor):
            return Tensor((a @ other._data).astype(np.float32))
        return NotImplemented

    @property
    def nnz(self) -> int:
        return len(self.data)

    def __repr__(self):
        return f'csr_matrix(shape={self.shape}, nnz={self.nnz})'


class csc_matrix:
    """Compressed Sparse Column matrix."""
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            arr = data
        elif isinstance(data, Tensor):
            arr = data._data
        else:
            arr = np.array(data)
        self.shape = arr.shape
        rows, cols = np.nonzero(arr)
        order = np.argsort(cols)
        self.data = arr[rows[order], cols[order]]
        self.indices = rows[order]
        self.indptr = np.zeros(arr.shape[1] + 1, dtype=int)
        for c in cols[order]:
            self.indptr[c + 1] += 1
        self.indptr = np.cumsum(self.indptr)

    def toarray(self) -> np.ndarray:
        result = np.zeros(self.shape)
        for j in range(self.shape[1]):
            start, end = self.indptr[j], self.indptr[j+1]
            result[self.indices[start:end], j] = self.data[start:end]
        return result

    def to_tensor(self) -> Tensor:
        return Tensor(self.toarray().astype(np.float32))

    @property
    def nnz(self) -> int:
        return len(self.data)


class coo_matrix:
    """Coordinate (COO) sparse matrix."""
    def __init__(self, data):
        if isinstance(data, tuple) and len(data) == 2:
            (vals, (rows, cols)), shape = data
            self.data = np.array(vals)
            self.row = np.array(rows, dtype=int)
            self.col = np.array(cols, dtype=int)
            self.shape = shape
        elif isinstance(data, np.ndarray):
            arr = data
            self.shape = arr.shape
            self.row, self.col = np.nonzero(arr)
            self.data = arr[self.row, self.col]
        else:
            arr = np.array(data)
            self.shape = arr.shape
            self.row, self.col = np.nonzero(arr)
            self.data = arr[self.row, self.col]

    def toarray(self) -> np.ndarray:
        result = np.zeros(self.shape)
        result[self.row, self.col] = self.data
        return result

    def tocsr(self) -> csr_matrix:
        return csr_matrix(self.toarray())

    def to_tensor(self) -> Tensor:
        return Tensor(self.toarray().astype(np.float32))

    @property
    def nnz(self) -> int:
        return len(self.data)
