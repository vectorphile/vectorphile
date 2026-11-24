"""Sparse tensor support with COO format and autograd-safe operations.

SparseTensor stores (indices, values) in COO (Coordinate) format.
Supports:
- to_dense() / to_sparse() conversion
- sparse_mm: sparse × dense matmul with grad flowing through dense
- sparse_embedding_lookup: index into an embedding table where only
  accessed rows receive gradient (sparse param update)
"""
from __future__ import annotations
import array as _arr
import math as _math
from typing import Dict, List, Optional, Tuple


class SparseTensor:
    """COO sparse tensor: stores non-zero values as (indices, values).

    Shape is (nrows, ncols) for 2-D (the common case for weight matrices
    and embedding tables).  Higher-rank sparse tensors store a flat
    linear index into the dense shape.

    Attributes:
        indices: list of (row, col) int tuples (COO format)
        values:  list of floats, one per index
        shape:   dense shape (nrows, ncols)
        requires_grad: whether this tensor participates in autograd
    """

    def __init__(self, indices: List[Tuple[int, ...]], values: List[float],
                 shape: Tuple[int, ...], requires_grad: bool = False):
        self.indices = list(indices)
        self.values = list(values)
        self.shape = tuple(shape)
        self.requires_grad = requires_grad
        self.grad: Optional['SparseTensor'] = None

    # ── Conversion ────────────────────────────────────────────────────────────

    def to_dense(self) -> 'Tensor':
        """Convert COO sparse tensor to a dense VP Tensor."""
        from vectorphile.core.tensor import Tensor
        from vectorphile._core.ndarray import NDArray as _NDA, _c_strides as _cs, _DType as _DT
        n = 1
        for s in self.shape: n *= s
        buf = _arr.array('f', [0.0] * n)
        strides = _cs(self.shape)
        for idx_tuple, val in zip(self.indices, self.values):
            flat = sum(idx_tuple[i] * strides[i] for i in range(len(idx_tuple)))
            if 0 <= flat < n:
                buf[flat] += val
        nd = _NDA(buf, self.shape, strides, _DT('float32'))
        return Tensor(nd, requires_grad=self.requires_grad)

    @classmethod
    def from_dense(cls, tensor: 'Tensor', threshold: float = 0.0,
                   requires_grad: bool = False) -> 'SparseTensor':
        """Convert a dense VP Tensor to COO SparseTensor."""
        from vectorphile._core.ndarray import _c_strides as _cs
        sh = tensor._data.shape if hasattr(tensor._data, 'shape') else (len(tensor._data),)
        flat = list(tensor._data.to_flat_list() if hasattr(tensor._data, 'to_flat_list') else tensor._data)
        strides = _cs(sh)
        indices, values = [], []
        for flat_i, v in enumerate(flat):
            if abs(v) > threshold:
                # Reconstruct multi-dim index from flat index
                idx = []
                rem = flat_i
                for d in range(len(sh)):
                    idx.append(rem // strides[d])
                    rem = rem % strides[d]
                indices.append(tuple(idx))
                values.append(float(v))
        return cls(indices, values, sh, requires_grad=requires_grad)

    def nnz(self) -> int:
        """Number of non-zero elements."""
        return len(self.values)

    def density(self) -> float:
        total = 1
        for s in self.shape: total *= s
        return self.nnz() / max(1, total)

    def __repr__(self) -> str:
        return (f'SparseTensor(shape={self.shape}, nnz={self.nnz()}, '
                f'density={self.density():.4f}, requires_grad={self.requires_grad})')

    # ── Arithmetic (COO + COO, COO * scalar) ─────────────────────────────────

    def _add_coo(self, other: 'SparseTensor') -> 'SparseTensor':
        assert self.shape == other.shape, "Shape mismatch in sparse add"
        # Merge by index
        idx_map: Dict[Tuple, float] = {}
        for idx, v in zip(self.indices, self.values):
            idx_map[tuple(idx)] = idx_map.get(tuple(idx), 0.0) + v
        for idx, v in zip(other.indices, other.values):
            idx_map[tuple(idx)] = idx_map.get(tuple(idx), 0.0) + v
        indices = list(idx_map.keys())
        values = [idx_map[k] for k in indices]
        return SparseTensor(indices, values, self.shape)

    def __add__(self, other: 'SparseTensor') -> 'SparseTensor':
        return self._add_coo(other)

    def __mul__(self, scalar: float) -> 'SparseTensor':
        return SparseTensor(list(self.indices), [v * scalar for v in self.values], self.shape)

    def __rmul__(self, scalar: float) -> 'SparseTensor':
        return self.__mul__(scalar)


# ── Sparse × dense matmul with autograd ──────────────────────────────────────

def sparse_mm(sparse: SparseTensor, dense: 'Tensor') -> 'Tensor':
    """Sparse-dense matrix multiplication: (M, K) × (K, N) → (M, N).

    Gradient flows through the dense matrix only.
    For sparse × dense:
      - d_loss/d_dense[k,n] += sparse[m,k] * upstream[m,n]  for each nnz (m,k)
    """
    from vectorphile.core.tensor import Tensor
    from vectorphile._core.ndarray import NDArray as _NDA, _c_strides as _cs, _DType as _DT

    assert len(sparse.shape) == 2, "sparse_mm requires 2-D sparse tensor"
    M, K = sparse.shape
    dense_sh = dense._data.shape if hasattr(dense._data, 'shape') else (len(dense._data),)
    assert dense_sh[0] == K, f"sparse_mm: K dim mismatch {K} vs {dense_sh[0]}"
    N = dense_sh[1] if len(dense_sh) > 1 else 1

    dense_flat = list(dense._data.to_flat_list() if hasattr(dense._data, 'to_flat_list') else dense._data)
    out = [0.0] * (M * N)

    for (m, k), sv in zip(sparse.indices, sparse.values):
        for n in range(N):
            out[m * N + n] += sv * dense_flat[k * N + n]

    out_sh = (M, N) if N > 1 else (M,)
    out_buf = _arr.array('f', [float(v) for v in out])
    out_nd = _NDA(out_buf, out_sh, _cs(out_sh), _DT('float32'))
    result = Tensor(out_nd, requires_grad=dense.requires_grad)

    if dense.requires_grad:
        from vectorphile.autograd.engine import Function

        class _SparseMmBackward(Function):
            def __init__(self):
                self.sparse = sparse
                self.dense_sh = dense_sh
                self.K = K
                self.N = N
                self.inputs = [dense]

            def backward(self, grad_out: Tensor):
                # d_dense[k,n] = sum_m sparse[m,k] * grad_out[m,n]
                from vectorphile._core.ndarray import NDArray as _BNDA, _c_strides as _bcs, _DType as _BDT
                g_flat = list(grad_out._data.to_flat_list() if hasattr(grad_out._data, 'to_flat_list') else grad_out._data)
                d_dense = [0.0] * (self.K * self.N)
                for (m, k), sv in zip(self.sparse.indices, self.sparse.values):
                    for n in range(self.N):
                        d_dense[k * self.N + n] += sv * g_flat[m * self.N + n]
                buf = _arr.array('f', [float(v) for v in d_dense])
                nd = _BNDA(buf, self.dense_sh, _bcs(self.dense_sh), _BDT('float32'))
                return [Tensor(nd)]

        result.grad_fn = _SparseMmBackward()
        result._is_leaf = False

    return result


# ── Sparse embedding lookup with sparse gradient ──────────────────────────────

def sparse_embedding_lookup(weight: 'Tensor', indices: List[int]) -> 'Tensor':
    """Look up rows of ``weight`` by ``indices``; gradient is sparse over weight.

    Equivalent to Embedding forward but accumulates gradient only for the
    rows that were accessed, leaving other rows at zero.  This matches
    PyTorch's sparse_grad=True behaviour.

    Args:
        weight: (vocab_size, embed_dim) Tensor
        indices: list of integer row indices to look up

    Returns:
        (len(indices), embed_dim) Tensor with requires_grad tracking weight.
    """
    from vectorphile.core.tensor import Tensor
    from vectorphile._core.ndarray import NDArray as _ENDA, _c_strides as _ecs, _DType as _EDT

    w_sh = weight._data.shape if hasattr(weight._data, 'shape') else (len(weight._data),)
    assert len(w_sh) == 2, "sparse_embedding_lookup: weight must be 2-D"
    vocab_size, embed_dim = w_sh
    w_flat = list(weight._data.to_flat_list() if hasattr(weight._data, 'to_flat_list') else weight._data)

    B = len(indices)
    out = []
    for idx in indices:
        idx = int(idx) % vocab_size
        out.extend(w_flat[idx * embed_dim: (idx + 1) * embed_dim])

    out_sh = (B, embed_dim)
    out_buf = _arr.array('f', [float(v) for v in out])
    out_nd = _ENDA(out_buf, out_sh, _ecs(out_sh), _EDT('float32'))
    result = Tensor(out_nd, requires_grad=weight.requires_grad)

    if weight.requires_grad:
        from vectorphile.autograd.engine import Function

        class _SparseEmbeddingBackward(Function):
            def __init__(self):
                self.indices = [int(i) % vocab_size for i in indices]
                self.vocab_size = vocab_size
                self.embed_dim = embed_dim
                self.inputs = [weight]

            def backward(self, grad_out: Tensor):
                from vectorphile._core.ndarray import NDArray as _GBDA, _c_strides as _gcs, _DType as _GDT
                g_flat = list(grad_out._data.to_flat_list() if hasattr(grad_out._data, 'to_flat_list') else grad_out._data)
                # Accumulate only for accessed rows
                d_weight = [0.0] * (self.vocab_size * self.embed_dim)
                for b, row_idx in enumerate(self.indices):
                    for d in range(self.embed_dim):
                        d_weight[row_idx * self.embed_dim + d] += g_flat[b * self.embed_dim + d]
                buf = _arr.array('f', [float(v) for v in d_weight])
                nd = _GBDA(buf, (self.vocab_size, self.embed_dim),
                           _gcs((self.vocab_size, self.embed_dim)), _GDT('float32'))
                return [Tensor(nd)]

        result.grad_fn = _SparseEmbeddingBackward()
        result._is_leaf = False

    return result


# ── Patch to_sparse / to_dense onto Tensor ───────────────────────────────────

def _tensor_to_sparse(self, threshold: float = 0.0) -> SparseTensor:
    """Convert this Tensor to COO SparseTensor."""
    return SparseTensor.from_dense(self, threshold=threshold,
                                   requires_grad=self.requires_grad)


def _tensor_to_dense(self):
    return self   # Dense tensor is already dense; no-op


try:
    from vectorphile.core.tensor import Tensor as _Tensor
    if not hasattr(_Tensor, 'to_sparse'):
        _Tensor.to_sparse = _tensor_to_sparse
except Exception:
    pass


__all__ = ['SparseTensor', 'sparse_mm', 'sparse_embedding_lookup']
