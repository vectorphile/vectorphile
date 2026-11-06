"""Linear algebra operations."""
import numpy as np
from vectorphile.core.tensor import Tensor


def det(a: Tensor) -> Tensor:
    return Tensor(np.linalg.det(a._data))

def inv(a: Tensor) -> Tensor:
    return Tensor(np.linalg.inv(a._data))

def svd(a: Tensor, full_matrices=True):
    U, S, Vh = np.linalg.svd(a._data, full_matrices=full_matrices)
    return Tensor(U), Tensor(S), Tensor(Vh)

def eig(a: Tensor):
    vals, vecs = np.linalg.eig(a._data)
    return Tensor(vals.real.astype(np.float32)), Tensor(vecs.real.astype(np.float32))

def eigh(a: Tensor):
    vals, vecs = np.linalg.eigh(a._data)
    return Tensor(vals.astype(np.float32)), Tensor(vecs.astype(np.float32))

def lstsq(a: Tensor, b: Tensor):
    result = np.linalg.lstsq(a._data, b._data, rcond=None)
    return Tensor(result[0])

def solve(a: Tensor, b: Tensor) -> Tensor:
    return Tensor(np.linalg.solve(a._data, b._data))

def norm(a: Tensor, ord=None, dim=None, keepdim=False) -> Tensor:
    return Tensor(np.linalg.norm(a._data, ord=ord, axis=dim, keepdims=keepdim))

def matrix_rank(a: Tensor, tol=None) -> Tensor:
    return Tensor(np.linalg.matrix_rank(a._data, tol=tol))

def qr(a: Tensor, mode='reduced'):
    Q, R = np.linalg.qr(a._data, mode=mode)
    return Tensor(Q), Tensor(R)

def cholesky(a: Tensor) -> Tensor:
    return Tensor(np.linalg.cholesky(a._data))

def pinv(a: Tensor) -> Tensor:
    return Tensor(np.linalg.pinv(a._data))

def trace(a: Tensor) -> Tensor:
    return Tensor(np.trace(a._data))

def matrix_power(a: Tensor, n: int) -> Tensor:
    return Tensor(np.linalg.matrix_power(a._data, n))

def cross(a: Tensor, b: Tensor, dim=-1) -> Tensor:
    return Tensor(np.cross(a._data, b._data, axis=dim))

def outer(a: Tensor, b: Tensor) -> Tensor:
    return Tensor(np.outer(a._data, b._data))

def kron(a: Tensor, b: Tensor) -> Tensor:
    return Tensor(np.kron(a._data, b._data))

def vander(x: Tensor, N=None, increasing=False) -> Tensor:
    return Tensor(np.vander(x._data, N, increasing))
