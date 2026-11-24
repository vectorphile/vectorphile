"""Simple thread-local buffer pool to avoid cold-cache allocations.

Re-uses previously allocated array.array buffers of the same size+typecode,
so repeated calls to ops don't pay the cache-cold penalty from fresh alloc.
"""
from __future__ import annotations
import array as _array
import threading

_local = threading.local()
_MAX_PER_KEY = 2

_DTYPE_TO_TC = {
    'float32': 'f', 'float64': 'd', 'int32': 'i', 'int64': 'l',
    'uint8': 'B', 'bool': 'b',
}


def _get_pool():
    if not hasattr(_local, 'pool'):
        _local.pool = {}
    return _local.pool


def _n_elements(shape):
    n = 1
    for s in shape: n *= s
    return n


def alloc(shape: tuple, dtype: str) -> _array.array:
    """Return a cache-warm array.array buffer; allocate fresh only if pool is empty."""
    pool = _get_pool()
    key = (shape, dtype)
    stack = pool.get(key)
    if stack:
        return stack.pop()
    tc = _DTYPE_TO_TC.get(dtype, 'f')
    n = _n_elements(shape)
    return _array.array(tc, bytes(n * _array.array(tc).itemsize))


def release(arr: _array.array) -> None:
    """Return `arr` to the pool for reuse."""
    if not isinstance(arr, _array.array):
        return
    pool = _get_pool()
    # reconstruct key from typecode — we can't recover shape; use size as proxy
    tc = arr.typecode
    dtype = {v: k for k, v in _DTYPE_TO_TC.items()}.get(tc, tc)
    key = (len(arr), dtype)
    stack = pool.setdefault(key, [])
    if len(stack) < _MAX_PER_KEY:
        stack.append(arr)
