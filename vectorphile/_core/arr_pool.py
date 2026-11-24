"""
Thread-local pool of Python array.array buffers — zero third-party dependencies.
Avoids the O(n) zero-init malloc on every tensor operation.

Hot-page cache: keeps a small set of array.array buffers permanently alive per
(n, typecode) key so the OS never unmaps those physical pages between calls —
eliminates the ~700 µs page-fault penalty on fresh large allocations.
"""
from __future__ import annotations
import threading
import array as _array
import ctypes as _ctypes
import sys as _sys

_local = threading.local()
_MAX_PER_KEY = 4   # keep up to 4 buffers per (typecode, n) key

# ── Hot-page array.array cache ───────────────────────────────────────────────
# Maps (n, typecode) → list of array.array buffers whose pages are mapped.
# Max 4 buffers per size. All stdlib — no numpy dependency.
_arr_hot: dict = {}
_ARR_MAX = 4

# bytes per element for supported typecodes
_ELEM_SIZE = {'f': 4, 'd': 8, 'i': 4, 'l': 8}

def alloc_np(n: int, dtype: str = 'float32'):
    """Return a hot array.array of length n for float32/float64 (buffer-protocol compatible)."""
    tc = 'f' if dtype == 'float32' else 'd'
    key = (n, tc)
    stack = _arr_hot.get(key)
    if stack:
        return stack.pop()
    # Fresh alloc — touch pages to wire them into physical memory
    arr = _array.array(tc, b'\x00' * (_ELEM_SIZE[tc] * n))
    return arr

def release_np(arr) -> None:
    """Return an array.array to the hot-page cache for reuse."""
    if arr is None:
        return
    tc = arr.typecode
    n = len(arr)
    key = (n, tc)
    stack = _arr_hot.setdefault(key, [])
    if len(stack) < _ARR_MAX:
        stack.append(arr)


def _get_pool():
    if not hasattr(_local, 'pool'):
        _local.pool = {}
    return _local.pool


def alloc(typecode: str, n: int) -> _array.array:
    """Return a recycled or freshly allocated array.array of length n."""
    pool = _get_pool()
    key = (typecode, n)
    stack = pool.get(key)
    if stack:
        return stack.pop()
    return _array.array(typecode, bytes(4 * n if typecode == 'f' else 8 * n))


def alloc_f32(n: int) -> _array.array:
    return alloc('f', n)


def alloc_f64(n: int) -> _array.array:
    return alloc('d', n)


def release(arr: _array.array) -> None:
    """Return arr to pool.  Caller must hold the ONLY reference besides ours."""
    pool = _get_pool()
    tc = arr.typecode
    n = len(arr)
    key = (tc, n)
    stack = pool.setdefault(key, [])
    if len(stack) < _MAX_PER_KEY:
        stack.append(arr)


def release_safe(arr: _array.array) -> None:
    """Release only when this is the last holder (rc == 2: us + getrefcount arg)."""
    if _sys.getrefcount(arr) <= 2:
        release(arr)
