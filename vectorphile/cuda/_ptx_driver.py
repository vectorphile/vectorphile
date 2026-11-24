"""CUDA Driver API wrapper for running compiled PTX kernels via ctypes.

Loads nvcuda.dll, creates a CUDA context, compiles the PTX module, and
exposes typed kernel launchers for relu, gelu, softmax, matmul, layernorm.
"""
from __future__ import annotations
import ctypes
import math
import os
import struct
from typing import List, Optional, Tuple

# ── CUDA Driver API types ─────────────────────────────────────────────────────

CUresult   = ctypes.c_int
CUdevice   = ctypes.c_int
CUcontext  = ctypes.c_void_p
CUmodule   = ctypes.c_void_p
CUfunction = ctypes.c_void_p
CUdeviceptr = ctypes.c_uint64


class _CUDADriver:
    """Thin ctypes wrapper around nvcuda.dll."""

    def __init__(self):
        self._lib = ctypes.windll.LoadLibrary("nvcuda.dll")
        self._ctx: Optional[CUcontext] = None
        self._mod: Optional[CUmodule] = None
        self._fns = {}

    def _check(self, res: int, op: str = ""):
        if res != 0:
            raise RuntimeError(f"CUDA Driver error {res} in {op}")

    def init(self, device: int = 0) -> None:
        self._check(self._lib.cuInit(0), "cuInit")
        dev = CUdevice(0)
        self._check(self._lib.cuDeviceGet(ctypes.byref(dev), device), "cuDeviceGet")
        ctx = CUcontext(0)
        self._check(self._lib.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")
        self._ctx = ctx

    def load_ptx(self, ptx_path: str) -> None:
        with open(ptx_path, "rb") as f:
            ptx_bytes = f.read() + b"\x00"
        mod = CUmodule(0)
        self._check(self._lib.cuModuleLoadData(ctypes.byref(mod), ptx_bytes), "cuModuleLoadData")
        self._mod = mod

    def _get_fn(self, name: str) -> CUfunction:
        if name not in self._fns:
            fn = CUfunction(0)
            self._check(
                self._lib.cuModuleGetFunction(ctypes.byref(fn), self._mod, name.encode()),
                f"cuModuleGetFunction({name})"
            )
            self._fns[name] = fn
        return self._fns[name]

    def alloc(self, nbytes: int) -> CUdeviceptr:
        ptr = CUdeviceptr(0)
        self._check(self._lib.cuMemAlloc_v2(ctypes.byref(ptr), nbytes), "cuMemAlloc")
        return ptr

    def free(self, ptr: CUdeviceptr) -> None:
        self._lib.cuMemFree_v2(ptr)

    def h2d(self, dst: CUdeviceptr, src: bytes, nbytes: int) -> None:
        self._check(
            self._lib.cuMemcpyHtoD_v2(dst, ctypes.cast(ctypes.c_char_p(src), ctypes.c_void_p), nbytes),
            "cuMemcpyHtoD"
        )

    def d2h(self, dst_buf: ctypes.Array, src: CUdeviceptr, nbytes: int) -> None:
        self._check(
            self._lib.cuMemcpyDtoH_v2(dst_buf, src, nbytes),
            "cuMemcpyDtoH"
        )

    def launch(self, fn_name: str,
               grid: Tuple[int,int,int], block: Tuple[int,int,int],
               *args) -> None:
        fn = self._get_fn(fn_name)
        # Build kernel argument array
        c_args = []
        ptrs   = []
        for a in args:
            if isinstance(a, CUdeviceptr):
                buf = ctypes.c_uint64(a.value)
            elif isinstance(a, int):
                buf = ctypes.c_int(a)
            elif isinstance(a, float):
                buf = ctypes.c_float(a)
            else:
                raise TypeError(f"Unsupported kernel arg type: {type(a)}")
            ptrs.append(buf)
            c_args.append(ctypes.cast(ctypes.byref(buf), ctypes.c_void_p))
        arr = (ctypes.c_void_p * len(c_args))(*c_args)
        self._check(
            self._lib.cuLaunchKernel(
                fn,
                grid[0], grid[1], grid[2],
                block[0], block[1], block[2],
                0, None, arr, None,
            ),
            f"cuLaunchKernel({fn_name})"
        )

    def sync(self) -> None:
        self._check(self._lib.cuCtxSynchronize(), "cuCtxSynchronize")


# ── Singleton driver ──────────────────────────────────────────────────────────

_drv: Optional[_CUDADriver] = None
_PTX_AVAILABLE = False

def _init_ptx() -> bool:
    global _drv, _PTX_AVAILABLE
    if _PTX_AVAILABLE:
        return True
    ptx_candidates = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "_core", "vp_kernels_sm89.ptx"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "_core", "vp_kernels_sm86.ptx"),
    ]
    ptx_path = next((p for p in ptx_candidates if os.path.isfile(p)), None)
    if ptx_path is None:
        return False
    try:
        d = _CUDADriver()
        d.init()
        d.load_ptx(ptx_path)
        _drv = d
        _PTX_AVAILABLE = True
        return True
    except Exception:
        return False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pack_f32(lst: List[float]) -> bytes:
    return struct.pack(f"{len(lst)}f", *lst)

def _pack_i32(lst: List[int]) -> bytes:
    return struct.pack(f"{len(lst)}i", *lst)

def _unpack_f32(b: bytes) -> List[float]:
    n = len(b) // 4
    return list(struct.unpack(f"{n}f", b))

_BLOCK = 256
_TILE  = 16


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ── Public kernel launchers ───────────────────────────────────────────────────

def ptx_relu(x: List[float]) -> List[float]:
    if not _init_ptx():
        return [max(0.0, v) for v in x]
    n = len(x)
    nb = n * 4
    xb = _pack_f32(x)
    dx = _drv.alloc(nb)
    dy = _drv.alloc(nb)
    try:
        _drv.h2d(dx, xb, nb)
        _drv.launch("vp_relu", (_ceil_div(n, _BLOCK), 1, 1), (_BLOCK, 1, 1), dx, dy, n)
        _drv.sync()
        out = (ctypes.c_char * nb)()
        _drv.d2h(out, dy, nb)
        return _unpack_f32(bytes(out))
    finally:
        _drv.free(dx); _drv.free(dy)


def ptx_gelu(x: List[float]) -> List[float]:
    if not _init_ptx():
        sq2pi = math.sqrt(2.0 / math.pi)
        return [0.5*v*(1+math.tanh(sq2pi*(v+0.044715*v**3))) for v in x]
    n = len(x); nb = n * 4
    xb = _pack_f32(x)
    dx = _drv.alloc(nb); dy = _drv.alloc(nb)
    try:
        _drv.h2d(dx, xb, nb)
        _drv.launch("vp_gelu", (_ceil_div(n, _BLOCK), 1, 1), (_BLOCK, 1, 1), dx, dy, n)
        _drv.sync()
        out = (ctypes.c_char * nb)()
        _drv.d2h(out, dy, nb)
        return _unpack_f32(bytes(out))
    finally:
        _drv.free(dx); _drv.free(dy)


def ptx_silu(x: List[float]) -> List[float]:
    if not _init_ptx():
        return [v / (1 + math.exp(-v)) for v in x]
    n = len(x); nb = n * 4
    xb = _pack_f32(x)
    dx = _drv.alloc(nb); dy = _drv.alloc(nb)
    try:
        _drv.h2d(dx, xb, nb)
        _drv.launch("vp_silu", (_ceil_div(n, _BLOCK), 1, 1), (_BLOCK, 1, 1), dx, dy, n)
        _drv.sync()
        out = (ctypes.c_char * nb)()
        _drv.d2h(out, dy, nb)
        return _unpack_f32(bytes(out))
    finally:
        _drv.free(dx); _drv.free(dy)


def ptx_softmax(x: List[float], N: int, C: int) -> List[float]:
    if not _init_ptx():
        return _cpu_softmax(x, N, C)
    nb_x = N * C * 4
    xb = _pack_f32(x)
    dx = _drv.alloc(nb_x); dy = _drv.alloc(nb_x)
    try:
        _drv.h2d(dx, xb, nb_x)
        _drv.launch("vp_softmax", (N, 1, 1), (_BLOCK, 1, 1), dx, dy, N, C)
        _drv.sync()
        out = (ctypes.c_char * nb_x)()
        _drv.d2h(out, dy, nb_x)
        return _unpack_f32(bytes(out))
    finally:
        _drv.free(dx); _drv.free(dy)


def ptx_matmul(a: List[float], M: int, K: int,
               b: List[float], N: int) -> List[float]:
    if not _init_ptx():
        return _cpu_matmul(a, M, K, b, N)
    nb_a = M * K * 4; nb_b = K * N * 4; nb_c = M * N * 4
    da = _drv.alloc(nb_a); db_ = _drv.alloc(nb_b); dc = _drv.alloc(nb_c)
    try:
        _drv.h2d(da, _pack_f32(a), nb_a)
        _drv.h2d(db_, _pack_f32(b), nb_b)
        gx = _ceil_div(N, _TILE); gy = _ceil_div(M, _TILE)
        _drv.launch("vp_matmul", (gx, gy, 1), (_TILE, _TILE, 1), da, db_, dc, M, K, N)
        _drv.sync()
        out = (ctypes.c_char * nb_c)()
        _drv.d2h(out, dc, nb_c)
        return _unpack_f32(bytes(out))
    finally:
        _drv.free(da); _drv.free(db_); _drv.free(dc)


def ptx_layernorm(x: List[float], w: List[float], b: List[float],
                  N: int, C: int, eps: float = 1e-5) -> List[float]:
    if not _init_ptx():
        return _cpu_layernorm(x, w, b, N, C, eps)
    nb = N * C * 4; wb = C * 4
    dx = _drv.alloc(nb); dw = _drv.alloc(wb); db_ = _drv.alloc(wb); dy = _drv.alloc(nb)
    try:
        _drv.h2d(dx, _pack_f32(x), nb)
        _drv.h2d(dw, _pack_f32(w), wb)
        _drv.h2d(db_, _pack_f32(b), wb)
        _drv.launch("vp_layernorm", (N, 1, 1), (_BLOCK, 1, 1), dx, dw, db_, dy, N, C, eps)
        _drv.sync()
        out = (ctypes.c_char * nb)()
        _drv.d2h(out, dy, nb)
        return _unpack_f32(bytes(out))
    finally:
        _drv.free(dx); _drv.free(dw); _drv.free(db_); _drv.free(dy)


# ── CPU fallbacks (used when PTX not available) ───────────────────────────────

def _cpu_softmax(x: List[float], N: int, C: int) -> List[float]:
    result = []
    for r in range(N):
        row = x[r*C:(r+1)*C]
        mx = max(row)
        exps = [math.exp(v - mx) for v in row]
        s = sum(exps)
        result.extend(e / s for e in exps)
    return result


def _cpu_matmul(a: List[float], M: int, K: int, b: List[float], N: int) -> List[float]:
    result = [0.0] * (M * N)
    for i in range(M):
        for j in range(N):
            s = 0.0
            for t in range(K):
                s += a[i*K+t] * b[t*N+j]
            result[i*N+j] = s
    return result


def _cpu_layernorm(x: List[float], w: List[float], b: List[float],
                   N: int, C: int, eps: float) -> List[float]:
    result = []
    for r in range(N):
        row = x[r*C:(r+1)*C]
        mean = sum(row) / C
        var  = sum((v - mean)**2 for v in row) / C
        inv  = 1.0 / math.sqrt(var + eps)
        result.extend((row[c] - mean) * inv * w[c % len(w)] + b[c % len(b)]
                       for c in range(C))
    return result
