"""
vectorphile NDArray — fully compiled Mojo struct.

Replaces Python's array.array with a heap-allocated typed buffer managed
entirely in Mojo. The Python side holds only a pointer (via ctypes c_void_p)
and calls exported C-ABI functions to operate on it.

No Python object overhead. No GIL. No boxing. Pure compiled memory.

Exported C ABI (all functions prefixed vp_nd_):
  Lifecycle:
    vp_nd_alloc(int64 size, int dtype) -> void*       allocate
    vp_nd_free(void* buf)                              free
    vp_nd_copy(void* dst, void* src, int64 n)         memcpy
    vp_nd_fill(void* buf, float val, int64 n)          fill with scalar
    vp_nd_zeros(void* buf, int64 n)                    zero-fill

  Arithmetic (all return new alloc'd buffer):
    vp_nd_add(void* a, void* b, int64 n) -> void*
    vp_nd_sub(void* a, void* b, int64 n) -> void*
    vp_nd_mul(void* a, void* b, int64 n) -> void*
    vp_nd_div(void* a, void* b, int64 n) -> void*
    vp_nd_add_scalar(void* a, float s, int64 n) -> void*
    vp_nd_mul_scalar(void* a, float s, int64 n) -> void*

  Reduction:
    vp_nd_sum(void* buf, int64 n) -> float
    vp_nd_max(void* buf, int64 n) -> float
    vp_nd_min(void* buf, int64 n) -> float
    vp_nd_argmax(void* buf, int64 n) -> int64
    vp_nd_argmin(void* buf, int64 n) -> int64
    vp_nd_dot(void* a, void* b, int64 n) -> float

  Unary math:
    vp_nd_exp(void* buf, int64 n) -> void*
    vp_nd_log(void* buf, int64 n) -> void*
    vp_nd_sqrt(void* buf, int64 n) -> void*
    vp_nd_abs(void* buf, int64 n) -> void*
    vp_nd_neg(void* buf, int64 n) -> void*
    vp_nd_relu(void* buf, int64 n) -> void*
    vp_nd_sigmoid(void* buf, int64 n) -> void*
    vp_nd_tanh(void* buf, int64 n) -> void*
    vp_nd_gelu(void* buf, int64 n) -> void*
    vp_nd_silu(void* buf, int64 n) -> void*
    vp_nd_softmax_inplace(void* buf, int64 n)

  GEMM:
    vp_nd_matmul(void* C, void* A, void* B, int M, int K, int N)

  Utils:
    vp_nd_set(void* buf, int64 idx, float val)
    vp_nd_get(void* buf, int64 idx) -> float
    vp_nd_copy_to_py(void* buf, float* dst, int64 n)   copy to Python buffer
    vp_nd_copy_from_py(void* buf, float* src, int64 n) copy from Python buffer
"""
from memory import UnsafePointer, memset_zero, malloc, free
from math import exp, log, sqrt, tanh, abs
from sys.info import simdwidthof
from algorithm import vectorize, parallelize

alias float32 = DType.float32
alias SIMD_W = simdwidthof[float32]()
alias NEG_INF = Float32(-3.4028235e+38)
alias POS_INF = Float32(3.4028235e+38)

# dtype codes match Python _TYPECODES in ndarray.py
alias DTYPE_F32 = 0
alias DTYPE_F64 = 1
alias DTYPE_I32 = 2
alias DTYPE_I64 = 3
alias DTYPE_I8  = 4
alias DTYPE_U8  = 5


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@export
fn vp_nd_alloc(size: Int64, dtype: Int32) -> UnsafePointer[Float32]:
    """Allocate a typed flat buffer. Returns float32* (caller casts as needed)."""
    var item_size: Int = 4  # default float32
    if dtype == DTYPE_F64 or dtype == DTYPE_I64: item_size = 8
    elif dtype == DTYPE_I8 or dtype == DTYPE_U8:  item_size = 1
    var nbytes = Int(size) * item_size
    return malloc[Float32](nbytes)


@export
fn vp_nd_free(buf: UnsafePointer[Float32]):
    free(buf)


@export
fn vp_nd_zeros(buf: UnsafePointer[Float32], n: Int64):
    memset_zero(buf, Int(n))


@export
fn vp_nd_fill(buf: UnsafePointer[Float32], val: Float32, n: Int64):
    var vv = SIMD[float32, SIMD_W](val)
    var i = 0
    while i + SIMD_W <= Int(n):
        buf.store[width=SIMD_W](i, vv); i += SIMD_W
    while i < Int(n):
        buf[i] = val; i += 1


@export
fn vp_nd_copy(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int64):
    var i = 0
    while i + SIMD_W <= Int(n):
        dst.store[width=SIMD_W](i, src.load[width=SIMD_W](i)); i += SIMD_W
    while i < Int(n):
        dst[i] = src[i]; i += 1


@export
fn vp_nd_set(buf: UnsafePointer[Float32], idx: Int64, val: Float32):
    buf[Int(idx)] = val


@export
fn vp_nd_get(buf: UnsafePointer[Float32], idx: Int64) -> Float32:
    return buf[Int(idx)]


@export
fn vp_nd_copy_to_py(buf: UnsafePointer[Float32], dst: UnsafePointer[Float32], n: Int64):
    vp_nd_copy(dst, buf, n)


@export
fn vp_nd_copy_from_py(buf: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int64):
    vp_nd_copy(buf, src, n)


# ── Binary arithmetic (returns NEW buffer) ────────────────────────────────────

@export
fn vp_nd_add(a: UnsafePointer[Float32], b: UnsafePointer[Float32], n: Int64) -> UnsafePointer[Float32]:
    var out = vp_nd_alloc(n, DTYPE_F32)
    @parameter
    fn k[w: Int](i: Int): out.store[width=w](i, a.load[width=w](i) + b.load[width=w](i))
    vectorize[k, SIMD_W](Int(n))
    return out

@export
fn vp_nd_sub(a: UnsafePointer[Float32], b: UnsafePointer[Float32], n: Int64) -> UnsafePointer[Float32]:
    var out = vp_nd_alloc(n, DTYPE_F32)
    @parameter
    fn k[w: Int](i: Int): out.store[width=w](i, a.load[width=w](i) - b.load[width=w](i))
    vectorize[k, SIMD_W](Int(n))
    return out

@export
fn vp_nd_mul(a: UnsafePointer[Float32], b: UnsafePointer[Float32], n: Int64) -> UnsafePointer[Float32]:
    var out = vp_nd_alloc(n, DTYPE_F32)
    @parameter
    fn k[w: Int](i: Int): out.store[width=w](i, a.load[width=w](i) * b.load[width=w](i))
    vectorize[k, SIMD_W](Int(n))
    return out

@export
fn vp_nd_div(a: UnsafePointer[Float32], b: UnsafePointer[Float32], n: Int64) -> UnsafePointer[Float32]:
    var out = vp_nd_alloc(n, DTYPE_F32)
    @parameter
    fn k[w: Int](i: Int): out.store[width=w](i, a.load[width=w](i) / b.load[width=w](i))
    vectorize[k, SIMD_W](Int(n))
    return out

@export
fn vp_nd_add_scalar(a: UnsafePointer[Float32], s: Float32, n: Int64) -> UnsafePointer[Float32]:
    var out = vp_nd_alloc(n, DTYPE_F32)
    @parameter
    fn k[w: Int](i: Int): out.store[width=w](i, a.load[width=w](i) + SIMD[float32, w](s))
    vectorize[k, SIMD_W](Int(n))
    return out

@export
fn vp_nd_mul_scalar(a: UnsafePointer[Float32], s: Float32, n: Int64) -> UnsafePointer[Float32]:
    var out = vp_nd_alloc(n, DTYPE_F32)
    @parameter
    fn k[w: Int](i: Int): out.store[width=w](i, a.load[width=w](i) * SIMD[float32, w](s))
    vectorize[k, SIMD_W](Int(n))
    return out


# ── Reductions ────────────────────────────────────────────────────────────────

@export
fn vp_nd_sum(buf: UnsafePointer[Float32], n: Int64) -> Float32:
    var acc = SIMD[float32, SIMD_W](0)
    var i = 0
    while i + SIMD_W <= Int(n): acc += buf.load[width=SIMD_W](i); i += SIMD_W
    var s = acc.reduce_add()
    while i < Int(n): s += buf[i]; i += 1
    return s

@export
fn vp_nd_max(buf: UnsafePointer[Float32], n: Int64) -> Float32:
    var acc = SIMD[float32, SIMD_W](NEG_INF)
    var i = 0
    while i + SIMD_W <= Int(n): acc = acc.max(buf.load[width=SIMD_W](i)); i += SIMD_W
    var m = acc.reduce_max()
    while i < Int(n): if buf[i] > m: m = buf[i]; i += 1
    return m

@export
fn vp_nd_min(buf: UnsafePointer[Float32], n: Int64) -> Float32:
    var acc = SIMD[float32, SIMD_W](POS_INF)
    var i = 0
    while i + SIMD_W <= Int(n): acc = acc.min(buf.load[width=SIMD_W](i)); i += SIMD_W
    var m = acc.reduce_min()
    while i < Int(n): if buf[i] < m: m = buf[i]; i += 1
    return m

@export
fn vp_nd_argmax(buf: UnsafePointer[Float32], n: Int64) -> Int64:
    var best_val = NEG_INF; var best_idx = Int64(0)
    for i in range(Int(n)):
        if buf[i] > best_val: best_val = buf[i]; best_idx = Int64(i)
    return best_idx

@export
fn vp_nd_argmin(buf: UnsafePointer[Float32], n: Int64) -> Int64:
    var best_val = POS_INF; var best_idx = Int64(0)
    for i in range(Int(n)):
        if buf[i] < best_val: best_val = buf[i]; best_idx = Int64(i)
    return best_idx

@export
fn vp_nd_dot(a: UnsafePointer[Float32], b: UnsafePointer[Float32], n: Int64) -> Float32:
    var acc = SIMD[float32, SIMD_W](0)
    var i = 0
    while i + SIMD_W <= Int(n): acc += a.load[width=SIMD_W](i) * b.load[width=SIMD_W](i); i += SIMD_W
    var s = acc.reduce_add()
    while i < Int(n): s += a[i] * b[i]; i += 1
    return s


# ── Unary math (returns NEW buffer) ──────────────────────────────────────────

@export
fn vp_nd_exp(buf: UnsafePointer[Float32], n: Int64) -> UnsafePointer[Float32]:
    var out = vp_nd_alloc(n, DTYPE_F32)
    @parameter
    fn k[w: Int](i: Int): out.store[width=w](i, exp(buf.load[width=w](i)))
    vectorize[k, SIMD_W](Int(n)); return out

@export
fn vp_nd_log(buf: UnsafePointer[Float32], n: Int64) -> UnsafePointer[Float32]:
    var out = vp_nd_alloc(n, DTYPE_F32)
    @parameter
    fn k[w: Int](i: Int): out.store[width=w](i, log(buf.load[width=w](i)))
    vectorize[k, SIMD_W](Int(n)); return out

@export
fn vp_nd_sqrt(buf: UnsafePointer[Float32], n: Int64) -> UnsafePointer[Float32]:
    var out = vp_nd_alloc(n, DTYPE_F32)
    @parameter
    fn k[w: Int](i: Int): out.store[width=w](i, sqrt(buf.load[width=w](i)))
    vectorize[k, SIMD_W](Int(n)); return out

@export
fn vp_nd_abs(buf: UnsafePointer[Float32], n: Int64) -> UnsafePointer[Float32]:
    var out = vp_nd_alloc(n, DTYPE_F32)
    @parameter
    fn k[w: Int](i: Int): out.store[width=w](i, abs(buf.load[width=w](i)))
    vectorize[k, SIMD_W](Int(n)); return out

@export
fn vp_nd_neg(buf: UnsafePointer[Float32], n: Int64) -> UnsafePointer[Float32]:
    var out = vp_nd_alloc(n, DTYPE_F32)
    @parameter
    fn k[w: Int](i: Int): out.store[width=w](i, -buf.load[width=w](i))
    vectorize[k, SIMD_W](Int(n)); return out

@export
fn vp_nd_relu(buf: UnsafePointer[Float32], n: Int64) -> UnsafePointer[Float32]:
    var out = vp_nd_alloc(n, DTYPE_F32)
    @parameter
    fn k[w: Int](i: Int): out.store[width=w](i, buf.load[width=w](i).max(SIMD[float32, w](0)))
    vectorize[k, SIMD_W](Int(n)); return out

@export
fn vp_nd_sigmoid(buf: UnsafePointer[Float32], n: Int64) -> UnsafePointer[Float32]:
    var out = vp_nd_alloc(n, DTYPE_F32)
    @parameter
    fn k[w: Int](i: Int):
        out.store[width=w](i, SIMD[float32, w](1) / (SIMD[float32, w](1) + exp(-buf.load[width=w](i))))
    vectorize[k, SIMD_W](Int(n)); return out

@export
fn vp_nd_tanh(buf: UnsafePointer[Float32], n: Int64) -> UnsafePointer[Float32]:
    var out = vp_nd_alloc(n, DTYPE_F32)
    @parameter
    fn k[w: Int](i: Int): out.store[width=w](i, tanh(buf.load[width=w](i)))
    vectorize[k, SIMD_W](Int(n)); return out

@export
fn vp_nd_gelu(buf: UnsafePointer[Float32], n: Int64) -> UnsafePointer[Float32]:
    alias C0 = Float32(0.7978845608028654)
    alias C1 = Float32(0.044715)
    var out = vp_nd_alloc(n, DTYPE_F32)
    @parameter
    fn k[w: Int](i: Int):
        var x = buf.load[width=w](i)
        out.store[width=w](i, SIMD[float32, w](0.5) * x * (SIMD[float32, w](1) + tanh(C0*(x + C1*x*x*x))))
    vectorize[k, SIMD_W](Int(n)); return out

@export
fn vp_nd_silu(buf: UnsafePointer[Float32], n: Int64) -> UnsafePointer[Float32]:
    var out = vp_nd_alloc(n, DTYPE_F32)
    @parameter
    fn k[w: Int](i: Int):
        var x = buf.load[width=w](i)
        out.store[width=w](i, x / (SIMD[float32, w](1) + exp(-x)))
    vectorize[k, SIMD_W](Int(n)); return out

@export
fn vp_nd_softmax_inplace(buf: UnsafePointer[Float32], n: Int64):
    var mx = vp_nd_max(buf, n)
    var acc = SIMD[float32, SIMD_W](0)
    var i = 0
    while i + SIMD_W <= Int(n):
        var v = exp(buf.load[width=SIMD_W](i) - SIMD[float32, SIMD_W](mx))
        buf.store[width=SIMD_W](i, v); acc += v; i += SIMD_W
    var s = acc.reduce_add()
    while i < Int(n):
        var v = exp(buf[i] - mx); buf[i] = v; s += v; i += 1
    var inv_s = SIMD[float32, SIMD_W](1 / s)
    i = 0
    while i + SIMD_W <= Int(n):
        buf.store[width=SIMD_W](i, buf.load[width=SIMD_W](i) * inv_s); i += SIMD_W
    while i < Int(n):
        buf[i] /= s; i += 1


# ── GEMM ──────────────────────────────────────────────────────────────────────

alias MC = 64; alias KC = 256; alias NC = 128

@export
fn vp_nd_matmul(
    C: UnsafePointer[Float32],
    A: UnsafePointer[Float32],
    B: UnsafePointer[Float32],
    M: Int, K: Int, N: Int
):
    memset_zero(C, M * N)
    var m0 = 0
    while m0 < M:
        var m_end = min(m0 + MC, M)
        var k0 = 0
        while k0 < K:
            var k_end = min(k0 + KC, K)
            var n0 = 0
            while n0 < N:
                var n_end = min(n0 + NC, N)
                for m in range(m0, m_end):
                    for k in range(k0, k_end):
                        var av = SIMD[float32, SIMD_W](A[m*K+k])
                        var n = n0
                        while n + SIMD_W <= n_end:
                            C.store[width=SIMD_W](m*N+n, C.load[width=SIMD_W](m*N+n) + av * B.load[width=SIMD_W](k*N+n))
                            n += SIMD_W
                        while n < n_end:
                            C[m*N+n] += A[m*K+k] * B[k*N+n]; n += 1
                n0 += NC
            k0 += KC
        m0 += MC
