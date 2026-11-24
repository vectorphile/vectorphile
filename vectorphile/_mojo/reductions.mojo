"""
vectorphile reductions kernel — SIMD tree-reduction via Mojo.

Uses SIMD horizontal reduction: accumulate SIMD_W lanes in parallel,
then reduce the vector once. Far faster than scalar loop for large arrays.

Exported C ABI:
  vp_sum(float* src, int n) -> float
  vp_mean(float* src, int n) -> float
  vp_max(float* src, int n) -> float
  vp_min(float* src, int n) -> float
  vp_argmax(float* src, int n) -> int
  vp_argmin(float* src, int n) -> int
  vp_sum_axis(float* dst, float* src, int outer, int axis_len, int inner)
  vp_max_axis(float* dst, float* src, int outer, int axis_len, int inner)
  vp_min_axis(float* dst, float* src, int outer, int axis_len, int inner)
  vp_norm_l2(float* src, int n) -> float
  vp_dot(float* a, float* b, int n) -> float
  vp_softmax_inplace(float* x, int n)
  vp_log_softmax_inplace(float* x, int n)
"""
from memory import UnsafePointer
from math import exp, log, sqrt
from sys.info import simdwidthof
from algorithm import vectorize
from sys import info

alias float32 = DType.float32
alias SIMD_W = simdwidthof[float32]()
alias NEG_INF = Float32(-3.4028235e+38)
alias POS_INF = Float32(3.4028235e+38)


@export
fn vp_sum(src: UnsafePointer[Float32], n: Int) -> Float32:
    var acc = SIMD[float32, SIMD_W](0)
    var i = 0
    while i + SIMD_W <= n:
        acc += src.load[width=SIMD_W](i)
        i += SIMD_W
    var s = acc.reduce_add()
    while i < n:
        s += src[i]; i += 1
    return s


@export
fn vp_mean(src: UnsafePointer[Float32], n: Int) -> Float32:
    return vp_sum(src, n) / Float32(n)


@export
fn vp_max(src: UnsafePointer[Float32], n: Int) -> Float32:
    var acc = SIMD[float32, SIMD_W](NEG_INF)
    var i = 0
    while i + SIMD_W <= n:
        acc = acc.max(src.load[width=SIMD_W](i))
        i += SIMD_W
    var m = acc.reduce_max()
    while i < n:
        if src[i] > m: m = src[i]
        i += 1
    return m


@export
fn vp_min(src: UnsafePointer[Float32], n: Int) -> Float32:
    var acc = SIMD[float32, SIMD_W](POS_INF)
    var i = 0
    while i + SIMD_W <= n:
        acc = acc.min(src.load[width=SIMD_W](i))
        i += SIMD_W
    var m = acc.reduce_min()
    while i < n:
        if src[i] < m: m = src[i]
        i += 1
    return m


@export
fn vp_argmax(src: UnsafePointer[Float32], n: Int) -> Int:
    var best_val = NEG_INF
    var best_idx = 0
    for i in range(n):
        if src[i] > best_val:
            best_val = src[i]; best_idx = i
    return best_idx


@export
fn vp_argmin(src: UnsafePointer[Float32], n: Int) -> Int:
    var best_val = POS_INF
    var best_idx = 0
    for i in range(n):
        if src[i] < best_val:
            best_val = src[i]; best_idx = i
    return best_idx


@export
fn vp_norm_l2(src: UnsafePointer[Float32], n: Int) -> Float32:
    var acc = SIMD[float32, SIMD_W](0)
    var i = 0
    while i + SIMD_W <= n:
        var v = src.load[width=SIMD_W](i)
        acc += v * v
        i += SIMD_W
    var s = acc.reduce_add()
    while i < n:
        s += src[i] * src[i]; i += 1
    return sqrt(s)


@export
fn vp_dot(a: UnsafePointer[Float32], b: UnsafePointer[Float32], n: Int) -> Float32:
    var acc = SIMD[float32, SIMD_W](0)
    var i = 0
    while i + SIMD_W <= n:
        acc += a.load[width=SIMD_W](i) * b.load[width=SIMD_W](i)
        i += SIMD_W
    var s = acc.reduce_add()
    while i < n:
        s += a[i] * b[i]; i += 1
    return s


# ── Axis reductions ───────────────────────────────────────────────────────────
# src shape: [outer, axis_len, inner]
# dst shape: [outer, inner]  (axis collapsed)

@export
fn vp_sum_axis(
    dst: UnsafePointer[Float32],
    src: UnsafePointer[Float32],
    outer: Int, axis_len: Int, inner: Int
):
    for o in range(outer):
        for i in range(inner):
            var s: Float32 = 0
            for k in range(axis_len):
                s += src[o * axis_len * inner + k * inner + i]
            dst[o * inner + i] = s


@export
fn vp_max_axis(
    dst: UnsafePointer[Float32],
    src: UnsafePointer[Float32],
    outer: Int, axis_len: Int, inner: Int
):
    for o in range(outer):
        for i in range(inner):
            var m = NEG_INF
            for k in range(axis_len):
                var v = src[o * axis_len * inner + k * inner + i]
                if v > m: m = v
            dst[o * inner + i] = m


@export
fn vp_min_axis(
    dst: UnsafePointer[Float32],
    src: UnsafePointer[Float32],
    outer: Int, axis_len: Int, inner: Int
):
    for o in range(outer):
        for i in range(inner):
            var m = POS_INF
            for k in range(axis_len):
                var v = src[o * axis_len * inner + k * inner + i]
                if v < m: m = v
            dst[o * inner + i] = m


# ── Softmax (numerically stable, single-pass) ─────────────────────────────────

@export
fn vp_softmax_inplace(x: UnsafePointer[Float32], n: Int):
    # Pass 1: find max
    var mx = vp_max(x, n)
    # Pass 2: exp(x - max), accumulate sum
    var acc = SIMD[float32, SIMD_W](0)
    var i = 0
    while i + SIMD_W <= n:
        var v = exp(x.load[width=SIMD_W](i) - SIMD[float32, SIMD_W](mx))
        x.store[width=SIMD_W](i, v)
        acc += v
        i += SIMD_W
    var s = acc.reduce_add()
    while i < n:
        var v = exp(x[i] - mx)
        x[i] = v; s += v; i += 1
    # Pass 3: divide by sum
    var inv_s = SIMD[float32, SIMD_W](1.0 / s)
    i = 0
    while i + SIMD_W <= n:
        x.store[width=SIMD_W](i, x.load[width=SIMD_W](i) * inv_s)
        i += SIMD_W
    while i < n:
        x[i] /= s; i += 1


@export
fn vp_log_softmax_inplace(x: UnsafePointer[Float32], n: Int):
    var mx = vp_max(x, n)
    # Compute log(sum(exp(x - max))) = log_sum_exp
    var acc = SIMD[float32, SIMD_W](0)
    var i = 0
    while i + SIMD_W <= n:
        acc += exp(x.load[width=SIMD_W](i) - SIMD[float32, SIMD_W](mx))
        i += SIMD_W
    var s = acc.reduce_add()
    while i < n:
        s += exp(x[i] - mx); i += 1
    var log_sum = log(s)
    # x[i] = x[i] - max - log_sum
    var shift = SIMD[float32, SIMD_W](mx + log_sum)
    i = 0
    while i + SIMD_W <= n:
        x.store[width=SIMD_W](i, x.load[width=SIMD_W](i) - shift)
        i += SIMD_W
    while i < n:
        x[i] -= mx + log_sum; i += 1
