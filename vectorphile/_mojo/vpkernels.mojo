"""
vectorphile Mojo CPU kernels — SIMD-vectorized build target.
Uses SIMD[DType.float32, 8] (256-bit AVX2) throughout for ~8× scalar throughput.
Exports all ops with C ABI for ctypes loading from Python.
"""

from memory import UnsafePointer
from math import exp, log, sqrt, tanh, pow
from sys.info import simdwidthof

alias F32 = DType.float32
alias W = 8  # AVX2 lane count for float32

# ── runtime init ──────────────────────────────────────────────────────────────

@export
fn vp_init() -> None:
    pass

# ── elementwise SIMD helpers ─────────────────────────────────────────────────

@always_inline
fn _simd_loop_binary[
    func: fn(SIMD[F32,W], SIMD[F32,W]) -> SIMD[F32,W]
](
    a: UnsafePointer[Float32],
    b: UnsafePointer[Float32],
    r: UnsafePointer[Float32],
    n: Int,
):
    var i = 0
    while i + W <= n:
        var va = SIMD[F32,W].load(a + i)
        var vb = SIMD[F32,W].load(b + i)
        func(va, vb).store(r + i)
        i += W
    while i < n:
        r[i] = Float32(func(SIMD[F32,W](a[i]), SIMD[F32,W](b[i]))[0])
        i += 1

@always_inline
fn _simd_loop_unary[
    func: fn(SIMD[F32,W]) -> SIMD[F32,W]
](
    a: UnsafePointer[Float32],
    r: UnsafePointer[Float32],
    n: Int,
):
    var i = 0
    while i + W <= n:
        func(SIMD[F32,W].load(a + i)).store(r + i)
        i += W
    while i < n:
        r[i] = Float32(func(SIMD[F32,W](a[i]))[0])
        i += 1

# ── elementwise ops ───────────────────────────────────────────────────────────

@export
fn vp_add_f32(a: UnsafePointer[Float32], b: UnsafePointer[Float32],
              result: UnsafePointer[Float32], n: Int32) -> None:
    _simd_loop_binary[fn(x: SIMD[F32,W], y: SIMD[F32,W]) -> SIMD[F32,W]: return x + y](
        a, b, result, Int(n))

@export
fn vp_sub_f32(a: UnsafePointer[Float32], b: UnsafePointer[Float32],
              result: UnsafePointer[Float32], n: Int32) -> None:
    _simd_loop_binary[fn(x: SIMD[F32,W], y: SIMD[F32,W]) -> SIMD[F32,W]: return x - y](
        a, b, result, Int(n))

@export
fn vp_mul_f32(a: UnsafePointer[Float32], b: UnsafePointer[Float32],
              result: UnsafePointer[Float32], n: Int32) -> None:
    _simd_loop_binary[fn(x: SIMD[F32,W], y: SIMD[F32,W]) -> SIMD[F32,W]: return x * y](
        a, b, result, Int(n))

@export
fn vp_div_f32(a: UnsafePointer[Float32], b: UnsafePointer[Float32],
              result: UnsafePointer[Float32], n: Int32) -> None:
    _simd_loop_binary[fn(x: SIMD[F32,W], y: SIMD[F32,W]) -> SIMD[F32,W]: return x / y](
        a, b, result, Int(n))

@export
fn vp_exp_f32(a: UnsafePointer[Float32], result: UnsafePointer[Float32], n: Int32) -> None:
    _simd_loop_unary[fn(x: SIMD[F32,W]) -> SIMD[F32,W]: return exp(x)](a, result, Int(n))

@export
fn vp_log_f32(a: UnsafePointer[Float32], result: UnsafePointer[Float32], n: Int32) -> None:
    _simd_loop_unary[fn(x: SIMD[F32,W]) -> SIMD[F32,W]: return log(x)](a, result, Int(n))

@export
fn vp_sqrt_f32(a: UnsafePointer[Float32], result: UnsafePointer[Float32], n: Int32) -> None:
    _simd_loop_unary[fn(x: SIMD[F32,W]) -> SIMD[F32,W]: return sqrt(x)](a, result, Int(n))

@export
fn vp_relu_f32(a: UnsafePointer[Float32], result: UnsafePointer[Float32], n: Int32) -> None:
    _simd_loop_unary[fn(x: SIMD[F32,W]) -> SIMD[F32,W]: return x.max(SIMD[F32,W](0.0))](
        a, result, Int(n))

@export
fn vp_gelu_f32(a: UnsafePointer[Float32], result: UnsafePointer[Float32], n: Int32) -> None:
    alias K: Float32 = 0.7978845608
    alias C: Float32 = 0.044715
    _simd_loop_unary[fn(x: SIMD[F32,W]) -> SIMD[F32,W]:
        return 0.5 * x * (1.0 + tanh(K * (x + C * x * x * x)))](
        a, result, Int(n))

@export
fn vp_silu_f32(a: UnsafePointer[Float32], result: UnsafePointer[Float32], n: Int32) -> None:
    _simd_loop_unary[fn(x: SIMD[F32,W]) -> SIMD[F32,W]:
        return x / (1.0 + exp(-x))](a, result, Int(n))

@export
fn vp_sigmoid_f32(a: UnsafePointer[Float32], result: UnsafePointer[Float32], n: Int32) -> None:
    _simd_loop_unary[fn(x: SIMD[F32,W]) -> SIMD[F32,W]:
        return 1.0 / (1.0 + exp(-x))](a, result, Int(n))

@export
fn vp_tanh_f32(a: UnsafePointer[Float32], result: UnsafePointer[Float32], n: Int32) -> None:
    _simd_loop_unary[fn(x: SIMD[F32,W]) -> SIMD[F32,W]: return tanh(x)](a, result, Int(n))

@export
fn vp_scalar_mul_f32(a: UnsafePointer[Float32], scalar: Float32,
                     result: UnsafePointer[Float32], n: Int32) -> None:
    var vs = SIMD[F32,W](scalar)
    var i = 0; var N = Int(n)
    while i + W <= N:
        (SIMD[F32,W].load(a + i) * vs).store(result + i)
        i += W
    while i < N:
        result[i] = a[i] * scalar
        i += 1

@export
fn vp_scalar_add_f32(a: UnsafePointer[Float32], scalar: Float32,
                     result: UnsafePointer[Float32], n: Int32) -> None:
    var vs = SIMD[F32,W](scalar)
    var i = 0; var N = Int(n)
    while i + W <= N:
        (SIMD[F32,W].load(a + i) + vs).store(result + i)
        i += W
    while i < N:
        result[i] = a[i] + scalar
        i += 1

# ── reductions (SIMD horizontal) ─────────────────────────────────────────────

@export
fn vp_sum_f32(a: UnsafePointer[Float32], n: Int32, result: UnsafePointer[Float32]) -> None:
    var N = Int(n)
    var acc = SIMD[F32,W](0.0)
    var i = 0
    while i + W <= N:
        acc += SIMD[F32,W].load(a + i)
        i += W
    var s: Float32 = acc.reduce_add()
    while i < N:
        s += a[i]; i += 1
    result[0] = s

@export
fn vp_max_f32(a: UnsafePointer[Float32], n: Int32, result: UnsafePointer[Float32]) -> None:
    var N = Int(n)
    var acc = SIMD[F32,W](a[0])
    var i = 0
    while i + W <= N:
        acc = acc.max(SIMD[F32,W].load(a + i))
        i += W
    var m: Float32 = acc.reduce_max()
    while i < N:
        if a[i] > m: m = a[i]; i += 1
    result[0] = m

@export
fn vp_min_f32(a: UnsafePointer[Float32], n: Int32, result: UnsafePointer[Float32]) -> None:
    var N = Int(n)
    var acc = SIMD[F32,W](a[0])
    var i = 0
    while i + W <= N:
        acc = acc.min(SIMD[F32,W].load(a + i))
        i += W
    var m: Float32 = acc.reduce_min()
    while i < N:
        if a[i] < m: m = a[i]; i += 1
    result[0] = m

@export
fn vp_argmax_f32(a: UnsafePointer[Float32], n: Int32,
                 result: UnsafePointer[Int32]) -> None:
    var N = Int(n)
    var best_val = a[0]
    var best_idx: Int32 = 0
    for i in range(1, N):
        if a[i] > best_val:
            best_val = a[i]; best_idx = Int32(i)
    result[0] = best_idx

# ── GEMM — cache-tiled with SIMD inner loop ──────────────────────────────────

@export
fn vp_gemm_f32(
    a: UnsafePointer[Float32], b: UnsafePointer[Float32],
    c: UnsafePointer[Float32],
    M: Int32, K: Int32, N: Int32,
) -> None:
    var m = Int(M); var k = Int(K); var n = Int(N)
    alias BM = 32; alias BK = 64; alias BN = 64
    for i in range(m * n):
        c[i] = Float32(0.0)
    var bm0 = 0
    while bm0 < m:
        var bm1 = min(bm0 + BM, m)
        var bk0 = 0
        while bk0 < k:
            var bk1 = min(bk0 + BK, k)
            var bn0 = 0
            while bn0 < n:
                var bn1 = min(bn0 + BN, n)
                for i in range(bm0, bm1):
                    var a_row = i * k
                    var c_row = i * n
                    for p in range(bk0, bk1):
                        var av = SIMD[F32,W](a[a_row + p])
                        var b_row = p * n
                        var j = bn0
                        while j + W <= bn1:
                            var cv = SIMD[F32,W].load(c + c_row + j)
                            var bv = SIMD[F32,W].load(b + b_row + j)
                            (cv + av * bv).store(c + c_row + j)
                            j += W
                        while j < bn1:
                            c[c_row + j] += a[a_row + p] * b[b_row + j]
                            j += 1
                bn0 = bn0 + BN
            bk0 = bk0 + BK
        bm0 = bm0 + BM

# ── fused ops ─────────────────────────────────────────────────────────────────

@export
fn vp_fused_linear_relu(
    x: UnsafePointer[Float32], weight: UnsafePointer[Float32],
    bias: UnsafePointer[Float32], result: UnsafePointer[Float32],
    batch: Int32, in_features: Int32, out_features: Int32,
) -> None:
    var B = Int(batch); var I = Int(in_features); var O = Int(out_features)
    for b in range(B):
        for o in range(O):
            var acc = SIMD[F32,W](0.0)
            var x_row = b * I; var w_row = o * I
            var i = 0
            while i + W <= I:
                acc += SIMD[F32,W].load(x + x_row + i) * SIMD[F32,W].load(weight + w_row + i)
                i += W
            var s: Float32 = acc.reduce_add() + bias[o]
            while i < I:
                s += x[x_row + i] * weight[w_row + i]; i += 1
            result[b * O + o] = s if s > Float32(0.0) else Float32(0.0)

@export
fn vp_fused_linear_gelu(
    x: UnsafePointer[Float32], weight: UnsafePointer[Float32],
    bias: UnsafePointer[Float32], result: UnsafePointer[Float32],
    batch: Int32, in_features: Int32, out_features: Int32,
) -> None:
    alias K: Float32 = 0.7978845608; alias C: Float32 = 0.044715
    var B = Int(batch); var I = Int(in_features); var O = Int(out_features)
    for b in range(B):
        for o in range(O):
            var acc = SIMD[F32,W](0.0)
            var x_row = b * I; var w_row = o * I
            var i = 0
            while i + W <= I:
                acc += SIMD[F32,W].load(x + x_row + i) * SIMD[F32,W].load(weight + w_row + i)
                i += W
            var s: Float32 = acc.reduce_add() + bias[o]
            while i < I:
                s += x[x_row + i] * weight[w_row + i]; i += 1
            result[b * O + o] = Float32(0.5) * s * (1.0 + tanh(K * (s + C * s * s * s)))

@export
fn vp_fused_linear_silu(
    x: UnsafePointer[Float32], weight: UnsafePointer[Float32],
    bias: UnsafePointer[Float32], result: UnsafePointer[Float32],
    batch: Int32, in_features: Int32, out_features: Int32,
) -> None:
    var B = Int(batch); var I = Int(in_features); var O = Int(out_features)
    for b in range(B):
        for o in range(O):
            var acc = SIMD[F32,W](0.0)
            var x_row = b * I; var w_row = o * I
            var i = 0
            while i + W <= I:
                acc += SIMD[F32,W].load(x + x_row + i) * SIMD[F32,W].load(weight + w_row + i)
                i += W
            var s: Float32 = acc.reduce_add() + bias[o]
            while i < I:
                s += x[x_row + i] * weight[w_row + i]; i += 1
            result[b * O + o] = s / (1.0 + exp(-s))

@export
fn vp_fused_rmsnorm(
    x: UnsafePointer[Float32], weight: UnsafePointer[Float32],
    result: UnsafePointer[Float32], n: Int32, eps: Float32,
) -> None:
    var N = Int(n)
    var acc = SIMD[F32,W](0.0)
    var i = 0
    while i + W <= N:
        var v = SIMD[F32,W].load(x + i)
        acc += v * v; i += W
    var ss: Float32 = acc.reduce_add()
    while i < N:
        ss += x[i] * x[i]; i += 1
    var scale = 1.0 / sqrt(ss / Float32(N) + eps)
    var vs = SIMD[F32,W](scale)
    i = 0
    while i + W <= N:
        (SIMD[F32,W].load(x + i) * vs * SIMD[F32,W].load(weight + i)).store(result + i)
        i += W
    while i < N:
        result[i] = x[i] * scale * weight[i]; i += 1

@export
fn vp_fused_layernorm(
    x: UnsafePointer[Float32], weight: UnsafePointer[Float32],
    bias: UnsafePointer[Float32], result: UnsafePointer[Float32],
    n: Int32, eps: Float32,
) -> None:
    var N = Int(n)
    # mean
    var mean_acc = SIMD[F32,W](0.0)
    var i = 0
    while i + W <= N:
        mean_acc += SIMD[F32,W].load(x + i); i += W
    var mean: Float32 = mean_acc.reduce_add()
    while i < N:
        mean += x[i]; i += 1
    mean /= Float32(N)
    # variance
    var var_acc = SIMD[F32,W](0.0)
    var vm = SIMD[F32,W](mean)
    i = 0
    while i + W <= N:
        var d = SIMD[F32,W].load(x + i) - vm
        var_acc += d * d; i += W
    var var_: Float32 = var_acc.reduce_add()
    while i < N:
        var d = x[i] - mean; var_ += d * d; i += 1
    var inv_std = 1.0 / sqrt(var_ / Float32(N) + eps)
    var vis = SIMD[F32,W](inv_std)
    i = 0
    while i + W <= N:
        var xv = SIMD[F32,W].load(x + i) - vm
        (xv * vis * SIMD[F32,W].load(weight + i) + SIMD[F32,W].load(bias + i)).store(result + i)
        i += W
    while i < N:
        result[i] = (x[i] - mean) * inv_std * weight[i] + bias[i]; i += 1

@export
fn vp_adam_step(
    param: UnsafePointer[Float32], grad: UnsafePointer[Float32],
    m: UnsafePointer[Float32], v: UnsafePointer[Float32],
    n: Int32, lr: Float32, beta1: Float32, beta2: Float32,
    eps: Float32, t: Int32,
) -> None:
    var N = Int(n)
    var b1t: Float32 = 1.0 - pow(beta1, Float32(Int(t)))
    var b2t: Float32 = 1.0 - pow(beta2, Float32(Int(t)))
    var vb1 = SIMD[F32,W](beta1); var v1b1 = SIMD[F32,W](1.0 - beta1)
    var vb2 = SIMD[F32,W](beta2); var v1b2 = SIMD[F32,W](1.0 - beta2)
    var vlr = SIMD[F32,W](lr)
    var vb1t = SIMD[F32,W](b1t); var vb2t = SIMD[F32,W](b2t)
    var veps = SIMD[F32,W](eps)
    var i = 0
    while i + W <= N:
        var g  = SIMD[F32,W].load(grad + i)
        var mi = vb1 * SIMD[F32,W].load(m + i) + v1b1 * g
        var vi = vb2 * SIMD[F32,W].load(v + i) + v1b2 * g * g
        mi.store(m + i); vi.store(v + i)
        var mh = mi / vb1t
        var vh = vi / vb2t
        var p = SIMD[F32,W].load(param + i) - vlr * mh / (sqrt(vh) + veps)
        p.store(param + i)
        i += W
    while i < N:
        var g = grad[i]
        m[i] = beta1 * m[i] + (1.0 - beta1) * g
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g
        var mhat = m[i] / b1t; var vhat = v[i] / b2t
        param[i] -= lr * mhat / (sqrt(vhat) + eps)
        i += 1

@export
fn vp_softmax_inplace(
    x: UnsafePointer[Float32], n: Int32,
) -> None:
    var N = Int(n)
    # max
    var max_acc = SIMD[F32,W](x[0])
    var i = 0
    while i + W <= N:
        max_acc = max_acc.max(SIMD[F32,W].load(x + i)); i += W
    var m: Float32 = max_acc.reduce_max()
    while i < N:
        if x[i] > m: m = x[i]; i += 1
    # exp(x - max)
    var vm = SIMD[F32,W](m)
    i = 0
    while i + W <= N:
        exp(SIMD[F32,W].load(x + i) - vm).store(x + i); i += W
    while i < N:
        x[i] = exp(x[i] - m); i += 1
    # sum
    var sum_acc = SIMD[F32,W](0.0)
    i = 0
    while i + W <= N:
        sum_acc += SIMD[F32,W].load(x + i); i += W
    var s: Float32 = sum_acc.reduce_add()
    while i < N:
        s += x[i]; i += 1
    # divide
    var vs = SIMD[F32,W](1.0 / s)
    i = 0
    while i + W <= N:
        (SIMD[F32,W].load(x + i) * vs).store(x + i); i += W
    while i < N:
        x[i] /= s; i += 1

@export
fn vp_fused_attention_scores(
    q: UnsafePointer[Float32], k: UnsafePointer[Float32],
    out: UnsafePointer[Float32],
    T: Int32, S: Int32, D: Int32, scale: Float32,
) -> None:
    # out[T,S] = (Q @ K^T) / scale  (no softmax)
    var t = Int(T); var s = Int(S); var d = Int(D)
    for i in range(t):
        for j in range(s):
            var acc = SIMD[F32,W](0.0)
            var q_row = i * d; var k_row = j * d
            var p = 0
            while p + W <= d:
                acc += SIMD[F32,W].load(q + q_row + p) * SIMD[F32,W].load(k + k_row + p)
                p += W
            var dot: Float32 = acc.reduce_add()
            while p < d:
                dot += q[q_row + p] * k[k_row + p]; p += 1
            out[i * s + j] = dot / scale
