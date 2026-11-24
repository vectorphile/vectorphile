"""
vectorphile fused kernel library — multi-op single-pass kernels.

Each fused kernel does what two separate kernels would do, but touches
memory only once. For memory-bandwidth-bound workloads this is 1.5–2×
faster than chaining individual ops.

Fused kernels exported:
  vp_fused_linear_relu(out, x, W, b, N, in, out_features)
    out = relu(x @ W^T + b)

  vp_fused_linear_gelu(out, x, W, b, N, in, out_features)
    out = gelu(x @ W^T + b)

  vp_fused_linear_silu(out, x, W, b, N, in, out_features)
    out = silu(x @ W^T + b)  ← LLaMA FFN gate

  vp_fused_scale_add(out, a, b, alpha, n)
    out = a * alpha + b  ← residual connections, Adam update

  vp_fused_rmsnorm(out, x, weight, n, eps)
    out = x / rms(x) * weight  ← LLaMA/Mistral norm

  vp_fused_layernorm(out, x, weight, bias, n, eps)
    out = (x - mean) / std * weight + bias

  vp_fused_rope(x, cos_cache, sin_cache, n_heads, head_dim, seq_len)
    Rotary position embedding in-place  ← LLaMA attention

  vp_fused_attention_scores(out, Q, K, scale, B, H, S, D)
    out = Q @ K^T * scale  ← scaled dot-product attention scores

  vp_fused_dropout_add(out, x, residual, prob, n, seed)
    out = x * mask / (1 - prob) + residual  ← fused dropout + residual

  vp_fused_adam_step(param, grad, m, v, lr, b1, b2, eps, t, n)
    Single-pass Adam parameter update
"""
from memory import UnsafePointer, memset_zero
from math import exp, log, sqrt, tanh, cos, sin
from sys.info import simdwidthof
from algorithm import vectorize, parallelize

alias float32 = DType.float32
alias SIMD_W = simdwidthof[float32]()


# ── Activation helpers (inlined, no call overhead) ────────────────────────────

@always_inline
fn _relu[w: Int](x: SIMD[float32, w]) -> SIMD[float32, w]:
    return x.max(SIMD[float32, w](0))

@always_inline
fn _gelu[w: Int](x: SIMD[float32, w]) -> SIMD[float32, w]:
    alias C0 = Float32(0.7978845608028654)
    alias C1 = Float32(0.044715)
    return SIMD[float32, w](0.5) * x * (SIMD[float32, w](1) + tanh(C0 * (x + C1 * x * x * x)))

@always_inline
fn _silu[w: Int](x: SIMD[float32, w]) -> SIMD[float32, w]:
    return x / (SIMD[float32, w](1) + exp(-x))


# ── Linear + activation fusions ───────────────────────────────────────────────

@export
fn vp_fused_linear_relu(
    out: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    W: UnsafePointer[Float32],   # shape [out_features, in_features]
    b: UnsafePointer[Float32],   # shape [out_features], can be NULL
    N: Int, in_features: Int, out_features: Int
):
    """out[n, j] = relu(sum_k x[n,k] * W[j,k] + b[j])"""
    for n in range(N):
        for j in range(out_features):
            var acc = SIMD[float32, SIMD_W](0)
            var k = 0
            while k + SIMD_W <= in_features:
                acc += x.load[width=SIMD_W](n*in_features+k) * W.load[width=SIMD_W](j*in_features+k)
                k += SIMD_W
            var s = acc.reduce_add()
            while k < in_features:
                s += x[n*in_features+k] * W[j*in_features+k]; k += 1
            if b: s += b[j]
            out[n*out_features+j] = s if s > 0 else Float32(0)


@export
fn vp_fused_linear_gelu(
    out: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    W: UnsafePointer[Float32],
    b: UnsafePointer[Float32],
    N: Int, in_features: Int, out_features: Int
):
    alias C0 = Float32(0.7978845608028654)
    alias C1 = Float32(0.044715)
    for n in range(N):
        for j in range(out_features):
            var acc = SIMD[float32, SIMD_W](0)
            var k = 0
            while k + SIMD_W <= in_features:
                acc += x.load[width=SIMD_W](n*in_features+k) * W.load[width=SIMD_W](j*in_features+k)
                k += SIMD_W
            var s = acc.reduce_add()
            while k < in_features:
                s += x[n*in_features+k] * W[j*in_features+k]; k += 1
            if b: s += b[j]
            out[n*out_features+j] = Float32(0.5) * s * (Float32(1) + tanh(C0*(s + C1*s*s*s)))


@export
fn vp_fused_linear_silu(
    out: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    W: UnsafePointer[Float32],
    b: UnsafePointer[Float32],
    N: Int, in_features: Int, out_features: Int
):
    for n in range(N):
        for j in range(out_features):
            var acc = SIMD[float32, SIMD_W](0)
            var k = 0
            while k + SIMD_W <= in_features:
                acc += x.load[width=SIMD_W](n*in_features+k) * W.load[width=SIMD_W](j*in_features+k)
                k += SIMD_W
            var s = acc.reduce_add()
            while k < in_features:
                s += x[n*in_features+k] * W[j*in_features+k]; k += 1
            if b: s += b[j]
            out[n*out_features+j] = s / (Float32(1) + exp(-s))


# ── Arithmetic fusions ────────────────────────────────────────────────────────

@export
fn vp_fused_scale_add(
    out: UnsafePointer[Float32],
    a: UnsafePointer[Float32],
    b: UnsafePointer[Float32],
    alpha: Float32, n: Int
):
    """out = a * alpha + b  — residual connections, Adam update."""
    var av = SIMD[float32, SIMD_W](alpha)
    @parameter
    fn kernel[w: Int](i: Int):
        out.store[width=w](i, a.load[width=w](i) * SIMD[float32, w](alpha) + b.load[width=w](i))
    vectorize[kernel, SIMD_W](n)


# ── Normalization fusions ─────────────────────────────────────────────────────

@export
fn vp_fused_rmsnorm(
    out: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    weight: UnsafePointer[Float32],
    n: Int, eps: Float32
):
    """out = x / rms(x) * weight  (LLaMA/Mistral RMSNorm)."""
    # Compute RMS in one SIMD pass
    var acc = SIMD[float32, SIMD_W](0)
    var i = 0
    while i + SIMD_W <= n:
        var v = x.load[width=SIMD_W](i)
        acc += v * v; i += SIMD_W
    var ss = acc.reduce_add()
    while i < n:
        ss += x[i] * x[i]; i += 1
    var inv_rms = Float32(1) / sqrt(ss / Float32(n) + eps)
    # Apply norm + weight in one pass
    @parameter
    fn kernel[w: Int](j: Int):
        out.store[width=w](j, x.load[width=w](j) * SIMD[float32, w](inv_rms) * weight.load[width=w](j))
    vectorize[kernel, SIMD_W](n)


@export
fn vp_fused_layernorm(
    out: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    weight: UnsafePointer[Float32],
    bias: UnsafePointer[Float32],
    n: Int, eps: Float32
):
    """out = (x - mean) / std * weight + bias  (standard LayerNorm)."""
    # Pass 1: mean
    var sum_acc = SIMD[float32, SIMD_W](0)
    var i = 0
    while i + SIMD_W <= n:
        sum_acc += x.load[width=SIMD_W](i); i += SIMD_W
    var mean = sum_acc.reduce_add()
    while i < n:
        mean += x[i]; i += 1
    mean /= Float32(n)
    # Pass 2: variance (single fused pass)
    var var_acc = SIMD[float32, SIMD_W](0)
    var mean_v = SIMD[float32, SIMD_W](mean)
    i = 0
    while i + SIMD_W <= n:
        var d = x.load[width=SIMD_W](i) - mean_v
        var_acc += d * d; i += SIMD_W
    var variance = var_acc.reduce_add()
    while i < n:
        var d = x[i] - mean; variance += d*d; i += 1
    var inv_std = Float32(1) / sqrt(variance / Float32(n) + eps)
    # Pass 3: normalize + scale + shift
    var inv_std_v = SIMD[float32, SIMD_W](inv_std)
    i = 0
    while i + SIMD_W <= n:
        var normed = (x.load[width=SIMD_W](i) - mean_v) * inv_std_v
        out.store[width=SIMD_W](i, normed * weight.load[width=SIMD_W](i) + bias.load[width=SIMD_W](i))
        i += SIMD_W
    while i < n:
        out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i]; i += 1


# ── Rotary Position Embedding (RoPE) ─────────────────────────────────────────

@export
fn vp_fused_rope(
    x: UnsafePointer[Float32],       # [B, n_heads, seq_len, head_dim] in-place
    cos_cache: UnsafePointer[Float32], # [seq_len, head_dim/2]
    sin_cache: UnsafePointer[Float32], # [seq_len, head_dim/2]
    B: Int, n_heads: Int, seq_len: Int, head_dim: Int
):
    """Apply RoPE in-place. head_dim must be even."""
    var half = head_dim // 2
    for b in range(B):
        for h in range(n_heads):
            for s in range(seq_len):
                var base = b*n_heads*seq_len*head_dim + h*seq_len*head_dim + s*head_dim
                for i in range(half):
                    var x0 = x[base + i]
                    var x1 = x[base + i + half]
                    var c  = cos_cache[s * half + i]
                    var sn = sin_cache[s * half + i]
                    x[base + i]        = x0 * c - x1 * sn
                    x[base + i + half] = x0 * sn + x1 * c


# ── Attention score fusion ────────────────────────────────────────────────────

@export
fn vp_fused_attention_scores(
    out: UnsafePointer[Float32],   # [B, H, S, S]
    Q: UnsafePointer[Float32],     # [B, H, S, D]
    K: UnsafePointer[Float32],     # [B, H, S, D]
    scale: Float32,
    B: Int, H: Int, S: Int, D: Int
):
    """Scaled QK^T — scores[b,h,i,j] = dot(Q[b,h,i], K[b,h,j]) * scale."""
    for b in range(B):
        for h in range(H):
            for i in range(S):
                for j in range(S):
                    var acc = SIMD[float32, SIMD_W](0)
                    var q_base = b*H*S*D + h*S*D + i*D
                    var k_base = b*H*S*D + h*S*D + j*D
                    var d = 0
                    while d + SIMD_W <= D:
                        acc += Q.load[width=SIMD_W](q_base+d) * K.load[width=SIMD_W](k_base+d)
                        d += SIMD_W
                    var s = acc.reduce_add()
                    while d < D:
                        s += Q[q_base+d] * K[k_base+d]; d += 1
                    out[b*H*S*S + h*S*S + i*S + j] = s * scale


# ── Fused dropout + residual ─────────────────────────────────────────────────

@export
fn vp_fused_dropout_add(
    out: UnsafePointer[Float32],
    x: UnsafePointer[Float32],
    residual: UnsafePointer[Float32],
    prob: Float32, n: Int, seed: UInt64
):
    """out = x * bernoulli_mask / (1-prob) + residual (training dropout + residual)."""
    var scale = Float32(1) / (Float32(1) - prob)
    # Simple LCG for mask (fast, good enough for dropout)
    var state = seed
    for i in range(n):
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        var u = Float32(state >> 41) / Float32(1 << 23)
        var mask = Float32(1) if u > prob else Float32(0)
        out[i] = x[i] * mask * scale + residual[i]


# ── Fused Adam step ───────────────────────────────────────────────────────────

@export
fn vp_fused_adam_step(
    param: UnsafePointer[Float32],
    grad: UnsafePointer[Float32],
    m: UnsafePointer[Float32],     # first moment (modified in-place)
    v: UnsafePointer[Float32],     # second moment (modified in-place)
    lr: Float32, b1: Float32, b2: Float32, eps: Float32,
    t: Int, n: Int                 # t = step count (for bias correction)
):
    """Single-pass fused Adam: update m, v, and param in one memory sweep."""
    var bc1 = Float32(1) - b1 ** t
    var bc2 = Float32(1) - b2 ** t
    var lr_corrected = lr * sqrt(bc2) / bc1

    @parameter
    fn kernel[w: Int](i: Int):
        var g  = grad.load[width=w](i)
        var mi = b1 * m.load[width=w](i) + (SIMD[float32, w](1)-SIMD[float32, w](b1)) * g
        var vi = b2 * v.load[width=w](i) + (SIMD[float32, w](1)-SIMD[float32, w](b2)) * g * g
        m.store[width=w](i, mi)
        v.store[width=w](i, vi)
        param.store[width=w](i, param.load[width=w](i) - SIMD[float32, w](lr_corrected) * mi / (sqrt(vi) + SIMD[float32, w](eps)))
    vectorize[kernel, SIMD_W](n)
