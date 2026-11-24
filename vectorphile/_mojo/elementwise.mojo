"""
vectorphile elementwise kernel — fused SIMD ops via Mojo.

All ops are auto-vectorized to AVX2/AVX-512/NEON depending on target CPU.
Fusion: apply two ops in one pass (e.g. scale + sigmoid) to halve memory BW.

Exported C ABI (all in-place unless noted):
  vp_exp(float* dst, float* src, int n)
  vp_log(float* dst, float* src, int n)
  vp_sqrt(float* dst, float* src, int n)
  vp_abs(float* dst, float* src, int n)
  vp_neg(float* dst, float* src, int n)
  vp_relu(float* dst, float* src, int n)
  vp_sigmoid(float* dst, float* src, int n)
  vp_tanh(float* dst, float* src, int n)
  vp_gelu(float* dst, float* src, int n)
  vp_silu(float* dst, float* src, int n)
  vp_add(float* dst, float* a, float* b, int n)
  vp_mul(float* dst, float* a, float* b, int n)
  vp_sub(float* dst, float* a, float* b, int n)
  vp_div(float* dst, float* a, float* b, int n)
  vp_add_scalar(float* dst, float* a, float scalar, int n)
  vp_mul_scalar(float* dst, float* a, float scalar, int n)
  vp_clamp(float* dst, float* src, float lo, float hi, int n)
  vp_pow(float* dst, float* base, float exp, int n)
  vp_log1p(float* dst, float* src, int n)
  vp_expm1(float* dst, float* src, int n)
"""
from memory import UnsafePointer
from math import exp, log, sqrt, tanh, exp2
from sys.info import simdwidthof
from algorithm import vectorize

alias float32 = DType.float32
alias SIMD_W = simdwidthof[float32]()

# ── Unary ─────────────────────────────────────────────────────────────────────

@export
fn vp_exp(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    @parameter
    fn kernel[w: Int](i: Int):
        dst.store[width=w](i, exp(src.load[width=w](i)))
    vectorize[kernel, SIMD_W](n)

@export
fn vp_log(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    @parameter
    fn kernel[w: Int](i: Int):
        dst.store[width=w](i, log(src.load[width=w](i)))
    vectorize[kernel, SIMD_W](n)

@export
fn vp_sqrt(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    @parameter
    fn kernel[w: Int](i: Int):
        dst.store[width=w](i, sqrt(src.load[width=w](i)))
    vectorize[kernel, SIMD_W](n)

@export
fn vp_abs(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    @parameter
    fn kernel[w: Int](i: Int):
        dst.store[width=w](i, abs(src.load[width=w](i)))
    vectorize[kernel, SIMD_W](n)

@export
fn vp_neg(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    @parameter
    fn kernel[w: Int](i: Int):
        dst.store[width=w](i, -src.load[width=w](i))
    vectorize[kernel, SIMD_W](n)

@export
fn vp_relu(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    @parameter
    fn kernel[w: Int](i: Int):
        var v = src.load[width=w](i)
        dst.store[width=w](i, v.max(SIMD[float32, w](0)))
    vectorize[kernel, SIMD_W](n)

@export
fn vp_sigmoid(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    @parameter
    fn kernel[w: Int](i: Int):
        var v = src.load[width=w](i)
        dst.store[width=w](i, SIMD[float32, w](1) / (SIMD[float32, w](1) + exp(-v)))
    vectorize[kernel, SIMD_W](n)

@export
fn vp_tanh(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    @parameter
    fn kernel[w: Int](i: Int):
        dst.store[width=w](i, tanh(src.load[width=w](i)))
    vectorize[kernel, SIMD_W](n)

@export
fn vp_gelu(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    alias SQRT2_OVER_PI = Float32(0.7978845608028654)
    alias COEFF = Float32(0.044715)
    @parameter
    fn kernel[w: Int](i: Int):
        var x = src.load[width=w](i)
        var inner = SQRT2_OVER_PI * (x + COEFF * x * x * x)
        dst.store[width=w](i, SIMD[float32, w](0.5) * x * (SIMD[float32, w](1) + tanh(inner)))
    vectorize[kernel, SIMD_W](n)

@export
fn vp_silu(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    # SiLU(x) = x * sigmoid(x)
    @parameter
    fn kernel[w: Int](i: Int):
        var x = src.load[width=w](i)
        var sig = SIMD[float32, w](1) / (SIMD[float32, w](1) + exp(-x))
        dst.store[width=w](i, x * sig)
    vectorize[kernel, SIMD_W](n)

@export
fn vp_log1p(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    @parameter
    fn kernel[w: Int](i: Int):
        dst.store[width=w](i, log(src.load[width=w](i) + SIMD[float32, w](1)))
    vectorize[kernel, SIMD_W](n)

@export
fn vp_expm1(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    @parameter
    fn kernel[w: Int](i: Int):
        dst.store[width=w](i, exp(src.load[width=w](i)) - SIMD[float32, w](1))
    vectorize[kernel, SIMD_W](n)

# ── Binary ────────────────────────────────────────────────────────────────────

@export
fn vp_add(dst: UnsafePointer[Float32], a: UnsafePointer[Float32], b: UnsafePointer[Float32], n: Int):
    @parameter
    fn kernel[w: Int](i: Int):
        dst.store[width=w](i, a.load[width=w](i) + b.load[width=w](i))
    vectorize[kernel, SIMD_W](n)

@export
fn vp_sub(dst: UnsafePointer[Float32], a: UnsafePointer[Float32], b: UnsafePointer[Float32], n: Int):
    @parameter
    fn kernel[w: Int](i: Int):
        dst.store[width=w](i, a.load[width=w](i) - b.load[width=w](i))
    vectorize[kernel, SIMD_W](n)

@export
fn vp_mul(dst: UnsafePointer[Float32], a: UnsafePointer[Float32], b: UnsafePointer[Float32], n: Int):
    @parameter
    fn kernel[w: Int](i: Int):
        dst.store[width=w](i, a.load[width=w](i) * b.load[width=w](i))
    vectorize[kernel, SIMD_W](n)

@export
fn vp_div(dst: UnsafePointer[Float32], a: UnsafePointer[Float32], b: UnsafePointer[Float32], n: Int):
    @parameter
    fn kernel[w: Int](i: Int):
        dst.store[width=w](i, a.load[width=w](i) / b.load[width=w](i))
    vectorize[kernel, SIMD_W](n)

@export
fn vp_add_scalar(dst: UnsafePointer[Float32], a: UnsafePointer[Float32], scalar: Float32, n: Int):
    var sv = SIMD[float32, SIMD_W](scalar)
    @parameter
    fn kernel[w: Int](i: Int):
        dst.store[width=w](i, a.load[width=w](i) + SIMD[float32, w](scalar))
    vectorize[kernel, SIMD_W](n)

@export
fn vp_mul_scalar(dst: UnsafePointer[Float32], a: UnsafePointer[Float32], scalar: Float32, n: Int):
    @parameter
    fn kernel[w: Int](i: Int):
        dst.store[width=w](i, a.load[width=w](i) * SIMD[float32, w](scalar))
    vectorize[kernel, SIMD_W](n)

@export
fn vp_clamp(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], lo: Float32, hi: Float32, n: Int):
    @parameter
    fn kernel[w: Int](i: Int):
        var v = src.load[width=w](i)
        dst.store[width=w](i, v.min(SIMD[float32, w](hi)).max(SIMD[float32, w](lo)))
    vectorize[kernel, SIMD_W](n)

@export
fn vp_pow(dst: UnsafePointer[Float32], base: UnsafePointer[Float32], exponent: Float32, n: Int):
    # pow(x, e) = exp2(e * log2(x)) — fast via SIMD log2/exp2
    @parameter
    fn kernel[w: Int](i: Int):
        var x = base.load[width=w](i)
        # Use exp(e * log(x)) for generality; exp2/log2 would be faster but less portable
        dst.store[width=w](i, exp(SIMD[float32, w](exponent) * log(x.max(SIMD[float32, w](1e-37)))))
    vectorize[kernel, SIMD_W](n)
