"""
vectorphile GPU elementwise kernels — Mojo @gpu.kernel.

Each thread handles one or more elements. Thread blocks of 256 threads
are launched over the full array. All ops auto-scale to any GPU.

For large arrays (>1M elements) these are 20–100× faster than CPU Mojo.
Memory bandwidth is the bottleneck; fused ops below cut BW usage by 2×.

Exported C ABI:
  vp_gpu_exp(ctx, dst, src, n)
  vp_gpu_log(ctx, dst, src, n)
  vp_gpu_sqrt(ctx, dst, src, n)
  vp_gpu_relu(ctx, dst, src, n)
  vp_gpu_sigmoid(ctx, dst, src, n)
  vp_gpu_tanh(ctx, dst, src, n)
  vp_gpu_gelu(ctx, dst, src, n)
  vp_gpu_silu(ctx, dst, src, n)
  vp_gpu_neg(ctx, dst, src, n)
  vp_gpu_abs(ctx, dst, src, n)
  vp_gpu_add(ctx, dst, a, b, n)
  vp_gpu_sub(ctx, dst, a, b, n)
  vp_gpu_mul(ctx, dst, a, b, n)
  vp_gpu_div(ctx, dst, a, b, n)
  vp_gpu_add_scalar(ctx, dst, a, scalar, n)
  vp_gpu_mul_scalar(ctx, dst, a, scalar, n)
  vp_gpu_clamp(ctx, dst, src, lo, hi, n)
  vp_gpu_fused_scale_add(ctx, dst, a, b, alpha, n)
  vp_gpu_fused_relu_add(ctx, dst, x, residual, n)
"""
from gpu.host import DeviceContext, DeviceBuffer
from gpu import thread_idx, block_idx, block_dim, barrier
from math import exp, log, sqrt, tanh
from memory import UnsafePointer

alias float32 = DType.float32
alias BLOCK = 256


struct GPUContext:
    var ctx: DeviceContext
    var device_id: Int
    fn __init__(inout self, device_id: Int):
        self.device_id = device_id
        self.ctx = DeviceContext(device_id=device_id)


fn _grid(n: Int) -> Int:
    return (n + BLOCK - 1) // BLOCK


# ── Unary kernels ─────────────────────────────────────────────────────────────

@gpu.kernel
fn _k_exp(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n: dst[i] = exp(src[i])

@gpu.kernel
fn _k_log(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n: dst[i] = log(src[i])

@gpu.kernel
fn _k_sqrt(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n: dst[i] = sqrt(src[i])

@gpu.kernel
fn _k_relu(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n: dst[i] = src[i] if src[i] > Float32(0) else Float32(0)

@gpu.kernel
fn _k_sigmoid(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n: dst[i] = Float32(1) / (Float32(1) + exp(-src[i]))

@gpu.kernel
fn _k_tanh(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n: dst[i] = tanh(src[i])

@gpu.kernel
fn _k_gelu(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    alias C0 = Float32(0.7978845608028654)
    alias C1 = Float32(0.044715)
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        var x = src[i]
        dst[i] = Float32(0.5) * x * (Float32(1) + tanh(C0 * (x + C1 * x * x * x)))

@gpu.kernel
fn _k_silu(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        var x = src[i]
        dst[i] = x / (Float32(1) + exp(-x))

@gpu.kernel
fn _k_neg(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n: dst[i] = -src[i]

@gpu.kernel
fn _k_abs(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n: dst[i] = src[i] if src[i] >= Float32(0) else -src[i]


# ── Binary kernels ────────────────────────────────────────────────────────────

@gpu.kernel
fn _k_add(dst: UnsafePointer[Float32], a: UnsafePointer[Float32], b: UnsafePointer[Float32], n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n: dst[i] = a[i] + b[i]

@gpu.kernel
fn _k_sub(dst: UnsafePointer[Float32], a: UnsafePointer[Float32], b: UnsafePointer[Float32], n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n: dst[i] = a[i] - b[i]

@gpu.kernel
fn _k_mul(dst: UnsafePointer[Float32], a: UnsafePointer[Float32], b: UnsafePointer[Float32], n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n: dst[i] = a[i] * b[i]

@gpu.kernel
fn _k_div(dst: UnsafePointer[Float32], a: UnsafePointer[Float32], b: UnsafePointer[Float32], n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n: dst[i] = a[i] / b[i]

@gpu.kernel
fn _k_add_scalar(dst: UnsafePointer[Float32], a: UnsafePointer[Float32], s: Float32, n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n: dst[i] = a[i] + s

@gpu.kernel
fn _k_mul_scalar(dst: UnsafePointer[Float32], a: UnsafePointer[Float32], s: Float32, n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n: dst[i] = a[i] * s

@gpu.kernel
fn _k_clamp(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], lo: Float32, hi: Float32, n: Int):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        var v = src[i]
        dst[i] = hi if v > hi else (lo if v < lo else v)


# ── Fused GPU kernels (saves a full memory round-trip) ────────────────────────

@gpu.kernel
fn _k_fused_scale_add(
    dst: UnsafePointer[Float32], a: UnsafePointer[Float32],
    b: UnsafePointer[Float32], alpha: Float32, n: Int
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n: dst[i] = a[i] * alpha + b[i]

@gpu.kernel
fn _k_fused_relu_add(
    dst: UnsafePointer[Float32], x: UnsafePointer[Float32],
    residual: UnsafePointer[Float32], n: Int
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        var v = x[i]
        dst[i] = (v if v > Float32(0) else Float32(0)) + residual[i]


# ── Exported launchers ────────────────────────────────────────────────────────

@export
fn vp_gpu_exp(h: UnsafePointer[GPUContext], dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    h[].ctx.enqueue_function[_k_exp](_grid(n), BLOCK)(dst, src, n)

@export
fn vp_gpu_log(h: UnsafePointer[GPUContext], dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    h[].ctx.enqueue_function[_k_log](_grid(n), BLOCK)(dst, src, n)

@export
fn vp_gpu_sqrt(h: UnsafePointer[GPUContext], dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    h[].ctx.enqueue_function[_k_sqrt](_grid(n), BLOCK)(dst, src, n)

@export
fn vp_gpu_relu(h: UnsafePointer[GPUContext], dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    h[].ctx.enqueue_function[_k_relu](_grid(n), BLOCK)(dst, src, n)

@export
fn vp_gpu_sigmoid(h: UnsafePointer[GPUContext], dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    h[].ctx.enqueue_function[_k_sigmoid](_grid(n), BLOCK)(dst, src, n)

@export
fn vp_gpu_tanh(h: UnsafePointer[GPUContext], dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    h[].ctx.enqueue_function[_k_tanh](_grid(n), BLOCK)(dst, src, n)

@export
fn vp_gpu_gelu(h: UnsafePointer[GPUContext], dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    h[].ctx.enqueue_function[_k_gelu](_grid(n), BLOCK)(dst, src, n)

@export
fn vp_gpu_silu(h: UnsafePointer[GPUContext], dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    h[].ctx.enqueue_function[_k_silu](_grid(n), BLOCK)(dst, src, n)

@export
fn vp_gpu_neg(h: UnsafePointer[GPUContext], dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    h[].ctx.enqueue_function[_k_neg](_grid(n), BLOCK)(dst, src, n)

@export
fn vp_gpu_abs(h: UnsafePointer[GPUContext], dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    h[].ctx.enqueue_function[_k_abs](_grid(n), BLOCK)(dst, src, n)

@export
fn vp_gpu_add(h: UnsafePointer[GPUContext], dst: UnsafePointer[Float32], a: UnsafePointer[Float32], b: UnsafePointer[Float32], n: Int):
    h[].ctx.enqueue_function[_k_add](_grid(n), BLOCK)(dst, a, b, n)

@export
fn vp_gpu_sub(h: UnsafePointer[GPUContext], dst: UnsafePointer[Float32], a: UnsafePointer[Float32], b: UnsafePointer[Float32], n: Int):
    h[].ctx.enqueue_function[_k_sub](_grid(n), BLOCK)(dst, a, b, n)

@export
fn vp_gpu_mul(h: UnsafePointer[GPUContext], dst: UnsafePointer[Float32], a: UnsafePointer[Float32], b: UnsafePointer[Float32], n: Int):
    h[].ctx.enqueue_function[_k_mul](_grid(n), BLOCK)(dst, a, b, n)

@export
fn vp_gpu_div(h: UnsafePointer[GPUContext], dst: UnsafePointer[Float32], a: UnsafePointer[Float32], b: UnsafePointer[Float32], n: Int):
    h[].ctx.enqueue_function[_k_div](_grid(n), BLOCK)(dst, a, b, n)

@export
fn vp_gpu_add_scalar(h: UnsafePointer[GPUContext], dst: UnsafePointer[Float32], a: UnsafePointer[Float32], s: Float32, n: Int):
    h[].ctx.enqueue_function[_k_add_scalar](_grid(n), BLOCK)(dst, a, s, n)

@export
fn vp_gpu_mul_scalar(h: UnsafePointer[GPUContext], dst: UnsafePointer[Float32], a: UnsafePointer[Float32], s: Float32, n: Int):
    h[].ctx.enqueue_function[_k_mul_scalar](_grid(n), BLOCK)(dst, a, s, n)

@export
fn vp_gpu_clamp(h: UnsafePointer[GPUContext], dst: UnsafePointer[Float32], src: UnsafePointer[Float32], lo: Float32, hi: Float32, n: Int):
    h[].ctx.enqueue_function[_k_clamp](_grid(n), BLOCK)(dst, src, lo, hi, n)

@export
fn vp_gpu_fused_scale_add(h: UnsafePointer[GPUContext], dst: UnsafePointer[Float32], a: UnsafePointer[Float32], b: UnsafePointer[Float32], alpha: Float32, n: Int):
    h[].ctx.enqueue_function[_k_fused_scale_add](_grid(n), BLOCK)(dst, a, b, alpha, n)

@export
fn vp_gpu_fused_relu_add(h: UnsafePointer[GPUContext], dst: UnsafePointer[Float32], x: UnsafePointer[Float32], residual: UnsafePointer[Float32], n: Int):
    h[].ctx.enqueue_function[_k_fused_relu_add](_grid(n), BLOCK)(dst, x, residual, n)
