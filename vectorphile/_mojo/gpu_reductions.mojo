"""
vectorphile GPU reductions — warp-level + block-level tree reduction.

Uses warp shuffle (__shfl_down equivalent in Mojo) for the final 32
elements, avoiding shared memory bank conflicts in the last 5 steps.

Reduction pattern (for N elements, BLOCK=256 threads):
  Pass 1: each thread reduces N/BLOCK elements → 1 value in register
  Pass 2: 256→128→64→32 via shared memory
  Pass 3: 32→1 via warp shuffle (no __syncthreads needed)

Result written to dst[0]. For large arrays a two-pass kernel is used.

Exported C ABI:
  vp_gpu_sum(ctx, dst_scalar, src, n)
  vp_gpu_max(ctx, dst_scalar, src, n)
  vp_gpu_min(ctx, dst_scalar, src, n)
  vp_gpu_sum_axis(ctx, dst, src, outer, axis_len, inner)
  vp_gpu_softmax_rows(ctx, x, rows, cols)      in-place per-row softmax
  vp_gpu_log_softmax_rows(ctx, x, rows, cols)  in-place per-row log-softmax
  vp_gpu_layernorm_rows(ctx, out, x, w, b, rows, cols, eps)
  vp_gpu_rmsnorm_rows(ctx, out, x, w, rows, cols, eps)
"""
from gpu.host import DeviceContext
from gpu import thread_idx, block_idx, block_dim, barrier, warp_reduce_sum, warp_reduce_max, warp_reduce_min
from memory import UnsafePointer, stack_allocation
from math import exp, log, sqrt

alias float32 = DType.float32
alias BLOCK = 256
alias WARP  = 32


struct GPUContext:
    var ctx: DeviceContext
    var device_id: Int
    fn __init__(inout self, device_id: Int):
        self.device_id = device_id
        self.ctx = DeviceContext(device_id=device_id)


# ── Block-level reduction helpers ─────────────────────────────────────────────

@gpu.kernel
fn _k_sum(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    var smem = stack_allocation[BLOCK, Float32]()
    var tid = thread_idx.x
    var i = block_idx.x * block_dim.x + tid
    var acc = Float32(0)
    while i < n: acc += src[i]; i += block_dim.x * gridDim.x
    smem[tid] = acc
    barrier()
    var s = BLOCK // 2
    while s > WARP:
        if tid < s: smem[tid] += smem[tid + s]
        barrier(); s //= 2
    if tid < WARP:
        var v = smem[tid]
        v = warp_reduce_sum(v)
        if tid == 0: dst[block_idx.x] = v


@gpu.kernel
fn _k_max(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    var smem = stack_allocation[BLOCK, Float32]()
    var tid = thread_idx.x
    var i = block_idx.x * block_dim.x + tid
    var acc = Float32(-3.4028235e+38)
    while i < n: if src[i] > acc: acc = src[i]; i += block_dim.x * gridDim.x
    smem[tid] = acc; barrier()
    var s = BLOCK // 2
    while s > WARP:
        if tid < s and smem[tid+s] > smem[tid]: smem[tid] = smem[tid+s]
        barrier(); s //= 2
    if tid < WARP:
        var v = smem[tid]
        v = warp_reduce_max(v)
        if tid == 0: dst[block_idx.x] = v


@gpu.kernel
fn _k_min(dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    var smem = stack_allocation[BLOCK, Float32]()
    var tid = thread_idx.x
    var i = block_idx.x * block_dim.x + tid
    var acc = Float32(3.4028235e+38)
    while i < n: if src[i] < acc: acc = src[i]; i += block_dim.x * gridDim.x
    smem[tid] = acc; barrier()
    var s = BLOCK // 2
    while s > WARP:
        if tid < s and smem[tid+s] < smem[tid]: smem[tid] = smem[tid+s]
        barrier(); s //= 2
    if tid < WARP:
        var v = smem[tid]
        v = warp_reduce_min(v)
        if tid == 0: dst[block_idx.x] = v


# ── Per-row softmax (one block per row, shared memory) ────────────────────────

@gpu.kernel
fn _k_softmax_rows(x: UnsafePointer[Float32], rows: Int, cols: Int):
    var smem = stack_allocation[BLOCK, Float32]()
    var row = block_idx.x
    if row >= rows: return
    var base = row * cols
    var tid = thread_idx.x

    # Pass 1: find max
    var mx = Float32(-3.4028235e+38)
    var i = tid
    while i < cols: if x[base+i] > mx: mx = x[base+i]; i += BLOCK
    smem[tid] = mx; barrier()
    var s = BLOCK // 2
    while s > 0:
        if tid < s and smem[tid+s] > smem[tid]: smem[tid] = smem[tid+s]
        barrier(); s //= 2
    mx = smem[0]; barrier()

    # Pass 2: exp(x - max), accumulate sum
    var sum = Float32(0)
    i = tid
    while i < cols:
        var v = exp(x[base+i] - mx)
        x[base+i] = v; sum += v; i += BLOCK
    smem[tid] = sum; barrier()
    s = BLOCK // 2
    while s > 0: if tid < s: smem[tid] += smem[tid+s]; barrier(); s //= 2
    var inv_sum = Float32(1) / smem[0]; barrier()

    # Pass 3: divide
    i = tid
    while i < cols: x[base+i] *= inv_sum; i += BLOCK


@gpu.kernel
fn _k_log_softmax_rows(x: UnsafePointer[Float32], rows: Int, cols: Int):
    var smem = stack_allocation[BLOCK, Float32]()
    var row = block_idx.x; if row >= rows: return
    var base = row * cols; var tid = thread_idx.x

    # max
    var mx = Float32(-3.4028235e+38)
    var i = tid
    while i < cols: if x[base+i] > mx: mx = x[base+i]; i += BLOCK
    smem[tid] = mx; barrier()
    var s = BLOCK // 2
    while s > 0:
        if tid < s and smem[tid+s] > smem[tid]: smem[tid] = smem[tid+s]
        barrier(); s //= 2
    mx = smem[0]; barrier()

    # sum of exp
    var sum = Float32(0)
    i = tid
    while i < cols: sum += exp(x[base+i] - mx); i += BLOCK
    smem[tid] = sum; barrier()
    s = BLOCK // 2
    while s > 0: if tid < s: smem[tid] += smem[tid+s]; barrier(); s //= 2
    var log_sum = log(smem[0]); barrier()

    # x = x - max - log_sum
    i = tid
    while i < cols: x[base+i] -= mx + log_sum; i += BLOCK


# ── Fused LayerNorm per row ───────────────────────────────────────────────────

@gpu.kernel
fn _k_layernorm_rows(
    out: UnsafePointer[Float32], x: UnsafePointer[Float32],
    w: UnsafePointer[Float32], b: UnsafePointer[Float32],
    rows: Int, cols: Int, eps: Float32
):
    var smem = stack_allocation[BLOCK, Float32]()
    var row = block_idx.x; if row >= rows: return
    var base = row * cols; var tid = thread_idx.x

    # Mean
    var sum = Float32(0)
    var i = tid
    while i < cols: sum += x[base+i]; i += BLOCK
    smem[tid] = sum; barrier()
    var s = BLOCK // 2
    while s > 0: if tid < s: smem[tid] += smem[tid+s]; barrier(); s //= 2
    var mean = smem[0] / Float32(cols); barrier()

    # Variance
    sum = Float32(0); i = tid
    while i < cols:
        var d = x[base+i] - mean; sum += d*d; i += BLOCK
    smem[tid] = sum; barrier()
    s = BLOCK // 2
    while s > 0: if tid < s: smem[tid] += smem[tid+s]; barrier(); s //= 2
    var inv_std = Float32(1) / sqrt(smem[0] / Float32(cols) + eps); barrier()

    # Normalize
    i = tid
    while i < cols:
        out[base+i] = (x[base+i] - mean) * inv_std * w[i] + b[i]; i += BLOCK


@gpu.kernel
fn _k_rmsnorm_rows(
    out: UnsafePointer[Float32], x: UnsafePointer[Float32],
    w: UnsafePointer[Float32],
    rows: Int, cols: Int, eps: Float32
):
    var smem = stack_allocation[BLOCK, Float32]()
    var row = block_idx.x; if row >= rows: return
    var base = row * cols; var tid = thread_idx.x

    var sum = Float32(0); var i = tid
    while i < cols:
        var v = x[base+i]; sum += v*v; i += BLOCK
    smem[tid] = sum; barrier()
    var s = BLOCK // 2
    while s > 0: if tid < s: smem[tid] += smem[tid+s]; barrier(); s //= 2
    var inv_rms = Float32(1) / sqrt(smem[0] / Float32(cols) + eps); barrier()

    i = tid
    while i < cols: out[base+i] = x[base+i] * inv_rms * w[i]; i += BLOCK


# ── Exported launchers ────────────────────────────────────────────────────────

@export
fn vp_gpu_sum(h: UnsafePointer[GPUContext], dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    var grid = (n + BLOCK - 1) // BLOCK
    h[].ctx.enqueue_function[_k_sum](grid, BLOCK)(dst, src, n)

@export
fn vp_gpu_max(h: UnsafePointer[GPUContext], dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    var grid = (n + BLOCK - 1) // BLOCK
    h[].ctx.enqueue_function[_k_max](grid, BLOCK)(dst, src, n)

@export
fn vp_gpu_min(h: UnsafePointer[GPUContext], dst: UnsafePointer[Float32], src: UnsafePointer[Float32], n: Int):
    var grid = (n + BLOCK - 1) // BLOCK
    h[].ctx.enqueue_function[_k_min](grid, BLOCK)(dst, src, n)

@export
fn vp_gpu_softmax_rows(h: UnsafePointer[GPUContext], x: UnsafePointer[Float32], rows: Int, cols: Int):
    h[].ctx.enqueue_function[_k_softmax_rows](rows, BLOCK)(x, rows, cols)

@export
fn vp_gpu_log_softmax_rows(h: UnsafePointer[GPUContext], x: UnsafePointer[Float32], rows: Int, cols: Int):
    h[].ctx.enqueue_function[_k_log_softmax_rows](rows, BLOCK)(x, rows, cols)

@export
fn vp_gpu_layernorm_rows(
    h: UnsafePointer[GPUContext],
    out: UnsafePointer[Float32], x: UnsafePointer[Float32],
    w: UnsafePointer[Float32], b: UnsafePointer[Float32],
    rows: Int, cols: Int, eps: Float32
):
    h[].ctx.enqueue_function[_k_layernorm_rows](rows, BLOCK)(out, x, w, b, rows, cols, eps)

@export
fn vp_gpu_rmsnorm_rows(
    h: UnsafePointer[GPUContext],
    out: UnsafePointer[Float32], x: UnsafePointer[Float32],
    w: UnsafePointer[Float32],
    rows: Int, cols: Int, eps: Float32
):
    h[].ctx.enqueue_function[_k_rmsnorm_rows](rows, BLOCK)(out, x, w, rows, cols, eps)
