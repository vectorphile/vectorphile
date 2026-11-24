"""
vp_mojo.mojo — vectorphile Mojo SIMD kernels for Linux/MAX backend.

Kernels (all float32, hand-written, no stdlib BLAS/ML deps):
  gemm_f32          — tiled SIMD GEMM with vectorize + parallelize
  gemm_act_f32      — GEMM + fused epilogue activation
  add_bias_rows_f32 — broadcast bias across M rows
  relu_f32          — element-wise ReLU
  gelu_f32          — GeLU (tanh approximation)
  silu_f32          — SiLU (x * sigmoid(x))
  sigmoid_f32       — fast sigmoid
  tanh_f32          — tanh
  relu_bwd_f32      — ReLU backward (gate by saved output)
  transpose_f32     — tiled cache-friendly transpose
  sum_rows_f32      — (M,N) → (N,) row reduction for bias grad

Build:
  cd vectorphile/_core/csrc
  mojo package vp_mojo.mojo -o ../vp_mojo.mojopkg

Usage from Python (via MAX Engine Python bindings):
  See vectorphile/_core/mojo_backend.py
"""

from sys.info import num_physical_cores
from algorithm import vectorize, parallelize, sync_parallelize
from math import sqrt, exp, tanh as _tanh
from memory import UnsafePointer

alias SIMD_WIDTH = 8          # AVX2: 8 x float32 per register
alias MC = 96                 # GEMM micro-panel row tile
alias KC = 256                # GEMM micro-panel depth tile
alias NR = 16                 # GEMM register column tile
alias MR = 6                  # GEMM register row tile

# Post-op constants (mirror vp_core.c)
alias POST_NONE    = 0
alias POST_RELU    = 1
alias POST_GELU    = 2
alias POST_SILU    = 3
alias POST_SIGMOID = 4
alias POST_TANH    = 5


# ── Activation helpers ───────────────────────────────────────────────────────

@always_inline
fn _relu(x: Float32) -> Float32:
    return x if x > 0.0 else 0.0

@always_inline
fn _sigmoid(x: Float32) -> Float32:
    return 1.0 / (1.0 + exp(-x))

@always_inline
fn _gelu(x: Float32) -> Float32:
    """Tanh-approximation GeLU."""
    let k: Float32 = 0.7978845608
    let c = k * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + _tanh(c))

@always_inline
fn _silu(x: Float32) -> Float32:
    return x * _sigmoid(x)

@always_inline
fn _apply_post(x: Float32, post_op: Int) -> Float32:
    if post_op == POST_RELU:    return _relu(x)
    if post_op == POST_GELU:    return _gelu(x)
    if post_op == POST_SILU:    return _silu(x)
    if post_op == POST_SIGMOID: return _sigmoid(x)
    if post_op == POST_TANH:    return _tanh(x)
    return x


# ── Element-wise kernels ─────────────────────────────────────────────────────

fn relu_f32(x: UnsafePointer[Float32], out: UnsafePointer[Float32], n: Int):
    @parameter
    fn _op[w: Int](i: Int):
        let v = x.load[width=w](i)
        out.store[width=w](i, (v > 0.0).select(v, SIMD[DType.float32, w](0.0)))
    vectorize[_op, SIMD_WIDTH](n)

fn gelu_f32(x: UnsafePointer[Float32], out: UnsafePointer[Float32], n: Int):
    for i in range(n):
        out[i] = _gelu(x[i])

fn silu_f32(x: UnsafePointer[Float32], out: UnsafePointer[Float32], n: Int):
    @parameter
    fn _op[w: Int](i: Int):
        let v = x.load[width=w](i)
        let sig = SIMD[DType.float32, w](1.0) / (SIMD[DType.float32, w](1.0) + exp(-v))
        out.store[width=w](i, v * sig)
    vectorize[_op, SIMD_WIDTH](n)

fn sigmoid_f32(x: UnsafePointer[Float32], out: UnsafePointer[Float32], n: Int):
    @parameter
    fn _op[w: Int](i: Int):
        let v = x.load[width=w](i)
        out.store[width=w](i, SIMD[DType.float32, w](1.0) / (SIMD[DType.float32, w](1.0) + exp(-v)))
    vectorize[_op, SIMD_WIDTH](n)

fn tanh_f32(x: UnsafePointer[Float32], out: UnsafePointer[Float32], n: Int):
    for i in range(n):
        out[i] = _tanh(x[i])

fn relu_bwd_f32(grad: UnsafePointer[Float32], saved_out: UnsafePointer[Float32],
                dx: UnsafePointer[Float32], n: Int):
    @parameter
    fn _op[w: Int](i: Int):
        let g = grad.load[width=w](i)
        let o = saved_out.load[width=w](i)
        dx.store[width=w](i, (o > 0.0).select(g, SIMD[DType.float32, w](0.0)))
    vectorize[_op, SIMD_WIDTH](n)


# ── Transpose (tiled, cache-friendly) ────────────────────────────────────────

fn transpose_f32(inp: UnsafePointer[Float32], out: UnsafePointer[Float32], M: Int, N: Int):
    alias T = 32
    for ii in range(0, M, T):
        for jj in range(0, N, T):
            let im = min(ii + T, M)
            let jm = min(jj + T, N)
            for i in range(ii, im):
                for j in range(jj, jm):
                    out[j * M + i] = inp[i * N + j]


# ── Add bias rows: C[i,j] += b[j] ────────────────────────────────────────────

fn add_bias_rows_f32(C: UnsafePointer[Float32], b: UnsafePointer[Float32], M: Int, N: Int):
    @parameter
    fn _row(i: Int):
        @parameter
        fn _op[w: Int](j: Int):
            C.store[width=w](i * N + j, C.load[width=w](i * N + j) + b.load[width=w](j))
        vectorize[_op, SIMD_WIDTH](N)
    parallelize[_row](M, num_physical_cores())


# ── Row-sum: (M,N) → (N,) ────────────────────────────────────────────────────

fn sum_rows_f32(x: UnsafePointer[Float32], out: UnsafePointer[Float32], M: Int, N: Int):
    # Zero output
    @parameter
    fn _zero[w: Int](j: Int):
        out.store[width=w](j, SIMD[DType.float32, w](0.0))
    vectorize[_zero, SIMD_WIDTH](N)
    # Accumulate
    for i in range(M):
        @parameter
        fn _acc[w: Int](j: Int):
            out.store[width=w](j, out.load[width=w](j) + x.load[width=w](i * N + j))
        vectorize[_acc, SIMD_WIDTH](N)


# ── GEMM: C = A @ B  (M×K) @ (K×N) → (M×N) ─────────────────────────────────
# Tiled micro-kernel approach: parallelize over MC-row panels,
# SIMD-vectorize the NR=16 column accumulate per MR=6 row micro-tile.

fn _micro_kernel(
    kc: Int,
    Ap: UnsafePointer[Float32],   # (MR, kc) packed panel
    Bp: UnsafePointer[Float32],   # (kc, NR) packed panel
    C:  UnsafePointer[Float32],   # output row (C + ir*N + jc)
    ldc: Int,
    mr: Int, nr: Int,
    post_op: Int,
    is_last_kc: Bool,
):
    """MR×NR micro-kernel: accumulate kc depth slices, write to C."""
    var acc = StaticTuple[SIMD[DType.float32, 8], 12]()
    for r in range(12):
        acc[r] = SIMD[DType.float32, 8](0.0)

    # Accumulate
    for p in range(kc):
        let b0 = Bp.load[width=8](p * NR)
        let b1 = Bp.load[width=8](p * NR + 8)
        for r in range(mr):
            let a_val = SIMD[DType.float32, 8](Ap[r * kc + p])
            acc[r * 2]     = a_val.fma(b0, acc[r * 2])
            acc[r * 2 + 1] = a_val.fma(b1, acc[r * 2 + 1])

    # Write + optional activation
    for r in range(mr):
        let row_ptr = C + r * ldc
        var v0 = acc[r * 2];     var v1 = acc[r * 2 + 1]
        # Add existing C (beta=1 for multi-KC pass)
        v0 = v0 + row_ptr.load[width=8](0)
        v1 = v1 + row_ptr.load[width=8](8)
        if is_last_kc and post_op != POST_NONE:
            # Apply activation scalar (vectorized form below for relu)
            if post_op == POST_RELU:
                let zero = SIMD[DType.float32, 8](0.0)
                v0 = (v0 > zero).select(v0, zero)
                v1 = (v1 > zero).select(v1, zero)
            else:
                # Scalar fallback for other activations
                for j in range(nr if nr < 8 else 8):
                    row_ptr[j] = _apply_post(v0[j], post_op)
                for j in range(max(0, nr - 8)):
                    row_ptr[8 + j] = _apply_post(v1[j], post_op)
                continue
        row_ptr.store[width=8](0, v0) if nr >= 8 else _store_partial(row_ptr, v0, nr)
        row_ptr.store[width=8](8, v1) if nr > 8 else None


fn _store_partial(ptr: UnsafePointer[Float32], v: SIMD[DType.float32, 8], n: Int):
    for i in range(n):
        ptr[i] = v[i]


fn gemm_f32(
    A: UnsafePointer[Float32],
    B: UnsafePointer[Float32],
    C: UnsafePointer[Float32],
    M: Int, K: Int, N: Int,
):
    gemm_act_f32(A, B, C, M, K, N, POST_NONE)


fn gemm_act_f32(
    A: UnsafePointer[Float32],
    B: UnsafePointer[Float32],
    C: UnsafePointer[Float32],
    M: Int, K: Int, N: Int,
    post_op: Int,
):
    """
    Tiled GEMM with optional fused epilogue activation.
    Tiles: MC×KC×NC. Parallelized over MC-row panels (outer rows).
    """
    alias NC = 256

    # Adaptive NC for small-M (doubles tile count → more parallelism)
    let nc_tile = NC // 2 if M <= MC else NC

    @parameter
    fn _panel_i(ic: Int):
        let mc = min(MC, M - ic * MC)
        for pc in range(0, K, KC):
            let kc = min(KC, K - pc)
            let is_last_kc = (pc + kc >= K)
            # Pack A panel: (mc, kc)
            var Ap_data = UnsafePointer[Float32].alloc(MC * KC)
            for i in range(mc):
                for p in range(kc):
                    Ap_data[i * kc + p] = A[(ic * MC + i) * K + pc + p]

            for jc in range(0, N, nc_tile):
                let nc = min(nc_tile, N - jc)
                # Pack B panel: (kc, nc)
                var Bp_data = UnsafePointer[Float32].alloc(KC * NC)
                for p in range(kc):
                    for j in range(nc):
                        Bp_data[p * nc + j] = B[(pc + p) * N + jc + j]

                # Micro-kernel loop
                var ir = 0
                while ir < mc:
                    let mr = min(MR, mc - ir)
                    var jr = 0
                    while jr < nc:
                        let nr = min(NR, nc - jr)
                        # Zero C tile on first KC pass
                        if pc == 0:
                            for i in range(mr):
                                for j in range(nr):
                                    C[(ic * MC + ir + i) * N + jc + jr + j] = 0.0
                        _micro_kernel(
                            kc,
                            Ap_data + ir * kc,
                            Bp_data + jr,          # strided: B cols jr..jr+nr
                            C + (ic * MC + ir) * N + jc + jr,
                            N,
                            mr, nr,
                            post_op if is_last_kc else POST_NONE,
                            is_last_kc,
                        )
                        jr += NR
                    ir += MR

                Bp_data.free()
            Ap_data.free()

    let n_panels = (M + MC - 1) // MC
    sync_parallelize[_panel_i](n_panels, num_physical_cores())
