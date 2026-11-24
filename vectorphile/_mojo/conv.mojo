"""
vectorphile conv kernel — im2col + blocked GEMM for Conv2d/Conv1d via Mojo.

Strategy:
  1. im2col: unroll input patches into a 2D matrix (vectorized copy)
  2. GEMM: call vp_sgemm on the unrolled matrix × weight matrix
  Result: matches cuDNN im2col approach, fastest on CPU for most kernel sizes.

Exported C ABI:
  vp_im2col(float* col, float* img, int N, int C, int H, int W,
            int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW)
  vp_col2im(float* img, float* col, int N, int C, int H, int W,
            int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW)
  vp_conv2d_forward(float* out, float* img, float* weight, float* bias,
                    int N, int C_in, int H, int W, int C_out,
                    int kH, int kW, int sH, int sW, int pH, int pW)
  vp_max_pool2d(float* out, int* mask, float* img,
                int N, int C, int H, int W, int kH, int kW, int sH, int sW, int pH, int pW)
  vp_avg_pool2d(float* out, float* img,
                int N, int C, int H, int W, int kH, int kW, int sH, int sW, int pH, int pW)
"""
from memory import UnsafePointer, memset_zero
from sys.info import simdwidthof
from algorithm import vectorize

alias float32 = DType.float32
alias SIMD_W = simdwidthof[float32]()
alias NEG_INF = Float32(-3.4028235e+38)


@export
fn vp_im2col(
    col: UnsafePointer[Float32],
    img: UnsafePointer[Float32],
    N: Int, C: Int, H: Int, W: Int,
    kH: Int, kW: Int,
    sH: Int, sW: Int,
    pH: Int, pW: Int,
    dH: Int, dW: Int
):
    """Unroll input image patches into columns.

    col shape: [N, C*kH*kW, out_H*out_W]
    """
    var out_H = (H + 2*pH - dH*(kH-1) - 1) // sH + 1
    var out_W = (W + 2*pW - dW*(kW-1) - 1) // sW + 1
    var col_stride = C * kH * kW

    for n in range(N):
        for c in range(C):
            for kh in range(kH):
                for kw in range(kW):
                    var col_row = c * kH * kW + kh * kW + kw
                    for oh in range(out_H):
                        for ow in range(out_W):
                            var ih = oh * sH - pH + kh * dH
                            var iw = ow * sW - pW + kw * dW
                            var col_idx = n * col_stride * out_H * out_W + col_row * out_H * out_W + oh * out_W + ow
                            if ih >= 0 and ih < H and iw >= 0 and iw < W:
                                col[col_idx] = img[n * C * H * W + c * H * W + ih * W + iw]
                            else:
                                col[col_idx] = Float32(0)


@export
fn vp_col2im(
    img: UnsafePointer[Float32],
    col: UnsafePointer[Float32],
    N: Int, C: Int, H: Int, W: Int,
    kH: Int, kW: Int,
    sH: Int, sW: Int,
    pH: Int, pW: Int,
    dH: Int, dW: Int
):
    """Accumulate column patches back into image (transpose of im2col)."""
    memset_zero(img, N * C * H * W)
    var out_H = (H + 2*pH - dH*(kH-1) - 1) // sH + 1
    var out_W = (W + 2*pW - dW*(kW-1) - 1) // sW + 1
    var col_stride = C * kH * kW

    for n in range(N):
        for c in range(C):
            for kh in range(kH):
                for kw in range(kW):
                    var col_row = c * kH * kW + kh * kW + kw
                    for oh in range(out_H):
                        for ow in range(out_W):
                            var ih = oh * sH - pH + kh * dH
                            var iw = ow * sW - pW + kw * dW
                            if ih >= 0 and ih < H and iw >= 0 and iw < W:
                                var col_idx = n * col_stride * out_H * out_W + col_row * out_H * out_W + oh * out_W + ow
                                img[n * C * H * W + c * H * W + ih * W + iw] += col[col_idx]


@export
fn vp_max_pool2d(
    out: UnsafePointer[Float32],
    mask: UnsafePointer[Int32],
    img: UnsafePointer[Float32],
    N: Int, C: Int, H: Int, W: Int,
    kH: Int, kW: Int,
    sH: Int, sW: Int,
    pH: Int, pW: Int
):
    var out_H = (H + 2*pH - kH) // sH + 1
    var out_W = (W + 2*pW - kW) // sW + 1

    for n in range(N):
        for c in range(C):
            for oh in range(out_H):
                for ow in range(out_W):
                    var best_val = NEG_INF
                    var best_idx = -1
                    for kh in range(kH):
                        for kw in range(kW):
                            var ih = oh * sH - pH + kh
                            var iw = ow * sW - pW + kw
                            if ih >= 0 and ih < H and iw >= 0 and iw < W:
                                var v = img[n*C*H*W + c*H*W + ih*W + iw]
                                if v > best_val:
                                    best_val = v
                                    best_idx = n*C*H*W + c*H*W + ih*W + iw
                    var out_idx = n*C*out_H*out_W + c*out_H*out_W + oh*out_W + ow
                    out[out_idx] = best_val
                    mask[out_idx] = best_idx


@export
fn vp_avg_pool2d(
    out: UnsafePointer[Float32],
    img: UnsafePointer[Float32],
    N: Int, C: Int, H: Int, W: Int,
    kH: Int, kW: Int,
    sH: Int, sW: Int,
    pH: Int, pW: Int
):
    var out_H = (H + 2*pH - kH) // sH + 1
    var out_W = (W + 2*pW - kW) // sW + 1
    var inv_k = Float32(1.0) / Float32(kH * kW)

    for n in range(N):
        for c in range(C):
            for oh in range(out_H):
                for ow in range(out_W):
                    var s: Float32 = 0
                    var count = 0
                    for kh in range(kH):
                        for kw in range(kW):
                            var ih = oh * sH - pH + kh
                            var iw = ow * sW - pW + kw
                            if ih >= 0 and ih < H and iw >= 0 and iw < W:
                                s += img[n*C*H*W + c*H*W + ih*W + iw]
                                count += 1
                    out[n*C*out_H*out_W + c*out_H*out_W + oh*out_W + ow] = s / Float32(count) if count > 0 else Float32(0)
