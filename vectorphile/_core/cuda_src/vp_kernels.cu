/*
 * vp_kernels.cu — vectorphile GPU kernels compiled to PTX.
 * Compile with:  nvcc -ptx -O3 -arch=compute_89 vp_kernels.cu -o vp_kernels.ptx
 * (no Python headers needed — this is device code only)
 */

#include <math.h>

#define BLOCK 256
#define TILE  16

extern "C" {

/* ── elementwise ─────────────────────────────────────────────────────────── */

__global__ void vp_relu(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.f ? x[i] : 0.f;
}

__global__ void vp_gelu(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = 0.5f * v * (1.f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

__global__ void vp_silu(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) { float v = x[i]; y[i] = v / (1.f + expf(-v)); }
}

__global__ void vp_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void vp_mul(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

__global__ void vp_scale(const float* x, float* y, float s, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) y[i] = x[i] * s;
}

/* ── softmax (row-wise) ───────────────────────────────────────────────────── */

__global__ void vp_softmax(const float* __restrict__ x, float* __restrict__ y,
                           int N, int C) {
    int row = blockIdx.x;
    if (row >= N) return;
    const float* rx = x + row * C;
    float* ry = y + row * C;
    __shared__ float smem[BLOCK];

    float mx = -1e38f;
    for (int c = threadIdx.x; c < C; c += BLOCK) mx = fmaxf(mx, rx[c]);
    smem[threadIdx.x] = mx; __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x+s]);
        __syncthreads();
    }
    mx = smem[0];

    float sum = 0.f;
    for (int c = threadIdx.x; c < C; c += BLOCK) sum += expf(rx[c] - mx);
    smem[threadIdx.x] = sum; __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x+s];
        __syncthreads();
    }
    float inv_s = 1.f / smem[0];
    for (int c = threadIdx.x; c < C; c += BLOCK) ry[c] = expf(rx[c] - mx) * inv_s;
}

/* ── cross-entropy forward ───────────────────────────────────────────────── */

__global__ void vp_ce_fwd(const float* __restrict__ logits,
                          const int*   __restrict__ targets,
                          float*       __restrict__ losses,
                          int N, int C) {
    int row = blockIdx.x;
    if (row >= N) return;
    const float* r = logits + row * C;
    float mx = -1e38f;
    for (int c = 0; c < C; c++) mx = fmaxf(mx, r[c]);
    float sum = 0.f;
    for (int c = 0; c < C; c++) sum += expf(r[c] - mx);
    losses[row] = -(r[targets[row]] - mx - logf(sum));
}

/* ── cross-entropy backward ──────────────────────────────────────────────── */

__global__ void vp_ce_bwd(const float* __restrict__ logits,
                          const int*   __restrict__ targets,
                          float*       __restrict__ grad,
                          int N, int C, float scale) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row >= N || col >= C) return;
    const float* r = logits + row * C;
    float mx = -1e38f;
    for (int c = 0; c < C; c++) mx = fmaxf(mx, r[c]);
    float sum = 0.f;
    for (int c = 0; c < C; c++) sum += expf(r[c] - mx);
    float p = expf(r[col] - mx) / sum;
    grad[row * C + col] = (p - (col == targets[row] ? 1.f : 0.f)) * scale;
}

/* ── tiled matmul ────────────────────────────────────────────────────────── */

__global__ void vp_matmul(const float* __restrict__ A,
                          const float* __restrict__ B,
                          float*       __restrict__ C,
                          int M, int K, int N) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int ak = t * TILE + threadIdx.x;
        int bk = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[row*K+ak] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[bk*N+col] : 0.f;
        __syncthreads();
        for (int i = 0; i < TILE; i++) acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row*N+col] = acc;
}

/* ── layer norm ──────────────────────────────────────────────────────────── */

__global__ void vp_layernorm(const float* __restrict__ x,
                             const float* __restrict__ w,
                             const float* __restrict__ b,
                             float* __restrict__ y,
                             int N, int C, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;
    const float* rx = x + row * C;
    float* ry = y + row * C;
    float mean = 0.f;
    for (int c = 0; c < C; c++) mean += rx[c];
    mean /= C;
    float var = 0.f;
    for (int c = 0; c < C; c++) { float d = rx[c]-mean; var += d*d; }
    float inv_std = rsqrtf(var/C + eps);
    for (int c = threadIdx.x; c < C; c += BLOCK)
        ry[c] = (rx[c]-mean)*inv_std * (w ? w[c] : 1.f) + (b ? b[c] : 0.f);
}

/* ── embedding ───────────────────────────────────────────────────────────── */

__global__ void vp_embedding(const float* __restrict__ weight,
                             const int*   __restrict__ idx,
                             float*       __restrict__ out,
                             int B, int D) {
    int b = blockIdx.x;
    if (b >= B) return;
    for (int d = threadIdx.x; d < D; d += BLOCK)
        out[b*D+d] = weight[idx[b]*D+d];
}

} /* extern "C" */
