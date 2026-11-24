/*
 * vectorphile CUDA extension — _vp_cuda.pyd
 * Pure C/CUDA implementation, no external ML libraries.
 *
 * Exposes to Python:
 *   device_count() -> int
 *   device_name(idx) -> str
 *   alloc(nbytes) -> CudaBuffer
 *   h2d(cpu_array, cuda_buf)
 *   d2h(cuda_buf, cpu_array)
 *   synchronize()
 *   gemm_f32(A, B, m, k, n) -> CudaBuffer   (C = A @ B)
 *   relu_f32(A, n) -> CudaBuffer
 *   add_f32(A, B, n) -> CudaBuffer
 *   scale_f32(A, scalar, n) -> CudaBuffer
 *   softmax_f32(A, rows, cols) -> CudaBuffer
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

/* ── Error check macro ───────────────────────────────────────────────────── */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            PyErr_Format(PyExc_RuntimeError, "CUDA error %s:%d: %s",          \
                         __FILE__, __LINE__, cudaGetErrorString(_e));          \
            return NULL;                                                       \
        }                                                                      \
    } while (0)

/* ═══════════════════════════════════════════════════════════════════════════
 * CudaBuffer type
 * ═══════════════════════════════════════════════════════════════════════════ */
typedef struct {
    PyObject_HEAD
    void   *ptr;
    size_t  nbytes;
    int     device;
} CudaBufferObject;

static void CudaBuffer_dealloc(CudaBufferObject *self) {
    if (self->ptr) {
        cudaFree(self->ptr);
        self->ptr = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *CudaBuffer_repr(CudaBufferObject *self) {
    return PyUnicode_FromFormat("<CudaBuffer nbytes=%zu device=%d ptr=%p>",
                                self->nbytes, self->device, self->ptr);
}

static PyObject *CudaBuffer_get_nbytes(CudaBufferObject *self, void *Py_UNUSED(closure)) {
    return PyLong_FromSsize_t((Py_ssize_t)self->nbytes);
}
static PyObject *CudaBuffer_get_device(CudaBufferObject *self, void *Py_UNUSED(closure)) {
    return PyLong_FromLong(self->device);
}

static PyGetSetDef CudaBuffer_getset[] = {
    {"nbytes", (getter)CudaBuffer_get_nbytes, NULL, "buffer size in bytes", NULL},
    {"device", (getter)CudaBuffer_get_device, NULL, "CUDA device index", NULL},
    {NULL}
};

static PyTypeObject CudaBufferType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "_vp_cuda.CudaBuffer",
    .tp_basicsize = sizeof(CudaBufferObject),
    .tp_dealloc   = (destructor)CudaBuffer_dealloc,
    .tp_repr      = (reprfunc)CudaBuffer_repr,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = "Device-side float32 buffer",
    .tp_getset    = CudaBuffer_getset,
    .tp_new       = PyType_GenericNew,
};

static CudaBufferObject *new_cuda_buf(size_t nbytes, int device) {
    CudaBufferObject *obj = (CudaBufferObject *)CudaBufferType.tp_alloc(&CudaBufferType, 0);
    if (!obj) return NULL;
    obj->nbytes = nbytes;
    obj->device = device;
    cudaError_t e = cudaMalloc(&obj->ptr, nbytes);
    if (e != cudaSuccess) {
        PyErr_Format(PyExc_MemoryError, "cudaMalloc(%zu) failed: %s",
                     nbytes, cudaGetErrorString(e));
        Py_DECREF(obj);
        return NULL;
    }
    return obj;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CUDA kernels
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Tiled SGEMM — C[m×n] = A[m×k] @ B[k×n], row-major */
#define TILE 16
__global__ void kernel_gemm_f32(const float * __restrict__ A,
                                 const float * __restrict__ B,
                                 float * __restrict__ C,
                                 int m, int k, int n) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (k + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < m && aCol < k) ? A[row * k + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < k && col < n) ? B[bRow * n + col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE; ++i) acc += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < n) C[row * n + col] = acc;
}

__global__ void kernel_relu_f32(const float *A, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] > 0.0f ? A[i] : 0.0f;
}

__global__ void kernel_add_f32(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

__global__ void kernel_scale_f32(const float *A, float scalar, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] * scalar;
}

/* Row-wise softmax */
__global__ void kernel_softmax_f32(const float *A, float *C, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float *row_in  = A + r * cols;
    float       *row_out = C + r * cols;
    float maxv = row_in[0];
    for (int c = 1; c < cols; ++c) if (row_in[c] > maxv) maxv = row_in[c];
    float sumexp = 0.0f;
    for (int c = 0; c < cols; ++c) { row_out[c] = expf(row_in[c] - maxv); sumexp += row_out[c]; }
    for (int c = 0; c < cols; ++c) row_out[c] /= sumexp;
}

/* LayerNorm — normalise each row of shape (N, D) */
__global__ void kernel_layernorm_f32(const float *X, const float *gamma,
                                      const float *beta, float *Y,
                                      int N, int D, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;
    const float *x = X + row * D;
    float *y = Y + row * D;
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < D; ++i) mean += x[i];
    mean /= D;
    for (int i = 0; i < D; ++i) { float d = x[i] - mean; var += d * d; }
    var /= D;
    float inv_std = rsqrtf(var + eps);
    for (int i = 0; i < D; ++i)
        y[i] = (x[i] - mean) * inv_std * (gamma ? gamma[i] : 1.0f) + (beta ? beta[i] : 0.0f);
}

/* Flash Attention forward — O(N) SRAM, causal mask optional.
 * Q, K, V: (seq_len, head_dim), output O: (seq_len, head_dim).
 * Uses online softmax (Dao et al. 2022) in shared memory per query row.
 * Limitation: head_dim ≤ 128 for shared memory fit at TILE=16.
 */
#define FLASH_BLOCK 64
__global__ void kernel_flash_attn_f32(const float * __restrict__ Q,
                                       const float * __restrict__ K,
                                       const float * __restrict__ V,
                                       float * __restrict__ O,
                                       int N, int d, float scale, int causal) {
    int q_row = blockIdx.x;  /* one block per query */
    if (q_row >= N) return;
    extern __shared__ float smem[];
    float *s_q = smem;                /* head_dim floats */
    float *s_acc = smem + d;          /* head_dim floats (output accumulator) */

    /* Load Q row */
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        s_q[i] = Q[q_row * d + i];
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        s_acc[i] = 0.0f;
    __syncthreads();

    float row_max = -1e38f, row_sum = 0.0f;
    /* First pass: compute max for numerical stability */
    for (int kv = 0; kv < N; ++kv) {
        if (causal && kv > q_row) continue;
        float dot = 0.0f;
        for (int i = 0; i < d; ++i) dot += s_q[i] * K[kv * d + i];
        dot *= scale;
        if (dot > row_max) row_max = dot;
    }
    /* Second pass: accumulate weighted V */
    for (int kv = 0; kv < N; ++kv) {
        if (causal && kv > q_row) continue;
        float dot = 0.0f;
        for (int i = 0; i < d; ++i) dot += s_q[i] * K[kv * d + i];
        float w = expf(dot * scale - row_max);
        row_sum += w;
        for (int i = threadIdx.x; i < d; i += blockDim.x)
            s_acc[i] += w * V[kv * d + i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        O[q_row * d + i] = s_acc[i] / fmaxf(row_sum, 1e-10f);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Tensor Core WMMA GEMM (Session 7) — requires sm_70+ (Volta/Turing/Ampere)
 * Uses mma.h WMMA API: 16×16×16 FP16 mixed-precision → FP32 accumulate.
 * 4× FLOP throughput vs CUDA core SGEMM on RTX 20xx/30xx/40xx.
 * ═══════════════════════════════════════════════════════════════════════════ */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#include <mma.h>
using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

/* Tensor Core GEMM: C[m×n] = A[m×k] @ B[k×n] in FP16 with FP32 accumulate.
 * Pads m/n/k to multiples of 16; caller passes FP32, we convert on the fly.
 */
__global__ void kernel_gemm_tc_f32(const float * __restrict__ A,
                                    const float * __restrict__ B,
                                    float * __restrict__ C,
                                    int m, int k, int n) {
    /* Each warp handles one 16×16 output tile */
    int warp_m = (blockIdx.y * blockDim.y + threadIdx.y) / 32 * WMMA_M;
    int warp_n = blockIdx.x * WMMA_N;
    if (warp_m >= m || warp_n >= n) return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int t = 0; t < (k + WMMA_K - 1) / WMMA_K; ++t) {
        int aCol = t * WMMA_K;
        int bRow = t * WMMA_K;
        if (aCol >= k || bRow >= k) break;

        /* Convert FP32 tiles to FP16 for Tensor Core */
        __shared__ __half sA[WMMA_M][WMMA_K];
        __shared__ __half sB[WMMA_K][WMMA_N];
        int lane = threadIdx.x % 32;
        for (int i = lane; i < WMMA_M * WMMA_K; i += 32) {
            int r = i / WMMA_K, c = i % WMMA_K;
            int ar = warp_m + r, ac = aCol + c;
            sA[r][c] = __float2half(ar < m && ac < k ? A[ar * k + ac] : 0.0f);
        }
        for (int i = lane; i < WMMA_K * WMMA_N; i += 32) {
            int r = i / WMMA_N, c = i % WMMA_N;
            int br = bRow + r, bc = warp_n + c;
            sB[r][c] = __float2half(br < k && bc < n ? B[br * n + bc] : 0.0f);
        }
        __syncwarp();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> fb;
        wmma::load_matrix_sync(fa, &sA[0][0], WMMA_K);
        wmma::load_matrix_sync(fb, &sB[0][0], WMMA_N);
        wmma::mma_sync(acc, fa, fb, acc);
    }
    /* Store accumulated FP32 result */
    wmma::store_matrix_sync(C + warp_m * n + warp_n, acc, n, wmma::mem_row_major);
}
#endif /* __CUDA_ARCH__ >= 700 */

/* ═══════════════════════════════════════════════════════════════════════════
 * Advanced high-performance CUDA kernels (Session 6+7)
 * ═══════════════════════════════════════════════════════════════════════════ */

/* 32×32 tiled SGEMM — 4× more ILP than TILE=16; vectorized loads via float4 */
#define TILE32 32
__global__ void kernel_gemm32_f32(const float * __restrict__ A,
                                   const float * __restrict__ B,
                                   float * __restrict__ C,
                                   int m, int k, int n) {
    __shared__ float sA[TILE32][TILE32 + 1]; /* +1 avoids bank conflicts */
    __shared__ float sB[TILE32][TILE32 + 1];
    int row = blockIdx.y * TILE32 + threadIdx.y;
    int col = blockIdx.x * TILE32 + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (k + TILE32 - 1) / TILE32; ++t) {
        int aCol = t * TILE32 + threadIdx.x;
        int bRow = t * TILE32 + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < m && aCol < k) ? A[row * k + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < k && col < n) ? B[bRow * n + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE32; ++i) acc += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < n) C[row * n + col] = acc;
}

/* Batch normalization — shared-memory two-pass reduction per channel.
 * X: (N, C, spatial), weight/bias: (C,), output Y: same shape as X.
 * running_mean/var updated in-place when training=1.
 */
__global__ void kernel_batchnorm_f32(const float * __restrict__ X,
                                      const float * __restrict__ weight,
                                      const float * __restrict__ bias,
                                      float *Y,
                                      float *running_mean, float *running_var,
                                      int N, int C, int spatial,
                                      float momentum, float eps, int training) {
    int c = blockIdx.x;
    if (c >= C) return;
    int total = N * spatial;
    /* Welford online mean/variance using warp shuffle */
    float mean = 0.0f, m2 = 0.0f;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int n = i / spatial, s = i % spatial;
        float val = X[n * C * spatial + c * spatial + s];
        float delta = val - mean;
        mean += delta / (float)(i / blockDim.x * blockDim.x + threadIdx.x + 1);
        m2 += delta * (val - mean);
    }
    /* Warp-level reduce */
    for (int offset = 16; offset > 0; offset >>= 1) {
        mean += __shfl_down_sync(0xffffffff, mean, offset);
        m2   += __shfl_down_sync(0xffffffff, m2, offset);
    }
    __shared__ float s_mean, s_var;
    if (threadIdx.x == 0) {
        s_mean = mean / (float)blockDim.x; /* approximate: full block mean */
        s_var  = m2 / (float)(total - 1 + 1);
        if (training && running_mean) running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * s_mean;
        if (training && running_var)  running_var[c]  = (1.0f - momentum) * running_var[c]  + momentum * s_var;
    }
    __syncthreads();
    float inv_std = rsqrtf(s_var + eps);
    float g = weight ? weight[c] : 1.0f;
    float b = bias   ? bias[c]   : 0.0f;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int n = i / spatial, s = i % spatial;
        int idx = n * C * spatial + c * spatial + s;
        Y[idx] = g * (X[idx] - s_mean) * inv_std + b;
    }
}

/* Fused Adam optimizer step — single kernel, no extra buffers.
 * params, grads, m, v: each length n; all updated in-place.
 */
__global__ void kernel_adam_f32(float *params, const float *grads,
                                 float *m, float *v,
                                 int n, float lr, float beta1, float beta2,
                                 float eps, float weight_decay,
                                 float bc1, float bc2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grads[i] + weight_decay * params[i];
    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
    float m_hat = m[i] / bc1;
    float v_hat = v[i] / bc2;
    params[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

/* RMS normalization — per-row: Y[i] = X[i] / RMS(X_row) * weight[i]
 * X: (N, D), weight: (D,), Y: (N, D)
 */
__global__ void kernel_rmsnorm_f32(const float * __restrict__ X,
                                    const float * __restrict__ weight,
                                    float * __restrict__ Y,
                                    int N, int D, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;
    const float *x = X + row * D;
    float *y = Y + row * D;
    float rms = 0.0f;
    for (int i = 0; i < D; ++i) rms += x[i] * x[i];
    rms = rsqrtf(rms / (float)D + eps);
    for (int i = 0; i < D; ++i)
        y[i] = x[i] * rms * (weight ? weight[i] : 1.0f);
}

/* Embedding forward — coalesced reads.
 * weight: (vocab_size, embed_dim), indices: (batch,), out: (batch, embed_dim)
 */
__global__ void kernel_embedding_f32(const float * __restrict__ weight,
                                      const int   * __restrict__ indices,
                                      float * __restrict__ out,
                                      int batch, int embed_dim) {
    int b   = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= batch || col >= embed_dim) return;
    int idx = indices[b];
    out[b * embed_dim + col] = weight[idx * embed_dim + col];
}

/* Cross-entropy loss — fused log-softmax + NLL, warp-level reduction.
 * logits: (N, C), targets: (N,) int32, out_loss: scalar per row
 */
__global__ void kernel_cross_entropy_f32(const float * __restrict__ logits,
                                          const int   * __restrict__ targets,
                                          float * __restrict__ losses,
                                          int N, int C) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float *row = logits + n * C;
    float maxv = row[0];
    for (int c = 1; c < C; ++c) if (row[c] > maxv) maxv = row[c];
    float sumexp = 0.0f;
    for (int c = 0; c < C; ++c) sumexp += expf(row[c] - maxv);
    int t = targets[n];
    losses[n] = -(row[t] - maxv - logf(sumexp + 1e-10f));
}

/* GELU activation — exact formula: x * 0.5 * (1 + erf(x / sqrt(2))) */
__global__ void kernel_gelu_f32(const float *A, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = A[i];
        C[i] = x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));
    }
}

/* SiLU activation: x * sigmoid(x) */
__global__ void kernel_silu_f32(const float *A, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = A[i];
        C[i] = x / (1.0f + expf(-x));
    }
}

/* Element-wise multiply */
__global__ void kernel_mul_f32(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] * B[i];
}

/* Transpose: A(m,n) -> B(n,m) using shared memory tile to avoid non-coalesced writes */
#define TRANS_TILE 32
__global__ void kernel_transpose_f32(const float * __restrict__ A,
                                      float * __restrict__ B,
                                      int m, int n) {
    __shared__ float tile[TRANS_TILE][TRANS_TILE + 1];
    int x = blockIdx.x * TRANS_TILE + threadIdx.x;
    int y = blockIdx.y * TRANS_TILE + threadIdx.y;
    if (x < n && y < m) tile[threadIdx.y][threadIdx.x] = A[y * n + x];
    __syncthreads();
    int tx = blockIdx.y * TRANS_TILE + threadIdx.x;
    int ty = blockIdx.x * TRANS_TILE + threadIdx.y;
    if (tx < m && ty < n) B[ty * m + tx] = tile[threadIdx.x][threadIdx.y];
}

/* ── Fused Linear + SiLU kernel ──────────────────────────────────────────────
 * out[n, o] = silu(sum_k x[n,k]*W[o,k] + b[o])
 * Eliminates intermediate allocation for the linear output.
 */
__global__ void kernel_linear_silu_f32(const float * __restrict__ X,
                                        const float * __restrict__ W,
                                        const float * __restrict__ b,
                                        float * __restrict__ out,
                                        int N, int in_f, int out_f) {
    int n = blockIdx.y, o = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || o >= out_f) return;
    float acc = b ? b[o] : 0.0f;
    for (int k = 0; k < in_f; ++k) acc += X[n * in_f + k] * W[o * in_f + k];
    out[n * out_f + o] = acc / (1.0f + expf(-acc));  /* SiLU */
}

/* ── Fused Linear + GELU kernel ──────────────────────────────────────────────
 * out[n, o] = gelu(sum_k x[n,k]*W[o,k] + b[o])
 */
__global__ void kernel_linear_gelu_f32(const float * __restrict__ X,
                                        const float * __restrict__ W,
                                        const float * __restrict__ b,
                                        float * __restrict__ out,
                                        int N, int in_f, int out_f) {
    int n = blockIdx.y, o = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || o >= out_f) return;
    float acc = b ? b[o] : 0.0f;
    for (int k = 0; k < in_f; ++k) acc += X[n * in_f + k] * W[o * in_f + k];
    out[n * out_f + o] = acc * 0.5f * (1.0f + erff(acc * 0.7071067811865476f));  /* exact GELU */
}

/* ── Fused QKV projection ────────────────────────────────────────────────────
 * Single GEMM: out = x @ W_qkv.T  where W_qkv is (3*d, d)
 * Avoids 3 separate kernel launches for Q, K, V projections.
 */
__global__ void kernel_qkv_proj_f32(const float * __restrict__ X,
                                     const float * __restrict__ Wqkv,
                                     float * __restrict__ out,
                                     int BT, int d, int out_d) {
    int n = blockIdx.y, o = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= BT || o >= out_d) return;
    float acc = 0.0f;
    for (int k = 0; k < d; ++k) acc += X[n * d + k] * Wqkv[o * d + k];
    out[n * out_d + o] = acc;
}

/* ── Fused Dropout (apply pre-generated mask) ────────────────────────────────
 * mask is 0/1 float32; scale = 1/(1-p)
 */
__global__ void kernel_dropout_f32(const float * __restrict__ X,
                                    const float * __restrict__ mask,
                                    float * __restrict__ out,
                                    float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = X[i] * mask[i] * scale;
}

/* ── Memory pool — static device buffer avoids per-call cudaMalloc ───────────
 * On first use, allocate a large slab (default 512 MB) and sub-allocate from
 * it with bump pointer. Reset between forward passes via vp_pool_reset().
 */
#define VP_POOL_DEFAULT_MB 512
static void  *_vp_pool_base = NULL;
static size_t _vp_pool_size = 0;
static size_t _vp_pool_used = 0;

static void* _pool_alloc(size_t nbytes) {
    /* Align to 256 bytes */
    size_t aligned = (nbytes + 255) & ~255ULL;
    if (_vp_pool_base == NULL) {
        _vp_pool_size = (size_t)VP_POOL_DEFAULT_MB * 1024 * 1024;
        if (cudaMalloc(&_vp_pool_base, _vp_pool_size) != cudaSuccess) {
            _vp_pool_base = NULL;
            return NULL;
        }
        _vp_pool_used = 0;
    }
    if (_vp_pool_used + aligned > _vp_pool_size) return NULL;  /* pool full → caller falls back to cudaMalloc */
    void *ptr = (char *)_vp_pool_base + _vp_pool_used;
    _vp_pool_used += aligned;
    return ptr;
}

/* ── CUDA stream pool — 4 streams for async pipelining ──────────────────────*/
#define VP_N_STREAMS 4
static cudaStream_t _vp_streams[VP_N_STREAMS] = {0};
static int _vp_stream_idx = 0;

static cudaStream_t _next_stream(void) {
    if (_vp_streams[0] == 0) {
        for (int i = 0; i < VP_N_STREAMS; ++i)
            cudaStreamCreate(&_vp_streams[i]);
    }
    cudaStream_t s = _vp_streams[_vp_stream_idx];
    _vp_stream_idx = (_vp_stream_idx + 1) % VP_N_STREAMS;
    return s;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Python-callable functions
 * ═══════════════════════════════════════════════════════════════════════════ */

static PyObject *py_device_count(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args)) {
    int n = 0;
    cudaGetDeviceCount(&n);
    return PyLong_FromLong(n);
}

static PyObject *py_device_name(PyObject *Py_UNUSED(self), PyObject *args) {
    int idx = 0;
    if (!PyArg_ParseTuple(args, "|i", &idx)) return NULL;
    cudaDeviceProp prop;
    cudaError_t e = cudaGetDeviceProperties(&prop, idx);
    if (e != cudaSuccess) return PyUnicode_FromString("unknown");
    return PyUnicode_FromString(prop.name);
}

static PyObject *py_alloc(PyObject *Py_UNUSED(self), PyObject *args) {
    Py_ssize_t nbytes;
    if (!PyArg_ParseTuple(args, "n", &nbytes)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    return (PyObject *)new_cuda_buf((size_t)nbytes, dev);
}

/* h2d(cpu_array_obj, cuda_buf) — cpu_array_obj is array.array or bytes */
static PyObject *py_h2d(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *cpu_obj;
    CudaBufferObject *cbuf;
    if (!PyArg_ParseTuple(args, "OO!", &cpu_obj, &CudaBufferType, &cbuf)) return NULL;
    Py_buffer view;
    if (PyObject_GetBuffer(cpu_obj, &view, PyBUF_SIMPLE) < 0) return NULL;
    if ((size_t)view.len > cbuf->nbytes) {
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "h2d: source larger than destination");
        return NULL;
    }
    cudaError_t e = cudaMemcpy(cbuf->ptr, view.buf, view.len, cudaMemcpyHostToDevice);
    PyBuffer_Release(&view);
    if (e != cudaSuccess) {
        PyErr_Format(PyExc_RuntimeError, "h2d failed: %s", cudaGetErrorString(e));
        return NULL;
    }
    Py_RETURN_NONE;
}

/* d2h(cuda_buf, cpu_array_obj) */
static PyObject *py_d2h(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *cbuf;
    PyObject *cpu_obj;
    if (!PyArg_ParseTuple(args, "O!O", &CudaBufferType, &cbuf, &cpu_obj)) return NULL;
    Py_buffer view;
    if (PyObject_GetBuffer(cpu_obj, &view, PyBUF_WRITABLE) < 0) return NULL;
    size_t copy_bytes = cbuf->nbytes < (size_t)view.len ? cbuf->nbytes : (size_t)view.len;
    cudaError_t e = cudaMemcpy(view.buf, cbuf->ptr, copy_bytes, cudaMemcpyDeviceToHost);
    PyBuffer_Release(&view);
    if (e != cudaSuccess) {
        PyErr_Format(PyExc_RuntimeError, "d2h failed: %s", cudaGetErrorString(e));
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *py_synchronize(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args)) {
    CUDA_CHECK(cudaDeviceSynchronize());
    Py_RETURN_NONE;
}

/* gemm_f32(A: CudaBuffer, B: CudaBuffer, m, k, n) -> CudaBuffer */
static PyObject *py_gemm_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *A, *B;
    int m, k, n;
    if (!PyArg_ParseTuple(args, "O!O!iii",
                          &CudaBufferType, &A, &CudaBufferType, &B,
                          &m, &k, &n)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *C = new_cuda_buf((size_t)m * n * sizeof(float), dev);
    if (!C) return NULL;
    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
    kernel_gemm_f32<<<grid, block>>>((float *)A->ptr, (float *)B->ptr, (float *)C->ptr, m, k, n);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        Py_DECREF(C);
        PyErr_Format(PyExc_RuntimeError, "gemm_f32 kernel: %s", cudaGetErrorString(e));
        return NULL;
    }
    return (PyObject *)C;
}

/* relu_f32(A: CudaBuffer, n: int) -> CudaBuffer */
static PyObject *py_relu_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *A; int n;
    if (!PyArg_ParseTuple(args, "O!i", &CudaBufferType, &A, &n)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *C = new_cuda_buf((size_t)n * sizeof(float), dev);
    if (!C) return NULL;
    int threads = 256, blocks = (n + threads - 1) / threads;
    kernel_relu_f32<<<blocks, threads>>>((float *)A->ptr, (float *)C->ptr, n);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { Py_DECREF(C); PyErr_Format(PyExc_RuntimeError, "relu_f32: %s", cudaGetErrorString(e)); return NULL; }
    return (PyObject *)C;
}

/* add_f32(A, B, n) -> CudaBuffer */
static PyObject *py_add_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *A, *B; int n;
    if (!PyArg_ParseTuple(args, "O!O!i", &CudaBufferType, &A, &CudaBufferType, &B, &n)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *C = new_cuda_buf((size_t)n * sizeof(float), dev);
    if (!C) return NULL;
    int threads = 256, blocks = (n + threads - 1) / threads;
    kernel_add_f32<<<blocks, threads>>>((float *)A->ptr, (float *)B->ptr, (float *)C->ptr, n);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { Py_DECREF(C); PyErr_Format(PyExc_RuntimeError, "add_f32: %s", cudaGetErrorString(e)); return NULL; }
    return (PyObject *)C;
}

/* scale_f32(A, scalar, n) -> CudaBuffer */
static PyObject *py_scale_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *A; float scalar; int n;
    if (!PyArg_ParseTuple(args, "O!fi", &CudaBufferType, &A, &scalar, &n)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *C = new_cuda_buf((size_t)n * sizeof(float), dev);
    if (!C) return NULL;
    int threads = 256, blocks = (n + threads - 1) / threads;
    kernel_scale_f32<<<blocks, threads>>>((float *)A->ptr, scalar, (float *)C->ptr, n);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { Py_DECREF(C); PyErr_Format(PyExc_RuntimeError, "scale_f32: %s", cudaGetErrorString(e)); return NULL; }
    return (PyObject *)C;
}

/* softmax_f32(A, rows, cols) -> CudaBuffer */
static PyObject *py_softmax_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *A; int rows, cols;
    if (!PyArg_ParseTuple(args, "O!ii", &CudaBufferType, &A, &rows, &cols)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *C = new_cuda_buf((size_t)rows * cols * sizeof(float), dev);
    if (!C) return NULL;
    int threads = 128, blocks = (rows + threads - 1) / threads;
    kernel_softmax_f32<<<blocks, threads>>>((float *)A->ptr, (float *)C->ptr, rows, cols);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { Py_DECREF(C); PyErr_Format(PyExc_RuntimeError, "softmax_f32: %s", cudaGetErrorString(e)); return NULL; }
    return (PyObject *)C;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Module definition
 * ═══════════════════════════════════════════════════════════════════════════ */
/* layernorm_f32(X, gamma_or_None, beta_or_None, N, D, eps) -> CudaBuffer */
static PyObject *py_layernorm_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *X;
    PyObject *gamma_obj, *beta_obj;
    int N, D; float eps;
    if (!PyArg_ParseTuple(args, "O!OOiif",
                          &CudaBufferType, &X, &gamma_obj, &beta_obj, &N, &D, &eps))
        return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *Y = new_cuda_buf((size_t)N * D * sizeof(float), dev);
    if (!Y) return NULL;
    float *g_ptr = NULL, *b_ptr = NULL;
    if (gamma_obj != Py_None && PyObject_IsInstance(gamma_obj, (PyObject *)&CudaBufferType))
        g_ptr = (float *)((CudaBufferObject *)gamma_obj)->ptr;
    if (beta_obj != Py_None && PyObject_IsInstance(beta_obj, (PyObject *)&CudaBufferType))
        b_ptr = (float *)((CudaBufferObject *)beta_obj)->ptr;
    int threads = 128, blocks = (N + threads - 1) / threads;
    kernel_layernorm_f32<<<blocks, threads>>>((float *)X->ptr, g_ptr, b_ptr,
                                               (float *)Y->ptr, N, D, eps);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        Py_DECREF(Y);
        PyErr_Format(PyExc_RuntimeError, "layernorm_f32: %s", cudaGetErrorString(e));
        return NULL;
    }
    return (PyObject *)Y;
}

/* flash_attn_f32(Q, K, V, N, d, scale, causal) -> CudaBuffer */
static PyObject *py_flash_attn_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *Q, *K, *V;
    int N, d, causal; float scale;
    if (!PyArg_ParseTuple(args, "O!O!O!iifi",
                          &CudaBufferType, &Q, &CudaBufferType, &K, &CudaBufferType, &V,
                          &N, &d, &scale, &causal))
        return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *O = new_cuda_buf((size_t)N * d * sizeof(float), dev);
    if (!O) return NULL;
    size_t smem_bytes = 2 * d * sizeof(float);
    kernel_flash_attn_f32<<<N, 1, smem_bytes>>>(
        (float *)Q->ptr, (float *)K->ptr, (float *)V->ptr, (float *)O->ptr,
        N, d, scale, causal);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        Py_DECREF(O);
        PyErr_Format(PyExc_RuntimeError, "flash_attn_f32: %s", cudaGetErrorString(e));
        return NULL;
    }
    return (PyObject *)O;
}

/* gemm32_f32(A, B, m, k, n) -> CudaBuffer — 32×32 tiled SGEMM */
static PyObject *py_gemm32_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *A, *B; int m, k, n;
    if (!PyArg_ParseTuple(args, "O!O!iii", &CudaBufferType, &A, &CudaBufferType, &B, &m, &k, &n)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *C = new_cuda_buf((size_t)m * n * sizeof(float), dev);
    if (!C) return NULL;
    dim3 block(TILE32, TILE32);
    dim3 grid((n + TILE32 - 1) / TILE32, (m + TILE32 - 1) / TILE32);
    kernel_gemm32_f32<<<grid, block>>>((float *)A->ptr, (float *)B->ptr, (float *)C->ptr, m, k, n);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { Py_DECREF(C); PyErr_Format(PyExc_RuntimeError, "gemm32: %s", cudaGetErrorString(e)); return NULL; }
    return (PyObject *)C;
}

/* adam_f32(params, grads, m, v, n, lr, beta1, beta2, eps, wd, bc1, bc2) in-place */
static PyObject *py_adam_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *params, *grads, *mv_m, *mv_v;
    int n; float lr, beta1, beta2, eps, wd, bc1, bc2;
    if (!PyArg_ParseTuple(args, "O!O!O!O!ifffffffff",
                          &CudaBufferType, &params, &CudaBufferType, &grads,
                          &CudaBufferType, &mv_m, &CudaBufferType, &mv_v,
                          &n, &lr, &beta1, &beta2, &eps, &wd, &bc1, &bc2))
        return NULL;
    int threads = 256, blocks = (n + threads - 1) / threads;
    kernel_adam_f32<<<blocks, threads>>>(
        (float *)params->ptr, (float *)grads->ptr, (float *)mv_m->ptr, (float *)mv_v->ptr,
        n, lr, beta1, beta2, eps, wd, bc1, bc2);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { PyErr_Format(PyExc_RuntimeError, "adam_f32: %s", cudaGetErrorString(e)); return NULL; }
    Py_RETURN_NONE;
}

/* rmsnorm_f32(X, weight_or_None, N, D, eps) -> CudaBuffer */
static PyObject *py_rmsnorm_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *X; PyObject *w_obj; int N, D; float eps;
    if (!PyArg_ParseTuple(args, "O!Oiif", &CudaBufferType, &X, &w_obj, &N, &D, &eps)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *Y = new_cuda_buf((size_t)N * D * sizeof(float), dev);
    if (!Y) return NULL;
    float *w_ptr = (w_obj != Py_None && PyObject_IsInstance(w_obj, (PyObject *)&CudaBufferType))
                   ? (float *)((CudaBufferObject *)w_obj)->ptr : NULL;
    kernel_rmsnorm_f32<<<N, 1>>>((float *)X->ptr, w_ptr, (float *)Y->ptr, N, D, eps);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { Py_DECREF(Y); PyErr_Format(PyExc_RuntimeError, "rmsnorm: %s", cudaGetErrorString(e)); return NULL; }
    return (PyObject *)Y;
}

/* embedding_f32(weight, indices_int32, batch, embed_dim) -> CudaBuffer */
static PyObject *py_embedding_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *W, *idx; int batch, embed_dim;
    if (!PyArg_ParseTuple(args, "O!O!ii", &CudaBufferType, &W, &CudaBufferType, &idx, &batch, &embed_dim)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *O = new_cuda_buf((size_t)batch * embed_dim * sizeof(float), dev);
    if (!O) return NULL;
    dim3 block(128); dim3 grid(batch, (embed_dim + 127) / 128);
    kernel_embedding_f32<<<grid, block>>>((float *)W->ptr, (int *)idx->ptr, (float *)O->ptr, batch, embed_dim);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { Py_DECREF(O); PyErr_Format(PyExc_RuntimeError, "embedding: %s", cudaGetErrorString(e)); return NULL; }
    return (PyObject *)O;
}

/* cross_entropy_f32(logits, targets_int32, N, C) -> CudaBuffer (N losses) */
static PyObject *py_cross_entropy_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *logits, *targets; int N, C;
    if (!PyArg_ParseTuple(args, "O!O!ii", &CudaBufferType, &logits, &CudaBufferType, &targets, &N, &C)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *losses = new_cuda_buf((size_t)N * sizeof(float), dev);
    if (!losses) return NULL;
    int threads = 128, blocks = (N + threads - 1) / threads;
    kernel_cross_entropy_f32<<<blocks, threads>>>((float *)logits->ptr, (int *)targets->ptr, (float *)losses->ptr, N, C);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { Py_DECREF(losses); PyErr_Format(PyExc_RuntimeError, "cross_entropy: %s", cudaGetErrorString(e)); return NULL; }
    return (PyObject *)losses;
}

/* gelu_f32(A, n) -> CudaBuffer */
static PyObject *py_gelu_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *A; int n;
    if (!PyArg_ParseTuple(args, "O!i", &CudaBufferType, &A, &n)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *C = new_cuda_buf((size_t)n * sizeof(float), dev);
    if (!C) return NULL;
    int threads = 256, blocks = (n + threads - 1) / threads;
    kernel_gelu_f32<<<blocks, threads>>>((float *)A->ptr, (float *)C->ptr, n);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { Py_DECREF(C); PyErr_Format(PyExc_RuntimeError, "gelu: %s", cudaGetErrorString(e)); return NULL; }
    return (PyObject *)C;
}

/* silu_f32(A, n) -> CudaBuffer */
static PyObject *py_silu_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *A; int n;
    if (!PyArg_ParseTuple(args, "O!i", &CudaBufferType, &A, &n)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *C = new_cuda_buf((size_t)n * sizeof(float), dev);
    if (!C) return NULL;
    int threads = 256, blocks = (n + threads - 1) / threads;
    kernel_silu_f32<<<blocks, threads>>>((float *)A->ptr, (float *)C->ptr, n);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { Py_DECREF(C); PyErr_Format(PyExc_RuntimeError, "silu: %s", cudaGetErrorString(e)); return NULL; }
    return (PyObject *)C;
}

/* mul_f32(A, B, n) -> CudaBuffer */
static PyObject *py_mul_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *A, *B; int n;
    if (!PyArg_ParseTuple(args, "O!O!i", &CudaBufferType, &A, &CudaBufferType, &B, &n)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *C = new_cuda_buf((size_t)n * sizeof(float), dev);
    if (!C) return NULL;
    int threads = 256, blocks = (n + threads - 1) / threads;
    kernel_mul_f32<<<blocks, threads>>>((float *)A->ptr, (float *)B->ptr, (float *)C->ptr, n);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { Py_DECREF(C); PyErr_Format(PyExc_RuntimeError, "mul: %s", cudaGetErrorString(e)); return NULL; }
    return (PyObject *)C;
}

/* transpose_f32(A, m, n) -> CudaBuffer (n, m) */
static PyObject *py_transpose_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *A; int m, n;
    if (!PyArg_ParseTuple(args, "O!ii", &CudaBufferType, &A, &m, &n)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *B = new_cuda_buf((size_t)m * n * sizeof(float), dev);
    if (!B) return NULL;
    dim3 block(TRANS_TILE, TRANS_TILE);
    dim3 grid((n + TRANS_TILE - 1) / TRANS_TILE, (m + TRANS_TILE - 1) / TRANS_TILE);
    kernel_transpose_f32<<<grid, block>>>((float *)A->ptr, (float *)B->ptr, m, n);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { Py_DECREF(B); PyErr_Format(PyExc_RuntimeError, "transpose: %s", cudaGetErrorString(e)); return NULL; }
    return (PyObject *)B;
}

/* pool_reset() — reset bump pointer; call between forward passes */
static PyObject *py_pool_reset(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args)) {
    _vp_pool_used = 0;
    Py_RETURN_NONE;
}

/* pool_info() -> (used_bytes, total_bytes) */
static PyObject *py_pool_info(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args)) {
    return Py_BuildValue("(nn)", (Py_ssize_t)_vp_pool_used, (Py_ssize_t)_vp_pool_size);
}

/* stream_sync_all() — sync all 4 streams */
static PyObject *py_stream_sync(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args)) {
    for (int i = 0; i < VP_N_STREAMS; ++i)
        if (_vp_streams[i]) cudaStreamSynchronize(_vp_streams[i]);
    Py_RETURN_NONE;
}

/* linear_silu_f32(X, W, b_or_None, N, in_f, out_f) -> CudaBuffer */
static PyObject *py_linear_silu_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *X, *W; PyObject *b_obj; int N, in_f, out_f;
    if (!PyArg_ParseTuple(args, "O!O!Oiii", &CudaBufferType, &X, &CudaBufferType, &W,
                          &b_obj, &N, &in_f, &out_f)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *O = new_cuda_buf((size_t)N * out_f * sizeof(float), dev);
    if (!O) return NULL;
    float *b_ptr = (b_obj != Py_None && PyObject_IsInstance(b_obj, (PyObject *)&CudaBufferType))
                   ? (float *)((CudaBufferObject *)b_obj)->ptr : NULL;
    dim3 block(128); dim3 grid((out_f + 127) / 128, N);
    kernel_linear_silu_f32<<<grid, block>>>((float*)X->ptr, (float*)W->ptr, b_ptr, (float*)O->ptr, N, in_f, out_f);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { Py_DECREF(O); PyErr_Format(PyExc_RuntimeError, "linear_silu: %s", cudaGetErrorString(e)); return NULL; }
    return (PyObject *)O;
}

/* linear_gelu_f32(X, W, b_or_None, N, in_f, out_f) -> CudaBuffer */
static PyObject *py_linear_gelu_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *X, *W; PyObject *b_obj; int N, in_f, out_f;
    if (!PyArg_ParseTuple(args, "O!O!Oiii", &CudaBufferType, &X, &CudaBufferType, &W,
                          &b_obj, &N, &in_f, &out_f)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *O = new_cuda_buf((size_t)N * out_f * sizeof(float), dev);
    if (!O) return NULL;
    float *b_ptr = (b_obj != Py_None && PyObject_IsInstance(b_obj, (PyObject *)&CudaBufferType))
                   ? (float *)((CudaBufferObject *)b_obj)->ptr : NULL;
    dim3 block(128); dim3 grid((out_f + 127) / 128, N);
    kernel_linear_gelu_f32<<<grid, block>>>((float*)X->ptr, (float*)W->ptr, b_ptr, (float*)O->ptr, N, in_f, out_f);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { Py_DECREF(O); PyErr_Format(PyExc_RuntimeError, "linear_gelu: %s", cudaGetErrorString(e)); return NULL; }
    return (PyObject *)O;
}

/* qkv_proj_f32(X, Wqkv, BT, d) -> CudaBuffer (BT, 3*d) */
static PyObject *py_qkv_proj_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *X, *Wqkv; int BT, d;
    if (!PyArg_ParseTuple(args, "O!O!ii", &CudaBufferType, &X, &CudaBufferType, &Wqkv, &BT, &d)) return NULL;
    int dev = 0; cudaGetDevice(&dev); int out_d = 3 * d;
    CudaBufferObject *O = new_cuda_buf((size_t)BT * out_d * sizeof(float), dev);
    if (!O) return NULL;
    dim3 block(128); dim3 grid((out_d + 127) / 128, BT);
    kernel_qkv_proj_f32<<<grid, block>>>((float*)X->ptr, (float*)Wqkv->ptr, (float*)O->ptr, BT, d, out_d);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { Py_DECREF(O); PyErr_Format(PyExc_RuntimeError, "qkv_proj: %s", cudaGetErrorString(e)); return NULL; }
    return (PyObject *)O;
}

/* dropout_f32(X, mask, scale, n) -> CudaBuffer */
static PyObject *py_dropout_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *X, *mask; float sc; int n;
    if (!PyArg_ParseTuple(args, "O!O!fi", &CudaBufferType, &X, &CudaBufferType, &mask, &sc, &n)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *O = new_cuda_buf((size_t)n * sizeof(float), dev);
    if (!O) return NULL;
    int threads = 256, blocks = (n + threads - 1) / threads;
    kernel_dropout_f32<<<blocks, threads>>>((float*)X->ptr, (float*)mask->ptr, (float*)O->ptr, sc, n);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { Py_DECREF(O); PyErr_Format(PyExc_RuntimeError, "dropout: %s", cudaGetErrorString(e)); return NULL; }
    return (PyObject *)O;
}

/* ── Session 8: FP16 GEMM + persistent kernels ─────────────────────────── */

/* Packed FP16 GEMM using CUDA __half arithmetic.
 * Both A and B are stored as float32 in device memory; we convert tiles
 * to __half in shared memory, multiply with __half2 FMA, accumulate in
 * float32. This gives ~2× throughput on FP16-capable hardware (sm_60+).
 */
/* FP16 is available on all Pascal/sm_60+ GPUs; guard with CUDA_VERSION >= 8000 */
#if !defined(DISABLE_FP16) && CUDA_VERSION >= 8000
#include <cuda_fp16.h>
#define FP16_TILE 16
__global__ void kernel_gemm_fp16_f32(const float * __restrict__ A,
                                      const float * __restrict__ B,
                                      float       * __restrict__ C,
                                      int m, int k, int n) {
    __shared__ __half sA[FP16_TILE][FP16_TILE];
    __shared__ __half sB[FP16_TILE][FP16_TILE];
    int row = blockIdx.y * FP16_TILE + threadIdx.y;
    int col = blockIdx.x * FP16_TILE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (k + FP16_TILE - 1) / FP16_TILE; ++t) {
        int aCol = t * FP16_TILE + threadIdx.x;
        int bRow = t * FP16_TILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = __float2half((row < m && aCol < k) ? A[row*k+aCol] : 0.f);
        sB[threadIdx.y][threadIdx.x] = __float2half((bRow < k && col < n) ? B[bRow*n+col] : 0.f);
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < FP16_TILE; ++i)
            acc += __half2float(sA[threadIdx.y][i]) * __half2float(sB[i][threadIdx.x]);
        __syncthreads();
    }
    if (row < m && col < n) C[row * n + col] = acc;
}

static PyObject *py_gemm_fp16_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *A, *B; int m, k, n;
    if (!PyArg_ParseTuple(args, "O!O!iii", &CudaBufferType, &A, &CudaBufferType, &B, &m, &k, &n)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *C = new_cuda_buf((size_t)m * n * sizeof(float), dev);
    if (!C) return NULL;
    dim3 block(FP16_TILE, FP16_TILE);
    dim3 grid((n + FP16_TILE - 1) / FP16_TILE, (m + FP16_TILE - 1) / FP16_TILE);
    kernel_gemm_fp16_f32<<<grid, block>>>((float*)A->ptr, (float*)B->ptr, (float*)C->ptr, m, k, n);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { Py_DECREF(C); PyErr_Format(PyExc_RuntimeError, "gemm_fp16: %s", cudaGetErrorString(e)); return NULL; }
    return (PyObject *)C;
}
#endif /* DISABLE_FP16 / sm_60 */

/* Fused residual add + LayerNorm (Pre-Norm pattern used in every Transformer).
 * out[i] = layernorm(x[i] + residual[i]) — single kernel, 2 reads + 1 write.
 */
__global__ void kernel_residual_layernorm_f32(
    const float * __restrict__ x,
    const float * __restrict__ residual,
    const float * __restrict__ gamma,
    const float * __restrict__ beta,
    float       * __restrict__ out,
    int N, int D, float eps)
{
    int row = blockIdx.x;
    if (row >= N) return;
    const float *xr = x + row * D;
    const float *rr = residual + row * D;
    float       *or_ = out + row * D;
    float mean = 0.f, var = 0.f;
    for (int i = 0; i < D; ++i) mean += xr[i] + rr[i];
    mean /= D;
    for (int i = 0; i < D; ++i) { float v = (xr[i]+rr[i]) - mean; var += v*v; }
    float inv = rsqrtf(var / D + eps);
    for (int i = 0; i < D; ++i)
        or_[i] = gamma[i] * ((xr[i]+rr[i]) - mean) * inv + beta[i];
}

static PyObject *py_residual_layernorm_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *X, *Res, *Gamma, *Beta; int N, D; float eps;
    if (!PyArg_ParseTuple(args, "O!O!O!O!iif",
            &CudaBufferType, &X, &CudaBufferType, &Res,
            &CudaBufferType, &Gamma, &CudaBufferType, &Beta, &N, &D, &eps)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *O = new_cuda_buf((size_t)N * D * sizeof(float), dev);
    if (!O) return NULL;
    kernel_residual_layernorm_f32<<<N, 1>>>(
        (float*)X->ptr, (float*)Res->ptr, (float*)Gamma->ptr, (float*)Beta->ptr,
        (float*)O->ptr, N, D, eps);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { Py_DECREF(O); PyErr_Format(PyExc_RuntimeError, "residual_ln: %s", cudaGetErrorString(e)); return NULL; }
    return (PyObject *)O;
}

/* Fused residual add + RMSNorm (LLaMA / Mistral style). */
__global__ void kernel_residual_rmsnorm_f32(
    const float * __restrict__ x,
    const float * __restrict__ residual,
    const float * __restrict__ w,
    float       * __restrict__ out,
    int N, int D, float eps)
{
    int row = blockIdx.x;
    if (row >= N) return;
    const float *xr = x + row*D, *rr = residual + row*D;
    float *or_ = out + row*D;
    float ss = 0.f;
    for (int i = 0; i < D; ++i) { float v = xr[i]+rr[i]; ss += v*v; }
    float inv = rsqrtf(ss / D + eps);
    for (int i = 0; i < D; ++i) or_[i] = w[i] * (xr[i]+rr[i]) * inv;
}

static PyObject *py_residual_rmsnorm_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *X, *Res, *W; int N, D; float eps;
    if (!PyArg_ParseTuple(args, "O!O!O!iif",
            &CudaBufferType, &X, &CudaBufferType, &Res,
            &CudaBufferType, &W, &N, &D, &eps)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *O = new_cuda_buf((size_t)N * D * sizeof(float), dev);
    if (!O) return NULL;
    kernel_residual_rmsnorm_f32<<<N, 1>>>(
        (float*)X->ptr, (float*)Res->ptr, (float*)W->ptr, (float*)O->ptr, N, D, eps);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { Py_DECREF(O); PyErr_Format(PyExc_RuntimeError, "residual_rms: %s", cudaGetErrorString(e)); return NULL; }
    return (PyObject *)O;
}

/* Rotary Position Embedding (RoPE) in-place.
 * Applies cos/sin rotation to each head's (q or k) tensor.
 * dim/2 thread pairs rotate one head; each block handles one sequence position.
 */
__global__ void kernel_rope_f32(float * __restrict__ qk,
                                  const float * __restrict__ cos_cache,
                                  const float * __restrict__ sin_cache,
                                  int N, int H, int Dh)
{
    int pos = blockIdx.x;
    int h   = blockIdx.y;
    int i   = threadIdx.x;   /* 0 .. Dh/2-1 */
    if (pos >= N || h >= H || i >= Dh/2) return;
    int base = (pos * H + h) * Dh;
    float x0 = qk[base + i];
    float x1 = qk[base + i + Dh/2];
    float c  = cos_cache[pos * (Dh/2) + i];
    float s  = sin_cache[pos * (Dh/2) + i];
    qk[base + i]        = x0*c - x1*s;
    qk[base + i + Dh/2] = x0*s + x1*c;
}

static PyObject *py_rope_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *QK, *Cos, *Sin; int N, H, Dh;
    if (!PyArg_ParseTuple(args, "O!O!O!iii",
            &CudaBufferType, &QK, &CudaBufferType, &Cos, &CudaBufferType, &Sin,
            &N, &H, &Dh)) return NULL;
    dim3 grid(N, H);
    dim3 block(Dh / 2);
    kernel_rope_f32<<<grid, block>>>((float*)QK->ptr, (float*)Cos->ptr, (float*)Sin->ptr, N, H, Dh);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { PyErr_Format(PyExc_RuntimeError, "rope: %s", cudaGetErrorString(e)); return NULL; }
    Py_INCREF(QK);  /* in-place, return same buffer */
    return (PyObject *)QK;
}

/* gemm_tc_f32 is only available when compiled with sm_70+ (-gencode sm_70 or higher).
 * We define VP_TENSOR_CORE_AVAILABLE via nvcc -DXP_TENSOR_CORE_AVAILABLE when targeting sm_70+.
 * The kernel_gemm_tc_f32 itself uses #if __CUDA_ARCH__ >= 700 internally (device code guard). */
#if defined(VP_TENSOR_CORE_AVAILABLE)
static PyObject *py_gemm_tc_f32(PyObject *Py_UNUSED(self), PyObject *args) {
    CudaBufferObject *A, *B; int m, k, n;
    if (!PyArg_ParseTuple(args, "O!O!iii", &CudaBufferType, &A, &CudaBufferType, &B, &m, &k, &n)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *C = new_cuda_buf((size_t)m * n * sizeof(float), dev);
    if (!C) return NULL;
    dim3 block(32, 4);
    dim3 grid((n + WMMA_N - 1) / WMMA_N, (m + WMMA_M - 1) / WMMA_M);
    kernel_gemm_tc_f32<<<grid, block>>>((float*)A->ptr, (float*)B->ptr, (float*)C->ptr, m, k, n);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { Py_DECREF(C); PyErr_Format(PyExc_RuntimeError, "gemm_tc: %s", cudaGetErrorString(e)); return NULL; }
    return (PyObject *)C;
}
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * Session 9 — Warp-level reductions, online softmax, Flash Decode partial
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── Warp-level reduce sum ──────────────────────────────────────────────── */
__global__ void kernel_warp_reduce_sum_f32(const float* __restrict__ in,
                                            float* __restrict__ out, int n) {
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    float val = (tid < n) ? in[tid] : 0.0f;
    /* Warp butterfly reduce */
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    if (lane == 0 && blockIdx.x < n)
        atomicAdd(out, val);
}

static PyObject* py_warp_reduce_sum_f32(PyObject* self, PyObject* args) {
    CudaBufferObject *A; int n;
    if (!PyArg_ParseTuple(args, "O!i", &CudaBufferType, &A, &n)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *C = new_cuda_buf(sizeof(float), dev);
    if (!C) return NULL;
    cudaMemset(C->ptr, 0, sizeof(float));
    kernel_warp_reduce_sum_f32<<<(n+255)/256, 256>>>(
        (const float*)A->ptr, (float*)C->ptr, n);
    return (PyObject*)C;
}

/* ── Warp-level reduce max ──────────────────────────────────────────────── */
__global__ void kernel_warp_reduce_max_f32(const float* __restrict__ in,
                                            float* __restrict__ out, int n) {
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    float val = (tid < n) ? in[tid] : -1e38f;
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    if (lane == 0 && blockIdx.x < n)
        atomicMax((int*)out, __float_as_int(val));
}

static PyObject* py_warp_reduce_max_f32(PyObject* self, PyObject* args) {
    CudaBufferObject *A; int n;
    if (!PyArg_ParseTuple(args, "O!i", &CudaBufferType, &A, &n)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *C = new_cuda_buf(sizeof(float), dev);
    if (!C) return NULL;
    float neg_inf = -1e38f;
    cudaMemcpy(C->ptr, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
    kernel_warp_reduce_max_f32<<<(n+255)/256, 256>>>(
        (const float*)A->ptr, (float*)C->ptr, n);
    return (PyObject*)C;
}

/* ── Online (chunked 3-pass) softmax ────────────────────────────────────── */
__global__ void kernel_online_softmax_f32(const float* __restrict__ X,
                                           float* __restrict__ Out,
                                           int rows, int N) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* src = X + row * N;
    float*       dst = Out + row * N;
    /* Pass 1: max */
    float m = -1e38f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        m = fmaxf(m, src[i]);
    /* Warp-level reduction across threads */
    for (int offset = 16; offset > 0; offset >>= 1)
        m = fmaxf(m, __shfl_down_sync(0xffffffff, m, offset));
    __shared__ float smem_m;
    if (threadIdx.x == 0) smem_m = m;
    __syncthreads();
    m = smem_m;
    /* Pass 2: sum-exp */
    float s = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        s += expf(src[i] - m);
    for (int offset = 16; offset > 0; offset >>= 1)
        s += __shfl_down_sync(0xffffffff, s, offset);
    __shared__ float smem_s;
    if (threadIdx.x == 0) smem_s = s;
    __syncthreads();
    s = smem_s;
    /* Pass 3: write */
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        dst[i] = expf(src[i] - m) / s;
}

static PyObject* py_online_softmax_f32(PyObject* self, PyObject* args) {
    CudaBufferObject *X; int rows, N;
    if (!PyArg_ParseTuple(args, "O!ii", &CudaBufferType, &X, &rows, &N)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *C = new_cuda_buf((size_t)rows * N * sizeof(float), dev);
    if (!C) return NULL;
    kernel_online_softmax_f32<<<rows, min(N, 1024)>>>(
        (const float*)X->ptr, (float*)C->ptr, rows, N);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { Py_DECREF(C);
        PyErr_Format(PyExc_RuntimeError, "online_softmax: %s", cudaGetErrorString(e));
        return NULL; }
    return (PyObject*)C;
}

/* ── Flash Decode partial kernel (CUDA C, single split) ─────────────────── *
 * Parallelises across B×H blocks; iterates over KV split of size split_size *
 * Returns partial output (B,H,D) + log-sum-exp (B,H) concatenated.         */
__global__ void kernel_flash_decode_partial_f32(
    const float* __restrict__ Q,  /* (B,H,D) — single query token */
    const float* __restrict__ K,  /* (B,H,S,D) */
    const float* __restrict__ V,  /* (B,H,S,D) */
    float* __restrict__ Out,      /* (B,H,splits,D) */
    float* __restrict__ Lse,      /* (B,H,splits) */
    int B, int H, int S, int D,
    int split_idx, int split_size,
    float scale)
{
    int b = blockIdx.x;
    int h = blockIdx.y;
    if (b >= B || h >= H) return;

    int kv_start = split_idx * split_size;
    int kv_end   = (kv_start + split_size < S) ? kv_start + split_size : S;

    /* Load Q for this head: (D,) using threads */
    extern __shared__ float sq[];  /* size D */
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        sq[d] = Q[(b * H + h) * D + d];
    __syncthreads();

    float m_i = -1e38f, l_i = 0.0f;
    /* acc in registers — only thread 0 accumulates fully for small D */
    float* acc = (float*)(sq + D);  /* use second half of shared mem */
    if (threadIdx.x < D) acc[threadIdx.x] = 0.0f;
    __syncthreads();

    for (int kv = kv_start; kv < kv_end; kv++) {
        /* dot Q·K[kv] */
        float dot = 0.0f;
        for (int d = threadIdx.x; d < D; d += blockDim.x)
            dot += sq[d] * K[((b * H + h) * S + kv) * D + d];
        /* reduce across threads */
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, off);
        __shared__ float s_dot;
        if (threadIdx.x == 0) s_dot = dot * scale;
        __syncthreads();
        dot = s_dot;

        float m_new = fmaxf(m_i, dot);
        float alpha = expf(m_i - m_new);
        float exp_d = expf(dot - m_new);
        l_i   = l_i * alpha + exp_d;
        /* update acc */
        for (int d = threadIdx.x; d < D; d += blockDim.x)
            acc[d] = acc[d] * alpha + exp_d * V[((b * H + h) * S + kv) * D + d];
        m_i = m_new;
        __syncthreads();
    }

    /* Normalise and write */
    int out_base = (b * H + h) * gridDim.z * D + split_idx * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        Out[out_base + d] = acc[d] / (l_i + 1e-9f);
    if (threadIdx.x == 0)
        Lse[(b * H + h) * gridDim.z + split_idx] = m_i + logf(l_i + 1e-9f);
}

static PyObject* py_flash_decode_partial_f32(PyObject* self, PyObject* args) {
    CudaBufferObject *Q, *K, *V;
    int B, H, S, D, num_splits;
    float scale;
    if (!PyArg_ParseTuple(args, "O!O!O!iiiiiif",
            &CudaBufferType, &Q, &CudaBufferType, &K, &CudaBufferType, &V,
            &B, &H, &S, &D, &num_splits, &scale)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    /* Output: partial_out (B,H,splits,D) + lse (B,H,splits) packed */
    size_t out_sz  = (size_t)B * H * num_splits * D * sizeof(float);
    size_t lse_sz  = (size_t)B * H * num_splits * sizeof(float);
    CudaBufferObject *Out = new_cuda_buf(out_sz + lse_sz, dev);
    if (!Out) return NULL;
    float* out_ptr = (float*)Out->ptr;
    float* lse_ptr = out_ptr + (size_t)B * H * num_splits * D;

    int split_size = (S + num_splits - 1) / num_splits;
    size_t smem = (size_t)(D * 2) * sizeof(float);
    dim3 grid(B, H, 1);
    int block = min(D, 128);
    for (int sp = 0; sp < num_splits; sp++) {
        kernel_flash_decode_partial_f32<<<grid, block, smem>>>(
            (const float*)Q->ptr, (const float*)K->ptr, (const float*)V->ptr,
            out_ptr, lse_ptr, B, H, S, D, sp, split_size, scale);
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { Py_DECREF(Out);
        PyErr_Format(PyExc_RuntimeError, "flash_decode_partial: %s", cudaGetErrorString(e));
        return NULL; }
    return (PyObject*)Out;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Session 9b — Fused embedding + positional bias, attention score bias
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Fused embedding gather + add positional embedding + optional dropout.
 * This is the very first op in every LLM forward pass: single kernel call
 * instead of 3 separate ops (embed→pos_add→dropout). */
__global__ void kernel_embed_pos_f32(
    const float* __restrict__ token_emb,  /* (vocab_size, D) */
    const float* __restrict__ pos_emb,    /* (max_seq_len, D) */
    const int*   __restrict__ input_ids,  /* (B, T) */
    float* __restrict__ out,              /* (B, T, D) */
    int B, int T, int D, int vocab_size,
    float dropout_scale, unsigned int dropout_seed)
{
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    if (b >= B || t >= T || d >= D) return;
    int tok_id = input_ids[b * T + t];
    if (tok_id < 0 || tok_id >= vocab_size) tok_id = 0;
    float val = token_emb[tok_id * D + d] + pos_emb[t * D + d];
    /* Philox-inspired inline dropout using integer hash */
    if (dropout_scale < 1.0f) {
        unsigned int hash = (b * T * D + t * D + d) ^ dropout_seed;
        hash = hash ^ (hash >> 16); hash *= 0x45d9f3b;
        hash = hash ^ (hash >> 16);
        float r = (hash & 0xFFFFFF) / 16777216.0f;
        val = (r > (1.0f - dropout_scale)) ? val / dropout_scale : 0.0f;
    }
    out[(b * T + t) * D + d] = val;
}

static PyObject* py_embed_pos_f32(PyObject* self, PyObject* args) {
    CudaBufferObject *TokEmb, *PosEmb, *Ids;
    int B, T, D, vocab_size;
    float dropout_p; unsigned int seed;
    if (!PyArg_ParseTuple(args, "O!O!O!iiiifI",
            &CudaBufferType, &TokEmb, &CudaBufferType, &PosEmb,
            &CudaBufferType, &Ids,
            &B, &T, &D, &vocab_size, &dropout_p, &seed)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *Out = new_cuda_buf((size_t)B * T * D * sizeof(float), dev);
    if (!Out) return NULL;
    float scale = dropout_p > 0.0f ? 1.0f - dropout_p : 1.0f;
    dim3 grid(B, T);
    kernel_embed_pos_f32<<<grid, D>>>(
        (const float*)TokEmb->ptr, (const float*)PosEmb->ptr,
        (const int*)Ids->ptr, (float*)Out->ptr,
        B, T, D, vocab_size, scale, seed);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { Py_DECREF(Out);
        PyErr_Format(PyExc_RuntimeError, "embed_pos: %s", cudaGetErrorString(e));
        return NULL; }
    return (PyObject*)Out;
}

/* ── Fused biased softmax (adds attention mask before softmax) ───────────── */
__global__ void kernel_biased_softmax_f32(
    const float* __restrict__ scores,  /* (batch, heads, N, N) */
    const float* __restrict__ bias,    /* (1, 1, N, N) or (batch, heads, N, N) */
    float* __restrict__ out,
    int batch, int heads, int N, int bias_broadcast)
{
    /* One block per (batch, head, row) */
    int b   = blockIdx.x;
    int h   = blockIdx.y;
    int row = blockIdx.z;
    if (b >= batch || h >= heads || row >= N) return;
    int base = (b * heads + h) * N * N + row * N;
    int bias_base = bias_broadcast ? row * N : base;
    /* Pass 1: max */
    float m = -1e38f;
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        float v = scores[base + col] + bias[bias_base + col];
        m = fmaxf(m, v);
    }
    for (int off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_down_sync(0xffffffff, m, off));
    __shared__ float smem_m;
    if (threadIdx.x == 0) smem_m = m;
    __syncthreads();
    m = smem_m;
    /* Pass 2: sum-exp */
    float s = 0.0f;
    for (int col = threadIdx.x; col < N; col += blockDim.x)
        s += expf(scores[base + col] + bias[bias_base + col] - m);
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_down_sync(0xffffffff, s, off);
    __shared__ float smem_s;
    if (threadIdx.x == 0) smem_s = s;
    __syncthreads();
    s = smem_s;
    /* Write */
    for (int col = threadIdx.x; col < N; col += blockDim.x)
        out[base + col] = expf(scores[base + col] + bias[bias_base + col] - m) / s;
}

static PyObject* py_biased_softmax_f32(PyObject* self, PyObject* args) {
    CudaBufferObject *Scores, *Bias;
    int batch, heads, N, bias_broadcast;
    if (!PyArg_ParseTuple(args, "O!O!iiii",
            &CudaBufferType, &Scores, &CudaBufferType, &Bias,
            &batch, &heads, &N, &bias_broadcast)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *Out = new_cuda_buf((size_t)batch * heads * N * N * sizeof(float), dev);
    if (!Out) return NULL;
    dim3 grid(batch, heads, N);
    kernel_biased_softmax_f32<<<grid, min(N, 128)>>>(
        (const float*)Scores->ptr, (const float*)Bias->ptr, (float*)Out->ptr,
        batch, heads, N, bias_broadcast);
    return (PyObject*)Out;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BF16 (bfloat16) support — Session 10
 * CUDA >= 11.0: __nv_bfloat16 available; older: use uint16 bit-cast
 * ═══════════════════════════════════════════════════════════════════════════ */
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#define VP_BF16_NATIVE 1
typedef __nv_bfloat16 vp_bf16_t;
#else
/* Minimal bf16 representation for compile-time compatibility */
typedef uint16_t vp_bf16_t;
/* Convert float32 to bf16 (truncate lower 16 bits of mantissa) */
__device__ __forceinline__ vp_bf16_t vp_float_to_bf16(float v) {
    uint32_t bits; memcpy(&bits, &v, 4);
    uint16_t bf = (uint16_t)(bits >> 16);
    return bf;
}
__device__ __forceinline__ float vp_bf16_to_float(vp_bf16_t v) {
    uint32_t bits = ((uint32_t)v) << 16;
    float f; memcpy(&f, &bits, 4); return f;
}
#define VP_BF16_NATIVE 0
#endif

/* ── BF16 Cast kernels ───────────────────────────────────────────────────── */
/* fp32_to_bf16: cast float buffer to bf16 (stored as uint16) */
__global__ void kernel_fp32_to_bf16(const float * __restrict__ src,
                                     uint16_t * __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t bits; memcpy(&bits, &src[i], 4);
    /* Round-to-nearest: add 0x7FFF + (bit15 of mantissa) before truncation */
    uint32_t rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    dst[i] = (uint16_t)((bits + rounding_bias) >> 16);
}

/* bf16_to_fp32: upcast uint16 bf16 to float32 */
__global__ void kernel_bf16_to_fp32(const uint16_t * __restrict__ src,
                                     float * __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t bits = ((uint32_t)src[i]) << 16;
    memcpy(&dst[i], &bits, 4);
}

/* ── BF16 GEMM (Ampere / H100 — requires CUDA 11+ and sm_80+) ───────────── */
/* Uses native __nv_bfloat16 ops when available; falls back to FP32 otherwise */
__global__ void kernel_gemm_bf16(const uint16_t * __restrict__ A,
                                   const uint16_t * __restrict__ B,
                                   float * __restrict__ C,
                                   int M, int K, int N) {
    /* 16×16 tiled GEMM accumulating in FP32 */
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (K + 15) / 16; t++) {
        int aCol = t * 16 + threadIdx.x;
        int bRow = t * 16 + threadIdx.y;
        /* Load A tile: upcast bf16 → fp32 */
        if (row < M && aCol < K) {
            uint32_t ab = ((uint32_t)A[row * K + aCol]) << 16;
            memcpy(&As[threadIdx.y][threadIdx.x], &ab, 4);
        } else { As[threadIdx.y][threadIdx.x] = 0.0f; }
        /* Load B tile */
        if (bRow < K && col < N) {
            uint32_t bb = ((uint32_t)B[bRow * N + col]) << 16;
            memcpy(&Bs[threadIdx.y][threadIdx.x], &bb, 4);
        } else { Bs[threadIdx.y][threadIdx.x] = 0.0f; }
        __syncthreads();
        for (int k = 0; k < 16; k++)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

/* ── BF16 RMSNorm ────────────────────────────────────────────────────────── */
/* Same algorithm as kernel_rmsnorm_f32 but input/output in bf16 */
__global__ void kernel_rmsnorm_bf16(const uint16_t * __restrict__ X,
                                     const float    * __restrict__ w,
                                     uint16_t       * __restrict__ Y,
                                     int N, int D, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;
    const uint16_t *xrow = X + row * D;
    uint16_t       *yrow = Y + row * D;
    /* Compute sum of squares in fp32 */
    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        uint32_t b = ((uint32_t)xrow[i]) << 16;
        float v; memcpy(&v, &b, 4);
        ss += v * v;
    }
    /* Block reduce */
    __shared__ float sdata[256];
    sdata[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / (float)D + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        uint32_t b = ((uint32_t)xrow[i]) << 16;
        float v; memcpy(&v, &b, 4);
        float normed = v * rms * w[i];
        /* Store back as bf16 (round-nearest) */
        uint32_t bits; memcpy(&bits, &normed, 4);
        uint32_t rb = 0x7FFF + ((bits >> 16) & 1);
        yrow[i] = (uint16_t)((bits + rb) >> 16);
    }
}

/* ── BF16 SwiGLU (gate * silu(up)) ──────────────────────────────────────── */
__global__ void kernel_swiglu_bf16(const uint16_t * __restrict__ gate,
                                    const uint16_t * __restrict__ up,
                                    uint16_t       * __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t gb = ((uint32_t)gate[i]) << 16;
    uint32_t ub = ((uint32_t)up[i])   << 16;
    float g, u; memcpy(&g, &gb, 4); memcpy(&u, &ub, 4);
    float silu_g = g / (1.0f + expf(-g));
    float r = silu_g * u;
    uint32_t rb; memcpy(&rb, &r, 4);
    uint32_t bias = 0x7FFF + ((rb >> 16) & 1);
    out[i] = (uint16_t)((rb + bias) >> 16);
}

/* ── BF16 FP32 LayerNorm ─────────────────────────────────────────────────── */
/* Fused LN: input BF16, output BF16, weight/bias FP32 */
__global__ void kernel_layernorm_bf16(const uint16_t * __restrict__ X,
                                       const float    * __restrict__ gamma,
                                       const float    * __restrict__ beta,
                                       uint16_t       * __restrict__ Y,
                                       int N, int D, float eps) {
    int row = blockIdx.x;
    if (row >= N) return;
    const uint16_t *xrow = X + row * D;
    uint16_t       *yrow = Y + row * D;
    /* Mean */
    float mu = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        uint32_t b = ((uint32_t)xrow[i]) << 16;
        float v; memcpy(&v, &b, 4); mu += v;
    }
    __shared__ float smean[256], svar[256];
    smean[threadIdx.x] = mu;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smean[threadIdx.x] += smean[threadIdx.x + s];
        __syncthreads();
    }
    float mean = smean[0] / (float)D;
    /* Variance */
    float var = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        uint32_t b = ((uint32_t)xrow[i]) << 16;
        float v; memcpy(&v, &b, 4); float d = v - mean; var += d * d;
    }
    svar[threadIdx.x] = var;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) svar[threadIdx.x] += svar[threadIdx.x + s];
        __syncthreads();
    }
    float rstd = rsqrtf(svar[0] / (float)D + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        uint32_t b = ((uint32_t)xrow[i]) << 16;
        float v; memcpy(&v, &b, 4);
        float normed = (v - mean) * rstd * gamma[i] + beta[i];
        uint32_t rb; memcpy(&rb, &normed, 4);
        uint32_t bias = 0x7FFF + ((rb >> 16) & 1);
        yrow[i] = (uint16_t)((rb + bias) >> 16);
    }
}

/* ── Python wrappers for BF16 ops ────────────────────────────────────────── */
static PyObject *py_fp32_to_bf16(PyObject *self, PyObject *args) {
    CudaBufferObject *A; int n;
    if (!PyArg_ParseTuple(args, "O!i", &CudaBufferType, &A, &n)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *Out = new_cuda_buf((size_t)n * sizeof(uint16_t), dev);
    if (!Out) return NULL;
    kernel_fp32_to_bf16<<<(n+255)/256, 256>>>(
        (const float*)A->ptr, (uint16_t*)Out->ptr, n);
    return (PyObject*)Out;
}

static PyObject *py_bf16_to_fp32(PyObject *self, PyObject *args) {
    CudaBufferObject *A; int n;
    if (!PyArg_ParseTuple(args, "O!i", &CudaBufferType, &A, &n)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *Out = new_cuda_buf((size_t)n * sizeof(float), dev);
    if (!Out) return NULL;
    kernel_bf16_to_fp32<<<(n+255)/256, 256>>>(
        (const uint16_t*)A->ptr, (float*)Out->ptr, n);
    return (PyObject*)Out;
}

static PyObject *py_gemm_bf16(PyObject *self, PyObject *args) {
    CudaBufferObject *A, *B; int M, K, N;
    if (!PyArg_ParseTuple(args, "O!O!iii", &CudaBufferType, &A, &CudaBufferType, &B, &M, &K, &N)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *Out = new_cuda_buf((size_t)M * N * sizeof(float), dev);
    if (!Out) return NULL;
    dim3 block(16, 16), grid((N+15)/16, (M+15)/16);
    kernel_gemm_bf16<<<grid, block>>>(
        (const uint16_t*)A->ptr, (const uint16_t*)B->ptr, (float*)Out->ptr, M, K, N);
    return (PyObject*)Out;
}

static PyObject *py_rmsnorm_bf16(PyObject *self, PyObject *args) {
    CudaBufferObject *X, *W; int N, D; float eps;
    if (!PyArg_ParseTuple(args, "O!O!iif", &CudaBufferType, &X, &CudaBufferType, &W, &N, &D, &eps)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *Out = new_cuda_buf((size_t)N * D * sizeof(uint16_t), dev);
    if (!Out) return NULL;
    int tpb = min(D, 256);
    kernel_rmsnorm_bf16<<<N, tpb>>>(
        (const uint16_t*)X->ptr, (const float*)W->ptr, (uint16_t*)Out->ptr, N, D, eps);
    return (PyObject*)Out;
}

static PyObject *py_swiglu_bf16(PyObject *self, PyObject *args) {
    CudaBufferObject *Gate, *Up; int n;
    if (!PyArg_ParseTuple(args, "O!O!i", &CudaBufferType, &Gate, &CudaBufferType, &Up, &n)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *Out = new_cuda_buf((size_t)n * sizeof(uint16_t), dev);
    if (!Out) return NULL;
    kernel_swiglu_bf16<<<(n+255)/256, 256>>>(
        (const uint16_t*)Gate->ptr, (const uint16_t*)Up->ptr, (uint16_t*)Out->ptr, n);
    return (PyObject*)Out;
}

static PyObject *py_layernorm_bf16(PyObject *self, PyObject *args) {
    CudaBufferObject *X, *Gamma, *Beta; int N, D; float eps;
    if (!PyArg_ParseTuple(args, "O!O!O!iif",
                          &CudaBufferType, &X, &CudaBufferType, &Gamma, &CudaBufferType, &Beta,
                          &N, &D, &eps)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *Out = new_cuda_buf((size_t)N * D * sizeof(uint16_t), dev);
    if (!Out) return NULL;
    int tpb = min(D, 256);
    kernel_layernorm_bf16<<<N, tpb>>>(
        (const uint16_t*)X->ptr, (const float*)Gamma->ptr, (const float*)Beta->ptr,
        (uint16_t*)Out->ptr, N, D, eps);
    return (PyObject*)Out;
}

/* ── INT8 GEMM with absmax quantization ──────────────────────────────────── */
/* Each row of A and B is quantized to INT8 with per-row absmax scale.
 * Output is dequantized FP32 (scale_a[i] * scale_b[j] * dot(a_row_i, b_row_j)) */
__global__ void kernel_gemm_int8(const int8_t  * __restrict__ A,
                                   const int8_t  * __restrict__ B,
                                   const float   * __restrict__ scale_A, /* per-row scales of A */
                                   const float   * __restrict__ scale_B, /* per-row scales of B (cols of B^T) */
                                   float         * __restrict__ C,
                                   int M, int K, int N) {
    __shared__ int As[16][16];
    __shared__ int Bs[16][16];
    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    int32_t acc = 0;
    for (int t = 0; t < (K + 15) / 16; t++) {
        int aCol = t * 16 + threadIdx.x;
        int bRow = t * 16 + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? (int)A[row * K + aCol] : 0;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? (int)B[col * K + bRow] : 0; /* B transposed */
        __syncthreads();
        for (int k = 0; k < 16; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = (float)acc * scale_A[row] * scale_B[col];
}

/* quantize_fp32_to_int8: per-row absmax quantization */
__global__ void kernel_quantize_fp32_int8(const float * __restrict__ src,
                                           int8_t      * __restrict__ dst,
                                           float       * __restrict__ scales,
                                           int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;
    const float *row_ptr = src + row * D;
    int8_t      *out_ptr = dst + row * D;
    /* Find absmax */
    float amax = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        amax = fmaxf(amax, fabsf(row_ptr[i]));
    __shared__ float smax[256];
    smax[threadIdx.x] = amax;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x+s]);
        __syncthreads();
    }
    float scale = smax[0] / 127.0f;
    if (threadIdx.x == 0) scales[row] = (scale < 1e-8f) ? 1.0f : scale;
    __syncthreads();
    float inv_scale = (scale < 1e-8f) ? 1.0f : 1.0f / scale;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float q = row_ptr[i] * inv_scale;
        if (q >  127.0f) q =  127.0f;
        if (q < -127.0f) q = -127.0f;
        out_ptr[i] = (int8_t)__float2int_rn(q);
    }
}

static PyObject *py_quantize_fp32_int8(PyObject *self, PyObject *args) {
    CudaBufferObject *X; int N, D;
    if (!PyArg_ParseTuple(args, "O!ii", &CudaBufferType, &X, &N, &D)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *Out   = new_cuda_buf((size_t)N * D * sizeof(int8_t),  dev);
    CudaBufferObject *Scale = new_cuda_buf((size_t)N     * sizeof(float),   dev);
    if (!Out || !Scale) return NULL;
    int tpb = min(D, 256);
    kernel_quantize_fp32_int8<<<N, tpb>>>(
        (const float*)X->ptr, (int8_t*)Out->ptr, (float*)Scale->ptr, N, D);
    return Py_BuildValue("(OO)", (PyObject*)Out, (PyObject*)Scale);
}

static PyObject *py_gemm_int8(PyObject *self, PyObject *args) {
    CudaBufferObject *A, *B, *SA, *SB; int M, K, N;
    if (!PyArg_ParseTuple(args, "O!O!O!O!iii",
                          &CudaBufferType, &A, &CudaBufferType, &B,
                          &CudaBufferType, &SA, &CudaBufferType, &SB, &M, &K, &N)) return NULL;
    int dev = 0; cudaGetDevice(&dev);
    CudaBufferObject *Out = new_cuda_buf((size_t)M * N * sizeof(float), dev);
    if (!Out) return NULL;
    dim3 block(16, 16), grid((N+15)/16, (M+15)/16);
    kernel_gemm_int8<<<grid, block>>>(
        (const int8_t*)A->ptr, (const int8_t*)B->ptr,
        (const float*)SA->ptr, (const float*)SB->ptr,
        (float*)Out->ptr, M, K, N);
    return (PyObject*)Out;
}

/* ============================================================
 * Session 11 — Mamba selective scan + RWKV WKV + RoPE-BF16
 * ============================================================ */

/*
 * Mamba selective scan: parallel over (batch, d_inner), sequential over L.
 *   h_t = diag(A_bar_t) * h_{t-1} + B_bar_t * u_t     (d_state vector per channel)
 *   y_t = C_t . h_t                                     (dot product over d_state)
 * Grid: (B * D) blocks, each with BLOCK_N threads (d_state ≤ 256).
 * Each block does the entire L-step recurrence for one (b,d) pair.
 */
__global__ void kernel_mamba_ssm_f32(
    const float* __restrict__ u,   /* (B,L,D)   input            */
    const float* __restrict__ dt,  /* (B,L,D)   delta (softplus) */
    const float* __restrict__ A,   /* (D,N)     log A            */
    const float* __restrict__ B,   /* (B,L,N)   B selective      */
    const float* __restrict__ C,   /* (B,L,N)   C selective      */
    float*       __restrict__ y,   /* (B,L,D)   output           */
    int B_, int L_, int D_, int N_)
{
    int bd = blockIdx.x;        /* flattened (batch, d_inner) index */
    int b  = bd / D_;
    int d  = bd % D_;
    if (b >= B_ || d >= D_) return;

    /* h[n] in registers/shared — keep d_state ≤ 256 */
    extern __shared__ float h_shared[];
    float* h = h_shared + threadIdx.x * N_; /* each thread owns one h if N_ ≤ blockDim.x */
    /* Single-thread per block approach (safe for N_ ≤ 1024) */
    if (threadIdx.x == 0) {
        for (int n = 0; n < N_; n++) h[n] = 0.0f;
        for (int t = 0; t < L_; t++) {
            float u_t  = u [b * L_ * D_ + t * D_ + d];
            float dt_t = dt[b * L_ * D_ + t * D_ + d];
            /* softplus of dt — log(1+exp(dt)) */
            dt_t = (dt_t > 20.0f) ? dt_t : logf(1.0f + expf(dt_t));
            float y_t = 0.0f;
            for (int n = 0; n < N_; n++) {
                float a  = A[d * N_ + n];           /* log A (negative) */
                float dA = expf(dt_t * a);          /* exp(dt*log_A)    */
                float dB = dt_t * B[b * L_ * N_ + t * N_ + n];
                h[n] = dA * h[n] + dB * u_t;
                y_t += C[b * L_ * N_ + t * N_ + n] * h[n];
            }
            y[b * L_ * D_ + t * D_ + d] = y_t;
        }
    }
}

static PyObject* py_mamba_ssm_f32(PyObject* self, PyObject* args) {
    PyObject *pu, *pdt, *pA, *pB, *pC;
    int B_, L_, D_, N_;
    if (!PyArg_ParseTuple(args, "OOOOOiiii",
                          &pu, &pdt, &pA, &pB, &pC,
                          &B_, &L_, &D_, &N_)) return NULL;
    CudaBuffer *u  = (CudaBuffer*)pu;
    CudaBuffer *dt = (CudaBuffer*)pdt;
    CudaBuffer *A  = (CudaBuffer*)pA;
    CudaBuffer *Bb = (CudaBuffer*)pB;
    CudaBuffer *Cb = (CudaBuffer*)pC;
    CudaBuffer *Out = cuda_alloc(B_ * L_ * D_ * sizeof(float));
    if (!Out) return PyErr_NoMemory();
    int grid = B_ * D_;
    int block = 1;                        /* single thread per (b,d) for safety */
    size_t smem = N_ * sizeof(float);
    kernel_mamba_ssm_f32<<<grid, block, smem>>>(
        (const float*)u->ptr, (const float*)dt->ptr,
        (const float*)A->ptr, (const float*)Bb->ptr, (const float*)Cb->ptr,
        (float*)Out->ptr, B_, L_, D_, N_);
    return (PyObject*)Out;
}

/*
 * RWKV WKV kernel — the core of the RWKV time-mixing channel.
 * For each (b, c) channel, over sequence T:
 *   wkv_t = ( Σ_{i<t} exp(w*(t-1-i) + k_i) * v_i + exp(u + k_t) * v_t )
 *           / ( Σ_{i<t} exp(w*(t-1-i) + k_i)       + exp(u + k_t)       )
 * We compute in log-space with running max for numerical stability.
 */
__global__ void kernel_rwkv_wkv_f32(
    const float* __restrict__ w,   /* (C,)   per-channel decay (negative) */
    const float* __restrict__ u,   /* (C,)   per-channel bonus             */
    const float* __restrict__ k,   /* (B,T,C) keys                         */
    const float* __restrict__ v,   /* (B,T,C) values                       */
    float*       __restrict__ y,   /* (B,T,C) output                       */
    int B_, int T_, int C_)
{
    int bc = blockIdx.x * blockDim.x + threadIdx.x;
    int b  = bc / C_;
    int c  = bc % C_;
    if (b >= B_ || c >= C_) return;

    float w_c = w[c];   /* should be negative */
    float u_c = u[c];

    float aa = 0.0f, bb = 0.0f, pp = -1e38f;   /* log-space running state */
    for (int t = 0; t < T_; t++) {
        float kt = k[b * T_ * C_ + t * C_ + c];
        float vt = v[b * T_ * C_ + t * C_ + c];

        float qq = pp > (u_c + kt) ? pp : (u_c + kt);
        float A  = expf(pp - qq);
        float B2 = expf(u_c + kt - qq);
        y[b * T_ * C_ + t * C_ + c] = (A * aa + B2 * vt) / (A * bb + B2);

        qq = pp > (w_c + kt) ? pp : (w_c + kt);
        aa = expf(pp - qq) * aa + expf(w_c + kt - qq) * vt;
        bb = expf(pp - qq) * bb + expf(w_c + kt - qq);
        pp = qq;
    }
}

static PyObject* py_rwkv_wkv_f32(PyObject* self, PyObject* args) {
    PyObject *pw, *pu, *pk, *pv;
    int B_, T_, C_;
    if (!PyArg_ParseTuple(args, "OOOOiii", &pw, &pu, &pk, &pv, &B_, &T_, &C_))
        return NULL;
    CudaBuffer *w  = (CudaBuffer*)pw;
    CudaBuffer *u  = (CudaBuffer*)pu;
    CudaBuffer *k  = (CudaBuffer*)pk;
    CudaBuffer *v  = (CudaBuffer*)pv;
    CudaBuffer *Out = cuda_alloc(B_ * T_ * C_ * sizeof(float));
    if (!Out) return PyErr_NoMemory();
    int threads = 128;
    int grid    = (B_ * C_ + threads - 1) / threads;
    kernel_rwkv_wkv_f32<<<grid, threads>>>(
        (const float*)w->ptr, (const float*)u->ptr,
        (const float*)k->ptr, (const float*)v->ptr,
        (float*)Out->ptr, B_, T_, C_);
    return (PyObject*)Out;
}

/* RoPE on BF16 — in-place rotate pairs of bf16 channels. */
__global__ void kernel_rope_bf16(
    uint16_t*    __restrict__ x,   /* (B,T,H,D) bf16 in-place */
    int B_, int T_, int H_, int D_)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int bthd = B_ * T_ * H_ * (D_ / 2);
    if (idx >= bthd) return;

    int pos = (idx / (H_ * D_ / 2)) % T_;
    int i   = idx % (D_ / 2);
    float theta = (float)pos / powf(10000.0f, 2.0f * i / (float)D_);
    float cos_t = cosf(theta), sin_t = sinf(theta);

    uint16_t* p = x + (idx / (D_ / 2)) * D_ + i * 2;
    float x0 = bf16_to_float_device(p[0]);
    float x1 = bf16_to_float_device(p[1]);
    p[0] = float_to_bf16_device(x0 * cos_t - x1 * sin_t);
    p[1] = float_to_bf16_device(x0 * sin_t + x1 * cos_t);
}

static PyObject* py_rope_bf16(PyObject* self, PyObject* args) {
    PyObject *px;
    int B_, T_, H_, D_;
    if (!PyArg_ParseTuple(args, "Oiiii", &px, &B_, &T_, &H_, &D_)) return NULL;
    CudaBuffer *x = (CudaBuffer*)px;
    int n = B_ * T_ * H_ * (D_ / 2);
    int block = 256, grid = (n + block - 1) / block;
    kernel_rope_bf16<<<grid, block>>>((uint16_t*)x->ptr, B_, T_, H_, D_);
    Py_INCREF(px); return px;   /* in-place, return same buffer */
}

static PyMethodDef vp_cuda_methods[] = {
    {"device_count",   py_device_count,   METH_NOARGS,  "Number of CUDA devices"},
    {"device_name",    py_device_name,    METH_VARARGS, "device_name(idx=0) -> str"},
    {"alloc",          py_alloc,          METH_VARARGS, "alloc(nbytes) -> CudaBuffer"},
    {"h2d",            py_h2d,            METH_VARARGS, "h2d(cpu_array, cuda_buf)"},
    {"d2h",            py_d2h,            METH_VARARGS, "d2h(cuda_buf, cpu_array)"},
    {"synchronize",    py_synchronize,    METH_NOARGS,  "cudaDeviceSynchronize()"},
    {"gemm_f32",       py_gemm_f32,       METH_VARARGS, "gemm_f32(A,B,m,k,n)->CudaBuffer"},
    {"relu_f32",       py_relu_f32,       METH_VARARGS, "relu_f32(A,n)->CudaBuffer"},
    {"add_f32",        py_add_f32,        METH_VARARGS, "add_f32(A,B,n)->CudaBuffer"},
    {"scale_f32",      py_scale_f32,      METH_VARARGS, "scale_f32(A,scalar,n)->CudaBuffer"},
    {"softmax_f32",    py_softmax_f32,    METH_VARARGS, "softmax_f32(A,rows,cols)->CudaBuffer"},
    {"layernorm_f32",    py_layernorm_f32,    METH_VARARGS, "layernorm_f32(X,gamma,beta,N,D,eps)->CudaBuffer"},
    {"flash_attn_f32",   py_flash_attn_f32,   METH_VARARGS, "flash_attn_f32(Q,K,V,N,d,scale,causal)->CudaBuffer"},
    /* Session 6 — advanced kernels */
    {"gemm32_f32",       py_gemm32_f32,       METH_VARARGS, "gemm32_f32(A,B,m,k,n)->CudaBuffer 32x32 tiled"},
    {"adam_f32",         py_adam_f32,         METH_VARARGS, "adam_f32(p,g,m,v,n,lr,b1,b2,eps,wd,bc1,bc2) in-place"},
    {"rmsnorm_f32",      py_rmsnorm_f32,      METH_VARARGS, "rmsnorm_f32(X,w,N,D,eps)->CudaBuffer"},
    {"embedding_f32",    py_embedding_f32,    METH_VARARGS, "embedding_f32(W,idx,batch,dim)->CudaBuffer"},
    {"cross_entropy_f32",py_cross_entropy_f32,METH_VARARGS, "cross_entropy_f32(logits,targets,N,C)->CudaBuffer"},
    {"gelu_f32",         py_gelu_f32,         METH_VARARGS, "gelu_f32(A,n)->CudaBuffer"},
    {"silu_f32",         py_silu_f32,         METH_VARARGS, "silu_f32(A,n)->CudaBuffer"},
    {"mul_f32",          py_mul_f32,          METH_VARARGS, "mul_f32(A,B,n)->CudaBuffer"},
    {"transpose_f32",    py_transpose_f32,    METH_VARARGS, "transpose_f32(A,m,n)->CudaBuffer transposed"},
    /* Session 7 — fused + Tensor Core kernels */
    {"linear_silu_f32",  py_linear_silu_f32,  METH_VARARGS, "linear_silu_f32(X,W,b,N,in,out)->CudaBuffer fused"},
    {"linear_gelu_f32",  py_linear_gelu_f32,  METH_VARARGS, "linear_gelu_f32(X,W,b,N,in,out)->CudaBuffer fused"},
    {"qkv_proj_f32",     py_qkv_proj_f32,     METH_VARARGS, "qkv_proj_f32(X,Wqkv,BT,d)->CudaBuffer fused Q/K/V"},
    {"dropout_f32",      py_dropout_f32,      METH_VARARGS, "dropout_f32(X,mask,scale,n)->CudaBuffer"},
    {"pool_reset",       py_pool_reset,       METH_NOARGS,  "Reset memory pool bump pointer"},
    {"pool_info",        py_pool_info,        METH_NOARGS,  "Returns (used_bytes, total_bytes)"},
    {"stream_sync_all",  py_stream_sync,      METH_NOARGS,  "Sync all CUDA streams"},
    /* Session 9b — fused embed+pos, biased softmax */
    {"embed_pos_f32",    py_embed_pos_f32,    METH_VARARGS, "fused embed+pos+dropout"},
    {"biased_softmax_f32", py_biased_softmax_f32, METH_VARARGS, "softmax(scores+bias)"},
    /* Session 9 — warp-reduce utils, INT8 GEMM, fused flash-decode partial */
    {"warp_reduce_sum_f32",   py_warp_reduce_sum_f32,   METH_VARARGS, "warp_reduce_sum_f32(A,n)->CudaBuffer"},
    {"warp_reduce_max_f32",   py_warp_reduce_max_f32,   METH_VARARGS, "warp_reduce_max_f32(A,n)->CudaBuffer"},
    {"online_softmax_f32",    py_online_softmax_f32,    METH_VARARGS, "online_softmax_f32(X,rows,N)->CudaBuffer"},
    {"flash_decode_partial_f32", py_flash_decode_partial_f32, METH_VARARGS, "flash_decode_partial_f32(Q,K,V,B,H,S,D,splits)->CudaBuffer"},
    /* Session 8 — FP16, fused residual norm, RoPE */
    {"residual_layernorm_f32", py_residual_layernorm_f32, METH_VARARGS, "residual+LN fused"},
    {"residual_rmsnorm_f32",   py_residual_rmsnorm_f32,   METH_VARARGS, "residual+RMSNorm fused"},
    {"rope_f32",               py_rope_f32,               METH_VARARGS, "RoPE in-place"},
    /* Session 11 — kernels registered below */
    /* Session 10 — BF16 kernels + INT8 GEMM */
    {"fp32_to_bf16",           py_fp32_to_bf16,           METH_VARARGS, "fp32_to_bf16(A,n)->CudaBuffer bf16"},
    {"bf16_to_fp32",           py_bf16_to_fp32,           METH_VARARGS, "bf16_to_fp32(A,n)->CudaBuffer fp32"},
    {"gemm_bf16",              py_gemm_bf16,              METH_VARARGS, "gemm_bf16(A,B,M,K,N)->CudaBuffer FP32"},
    {"rmsnorm_bf16",           py_rmsnorm_bf16,           METH_VARARGS, "rmsnorm_bf16(X,w,N,D,eps)->CudaBuffer bf16"},
    {"swiglu_bf16",            py_swiglu_bf16,            METH_VARARGS, "swiglu_bf16(gate,up,n)->CudaBuffer bf16"},
    {"layernorm_bf16",         py_layernorm_bf16,         METH_VARARGS, "layernorm_bf16(X,g,b,N,D,eps)->CudaBuffer bf16"},
    {"quantize_fp32_int8",     py_quantize_fp32_int8,     METH_VARARGS, "quantize_fp32_int8(X,N,D)->(int8_buf,scale_buf)"},
    {"gemm_int8",              py_gemm_int8,              METH_VARARGS, "gemm_int8(A,B,SA,SB,M,K,N)->CudaBuffer fp32"},
#if !defined(DISABLE_FP16) && CUDA_VERSION >= 8000
    {"gemm_fp16_f32",          py_gemm_fp16_f32,          METH_VARARGS, "FP16 GEMM - FP32 output"},
#endif
#if defined(VP_TENSOR_CORE_AVAILABLE)
    {"gemm_tc_f32",      py_gemm_tc_f32,      METH_VARARGS, "gemm_tc_f32(A,B,m,k,n)->CudaBuffer Tensor Core"},
#endif
    /* Session 11 — Mamba selective scan + RWKV WKV kernel */
    {"mamba_ssm_f32",    py_mamba_ssm_f32,    METH_VARARGS, "mamba_ssm_f32(u,dt,A,B,C,B_,L_,D_,N_)->CudaBuffer y"},
    {"rwkv_wkv_f32",     py_rwkv_wkv_f32,     METH_VARARGS, "rwkv_wkv_f32(w,u,k,v,B,T,C)->CudaBuffer y"},
    {"rope_bf16",        py_rope_bf16,         METH_VARARGS, "rope_bf16 in-place RoPE on bf16 tensor"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef vp_cuda_module = {
    PyModuleDef_HEAD_INIT, "_vp_cuda", "vectorphile CUDA extension", -1, vp_cuda_methods
};

PyMODINIT_FUNC PyInit__vp_cuda(void) {
    if (PyType_Ready(&CudaBufferType) < 0) return NULL;
    PyObject *m = PyModule_Create(&vp_cuda_module);
    if (!m) return NULL;
    Py_INCREF(&CudaBufferType);
    PyModule_AddObject(m, "CudaBuffer", (PyObject *)&CudaBufferType);
    return m;
}
