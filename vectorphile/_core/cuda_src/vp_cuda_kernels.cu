/*
 * vp_cuda_kernels.cu — vectorphile native CUDA kernels (no Triton dependency)
 *
 * Compiled with:
 *   nvcc -O3 -arch=sm_89 --shared -Xcompiler /LD -o vp_cuda_ext.pyd
 *        -I<Python include> vp_cuda_kernels.cu <Python lib>
 *
 * Exposed Python functions (via Python C API):
 *   matmul_f32(a_ptr, b_ptr, c_ptr, M, K, N)
 *   softmax_f32(x_ptr, out_ptr, N, C)
 *   cross_entropy_fwd(logits_ptr, targets_ptr, loss_ptr, N, C)
 *   cross_entropy_bwd(logits_ptr, targets_ptr, grad_ptr, N, C, scale)
 *   relu_f32(x_ptr, out_ptr, n)
 *   gelu_f32(x_ptr, out_ptr, n)
 *   add_f32(a_ptr, b_ptr, out_ptr, n)
 *   mul_f32(a_ptr, b_ptr, out_ptr, n)
 *   layernorm_f32(x_ptr, w_ptr, b_ptr, out_ptr, N, C, eps)
 *   embedding_fwd(weight_ptr, idx_ptr, out_ptr, B, D)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

/* ── helpers ─────────────────────────────────────────────────────────────── */

static void* _get_ptr(PyObject* capsule_or_int) {
    if (PyCapsule_CheckExact(capsule_or_int))
        return PyCapsule_GetPointer(capsule_or_int, NULL);
    return (void*)(uintptr_t)PyLong_AsUnsignedLongLong(capsule_or_int);
}

#define BLOCK 256
#define WARP  32

/* ── elementwise kernels ─────────────────────────────────────────────────── */

__global__ void k_relu(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.f ? x[i] : 0.f;
}

__global__ void k_gelu(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = 0.5f * v * (1.f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

__global__ void k_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void k_mul(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

/* ── softmax (row-wise, shared memory reduce) ────────────────────────────── */

__global__ void k_softmax(const float* __restrict__ x, float* __restrict__ y,
                          int N, int C) {
    int row = blockIdx.x;
    if (row >= N) return;
    const float* row_x = x + row * C;
    float* row_y = y + row * C;

    // Max reduce
    float mx = -1e38f;
    for (int c = threadIdx.x; c < C; c += BLOCK) mx = fmaxf(mx, row_x[c]);
    __shared__ float smem[BLOCK];
    smem[threadIdx.x] = mx;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x+s]);
        __syncthreads();
    }
    mx = smem[0];

    // Sum of exp
    float sum = 0.f;
    for (int c = threadIdx.x; c < C; c += BLOCK) sum += expf(row_x[c] - mx);
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = BLOCK/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x+s];
        __syncthreads();
    }
    float inv_sum = 1.f / smem[0];

    for (int c = threadIdx.x; c < C; c += BLOCK)
        row_y[c] = expf(row_x[c] - mx) * inv_sum;
}

/* ── cross-entropy forward ───────────────────────────────────────────────── */

__global__ void k_ce_fwd(const float* __restrict__ logits,
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
    int t = targets[row];
    losses[row] = -(r[t] - mx - logf(sum));
}

/* ── cross-entropy backward ──────────────────────────────────────────────── */

__global__ void k_ce_bwd(const float* __restrict__ logits,
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

/* ── tiled matmul (C = A @ B, A: M×K, B: K×N) ───────────────────────────── */

#define TILE 16

__global__ void k_matmul(const float* __restrict__ A,
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
        As[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[row * K + ak] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[bk * N + col] : 0.f;
        __syncthreads();
        for (int i = 0; i < TILE; i++) acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

/* ── layer norm ──────────────────────────────────────────────────────────── */

__global__ void k_layernorm(const float* __restrict__ x,
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
    for (int c = 0; c < C; c++) { float d = rx[c] - mean; var += d*d; }
    var = var / C + eps;
    float inv_std = rsqrtf(var);
    for (int c = threadIdx.x; c < C; c += BLOCK)
        ry[c] = (rx[c] - mean) * inv_std * (w ? w[c] : 1.f) + (b ? b[c] : 0.f);
}

/* ── embedding forward ───────────────────────────────────────────────────── */

__global__ void k_embedding(const float* __restrict__ weight,
                            const int*   __restrict__ idx,
                            float*       __restrict__ out,
                            int B, int D) {
    int b = blockIdx.x;
    if (b >= B) return;
    int i = idx[b];
    for (int d = threadIdx.x; d < D; d += BLOCK)
        out[b * D + d] = weight[i * D + d];
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Python-callable wrappers
 * Each function:
 *   - accepts Python float/bytes buffer objects OR integer device pointers
 *   - copies host→device if needed, runs kernel, copies device→host
 *   - returns Python bytes of result floats
 * ═══════════════════════════════════════════════════════════════════════════ */

/* helper: alloc device, copy from host bytes buffer */
static float* _h2d_f32(const float* host, int n) {
    float* dev; cudaMalloc(&dev, n * sizeof(float));
    cudaMemcpy(dev, host, n * sizeof(float), cudaMemcpyHostToDevice);
    return dev;
}
static int* _h2d_i32(const int* host, int n) {
    int* dev; cudaMalloc(&dev, n * sizeof(int));
    cudaMemcpy(dev, host, n * sizeof(int), cudaMemcpyHostToDevice);
    return dev;
}
static PyObject* _d2py(float* dev, int n) {
    float* host = (float*)malloc(n * sizeof(float));
    cudaMemcpy(host, dev, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev);
    PyObject* ret = PyBytes_FromStringAndSize((char*)host, n * sizeof(float));
    free(host);
    return ret;
}

/* relu_f32(x_bytes, n) -> bytes */
static PyObject* py_relu(PyObject* self, PyObject* args) {
    Py_buffer x; int n;
    if (!PyArg_ParseTuple(args, "y*i", &x, &n)) return NULL;
    float* dx = _h2d_f32((const float*)x.buf, n);
    float* dy; cudaMalloc(&dy, n * sizeof(float));
    k_relu<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(dx, dy, n);
    cudaFree(dx);
    PyBuffer_Release(&x);
    return _d2py(dy, n);
}

/* gelu_f32(x_bytes, n) -> bytes */
static PyObject* py_gelu(PyObject* self, PyObject* args) {
    Py_buffer x; int n;
    if (!PyArg_ParseTuple(args, "y*i", &x, &n)) return NULL;
    float* dx = _h2d_f32((const float*)x.buf, n);
    float* dy; cudaMalloc(&dy, n * sizeof(float));
    k_gelu<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(dx, dy, n);
    cudaFree(dx);
    PyBuffer_Release(&x);
    return _d2py(dy, n);
}

/* add_f32(a_bytes, b_bytes, n) -> bytes */
static PyObject* py_add(PyObject* self, PyObject* args) {
    Py_buffer a, b; int n;
    if (!PyArg_ParseTuple(args, "y*y*i", &a, &b, &n)) return NULL;
    float* da = _h2d_f32((const float*)a.buf, n);
    float* db = _h2d_f32((const float*)b.buf, n);
    float* dc; cudaMalloc(&dc, n * sizeof(float));
    k_add<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(da, db, dc, n);
    cudaFree(da); cudaFree(db);
    PyBuffer_Release(&a); PyBuffer_Release(&b);
    return _d2py(dc, n);
}

/* mul_f32(a_bytes, b_bytes, n) -> bytes */
static PyObject* py_mul(PyObject* self, PyObject* args) {
    Py_buffer a, b; int n;
    if (!PyArg_ParseTuple(args, "y*y*i", &a, &b, &n)) return NULL;
    float* da = _h2d_f32((const float*)a.buf, n);
    float* db = _h2d_f32((const float*)b.buf, n);
    float* dc; cudaMalloc(&dc, n * sizeof(float));
    k_mul<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(da, db, dc, n);
    cudaFree(da); cudaFree(db);
    PyBuffer_Release(&a); PyBuffer_Release(&b);
    return _d2py(dc, n);
}

/* softmax_f32(x_bytes, N, C) -> bytes */
static PyObject* py_softmax(PyObject* self, PyObject* args) {
    Py_buffer x; int N, C;
    if (!PyArg_ParseTuple(args, "y*ii", &x, &N, &C)) return NULL;
    float* dx = _h2d_f32((const float*)x.buf, N*C);
    float* dy; cudaMalloc(&dy, N*C * sizeof(float));
    k_softmax<<<N, BLOCK>>>(dx, dy, N, C);
    cudaFree(dx);
    PyBuffer_Release(&x);
    return _d2py(dy, N*C);
}

/* cross_entropy_fwd(logits_bytes, targets_bytes, N, C) -> loss_bytes (N floats) */
static PyObject* py_ce_fwd(PyObject* self, PyObject* args) {
    Py_buffer logits, tgts; int N, C;
    if (!PyArg_ParseTuple(args, "y*y*ii", &logits, &tgts, &N, &C)) return NULL;
    float* dl = _h2d_f32((const float*)logits.buf, N*C);
    int*   dt = _h2d_i32((const int*)tgts.buf, N);
    float* dout; cudaMalloc(&dout, N * sizeof(float));
    k_ce_fwd<<<N, 1>>>(dl, dt, dout, N, C);
    cudaFree(dl); cudaFree(dt);
    PyBuffer_Release(&logits); PyBuffer_Release(&tgts);
    return _d2py(dout, N);
}

/* cross_entropy_bwd(logits_bytes, targets_bytes, N, C, scale) -> grad_bytes (N*C floats) */
static PyObject* py_ce_bwd(PyObject* self, PyObject* args) {
    Py_buffer logits, tgts; int N, C; float scale;
    if (!PyArg_ParseTuple(args, "y*y*iif", &logits, &tgts, &N, &C, &scale)) return NULL;
    float* dl = _h2d_f32((const float*)logits.buf, N*C);
    int*   dt = _h2d_i32((const int*)tgts.buf, N);
    float* dgrad; cudaMalloc(&dgrad, N*C * sizeof(float));
    /* launch: N blocks, min(C,BLOCK) threads */
    int thr = C < BLOCK ? C : BLOCK;
    k_ce_bwd<<<N, thr>>>(dl, dt, dgrad, N, C, scale);
    cudaFree(dl); cudaFree(dt);
    PyBuffer_Release(&logits); PyBuffer_Release(&tgts);
    return _d2py(dgrad, N*C);
}

/* matmul_f32(a_bytes, b_bytes, M, K, N) -> bytes */
static PyObject* py_matmul(PyObject* self, PyObject* args) {
    Py_buffer a, b; int M, K, N;
    if (!PyArg_ParseTuple(args, "y*y*iii", &a, &b, &M, &K, &N)) return NULL;
    float* da = _h2d_f32((const float*)a.buf, M*K);
    float* db = _h2d_f32((const float*)b.buf, K*N);
    float* dc; cudaMalloc(&dc, M*N * sizeof(float));
    dim3 block(TILE, TILE), grid((N+TILE-1)/TILE, (M+TILE-1)/TILE);
    k_matmul<<<grid, block>>>(da, db, dc, M, K, N);
    cudaFree(da); cudaFree(db);
    PyBuffer_Release(&a); PyBuffer_Release(&b);
    return _d2py(dc, M*N);
}

/* layernorm_f32(x, w_or_None, b_or_None, N, C, eps) -> bytes */
static PyObject* py_layernorm(PyObject* self, PyObject* args) {
    Py_buffer x; PyObject *w_obj, *b_obj; int N, C; float eps;
    if (!PyArg_ParseTuple(args, "y*OOiif", &x, &w_obj, &b_obj, &N, &C, &eps)) return NULL;
    float* dx = _h2d_f32((const float*)x.buf, N*C);
    Py_buffer wb, bb;
    float *dw = NULL, *db_d = NULL;
    int has_w = (w_obj != Py_None && PyObject_GetBuffer(w_obj, &wb, PyBUF_SIMPLE) == 0);
    int has_b = (b_obj != Py_None && PyObject_GetBuffer(b_obj, &bb, PyBUF_SIMPLE) == 0);
    if (has_w) dw = _h2d_f32((const float*)wb.buf, C);
    if (has_b) db_d = _h2d_f32((const float*)bb.buf, C);
    float* dy; cudaMalloc(&dy, N*C * sizeof(float));
    k_layernorm<<<N, BLOCK>>>(dx, dw, db_d, dy, N, C, eps);
    cudaFree(dx);
    if (dw) cudaFree(dw);
    if (db_d) cudaFree(db_d);
    PyBuffer_Release(&x);
    if (has_w) PyBuffer_Release(&wb);
    if (has_b) PyBuffer_Release(&bb);
    return _d2py(dy, N*C);
}

/* embedding_fwd(weight_bytes, idx_bytes, B, D) -> bytes */
static PyObject* py_embedding(PyObject* self, PyObject* args) {
    Py_buffer w, idx; int B, D;
    if (!PyArg_ParseTuple(args, "y*y*ii", &w, &idx, &B, &D)) return NULL;
    int V = (int)(w.len / sizeof(float)) / D;
    float* dw = _h2d_f32((const float*)w.buf, V*D);
    int*   di = _h2d_i32((const int*)idx.buf, B);
    float* dout; cudaMalloc(&dout, B*D * sizeof(float));
    k_embedding<<<B, BLOCK>>>(dw, di, dout, B, D);
    cudaFree(dw); cudaFree(di);
    PyBuffer_Release(&w); PyBuffer_Release(&idx);
    return _d2py(dout, B*D);
}

/* ── module definition ───────────────────────────────────────────────────── */

static PyMethodDef methods[] = {
    {"relu_f32",        py_relu,      METH_VARARGS, "ReLU on GPU"},
    {"gelu_f32",        py_gelu,      METH_VARARGS, "GELU on GPU"},
    {"add_f32",         py_add,       METH_VARARGS, "elementwise add on GPU"},
    {"mul_f32",         py_mul,       METH_VARARGS, "elementwise mul on GPU"},
    {"softmax_f32",     py_softmax,   METH_VARARGS, "row softmax on GPU"},
    {"cross_entropy_fwd", py_ce_fwd,  METH_VARARGS, "cross-entropy forward on GPU"},
    {"cross_entropy_bwd", py_ce_bwd,  METH_VARARGS, "cross-entropy backward on GPU"},
    {"matmul_f32",      py_matmul,    METH_VARARGS, "tiled matmul on GPU"},
    {"layernorm_f32",   py_layernorm, METH_VARARGS, "LayerNorm on GPU"},
    {"embedding_fwd",   py_embedding, METH_VARARGS, "embedding lookup on GPU"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moddef = {
    PyModuleDef_HEAD_INIT, "vp_cuda_ext", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_vp_cuda_ext(void) {
    return PyModule_Create(&moddef);
}
