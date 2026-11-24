"""
vectorphile OpenCL GPU backend — pure ctypes, zero third-party dependencies.

Wraps the OpenCL 1.2 C API via ctypes.WinDLL('OpenCL') (Windows) or
ctypes.CDLL('libOpenCL.so') (Linux/macOS).  JIT-compiles kernel source
strings on first use and caches the compiled programs.

Public API
----------
device_info()   → dict with platform/device name, compute units, memory
is_available()  → bool
exp_f32         (src, dst, n)
tanh_f32        (src, dst, n)
sigmoid_f32     (src, dst, n)
relu_f32        (src, dst, n)
silu_f32        (src, dst, n)
gelu_f32        (src, dst, n)
log_f32         (src, dst, n)
sqrt_f32        (src, dst, n)
abs_f32         (src, dst, n)
neg_f32         (src, dst, n)
fused_chain_f32 (src, dst, n, op_codes)
matmul_f32      (A, B, C, M, K, N)
softmax_f32     (src, dst, rows, cols)
layer_norm_f32  (x, w, b, dst, N, D, eps)

All array arguments are ctypes pointers to float32 data (use
ctypes.cast(arr.buffer_info()[0], ctypes.POINTER(ctypes.c_float))).
"""
from __future__ import annotations

import ctypes
import ctypes.util
import sys
import struct
import array as _array
from typing import Optional

# ── OpenCL constants ──────────────────────────────────────────────────────────
CL_SUCCESS                   = 0
CL_DEVICE_TYPE_GPU           = 4
CL_DEVICE_TYPE_ALL           = 0xFFFFFFFF
CL_DEVICE_NAME               = 0x102B
CL_DEVICE_MAX_COMPUTE_UNITS  = 0x1002
CL_DEVICE_GLOBAL_MEM_SIZE    = 0x101F
CL_DEVICE_LOCAL_MEM_SIZE     = 0x1023
CL_DEVICE_MAX_WORK_GROUP_SIZE= 0x1004
CL_DEVICE_VENDOR             = 0x102C
CL_MEM_READ_ONLY             = (1 << 2)
CL_MEM_WRITE_ONLY            = (1 << 1)
CL_MEM_READ_WRITE            = (1 << 0)
CL_MEM_COPY_HOST_PTR         = (1 << 5)
CL_MEM_USE_HOST_PTR          = (1 << 3)
CL_MEM_ALLOC_HOST_PTR        = (1 << 4)
CL_QUEUE_PROFILING_ENABLE    = (1 << 0)
CL_PLATFORM_NAME             = 0x0902

# ── Load OpenCL shared library ────────────────────────────────────────────────
def _load_opencl():
    if sys.platform == "win32":
        return ctypes.WinDLL("OpenCL")
    lib_path = ctypes.util.find_library("OpenCL")
    if lib_path:
        return ctypes.CDLL(lib_path)
    for path in ["/usr/lib/libOpenCL.so", "/usr/local/lib/libOpenCL.so",
                 "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1"]:
        try:
            return ctypes.CDLL(path)
        except OSError:
            pass
    return None

try:
    _ocl = _load_opencl()
    _OCL_AVAILABLE = (_ocl is not None)
except Exception:
    _ocl = None
    _OCL_AVAILABLE = False

# ── Declare argtypes so ctypes correctly marshals 64-bit pointer values ───────
# Without argtypes, large 64-bit cl_platform_id / cl_device_id values overflow
# ctypes' default c_long (32-bit on Windows).

def _setup_ocl_signatures(lib) -> None:
    """Assign argtypes + restype for every OpenCL entry point we use."""
    vp = ctypes.c_void_p
    i32 = ctypes.c_int32
    u32 = ctypes.c_uint32
    u64 = ctypes.c_uint64
    sz  = ctypes.c_size_t
    pp  = ctypes.POINTER(ctypes.c_void_p)

    def _sig(name, args, res=i32):
        try:
            fn = getattr(lib, name)
            fn.argtypes = args
            fn.restype  = res
        except AttributeError:
            pass  # older OpenCL — skip gracefully

    _sig("clGetPlatformIDs",    [u32, pp, ctypes.POINTER(u32)])
    _sig("clGetPlatformInfo",   [vp, u32, sz, ctypes.c_void_p, ctypes.POINTER(sz)])
    _sig("clGetDeviceIDs",      [vp, u64, u32, pp, ctypes.POINTER(u32)])
    _sig("clGetDeviceInfo",     [vp, u32, sz, ctypes.c_void_p, ctypes.POINTER(sz)])
    _sig("clCreateContext",     [ctypes.c_void_p, u32, pp, ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.POINTER(i32)], vp)
    _sig("clCreateCommandQueue",[vp, vp, u64, ctypes.POINTER(i32)], vp)
    _sig("clReleaseContext",    [vp])
    _sig("clReleaseCommandQueue",[vp])
    _sig("clCreateBuffer",      [vp, u64, sz, ctypes.c_void_p, ctypes.POINTER(i32)], vp)
    _sig("clEnqueueWriteBuffer",[vp, vp, u32, sz, sz, ctypes.c_void_p,
                                  u32, ctypes.c_void_p, ctypes.c_void_p])
    _sig("clEnqueueReadBuffer", [vp, vp, u32, sz, sz, ctypes.c_void_p,
                                  u32, ctypes.c_void_p, ctypes.c_void_p])
    _sig("clReleaseMemObject",  [vp])
    _sig("clCreateProgramWithSource",[vp, u32, ctypes.POINTER(ctypes.c_char_p),
                                       ctypes.POINTER(sz), ctypes.POINTER(i32)], vp)
    _sig("clBuildProgram",      [vp, u32, pp, ctypes.c_char_p,
                                  ctypes.c_void_p, ctypes.c_void_p])
    _sig("clGetProgramBuildInfo",[vp, vp, u32, sz, ctypes.c_void_p, ctypes.POINTER(sz)])
    _sig("clCreateKernel",      [vp, ctypes.c_char_p, ctypes.POINTER(i32)], vp)
    _sig("clSetKernelArg",      [vp, u32, sz, ctypes.c_void_p])
    _sig("clEnqueueNDRangeKernel",[vp, vp, u32, ctypes.POINTER(sz),
                                    ctypes.POINTER(sz), ctypes.POINTER(sz),
                                    u32, ctypes.c_void_p, ctypes.c_void_p])
    _sig("clFinish",            [vp])
    _sig("clReleaseKernel",     [vp])
    _sig("clReleaseProgram",    [vp])

if _OCL_AVAILABLE and _ocl is not None:
    _setup_ocl_signatures(_ocl)

# ── OpenCL kernel source strings ─────────────────────────────────────────────

_UNARY_KERNEL_SRC = r"""
__kernel void vp_exp   (__global const float* s, __global float* d, int n){int i=get_global_id(0);if(i<n)d[i]=exp(s[i]);}
__kernel void vp_tanh  (__global const float* s, __global float* d, int n){int i=get_global_id(0);if(i<n)d[i]=tanh(s[i]);}
__kernel void vp_sigmoid(__global const float* s,__global float* d, int n){int i=get_global_id(0);if(i<n)d[i]=1.0f/(1.0f+exp(-s[i]));}
__kernel void vp_relu  (__global const float* s, __global float* d, int n){int i=get_global_id(0);if(i<n)d[i]=max(s[i],0.0f);}
__kernel void vp_silu  (__global const float* s, __global float* d, int n){int i=get_global_id(0);if(i<n){float x=s[i];d[i]=x/(1.0f+exp(-x));}}
__kernel void vp_gelu  (__global const float* s, __global float* d, int n){int i=get_global_id(0);if(i<n){float x=s[i];d[i]=x*0.5f*(1.0f+erf(x*0.7071067811865476f));}}
__kernel void vp_log   (__global const float* s, __global float* d, int n){int i=get_global_id(0);if(i<n)d[i]=log(s[i]);}
__kernel void vp_sqrt  (__global const float* s, __global float* d, int n){int i=get_global_id(0);if(i<n)d[i]=sqrt(s[i]);}
__kernel void vp_abs_f (__global const float* s, __global float* d, int n){int i=get_global_id(0);if(i<n)d[i]=fabs(s[i]);}
__kernel void vp_neg_f (__global const float* s, __global float* d, int n){int i=get_global_id(0);if(i<n)d[i]=-s[i];}
"""

# Chain kernel: applies a sequence of up to 8 unary ops in one pass.
# Op codes mirror VP_CHAIN_* in vp_core.c.
_CHAIN_KERNEL_SRC = r"""
#define VP_RELU    0
#define VP_SIGMOID 1
#define VP_TANH    2
#define VP_EXP     3
#define VP_LOG     4
#define VP_SQRT    5
#define VP_ABS     6
#define VP_NEG     7
#define VP_GELU    8
#define VP_SILU    9
#define VP_SIGN   10
#define VP_ERF    11

__kernel void vp_fused_chain(
    __global const float* src,
    __global float*       dst,
    int n,
    int op0, int op1, int op2, int op3,
    int op4, int op5, int op6, int op7,
    int nops)
{
    int i = get_global_id(0);
    if (i >= n) return;
    float x = src[i];
    int ops[8] = {op0, op1, op2, op3, op4, op5, op6, op7};
    for (int k = 0; k < nops; k++) {
        switch (ops[k]) {
        case VP_RELU:    x = max(x, 0.0f);                               break;
        case VP_SIGMOID: x = 1.0f / (1.0f + exp(-x));                   break;
        case VP_TANH:    x = tanh(x);                                    break;
        case VP_EXP:     x = exp(x);                                     break;
        case VP_LOG:     x = log(x);                                     break;
        case VP_SQRT:    x = sqrt(x);                                    break;
        case VP_ABS:     x = fabs(x);                                    break;
        case VP_NEG:     x = -x;                                         break;
        case VP_GELU:    x = x * 0.5f * (1.0f + erf(x * 0.7071067f)); break;
        case VP_SILU:    x = x / (1.0f + exp(-x));                      break;
        case VP_SIGN:    x = (x > 0.0f) ? 1.0f : (x < 0.0f ? -1.0f : 0.0f); break;
        case VP_ERF:     x = erf(x);                                     break;
        }
    }
    dst[i] = x;
}
"""

# Tiled matrix multiply: C(M,N) = A(M,K) * B(K,N)
_MATMUL_KERNEL_SRC = r"""
#define TILE 16

__kernel __attribute__((reqd_work_group_size(TILE, TILE, 1)))
void vp_matmul(
    __global const float* A,
    __global const float* B,
    __global float*       C,
    int M, int K, int N)
{
    __local float sA[TILE][TILE];
    __local float sB[TILE][TILE];

    int row  = get_global_id(1);
    int col  = get_global_id(0);
    int lrow = get_local_id(1);
    int lcol = get_local_id(0);

    float acc = 0.0f;
    int ntiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < ntiles; t++) {
        int ar = row,       ac = t * TILE + lcol;
        int br = t * TILE + lrow, bc = col;
        sA[lrow][lcol] = (ar < M && ac < K) ? A[ar * K + ac] : 0.0f;
        sB[lrow][lcol] = (br < K && bc < N) ? B[br * N + bc] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE; k++) acc += sA[lrow][k] * sB[k][lcol];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
"""

# Row-wise softmax: each work-group handles one row
_SOFTMAX_KERNEL_SRC = r"""
__kernel void vp_softmax(
    __global const float* src,
    __global float*       dst,
    int rows, int cols)
{
    int row = get_global_id(0);
    if (row >= rows) return;
    const __global float* s = src + row * cols;
    __global float*       d = dst + row * cols;

    /* max for numerical stability */
    float mx = s[0];
    for (int j = 1; j < cols; j++) mx = max(mx, s[j]);

    float sum = 0.0f;
    for (int j = 0; j < cols; j++) { d[j] = exp(s[j] - mx); sum += d[j]; }
    float inv = 1.0f / (sum + 1e-9f);
    for (int j = 0; j < cols; j++) d[j] *= inv;
}
"""

# Layer norm: one work-item per row, weight/bias applied per column
_LAYER_NORM_SRC = r"""
__kernel void vp_layer_norm(
    __global const float* x,
    __global const float* w,
    __global const float* b,
    __global float*       dst,
    int N, int D, float eps)
{
    int row = get_global_id(0);
    if (row >= N) return;
    const __global float* xr = x   + row * D;
    __global float*       dr = dst + row * D;

    float mean = 0.0f;
    for (int d = 0; d < D; d++) mean += xr[d];
    mean /= D;

    float var = 0.0f;
    for (int d = 0; d < D; d++) { float diff = xr[d] - mean; var += diff * diff; }
    float inv_std = 1.0f / sqrt(var / D + eps);

    for (int d = 0; d < D; d++)
        dr[d] = (xr[d] - mean) * inv_std * w[d] + b[d];
}
"""

# Fused AdamW update — one kernel, all 4 arrays updated in a single pass.
_ADAM_KERNEL_SRC = r"""
__kernel void vp_adam_step(
    __global float* p,
    __global const float* g,
    __global float* m,
    __global float* v,
    float lr, float beta1, float beta2, float eps, float wd,
    float inv_bc1, float inv_bc2,
    int n)
{
    int i = get_global_id(0);
    if (i >= n) return;
    float gi = g[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * gi;
    float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    float mhat = mi * inv_bc1;
    float vhat = vi * inv_bc2;
    p[i] -= lr * (mhat / (sqrt(vhat) + eps) + wd * p[i]);
}
"""

# Flash Attention: online softmax over tiles, O(1) HBM traffic per Q block.
# Each work-item handles one (batch*head, query_row) pair.
# TILE_K = 64: processes 64 K/V rows per inner loop iteration.
_FLASH_ATTN_SRC = r"""
#define TILE_K 64

__kernel void vp_flash_attn(
    __global const float* Q,     /* [B*H, Sq, D] */
    __global const float* K,     /* [B*H, Sk, D] */
    __global const float* V,     /* [B*H, Sk, D] */
    __global float*       O,     /* [B*H, Sq, D] */
    int Sq, int Sk, int D,
    float scale)                 /* 1/sqrt(D) */
{
    int bh  = get_global_id(1);  /* batch*head index */
    int q_i = get_global_id(0);  /* query row */
    if (q_i >= Sq) return;

    int Q_off = bh * Sq * D + q_i * D;
    int K_off = bh * Sk * D;
    int V_off = bh * Sk * D;
    int O_off = bh * Sq * D + q_i * D;

    /* Online softmax state */
    float m = -1e38f;   /* running max */
    float l = 0.0f;     /* running sum of exp(s - m) */

    /* Accumulator for output row */
    float acc[256];     /* D <= 256 */
    for (int d = 0; d < D; d++) acc[d] = 0.0f;

    /* Tile over K dimension */
    for (int k_start = 0; k_start < Sk; k_start += TILE_K) {
        int k_end = k_start + TILE_K;
        if (k_end > Sk) k_end = Sk;

        /* Compute QK^T scores for this tile */
        float scores[TILE_K];
        float tile_max = -1e38f;
        for (int k = k_start; k < k_end; k++) {
            float dot = 0.0f;
            int K_row = K_off + k * D;
            for (int d = 0; d < D; d++)
                dot += Q[Q_off + d] * K[K_row + d];
            dot *= scale;
            scores[k - k_start] = dot;
            if (dot > tile_max) tile_max = dot;
        }

        /* Update running max and rescale accumulator */
        float m_new = m > tile_max ? m : tile_max;
        float scale_old = exp(m - m_new);
        float l_new = l * scale_old;
        for (int d = 0; d < D; d++) acc[d] *= scale_old;

        /* Accumulate V rows weighted by exp(score - m_new) */
        for (int k = k_start; k < k_end; k++) {
            float w = exp(scores[k - k_start] - m_new);
            l_new += w;
            int V_row = V_off + k * D;
            for (int d = 0; d < D; d++)
                acc[d] += w * V[V_row + d];
        }

        m = m_new;
        l = l_new;
    }

    /* Normalize and write output */
    float inv_l = 1.0f / l;
    for (int d = 0; d < D; d++)
        O[O_off + d] = acc[d] * inv_l;
}
"""

# ── OpenCL device + context manager ──────────────────────────────────────────

class _VPOpenCLDevice:
    """Singleton: one context + command queue for the NVIDIA GPU."""

    def __init__(self):
        self._ctx   = None
        self._queue = None
        self._dev   = None
        self._programs: dict = {}   # source_key → cl_program
        self._kernels: dict  = {}   # (source_key, name) → cl_kernel
        self._info   = {}
        self._ready  = False
        self._init()

    # ── OpenCL helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _check(ret, msg="OpenCL error"):
        if ret != CL_SUCCESS:
            raise RuntimeError(f"{msg}: error code {ret}")

    def _get_device_str(self, dev, param) -> str:
        buf = ctypes.create_string_buffer(256)
        sz  = ctypes.c_size_t(0)
        _ocl.clGetDeviceInfo(dev, ctypes.c_uint32(param),
                              ctypes.c_size_t(256), buf, ctypes.byref(sz))
        return buf.value.decode("utf-8", errors="replace").strip()

    def _get_device_uint64(self, dev, param) -> int:
        val = ctypes.c_uint64(0)
        _ocl.clGetDeviceInfo(dev, ctypes.c_uint32(param),
                              ctypes.c_size_t(8), ctypes.byref(val), None)
        return val.value

    def _get_device_uint(self, dev, param) -> int:
        val = ctypes.c_uint32(0)
        _ocl.clGetDeviceInfo(dev, ctypes.c_uint32(param),
                              ctypes.c_size_t(4), ctypes.byref(val), None)
        return val.value

    # ── Initialization ────────────────────────────────────────────────────────

    def _init(self):
        if not _OCL_AVAILABLE:
            return
        try:
            self._init_device()
            self._ready = True
        except Exception as e:
            self._ready = False

    def _init_device(self):
        # Enumerate platforms
        n_plat = ctypes.c_uint32(0)
        _ocl.clGetPlatformIDs(0, None, ctypes.byref(n_plat))
        if n_plat.value == 0:
            raise RuntimeError("No OpenCL platforms found")

        PlatformArray = ctypes.c_void_p * n_plat.value
        platforms = PlatformArray()
        _ocl.clGetPlatformIDs(n_plat.value, ctypes.cast(platforms, ctypes.POINTER(ctypes.c_void_p)), None)

        # Prefer the NVIDIA platform; fall back to any GPU
        chosen_dev_vp: Optional[ctypes.c_void_p] = None
        chosen_plat_vp: Optional[ctypes.c_void_p] = None

        for idx in range(n_plat.value):
            plat_vp = ctypes.c_void_p(platforms[idx])
            # Get platform name
            buf = ctypes.create_string_buffer(256)
            sz  = ctypes.c_size_t(0)
            _ocl.clGetPlatformInfo(plat_vp, CL_PLATFORM_NAME, ctypes.c_size_t(256),
                                    buf, ctypes.byref(sz))
            pname = buf.value.decode("utf-8", errors="replace")

            n_dev = ctypes.c_uint32(0)
            ret = _ocl.clGetDeviceIDs(plat_vp, ctypes.c_uint64(CL_DEVICE_TYPE_GPU),
                                       ctypes.c_uint32(0), None, ctypes.byref(n_dev))
            if ret != CL_SUCCESS or n_dev.value == 0:
                continue

            DevArray = ctypes.c_void_p * n_dev.value
            devs = DevArray()
            _ocl.clGetDeviceIDs(plat_vp, ctypes.c_uint64(CL_DEVICE_TYPE_GPU),
                                 ctypes.c_uint32(n_dev.value),
                                 ctypes.cast(devs, ctypes.POINTER(ctypes.c_void_p)), None)

            dev_vp = ctypes.c_void_p(devs[0])
            if chosen_dev_vp is None or "NVIDIA" in pname or "nvidia" in pname.lower():
                chosen_dev_vp  = dev_vp
                chosen_plat_vp = plat_vp

        if chosen_dev_vp is None:
            raise RuntimeError("No GPU device found via OpenCL")

        self._dev      = chosen_dev_vp
        self._platform = chosen_plat_vp

        # Build info dict (pass c_void_p to device-info helpers)
        self._info = {
            "name":          self._get_device_str(chosen_dev_vp, CL_DEVICE_NAME),
            "vendor":        self._get_device_str(chosen_dev_vp, CL_DEVICE_VENDOR),
            "compute_units": self._get_device_uint(chosen_dev_vp, CL_DEVICE_MAX_COMPUTE_UNITS),
            "global_mem_mb": self._get_device_uint64(chosen_dev_vp, CL_DEVICE_GLOBAL_MEM_SIZE) // (1024*1024),
            "local_mem_kb":  self._get_device_uint64(chosen_dev_vp, CL_DEVICE_LOCAL_MEM_SIZE) // 1024,
            "max_wg_size":   self._get_device_uint(chosen_dev_vp, CL_DEVICE_MAX_WORK_GROUP_SIZE),
        }

        # Create context — keep a stable array for the device pointer
        err = ctypes.c_int32(0)
        dev_arr = (ctypes.c_void_p * 1)(chosen_dev_vp)
        ctx_int = _ocl.clCreateContext(
            None, ctypes.c_uint32(1),
            ctypes.cast(dev_arr, ctypes.POINTER(ctypes.c_void_p)),
            None, None, ctypes.byref(err))
        self._check(err.value, "clCreateContext")
        self._ctx = ctypes.c_void_p(ctx_int)

        # Create command queue (CL 1.2 style)
        q_int = _ocl.clCreateCommandQueue(
            self._ctx, chosen_dev_vp, ctypes.c_uint64(0), ctypes.byref(err))
        self._check(err.value, "clCreateCommandQueue")
        self._queue = ctypes.c_void_p(q_int)

    # ── Kernel compilation ────────────────────────────────────────────────────

    def _compile_program(self, src_key: str, src: str):
        if src_key in self._programs:
            return
        err = ctypes.c_int32(0)
        c_src = src.encode("utf-8")
        c_src_ptr   = ctypes.c_char_p(c_src)
        c_src_len   = ctypes.c_size_t(len(c_src))
        prog_int = _ocl.clCreateProgramWithSource(
            self._ctx, ctypes.c_uint32(1),
            ctypes.byref(c_src_ptr),
            ctypes.byref(c_src_len),
            ctypes.byref(err))
        self._check(err.value, "clCreateProgramWithSource")
        prog = ctypes.c_void_p(prog_int)

        ret = _ocl.clBuildProgram(prog, ctypes.c_uint32(0), None,
                                   b"-cl-fast-relaxed-math", None, None)
        if ret != CL_SUCCESS:
            # Get build log
            log_sz = ctypes.c_size_t(0)
            _ocl.clGetProgramBuildInfo(prog, self._dev, ctypes.c_uint32(0x1183),
                                        ctypes.c_size_t(0), None, ctypes.byref(log_sz))
            log_buf = ctypes.create_string_buffer(log_sz.value)
            _ocl.clGetProgramBuildInfo(prog, self._dev, ctypes.c_uint32(0x1183),
                                        log_sz, log_buf, None)
            raise RuntimeError(f"OpenCL build error:\n{log_buf.value.decode()}")

        self._programs[src_key] = prog

    def _get_kernel(self, src_key: str, src: str, name: str):
        key = (src_key, name)
        if key not in self._kernels:
            self._compile_program(src_key, src)
            err = ctypes.c_int32(0)
            k_int = _ocl.clCreateKernel(
                self._programs[src_key], name.encode("utf-8"), ctypes.byref(err))
            self._check(err.value, f"clCreateKernel({name})")
            # clCreateKernel returns Python int when restype=c_void_p; wrap it
            self._kernels[key] = ctypes.c_void_p(k_int)
        return self._kernels[key]

    # ── Buffer management ─────────────────────────────────────────────────────

    def alloc(self, n_floats: int, flags=CL_MEM_READ_WRITE) -> ctypes.c_void_p:
        err = ctypes.c_int32(0)
        buf_int = _ocl.clCreateBuffer(
            self._ctx, ctypes.c_uint64(flags),
            ctypes.c_size_t(n_floats * 4), None, ctypes.byref(err))
        self._check(err.value, "clCreateBuffer")
        # clCreateBuffer returns a Python int when restype=c_void_p; wrap it so
        # _set_arg's isinstance(val, c_void_p) branch fires and passes 8 bytes.
        return ctypes.c_void_p(buf_int)

    def upload(self, cl_buf: ctypes.c_void_p, host_ptr, n_floats: int):
        self._check(
            _ocl.clEnqueueWriteBuffer(
                self._queue, cl_buf, ctypes.c_uint32(1),
                ctypes.c_size_t(0), ctypes.c_size_t(n_floats * 4),
                host_ptr, ctypes.c_uint32(0), None, None),
            "clEnqueueWriteBuffer")

    def download(self, cl_buf: ctypes.c_void_p, host_ptr, n_floats: int):
        self._check(
            _ocl.clEnqueueReadBuffer(
                self._queue, cl_buf, ctypes.c_uint32(1),
                ctypes.c_size_t(0), ctypes.c_size_t(n_floats * 4),
                host_ptr, ctypes.c_uint32(0), None, None),
            "clEnqueueReadBuffer")

    def release_buf(self, cl_buf: ctypes.c_void_p):
        _ocl.clReleaseMemObject(cl_buf)

    def finish(self):
        _ocl.clFinish(self._queue)

    # ── Kernel dispatch helpers ───────────────────────────────────────────────

    def _set_arg(self, kernel, idx: int, val):
        """Set a kernel argument. Handles cl_mem (c_void_p) and scalars."""
        ui = ctypes.c_uint32(idx)
        if isinstance(val, ctypes.c_void_p):
            # cl_mem is a pointer — pass by-value as a pointer-sized word
            _ocl.clSetKernelArg(kernel, ui, ctypes.c_size_t(ctypes.sizeof(ctypes.c_void_p)),
                                  ctypes.byref(val))
        elif isinstance(val, ctypes.c_float):
            _ocl.clSetKernelArg(kernel, ui, ctypes.c_size_t(4), ctypes.byref(val))
        elif isinstance(val, float):
            v = ctypes.c_float(val)
            _ocl.clSetKernelArg(kernel, ui, ctypes.c_size_t(4), ctypes.byref(v))
        else:
            v = ctypes.c_int32(int(val))
            _ocl.clSetKernelArg(kernel, ui, ctypes.c_size_t(4), ctypes.byref(v))

    def _launch_1d(self, kernel, global_size: int, local_size: int = 256):
        gs_arr = (ctypes.c_size_t * 1)(((global_size + local_size - 1) // local_size) * local_size)
        ls_arr = (ctypes.c_size_t * 1)(local_size)
        self._check(
            _ocl.clEnqueueNDRangeKernel(
                self._queue, kernel, ctypes.c_uint32(1), None,
                gs_arr, ls_arr, ctypes.c_uint32(0), None, None),
            "clEnqueueNDRangeKernel(1D)")

    def _launch_2d(self, kernel, gx: int, gy: int, lx: int = 16, ly: int = 16):
        gsize = (ctypes.c_size_t * 2)(
            ((gx + lx - 1) // lx) * lx,
            ((gy + ly - 1) // ly) * ly)
        lsize = (ctypes.c_size_t * 2)(lx, ly)
        self._check(
            _ocl.clEnqueueNDRangeKernel(
                self._queue, kernel, ctypes.c_uint32(2), None,
                gsize, lsize, ctypes.c_uint32(0), None, None),
            "clEnqueueNDRangeKernel(2D)")

    # ── High-level ops ────────────────────────────────────────────────────────

    def _unary(self, kname: str, src_ptr, dst_ptr, n: int):
        k = self._get_kernel("unary", _UNARY_KERNEL_SRC, kname)
        cl_src = self.alloc(n, CL_MEM_READ_ONLY)
        cl_dst = self.alloc(n, CL_MEM_WRITE_ONLY)
        self.upload(cl_src, src_ptr, n)
        self._set_arg(k, 0, cl_src)
        self._set_arg(k, 1, cl_dst)
        self._set_arg(k, 2, n)
        self._launch_1d(k, n)
        self.download(cl_dst, dst_ptr, n)
        self.release_buf(cl_src)
        self.release_buf(cl_dst)

    def exp_f32(self, src_ptr, dst_ptr, n: int):   self._unary("vp_exp",    src_ptr, dst_ptr, n)
    def tanh_f32(self, src_ptr, dst_ptr, n: int):  self._unary("vp_tanh",   src_ptr, dst_ptr, n)
    def sigmoid_f32(self, src_ptr, dst_ptr, n: int): self._unary("vp_sigmoid", src_ptr, dst_ptr, n)
    def relu_f32(self, src_ptr, dst_ptr, n: int):  self._unary("vp_relu",   src_ptr, dst_ptr, n)
    def silu_f32(self, src_ptr, dst_ptr, n: int):  self._unary("vp_silu",   src_ptr, dst_ptr, n)
    def gelu_f32(self, src_ptr, dst_ptr, n: int):  self._unary("vp_gelu",   src_ptr, dst_ptr, n)
    def log_f32(self, src_ptr, dst_ptr, n: int):   self._unary("vp_log",    src_ptr, dst_ptr, n)
    def sqrt_f32(self, src_ptr, dst_ptr, n: int):  self._unary("vp_sqrt",   src_ptr, dst_ptr, n)
    def abs_f32(self, src_ptr, dst_ptr, n: int):   self._unary("vp_abs_f",  src_ptr, dst_ptr, n)
    def neg_f32(self, src_ptr, dst_ptr, n: int):   self._unary("vp_neg_f",  src_ptr, dst_ptr, n)

    def fused_chain_f32(self, src_ptr, dst_ptr, n: int, op_codes: list):
        k = self._get_kernel("chain", _CHAIN_KERNEL_SRC, "vp_fused_chain")
        cl_src = self.alloc(n, CL_MEM_READ_ONLY)
        cl_dst = self.alloc(n, CL_MEM_WRITE_ONLY)
        self.upload(cl_src, src_ptr, n)
        self._set_arg(k, 0, cl_src)
        self._set_arg(k, 1, cl_dst)
        self._set_arg(k, 2, n)
        ops8 = (list(op_codes) + [-1] * 8)[:8]
        for idx, op in enumerate(ops8):
            self._set_arg(k, 3 + idx, op)
        self._set_arg(k, 11, len(op_codes))
        self._launch_1d(k, n)
        self.download(cl_dst, dst_ptr, n)
        self.release_buf(cl_src)
        self.release_buf(cl_dst)

    def matmul_f32(self, A_ptr, B_ptr, C_ptr, M: int, K: int, N: int):
        k = self._get_kernel("matmul", _MATMUL_KERNEL_SRC, "vp_matmul")
        cl_A = self.alloc(M * K, CL_MEM_READ_ONLY)
        cl_B = self.alloc(K * N, CL_MEM_READ_ONLY)
        cl_C = self.alloc(M * N, CL_MEM_WRITE_ONLY)
        self.upload(cl_A, A_ptr, M * K)
        self.upload(cl_B, B_ptr, K * N)
        self._set_arg(k, 0, cl_A)
        self._set_arg(k, 1, cl_B)
        self._set_arg(k, 2, cl_C)
        self._set_arg(k, 3, M)
        self._set_arg(k, 4, K)
        self._set_arg(k, 5, N)
        self._launch_2d(k, N, M)
        self.download(cl_C, C_ptr, M * N)
        self.release_buf(cl_A)
        self.release_buf(cl_B)
        self.release_buf(cl_C)

    def softmax_f32(self, src_ptr, dst_ptr, rows: int, cols: int):
        k = self._get_kernel("softmax", _SOFTMAX_KERNEL_SRC, "vp_softmax")
        cl_src = self.alloc(rows * cols, CL_MEM_READ_ONLY)
        cl_dst = self.alloc(rows * cols, CL_MEM_WRITE_ONLY)
        self.upload(cl_src, src_ptr, rows * cols)
        self._set_arg(k, 0, cl_src)
        self._set_arg(k, 1, cl_dst)
        self._set_arg(k, 2, rows)
        self._set_arg(k, 3, cols)
        self._launch_1d(k, rows, min(256, rows))
        self.download(cl_dst, dst_ptr, rows * cols)
        self.release_buf(cl_src)
        self.release_buf(cl_dst)

    def layer_norm_f32(self, x_ptr, w_ptr, b_ptr, dst_ptr,
                        N: int, D: int, eps: float = 1e-5):
        k = self._get_kernel("layer_norm", _LAYER_NORM_SRC, "vp_layer_norm")
        cl_x   = self.alloc(N * D, CL_MEM_READ_ONLY)
        cl_w   = self.alloc(D,     CL_MEM_READ_ONLY)
        cl_b   = self.alloc(D,     CL_MEM_READ_ONLY)
        cl_dst = self.alloc(N * D, CL_MEM_WRITE_ONLY)
        self.upload(cl_x,   x_ptr,   N * D)
        self.upload(cl_w,   w_ptr,   D)
        self.upload(cl_b,   b_ptr,   D)
        self._set_arg(k, 0, cl_x)
        self._set_arg(k, 1, cl_w)
        self._set_arg(k, 2, cl_b)
        self._set_arg(k, 3, cl_dst)
        self._set_arg(k, 4, N)
        self._set_arg(k, 5, D)
        self._set_arg(k, 6, ctypes.c_float(eps))
        self._launch_1d(k, N)
        self.download(cl_dst, dst_ptr, N * D)
        self.release_buf(cl_x)
        self.release_buf(cl_w)
        self.release_buf(cl_b)
        self.release_buf(cl_dst)

    def adam_step_f32(self, p_ptr, g_ptr, m_ptr, v_ptr, n: int,
                       lr: float, beta1: float, beta2: float,
                       eps: float, wd: float, t: int):
        """Fused AdamW step entirely on GPU — no host round-trip needed."""
        import math as _math
        bc1 = 1.0 - beta1 ** t;  bc2 = 1.0 - beta2 ** t
        inv_bc1 = 1.0 / bc1;     inv_bc2 = 1.0 / bc2
        k = self._get_kernel("adam", _ADAM_KERNEL_SRC, "vp_adam_step")
        cl_p = self.alloc(n, CL_MEM_READ_WRITE); self.upload(cl_p, p_ptr, n)
        cl_g = self.alloc(n, CL_MEM_READ_ONLY);  self.upload(cl_g, g_ptr, n)
        cl_m = self.alloc(n, CL_MEM_READ_WRITE); self.upload(cl_m, m_ptr, n)
        cl_v = self.alloc(n, CL_MEM_READ_WRITE); self.upload(cl_v, v_ptr, n)
        self._set_arg(k, 0, cl_p); self._set_arg(k, 1, cl_g)
        self._set_arg(k, 2, cl_m); self._set_arg(k, 3, cl_v)
        self._set_arg(k, 4, ctypes.c_float(lr));      self._set_arg(k, 5, ctypes.c_float(beta1))
        self._set_arg(k, 6, ctypes.c_float(beta2));   self._set_arg(k, 7, ctypes.c_float(eps))
        self._set_arg(k, 8, ctypes.c_float(wd));      self._set_arg(k, 9, ctypes.c_float(inv_bc1))
        self._set_arg(k, 10, ctypes.c_float(inv_bc2)); self._set_arg(k, 11, n)
        self._launch_1d(k, n)
        self.download(cl_p, p_ptr, n); self.download(cl_m, m_ptr, n); self.download(cl_v, v_ptr, n)
        for buf in (cl_p, cl_g, cl_m, cl_v): self.release_buf(buf)

    def flash_attn_f32(self, Q_ptr, K_ptr, V_ptr, O_ptr,
                        BH: int, Sq: int, Sk: int, D: int):
        """Flash Attention: O = softmax(QK^T/sqrt(D)) * V, online tiled algorithm."""
        import math
        scale = 1.0 / math.sqrt(D)
        k = self._get_kernel("flash_attn", _FLASH_ATTN_SRC, "vp_flash_attn")
        n = BH * Sq * D; nk = BH * Sk * D
        cl_Q = self.alloc(n,  CL_MEM_READ_ONLY);  self.upload(cl_Q, Q_ptr, n)
        cl_K = self.alloc(nk, CL_MEM_READ_ONLY);  self.upload(cl_K, K_ptr, nk)
        cl_V = self.alloc(nk, CL_MEM_READ_ONLY);  self.upload(cl_V, V_ptr, nk)
        cl_O = self.alloc(n,  CL_MEM_WRITE_ONLY)
        self._set_arg(k, 0, cl_Q); self._set_arg(k, 1, cl_K)
        self._set_arg(k, 2, cl_V); self._set_arg(k, 3, cl_O)
        self._set_arg(k, 4, Sq);   self._set_arg(k, 5, Sk)
        self._set_arg(k, 6, D);    self._set_arg(k, 7, ctypes.c_float(scale))
        # 2D launch: dim0=Sq (query rows), dim1=BH (batch*heads)
        lx = min(64, Sq); ly = 1
        self._launch_2d(k, Sq, BH, lx, ly)
        self.download(cl_O, O_ptr, n)
        for buf in (cl_Q, cl_K, cl_V, cl_O):
            self.release_buf(buf)

    def device_info(self) -> dict:
        return dict(self._info)


# ── Module-level singleton + public API ──────────────────────────────────────

_device: Optional[_VPOpenCLDevice] = None

def _get_device() -> _VPOpenCLDevice:
    global _device
    if _device is None:
        _device = _VPOpenCLDevice()
    return _device

def is_available() -> bool:
    """Return True if an OpenCL GPU device was found and initialised."""
    try:
        return _get_device()._ready
    except Exception:
        return False

def device_info() -> dict:
    """Return device capabilities dict."""
    if not is_available():
        return {"name": "CPU (no OpenCL GPU)"}
    return _get_device().device_info()


def _arr_ptr(arr: _array.array):
    """Return a ctypes void* to an array.array's internal buffer."""
    addr, _ = arr.buffer_info()
    return ctypes.cast(addr, ctypes.c_void_p)


# Element-wise ops — accept array.array('f', ...) and write result into dst.
def exp_f32(src: _array.array, dst: _array.array) -> None:
    _get_device().exp_f32(_arr_ptr(src), _arr_ptr(dst), len(src))

def tanh_f32(src: _array.array, dst: _array.array) -> None:
    _get_device().tanh_f32(_arr_ptr(src), _arr_ptr(dst), len(src))

def sigmoid_f32(src: _array.array, dst: _array.array) -> None:
    _get_device().sigmoid_f32(_arr_ptr(src), _arr_ptr(dst), len(src))

def relu_f32(src: _array.array, dst: _array.array) -> None:
    _get_device().relu_f32(_arr_ptr(src), _arr_ptr(dst), len(src))

def silu_f32(src: _array.array, dst: _array.array) -> None:
    _get_device().silu_f32(_arr_ptr(src), _arr_ptr(dst), len(src))

def gelu_f32(src: _array.array, dst: _array.array) -> None:
    _get_device().gelu_f32(_arr_ptr(src), _arr_ptr(dst), len(src))

def log_f32(src: _array.array, dst: _array.array) -> None:
    _get_device().log_f32(_arr_ptr(src), _arr_ptr(dst), len(src))

def sqrt_f32(src: _array.array, dst: _array.array) -> None:
    _get_device().sqrt_f32(_arr_ptr(src), _arr_ptr(dst), len(src))

def fused_chain_f32(src: _array.array, dst: _array.array, op_codes: list) -> None:
    _get_device().fused_chain_f32(_arr_ptr(src), _arr_ptr(dst), len(src), op_codes)

def matmul_f32(A: _array.array, B: _array.array, C: _array.array,
               M: int, K: int, N: int) -> None:
    _get_device().matmul_f32(
        _arr_ptr(A), _arr_ptr(B), _arr_ptr(C), M, K, N)

def softmax_f32(src: _array.array, dst: _array.array,
                rows: int, cols: int) -> None:
    _get_device().softmax_f32(_arr_ptr(src), _arr_ptr(dst), rows, cols)

def layer_norm_f32(x: _array.array, w: _array.array, b: _array.array,
                   dst: _array.array, N: int, D: int, eps: float = 1e-5) -> None:
    _get_device().layer_norm_f32(
        _arr_ptr(x), _arr_ptr(w), _arr_ptr(b), _arr_ptr(dst), N, D, eps)

def flash_attn_f32(Q: _array.array, K: _array.array, V: _array.array,
                   O: _array.array, BH: int, Sq: int, Sk: int, D: int) -> None:
    """Flash Attention forward: O[BH,Sq,D] = softmax(Q[BH,Sq,D]@K^T/sqrt(D))@V[BH,Sk,D]"""
    _get_device().flash_attn_f32(
        _arr_ptr(Q), _arr_ptr(K), _arr_ptr(V), _arr_ptr(O), BH, Sq, Sk, D)

def adam_step_f32(p: _array.array, g: _array.array,
                  m: _array.array, v: _array.array,
                  lr: float, beta1: float, beta2: float,
                  eps: float, wd: float, t: int) -> None:
    """Fused AdamW step on GPU. p, m, v updated in place."""
    _get_device().adam_step_f32(
        _arr_ptr(p), _arr_ptr(g), _arr_ptr(m), _arr_ptr(v),
        len(p), lr, beta1, beta2, eps, wd, t)


class GPUTensor:
    """
    Persistent GPU buffer — upload once, run many kernels, download when needed.

    Example
    -------
    gQ = GPUTensor.from_array(Q, read_only=True)
    gK = GPUTensor.from_array(K, read_only=True)
    gV = GPUTensor.from_array(V, read_only=True)
    gO = GPUTensor.zeros(n)
    GPUTensor.flash_attn(gQ, gK, gV, gO, BH, Sq, Sk, D)
    result = gO.to_array()
    """
    def __init__(self, cl_buf, n: int, read_only: bool = False):
        self._buf = cl_buf    # ctypes.c_void_p OpenCL cl_mem handle
        self.n    = n         # number of float32 elements
        self._ro  = read_only

    def __del__(self):
        try:
            _get_device().release_buf(self._buf)
        except Exception:
            pass

    @classmethod
    def from_array(cls, arr: _array.array, read_only: bool = False) -> "GPUTensor":
        dev = _get_device()
        flag = CL_MEM_READ_ONLY if read_only else CL_MEM_READ_WRITE
        buf  = dev.alloc(len(arr), flag)
        dev.upload(buf, _arr_ptr(arr), len(arr))
        return cls(buf, len(arr), read_only)

    @classmethod
    def zeros(cls, n: int) -> "GPUTensor":
        dev = _get_device()
        buf = dev.alloc(n, CL_MEM_READ_WRITE)
        return cls(buf, n, False)

    def to_array(self) -> _array.array:
        dev = _get_device()
        out = _array.array('f', bytes(self.n * 4))
        dev.download(self._buf, _arr_ptr(out), self.n)
        return out

    def upload(self, arr: _array.array) -> None:
        _get_device().upload(self._buf, _arr_ptr(arr), self.n)

    @staticmethod
    def flash_attn(gQ: "GPUTensor", gK: "GPUTensor", gV: "GPUTensor",
                   gO: "GPUTensor", BH: int, Sq: int, Sk: int, D: int) -> None:
        """Run Flash Attention with pre-uploaded GPU tensors — no PCIe transfers."""
        import math as _math
        dev = _get_device()
        scale = 1.0 / _math.sqrt(D)
        k = dev._get_kernel("flash_attn", _FLASH_ATTN_SRC, "vp_flash_attn")
        dev._set_arg(k, 0, gQ._buf); dev._set_arg(k, 1, gK._buf)
        dev._set_arg(k, 2, gV._buf); dev._set_arg(k, 3, gO._buf)
        dev._set_arg(k, 4, Sq);      dev._set_arg(k, 5, Sk)
        dev._set_arg(k, 6, D);       dev._set_arg(k, 7, ctypes.c_float(scale))
        dev._launch_2d(k, Sq, BH, min(64, Sq), 1)

    @staticmethod
    def matmul(gA: "GPUTensor", gB: "GPUTensor", gC: "GPUTensor",
               M: int, K: int, N: int) -> None:
        """Run GPU matmul with pre-uploaded buffers."""
        from vectorphile._core.vp_opencl import _MATMUL_KERNEL_SRC
        dev = _get_device()
        k = dev._get_kernel("matmul", _MATMUL_KERNEL_SRC, "vp_matmul")
        dev._set_arg(k, 0, gA._buf); dev._set_arg(k, 1, gB._buf)
        dev._set_arg(k, 2, gC._buf)
        dev._set_arg(k, 3, M); dev._set_arg(k, 4, K); dev._set_arg(k, 5, N)
        dev._launch_2d(k, N, M, 16, 16)
