"""
Build vectorphile C kernels into a loadable shared library using cffi.

Usage:
    python build_c_kernels.py

Produces vpkernels.dll (Windows) / libvpkernels.so (Linux/Mac).
No dependencies beyond cffi (which is already installed).
"""
from __future__ import annotations
import sys
import os
import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).parent.resolve()
C_SRC = HERE / "vpkernels.c"


CDEF = """
/* GEMM */
void vp_sgemm(float* C, const float* A, const float* B, int M, int K, int N);
void vp_gemm_f32(float* C, const float* A, const float* B, int M, int K, int N);
void vp_sgemm_transB(float* C, const float* A, const float* B, int M, int K, int N);
void vp_sgemm_batched(float* C, const float* A, const float* B, int batch, int M, int K, int N);
void vp_nd_matmul(float* C, const float* A, const float* B, int M, int K, int N);

/* Elementwise unary */
void vp_exp(float* dst, const float* src, int n);
void vp_exp_f32(float* dst, const float* src, int n);
void vp_log(float* dst, const float* src, int n);
void vp_log_f32(float* dst, const float* src, int n);
void vp_sqrt(float* dst, const float* src, int n);
void vp_sqrt_f32(float* dst, const float* src, int n);
void vp_abs(float* dst, const float* src, int n);
void vp_neg(float* dst, const float* src, int n);
void vp_relu(float* dst, const float* src, int n);
void vp_relu_f32(float* dst, const float* src, int n);
void vp_sigmoid(float* dst, const float* src, int n);
void vp_sigmoid_f32(float* dst, const float* src, int n);
void vp_tanh(float* dst, const float* src, int n);
void vp_tanh_f32(float* dst, const float* src, int n);
void vp_gelu(float* dst, const float* src, int n);
void vp_gelu_f32(float* dst, const float* src, int n);
void vp_silu(float* dst, const float* src, int n);
void vp_silu_f32(float* dst, const float* src, int n);
void vp_log1p(float* dst, const float* src, int n);
void vp_expm1(float* dst, const float* src, int n);
void vp_erf(float* dst, const float* src, int n);

/* Elementwise binary */
void vp_add(float* dst, const float* a, const float* b, int n);
void vp_add_f32(float* dst, const float* a, const float* b, int n);
void vp_sub(float* dst, const float* a, const float* b, int n);
void vp_sub_f32(float* dst, const float* a, const float* b, int n);
void vp_mul(float* dst, const float* a, const float* b, int n);
void vp_mul_f32(float* dst, const float* a, const float* b, int n);
void vp_div(float* dst, const float* a, const float* b, int n);
void vp_div_f32(float* dst, const float* a, const float* b, int n);
void vp_add_scalar(float* dst, const float* a, float scalar, int n);
void vp_scalar_add_f32(float* src, float scalar, float* dst, int n);
void vp_mul_scalar(float* dst, const float* a, float scalar, int n);
void vp_scalar_mul_f32(float* src, float scalar, float* dst, int n);
void vp_broadcast_binary(float* dst, const float* a, const float* b,
                         const int* out_shape, const int* a_strides,
                         const int* b_strides, int ndim, int op);
void vp_relu_backward(float* dst, const float* grad, const float* x, int n);
void vp_sigmoid_backward(float* dst, const float* grad, const float* sig, int n);
void vp_tanh_backward(float* dst, const float* grad, const float* tanh_out, int n);
void vp_log_backward(float* dst, const float* grad, const float* inp, int n, float eps);
void vp_sqrt_backward(float* dst, const float* grad, const float* sqrt_out, int n, float eps);
void vp_exp_backward(float* dst, const float* grad, const float* exp_out, int n);
void vp_clamp(float* dst, const float* src, float lo, float hi, int n);
void vp_pow(float* dst, const float* base, float exp_val, int n);

/* Reductions */
float vp_sum(const float* a, int n);
void vp_sum_f32(const float* a, int n, float* out);
float vp_mean(const float* a, int n);
float vp_max(const float* a, int n);
void vp_max_f32(const float* a, int n, float* out);
float vp_min(const float* a, int n);
void vp_min_f32(const float* a, int n, float* out);
float vp_norm_l2(const float* a, int n);
float vp_dot(const float* a, const float* b, int n);
int vp_argmax(const float* a, int n);
void vp_argmax_f32(const float* a, int n, int* out);
int vp_argmin(const float* a, int n);
void vp_sum_axis(const float* a, float* out, int outer, int axis_len, int inner);
void vp_max_axis(const float* a, float* out, int outer, int axis_len, int inner);
void vp_min_axis(const float* a, float* out, int outer, int axis_len, int inner);
void vp_softmax_inplace(float* x, int n);
void vp_log_softmax_inplace(float* x, int n);

/* NDArray buffer ops */
float* vp_nd_alloc(int64_t n, int dtype_size);
void vp_nd_free(float* ptr);
void vp_nd_zeros(float* ptr, int64_t n);
void vp_nd_fill(float* ptr, float val, int64_t n);
void vp_nd_copy(float* dst, const float* src, int64_t n);
void vp_nd_set(float* ptr, int64_t idx, float val);
float vp_nd_get(const float* ptr, int64_t idx);
float* vp_nd_add(const float* a, const float* b, int64_t n);
float* vp_nd_sub(const float* a, const float* b, int64_t n);
float* vp_nd_mul(const float* a, const float* b, int64_t n);
float* vp_nd_div(const float* a, const float* b, int64_t n);
float* vp_nd_add_scalar(const float* a, float s, int64_t n);
float* vp_nd_mul_scalar(const float* a, float s, int64_t n);
float vp_nd_sum(const float* a, int64_t n);
float vp_nd_max(const float* a, int64_t n);
float vp_nd_min(const float* a, int64_t n);
float vp_nd_dot(const float* a, const float* b, int64_t n);
int64_t vp_nd_argmax(const float* a, int64_t n);
int64_t vp_nd_argmin(const float* a, int64_t n);
float* vp_nd_exp(const float* a, int64_t n);
float* vp_nd_log(const float* a, int64_t n);
float* vp_nd_sqrt(const float* a, int64_t n);
float* vp_nd_abs(const float* a, int64_t n);
float* vp_nd_neg(const float* a, int64_t n);
float* vp_nd_relu(const float* a, int64_t n);
float* vp_nd_sigmoid(const float* a, int64_t n);
float* vp_nd_tanh(const float* a, int64_t n);
float* vp_nd_gelu(const float* a, int64_t n);
float* vp_nd_silu(const float* a, int64_t n);
void vp_nd_softmax_inplace(float* x, int64_t n);

/* Random */
void vp_rand_uniform(float* out, int n, uint64_t seed);
void vp_rand_normal(float* out, int n, uint64_t seed, float mean, float std);
void vp_rand_int(int* out, int n, uint64_t seed, int lo, int hi);
void vp_rand_bernoulli(float* out, int n, uint64_t seed, float p);

/* Conv / Pool */
void vp_im2col(float* col, const float* img, int N, int C, int H, int W,
               int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int a, int b);
void vp_col2im(float* img, const float* col, int N, int C, int H, int W,
               int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW, int a, int b);
void vp_max_pool2d(float* out, int* mask, const float* img, int N, int C, int H, int W,
                   int kH, int kW, int sH, int sW, int pH, int pW, int a, int b, int c, int d);
void vp_avg_pool2d(float* out, const float* img, int N, int C, int H, int W,
                   int kH, int kW, int sH, int sW, int pH, int pW, int a, int b, int c, int d);

/* Fused */
void vp_fused_linear_relu(float* out, const float* x, const float* W, const float* b,
                          int N, int in_f, int out_f);
void vp_fused_linear_gelu(float* out, const float* x, const float* W, const float* b,
                          int N, int in_f, int out_f);
void vp_fused_linear_silu(float* out, const float* x, const float* W, const float* b,
                          int N, int in_f, int out_f);
void vp_fused_scale_add(float* out, const float* a, const float* b, float alpha, int n);
void vp_fused_rmsnorm(float* out, const float* x, const float* weight, int n, float eps);
void vp_fused_layernorm(float* out, const float* x, const float* weight, const float* bias,
                        int n, float eps);
void vp_fused_rope(float* x, const float* cos_c, const float* sin_c,
                   int B, int n_heads, int seq_len, int head_dim);
void vp_fused_attention_scores(float* out, const float* Q, const float* K,
                               float scale, int B, int H, int S, int D);
void vp_fused_dropout_add(float* out, const float* x, const float* residual,
                          float prob, int n, uint64_t seed);
void vp_fused_adam_step(float* param, const float* grad, float* m, float* v,
                        float lr, float b1, float b2, float eps, int t, int n);
void vp_adam_step(float* param, const float* grad, float* m, float* v,
                  int t, float lr, float b1, float b2, float eps, int n);
"""


def build():
    try:
        from cffi import FFI
    except ImportError:
        print("cffi not found. Install with: pip install cffi")
        return False

    ffi = FFI()
    ffi.cdef(CDEF)

    c_code = C_SRC.read_text(encoding="utf-8")

    # Extra compile args for fast math + AVX2
    if sys.platform == "win32":
        extra_compile_args = ["/O2", "/arch:AVX2", "/fp:fast", "/openmp"]
        extra_link_args = []
    else:
        extra_compile_args = ["-O3", "-march=native", "-ffast-math", "-funroll-loops"]
        extra_link_args = ["-lm"]

    ffi.set_source(
        "_vp_kernels",
        c_code,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    tmpdir = tempfile.mkdtemp(prefix="vp_build_")
    try:
        print(f"Building C kernels in {tmpdir}...")
        pyd_path = ffi.compile(tmpdir=tmpdir, verbose=True)
        print(f"Built: {pyd_path}")

        # Copy the .pyd/.so to the _mojo directory where the loader looks
        dest = HERE / Path(pyd_path).name
        shutil.copy2(pyd_path, dest)
        print(f"Installed to: {dest}")

        # Also write a small .pth so Python can find it
        _write_cffi_loader(HERE, ffi)
        return True
    except Exception as e:
        print(f"Build failed: {e}")
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _write_cffi_loader(dest_dir: Path, ffi):
    """Write _vp_cffi_loader.py so the main loader can import kernels via cffi."""
    loader_code = '''"""Auto-generated cffi loader for vectorphile C kernels."""
from __future__ import annotations
import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
try:
    import _vp_kernels
    _ffi = _vp_kernels.ffi
    _lib = _vp_kernels.lib

    def available(): return True

    def get(name: str):
        try:
            return getattr(_lib, name)
        except AttributeError:
            return None

    def ptr(arr):
        """Return cffi float* pointing at array.array buffer."""
        import ctypes
        addr, _ = arr.buffer_info()
        return _ffi.cast("float*", addr)

    def iptr(arr):
        """Return cffi int* pointing at array.array buffer."""
        import ctypes
        addr, _ = arr.buffer_info()
        return _ffi.cast("int*", addr)

    def i64ptr(arr):
        """Return cffi int64_t* pointing at array.array buffer."""
        import ctypes
        addr, _ = arr.buffer_info()
        return _ffi.cast("int64_t*", addr)

except ImportError:
    _ffi = None; _lib = None
    def available(): return False
    def get(name): return None
    def ptr(arr): return None
    def iptr(arr): return None
    def i64ptr(arr): return None
'''
    (dest_dir / "_vp_cffi_loader.py").write_text(loader_code)
    print(f"Wrote cffi loader: {dest_dir / '_vp_cffi_loader.py'}")


if __name__ == "__main__":
    success = build()
    sys.exit(0 if success else 1)
