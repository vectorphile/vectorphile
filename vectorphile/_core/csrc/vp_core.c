/*
 * vp_core.c — vectorphile core C extension
 *
 * Implements AVX2/FMA-vectorized array operations on Python array.array buffers.
 * Zero third-party dependencies: uses only the Python C API and libc.
 *
 * Ops: relu, exp, log, sigmoid, tanh, sqrt, abs, neg, sign, erf,
 *      add, sub, mul, div (element-wise and scalar),
 *      sum, max, min, argmax, argmin (reductions),
 *      matmul (tiled micro-kernel, FMA),
 *      softmax, log_softmax.
 *
 * Compile flags:
 *   MSVC  : /O2 /arch:AVX2
 *   GCC   : -O3 -mavx2 -mfma -ffast-math
 *   Clang : -O3 -mavx2 -mfma -ffast-math
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>

/* ── Early restrict/inline macros (needed by thread pool before AVX2 section) ── */
#ifdef _MSC_VER
#  ifndef VP_RESTRICT
#    define VP_RESTRICT __restrict
#  endif
#  ifndef VP_INLINE
#    define VP_INLINE __forceinline
#  endif
#else
#  ifndef VP_RESTRICT
#    define VP_RESTRICT __restrict__
#  endif
#  ifndef VP_INLINE
#    define VP_INLINE static inline __attribute__((always_inline))
#  endif
#endif

/* ── Persistent thread pool (Windows) ───────────────────────────────────────
 * Keeps worker threads spinning so fork/join is ~1-2μs instead of VCOMP's ~60μs.
 * Used for lightweight ops (relu, abs, neg, sqrt) where VCOMP overhead dominates.
 * Automatically sized to physical P-cores (max 8 on i5-13450HX = 6P+4E).
 */
#ifdef _WIN32
#  define VP_HAVE_TPOOL 1
#  include <windows.h>

#  define VP_TPOOL_MAX  8
#  define VP_TPOOL_SPIN 50000  /* spin cycles before sleeping */

typedef void (*vp_tpool_fn)(const float*, float*, Py_ssize_t, Py_ssize_t);

typedef struct {
    volatile LONG      ready;    /* 0=idle, 1=work_pending, 2=done */
    volatile LONG      running;  /* worker alive flag */
    const float*       src;
    float*             dst;
    Py_ssize_t         start;
    Py_ssize_t         end;
    vp_tpool_fn        fn;
} vp_worker_t;

static vp_worker_t g_workers[VP_TPOOL_MAX];
static HANDLE      g_threads[VP_TPOOL_MAX];
static int         g_nthreads = 0;

static DWORD WINAPI vp_worker_thread(LPVOID arg) {
    vp_worker_t* w = (vp_worker_t*)arg;
    while (w->running) {
        /* Spin-wait for work */
        int spin = 0;
        while (InterlockedCompareExchange(&w->ready, 1, 1) != 1) {
            if (!w->running) return 0;
            if (++spin > VP_TPOOL_SPIN) { Sleep(0); spin = 0; }
        }
        /* Execute work slice */
        w->fn(w->src, w->dst, w->start, w->end);
        InterlockedExchange(&w->ready, 2);  /* signal done */
    }
    return 0;
}

static void vp_tpool_init(void) {
    if (g_nthreads > 0) return;
    SYSTEM_INFO si; GetSystemInfo(&si);
    int ncores = (int)si.dwNumberOfProcessors;
    if (ncores > VP_TPOOL_MAX) ncores = VP_TPOOL_MAX;
    g_nthreads = ncores > 1 ? ncores : 1;
    for (int i = 0; i < g_nthreads; i++) {
        g_workers[i].ready   = 0;
        g_workers[i].running = 1;
        g_threads[i] = CreateThread(NULL, 0, vp_worker_thread,
                                     &g_workers[i], 0, NULL);
        SetThreadPriority(g_threads[i], THREAD_PRIORITY_ABOVE_NORMAL);
    }
}

/* Dispatch a parallel op across the pool.
 * fn(src, dst, start, end) processes elements [start, end). */
static void vp_tpool_run(vp_tpool_fn fn,
                          const float* VP_RESTRICT src,
                          float*       VP_RESTRICT dst,
                          Py_ssize_t n) {
    if (g_nthreads <= 1 || n < 262144) {  /* < 256K: single thread */
        fn(src, dst, 0, n);
        return;
    }
    int nt = g_nthreads;
    Py_ssize_t chunk = (n + nt - 1) / nt;
    /* Submit work to each worker */
    for (int t = 0; t < nt; t++) {
        vp_worker_t* w = &g_workers[t];
        w->src   = src;
        w->dst   = dst;
        w->fn    = fn;
        w->start = (Py_ssize_t)t * chunk;
        w->end   = w->start + chunk < n ? w->start + chunk : n;
        InterlockedExchange(&w->ready, 1);  /* post work */
    }
    /* Spin-wait for all to finish */
    for (int t = 0; t < nt; t++) {
        while (InterlockedCompareExchange(&g_workers[t].ready, 2, 2) != 2)
            _mm_pause();
        InterlockedExchange(&g_workers[t].ready, 0);
    }
}

/* Per-thread relu slice — called by both tpool and single-thread paths */
static void vp_relu_slice(const float* VP_RESTRICT src,
                            float*       VP_RESTRICT dst,
                            Py_ssize_t start, Py_ssize_t end) {
    const __m256 zero = _mm256_setzero_ps();
    Py_ssize_t i = start;
    for (; i <= end - 32; i += 32) {
        _mm_prefetch((const char*)(src + i + 256), _MM_HINT_T0);
        _mm256_storeu_ps(dst + i,      _mm256_max_ps(_mm256_loadu_ps(src + i),      zero));
        _mm256_storeu_ps(dst + i +  8, _mm256_max_ps(_mm256_loadu_ps(src + i +  8), zero));
        _mm256_storeu_ps(dst + i + 16, _mm256_max_ps(_mm256_loadu_ps(src + i + 16), zero));
        _mm256_storeu_ps(dst + i + 24, _mm256_max_ps(_mm256_loadu_ps(src + i + 24), zero));
    }
    for (; i < end; i++) dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
}

static void vp_abs_slice(const float* VP_RESTRICT src,
                           float*       VP_RESTRICT dst,
                           Py_ssize_t start, Py_ssize_t end) {
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    Py_ssize_t i = start;
    for (; i <= end - 32; i += 32) {
        _mm_prefetch((const char*)(src + i + 256), _MM_HINT_T0);
        _mm256_storeu_ps(dst + i,      _mm256_and_ps(_mm256_loadu_ps(src + i),      sign_mask));
        _mm256_storeu_ps(dst + i +  8, _mm256_and_ps(_mm256_loadu_ps(src + i +  8), sign_mask));
        _mm256_storeu_ps(dst + i + 16, _mm256_and_ps(_mm256_loadu_ps(src + i + 16), sign_mask));
        _mm256_storeu_ps(dst + i + 24, _mm256_and_ps(_mm256_loadu_ps(src + i + 24), sign_mask));
    }
    for (; i < end; i++) dst[i] = fabsf(src[i]);
}

static void vp_neg_slice(const float* VP_RESTRICT src,
                           float*       VP_RESTRICT dst,
                           Py_ssize_t start, Py_ssize_t end) {
    const __m256 zero = _mm256_setzero_ps();
    Py_ssize_t i = start;
    for (; i <= end - 32; i += 32) {
        _mm_prefetch((const char*)(src + i + 256), _MM_HINT_T0);
        _mm256_storeu_ps(dst + i,      _mm256_sub_ps(zero, _mm256_loadu_ps(src + i)));
        _mm256_storeu_ps(dst + i +  8, _mm256_sub_ps(zero, _mm256_loadu_ps(src + i +  8)));
        _mm256_storeu_ps(dst + i + 16, _mm256_sub_ps(zero, _mm256_loadu_ps(src + i + 16)));
        _mm256_storeu_ps(dst + i + 24, _mm256_sub_ps(zero, _mm256_loadu_ps(src + i + 24)));
    }
    for (; i < end; i++) dst[i] = -src[i];
}

/* AVX2 sqrt via rsqrt+NR: ~11 cycles vs _mm256_sqrt_ps ~25 cycles.
 * sqrt(x) = x * rsqrt(x),  NR: r' = r*(1.5 - 0.5*x*r^2)  then  x*r' */
static VP_INLINE __m256 vp_avx2_sqrt(__m256 x) {
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 c1p5 = _mm256_set1_ps(1.5f);
    __m256 r  = _mm256_rsqrt_ps(x);                        /* 11-bit approx */
    __m256 xh = _mm256_mul_ps(half, x);
    __m256 r2 = _mm256_mul_ps(r, r);
    r = _mm256_mul_ps(r, _mm256_fnmadd_ps(xh, r2, c1p5)); /* NR → 23-bit */
    return _mm256_mul_ps(x, r);                             /* sqrt = x/rsqrt */
}
static void vp_sqrt_slice(const float* VP_RESTRICT src,
                            float*       VP_RESTRICT dst,
                            Py_ssize_t start, Py_ssize_t end) {
    Py_ssize_t i = start;
    for (; i <= end - 32; i += 32) {
        _mm_prefetch((const char*)(src + i + 256), _MM_HINT_T0);
        _mm256_storeu_ps(dst + i,      vp_avx2_sqrt(_mm256_loadu_ps(src + i)));
        _mm256_storeu_ps(dst + i +  8, vp_avx2_sqrt(_mm256_loadu_ps(src + i +  8)));
        _mm256_storeu_ps(dst + i + 16, vp_avx2_sqrt(_mm256_loadu_ps(src + i + 16)));
        _mm256_storeu_ps(dst + i + 24, vp_avx2_sqrt(_mm256_loadu_ps(src + i + 24)));
    }
    for (; i < end; i++) dst[i] = sqrtf(src[i]);
}

#endif /* _WIN32 */

/* ── All ops use vectorphile's own AVX2 kernels — no third-party BLAS/VML ── */

/* CBLAS constants (same values on all platforms) */
#define CBLAS_ROW_MAJOR 101
#define CBLAS_NO_TRANS  111

/* ── AVX-512F detection ──────────────────────────────────────────────────────── */
#if defined(__AVX512F__)
#  include <immintrin.h>
#  define VP_AVX512F 1
#else
#  define VP_AVX512F 0
#endif

/* ── AVX2 detection ─────────────────────────────────────────────────────────── */
#if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
#  ifndef _IMMINTRIN_H_INCLUDED
#  include <immintrin.h>
#  define _IMMINTRIN_H_INCLUDED
#  endif
#  define VP_AVX2 1
#else
#  define VP_AVX2 0
#endif

/* ── MSVC compatibility ──────────────────────────────────────────────────────── */
#ifdef _MSC_VER
#  define VP_RESTRICT __restrict
#  define VP_INLINE   __forceinline
#else
#  define VP_RESTRICT __restrict__
#  define VP_INLINE   static inline __attribute__((always_inline))
#endif

/* ── Portable aligned local variable ────────────────────────────────────────── */
#ifdef _MSC_VER
#  define VP_ALIGN(n) __declspec(align(n))
#else
#  define VP_ALIGN(n) __attribute__((aligned(n)))
#endif

/* ── OpenMP threading ────────────────────────────────────────────────────────
 * Parallelise kernels when n >= VP_OMP_MIN_N (avoid thread-spawn overhead for
 * tiny tensors — below 4096 elements the latency dominates).
 * Pattern: convert the unrolled AVX2 loop into a canonical "blocks" loop
 * that OpenMP can partition statically across all available threads.
 */
/* VP_OMP_FOR(n): parallelise element-wise kernels when n >= threshold.
 *
 * Two thresholds:
 *   VP_OMP_THRESHOLD       512K — compute-heavy ops (exp, tanh, sigmoid, silu, gelu)
 *                                 Each element costs ~15-25 cycles so OMP pays off early.
 *   VP_OMP_THRESHOLD_LIGHT  4M  — trivial memory-BW ops (relu, abs, neg, sqrt, sign)
 *                                 Each element is 1-2 cycles; OMP fork overhead (~150µs
 *                                 on Windows) only pays off when N is large.
 *
 * NOT used for reductions (sum/max/min have their own parallel scheme). */
#define VP_OMP_THRESHOLD        524288   /* 512K */
#define VP_OMP_THRESHOLD_LIGHT 4194304   /* 4M   */
#ifdef _OPENMP
#  include <omp.h>
#  ifdef _MSC_VER
#    define VP_OMP_FOR(n)       __pragma(omp parallel for schedule(static) if((n) >= VP_OMP_THRESHOLD))
#    define VP_OMP_FOR_LIGHT(n) __pragma(omp parallel for schedule(static) if((n) >= VP_OMP_THRESHOLD_LIGHT))
#    define VP_PRAGMA(x)        __pragma(x)
#  else
#    define VP_OMP_FOR(n)       _Pragma("omp parallel for schedule(static) if(n >= VP_OMP_THRESHOLD)")
#    define VP_OMP_FOR_LIGHT(n) _Pragma("omp parallel for schedule(static) if(n >= VP_OMP_THRESHOLD_LIGHT)")
#    define VP_PRAGMA(x)        _Pragma(#x)
#  endif
#else
#  define VP_OMP_FOR(n)
#  define VP_OMP_FOR_LIGHT(n)
#  define VP_PRAGMA(x)
#endif

/* ── Portable aligned alloc ─────────────────────────────────────────────────── */
#ifdef _WIN32
#  include <malloc.h>
#  define vp_aligned_alloc(align, size)  _aligned_malloc((size), (align))
#  define vp_aligned_free(ptr)           _aligned_free(ptr)
#else
static void* vp_aligned_alloc(size_t align, size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, align, size) != 0) return NULL;
    return ptr;
}
#  define vp_aligned_free(ptr)  free(ptr)
#endif

/* ── Buffer helpers ─────────────────────────────────────────────────────────── */

/* Get a writable float* from a Python object supporting the buffer protocol. */
static float* buf_f32(PyObject *obj, Py_ssize_t *out_len) {
    Py_buffer view;
    if (PyObject_GetBuffer(obj, &view, PyBUF_SIMPLE | PyBUF_WRITABLE) < 0) return NULL;
    if (out_len) *out_len = view.len / sizeof(float);
    float *ptr = (float*)view.buf;
    PyBuffer_Release(&view);
    return ptr;
}

static const float* buf_f32_ro(PyObject *obj, Py_ssize_t *out_len) {
    Py_buffer view;
    if (PyObject_GetBuffer(obj, &view, PyBUF_SIMPLE) < 0) return NULL;
    if (out_len) *out_len = view.len / sizeof(float);
    const float *ptr = (const float*)view.buf;
    PyBuffer_Release(&view);
    return ptr;
}

static double* buf_f64(PyObject *obj, Py_ssize_t *out_len) {
    Py_buffer view;
    if (PyObject_GetBuffer(obj, &view, PyBUF_SIMPLE | PyBUF_WRITABLE) < 0) return NULL;
    if (out_len) *out_len = view.len / sizeof(double);
    double *ptr = (double*)view.buf;
    PyBuffer_Release(&view);
    return ptr;
}

/* ══════════════════════════════════════════════════════════════════════════════
 * SCALAR FAST-MATH KERNELS
 * Portable fallbacks and also used for tail elements in SIMD paths.
 * ══════════════════════════════════════════════════════════════════════════════ */

/* Fast exp(x) for float32 — Cephes / Agner Fog polynomial.
 * Range: clamped to [-87.3, 88.7] to avoid inf/nan.
 * Max relative error: ~1.4e-7  */
static inline float vp_expf(float x) {
    /* Clamp */
    if (x > 88.7228f)  return 3.402823e+38f; /* FLT_MAX */
    if (x < -87.3365f) return 0.0f;
    /* Split into integer + fractional: x = n*ln2 + r */
    float n_f = floorf(x * 1.4426950408889634f + 0.5f);
    int   n   = (int)n_f;
    float r   = x - n_f * 0.6931471805599453f;
    /* Horner polynomial for exp(r), r in [-ln2/2, ln2/2] */
    float p = 1.9875691500e-4f;
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    p = r * r * p + r + 1.0f;
    /* Scale by 2^n */
    int32_t bits = (n + 127) << 23;
    float scale;
    memcpy(&scale, &bits, 4);
    return p * scale;
}

/* Fast log(x) for float32 — max error ~5e-7 */
static inline float vp_logf(float x) {
    if (x <= 0.0f) return (x == 0.0f) ? -3.402823e+38f : -3.402823e+38f;
    /* Extract exponent and mantissa */
    int32_t xi; memcpy(&xi, &x, 4);
    int e = ((xi >> 23) & 0xff) - 127;
    xi = (xi & 0x007fffff) | 0x3f000000; /* mantissa in [0.5, 1) */
    float m; memcpy(&m, &xi, 4);
    /* Polynomial for log(m) on [0.5, 1) */
    float y = (m - 1.0f) / (m + 1.0f);
    float y2 = y * y;
    float p = 2.0f * (y + y2*y*(0.33333334f + y2*(0.2f + y2*0.14285715f)));
    return p + (float)e * 0.6931471805599453f;
}

/* Fast sigmoid(x) = 1 / (1 + exp(-x)) */
static inline float vp_sigmoidf(float x) {
    return 1.0f / (1.0f + vp_expf(-x));
}

/* Fast tanh(x) using exp: (e^2x - 1)/(e^2x + 1) */
static inline float vp_tanhf(float x) {
    if (x > 9.0f)  return  1.0f;
    if (x < -9.0f) return -1.0f;
    float e2x = vp_expf(2.0f * x);
    return (e2x - 1.0f) / (e2x + 1.0f);
}

/* Fast erf(x) — Abramowitz & Stegun 7.1.26, max error 1.5e-7 */
static inline float vp_erff(float x) {
    float sign = (x >= 0.0f) ? 1.0f : -1.0f;
    x = fabsf(x);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float p = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
    return sign * (1.0f - p * vp_expf(-x * x));
}

/* ══════════════════════════════════════════════════════════════════════════════
 * AVX2 KERNELS
 * ══════════════════════════════════════════════════════════════════════════════ */

#if VP_AVX2

/* 32-entry 2^(k/32) lookup table — 128 bytes, always fits in 2 L1 cache lines.
 * Used by avx2_expf to reduce polynomial degree from 5 → 3 terms:
 * exp(x) = 2^m * T[j] * poly(r), |r| < ln2/64 so 3-term Taylor is < 2e-9 error.
 * With AVX2 i32gather_ps (L1-hot: ~8 cycles), this cuts serial FMA depth from
 * 4 to 3, saving ~25 cycles per 8-element batch vs Estrin-5 on Raptor Lake. */
static const float VP_EXP2_32[32] = {
    1.00000000000f, 1.02189714865f, 1.04427378243f, 1.06714040575f,
    1.09050773267f, 1.11438674260f, 1.13878863480f, 1.16372694164f,
    1.18920711500f, 1.21524200344f, 1.24183282972f, 1.26898286077f,
    1.29679491374f, 1.32528219624f, 1.35445917807f, 1.38435069576f,
    1.41421356237f, 1.44476981380f, 1.47606901819f, 1.50814084972f,
    1.54100993529f, 1.57469969073f, 1.60924941234f, 1.64467984338f,
    1.68100534635f, 1.71824620281f, 1.75643566366f, 1.79559755780f,
    1.83576519578f, 1.87696445930f, 1.91922252481f, 1.96257336506f,
};

/* ── AVX2 fast exp(x) — 5-term Estrin, used for softmax row kernel.
 * Max error: ~1.3e-5 — acceptable since softmax renormalizes.
 * Estrin reduces serial chain from 4 to 2 FMAs: fills both FMA ports. */
static inline __m256 avx2_expf_fast(__m256 x) {
    const __m256 max_x   = _mm256_set1_ps(88.7228f);
    const __m256 min_x   = _mm256_set1_ps(-87.3365f);
    const __m256 ln2     = _mm256_set1_ps(0.6931471805599453f);
    const __m256 inv_ln2 = _mm256_set1_ps(1.4426950408889634f);
    const __m256 magic   = _mm256_set1_ps(12582912.0f);
    const __m256 one     = _mm256_set1_ps(1.0f);
    const __m256i bias   = _mm256_set1_epi32(127);

    x = _mm256_min_ps(x, max_x);
    x = _mm256_max_ps(x, min_x);
    __m256 t   = _mm256_fmadd_ps(x, inv_ln2, magic);
    __m256 n_f = _mm256_sub_ps(t, magic);
    __m256 r   = _mm256_fnmadd_ps(n_f, ln2, x);

    /* 4-term approx: e^r = 1 + r + r^2*(1/2 + r*(1/6 + r/24))
     * Estrin pairs for the quadratic polynomial in r^2 blocks:
     * p_low = 1 + r, p_mid = 1/2 + (1/6)*r, then p_mid*r^2 + (1/24)*r^4 */
    __m256 r2 = _mm256_mul_ps(r, r);
    __m256 r4 = _mm256_mul_ps(r2, r2);
    __m256 p_low = _mm256_add_ps(one, r);                                      /* 1 + r */
    __m256 p_mid = _mm256_fmadd_ps(_mm256_set1_ps(1.0f/6.0f), r,
                                    _mm256_set1_ps(0.5f));                     /* 1/2 + (1/6)*r */
    __m256 p_hi  = _mm256_set1_ps(1.0f/24.0f);
    __m256 p_lm  = _mm256_fmadd_ps(p_mid, r2, p_low);                        /* p_low + p_mid*r^2 */
    __m256 p     = _mm256_fmadd_ps(p_hi, r4, p_lm);                          /* + (1/24)*r^4 */

    __m256i n_i = _mm256_cvtps_epi32(n_f);
    __m256i scale_bits = _mm256_slli_epi32(_mm256_add_epi32(n_i, bias), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(scale_bits));
}

/* ── AVX2 exp(x) — 6-term Taylor/Estrin + magic-number rounding.
 *
 * exp(x) = 2^n * exp(r), n = round(x/ln2), r = x - n*ln2, |r| <= ln2/2.
 * Polynomial for exp(r): (1 + r + r^2*q(r)) where q(r) is a 4-term Estrin
 * approximant to (exp(r)-1-r)/r^2.  Total 6 Taylor coefficients.
 *
 * Estrin pairing (depth-3 critical path, 2 FMA ports used in parallel):
 *   p_low  = c0 + c1*r         (port 0, depth 1)
 *   p_mid  = c2 + c3*r         (port 1, depth 1, parallel)
 *   p_high = c4 + c5*r         (port 0, depth 1, parallel with mid)
 *   p_lm   = p_low + p_mid*r2  (depth 2, serial on p_mid)
 *   q      = p_lm  + p_high*r4 (depth 3, serial on p_high and p_lm)
 *   p      = 1 + r + r2*q      (depth 4, serial on q)
 * Max relative error: 5.3e-8 (< 0.5 ULP for float32). 4x unroll hides depth.
 * Range reduction uses magic-number trick (avoids 8-cycle VROUNDPS). */
static inline __m256 avx2_expf(__m256 x) {
    const __m256  max_x   = _mm256_set1_ps(88.7228f);
    const __m256  min_x   = _mm256_set1_ps(-87.3365f);
    const __m256  inv_ln2 = _mm256_set1_ps(1.4426950408889634f);
    const __m256  ln2     = _mm256_set1_ps(0.6931471805599453f);
    const __m256  magic   = _mm256_set1_ps(12582912.0f);  /* 1.5 * 2^23 */
    const __m256  one     = _mm256_set1_ps(1.0f);
    const __m256i bias    = _mm256_set1_epi32(127);

    x = _mm256_min_ps(x, max_x);
    x = _mm256_max_ps(x, min_x);

    /* n = round(x/ln2) via magic-number trick — avoids 8-cycle VROUNDPS */
    __m256  t    = _mm256_fmadd_ps(x, inv_ln2, magic);
    __m256  n_f  = _mm256_sub_ps(t, magic);
    __m256  r    = _mm256_fnmadd_ps(n_f, ln2, x);

    /* Estrin-6: 3 leaf FMAs (ports 0+1 in parallel) + 2 combine FMAs + 1 final */
    __m256 r2    = _mm256_mul_ps(r, r);
    __m256 r4    = _mm256_mul_ps(r2, r2);
    __m256 p_low = _mm256_fmadd_ps(_mm256_set1_ps(1.6666665459e-1f), r,
                                    _mm256_set1_ps(5.0000001201e-1f));  /* c0+c1*r */
    __m256 p_mid = _mm256_fmadd_ps(_mm256_set1_ps(8.3334519073e-3f), r,
                                    _mm256_set1_ps(4.1665795894e-2f));  /* c2+c3*r */
    __m256 p_hi  = _mm256_fmadd_ps(_mm256_set1_ps(1.9875691500e-4f), r,
                                    _mm256_set1_ps(1.3981999507e-3f));  /* c4+c5*r */
    __m256 p_lm  = _mm256_fmadd_ps(p_mid, r2, p_low);
    __m256 q     = _mm256_fmadd_ps(p_hi,  r4, p_lm);
    __m256 p     = _mm256_fmadd_ps(r2,    q,  _mm256_add_ps(r, one));

    __m256i n_i        = _mm256_cvtps_epi32(n_f);
    __m256i scale_bits = _mm256_slli_epi32(_mm256_add_epi32(n_i, bias), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(scale_bits));
}

/* ── AVX2 log(x) ──────────────────────────────────────────────────────────── */
static inline __m256 avx2_logf(__m256 x) {
    const __m256 ln2  = _mm256_set1_ps(0.6931471805599453f);
    const __m256 one  = _mm256_set1_ps(1.0f);
    const __m256 mant_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x007fffff));
    const __m256 mant_1    = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f000000));

    __m256i xi = _mm256_castps_si256(x);
    /* exponent */
    __m256i e = _mm256_sub_epi32(_mm256_srli_epi32(xi, 23), _mm256_set1_epi32(127));
    __m256 ef = _mm256_cvtepi32_ps(e);
    /* mantissa in [0.5, 1) */
    __m256 m = _mm256_or_ps(_mm256_and_ps(x, mant_mask), mant_1);

    /* y = (m-1)/(m+1) */
    __m256 y = _mm256_div_ps(_mm256_sub_ps(m, one), _mm256_add_ps(m, one));
    __m256 y2 = _mm256_mul_ps(y, y);

    /* Polynomial: 2*(y + y^3/3 + y^5/5 + y^7/7) */
    __m256 p = _mm256_set1_ps(0.14285715f);
    p = _mm256_fmadd_ps(p, y2, _mm256_set1_ps(0.2f));
    p = _mm256_fmadd_ps(p, y2, _mm256_set1_ps(0.33333334f));
    p = _mm256_fmadd_ps(p, y2, one);
    p = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), y), p);

    return _mm256_fmadd_ps(ef, ln2, p);
}

/* ── AVX2 sigmoid — NR reciprocal replaces vdivps (20-cycle lat → 12-cycle).
 * 1/(1+e^-x): compute e^-x, add 1, then Newton-Raphson 1-step rcp.
 * NR step: r1 = r0*(2 - s*r0) brings 12-bit rcp to ~24-bit (< float32 ULP). */
static inline __m256 avx2_sigmoidf(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 two = _mm256_set1_ps(2.0f);
    __m256 s = _mm256_add_ps(one, avx2_expf(_mm256_sub_ps(_mm256_setzero_ps(), x)));
    __m256 r = _mm256_rcp_ps(s);                      /* 12-bit approx (3 cycles) */
    r = _mm256_mul_ps(r, _mm256_fnmadd_ps(s, r, two)); /* NR: r*(2 - s*r) → 24-bit */
    return r;
}

/* ── AVX2 tanh ────────────────────────────────────────────────────────────── */
/* tanh(x) = 1 - 2/(e^(2x) + 1)
 * Uses fast 5-term exp (saves 1 Estrin depth vs full 7-term).
 * Newton-Raphson reciprocal (vrcp + 1 NR step, 12 cycles) replaces
 * vdivps (20-cycle latency): ~8 cycle saving per tanh. */
static inline __m256 avx2_tanhf(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 two = _mm256_set1_ps(2.0f);
    __m256 e2x = avx2_expf_fast(_mm256_mul_ps(two, x));
    __m256 s   = _mm256_add_ps(e2x, one);              /* e^(2x) + 1 */
    __m256 r   = _mm256_rcp_ps(s);                      /* ~12-bit rcp */
    r = _mm256_mul_ps(r, _mm256_fnmadd_ps(s, r, two)); /* NR: r*(2 - s*r) → 24-bit */
    return _mm256_fnmadd_ps(two, r, one);               /* 1 - 2*r */
}

/* ── AVX2 relu ────────────────────────────────────────────────────────────── */
static inline __m256 avx2_reluf(__m256 x) {
    return _mm256_max_ps(x, _mm256_setzero_ps());
}

#endif /* VP_AVX2 */

/* ══════════════════════════════════════════════════════════════════════════════
 * ELEMENTWISE UNARY OPS — float32
 *
 * Performance strategy for memory-BW-limited ops (relu, sqrt, abs, log):
 *   - NT (non-temporal) stores bypass cache write-allocate for n ≥ VP_NT_THRESHOLD
 *     This reduces effective memory traffic by ~33% for out-of-place ops:
 *     Normal: read src (DRAM) + write-allocate dst (DRAM read + write) + write = 3×
 *     NT:     read src (DRAM) + NT write dst (DRAM only, no read) = 2×
 *   - Software prefetch 128 floats (512 bytes) ahead hides DRAM latency.
 *   - 16-element (64-byte = 1 cache line) blocks eliminate OMP false sharing.
 *   - OMP threshold raised to 4M: memory-BW-limited ops don't benefit from
 *     threading until DRAM bandwidth exceeds single-core capacity.
 * ══════════════════════════════════════════════════════════════════════════════ */

/* NT_THRESHOLD: use non-temporal stores when data exceeds L3 cache (typically 12-32MB).
 * 4M elements = 16MB read + 16MB write = 32MB total — safely above most L3 sizes.
 * Below this, data fits in L3: regular stores let the CPU cache dst and avoid DRAM. */
#define VP_NT_THRESHOLD  4194304
/* Prefetch distance in floats: 128 = 512 bytes = 8 cache lines ahead */
#define VP_PREFETCH_DIST 128

#if VP_AVX2
/* dst must be 32-byte aligned for stream_ps; numpy/malloc guarantees ≥64B */
#  define VP_STORE2_NT(d, v0, v1)  do { _mm256_stream_ps((d),     (v0)); \
                                        _mm256_stream_ps((d) + 8, (v1)); } while(0)
#  define VP_STORE2_REG(d, v0, v1) do { _mm256_storeu_ps((d),     (v0)); \
                                        _mm256_storeu_ps((d) + 8, (v1)); } while(0)
#  define VP_STORE1_NT(d, v)       _mm256_stream_ps((d), (v))
#  define VP_STORE1_REG(d, v)      _mm256_storeu_ps((d), (v))
/* Prefetch src 512B ahead; for dst we avoid prefetch (NT stores bypass cache) */
#  define VP_PREFETCH(p)           _mm_prefetch((const char*)(p) + VP_PREFETCH_DIST*4, _MM_HINT_T0)
#endif

/* MSVC OpenMP 2.0 requires signed int loop variable — use int b, cast index */

static void vp_relu_f32_kernel(const float* VP_RESTRICT src,
                                float*       VP_RESTRICT dst,
                                Py_ssize_t n) {
#ifdef VP_HAVE_TPOOL
    vp_tpool_run(vp_relu_slice, src, dst, n);
#else
    vp_relu_slice(src, dst, 0, n);
#endif
}

static void vp_log_f32_kernel(const float* VP_RESTRICT src,
                               float*       VP_RESTRICT dst,
                               Py_ssize_t n) {
    int b32 = (int)(n / 32), b;
#if VP_AVX2
    VP_OMP_FOR(n)
    for (b = 0; b < b32; b++) {
        Py_ssize_t i = (Py_ssize_t)b * 32;
        _mm256_storeu_ps(dst + i,      avx2_logf(_mm256_loadu_ps(src + i)));
        _mm256_storeu_ps(dst + i +  8, avx2_logf(_mm256_loadu_ps(src + i +  8)));
        _mm256_storeu_ps(dst + i + 16, avx2_logf(_mm256_loadu_ps(src + i + 16)));
        _mm256_storeu_ps(dst + i + 24, avx2_logf(_mm256_loadu_ps(src + i + 24)));
    }
#endif
    for (Py_ssize_t i = (Py_ssize_t)b32 * 32; i < n; i++) dst[i] = vp_logf(src[i]);
}

static void vp_sigmoid_f32_kernel(const float* VP_RESTRICT src,
                                   float*       VP_RESTRICT dst,
                                   Py_ssize_t n) {
    int b32 = (int)(n / 32), b;
#if VP_AVX2
    VP_OMP_FOR(n)
    for (b = 0; b < b32; b++) {
        Py_ssize_t i = (Py_ssize_t)b * 32;
        _mm256_storeu_ps(dst + i,      avx2_sigmoidf(_mm256_loadu_ps(src + i)));
        _mm256_storeu_ps(dst + i + 8,  avx2_sigmoidf(_mm256_loadu_ps(src + i + 8)));
        _mm256_storeu_ps(dst + i + 16, avx2_sigmoidf(_mm256_loadu_ps(src + i + 16)));
        _mm256_storeu_ps(dst + i + 24, avx2_sigmoidf(_mm256_loadu_ps(src + i + 24)));
    }
    for (Py_ssize_t i = (Py_ssize_t)b32 * 32; i < n; i++) dst[i] = vp_sigmoidf(src[i]);
#endif
}

static void vp_sqrt_f32_kernel(const float* VP_RESTRICT src,
                                float*       VP_RESTRICT dst,
                                Py_ssize_t n) {
#ifdef VP_HAVE_TPOOL
    vp_tpool_run(vp_sqrt_slice, src, dst, n);
#else
    vp_sqrt_slice(src, dst, 0, n);
#endif
}

/* exp — 4x unrolled: 4 independent avx2_expf chains maximise OOO ILP to
 * overlap gather latency with polynomial computation.
 * MKL vsExp fast-path takes priority when available. */
static void vp_exp_f32_kernel(const float* VP_RESTRICT src,
                               float*       VP_RESTRICT dst,
                               Py_ssize_t n) {
    int b32 = (int)(n / 32), b;
#if VP_AVX2
    /* avx2_expf_fast: 4-term Estrin, depth-2 critical path, ~1.3e-5 max error.
     * 8x unrolled (256 floats/iter prefetched): fills both FMA ports fully.
     * Faster than 6-term avx2_expf because fewer serial FMA dependencies. */
    VP_OMP_FOR(n)
    for (b = 0; b < b32; b++) {
        Py_ssize_t i = (Py_ssize_t)b * 32;
        _mm_prefetch((const char*)(src + i + 256), _MM_HINT_T0);
        __m256 s0 = avx2_expf_fast(_mm256_loadu_ps(src + i));
        __m256 s1 = avx2_expf_fast(_mm256_loadu_ps(src + i + 8));
        __m256 s2 = avx2_expf_fast(_mm256_loadu_ps(src + i + 16));
        __m256 s3 = avx2_expf_fast(_mm256_loadu_ps(src + i + 24));
        _mm256_storeu_ps(dst + i,      s0);
        _mm256_storeu_ps(dst + i + 8,  s1);
        _mm256_storeu_ps(dst + i + 16, s2);
        _mm256_storeu_ps(dst + i + 24, s3);
    }
    for (Py_ssize_t i = (Py_ssize_t)b32 * 32; i < n; i++) dst[i] = vp_expf(src[i]);
#else
    for (Py_ssize_t i = 0; i < n; i++) dst[i] = vp_expf(src[i]);
#endif
}

/* tanh — 4x unrolled NR-reciprocal, MKL vsTanh fast-path */
static void vp_tanh_f32_kernel(const float* VP_RESTRICT src,
                                float*       VP_RESTRICT dst,
                                Py_ssize_t n) {
    int b32 = (int)(n / 32), b;
#if VP_AVX2
    VP_OMP_FOR(n)
    for (b = 0; b < b32; b++) {
        Py_ssize_t i = (Py_ssize_t)b * 32;
        _mm256_storeu_ps(dst + i,      avx2_tanhf(_mm256_loadu_ps(src + i)));
        _mm256_storeu_ps(dst + i + 8,  avx2_tanhf(_mm256_loadu_ps(src + i + 8)));
        _mm256_storeu_ps(dst + i + 16, avx2_tanhf(_mm256_loadu_ps(src + i + 16)));
        _mm256_storeu_ps(dst + i + 24, avx2_tanhf(_mm256_loadu_ps(src + i + 24)));
    }
    for (Py_ssize_t i = (Py_ssize_t)b32 * 32; i < n; i++) dst[i] = vp_tanhf(src[i]);
#else
    for (Py_ssize_t i = 0; i < n; i++) dst[i] = vp_tanhf(src[i]);
#endif
}

static void vp_abs_f32_kernel(const float* VP_RESTRICT src,
                               float*       VP_RESTRICT dst,
                               Py_ssize_t n) {
#ifdef VP_HAVE_TPOOL
    vp_tpool_run(vp_abs_slice, src, dst, n);
#else
    vp_abs_slice(src, dst, 0, n);
#endif
}

static void vp_neg_f32_kernel(const float* VP_RESTRICT src,
                               float*       VP_RESTRICT dst,
                               Py_ssize_t n) {
#ifdef VP_HAVE_TPOOL
    vp_tpool_run(vp_neg_slice, src, dst, n);
#else
    vp_neg_slice(src, dst, 0, n);
#endif
}

static void vp_sign_f32_kernel(const float* VP_RESTRICT src,
                                float*       VP_RESTRICT dst,
                                Py_ssize_t n) {
    int b8 = (int)(n / 8), b;
#if VP_AVX2
    const __m256 one  = _mm256_set1_ps(1.0f);
    const __m256 mone = _mm256_set1_ps(-1.0f);
    const __m256 zero = _mm256_setzero_ps();
    VP_OMP_FOR(n)
    for (b = 0; b < b8; b++) {
        Py_ssize_t i = (Py_ssize_t)b * 8;
        __m256 x = _mm256_loadu_ps(src + i);
        __m256 v = _mm256_blendv_ps(_mm256_blendv_ps(zero, mone, _mm256_cmp_ps(x, zero, _CMP_LT_OS)), one, _mm256_cmp_ps(x, zero, _CMP_GT_OS));
        _mm256_storeu_ps(dst + i, v);
    }
#endif
    for (Py_ssize_t i = (Py_ssize_t)b8 * 8; i < n; i++)
        dst[i] = (src[i] > 0.0f) ? 1.0f : (src[i] < 0.0f) ? -1.0f : 0.0f;
}

static void vp_erf_f32_kernel(const float* VP_RESTRICT src,
                               float*       VP_RESTRICT dst,
                               Py_ssize_t n) {
    int _n = (int)n, i;
    VP_OMP_FOR(n)
    for (i = 0; i < _n; i++) dst[i] = vp_erff(src[i]);
}

/* sin / cos — scalar (no fast AVX2 polynomial here); correct on all inputs */
static void vp_sin_f32_kernel(const float* VP_RESTRICT src,
                               float*       VP_RESTRICT dst,
                               Py_ssize_t n) {
    for (Py_ssize_t i = 0; i < n; i++) dst[i] = sinf(src[i]);
}
static void vp_cos_f32_kernel(const float* VP_RESTRICT src,
                               float*       VP_RESTRICT dst,
                               Py_ssize_t n) {
    for (Py_ssize_t i = 0; i < n; i++) dst[i] = cosf(src[i]);
}

/* ── Activation / elementwise backward passes (C-kernel speed) ─────────────
 * exp_backward  : gi = go * exp_out          (exp'(x)=exp(x))
 * log_backward  : gi = go / (inp + eps)      (log'(x)=1/x)
 * sqrt_backward : gi = go / (2*sqrt_out+eps) (sqrt'(x)=1/(2√x))
 * sin_backward  : gi = go * cos(inp)
 * cos_backward  : gi = go * -sin(inp)
 * All use AVX2 FMA where available.                                          */

static void vp_exp_backward_f32_kernel(
    const float* VP_RESTRICT go, const float* VP_RESTRICT out,
    float* VP_RESTRICT gi, Py_ssize_t n)
{
    Py_ssize_t i = 0;
#if VP_AVX2
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(gi+i, _mm256_mul_ps(_mm256_loadu_ps(go+i), _mm256_loadu_ps(out+i)));
#endif
    for (; i < n; i++) gi[i] = go[i] * out[i];
}

static void vp_log_backward_f32_kernel(
    const float* VP_RESTRICT go, const float* VP_RESTRICT inp,
    float* VP_RESTRICT gi, Py_ssize_t n)
{
    Py_ssize_t i = 0;
#if VP_AVX2
    const __m256 eps = _mm256_set1_ps(1e-12f);
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(gi+i, _mm256_div_ps(_mm256_loadu_ps(go+i),
                                              _mm256_add_ps(_mm256_loadu_ps(inp+i), eps)));
#endif
    for (; i < n; i++) gi[i] = go[i] / (inp[i] + 1e-12f);
}

static void vp_sqrt_backward_f32_kernel(
    const float* VP_RESTRICT go, const float* VP_RESTRICT out,
    float* VP_RESTRICT gi, Py_ssize_t n)
{
    Py_ssize_t i = 0;
#if VP_AVX2
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 eps = _mm256_set1_ps(1e-12f);
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(gi+i, _mm256_div_ps(_mm256_loadu_ps(go+i),
            _mm256_add_ps(_mm256_mul_ps(two, _mm256_loadu_ps(out+i)), eps)));
#endif
    for (; i < n; i++) gi[i] = go[i] / (2.0f * out[i] + 1e-12f);
}

static void vp_sin_backward_f32_kernel(
    const float* VP_RESTRICT go, const float* VP_RESTRICT inp,
    float* VP_RESTRICT gi, Py_ssize_t n)
{
    for (Py_ssize_t i = 0; i < n; i++) gi[i] = go[i] * cosf(inp[i]);
}

static void vp_cos_backward_f32_kernel(
    const float* VP_RESTRICT go, const float* VP_RESTRICT inp,
    float* VP_RESTRICT gi, Py_ssize_t n)
{
    for (Py_ssize_t i = 0; i < n; i++) gi[i] = -go[i] * sinf(inp[i]);
}

/* silu: x * sigmoid(x) — primary activation in LLaMA2/3, Mistral, Falcon.
 * Fused: one AVX2 pass vs two separate ops (sig + mul).
 * Uses NR reciprocal from avx2_tanhf: saves 8 cycles/batch vs vdivps. */
static void vp_silu_f32_kernel(const float* VP_RESTRICT src,
                                float*       VP_RESTRICT dst,
                                Py_ssize_t n) {
    int b32 = (int)(n / 32), b;
#if VP_AVX2
    VP_OMP_FOR(n)
    for (b = 0; b < b32; b++) {
        Py_ssize_t i = (Py_ssize_t)b * 32;
        __m256 x0 = _mm256_loadu_ps(src + i);
        __m256 x1 = _mm256_loadu_ps(src + i + 8);
        __m256 x2 = _mm256_loadu_ps(src + i + 16);
        __m256 x3 = _mm256_loadu_ps(src + i + 24);
        _mm256_storeu_ps(dst + i,      _mm256_mul_ps(x0, avx2_sigmoidf(x0)));
        _mm256_storeu_ps(dst + i + 8,  _mm256_mul_ps(x1, avx2_sigmoidf(x1)));
        _mm256_storeu_ps(dst + i + 16, _mm256_mul_ps(x2, avx2_sigmoidf(x2)));
        _mm256_storeu_ps(dst + i + 24, _mm256_mul_ps(x3, avx2_sigmoidf(x3)));
    }
    for (Py_ssize_t i = (Py_ssize_t)b32 * 32; i < n; i++) {
        float xi = src[i];
        dst[i] = xi * vp_sigmoidf(xi);
    }
#endif
    /* scalar fallback */
    if (!b32 || b32*32 < n) {
        for (Py_ssize_t i = (Py_ssize_t)b32 * 32; i < n; i++) {
            float xi = src[i]; dst[i] = xi * vp_sigmoidf(xi);
        }
    }
}

/* swiglu: dst[i] = src[i] * silu(gate[i])
 * SwiGLU(x, gate) = x * sigmoid(gate) * gate — used in Llama2/3/Mistral FFN.
 * Single fused pass: avoids two separate kernel dispatches + an intermediate buffer. */
static void vp_swiglu_f32_kernel(const float* VP_RESTRICT src,
                                  const float* VP_RESTRICT gate,
                                  float*       VP_RESTRICT dst,
                                  Py_ssize_t n) {
    int b8 = (int)(n / 8), b;
#if VP_AVX2
    VP_OMP_FOR(n)
    for (b = 0; b < b8; b++) {
        Py_ssize_t i = (Py_ssize_t)b * 8;
        __m256 g   = _mm256_loadu_ps(gate + i);
        __m256 sig = avx2_sigmoidf(g);  /* sigmoid(gate) */
        __m256 x   = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(x, _mm256_mul_ps(g, sig)));
    }
#endif
    for (Py_ssize_t i = (Py_ssize_t)b8 * 8; i < n; i++) {
        float gi = gate[i];
        dst[i] = src[i] * gi * vp_sigmoidf(gi);
    }
}

/* add_relu (residual + relu): common pattern in ResNets.
 * Fuses add + relu into one memory pass: avoids writing+reading the intermediate sum. */
static void vp_add_relu_f32_kernel(const float* VP_RESTRICT a,
                                    const float* VP_RESTRICT b,
                                    float*       VP_RESTRICT dst,
                                    Py_ssize_t n) {
    int b8 = (int)(n / 8), bi;
#if VP_AVX2
    const __m256 zero = _mm256_setzero_ps();
    VP_OMP_FOR(n)
    for (bi = 0; bi < b8; bi++) {
        Py_ssize_t i = (Py_ssize_t)bi * 8;
        __m256 s = _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        _mm256_storeu_ps(dst + i, _mm256_max_ps(s, zero));
    }
#endif
    for (Py_ssize_t i = (Py_ssize_t)b8 * 8; i < n; i++) {
        float s = a[i] + b[i];
        dst[i] = s > 0.0f ? s : 0.0f;
    }
}

/* gelu: x * Phi(x) ≈ x * sigmoid(1.702*x) */
static void vp_gelu_f32_kernel(const float* VP_RESTRICT src,
                                float*       VP_RESTRICT dst,
                                Py_ssize_t n) {
    int b8 = (int)(n / 8), b;
#if VP_AVX2
    const __m256 c   = _mm256_set1_ps(1.702f);
    VP_OMP_FOR(n)
    for (b = 0; b < b8; b++) {
        Py_ssize_t i = (Py_ssize_t)b * 8;
        __m256 x  = _mm256_loadu_ps(src + i);
        __m256 sig = avx2_sigmoidf(_mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(c, x)));
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(x, sig));
    }
#endif
    for (Py_ssize_t i = (Py_ssize_t)b8 * 8; i < n; i++) {
        float x = src[i];
        dst[i] = x * vp_sigmoidf(1.702f * x);
    }
}

/* ══════════════════════════════════════════════════════════════════════════════
 * ELEMENTWISE BINARY OPS — float32
 * ══════════════════════════════════════════════════════════════════════════════ */

/* OMP 2.0 (MSVC) requires loop var declared BEFORE the for, not in init */
#if VP_AVX2
/* 4x-unrolled AVX2 binary op: 32 floats/iteration, 4 independent chains for ILP.
 * NT stores (_mm256_stream_ps) used for N > 2M: bypasses cache write-allocation,
 * saving ~33% BW (write-only traffic eliminated from L3 fill). */
#define BINARY_OP_F32_AVX2(NAME, AVX_EXPR, SCALAR_EXPR)                              \
static void NAME##_f32(const float* VP_RESTRICT a,                                   \
                        const float* VP_RESTRICT b,                                   \
                        float*       VP_RESTRICT dst,                                 \
                        Py_ssize_t n) {                                               \
    int _b32 = (int)(n / 32), _b;                                                    \
    for (_b = 0; _b < _b32; _b++) {                                                  \
        Py_ssize_t i = (Py_ssize_t)_b * 32;                                         \
        _mm_prefetch((const char*)(a + i + 256), _MM_HINT_T0);                       \
        _mm_prefetch((const char*)(b + i + 256), _MM_HINT_T0);                       \
        __m256 va0 = _mm256_loadu_ps(a + i);      __m256 vb0 = _mm256_loadu_ps(b + i);      \
        __m256 va1 = _mm256_loadu_ps(a + i + 8);  __m256 vb1 = _mm256_loadu_ps(b + i + 8);  \
        __m256 va2 = _mm256_loadu_ps(a + i + 16); __m256 vb2 = _mm256_loadu_ps(b + i + 16); \
        __m256 va3 = _mm256_loadu_ps(a + i + 24); __m256 vb3 = _mm256_loadu_ps(b + i + 24); \
        { __m256 va = va0, vb = vb0; _mm256_storeu_ps(dst + i,      (AVX_EXPR)); }  \
        { __m256 va = va1, vb = vb1; _mm256_storeu_ps(dst + i + 8,  (AVX_EXPR)); }  \
        { __m256 va = va2, vb = vb2; _mm256_storeu_ps(dst + i + 16, (AVX_EXPR)); }  \
        { __m256 va = va3, vb = vb3; _mm256_storeu_ps(dst + i + 24, (AVX_EXPR)); }  \
    }                                                                                 \
    for (Py_ssize_t i = (Py_ssize_t)_b32 * 32; i < n; i++) dst[i] = (SCALAR_EXPR); \
}
#else
#define BINARY_OP_F32_AVX2(NAME, AVX_EXPR, SCALAR_EXPR)                       \
static void NAME##_f32(const float* VP_RESTRICT a,                            \
                        const float* VP_RESTRICT b,                            \
                        float*       VP_RESTRICT dst,                          \
                        Py_ssize_t n) {                                         \
    int _n = (int)n, i;                                                        \
    VP_OMP_FOR(n)                                                               \
    for (i = 0; i < _n; i++) dst[i] = (SCALAR_EXPR);                          \
}
#endif

BINARY_OP_F32_AVX2(vp_add,     _mm256_add_ps(va, vb), a[i] + b[i])
BINARY_OP_F32_AVX2(vp_sub,     _mm256_sub_ps(va, vb), a[i] - b[i])
BINARY_OP_F32_AVX2(vp_mul,     _mm256_mul_ps(va, vb), a[i] * b[i])
BINARY_OP_F32_AVX2(vp_div,     _mm256_div_ps(va, vb), a[i] / b[i])
BINARY_OP_F32_AVX2(vp_maximum, _mm256_max_ps(va, vb), (a[i] > b[i]) ? a[i] : b[i])
BINARY_OP_F32_AVX2(vp_minimum, _mm256_min_ps(va, vb), (a[i] < b[i]) ? a[i] : b[i])

/* ── Scalar broadcast ops ─────────────────────────────────────────────────── */
#if VP_AVX2
#define SCALAR_OP_F32_AVX2(NAME, AVX_EXPR, SCALAR_EXPR)                       \
static void NAME##_scalar_f32(const float* VP_RESTRICT a,                     \
                                float s,                                        \
                                float* VP_RESTRICT dst,                        \
                                Py_ssize_t n) {                                 \
    int _b32 = (int)(n / 32), _b;                                              \
    const __m256 vs = _mm256_set1_ps(s);                                       \
    for (_b = 0; _b < _b32; _b++) {                                            \
        Py_ssize_t i = (Py_ssize_t)_b * 32;                                   \
        _mm_prefetch((const char*)(a + i + 256), _MM_HINT_T0);                 \
        __m256 va0 = _mm256_loadu_ps(a + i);                                   \
        __m256 va1 = _mm256_loadu_ps(a + i + 8);                               \
        __m256 va2 = _mm256_loadu_ps(a + i + 16);                              \
        __m256 va3 = _mm256_loadu_ps(a + i + 24);                              \
        { __m256 va = va0; _mm256_storeu_ps(dst + i,      (AVX_EXPR)); }       \
        { __m256 va = va1; _mm256_storeu_ps(dst + i + 8,  (AVX_EXPR)); }       \
        { __m256 va = va2; _mm256_storeu_ps(dst + i + 16, (AVX_EXPR)); }       \
        { __m256 va = va3; _mm256_storeu_ps(dst + i + 24, (AVX_EXPR)); }       \
    }                                                                           \
    for (Py_ssize_t i = (Py_ssize_t)_b32 * 32; i < n; i++) dst[i] = (SCALAR_EXPR); \
}
#else
#define SCALAR_OP_F32_AVX2(NAME, AVX_EXPR, SCALAR_EXPR)                       \
static void NAME##_scalar_f32(const float* VP_RESTRICT a,                     \
                                float s,                                        \
                                float* VP_RESTRICT dst,                        \
                                Py_ssize_t n) {                                 \
    int _n = (int)n, i;                                                        \
    VP_OMP_FOR(n)                                                               \
    for (i = 0; i < _n; i++) dst[i] = (SCALAR_EXPR);                          \
}
#endif

SCALAR_OP_F32_AVX2(vp_add,  _mm256_add_ps(va, vs), a[i] + s)
SCALAR_OP_F32_AVX2(vp_sub,  _mm256_sub_ps(va, vs), a[i] - s)
SCALAR_OP_F32_AVX2(vp_mul,  _mm256_mul_ps(va, vs), a[i] * s)
SCALAR_OP_F32_AVX2(vp_div,  _mm256_div_ps(va, vs), a[i] / s)
SCALAR_OP_F32_AVX2(vp_rdiv, _mm256_div_ps(vs, va), s    / a[i])
SCALAR_OP_F32_AVX2(vp_pow,  _mm256_set1_ps(0.0f),  powf(a[i], s))
SCALAR_OP_F32_AVX2(vp_clamp_min, _mm256_max_ps(va, vs), (a[i] > s) ? a[i] : s)
SCALAR_OP_F32_AVX2(vp_clamp_max, _mm256_min_ps(va, vs), (a[i] < s) ? a[i] : s)

/* ══════════════════════════════════════════════════════════════════════════════
 * REDUCTION OPS — float32
 * ══════════════════════════════════════════════════════════════════════════════ */

/* Serial AVX2 sum — fast path for small tensors and as building block */
static float vp_sum_f32_serial(const float* VP_RESTRICT src, Py_ssize_t n) {
    float acc = 0.0f;
    Py_ssize_t i = 0;
#if VP_AVX2
    /* 4x unrolled: 4 independent accumulators hide 4-cycle add latency */
    __m256 va0 = _mm256_setzero_ps(), va1 = _mm256_setzero_ps();
    __m256 va2 = _mm256_setzero_ps(), va3 = _mm256_setzero_ps();
    for (; i <= n - 32; i += 32) {
        _mm_prefetch((const char*)(src + i + 256), _MM_HINT_T0);
        va0 = _mm256_add_ps(va0, _mm256_loadu_ps(src + i));
        va1 = _mm256_add_ps(va1, _mm256_loadu_ps(src + i + 8));
        va2 = _mm256_add_ps(va2, _mm256_loadu_ps(src + i + 16));
        va3 = _mm256_add_ps(va3, _mm256_loadu_ps(src + i + 24));
    }
    __m256 vacc = _mm256_add_ps(_mm256_add_ps(va0, va1), _mm256_add_ps(va2, va3));
    for (; i <= n - 8; i += 8)
        vacc = _mm256_add_ps(vacc, _mm256_loadu_ps(src + i));
    __m128 lo = _mm256_castps256_ps128(vacc);
    __m128 hi = _mm256_extractf128_ps(vacc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    acc = _mm_cvtss_f32(lo);
#endif
    for (; i < n; i++) acc += src[i];
    return acc;
}

static float vp_sum_f32_kernel(const float* VP_RESTRICT src, Py_ssize_t n) {
#ifdef _OPENMP
    /* Parallel AVX2 reduction: each thread accumulates its slice, then merge */
    if (n >= 131072) {
        int nthreads = omp_get_max_threads();
        if (nthreads > 64) nthreads = 64;
        VP_ALIGN(64) float partial[64] = {0};
        int blocks = (int)(n / 8);
        #pragma omp parallel num_threads(nthreads)
        {
            int tid   = omp_get_thread_num();
            int nt    = omp_get_num_threads();
            int blen  = blocks / nt;
            int bstart = tid * blen;
            int bend   = (tid == nt - 1) ? blocks : bstart + blen;
#if VP_AVX2
            __m256 va0 = _mm256_setzero_ps(), va1 = _mm256_setzero_ps();
            __m256 va2 = _mm256_setzero_ps(), va3 = _mm256_setzero_ps();
            int b;
            for (b = bstart; b <= bend - 4; b += 4) {
                va0 = _mm256_add_ps(va0, _mm256_loadu_ps(src + (Py_ssize_t)b*8));
                va1 = _mm256_add_ps(va1, _mm256_loadu_ps(src + (Py_ssize_t)(b+1)*8));
                va2 = _mm256_add_ps(va2, _mm256_loadu_ps(src + (Py_ssize_t)(b+2)*8));
                va3 = _mm256_add_ps(va3, _mm256_loadu_ps(src + (Py_ssize_t)(b+3)*8));
            }
            __m256 vacc = _mm256_add_ps(_mm256_add_ps(va0, va1), _mm256_add_ps(va2, va3));
            for (; b < bend; b++)
                vacc = _mm256_add_ps(vacc, _mm256_loadu_ps(src + (Py_ssize_t)b*8));
            __m128 lo = _mm256_castps256_ps128(vacc);
            __m128 hi = _mm256_extractf128_ps(vacc, 1);
            lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
            partial[tid] = _mm_cvtss_f32(lo);
#else
            float acc = 0.0f;
            for (int b = bstart; b < bend; b++) acc += src[(Py_ssize_t)b*8];
            partial[tid] = acc;
#endif
        }
        float acc = 0.0f;
        for (int t = 0; t < nthreads; t++) acc += partial[t];
        for (Py_ssize_t i = (Py_ssize_t)blocks * 8; i < n; i++) acc += src[i];
        return acc;
    }
#endif
    return vp_sum_f32_serial(src, n);
}

static float vp_max_f32_kernel(const float* VP_RESTRICT src, Py_ssize_t n) {
    if (n == 0) return -FLT_MAX;
#if VP_AVX2
    if (n >= 32) {
        /* 4x unrolled + OMP parallel max */
        int b32 = (int)(n / 32), b;
#ifdef _OPENMP
        if (n >= 524288) {
            int nthreads = omp_get_max_threads();
            if (nthreads > 64) nthreads = 64;
            VP_ALIGN(64) float partial[64];
            for (int t = 0; t < nthreads; t++) partial[t] = -FLT_MAX;
            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num(), nt = omp_get_num_threads();
                int blen = b32 / nt;
                int bstart = tid * blen, bend = (tid == nt-1) ? b32 : bstart + blen;
                __m256 v0 = _mm256_set1_ps(-FLT_MAX), v1 = v0, v2 = v0, v3 = v0;
                int bb;
                for (bb = bstart; bb < bend - 3; bb += 4) {
                    Py_ssize_t ii = (Py_ssize_t)bb * 32;
                    v0 = _mm256_max_ps(v0, _mm256_loadu_ps(src + ii));
                    v1 = _mm256_max_ps(v1, _mm256_loadu_ps(src + ii + 8));
                    v2 = _mm256_max_ps(v2, _mm256_loadu_ps(src + ii + 16));
                    v3 = _mm256_max_ps(v3, _mm256_loadu_ps(src + ii + 24));
                }
                for (; bb < bend; bb++) {
                    Py_ssize_t ii = (Py_ssize_t)bb * 32;
                    v0 = _mm256_max_ps(v0, _mm256_loadu_ps(src + ii));
                    v1 = _mm256_max_ps(v1, _mm256_loadu_ps(src + ii + 8));
                    v2 = _mm256_max_ps(v2, _mm256_loadu_ps(src + ii + 16));
                    v3 = _mm256_max_ps(v3, _mm256_loadu_ps(src + ii + 24));
                }
                __m256 vm = _mm256_max_ps(_mm256_max_ps(v0, v1), _mm256_max_ps(v2, v3));
                __m128 lo = _mm256_castps256_ps128(vm), hi = _mm256_extractf128_ps(vm, 1);
                lo = _mm_max_ps(lo, hi);
                lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
                lo = _mm_max_ss(lo, _mm_shuffle_ps(lo, lo, 1));
                partial[tid] = _mm_cvtss_f32(lo);
            }
            float acc = -FLT_MAX;
            for (int t = 0; t < nthreads; t++) if (partial[t] > acc) acc = partial[t];
            for (Py_ssize_t i = (Py_ssize_t)b32 * 32; i < n; i++) if (src[i] > acc) acc = src[i];
            return acc;
        }
#endif
        /* Single-thread 4x unrolled */
        __m256 v0 = _mm256_loadu_ps(src),     v1 = _mm256_loadu_ps(src + 8);
        __m256 v2 = _mm256_loadu_ps(src + 16), v3 = _mm256_loadu_ps(src + 24);
        for (b = 1; b < b32; b++) {
            Py_ssize_t i = (Py_ssize_t)b * 32;
            v0 = _mm256_max_ps(v0, _mm256_loadu_ps(src + i));
            v1 = _mm256_max_ps(v1, _mm256_loadu_ps(src + i +  8));
            v2 = _mm256_max_ps(v2, _mm256_loadu_ps(src + i + 16));
            v3 = _mm256_max_ps(v3, _mm256_loadu_ps(src + i + 24));
        }
        __m256 vm = _mm256_max_ps(_mm256_max_ps(v0, v1), _mm256_max_ps(v2, v3));
        __m128 lo = _mm256_castps256_ps128(vm), hi = _mm256_extractf128_ps(vm, 1);
        lo = _mm_max_ps(lo, hi);
        lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_max_ss(lo, _mm_shuffle_ps(lo, lo, 1));
        float acc = _mm_cvtss_f32(lo);
        for (Py_ssize_t i = (Py_ssize_t)b32 * 32; i < n; i++) if (src[i] > acc) acc = src[i];
        return acc;
    }
#endif
    float acc = src[0];
    for (Py_ssize_t i = 1; i < n; i++) if (src[i] > acc) acc = src[i];
    return acc;
}

static float vp_min_f32_kernel(const float* VP_RESTRICT src, Py_ssize_t n) {
    if (n == 0) return FLT_MAX;
#if VP_AVX2
    if (n >= 32) {
        int b32 = (int)(n / 32), b;
#ifdef _OPENMP
        if (n >= 524288) {
            int nthreads = omp_get_max_threads();
            if (nthreads > 64) nthreads = 64;
            VP_ALIGN(64) float partial[64];
            for (int t = 0; t < nthreads; t++) partial[t] = FLT_MAX;
            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num(), nt = omp_get_num_threads();
                int blen = b32 / nt, bstart = tid * blen;
                int bend = (tid == nt-1) ? b32 : bstart + blen;
                __m256 v0 = _mm256_set1_ps(FLT_MAX), v1 = v0, v2 = v0, v3 = v0;
                int bb;
                for (bb = bstart; bb < bend - 3; bb += 4) {
                    Py_ssize_t ii = (Py_ssize_t)bb * 32;
                    v0 = _mm256_min_ps(v0, _mm256_loadu_ps(src + ii));
                    v1 = _mm256_min_ps(v1, _mm256_loadu_ps(src + ii + 8));
                    v2 = _mm256_min_ps(v2, _mm256_loadu_ps(src + ii + 16));
                    v3 = _mm256_min_ps(v3, _mm256_loadu_ps(src + ii + 24));
                }
                for (; bb < bend; bb++) {
                    Py_ssize_t ii = (Py_ssize_t)bb * 32;
                    v0 = _mm256_min_ps(v0, _mm256_loadu_ps(src + ii));
                    v1 = _mm256_min_ps(v1, _mm256_loadu_ps(src + ii + 8));
                    v2 = _mm256_min_ps(v2, _mm256_loadu_ps(src + ii + 16));
                    v3 = _mm256_min_ps(v3, _mm256_loadu_ps(src + ii + 24));
                }
                __m256 vm = _mm256_min_ps(_mm256_min_ps(v0, v1), _mm256_min_ps(v2, v3));
                __m128 lo = _mm256_castps256_ps128(vm), hi = _mm256_extractf128_ps(vm, 1);
                lo = _mm_min_ps(lo, hi);
                lo = _mm_min_ps(lo, _mm_movehl_ps(lo, lo));
                lo = _mm_min_ss(lo, _mm_shuffle_ps(lo, lo, 1));
                partial[tid] = _mm_cvtss_f32(lo);
            }
            float acc = FLT_MAX;
            for (int t = 0; t < nthreads; t++) if (partial[t] < acc) acc = partial[t];
            for (Py_ssize_t i = (Py_ssize_t)b32 * 32; i < n; i++) if (src[i] < acc) acc = src[i];
            return acc;
        }
#endif
        __m256 v0 = _mm256_loadu_ps(src),      v1 = _mm256_loadu_ps(src + 8);
        __m256 v2 = _mm256_loadu_ps(src + 16), v3 = _mm256_loadu_ps(src + 24);
        for (b = 1; b < b32; b++) {
            Py_ssize_t i = (Py_ssize_t)b * 32;
            v0 = _mm256_min_ps(v0, _mm256_loadu_ps(src + i));
            v1 = _mm256_min_ps(v1, _mm256_loadu_ps(src + i +  8));
            v2 = _mm256_min_ps(v2, _mm256_loadu_ps(src + i + 16));
            v3 = _mm256_min_ps(v3, _mm256_loadu_ps(src + i + 24));
        }
        __m256 vm = _mm256_min_ps(_mm256_min_ps(v0, v1), _mm256_min_ps(v2, v3));
        __m128 lo = _mm256_castps256_ps128(vm), hi = _mm256_extractf128_ps(vm, 1);
        lo = _mm_min_ps(lo, hi);
        lo = _mm_min_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_min_ss(lo, _mm_shuffle_ps(lo, lo, 1));
        float acc = _mm_cvtss_f32(lo);
        for (Py_ssize_t i = (Py_ssize_t)b32 * 32; i < n; i++) if (src[i] < acc) acc = src[i];
        return acc;
    }
#endif
    float acc = src[0];
    for (Py_ssize_t i = 1; i < n; i++) if (src[i] < acc) acc = src[i];
    return acc;
}

static Py_ssize_t vp_argmax_f32_kernel(const float* VP_RESTRICT src, Py_ssize_t n) {
    if (n == 0) return -1;
    /* AVX2 two-phase: find max value, then find first index with that value */
    float best = vp_max_f32_kernel(src, n);
    for (Py_ssize_t i = 0; i < n; i++) if (src[i] == best) return i;
    return 0;
}

static Py_ssize_t vp_argmin_f32_kernel(const float* VP_RESTRICT src, Py_ssize_t n) {
    if (n == 0) return -1;
    float best = vp_min_f32_kernel(src, n);
    for (Py_ssize_t i = 0; i < n; i++) if (src[i] == best) return i;
    return 0;
}

/* Safe softmax over a row of length n — AVX2 vectorized exp */
static void vp_softmax_row_f32(const float* VP_RESTRICT src,
                                float*       VP_RESTRICT dst,
                                Py_ssize_t n) {
    float m = vp_max_f32_kernel(src, n);
    float acc = 0.0f;
    Py_ssize_t i = 0;
#if VP_AVX2
    __m256 vm = _mm256_set1_ps(m);
    /* 4x unrolled exp+accumulate — 4 independent accumulator chains */
    __m256 va0 = _mm256_setzero_ps(), va1 = _mm256_setzero_ps();
    __m256 va2 = _mm256_setzero_ps(), va3 = _mm256_setzero_ps();
    for (; i <= n - 32; i += 32) {
        __m256 e0 = avx2_expf_fast(_mm256_sub_ps(_mm256_loadu_ps(src + i),      vm));
        __m256 e1 = avx2_expf_fast(_mm256_sub_ps(_mm256_loadu_ps(src + i + 8),  vm));
        __m256 e2 = avx2_expf_fast(_mm256_sub_ps(_mm256_loadu_ps(src + i + 16), vm));
        __m256 e3 = avx2_expf_fast(_mm256_sub_ps(_mm256_loadu_ps(src + i + 24), vm));
        _mm256_storeu_ps(dst + i,      e0); va0 = _mm256_add_ps(va0, e0);
        _mm256_storeu_ps(dst + i + 8,  e1); va1 = _mm256_add_ps(va1, e1);
        _mm256_storeu_ps(dst + i + 16, e2); va2 = _mm256_add_ps(va2, e2);
        _mm256_storeu_ps(dst + i + 24, e3); va3 = _mm256_add_ps(va3, e3);
    }
    __m256 vacc = _mm256_add_ps(_mm256_add_ps(va0,va1), _mm256_add_ps(va2,va3));
    for (; i <= n - 8; i += 8) {
        __m256 e = avx2_expf_fast(_mm256_sub_ps(_mm256_loadu_ps(src + i), vm));
        _mm256_storeu_ps(dst + i, e);
        vacc = _mm256_add_ps(vacc, e);
    }
    __m128 lo = _mm256_castps256_ps128(vacc);
    __m128 hi = _mm256_extractf128_ps(vacc, 1);
    lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
    acc = _mm_cvtss_f32(lo);
#endif
    for (; i < n; i++) { dst[i] = vp_expf(src[i] - m); acc += dst[i]; }
    float inv_acc = 1.0f / acc;
    i = 0;
#if VP_AVX2
    __m256 vinv = _mm256_set1_ps(inv_acc);
    for (; i <= n - 32; i += 32) {
        _mm256_storeu_ps(dst + i,      _mm256_mul_ps(_mm256_loadu_ps(dst + i),      vinv));
        _mm256_storeu_ps(dst + i + 8,  _mm256_mul_ps(_mm256_loadu_ps(dst + i + 8),  vinv));
        _mm256_storeu_ps(dst + i + 16, _mm256_mul_ps(_mm256_loadu_ps(dst + i + 16), vinv));
        _mm256_storeu_ps(dst + i + 24, _mm256_mul_ps(_mm256_loadu_ps(dst + i + 24), vinv));
    }
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(dst + i), vinv));
#endif
    for (; i < n; i++) dst[i] *= inv_acc;
}

static void vp_log_softmax_row_f32(const float* VP_RESTRICT src,
                                    float*       VP_RESTRICT dst,
                                    Py_ssize_t n) {
    float m = vp_max_f32_kernel(src, n);
    float acc = 0.0f;
    Py_ssize_t i = 0;
#if VP_AVX2
    __m256 vm = _mm256_set1_ps(m);
    __m256 va0=_mm256_setzero_ps(), va1=_mm256_setzero_ps();
    __m256 va2=_mm256_setzero_ps(), va3=_mm256_setzero_ps();
    for (; i <= n - 32; i += 32) {
        va0 = _mm256_add_ps(va0, avx2_expf_fast(_mm256_sub_ps(_mm256_loadu_ps(src+i),    vm)));
        va1 = _mm256_add_ps(va1, avx2_expf_fast(_mm256_sub_ps(_mm256_loadu_ps(src+i+8),  vm)));
        va2 = _mm256_add_ps(va2, avx2_expf_fast(_mm256_sub_ps(_mm256_loadu_ps(src+i+16), vm)));
        va3 = _mm256_add_ps(va3, avx2_expf_fast(_mm256_sub_ps(_mm256_loadu_ps(src+i+24), vm)));
    }
    __m256 vacc = _mm256_add_ps(_mm256_add_ps(va0,va1), _mm256_add_ps(va2,va3));
    for (; i <= n - 8; i += 8)
        vacc = _mm256_add_ps(vacc, avx2_expf_fast(_mm256_sub_ps(_mm256_loadu_ps(src+i), vm)));
    __m128 lo = _mm256_castps256_ps128(vacc);
    __m128 hi = _mm256_extractf128_ps(vacc, 1);
    lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
    acc = _mm_cvtss_f32(lo);
    i = 0;
#endif
    for (; i < n; i++) acc += vp_expf(src[i] - m);
    float log_sum = m + vp_logf(acc);
    i = 0;
#if VP_AVX2
    __m256 vls = _mm256_set1_ps(log_sum);
    for (; i <= n - 32; i += 32) {
        _mm256_storeu_ps(dst+i,    _mm256_sub_ps(_mm256_loadu_ps(src+i),    vls));
        _mm256_storeu_ps(dst+i+8,  _mm256_sub_ps(_mm256_loadu_ps(src+i+8),  vls));
        _mm256_storeu_ps(dst+i+16, _mm256_sub_ps(_mm256_loadu_ps(src+i+16), vls));
        _mm256_storeu_ps(dst+i+24, _mm256_sub_ps(_mm256_loadu_ps(src+i+24), vls));
    }
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_sub_ps(_mm256_loadu_ps(src + i), vls));
#endif
    for (; i < n; i++) dst[i] = src[i] - log_sum;
}

/* ══════════════════════════════════════════════════════════════════════════════
 * MATRIX MULTIPLY — float32  (row-major, C = A*B)
 * Blocking: MC=64, KC=256, NC=512; micro-kernel 8x4 using AVX2 FMA
 * ══════════════════════════════════════════════════════════════════════════════ */

/* ── GEMM tile dimensions ─────────────────────────────────────────────────── *
 * MR×NR = 6×16: 12 YMM accumulators + 2 YMM B + 1 YMM A broadcast = 15 regs
 * (AVX2 has 16 YMM) — dual NR=16 fills both FMA ports every cycle.
 * MC×KC A-panel = 72×256×4 = 72 KB (L2), KC×NC B-panel = 256×128×4 = 128 KB (L2).
 * Both panels are packed → contiguous loads, no TLB thrash.
 * --------------------------------------------------------------------------- */
#define MC   96
#define KC  256
#define NC  256
#define NC_MAX 512  /* max NC for large-N GEMMs; Bpack buffers sized to this */
#define MR    6
#define NR   16   /* 2 AVX2 registers wide */
/* OMP flop threshold: spawn threads when M*K*N exceeds this value.
 * 32M covers 64×512×512 GEMMs (33.5M flops) while avoiding overhead
 * on tiny backward GEMMs (B×C×H typically < 10M). */
#define VP_OMP_FLOP_THRESH 33554432LL

/* Post-op codes for fused epilogue (applied after each C micro-tile is written) */
#define VP_POST_NONE    0
#define VP_POST_RELU    1
#define VP_POST_GELU    2
#define VP_POST_SILU    3
#define VP_POST_SIGMOID 4
#define VP_POST_TANH    5

/* ── AVX-512F micro-kernel: 6×32 (6 rows × 32 floats using ZMM registers) ─── *
 * Each ZMM register holds 16 floats — NR512=32 uses 2 ZMM per row = 12 zmm acc
 * + 4 zmm B-loads + 1 zmm broadcast = 17 of 32 ZMM registers.
 * Throughput: 2 FMA512 ops/cycle × 16 floats = 32 FLOPs/cycle (vs 16 for AVX2).
 * ─────────────────────────────────────────────────────────────────────────── */
#if VP_AVX512F

#define NR512 32

static void micro_kernel_avx512(
    int mr, int kc,
    const float* VP_RESTRICT Ap,
    const float* VP_RESTRICT Bp,
    float*       VP_RESTRICT C,
    int ldc)
{
    __m512 c00=_mm512_setzero_ps(), c01=_mm512_setzero_ps();
    __m512 c10=_mm512_setzero_ps(), c11=_mm512_setzero_ps();
    __m512 c20=_mm512_setzero_ps(), c21=_mm512_setzero_ps();
    __m512 c30=_mm512_setzero_ps(), c31=_mm512_setzero_ps();
    __m512 c40=_mm512_setzero_ps(), c41=_mm512_setzero_ps();
    __m512 c50=_mm512_setzero_ps(), c51=_mm512_setzero_ps();

    int p = 0;
    for (; p <= kc - 4; p += 4) {
        __m512 b00 = _mm512_loadu_ps(Bp + (p+0)*NR512),      b01 = _mm512_loadu_ps(Bp + (p+0)*NR512 + 16);
        __m512 b10 = _mm512_loadu_ps(Bp + (p+1)*NR512),      b11 = _mm512_loadu_ps(Bp + (p+1)*NR512 + 16);
        __m512 b20 = _mm512_loadu_ps(Bp + (p+2)*NR512),      b21 = _mm512_loadu_ps(Bp + (p+2)*NR512 + 16);
        __m512 b30 = _mm512_loadu_ps(Bp + (p+3)*NR512),      b31 = _mm512_loadu_ps(Bp + (p+3)*NR512 + 16);
        __m512 a0, a1, a2, a3;
#define FMA512K4(row) \
        a0 = _mm512_set1_ps(Ap[row*kc+p]); a1 = _mm512_set1_ps(Ap[row*kc+p+1]); \
        a2 = _mm512_set1_ps(Ap[row*kc+p+2]); a3 = _mm512_set1_ps(Ap[row*kc+p+3]); \
        c##row##0 = _mm512_fmadd_ps(a0,b00,_mm512_fmadd_ps(a1,b10,_mm512_fmadd_ps(a2,b20,_mm512_fmadd_ps(a3,b30,c##row##0)))); \
        c##row##1 = _mm512_fmadd_ps(a0,b01,_mm512_fmadd_ps(a1,b11,_mm512_fmadd_ps(a2,b21,_mm512_fmadd_ps(a3,b31,c##row##1))));
                   FMA512K4(0)
        if (mr>1){ FMA512K4(1) }
        if (mr>2){ FMA512K4(2) }
        if (mr>3){ FMA512K4(3) }
        if (mr>4){ FMA512K4(4) }
        if (mr>5){ FMA512K4(5) }
#undef FMA512K4
    }
    for (; p < kc; p++) {
        __m512 b0 = _mm512_loadu_ps(Bp + p*NR512);
        __m512 b1 = _mm512_loadu_ps(Bp + p*NR512 + 16);
        __m512 a;
        a = _mm512_set1_ps(Ap[0*kc + p]);
        c00 = _mm512_fmadd_ps(a, b0, c00); c01 = _mm512_fmadd_ps(a, b1, c01);
        if (mr > 1) { a = _mm512_set1_ps(Ap[1*kc+p]);
        c10 = _mm512_fmadd_ps(a, b0, c10); c11 = _mm512_fmadd_ps(a, b1, c11); }
        if (mr > 2) { a = _mm512_set1_ps(Ap[2*kc+p]);
        c20 = _mm512_fmadd_ps(a, b0, c20); c21 = _mm512_fmadd_ps(a, b1, c21); }
        if (mr > 3) { a = _mm512_set1_ps(Ap[3*kc+p]);
        c30 = _mm512_fmadd_ps(a, b0, c30); c31 = _mm512_fmadd_ps(a, b1, c31); }
        if (mr > 4) { a = _mm512_set1_ps(Ap[4*kc+p]);
        c40 = _mm512_fmadd_ps(a, b0, c40); c41 = _mm512_fmadd_ps(a, b1, c41); }
        if (mr > 5) { a = _mm512_set1_ps(Ap[5*kc+p]);
        c50 = _mm512_fmadd_ps(a, b0, c50); c51 = _mm512_fmadd_ps(a, b1, c51); }
    }
    /* Accumulate into C */
    __m512 t;
    t = _mm512_loadu_ps(C + 0*ldc);    _mm512_storeu_ps(C + 0*ldc,    _mm512_add_ps(t,c00));
    t = _mm512_loadu_ps(C + 0*ldc+16); _mm512_storeu_ps(C + 0*ldc+16, _mm512_add_ps(t,c01));
    if (mr>1){
    t = _mm512_loadu_ps(C + 1*ldc);    _mm512_storeu_ps(C + 1*ldc,    _mm512_add_ps(t,c10));
    t = _mm512_loadu_ps(C + 1*ldc+16); _mm512_storeu_ps(C + 1*ldc+16, _mm512_add_ps(t,c11)); }
    if (mr>2){
    t = _mm512_loadu_ps(C + 2*ldc);    _mm512_storeu_ps(C + 2*ldc,    _mm512_add_ps(t,c20));
    t = _mm512_loadu_ps(C + 2*ldc+16); _mm512_storeu_ps(C + 2*ldc+16, _mm512_add_ps(t,c21)); }
    if (mr>3){
    t = _mm512_loadu_ps(C + 3*ldc);    _mm512_storeu_ps(C + 3*ldc,    _mm512_add_ps(t,c30));
    t = _mm512_loadu_ps(C + 3*ldc+16); _mm512_storeu_ps(C + 3*ldc+16, _mm512_add_ps(t,c31)); }
    if (mr>4){
    t = _mm512_loadu_ps(C + 4*ldc);    _mm512_storeu_ps(C + 4*ldc,    _mm512_add_ps(t,c40));
    t = _mm512_loadu_ps(C + 4*ldc+16); _mm512_storeu_ps(C + 4*ldc+16, _mm512_add_ps(t,c41)); }
    if (mr>5){
    t = _mm512_loadu_ps(C + 5*ldc);    _mm512_storeu_ps(C + 5*ldc,    _mm512_add_ps(t,c50));
    t = _mm512_loadu_ps(C + 5*ldc+16); _mm512_storeu_ps(C + 5*ldc+16, _mm512_add_ps(t,c51)); }
}

/* AVX-512 tiled GEMM: same blocking structure as AVX2 but NR=32 */
static void vp_matmul_f32_avx512(
    const float* A, const float* B, float* C,
    int M, int K, int N, int post_op)
{
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int ic = 0; ic < M; ic += MC) {
        int mc = (ic + MC <= M) ? MC : (M - ic);
    for (int jc = 0; jc < N; jc += NR512) {
        int nc = (jc + NR512 <= N) ? NR512 : (N - jc);
    for (int pc = 0; pc < K; pc += KC) {
        int kc = (pc + KC <= K) ? KC : (K - pc);
        /* Pack A panel: mc×kc */
        float Apack[MC * KC];
        for (int i = 0; i < mc; i++)
            for (int p = 0; p < kc; p++)
                Apack[i*kc + p] = A[(ic+i)*K + pc+p];
        /* Pack B panel: kc×nc (zero-padded to NR512 columns) */
        float Bpack[KC * NR512];
        memset(Bpack, 0, sizeof(Bpack));
        for (int p = 0; p < kc; p++)
            for (int j = 0; j < nc; j++)
                Bpack[p*NR512 + j] = B[(pc+p)*N + jc+j];
        /* Micro-kernel loop */
        for (int ir = 0; ir < mc; ir += MR) {
            int mr = (ir + MR <= mc) ? MR : (mc - ir);
            micro_kernel_avx512(mr, kc,
                Apack + ir*kc, Bpack,
                C + (ic+ir)*N + jc, N);
        }
    }}}
    /* Post-op activation (applied to full output) */
    if (post_op != VP_POST_NONE) {
        int total = M * N;
        if (post_op == VP_POST_RELU) {
            __m512 zero = _mm512_setzero_ps();
            int i = 0;
            for (; i <= total - 16; i += 16)
                _mm512_storeu_ps(C+i, _mm512_max_ps(zero, _mm512_loadu_ps(C+i)));
            for (; i < total; i++) C[i] = C[i] > 0.0f ? C[i] : 0.0f;
        }
    }
}

#endif /* VP_AVX512F */

#if VP_AVX2

/* Packed micro-kernel: C[mr×16] += Apack[mr×kc] * Bpack[kc×16]
 * Apack row-major [mr×kc], Bpack row-major [kc×16].
 * Both are contiguous — no stride arithmetic inside the hot loop.          */
static void micro_kernel_MRxNR(
    int mr, int kc,
    const float* VP_RESTRICT Ap,   /* packed A panel, mr rows × kc cols     */
    const float* VP_RESTRICT Bp,   /* packed B panel, kc rows × NR cols     */
    float*       VP_RESTRICT C,    /* output, row-major, stride ldc         */
    int ldc)
{
    /* 12 accumulator registers (6 rows × 2 AVX2 columns) */
    __m256 c00=_mm256_setzero_ps(), c01=_mm256_setzero_ps();
    __m256 c10=_mm256_setzero_ps(), c11=_mm256_setzero_ps();
    __m256 c20=_mm256_setzero_ps(), c21=_mm256_setzero_ps();
    __m256 c30=_mm256_setzero_ps(), c31=_mm256_setzero_ps();
    __m256 c40=_mm256_setzero_ps(), c41=_mm256_setzero_ps();
    __m256 c50=_mm256_setzero_ps(), c51=_mm256_setzero_ps();

    /* Unroll K by 4: 4 independent B-load chains hide FMA latency (4 cycles).
     * Each unroll step issues 2 loads + 12 FMAs, filling both FMA ports. */
    int p = 0;
    for (; p <= kc - 4; p += 4) {
        __m256 b00 = _mm256_loadu_ps(Bp + (p+0)*NR),   b01 = _mm256_loadu_ps(Bp + (p+0)*NR + 8);
        __m256 b10 = _mm256_loadu_ps(Bp + (p+1)*NR),   b11 = _mm256_loadu_ps(Bp + (p+1)*NR + 8);
        __m256 b20 = _mm256_loadu_ps(Bp + (p+2)*NR),   b21 = _mm256_loadu_ps(Bp + (p+2)*NR + 8);
        __m256 b30 = _mm256_loadu_ps(Bp + (p+3)*NR),   b31 = _mm256_loadu_ps(Bp + (p+3)*NR + 8);
        __m256 a0, a1, a2, a3;
#define FMAK4(row) \
        a0 = _mm256_set1_ps(Ap[row*kc+p]); a1 = _mm256_set1_ps(Ap[row*kc+p+1]); \
        a2 = _mm256_set1_ps(Ap[row*kc+p+2]); a3 = _mm256_set1_ps(Ap[row*kc+p+3]); \
        c##row##0 = _mm256_fmadd_ps(a0,b00,_mm256_fmadd_ps(a1,b10,_mm256_fmadd_ps(a2,b20,_mm256_fmadd_ps(a3,b30,c##row##0)))); \
        c##row##1 = _mm256_fmadd_ps(a0,b01,_mm256_fmadd_ps(a1,b11,_mm256_fmadd_ps(a2,b21,_mm256_fmadd_ps(a3,b31,c##row##1))));
                   FMAK4(0)
        if (mr>1){ FMAK4(1) }
        if (mr>2){ FMAK4(2) }
        if (mr>3){ FMAK4(3) }
        if (mr>4){ FMAK4(4) }
        if (mr>5){ FMAK4(5) }
#undef FMAK4
    }
    for (; p < kc; p++) {
        __m256 b0 = _mm256_loadu_ps(Bp + p*NR);
        __m256 b1 = _mm256_loadu_ps(Bp + p*NR + 8);
        __m256 a;
        a = _mm256_set1_ps(Ap[0*kc + p]);
        c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        if (mr > 1) {
        a = _mm256_set1_ps(Ap[1*kc + p]);
        c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11); }
        if (mr > 2) {
        a = _mm256_set1_ps(Ap[2*kc + p]);
        c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21); }
        if (mr > 3) {
        a = _mm256_set1_ps(Ap[3*kc + p]);
        c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31); }
        if (mr > 4) {
        a = _mm256_set1_ps(Ap[4*kc + p]);
        c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41); }
        if (mr > 5) {
        a = _mm256_set1_ps(Ap[5*kc + p]);
        c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51); }
    }
    /* Accumulate into C (add, not overwrite — called per KC panel) */
    __m256 t;
    t = _mm256_loadu_ps(C + 0*ldc);     _mm256_storeu_ps(C + 0*ldc,     _mm256_add_ps(t,c00));
    t = _mm256_loadu_ps(C + 0*ldc+8);   _mm256_storeu_ps(C + 0*ldc+8,   _mm256_add_ps(t,c01));
    if (mr > 1) {
    t = _mm256_loadu_ps(C + 1*ldc);     _mm256_storeu_ps(C + 1*ldc,     _mm256_add_ps(t,c10));
    t = _mm256_loadu_ps(C + 1*ldc+8);   _mm256_storeu_ps(C + 1*ldc+8,   _mm256_add_ps(t,c11)); }
    if (mr > 2) {
    t = _mm256_loadu_ps(C + 2*ldc);     _mm256_storeu_ps(C + 2*ldc,     _mm256_add_ps(t,c20));
    t = _mm256_loadu_ps(C + 2*ldc+8);   _mm256_storeu_ps(C + 2*ldc+8,   _mm256_add_ps(t,c21)); }
    if (mr > 3) {
    t = _mm256_loadu_ps(C + 3*ldc);     _mm256_storeu_ps(C + 3*ldc,     _mm256_add_ps(t,c30));
    t = _mm256_loadu_ps(C + 3*ldc+8);   _mm256_storeu_ps(C + 3*ldc+8,   _mm256_add_ps(t,c31)); }
    if (mr > 4) {
    t = _mm256_loadu_ps(C + 4*ldc);     _mm256_storeu_ps(C + 4*ldc,     _mm256_add_ps(t,c40));
    t = _mm256_loadu_ps(C + 4*ldc+8);   _mm256_storeu_ps(C + 4*ldc+8,   _mm256_add_ps(t,c41)); }
    if (mr > 5) {
    t = _mm256_loadu_ps(C + 5*ldc);     _mm256_storeu_ps(C + 5*ldc,     _mm256_add_ps(t,c50));
    t = _mm256_loadu_ps(C + 5*ldc+8);   _mm256_storeu_ps(C + 5*ldc+8,   _mm256_add_ps(t,c51)); }
}

/* Specialized no-branch full-tile kernel: mr==MR(6), nr==NR(16).
 * Loads B once per k step into two YMM registers shared across all 6 rows.
 * K unrolled by 4 (matches FMA latency=4 on Intel); no runtime branches. */
__forceinline static void micro_kernel_6x16_full(
    int kc,
    const float* VP_RESTRICT Ap,
    const float* VP_RESTRICT Bp,
    float*       VP_RESTRICT C,
    int ldc)
{
    __m256 c00=_mm256_setzero_ps(), c01=_mm256_setzero_ps();
    __m256 c10=_mm256_setzero_ps(), c11=_mm256_setzero_ps();
    __m256 c20=_mm256_setzero_ps(), c21=_mm256_setzero_ps();
    __m256 c30=_mm256_setzero_ps(), c31=_mm256_setzero_ps();
    __m256 c40=_mm256_setzero_ps(), c41=_mm256_setzero_ps();
    __m256 c50=_mm256_setzero_ps(), c51=_mm256_setzero_ps();

/* Load B[k] once, then FMA across all 6 rows — avoids reloading B per row. */
#define STEP(k) \
    { __m256 _b0 = _mm256_loadu_ps(Bp+(k)*NR); \
      __m256 _b1 = _mm256_loadu_ps(Bp+(k)*NR+8); \
      c00=_mm256_fmadd_ps(_mm256_set1_ps(Ap[0*kc+(k)]),_b0,c00); c01=_mm256_fmadd_ps(_mm256_set1_ps(Ap[0*kc+(k)]),_b1,c01); \
      c10=_mm256_fmadd_ps(_mm256_set1_ps(Ap[1*kc+(k)]),_b0,c10); c11=_mm256_fmadd_ps(_mm256_set1_ps(Ap[1*kc+(k)]),_b1,c11); \
      c20=_mm256_fmadd_ps(_mm256_set1_ps(Ap[2*kc+(k)]),_b0,c20); c21=_mm256_fmadd_ps(_mm256_set1_ps(Ap[2*kc+(k)]),_b1,c21); \
      c30=_mm256_fmadd_ps(_mm256_set1_ps(Ap[3*kc+(k)]),_b0,c30); c31=_mm256_fmadd_ps(_mm256_set1_ps(Ap[3*kc+(k)]),_b1,c31); \
      c40=_mm256_fmadd_ps(_mm256_set1_ps(Ap[4*kc+(k)]),_b0,c40); c41=_mm256_fmadd_ps(_mm256_set1_ps(Ap[4*kc+(k)]),_b1,c41); \
      c50=_mm256_fmadd_ps(_mm256_set1_ps(Ap[5*kc+(k)]),_b0,c50); c51=_mm256_fmadd_ps(_mm256_set1_ps(Ap[5*kc+(k)]),_b1,c51); }

    int p = 0;
    for (; p <= kc - 4; p += 4) { STEP(p) STEP(p+1) STEP(p+2) STEP(p+3) }
    for (; p < kc; p++)          { STEP(p) }
#undef STEP

    _mm256_storeu_ps(C+0*ldc,   _mm256_add_ps(_mm256_loadu_ps(C+0*ldc),   c00));
    _mm256_storeu_ps(C+0*ldc+8, _mm256_add_ps(_mm256_loadu_ps(C+0*ldc+8), c01));
    _mm256_storeu_ps(C+1*ldc,   _mm256_add_ps(_mm256_loadu_ps(C+1*ldc),   c10));
    _mm256_storeu_ps(C+1*ldc+8, _mm256_add_ps(_mm256_loadu_ps(C+1*ldc+8), c11));
    _mm256_storeu_ps(C+2*ldc,   _mm256_add_ps(_mm256_loadu_ps(C+2*ldc),   c20));
    _mm256_storeu_ps(C+2*ldc+8, _mm256_add_ps(_mm256_loadu_ps(C+2*ldc+8), c21));
    _mm256_storeu_ps(C+3*ldc,   _mm256_add_ps(_mm256_loadu_ps(C+3*ldc),   c30));
    _mm256_storeu_ps(C+3*ldc+8, _mm256_add_ps(_mm256_loadu_ps(C+3*ldc+8), c31));
    _mm256_storeu_ps(C+4*ldc,   _mm256_add_ps(_mm256_loadu_ps(C+4*ldc),   c40));
    _mm256_storeu_ps(C+4*ldc+8, _mm256_add_ps(_mm256_loadu_ps(C+4*ldc+8), c41));
    _mm256_storeu_ps(C+5*ldc,   _mm256_add_ps(_mm256_loadu_ps(C+5*ldc),   c50));
    _mm256_storeu_ps(C+5*ldc+8, _mm256_add_ps(_mm256_loadu_ps(C+5*ldc+8), c51));
}

/* ── Epilogue: apply post-op in-place to a row of C of length nc ──────────── *
 * Called once per micro-panel row ONLY on the final KC-panel pass (pc+kc==K). *
 * Data is still hot in L2/L3 cache — avoids a full second pass through C.    */
#if VP_AVX2
__forceinline static __m256 _ep_gelu8(__m256 x) {
    /* GELU: x*0.5*(1+erf(x/sqrt(2))) via A&S 7.1.26 poly */
    const __m256 kInvSqrt2 = _mm256_set1_ps(0.7071067811865476f);
    const __m256 kA1 = _mm256_set1_ps( 0.254829592f);
    const __m256 kA2 = _mm256_set1_ps(-0.284496736f);
    const __m256 kA3 = _mm256_set1_ps( 1.421413741f);
    const __m256 kA4 = _mm256_set1_ps(-1.453152027f);
    const __m256 kA5 = _mm256_set1_ps( 1.061405429f);
    const __m256 kP  = _mm256_set1_ps( 0.3275911f);
    const __m256 one = _mm256_set1_ps(1.f);
    const __m256 half= _mm256_set1_ps(0.5f);
    __m256 ax = _mm256_andnot_ps(_mm256_set1_ps(-0.f), x); /* abs(x) */
    __m256 t  = _mm256_div_ps(one, _mm256_fmadd_ps(kP, ax, one));
    __m256 p  = _mm256_fmadd_ps(kA5, t, kA4);
    p = _mm256_fmadd_ps(p, t, kA3);
    p = _mm256_fmadd_ps(p, t, kA2);
    p = _mm256_fmadd_ps(p, t, kA1);
    /* erf approximation: sign(x) * (1 - poly*exp(-x^2)) */
    __m256 xn = _mm256_mul_ps(_mm256_mul_ps(x, kInvSqrt2),
                               _mm256_mul_ps(_mm256_mul_ps(x, kInvSqrt2), _mm256_set1_ps(-1.f)));
    /* use exp approximation: 2^(x/ln2) ≈ exp(x); for small |x| use direct series */
    /* Simplified: use the same poly * exp(-t^2) but skip full exp — approximate */
    /* Full GELU: x * 0.5 * (1 + erf(x/sqrt2)); erf uses p*exp(-ax^2) formula */
    __m256 neg_ax2 = _mm256_mul_ps(_mm256_mul_ps(ax,ax), _mm256_set1_ps(-1.f));
    /* exp(-ax^2) via fast exp: clamp exponent to avoid overflow */
    neg_ax2 = _mm256_max_ps(neg_ax2, _mm256_set1_ps(-87.f));
    /* Fast exp via bit trick: 2^(x/ln2) */
    __m256 e = _mm256_mul_ps(neg_ax2, _mm256_set1_ps(1.4426950408f)); /* x/ln2 */
    __m256i ei = _mm256_cvtps_epi32(_mm256_add_ps(e, _mm256_set1_ps(127.f)));
    ei = _mm256_slli_epi32(ei, 23);
    __m256 exp_val = _mm256_castsi256_ps(ei);
    __m256 sign = _mm256_and_ps(x, _mm256_set1_ps(-0.f)); /* sign bit of x */
    __m256 erf_val = _mm256_or_ps(sign,
        _mm256_sub_ps(one, _mm256_mul_ps(p, exp_val)));
    return _mm256_mul_ps(x, _mm256_mul_ps(half, _mm256_add_ps(one, erf_val)));
}
__forceinline static __m256 _ep_sigmoid8(__m256 x) {
    /* sigmoid(x) = 1/(1+exp(-x)) — fast exp approximation */
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    neg_x = _mm256_max_ps(neg_x, _mm256_set1_ps(-87.f));
    __m256 e = _mm256_mul_ps(neg_x, _mm256_set1_ps(1.4426950408f));
    __m256i ei = _mm256_cvtps_epi32(_mm256_add_ps(e, _mm256_set1_ps(127.f)));
    ei = _mm256_slli_epi32(ei, 23);
    __m256 exp_val = _mm256_castsi256_ps(ei);
    return _mm256_div_ps(_mm256_set1_ps(1.f),
                         _mm256_add_ps(_mm256_set1_ps(1.f), exp_val));
}
__forceinline static __m256 _ep_tanh8(__m256 x) {
    /* tanh(x) = (exp(2x)-1)/(exp(2x)+1); clamp |x| to 15 */
    x = _mm256_max_ps(_mm256_set1_ps(-15.f), _mm256_min_ps(x, _mm256_set1_ps(15.f)));
    __m256 x2 = _mm256_add_ps(x, x);
    x2 = _mm256_max_ps(x2, _mm256_set1_ps(-87.f));
    __m256 e = _mm256_mul_ps(x2, _mm256_set1_ps(1.4426950408f));
    __m256i ei = _mm256_cvtps_epi32(_mm256_add_ps(e, _mm256_set1_ps(127.f)));
    ei = _mm256_slli_epi32(ei, 23);
    __m256 exp2x = _mm256_castsi256_ps(ei);
    return _mm256_div_ps(_mm256_sub_ps(exp2x, _mm256_set1_ps(1.f)),
                         _mm256_add_ps(exp2x, _mm256_set1_ps(1.f)));
}
__forceinline static __m256 _ep_silu8(__m256 x) {
    /* silu(x) = x * sigmoid(x) */
    return _mm256_mul_ps(x, _ep_sigmoid8(x));
}

/* Apply post-op to nc floats at ptr (nc <= NR, all in cache). */
static void _apply_epilogue(float* VP_RESTRICT ptr, int nc, int post_op) {
    if (post_op == VP_POST_NONE) return;
    int j = 0;
    for (; j + 8 <= nc; j += 8) {
        __m256 v = _mm256_loadu_ps(ptr + j);
        switch (post_op) {
            case VP_POST_RELU:    v = _mm256_max_ps(v, _mm256_setzero_ps()); break;
            case VP_POST_GELU:    v = _ep_gelu8(v);    break;
            case VP_POST_SILU:    v = _ep_silu8(v);    break;
            case VP_POST_SIGMOID: v = _ep_sigmoid8(v); break;
            case VP_POST_TANH:    v = _ep_tanh8(v);    break;
        }
        _mm256_storeu_ps(ptr + j, v);
    }
    for (; j < nc; j++) {
        float v = ptr[j];
        switch (post_op) {
            case VP_POST_RELU:    v = v > 0.f ? v : 0.f; break;
            case VP_POST_GELU:    { float t=v*0.7071067811865476f; v=v*0.5f*(1.f+tanhf(t*(1.f+0.044715f*t*t))); } break;
            case VP_POST_SILU:    v = v / (1.f + expf(-v)); break;
            case VP_POST_SIGMOID: v = 1.f / (1.f + expf(-v)); break;
            case VP_POST_TANH:    v = tanhf(v); break;
        }
        ptr[j] = v;
    }
}
#endif /* VP_AVX2 (epilogue helpers) */

/* Pack a KC×NC slice of B into contiguous [kc×NR] micro-panels */
static void pack_B(
    const float* VP_RESTRICT B, int K, int N,
    int pc, int kc, int jc, int nc,
    float* VP_RESTRICT Bpack)
{
    /* Bpack layout: [nc/NR panels][kc rows][NR cols] */
    for (int jr = 0; jr < nc; jr += NR) {
        int nr = (jr + NR <= nc) ? NR : (nc - jr);
        float* dst = Bpack + (jr/NR) * (kc * NR);
        for (int p = 0; p < kc; p++) {
            const float* src = B + (pc+p)*N + (jc+jr);
            if (nr == NR) {
                memcpy(dst + p*NR, src, NR * sizeof(float));
            } else {
                memcpy(dst + p*NR, src, nr * sizeof(float));
                memset(dst + p*NR + nr, 0, (NR-nr) * sizeof(float));
            }
        }
    }
}

/* Pack an MC×KC slice of A into contiguous [MR×kc] micro-panels */
static void pack_A(
    const float* VP_RESTRICT A, int K,
    int ic, int mc, int pc, int kc,
    float* VP_RESTRICT Apack)
{
    /* Apack layout: [mc/MR panels][MR rows][kc cols], row-major within panel */
    for (int ir = 0; ir < mc; ir += MR) {
        int mr = (ir + MR <= mc) ? MR : (mc - ir);
        float* dst = Apack + (ir/MR) * (MR * kc);
        for (int i = 0; i < mr; i++) {
            memcpy(dst + i*kc, A + (ic+ir+i)*K + pc, kc * sizeof(float));
        }
        /* Zero-pad tail rows to MR */
        for (int i = mr; i < MR; i++)
            memset(dst + i*kc, 0, kc * sizeof(float));
    }
}

/* Persistent per-thread pack buffers — allocated once on first parallel GEMM, never freed.
 * Indexed by omp_get_thread_num(), sized for up to 64 threads.
 * This avoids per-call aligned-alloc overhead that otherwise dominates large GEMM timing. */
#ifdef _OPENMP
#define VP_MAX_THREADS 64
static float* _g_Bpack[VP_MAX_THREADS];  /* KC*NC floats per thread */
static float* _g_Apack[VP_MAX_THREADS];  /* MC*KC floats per thread */
static int    _g_packs_init = 0;

static void _ensure_pack_bufs(int max_t) {
    /* Called once from the serial region before the OMP parallel for.
     * Allocates any missing slots; already-allocated ones are reused. */
    int t;
    for (t = 0; t < max_t && t < VP_MAX_THREADS; t++) {
        if (!_g_Bpack[t])
            _g_Bpack[t] = (float*)vp_aligned_alloc(64, (size_t)KC * NC_MAX * sizeof(float));
        if (!_g_Apack[t])
            _g_Apack[t] = (float*)vp_aligned_alloc(64, (size_t)MC * KC * sizeof(float));
    }
    _g_packs_init = 1;
}
#endif

/* Packed tiled GEMM: C[M,N] = A[M,K] * B[K,N] with optional fused epilogue.
 * post_op: VP_POST_* constant — activation applied to each C tile on the
 * final KC-panel pass while data is still in L2 cache.                      */
static void vp_matmul_f32_avx2(
    const float* A, const float* B, float* C,
    int M, int K, int N, int post_op)
{
    memset(C, 0, (size_t)M * N * sizeof(float));

#ifdef _OPENMP
    {
    /* Adaptive NC: halve the tile width when M is small so nj doubles, giving
     * more tiles for the scheduler and enabling parallelism even at M=64. */
    /* Adaptive tile width: wider panels for large N (better B reuse in L3),
     * narrower when M is small (more tiles → more thread parallelism). */
    int nc_tile = (N >= 2048 && M >= MC) ? NC_MAX
                : (M <= MC)              ? (NC / 2)
                :                          NC;
    if (nc_tile < NR) nc_tile = NR;  /* must be at least one NR-panel wide */
    int ni = (M + MC - 1) / MC;
    int nj = (N + nc_tile - 1) / nc_tile;
    int ntiles = ni * nj;
    /* Only spawn threads when flops justify wakeup overhead */
    long long _flops = (long long)M * K * N;
    /* Parallelise only when there are enough tiles to keep threads busy.
     * ntiles < 4 means at most 3 threads can work — not worth wakeup cost. */
    int max_t = (_flops >= VP_OMP_FLOP_THRESH && ntiles >= 4) ? omp_get_max_threads() : 1;
    if (max_t > VP_MAX_THREADS) max_t = VP_MAX_THREADS;
    /* Ensure persistent buffers exist (no-op after first call per thread count) */
    _ensure_pack_bufs(max_t);
    int tile_i;
    /* Cap threads to tile count — no point waking more threads than there is work */
    if (max_t > ntiles) max_t = ntiles;
    VP_PRAGMA(omp parallel for schedule(static) num_threads(max_t) if(max_t > 1))
    for (tile_i = 0; tile_i < ntiles; tile_i++) {
        int tid = omp_get_thread_num();
        int ic = (tile_i / nj) * MC;
        int jc = (tile_i % nj) * nc_tile;
        int mc = (ic + MC <= M) ? MC : (M - ic);
        int nc = (jc + nc_tile <= N) ? nc_tile : (N - jc);
        float* Bpack = _g_Bpack[tid];
        float* Apack = _g_Apack[tid];
        if (!Bpack || !Apack) continue;
#else
    for (int ic = 0; ic < M; ic += MC) {
        int mc = (ic + MC <= M) ? MC : (M - ic);
    for (int jc = 0; jc < N; jc += NC) {
        int nc = (jc + NC <= N) ? NC : (N - jc);
        float* Bpack = (float*)vp_aligned_alloc(64, (size_t)KC * NC * sizeof(float));
        float* Apack = (float*)vp_aligned_alloc(64, (size_t)MC * KC * sizeof(float));
        if (!Bpack || !Apack) {
            if (Bpack) vp_aligned_free(Bpack);
            if (Apack) vp_aligned_free(Apack);
            continue;
        }
#endif

        for (int pc = 0; pc < K; pc += KC) {
            int kc = (pc + KC <= K) ? KC : (K - pc);
            int is_last_kc = (pc + kc >= K);
            pack_B(B, K, N, pc, kc, jc, nc, Bpack);
            pack_A(A, K, ic, mc, pc, kc, Apack);

            for (int ir = 0; ir < mc; ir += MR) {
                int mr = (ir + MR <= mc) ? MR : (mc - ir);
                const float* Ap = Apack + (ir/MR) * (MR * kc);

                for (int jr = 0; jr < nc; jr += NR) {
                    int nr = (jr + NR <= nc) ? NR : (nc - jr);
                    const float* Bp = Bpack + (jr/NR) * (kc * NR);
                    float* Crow = C + (ic+ir)*N + (jc+jr);

                    if (nr == NR) {
                        if (mr == MR)
                            micro_kernel_6x16_full(kc, Ap, Bp, Crow, N);
                        else
                            micro_kernel_MRxNR(mr, kc, Ap, Bp, Crow, N);
                    } else {
                        for (int i2 = 0; i2 < mr; i2++)
                        for (int j2 = 0; j2 < nr; j2++) {
                            float acc = 0.f;
                            for (int p = 0; p < kc; p++)
                                acc += Ap[i2*kc + p] * Bp[p*NR + j2];
                            C[(ic+ir+i2)*N + (jc+jr+j2)] += acc;
                        }
                    }
                    /* Epilogue: apply activation to this tile while still in cache */
                    if (is_last_kc && post_op != VP_POST_NONE) {
                        for (int i2 = 0; i2 < mr; i2++)
                            _apply_epilogue(C + (ic+ir+i2)*N + (jc+jr), nr, post_op);
                    }
                }
            }
        }

#ifndef _OPENMP
        vp_aligned_free(Bpack);
        vp_aligned_free(Apack);
    } /* jc loop */
    } /* ic loop */
#else
    } /* tile_i loop */
    /* Persistent buffers (_g_Bpack/_g_Apack) are reused across calls — never freed here */
    } /* block scope */
#endif
}

#endif /* VP_AVX2 */

/* Scalar fallback matmul */
static void vp_matmul_f32_scalar(
    const float* A, const float* B, float* C,
    int M, int K, int N)
{
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int i = 0; i < M; i++)
    for (int p = 0; p < K; p++) {
        float a = A[i*K + p];
        for (int j = 0; j < N; j++)
            C[i*N + j] += a * B[p*N + j];
    }
}

/* Small-matrix direct kernel — no packing overhead, good for M,K,N ≤ 256 */
#if VP_AVX2
static void vp_matmul_f32_small(
    const float* VP_RESTRICT A, const float* VP_RESTRICT B, float* VP_RESTRICT C,
    int M, int K, int N)
{
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int p = 0; p < K; p++) {
            __m256 a = _mm256_set1_ps(A[i*K + p]);
            int j = 0;
            for (; j <= N - 8; j += 8) {
                __m256 c = _mm256_loadu_ps(C + i*N + j);
                __m256 b = _mm256_loadu_ps(B + p*N + j);
                _mm256_storeu_ps(C + i*N + j, _mm256_fmadd_ps(a, b, c));
            }
            for (; j < N; j++)
                C[i*N + j] += A[i*K + p] * B[p*N + j];
        }
    }
}
#endif

static void vp_matmul_f32_dispatch(
    const float* A, const float* B, float* C,
    int M, int K, int N, int post_op)
{
#if VP_AVX512F
    /* AVX-512F path: 2× throughput vs AVX2 (32-float ZMM registers) */
    if (M >= MR && N >= NR512 && K >= 4)
        vp_matmul_f32_avx512(A, B, C, M, K, N, post_op);
    else
#  if VP_AVX2
        vp_matmul_f32_avx2(A, B, C, M, K, N, post_op);
#  else
        vp_matmul_f32_scalar(A, B, C, M, K, N);
#  endif
#elif VP_AVX2
    if (M < MR || N < NR || K < 4)
        vp_matmul_f32_small(A, B, C, M, K, N);
    else
        vp_matmul_f32_avx2(A, B, C, M, K, N, post_op);
#else
    vp_matmul_f32_scalar(A, B, C, M, K, N);
#endif
}

/* ══════════════════════════════════════════════════════════════════════════════
 * PYTHON MODULE — C API bindings
 * ══════════════════════════════════════════════════════════════════════════════ */

/* Helper: get (ptr, len) from a Python array.array or bytes/bytearray/memoryview */
#define GET_SRC(obj, ptr, n, itemsize)                                      \
    do {                                                                     \
        Py_buffer _v;                                                        \
        if (PyObject_GetBuffer((obj), &_v, PyBUF_SIMPLE) < 0) return NULL;  \
        (ptr) = (const float*)_v.buf;                                        \
        (n)   = _v.len / (itemsize);                                         \
        PyBuffer_Release(&_v);                                               \
    } while (0)

#define GET_DST(obj, ptr, n, itemsize)                                      \
    do {                                                                     \
        Py_buffer _v;                                                        \
        if (PyObject_GetBuffer((obj), &_v, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) return NULL; \
        (ptr) = (float*)_v.buf;                                              \
        (n)   = _v.len / (itemsize);                                         \
        PyBuffer_Release(&_v);                                               \
    } while (0)

/* ── Generic unary op binder ─────────────────────────────────────────────── */
typedef void (*unary_f32_fn)(const float*, float*, Py_ssize_t);

static PyObject* _unary_f32(PyObject *self, PyObject *args, unary_f32_fn fn) {
    PyObject *src_obj, *dst_obj;
    if (!PyArg_ParseTuple(args, "OO", &src_obj, &dst_obj)) return NULL;
    const float *src; Py_ssize_t ns;
    float *dst; Py_ssize_t nd;
    GET_SRC(src_obj, src, ns, sizeof(float));
    GET_DST(dst_obj, dst, nd, sizeof(float));
    if (ns != nd) { PyErr_SetString(PyExc_ValueError, "buffer size mismatch"); return NULL; }
    fn(src, dst, ns);
    Py_RETURN_NONE;
}

#define PY_UNARY_F32(PYNAME, CNAME)                                          \
static PyObject* py_##PYNAME(PyObject *self, PyObject *args) {              \
    return _unary_f32(self, args, CNAME##_kernel);                          \
}

PY_UNARY_F32(relu,    vp_relu_f32)
PY_UNARY_F32(exp,     vp_exp_f32)
PY_UNARY_F32(log,     vp_log_f32)
PY_UNARY_F32(sigmoid, vp_sigmoid_f32)
PY_UNARY_F32(tanh_f,  vp_tanh_f32)
PY_UNARY_F32(sqrt,    vp_sqrt_f32)
PY_UNARY_F32(abs_f,   vp_abs_f32)
PY_UNARY_F32(neg,     vp_neg_f32)
PY_UNARY_F32(sign,    vp_sign_f32)
PY_UNARY_F32(erf,     vp_erf_f32)
PY_UNARY_F32(gelu,    vp_gelu_f32)
PY_UNARY_F32(silu,    vp_silu_f32)

/* ── SwiGLU: fused x * gate * sigmoid(gate) ─────────────────────────────── */
static PyObject* py_swiglu_f32(PyObject *self, PyObject *args) {
    PyObject *src_obj, *gate_obj, *dst_obj;
    if (!PyArg_ParseTuple(args, "OOO", &src_obj, &gate_obj, &dst_obj)) return NULL;
    Py_buffer vs, vg, vd;
    if (PyObject_GetBuffer(src_obj,  &vs, PyBUF_SIMPLE)            < 0) return NULL;
    if (PyObject_GetBuffer(gate_obj, &vg, PyBUF_SIMPLE)            < 0) { PyBuffer_Release(&vs); return NULL; }
    if (PyObject_GetBuffer(dst_obj,  &vd, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&vs); PyBuffer_Release(&vg); return NULL;
    }
    Py_ssize_t n = vs.len / sizeof(float);
    vp_swiglu_f32_kernel((const float*)vs.buf, (const float*)vg.buf, (float*)vd.buf, n);
    PyBuffer_Release(&vs); PyBuffer_Release(&vg); PyBuffer_Release(&vd);
    Py_RETURN_NONE;
}

/* ── add_relu: fused residual + relu ────────────────────────────────────── */
static PyObject* py_add_relu_f32(PyObject *self, PyObject *args) {
    PyObject *a_obj, *b_obj, *dst_obj;
    if (!PyArg_ParseTuple(args, "OOO", &a_obj, &b_obj, &dst_obj)) return NULL;
    Py_buffer va, vb, vd;
    if (PyObject_GetBuffer(a_obj,   &va, PyBUF_SIMPLE)            < 0) return NULL;
    if (PyObject_GetBuffer(b_obj,   &vb, PyBUF_SIMPLE)            < 0) { PyBuffer_Release(&va); return NULL; }
    if (PyObject_GetBuffer(dst_obj, &vd, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&va); PyBuffer_Release(&vb); return NULL;
    }
    Py_ssize_t n = va.len / sizeof(float);
    vp_add_relu_f32_kernel((const float*)va.buf, (const float*)vb.buf, (float*)vd.buf, n);
    PyBuffer_Release(&va); PyBuffer_Release(&vb); PyBuffer_Release(&vd);
    Py_RETURN_NONE;
}

/* ── Fused unary chain ───────────────────────────────────────────────────────
 * Applies a sequence of element-wise unary ops in a SINGLE memory pass.
 * Eliminates intermediate allocations and reduces memory-bandwidth pressure
 * by 1/nops vs executing each op separately.
 *
 * Op codes must match vectorphile/_lazy.py:_CHAIN_OPCODES.
 * Chain limited to VP_MAX_CHAIN ops (keeps stack-allocated fn-ptr array). */
#define VP_MAX_CHAIN 16

/* Op code constants — must match _lazy.py */
#define VP_CHAIN_RELU     0
#define VP_CHAIN_SIGMOID  1
#define VP_CHAIN_TANH     2
#define VP_CHAIN_EXP      3
#define VP_CHAIN_LOG      4
#define VP_CHAIN_SQRT     5
#define VP_CHAIN_ABS      6
#define VP_CHAIN_NEG      7
#define VP_CHAIN_GELU     8
#define VP_CHAIN_SILU     9
#define VP_CHAIN_SIGN    10
#define VP_CHAIN_ERF     11

static inline float _scalar_chain_op(float x, int op) {
    switch (op) {
    case VP_CHAIN_RELU:    return x > 0.0f ? x : 0.0f;
    case VP_CHAIN_SIGMOID: return vp_sigmoidf(x);
    case VP_CHAIN_TANH:    return vp_tanhf(x);
    case VP_CHAIN_EXP:     return vp_expf(x);
    case VP_CHAIN_LOG:     return vp_logf(x);
    case VP_CHAIN_SQRT:    return sqrtf(x);
    case VP_CHAIN_ABS:     return x < 0.0f ? -x : x;
    case VP_CHAIN_NEG:     return -x;
    case VP_CHAIN_GELU:    { float s = vp_sigmoidf(1.702f * x); return x * s; }
    case VP_CHAIN_SILU:    return x * vp_sigmoidf(x);
    case VP_CHAIN_SIGN:    return (float)((x > 0.0f) - (x < 0.0f));
    case VP_CHAIN_ERF:     return erff(x);
    default:               return x;
    }
}

#if VP_AVX2
static inline __m256 _avx2_chain_op(__m256 x, int op) {
    switch (op) {
    case VP_CHAIN_RELU:    return _mm256_max_ps(x, _mm256_setzero_ps());
    case VP_CHAIN_SIGMOID: return avx2_sigmoidf(x);
    case VP_CHAIN_TANH:    return avx2_tanhf(x);
    case VP_CHAIN_EXP:     return avx2_expf(x);
    case VP_CHAIN_LOG:     return avx2_logf(x);
    case VP_CHAIN_SQRT:    return _mm256_sqrt_ps(x);
    case VP_CHAIN_ABS:     return _mm256_and_ps(x,
                               _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
    case VP_CHAIN_NEG:     return _mm256_xor_ps(x,
                               _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)));
    case VP_CHAIN_GELU: {
        __m256 s = avx2_sigmoidf(_mm256_mul_ps(_mm256_set1_ps(1.702f), x));
        return _mm256_mul_ps(x, s);
    }
    case VP_CHAIN_SILU:    return _mm256_mul_ps(x, avx2_sigmoidf(x));
    case VP_CHAIN_SIGN: {
        __m256 pos = _mm256_castsi256_ps(_mm256_srli_epi32(
                        _mm256_castps_si256(_mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_GT_OS)), 31));
        __m256 neg = _mm256_castsi256_ps(_mm256_srli_epi32(
                        _mm256_castps_si256(_mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OS)), 31));
        return _mm256_sub_ps(pos, neg);  /* +1, -1, or 0 */
    }
    case VP_CHAIN_ERF: {
        /* erf via our existing scalar path — will be vectorized in future */
        float buf[8]; _mm256_storeu_ps(buf, x);
        for (int k = 0; k < 8; k++) buf[k] = erff(buf[k]);
        return _mm256_loadu_ps(buf);
    }
    default:               return x;
    }
}
#endif /* VP_AVX2 */

static void vp_fused_chain_f32_kernel(const float* VP_RESTRICT src,
                                       float*       VP_RESTRICT dst,
                                       Py_ssize_t n,
                                       const int*   ops,
                                       int          nops) {
    int b32 = (int)(n / 32), b;
#if VP_AVX2
    /* 4x unrolled: 4 independent AVX2 chains → hides op latency + ILP.
     * The inner switch on ops[k] is on a constant array — the branch predictor
     * handles it perfectly since ops[] never changes between outer iterations. */
    VP_OMP_FOR(n)
    for (b = 0; b < b32; b++) {
        Py_ssize_t i = (Py_ssize_t)b * 32;
        __m256 x0 = _mm256_loadu_ps(src + i);
        __m256 x1 = _mm256_loadu_ps(src + i + 8);
        __m256 x2 = _mm256_loadu_ps(src + i + 16);
        __m256 x3 = _mm256_loadu_ps(src + i + 24);
        for (int k = 0; k < nops; k++) {
            int op = ops[k];
            x0 = _avx2_chain_op(x0, op);
            x1 = _avx2_chain_op(x1, op);
            x2 = _avx2_chain_op(x2, op);
            x3 = _avx2_chain_op(x3, op);
        }
        _mm256_storeu_ps(dst + i,      x0);
        _mm256_storeu_ps(dst + i + 8,  x1);
        _mm256_storeu_ps(dst + i + 16, x2);
        _mm256_storeu_ps(dst + i + 24, x3);
    }
    for (Py_ssize_t i = (Py_ssize_t)b32 * 32; i < n; i++) {
        float v = src[i];
        for (int k = 0; k < nops; k++) v = _scalar_chain_op(v, ops[k]);
        dst[i] = v;
    }
#else
    for (Py_ssize_t i = 0; i < n; i++) {
        float v = src[i];
        for (int k = 0; k < nops; k++) v = _scalar_chain_op(v, ops[k]);
        dst[i] = v;
    }
#endif
}

static PyObject* py_fused_chain_f32(PyObject *self, PyObject *args) {
    PyObject *src_obj, *dst_obj, *ops_obj;
    if (!PyArg_ParseTuple(args, "OOO", &src_obj, &ops_obj, &dst_obj)) return NULL;
    Py_buffer vs, vd, vo;
    if (PyObject_GetBuffer(src_obj, &vs, PyBUF_SIMPLE) < 0) return NULL;
    if (PyObject_GetBuffer(dst_obj, &vd, PyBUF_SIMPLE) < 0) {
        PyBuffer_Release(&vs); return NULL;
    }
    if (PyObject_GetBuffer(ops_obj, &vo, PyBUF_SIMPLE) < 0) {
        PyBuffer_Release(&vs); PyBuffer_Release(&vd); return NULL;
    }
    Py_ssize_t n    = vs.len / sizeof(float);
    int        nops = (int)(vo.len / sizeof(int));
    if (nops > VP_MAX_CHAIN) nops = VP_MAX_CHAIN;
    vp_fused_chain_f32_kernel(
        (const float*)vs.buf, (float*)vd.buf, n, (const int*)vo.buf, nops);
    PyBuffer_Release(&vs); PyBuffer_Release(&vd); PyBuffer_Release(&vo);
    Py_RETURN_NONE;
}

/* ── Binary ops ──────────────────────────────────────────────────────────── */
typedef void (*binary_f32_fn)(const float*, const float*, float*, Py_ssize_t);

static PyObject* _binary_f32(PyObject *self, PyObject *args, binary_f32_fn fn) {
    PyObject *a_obj, *b_obj, *dst_obj;
    if (!PyArg_ParseTuple(args, "OOO", &a_obj, &b_obj, &dst_obj)) return NULL;
    Py_buffer va, vb, vd;
    if (PyObject_GetBuffer(a_obj,   &va, PyBUF_SIMPLE)           < 0) return NULL;
    if (PyObject_GetBuffer(b_obj,   &vb, PyBUF_SIMPLE)           < 0) { PyBuffer_Release(&va); return NULL; }
    if (PyObject_GetBuffer(dst_obj, &vd, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&va); PyBuffer_Release(&vb); return NULL;
    }
    Py_ssize_t n = va.len / sizeof(float);
    fn((const float*)va.buf, (const float*)vb.buf, (float*)vd.buf, n);
    PyBuffer_Release(&va); PyBuffer_Release(&vb); PyBuffer_Release(&vd);
    Py_RETURN_NONE;
}

#define PY_BINARY_F32(PYNAME, CNAME)                                         \
static PyObject* py_##PYNAME(PyObject *self, PyObject *args) {              \
    return _binary_f32(self, args, CNAME##_f32);                            \
}

PY_BINARY_F32(add,     vp_add)
PY_BINARY_F32(sub,     vp_sub)
PY_BINARY_F32(mul,     vp_mul)
PY_BINARY_F32(div,     vp_div)
PY_BINARY_F32(maximum, vp_maximum)
PY_BINARY_F32(minimum, vp_minimum)

/* ── Scalar broadcast ops ─────────────────────────────────────────────────── */
typedef void (*scalar_f32_fn)(const float*, float, float*, Py_ssize_t);

static PyObject* _scalar_f32(PyObject *self, PyObject *args, scalar_f32_fn fn) {
    PyObject *a_obj, *dst_obj; float s;
    if (!PyArg_ParseTuple(args, "OfO", &a_obj, &s, &dst_obj)) return NULL;
    Py_buffer va, vd;
    if (PyObject_GetBuffer(a_obj,   &va, PyBUF_SIMPLE)           < 0) return NULL;
    if (PyObject_GetBuffer(dst_obj, &vd, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&va); return NULL;
    }
    Py_ssize_t n = va.len / sizeof(float);
    fn((const float*)va.buf, s, (float*)vd.buf, n);
    PyBuffer_Release(&va); PyBuffer_Release(&vd);
    Py_RETURN_NONE;
}

#define PY_SCALAR_F32(PYNAME, CNAME)                                         \
static PyObject* py_##PYNAME(PyObject *self, PyObject *args) {              \
    return _scalar_f32(self, args, CNAME##_scalar_f32);                     \
}

PY_SCALAR_F32(add_scalar,      vp_add)
PY_SCALAR_F32(sub_scalar,      vp_sub)
PY_SCALAR_F32(mul_scalar,      vp_mul)
PY_SCALAR_F32(div_scalar,      vp_div)
PY_SCALAR_F32(rdiv_scalar,     vp_rdiv)
PY_SCALAR_F32(pow_scalar,      vp_pow)
PY_SCALAR_F32(clamp_min,       vp_clamp_min)
PY_SCALAR_F32(clamp_max,       vp_clamp_max)

/* ── Reductions ──────────────────────────────────────────────────────────── */
static PyObject* py_sum_f32(PyObject *self, PyObject *args) {
    PyObject *obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;
    Py_buffer v;
    if (PyObject_GetBuffer(obj, &v, PyBUF_SIMPLE) < 0) return NULL;
    float r = vp_sum_f32_kernel((const float*)v.buf, v.len / sizeof(float));
    PyBuffer_Release(&v);
    return PyFloat_FromDouble((double)r);
}

static PyObject* py_max_f32(PyObject *self, PyObject *args) {
    PyObject *obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;
    Py_buffer v;
    if (PyObject_GetBuffer(obj, &v, PyBUF_SIMPLE) < 0) return NULL;
    float r = vp_max_f32_kernel((const float*)v.buf, v.len / sizeof(float));
    PyBuffer_Release(&v);
    return PyFloat_FromDouble((double)r);
}

static PyObject* py_min_f32(PyObject *self, PyObject *args) {
    PyObject *obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;
    Py_buffer v;
    if (PyObject_GetBuffer(obj, &v, PyBUF_SIMPLE) < 0) return NULL;
    float r = vp_min_f32_kernel((const float*)v.buf, v.len / sizeof(float));
    PyBuffer_Release(&v);
    return PyFloat_FromDouble((double)r);
}

static PyObject* py_argmax_f32(PyObject *self, PyObject *args) {
    PyObject *obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;
    Py_buffer v;
    if (PyObject_GetBuffer(obj, &v, PyBUF_SIMPLE) < 0) return NULL;
    Py_ssize_t r = vp_argmax_f32_kernel((const float*)v.buf, v.len / sizeof(float));
    PyBuffer_Release(&v);
    return PyLong_FromSsize_t(r);
}

static PyObject* py_argmin_f32(PyObject *self, PyObject *args) {
    PyObject *obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;
    Py_buffer v;
    if (PyObject_GetBuffer(obj, &v, PyBUF_SIMPLE) < 0) return NULL;
    Py_ssize_t r = vp_argmin_f32_kernel((const float*)v.buf, v.len / sizeof(float));
    PyBuffer_Release(&v);
    return PyLong_FromSsize_t(r);
}

/* ── Softmax / log-softmax ───────────────────────────────────────────────── */

/* softmax_rows(src, dst, rows, cols) — apply softmax over each row */
static PyObject* py_softmax_rows(PyObject *self, PyObject *args) {
    PyObject *src_obj, *dst_obj; int rows, cols;
    if (!PyArg_ParseTuple(args, "OOii", &src_obj, &dst_obj, &rows, &cols)) return NULL;
    Py_buffer vs, vd;
    if (PyObject_GetBuffer(src_obj, &vs, PyBUF_SIMPLE)            < 0) return NULL;
    if (PyObject_GetBuffer(dst_obj, &vd, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&vs); return NULL;
    }
    const float *src = (const float*)vs.buf;
    float *dst       = (float*)vd.buf;
    int r;
#ifdef _OPENMP
    VP_PRAGMA(omp parallel for schedule(static) if(rows >= 4))
#endif
    for (r = 0; r < rows; r++)
        vp_softmax_row_f32(src + (Py_ssize_t)r*cols, dst + (Py_ssize_t)r*cols, cols);
    PyBuffer_Release(&vs); PyBuffer_Release(&vd);
    Py_RETURN_NONE;
}

static PyObject* py_log_softmax_rows(PyObject *self, PyObject *args) {
    PyObject *src_obj, *dst_obj; int rows, cols;
    if (!PyArg_ParseTuple(args, "OOii", &src_obj, &dst_obj, &rows, &cols)) return NULL;
    Py_buffer vs, vd;
    if (PyObject_GetBuffer(src_obj, &vs, PyBUF_SIMPLE)            < 0) return NULL;
    if (PyObject_GetBuffer(dst_obj, &vd, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&vs); return NULL;
    }
    const float *src = (const float*)vs.buf;
    float *dst       = (float*)vd.buf;
    int r;
#ifdef _OPENMP
    VP_PRAGMA(omp parallel for schedule(static) if(rows >= 4))
#endif
    for (r = 0; r < rows; r++)
        vp_log_softmax_row_f32(src + (Py_ssize_t)r*cols, dst + (Py_ssize_t)r*cols, cols);
    PyBuffer_Release(&vs); PyBuffer_Release(&vd);
    Py_RETURN_NONE;
}

/* ── Matrix multiply ─────────────────────────────────────────────────────── */
/* matmul_f32(A, B, C, M, K, N) — C[M,N] = A[M,K] * B[K,N] */
static PyObject* py_matmul_f32(PyObject *self, PyObject *args) {
    PyObject *a_obj, *b_obj, *c_obj;
    int M, K, N;
    if (!PyArg_ParseTuple(args, "OOOiii", &a_obj, &b_obj, &c_obj, &M, &K, &N)) return NULL;
    Py_buffer va, vb, vc;
    if (PyObject_GetBuffer(a_obj, &va, PyBUF_SIMPLE) < 0) return NULL;
    if (PyObject_GetBuffer(b_obj, &vb, PyBUF_SIMPLE) < 0) { PyBuffer_Release(&va); return NULL; }
    if (PyObject_GetBuffer(c_obj, &vc, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&va); PyBuffer_Release(&vb); return NULL;
    }
    vp_matmul_f32_dispatch(
        (const float*)va.buf, (const float*)vb.buf, (float*)vc.buf, M, K, N, VP_POST_NONE);
    PyBuffer_Release(&va); PyBuffer_Release(&vb); PyBuffer_Release(&vc);
    Py_RETURN_NONE;
}

/* matmul_act_f32(A, B, C, M, K, N, post_op) — fused GEMM + activation epilogue */
static PyObject* py_matmul_act_f32(PyObject *self, PyObject *args) {
    PyObject *a_obj, *b_obj, *c_obj;
    int M, K, N, post_op;
    if (!PyArg_ParseTuple(args, "OOOiiii", &a_obj, &b_obj, &c_obj, &M, &K, &N, &post_op)) return NULL;
    Py_buffer va, vb, vc;
    if (PyObject_GetBuffer(a_obj, &va, PyBUF_SIMPLE) < 0) return NULL;
    if (PyObject_GetBuffer(b_obj, &vb, PyBUF_SIMPLE) < 0) { PyBuffer_Release(&va); return NULL; }
    if (PyObject_GetBuffer(c_obj, &vc, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&va); PyBuffer_Release(&vb); return NULL;
    }
    vp_matmul_f32_dispatch(
        (const float*)va.buf, (const float*)vb.buf, (float*)vc.buf, M, K, N, post_op);
    PyBuffer_Release(&va); PyBuffer_Release(&vb); PyBuffer_Release(&vc);
    Py_RETURN_NONE;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Flash Attention v2 — O(N) memory, tiled softmax, AVX2-vectorised
 *
 *  Forward:  O = FlashAttn(Q, K, V)   scale = 1/sqrt(d)
 *  Backward: dQ, dK, dV from dO, Q, K, V, O, L (logsumexp)
 *
 *  All inputs float32, row-major:
 *    Q,K,V,O : (N, d)   — single attention head
 *    L       : (N,)     — saved logsumexp for backward
 *    dO      : (N, d)   — upstream gradient
 *    dQ,dK,dV: (N, d)   — output gradients
 *
 *  Tile sizes (tuned for 256KB L2 working set):
 *    Br = 64 rows of Q  (query tile)
 *    Bc = 64 rows of K  (key tile)
 * ══════════════════════════════════════════════════════════════════════════*/

#define FA_BR 64
#define FA_BC 64
#define FA_NEG_INF (-1e38f)

/* fast scalar exp approximation (matches degree-5 minimax, error < 2^-23) */
static __forceinline float _fa_exp(float x) {
    /* clamp to avoid NaN on extreme values */
    if (x < -88.0f) return 0.0f;
    return expf(x);
}

/*
 * vp_flash_attn_fwd_f32
 *
 * Implements Algorithm 2 from FlashAttention-2 (Tri Dao, 2023).
 * O(N) HBM memory — only tiles of size Br×d and Bc×d are live at once.
 *
 * Q,K,V: (N, d) input matrices
 * O    : (N, d) output
 * L    : (N,)   logsumexp — needed for backward
 * N    : sequence length
 * d    : head dimension
 * scale: 1/sqrt(d)  (passed in to avoid recompute)
 */
/* Horizontal sum of 8-wide AVX2 register — shared by Flash Attn QK^T loop */
static VP_INLINE float _fa_hsum256(__m256 v) {
#ifdef __AVX2__
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
#else
    float buf[8]; memcpy(buf, &v, 32);
    return buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7];
#endif
}

static void vp_flash_attn_fwd_f32(
    const float* VP_RESTRICT Q,
    const float* VP_RESTRICT K,
    const float* VP_RESTRICT V,
          float* VP_RESTRICT O,
          float* VP_RESTRICT L,
    int N, int d, float scale)
{
    int Tr = (N + FA_BR - 1) / FA_BR;
    int Tc = (N + FA_BC - 1) / FA_BC;
    int i; /* MSVC OpenMP 2.0 requires loop var declared outside for */

#ifdef _OPENMP
    int max_fa_t = (Tr >= 4) ? omp_get_max_threads() : 1;
    if (max_fa_t > Tr) max_fa_t = Tr;
#   define FA_OMP_PRAGMA VP_PRAGMA(omp parallel for schedule(dynamic) num_threads(max_fa_t) if(max_fa_t > 1))
#else
#   define FA_OMP_PRAGMA
#endif

    FA_OMP_PRAGMA
    for (i = 0; i < Tr; i++) {
        /* Thread-local scratch — independent per Q tile, safe under OMP */
        float* S   = (float*)malloc(FA_BR * FA_BC * sizeof(float));
        float* P   = (float*)malloc(FA_BR * FA_BC * sizeof(float));
        float* Oi  = (float*)malloc(FA_BR * d     * sizeof(float));
        float* mi  = (float*)malloc(FA_BR         * sizeof(float));
        float* li  = (float*)malloc(FA_BR         * sizeof(float));
        if (!S || !P || !Oi || !mi || !li) {
            free(S); free(P); free(Oi); free(mi); free(li); continue;
        }

        int br = (i * FA_BR + FA_BR <= N) ? FA_BR : (N - i * FA_BR);
        const float* Qi = Q + i * FA_BR * d;

        for (int r = 0; r < br; r++) { mi[r] = FA_NEG_INF; li[r] = 0.0f; }
        memset(Oi, 0, br * d * sizeof(float));

        for (int j = 0; j < Tc; j++) {
            int bc = (j * FA_BC + FA_BC <= N) ? FA_BC : (N - j * FA_BC);
            const float* Kj = K + j * FA_BC * d;
            const float* Vj = V + j * FA_BC * d;

            /* ── S = Qi @ Kj^T * scale  (br x bc)
             * Key optimisation: for each Kj row (column of output), load it
             * ONCE and dot with 4 consecutive Qi rows sharing the same load.
             * 4× fewer memory accesses to K vs. the naive double loop.          */
            memset(S, 0, br * bc * sizeof(float));
            for (int c = 0; c < bc; c++) {
                const float* Krow = Kj + c * d;
                int r = 0;
#ifdef __AVX2__
                for (; r + 4 <= br; r += 4) {
                    const float* Q0 = Qi + (r+0)*d;
                    const float* Q1 = Qi + (r+1)*d;
                    const float* Q2 = Qi + (r+2)*d;
                    const float* Q3 = Qi + (r+3)*d;
                    __m256 a0=_mm256_setzero_ps(), a1=_mm256_setzero_ps();
                    __m256 a2=_mm256_setzero_ps(), a3=_mm256_setzero_ps();
                    for (int p = 0; p <= d - 8; p += 8) {
                        __m256 k = _mm256_loadu_ps(Krow + p);
                        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(Q0+p), k, a0);
                        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(Q1+p), k, a1);
                        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(Q2+p), k, a2);
                        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(Q3+p), k, a3);
                    }
                    S[(r+0)*bc + c] = _fa_hsum256(a0) * scale;
                    S[(r+1)*bc + c] = _fa_hsum256(a1) * scale;
                    S[(r+2)*bc + c] = _fa_hsum256(a2) * scale;
                    S[(r+3)*bc + c] = _fa_hsum256(a3) * scale;
                    /* scalar tail (d not multiple of 8) */
                    for (int p = (d & ~7); p < d; p++) {
                        S[(r+0)*bc+c] += Q0[p] * Krow[p] * scale;
                        S[(r+1)*bc+c] += Q1[p] * Krow[p] * scale;
                        S[(r+2)*bc+c] += Q2[p] * Krow[p] * scale;
                        S[(r+3)*bc+c] += Q3[p] * Krow[p] * scale;
                    }
                }
#endif
                for (; r < br; r++) {
                    float dot = 0.f;
                    for (int p = 0; p < d; p++) dot += Qi[r*d+p] * Krow[p];
                    S[r*bc + c] = dot * scale;
                }
            }

            /* ── Row-wise online softmax + rescale + Oi += P @ Vj ── */
            for (int r = 0; r < br; r++) {
                float mij = FA_NEG_INF;
                for (int c = 0; c < bc; c++) if (S[r*bc+c] > mij) mij = S[r*bc+c];

                float mi_new = (mi[r] > mij) ? mi[r] : mij;
                float lij = 0.0f;
                for (int c = 0; c < bc; c++) {
                    P[r*bc+c] = _fa_exp(S[r*bc+c] - mi_new);
                    lij += P[r*bc+c];
                }

                float alpha = _fa_exp(mi[r] - mi_new);
                int k = 0;
#ifdef __AVX2__
                { __m256 va = _mm256_set1_ps(alpha);
                  for (; k <= d - 8; k += 8)
                      _mm256_storeu_ps(Oi+r*d+k, _mm256_mul_ps(va, _mm256_loadu_ps(Oi+r*d+k))); }
#endif
                for (; k < d; k++) Oi[r*d+k] *= alpha;

                /* Oi += P[r,:] @ Vj  with AVX2 broadcast-FMA */
                for (int c = 0; c < bc; c++) {
                    float prc = P[r*bc+c];
                    k = 0;
#ifdef __AVX2__
                    __m256 vp = _mm256_set1_ps(prc);
                    for (; k <= d - 8; k += 8)
                        _mm256_storeu_ps(Oi+r*d+k,
                            _mm256_fmadd_ps(vp, _mm256_loadu_ps(Vj+c*d+k),
                                                 _mm256_loadu_ps(Oi+r*d+k)));
#endif
                    for (; k < d; k++) Oi[r*d+k] += prc * Vj[c*d+k];
                }

                li[r] = alpha * li[r] + lij;
                mi[r] = mi_new;
            }
        } /* end K tiles */

        for (int r = 0; r < br; r++) {
            float inv_li = 1.0f / li[r];
            float* Orow = O + (i * FA_BR + r) * d;
            int k = 0;
#ifdef __AVX2__
            { __m256 vi = _mm256_set1_ps(inv_li);
              for (; k <= d - 8; k += 8)
                  _mm256_storeu_ps(Orow+k, _mm256_mul_ps(vi, _mm256_loadu_ps(Oi+r*d+k))); }
#endif
            for (; k < d; k++) Orow[k] = Oi[r*d+k] * inv_li;
            L[i * FA_BR + r] = mi[r] + logf(li[r]);
        }

        free(S); free(P); free(Oi); free(mi); free(li);
    } /* end Q tiles (OMP parallel) */
#undef FA_OMP_PRAGMA
}

/*
 * vp_flash_attn_bwd_f32
 *
 * Recompute-based backward pass (Algorithm 4, FlashAttention-2).
 * No N² materialisation — recomputes P tiles on-the-fly from Q,K,L.
 *
 * Inputs:  dO (N,d), Q (N,d), K (N,d), V (N,d), O (N,d), L (N,)
 * Outputs: dQ (N,d), dK (N,d), dV (N,d)
 */
static void vp_flash_attn_bwd_f32(
    const float* VP_RESTRICT dO,
    const float* VP_RESTRICT Q,
    const float* VP_RESTRICT K,
    const float* VP_RESTRICT V,
    const float* VP_RESTRICT O,
    const float* VP_RESTRICT L,
          float* VP_RESTRICT dQ,
          float* VP_RESTRICT dK,
          float* VP_RESTRICT dV,
    int N, int d, float scale)
{
    float* S  = (float*)malloc(FA_BR * FA_BC * sizeof(float));
    float* P  = (float*)malloc(FA_BR * FA_BC * sizeof(float));
    float* dS = (float*)malloc(FA_BR * FA_BC * sizeof(float));
    float* Di = (float*)malloc(FA_BR * sizeof(float)); /* D_i = rowsum(dO * O) */
    if (!S || !P || !dS || !Di) {
        free(S); free(P); free(dS); free(Di); return;
    }

    /* Zero output grads */
    memset(dQ, 0, N * d * sizeof(float));
    memset(dK, 0, N * d * sizeof(float));
    memset(dV, 0, N * d * sizeof(float));

    int Tr = (N + FA_BR - 1) / FA_BR;
    int Tc = (N + FA_BC - 1) / FA_BC;

    for (int i = 0; i < Tr; i++) {
        int br = (i * FA_BR + FA_BR <= N) ? FA_BR : (N - i * FA_BR);
        const float* Qi  = Q  + i * FA_BR * d;
        const float* dOi = dO + i * FA_BR * d;
        const float* Oi  = O  + i * FA_BR * d;
        const float* Li  = L  + i * FA_BR;
        float*       dQi = dQ + i * FA_BR * d;

        /* D_i = rowsum(dO_i * O_i) */
        for (int r = 0; r < br; r++) {
            float s = 0.0f;
            for (int k = 0; k < d; k++) s += dOi[r * d + k] * Oi[r * d + k];
            Di[r] = s;
        }

        for (int j = 0; j < Tc; j++) {
            int bc = (j * FA_BC + FA_BC <= N) ? FA_BC : (N - j * FA_BC);
            const float* Kj = K + j * FA_BC * d;
            const float* Vj = V + j * FA_BC * d;
            float*       dKj = dK + j * FA_BC * d;
            float*       dVj = dV + j * FA_BC * d;

            /* Recompute S, P */
            for (int r = 0; r < br; r++) {
                for (int c = 0; c < bc; c++) {
                    float dot = 0.0f;
                    for (int k = 0; k < d; k++) dot += Qi[r * d + k] * Kj[c * d + k];
                    S[r * bc + c] = dot * scale;
                }
            }
            for (int r = 0; r < br; r++)
                for (int c = 0; c < bc; c++)
                    P[r * bc + c] = _fa_exp(S[r * bc + c] - Li[r]);

            /* dV_j += P^T @ dO_i   (bc x br) @ (br x d) = (bc x d) */
            for (int c = 0; c < bc; c++)
                for (int r = 0; r < br; r++) {
                    float prc = P[r * bc + c];
                    for (int k = 0; k < d; k++) dVj[c * d + k] += prc * dOi[r * d + k];
                }

            /* dS = P * (dO_i @ V_j^T - D_i)  — (br x bc) */
            for (int r = 0; r < br; r++) {
                for (int c = 0; c < bc; c++) {
                    float dp = 0.0f;
                    for (int k = 0; k < d; k++) dp += dOi[r * d + k] * Vj[c * d + k];
                    dS[r * bc + c] = P[r * bc + c] * (dp - Di[r]);
                }
            }

            /* dQ_i += scale * dS @ K_j   (br x bc) @ (bc x d) = (br x d) */
            for (int r = 0; r < br; r++)
                for (int c = 0; c < bc; c++) {
                    float ds = dS[r * bc + c] * scale;
                    for (int k = 0; k < d; k++) dQi[r * d + k] += ds * Kj[c * d + k];
                }

            /* dK_j += scale * dS^T @ Q_i   (bc x br) @ (br x d) = (bc x d) */
            for (int c = 0; c < bc; c++)
                for (int r = 0; r < br; r++) {
                    float ds = dS[r * bc + c] * scale;
                    for (int k = 0; k < d; k++) dKj[c * d + k] += ds * Qi[r * d + k];
                }
        }
    }

    free(S); free(P); free(dS); free(Di);
}

/* ── Python bindings for Flash Attention ── */

static PyObject* py_flash_attn_fwd_f32(PyObject *self, PyObject *args) {
    PyObject *q_obj, *k_obj, *v_obj, *o_obj, *l_obj;
    int N, d;
    float scale;
    if (!PyArg_ParseTuple(args, "OOOOOiif", &q_obj, &k_obj, &v_obj, &o_obj, &l_obj,
                          &N, &d, &scale)) return NULL;
    Py_buffer vq, vk, vv, vo, vl;
    if (PyObject_GetBuffer(q_obj, &vq, PyBUF_SIMPLE) < 0) return NULL;
#define FA_REL(n) PyBuffer_Release(&vq); if (n >= 1) PyBuffer_Release(&vk); \
    if (n >= 2) PyBuffer_Release(&vv); if (n >= 3) PyBuffer_Release(&vo); \
    if (n >= 4) PyBuffer_Release(&vl);
    if (PyObject_GetBuffer(k_obj, &vk, PyBUF_SIMPLE) < 0) { FA_REL(0); return NULL; }
    if (PyObject_GetBuffer(v_obj, &vv, PyBUF_SIMPLE) < 0) { FA_REL(1); return NULL; }
    if (PyObject_GetBuffer(o_obj, &vo, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) { FA_REL(2); return NULL; }
    if (PyObject_GetBuffer(l_obj, &vl, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) { FA_REL(3); return NULL; }
    vp_flash_attn_fwd_f32(
        (const float*)vq.buf, (const float*)vk.buf, (const float*)vv.buf,
        (float*)vo.buf, (float*)vl.buf, N, d, scale);
    FA_REL(4);
    Py_RETURN_NONE;
}

static PyObject* py_flash_attn_bwd_f32(PyObject *self, PyObject *args) {
    PyObject *do_obj, *q_obj, *k_obj, *v_obj, *o_obj, *l_obj,
             *dq_obj, *dk_obj, *dv_obj;
    int N, d;
    float scale;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOiif",
                          &do_obj, &q_obj, &k_obj, &v_obj, &o_obj, &l_obj,
                          &dq_obj, &dk_obj, &dv_obj, &N, &d, &scale))
        return NULL;
    Py_buffer vdo, vq, vk, vv, vo, vl, vdq, vdk, vdv;
    /* acquire all read buffers */
    if (PyObject_GetBuffer(do_obj, &vdo, PyBUF_SIMPLE) < 0) return NULL;
    if (PyObject_GetBuffer(q_obj,  &vq,  PyBUF_SIMPLE) < 0) { PyBuffer_Release(&vdo); return NULL; }
    if (PyObject_GetBuffer(k_obj,  &vk,  PyBUF_SIMPLE) < 0) { PyBuffer_Release(&vdo); PyBuffer_Release(&vq); return NULL; }
    if (PyObject_GetBuffer(v_obj,  &vv,  PyBUF_SIMPLE) < 0) { PyBuffer_Release(&vdo); PyBuffer_Release(&vq); PyBuffer_Release(&vk); return NULL; }
    if (PyObject_GetBuffer(o_obj,  &vo,  PyBUF_SIMPLE) < 0) { PyBuffer_Release(&vdo); PyBuffer_Release(&vq); PyBuffer_Release(&vk); PyBuffer_Release(&vv); return NULL; }
    if (PyObject_GetBuffer(l_obj,  &vl,  PyBUF_SIMPLE) < 0) { PyBuffer_Release(&vdo); PyBuffer_Release(&vq); PyBuffer_Release(&vk); PyBuffer_Release(&vv); PyBuffer_Release(&vo); return NULL; }
    if (PyObject_GetBuffer(dq_obj, &vdq, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) { PyBuffer_Release(&vdo); PyBuffer_Release(&vq); PyBuffer_Release(&vk); PyBuffer_Release(&vv); PyBuffer_Release(&vo); PyBuffer_Release(&vl); return NULL; }
    if (PyObject_GetBuffer(dk_obj, &vdk, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) { PyBuffer_Release(&vdo); PyBuffer_Release(&vq); PyBuffer_Release(&vk); PyBuffer_Release(&vv); PyBuffer_Release(&vo); PyBuffer_Release(&vl); PyBuffer_Release(&vdq); return NULL; }
    if (PyObject_GetBuffer(dv_obj, &vdv, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) { PyBuffer_Release(&vdo); PyBuffer_Release(&vq); PyBuffer_Release(&vk); PyBuffer_Release(&vv); PyBuffer_Release(&vo); PyBuffer_Release(&vl); PyBuffer_Release(&vdq); PyBuffer_Release(&vdk); return NULL; }
    vp_flash_attn_bwd_f32(
        (const float*)vdo.buf, (const float*)vq.buf, (const float*)vk.buf,
        (const float*)vv.buf,  (const float*)vo.buf, (const float*)vl.buf,
        (float*)vdq.buf, (float*)vdk.buf, (float*)vdv.buf, N, d, scale);
    PyBuffer_Release(&vdo); PyBuffer_Release(&vq); PyBuffer_Release(&vk); PyBuffer_Release(&vv);
    PyBuffer_Release(&vo);  PyBuffer_Release(&vl);
    PyBuffer_Release(&vdq); PyBuffer_Release(&vdk); PyBuffer_Release(&vdv);
    Py_RETURN_NONE;
}
#undef FA_REL

/* ══════════════════════════════════════════════════════════════════════════
 * im2col / col2im  — convolution ↔ GEMM format conversion
 *
 * im2col: input (N,C,H,W) → col (N*H_out*W_out, C*Kh*Kw)
 *   Each row of col is one receptive field patch, in C×Kh×Kw order.
 *   Zero-pads for out-of-bounds positions (instead of explicit padding).
 *
 * col2im: col → input_grad (N,C,H,W)   (accumulates, caller zeros first)
 * ══════════════════════════════════════════════════════════════════════════ */
static void vp_im2col_f32(
    const float* VP_RESTRICT input,
    float*       VP_RESTRICT col,
    int N, int C, int H, int W,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw)
{
    int H_out = (H + 2*Ph - Kh) / Sh + 1;
    int W_out = (W + 2*Pw - Kw) / Sw + 1;
    int col_w = C * Kh * Kw;
    for (int n = 0; n < N; n++) {
        const float* inp_n = input + (Py_ssize_t)n * C * H * W;
        for (int oh = 0; oh < H_out; oh++) {
            for (int ow = 0; ow < W_out; ow++) {
                float* col_row = col + ((Py_ssize_t)(n * H_out + oh) * W_out + ow) * col_w;
                for (int c = 0; c < C; c++) {
                    const float* inp_c = inp_n + (Py_ssize_t)c * H * W;
                    for (int kh = 0; kh < Kh; kh++) {
                        int ih = oh * Sh - Ph + kh;
                        for (int kw = 0; kw < Kw; kw++) {
                            int iw = ow * Sw - Pw + kw;
                            int idx = (c * Kh + kh) * Kw + kw;
                            col_row[idx] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                           ? inp_c[ih * W + iw]
                                           : 0.0f;
                        }
                    }
                }
            }
        }
    }
}

static void vp_col2im_f32(
    const float* VP_RESTRICT col,
    float*       VP_RESTRICT input_grad,
    int N, int C, int H, int W,
    int Kh, int Kw, int Sh, int Sw, int Ph, int Pw)
{
    int H_out = (H + 2*Ph - Kh) / Sh + 1;
    int W_out = (W + 2*Pw - Kw) / Sw + 1;
    int col_w = C * Kh * Kw;
    for (int n = 0; n < N; n++) {
        float* grad_n = input_grad + (Py_ssize_t)n * C * H * W;
        for (int oh = 0; oh < H_out; oh++) {
            for (int ow = 0; ow < W_out; ow++) {
                const float* col_row = col + ((Py_ssize_t)(n * H_out + oh) * W_out + ow) * col_w;
                for (int c = 0; c < C; c++) {
                    float* grad_c = grad_n + (Py_ssize_t)c * H * W;
                    for (int kh = 0; kh < Kh; kh++) {
                        int ih = oh * Sh - Ph + kh;
                        for (int kw = 0; kw < Kw; kw++) {
                            int iw = ow * Sw - Pw + kw;
                            int idx = (c * Kh + kh) * Kw + kw;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                grad_c[ih * W + iw] += col_row[idx];
                        }
                    }
                }
            }
        }
    }
}

/* Python bindings for im2col / col2im */
static PyObject* py_im2col_f32(PyObject *self, PyObject *args) {
    PyObject *inp_obj, *col_obj;
    int N, C, H, W, Kh, Kw, Sh, Sw, Ph, Pw;
    if (!PyArg_ParseTuple(args, "OOiiiiiiiiii",
                          &inp_obj, &col_obj, &N, &C, &H, &W,
                          &Kh, &Kw, &Sh, &Sw, &Ph, &Pw)) return NULL;
    Py_buffer vi, vc;
    if (PyObject_GetBuffer(inp_obj, &vi, PyBUF_SIMPLE) < 0) return NULL;
    if (PyObject_GetBuffer(col_obj, &vc, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&vi); return NULL;
    }
    vp_im2col_f32((const float*)vi.buf, (float*)vc.buf,
                  N, C, H, W, Kh, Kw, Sh, Sw, Ph, Pw);
    PyBuffer_Release(&vi); PyBuffer_Release(&vc);
    Py_RETURN_NONE;
}

static PyObject* py_col2im_f32(PyObject *self, PyObject *args) {
    PyObject *col_obj, *grad_obj;
    int N, C, H, W, Kh, Kw, Sh, Sw, Ph, Pw;
    if (!PyArg_ParseTuple(args, "OOiiiiiiiiii",
                          &col_obj, &grad_obj, &N, &C, &H, &W,
                          &Kh, &Kw, &Sh, &Sw, &Ph, &Pw)) return NULL;
    Py_buffer vc, vg;
    if (PyObject_GetBuffer(col_obj,  &vc, PyBUF_SIMPLE) < 0) return NULL;
    if (PyObject_GetBuffer(grad_obj, &vg, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&vc); return NULL;
    }
    vp_col2im_f32((const float*)vc.buf, (float*)vg.buf,
                  N, C, H, W, Kh, Kw, Sh, Sw, Ph, Pw);
    PyBuffer_Release(&vc); PyBuffer_Release(&vg);
    Py_RETURN_NONE;
}

/* ── BF16 GEMM: C[M,N] = A_bf16[M,K] @ B_bf16[K,N], accumulate in FP32 ──
 * BF16 = float32 with low 16 mantissa bits zeroed. Convert: (uint16) << 16.
 * No AVX-512 BF16 needed — works on any AVX2 processor.                     */
static VP_INLINE float _bf16_to_f32(uint16_t x) {
    uint32_t bits = (uint32_t)x << 16;
    float f; memcpy(&f, &bits, 4); return f;
}

static void vp_gemm_bf16_f32(
    const uint16_t* A_bf16,  /* (M, K) BF16 row-major */
    const uint16_t* B_bf16,  /* (K, N) BF16 row-major */
    float*          C,       /* (M, N) FP32 output, zeroed by caller */
    int M, int K, int N)
{
    /* Convert both inputs to FP32 then delegate to the AVX2 blocked GEMM.
     * This gets the full tiled+packed GEMM performance automatically.        */
    float* Af32 = (float*)malloc((size_t)M * K * sizeof(float));
    float* Bf32 = (float*)malloc((size_t)K * N * sizeof(float));
    if (!Af32 || !Bf32) { free(Af32); free(Bf32); return; }
    for (Py_ssize_t i = 0; i < (Py_ssize_t)M * K; i++) Af32[i] = _bf16_to_f32(A_bf16[i]);
    for (Py_ssize_t i = 0; i < (Py_ssize_t)K * N; i++) Bf32[i] = _bf16_to_f32(B_bf16[i]);
    vp_matmul_f32_avx2(Af32, Bf32, C, M, K, N, VP_POST_NONE);
    free(Af32); free(Bf32);
}

static PyObject* py_gemm_bf16_f32(PyObject *self, PyObject *args) {
    PyObject *a_obj, *b_obj, *c_obj;
    int M, K, N;
    if (!PyArg_ParseTuple(args, "OOOiii", &a_obj, &b_obj, &c_obj, &M, &K, &N)) return NULL;
    Py_buffer va, vb, vc;
    if (PyObject_GetBuffer(a_obj, &va, PyBUF_SIMPLE) < 0) return NULL;
    if (PyObject_GetBuffer(b_obj, &vb, PyBUF_SIMPLE) < 0) { PyBuffer_Release(&va); return NULL; }
    if (PyObject_GetBuffer(c_obj, &vc, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&va); PyBuffer_Release(&vb); return NULL;
    }
    vp_gemm_bf16_f32((const uint16_t*)va.buf, (const uint16_t*)vb.buf,
                     (float*)vc.buf, M, K, N);
    PyBuffer_Release(&va); PyBuffer_Release(&vb); PyBuffer_Release(&vc);
    Py_RETURN_NONE;
}

/* Python wrappers for new unary kernels */
#define UNARY_KERNEL_PY(name, kern) \
static PyObject* py_##name(PyObject *self, PyObject *args) { \
    PyObject *s_obj, *d_obj; \
    if (!PyArg_ParseTuple(args, "OO", &s_obj, &d_obj)) return NULL; \
    Py_buffer vs, vd; \
    if (PyObject_GetBuffer(s_obj, &vs, PyBUF_SIMPLE) < 0) return NULL; \
    if (PyObject_GetBuffer(d_obj, &vd, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) { PyBuffer_Release(&vs); return NULL; } \
    kern((const float*)vs.buf, (float*)vd.buf, vs.len / sizeof(float)); \
    PyBuffer_Release(&vs); PyBuffer_Release(&vd); Py_RETURN_NONE; }

UNARY_KERNEL_PY(sin_f32,  vp_sin_f32_kernel)
UNARY_KERNEL_PY(cos_f32,  vp_cos_f32_kernel)

/* Python wrappers for backward kernels (go, inp_or_out, gi) */
#define BWD_KERNEL_PY(name, kern) \
static PyObject* py_##name(PyObject *self, PyObject *args) { \
    PyObject *go_obj, *io_obj, *gi_obj; \
    if (!PyArg_ParseTuple(args, "OOO", &go_obj, &io_obj, &gi_obj)) return NULL; \
    Py_buffer vgo, vio, vgi; \
    if (PyObject_GetBuffer(go_obj, &vgo, PyBUF_SIMPLE) < 0) return NULL; \
    if (PyObject_GetBuffer(io_obj, &vio, PyBUF_SIMPLE) < 0) { PyBuffer_Release(&vgo); return NULL; } \
    if (PyObject_GetBuffer(gi_obj, &vgi, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) { PyBuffer_Release(&vgo); PyBuffer_Release(&vio); return NULL; } \
    kern((const float*)vgo.buf, (const float*)vio.buf, (float*)vgi.buf, vgo.len / sizeof(float)); \
    PyBuffer_Release(&vgo); PyBuffer_Release(&vio); PyBuffer_Release(&vgi); Py_RETURN_NONE; }

BWD_KERNEL_PY(exp_backward_f32,  vp_exp_backward_f32_kernel)
BWD_KERNEL_PY(log_backward_f32,  vp_log_backward_f32_kernel)
BWD_KERNEL_PY(sqrt_backward_f32, vp_sqrt_backward_f32_kernel)
BWD_KERNEL_PY(sin_backward_f32,  vp_sin_backward_f32_kernel)
BWD_KERNEL_PY(cos_backward_f32,  vp_cos_backward_f32_kernel)

/* ── dot product ─────────────────────────────────────────────────────────── */
static PyObject* py_dot_f32(PyObject *self, PyObject *args) {
    PyObject *a_obj, *b_obj;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return NULL;
    Py_buffer va, vb;
    if (PyObject_GetBuffer(a_obj, &va, PyBUF_SIMPLE) < 0) return NULL;
    if (PyObject_GetBuffer(b_obj, &vb, PyBUF_SIMPLE) < 0) { PyBuffer_Release(&va); return NULL; }
    Py_ssize_t n = va.len / sizeof(float);
    const float *a = (const float*)va.buf;
    const float *b = (const float*)vb.buf;
    double acc = 0.0;
    Py_ssize_t i = 0;
#if VP_AVX2
    __m256d vacc = _mm256_setzero_pd();
    for (; i <= n - 8; i += 8) {
        __m256 va8 = _mm256_loadu_ps(a + i);
        __m256 vb8 = _mm256_loadu_ps(b + i);
        __m256 vp  = _mm256_mul_ps(va8, vb8);
        /* split 8xf32 sum into 4xf64 for precision */
        __m128 lo = _mm256_castps256_ps128(vp);
        __m128 hi = _mm256_extractf128_ps(vp, 1);
        vacc = _mm256_add_pd(vacc, _mm256_cvtps_pd(lo));
        vacc = _mm256_add_pd(vacc, _mm256_cvtps_pd(hi));
    }
    __m128d lo2 = _mm256_castpd256_pd128(vacc);
    __m128d hi2 = _mm256_extractf128_pd(vacc, 1);
    lo2 = _mm_add_pd(lo2, hi2);
    lo2 = _mm_hadd_pd(lo2, lo2);
    acc = _mm_cvtsd_f64(lo2);
#endif
    for (; i < n; i++) acc += (double)a[i] * (double)b[i];
    PyBuffer_Release(&va); PyBuffer_Release(&vb);
    return PyFloat_FromDouble(acc);
}

/* ── has_avx2 query ──────────────────────────────────────────────────────── */
static PyObject* py_has_avx2(PyObject *self, PyObject *args) {
#if VP_AVX2
    Py_RETURN_TRUE;
#else
    Py_RETURN_FALSE;
#endif
}

/* ── Fill / clamp helpers ────────────────────────────────────────────────── */
static PyObject* py_fill_f32(PyObject *self, PyObject *args) {
    PyObject *obj; float val;
    if (!PyArg_ParseTuple(args, "Of", &obj, &val)) return NULL;
    Py_buffer v;
    if (PyObject_GetBuffer(obj, &v, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) return NULL;
    float *dst = (float*)v.buf;
    Py_ssize_t n = v.len / sizeof(float);
    Py_ssize_t i = 0;
#if VP_AVX2
    __m256 vv = _mm256_set1_ps(val);
    for (; i <= n - 8; i += 8) _mm256_storeu_ps(dst + i, vv);
#endif
    for (; i < n; i++) dst[i] = val;
    PyBuffer_Release(&v);
    Py_RETURN_NONE;
}

/* ── axpy: dst += alpha * src ────────────────────────────────────────────── */
static PyObject* py_axpy_f32(PyObject *self, PyObject *args) {
    PyObject *src_obj, *dst_obj; float alpha;
    if (!PyArg_ParseTuple(args, "OfO", &src_obj, &alpha, &dst_obj)) return NULL;
    Py_buffer vs, vd;
    if (PyObject_GetBuffer(src_obj, &vs, PyBUF_SIMPLE) < 0) return NULL;
    if (PyObject_GetBuffer(dst_obj, &vd, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&vs); return NULL;
    }
    const float *src = (const float*)vs.buf;
    float *dst = (float*)vd.buf;
    Py_ssize_t n = vs.len / sizeof(float);
    Py_ssize_t i = 0;
#if VP_AVX2
    __m256 va = _mm256_set1_ps(alpha);
    for (; i <= n - 8; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(src + i),
                                                       _mm256_loadu_ps(dst + i)));
#endif
    for (; i < n; i++) dst[i] += alpha * src[i];
    PyBuffer_Release(&vs); PyBuffer_Release(&vd);
    Py_RETURN_NONE;
}

/* ══════════════════════════════════════════════════════════════════════════════
 * FLOAT64 ops (key ops only — exp, log, sum, matmul)
 * ══════════════════════════════════════════════════════════════════════════════ */

static PyObject* py_exp_f64(PyObject *self, PyObject *args) {
    PyObject *src_obj, *dst_obj;
    if (!PyArg_ParseTuple(args, "OO", &src_obj, &dst_obj)) return NULL;
    Py_buffer vs, vd;
    if (PyObject_GetBuffer(src_obj, &vs, PyBUF_SIMPLE)            < 0) return NULL;
    if (PyObject_GetBuffer(dst_obj, &vd, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&vs); return NULL;
    }
    const double *src = (const double*)vs.buf;
    double *dst       = (double*)vd.buf;
    Py_ssize_t n      = vs.len / sizeof(double);
    for (Py_ssize_t i = 0; i < n; i++) dst[i] = exp(src[i]);
    PyBuffer_Release(&vs); PyBuffer_Release(&vd);
    Py_RETURN_NONE;
}

static PyObject* py_log_f64(PyObject *self, PyObject *args) {
    PyObject *src_obj, *dst_obj;
    if (!PyArg_ParseTuple(args, "OO", &src_obj, &dst_obj)) return NULL;
    Py_buffer vs, vd;
    if (PyObject_GetBuffer(src_obj, &vs, PyBUF_SIMPLE)            < 0) return NULL;
    if (PyObject_GetBuffer(dst_obj, &vd, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&vs); return NULL;
    }
    const double *src = (const double*)vs.buf;
    double *dst       = (double*)vd.buf;
    Py_ssize_t n      = vs.len / sizeof(double);
    for (Py_ssize_t i = 0; i < n; i++) dst[i] = log(src[i]);
    PyBuffer_Release(&vs); PyBuffer_Release(&vd);
    Py_RETURN_NONE;
}

static PyObject* py_sum_f64(PyObject *self, PyObject *args) {
    PyObject *obj;
    if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;
    Py_buffer v;
    if (PyObject_GetBuffer(obj, &v, PyBUF_SIMPLE) < 0) return NULL;
    const double *src = (const double*)v.buf;
    Py_ssize_t n = v.len / sizeof(double);
    double acc = 0.0;
#if VP_AVX2
    __m256d vacc = _mm256_setzero_pd();
    Py_ssize_t i = 0;
    for (; i <= n - 4; i += 4)
        vacc = _mm256_add_pd(vacc, _mm256_loadu_pd(src + i));
    __m128d lo = _mm256_castpd256_pd128(vacc);
    __m128d hi = _mm256_extractf128_pd(vacc, 1);
    lo = _mm_add_pd(lo, hi);
    lo = _mm_hadd_pd(lo, lo);
    acc = _mm_cvtsd_f64(lo);
    for (; i < n; i++) acc += src[i];
#else
    for (Py_ssize_t i = 0; i < n; i++) acc += src[i];
#endif
    PyBuffer_Release(&v);
    return PyFloat_FromDouble(acc);
}

static PyObject* py_matmul_f64(PyObject *self, PyObject *args) {
    PyObject *a_obj, *b_obj, *c_obj;
    int M, K, N;
    if (!PyArg_ParseTuple(args, "OOOiii", &a_obj, &b_obj, &c_obj, &M, &K, &N)) return NULL;
    Py_buffer va, vb, vc;
    if (PyObject_GetBuffer(a_obj, &va, PyBUF_SIMPLE) < 0) return NULL;
    if (PyObject_GetBuffer(b_obj, &vb, PyBUF_SIMPLE) < 0) { PyBuffer_Release(&va); return NULL; }
    if (PyObject_GetBuffer(c_obj, &vc, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&va); PyBuffer_Release(&vb); return NULL;
    }
    const double *A = (const double*)va.buf;
    const double *B = (const double*)vb.buf;
    double *C       = (double*)vc.buf;
    memset(C, 0, (size_t)M * N * sizeof(double));
    for (int i = 0; i < M; i++)
    for (int p = 0; p < K; p++) {
        double a = A[i*K + p];
        for (int j = 0; j < N; j++) C[i*N + j] += a * B[p*N + j];
    }
    PyBuffer_Release(&va); PyBuffer_Release(&vb); PyBuffer_Release(&vc);
    Py_RETURN_NONE;
}

/* ── leaky_relu ──────────────────────────────────────────────────────────── */
static PyObject* py_leaky_relu_f32(PyObject *self, PyObject *args) {
    PyObject *src_obj, *dst_obj; float alpha;
    if (!PyArg_ParseTuple(args, "OOF", &src_obj, &dst_obj, &alpha)) {
        PyErr_Clear();
        if (!PyArg_ParseTuple(args, "OOf", &src_obj, &dst_obj, &alpha)) return NULL;
    }
    Py_buffer vs, vd;
    if (PyObject_GetBuffer(src_obj, &vs, PyBUF_SIMPLE)            < 0) return NULL;
    if (PyObject_GetBuffer(dst_obj, &vd, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&vs); return NULL;
    }
    const float *src = (const float*)vs.buf;
    float *dst       = (float*)vd.buf;
    Py_ssize_t n     = vs.len / sizeof(float);
    Py_ssize_t i = 0;
#if VP_AVX2
    __m256 va = _mm256_set1_ps(alpha);
    __m256 z  = _mm256_setzero_ps();
    for (; i <= n - 8; i += 8) {
        __m256 x  = _mm256_loadu_ps(src + i);
        __m256 pos = _mm256_max_ps(x, z);
        __m256 neg = _mm256_mul_ps(_mm256_min_ps(x, z), va);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(pos, neg));
    }
#endif
    for (; i < n; i++) dst[i] = src[i] >= 0.0f ? src[i] : alpha * src[i];
    PyBuffer_Release(&vs); PyBuffer_Release(&vd);
    Py_RETURN_NONE;
}

/* ══════════════════════════════════════════════════════════════════════════════
 * ADVANCED KERNELS — variance, layer_norm, transpose, axial reductions,
 *                    where, sigmoid/relu/gelu backward, element-wise pow
 * ══════════════════════════════════════════════════════════════════════════════ */

/* ── Variance (single-pass Welford) ──────────────────────────────────────── */
/* Returns (mean, variance) via two output pointers; unbiased=1 for N-1 denom */
static void vp_var_f32_kernel(const float* VP_RESTRICT src, Py_ssize_t n,
                               float *out_mean, float *out_var, int unbiased) {
    if (n == 0) { *out_mean = 0.0f; *out_var = 0.0f; return; }
    /* Two-pass for numerical stability: mean first, then variance */
    float mean = vp_sum_f32_kernel(src, n) / (float)n;
    float acc = 0.0f;
    Py_ssize_t i = 0;
#if VP_AVX2
    __m256 vm   = _mm256_set1_ps(mean);
    __m256 vacc = _mm256_setzero_ps();
    for (; i <= n - 8; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(src + i), vm);
        vacc = _mm256_fmadd_ps(d, d, vacc);
    }
    __m128 lo = _mm256_castps256_ps128(vacc);
    __m128 hi = _mm256_extractf128_ps(vacc, 1);
    lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
    acc = _mm_cvtss_f32(lo);
#endif
    for (; i < n; i++) { float d = src[i] - mean; acc += d * d; }
    *out_mean = mean;
    *out_var  = acc / (float)(unbiased ? (n - 1) : n);
}

static PyObject* py_var_f32(PyObject *self, PyObject *args) {
    PyObject *obj; int unbiased = 0;
    if (!PyArg_ParseTuple(args, "O|i", &obj, &unbiased)) return NULL;
    Py_buffer v;
    if (PyObject_GetBuffer(obj, &v, PyBUF_SIMPLE) < 0) return NULL;
    float mean, var;
    vp_var_f32_kernel((const float*)v.buf, v.len / sizeof(float), &mean, &var, unbiased);
    PyBuffer_Release(&v);
    return PyTuple_Pack(2, PyFloat_FromDouble(mean), PyFloat_FromDouble(var));
}

/* ── Layer Norm ────────────────────────────────────────────────────────────
 * y = (x - mean) / sqrt(var + eps) * gamma + beta
 * Applied independently to each row of length `norm_size`.
 * gamma/beta may be NULL (no affine scaling/shifting). */
static PyObject* py_layer_norm_f32(PyObject *self, PyObject *args) {
    PyObject *src_obj, *dst_obj, *gamma_obj, *beta_obj;
    int rows, norm_size; float eps;
    if (!PyArg_ParseTuple(args, "OOOOiif",
                          &src_obj, &dst_obj, &gamma_obj, &beta_obj,
                          &rows, &norm_size, &eps)) return NULL;

    Py_buffer vs, vd, vg_buf, vb_buf;
    int has_gamma = (gamma_obj != Py_None);
    int has_beta  = (beta_obj  != Py_None);

    if (PyObject_GetBuffer(src_obj, &vs, PyBUF_SIMPLE) < 0) return NULL;
    if (PyObject_GetBuffer(dst_obj, &vd, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&vs); return NULL;
    }
    if (has_gamma && PyObject_GetBuffer(gamma_obj, &vg_buf, PyBUF_SIMPLE) < 0) {
        PyBuffer_Release(&vs); PyBuffer_Release(&vd); return NULL;
    }
    if (has_beta && PyObject_GetBuffer(beta_obj, &vb_buf, PyBUF_SIMPLE) < 0) {
        PyBuffer_Release(&vs); PyBuffer_Release(&vd);
        if (has_gamma) PyBuffer_Release(&vg_buf);
        return NULL;
    }

    const float *src   = (const float*)vs.buf;
    float *dst         = (float*)vd.buf;
    const float *gamma = has_gamma ? (const float*)vg_buf.buf : NULL;
    const float *beta  = has_beta  ? (const float*)vb_buf.buf : NULL;

    for (int r = 0; r < rows; r++) {
        const float *row = src + (Py_ssize_t)r * norm_size;
        float *drow      = dst + (Py_ssize_t)r * norm_size;
        float mean, var;
        vp_var_f32_kernel(row, norm_size, &mean, &var, 0);
        float inv_std = 1.0f / sqrtf(var + eps);
        Py_ssize_t i = 0;
#if VP_AVX2
        __m256 vm  = _mm256_set1_ps(mean);
        __m256 vis = _mm256_set1_ps(inv_std);
        for (; i <= norm_size - 8; i += 8) {
            __m256 x  = _mm256_loadu_ps(row + i);
            __m256 xn = _mm256_mul_ps(_mm256_sub_ps(x, vm), vis);
            if (gamma) xn = _mm256_mul_ps(xn, _mm256_loadu_ps(gamma + i));
            if (beta)  xn = _mm256_add_ps(xn, _mm256_loadu_ps(beta + i));
            _mm256_storeu_ps(drow + i, xn);
        }
#endif
        for (; i < norm_size; i++) {
            float xn = (row[i] - mean) * inv_std;
            if (gamma) xn *= gamma[i];
            if (beta)  xn += beta[i];
            drow[i] = xn;
        }
    }

    if (has_beta)  PyBuffer_Release(&vb_buf);
    if (has_gamma) PyBuffer_Release(&vg_buf);
    PyBuffer_Release(&vd); PyBuffer_Release(&vs);
    Py_RETURN_NONE;
}

/* ── Batch Norm 1D ───────────────────────────────────────────────────────── */
/* Forward: input (N, C). Row-major two-pass AVX2 implementation.
   Pass 1: accumulate sum/sum-of-squares per feature (row-major, cache-friendly).
   Pass 2: normalize + scale/shift with AVX2 FMA.
   Signature: batch_norm_f32(x, out, mean, rstd, gamma, beta, xhat_or_None, N, C, eps) */
static PyObject* py_batch_norm_f32(PyObject *self, PyObject *args) {
    PyObject *x_obj, *out_obj, *mean_obj, *rstd_obj, *gamma_obj, *beta_obj, *xhat_obj;
    int N, C; float eps;
    if (!PyArg_ParseTuple(args, "OOOOOOOiif",
                          &x_obj, &out_obj, &mean_obj, &rstd_obj,
                          &gamma_obj, &beta_obj, &xhat_obj, &N, &C, &eps)) return NULL;
    Py_buffer vx, vo, vm, vr, vg, vb, vxh;
    if (PyObject_GetBuffer(x_obj,    &vx, PyBUF_SIMPLE) < 0) return NULL;
    if (PyObject_GetBuffer(out_obj,  &vo, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) { PyBuffer_Release(&vx); return NULL; }
    if (PyObject_GetBuffer(mean_obj, &vm, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) { PyBuffer_Release(&vx); PyBuffer_Release(&vo); return NULL; }
    if (PyObject_GetBuffer(rstd_obj, &vr, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) { PyBuffer_Release(&vx); PyBuffer_Release(&vo); PyBuffer_Release(&vm); return NULL; }
    int has_g = (gamma_obj != Py_None), has_b = (beta_obj != Py_None);
    int has_xh = (xhat_obj != Py_None);
    if (has_g && PyObject_GetBuffer(gamma_obj, &vg, PyBUF_SIMPLE) < 0) { PyBuffer_Release(&vx); PyBuffer_Release(&vo); PyBuffer_Release(&vm); PyBuffer_Release(&vr); return NULL; }
    if (has_b && PyObject_GetBuffer(beta_obj,  &vb, PyBUF_SIMPLE) < 0) { PyBuffer_Release(&vx); PyBuffer_Release(&vo); PyBuffer_Release(&vm); PyBuffer_Release(&vr); if (has_g) PyBuffer_Release(&vg); return NULL; }
    if (has_xh && PyObject_GetBuffer(xhat_obj, &vxh, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) { PyBuffer_Release(&vx); PyBuffer_Release(&vo); PyBuffer_Release(&vm); PyBuffer_Release(&vr); if (has_g) PyBuffer_Release(&vg); if (has_b) PyBuffer_Release(&vb); return NULL; }
    const float *x     = (const float*)vx.buf;
    float       *out   = (float*)vo.buf;
    float       *mean  = (float*)vm.buf;
    float       *rstd  = (float*)vr.buf;
    const float *gamma = has_g ? (const float*)vg.buf : NULL;
    const float *beta  = has_b ? (const float*)vb.buf : NULL;
    float       *xhat  = has_xh ? (float*)vxh.buf : NULL;

#if VP_AVX2
    /* AVX2 row-major two-pass: much better cache utilization than column-major */
    float *sum_c = (float*)vp_aligned_alloc(32, (size_t)C * sizeof(float));
    float *ssq_c = (float*)vp_aligned_alloc(32, (size_t)C * sizeof(float));
    if (!sum_c || !ssq_c) {
        if (sum_c) vp_aligned_free(sum_c);
        if (ssq_c) vp_aligned_free(ssq_c);
        PyErr_NoMemory(); goto bn_cleanup;
    }
    memset(sum_c, 0, (size_t)C * sizeof(float));
    memset(ssq_c, 0, (size_t)C * sizeof(float));

    /* Pass 1: accumulate sum and sum-of-squares per feature (row-major, cache-friendly) */
    for (int n = 0; n < N; n++) {
        const float *row = x + (Py_ssize_t)n * C;
        int c = 0;
        for (; c <= C - 8; c += 8) {
            __m256 v   = _mm256_loadu_ps(row + c);
            __m256 s   = _mm256_loadu_ps(sum_c + c);
            __m256 sq  = _mm256_loadu_ps(ssq_c + c);
            _mm256_storeu_ps(sum_c + c, _mm256_add_ps(s, v));
            _mm256_storeu_ps(ssq_c + c, _mm256_fmadd_ps(v, v, sq));
        }
        for (; c < C; c++) { sum_c[c] += row[c]; ssq_c[c] += row[c] * row[c]; }
    }

    /* Compute mean and rstd per feature */
    float inv_N = 1.0f / (float)N;
    {
        int c = 0;
        __m256 vinvN = _mm256_set1_ps(inv_N);
        __m256 veps  = _mm256_set1_ps(eps);
        for (; c <= C - 8; c += 8) {
            __m256 s  = _mm256_loadu_ps(sum_c + c);
            __m256 sq = _mm256_loadu_ps(ssq_c + c);
            __m256 m  = _mm256_mul_ps(s, vinvN);
            __m256 var = _mm256_sub_ps(_mm256_mul_ps(sq, vinvN), _mm256_mul_ps(m, m));
            var = _mm256_max_ps(var, _mm256_setzero_ps());
            __m256 rs = _mm256_rsqrt_ps(_mm256_add_ps(var, veps));
            /* One Newton-Raphson refinement for rsqrt accuracy */
            __m256 half = _mm256_set1_ps(0.5f);
            __m256 three = _mm256_set1_ps(3.0f);
            rs = _mm256_mul_ps(half, _mm256_mul_ps(rs, _mm256_fnmadd_ps(_mm256_mul_ps(rs, rs), _mm256_add_ps(var, veps), three)));
            _mm256_storeu_ps(mean + c, m);
            _mm256_storeu_ps(rstd + c, rs);
        }
        for (; c < C; c++) {
            float m = sum_c[c] * inv_N;
            float var = ssq_c[c] * inv_N - m * m;
            if (var < 0.0f) var = 0.0f;
            mean[c] = m;
            rstd[c] = 1.0f / sqrtf(var + eps);
        }
    }
    vp_aligned_free(sum_c); vp_aligned_free(ssq_c);

    /* Pass 2: normalize + optional scale/shift (row-major, cache-friendly) */
    for (int n = 0; n < N; n++) {
        const float *row   = x   + (Py_ssize_t)n * C;
        float       *orow  = out + (Py_ssize_t)n * C;
        float       *xhrow = xhat ? xhat + (Py_ssize_t)n * C : NULL;
        int c = 0;
        for (; c <= C - 8; c += 8) {
            __m256 v  = _mm256_loadu_ps(row + c);
            __m256 m  = _mm256_loadu_ps(mean + c);
            __m256 rs = _mm256_loadu_ps(rstd + c);
            __m256 xh = _mm256_mul_ps(_mm256_sub_ps(v, m), rs);
            if (xhrow) _mm256_storeu_ps(xhrow + c, xh);
            if (gamma && beta) {
                __m256 g = _mm256_loadu_ps(gamma + c);
                __m256 b = _mm256_loadu_ps(beta  + c);
                xh = _mm256_fmadd_ps(xh, g, b);
            }
            _mm256_storeu_ps(orow + c, xh);
        }
        for (; c < C; c++) {
            float xh = (row[c] - mean[c]) * rstd[c];
            if (xhrow) xhrow[c] = xh;
            orow[c] = gamma ? gamma[c] * xh + beta[c] : xh;
        }
    }
    goto bn_done;
bn_cleanup:;
#endif /* VP_AVX2 */

    /* Scalar fallback (column-major) */
    {
        float inv_N2 = 1.0f / (float)N;
        for (int c = 0; c < C; c++) {
            float s = 0.0f;
            for (int n = 0; n < N; n++) s += x[(Py_ssize_t)n * C + c];
            float m = s * inv_N2;
            float v = 0.0f;
            for (int n = 0; n < N; n++) { float d = x[(Py_ssize_t)n * C + c] - m; v += d * d; }
            float rs = 1.0f / sqrtf(v * inv_N2 + eps);
            mean[c] = m; rstd[c] = rs;
            float g = has_g ? gamma[c] : 1.0f;
            float b = has_b ? beta[c]  : 0.0f;
            for (int n = 0; n < N; n++) {
                Py_ssize_t idx = (Py_ssize_t)n * C + c;
                float xh = (x[idx] - m) * rs;
                if (xhat) xhat[idx] = xh;
                out[idx] = g * xh + b;
            }
        }
    }

#if VP_AVX2
bn_done:;
#endif
    if (has_xh) PyBuffer_Release(&vxh);
    if (has_b) PyBuffer_Release(&vb);
    if (has_g) PyBuffer_Release(&vg);
    PyBuffer_Release(&vr); PyBuffer_Release(&vm); PyBuffer_Release(&vo); PyBuffer_Release(&vx);
    Py_RETURN_NONE;
}

/* Update running mean and variance from per-batch mean/rstd.
   Signature: bn_update_running_f32(mean_buf, rstd_buf, rm_buf, rv_buf, C, momentum, eps, N)
   rv_buf updated with Bessel-corrected variance (unbiased estimator). */
static PyObject* py_bn_update_running_f32(PyObject *self, PyObject *args) {
    PyObject *mean_obj, *rstd_obj, *rm_obj, *rv_obj;
    int C, N; float momentum, eps;
    if (!PyArg_ParseTuple(args, "OOOOiffi", &mean_obj, &rstd_obj, &rm_obj, &rv_obj,
                          &C, &momentum, &eps, &N)) return NULL;
    Py_buffer vm, vrs, vrm, vrv;
    if (PyObject_GetBuffer(mean_obj, &vm,  PyBUF_SIMPLE) < 0) return NULL;
    if (PyObject_GetBuffer(rstd_obj, &vrs, PyBUF_SIMPLE) < 0) { PyBuffer_Release(&vm); return NULL; }
    if (PyObject_GetBuffer(rm_obj,   &vrm, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) { PyBuffer_Release(&vm); PyBuffer_Release(&vrs); return NULL; }
    if (PyObject_GetBuffer(rv_obj,   &vrv, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) { PyBuffer_Release(&vm); PyBuffer_Release(&vrs); PyBuffer_Release(&vrm); return NULL; }
    const float *mean = (const float*)vm.buf;
    const float *rstd = (const float*)vrs.buf;
    float       *rm   = (float*)vrm.buf;
    float       *rv   = (float*)vrv.buf;
    float one_m = 1.0f - momentum;
    float scale_var = (N > 1) ? (float)N / (float)(N - 1) : 1.0f;
#if VP_AVX2
    __m256 vmom  = _mm256_set1_ps(momentum);
    __m256 v1m   = _mm256_set1_ps(one_m);
    __m256 vsv   = _mm256_set1_ps(scale_var);
    __m256 veps  = _mm256_set1_ps(eps);
    int c = 0;
    for (; c <= C - 8; c += 8) {
        __m256 m   = _mm256_loadu_ps(mean + c);
        __m256 rs  = _mm256_loadu_ps(rstd + c);
        /* var = 1/(rstd^2) - eps */
        __m256 var = _mm256_sub_ps(_mm256_rcp_ps(_mm256_mul_ps(rs, rs)), veps);
        var = _mm256_max_ps(var, _mm256_setzero_ps());
        /* new_rm = one_m * rm + momentum * m */
        __m256 new_rm = _mm256_fmadd_ps(vmom, m, _mm256_mul_ps(v1m, _mm256_loadu_ps(rm + c)));
        /* new_rv = one_m * rv + momentum * var * scale_var */
        __m256 new_rv = _mm256_fmadd_ps(vmom, _mm256_mul_ps(var, vsv), _mm256_mul_ps(v1m, _mm256_loadu_ps(rv + c)));
        _mm256_storeu_ps(rm + c, new_rm);
        _mm256_storeu_ps(rv + c, new_rv);
    }
    for (; c < C; c++) {
        float var_i = 1.0f / (rstd[c] * rstd[c]) - eps;
        if (var_i < 0.0f) var_i = 0.0f;
        rm[c] = one_m * rm[c] + momentum * mean[c];
        rv[c] = one_m * rv[c] + momentum * var_i * scale_var;
    }
#else
    for (int c = 0; c < C; c++) {
        float var_i = 1.0f / (rstd[c] * rstd[c]) - eps;
        if (var_i < 0.0f) var_i = 0.0f;
        rm[c] = one_m * rm[c] + momentum * mean[c];
        rv[c] = one_m * rv[c] + momentum * var_i * scale_var;
    }
#endif
    PyBuffer_Release(&vrv); PyBuffer_Release(&vrm); PyBuffer_Release(&vrs); PyBuffer_Release(&vm);
    Py_RETURN_NONE;
}

/* Backward: dy (N,C), x_hat (N,C), rstd (C), gamma (C) -> dx (N,C), dgamma (C), dbeta (C) */
static PyObject* py_batch_norm_backward_f32(PyObject *self, PyObject *args) {
    PyObject *dy_obj, *xhat_obj, *rstd_obj, *gamma_obj, *dx_obj, *dg_obj, *db_obj;
    int N, C;
    if (!PyArg_ParseTuple(args, "OOOOOOOii",
                          &dy_obj, &xhat_obj, &rstd_obj, &gamma_obj,
                          &dx_obj, &dg_obj, &db_obj, &N, &C)) return NULL;
    Py_buffer vdy, vxh, vrs, vgm, vdx, vdg, vdb;
    if (PyObject_GetBuffer(dy_obj,    &vdy, PyBUF_SIMPLE) < 0) return NULL;
    if (PyObject_GetBuffer(xhat_obj,  &vxh, PyBUF_SIMPLE) < 0) { PyBuffer_Release(&vdy); return NULL; }
    if (PyObject_GetBuffer(rstd_obj,  &vrs, PyBUF_SIMPLE) < 0) { PyBuffer_Release(&vdy); PyBuffer_Release(&vxh); return NULL; }
    int has_g = (gamma_obj != Py_None);
    if (has_g && PyObject_GetBuffer(gamma_obj, &vgm, PyBUF_SIMPLE) < 0) { PyBuffer_Release(&vdy); PyBuffer_Release(&vxh); PyBuffer_Release(&vrs); return NULL; }
    if (PyObject_GetBuffer(dx_obj, &vdx, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) { PyBuffer_Release(&vdy); PyBuffer_Release(&vxh); PyBuffer_Release(&vrs); if (has_g) PyBuffer_Release(&vgm); return NULL; }
    if (PyObject_GetBuffer(dg_obj, &vdg, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) { PyBuffer_Release(&vdy); PyBuffer_Release(&vxh); PyBuffer_Release(&vrs); if (has_g) PyBuffer_Release(&vgm); PyBuffer_Release(&vdx); return NULL; }
    if (PyObject_GetBuffer(db_obj, &vdb, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) { PyBuffer_Release(&vdy); PyBuffer_Release(&vxh); PyBuffer_Release(&vrs); if (has_g) PyBuffer_Release(&vgm); PyBuffer_Release(&vdx); PyBuffer_Release(&vdg); return NULL; }
    const float *dy   = (const float*)vdy.buf;
    const float *xhat = (const float*)vxh.buf;
    const float *rstd = (const float*)vrs.buf;
    const float *gam  = has_g ? (const float*)vgm.buf : NULL;
    float *dx   = (float*)vdx.buf;
    float *dgam = (float*)vdg.buf;
    float *dbeta= (float*)vdb.buf;
    float inv_N = 1.0f / (float)N;
    for (int c = 0; c < C; c++) {
        float g = has_g ? gam[c] : 1.0f;
        float rs = rstd[c];
        /* accumulate dgamma and dbeta */
        float dg_acc = 0.0f, db_acc = 0.0f;
        float sum_dy = 0.0f, sum_dy_xhat = 0.0f;
        for (int n = 0; n < N; n++) {
            float d = dy[(Py_ssize_t)n * C + c];
            float xh = xhat[(Py_ssize_t)n * C + c];
            dg_acc += d * xh; db_acc += d;
            sum_dy += d; sum_dy_xhat += d * xh;
        }
        dgam[c]  = dg_acc;
        dbeta[c] = db_acc;
        float m_dy = sum_dy * inv_N;
        float m_dy_xhat = sum_dy_xhat * inv_N;
        for (int n = 0; n < N; n++) {
            float d  = dy[(Py_ssize_t)n * C + c];
            float xh = xhat[(Py_ssize_t)n * C + c];
            dx[(Py_ssize_t)n * C + c] = g * rs * (d - m_dy - xh * m_dy_xhat);
        }
    }
    PyBuffer_Release(&vdb); PyBuffer_Release(&vdg); PyBuffer_Release(&vdx);
    if (has_g) PyBuffer_Release(&vgm);
    PyBuffer_Release(&vrs); PyBuffer_Release(&vxh); PyBuffer_Release(&vdy);
    Py_RETURN_NONE;
}

/* ── RMS Norm (used in LLaMA-style transformers) ─────────────────────────── */
static PyObject* py_rms_norm_f32(PyObject *self, PyObject *args) {
    PyObject *src_obj, *dst_obj, *weight_obj;
    int rows, norm_size; float eps;
    if (!PyArg_ParseTuple(args, "OOOiif",
                          &src_obj, &dst_obj, &weight_obj, &rows, &norm_size, &eps)) return NULL;
    Py_buffer vs, vd, vw;
    int has_w = (weight_obj != Py_None);
    if (PyObject_GetBuffer(src_obj, &vs, PyBUF_SIMPLE) < 0) return NULL;
    if (PyObject_GetBuffer(dst_obj, &vd, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&vs); return NULL;
    }
    if (has_w && PyObject_GetBuffer(weight_obj, &vw, PyBUF_SIMPLE) < 0) {
        PyBuffer_Release(&vs); PyBuffer_Release(&vd); return NULL;
    }
    const float *src    = (const float*)vs.buf;
    float *dst          = (float*)vd.buf;
    const float *weight = has_w ? (const float*)vw.buf : NULL;
    for (int r = 0; r < rows; r++) {
        const float *row = src + (Py_ssize_t)r * norm_size;
        float *drow      = dst + (Py_ssize_t)r * norm_size;
        /* E[x^2] via AVX2 */
        float ss = 0.0f;
        Py_ssize_t i = 0;
#if VP_AVX2
        __m256 vacc = _mm256_setzero_ps();
        for (; i <= norm_size - 8; i += 8) {
            __m256 x = _mm256_loadu_ps(row + i);
            vacc = _mm256_fmadd_ps(x, x, vacc);
        }
        __m128 lo = _mm256_castps256_ps128(vacc);
        __m128 hi = _mm256_extractf128_ps(vacc, 1);
        lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
        ss = _mm_cvtss_f32(lo);
#endif
        for (; i < norm_size; i++) ss += row[i] * row[i];
        float inv_rms = 1.0f / sqrtf(ss / (float)norm_size + eps);
        i = 0;
#if VP_AVX2
        __m256 vir = _mm256_set1_ps(inv_rms);
        for (; i <= norm_size - 8; i += 8) {
            __m256 x  = _mm256_mul_ps(_mm256_loadu_ps(row + i), vir);
            if (weight) x = _mm256_mul_ps(x, _mm256_loadu_ps(weight + i));
            _mm256_storeu_ps(drow + i, x);
        }
#endif
        for (; i < norm_size; i++) {
            float x = row[i] * inv_rms;
            drow[i] = weight ? x * weight[i] : x;
        }
    }
    if (has_w) PyBuffer_Release(&vw);
    PyBuffer_Release(&vd); PyBuffer_Release(&vs);
    Py_RETURN_NONE;
}

/* ── Fused Linear + ReLU ─────────────────────────────────────────────────── */
/* linear_relu_f32(x, wT, bias_or_None, out, N, in_f, out_f)
   Computes out[N,out_f] = relu(x[N,in_f] @ wT[in_f,out_f] + bias[out_f])
   wT is already transposed (shape in_f x out_f, row-major). */
static PyObject* py_linear_relu_f32(PyObject *self, PyObject *args) {
    PyObject *x_obj, *wT_obj, *bias_obj, *out_obj;
    int N, in_f, out_f;
    if (!PyArg_ParseTuple(args, "OOOOiii", &x_obj, &wT_obj, &bias_obj, &out_obj, &N, &in_f, &out_f)) return NULL;
    Py_buffer vx, vw, vo; Py_buffer vbias;
    if (PyObject_GetBuffer(x_obj,   &vx, PyBUF_SIMPLE) < 0) return NULL;
    if (PyObject_GetBuffer(wT_obj,  &vw, PyBUF_SIMPLE) < 0) { PyBuffer_Release(&vx); return NULL; }
    if (PyObject_GetBuffer(out_obj, &vo, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) { PyBuffer_Release(&vx); PyBuffer_Release(&vw); return NULL; }
    int has_b = (bias_obj != Py_None);
    if (has_b && PyObject_GetBuffer(bias_obj, &vbias, PyBUF_SIMPLE) < 0) { PyBuffer_Release(&vx); PyBuffer_Release(&vw); PyBuffer_Release(&vo); return NULL; }
    const float *x = (const float*)vx.buf, *wT = (const float*)vw.buf;
    const float *bias = has_b ? (const float*)vbias.buf : NULL;
    float *out = (float*)vo.buf;
    /* GEMM: out = x @ wT  (wT is [in_f, out_f] row-major = W^T where W is [out_f, in_f]) */
    vp_matmul_f32_dispatch(x, wT, out, N, in_f, out_f, VP_POST_NONE);
    /* Fused bias + ReLU in one AVX2 pass */
#if VP_AVX2
    const __m256 vzero = _mm256_setzero_ps();
    for (int n = 0; n < N; n++) {
        float *row = out + (Py_ssize_t)n * out_f;
        Py_ssize_t i = 0;
        for (; i <= out_f - 8; i += 8) {
            __m256 v = _mm256_loadu_ps(row + i);
            if (bias) v = _mm256_add_ps(v, _mm256_loadu_ps(bias + i));
            _mm256_storeu_ps(row + i, _mm256_max_ps(v, vzero));
        }
        for (; i < out_f; i++) {
            float v = row[i] + (bias ? bias[i] : 0.0f);
            row[i] = v > 0.0f ? v : 0.0f;
        }
    }
#else
    for (int n = 0; n < N; n++) {
        float *row = out + (Py_ssize_t)n * out_f;
        for (int i = 0; i < out_f; i++) {
            float v = row[i] + (bias ? bias[i] : 0.0f);
            row[i] = v > 0.0f ? v : 0.0f;
        }
    }
#endif
    if (has_b) PyBuffer_Release(&vbias);
    PyBuffer_Release(&vo); PyBuffer_Release(&vw); PyBuffer_Release(&vx);
    Py_RETURN_NONE;
}

/* ── Transpose (cache-oblivious blocked) ──────────────────────────────────── */
/* AVX2 8×8 in-register transpose.
 * Load 8 rows × 8 floats, transpose in-place using 16 shuffles, store 8 columns. */
#if VP_AVX2
static VP_INLINE void vp_transpose_8x8(
    const float* VP_RESTRICT src, int src_stride,
    float*       VP_RESTRICT dst, int dst_stride)
{
    /* Load 8 rows of 8 floats each */
    __m256 r0 = _mm256_loadu_ps(src + 0 * src_stride);
    __m256 r1 = _mm256_loadu_ps(src + 1 * src_stride);
    __m256 r2 = _mm256_loadu_ps(src + 2 * src_stride);
    __m256 r3 = _mm256_loadu_ps(src + 3 * src_stride);
    __m256 r4 = _mm256_loadu_ps(src + 4 * src_stride);
    __m256 r5 = _mm256_loadu_ps(src + 5 * src_stride);
    __m256 r6 = _mm256_loadu_ps(src + 6 * src_stride);
    __m256 r7 = _mm256_loadu_ps(src + 7 * src_stride);
    /* Unpack pairs into interleaved lo/hi halves */
    __m256 t0 = _mm256_unpacklo_ps(r0, r1);
    __m256 t1 = _mm256_unpackhi_ps(r0, r1);
    __m256 t2 = _mm256_unpacklo_ps(r2, r3);
    __m256 t3 = _mm256_unpackhi_ps(r2, r3);
    __m256 t4 = _mm256_unpacklo_ps(r4, r5);
    __m256 t5 = _mm256_unpackhi_ps(r4, r5);
    __m256 t6 = _mm256_unpacklo_ps(r6, r7);
    __m256 t7 = _mm256_unpackhi_ps(r6, r7);
    /* Shuffle 32-bit groups */
    __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
    __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
    __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
    __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
    __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xEE);
    /* Permute 128-bit lanes to bring lo/hi into correct output rows */
    _mm256_storeu_ps(dst + 0 * dst_stride, _mm256_permute2f128_ps(s0, s4, 0x20));
    _mm256_storeu_ps(dst + 1 * dst_stride, _mm256_permute2f128_ps(s1, s5, 0x20));
    _mm256_storeu_ps(dst + 2 * dst_stride, _mm256_permute2f128_ps(s2, s6, 0x20));
    _mm256_storeu_ps(dst + 3 * dst_stride, _mm256_permute2f128_ps(s3, s7, 0x20));
    _mm256_storeu_ps(dst + 4 * dst_stride, _mm256_permute2f128_ps(s0, s4, 0x31));
    _mm256_storeu_ps(dst + 5 * dst_stride, _mm256_permute2f128_ps(s1, s5, 0x31));
    _mm256_storeu_ps(dst + 6 * dst_stride, _mm256_permute2f128_ps(s2, s6, 0x31));
    _mm256_storeu_ps(dst + 7 * dst_stride, _mm256_permute2f128_ps(s3, s7, 0x31));
}
#endif

#define TRANSPOSE_TILE 64
static PyObject* py_transpose_f32(PyObject *self, PyObject *args) {
    PyObject *src_obj, *dst_obj; int rows, cols;
    if (!PyArg_ParseTuple(args, "OOii", &src_obj, &dst_obj, &rows, &cols)) return NULL;
    Py_buffer vs, vd;
    if (PyObject_GetBuffer(src_obj, &vs, PyBUF_SIMPLE) < 0) return NULL;
    if (PyObject_GetBuffer(dst_obj, &vd, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&vs); return NULL;
    }
    const float *src = (const float*)vs.buf;
    float *dst       = (float*)vd.buf;
    int tiles_r = (rows + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE;
    int tiles_c = (cols + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE;
    int ntiles  = tiles_r * tiles_c;
    int tile_idx;
#ifdef _OPENMP
    VP_PRAGMA(omp parallel for schedule(static) if(ntiles > 4))
    for (tile_idx = 0; tile_idx < ntiles; tile_idx++) {
        int ii = (tile_idx / tiles_c) * TRANSPOSE_TILE;
        int jj = (tile_idx % tiles_c) * TRANSPOSE_TILE;
#else
    for (int ii = 0; ii < rows; ii += TRANSPOSE_TILE)
    for (int jj = 0; jj < cols; jj += TRANSPOSE_TILE) {
#endif
        int imax = ii + TRANSPOSE_TILE < rows ? ii + TRANSPOSE_TILE : rows;
        int jmax = jj + TRANSPOSE_TILE < cols ? jj + TRANSPOSE_TILE : cols;
        int ti = ii, tj = jj;
#if VP_AVX2
        /* Fast path: 8×8 AVX2 blocks within each tile */
        for (ti = ii; ti + 8 <= imax; ti += 8) {
            int tj2 = jj;
            for (; tj2 + 8 <= jmax; tj2 += 8) {
                vp_transpose_8x8(
                    src + (Py_ssize_t)ti * cols + tj2, cols,
                    dst + (Py_ssize_t)tj2 * rows + ti, rows);
            }
            /* Scalar tail for remaining cols */
            for (int i = ti; i < ti + 8; i++)
                for (int j = tj2; j < jmax; j++)
                    dst[(Py_ssize_t)j * rows + i] = src[(Py_ssize_t)i * cols + j];
        }
        /* Scalar tail for remaining rows */
        for (int i = ti; i < imax; i++)
            for (int j = jj; j < jmax; j++)
                dst[(Py_ssize_t)j * rows + i] = src[(Py_ssize_t)i * cols + j];
#else
        for (int i = ti; i < imax; i++)
            for (int j = tj; j < jmax; j++)
                dst[(Py_ssize_t)j * rows + i] = src[(Py_ssize_t)i * cols + j];
#endif
    }
    PyBuffer_Release(&vs); PyBuffer_Release(&vd);
    Py_RETURN_NONE;
}

/* ── Axial reductions: sum/max/min along axis 0 or 1 ─────────────────────── */
/* axis=1: sum across cols → dst[r] = sum(src[r,:])    shape [rows] */
/* axis=0: sum down rows  → dst[c] = sum(src[:,c])     shape [cols] */
static PyObject* py_sum_axis_f32(PyObject *self, PyObject *args) {
    PyObject *src_obj, *dst_obj; int rows, cols, axis;
    if (!PyArg_ParseTuple(args, "OOiii", &src_obj, &dst_obj, &rows, &cols, &axis)) return NULL;
    Py_buffer vs, vd;
    if (PyObject_GetBuffer(src_obj, &vs, PyBUF_SIMPLE) < 0) return NULL;
    if (PyObject_GetBuffer(dst_obj, &vd, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&vs); return NULL;
    }
    const float *src = (const float*)vs.buf;
    float *dst       = (float*)vd.buf;
    if (axis == 1) {
        /* sum each row */
        for (int r = 0; r < rows; r++)
            dst[r] = vp_sum_f32_kernel(src + (Py_ssize_t)r * cols, cols);
    } else {
        /* sum each column — stride access, do it in cache-friendly blocks */
        memset(dst, 0, (size_t)cols * sizeof(float));
        for (int r = 0; r < rows; r++) {
            const float *row = src + (Py_ssize_t)r * cols;
            Py_ssize_t j = 0;
#if VP_AVX2
            for (; j <= cols - 8; j += 8)
                _mm256_storeu_ps(dst + j,
                    _mm256_add_ps(_mm256_loadu_ps(dst + j), _mm256_loadu_ps(row + j)));
#endif
            for (; j < cols; j++) dst[j] += row[j];
        }
    }
    PyBuffer_Release(&vs); PyBuffer_Release(&vd);
    Py_RETURN_NONE;
}

static PyObject* py_max_axis_f32(PyObject *self, PyObject *args) {
    PyObject *src_obj, *dst_obj; int rows, cols, axis;
    if (!PyArg_ParseTuple(args, "OOiii", &src_obj, &dst_obj, &rows, &cols, &axis)) return NULL;
    Py_buffer vs, vd;
    if (PyObject_GetBuffer(src_obj, &vs, PyBUF_SIMPLE) < 0) return NULL;
    if (PyObject_GetBuffer(dst_obj, &vd, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&vs); return NULL;
    }
    const float *src = (const float*)vs.buf;
    float *dst       = (float*)vd.buf;
    if (axis == 1) {
        for (int r = 0; r < rows; r++)
            dst[r] = vp_max_f32_kernel(src + (Py_ssize_t)r * cols, cols);
    } else {
        for (int j = 0; j < cols; j++) dst[j] = -FLT_MAX;
        for (int r = 0; r < rows; r++) {
            const float *row = src + (Py_ssize_t)r * cols;
            Py_ssize_t j = 0;
#if VP_AVX2
            for (; j <= cols - 8; j += 8)
                _mm256_storeu_ps(dst + j, _mm256_max_ps(_mm256_loadu_ps(dst + j),
                                                         _mm256_loadu_ps(row + j)));
#endif
            for (; j < cols; j++) if (row[j] > dst[j]) dst[j] = row[j];
        }
    }
    PyBuffer_Release(&vs); PyBuffer_Release(&vd);
    Py_RETURN_NONE;
}

/* ── Where / conditional select ───────────────────────────────────────────── */
/* where_f32(cond, a, b, dst, n): dst[i] = a[i] if cond[i] > 0 else b[i] */
static PyObject* py_where_f32(PyObject *self, PyObject *args) {
    PyObject *cond_obj, *a_obj, *b_obj, *dst_obj;
    if (!PyArg_ParseTuple(args, "OOOO", &cond_obj, &a_obj, &b_obj, &dst_obj)) return NULL;
    Py_buffer vc, va, vb, vd;
    if (PyObject_GetBuffer(cond_obj, &vc, PyBUF_SIMPLE)            < 0) return NULL;
    if (PyObject_GetBuffer(a_obj,    &va, PyBUF_SIMPLE)            < 0) { PyBuffer_Release(&vc); return NULL; }
    if (PyObject_GetBuffer(b_obj,    &vb, PyBUF_SIMPLE)            < 0) { PyBuffer_Release(&vc); PyBuffer_Release(&va); return NULL; }
    if (PyObject_GetBuffer(dst_obj,  &vd, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&vc); PyBuffer_Release(&va); PyBuffer_Release(&vb); return NULL;
    }
    const float *cond = (const float*)vc.buf;
    const float *a    = (const float*)va.buf;
    const float *b    = (const float*)vb.buf;
    float *dst        = (float*)vd.buf;
    Py_ssize_t n = vc.len / sizeof(float);
    Py_ssize_t i = 0;
#if VP_AVX2
    const __m256 zero = _mm256_setzero_ps();
    for (; i <= n - 8; i += 8) {
        __m256 mask = _mm256_cmp_ps(_mm256_loadu_ps(cond + i), zero, _CMP_GT_OS);
        __m256 res  = _mm256_blendv_ps(_mm256_loadu_ps(b + i), _mm256_loadu_ps(a + i), mask);
        _mm256_storeu_ps(dst + i, res);
    }
#endif
    for (; i < n; i++) dst[i] = cond[i] > 0.0f ? a[i] : b[i];
    PyBuffer_Release(&vc); PyBuffer_Release(&va); PyBuffer_Release(&vb); PyBuffer_Release(&vd);
    Py_RETURN_NONE;
}

/* ── Comparison ops (return 1.0 or 0.0 float mask) ─────────────────────── */
#if VP_AVX2
#define CMP_OP_AVX2(NAME, AVX_CMP, SCALAR_CMP)                                \
static PyObject* py_##NAME##_f32(PyObject *self, PyObject *args) {            \
    PyObject *a_obj, *b_obj, *dst_obj;                                        \
    if (!PyArg_ParseTuple(args, "OOO", &a_obj, &b_obj, &dst_obj)) return NULL;\
    Py_buffer va, vb, vd;                                                     \
    if (PyObject_GetBuffer(a_obj,   &va, PyBUF_SIMPLE)            < 0) return NULL; \
    if (PyObject_GetBuffer(b_obj,   &vb, PyBUF_SIMPLE)            < 0) { PyBuffer_Release(&va); return NULL; } \
    if (PyObject_GetBuffer(dst_obj, &vd, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) { \
        PyBuffer_Release(&va); PyBuffer_Release(&vb); return NULL; }          \
    const float *a = (const float*)va.buf;                                    \
    const float *b = (const float*)vb.buf;                                    \
    float *dst     = (float*)vd.buf;                                          \
    Py_ssize_t n = va.len / sizeof(float), i = 0;                             \
    const __m256 one = _mm256_set1_ps(1.0f);                                  \
    const __m256 zer = _mm256_setzero_ps();                                   \
    for (; i <= n - 8; i += 8) {                                              \
        __m256 mask = _mm256_cmp_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),(AVX_CMP)); \
        _mm256_storeu_ps(dst + i, _mm256_blendv_ps(zer, one, mask));          \
    }                                                                         \
    for (; i < n; i++) dst[i] = (a[i] SCALAR_CMP b[i]) ? 1.0f : 0.0f;       \
    PyBuffer_Release(&va); PyBuffer_Release(&vb); PyBuffer_Release(&vd);      \
    Py_RETURN_NONE; \
}
#else
#define CMP_OP_AVX2(NAME, AVX_CMP, SCALAR_CMP)                                \
static PyObject* py_##NAME##_f32(PyObject *self, PyObject *args) {            \
    PyObject *a_obj, *b_obj, *dst_obj;                                        \
    if (!PyArg_ParseTuple(args, "OOO", &a_obj, &b_obj, &dst_obj)) return NULL;\
    Py_buffer va, vb, vd;                                                     \
    if (PyObject_GetBuffer(a_obj,   &va, PyBUF_SIMPLE)            < 0) return NULL; \
    if (PyObject_GetBuffer(b_obj,   &vb, PyBUF_SIMPLE)            < 0) { PyBuffer_Release(&va); return NULL; } \
    if (PyObject_GetBuffer(dst_obj, &vd, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) { \
        PyBuffer_Release(&va); PyBuffer_Release(&vb); return NULL; }          \
    const float *a = (const float*)va.buf;                                    \
    const float *b = (const float*)vb.buf;                                    \
    float *dst     = (float*)vd.buf;                                          \
    Py_ssize_t n = va.len / sizeof(float);                                    \
    for (Py_ssize_t i = 0; i < n; i++) dst[i] = (a[i] SCALAR_CMP b[i]) ? 1.0f : 0.0f; \
    PyBuffer_Release(&va); PyBuffer_Release(&vb); PyBuffer_Release(&vd);      \
    Py_RETURN_NONE; \
}
#endif
CMP_OP_AVX2(eq,  _CMP_EQ_OS,  ==)
CMP_OP_AVX2(ne,  _CMP_NEQ_OS, !=)
CMP_OP_AVX2(lt,  _CMP_LT_OS,  <)
CMP_OP_AVX2(le,  _CMP_LE_OS,  <=)
CMP_OP_AVX2(gt,  _CMP_GT_OS,  >)
CMP_OP_AVX2(ge,  _CMP_GE_OS,  >=)

/* ── Activation backward passes ─────────────────────────────────────────── */
/* relu_backward: grad_input = grad_output * (output > 0) */
static PyObject* py_relu_backward_f32(PyObject *self, PyObject *args) {
    PyObject *go_obj, *out_obj, *gi_obj;
    if (!PyArg_ParseTuple(args, "OOO", &go_obj, &out_obj, &gi_obj)) return NULL;
    Py_buffer vgo, vo, vgi;
    if (PyObject_GetBuffer(go_obj,  &vgo, PyBUF_SIMPLE)            < 0) return NULL;
    if (PyObject_GetBuffer(out_obj, &vo,  PyBUF_SIMPLE)            < 0) { PyBuffer_Release(&vgo); return NULL; }
    if (PyObject_GetBuffer(gi_obj,  &vgi, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&vgo); PyBuffer_Release(&vo); return NULL;
    }
    const float *go  = (const float*)vgo.buf;
    const float *out = (const float*)vo.buf;
    float *gi        = (float*)vgi.buf;
    Py_ssize_t n = vgo.len / sizeof(float), i = 0;
#if VP_AVX2
    const __m256 zero = _mm256_setzero_ps();
    for (; i <= n - 8; i += 8) {
        __m256 mask = _mm256_cmp_ps(_mm256_loadu_ps(out + i), zero, _CMP_GT_OS);
        _mm256_storeu_ps(gi + i, _mm256_and_ps(_mm256_loadu_ps(go + i), mask));
    }
#endif
    for (; i < n; i++) gi[i] = out[i] > 0.0f ? go[i] : 0.0f;
    PyBuffer_Release(&vgo); PyBuffer_Release(&vo); PyBuffer_Release(&vgi);
    Py_RETURN_NONE;
}

/* sigmoid_backward: grad = go * sigmoid_out * (1 - sigmoid_out) */
static PyObject* py_sigmoid_backward_f32(PyObject *self, PyObject *args) {
    PyObject *go_obj, *out_obj, *gi_obj;
    if (!PyArg_ParseTuple(args, "OOO", &go_obj, &out_obj, &gi_obj)) return NULL;
    Py_buffer vgo, vo, vgi;
    if (PyObject_GetBuffer(go_obj,  &vgo, PyBUF_SIMPLE)            < 0) return NULL;
    if (PyObject_GetBuffer(out_obj, &vo,  PyBUF_SIMPLE)            < 0) { PyBuffer_Release(&vgo); return NULL; }
    if (PyObject_GetBuffer(gi_obj,  &vgi, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&vgo); PyBuffer_Release(&vo); return NULL;
    }
    const float *go  = (const float*)vgo.buf;
    const float *out = (const float*)vo.buf;
    float *gi        = (float*)vgi.buf;
    Py_ssize_t n = vgo.len / sizeof(float), i = 0;
#if VP_AVX2
    const __m256 one = _mm256_set1_ps(1.0f);
    for (; i <= n - 8; i += 8) {
        __m256 s  = _mm256_loadu_ps(out + i);
        __m256 ds = _mm256_mul_ps(s, _mm256_sub_ps(one, s));
        _mm256_storeu_ps(gi + i, _mm256_mul_ps(_mm256_loadu_ps(go + i), ds));
    }
#endif
    for (; i < n; i++) { float s = out[i]; gi[i] = go[i] * s * (1.0f - s); }
    PyBuffer_Release(&vgo); PyBuffer_Release(&vo); PyBuffer_Release(&vgi);
    Py_RETURN_NONE;
}

/* tanh_backward: grad = go * (1 - tanh_out^2) */
static PyObject* py_tanh_backward_f32(PyObject *self, PyObject *args) {
    PyObject *go_obj, *out_obj, *gi_obj;
    if (!PyArg_ParseTuple(args, "OOO", &go_obj, &out_obj, &gi_obj)) return NULL;
    Py_buffer vgo, vo, vgi;
    if (PyObject_GetBuffer(go_obj,  &vgo, PyBUF_SIMPLE)            < 0) return NULL;
    if (PyObject_GetBuffer(out_obj, &vo,  PyBUF_SIMPLE)            < 0) { PyBuffer_Release(&vgo); return NULL; }
    if (PyObject_GetBuffer(gi_obj,  &vgi, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&vgo); PyBuffer_Release(&vo); return NULL;
    }
    const float *go  = (const float*)vgo.buf;
    const float *out = (const float*)vo.buf;
    float *gi        = (float*)vgi.buf;
    Py_ssize_t n = vgo.len / sizeof(float), i = 0;
#if VP_AVX2
    const __m256 one = _mm256_set1_ps(1.0f);
    for (; i <= n - 8; i += 8) {
        __m256 t  = _mm256_loadu_ps(out + i);
        __m256 dt = _mm256_fnmadd_ps(t, t, one); /* 1 - t^2 */
        _mm256_storeu_ps(gi + i, _mm256_mul_ps(_mm256_loadu_ps(go + i), dt));
    }
#endif
    for (; i < n; i++) { float t = out[i]; gi[i] = go[i] * (1.0f - t * t); }
    PyBuffer_Release(&vgo); PyBuffer_Release(&vo); PyBuffer_Release(&vgi);
    Py_RETURN_NONE;
}

/* ── Element-wise pow: dst = a^b (both tensors) ──────────────────────────── */
/* Uses exp(b * log(a)) under AVX2 — calls our fast log + fast exp kernels */
static PyObject* py_pow_f32(PyObject *self, PyObject *args) {
    PyObject *a_obj, *b_obj, *dst_obj;
    if (!PyArg_ParseTuple(args, "OOO", &a_obj, &b_obj, &dst_obj)) return NULL;
    Py_buffer va, vb, vd;
    if (PyObject_GetBuffer(a_obj,   &va, PyBUF_SIMPLE)            < 0) return NULL;
    if (PyObject_GetBuffer(b_obj,   &vb, PyBUF_SIMPLE)            < 0) { PyBuffer_Release(&va); return NULL; }
    if (PyObject_GetBuffer(dst_obj, &vd, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&va); PyBuffer_Release(&vb); return NULL;
    }
    const float *a = (const float*)va.buf;
    const float *b = (const float*)vb.buf;
    float *dst     = (float*)vd.buf;
    Py_ssize_t n = va.len / sizeof(float), i = 0;
#if VP_AVX2
    for (; i <= n - 8; i += 8) {
        __m256 loga = avx2_logf(_mm256_loadu_ps(a + i));
        __m256 bloga = _mm256_mul_ps(_mm256_loadu_ps(b + i), loga);
        _mm256_storeu_ps(dst + i, avx2_expf(bloga));
    }
#endif
    for (; i < n; i++) dst[i] = powf(a[i], b[i]);
    PyBuffer_Release(&va); PyBuffer_Release(&vb); PyBuffer_Release(&vd);
    Py_RETURN_NONE;
}

/* ── Softmax backward: grad_input = softmax_out * (grad_out - dot(grad_out, softmax_out)) */
static PyObject* py_softmax_backward_f32(PyObject *self, PyObject *args) {
    PyObject *go_obj, *out_obj, *gi_obj;
    int rows, cols;
    if (!PyArg_ParseTuple(args, "OOOii", &go_obj, &out_obj, &gi_obj, &rows, &cols)) return NULL;
    Py_buffer vgo, vo, vgi;
    if (PyObject_GetBuffer(go_obj,  &vgo, PyBUF_SIMPLE)            < 0) return NULL;
    if (PyObject_GetBuffer(out_obj, &vo,  PyBUF_SIMPLE)            < 0) { PyBuffer_Release(&vgo); return NULL; }
    if (PyObject_GetBuffer(gi_obj,  &vgi, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&vgo); PyBuffer_Release(&vo); return NULL;
    }
    const float *go  = (const float*)vgo.buf;
    const float *out = (const float*)vo.buf;
    float *gi        = (float*)vgi.buf;
    for (int r = 0; r < rows; r++) {
        const float *gorow  = go  + (Py_ssize_t)r * cols;
        const float *outrow = out + (Py_ssize_t)r * cols;
        float *girow        = gi  + (Py_ssize_t)r * cols;
        /* dot = sum(go * out) for this row */
        float dot = 0.0f;
        Py_ssize_t i = 0;
#if VP_AVX2
        __m256 vdot = _mm256_setzero_ps();
        for (; i <= cols - 8; i += 8)
            vdot = _mm256_fmadd_ps(_mm256_loadu_ps(gorow + i),
                                    _mm256_loadu_ps(outrow + i), vdot);
        __m128 lo = _mm256_castps256_ps128(vdot);
        __m128 hi = _mm256_extractf128_ps(vdot, 1);
        lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
        dot = _mm_cvtss_f32(lo);
#endif
        for (; i < cols; i++) dot += gorow[i] * outrow[i];
        /* gi = out * (go - dot) */
        i = 0;
#if VP_AVX2
        __m256 vd = _mm256_set1_ps(dot);
        for (; i <= cols - 8; i += 8)
            _mm256_storeu_ps(girow + i,
                _mm256_mul_ps(_mm256_loadu_ps(outrow + i),
                              _mm256_sub_ps(_mm256_loadu_ps(gorow + i), vd)));
#endif
        for (; i < cols; i++) girow[i] = outrow[i] * (gorow[i] - dot);
    }
    PyBuffer_Release(&vgo); PyBuffer_Release(&vo); PyBuffer_Release(&vgi);
    Py_RETURN_NONE;
}

/* ── Cross-entropy loss (fused NLL + log_softmax) ────────────────────────── */
/* cross_entropy_f32(logits, targets, out_loss, rows, cols)
 * targets: int32 buffer of class indices; out_loss: scalar float buffer */
static PyObject* py_cross_entropy_f32(PyObject *self, PyObject *args) {
    PyObject *logits_obj, *targets_obj, *loss_obj;
    int rows, cols;
    if (!PyArg_ParseTuple(args, "OOOii", &logits_obj, &targets_obj, &loss_obj, &rows, &cols)) return NULL;
    Py_buffer vl, vt, vo;
    if (PyObject_GetBuffer(logits_obj,  &vl, PyBUF_SIMPLE)            < 0) return NULL;
    if (PyObject_GetBuffer(targets_obj, &vt, PyBUF_SIMPLE)            < 0) { PyBuffer_Release(&vl); return NULL; }
    if (PyObject_GetBuffer(loss_obj,    &vo, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&vl); PyBuffer_Release(&vt); return NULL;
    }
    const float   *logits  = (const float*)vl.buf;
    const int32_t *targets = (const int32_t*)vt.buf;
    float *loss_out        = (float*)vo.buf;
    float total_loss = 0.0f;
    for (int r = 0; r < rows; r++) {
        const float *row = logits + (Py_ssize_t)r * cols;
        float m = vp_max_f32_kernel(row, cols);
        float acc = 0.0f;
        Py_ssize_t j = 0;
#if VP_AVX2
        __m256 vm = _mm256_set1_ps(m);
        __m256 vacc = _mm256_setzero_ps();
        for (; j <= cols - 8; j += 8)
            vacc = _mm256_add_ps(vacc, avx2_expf_fast(_mm256_sub_ps(_mm256_loadu_ps(row + j), vm)));
        __m128 lo = _mm256_castps256_ps128(vacc);
        __m128 hi = _mm256_extractf128_ps(vacc, 1);
        lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
        acc = _mm_cvtss_f32(lo);
#endif
        for (; j < cols; j++) acc += vp_expf(row[j] - m);
        int t = targets[r];
        if (t >= 0 && t < cols)
            total_loss -= (row[t] - m) - vp_logf(acc);
    }
    loss_out[0] = total_loss / (float)rows;
    PyBuffer_Release(&vl); PyBuffer_Release(&vt); PyBuffer_Release(&vo);
    Py_RETURN_NONE;
}

/* ══════════════════════════════════════════════════════════════════════════════
 * OPTIMIZER KERNELS — fused single-pass AVX2
 * ══════════════════════════════════════════════════════════════════════════════
 *
 * adam_step_f32: fused AdamW parameter update — one pass over all params.
 *   p  -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * p)
 *   m   = β1*m + (1-β1)*g
 *   v   = β2*v + (1-β2)*g²
 *
 * All six arrays (p, g, m, v) are modified in place.
 * bias-correction factors bc1 = 1-β1^t, bc2 = 1-β2^t are pre-computed by caller.
 */
static PyObject* py_adam_step_f32(PyObject *self, PyObject *args) {
    PyObject *p_obj, *g_obj, *m_obj, *v_obj;
    float lr, beta1, beta2, eps, wd, bc1, bc2;
    if (!PyArg_ParseTuple(args, "OOOOfffffff",
                          &p_obj, &g_obj, &m_obj, &v_obj,
                          &lr, &beta1, &beta2, &eps, &wd, &bc1, &bc2))
        return NULL;

    Py_buffer vp2, vg, vm2, vv;
    if (PyObject_GetBuffer(p_obj,  &vp2, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) return NULL;
    if (PyObject_GetBuffer(g_obj,  &vg,  PyBUF_SIMPLE) < 0)
        { PyBuffer_Release(&vp2); return NULL; }
    if (PyObject_GetBuffer(m_obj,  &vm2, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0)
        { PyBuffer_Release(&vp2); PyBuffer_Release(&vg); return NULL; }
    if (PyObject_GetBuffer(v_obj,  &vv,  PyBUF_SIMPLE|PyBUF_WRITABLE) < 0)
        { PyBuffer_Release(&vp2); PyBuffer_Release(&vg); PyBuffer_Release(&vm2); return NULL; }

    float* VP_RESTRICT p = (float*)vp2.buf;
    const float* VP_RESTRICT g = (const float*)vg.buf;
    float* VP_RESTRICT m = (float*)vm2.buf;
    float* VP_RESTRICT v = (float*)vv.buf;
    Py_ssize_t n = vp2.len / sizeof(float);

    float inv_bc1 = 1.0f / bc1;
    float inv_bc2 = 1.0f / bc2;
    float one_m_b1 = 1.0f - beta1;
    float one_m_b2 = 1.0f - beta2;

#if VP_AVX2
    __m256 vb1   = _mm256_set1_ps(beta1);
    __m256 vb2   = _mm256_set1_ps(beta2);
    __m256 vomb1 = _mm256_set1_ps(one_m_b1);
    __m256 vomb2 = _mm256_set1_ps(one_m_b2);
    __m256 vibc1 = _mm256_set1_ps(inv_bc1);
    __m256 vibc2 = _mm256_set1_ps(inv_bc2);
    __m256 veps  = _mm256_set1_ps(eps);
    __m256 vlr   = _mm256_set1_ps(lr);
    __m256 vwd   = _mm256_set1_ps(wd);

    Py_ssize_t i = 0;
    int b32 = (int)(n / 32), b;
    VP_OMP_FOR(n)
    for (b = 0; b < b32; b++) {
        i = (Py_ssize_t)b * 32;
        _mm_prefetch((const char*)(g + i + 256), _MM_HINT_T0);
        _mm_prefetch((const char*)(p + i + 256), _MM_HINT_T0);
        /* Unroll 4x — independent chains, hides FMA latency */
        for (int u = 0; u < 4; u++) {
            Py_ssize_t j = i + u * 8;
            __m256 vg_ = _mm256_loadu_ps(g + j);
            __m256 vm_ = _mm256_loadu_ps(m + j);
            __m256 vv_ = _mm256_loadu_ps(v + j);
            __m256 vp_ = _mm256_loadu_ps(p + j);
            /* m = β1*m + (1-β1)*g */
            vm_ = _mm256_fmadd_ps(vb1, vm_, _mm256_mul_ps(vomb1, vg_));
            /* v = β2*v + (1-β2)*g² */
            vv_ = _mm256_fmadd_ps(vb2, vv_, _mm256_mul_ps(vomb2, _mm256_mul_ps(vg_, vg_)));
            /* m_hat = m / bc1,  v_hat = v / bc2 */
            __m256 mhat = _mm256_mul_ps(vm_, vibc1);
            __m256 vhat = _mm256_mul_ps(vv_, vibc2);
            /* step = lr * (mhat / (sqrt(vhat) + eps) + wd*p) */
            __m256 denom = _mm256_add_ps(vp_avx2_sqrt(vhat), veps);
            __m256 upd = _mm256_fmadd_ps(vwd, vp_, _mm256_div_ps(mhat, denom));
            vp_ = _mm256_fnmadd_ps(vlr, upd, vp_);  /* p -= lr * upd */
            _mm256_storeu_ps(m + j, vm_);
            _mm256_storeu_ps(v + j, vv_);
            _mm256_storeu_ps(p + j, vp_);
        }
    }
    i = (Py_ssize_t)b32 * 32;
#else
    Py_ssize_t i = 0;
#endif
    /* scalar tail */
    for (; i < n; i++) {
        m[i] = beta1 * m[i] + one_m_b1 * g[i];
        v[i] = beta2 * v[i] + one_m_b2 * g[i] * g[i];
        float mhat = m[i] * inv_bc1;
        float vhat = v[i] * inv_bc2;
        p[i] -= lr * (mhat / (sqrtf(vhat) + eps) + wd * p[i]);
    }

    PyBuffer_Release(&vp2); PyBuffer_Release(&vg);
    PyBuffer_Release(&vm2); PyBuffer_Release(&vv);
    Py_RETURN_NONE;
}

/* clip_grad_norm_f32: compute L2 norm of all grads, then scale if > max_norm.
 * Returns the unclipped norm as a float. */
static PyObject* py_clip_grad_norm_f32(PyObject *self, PyObject *args) {
    PyObject *g_obj; float max_norm;
    if (!PyArg_ParseTuple(args, "Of", &g_obj, &max_norm)) return NULL;
    Py_buffer vg;
    if (PyObject_GetBuffer(g_obj, &vg, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) return NULL;
    float* VP_RESTRICT g = (float*)vg.buf;
    Py_ssize_t n = vg.len / sizeof(float);

    /* Compute sum of squares — 4x unrolled AVX2 */
    float norm2 = 0.0f;
    Py_ssize_t i = 0;
#if VP_AVX2
    __m256 va0 = _mm256_setzero_ps(), va1 = _mm256_setzero_ps();
    __m256 va2 = _mm256_setzero_ps(), va3 = _mm256_setzero_ps();
    for (; i <= n - 32; i += 32) {
        __m256 g0 = _mm256_loadu_ps(g + i);
        __m256 g1 = _mm256_loadu_ps(g + i + 8);
        __m256 g2 = _mm256_loadu_ps(g + i + 16);
        __m256 g3 = _mm256_loadu_ps(g + i + 24);
        va0 = _mm256_fmadd_ps(g0, g0, va0);
        va1 = _mm256_fmadd_ps(g1, g1, va1);
        va2 = _mm256_fmadd_ps(g2, g2, va2);
        va3 = _mm256_fmadd_ps(g3, g3, va3);
    }
    __m256 vacc = _mm256_add_ps(_mm256_add_ps(va0, va1), _mm256_add_ps(va2, va3));
    for (; i <= n - 8; i += 8) {
        __m256 gi = _mm256_loadu_ps(g + i);
        vacc = _mm256_fmadd_ps(gi, gi, vacc);
    }
    __m128 lo = _mm256_castps256_ps128(vacc);
    __m128 hi = _mm256_extractf128_ps(vacc, 1);
    lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
    norm2 = _mm_cvtss_f32(lo);
#endif
    for (; i < n; i++) norm2 += g[i] * g[i];

    float norm = sqrtf(norm2);
    if (norm > max_norm) {
        float scale = max_norm / (norm + 1e-6f);
        i = 0;
#if VP_AVX2
        __m256 vs = _mm256_set1_ps(scale);
        int b32 = (int)(n / 32), b;
        VP_OMP_FOR_LIGHT(n)
        for (b = 0; b < b32; b++) {
            Py_ssize_t j = (Py_ssize_t)b * 32;
            _mm256_storeu_ps(g + j,      _mm256_mul_ps(_mm256_loadu_ps(g + j),      vs));
            _mm256_storeu_ps(g + j + 8,  _mm256_mul_ps(_mm256_loadu_ps(g + j + 8),  vs));
            _mm256_storeu_ps(g + j + 16, _mm256_mul_ps(_mm256_loadu_ps(g + j + 16), vs));
            _mm256_storeu_ps(g + j + 24, _mm256_mul_ps(_mm256_loadu_ps(g + j + 24), vs));
        }
        i = (Py_ssize_t)b32 * 32;
#endif
        for (; i < n; i++) g[i] *= scale;
    }

    PyBuffer_Release(&vg);
    return PyFloat_FromDouble((double)norm);
}

/* addcmul_f32: dst = a + alpha * b * c  — fused multiply-add for grad accumulation */
static PyObject* py_addcmul_f32(PyObject *self, PyObject *args) {
    PyObject *a_obj, *b_obj, *c_obj, *dst_obj; float alpha;
    if (!PyArg_ParseTuple(args, "OOOfO", &a_obj, &b_obj, &c_obj, &alpha, &dst_obj))
        return NULL;
    Py_buffer va, vb, vc, vd;
    if (PyObject_GetBuffer(a_obj,   &va, PyBUF_SIMPLE) < 0) return NULL;
    if (PyObject_GetBuffer(b_obj,   &vb, PyBUF_SIMPLE) < 0) { PyBuffer_Release(&va); return NULL; }
    if (PyObject_GetBuffer(c_obj,   &vc, PyBUF_SIMPLE) < 0)
        { PyBuffer_Release(&va); PyBuffer_Release(&vb); return NULL; }
    if (PyObject_GetBuffer(dst_obj, &vd, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0)
        { PyBuffer_Release(&va); PyBuffer_Release(&vb); PyBuffer_Release(&vc); return NULL; }

    const float* VP_RESTRICT a   = (const float*)va.buf;
    const float* VP_RESTRICT b   = (const float*)vb.buf;
    const float* VP_RESTRICT c   = (const float*)vc.buf;
    float*       VP_RESTRICT dst = (float*)vd.buf;
    Py_ssize_t n = va.len / sizeof(float);

    Py_ssize_t i = 0;
#if VP_AVX2
    __m256 valpha = _mm256_set1_ps(alpha);
    int b32 = (int)(n / 32), bl;
    VP_OMP_FOR(n)
    for (bl = 0; bl < b32; bl++) {
        i = (Py_ssize_t)bl * 32;
        _mm_prefetch((const char*)(b + i + 256), _MM_HINT_T0);
        _mm_prefetch((const char*)(c + i + 256), _MM_HINT_T0);
        for (int u = 0; u < 4; u++) {
            Py_ssize_t j = i + u * 8;
            __m256 va_ = _mm256_loadu_ps(a + j);
            __m256 vb_ = _mm256_loadu_ps(b + j);
            __m256 vc_ = _mm256_loadu_ps(c + j);
            /* dst = a + alpha * b * c */
            _mm256_storeu_ps(dst + j, _mm256_fmadd_ps(valpha, _mm256_mul_ps(vb_, vc_), va_));
        }
    }
    i = (Py_ssize_t)b32 * 32;
#endif
    for (; i < n; i++) dst[i] = a[i] + alpha * b[i] * c[i];

    PyBuffer_Release(&va); PyBuffer_Release(&vb);
    PyBuffer_Release(&vc); PyBuffer_Release(&vd);
    Py_RETURN_NONE;
}

/* ── Bias broadcast add: out[n*C : (n+1)*C] += bias[C], for all N rows ─────── */
static PyObject* py_add_bias_rows_f32(PyObject *self, PyObject *args) {
    PyObject *out_obj, *bias_obj;
    int N, C;
    if (!PyArg_ParseTuple(args, "OOii", &out_obj, &bias_obj, &N, &C)) return NULL;
    Py_buffer vo, vb;
    if (PyObject_GetBuffer(out_obj,  &vo, PyBUF_SIMPLE|PyBUF_WRITABLE) < 0) return NULL;
    if (PyObject_GetBuffer(bias_obj, &vb, PyBUF_SIMPLE) < 0) { PyBuffer_Release(&vo); return NULL; }
    float       *out  = (float*)vo.buf;
    const float *bias = (const float*)vb.buf;
    for (int n = 0; n < N; n++) {
        float *row = out + (Py_ssize_t)n * C;
        Py_ssize_t i = 0;
#if VP_AVX2
        for (; i <= C - 8; i += 8) {
            __m256 vr = _mm256_loadu_ps(row + i);
            __m256 vbias = _mm256_loadu_ps(bias + i);
            _mm256_storeu_ps(row + i, _mm256_add_ps(vr, vbias));
        }
#endif
        for (; i < C; i++) row[i] += bias[i];
    }
    PyBuffer_Release(&vb); PyBuffer_Release(&vo);
    Py_RETURN_NONE;
}

/* ══════════════════════════════════════════════════════════════════════════════
 * METHOD TABLE
 * ══════════════════════════════════════════════════════════════════════════════ */

static PyMethodDef vp_core_methods[] = {
    /* Unary float32 */
    {"relu_f32",           py_relu,              METH_VARARGS, "relu(src, dst)"},
    {"exp_f32",            py_exp,               METH_VARARGS, "exp(src, dst)"},
    {"log_f32",            py_log,               METH_VARARGS, "log(src, dst)"},
    {"sigmoid_f32",        py_sigmoid,           METH_VARARGS, "sigmoid(src, dst)"},
    {"tanh_f32",           py_tanh_f,            METH_VARARGS, "tanh(src, dst)"},
    {"sqrt_f32",           py_sqrt,              METH_VARARGS, "sqrt(src, dst)"},
    {"abs_f32",            py_abs_f,             METH_VARARGS, "abs(src, dst)"},
    {"neg_f32",            py_neg,               METH_VARARGS, "neg(src, dst)"},
    {"sign_f32",           py_sign,              METH_VARARGS, "sign(src, dst)"},
    {"erf_f32",            py_erf,               METH_VARARGS, "erf(src, dst)"},
    {"gelu_f32",           py_gelu,              METH_VARARGS, "gelu(src, dst)"},
    {"silu_f32",           py_silu,              METH_VARARGS, "silu(src, dst)"},
    {"leaky_relu_f32",     py_leaky_relu_f32,    METH_VARARGS, "leaky_relu(src, dst, alpha)"},
    {"swiglu_f32",         py_swiglu_f32,        METH_VARARGS, "swiglu(src, gate, dst) — x*gate*sigmoid(gate)"},
    {"add_relu_f32",       py_add_relu_f32,      METH_VARARGS, "add_relu(a, b, dst) — fused residual+relu"},
    {"fused_chain_f32",   py_fused_chain_f32,  METH_VARARGS, "fused_chain(src, ops, dst) — single-pass unary chain"},
    /* Binary float32 */
    {"add_f32",            py_add,               METH_VARARGS, "add(a, b, dst)"},
    {"sub_f32",            py_sub,               METH_VARARGS, "sub(a, b, dst)"},
    {"mul_f32",            py_mul,               METH_VARARGS, "mul(a, b, dst)"},
    {"div_f32",            py_div,               METH_VARARGS, "div(a, b, dst)"},
    {"maximum_f32",        py_maximum,           METH_VARARGS, "maximum(a, b, dst)"},
    {"minimum_f32",        py_minimum,           METH_VARARGS, "minimum(a, b, dst)"},
    /* Scalar broadcast float32 */
    {"add_scalar_f32",     py_add_scalar,        METH_VARARGS, "add_scalar(a, s, dst)"},
    {"sub_scalar_f32",     py_sub_scalar,        METH_VARARGS, "sub_scalar(a, s, dst)"},
    {"mul_scalar_f32",     py_mul_scalar,        METH_VARARGS, "mul_scalar(a, s, dst)"},
    {"div_scalar_f32",     py_div_scalar,        METH_VARARGS, "div_scalar(a, s, dst)"},
    {"rdiv_scalar_f32",    py_rdiv_scalar,       METH_VARARGS, "rdiv_scalar(a, s, dst)"},
    {"pow_scalar_f32",     py_pow_scalar,        METH_VARARGS, "pow_scalar(a, s, dst)"},
    {"clamp_min_f32",      py_clamp_min,         METH_VARARGS, "clamp_min(a, s, dst)"},
    {"clamp_max_f32",      py_clamp_max,         METH_VARARGS, "clamp_max(a, s, dst)"},
    {"fill_f32",           py_fill_f32,          METH_VARARGS, "fill(dst, val)"},
    {"axpy_f32",           py_axpy_f32,          METH_VARARGS, "axpy(src, alpha, dst)"},
    /* Reductions float32 */
    {"sum_f32",            py_sum_f32,           METH_VARARGS, "sum(src) -> float"},
    {"max_f32",            py_max_f32,           METH_VARARGS, "max(src) -> float"},
    {"min_f32",            py_min_f32,           METH_VARARGS, "min(src) -> float"},
    {"argmax_f32",         py_argmax_f32,        METH_VARARGS, "argmax(src) -> int"},
    {"argmin_f32",         py_argmin_f32,        METH_VARARGS, "argmin(src) -> int"},
    /* Softmax float32 */
    {"softmax_rows_f32",     py_softmax_rows,      METH_VARARGS, "softmax_rows(src, dst, rows, cols)"},
    {"log_softmax_rows_f32", py_log_softmax_rows,  METH_VARARGS, "log_softmax_rows(src, dst, rows, cols)"},
    /* MatMul */
    {"matmul_f32",         py_matmul_f32,        METH_VARARGS, "matmul_f32(A, B, C, M, K, N)"},
    {"matmul_act_f32",     py_matmul_act_f32,    METH_VARARGS, "matmul_act_f32(A,B,C,M,K,N,post_op)"},
    {"flash_attn_fwd_f32", py_flash_attn_fwd_f32, METH_VARARGS, "flash_attn_fwd_f32(Q,K,V,O,L,N,d,scale)"},
    {"flash_attn_bwd_f32", py_flash_attn_bwd_f32, METH_VARARGS, "flash_attn_bwd_f32(dO,Q,K,V,O,L,dQ,dK,dV,N,d,scale)"},
    {"dot_f32",            py_dot_f32,           METH_VARARGS, "dot_f32(a, b) -> float"},
    /* Float64 key ops */
    {"exp_f64",            py_exp_f64,           METH_VARARGS, "exp(src, dst)"},
    {"log_f64",            py_log_f64,           METH_VARARGS, "log(src, dst)"},
    {"sum_f64",            py_sum_f64,           METH_VARARGS, "sum(src) -> float"},
    {"matmul_f64",         py_matmul_f64,        METH_VARARGS, "matmul_f64(A, B, C, M, K, N)"},
    /* Normalization */
    {"var_f32",              py_var_f32,              METH_VARARGS, "var(src[,unbiased]) -> (mean,var)"},
    {"layer_norm_f32",       py_layer_norm_f32,       METH_VARARGS, "layer_norm(src,dst,gamma,beta,rows,norm_size,eps)"},
    {"rms_norm_f32",         py_rms_norm_f32,         METH_VARARGS, "rms_norm(src,dst,weight,rows,norm_size,eps)"},
    {"batch_norm_f32",       py_batch_norm_f32,       METH_VARARGS, "batch_norm(x,out,mean,rstd,gamma,beta,xhat,N,C,eps)"},
    {"bn_update_running_f32", py_bn_update_running_f32, METH_VARARGS, "bn_update_running(mean,rstd,rm,rv,C,momentum,eps,N)"},
    {"batch_norm_backward_f32", py_batch_norm_backward_f32, METH_VARARGS, "batch_norm_bwd(dy,xhat,rstd,gamma,dx,dgamma,dbeta,N,C)"},
    {"add_bias_rows_f32",    py_add_bias_rows_f32,    METH_VARARGS, "add_bias_rows(out,bias,N,C) — out[n] += bias for all rows"},
    {"linear_relu_f32",     py_linear_relu_f32,      METH_VARARGS, "linear_relu(x,wT,bias_or_None,out,N,in_f,out_f) — fused GEMM+bias+ReLU"},
    /* Transpose + axial */
    {"transpose_f32",        py_transpose_f32,        METH_VARARGS, "transpose(src,dst,rows,cols)"},
    {"sum_axis_f32",         py_sum_axis_f32,         METH_VARARGS, "sum_axis(src,dst,rows,cols,axis)"},
    {"max_axis_f32",         py_max_axis_f32,         METH_VARARGS, "max_axis(src,dst,rows,cols,axis)"},
    /* Where + comparison */
    {"where_f32",            py_where_f32,            METH_VARARGS, "where(cond,a,b,dst)"},
    {"eq_f32",               py_eq_f32,               METH_VARARGS, "eq(a,b,dst)"},
    {"ne_f32",               py_ne_f32,               METH_VARARGS, "ne(a,b,dst)"},
    {"lt_f32",               py_lt_f32,               METH_VARARGS, "lt(a,b,dst)"},
    {"le_f32",               py_le_f32,               METH_VARARGS, "le(a,b,dst)"},
    {"gt_f32",               py_gt_f32,               METH_VARARGS, "gt(a,b,dst)"},
    {"ge_f32",               py_ge_f32,               METH_VARARGS, "ge(a,b,dst)"},
    /* Activation backward */
    {"relu_backward_f32",    py_relu_backward_f32,    METH_VARARGS, "relu_backward(go,out,gi)"},
    {"sigmoid_backward_f32", py_sigmoid_backward_f32, METH_VARARGS, "sigmoid_backward(go,out,gi)"},
    {"tanh_backward_f32",    py_tanh_backward_f32,    METH_VARARGS, "tanh_backward(go,out,gi)"},
    {"softmax_backward_f32", py_softmax_backward_f32, METH_VARARGS, "softmax_backward(go,out,gi,rows,cols)"},
    /* Pow + fused loss */
    {"pow_f32",              py_pow_f32,              METH_VARARGS, "pow(a,b,dst)"},
    {"cross_entropy_f32",    py_cross_entropy_f32,    METH_VARARGS, "cross_entropy(logits,targets,loss,rows,cols)"},
    /* Optimizer kernels */
    {"adam_step_f32",        py_adam_step_f32,        METH_VARARGS, "adam_step(p,g,m,v,lr,b1,b2,eps,wd,bc1,bc2)"},
    {"clip_grad_norm_f32",   py_clip_grad_norm_f32,   METH_VARARGS, "clip_grad_norm(g, max_norm) -> float norm"},
    {"addcmul_f32",          py_addcmul_f32,          METH_VARARGS, "addcmul(a, b, c, alpha, dst) — dst=a+alpha*b*c"},
    /* Capability query */
    {"has_avx2",             py_has_avx2,             METH_NOARGS,  "has_avx2() -> bool"},
    /* Trig */
    {"sin_f32",              py_sin_f32,              METH_VARARGS, "sin(src, dst)"},
    {"cos_f32",              py_cos_f32,              METH_VARARGS, "cos(src, dst)"},
    /* Activation backward (C-speed) */
    {"exp_backward_f32",    py_exp_backward_f32,     METH_VARARGS, "exp_backward(go, out, gi)"},
    {"log_backward_f32",    py_log_backward_f32,     METH_VARARGS, "log_backward(go, inp, gi)"},
    {"sqrt_backward_f32",   py_sqrt_backward_f32,    METH_VARARGS, "sqrt_backward(go, out, gi)"},
    {"sin_backward_f32",    py_sin_backward_f32,     METH_VARARGS, "sin_backward(go, inp, gi)"},
    {"cos_backward_f32",    py_cos_backward_f32,     METH_VARARGS, "cos_backward(go, inp, gi)"},
    /* im2col / col2im for Conv2d */
    {"im2col_f32",          py_im2col_f32,           METH_VARARGS, "im2col(inp,col,N,C,H,W,Kh,Kw,Sh,Sw,Ph,Pw)"},
    {"col2im_f32",          py_col2im_f32,           METH_VARARGS, "col2im(col,grad,N,C,H,W,Kh,Kw,Sh,Sw,Ph,Pw)"},
    /* BF16 GEMM */
    {"gemm_bf16_f32",       py_gemm_bf16_f32,        METH_VARARGS, "gemm_bf16_f32(A_bf16,B_bf16,C_f32,M,K,N)"},
    {NULL, NULL, 0, NULL}
};

/* ---- ISPC bridge (compiled separately, linked in) ---- */
#ifdef VP_HAVE_ISPC
#include "../ispc/vp_ispc_bridge.c"
#endif

/* Merged method table: AVX2 methods + ISPC methods (when available) */
static PyMethodDef* _build_methods(void) {
#ifdef VP_HAVE_ISPC
    /* Count entries in each table */
    int n_core = 0, n_ispc = 0;
    while (vp_core_methods[n_core].ml_name) n_core++;
    while (vp_ispc_methods[n_ispc].ml_name) n_ispc++;
    int total = n_core + n_ispc + 1; /* +1 for sentinel */
    PyMethodDef *merged = (PyMethodDef*)malloc(total * sizeof(PyMethodDef));
    if (!merged) return vp_core_methods; /* fallback */
    memcpy(merged,          vp_core_methods, n_core  * sizeof(PyMethodDef));
    memcpy(merged + n_core, vp_ispc_methods, n_ispc  * sizeof(PyMethodDef));
    merged[n_core + n_ispc] = (PyMethodDef){NULL, NULL, 0, NULL};
    return merged;
#else
    return vp_core_methods;
#endif
}

static struct PyModuleDef vp_core_module = {
    PyModuleDef_HEAD_INIT,
    "_vp_core",
    "vectorphile AVX2/FMA + ISPC core kernels",
    -1,
    NULL   /* filled in PyInit__vp_core below */
};

/* Warm the OpenMP thread pool at module load: a no-op parallel for so the
 * OS has already created all worker threads before the first real kernel call.
 * Without this, the first parallel region incurs 100-200 µs thread-spawn cost. */
static void vp_omp_warmup(void) {
#ifdef _OPENMP
    /* Fire a tiny parallel region unconditionally so the OS creates all OMP
     * worker threads now. Without this the first real parallel kernel call
     * incurs a 100-200 µs thread-spawn penalty on Windows. */
    volatile int x = 0;
    VP_PRAGMA(omp parallel) { VP_PRAGMA(omp atomic) x++; }
    (void)x;
#endif
}

PyMODINIT_FUNC PyInit__vp_core(void) {
    vp_omp_warmup();   /* prime OMP thread pool: eliminates 100-200µs first-call latency */
#ifdef VP_HAVE_TPOOL
    vp_tpool_init();   /* start persistent worker threads for lightweight ops */
#endif
    vp_core_module.m_methods = _build_methods();
    PyObject *m = PyModule_Create(&vp_core_module);
    if (!m) return NULL;
    /* Expose VP_POST_* constants so Python can pass them to matmul_act_f32 */
    PyModule_AddIntConstant(m, "POST_NONE",    VP_POST_NONE);
    PyModule_AddIntConstant(m, "POST_RELU",    VP_POST_RELU);
    PyModule_AddIntConstant(m, "POST_GELU",    VP_POST_GELU);
    PyModule_AddIntConstant(m, "POST_SILU",    VP_POST_SILU);
    PyModule_AddIntConstant(m, "POST_SIGMOID", VP_POST_SIGMOID);
    PyModule_AddIntConstant(m, "POST_TANH",    VP_POST_TANH);
    return m;
}
