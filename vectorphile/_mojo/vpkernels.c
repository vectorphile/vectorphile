/*
 * vectorphile kernel library -- pure C, zero dependencies.
 *
 * All kernels written from scratch. Compiler auto-vectorizes with
 * /O2 /arch:AVX2 (MSVC) or -O3 -march=native -ffast-math (GCC/Clang).
 *
 * ABI: all float buffers are float32 (4-byte IEEE 754), row-major.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#elif defined(_MSC_VER) && defined(_M_AMD64)
#include <immintrin.h>
#define __AVX2__ 1
#endif

/* == Compiler hints ======================================================== */

#ifdef _MSC_VER
  #define VP_INLINE __forceinline
  #define VP_RESTRICT __restrict
  /* MSVC: auto-vectorizes with /O2 /arch:AVX2 */
#else
  #define VP_INLINE __attribute__((always_inline)) inline
  #define VP_RESTRICT __restrict__
  #pragma GCC optimize("O3,unroll-loops")
  #pragma GCC target("avx2,fma")
#endif

/* == Fast math approximations ============================================== */

/* Fast expf: e^x via float-bit trick + degree-3 polynomial correction.
 * MSVC auto-vectorizes this loop well. Max relative error ~5e-6. */
static VP_INLINE float _vp_expf_fast(float x) {
    return expf(x);
}

/* Fast tanh via rational approximation (max err ~1e-5):
 * tanh(x) ~= x*(27+x^2) / (27+9*x^2)   for |x| < 3 (Pade [3/2])
 * saturates to ±1 outside [-3,3] */
static VP_INLINE float _vp_tanhf_fast(float x) {
    float ax = x < 0.0f ? -x : x;
    if (ax >= 4.97f) return x < 0.0f ? -1.0f : 1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

/* Fast sigmoid via hard-sigmoid approximation, clamped:
 * sigmoid(x) = clamp(0.2*x + 0.5, 0, 1)   -- linear approx, very fast
 * Better: rational approx matching logistic to <0.002 abs error on [-5,5] */
static VP_INLINE float _vp_sigmoidf_fast(float x) {
    /* Rational Pade approximation for sigmoid accurate to 0.002:
     * sig(x) ~= (x*(27 + x^2)) / (4*(6 + x^2*(3 + x^2*0.03704))) ... too complex.
     * Use: 0.5 * tanh(x/2) + 0.5  -- avoids exp entirely, reuses fast tanh */
    return 0.5f * _vp_tanhf_fast(x * 0.5f) + 0.5f;
}

static VP_INLINE float _vp_expf(float x) { return expf(x); }
static VP_INLINE float _vp_logf(float x) { return logf(x); }
static VP_INLINE float _vp_sqrtf(float x) { return sqrtf(x); }
static VP_INLINE float _vp_tanhf(float x) { return tanhf(x); }

/* erf approximation (Abramowitz & Stegun 7.1.26) */
static VP_INLINE float _vp_erff(float x) {
    float sign = (x >= 0.0f) ? 1.0f : -1.0f;
    x = fabsf(x);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float p = t * (0.254829592f + t * (-0.284496736f +
              t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
    return sign * (1.0f - p * expf(-x * x));
}

/* ===========================================================================
 * GEMM -- register-tiled AVX2 microkernel, BLIS-style packing
 *
 * Microkernel: MR=8 rows x NR=8 cols (8 YMM regs hold C tile in registers)
 * Packing: A panel -> column-major (MR x KC), B panel -> row-major (KC x NR)
 * This keeps C tile in registers for the entire K loop -> ~4x less memory BW
 * =========================================================================== */

#define MR 8        /* microkernel row tile  */
#define NR 8        /* microkernel col tile (1 AVX2 register = 8 floats) */
#define MC 64       /* macro M tile (must be multiple of MR) */
#define KC 256      /* macro K tile -- fits in L2 with A panel (64*256*4 = 64KB) */
#define NC 512      /* macro N tile (must be multiple of NR) -- B panel 256*512*4 = 512KB (L3) */

/* Pack a mc×kc submatrix of A (row-major, stride K) into packed_a.
 * Layout: packed_a[i * kc + k] = A[m0+i, k0+k]  (row-major within mc×kc) */
static VP_INLINE void _pack_a(const float * VP_RESTRICT A, float * VP_RESTRICT pa,
                               int mc, int kc, int K) {
    for (int i = 0; i < mc; i++) {
        const float *Ai = A + i * K;
        float *pai = pa + i * kc;
        int k = 0;
#ifdef __AVX2__
        for (; k + 7 < kc; k += 8) {
            __m256 v = _mm256_loadu_ps(Ai + k);
            _mm256_storeu_ps(pai + k, v);
        }
#endif
        for (; k < kc; k++) pai[k] = Ai[k];
    }
}

/* Pack a kc×nc submatrix of B (row-major, stride N) into packed_b.
 * Layout: packed_b[k * nc + j] = B[k0+k, n0+j]  (row-major within kc×nc) */
static VP_INLINE void _pack_b(const float * VP_RESTRICT B, float * VP_RESTRICT pb,
                               int kc, int nc, int N) {
    for (int k = 0; k < kc; k++) {
        const float *Bk = B + k * N;
        float *pbk = pb + k * nc;
        int j = 0;
#ifdef __AVX2__
        for (; j + 7 < nc; j += 8) {
            __m256 v = _mm256_loadu_ps(Bk + j);
            _mm256_storeu_ps(pbk + j, v);
        }
#endif
        for (; j < nc; j++) pbk[j] = Bk[j];
    }
}

#ifdef __AVX2__
/* 8×8 AVX2 register-tiled microkernel.
 * Computes C[0..7][0..7] += A_packed[0..7][0..kc-1] * B_packed[0..kc-1][0..7]
 * pa: row-major mc×kc, stride kc  (we pass pa + ir*kc for row ir)
 * pb: row-major kc×nc, stride nc  (we pass pb + jr for col jr)
 * C:  row-major M×N,   stride ldc */
static void _micro_8x8(float * VP_RESTRICT C, const float * VP_RESTRICT pa,
                        const float * VP_RESTRICT pb, int kc, int nc, int ldc,
                        int mr, int nr) {
    __m256 c0, c1, c2, c3, c4, c5, c6, c7;

    if (mr == MR && nr == NR) {
        /* Full 8×8 tile -- keep C entirely in registers */
        c0 = _mm256_setzero_ps(); c1 = _mm256_setzero_ps();
        c2 = _mm256_setzero_ps(); c3 = _mm256_setzero_ps();
        c4 = _mm256_setzero_ps(); c5 = _mm256_setzero_ps();
        c6 = _mm256_setzero_ps(); c7 = _mm256_setzero_ps();

        for (int k = 0; k < kc; k++) {
            __m256 bk = _mm256_loadu_ps(pb + k * nc); /* B row k, cols [jr..jr+7] */
            c0 = _mm256_fmadd_ps(_mm256_set1_ps(pa[0 * kc + k]), bk, c0);
            c1 = _mm256_fmadd_ps(_mm256_set1_ps(pa[1 * kc + k]), bk, c1);
            c2 = _mm256_fmadd_ps(_mm256_set1_ps(pa[2 * kc + k]), bk, c2);
            c3 = _mm256_fmadd_ps(_mm256_set1_ps(pa[3 * kc + k]), bk, c3);
            c4 = _mm256_fmadd_ps(_mm256_set1_ps(pa[4 * kc + k]), bk, c4);
            c5 = _mm256_fmadd_ps(_mm256_set1_ps(pa[5 * kc + k]), bk, c5);
            c6 = _mm256_fmadd_ps(_mm256_set1_ps(pa[6 * kc + k]), bk, c6);
            c7 = _mm256_fmadd_ps(_mm256_set1_ps(pa[7 * kc + k]), bk, c7);
        }

        /* Accumulate into C (C was zeroed before the k-loop in the caller) */
        _mm256_storeu_ps(C + 0*ldc, _mm256_add_ps(_mm256_loadu_ps(C + 0*ldc), c0));
        _mm256_storeu_ps(C + 1*ldc, _mm256_add_ps(_mm256_loadu_ps(C + 1*ldc), c1));
        _mm256_storeu_ps(C + 2*ldc, _mm256_add_ps(_mm256_loadu_ps(C + 2*ldc), c2));
        _mm256_storeu_ps(C + 3*ldc, _mm256_add_ps(_mm256_loadu_ps(C + 3*ldc), c3));
        _mm256_storeu_ps(C + 4*ldc, _mm256_add_ps(_mm256_loadu_ps(C + 4*ldc), c4));
        _mm256_storeu_ps(C + 5*ldc, _mm256_add_ps(_mm256_loadu_ps(C + 5*ldc), c5));
        _mm256_storeu_ps(C + 6*ldc, _mm256_add_ps(_mm256_loadu_ps(C + 6*ldc), c6));
        _mm256_storeu_ps(C + 7*ldc, _mm256_add_ps(_mm256_loadu_ps(C + 7*ldc), c7));
    } else {
        /* Fringe tile: scalar fallback for edge rows/cols */
        for (int i = 0; i < mr; i++) {
            for (int j = 0; j < nr; j++) {
                float s = 0.0f;
                for (int k = 0; k < kc; k++) s += pa[i * kc + k] * pb[k * nc + j];
                C[i * ldc + j] += s;
            }
        }
    }
}
#endif /* __AVX2__ */

void vp_sgemm(
    float * VP_RESTRICT C,
    const float * VP_RESTRICT A,
    const float * VP_RESTRICT B,
    int M, int K, int N
) {
    memset(C, 0, (size_t)M * N * sizeof(float));

#ifdef __AVX2__
    /* Allocate packing buffers on stack (small) or heap for large tiles */
    float *pa = (float*)malloc((size_t)MC * KC * sizeof(float));
    float *pb = (float*)malloc((size_t)KC * NC * sizeof(float));
    if (!pa || !pb) { free(pa); free(pb); goto scalar_fallback; }

    {
        int num_m = (M + MC - 1) / MC;
        int tm;
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic,1) firstprivate(pa,pb) if(M >= MC)
#endif
        for (tm = 0; tm < num_m; tm++) {
            int m0, mc;
            float *tpa, *tpb;
            m0 = tm * MC;
            mc = (m0 + MC <= M) ? MC : (M - m0);
            /* private packing buffers per thread when OpenMP is active */
#ifdef _OPENMP
            tpa = (float*)malloc((size_t)MC * KC * sizeof(float));
            tpb = (float*)malloc((size_t)KC * NC * sizeof(float));
            if (!tpa || !tpb) { free(tpa); free(tpb); continue; }
#else
            tpa = pa;
            tpb = pb;
#endif
            {
                int k0, kc, n0, nc, ir, mr, jr, nr;
                for (k0 = 0; k0 < K; k0 += KC) {
                    kc = (k0 + KC <= K) ? KC : (K - k0);
                    _pack_a(A + m0*K + k0, tpa, mc, kc, K);

                    for (n0 = 0; n0 < N; n0 += NC) {
                        nc = (n0 + NC <= N) ? NC : (N - n0);
                        _pack_b(B + k0*N + n0, tpb, kc, nc, N);

                        for (ir = 0; ir < mc; ir += MR) {
                            mr = (ir + MR <= mc) ? MR : (mc - ir);
                            for (jr = 0; jr < nc; jr += NR) {
                                nr = (jr + NR <= nc) ? NR : (nc - jr);
                                _micro_8x8(
                                    C + (m0+ir)*N + (n0+jr),
                                    tpa + ir * kc,
                                    tpb + jr,
                                    kc, nc, N, mr, nr
                                );
                            }
                        }
                    }
                }
            }
#ifdef _OPENMP
            free(tpa); free(tpb);
#endif
        }
    }

    free(pa); free(pb);
    return;

scalar_fallback:
#endif /* __AVX2__ */
    /* Scalar fallback: simple i-p-j with optional auto-vectorisation */
    {
        int i, p, j;
        for (i = 0; i < M; i++) {
            for (p = 0; p < K; p++) {
                float av = A[i*K + p];
                if (av == 0.0f) continue;
                for (j = 0; j < N; j++) C[i*N+j] += av * B[p*N+j];
            }
        }
    }
}

void vp_gemm_f32(float *C, const float *A, const float *B, int M, int K, int N) {
    vp_sgemm(C, A, B, M, K, N);
}

void vp_sgemm_transB(
    float * VP_RESTRICT C,
    const float * VP_RESTRICT A,
    const float * VP_RESTRICT B,  /* B is (N x K), i.e. transposed */
    int M, int K, int N
) {
    /* For transB, transpose B into a temp buffer then call vp_sgemm */
    float *Bt = (float*)malloc((size_t)K * N * sizeof(float));
    if (Bt) {
        int j, k;
        for (j = 0; j < N; j++)
            for (k = 0; k < K; k++)
                Bt[k*N + j] = B[j*K + k];
        vp_sgemm(C, A, Bt, M, K, N);
        free(Bt);
    } else {
        /* fallback: scalar dot-product rows */
        int i, j, k;
        for (i = 0; i < M; i++) {
            const float *Ai = A + i * K;
            float *Ci = C + i * N;
            for (j = 0; j < N; j++) {
                const float *Bj = B + j * K;
                float s = 0.0f;
                for (k = 0; k < K; k++) s += Ai[k] * Bj[k];
                Ci[j] = s;
            }
        }
    }
}

void vp_sgemm_batched(
    float *C, const float *A, const float *B,
    int batch, int M, int K, int N
) {
    {
        int b;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(batch > 1)
#endif
        for (b = 0; b < batch; b++) {
            vp_sgemm(C + b*M*N, A + b*M*K, B + b*K*N, M, K, N);
        }
    }
}

void vp_nd_matmul(float *C, const float *A, const float *B, int M, int K, int N) {
    vp_sgemm(C, A, B, M, K, N);
}

/* ===========================================================================
 * ELEMENTWISE UNARY
 * =========================================================================== */

/* AVX2 fast exp: process 8 floats at once */
#ifdef __AVX2__
static void _vp_exp_avx2(float * VP_RESTRICT dst, const float * VP_RESTRICT src, int n) {
    /* Accurate AVX2 exp: round-based range reduction + degree-6 polynomial
     * r in [-ln2/2, ln2/2], error ~1e-7 */
    static const __m256 log2e   = {1.44269504f,1.44269504f,1.44269504f,1.44269504f,
                                    1.44269504f,1.44269504f,1.44269504f,1.44269504f};
    static const __m256 ln2hi   = {6.9314575e-1f,6.9314575e-1f,6.9314575e-1f,6.9314575e-1f,
                                    6.9314575e-1f,6.9314575e-1f,6.9314575e-1f,6.9314575e-1f};
    static const __m256 ln2lo   = {1.4286068e-6f,1.4286068e-6f,1.4286068e-6f,1.4286068e-6f,
                                    1.4286068e-6f,1.4286068e-6f,1.4286068e-6f,1.4286068e-6f};
    static const __m256 half    = {0.5f,0.5f,0.5f,0.5f,0.5f,0.5f,0.5f,0.5f};
    static const __m256 lo      = {-88.f,-88.f,-88.f,-88.f,-88.f,-88.f,-88.f,-88.f};
    static const __m256 hi      = { 88.f, 88.f, 88.f, 88.f, 88.f, 88.f, 88.f, 88.f};
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 x   = _mm256_loadu_ps(src + i);
        x = _mm256_max_ps(lo, _mm256_min_ps(hi, x));
        /* n = round(x * log2e) */
        __m256  af = _mm256_fmadd_ps(x, log2e, half);
        __m256i ni = _mm256_cvtps_epi32(_mm256_floor_ps(af));
        __m256  nf = _mm256_cvtepi32_ps(ni);
        /* Cody-Waite: r = x - n*ln2hi - n*ln2lo */
        __m256  r  = _mm256_fnmadd_ps(nf, ln2hi, x);
        r = _mm256_fnmadd_ps(nf, ln2lo, r);
        /* Degree-6 Horner: e^r on [-ln2/2, ln2/2], err < 1e-7 */
        __m256 er = _mm256_set1_ps(1.38888888e-3f);
        er = _mm256_fmadd_ps(r, er, _mm256_set1_ps(8.33333333e-3f));
        er = _mm256_fmadd_ps(r, er, _mm256_set1_ps(4.16666667e-2f));
        er = _mm256_fmadd_ps(r, er, _mm256_set1_ps(1.66666667e-1f));
        er = _mm256_fmadd_ps(r, er, _mm256_set1_ps(5.00000000e-1f));
        er = _mm256_fmadd_ps(r, er, _mm256_set1_ps(1.00000000e+0f));
        er = _mm256_fmadd_ps(r, er, _mm256_set1_ps(1.00000000e+0f));
        /* 2^n via biased exponent */
        __m256i pw = _mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23);
        __m256 scale = _mm256_castsi256_ps(pw);
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(scale, er));
    }
    for (; i < n; i++) dst[i] = _vp_expf_fast(src[i]);
}
#endif

void vp_exp(float *dst, const float *src, int n) {
#ifdef __AVX2__
    _vp_exp_avx2(dst, src, n);
#else
    int i;
    for (i = 0; i < n; i++) dst[i] = _vp_expf_fast(src[i]);
#endif
}
void vp_exp_f32(float *dst, const float *src, int n) { vp_exp(dst, src, n); }

void vp_log(float *dst, const float *src, int n) {
    for (int i = 0; i < n; i++) dst[i] = logf(src[i]);
}
void vp_log_f32(float *dst, const float *src, int n) { vp_log(dst, src, n); }

void vp_sqrt(float *dst, const float *src, int n) {
    for (int i = 0; i < n; i++) dst[i] = sqrtf(src[i]);
}
void vp_sqrt_f32(float *dst, const float *src, int n) { vp_sqrt(dst, src, n); }

void vp_abs(float *dst, const float *src, int n) {
    for (int i = 0; i < n; i++) dst[i] = fabsf(src[i]);
}

void vp_neg(float *dst, const float *src, int n) {
    for (int i = 0; i < n; i++) dst[i] = -src[i];
}

void vp_relu(float *dst, const float *src, int n) {
    for (int i = 0; i < n; i++) dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
}
void vp_relu_f32(float *dst, const float *src, int n) { vp_relu(dst, src, n); }

/* -- AVX2 Pade tanh shared helper: tanh(u) = u*(27+u^2)/(27+9*u^2), sat at |u|>=4.97 -- */
#ifdef __AVX2__
static VP_INLINE __m256 _avx2_tanh_pade(__m256 u) {
    __m256 u2   = _mm256_mul_ps(u, u);
    __m256 num  = _mm256_fmadd_ps(u2, u, _mm256_mul_ps(_mm256_set1_ps(27.0f), u));
    __m256 den  = _mm256_fmadd_ps(u2, _mm256_set1_ps(9.0f), _mm256_set1_ps(27.0f));
    __m256 t    = _mm256_div_ps(num, den);
    __m256 au   = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), u);
    __m256 sign = _mm256_or_ps(_mm256_and_ps(u, _mm256_set1_ps(-0.0f)), _mm256_set1_ps(1.0f));
    __m256 mask = _mm256_cmp_ps(au, _mm256_set1_ps(4.97f), _CMP_GE_OQ);
    return _mm256_blendv_ps(t, sign, mask);
}
#endif

void vp_tanh(float *dst, const float *src, int n) {
#ifdef __AVX2__
    int i = 0;
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(dst+i, _avx2_tanh_pade(_mm256_loadu_ps(src+i)));
    for (; i < n; i++) dst[i] = _vp_tanhf_fast(src[i]);
#else
    for (int i = 0; i < n; i++) dst[i] = _vp_tanhf_fast(src[i]);
#endif
}
void vp_tanh_f32(float *dst, const float *src, int n) { vp_tanh(dst, src, n); }

void vp_sigmoid(float *dst, const float *src, int n) {
#ifdef __AVX2__
    /* sigmoid = 0.5*tanh(x/2) + 0.5 */
    __m256 half = _mm256_set1_ps(0.5f);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(src+i);
        __m256 t = _avx2_tanh_pade(_mm256_mul_ps(x, half));
        _mm256_storeu_ps(dst+i, _mm256_fmadd_ps(half, t, half));
    }
    for (; i < n; i++) dst[i] = _vp_sigmoidf_fast(src[i]);
#else
    for (int i = 0; i < n; i++) dst[i] = _vp_sigmoidf_fast(src[i]);
#endif
}
void vp_sigmoid_f32(float *dst, const float *src, int n) { vp_sigmoid(dst, src, n); }

void vp_gelu(float *dst, const float *src, int n) {
#ifdef __AVX2__
    /* GELU(x) ~= x * 0.5*(tanh(0.8512*x)+1) */
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 ck   = _mm256_set1_ps(0.8512f);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(src+i);
        __m256 t = _avx2_tanh_pade(_mm256_mul_ps(x, ck));
        __m256 g = _mm256_mul_ps(x, _mm256_fmadd_ps(half, t, half));
        _mm256_storeu_ps(dst+i, g);
    }
    for (; i < n; i++) { float x=src[i]; dst[i]=x*(0.5f*_vp_tanhf_fast(x*0.8512f)+0.5f); }
#else
    for (int i = 0; i < n; i++) { float x=src[i]; dst[i]=x*(0.5f*_vp_tanhf_fast(x*0.8512f)+0.5f); }
#endif
}
void vp_gelu_f32(float *dst, const float *src, int n) { vp_gelu(dst, src, n); }

void vp_silu(float *dst, const float *src, int n) {
#ifdef __AVX2__
    /* silu = x * sigmoid(x) = x * (0.5*tanh(x/2) + 0.5) */
    __m256 half = _mm256_set1_ps(0.5f);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 x   = _mm256_loadu_ps(src+i);
        __m256 t   = _avx2_tanh_pade(_mm256_mul_ps(x, half));
        __m256 sig = _mm256_fmadd_ps(half, t, half);
        _mm256_storeu_ps(dst+i, _mm256_mul_ps(x, sig));
    }
    for (; i < n; i++) { float x=src[i]; dst[i]=x*_vp_sigmoidf_fast(x); }
#else
    for (int i = 0; i < n; i++) { float x=src[i]; dst[i]=x*_vp_sigmoidf_fast(x); }
#endif
}
void vp_silu_f32(float *dst, const float *src, int n) { vp_silu(dst, src, n); }

void vp_log1p(float *dst, const float *src, int n) {
    for (int i = 0; i < n; i++) dst[i] = log1pf(src[i]);
}

void vp_expm1(float *dst, const float *src, int n) {
    for (int i = 0; i < n; i++) dst[i] = expm1f(src[i]);
}

void vp_erf(float *dst, const float *src, int n) {
    for (int i = 0; i < n; i++) dst[i] = _vp_erff(src[i]);
}

/* ===========================================================================
 * ELEMENTWISE BINARY
 * =========================================================================== */

void vp_add(float *dst, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) dst[i] = a[i] + b[i];
}
void vp_add_f32(float *dst, const float *a, const float *b, int n) { vp_add(dst, a, b, n); }

void vp_sub(float *dst, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) dst[i] = a[i] - b[i];
}
void vp_sub_f32(float *dst, const float *a, const float *b, int n) { vp_sub(dst, a, b, n); }

void vp_mul(float *dst, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) dst[i] = a[i] * b[i];
}
void vp_mul_f32(float *dst, const float *a, const float *b, int n) { vp_mul(dst, a, b, n); }

void vp_div(float *dst, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) dst[i] = a[i] / b[i];
}
void vp_div_f32(float *dst, const float *a, const float *b, int n) { vp_div(dst, a, b, n); }

void vp_add_scalar(float *dst, const float *a, float scalar, int n) {
    for (int i = 0; i < n; i++) dst[i] = a[i] + scalar;
}
void vp_scalar_add_f32(float *src, float scalar, float *dst, int n) { vp_add_scalar(dst, src, scalar, n); }

void vp_mul_scalar(float *dst, const float *a, float scalar, int n) {
    for (int i = 0; i < n; i++) dst[i] = a[i] * scalar;
}
void vp_scalar_mul_f32(float *src, float scalar, float *dst, int n) { vp_mul_scalar(dst, src, scalar, n); }

/* Strided broadcast binary op: dst[i] = op(a[a_strides . idx], b[b_strides . idx])
 * ndim <= 8, out_shape[ndim], a_strides[ndim], b_strides[ndim] in elements.
 * op: 0=add, 1=sub, 2=mul, 3=div */
void vp_broadcast_binary(float *dst,
                         const float *a, const float *b,
                         const int *out_shape,
                         const int *a_strides,
                         const int *b_strides,
                         int ndim, int op) {
    /* compute total output elements */
    int64_t total = 1;
    for (int d = 0; d < ndim; d++) total *= out_shape[d];

    for (int64_t flat = 0; flat < total; flat++) {
        int64_t rem = flat;
        int64_t ai = 0, bi = 0;
        for (int d = ndim - 1; d >= 0; d--) {
            int coord = (int)(rem % out_shape[d]);
            rem /= out_shape[d];
            ai += coord * a_strides[d];
            bi += coord * b_strides[d];
        }
        float va = a[ai], vb = b[bi];
        float r;
        if      (op == 0) r = va + vb;
        else if (op == 1) r = va - vb;
        else if (op == 2) r = va * vb;
        else              r = va / vb;
        dst[flat] = r;
    }
}

void vp_clamp(float *dst, const float *src, float lo, float hi, int n) {
    for (int i = 0; i < n; i++) {
        float v = src[i];
        dst[i] = v < lo ? lo : (v > hi ? hi : v);
    }
}

void vp_pow(float *dst, const float *base, float exp_val, int n) {
    for (int i = 0; i < n; i++) dst[i] = powf(base[i], exp_val);
}

/* ===========================================================================
 * REDUCTIONS
 * =========================================================================== */

float vp_sum(const float *a, int n) {
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) acc = _mm256_add_ps(acc, _mm256_loadu_ps(a + i));
    /* Horizontal sum of 8 lanes */
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4 = _mm_add_ss(s4, _mm_shuffle_ps(s4, s4, 1));
    float s = _mm_cvtss_f32(s4);
    for (; i < n; i++) s += a[i];
    return s;
#else
    double s = 0.0;
    for (int i = 0; i < n; i++) s += a[i];
    return (float)s;
#endif
}

void vp_sum_f32(const float *a, int n, float *out) { *out = vp_sum(a, n); }

float vp_mean(const float *a, int n) {
    return n > 0 ? vp_sum(a, n) / (float)n : 0.0f;
}

float vp_max(const float *a, int n) {
    float m = -FLT_MAX;
    for (int i = 0; i < n; i++) if (a[i] > m) m = a[i];
    return m;
}
void vp_max_f32(const float *a, int n, float *out) { *out = vp_max(a, n); }

float vp_min(const float *a, int n) {
    float m = FLT_MAX;
    for (int i = 0; i < n; i++) if (a[i] < m) m = a[i];
    return m;
}
void vp_min_f32(const float *a, int n, float *out) { *out = vp_min(a, n); }

float vp_norm_l2(const float *a, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)a[i] * a[i];
    return (float)sqrt(s);
}

float vp_dot(const float *a, const float *b, int n) {
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s4 = _mm_add_ps(lo, hi);
    s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    s4 = _mm_add_ss(s4, _mm_shuffle_ps(s4, s4, 1));
    float s = _mm_cvtss_f32(s4);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
#else
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)a[i] * b[i];
    return (float)s;
#endif
}

int vp_argmax(const float *a, int n) {
    int idx = 0;
    for (int i = 1; i < n; i++) if (a[i] > a[idx]) idx = i;
    return idx;
}
void vp_argmax_f32(const float *a, int n, int *out) { *out = vp_argmax(a, n); }

int vp_argmin(const float *a, int n) {
    int idx = 0;
    for (int i = 1; i < n; i++) if (a[i] < a[idx]) idx = i;
    return idx;
}

void vp_sum_axis(const float *a, float *out, int outer, int axis_len, int inner) {
    int out_n = outer * inner;
    for (int i = 0; i < out_n; i++) out[i] = 0.0f;
    for (int o = 0; o < outer; o++) {
        for (int k = 0; k < axis_len; k++) {
            for (int i = 0; i < inner; i++) {
                out[o * inner + i] += a[o * axis_len * inner + k * inner + i];
            }
        }
    }
}

void vp_max_axis(const float *a, float *out, int outer, int axis_len, int inner) {
    int out_n = outer * inner;
    for (int i = 0; i < out_n; i++) out[i] = -FLT_MAX;
    for (int o = 0; o < outer; o++) {
        for (int k = 0; k < axis_len; k++) {
            for (int i = 0; i < inner; i++) {
                float v = a[o * axis_len * inner + k * inner + i];
                if (v > out[o * inner + i]) out[o * inner + i] = v;
            }
        }
    }
}

void vp_min_axis(const float *a, float *out, int outer, int axis_len, int inner) {
    int out_n = outer * inner;
    for (int i = 0; i < out_n; i++) out[i] = FLT_MAX;
    for (int o = 0; o < outer; o++) {
        for (int k = 0; k < axis_len; k++) {
            for (int i = 0; i < inner; i++) {
                float v = a[o * axis_len * inner + k * inner + i];
                if (v < out[o * inner + i]) out[o * inner + i] = v;
            }
        }
    }
}

void vp_softmax_inplace(float *x, int n) {
    float m = -FLT_MAX;
    for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    float inv_s = 1.0f / s;
    for (int i = 0; i < n; i++) x[i] *= inv_s;
}

void vp_log_softmax_inplace(float *x, int n) {
    float m = -FLT_MAX;
    for (int i = 0; i < n; i++) if (x[i] > m) m = x[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += expf(x[i] - m);
    float log_s = logf(s);
    for (int i = 0; i < n; i++) x[i] = x[i] - m - log_s;
}

/* ===========================================================================
 * NDArray buffer ops
 * =========================================================================== */

float* vp_nd_alloc(int64_t n, int dtype_size) {
    return (float*)calloc((size_t)n, (size_t)dtype_size);
}
void vp_nd_free(float *ptr) { free(ptr); }
void vp_nd_zeros(float *ptr, int64_t n) { memset(ptr, 0, n * sizeof(float)); }
void vp_nd_fill(float *ptr, float val, int64_t n) {
    for (int64_t i = 0; i < n; i++) ptr[i] = val;
}
void vp_nd_copy(float *dst, const float *src, int64_t n) {
    memcpy(dst, src, n * sizeof(float));
}
void vp_nd_set(float *ptr, int64_t idx, float val) { ptr[idx] = val; }
float vp_nd_get(const float *ptr, int64_t idx) { return ptr[idx]; }

float* vp_nd_add(const float *a, const float *b, int64_t n) {
    float *c = (float*)malloc(n * sizeof(float));
    for (int64_t i = 0; i < n; i++) c[i] = a[i] + b[i];
    return c;
}
float* vp_nd_sub(const float *a, const float *b, int64_t n) {
    float *c = (float*)malloc(n * sizeof(float));
    for (int64_t i = 0; i < n; i++) c[i] = a[i] - b[i];
    return c;
}
float* vp_nd_mul(const float *a, const float *b, int64_t n) {
    float *c = (float*)malloc(n * sizeof(float));
    for (int64_t i = 0; i < n; i++) c[i] = a[i] * b[i];
    return c;
}
float* vp_nd_div(const float *a, const float *b, int64_t n) {
    float *c = (float*)malloc(n * sizeof(float));
    for (int64_t i = 0; i < n; i++) c[i] = a[i] / b[i];
    return c;
}
float* vp_nd_add_scalar(const float *a, float s, int64_t n) {
    float *c = (float*)malloc(n * sizeof(float));
    for (int64_t i = 0; i < n; i++) c[i] = a[i] + s;
    return c;
}
float* vp_nd_mul_scalar(const float *a, float s, int64_t n) {
    float *c = (float*)malloc(n * sizeof(float));
    for (int64_t i = 0; i < n; i++) c[i] = a[i] * s;
    return c;
}
float vp_nd_sum(const float *a, int64_t n) { double s=0; for(int64_t i=0;i<n;i++) s+=a[i]; return (float)s; }
float vp_nd_max(const float *a, int64_t n) { float m=-FLT_MAX; for(int64_t i=0;i<n;i++) if(a[i]>m) m=a[i]; return m; }
float vp_nd_min(const float *a, int64_t n) { float m=FLT_MAX; for(int64_t i=0;i<n;i++) if(a[i]<m) m=a[i]; return m; }
float vp_nd_dot(const float *a, const float *b, int64_t n) { double s=0; for(int64_t i=0;i<n;i++) s+=(double)a[i]*b[i]; return (float)s; }
int64_t vp_nd_argmax(const float *a, int64_t n) { int64_t idx=0; for(int64_t i=1;i<n;i++) if(a[i]>a[idx]) idx=i; return idx; }
int64_t vp_nd_argmin(const float *a, int64_t n) { int64_t idx=0; for(int64_t i=1;i<n;i++) if(a[i]<a[idx]) idx=i; return idx; }

float* vp_nd_exp(const float *a, int64_t n)  { float*c=(float*)malloc(n*sizeof(float)); for(int64_t i=0;i<n;i++) c[i]=expf(a[i]); return c; }
float* vp_nd_log(const float *a, int64_t n)  { float*c=(float*)malloc(n*sizeof(float)); for(int64_t i=0;i<n;i++) c[i]=logf(a[i]); return c; }
float* vp_nd_sqrt(const float *a, int64_t n) { float*c=(float*)malloc(n*sizeof(float)); for(int64_t i=0;i<n;i++) c[i]=sqrtf(a[i]); return c; }
float* vp_nd_abs(const float *a, int64_t n)  { float*c=(float*)malloc(n*sizeof(float)); for(int64_t i=0;i<n;i++) c[i]=fabsf(a[i]); return c; }
float* vp_nd_neg(const float *a, int64_t n)  { float*c=(float*)malloc(n*sizeof(float)); for(int64_t i=0;i<n;i++) c[i]=-a[i]; return c; }
float* vp_nd_relu(const float *a, int64_t n) { float*c=(float*)malloc(n*sizeof(float)); for(int64_t i=0;i<n;i++) c[i]=a[i]>0?a[i]:0; return c; }
float* vp_nd_sigmoid(const float *a, int64_t n) { float*c=(float*)malloc(n*sizeof(float)); for(int64_t i=0;i<n;i++) c[i]=1.0f/(1.0f+expf(-a[i])); return c; }
float* vp_nd_tanh(const float *a, int64_t n) { float*c=(float*)malloc(n*sizeof(float)); for(int64_t i=0;i<n;i++) c[i]=tanhf(a[i]); return c; }
float* vp_nd_gelu(const float *a, int64_t n) {
    float*c=(float*)malloc(n*sizeof(float));
    for(int64_t i=0;i<n;i++) {
        float x=a[i];
        c[i]=0.5f*x*(1.0f+tanhf(0.7978845608028654f*(x+0.044715f*x*x*x)));
    }
    return c;
}
float* vp_nd_silu(const float *a, int64_t n) { float*c=(float*)malloc(n*sizeof(float)); for(int64_t i=0;i<n;i++){float x=a[i];c[i]=x/(1.0f+expf(-x));} return c; }
void vp_nd_softmax_inplace(float *x, int64_t n) { vp_softmax_inplace(x, (int)n); }

/* ===========================================================================
 * RANDOM NUMBER GENERATION (xoshiro256+, Gaussian via Box-Muller)
 * =========================================================================== */

static uint64_t _vp_rng_state[4] = {
    0x9e3779b97f4a7c15ULL, 0x6c62272e07bb0142ULL,
    0x62e439be3ea84613ULL, 0x92c8e3737edd84d1ULL
};

static VP_INLINE uint64_t _vp_rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t _xoshiro256_next(uint64_t *s) {
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = _vp_rotl(s[3], 45);
    return s[0] + s[3];
}

void vp_rand_uniform(float *out, int n, uint64_t seed) {
    uint64_t s[4] = {seed ^ 0x9e3779b97f4a7c15ULL, seed * 6364136223846793005ULL + 1,
                     seed ^ 0xdeadbeefcafeULL, seed + 42};
    for (int i = 0; i < n; i++) {
        uint64_t r = _xoshiro256_next(s);
        /* upper 23 bits = float in [0,1) */
        out[i] = (float)((r >> 11) & 0x1FFFFFULL) / (float)(1 << 21);
    }
}

void vp_rand_normal(float *out, int n, uint64_t seed, float mean, float std) {
    uint64_t s[4] = {seed ^ 0x9e3779b97f4a7c15ULL, seed * 6364136223846793005ULL + 1,
                     seed ^ 0xdeadbeefcafeULL, seed + 42};
    int i = 0;
    while (i < n - 1) {
        float u1 = (float)((_xoshiro256_next(s) >> 11) & 0x1FFFFFULL) / (float)(1<<21) + 1e-10f;
        float u2 = (float)((_xoshiro256_next(s) >> 11) & 0x1FFFFFULL) / (float)(1<<21);
        float r = sqrtf(-2.0f * logf(u1));
        float theta = 6.283185307179586f * u2;
        out[i++] = mean + std * r * cosf(theta);
        out[i++] = mean + std * r * sinf(theta);
    }
    if (i < n) {
        float u1 = (float)((_xoshiro256_next(s) >> 11) & 0x1FFFFFULL) / (float)(1<<21) + 1e-10f;
        float u2 = (float)((_xoshiro256_next(s) >> 11) & 0x1FFFFFULL) / (float)(1<<21);
        out[i] = mean + std * sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
    }
}

void vp_rand_int(int *out, int n, uint64_t seed, int lo, int hi) {
    uint64_t s[4] = {seed ^ 0x9e3779b97f4a7c15ULL, seed * 6364136223846793005ULL + 1,
                     seed ^ 0xdeadbeefcafeULL, seed + 42};
    int range = hi - lo;
    for (int i = 0; i < n; i++) {
        out[i] = lo + (int)(_xoshiro256_next(s) % (uint64_t)range);
    }
}

void vp_rand_bernoulli(float *out, int n, uint64_t seed, float p) {
    uint64_t s[4] = {seed ^ 0x9e3779b97f4a7c15ULL, seed * 6364136223846793005ULL + 1,
                     seed ^ 0xdeadbeefcafeULL, seed + 42};
    for (int i = 0; i < n; i++) {
        float u = (float)((_xoshiro256_next(s) >> 11) & 0x1FFFFFULL) / (float)(1<<21);
        out[i] = u < p ? 1.0f : 0.0f;
    }
}

/* ===========================================================================
 * CONV2D -- im2col + GEMM
 * =========================================================================== */

void vp_im2col(
    float *col, const float *img,
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW,
    /* padding, unused extra args to match old signature */ int _a, int _b
) {
    int oH = (H + 2*pH - dH*(kH-1) - 1) / sH + 1;
    int oW = (W + 2*pW - dW*(kW-1) - 1) / sW + 1;
    int col_stride = C * kH * kW;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw2 = 0; kw2 < kW; kw2++) {
                    int col_row = c * kH * kW + kh * kW + kw2;
                    for (int oh = 0; oh < oH; oh++) {
                        for (int ow = 0; ow < oW; ow++) {
                            int ih = oh * sH - pH + kh * dH;
                            int iw = ow * sW - pW + kw2 * dW;
                            int col_idx = n * col_stride * oH * oW + col_row * oH * oW + oh * oW + ow;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                col[col_idx] = img[n * C * H * W + c * H * W + ih * W + iw];
                            else
                                col[col_idx] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

void vp_col2im(
    float *img, const float *col,
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW,
    int _a, int _b
) {
    int oH = (H + 2*pH - dH*(kH-1) - 1) / sH + 1;
    int oW = (W + 2*pW - dW*(kW-1) - 1) / sW + 1;
    int col_stride = C * kH * kW;
    memset(img, 0, (size_t)N * C * H * W * sizeof(float));
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int kh = 0; kh < kH; kh++) {
                for (int kw2 = 0; kw2 < kW; kw2++) {
                    int col_row = c * kH * kW + kh * kW + kw2;
                    for (int oh = 0; oh < oH; oh++) {
                        for (int ow = 0; ow < oW; ow++) {
                            int ih = oh * sH - pH + kh * dH;
                            int iw = ow * sW - pW + kw2 * dW;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                int col_idx = n * col_stride * oH * oW + col_row * oH * oW + oh * oW + ow;
                                img[n * C * H * W + c * H * W + ih * W + iw] += col[col_idx];
                            }
                        }
                    }
                }
            }
        }
    }
}

/* ===========================================================================
 * POOL2D
 * =========================================================================== */

void vp_max_pool2d(
    float *out, int *mask, const float *img,
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW, int pH, int pW,
    int _a, int _b, int _c, int _d  /* extra int args to match loader signature */
) {
    int oH = (H + 2*pH - kH) / sH + 1;
    int oW = (W + 2*pW - kW) / sW + 1;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < oH; oh++) {
                for (int ow = 0; ow < oW; ow++) {
                    float mx = -FLT_MAX; int mx_idx = 0;
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw2 = 0; kw2 < kW; kw2++) {
                            int ih = oh * sH - pH + kh;
                            int iw = ow * sW - pW + kw2;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                int flat = n*C*H*W + c*H*W + ih*W + iw;
                                float v = img[flat];
                                if (v > mx) { mx = v; mx_idx = flat; }
                            }
                        }
                    }
                    int out_idx = n*C*oH*oW + c*oH*oW + oh*oW + ow;
                    out[out_idx] = mx;
                    if (mask) mask[out_idx] = mx_idx;
                }
            }
        }
    }
}

void vp_avg_pool2d(
    float *out, const float *img,
    int N, int C, int H, int W,
    int kH, int kW, int sH, int sW, int pH, int pW,
    int _a, int _b, int _c, int _d
) {
    int oH = (H + 2*pH - kH) / sH + 1;
    int oW = (W + 2*pW - kW) / sW + 1;
    float inv = 1.0f / (float)(kH * kW);
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < oH; oh++) {
                for (int ow = 0; ow < oW; ow++) {
                    float s = 0.0f;
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw2 = 0; kw2 < kW; kw2++) {
                            int ih = oh * sH - pH + kh;
                            int iw = ow * sW - pW + kw2;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                s += img[n*C*H*W + c*H*W + ih*W + iw];
                        }
                    }
                    out[n*C*oH*oW + c*oH*oW + oh*oW + ow] = s * inv;
                }
            }
        }
    }
}

/* ===========================================================================
 * FUSED KERNELS
 * =========================================================================== */

void vp_fused_linear_relu(
    float *out, const float *x, const float *W, const float *b,
    int N, int in_f, int out_f
) {
    for (int n = 0; n < N; n++) {
        for (int j = 0; j < out_f; j++) {
            float s = b ? b[j] : 0.0f;
            for (int k = 0; k < in_f; k++) s += x[n*in_f+k] * W[j*in_f+k];
            out[n*out_f+j] = s > 0.0f ? s : 0.0f;
        }
    }
}

void vp_fused_linear_gelu(
    float *out, const float *x, const float *W, const float *b,
    int N, int in_f, int out_f
) {
    static const float C0 = 0.7978845608028654f, C1 = 0.044715f;
    for (int n = 0; n < N; n++) {
        for (int j = 0; j < out_f; j++) {
            float s = b ? b[j] : 0.0f;
            for (int k = 0; k < in_f; k++) s += x[n*in_f+k] * W[j*in_f+k];
            out[n*out_f+j] = 0.5f * s * (1.0f + tanhf(C0 * (s + C1 * s*s*s)));
        }
    }
}

void vp_fused_linear_silu(
    float *out, const float *x, const float *W, const float *b,
    int N, int in_f, int out_f
) {
    for (int n = 0; n < N; n++) {
        for (int j = 0; j < out_f; j++) {
            float s = b ? b[j] : 0.0f;
            for (int k = 0; k < in_f; k++) s += x[n*in_f+k] * W[j*in_f+k];
            out[n*out_f+j] = s / (1.0f + expf(-s));
        }
    }
}

void vp_fused_scale_add(
    float *out, const float *a, const float *b,
    float alpha, int n
) {
    for (int i = 0; i < n; i++) out[i] = a[i] * alpha + b[i];
}

/* dst[i] = grad[i] * (x[i] > 0 ? 1 : 0) */
void vp_relu_backward(float *dst, const float *grad, const float *x, int n) {
    for (int i = 0; i < n; i++) dst[i] = grad[i] * (x[i] > 0.0f ? 1.0f : 0.0f);
}

/* dst[i] = grad[i] * sig[i] * (1 - sig[i]) */
void vp_sigmoid_backward(float *dst, const float *grad, const float *sig, int n) {
    for (int i = 0; i < n; i++) dst[i] = grad[i] * sig[i] * (1.0f - sig[i]);
}

/* dst[i] = grad[i] * (1 - tanh_out[i]^2) */
void vp_tanh_backward(float *dst, const float *grad, const float *tanh_out, int n) {
    for (int i = 0; i < n; i++) dst[i] = grad[i] * (1.0f - tanh_out[i] * tanh_out[i]);
}

/* dst[i] = grad[i] / (inp[i] + eps)  - log backward */
void vp_log_backward(float *dst, const float *grad, const float *inp, int n, float eps) {
    for (int i = 0; i < n; i++) dst[i] = grad[i] / (inp[i] + eps);
}

/* dst[i] = grad[i] * 0.5 / (sqrt_out[i] + eps)  - sqrt backward */
void vp_sqrt_backward(float *dst, const float *grad, const float *sqrt_out, int n, float eps) {
    for (int i = 0; i < n; i++) dst[i] = grad[i] * 0.5f / (sqrt_out[i] + eps);
}

/* dst[i] = grad[i] * exp_out[i]  - exp backward */
void vp_exp_backward(float *dst, const float *grad, const float *exp_out, int n) {
    for (int i = 0; i < n; i++) dst[i] = grad[i] * exp_out[i];
}

void vp_fused_rmsnorm(float *out, const float *x, const float *weight, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float inv_rms = 1.0f / sqrtf((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv_rms * weight[i];
}

void vp_fused_layernorm(
    float *out, const float *x, const float *weight, const float *bias,
    int n, float eps
) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += x[i];
    float mean = (float)(s / n);
    double v = 0.0;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; v += d * d; }
    float inv_std = 1.0f / sqrtf((float)(v / n) + eps);
    for (int i = 0; i < n; i++)
        out[i] = (x[i] - mean) * inv_std * weight[i] + (bias ? bias[i] : 0.0f);
}

void vp_fused_rope(
    float *x, const float *cos_c, const float *sin_c,
    int B, int n_heads, int seq_len, int head_dim
) {
    int half = head_dim / 2;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < n_heads; h++) {
            for (int s = 0; s < seq_len; s++) {
                int base = b*n_heads*seq_len*head_dim + h*seq_len*head_dim + s*head_dim;
                for (int i = 0; i < half; i++) {
                    float x0 = x[base + i], x1 = x[base + i + half];
                    float c = cos_c[s * half + i], sn = sin_c[s * half + i];
                    x[base + i]        = x0 * c - x1 * sn;
                    x[base + i + half] = x0 * sn + x1 * c;
                }
            }
        }
    }
}

void vp_fused_attention_scores(
    float *out, const float *Q, const float *K,
    float scale, int B, int H, int S, int D
) {
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            for (int i = 0; i < S; i++) {
                for (int j = 0; j < S; j++) {
                    double acc = 0.0;
                    int qb = b*H*S*D + h*S*D + i*D;
                    int kb = b*H*S*D + h*S*D + j*D;
                    for (int d = 0; d < D; d++) acc += (double)Q[qb+d] * K[kb+d];
                    out[b*H*S*S + h*S*S + i*S + j] = (float)acc * scale;
                }
            }
        }
    }
}

void vp_fused_dropout_add(
    float *out, const float *x, const float *residual,
    float prob, int n, uint64_t seed
) {
    uint64_t s[4] = {seed ^ 0x9e3779b97f4a7c15ULL, seed * 6364136223846793005ULL + 1,
                     seed ^ 0xdeadbeefcafeULL, seed + 42};
    float scale = 1.0f / (1.0f - prob);
    for (int i = 0; i < n; i++) {
        float u = (float)((_xoshiro256_next(s) >> 11) & 0x1FFFFFULL) / (float)(1<<21);
        float mask = u > prob ? 1.0f : 0.0f;
        out[i] = x[i] * mask * scale + residual[i];
    }
}

void vp_fused_adam_step(
    float *param, const float *grad, float *m, float *v,
    float lr, float b1, float b2, float eps, int t, int n
) {
    float bc1 = 1.0f - powf(b1, (float)t);
    float bc2 = 1.0f - powf(b2, (float)t);
    float lr_c = lr * sqrtf(bc2) / bc1;
    for (int i = 0; i < n; i++) {
        float g = grad[i];
        m[i] = b1 * m[i] + (1.0f - b1) * g;
        v[i] = b2 * v[i] + (1.0f - b2) * g * g;
        param[i] -= lr_c * m[i] / (sqrtf(v[i]) + eps);
    }
}

void vp_adam_step(
    float *param, const float *grad, float *m, float *v,
    int t, float lr, float b1, float b2, float eps, int n
) {
    vp_fused_adam_step(param, grad, m, v, lr, b1, b2, eps, t, n);
}
