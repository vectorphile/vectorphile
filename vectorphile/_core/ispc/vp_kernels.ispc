/* vp_kernels.ispc — vectorphile ISPC kernel library
 * Compiled with: ispc vp_kernels.ispc -o vp_kernels.obj
 *                --target=avx2-i32x8 --arch=x86-64 --math-lib=svml
 *
 * All kernels assume float32, C-contiguous (row-major) layout.
 * "uniform" = same value across all SIMD lanes (scalar in disguise).
 * "varying" (default) = one value per lane — the SIMD dimension.
 */

/* ------------------------------------------------------------------ */
/* Elementwise activations                                              */
/* ------------------------------------------------------------------ */

export void ispc_relu_f32(
    uniform const float src[],
    uniform float       dst[],
    uniform int         n)
{
    foreach (i = 0 ... n)
        dst[i] = max(src[i], 0.f);
}

export void ispc_relu_backward_f32(
    uniform const float grad_out[],
    uniform const float fwd_in[],
    uniform float       grad_in[],
    uniform int         n)
{
    foreach (i = 0 ... n)
        grad_in[i] = (fwd_in[i] > 0.f) ? grad_out[i] : 0.f;
}

export void ispc_sigmoid_f32(
    uniform const float src[],
    uniform float       dst[],
    uniform int         n)
{
    foreach (i = 0 ... n)
        dst[i] = 1.f / (1.f + exp(-src[i]));
}

/* tanh via identity: tanh(x) = (exp(2x)-1)/(exp(2x)+1) */
static inline varying float _tanh_f32(varying float x) {
    /* Clamp to avoid overflow in exp */
    x = clamp(x, -15.f, 15.f);
    float e2x = exp(2.f * x);
    return (e2x - 1.f) / (e2x + 1.f);
}

export void ispc_tanh_f32(
    uniform const float src[],
    uniform float       dst[],
    uniform int         n)
{
    foreach (i = 0 ... n)
        dst[i] = _tanh_f32(src[i]);
}

/* erf polynomial approximation (Abramowitz & Stegun 7.1.26, max error 1.5e-7) */
static inline varying float _erf_f32(varying float x) {
    uniform const float kA1 =  0.254829592f;
    uniform const float kA2 = -0.284496736f;
    uniform const float kA3 =  1.421413741f;
    uniform const float kA4 = -1.453152027f;
    uniform const float kA5 =  1.061405429f;
    uniform const float kP  =  0.3275911f;
    float sign = (x >= 0.f) ? 1.f : -1.f;
    x = abs(x);
    float t = 1.f / (1.f + kP * x);
    float y = 1.f - (((((kA5*t + kA4)*t) + kA3)*t + kA2)*t + kA1)*t * exp(-x*x);
    return sign * y;
}

/* GELU: x * 0.5 * (1 + erf(x / sqrt(2))) */
export void ispc_gelu_f32(
    uniform const float src[],
    uniform float       dst[],
    uniform int         n)
{
    uniform const float kInvSqrt2 = 0.7071067811865476f;
    foreach (i = 0 ... n) {
        float x = src[i];
        dst[i] = x * 0.5f * (1.f + _erf_f32(x * kInvSqrt2));
    }
}

/* Fast GELU (tanh approximation, GPT-2 style) */
export void ispc_gelu_fast_f32(
    uniform const float src[],
    uniform float       dst[],
    uniform int         n)
{
    uniform const float kC1 = 0.7978845608028654f;   /* sqrt(2/pi) */
    uniform const float kC2 = 0.044715f;
    foreach (i = 0 ... n) {
        float x = src[i];
        float inner = kC1 * (x + kC2 * x * x * x);
        dst[i] = 0.5f * x * (1.f + _tanh_f32(inner));
    }
}

export void ispc_silu_f32(
    uniform const float src[],
    uniform float       dst[],
    uniform int         n)
{
    foreach (i = 0 ... n) {
        float x = src[i];
        dst[i] = x / (1.f + exp(-x));
    }
}

/* ------------------------------------------------------------------ */
/* Softmax / LogSoftmax — row-wise, vectorised over C                  */
/* ------------------------------------------------------------------ */

/* softmax(x)[i,j] = exp(x[i,j]-max_i) / sum_j exp(x[i,j]-max_i)     */
export void ispc_softmax_rows_f32(
    uniform const float src[],
    uniform float       dst[],
    uniform int         N,
    uniform int         C)
{
    for (uniform int row = 0; row < N; row++) {
        uniform int base = row * C;

        /* Pass 1 — find row max (numerical stability) */
        float vmax = -1e38f;
        foreach (j = 0 ... C)
            vmax = max(vmax, src[base + j]);
        uniform float rmax = reduce_max(vmax);

        /* Pass 2 — exp(x-max), accumulate sum */
        float vsum = 0.f;
        foreach (j = 0 ... C) {
            float e = exp(src[base + j] - rmax);
            dst[base + j] = e;
            vsum += e;
        }
        uniform float inv_sum = 1.f / reduce_add(vsum);

        /* Pass 3 — normalise */
        foreach (j = 0 ... C)
            dst[base + j] *= inv_sum;
    }
}

/* log_softmax(x)[i,j] = x[i,j] - max_i - log(sum exp(x-max))        */
export void ispc_log_softmax_rows_f32(
    uniform const float src[],
    uniform float       dst[],
    uniform int         N,
    uniform int         C)
{
    for (uniform int row = 0; row < N; row++) {
        uniform int base = row * C;

        float vmax = -1e38f;
        foreach (j = 0 ... C)
            vmax = max(vmax, src[base + j]);
        uniform float rmax = reduce_max(vmax);

        float vsum = 0.f;
        foreach (j = 0 ... C)
            vsum += exp(src[base + j] - rmax);
        uniform float log_sum_exp = log(reduce_add(vsum));
        uniform float offset = rmax + log_sum_exp;

        foreach (j = 0 ... C)
            dst[base + j] = src[base + j] - offset;
    }
}

/* ------------------------------------------------------------------ */
/* Cross-entropy loss (fused — no intermediate softmax tensor)         */
/* targets: integer class indices [0, C)                               */
/* ------------------------------------------------------------------ */

export void ispc_cross_entropy_fwd_f32(
    uniform const float logits[],   /* N x C row-major */
    uniform const int   targets[],  /* N               */
    uniform float       losses[],   /* N per-sample losses (caller averages) */
    uniform int         N,
    uniform int         C)
{
    for (uniform int row = 0; row < N; row++) {
        uniform int base = row * C;

        float vmax = -1e38f;
        foreach (j = 0 ... C)
            vmax = max(vmax, logits[base + j]);
        uniform float rmax = reduce_max(vmax);

        float vsum = 0.f;
        foreach (j = 0 ... C)
            vsum += exp(logits[base + j] - rmax);
        uniform float log_sum_exp = log(reduce_add(vsum)) + rmax;

        uniform int t = targets[row];
        losses[row] = log_sum_exp - logits[base + t];
    }
}

/* Fused cross-entropy backward: grad_logits = softmax(logits) - one_hot(target)
 * Writes directly into grad_logits (N x C), scaled by 1/N.             */
export void ispc_cross_entropy_bwd_f32(
    uniform const float logits[],
    uniform const int   targets[],
    uniform float       grad[],     /* N x C output */
    uniform int         N,
    uniform int         C)
{
    uniform float scale = 1.f / (uniform float)N;
    for (uniform int row = 0; row < N; row++) {
        uniform int base = row * C;

        /* Compute softmax in-place into grad */
        float vmax = -1e38f;
        foreach (j = 0 ... C)
            vmax = max(vmax, logits[base + j]);
        uniform float rmax = reduce_max(vmax);

        float vsum = 0.f;
        foreach (j = 0 ... C) {
            float e = exp(logits[base + j] - rmax);
            grad[base + j] = e;
            vsum += e;
        }
        uniform float inv_sum = scale / reduce_add(vsum);

        foreach (j = 0 ... C)
            grad[base + j] *= inv_sum;

        /* Subtract 1/N at the target class */
        grad[base + targets[row]] -= scale;
    }
}

/* ------------------------------------------------------------------ */
/* Elementwise binary ops (used to avoid numpy in Tensor arithmetic)   */
/* ------------------------------------------------------------------ */

export void ispc_add_f32(
    uniform const float a[],
    uniform const float b[],
    uniform float       c[],
    uniform int         n)
{
    foreach (i = 0 ... n)
        c[i] = a[i] + b[i];
}

export void ispc_mul_f32(
    uniform const float a[],
    uniform const float b[],
    uniform float       c[],
    uniform int         n)
{
    foreach (i = 0 ... n)
        c[i] = a[i] * b[i];
}

export void ispc_sub_f32(
    uniform const float a[],
    uniform const float b[],
    uniform float       c[],
    uniform int         n)
{
    foreach (i = 0 ... n)
        c[i] = a[i] - b[i];
}

export void ispc_div_f32(
    uniform const float a[],
    uniform const float b[],
    uniform float       c[],
    uniform int         n)
{
    foreach (i = 0 ... n)
        c[i] = a[i] / b[i];
}

/* scalar broadcast versions */
export void ispc_add_scalar_f32(
    uniform const float src[],
    uniform float       scalar,
    uniform float       dst[],
    uniform int         n)
{
    foreach (i = 0 ... n)
        dst[i] = src[i] + scalar;
}

export void ispc_mul_scalar_f32(
    uniform const float src[],
    uniform float       scalar,
    uniform float       dst[],
    uniform int         n)
{
    foreach (i = 0 ... n)
        dst[i] = src[i] * scalar;
}

/* ------------------------------------------------------------------ */
/* Layer Norm (single row, vectorised over C)                          */
/* ------------------------------------------------------------------ */

export void ispc_layer_norm_f32(
    uniform const float src[],      /* 1 x C */
    uniform float       dst[],
    uniform const float weight[],   /* gamma, C */
    uniform const float bias_buf[], /* beta,  C (NULL-safe via pointer) */
    uniform int         C,
    uniform float       eps)
{
    /* Mean */
    float vsum = 0.f;
    foreach (j = 0 ... C)
        vsum += src[j];
    uniform float mean = reduce_add(vsum) / (uniform float)C;

    /* Variance */
    float vvar = 0.f;
    foreach (j = 0 ... C) {
        float d = src[j] - mean;
        vvar += d * d;
    }
    uniform float rstd = 1.f / sqrt(reduce_add(vvar) / (uniform float)C + eps);

    /* Normalise + affine */
    if (bias_buf != NULL) {
        foreach (j = 0 ... C)
            dst[j] = (src[j] - mean) * rstd * weight[j] + bias_buf[j];
    } else {
        foreach (j = 0 ... C)
            dst[j] = (src[j] - mean) * rstd * weight[j];
    }
}

/* Batch layer norm: applies ispc_layer_norm_f32 over N rows */
export void ispc_layer_norm_rows_f32(
    uniform const float src[],
    uniform float       dst[],
    uniform const float weight[],
    uniform const float bias_buf[],
    uniform int         N,
    uniform int         C,
    uniform float       eps)
{
    for (uniform int row = 0; row < N; row++) {
        uniform int base = row * C;
        float vsum = 0.f;
        foreach (j = 0 ... C)
            vsum += src[base + j];
        uniform float mean = reduce_add(vsum) / (uniform float)C;

        float vvar = 0.f;
        foreach (j = 0 ... C) {
            float d = src[base + j] - mean;
            vvar += d * d;
        }
        uniform float rstd = 1.f / sqrt(reduce_add(vvar) / (uniform float)C + eps);

        if (bias_buf != NULL) {
            foreach (j = 0 ... C)
                dst[base + j] = (src[base + j] - mean) * rstd * weight[j] + bias_buf[j];
        } else {
            foreach (j = 0 ... C)
                dst[base + j] = (src[base + j] - mean) * rstd * weight[j];
        }
    }
}

/* ------------------------------------------------------------------ */
/* Clip / clamp                                                         */
/* ------------------------------------------------------------------ */

export void ispc_clamp_f32(
    uniform const float src[],
    uniform float       dst[],
    uniform float       lo,
    uniform float       hi,
    uniform int         n)
{
    foreach (i = 0 ... n)
        dst[i] = clamp(src[i], lo, hi);
}

/* ------------------------------------------------------------------ */
/* Reduction: sum, mean, max                                            */
/* ------------------------------------------------------------------ */

export uniform float ispc_sum_f32(
    uniform const float src[],
    uniform int         n)
{
    float acc = 0.f;
    foreach (i = 0 ... n)
        acc += src[i];
    return reduce_add(acc);
}

export uniform float ispc_max_f32(
    uniform const float src[],
    uniform int         n)
{
    float acc = -1e38f;
    foreach (i = 0 ... n)
        acc = max(acc, src[i]);
    return reduce_max(acc);
}
