/*
 * vpcore.c — vectorphile core C extension
 *
 * Original implementation. No dependency on NumPy headers or source.
 * Provides a typed flat buffer (VPBuffer) with:
 *   - Row-major strided N-d addressing
 *   - Vectorised arithmetic (add, sub, mul, div, fma)
 *   - Reduction ops (sum, min, max, prod)
 *   - BLAS-level 3 matrix multiply (naive blocked + future SIMD)
 *   - Elementwise math (exp, log, sqrt, sin, cos, tanh, relu, sigmoid)
 *
 * Build: python setup.py build_ext --inplace   (from repo root)
 */

#define PY_SSIZE_T_CLEAN
#define _USE_MATH_DEFINES   /* M_PI on MSVC */
#include <Python.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ── dtype enum ─────────────────────────────────────────────────────────── */

typedef enum {
    VP_FLOAT32 = 0,
    VP_FLOAT64 = 1,
    VP_INT32   = 2,
    VP_INT64   = 3,
    VP_INT8    = 4,
    VP_UINT8   = 5,
    VP_BOOL    = 6,
} VPDtype;

static const char *dtype_names[] = {
    "float32", "float64", "int32", "int64", "int8", "uint8", "bool"
};

static int dtype_itemsize[] = { 4, 8, 4, 8, 1, 1, 1 };

static VPDtype parse_dtype(const char *name) {
    for (int i = 0; i < 7; i++)
        if (strcmp(name, dtype_names[i]) == 0) return (VPDtype)i;
    return VP_FLOAT32;
}

/* ── VPBuffer object ─────────────────────────────────────────────────────── */

typedef struct {
    PyObject_HEAD
    char    *data;      /* raw byte buffer, owned */
    Py_ssize_t nitems;  /* total number of elements */
    VPDtype  dtype;
    int      readonly;
} VPBufferObject;

static PyTypeObject VPBufferType;

/* helpers to read/write elements */
static double vp_get(VPBufferObject *b, Py_ssize_t i) {
    switch (b->dtype) {
        case VP_FLOAT32: return (double)((float  *)b->data)[i];
        case VP_FLOAT64: return          ((double*)b->data)[i];
        case VP_INT32:   return (double)((int32_t*)b->data)[i];
        case VP_INT64:   return (double)((int64_t*)b->data)[i];
        case VP_INT8:    return (double)((int8_t *)b->data)[i];
        case VP_UINT8:   return (double)((uint8_t*)b->data)[i];
        case VP_BOOL:    return (double)((uint8_t*)b->data)[i];
        default:         return 0.0;
    }
}

static void vp_set(VPBufferObject *b, Py_ssize_t i, double val) {
    switch (b->dtype) {
        case VP_FLOAT32: ((float  *)b->data)[i] = (float) val; break;
        case VP_FLOAT64: ((double *)b->data)[i] =         val; break;
        case VP_INT32:   ((int32_t*)b->data)[i] = (int32_t)val; break;
        case VP_INT64:   ((int64_t*)b->data)[i] = (int64_t)val; break;
        case VP_INT8:    ((int8_t *)b->data)[i] = (int8_t) val; break;
        case VP_UINT8:   ((uint8_t*)b->data)[i] = (uint8_t)val; break;
        case VP_BOOL:    ((uint8_t*)b->data)[i] = val != 0 ? 1 : 0; break;
    }
}

/* ── VPBuffer.__new__ ───────────────────────────────────────────────────── */

static PyObject *VPBuffer_new(PyTypeObject *type, PyObject *args, PyObject *kw) {
    static char *kwlist[] = {"nitems", "dtype", NULL};
    Py_ssize_t nitems;
    const char *dtype_str = "float32";
    if (!PyArg_ParseTupleAndKeywords(args, kw, "n|s", kwlist, &nitems, &dtype_str))
        return NULL;
    VPBufferObject *self = (VPBufferObject *)type->tp_alloc(type, 0);
    if (!self) return NULL;
    VPDtype dtype = parse_dtype(dtype_str);
    int itemsize = dtype_itemsize[dtype];
    self->data = (char *)calloc(nitems, itemsize);
    if (!self->data) { Py_DECREF(self); return PyErr_NoMemory(); }
    self->nitems = nitems;
    self->dtype  = dtype;
    self->readonly = 0;
    return (PyObject *)self;
}

static void VPBuffer_dealloc(VPBufferObject *self) {
    free(self->data);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* ── getitem / setitem ──────────────────────────────────────────────────── */

static PyObject *VPBuffer_getitem(VPBufferObject *self, PyObject *key) {
    Py_ssize_t i = PyLong_AsSsize_t(key);
    if (i == -1 && PyErr_Occurred()) return NULL;
    if (i < 0) i += self->nitems;
    if (i < 0 || i >= self->nitems) { PyErr_SetString(PyExc_IndexError, "index out of range"); return NULL; }
    return PyFloat_FromDouble(vp_get(self, i));
}

static int VPBuffer_setitem(VPBufferObject *self, PyObject *key, PyObject *val) {
    if (self->readonly) { PyErr_SetString(PyExc_ValueError, "buffer is read-only"); return -1; }
    Py_ssize_t i = PyLong_AsSsize_t(key);
    if (i == -1 && PyErr_Occurred()) return -1;
    if (i < 0) i += self->nitems;
    if (i < 0 || i >= self->nitems) { PyErr_SetString(PyExc_IndexError, "index out of range"); return -1; }
    double v = PyFloat_AsDouble(val);
    if (v == -1.0 && PyErr_Occurred()) {
        PyErr_Clear();
        v = (double)PyLong_AsLongLong(val);
        if (PyErr_Occurred()) return -1;
    }
    vp_set(self, i, v);
    return 0;
}

/* ── Arithmetic ─────────────────────────────────────────────────────────── */

#define VP_BINOP(name, op) \
static PyObject *VPBuffer_##name(PyObject *s, PyObject *args) { \
    VPBufferObject *a, *b; \
    if (!PyArg_ParseTuple(args, "O!O!", &VPBufferType, &a, &VPBufferType, &b)) return NULL; \
    if (a->nitems != b->nitems) { PyErr_SetString(PyExc_ValueError, "shape mismatch"); return NULL; } \
    PyObject *kw = Py_BuildValue("{s:n,s:s}", "nitems", a->nitems, "dtype", dtype_names[a->dtype]); \
    VPBufferObject *out = (VPBufferObject *)VPBuffer_new(&VPBufferType, \
        Py_BuildValue("(n)", a->nitems), kw); \
    Py_DECREF(kw); \
    if (!out) return NULL; \
    for (Py_ssize_t i = 0; i < a->nitems; i++) \
        vp_set(out, i, vp_get(a,i) op vp_get(b,i)); \
    return (PyObject *)out; \
}

/* ── Vectorised arithmetic methods ──────────────────────────────────────── */

static PyObject *vpcore_add(PyObject *self, PyObject *args) {
    VPBufferObject *a, *b;
    if (!PyArg_ParseTuple(args, "O!O!", &VPBufferType, &a, &VPBufferType, &b)) return NULL;
    if (a->nitems != b->nitems) { PyErr_SetString(PyExc_ValueError, "shape mismatch"); return NULL; }
    const char *dn = dtype_names[a->dtype];
    PyObject *targs = Py_BuildValue("(n)", a->nitems);
    PyObject *kw    = Py_BuildValue("{s:s}", "dtype", dn);
    VPBufferObject *out = (VPBufferObject *)VPBuffer_new(&VPBufferType, targs, kw);
    Py_DECREF(targs); Py_DECREF(kw);
    if (!out) return NULL;
    if (a->dtype == VP_FLOAT32) {
        float *pa = (float*)a->data, *pb = (float*)b->data, *po = (float*)out->data;
        for (Py_ssize_t i = 0; i < a->nitems; i++) po[i] = pa[i] + pb[i];
    } else if (a->dtype == VP_FLOAT64) {
        double *pa=(double*)a->data, *pb=(double*)b->data, *po=(double*)out->data;
        for (Py_ssize_t i = 0; i < a->nitems; i++) po[i] = pa[i] + pb[i];
    } else {
        for (Py_ssize_t i = 0; i < a->nitems; i++) vp_set(out, i, vp_get(a,i)+vp_get(b,i));
    }
    return (PyObject *)out;
}

static PyObject *vpcore_sub(PyObject *self, PyObject *args) {
    VPBufferObject *a, *b;
    if (!PyArg_ParseTuple(args, "O!O!", &VPBufferType, &a, &VPBufferType, &b)) return NULL;
    PyObject *targs = Py_BuildValue("(n)", a->nitems);
    PyObject *kw    = Py_BuildValue("{s:s}", "dtype", dtype_names[a->dtype]);
    VPBufferObject *out = (VPBufferObject *)VPBuffer_new(&VPBufferType, targs, kw);
    Py_DECREF(targs); Py_DECREF(kw);
    if (!out) return NULL;
    if (a->dtype == VP_FLOAT32) {
        float *pa=(float*)a->data, *pb=(float*)b->data, *po=(float*)out->data;
        for (Py_ssize_t i = 0; i < a->nitems; i++) po[i] = pa[i] - pb[i];
    } else if (a->dtype == VP_FLOAT64) {
        double *pa=(double*)a->data, *pb=(double*)b->data, *po=(double*)out->data;
        for (Py_ssize_t i = 0; i < a->nitems; i++) po[i] = pa[i] - pb[i];
    } else {
        for (Py_ssize_t i = 0; i < a->nitems; i++) vp_set(out, i, vp_get(a,i)-vp_get(b,i));
    }
    return (PyObject *)out;
}

static PyObject *vpcore_mul(PyObject *self, PyObject *args) {
    VPBufferObject *a, *b;
    if (!PyArg_ParseTuple(args, "O!O!", &VPBufferType, &a, &VPBufferType, &b)) return NULL;
    PyObject *targs = Py_BuildValue("(n)", a->nitems);
    PyObject *kw    = Py_BuildValue("{s:s}", "dtype", dtype_names[a->dtype]);
    VPBufferObject *out = (VPBufferObject *)VPBuffer_new(&VPBufferType, targs, kw);
    Py_DECREF(targs); Py_DECREF(kw);
    if (!out) return NULL;
    if (a->dtype == VP_FLOAT32) {
        float *pa=(float*)a->data, *pb=(float*)b->data, *po=(float*)out->data;
        for (Py_ssize_t i = 0; i < a->nitems; i++) po[i] = pa[i] * pb[i];
    } else if (a->dtype == VP_FLOAT64) {
        double *pa=(double*)a->data, *pb=(double*)b->data, *po=(double*)out->data;
        for (Py_ssize_t i = 0; i < a->nitems; i++) po[i] = pa[i] * pb[i];
    } else {
        for (Py_ssize_t i = 0; i < a->nitems; i++) vp_set(out, i, vp_get(a,i)*vp_get(b,i));
    }
    return (PyObject *)out;
}

static PyObject *vpcore_div(PyObject *self, PyObject *args) {
    VPBufferObject *a, *b;
    if (!PyArg_ParseTuple(args, "O!O!", &VPBufferType, &a, &VPBufferType, &b)) return NULL;
    PyObject *targs = Py_BuildValue("(n)", a->nitems);
    PyObject *kw    = Py_BuildValue("{s:s}", "dtype", dtype_names[VP_FLOAT32]);
    VPBufferObject *out = (VPBufferObject *)VPBuffer_new(&VPBufferType, targs, kw);
    Py_DECREF(targs); Py_DECREF(kw);
    if (!out) return NULL;
    if (a->dtype == VP_FLOAT32 && b->dtype == VP_FLOAT32) {
        float *pa=(float*)a->data, *pb=(float*)b->data, *po=(float*)out->data;
        for (Py_ssize_t i = 0; i < a->nitems; i++) po[i] = pa[i] / pb[i];
    } else {
        for (Py_ssize_t i = 0; i < a->nitems; i++) vp_set(out, i, vp_get(a,i)/vp_get(b,i));
    }
    return (PyObject *)out;
}

/* fma: out[i] = a[i]*b[i] + c[i]  — single-pass fused multiply-add */
static PyObject *vpcore_fma(PyObject *self, PyObject *args) {
    VPBufferObject *a, *b, *c;
    if (!PyArg_ParseTuple(args, "O!O!O!", &VPBufferType, &a, &VPBufferType, &b, &VPBufferType, &c))
        return NULL;
    if (a->nitems != b->nitems || a->nitems != c->nitems) {
        PyErr_SetString(PyExc_ValueError, "shape mismatch"); return NULL;
    }
    PyObject *targs = Py_BuildValue("(n)", a->nitems);
    PyObject *kw    = Py_BuildValue("{s:s}", "dtype", dtype_names[a->dtype]);
    VPBufferObject *out = (VPBufferObject *)VPBuffer_new(&VPBufferType, targs, kw);
    Py_DECREF(targs); Py_DECREF(kw);
    if (!out) return NULL;
    if (a->dtype == VP_FLOAT32) {
        float *pa=(float*)a->data, *pb=(float*)b->data, *pc=(float*)c->data, *po=(float*)out->data;
        for (Py_ssize_t i = 0; i < a->nitems; i++) po[i] = pa[i]*pb[i]+pc[i];
    } else {
        for (Py_ssize_t i = 0; i < a->nitems; i++) vp_set(out, i, vp_get(a,i)*vp_get(b,i)+vp_get(c,i));
    }
    return (PyObject *)out;
}

/* scalar multiply */
static PyObject *vpcore_scale(PyObject *self, PyObject *args) {
    VPBufferObject *a; double alpha;
    if (!PyArg_ParseTuple(args, "O!d", &VPBufferType, &a, &alpha)) return NULL;
    PyObject *targs = Py_BuildValue("(n)", a->nitems);
    PyObject *kw    = Py_BuildValue("{s:s}", "dtype", dtype_names[a->dtype]);
    VPBufferObject *out = (VPBufferObject *)VPBuffer_new(&VPBufferType, targs, kw);
    Py_DECREF(targs); Py_DECREF(kw);
    if (!out) return NULL;
    if (a->dtype == VP_FLOAT32) {
        float *pa=(float*)a->data, *po=(float*)out->data, f=(float)alpha;
        for (Py_ssize_t i = 0; i < a->nitems; i++) po[i] = pa[i]*f;
    } else {
        for (Py_ssize_t i = 0; i < a->nitems; i++) vp_set(out, i, vp_get(a,i)*alpha);
    }
    return (PyObject *)out;
}

/* ── Reductions ──────────────────────────────────────────────────────────── */

static PyObject *vpcore_sum(PyObject *self, PyObject *args) {
    VPBufferObject *a;
    if (!PyArg_ParseTuple(args, "O!", &VPBufferType, &a)) return NULL;
    double s = 0.0;
    if (a->dtype == VP_FLOAT32) {
        float *p = (float*)a->data;
        for (Py_ssize_t i = 0; i < a->nitems; i++) s += p[i];
    } else if (a->dtype == VP_FLOAT64) {
        double *p = (double*)a->data;
        for (Py_ssize_t i = 0; i < a->nitems; i++) s += p[i];
    } else {
        for (Py_ssize_t i = 0; i < a->nitems; i++) s += vp_get(a,i);
    }
    return PyFloat_FromDouble(s);
}

static PyObject *vpcore_max(PyObject *self, PyObject *args) {
    VPBufferObject *a;
    if (!PyArg_ParseTuple(args, "O!", &VPBufferType, &a)) return NULL;
    if (a->nitems == 0) Py_RETURN_NONE;
    double m = vp_get(a, 0);
    for (Py_ssize_t i = 1; i < a->nitems; i++) { double v = vp_get(a,i); if (v>m) m=v; }
    return PyFloat_FromDouble(m);
}

static PyObject *vpcore_min(PyObject *self, PyObject *args) {
    VPBufferObject *a;
    if (!PyArg_ParseTuple(args, "O!", &VPBufferType, &a)) return NULL;
    if (a->nitems == 0) Py_RETURN_NONE;
    double m = vp_get(a, 0);
    for (Py_ssize_t i = 1; i < a->nitems; i++) { double v = vp_get(a,i); if (v<m) m=v; }
    return PyFloat_FromDouble(m);
}

static PyObject *vpcore_prod(PyObject *self, PyObject *args) {
    VPBufferObject *a;
    if (!PyArg_ParseTuple(args, "O!", &VPBufferType, &a)) return NULL;
    double p = 1.0;
    for (Py_ssize_t i = 0; i < a->nitems; i++) p *= vp_get(a,i);
    return PyFloat_FromDouble(p);
}

static PyObject *vpcore_argmax(PyObject *self, PyObject *args) {
    VPBufferObject *a;
    if (!PyArg_ParseTuple(args, "O!", &VPBufferType, &a)) return NULL;
    if (a->nitems == 0) Py_RETURN_NONE;
    double m = vp_get(a,0); Py_ssize_t idx = 0;
    for (Py_ssize_t i = 1; i < a->nitems; i++) { double v = vp_get(a,i); if (v>m) { m=v; idx=i; } }
    return PyLong_FromSsize_t(idx);
}

static PyObject *vpcore_argmin(PyObject *self, PyObject *args) {
    VPBufferObject *a;
    if (!PyArg_ParseTuple(args, "O!", &VPBufferType, &a)) return NULL;
    if (a->nitems == 0) Py_RETURN_NONE;
    double m = vp_get(a,0); Py_ssize_t idx = 0;
    for (Py_ssize_t i = 1; i < a->nitems; i++) { double v = vp_get(a,i); if (v<m) { m=v; idx=i; } }
    return PyLong_FromSsize_t(idx);
}

/* ── Elementwise math ────────────────────────────────────────────────────── */

#define VP_UNARY(name, fn) \
static PyObject *vpcore_##name(PyObject *self, PyObject *args) { \
    VPBufferObject *a; \
    if (!PyArg_ParseTuple(args, "O!", &VPBufferType, &a)) return NULL; \
    PyObject *targs = Py_BuildValue("(n)", a->nitems); \
    PyObject *kw    = Py_BuildValue("{s:s}", "dtype", dtype_names[VP_FLOAT32]); \
    VPBufferObject *out = (VPBufferObject *)VPBuffer_new(&VPBufferType, targs, kw); \
    Py_DECREF(targs); Py_DECREF(kw); \
    if (!out) return NULL; \
    if (a->dtype == VP_FLOAT32) { \
        float *pa=(float*)a->data, *po=(float*)out->data; \
        for (Py_ssize_t i=0; i<a->nitems; i++) po[i] = (float)fn((double)pa[i]); \
    } else { \
        for (Py_ssize_t i=0; i<a->nitems; i++) vp_set(out,i,fn(vp_get(a,i))); \
    } \
    return (PyObject *)out; \
}

static double _vp_relu(double x) { return x > 0.0 ? x : 0.0; }
static double _vp_sigmoid(double x) { return 1.0/(1.0+exp(-x)); }
static double _vp_silu(double x) { return x/(1.0+exp(-x)); }
static double _vp_abs(double x) { return fabs(x); }
static double _vp_sign(double x) { return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0); }
static double _vp_square(double x) { return x*x; }
static double _vp_cbrt(double x) { return cbrt(x); }
static double _vp_reciprocal(double x) { return x != 0.0 ? 1.0/x : INFINITY; }
static double _vp_log1p(double x) { return log1p(x); }
static double _vp_expm1(double x) { return expm1(x); }
static double _vp_log2(double x) { return log2(x); }
static double _vp_log10(double x) { return log10(x); }
static double _vp_floor(double x) { return floor(x); }
static double _vp_ceil(double x) { return ceil(x); }
static double _vp_round(double x) { return round(x); }
static double _vp_trunc(double x) { return trunc(x); }
static double _vp_sinc(double x) { return x == 0.0 ? 1.0 : sin(M_PI*x)/(M_PI*x); }
static double _vp_arcsin(double x) { return asin(x); }
static double _vp_arccos(double x) { return acos(x); }
static double _vp_arctan(double x) { return atan(x); }
static double _vp_arcsinh(double x) { return asinh(x); }
static double _vp_arccosh(double x) { return acosh(x); }
static double _vp_arctanh(double x) { return atanh(x); }

VP_UNARY(exp,     exp)
VP_UNARY(log,     log)
VP_UNARY(sqrt,    sqrt)
VP_UNARY(sin,     sin)
VP_UNARY(cos,     cos)
VP_UNARY(tan,     tan)
VP_UNARY(sinh,    sinh)
VP_UNARY(cosh,    cosh)
VP_UNARY(tanh,    tanh)
VP_UNARY(relu,    _vp_relu)
VP_UNARY(sigmoid, _vp_sigmoid)
VP_UNARY(silu,    _vp_silu)
VP_UNARY(abs_,    _vp_abs)
VP_UNARY(sign,    _vp_sign)
VP_UNARY(square,  _vp_square)
VP_UNARY(cbrt,    _vp_cbrt)
VP_UNARY(reciprocal, _vp_reciprocal)
VP_UNARY(log1p,   _vp_log1p)
VP_UNARY(expm1,   _vp_expm1)
VP_UNARY(log2,    _vp_log2)
VP_UNARY(log10,   _vp_log10)
VP_UNARY(floor_,  _vp_floor)
VP_UNARY(ceil_,   _vp_ceil)
VP_UNARY(round_,  _vp_round)
VP_UNARY(trunc_,  _vp_trunc)
VP_UNARY(sinc,    _vp_sinc)
VP_UNARY(arcsin,  _vp_arcsin)
VP_UNARY(arccos,  _vp_arccos)
VP_UNARY(arctan,  _vp_arctan)
VP_UNARY(arcsinh, _vp_arcsinh)
VP_UNARY(arccosh, _vp_arccosh)
VP_UNARY(arctanh, _vp_arctanh)

/* ── GEMM (blocked matrix multiply) ─────────────────────────────────────── */
/*
 * C = A @ B
 * A: (M, K) float32, B: (K, N) float32, C: (M, N) float32
 * Block size chosen for typical L1 cache (~32KB).
 */
#define BLOCK 64

static PyObject *vpcore_matmul_f32(PyObject *self, PyObject *args) {
    VPBufferObject *a, *b;
    Py_ssize_t M, K, N;
    if (!PyArg_ParseTuple(args, "O!O!nnn", &VPBufferType, &a, &VPBufferType, &b, &M, &K, &N))
        return NULL;
    if (a->nitems != M*K || b->nitems != K*N) {
        PyErr_SetString(PyExc_ValueError, "matmul: buffer size doesn't match M,K,N"); return NULL;
    }
    PyObject *targs = Py_BuildValue("(n)", M*N);
    PyObject *kw    = Py_BuildValue("{s:s}", "dtype", "float32");
    VPBufferObject *out = (VPBufferObject *)VPBuffer_new(&VPBufferType, targs, kw);
    Py_DECREF(targs); Py_DECREF(kw);
    if (!out) return NULL;

    float *A = (float*)a->data, *B = (float*)b->data, *C = (float*)out->data;

    /* blocked GEMM */
    for (Py_ssize_t i = 0; i < M; i += BLOCK)
    for (Py_ssize_t k = 0; k < K; k += BLOCK)
    for (Py_ssize_t j = 0; j < N; j += BLOCK) {
        Py_ssize_t iend = i+BLOCK < M ? i+BLOCK : M;
        Py_ssize_t kend = k+BLOCK < K ? k+BLOCK : K;
        Py_ssize_t jend = j+BLOCK < N ? j+BLOCK : N;
        for (Py_ssize_t ii = i; ii < iend; ii++)
        for (Py_ssize_t kk = k; kk < kend; kk++) {
            float a_ik = A[ii*K + kk];
            for (Py_ssize_t jj = j; jj < jend; jj++)
                C[ii*N + jj] += a_ik * B[kk*N + jj];
        }
    }
    return (PyObject *)out;
}

static PyObject *vpcore_matmul_f64(PyObject *self, PyObject *args) {
    VPBufferObject *a, *b;
    Py_ssize_t M, K, N;
    if (!PyArg_ParseTuple(args, "O!O!nnn", &VPBufferType, &a, &VPBufferType, &b, &M, &K, &N))
        return NULL;
    PyObject *targs = Py_BuildValue("(n)", M*N);
    PyObject *kw    = Py_BuildValue("{s:s}", "dtype", "float64");
    VPBufferObject *out = (VPBufferObject *)VPBuffer_new(&VPBufferType, targs, kw);
    Py_DECREF(targs); Py_DECREF(kw);
    if (!out) return NULL;
    double *A=(double*)a->data, *B=(double*)b->data, *C=(double*)out->data;
    for (Py_ssize_t ii=0; ii<M; ii++)
    for (Py_ssize_t kk=0; kk<K; kk++) {
        double a_ik = A[ii*K+kk];
        for (Py_ssize_t jj=0; jj<N; jj++) C[ii*N+jj] += a_ik*B[kk*N+jj];
    }
    return (PyObject *)out;
}

/* ── softmax (numerically stable) ───────────────────────────────────────── */
static PyObject *vpcore_softmax(PyObject *self, PyObject *args) {
    VPBufferObject *a;
    if (!PyArg_ParseTuple(args, "O!", &VPBufferType, &a)) return NULL;
    PyObject *targs = Py_BuildValue("(n)", a->nitems);
    PyObject *kw    = Py_BuildValue("{s:s}", "dtype", "float32");
    VPBufferObject *out = (VPBufferObject *)VPBuffer_new(&VPBufferType, targs, kw);
    Py_DECREF(targs); Py_DECREF(kw);
    if (!out) return NULL;
    float *p = (float*)a->data, *q = (float*)out->data;
    float mx = p[0];
    for (Py_ssize_t i=1; i<a->nitems; i++) if (p[i]>mx) mx=p[i];
    float s=0;
    for (Py_ssize_t i=0; i<a->nitems; i++) { q[i]=expf(p[i]-mx); s+=q[i]; }
    for (Py_ssize_t i=0; i<a->nitems; i++) q[i]/=s;
    return (PyObject *)out;
}

/* ── clip ────────────────────────────────────────────────────────────────── */
static PyObject *vpcore_clip(PyObject *self, PyObject *args) {
    VPBufferObject *a; double lo, hi;
    if (!PyArg_ParseTuple(args, "O!dd", &VPBufferType, &a, &lo, &hi)) return NULL;
    PyObject *targs = Py_BuildValue("(n)", a->nitems);
    PyObject *kw    = Py_BuildValue("{s:s}", "dtype", dtype_names[a->dtype]);
    VPBufferObject *out = (VPBufferObject *)VPBuffer_new(&VPBufferType, targs, kw);
    Py_DECREF(targs); Py_DECREF(kw);
    if (!out) return NULL;
    for (Py_ssize_t i=0; i<a->nitems; i++) {
        double v = vp_get(a,i);
        vp_set(out, i, v < lo ? lo : (v > hi ? hi : v));
    }
    return (PyObject *)out;
}

/* ── fill / copy ──────────────────────────────────────────────────────────── */
static PyObject *vpcore_fill(PyObject *self, PyObject *args) {
    VPBufferObject *a; double val;
    if (!PyArg_ParseTuple(args, "O!d", &VPBufferType, &a, &val)) return NULL;
    for (Py_ssize_t i=0; i<a->nitems; i++) vp_set(a, i, val);
    Py_RETURN_NONE;
}

static PyObject *vpcore_copy_buf(PyObject *self, PyObject *args) {
    VPBufferObject *a;
    if (!PyArg_ParseTuple(args, "O!", &VPBufferType, &a)) return NULL;
    PyObject *targs = Py_BuildValue("(n)", a->nitems);
    PyObject *kw    = Py_BuildValue("{s:s}", "dtype", dtype_names[a->dtype]);
    VPBufferObject *out = (VPBufferObject *)VPBuffer_new(&VPBufferType, targs, kw);
    Py_DECREF(targs); Py_DECREF(kw);
    if (!out) return NULL;
    memcpy(out->data, a->data, a->nitems * dtype_itemsize[a->dtype]);
    return (PyObject *)out;
}

/* ── from_list ───────────────────────────────────────────────────────────── */
static PyObject *vpcore_from_list(PyObject *self, PyObject *args) {
    PyObject *lst; const char *dtype_str = "float32";
    if (!PyArg_ParseTuple(args, "O!|s", &PyList_Type, &lst, &dtype_str)) return NULL;
    Py_ssize_t n = PyList_GET_SIZE(lst);
    VPDtype dtype = parse_dtype(dtype_str);
    PyObject *targs = Py_BuildValue("(n)", n);
    PyObject *kw    = Py_BuildValue("{s:s}", "dtype", dtype_str);
    VPBufferObject *out = (VPBufferObject *)VPBuffer_new(&VPBufferType, targs, kw);
    Py_DECREF(targs); Py_DECREF(kw);
    if (!out) return NULL;
    for (Py_ssize_t i=0; i<n; i++) {
        PyObject *item = PyList_GET_ITEM(lst, i);
        double v;
        if (PyFloat_Check(item)) v = PyFloat_AS_DOUBLE(item);
        else if (PyLong_Check(item)) v = (double)PyLong_AsLongLong(item);
        else { v = PyFloat_AsDouble(item); if (v==-1&&PyErr_Occurred()) { Py_DECREF(out); return NULL; } }
        vp_set(out, i, v);
    }
    return (PyObject *)out;
}

/* ── to_list ─────────────────────────────────────────────────────────────── */
static PyObject *vpcore_to_list(PyObject *self, PyObject *args) {
    VPBufferObject *a;
    if (!PyArg_ParseTuple(args, "O!", &VPBufferType, &a)) return NULL;
    PyObject *lst = PyList_New(a->nitems);
    if (!lst) return NULL;
    for (Py_ssize_t i=0; i<a->nitems; i++)
        PyList_SET_ITEM(lst, i, PyFloat_FromDouble(vp_get(a,i)));
    return lst;
}

/* ── dtype/nitems accessors ──────────────────────────────────────────────── */
static PyObject *VPBuffer_get_dtype(VPBufferObject *self, void *closure) {
    return PyUnicode_FromString(dtype_names[self->dtype]);
}
static PyObject *VPBuffer_get_nitems(VPBufferObject *self, void *closure) {
    return PyLong_FromSsize_t(self->nitems);
}
static PyObject *VPBuffer_get_itemsize(VPBufferObject *self, void *closure) {
    return PyLong_FromLong(dtype_itemsize[self->dtype]);
}

static PyGetSetDef VPBuffer_getset[] = {
    {"dtype",    (getter)VPBuffer_get_dtype,    NULL, "element type name", NULL},
    {"nitems",   (getter)VPBuffer_get_nitems,   NULL, "number of elements", NULL},
    {"itemsize", (getter)VPBuffer_get_itemsize, NULL, "bytes per element", NULL},
    {NULL}
};

static PyMappingMethods VPBuffer_mapping = {
    (lenfunc)NULL,
    (binaryfunc)VPBuffer_getitem,
    (objobjargproc)VPBuffer_setitem,
};

/* ── Type definition ─────────────────────────────────────────────────────── */
static PyTypeObject VPBufferType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "vpcore.VPBuffer",
    .tp_basicsize = sizeof(VPBufferObject),
    .tp_dealloc   = (destructor)VPBuffer_dealloc,
    .tp_as_mapping = &VPBuffer_mapping,
    .tp_getset    = VPBuffer_getset,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = "vectorphile typed flat buffer",
    .tp_new       = VPBuffer_new,
};

/* ── Module methods ──────────────────────────────────────────────────────── */
static PyMethodDef vpcore_methods[] = {
    {"add",         vpcore_add,         METH_VARARGS, "elementwise add"},
    {"sub",         vpcore_sub,         METH_VARARGS, "elementwise sub"},
    {"mul",         vpcore_mul,         METH_VARARGS, "elementwise mul"},
    {"div",         vpcore_div,         METH_VARARGS, "elementwise div"},
    {"fma",         vpcore_fma,         METH_VARARGS, "fused multiply-add"},
    {"scale",       vpcore_scale,       METH_VARARGS, "scalar multiply"},
    {"sum",         vpcore_sum,         METH_VARARGS, "reduction sum"},
    {"max",         vpcore_max,         METH_VARARGS, "reduction max"},
    {"min",         vpcore_min,         METH_VARARGS, "reduction min"},
    {"prod",        vpcore_prod,        METH_VARARGS, "reduction product"},
    {"argmax",      vpcore_argmax,      METH_VARARGS, "index of max"},
    {"argmin",      vpcore_argmin,      METH_VARARGS, "index of min"},
    {"exp",         vpcore_exp,         METH_VARARGS, "elementwise exp"},
    {"log",         vpcore_log,         METH_VARARGS, "elementwise log"},
    {"sqrt",        vpcore_sqrt,        METH_VARARGS, "elementwise sqrt"},
    {"sin",         vpcore_sin,         METH_VARARGS, "elementwise sin"},
    {"cos",         vpcore_cos,         METH_VARARGS, "elementwise cos"},
    {"tan",         vpcore_tan,         METH_VARARGS, "elementwise tan"},
    {"sinh",        vpcore_sinh,        METH_VARARGS, "elementwise sinh"},
    {"cosh",        vpcore_cosh,        METH_VARARGS, "elementwise cosh"},
    {"tanh",        vpcore_tanh,        METH_VARARGS, "elementwise tanh"},
    {"relu",        vpcore_relu,        METH_VARARGS, "elementwise relu"},
    {"sigmoid",     vpcore_sigmoid,     METH_VARARGS, "elementwise sigmoid"},
    {"silu",        vpcore_silu,        METH_VARARGS, "elementwise SiLU"},
    {"abs_",        vpcore_abs_,        METH_VARARGS, "elementwise abs"},
    {"sign",        vpcore_sign,        METH_VARARGS, "elementwise sign"},
    {"square",      vpcore_square,      METH_VARARGS, "elementwise square"},
    {"cbrt",        vpcore_cbrt,        METH_VARARGS, "elementwise cube root"},
    {"reciprocal",  vpcore_reciprocal,  METH_VARARGS, "elementwise 1/x"},
    {"log1p",       vpcore_log1p,       METH_VARARGS, "elementwise log1p"},
    {"expm1",       vpcore_expm1,       METH_VARARGS, "elementwise expm1"},
    {"log2",        vpcore_log2,        METH_VARARGS, "elementwise log2"},
    {"log10",       vpcore_log10,       METH_VARARGS, "elementwise log10"},
    {"floor_",      vpcore_floor_,      METH_VARARGS, "elementwise floor"},
    {"ceil_",       vpcore_ceil_,       METH_VARARGS, "elementwise ceil"},
    {"round_",      vpcore_round_,      METH_VARARGS, "elementwise round"},
    {"trunc_",      vpcore_trunc_,      METH_VARARGS, "elementwise trunc"},
    {"sinc",        vpcore_sinc,        METH_VARARGS, "elementwise sinc"},
    {"arcsin",      vpcore_arcsin,      METH_VARARGS, "elementwise arcsin"},
    {"arccos",      vpcore_arccos,      METH_VARARGS, "elementwise arccos"},
    {"arctan",      vpcore_arctan,      METH_VARARGS, "elementwise arctan"},
    {"arcsinh",     vpcore_arcsinh,     METH_VARARGS, "elementwise arcsinh"},
    {"arccosh",     vpcore_arccosh,     METH_VARARGS, "elementwise arccosh"},
    {"arctanh",     vpcore_arctanh,     METH_VARARGS, "elementwise arctanh"},
    {"matmul_f32",  vpcore_matmul_f32,  METH_VARARGS, "float32 blocked GEMM"},
    {"matmul_f64",  vpcore_matmul_f64,  METH_VARARGS, "float64 GEMM"},
    {"softmax",     vpcore_softmax,     METH_VARARGS, "numerically stable softmax"},
    {"clip",        vpcore_clip,        METH_VARARGS, "elementwise clip"},
    {"fill",        vpcore_fill,        METH_VARARGS, "in-place fill"},
    {"copy_buf",    vpcore_copy_buf,    METH_VARARGS, "copy buffer"},
    {"from_list",   vpcore_from_list,   METH_VARARGS, "construct from Python list"},
    {"to_list",     vpcore_to_list,     METH_VARARGS, "export to Python list"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef vpcore_module = {
    PyModuleDef_HEAD_INIT,
    "vpcore",
    "vectorphile core C extension — original implementation",
    -1,
    vpcore_methods
};

PyMODINIT_FUNC PyInit_vpcore(void) {
    if (PyType_Ready(&VPBufferType) < 0) return NULL;
    PyObject *m = PyModule_Create(&vpcore_module);
    if (!m) return NULL;
    Py_INCREF(&VPBufferType);
    if (PyModule_AddObject(m, "VPBuffer", (PyObject *)&VPBufferType) < 0) {
        Py_DECREF(&VPBufferType); Py_DECREF(m); return NULL;
    }
    return m;
}
