/* vp_ispc_bridge.c — Python C-API wrappers for ISPC kernels
 *
 * Compiled and linked together with vp_core.c.
 * ISPC-generated header (vp_kernels_ispc.h) is auto-produced by the
 * ISPC compiler and placed next to this file during the build.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "vp_kernels_ispc.h"

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

static int _get_buf_f32(PyObject *obj, Py_buffer *view) {
    return PyObject_GetBuffer(obj, view, PyBUF_SIMPLE | PyBUF_FORMAT);
}

static int _get_buf_i32(PyObject *obj, Py_buffer *view) {
    return PyObject_GetBuffer(obj, view, PyBUF_SIMPLE | PyBUF_FORMAT);
}

#define RELEASE(v) PyBuffer_Release(&(v))

/* ------------------------------------------------------------------ */
/* relu                                                                 */
/* ------------------------------------------------------------------ */

PyObject* vp_ispc_relu_f32(PyObject *self, PyObject *args) {
    PyObject *src_obj, *dst_obj;
    if (!PyArg_ParseTuple(args, "OO", &src_obj, &dst_obj)) return NULL;
    Py_buffer src, dst;
    if (_get_buf_f32(src_obj, &src)) return NULL;
    if (_get_buf_f32(dst_obj, &dst)) { RELEASE(src); return NULL; }
    Py_ssize_t n = src.len / sizeof(float);
    ispc_relu_f32((float*)src.buf, (float*)dst.buf, (int)n);
    RELEASE(src); RELEASE(dst);
    Py_RETURN_NONE;
}

PyObject* vp_ispc_relu_backward_f32(PyObject *self, PyObject *args) {
    PyObject *go, *fi, *gi;
    if (!PyArg_ParseTuple(args, "OOO", &go, &fi, &gi)) return NULL;
    Py_buffer bgo, bfi, bgi;
    if (_get_buf_f32(go, &bgo)) return NULL;
    if (_get_buf_f32(fi, &bfi)) { RELEASE(bgo); return NULL; }
    if (_get_buf_f32(gi, &bgi)) { RELEASE(bgo); RELEASE(bfi); return NULL; }
    Py_ssize_t n = bgo.len / sizeof(float);
    ispc_relu_backward_f32((float*)bgo.buf, (float*)bfi.buf, (float*)bgi.buf, (int)n);
    RELEASE(bgo); RELEASE(bfi); RELEASE(bgi);
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* sigmoid / tanh / gelu / silu                                        */
/* ------------------------------------------------------------------ */

#define MAKE_UNARY(name, ispc_fn)                                          \
PyObject* vp_ispc_##name(PyObject *self, PyObject *args) {                 \
    PyObject *src_obj, *dst_obj;                                           \
    if (!PyArg_ParseTuple(args, "OO", &src_obj, &dst_obj)) return NULL;   \
    Py_buffer src, dst;                                                    \
    if (_get_buf_f32(src_obj, &src)) return NULL;                         \
    if (_get_buf_f32(dst_obj, &dst)) { RELEASE(src); return NULL; }       \
    Py_ssize_t n = src.len / sizeof(float);                                \
    ispc_fn((float*)src.buf, (float*)dst.buf, (int)n);                    \
    RELEASE(src); RELEASE(dst);                                            \
    Py_RETURN_NONE;                                                        \
}

MAKE_UNARY(sigmoid_f32, ispc_sigmoid_f32)
MAKE_UNARY(tanh_f32,    ispc_tanh_f32)
MAKE_UNARY(gelu_f32,    ispc_gelu_f32)
MAKE_UNARY(gelu_fast_f32, ispc_gelu_fast_f32)
MAKE_UNARY(silu_f32,    ispc_silu_f32)

/* ------------------------------------------------------------------ */
/* softmax / log_softmax                                               */
/* ------------------------------------------------------------------ */

PyObject* vp_ispc_softmax_rows_f32(PyObject *self, PyObject *args) {
    PyObject *src_obj, *dst_obj;
    int N, C;
    if (!PyArg_ParseTuple(args, "OOii", &src_obj, &dst_obj, &N, &C)) return NULL;
    Py_buffer src, dst;
    if (_get_buf_f32(src_obj, &src)) return NULL;
    if (_get_buf_f32(dst_obj, &dst)) { RELEASE(src); return NULL; }
    ispc_softmax_rows_f32((float*)src.buf, (float*)dst.buf, N, C);
    RELEASE(src); RELEASE(dst);
    Py_RETURN_NONE;
}

PyObject* vp_ispc_log_softmax_rows_f32(PyObject *self, PyObject *args) {
    PyObject *src_obj, *dst_obj;
    int N, C;
    if (!PyArg_ParseTuple(args, "OOii", &src_obj, &dst_obj, &N, &C)) return NULL;
    Py_buffer src, dst;
    if (_get_buf_f32(src_obj, &src)) return NULL;
    if (_get_buf_f32(dst_obj, &dst)) { RELEASE(src); return NULL; }
    ispc_log_softmax_rows_f32((float*)src.buf, (float*)dst.buf, N, C);
    RELEASE(src); RELEASE(dst);
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* Cross-entropy                                                        */
/* ------------------------------------------------------------------ */

PyObject* vp_ispc_cross_entropy_fwd_f32(PyObject *self, PyObject *args) {
    PyObject *logits_obj, *targets_obj, *losses_obj;
    int N, C;
    if (!PyArg_ParseTuple(args, "OOOii",
            &logits_obj, &targets_obj, &losses_obj, &N, &C)) return NULL;
    Py_buffer bl, bt, blo;
    if (_get_buf_f32(logits_obj,  &bl))  return NULL;
    if (_get_buf_i32(targets_obj, &bt))  { RELEASE(bl); return NULL; }
    if (_get_buf_f32(losses_obj,  &blo)) { RELEASE(bl); RELEASE(bt); return NULL; }
    ispc_cross_entropy_fwd_f32(
        (float*)bl.buf, (int*)bt.buf, (float*)blo.buf, N, C);
    RELEASE(bl); RELEASE(bt); RELEASE(blo);
    Py_RETURN_NONE;
}

PyObject* vp_ispc_cross_entropy_bwd_f32(PyObject *self, PyObject *args) {
    PyObject *logits_obj, *targets_obj, *grad_obj;
    int N, C;
    if (!PyArg_ParseTuple(args, "OOOii",
            &logits_obj, &targets_obj, &grad_obj, &N, &C)) return NULL;
    Py_buffer bl, bt, bg;
    if (_get_buf_f32(logits_obj,  &bl)) return NULL;
    if (_get_buf_i32(targets_obj, &bt)) { RELEASE(bl); return NULL; }
    if (_get_buf_f32(grad_obj,    &bg)) { RELEASE(bl); RELEASE(bt); return NULL; }
    ispc_cross_entropy_bwd_f32(
        (float*)bl.buf, (int*)bt.buf, (float*)bg.buf, N, C);
    RELEASE(bl); RELEASE(bt); RELEASE(bg);
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* Layer norm                                                           */
/* ------------------------------------------------------------------ */

PyObject* vp_ispc_layer_norm_rows_f32(PyObject *self, PyObject *args) {
    PyObject *src_obj, *dst_obj, *w_obj, *b_obj;
    int N, C;
    float eps;
    if (!PyArg_ParseTuple(args, "OOOOiif",
            &src_obj, &dst_obj, &w_obj, &b_obj, &N, &C, &eps)) return NULL;
    Py_buffer bs, bd, bw;
    if (_get_buf_f32(src_obj, &bs)) return NULL;
    if (_get_buf_f32(dst_obj, &bd)) { RELEASE(bs); return NULL; }
    if (_get_buf_f32(w_obj,   &bw)) { RELEASE(bs); RELEASE(bd); return NULL; }
    float* bias_ptr = NULL;
    Py_buffer bb;
    int has_bias = (b_obj != Py_None);
    if (has_bias) {
        if (_get_buf_f32(b_obj, &bb)) { RELEASE(bs); RELEASE(bd); RELEASE(bw); return NULL; }
        bias_ptr = (float*)bb.buf;
    }
    ispc_layer_norm_rows_f32(
        (float*)bs.buf, (float*)bd.buf,
        (float*)bw.buf, bias_ptr,
        N, C, eps);
    RELEASE(bs); RELEASE(bd); RELEASE(bw);
    if (has_bias) RELEASE(bb);
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* Clamp                                                                */
/* ------------------------------------------------------------------ */

PyObject* vp_ispc_clamp_f32(PyObject *self, PyObject *args) {
    PyObject *src_obj, *dst_obj;
    float lo, hi;
    if (!PyArg_ParseTuple(args, "OOff", &src_obj, &dst_obj, &lo, &hi)) return NULL;
    Py_buffer src, dst;
    if (_get_buf_f32(src_obj, &src)) return NULL;
    if (_get_buf_f32(dst_obj, &dst)) { RELEASE(src); return NULL; }
    Py_ssize_t n = src.len / sizeof(float);
    ispc_clamp_f32((float*)src.buf, (float*)dst.buf, lo, hi, (int)n);
    RELEASE(src); RELEASE(dst);
    Py_RETURN_NONE;
}

/* ------------------------------------------------------------------ */
/* Method table — appended to vp_core's PyMethodDef table at link time */
/* ------------------------------------------------------------------ */

PyMethodDef vp_ispc_methods[] = {
    {"ispc_relu_f32",             vp_ispc_relu_f32,             METH_VARARGS, "ISPC relu"},
    {"ispc_relu_backward_f32",    vp_ispc_relu_backward_f32,    METH_VARARGS, "ISPC relu backward"},
    {"ispc_sigmoid_f32",          vp_ispc_sigmoid_f32,          METH_VARARGS, "ISPC sigmoid"},
    {"ispc_tanh_f32",             vp_ispc_tanh_f32,             METH_VARARGS, "ISPC tanh"},
    {"ispc_gelu_f32",             vp_ispc_gelu_f32,             METH_VARARGS, "ISPC gelu (exact)"},
    {"ispc_gelu_fast_f32",        vp_ispc_gelu_fast_f32,        METH_VARARGS, "ISPC gelu (tanh approx)"},
    {"ispc_silu_f32",             vp_ispc_silu_f32,             METH_VARARGS, "ISPC silu"},
    {"ispc_softmax_rows_f32",     vp_ispc_softmax_rows_f32,     METH_VARARGS, "ISPC softmax (row-wise)"},
    {"ispc_log_softmax_rows_f32", vp_ispc_log_softmax_rows_f32, METH_VARARGS, "ISPC log_softmax (row-wise)"},
    {"ispc_cross_entropy_fwd_f32",vp_ispc_cross_entropy_fwd_f32,METH_VARARGS, "ISPC cross-entropy forward"},
    {"ispc_cross_entropy_bwd_f32",vp_ispc_cross_entropy_bwd_f32,METH_VARARGS, "ISPC cross-entropy backward"},
    {"ispc_layer_norm_rows_f32",  vp_ispc_layer_norm_rows_f32,  METH_VARARGS, "ISPC layer norm (row-wise)"},
    {"ispc_clamp_f32",            vp_ispc_clamp_f32,            METH_VARARGS, "ISPC clamp"},
    {NULL, NULL, 0, NULL}
};
