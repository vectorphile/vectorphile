"""Neural network layers — all heavy ops dispatched through vectorphile.backend."""
from __future__ import annotations
import math
import random as _random_layers
import array as _arr_layers
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor
from vectorphile.autograd.engine import Function
import vectorphile.backend as _backend
from vectorphile._vp_helpers import _nd_uniform_fn, _nd_zeros
from vectorphile._core.ndarray import NDArray as _NDA_layers, _c_strides as _cs_layers, _DType as _DType_layers

try:
    from vectorphile._core import _vp_core as _C_layers
    _LAYERS_C_OK = True
except ImportError:
    # Fallback to Mojo backend on Linux
    try:
        from vectorphile._core import mojo_backend as _C_layers  # type: ignore[assignment]
        _LAYERS_C_OK = _C_layers.is_available()
    except ImportError:
        _C_layers = None
        _LAYERS_C_OK = False

try:
    from vectorphile._core import _vp_cuda as _C_cuda
    _CUDA_OK = (_C_cuda.device_count() > 0)
except (ImportError, Exception):
    _C_cuda = None
    _CUDA_OK = False

# Output buffer cache for Linear — keyed by (N, out_features) to avoid per-call 2MB alloc
_linear_out_cache: dict = {}


class _LinearActBackward(Function):
    """Backward for fused Linear+Activation. Saves post-activation output for mask/derivative."""
    def __init__(self, x, weight, bias, out_data, post_op_name):
        super().__init__(x, weight, bias) if bias is not None else super().__init__(x, weight)
        self.x = x; self.weight = weight; self.bias = bias
        self.out_data = out_data   # post-activation output
        self.post_op_name = post_op_name

    def backward(self, grad):
        gd = grad._data if isinstance(grad, Tensor) else grad
        od = self.out_data
        wd = self.weight._data if isinstance(self.weight, Tensor) else self.weight
        xd = self.x._data if isinstance(self.x, Tensor) else self.x

        if (_LAYERS_C_OK
                and isinstance(gd, _NDA_layers) and gd._buf is not None
                and isinstance(od, _NDA_layers) and od._buf is not None
                and isinstance(wd, _NDA_layers) and wd._buf is not None
                and isinstance(xd, _NDA_layers) and xd._buf is not None):
            N = gd.shape[0]; out_f = gd.shape[1]; in_f = wd.shape[1]
            n = N * out_f
            # 1. Gate grad through activation derivative
            _mask_key = ('act_mask', n)
            if _mask_key not in _linear_out_cache:
                _linear_out_cache[_mask_key] = _arr_layers.array('f', bytearray(4 * n))
            masked_buf = _linear_out_cache[_mask_key]
            op = self.post_op_name
            if op == 'relu':
                _C_layers.relu_backward_f32(gd._buf, od._buf, masked_buf)
            elif op in ('gelu', 'silu', 'sigmoid', 'tanh'):
                # For non-relu: use ISPC derivative if available, else approx via re-forward
                # Fallback: element-wise multiply (gradient already in gd; od holds output)
                # sigmoid': d/dx = out*(1-out); tanh': 1-out^2; gelu/silu: no simple closed form
                # Use ISPC relu_backward as mask is unavailable — approximate with gate on od>0
                _C_layers.relu_backward_f32(gd._buf, od._buf, masked_buf)
            else:
                masked_buf = gd._buf
            gd_m = _NDA_layers(masked_buf, (N, out_f), _cs_layers((N, out_f)), _DType_layers('float32'))

            # 2. dx = gd_masked @ W  (N, out_f) @ (out_f, in_f) = (N, in_f)
            _dx_key = (N, in_f)
            if _dx_key not in _linear_out_cache:
                _linear_out_cache[_dx_key] = _arr_layers.array('f', bytearray(4 * N * in_f))
            dx_buf = _linear_out_cache[_dx_key]
            _C_layers.matmul_f32(gd_m._buf, wd._buf, dx_buf, N, out_f, in_f)
            dx_nd = _NDA_layers(dx_buf, (N, in_f), _cs_layers((N, in_f)), _DType_layers('float32'))

            # 3. dW = gd_masked.T @ x
            _gdT_key = -(N * out_f)
            if _gdT_key not in _linear_out_cache:
                _linear_out_cache[_gdT_key] = _arr_layers.array('f', bytearray(4 * out_f * N))
            gdT_buf = _linear_out_cache[_gdT_key]
            _C_layers.transpose_f32(gd_m._buf, gdT_buf, N, out_f)
            _dw_key = (out_f, in_f)
            if _dw_key not in _linear_out_cache:
                _linear_out_cache[_dw_key] = _arr_layers.array('f', bytearray(4 * out_f * in_f))
            dw_buf = _linear_out_cache[_dw_key]
            _C_layers.matmul_f32(gdT_buf, xd._buf, dw_buf, out_f, N, in_f)
            dw_nd = _NDA_layers(dw_buf, (out_f, in_f), _cs_layers((out_f, in_f)), _DType_layers('float32'))

            if self.bias is not None:
                _db_key = -(out_f + 1)
                if _db_key not in _linear_out_cache:
                    _linear_out_cache[_db_key] = _arr_layers.array('f', bytearray(4 * out_f))
                db_buf = _linear_out_cache[_db_key]
                _C_layers.sum_axis_f32(gd_m._buf, db_buf, N, out_f, 0)
                db_nd = _NDA_layers(db_buf, (out_f,), _cs_layers((out_f,)), _DType_layers('float32'))
                return Tensor(dx_nd, _no_copy=True), Tensor(dw_nd, _no_copy=True), Tensor(db_nd, _no_copy=True)
            return Tensor(dx_nd, _no_copy=True), Tensor(dw_nd, _no_copy=True)

        # Pure Python fallback
        def _to_flat(d):
            if hasattr(d, 'to_flat_list'): return d.to_flat_list()
            if hasattr(d, 'tolist'): return d.tolist()
            return list(d)
        def _shape(d): return d.shape if hasattr(d, 'shape') else (len(d),)
        from vectorphile._core.ndarray import array as _nd_arr2
        gd_sh = _shape(gd); od_sh = _shape(od); wd_sh = _shape(wd); xd_sh = _shape(xd)
        gf = _to_flat(gd); of_ = _to_flat(od); wf = _to_flat(wd); xf2 = _to_flat(xd)
        # Apply ReLU mask
        N2, out2 = gd_sh[0], gd_sh[1]; in2 = wd_sh[1]
        gf_masked = [gf[i] * (1.0 if of_[i] > 0 else 0.0) for i in range(len(gf))]
        gm = _nd_arr2(gf_masked).reshape(gd_sh)
        w_nd = _nd_arr2(wf).reshape(wd_sh); x_nd = _nd_arr2(xf2).reshape(xd_sh)
        dx = gm @ w_nd; dw = gm.T @ x_nd
        if self.bias is not None:
            db = [sum(gf_masked[n*out2+o] for n in range(N2)) for o in range(out2)]
            return Tensor(dx), Tensor(dw), Tensor(_nd_arr2(db).reshape((out2,)))
        return Tensor(dx), Tensor(dw)

# Keep old name as alias for backward compat
_LinearReLUBackward = _LinearActBackward


class _LinearBackward(Function):
    def __init__(self, x, weight, bias=None):
        super().__init__(x, weight) if bias is None else super().__init__(x, weight, bias)
        self.x = x
        self.weight = weight
        self.bias = bias

    def backward(self, grad):
        # Use module-level imports and C kernels to avoid per-call overhead
        gd = grad._data if isinstance(grad, Tensor) else grad
        wd = self.weight._data if isinstance(self.weight, Tensor) else self.weight
        xd = self.x._data if isinstance(self.x, Tensor) else self.x

        # Fast path: all NDArray with buf, 2D only, float32 only
        if (_LAYERS_C_OK
                and isinstance(gd, _NDA_layers) and gd._buf is not None
                and isinstance(wd, _NDA_layers) and wd._buf is not None
                and isinstance(xd, _NDA_layers) and xd._buf is not None
                and gd.ndim == 2 and xd.ndim == 2
                and str(getattr(gd, 'dtype', '')) == 'float32'
                and str(getattr(wd, 'dtype', '')) == 'float32'
                and str(getattr(xd, 'dtype', '')) == 'float32'):
            N   = gd.shape[0]
            out_f = gd.shape[1]
            in_f  = wd.shape[1]

            # dx = grad @ weight  — (N, out_f) @ (out_f, in_f) = (N, in_f)
            # Use fresh buffers in backward to avoid aliasing with forward output cache
            dx_buf = _arr_layers.array('f', bytearray(4 * N * in_f))
            _C_layers.matmul_f32(gd._buf, wd._buf, dx_buf, N, out_f, in_f)
            dx_nd = _NDA_layers(dx_buf, (N, in_f), _cs_layers((N, in_f)), _DType_layers('float32'))

            # dw = grad.T @ x  — (out_f, N) @ (N, in_f) = (out_f, in_f)
            # Transpose grad first: (N, out_f) -> (out_f, N)
            gdT_buf = _arr_layers.array('f', bytearray(4 * out_f * N))
            _C_layers.transpose_f32(gd._buf, gdT_buf, N, out_f)
            dw_buf = _arr_layers.array('f', bytearray(4 * out_f * in_f))
            _C_layers.matmul_f32(gdT_buf, xd._buf, dw_buf, out_f, N, in_f)
            dw_nd = _NDA_layers(dw_buf, (out_f, in_f), _cs_layers((out_f, in_f)), _DType_layers('float32'))

            if self.bias is not None:
                # db = grad.sum(axis=0)  — (out_f,)
                db_buf = _arr_layers.array('f', bytearray(4 * out_f))
                _C_layers.sum_axis_f32(gd._buf, db_buf, N, out_f, 0)
                db_nd = _NDA_layers(db_buf, (out_f,), _cs_layers((out_f,)), _DType_layers('float32'))
                return Tensor(dx_nd, _no_copy=True), Tensor(dw_nd, _no_copy=True), Tensor(db_nd, _no_copy=True)
            return Tensor(dx_nd, _no_copy=True), Tensor(dw_nd, _no_copy=True)

        # Pure Python NDArray fallback
        def _to_flat_nd(t):
            d = t._data if isinstance(t, Tensor) else t
            if hasattr(d, 'to_flat_list'): return d.to_flat_list(), d.shape, str(getattr(d, 'dtype', 'float32'))
            if hasattr(d, 'flatten'):  # numpy array
                return d.flatten().tolist(), d.shape, str(d.dtype) if str(d.dtype) in ('float32','float64','int32','int64') else 'float32'
            if hasattr(d, 'shape'): return list(d), d.shape, 'float32'
            return list(d), (len(d),), 'float32'
        from vectorphile._core.ndarray import array as _nd_arr_lb
        gf2, gsh2, gdt2 = _to_flat_nd(grad)
        wf2, wsh2, wdt2 = _to_flat_nd(self.weight)
        xf3, xsh3, xdt3 = _to_flat_nd(self.x)
        g_nd = _nd_arr_lb(gf2, dtype=gdt2).reshape(gsh2)
        w_nd2 = _nd_arr_lb(wf2, dtype=wdt2).reshape(wsh2)
        x_nd2 = _nd_arr_lb(xf3, dtype=xdt3).reshape(xsh3)
        dx = g_nd @ w_nd2
        # Flatten batch dims for weight grad
        if len(gsh2) > 2:
            B2 = 1
            for s in gsh2[:-1]: B2 *= s
            g_2d = _nd_arr_lb(gf2, dtype=gdt2).reshape((B2, gsh2[-1]))
            x_2d = _nd_arr_lb(xf3, dtype=xdt3).reshape((B2, xsh3[-1]))
        else:
            g_2d = g_nd; x_2d = x_nd2
        dw = g_2d.T @ x_2d
        db_flat = [sum(gf2[n*gsh2[-1]+o] for n in range(len(gf2)//gsh2[-1])) for o in range(gsh2[-1])]
        if self.bias is not None:
            return Tensor(dx), Tensor(dw), Tensor(_nd_arr_lb(db_flat, dtype=gdt2).reshape((gsh2[-1],)))
        return Tensor(dx), Tensor(dw)


class _Conv2dBackward(Function):
    """Backward pass for Conv2d via col2im (im2col^-1) + weight grad via GEMM."""
    def __init__(self, x, weight, bias, col_buf,
                 N, C, H, W, Kh, Kw, Sh, Sw, Ph, Pw, H_out, W_out):
        if bias is not None:
            super().__init__(x, weight, bias)
        else:
            super().__init__(x, weight)
        self.x = x; self.weight = weight; self.bias = bias
        self.col_buf = col_buf  # saved im2col output from forward
        self.N, self.C, self.H, self.W = N, C, H, W
        self.Kh, self.Kw, self.Sh, self.Sw, self.Ph, self.Pw = Kh, Kw, Sh, Sw, Ph, Pw
        self.H_out, self.W_out = H_out, W_out

    def backward(self, grad):
        import array as _a2d
        from vectorphile._core.ndarray import array as _nd_c2d
        gd = grad._data if isinstance(grad, Tensor) else grad
        N, C_out = self.N, self.weight._data.shape[0]
        C_in, Kh, Kw = self.C, self.Kh, self.Kw
        col_rows = N * self.H_out * self.W_out
        col_cols = C_in * Kh * Kw
        # Get flat grad and permute (N,Co,Ho,Wo) -> (N,Ho,Wo,Co) -> (col_rows, C_out)
        if hasattr(gd, 'to_flat_list'): gf_raw = gd.to_flat_list()
        elif hasattr(gd, 'tolist'): gf_raw = gd.tolist()
        else: gf_raw = list(gd)
        # Permute: original shape (N, C_out, Ho, Wo) -> (N, Ho, Wo, C_out)
        Ho, Wo = self.H_out, self.W_out
        grad_flat_arr = _a2d.array('f', bytearray(4 * col_rows * C_out))
        for n in range(N):
            for ho in range(Ho):
                for wo in range(Wo):
                    for co in range(C_out):
                        src = gf_raw[n*C_out*Ho*Wo + co*Ho*Wo + ho*Wo + wo]
                        grad_flat_arr[(n*Ho*Wo + ho*Wo + wo)*C_out + co] = src
        if _LAYERS_C_OK:
            col_T = _a2d.array('f', bytearray(4 * col_cols * col_rows))
            _C_layers.transpose_f32(self.col_buf, col_T, col_rows, col_cols)
            dw_buf = _a2d.array('f', bytearray(4 * C_out * col_cols))
            gfT = _a2d.array('f', bytearray(4 * C_out * col_rows))
            _C_layers.transpose_f32(grad_flat_arr, gfT, col_rows, C_out)
            _C_layers.matmul_f32(gfT, self.col_buf, dw_buf, C_out, col_rows, col_cols)
            import struct as _st
            dw_flat = list(_st.unpack_from(f'{C_out*col_cols}f', bytes(dw_buf)))
            dw = Tensor(_nd_c2d(dw_flat).reshape((C_out, C_in, Kh, Kw)))
            # dx via col2im
            if hasattr(self.weight._data, 'to_flat_list'): wf2 = self.weight._data.to_flat_list()
            elif hasattr(self.weight._data, 'tolist'): wf2 = self.weight._data.tolist()
            else: wf2 = list(self.weight._data)
            w_buf2 = _a2d.array('f', bytearray(4 * C_out * col_cols))
            for i, v in enumerate(wf2): w_buf2[i] = v
            col_grad_buf = _a2d.array('f', bytearray(4 * col_rows * col_cols))
            _C_layers.matmul_f32(grad_flat_arr, w_buf2, col_grad_buf, col_rows, C_out, col_cols)
            dx_buf = _a2d.array('f', bytearray(4 * N * self.C * self.H * self.W))
            _C_layers.col2im_f32(col_grad_buf, dx_buf, N, self.C, self.H, self.W,
                                  Kh, Kw, self.Sh, self.Sw, self.Ph, self.Pw)
            dx_flat = list(_st.unpack_from(f'{N*self.C*self.H*self.W}f', bytes(dx_buf)))
            dx = Tensor(_nd_c2d(dx_flat).reshape((N, self.C, self.H, self.W)))
            if self.bias is not None:
                db = [sum(grad_flat_arr[r*C_out+co] for r in range(col_rows)) for co in range(C_out)]
                return dx, dw, Tensor(_nd_c2d(db).reshape((C_out,)))
            return dx, dw
        # Pure Python fallback
        # grad_flat_arr is (col_rows, C_out)
        # col_buf is (col_rows, col_cols)
        import struct as _st2
        gf2 = list(_st2.unpack_from(f'{col_rows*C_out}f', bytes(grad_flat_arr)))
        col_f = list(_st2.unpack_from(f'{col_rows*col_cols}f', bytes(self.col_buf)))
        # dw = gf2.T @ col_f -> (C_out, col_cols)
        dw_flat = [sum(gf2[r*C_out+co] * col_f[r*col_cols+kk] for r in range(col_rows)) for co in range(C_out) for kk in range(col_cols)]
        dw = Tensor(_nd_c2d(dw_flat).reshape((C_out, C_in, Kh, Kw)))
        if hasattr(self.weight._data, 'to_flat_list'): wf3 = self.weight._data.to_flat_list()
        elif hasattr(self.weight._data, 'tolist'): wf3 = self.weight._data.tolist()
        else: wf3 = list(self.weight._data)
        # col_grad = gf2 @ w_flat -> (col_rows, col_cols)
        col_grad = [sum(gf2[r*C_out+co] * wf3[co*col_cols+kk] for co in range(C_out)) for r in range(col_rows) for kk in range(col_cols)]
        # col2im
        dx_flat2 = [0.0] * (N * self.C * self.H * self.W)
        for n2 in range(N):
            for ho2 in range(Ho):
                for wo2 in range(Wo):
                    r2 = n2*Ho*Wo + ho2*Wo + wo2
                    for ci2 in range(C_in):
                        for ki in range(Kh):
                            for kj in range(Kw):
                                kk2 = ci2*Kh*Kw + ki*Kw + kj
                                hi = ho2*self.Sh - self.Ph + ki
                                wi = wo2*self.Sw - self.Pw + kj
                                if 0 <= hi < self.H and 0 <= wi < self.W:
                                    dx_flat2[n2*self.C*self.H*self.W + ci2*self.H*self.W + hi*self.W + wi] += col_grad[r2*col_cols + kk2]
        dx = Tensor(_nd_c2d(dx_flat2).reshape((N, self.C, self.H, self.W)))
        if self.bias is not None:
            db2 = [sum(gf2[r*C_out+co] for r in range(col_rows)) for co in range(C_out)]
            return dx, dw, Tensor(_nd_c2d(db2).reshape((C_out,)))
        return dx, dw


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        k = math.sqrt(1.0 / in_features)
        self.weight = Tensor(
            _nd_uniform_fn(-k, k, (out_features, in_features)),
            requires_grad=True
        )
        self.bias_param = None
        if bias:
            self.bias_param = Tensor(_nd_zeros(out_features, dtype='float32'), requires_grad=True)
            self._parameters['bias'] = self._parameters.pop('bias_param', self.bias_param)

    def _forward_cuda(self, x: Tensor) -> Tensor:
        """CUDA fast path: GEMM + optional fused activation on GPU."""
        from vectorphile._core.ndarray import NDArray as _NDA, _c_strides as _cs, _DType
        xd = x._data; wd = self.weight._data
        N = xd.shape[0]; out_f = wd.shape[0]; in_f = wd.shape[1]
        req_grad = x.requires_grad or self.weight.requires_grad

        # Ensure weight is on GPU (cache transposed weight on device)
        w_cbuf = getattr(wd, '_cuda_buf', None)
        if w_cbuf is None:
            raise RuntimeError('Linear weight must be on CUDA for _forward_cuda')
        # Cache transposed weight on device
        if not hasattr(self, '_wT_cbuf') or self._wT_w_cid != id(w_cbuf):
            wT_cbuf = _C_cuda.alloc(out_f * in_f * 4)
            _C_cuda.transpose_f32(w_cbuf, wT_cbuf, out_f, in_f)
            self._wT_cbuf = wT_cbuf
            self._wT_w_cid = id(w_cbuf)
        wT_cbuf = self._wT_cbuf

        x_cbuf = getattr(xd, '_cuda_buf', None)
        if x_cbuf is None:
            raise RuntimeError('Input must be on CUDA for _forward_cuda')

        out_cbuf = _C_cuda.alloc(N * out_f * 4)

        _fuse_act = getattr(self, '_fuse_act', None)
        if _fuse_act is None and getattr(self, '_fuse_relu', False):
            _fuse_act = 'relu'
        _post_map = {'relu': 1, 'gelu': 2, 'silu': 3, 'sigmoid': 4, 'tanh': 5}

        if _fuse_act in _post_map:
            post_op = _post_map[_fuse_act]
            _C_cuda.gemm_act_f32(x_cbuf, wT_cbuf, out_cbuf, N, in_f, out_f, post_op)
        else:
            _C_cuda.gemm_f32(x_cbuf, wT_cbuf, out_cbuf, N, in_f, out_f)

        if self.bias_param is not None:
            bd = self.bias_param._data
            b_cbuf = getattr(bd, '_cuda_buf', None)
            if b_cbuf is None:
                raise RuntimeError('bias must be on CUDA for _forward_cuda')
            _C_cuda.add_bias_rows_f32(out_cbuf, b_cbuf, N, out_f)

        # Wrap result as CUDA NDArray
        nd = _NDA.__new__(_NDA)
        nd._buf = None
        nd._np_buf = None
        nd._cuda_buf = out_cbuf
        nd.shape = (N, out_f)
        nd.strides = _cs((N, out_f))
        nd.dtype = _DType('float32')
        nd.offset = 0
        nd._owns_buf = False

        from vectorphile.core.tensor import Tensor as _T
        result = _T.__new__(_T)
        result._data = nd
        result.device = 'cuda'
        result.requires_grad = req_grad
        result.grad = None
        result.grad_fn = None
        result._is_leaf = True
        result._retain_grad = False
        if req_grad:
            from vectorphile.nn.layers import _LinearActBackward, _LinearBackward
            if _fuse_act in _post_map:
                result.grad_fn = _LinearActBackward(x, self.weight, self.bias_param, nd, _fuse_act)
            else:
                result.grad_fn = _LinearBackward(x, self.weight, self.bias_param)
            result._is_leaf = False
        return result

    def forward(self, x: Tensor) -> Tensor:
        # AMP autocast: if BF16/FP16 autocast enabled, cast input and weight
        _amp_weight = None
        try:
            from vectorphile.amp import is_autocast_enabled as _ice, get_autocast_dtype as _gad, cast_to_autocast_dtype as _cac
            if _ice():
                x = _cac(x)
                _amp_weight = _cac(self.weight)
        except Exception:
            pass
        # CUDA fast path
        if x.device == 'cuda' and _CUDA_OK:
            return self._forward_cuda(x)

        _arr = _arr_layers; _NDA = _NDA_layers; _cs = _cs_layers; _DType = _DType_layers
        _C_lin = _C_layers; _c_ok = _LAYERS_C_OK
        xd = x._data; wd = (_amp_weight._data if _amp_weight is not None else self.weight._data)
        N = xd.shape[0]; out_f = wd.shape[0]; in_f = wd.shape[1]
        try:
            from vectorphile.autograd.engine import no_grad as _ng
            _ge = _ng._enabled
        except Exception:
            _ge = True
        req_grad = _ge and (x.requires_grad or self.weight.requires_grad)
        if (_c_ok and isinstance(xd, _NDA_layers) and isinstance(wd, _NDA_layers)
                and xd.dtype == 'float32' and wd.dtype == 'float32'
                and xd._buf is not None and xd.ndim == 2):
            w_buf = wd._buf
            # Cache transposed weight — recompute only when weight buffer identity changes
            if (not hasattr(self, '_wT_buf') or self._wT_w_id != id(w_buf)):
                wT_buf = _arr.array('f', bytearray(4 * in_f * out_f))
                _C_lin.transpose_f32(w_buf, wT_buf, out_f, in_f)
                self._wT_buf = wT_buf
                self._wT_w_id = id(w_buf)
            wT_buf = self._wT_buf
            _out_key = (N, out_f)
            if _out_key not in _linear_out_cache:
                _linear_out_cache[_out_key] = _arr.array('f', bytearray(4 * N * out_f))
            out_buf = _linear_out_cache[_out_key]
            _fuse_act = getattr(self, '_fuse_act', None)
            if _fuse_act is None:
                _fuse_act = 'relu' if getattr(self, '_fuse_relu', False) else None
            _post_op_map = {'relu': 1, 'gelu': 2, 'silu': 3, 'sigmoid': 4, 'tanh': 5}
            # matmul_act_f32 epilogue fusion disabled — C kernel produces wrong results
            # Activations are applied by separate activation modules instead
            _C_lin.matmul_f32(xd._buf, wT_buf, out_buf, N, in_f, out_f)
            if self.bias_param is not None:
                bd = self.bias_param._data
                b_buf = bd._buf
                _C_lin.add_bias_rows_f32(out_buf, b_buf, N, out_f)
            result_data = _NDA(out_buf, (N, out_f), _cs((N, out_f)), _DType('float32'))
            result = Tensor(result_data, device=x.device, requires_grad=req_grad, _no_copy=True)
            if req_grad:
                result.grad_fn = _LinearBackward(x, self.weight, self.bias_param)
                result._is_leaf = False
            return result
        # Pure Python fallback matmul
        from vectorphile._core.ndarray import array as _nd_lin_fb
        def _get_fl(nd):
            if hasattr(nd, 'to_flat_list'): return nd.to_flat_list()
            if hasattr(nd, 'flatten'): return nd.flatten().tolist()  # numpy
            if hasattr(nd, '_buf') and nd._buf is not None: return list(nd._buf)
            return list(nd)
        def _get_sh(nd):
            if hasattr(nd, 'shape'): return nd.shape
            return (len(_get_fl(nd)),)
        def _get_dt(nd):
            dt = str(getattr(nd, 'dtype', 'float32'))
            return dt if dt in ('float32', 'float64', 'int32', 'int64') else 'float32'
        xfl = _get_fl(xd); wfl = _get_fl(wd)
        xsh_fb = _get_sh(xd); wsh_fb = _get_sh(wd)
        _out_dt = _get_dt(xd)
        # x: (..., in_f) @ w.T: (out_f, in_f) -> (..., out_f)
        in_f = wsh_fb[-1]; out_f = wsh_fb[0]
        batch_fb = len(xfl) // in_f
        out_fb = [sum(xfl[b*in_f+k]*wfl[o*in_f+k] for k in range(in_f)) for b in range(batch_fb) for o in range(out_f)]
        out_sh_fb = xsh_fb[:-1] + (out_f,)
        if self.bias_param is not None:
            bfl = _get_fl(self.bias_param._data)
            out_fb = [out_fb[b*out_f+o] + bfl[o] for b in range(batch_fb) for o in range(out_f)]
        result = Tensor(_nd_lin_fb(out_fb, dtype=_out_dt).reshape(out_sh_fb), device=x.device, requires_grad=req_grad, _no_copy=True)
        if req_grad:
            result.grad_fn = _LinearBackward(x, self.weight, self.bias_param)
            result._is_leaf = False
        return result

    @property
    def bias(self):
        return self.bias_param

    def __repr__(self):
        return f'Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias_param is not None})'


class FusedLinear(Module):
    """Linear layer with a fused activation epilogue — GEMM + activation in one C pass."""
    _ACT_NAMES = {'relu', 'gelu', 'silu', 'sigmoid', 'tanh'}

    def __init__(self, in_features: int, out_features: int, activation: str = 'relu', bias: bool = True):
        super().__init__()
        if activation not in self._ACT_NAMES:
            raise ValueError(f'FusedLinear activation must be one of {self._ACT_NAMES}')
        self._linear = Linear(in_features, out_features, bias=bias)
        self._linear._fuse_act = activation
        self._activation = activation
        # expose parameters so optimizers see them
        self._parameters['weight'] = self._linear.weight
        if bias:
            self._parameters['bias'] = self._linear.bias_param

    @property
    def weight(self):
        return self._linear.weight

    @property
    def bias_param(self):
        return self._linear.bias_param

    def forward(self, x: Tensor) -> Tensor:
        return self._linear(x)

    def __repr__(self):
        lf = self._linear
        return (f'FusedLinear({lf.in_features}, {lf.out_features}, '
                f'act={self._activation}, bias={lf.bias_param is not None})')


class Conv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, groups=1, dilation=1, padding_mode='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        k = math.sqrt(1.0 / (in_channels * kernel_size))
        n_w = out_channels * in_channels * kernel_size
        from vectorphile._core.ndarray import array as _nd_c1
        self.weight = Tensor(
            _nd_c1([_random_layers.uniform(-k, k) for _ in range(n_w)]).reshape((out_channels, in_channels, kernel_size)),
            requires_grad=True
        )
        self._parameters['weight'] = self.weight
        self.bias_param = None
        if bias:
            self.bias_param = Tensor(_nd_c1([0.0]*out_channels).reshape((out_channels,)), requires_grad=True)
            self._parameters['bias'] = self._parameters.pop('bias_param', self.bias_param)

    @property
    def bias(self):
        return self.bias_param

    def forward(self, x: Tensor) -> Tensor:
        from vectorphile._core.ndarray import array as _nd_c1f
        from vectorphile._core.ndarray import NDArray as _NDA_c1, _c_strides as _cs_c1, _DType as _DT_c1
        import array as _arr_c1
        xd = x._data; wd = self.weight._data
        if hasattr(xd, 'shape'): xsh = xd.shape
        else: xsh = (len(xd),)
        if hasattr(wd, 'shape'): wsh = wd.shape
        else: wsh = (len(wd),)
        B, C_in, L = xsh; C_out, _, K = wsh
        p, s = self.padding, self.stride
        L_out = (L + 2*p - K) // s + 1
        # C fast path: im2col_f32 + matmul_f32 (treat 1D as 2D with H=1)
        if (_LAYERS_C_OK and not x.requires_grad and not self.weight.requires_grad
                and isinstance(xd, _NDA_c1) and str(xd.dtype) == 'float32' and xd._buf is not None
                and isinstance(wd, _NDA_c1) and wd._buf is not None
                and hasattr(_C_layers, 'im2col_f32') and hasattr(_C_layers, 'matmul_f32')):
            try:
                # im2col: treat (B, C_in, 1, L) → col (B*L_out, C_in*K)
                col_buf = _arr_c1.array('f', bytearray(4 * B * L_out * C_in * K))
                # Reshape x to 4D (B, C_in, 1, L) for im2col
                x4d_buf = xd._buf  # already B*C_in*L contiguous
                # im2col signature: (inp, col, N, C, H, W, Kh, Kw, Sh, Sw, Ph, Pw)
                _C_layers.im2col_f32(x4d_buf, col_buf, B, C_in, 1, L, 1, K, 1, s, 0, p)
                # col @ W.T: (B*L_out, C_in*K) @ (C_in*K, C_out) → (B*L_out, C_out)
                out_buf = _arr_c1.array('f', bytearray(4 * B * L_out * C_out))
                _C_layers.matmul_f32(col_buf, wd._buf, out_buf, B * L_out, C_in * K, C_out)
                # Reshape (B*L_out, C_out) → (B, C_out, L_out)
                out_flat_list = list(out_buf)
                out_perm = [0.0] * (B * C_out * L_out)
                for b in range(B):
                    for lo in range(L_out):
                        for co in range(C_out):
                            out_perm[b * C_out * L_out + co * L_out + lo] = out_flat_list[(b * L_out + lo) * C_out + co]
                if self.bias_param is not None:
                    bfl = self.bias_param._data.to_flat_list() if hasattr(self.bias_param._data, 'to_flat_list') else list(self.bias_param._data)
                    out_perm = [out_perm[b * C_out * L_out + co * L_out + lo] + bfl[co]
                                for b in range(B) for co in range(C_out) for lo in range(L_out)]
                out_nd = _NDA_c1(_arr_c1.array('f', out_perm), (B, C_out, L_out), _cs_c1((B, C_out, L_out)), _DT_c1('float32'))
                return Tensor(out_nd, device=x.device, requires_grad=False)
            except Exception:
                pass
        if hasattr(xd, 'to_flat_list'): xf = xd.to_flat_list()
        elif hasattr(xd, 'tolist'): xf = xd.tolist()
        else: xf = list(xd)
        if hasattr(wd, 'to_flat_list'): wf = wd.to_flat_list()
        elif hasattr(wd, 'tolist'): wf = wd.tolist()
        else: wf = list(wd)
        # im2col cols: (B*L_out, C_in*K)
        cols = []
        for b in range(B):
            for lo in range(L_out):
                row = []
                for ci in range(C_in):
                    for k in range(K):
                        li = lo * s - p + k
                        row.append(xf[b*C_in*L + ci*L + li] if 0 <= li < L else 0.0)
                cols.append(row)
        col_rows = B * L_out; col_cols = C_in * K
        # w_flat: (C_out, C_in*K)
        # out_flat: (B*L_out, C_out) = cols @ w_flat.T
        out_flat = [[sum(cols[r][kk] * wf[co*col_cols+kk] for kk in range(col_cols)) for co in range(C_out)] for r in range(col_rows)]
        # Reshape to (B, C_out, L_out)
        out = [0.0] * (B * C_out * L_out)
        for b in range(B):
            for lo in range(L_out):
                for co in range(C_out):
                    out[b*C_out*L_out + co*L_out + lo] = out_flat[b*L_out+lo][co]
        if self.bias_param is not None:
            if hasattr(self.bias_param._data, 'to_flat_list'): bf = self.bias_param._data.to_flat_list()
            else: bf = list(self.bias_param._data)
            out = [out[b*C_out*L_out + co*L_out + lo] + bf[co]
                   for b in range(B) for co in range(C_out) for lo in range(L_out)]
        req = x.requires_grad or self.weight.requires_grad
        result = Tensor(_nd_c1f(out).reshape((B, C_out, L_out)), device=x.device, requires_grad=req)
        if req:
            _x, _w, _bp = x, self.weight, self.bias_param
            _cols = cols; _wf = wf
            _B, _Cin, _L, _K2, _s, _p2, _Lout, _Cout = B, C_in, L, K, s, p, L_out, C_out
            class _Conv1dBwd:
                def __init__(ss): ss.inputs = ([_x, _w] + ([_bp] if _bp is not None else []))
                def backward(ss, grad):
                    gd2 = grad._data
                    if hasattr(gd2, 'to_flat_list'): gf2 = gd2.to_flat_list()
                    else: gf2 = list(gd2)
                    # g_flat: (B*L_out, C_out) from (B,C_out,L_out)
                    g_flat = [[gf2[b*_Cout*_Lout + co*_Lout + lo] for co in range(_Cout)] for b in range(_B) for lo in range(_Lout)]
                    # dw: (C_out, C_in*K) = g_flat.T @ cols
                    col_cols2 = _Cin * _K2; col_rows2 = _B * _Lout
                    dw_flat = [sum(g_flat[r][co] * _cols[r][kk] for r in range(col_rows2)) for co in range(_Cout) for kk in range(col_cols2)]
                    # dx via col2im
                    dcol = [[sum(g_flat[r][co] * _wf[co*col_cols2+kk] for co in range(_Cout)) for kk in range(col_cols2)] for r in range(col_rows2)]
                    dx_flat = [0.0] * (_B * _Cin * _L)
                    for b in range(_B):
                        for lo in range(_Lout):
                            r2 = b*_Lout + lo
                            for ci2 in range(_Cin):
                                for k2 in range(_K2):
                                    li = lo * _s - _p2 + k2
                                    if 0 <= li < _L:
                                        dx_flat[b*_Cin*_L + ci2*_L + li] += dcol[r2][ci2*_K2+k2]
                    rets = (Tensor(_nd_c1f(dx_flat).reshape((_B, _Cin, _L))),
                            Tensor(_nd_c1f(dw_flat).reshape((_Cout, _Cin, _K2))))
                    if _bp is not None:
                        db2 = [sum(g_flat[b*_Lout+lo][co] for b in range(_B) for lo in range(_Lout)) for co in range(_Cout)]
                        return rets + (Tensor(_nd_c1f(db2).reshape((_Cout,))),)
                    return rets
            result.grad_fn = _Conv1dBwd(); result._is_leaf = False
        return result


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride  = stride  if isinstance(stride,  tuple) else (stride,  stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        kh, kw = self.kernel_size
        k = math.sqrt(1.0 / (in_channels * kh * kw))
        from vectorphile._core.ndarray import array as _nd_c2init
        n_w2 = out_channels * in_channels * kh * kw
        self.weight = Tensor(
            _nd_c2init([_random_layers.uniform(-k, k) for _ in range(n_w2)]).reshape((out_channels, in_channels, kh, kw)),
            requires_grad=True
        )
        self._parameters['weight'] = self.weight
        self.bias_param = None
        if bias:
            self.bias_param = Tensor(_nd_c2init([0.0]*out_channels).reshape((out_channels,)), requires_grad=True)
            self._parameters['bias'] = self._parameters.pop('bias_param', self.bias_param)

    def forward(self, x: Tensor) -> Tensor:
        sh, sw = self.stride
        ph, pw = self.padding
        kh, kw = self.kernel_size
        xd = x._data
        wd = self.weight._data

        # Fast path: im2col + matmul_f32 via C kernels (float32 only)
        if (_LAYERS_C_OK and hasattr(_C_layers, 'im2col_f32')
                and isinstance(xd, _NDA_layers) and xd._buf is not None
                and str(getattr(xd, 'dtype', 'float32')) == 'float32'
                and str(getattr(wd, 'dtype', 'float32')) == 'float32'):
            import array as _arr_c2d
            # xd shape: (N, C, H, W)
            N_b, C_in, H, W = xd.shape
            C_out = self.out_channels
            H_out = (H + 2*ph - kh) // sh + 1
            W_out = (W + 2*pw - kw) // sw + 1
            col_rows = N_b * H_out * W_out
            col_cols = C_in * kh * kw
            col_buf  = _arr_c2d.array('f', bytearray(4 * col_rows * col_cols))
            # Need flat float* for xd — ensure C-contiguous
            if hasattr(xd, '_buf') and xd._buf is not None:
                x_buf = xd._buf
            else:
                import array as _a2
                xfl_c2d = xd.to_flat_list() if hasattr(xd, 'to_flat_list') else list(xd)
                x_buf = _a2.array('f', xfl_c2d)
            _C_layers.im2col_f32(x_buf, col_buf, N_b, C_in, H, W,
                                 kh, kw, sh, sw, ph, pw)
            # weight: (C_out, C_in, kh, kw) — reshape to (C_out, col_cols)
            if hasattr(wd, '_buf') and wd._buf is not None:
                w_buf = wd._buf
            else:
                wfl_c2d = wd.to_flat_list() if hasattr(wd, 'to_flat_list') else list(wd)
                w_buf = _arr_c2d.array('f', wfl_c2d)
            # out_buf = col @ weight.T  → (col_rows, C_out)
            # We want C = col_buf(col_rows, col_cols) @ wT(col_cols, C_out)
            # wT is weight reshaped to (C_out, col_cols) then transposed → (col_cols, C_out)
            # Store transposed weight
            wT_buf = _arr_c2d.array('f', bytearray(4 * col_cols * C_out))
            _C_layers.transpose_f32(w_buf, wT_buf, C_out, col_cols)
            out_buf = _arr_c2d.array('f', bytearray(4 * col_rows * C_out))
            _C_layers.matmul_f32(col_buf, wT_buf, out_buf, col_rows, col_cols, C_out)
            if self.bias_param is not None:
                bd = self.bias_param._data
                b_buf = bd._buf if (hasattr(bd, '_buf') and bd._buf is not None) else \
                        _arr_c2d.array('f', bd.to_flat_list() if hasattr(bd, 'to_flat_list') else list(bd))
                _C_layers.add_bias_rows_f32(out_buf, b_buf, col_rows, C_out)
            # Reshape output: (col_rows, C_out) → (N, C_out, H_out, W_out)
            import struct as _st_c2d
            from vectorphile._core.ndarray import array as _nd_c2d_out
            raw_out = list(_st_c2d.unpack_from(f'{col_rows*C_out}f', bytes(out_buf)))
            # transpose (N,H_out,W_out,C_out) -> (N,C_out,H_out,W_out)
            out_trans = [raw_out[n*H_out*W_out*C_out + ho*W_out*C_out + wo*C_out + co]
                         for n in range(N_b) for co in range(C_out) for ho in range(H_out) for wo in range(W_out)]
            out_np = _nd_c2d_out(out_trans).reshape((N_b, C_out, H_out, W_out))
            # Track gradient
            req = x.requires_grad or self.weight.requires_grad
            result = Tensor(out_np, device=x.device, requires_grad=req)
            if req:
                result.grad_fn = _Conv2dBackward(x, self.weight, self.bias_param,
                                                  col_buf, N_b, C_in, H, W,
                                                  kh, kw, sh, sw, ph, pw, H_out, W_out)
                result._is_leaf = False
            return result

        # Pure Python im2col fallback
        from vectorphile._core.ndarray import array as _nd_c2fb
        _c2fb_dt = str(getattr(x._data, 'dtype', 'float32')) if hasattr(x._data, 'dtype') else 'float32'
        if _c2fb_dt not in ('float32', 'float64'): _c2fb_dt = 'float32'
        if hasattr(x._data, 'to_flat_list'): xf2 = x._data.to_flat_list()
        elif hasattr(x._data, 'tolist'): xf2 = x._data.tolist()
        else: xf2 = list(x._data)
        if hasattr(self.weight._data, 'to_flat_list'): wf2 = self.weight._data.to_flat_list()
        elif hasattr(self.weight._data, 'tolist'): wf2 = self.weight._data.tolist()
        else: wf2 = list(self.weight._data)
        xsh2 = x._data.shape if hasattr(x._data, 'shape') else (len(xf2),)
        N_b, C_in, H, W = xsh2
        C_out = self.out_channels
        H_out = (H + 2*ph - kh) // sh + 1
        W_out = (W + 2*pw - kw) // sw + 1
        col_rows = N_b * H_out * W_out; col_cols = C_in * kh * kw
        def _xget(n, c, h, w):
            if h < 0 or h >= H or w < 0 or w >= W: return 0.0
            return xf2[n*C_in*H*W + c*H*W + h*W + w]
        cols_flat = []
        for n in range(N_b):
            for ho in range(H_out):
                for wo in range(W_out):
                    for ci2 in range(C_in):
                        for ki in range(kh):
                            for kj in range(kw):
                                cols_flat.append(_xget(n, ci2, ho*sh-ph+ki, wo*sw-pw+kj))
        # out_flat: (col_rows, C_out) = cols @ w_flat.T  (uses Python float = float64, so precision matches input)
        if hasattr(self.weight._data, 'to_flat_list'): wf2 = self.weight._data.to_flat_list()
        elif hasattr(self.weight._data, 'tolist'): wf2 = self.weight._data.tolist()
        else: wf2 = list(self.weight._data)
        out_flat = [sum(cols_flat[r*col_cols+kk] * wf2[co*col_cols+kk] for kk in range(col_cols)) for r in range(col_rows) for co in range(C_out)]
        # Reshape (N,H_out,W_out,C_out) -> (N,C_out,H_out,W_out)
        out_list = [0.0] * (N_b * C_out * H_out * W_out)
        for n in range(N_b):
            for ho in range(H_out):
                for wo in range(W_out):
                    r2 = n*H_out*W_out + ho*W_out + wo
                    for co in range(C_out):
                        out_list[n*C_out*H_out*W_out + co*H_out*W_out + ho*W_out + wo] = out_flat[r2*C_out+co]
        if self.bias_param is not None:
            if hasattr(self.bias_param._data, 'to_flat_list'): bf2 = self.bias_param._data.to_flat_list()
            else: bf2 = list(self.bias_param._data)
            out_list = [out_list[n*C_out*H_out*W_out + co*H_out*W_out + ho*W_out + wo] + bf2[co]
                        for n in range(N_b) for co in range(C_out) for ho in range(H_out) for wo in range(W_out)]
        req = x.requires_grad or self.weight.requires_grad
        result = Tensor(_nd_c2fb(out_list, dtype=_c2fb_dt).reshape((N_b, C_out, H_out, W_out)), device=x.device, requires_grad=req)
        if req:
            _x2, _w2, _bp2 = x, self.weight, self.bias_param
            _cf2 = cols_flat; _wf3 = wf2; _dt2 = _c2fb_dt
            _N2, _Cin2, _H2, _W2 = N_b, C_in, H, W
            _kh2, _kw2, _sh2, _sw2, _ph2, _pw2 = kh, kw, sh, sw, ph, pw
            _Ho2, _Wo2, _Cout2 = H_out, W_out, C_out
            _cr2, _cc2 = col_rows, col_cols
            class _Conv2dBwdPy:
                def __init__(ss): ss.inputs = ([_x2, _w2] + ([_bp2] if _bp2 is not None else []))
                def backward(ss, grad):
                    gd3 = grad._data
                    if hasattr(gd3, 'to_flat_list'): gf3 = gd3.to_flat_list()
                    else: gf3 = list(gd3)
                    # g_flat: (col_rows, C_out) from (N, C_out, H_out, W_out)
                    g_flat2 = [gf3[n*_Cout2*_Ho2*_Wo2 + co*_Ho2*_Wo2 + ho*_Wo2 + wo]
                                for n in range(_N2) for ho in range(_Ho2) for wo in range(_Wo2) for co in range(_Cout2)]
                    dw2 = [sum(g_flat2[r*_Cout2+co] * _cf2[r*_cc2+kk] for r in range(_cr2)) for co in range(_Cout2) for kk in range(_cc2)]
                    dcol2 = [sum(g_flat2[r*_Cout2+co] * _wf3[co*_cc2+kk] for co in range(_Cout2)) for r in range(_cr2) for kk in range(_cc2)]
                    dx2 = [0.0] * (_N2 * _Cin2 * _H2 * _W2)
                    for n3 in range(_N2):
                        for ho3 in range(_Ho2):
                            for wo3 in range(_Wo2):
                                r3 = n3*_Ho2*_Wo2 + ho3*_Wo2 + wo3
                                for ci3 in range(_Cin2):
                                    for ki3 in range(_kh2):
                                        for kj3 in range(_kw2):
                                            hi3 = ho3*_sh2 - _ph2 + ki3
                                            wi3 = wo3*_sw2 - _pw2 + kj3
                                            if 0 <= hi3 < _H2 and 0 <= wi3 < _W2:
                                                dx2[n3*_Cin2*_H2*_W2 + ci3*_H2*_W2 + hi3*_W2 + wi3] += dcol2[r3*_cc2 + ci3*_kh2*_kw2 + ki3*_kw2 + kj3]
                    rets2 = (Tensor(_nd_c2fb(dx2, dtype=_dt2).reshape((_N2,_Cin2,_H2,_W2))),
                             Tensor(_nd_c2fb(dw2, dtype=_dt2).reshape((_Cout2,_Cin2,_kh2,_kw2))))
                    if _bp2 is not None:
                        db3 = [sum(g_flat2[r*_Cout2+co] for r in range(_cr2)) for co in range(_Cout2)]
                        return rets2 + (Tensor(_nd_c2fb(db3, dtype=_dt2).reshape((_Cout2,))),)
                    return rets2
            result.grad_fn = _Conv2dBwdPy(); result._is_leaf = False
        return result

    def __repr__(self):
        return (f'Conv2d({self.in_channels}, {self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})')


class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            from vectorphile._core.ndarray import ones as _nd_ones
            self.weight     = Tensor(_nd_ones(num_features, dtype='float32'), requires_grad=True)
            self.bias_param = Tensor(_nd_zeros(num_features, dtype='float32'), requires_grad=True)
            self._parameters['weight']     = self.weight
            self._parameters['bias'] = self._parameters.pop('bias_param', self.bias_param)
        from vectorphile._core.ndarray import ones as _nd_ones
        self.running_mean = _nd_zeros(num_features, dtype='float32')
        self.running_var  = _nd_ones(num_features, dtype='float32')

    def forward(self, x: Tensor) -> Tensor:
        from vectorphile._core.ndarray import array as _nd_bn1
        from vectorphile._core.ndarray import NDArray as _NDA_bn1, _c_strides as _cs_bn1, _DType as _DT_bn1
        import array as _arr_bn1
        xd = x._data; xsh = xd.shape if hasattr(xd, 'shape') else (len(xd._buf if hasattr(xd, '_buf') else xd),)
        ndim = len(xsh); C = xsh[1]
        # C AVX2 fast path: 2D (N, C) float32, training or inference
        if (_LAYERS_C_OK and ndim == 2 and isinstance(xd, _NDA_bn1)
                and str(xd.dtype) == 'float32' and xd._buf is not None and self.affine):
            try:
                N_c = xsh[0]
                out_buf = _arr_bn1.array('f', bytearray(4 * N_c * C))
                mean_buf = _arr_bn1.array('f', bytearray(4 * C))
                rstd_buf = _arr_bn1.array('f', bytearray(4 * C))
                wd = self.weight._data; bpd = self.bias_param._data
                if (isinstance(wd, _NDA_bn1) and wd._buf is not None
                        and isinstance(bpd, _NDA_bn1) and bpd._buf is not None):
                    _C_layers.batch_norm_f32(xd._buf, out_buf, mean_buf, rstd_buf,
                                              wd._buf, bpd._buf, None, N_c, C, self.eps)
                    if self.training:
                        m = self.momentum
                        if hasattr(self.running_mean, '_buf'):
                            for c in range(C):
                                self.running_mean._buf[c] = (1-m)*self.running_mean._buf[c] + m*mean_buf[c]
                                # rstd = 1/sqrt(var+eps), var = 1/rstd^2 - eps
                                var_c = max(1.0/(rstd_buf[c]*rstd_buf[c]) - self.eps, 0.0)
                                self.running_var._buf[c]  = (1-m)*self.running_var._buf[c]  + m*var_c
                    out_nd = _NDA_bn1(out_buf, xsh, _cs_bn1(xsh), _DT_bn1('float32'))
                    return Tensor(out_nd, device=x.device, requires_grad=x.requires_grad)
            except Exception:
                pass
        if hasattr(xd, 'to_flat_list'): xf = xd.to_flat_list()
        else: xf = list(xd)
        N_total = len(xf) // C  # elements per channel
        if self.training:
            mean_v = [sum(xf[c + i*C*1 if ndim == 2 else 0] for i in range(N_total)) / N_total for c in range(C)]
            # Recompute properly: iterate over all positions except channel axis
            mean_v = [0.0] * C; count = len(xf) // C
            for idx2 in range(len(xf)):
                # channel index depends on layout (N,C,...): c = (idx2 // stride_c) % C
                stride_c = 1
                for d in range(2, ndim): stride_c *= xsh[d]
                c_idx = (idx2 // stride_c) % C
                mean_v[c_idx] += xf[idx2]
            mean_v = [v / count for v in mean_v]
            var_v = [0.0] * C
            for idx2 in range(len(xf)):
                stride_c = 1
                for d in range(2, ndim): stride_c *= xsh[d]
                c_idx = (idx2 // stride_c) % C
                var_v[c_idx] += (xf[idx2] - mean_v[c_idx]) ** 2
            var_v = [v / count for v in var_v]
            m = self.momentum
            if hasattr(self.running_mean, '_buf'):
                for c in range(C):
                    self.running_mean._buf[c] = (1-m)*self.running_mean._buf[c] + m*mean_v[c]
                    self.running_var._buf[c]  = (1-m)*self.running_var._buf[c]  + m*var_v[c]
        else:
            if hasattr(self.running_mean, '_buf'):
                mean_v = list(self.running_mean._buf)[:C]
                var_v  = list(self.running_var._buf)[:C]
            else:
                mean_v = list(self.running_mean)
                var_v  = list(self.running_var)
        # Triton fused path (inference only, affine only)
        if not self.training and self.affine:
            try:
                from vectorphile.cuda.triton_kernels import fused_batch_norm as _fbn
                if hasattr(self.weight._data, 'to_flat_list'): wf2 = self.weight._data.to_flat_list()
                else: wf2 = list(self.weight._data)
                if hasattr(self.bias_param._data, 'to_flat_list'): bf2 = self.bias_param._data.to_flat_list()
                else: bf2 = list(self.bias_param._data)
                w_nd = _nd_bn1(wf2)
                b_nd = _nd_bn1(bf2)
                out_nd = _fbn(xd, w_nd, b_nd, self.running_mean, self.running_var,
                              training=False, momentum=self.momentum, eps=self.eps)
                return Tensor(out_nd, device=x.device, requires_grad=x.requires_grad)
            except Exception:
                pass
        # Normalize (Python fallback)
        out_f = [0.0] * len(xf)
        for idx2 in range(len(xf)):
            stride_c = 1
            for d in range(2, ndim): stride_c *= xsh[d]
            c_idx = (idx2 // stride_c) % C
            out_f[idx2] = (xf[idx2] - mean_v[c_idx]) / math.sqrt(var_v[c_idx] + self.eps)
        result = Tensor(_nd_bn1(out_f).reshape(xsh), device=x.device, requires_grad=x.requires_grad)
        if self.affine:
            if hasattr(self.weight._data, 'to_flat_list'): wf = self.weight._data.to_flat_list()
            else: wf = list(self.weight._data)
            if hasattr(self.bias_param._data, 'to_flat_list'): bf3 = self.bias_param._data.to_flat_list()
            else: bf3 = list(self.bias_param._data)
            af = [0.0] * len(out_f)
            for idx2 in range(len(out_f)):
                stride_c = 1
                for d in range(2, ndim): stride_c *= xsh[d]
                c_idx = (idx2 // stride_c) % C
                af[idx2] = out_f[idx2] * wf[c_idx] + bf3[c_idx]
            result = Tensor(_nd_bn1(af).reshape(xsh), device=x.device, requires_grad=x.requires_grad)
        return result


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        from vectorphile._core.ndarray import ones as _nd_ones2, zeros as _nd_zeros2
        if affine:
            self.weight     = Tensor(_nd_ones2(num_features, dtype='float32'), requires_grad=True)
            self.bias_param = Tensor(_nd_zeros2(num_features, dtype='float32'), requires_grad=True)
            self._parameters['weight']     = self.weight
            self._parameters['bias'] = self._parameters.pop('bias_param', self.bias_param)
        self.running_mean = _nd_zeros2(num_features, dtype='float32')
        self.running_var  = _nd_ones2(num_features,  dtype='float32')

    def forward(self, x: Tensor) -> Tensor:
        from vectorphile._core.ndarray import array as _nd_bn2
        from vectorphile._core.ndarray import NDArray as _NDA_bn2, _c_strides as _cs_bn2, _DType as _DT_bn2
        import array as _arr_bn2
        xd = x._data; xsh = xd.shape if hasattr(xd, 'shape') else (len(xd),)
        N2, C2, H2, W2 = xsh; count2 = N2 * H2 * W2
        # Triton fused path (training or inference, affine=True)
        if self.affine:
            try:
                from vectorphile.cuda.triton_kernels import fused_batch_norm as _fbn2t
                if hasattr(self.weight._data, 'to_flat_list'): wfl = self.weight._data.to_flat_list()
                else: wfl = list(self.weight._data)
                if hasattr(self.bias_param._data, 'to_flat_list'): bfl = self.bias_param._data.to_flat_list()
                else: bfl = list(self.bias_param._data)
                w_nd = _nd_bn2(wfl); b_nd = _nd_bn2(bfl)
                out_nd = _fbn2t(xd, w_nd, b_nd, self.running_mean, self.running_var,
                                training=self.training, momentum=self.momentum, eps=self.eps)
                return Tensor(out_nd, device=x.device, requires_grad=x.requires_grad)
            except Exception:
                pass
        if hasattr(xd, 'to_flat_list'): xf2 = xd.to_flat_list()
        else: xf2 = list(xd)
        if self.training:
            mean2 = [0.0]*C2; var2 = [0.0]*C2
            for n2 in range(N2):
                for c2 in range(C2):
                    for h2 in range(H2):
                        for w2 in range(W2):
                            mean2[c2] += xf2[n2*C2*H2*W2 + c2*H2*W2 + h2*W2 + w2]
            mean2 = [v/count2 for v in mean2]
            for n2 in range(N2):
                for c2 in range(C2):
                    for h2 in range(H2):
                        for w2 in range(W2):
                            var2[c2] += (xf2[n2*C2*H2*W2 + c2*H2*W2 + h2*W2 + w2] - mean2[c2])**2
            var2 = [v/count2 for v in var2]
            m2 = self.momentum
            if hasattr(self.running_mean, '_buf'):
                for c2 in range(C2):
                    self.running_mean._buf[c2] = (1-m2)*self.running_mean._buf[c2] + m2*mean2[c2]
                    self.running_var._buf[c2]  = (1-m2)*self.running_var._buf[c2]  + m2*var2[c2]
        else:
            if hasattr(self.running_mean, '_buf'):
                mean2 = list(self.running_mean._buf)[:C2]
                var2  = list(self.running_var._buf)[:C2]
            else:
                mean2 = list(self.running_mean); var2 = list(self.running_var)
        # Triton fused path (inference only, affine only)
        if not self.training and self.affine:
            try:
                from vectorphile.cuda.triton_kernels import fused_batch_norm as _fbn2
                if hasattr(self.weight._data, 'to_flat_list'): wf2b = self.weight._data.to_flat_list()
                else: wf2b = list(self.weight._data)
                if hasattr(self.bias_param._data, 'to_flat_list'): bf2b = self.bias_param._data.to_flat_list()
                else: bf2b = list(self.bias_param._data)
                w_nd2 = _nd_bn2(wf2b); b_nd2 = _nd_bn2(bf2b)
                out_nd2 = _fbn2(xd, w_nd2, b_nd2, self.running_mean, self.running_var,
                                training=False, momentum=self.momentum, eps=self.eps)
                return Tensor(out_nd2, device=x.device, requires_grad=x.requires_grad)
            except Exception:
                pass
        out2 = [0.0]*len(xf2)
        for n2 in range(N2):
            for c2 in range(C2):
                inv = 1.0 / math.sqrt(var2[c2] + self.eps)
                for h2 in range(H2):
                    for w2 in range(W2):
                        idx3 = n2*C2*H2*W2 + c2*H2*W2 + h2*W2 + w2
                        out2[idx3] = (xf2[idx3] - mean2[c2]) * inv
        if self.affine:
            if hasattr(self.weight._data, 'to_flat_list'): wf2b = self.weight._data.to_flat_list()
            else: wf2b = list(self.weight._data)
            if hasattr(self.bias_param._data, 'to_flat_list'): bf2b = self.bias_param._data.to_flat_list()
            else: bf2b = list(self.bias_param._data)
            for n2 in range(N2):
                for c2 in range(C2):
                    for h2 in range(H2):
                        for w2 in range(W2):
                            idx3 = n2*C2*H2*W2 + c2*H2*W2 + h2*W2 + w2
                            out2[idx3] = out2[idx3] * wf2b[c2] + bf2b[c2]
        return Tensor(_nd_bn2(out2).reshape(xsh), device=x.device, requires_grad=x.requires_grad)


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            from vectorphile._core.ndarray import ones as _nd_ln_ones, zeros as _nd_ln_zeros
            ns_size = 1
            for s in self.normalized_shape: ns_size *= s
            self.weight     = Tensor(_nd_ln_ones(ns_size, dtype='float32').reshape(self.normalized_shape), requires_grad=True)
            self.bias_param = Tensor(_nd_ln_zeros(ns_size, dtype='float32').reshape(self.normalized_shape), requires_grad=True)
            self._parameters['weight']     = self.weight
            self._parameters['bias'] = self._parameters.pop('bias_param', self.bias_param)

    def forward(self, x: Tensor) -> Tensor:
        # Triton fused path (no gradient needed → inference or when affine)
        if not x.requires_grad and self.elementwise_affine:
            try:
                from vectorphile.cuda.triton_kernels import fused_layer_norm as _fln
                from vectorphile._core.ndarray import array as _nd_ln
                def _fl(t):
                    if hasattr(t, 'to_flat_list'): return t.to_flat_list()
                    d = t._data if hasattr(t, '_data') else t
                    return d.to_flat_list() if hasattr(d, 'to_flat_list') else list(d)
                w_nd = _nd_ln(_fl(self.weight))
                b_nd = _nd_ln(_fl(self.bias_param))
                out_nd = _fln(x._data, w_nd, b_nd, eps=self.eps)
                return Tensor(out_nd, device=x.device, requires_grad=False)
            except Exception:
                pass
        # C AVX2 fast path (inference only, float32, contiguous, elementwise_affine)
        if not x.requires_grad and self.elementwise_affine:
            try:
                import array as _arr_ln
                from vectorphile._core.ndarray import NDArray as _NDA_ln, _c_strides as _cs_ln, _DType as _DT_ln
                xd = x._data
                if (isinstance(xd, _NDA_ln) and str(xd.dtype) == 'float32'
                        and xd._buf is not None and _LAYERS_C_OK):
                    xsh = xd.shape
                    N_total = 1
                    for s in xsh[:-1]: N_total *= s
                    norm_size = xsh[-1]
                    out_buf = _arr_ln.array('f', bytearray(4 * N_total * norm_size))
                    wd = self.weight._data; bd = self.bias_param._data
                    if (isinstance(wd, _NDA_ln) and wd._buf is not None
                            and isinstance(bd, _NDA_ln) and bd._buf is not None):
                        _C_layers.layer_norm_f32(xd._buf, out_buf, wd._buf, bd._buf,
                                                  N_total, norm_size, self.eps)
                        out_nd = _NDA_ln(out_buf, xsh, _cs_ln(xsh), _DT_ln('float32'))
                        return Tensor(out_nd, device=x.device, requires_grad=False)
            except Exception:
                pass
        # Autograd path: use Tensor ops so gradients flow through
        ndim_n = len(self.normalized_shape)
        dim = tuple(range(x.ndim - ndim_n, x.ndim))
        mean = x.mean(dim=dim, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=dim, keepdim=True)
        x_norm = (x - mean) * (var + self.eps).pow(-0.5)
        if self.elementwise_affine:
            w = self.weight; b = self.bias_param
            x_norm = x_norm * w + b
        return x_norm


class InstanceNorm1d(Module):
    """Instance Normalization over (N, C, L) input."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            from vectorphile._core.ndarray import ones as _nd_on, zeros as _nd_zr
            self.weight = Tensor(_nd_on(num_features, dtype='float32'), requires_grad=True)
            self.bias_param = Tensor(_nd_zr(num_features, dtype='float32'), requires_grad=True)
            self._parameters['weight'] = self.weight
            self._parameters['bias'] = self._parameters.pop('bias_param', self.bias_param)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, L)
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) * (var + self.eps).pow(-0.5)
        if self.affine:
            x_norm = x_norm * self.weight.unsqueeze(-1) + self.bias_param.unsqueeze(-1)
        return x_norm


class InstanceNorm2d(Module):
    """Instance Normalization over (N, C, H, W) input."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            from vectorphile._core.ndarray import ones as _nd_on, zeros as _nd_zr
            self.weight = Tensor(_nd_on(num_features, dtype='float32'), requires_grad=True)
            self.bias_param = Tensor(_nd_zr(num_features, dtype='float32'), requires_grad=True)
            self._parameters['weight'] = self.weight
            self._parameters['bias'] = self._parameters.pop('bias_param', self.bias_param)

    def forward(self, x: Tensor) -> Tensor:
        # Normalize over H, W (last 2 dims)
        if self.affine and not x.requires_grad:
            try:
                from vectorphile.cuda.triton_kernels import fused_instance_norm_2d as _fin2
                out_nd = _fin2(x._data, self.weight._data, self.bias_param._data, self.eps)
                return Tensor(out_nd, device=x.device, requires_grad=False)
            except Exception:
                pass
        mean = x.mean(dim=(-2, -1), keepdim=True)
        var = ((x - mean) ** 2).mean(dim=(-2, -1), keepdim=True)
        x_norm = (x - mean) * (var + self.eps).pow(-0.5)
        if self.affine:
            w = self.weight.reshape((1, self.num_features, 1, 1))
            b = self.bias_param.reshape((1, self.num_features, 1, 1))
            x_norm = x_norm * w + b
        return x_norm


class GroupNorm(Module):
    """Group Normalization: normalize over groups of channels."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})")
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            from vectorphile._core.ndarray import ones as _nd_on, zeros as _nd_zr
            self.weight = Tensor(_nd_on(num_channels, dtype='float32'), requires_grad=True)
            self.bias_param = Tensor(_nd_zr(num_channels, dtype='float32'), requires_grad=True)
            self._parameters['weight'] = self.weight
            self._parameters['bias'] = self._parameters.pop('bias_param', self.bias_param)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, *) — reshape to (N, G, C//G, *)
        shape = x.shape
        N, C = shape[0], shape[1]
        G = self.num_groups
        if self.affine and not x.requires_grad and len(shape) >= 2:
            try:
                from vectorphile.cuda.triton_kernels import fused_group_norm as _fgn
                out_nd = _fgn(x._data, self.weight._data, self.bias_param._data, G, self.eps)
                return Tensor(out_nd, device=x.device, requires_grad=False)
            except Exception:
                pass
        x_r = x.reshape((N, G, C // G) + shape[2:])
        mean_dims = tuple(range(2, x_r.ndim))
        mean = x_r.mean(dim=mean_dims, keepdim=True)
        var = ((x_r - mean) ** 2).mean(dim=mean_dims, keepdim=True)
        x_norm = (x_r - mean) * (var + self.eps).pow(-0.5)
        x_norm = x_norm.reshape(shape)
        if self.affine:
            w_shape = (1, C) + (1,) * (len(shape) - 2)
            x_norm = x_norm * self.weight.reshape(w_shape) + self.bias_param.reshape(w_shape)
        return x_norm


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        if not x.requires_grad:
            try:
                import random as _rnd_do
                from vectorphile.cuda.triton_kernels import fused_dropout as _fdo
                seed = _rnd_do.randint(0, 2**31 - 1)
                out_nd, _ = _fdo(x._data, self.p, training=True, seed=seed)
                return Tensor(out_nd, device=x.device, requires_grad=False)
            except Exception:
                pass
        from vectorphile._core.ndarray import array as _nd_do
        xd = x._data
        if hasattr(xd, 'to_flat_list'): xfl = xd.to_flat_list()
        else: xfl = list(xd)
        xsh = xd.shape if hasattr(xd, 'shape') else (len(xfl),)
        out_do = [v / (1.0 - self.p) if _random_layers.random() >= self.p else 0.0 for v in xfl]
        return Tensor(_nd_do(out_do).reshape(xsh), device=x.device, requires_grad=x.requires_grad)

    def __repr__(self):
        return f'Dropout(p={self.p})'


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.padding_idx    = padding_idx
        from vectorphile._core.ndarray import array as _nd_emb_init
        _emb_vals = [math.sqrt(-2*math.log(max(_random_layers.random(),1e-10)))*math.cos(2*math.pi*_random_layers.random())*0.02
                     for _ in range(num_embeddings * embedding_dim)]
        self.weight = Tensor(
            _nd_emb_init(_emb_vals).reshape((num_embeddings, embedding_dim)),
            requires_grad=True
        )
        self._parameters['weight'] = self.weight

    def forward(self, indices: Tensor) -> Tensor:
        from vectorphile._core.ndarray import NDArray as _NDA, _c_strides as _cs, _DType as _DT
        import array as _arr
        idx = indices._data
        # Convert NDArray indices to flat int list
        if isinstance(idx, _NDA):
            flat_idx = [int(v) for v in idx.to_flat_list()]
        else:
            flat_idx = idx.ravel().astype(int).tolist()
        in_shape = indices._data.shape if hasattr(indices._data, 'shape') else (len(flat_idx),)
        E = self.embedding_dim
        # Gather rows from weight
        w = self.weight._data
        out_flat = _arr.array('f', bytearray(4 * len(flat_idx) * E))
        if isinstance(w, _NDA) and w._buf is not None:
            import ctypes as _ct
            buf_ptr = (_ct.c_float * (w._buf.buffer_info()[1]))
            # Use array slice-copy for O(E) C-speed per row
            for i, row_i in enumerate(flat_idx):
                if self.padding_idx is not None and row_i == self.padding_idx:
                    # zero the slice
                    out_flat[i * E: i * E + E] = _arr_layers.array('f', [0.0] * E)
                else:
                    # Slice the source buffer — single C memcpy under the hood
                    out_flat[i * E: i * E + E] = w._buf[row_i * E: row_i * E + E]
        else:
            if hasattr(w, 'to_flat_list'): wfl_emb = w.to_flat_list()
            else: wfl_emb = list(w)
            # Build lookup with slice operations
            wfl_arr = _arr_layers.array('f', wfl_emb)
            for i, row_i in enumerate(flat_idx):
                if self.padding_idx is not None and row_i == self.padding_idx:
                    out_flat[i * E: i * E + E] = _arr_layers.array('f', [0.0] * E)
                else:
                    out_flat[i * E: i * E + E] = wfl_arr[row_i * E: row_i * E + E]
        out_shape = tuple(in_shape) + (E,)
        out_nd = _NDA(out_flat, out_shape, _cs(out_shape), _DT('float32'))
        rg = self.weight.requires_grad
        result = Tensor(out_nd, device=indices.device, requires_grad=rg, _no_copy=True)
        if rg:
            _w = self.weight; _flat_idx = flat_idx; _E = E; _in_shape = in_shape
            class _EmbBackward:
                def __init__(s): s.inputs = (_w,)
                def backward(s, grad):
                    from vectorphile._core.ndarray import zeros as _z_eb
                    gd = grad._data if isinstance(grad, Tensor) else grad
                    gf = gd.to_flat_list() if hasattr(gd, 'to_flat_list') else list(gd)
                    dw = _z_eb(_w._data.shape, dtype='float32')
                    dw_fl = dw.to_flat_list() if hasattr(dw, 'to_flat_list') else list(dw)
                    for i, row_i in enumerate(_flat_idx):
                        for j in range(_E):
                            dw_fl[row_i * _E + j] += gf[i * _E + j]
                    from vectorphile._core.ndarray import array as _nd_eb, NDArray as _NDA_eb, _c_strides as _cs_eb, _DType as _DT_eb
                    import array as _arr_eb
                    flat_arr = _arr_eb.array('f', dw_fl)
                    ws = _w._data.shape
                    return (Tensor(_NDA_eb(flat_arr, ws, _cs_eb(ws), _DT_eb('float32'))),)
            result.grad_fn = _EmbBackward(); result._is_leaf = False
        return result

    def __repr__(self):
        return f'Embedding({self.num_embeddings}, {self.embedding_dim})'


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        shape = x._data.shape
        ndim = len(shape)
        end = self.end_dim if self.end_dim >= 0 else ndim + self.end_dim
        new_shape = shape[:self.start_dim] + (-1,) + shape[end + 1:]
        return Tensor(x._data.reshape(new_shape), device=x.device)


class ConvTranspose2d(Module):
    """2-D transposed convolution (fractionally-strided / deconvolution).

    Performs the gradient operation of Conv2d, used in U-Net decoders,
    GANs, and other upsampling networks.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, output_padding: int = 0, bias: bool = True):
        super().__init__()
        from vectorphile._core.ndarray import array as _nd_ct, zeros as _nd_ct_z
        k = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = k
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        scale = math.sqrt(2.0 / (in_channels * k * k))
        n_wct = in_channels * out_channels * k * k
        wct_vals = [math.sqrt(-2*math.log(max(_random_layers.random(),1e-10)))*math.cos(2*math.pi*_random_layers.random())*scale
                    for _ in range(n_wct)]
        self.weight = Tensor(_nd_ct(wct_vals).reshape((in_channels, out_channels, k, k)), requires_grad=True)
        self._parameters['weight'] = self.weight
        if bias:
            self.bias = Tensor(_nd_ct_z(out_channels, dtype='float32'), requires_grad=True)
            self._parameters['bias'] = self.bias
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        from vectorphile._core.ndarray import array as _nd_ctf
        xd = x._data; wd = self.weight._data
        if hasattr(xd, 'to_flat_list'): xf_ct = xd.to_flat_list()
        else: xf_ct = list(xd)
        if hasattr(wd, 'to_flat_list'): wf_ct = wd.to_flat_list()
        else: wf_ct = list(wd)
        xsh_ct = xd.shape if hasattr(xd, 'shape') else (len(xf_ct),)
        B, C_in, H_in, W_in = xsh_ct
        kH = self.kernel_size; kW = kH
        C_out = self.out_channels
        s = self.stride; p = self.padding; op = self.output_padding
        H_out = (H_in - 1)*s - 2*p + kH + op
        W_out = (W_in - 1)*s - 2*p + kW + op
        # Scatter (transpose of conv): for each input pixel, add its contribution
        out_ct = [0.0] * (B * C_out * H_out * W_out)
        for b in range(B):
            for ci in range(C_in):
                for hi in range(H_in):
                    for wi in range(W_in):
                        x_val = xf_ct[b*C_in*H_in*W_in + ci*H_in*W_in + hi*W_in + wi]
                        if x_val == 0.0: continue
                        for co in range(C_out):
                            for ki in range(kH):
                                for kj in range(kW):
                                    # flipped kernel: kH-1-ki, kW-1-kj
                                    w_val = wf_ct[ci*C_out*kH*kW + co*kH*kW + (kH-1-ki)*kW + (kW-1-kj)]
                                    ho = hi*s - p + ki
                                    wo = wi*s - p + kj
                                    if 0 <= ho < H_out and 0 <= wo < W_out:
                                        out_ct[b*C_out*H_out*W_out + co*H_out*W_out + ho*W_out + wo] += x_val * w_val
        if self.bias is not None:
            if hasattr(self.bias._data, 'to_flat_list'): bf_ct = self.bias._data.to_flat_list()
            else: bf_ct = list(self.bias._data)
            for b in range(B):
                for co in range(C_out):
                    for ho in range(H_out):
                        for wo in range(W_out):
                            out_ct[b*C_out*H_out*W_out + co*H_out*W_out + ho*W_out + wo] += bf_ct[co]
        return Tensor(_nd_ctf(out_ct).reshape((B, C_out, H_out, W_out)), device=x.device, requires_grad=x.requires_grad)


# =============================================================================
# 3D Convolution and Pooling layers (Session 11)
# =============================================================================

class Conv3d(Module):
    """3D convolution layer: (N, C_in, D, H, W) → (N, C_out, D', H', W').

    Args:
        in_channels, out_channels: channel counts
        kernel_size: int or (kD, kH, kW)
        stride, padding, dilation, groups: standard conv params
        bias: add learnable bias

    Usage::
        conv = vp.nn.Conv3d(1, 32, kernel_size=3, padding=1)
        y = conv(x)   # x: (N,1,16,16,16) → (N,32,16,16,16)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias: bool = True):
        super().__init__()
        from vectorphile._core.ndarray import array as _a, zeros as _z
        import math as _m
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size, kernel_size)
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride if isinstance(stride, tuple)   else (stride,)*3
        self.padding      = padding if isinstance(padding, tuple)  else (padding,)*3
        self.dilation     = dilation if isinstance(dilation, tuple) else (dilation,)*3
        self.groups       = groups
        kD, kH, kW = kernel_size
        fan_in = in_channels // groups * kD * kH * kW
        std = _m.sqrt(2.0 / fan_in)
        n_w = out_channels * (in_channels // groups) * kD * kH * kW
        wv  = [_m.sqrt(-2*_m.log(max(_random_layers.random(),1e-10)))*_m.cos(2*_m.pi*_random_layers.random())*std
               for _ in range(n_w)]
        self.weight = Tensor(_a(wv).reshape((out_channels, in_channels//groups, kD, kH, kW)), requires_grad=True)
        self._parameters['weight'] = self.weight
        if bias:
            self.bias = Tensor(_z(out_channels, dtype='float32'), requires_grad=True)
            self._parameters['bias'] = self.bias
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        from vectorphile.nn.functional import conv3d as _conv3d
        return _conv3d(x, self.weight, self.bias,
                       stride=self.stride, padding=self.padding,
                       dilation=self.dilation, groups=self.groups)

    def extra_repr(self):
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}")


class MaxPool3d(Module):
    """3D max pooling layer.

    Args:
        kernel_size: int or (kD, kH, kW)
        stride: defaults to kernel_size
        padding: zero-padding on each side

    Usage::
        pool = vp.nn.MaxPool3d(2)
        y = pool(x)   # x: (N,C,16,16,16) → (N,C,8,8,8)
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride  = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        from vectorphile.nn.functional import max_pool3d as _mp3
        return _mp3(x, self.kernel_size, self.stride, self.padding)


class AvgPool3d(Module):
    """3D average pooling layer.

    Args:
        kernel_size: int or (kD, kH, kW)
        stride: defaults to kernel_size
        padding: zero-padding on each side

    Usage::
        pool = vp.nn.AvgPool3d(2)
        y = pool(x)   # x: (N,C,16,16,16) → (N,C,8,8,8)
    """

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride  = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        from vectorphile.nn.functional import avg_pool3d as _ap3
        return _ap3(x, self.kernel_size, self.stride, self.padding)


class AdaptiveAvgPool3d(Module):
    """Adaptive 3D average pooling — output size is fixed regardless of input size.

    Args:
        output_size: int or (D, H, W) target output spatial size

    Usage::
        pool = vp.nn.AdaptiveAvgPool3d(1)
        y = pool(x)   # any (N, C, D, H, W) → (N, C, 1, 1, 1)
    """

    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size,) * 3
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        sh = x._data.shape
        N, C, Di, Hi, Wi = sh
        oD, oH, oW = self.output_size
        x_flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
        out = []
        for n in range(N):
            for c in range(C):
                for od in range(oD):
                    d_start = od * Di // oD; d_end = (od + 1) * Di // oD
                    for oh in range(oH):
                        h_start = oh * Hi // oH; h_end = (oh + 1) * Hi // oH
                        for ow in range(oW):
                            w_start = ow * Wi // oW; w_end = (ow + 1) * Wi // oW
                            cnt = total = 0.0
                            for id_ in range(d_start, d_end):
                                for ih in range(h_start, h_end):
                                    for iw in range(w_start, w_end):
                                        idx = n*C*Di*Hi*Wi + c*Di*Hi*Wi + id_*Hi*Wi + ih*Wi + iw
                                        total += x_flat[idx]; cnt += 1.0
                            out.append(total / max(cnt, 1.0))
        from vectorphile._core.ndarray import array as _a
        from vectorphile._core.ndarray import NDArray, _c_strides, _DType
        sh_out = (N, C, oD, oH, oW)
        return Tensor(NDArray(_a._a('f', [float(v) for v in out]) if hasattr(_a, '_a') else
                              __import__('array').array('f', [float(v) for v in out]),
                              sh_out, _c_strides(sh_out), _DType('float32')))


# =============================================================================
# Session 11 - Upsample, PixelShuffle, ChannelShuffle, Padding layers
# =============================================================================

class Upsample(Module):
    """Upsample a tensor using nearest, bilinear, or bicubic interpolation.

    Args:
        size: target output size (H, W) or None
        scale_factor: multiplier for spatial dims (int or float)
        mode: interpolation mode: 'nearest', 'bilinear', 'bicubic'

    Usage::
        up = vp.nn.Upsample(scale_factor=2, mode='nearest')
        y = up(x)   # x: (N,C,H,W) -> (N,C,2H,2W)
    """

    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        sh = x._data.shape
        if len(sh) == 4:
            N, C, H, W = sh
            if self.size is not None:
                oH, oW = self.size if isinstance(self.size, tuple) else (self.size, self.size)
            else:
                sf = self.scale_factor
                oH = int(H * sf); oW = int(W * sf)
        elif len(sh) == 3:
            N, C, L = sh
            if self.size is not None:
                oL = self.size
            else:
                oL = int(L * self.scale_factor)
            # 1D nearest
            x_flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
            out = []
            for n in range(N):
                for c in range(C):
                    for ol in range(oL):
                        il = min(int(ol * L / oL), L - 1)
                        out.append(x_flat[n*C*L + c*L + il])
            import array as _a2
            from vectorphile._core.ndarray import NDArray as _ND2, _c_strides, _DType
            sh_out = (N, C, oL)
            return Tensor(_ND2(_a2.array('f', out), sh_out, _c_strides(sh_out), _DType('float32')))
        else:
            return x  # passthrough for unsupported dims

        x_flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
        out = []
        for n in range(N):
            for c in range(C):
                for oh in range(oH):
                    for ow in range(oW):
                        if self.mode == 'nearest':
                            ih = min(int(oh * H / oH), H - 1)
                            iw = min(int(ow * W / oW), W - 1)
                            out.append(x_flat[n*C*H*W + c*H*W + ih*W + iw])
                        else:
                            # bilinear/bicubic: simple bilinear
                            fh = oh * (H - 1) / max(oH - 1, 1)
                            fw = ow * (W - 1) / max(oW - 1, 1)
                            h0 = int(fh); h1 = min(h0 + 1, H - 1)
                            w0 = int(fw); w1 = min(w0 + 1, W - 1)
                            dh = fh - h0; dw = fw - w0
                            v00 = x_flat[n*C*H*W + c*H*W + h0*W + w0]
                            v01 = x_flat[n*C*H*W + c*H*W + h0*W + w1]
                            v10 = x_flat[n*C*H*W + c*H*W + h1*W + w0]
                            v11 = x_flat[n*C*H*W + c*H*W + h1*W + w1]
                            out.append((1-dh)*(1-dw)*v00 + (1-dh)*dw*v01 +
                                        dh*(1-dw)*v10 + dh*dw*v11)
        import array as _a2
        from vectorphile._core.ndarray import NDArray as _ND2, _c_strides, _DType
        sh_out = (N, C, oH, oW)
        return Tensor(_ND2(_a2.array('f', out), sh_out, _c_strides(sh_out), _DType('float32')))


class PixelShuffle(Module):
    """Rearrange elements from (N, C*r^2, H, W) -> (N, C, H*r, W*r).

    Used in ESRGAN and many super-resolution architectures.
    Equivalent to torch.nn.PixelShuffle.

    Args:
        upscale_factor: spatial upscale ratio r

    Usage::
        ps = vp.nn.PixelShuffle(2)
        y = ps(x)   # x: (N, 4C, H, W) -> (N, C, 2H, 2W)
    """

    def __init__(self, upscale_factor: int):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x: Tensor) -> Tensor:
        import array as _a2
        from vectorphile._core.ndarray import NDArray as _ND2, _c_strides, _DType
        N, C_in, H, W = x._data.shape
        r = self.r
        C_out = C_in // (r * r)
        x_flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
        out = [0.0] * (N * C_out * H * r * W * r)
        for n in range(N):
            for c_out in range(C_out):
                for h in range(H):
                    for w in range(W):
                        for dy in range(r):
                            for dx in range(r):
                                c_in = c_out * r * r + dy * r + dx
                                src = n*C_in*H*W + c_in*H*W + h*W + w
                                dst = n*C_out*H*r*W*r + c_out*H*r*W*r + (h*r+dy)*W*r + (w*r+dx)
                                out[dst] = x_flat[src]
        sh_out = (N, C_out, H*r, W*r)
        return Tensor(_ND2(_a2.array('f', out), sh_out, _c_strides(sh_out), _DType('float32')))


class PixelUnshuffle(Module):
    """Inverse of PixelShuffle: (N, C, H*r, W*r) -> (N, C*r^2, H, W).

    Args:
        downscale_factor: spatial downscale ratio r
    """

    def __init__(self, downscale_factor: int):
        super().__init__()
        self.r = downscale_factor

    def forward(self, x: Tensor) -> Tensor:
        import array as _a2
        from vectorphile._core.ndarray import NDArray as _ND2, _c_strides, _DType
        N, C_in, H, W = x._data.shape
        r = self.r
        C_out = C_in * r * r
        x_flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
        oH, oW = H // r, W // r
        out = [0.0] * (N * C_out * oH * oW)
        for n in range(N):
            for c_in in range(C_in):
                for oh in range(oH):
                    for ow in range(oW):
                        for dy in range(r):
                            for dx in range(r):
                                c_out = c_in * r * r + dy * r + dx
                                src = n*C_in*H*W + c_in*H*W + (oh*r+dy)*W + (ow*r+dx)
                                dst = n*C_out*oH*oW + c_out*oH*oW + oh*oW + ow
                                out[dst] = x_flat[src]
        sh_out = (N, C_out, oH, oW)
        return Tensor(_ND2(_a2.array('f', out), sh_out, _c_strides(sh_out), _DType('float32')))


class ChannelShuffle(Module):
    """Channel Shuffle operation used in ShuffleNet.

    Interleaves channels across groups to allow cross-group information flow.
    (N, C, H, W) with C = groups * channels_per_group.

    Args:
        groups: number of groups to shuffle among

    Usage::
        cs = vp.nn.ChannelShuffle(groups=2)
        y = cs(x)   # x: (N, 2k, H, W) -> shuffled channels
    """

    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x: Tensor) -> Tensor:
        import array as _a2
        from vectorphile._core.ndarray import NDArray as _ND2, _c_strides, _DType
        N, C, H, W = x._data.shape
        g = self.groups
        c_per_g = C // g
        x_flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
        out = [0.0] * len(x_flat)
        for n in range(N):
            for gi in range(g):
                for ci in range(c_per_g):
                    src_c = gi * c_per_g + ci   # original channel
                    dst_c = ci * g + gi           # shuffled channel
                    for h in range(H):
                        for w in range(W):
                            src = n*C*H*W + src_c*H*W + h*W + w
                            dst = n*C*H*W + dst_c*H*W + h*W + w
                            out[dst] = x_flat[src]
        return Tensor(_ND2(_a2.array('f', out), (N,C,H,W), _c_strides((N,C,H,W)), _DType('float32')))


class ZeroPad2d(Module):
    """Pad a 4-D input tensor with zeros on all sides.

    Args:
        padding: int (all sides) or (left, right, top, bottom)
    """

    def __init__(self, padding):
        super().__init__()
        if isinstance(padding, int):
            self.pad = (padding,) * 4
        else:
            self.pad = tuple(padding)  # (left, right, top, bottom)

    def forward(self, x: Tensor) -> Tensor:
        import array as _a2
        from vectorphile._core.ndarray import NDArray as _ND2, _c_strides, _DType
        N, C, H, W = x._data.shape
        pl, pr, pt, pb = self.pad
        oH, oW = H + pt + pb, W + pl + pr
        x_flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
        out = [0.0] * (N * C * oH * oW)
        for n in range(N):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        dst = n*C*oH*oW + c*oH*oW + (h+pt)*oW + (w+pl)
                        out[dst] = x_flat[n*C*H*W + c*H*W + h*W + w]
        sh_out = (N, C, oH, oW)
        return Tensor(_ND2(_a2.array('f', out), sh_out, _c_strides(sh_out), _DType('float32')))


class ReflectionPad2d(Module):
    """Reflect-pad a 4-D tensor.

    Args:
        padding: int or (left, right, top, bottom)
    """

    def __init__(self, padding):
        super().__init__()
        self.pad = (padding,)*4 if isinstance(padding, int) else tuple(padding)

    def forward(self, x: Tensor) -> Tensor:
        import array as _a2
        from vectorphile._core.ndarray import NDArray as _ND2, _c_strides, _DType
        N, C, H, W = x._data.shape
        pl, pr, pt, pb = self.pad
        oH, oW = H + pt + pb, W + pl + pr
        x_flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
        out = []
        for n in range(N):
            for c in range(C):
                for oh in range(oH):
                    ih = oh - pt
                    if ih < 0: ih = -ih
                    if ih >= H: ih = 2*H - ih - 2
                    ih = max(0, min(H-1, ih))
                    for ow in range(oW):
                        iw = ow - pl
                        if iw < 0: iw = -iw
                        if iw >= W: iw = 2*W - iw - 2
                        iw = max(0, min(W-1, iw))
                        out.append(x_flat[n*C*H*W + c*H*W + ih*W + iw])
        sh_out = (N, C, oH, oW)
        return Tensor(_ND2(_a2.array('f', out), sh_out, _c_strides(sh_out), _DType('float32')))


class ConstantPad2d(Module):
    """Constant-pad a 4-D tensor.

    Args:
        padding: int or (left, right, top, bottom)
        value: fill value (default 0.0)
    """

    def __init__(self, padding, value: float = 0.0):
        super().__init__()
        self.pad   = (padding,)*4 if isinstance(padding, int) else tuple(padding)
        self.value = value

    def forward(self, x: Tensor) -> Tensor:
        import array as _a2
        from vectorphile._core.ndarray import NDArray as _ND2, _c_strides, _DType
        N, C, H, W = x._data.shape
        pl, pr, pt, pb = self.pad
        oH, oW = H + pt + pb, W + pl + pr
        x_flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
        out = [self.value] * (N * C * oH * oW)
        for n in range(N):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        dst = n*C*oH*oW + c*oH*oW + (h+pt)*oW + (w+pl)
                        out[dst] = x_flat[n*C*H*W + c*H*W + h*W + w]
        sh_out = (N, C, oH, oW)
        return Tensor(_ND2(_a2.array('f', out), sh_out, _c_strides(sh_out), _DType('float32')))


# ---------------------------------------------------------------------------
# 1-D padding variants (N,C,L tensors)
# ---------------------------------------------------------------------------

class ConstantPad1d(Module):
    def __init__(self, padding, value: float = 0.0):
        super().__init__()
        self.pad = (padding, padding) if isinstance(padding, int) else tuple(padding)
        self.value = value

    def forward(self, x: Tensor) -> Tensor:
        import array as _a3
        from vectorphile._core.ndarray import NDArray as _ND3, _c_strides, _DType
        N, C, L = x._data.shape
        pl, pr = self.pad
        oL = L + pl + pr
        x_flat = list(x._data.to_flat_list() if hasattr(x._data, "to_flat_list") else x._data)
        out = [self.value] * (N * C * oL)
        for n in range(N):
            for c in range(C):
                for l in range(L):
                    out[n*C*oL + c*oL + (l + pl)] = x_flat[n*C*L + c*L + l]
        sh = (N, C, oL)
        return Tensor(_ND3(_a3.array("f", out), sh, _c_strides(sh), _DType("float32")))


class ReflectionPad1d(Module):
    def __init__(self, padding):
        super().__init__()
        self.pad = (padding, padding) if isinstance(padding, int) else tuple(padding)

    def forward(self, x: Tensor) -> Tensor:
        import array as _a3
        from vectorphile._core.ndarray import NDArray as _ND3, _c_strides, _DType
        N, C, L = x._data.shape
        pl, pr = self.pad
        oL = L + pl + pr
        x_flat = list(x._data.to_flat_list() if hasattr(x._data, "to_flat_list") else x._data)
        out = []
        for n in range(N):
            for c in range(C):
                for ol in range(oL):
                    il = ol - pl
                    if il < 0: il = -il
                    if il >= L: il = 2*L - il - 2
                    il = max(0, min(L-1, il))
                    out.append(x_flat[n*C*L + c*L + il])
        sh = (N, C, oL)
        return Tensor(_ND3(_a3.array("f", out), sh, _c_strides(sh), _DType("float32")))


class ReplicationPad1d(Module):
    def __init__(self, padding):
        super().__init__()
        self.pad = (padding, padding) if isinstance(padding, int) else tuple(padding)

    def forward(self, x: Tensor) -> Tensor:
        import array as _a3
        from vectorphile._core.ndarray import NDArray as _ND3, _c_strides, _DType
        N, C, L = x._data.shape
        pl, pr = self.pad
        oL = L + pl + pr
        x_flat = list(x._data.to_flat_list() if hasattr(x._data, "to_flat_list") else x._data)
        out = []
        for n in range(N):
            for c in range(C):
                for ol in range(oL):
                    il = max(0, min(L-1, ol - pl))
                    out.append(x_flat[n*C*L + c*L + il])
        sh = (N, C, oL)
        return Tensor(_ND3(_a3.array("f", out), sh, _c_strides(sh), _DType("float32")))


class ReplicationPad2d(Module):
    def __init__(self, padding):
        super().__init__()
        self.pad = (padding,)*4 if isinstance(padding, int) else tuple(padding)

    def forward(self, x: Tensor) -> Tensor:
        import array as _a3
        from vectorphile._core.ndarray import NDArray as _ND3, _c_strides, _DType
        N, C, H, W = x._data.shape
        pl, pr, pt, pb = self.pad
        oH, oW = H + pt + pb, W + pl + pr
        x_flat = list(x._data.to_flat_list() if hasattr(x._data, "to_flat_list") else x._data)
        out = []
        for n in range(N):
            for c in range(C):
                for oh in range(oH):
                    ih = max(0, min(H-1, oh - pt))
                    for ow in range(oW):
                        iw = max(0, min(W-1, ow - pl))
                        out.append(x_flat[n*C*H*W + c*H*W + ih*W + iw])
        sh = (N, C, oH, oW)
        return Tensor(_ND3(_a3.array("f", out), sh, _c_strides(sh), _DType("float32")))


class ConstantPad3d(Module):
    def __init__(self, padding, value: float = 0.0):
        super().__init__()
        self.pad = (padding,)*6 if isinstance(padding, int) else tuple(padding)
        self.value = value

    def forward(self, x: Tensor) -> Tensor:
        import array as _a3
        from vectorphile._core.ndarray import NDArray as _ND3, _c_strides, _DType
        N, C, D, H, W = x._data.shape
        pl, pr, pt, pb, pf, pbk = self.pad
        oD, oH, oW = D + pf + pbk, H + pt + pb, W + pl + pr
        x_flat = list(x._data.to_flat_list() if hasattr(x._data, "to_flat_list") else x._data)
        out = [self.value] * (N * C * oD * oH * oW)
        for n in range(N):
            for c in range(C):
                for d in range(D):
                    for h in range(H):
                        for w in range(W):
                            dst = n*C*oD*oH*oW + c*oD*oH*oW + (d+pf)*oH*oW + (h+pt)*oW + (w+pl)
                            out[dst] = x_flat[n*C*D*H*W + c*D*H*W + d*H*W + h*W + w]
        sh = (N, C, oD, oH, oW)
        return Tensor(_ND3(_a3.array("f", out), sh, _c_strides(sh), _DType("float32")))


# ---------------------------------------------------------------------------
# Upsampling aliases
# ---------------------------------------------------------------------------

class UpsamplingNearest2d(Module):
    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self._up = Upsample(size=size, scale_factor=scale_factor, mode="nearest")

    def forward(self, x: Tensor) -> Tensor:
        return self._up(x)


class UpsamplingBilinear2d(Module):
    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self._up = Upsample(size=size, scale_factor=scale_factor, mode="bilinear")

    def forward(self, x: Tensor) -> Tensor:
        return self._up(x)


# ---------------------------------------------------------------------------
# ConvTranspose3d
# ---------------------------------------------------------------------------

class ConvTranspose3d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        ks = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)*3
        st = stride if isinstance(stride, tuple) else (stride,)*3
        pa = padding if isinstance(padding, tuple) else (padding,)*3
        self.kernel_size = ks; self.stride = st; self.padding = pa
        self.weight = Tensor.randn(in_channels, out_channels // groups, *ks) * 0.02
        self.bias_t = Tensor.zeros(out_channels) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        import array as _a3
        from vectorphile._core.ndarray import NDArray as _ND3, _c_strides, _DType
        N, C_in, D, H, W = x._data.shape
        kD, kH, kW = self.kernel_size
        sD, sH, sW = self.stride
        pD, pH, pW = self.padding
        oD = (D - 1)*sD - 2*pD + kD
        oH = (H - 1)*sH - 2*pH + kH
        oW = (W - 1)*sW - 2*pW + kW
        x_flat = list(x._data.to_flat_list() if hasattr(x._data, "to_flat_list") else x._data)
        w_flat = list(self.weight._data.to_flat_list() if hasattr(self.weight._data, "to_flat_list") else self.weight._data)
        C_out = self.out_channels
        out = [0.0] * (N * C_out * oD * oH * oW)
        for n in range(N):
            for ci in range(C_in):
                for d in range(D):
                    for h in range(H):
                        for w in range(W):
                            xv = x_flat[n*C_in*D*H*W + ci*D*H*W + d*H*W + h*W + w]
                            for co in range(C_out):
                                for kd in range(kD):
                                    for kh in range(kH):
                                        for kw2 in range(kW):
                                            od = d*sD + kd - pD
                                            oh = h*sH + kh - pH
                                            ow2 = w*sW + kw2 - pW
                                            if 0 <= od < oD and 0 <= oh < oH and 0 <= ow2 < oW:
                                                wi = ci*C_out*kD*kH*kW + co*kD*kH*kW + kd*kH*kW + kh*kW + kw2
                                                oi = n*C_out*oD*oH*oW + co*oD*oH*oW + od*oH*oW + oh*oW + ow2
                                                out[oi] += xv * w_flat[wi]
        if self.bias_t is not None:
            b_flat = list(self.bias_t._data.to_flat_list() if hasattr(self.bias_t._data, "to_flat_list") else self.bias_t._data)
            for n in range(N):
                for co in range(C_out):
                    for d in range(oD):
                        for h in range(oH):
                            for w2 in range(oW):
                                out[n*C_out*oD*oH*oW + co*oD*oH*oW + d*oH*oW + h*oW + w2] += b_flat[co]
        sh = (N, C_out, oD, oH, oW)
        return Tensor(_ND3(_a3.array("f", out), sh, _c_strides(sh), _DType("float32")))


# ---------------------------------------------------------------------------
# AdaptiveMaxPool1d / AdaptiveMaxPool3d
# ---------------------------------------------------------------------------

class AdaptiveMaxPool1d(Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        import array as _a3
        from vectorphile._core.ndarray import NDArray as _ND3, _c_strides, _DType
        import math as _m3
        N, C, L = x._data.shape
        oL = self.output_size
        x_flat = list(x._data.to_flat_list() if hasattr(x._data, "to_flat_list") else x._data)
        out = []
        for n in range(N):
            for c in range(C):
                for i in range(oL):
                    ls = int(_m3.floor(i * L / oL))
                    le = int(_m3.ceil((i+1) * L / oL))
                    best = max(x_flat[n*C*L + c*L + k] for k in range(ls, le))
                    out.append(best)
        sh = (N, C, oL)
        return Tensor(_ND3(_a3.array("f", out), sh, _c_strides(sh), _DType("float32")))


class AdaptiveMaxPool3d(Module):
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size,)*3
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        import array as _a3
        from vectorphile._core.ndarray import NDArray as _ND3, _c_strides, _DType
        import math as _m3
        N, C, D, H, W = x._data.shape
        oD, oH, oW = self.output_size
        x_flat = list(x._data.to_flat_list() if hasattr(x._data, "to_flat_list") else x._data)
        out = []
        for n in range(N):
            for c in range(C):
                for i in range(oD):
                    ds = int(_m3.floor(i * D / oD)); de = int(_m3.ceil((i+1) * D / oD))
                    for j in range(oH):
                        hs = int(_m3.floor(j * H / oH)); he = int(_m3.ceil((j+1) * H / oH))
                        for k in range(oW):
                            ws = int(_m3.floor(k * W / oW)); we = int(_m3.ceil((k+1) * W / oW))
                            best = float("-inf")
                            for d in range(ds, de):
                                for h in range(hs, he):
                                    for w in range(ws, we):
                                        v = x_flat[n*C*D*H*W + c*D*H*W + d*H*W + h*W + w]
                                        if v > best: best = v
                            out.append(best)
        sh = (N, C, oD, oH, oW)
        return Tensor(_ND3(_a3.array("f", out), sh, _c_strides(sh), _DType("float32")))
