"""Reverse-mode automatic differentiation engine."""
from __future__ import annotations
import array as _arr_eng
import math as _math_eng
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorphile.core.tensor import Tensor

try:
    from vectorphile._core import _vp_core as _C_eng
    from vectorphile._core.ndarray import NDArray as _NDA_eng, _c_strides as _cs_eng, _DType as _DType_eng
    _ENG_C_OK = True
    _ENG_ISPC_OK = hasattr(_C_eng, 'ispc_relu_backward_f32')
except ImportError:
    # Fallback to Mojo backend on Linux
    try:
        from vectorphile._core import mojo_backend as _C_eng  # type: ignore[assignment]
        from vectorphile._core.ndarray import NDArray as _NDA_eng, _c_strides as _cs_eng, _DType as _DType_eng
        _ENG_C_OK = _C_eng.is_available()
        _ENG_ISPC_OK = False
    except ImportError:
        _C_eng = None
        _ENG_C_OK = False
        _ENG_ISPC_OK = False


def _topo_sort(root: 'Tensor'):
    visited, order = set(), []
    def visit(node):
        if id(node) not in visited:
            visited.add(id(node))
            if node.grad_fn is not None:
                for inp in node.grad_fn.inputs:
                    if hasattr(inp, 'grad_fn'):
                        visit(inp)
            order.append(node)
    visit(root)
    return reversed(order)


def backward(root: 'Tensor', gradient: 'Tensor'):
    from vectorphile.core.tensor import Tensor
    from vectorphile._core.ndarray import NDArray as _NDA, zeros as _nd_zeros
    root.grad = gradient
    for node in _topo_sort(root):
        if node.grad_fn is None:
            continue
        grads = node.grad_fn.backward(node.grad)
        # Free intermediate gradient unless leaf or explicitly retained
        if not node._is_leaf and not getattr(node, '_retain_grad', False):
            node.grad = None
        for inp, g in zip(node.grad_fn.inputs, grads):
            if not isinstance(inp, Tensor) or not inp.requires_grad:
                continue
            if g is None:
                continue
            from vectorphile._core.ndarray import ComplexNDArray as _CNDA
            g_data = g._data if isinstance(g, Tensor) else g
            # For complex grads, extract real part (Wirtinger: d/dz* of real loss = conj(df/dz))
            if isinstance(g_data, _CNDA) and not isinstance(inp._data, _CNDA):
                g_data = g_data.real
            # Sum gradient over broadcast dimensions (preserves type)
            g_data = _sum_to_shape(g_data, inp._data.shape)
            from vectorphile._core.ndarray import ComplexNDArray as _CNDA2
            if inp.grad is None:
                # Initialize grad with matching type to param
                if isinstance(inp._data, _NDA):
                    _dt_grad = str(inp._data.dtype) if hasattr(inp._data, 'dtype') else 'float32'
                    inp.grad = Tensor(_nd_zeros(inp._data.shape, dtype=_dt_grad))
                elif isinstance(inp._data, _CNDA2):
                    inp.grad = Tensor(_CNDA2(_nd_zeros(inp._data.shape, dtype='float32'), _nd_zeros(inp._data.shape, dtype='float32')))
                else:
                    n = 1
                    for s in (inp._data.shape if hasattr(inp._data, 'shape') else (len(inp._data),)): n *= s
                    inp.grad = Tensor(_nd_zeros((n,), dtype='float32'))
            # Accumulate — keep NDArray if param is NDArray
            if isinstance(inp._data, _NDA) and isinstance(g_data, _NDA):
                inp.grad._data += g_data
            elif isinstance(inp._data, _NDA) and not isinstance(g_data, _NDA):
                import array as _a
                from vectorphile._core.ndarray import _c_strides as _cs, _DType
                if hasattr(g_data, 'to_flat_list'):
                    g_flat = list(g_data.to_flat_list())
                elif hasattr(g_data, 'tolist'):
                    v = g_data.tolist()
                    g_flat = [item for row in v for item in row] if v and isinstance(v[0], list) else list(v)
                else:
                    g_flat = [float(x) for x in g_data] if hasattr(g_data, '__iter__') else [float(g_data)]
                g_buf = _a.array('f', [float(v) for v in g_flat])
                g_nda = _NDA(g_buf, inp._data.shape, _cs(inp._data.shape), _DType('float32'))
                inp.grad._data += g_nda
            else:
                inp.grad._data += g_data


def _sum_to_shape(g, shape):
    """Sum gradient g down to target shape, preserving NDArray type when possible."""
    from vectorphile._core.ndarray import NDArray as _NDA
    if isinstance(g, _NDA):
        while g.ndim > len(shape):
            g = g.sum(axis=0)
        for i, (gs, s) in enumerate(zip(g.shape, shape)):
            if s == 1 and gs != 1:
                g = g.sum(axis=i, keepdims=True)
        return g
    # numpy path
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, (gs, s) in enumerate(zip(g.shape, shape)):
        if s == 1 and gs != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class _FunctionCtx:
    """Context object passed to static forward/backward in PyTorch-compat style."""
    def __init__(self):
        self.saved = ()
        self.needs_input_grad = ()

    def save_for_backward(self, *tensors):
        self.saved = tensors

    @property
    def saved_tensors(self):
        return self.saved


class _CustomBackward:
    """Autograd node created by Function.apply()."""
    def __init__(self, ctx, backward_fn, inputs):
        self.ctx = ctx
        self._backward_fn = backward_fn
        self.inputs = inputs
        self.needs_input_grad = tuple(getattr(t, 'requires_grad', False) for t in inputs)

    def backward(self, grad):
        return self._backward_fn(self.ctx, grad)


class Function:
    """Base class for custom autograd functions.

    Subclass with static forward() and backward() for PyTorch-compatible
    custom ops:

        class MyOp(vp.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x, y)
                return x + y

            @staticmethod
            def backward(ctx, grad):
                x, y = ctx.saved_tensors
                return grad, grad

        result = MyOp.apply(a, b)
    """

    def __init__(self, *inputs):
        self.inputs = inputs

    def backward(self, grad):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args):
        """Call forward with a fresh ctx, wire backward into the autograd graph."""
        from vectorphile.core.tensor import Tensor as _T
        ctx = _FunctionCtx()
        ctx.needs_input_grad = tuple(getattr(a, 'requires_grad', False) for a in args)
        result = cls.forward(ctx, *args)
        if not isinstance(result, _T):
            result = _T(result)
        requires_grad = any(getattr(a, 'requires_grad', False) for a in args)
        if requires_grad:
            result.requires_grad = True
            result._is_leaf = False
            result.grad_fn = _CustomBackward(ctx, cls.backward, args)
        return result

    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class AddBackward(Function):
    def backward(self, grad):
        return grad, grad


class SubBackward(Function):
    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        neg = Tensor(-grad._data, device=grad.device)
        return grad, neg


class MulBackward(Function):
    def __init__(self, a, b):
        super().__init__(a, b)
        self.a, self.b = a, b

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        a_data = self.a._data if hasattr(self.a, '_data') else self.a
        b_data = self.b._data if hasattr(self.b, '_data') else self.b
        return Tensor(grad._data * b_data), Tensor(grad._data * a_data)


class DivBackward(Function):
    def __init__(self, a, b):
        super().__init__(a, b)
        self.a, self.b = a, b

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        a_data = self.a._data if hasattr(self.a, '_data') else self.a
        b_data = self.b._data if hasattr(self.b, '_data') else self.b
        da = grad._data / b_data
        db = -grad._data * a_data / (b_data ** 2)
        return Tensor(da), Tensor(db)


class MatMulBackward(Function):
    def __init__(self, a, b):
        super().__init__(a, b)
        self.a, self.b = a, b

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        gd = grad._data; ad = self.a._data; bd = self.b._data
        # Fast path: use C kernel matmul with explicit transposes (no numpy)
        if (_ENG_C_OK
                and isinstance(gd, _NDA_eng) and gd._buf is not None
                and isinstance(ad, _NDA_eng) and ad._buf is not None
                and isinstance(bd, _NDA_eng) and bd._buf is not None
                and str(getattr(gd,'dtype','')) == 'float32'
                and str(getattr(ad,'dtype','')) == 'float32'
                and str(getattr(bd,'dtype','')) == 'float32'
                and gd.ndim == 2 and ad.ndim == 2 and bd.ndim == 2):
            M, K2 = gd.shape; K1, N = bd.shape  # grad=(M,K2), b=(K1,N), expect K2==N
            # da = grad @ b.T → (M,K2) @ (N,K1) = (M,K1)
            bT_buf = _arr_eng.array('f', bytearray(4 * K1 * N))
            _C_eng.transpose_f32(bd._buf, bT_buf, K1, N)
            da_buf = _arr_eng.array('f', bytearray(4 * M * K1))
            _C_eng.matmul_f32(gd._buf, bT_buf, da_buf, M, K2, K1)
            da = _NDA_eng(da_buf, (M, K1), _cs_eng((M, K1)), _DType_eng('float32'))
            # db = a.T @ grad → (K1,M) @ (M,K2) = (K1,K2)
            aT_buf = _arr_eng.array('f', bytearray(4 * K1 * M))
            _C_eng.transpose_f32(ad._buf, aT_buf, M, K1)
            db_buf = _arr_eng.array('f', bytearray(4 * K1 * K2))
            _C_eng.matmul_f32(aT_buf, gd._buf, db_buf, K1, M, K2)
            db = _NDA_eng(db_buf, (K1, K2), _cs_eng((K1, K2)), _DType_eng('float32'))
            return Tensor(da, _no_copy=True), Tensor(db, _no_copy=True)
        # Pure NDArray fallback — 2D only
        def _to_nda(d):
            from vectorphile._core.ndarray import array as _nd_arr
            if isinstance(d, _NDA_eng):
                return d
            _dt = str(getattr(d, 'dtype', 'float64'))
            if hasattr(d, 'to_flat_list'):
                flat = list(d.to_flat_list())
            elif hasattr(d, 'tolist'):
                v = d.tolist()
                flat = [item for row in v for item in row] if v and isinstance(v[0], list) else list(v)
            else:
                flat = [float(d)]
            sh = d.shape if hasattr(d, 'shape') else (len(flat),)
            return _nd_arr(flat, dtype=_dt).reshape(sh)
        g_nda = _to_nda(gd); a_nda = _to_nda(ad); b_nda = _to_nda(bd)
        # For batched matmul (ndim >= 3), transpose only last two axes
        def _mat_transpose(x):
            ndim = len(x.shape)
            if ndim < 2: return x
            axes = list(range(ndim))
            axes[-2], axes[-1] = axes[-1], axes[-2]
            return x.transpose(axes)
        da = g_nda @ _mat_transpose(b_nda)
        # For batched case (a is ND, b is 2D), sum db over batch dims
        if g_nda.ndim > 2 and b_nda.ndim == 2:
            # db = a.reshape(-1, K).T @ g.reshape(-1, N)
            K = a_nda.shape[-1]; N = g_nda.shape[-1]
            BM = 1
            for s in a_nda.shape[:-1]: BM *= s
            db = _mat_transpose(a_nda.reshape((BM, K))) @ g_nda.reshape((BM, N))
        elif g_nda.ndim > 2 and a_nda.ndim == 2:
            # a is 2D, g is ND: db = sum over batch of a.T @ g[b]
            K = a_nda.shape[-1]; N = g_nda.shape[-1]
            BM = 1
            for s in g_nda.shape[:-1]: BM *= s
            db = _mat_transpose(g_nda.reshape((BM, N))) @ a_nda.reshape((BM, K))
        else:
            db = _mat_transpose(a_nda) @ g_nda
        return Tensor(da), Tensor(db)


def _c_bwd2(fn_name, go_data, io_data, shape):
    """Call a 2-input C backward kernel (go, inp_or_out → gi) if both are NDArray."""
    if (_ENG_C_OK and isinstance(go_data, _NDA_eng) and go_data._buf is not None
            and isinstance(io_data, _NDA_eng) and io_data._buf is not None
            and str(go_data.dtype) == 'float32' and str(io_data.dtype) == 'float32'):
        n = 1
        for s in go_data.shape: n *= s
        out_buf = _arr_eng.array('f', bytearray(4 * n))
        getattr(_C_eng, fn_name)(go_data._buf, io_data._buf, out_buf)
        return _NDA_eng(out_buf, go_data.shape, _cs_eng(go_data.shape), _DType_eng('float32'))
    return None


class ExpBackward(Function):
    def __init__(self, inp, out):
        super().__init__(inp)
        self.out = out

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        result = _c_bwd2('exp_backward_f32', grad._data, self.out._data, grad._data.shape)
        if result is not None:
            return (Tensor(result),)
        return (Tensor(grad._data * self.out._data),)


class LogBackward(Function):
    def __init__(self, inp):
        super().__init__(inp)
        self.inp = inp

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        result = _c_bwd2('log_backward_f32', grad._data, self.inp._data, grad._data.shape)
        if result is not None:
            return (Tensor(result),)
        return (Tensor(grad._data / (self.inp._data + 1e-12)),)


class SqrtBackward(Function):
    def __init__(self, inp, out):
        super().__init__(inp)
        self.out = out

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        result = _c_bwd2('sqrt_backward_f32', grad._data, self.out._data, grad._data.shape)
        if result is not None:
            return (Tensor(result),)
        return (Tensor(0.5 * grad._data / (self.out._data + 1e-12)),)


def _nd_broadcast_to(g, target_shape):
    """Broadcast NDArray g to target_shape by repeating along size-1 dims."""
    if isinstance(g, _NDA_eng) and g._buf is not None and str(getattr(g, 'dtype', 'float32')) == 'float32':
        # Insert missing leading dims
        while g.ndim < len(target_shape):
            new_shape = (1,) + g.shape
            g = _NDA_eng(g._buf, new_shape, _cs_eng(new_shape), _DType_eng('float32'))
        # Repeat along each dim where g has 1 but target has > 1
        for i, (gs, ts) in enumerate(zip(g.shape, target_shape)):
            if gs == 1 and ts > 1:
                prefix = 1
                for s in g.shape[:i]: prefix *= s
                suffix = 1
                for s in g.shape[i+1:]: suffix *= s
                new_shape = g.shape[:i] + (ts,) + g.shape[i+1:]
                n_out = prefix * ts * suffix
                out_buf = _arr_eng.array('f', bytearray(4 * n_out))
                if suffix == 1 and _ENG_C_OK:
                    # (prefix, 1) @ ones(1, ts) → (prefix, ts): fast path
                    ones_buf = _arr_eng.array('f', [1.0] * ts)
                    _C_eng.matmul_f32(g._buf, ones_buf, out_buf, prefix, 1, ts)
                else:
                    # General: out[a, t, b] = g[a, 0, b] for t in range(ts)
                    for a in range(prefix):
                        for b in range(suffix):
                            v = g._buf[a * suffix + b]
                            for t in range(ts):
                                out_buf[(a * ts + t) * suffix + b] = v
                g = _NDA_eng(out_buf, new_shape, _cs_eng(new_shape), _DType_eng('float32'))
        return g
    # Pure Python fallback for non-NDArray or non-float32
    _dt_bc = str(getattr(g, 'dtype', 'float64')) if hasattr(g, 'dtype') else 'float64'
    if hasattr(g, 'to_flat_list'):
        flat = list(g.to_flat_list())
        g_shape = g.shape
    elif hasattr(g, 'tolist'):
        v = g.tolist()
        flat = [item for row in v for item in row] if v and isinstance(v[0], list) else list(v)
        g_shape = g.shape if hasattr(g, 'shape') else (len(flat),)
    else:
        flat = [float(g)]
        g_shape = ()
    n = 1
    for s in target_shape: n *= s
    if len(flat) == 1:
        result_flat = [flat[0]] * n
    else:
        result_flat = flat
    from vectorphile._core.ndarray import array as _nd_bc
    return _nd_bc(result_flat, dtype=_dt_bc).reshape(target_shape) if len(result_flat) == n else _nd_bc(flat, dtype=_dt_bc).reshape(g_shape)


class SumBackward(Function):
    def __init__(self, inp, dim, keepdim):
        super().__init__(inp)
        self.inp = inp
        self.dim = dim
        self.keepdim = keepdim

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        in_shape = self.inp._data.shape if hasattr(self.inp._data, 'shape') else (len(self.inp._data),)
        # Use NDArray expand/broadcast
        g = grad._data
        if isinstance(g, _NDA_eng):
            if self.dim is not None and not self.keepdim:
                dims = (self.dim,) if isinstance(self.dim, int) else tuple(sorted(self.dim))
                for d in dims:
                    g = g.unsqueeze(d) if hasattr(g, 'unsqueeze') else g
            return (_nd_broadcast_to(g, in_shape),) if g is not None else (Tensor(_NDA_eng),)
        return (Tensor(_nd_broadcast_to(g, in_shape)),)


class MeanBackward(Function):
    def __init__(self, inp, dim, keepdim):
        super().__init__(inp)
        self.inp = inp
        self.dim = dim
        self.keepdim = keepdim

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        in_shape = self.inp._data.shape if hasattr(self.inp._data, 'shape') else (len(self.inp._data),)
        if self.dim is None:
            n = 1
            for s in in_shape: n *= s
        elif isinstance(self.dim, (tuple, list)):
            n = 1
            for d in self.dim: n *= in_shape[d]
        else:
            n = in_shape[self.dim]
        g = grad._data
        from vectorphile._core.ndarray import array as _nd_arr
        g_flat = g.to_flat_list() if hasattr(g, 'to_flat_list') else [float(g)]
        _dt_mean = str(self.inp._data.dtype) if hasattr(self.inp._data, 'dtype') else 'float32'
        g_scaled = _nd_arr([v / n for v in g_flat], dtype=_dt_mean).reshape(g.shape if hasattr(g, 'shape') else (1,))
        # Re-insert reduced dims as size-1 so broadcasting aligns correctly
        if self.dim is not None and not self.keepdim:
            dims = (self.dim,) if isinstance(self.dim, int) else tuple(sorted(self.dim))
            for d in dims:
                g_scaled = g_scaled.unsqueeze(d) if hasattr(g_scaled, 'unsqueeze') else g_scaled
        return (Tensor(_nd_broadcast_to(g_scaled, in_shape)),)


class ReLUBackward(Function):
    def __init__(self, inp):
        super().__init__(inp)
        self.inp = inp

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor, _relu_buf_cache
        gd = grad._data; xd = self.inp._data
        if (_ENG_C_OK and isinstance(xd, _NDA_eng) and xd._buf is not None
                and isinstance(gd, _NDA_eng) and gd._buf is not None
                and str(xd.dtype) == 'float32'):
            n = 1
            for s in xd.shape: n *= s
            _key = -n
            if _key not in _relu_buf_cache:
                _relu_buf_cache[_key] = _arr_eng.array('f', bytearray(4 * n))
            out_buf = _relu_buf_cache[_key]
            fn = _C_eng.ispc_relu_backward_f32 if _ENG_ISPC_OK else _C_eng.relu_backward_f32
            fn(gd._buf, xd._buf, out_buf)
            return (Tensor(_NDA_eng(out_buf, xd.shape, _cs_eng(xd.shape), _DType_eng('float32'))),)
        from vectorphile._core.ndarray import array as _nd_arr
        if hasattr(xd, 'to_flat_list'):
            xf = list(xd.to_flat_list()); gf = list(gd.to_flat_list()) if hasattr(gd, 'to_flat_list') else [float(gd)]
        else:
            xf = [float(v) for v in xd] if hasattr(xd, '__iter__') else [float(xd)]
            gf = [float(v) for v in gd] if hasattr(gd, '__iter__') else [float(gd)]
        result = [gf[i] if xf[i] > 0 else 0.0 for i in range(len(xf))]
        sh = xd.shape if hasattr(xd, 'shape') else (len(result),)
        return (Tensor(_nd_arr(result).reshape(sh)),)


class SigmoidBackward(Function):
    def __init__(self, inp, out):
        super().__init__(inp)
        self.out = out

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        result = _c_bwd2('sigmoid_backward_f32', grad._data, self.out._data, grad._data.shape)
        if result is not None:
            return (Tensor(result),)
        s = self.out._data
        return (Tensor(grad._data * s * (1 - s)),)


class TanhBackward(Function):
    def __init__(self, inp, out):
        super().__init__(inp)
        self.out = out

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        result = _c_bwd2('tanh_backward_f32', grad._data, self.out._data, grad._data.shape)
        if result is not None:
            return (Tensor(result),)
        return (Tensor(grad._data * (1 - self.out._data ** 2)),)


class SinBackward(Function):
    def __init__(self, inp):
        super().__init__(inp)
        self.inp = inp

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        result = _c_bwd2('sin_backward_f32', grad._data, self.inp._data, grad._data.shape)
        if result is not None:
            return (Tensor(result),)
        from vectorphile._core.ndarray import array as _nd_arr
        xf = list(self.inp._data.to_flat_list()) if hasattr(self.inp._data, 'to_flat_list') else [float(self.inp._data)]
        gf = list(grad._data.to_flat_list()) if hasattr(grad._data, 'to_flat_list') else [float(grad._data)]
        sh = self.inp._data.shape if hasattr(self.inp._data, 'shape') else (len(xf),)
        return (Tensor(_nd_arr([gf[i] * _math_eng.cos(xf[i]) for i in range(len(xf))]).reshape(sh)),)


class CosBackward(Function):
    def __init__(self, inp):
        super().__init__(inp)
        self.inp = inp

    def backward(self, grad):
        from vectorphile.core.tensor import Tensor
        result = _c_bwd2('cos_backward_f32', grad._data, self.inp._data, grad._data.shape)
        if result is not None:
            return (Tensor(result),)
        from vectorphile._core.ndarray import array as _nd_arr
        xf = list(self.inp._data.to_flat_list()) if hasattr(self.inp._data, 'to_flat_list') else [float(self.inp._data)]
        gf = list(grad._data.to_flat_list()) if hasattr(grad._data, 'to_flat_list') else [float(grad._data)]
        sh = self.inp._data.shape if hasattr(self.inp._data, 'shape') else (len(xf),)
        return (Tensor(_nd_arr([-gf[i] * _math_eng.sin(xf[i]) for i in range(len(xf))]).reshape(sh)),)


class BatchNormBackward(Function):
    """Fused BatchNorm1d backward via C kernel."""
    def __init__(self, inp, x_hat, rstd, gamma, has_affine):
        super().__init__(inp)
        self.x_hat = x_hat
        self.rstd = rstd
        self.gamma = gamma
        self.has_affine = has_affine

    def backward(self, grad):
        import array as _arr
        from vectorphile._core import _vp_core as _C
        from vectorphile._core.ndarray import NDArray as _NDA, _c_strides as _cs, _DType
        from vectorphile.core.tensor import Tensor
        gd = grad._data
        xh = self.x_hat
        rs = self.rstd
        N, C = gd.shape
        from vectorphile._core.ndarray import NDArray as _NDA_bn, _c_strides as _cs_bn, _DType as _DType_bn, array as _nd_arr_bn
        def _get_buf(nd):
            if hasattr(nd, '_buf') and nd._buf is not None:
                return nd._buf
            if hasattr(nd, 'to_flat_list'):
                return _arr.array('f', [float(v) for v in nd.to_flat_list()])
            if hasattr(nd, 'tolist'):
                v = nd.tolist()
                flat = [item for row in v for item in row] if v and isinstance(v[0], list) else list(v)
                return _arr.array('f', [float(x) for x in flat])
            return _arr.array('f', [float(nd)])
        dy_buf   = _get_buf(gd)
        xhat_buf = _get_buf(xh)
        rstd_buf = _get_buf(rs)
        gamma_buf = _get_buf(self.gamma) if self.gamma is not None else None
        dx_buf   = _arr.array('f', bytearray(4 * N * C))
        dg_buf   = _arr.array('f', bytearray(4 * C))
        db_buf   = _arr.array('f', bytearray(4 * C))
        _C.batch_norm_backward_f32(dy_buf, xhat_buf, rstd_buf, gamma_buf,
                                   dx_buf, dg_buf, db_buf, N, C)
        dx_nda = _NDA_bn(dx_buf, (N, C), _cs_bn((N, C)), _DType_bn('float32'))
        dx = Tensor(dx_nda)
        if self.has_affine:
            dg_nda = _NDA_bn(dg_buf, (C,), _cs_bn((C,)), _DType_bn('float32'))
            db_nda = _NDA_bn(db_buf, (C,), _cs_bn((C,)), _DType_bn('float32'))
            return (dx, Tensor(dg_nda), Tensor(db_nda))
        return (dx,)


class no_grad:
    """Context manager to disable gradient tracking."""
    _enabled = True

    def __enter__(self):
        no_grad._enabled = False

    def __exit__(self, *args):
        no_grad._enabled = True


class enable_grad:
    """Context manager to re-enable gradient tracking inside a no_grad block."""
    def __enter__(self):
        self._prev = no_grad._enabled
        no_grad._enabled = True

    def __exit__(self, *args):
        no_grad._enabled = self._prev
