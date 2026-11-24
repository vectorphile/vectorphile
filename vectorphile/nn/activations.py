"""Activation function modules — heavy ops dispatched through vectorphile.backend."""
import math
import array as _arr
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor, _relu_buf_cache
import vectorphile.backend as _backend

try:
    from vectorphile._core import _vp_core as _C_act
    from vectorphile._core.ndarray import NDArray as _NDA_act, _c_strides as _cs_act, _DType as _DType_act
    _ACT_C_OK = True
    _ACT_ISPC_OK = hasattr(_C_act, 'ispc_sigmoid_f32')
except ImportError:
    _ACT_C_OK = False
    _ACT_ISPC_OK = False

_act_buf_cache: dict = {}

def _ispc_unary(fn, x: Tensor) -> Tensor:
    xd = x._data
    if not (isinstance(xd, _NDA_act) and xd._buf is not None and xd.dtype == 'float32'):
        return None
    n = 1
    for s in xd.shape: n *= s
    out_buf = _arr.array('f', bytearray(4 * n))
    fn(xd._buf, out_buf)
    result = Tensor(_NDA_act(out_buf, xd.shape, _cs_act(xd.shape), _DType_act('float32')),
                    device=x.device, requires_grad=x.requires_grad, _no_copy=True)
    return result


def _get_flat(x):
    if hasattr(x, 'to_flat_list'): return x.to_flat_list()
    if hasattr(x, 'tolist'): return x.tolist()
    if isinstance(x, list): return x
    return list(x)


class ReLU(Module):
    def __init__(self, inplace=False):
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()
    def __repr__(self): return 'ReLU()'


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope
    def forward(self, x: Tensor) -> Tensor:
        flat = _get_flat(x._data)
        ns = self.negative_slope
        out = [v if v >= 0 else ns * v for v in flat]
        rg = x.requires_grad
        sh = x._data.shape if hasattr(x._data, 'shape') else (len(flat),)
        _dt_lr = str(getattr(x._data, 'dtype', 'float32'))
        if _dt_lr not in ('float32', 'float64'): _dt_lr = 'float32'
        from vectorphile._core.ndarray import array as _nd
        result = Tensor(_nd(out, dtype=_dt_lr).reshape(sh), device=x.device, requires_grad=rg)
        if rg:
            _inp = x
            class _LReLUBwd:
                def __init__(s): s.inputs = (_inp,)
                def backward(s, grad):
                    gf = _get_flat(grad._data)
                    mask = [1.0 if v >= 0 else ns for v in flat]
                    return (Tensor(_nd([gf[i]*mask[i] for i in range(len(gf))], dtype=_dt_lr).reshape(sh)),)
            result.grad_fn = _LReLUBwd(); result._is_leaf = False
        return result
    def __repr__(self): return f'LeakyReLU(negative_slope={self.negative_slope})'


class ELU(Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x: Tensor) -> Tensor:
        flat = _get_flat(x._data)
        alpha = self.alpha
        out = [v if v >= 0 else alpha * (math.exp(v) - 1) for v in flat]
        rg = x.requires_grad
        sh = x._data.shape if hasattr(x._data, 'shape') else (len(flat),)
        from vectorphile._core.ndarray import array as _nd
        result = Tensor(_nd(out).reshape(sh), device=x.device, requires_grad=rg)
        if rg:
            _inp = x
            class _ELUBwd:
                def __init__(s): s.inputs = (_inp,)
                def backward(s, grad):
                    gf = _get_flat(grad._data)
                    mask = [1.0 if v >= 0 else alpha * math.exp(v) for v in flat]
                    return (Tensor(_nd([gf[i]*mask[i] for i in range(len(gf))]).reshape(sh)),)
            result.grad_fn = _ELUBwd(); result._is_leaf = False
        return result
    def __repr__(self): return f'ELU(alpha={self.alpha})'


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        from vectorphile.nn import functional as F
        return F.gelu(x)
    def __repr__(self): return 'GELU()'


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()
    def __repr__(self): return 'Sigmoid()'


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()
    def __repr__(self): return 'Tanh()'


class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, x: Tensor) -> Tensor:
        xd = x._data
        if (_ACT_ISPC_OK and isinstance(xd, _NDA_act) and xd._buf is not None
                and xd.dtype == 'float32' and xd.ndim == 2 and self.dim in (-1, 1)):
            N, C = xd.shape
            _key = (id(_C_act.ispc_softmax_rows_f32), N * C)
            if _key not in _act_buf_cache:
                _act_buf_cache[_key] = _arr.array('f', bytearray(4 * N * C))
            out_buf = _act_buf_cache[_key]
            _C_act.ispc_softmax_rows_f32(xd._buf, out_buf, N, C)
            return Tensor(_NDA_act(out_buf, xd.shape, _cs_act(xd.shape), _DType_act('float32')),
                          device=x.device, _no_copy=True)
        return Tensor(_backend.softmax(x._data, self.dim), device=x.device)
    def __repr__(self): return f'Softmax(dim={self.dim})'


class LogSoftmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, x: Tensor) -> Tensor:
        xd = x._data
        if (_ACT_ISPC_OK and isinstance(xd, _NDA_act) and xd._buf is not None
                and xd.dtype == 'float32' and xd.ndim == 2 and self.dim in (-1, 1)):
            N, C = xd.shape
            _key = (id(_C_act.ispc_log_softmax_rows_f32), N * C)
            if _key not in _act_buf_cache:
                _act_buf_cache[_key] = _arr.array('f', bytearray(4 * N * C))
            out_buf = _act_buf_cache[_key]
            _C_act.ispc_log_softmax_rows_f32(xd._buf, out_buf, N, C)
            return Tensor(_NDA_act(out_buf, xd.shape, _cs_act(xd.shape), _DType_act('float32')),
                          device=x.device, _no_copy=True)
        from vectorphile.core.tensor import log_softmax
        return log_softmax(x, self.dim)
    def __repr__(self): return f'LogSoftmax(dim={self.dim})'


class SELU(Module):
    _alpha = 1.6732632423543772
    _scale = 1.0507009873554805
    def forward(self, x: Tensor) -> Tensor:
        flat = _get_flat(x._data)
        a, s = self._alpha, self._scale
        out = [s * v if v >= 0 else s * a * (math.exp(v) - 1) for v in flat]
        from vectorphile._core.ndarray import array as _nd
        sh = x._data.shape if hasattr(x._data, 'shape') else (len(flat),)
        return Tensor(_nd(out).reshape(sh), device=x.device)
    def __repr__(self): return 'SELU()'


class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        if _ACT_ISPC_OK:
            r = _ispc_unary(_C_act.ispc_silu_f32, x)
            if r is not None: return r
        return Tensor(_backend.silu(x._data), device=x.device)
    def __repr__(self): return 'SiLU()'


class Mish(Module):
    def forward(self, x: Tensor) -> Tensor:
        flat = _get_flat(x._data)
        import math as _m
        def _softplus(v): return _m.log1p(_m.exp(max(-88.0, v)))
        out = [v * _m.tanh(_softplus(v)) for v in flat]
        from vectorphile._core.ndarray import array as _nd
        sh = x._data.shape if hasattr(x._data, 'shape') else (len(flat),)
        return Tensor(_nd(out).reshape(sh), device=x.device)
    def __repr__(self): return 'Mish()'


class Hardswish(Module):
    def forward(self, x: Tensor) -> Tensor:
        flat = _get_flat(x._data)
        out = [v * max(0.0, min(6.0, v + 3)) / 6.0 for v in flat]
        from vectorphile._core.ndarray import array as _nd
        sh = x._data.shape if hasattr(x._data, 'shape') else (len(flat),)
        return Tensor(_nd(out).reshape(sh), device=x.device)
    def __repr__(self): return 'Hardswish()'


class ReLU6(Module):
    """ReLU capped at 6."""
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        flat = _get_flat(x._data)
        out = [min(max(v, 0.0), 6.0) for v in flat]
        from vectorphile._core.ndarray import array as _nd
        sh = x._data.shape if hasattr(x._data, 'shape') else (len(flat),)
        return Tensor(_nd(out).reshape(sh), device=x.device)

    def __repr__(self): return f'ReLU6(inplace={self.inplace})'


class Identity(Module):
    """Identity module that returns input unchanged."""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x

    def __repr__(self): return 'Identity()'
