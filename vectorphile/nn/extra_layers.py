"""Extra nn layers — ConvTranspose1d, DepthwiseSepConv, PReLU, GLU, ChannelShuffle,
PixelShuffle, Unfold, Fold, CoordConv, MixerMLP, etc.

All built from scratch using vectorphile Tensors.
"""
from __future__ import annotations
import math
import random as _random_el
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor, cat as _t_cat, stack as _t_stack


# Pure-Python + vectorphile shim replacing numpy in this file
class _NpShim:
    """Minimal numpy-compatible shim backed by pure Python + vectorphile NDArray."""
    @staticmethod
    def _fl(d):
        if hasattr(d, 'to_flat_list'): return d.to_flat_list()
        if hasattr(d, '_buf') and d._buf is not None: return list(d._buf)
        if isinstance(d, list): return d
        return list(d)
    @staticmethod
    def _nd(lst, shape=None):
        from vectorphile._core.ndarray import array as _a
        r = _a(lst if isinstance(lst, list) else list(lst))
        if shape: r = r.reshape(shape)
        return r
    @staticmethod
    def asarray(d, dtype=None):
        if hasattr(d, 'shape'): return d
        from vectorphile._core.ndarray import array as _a
        fl = _NpShim._fl(d) if not isinstance(d, list) else d
        return _a(fl)
    @staticmethod
    def zeros(shape, dtype='float32'):
        from vectorphile._core.ndarray import zeros as _z
        if isinstance(shape, int): shape = (shape,)
        return _z(shape, dtype=dtype)
    @staticmethod
    def zeros_like(d):
        sh = d.shape if hasattr(d, 'shape') else (len(_NpShim._fl(d)),)
        from vectorphile._core.ndarray import zeros as _z
        return _z(sh, dtype='float32')
    @staticmethod
    def ones(shape, dtype='float32'):
        from vectorphile._core.ndarray import ones as _o
        if isinstance(shape, int): shape = (shape,)
        return _o(shape, dtype=dtype)
    @staticmethod
    def eye(N, dtype='float32'):
        fl = [1.0 if i == j else 0.0 for i in range(N) for j in range(N)]
        return _NpShim._nd(fl, (N, N))
    @staticmethod
    def exp(d):
        fl = _NpShim._fl(d)
        sh = d.shape if hasattr(d, 'shape') else (len(fl),)
        return _NpShim._nd([math.exp(max(-88.0, min(88.0, v))) for v in fl], sh)
    @staticmethod
    def log(d):
        fl = _NpShim._fl(d)
        sh = d.shape if hasattr(d, 'shape') else (len(fl),)
        return _NpShim._nd([math.log(max(1e-30, v)) for v in fl], sh)
    @staticmethod
    def log1p(d):
        fl = _NpShim._fl(d)
        sh = d.shape if hasattr(d, 'shape') else (len(fl),)
        return _NpShim._nd([math.log1p(v) for v in fl], sh)
    @staticmethod
    def sqrt(d):
        if isinstance(d, (int, float)): return math.sqrt(max(0.0, d))
        fl = _NpShim._fl(d)
        sh = d.shape if hasattr(d, 'shape') else (len(fl),)
        return _NpShim._nd([math.sqrt(max(0.0, v)) for v in fl], sh)
    @staticmethod
    def tanh(d):
        fl = _NpShim._fl(d)
        sh = d.shape if hasattr(d, 'shape') else (len(fl),)
        return _NpShim._nd([math.tanh(v) for v in fl], sh)
    @staticmethod
    def abs(d):
        fl = _NpShim._fl(d)
        sh = d.shape if hasattr(d, 'shape') else (len(fl),)
        return _NpShim._nd([abs(v) for v in fl], sh)
    @staticmethod
    def maximum(a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)): return max(a, b)
        fa = _NpShim._fl(a) if not isinstance(a, (int, float)) else None
        fb = _NpShim._fl(b) if not isinstance(b, (int, float)) else None
        sh = a.shape if hasattr(a, 'shape') else (b.shape if hasattr(b, 'shape') else (len(fa or fb),))
        if fa is None: return _NpShim._nd([max(a, v) for v in fb], sh)
        if fb is None: return _NpShim._nd([max(v, b) for v in fa], sh)
        return _NpShim._nd([max(x, y) for x, y in zip(fa, fb)], sh)
    @staticmethod
    def minimum(a, b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)): return min(a, b)
        fa = _NpShim._fl(a) if not isinstance(a, (int, float)) else None
        fb = _NpShim._fl(b) if not isinstance(b, (int, float)) else None
        sh = a.shape if hasattr(a, 'shape') else (b.shape if hasattr(b, 'shape') else (len(fa or fb),))
        if fa is None: return _NpShim._nd([min(a, v) for v in fb], sh)
        if fb is None: return _NpShim._nd([min(v, b) for v in fa], sh)
        return _NpShim._nd([min(x, y) for x, y in zip(fa, fb)], sh)
    @staticmethod
    def clip(d, lo, hi):
        fl = _NpShim._fl(d)
        sh = d.shape if hasattr(d, 'shape') else (len(fl),)
        return _NpShim._nd([max(lo, min(hi, v)) for v in fl], sh)
    @staticmethod
    def where(cond, a, b):
        fc = _NpShim._fl(cond)
        fa = _NpShim._fl(a) if not isinstance(a, (int, float)) else [a]*len(fc)
        fb = _NpShim._fl(b) if not isinstance(b, (int, float)) else [b]*len(fc)
        sh = cond.shape if hasattr(cond, 'shape') else (len(fc),)
        return _NpShim._nd([x if c else y for c, x, y in zip(fc, fa, fb)], sh)
    @staticmethod
    def concatenate(arrays, axis=0):
        # Only supports axis=0 or axis=-1 (last)
        from vectorphile._core.ndarray import array as _a
        parts = []
        for arr in arrays:
            parts.extend(_NpShim._fl(arr))
        total = sum(len(_NpShim._fl(a)) for a in arrays)
        # Return flat if axis ambiguous
        sh0 = arrays[0].shape if hasattr(arrays[0], 'shape') else (len(_NpShim._fl(arrays[0])),)
        if axis == 0:
            new_sh = (sum(a.shape[0] if hasattr(a, 'shape') else 1 for a in arrays),) + sh0[1:]
        else:
            new_sh = sh0[:-1] + (sum(a.shape[-1] if hasattr(a, 'shape') else len(_NpShim._fl(a)) for a in arrays),)
        return _NpShim._nd(parts, new_sh)
    @staticmethod
    def stack(arrays, axis=0):
        from vectorphile._core.ndarray import array as _a
        parts = []
        for arr in arrays:
            parts.extend(_NpShim._fl(arr))
        n = len(arrays)
        sh0 = arrays[0].shape if hasattr(arrays[0], 'shape') else (len(_NpShim._fl(arrays[0])),)
        if axis == 0:
            new_sh = (n,) + sh0
        else:
            new_sh = sh0[:axis] + (n,) + sh0[axis:]
        return _NpShim._nd(parts, new_sh)
    @staticmethod
    def arange(n):
        return _NpShim._nd(list(range(n)), (n,))
    @staticmethod
    def argsort(d, axis=1):
        fl = _NpShim._fl(d)
        sh = d.shape if hasattr(d, 'shape') else (len(fl),)
        # Only supports 2D axis=1
        if len(sh) == 2:
            rows, cols = sh
            result = []
            for r in range(rows):
                row = fl[r*cols:(r+1)*cols]
                result.extend(sorted(range(cols), key=lambda i: row[i]))
            return _NpShim._nd([float(v) for v in result], sh)
        idx = sorted(range(len(fl)), key=lambda i: fl[i])
        return _NpShim._nd([float(v) for v in idx], sh)
    @staticmethod
    def sum(d, axis=None):
        fl = _NpShim._fl(d)
        if axis is None: return sum(fl)
        sh = d.shape if hasattr(d, 'shape') else (len(fl),)
        ndim = len(sh)
        ax = axis if axis >= 0 else ndim + axis
        new_sh = sh[:ax] + sh[ax+1:]
        outer = 1
        for i in range(ax): outer *= sh[i]
        inner = 1
        for i in range(ax+1, ndim): inner *= sh[i]
        dim_sz = sh[ax]
        result = [sum(fl[o*dim_sz*inner + k*inner + i] for k in range(dim_sz))
                  for o in range(outer) for i in range(inner)]
        return _NpShim._nd(result, new_sh)
    @staticmethod
    def mean(d, axis=None, keepdims=False):
        fl = _NpShim._fl(d)
        if axis is None:
            m = sum(fl) / len(fl)
            return m
        sh = d.shape if hasattr(d, 'shape') else (len(fl),)
        s = _NpShim.sum(d, axis=axis)
        dim_sz = sh[axis if axis >= 0 else len(sh)+axis]
        sf = _NpShim._fl(s)
        return _NpShim._nd([v / dim_sz for v in sf], s.shape if hasattr(s, 'shape') else (len(sf),))
    @staticmethod
    def real(d):
        fl = _NpShim._fl(d)
        sh = d.shape if hasattr(d, 'shape') else (len(fl),)
        return _NpShim._nd([v.real if hasattr(v, 'real') else v for v in fl], sh)
    @staticmethod
    def take(d, indices, axis=0):
        fl = _NpShim._fl(d)
        sh = d.shape if hasattr(d, 'shape') else (len(fl),)
        idx_list = _NpShim._fl(indices) if not isinstance(indices, list) else indices
        idx_int = [int(v) for v in idx_list]
        ndim = len(sh)
        ax = axis if axis >= 0 else ndim + axis
        # stride for axis
        inner = 1
        for i in range(ax+1, ndim): inner *= sh[i]
        outer = 1
        for i in range(ax): outer *= sh[i]
        dim_sz = sh[ax]
        new_sh = sh[:ax] + (len(idx_int),) + sh[ax+1:]
        result = [fl[o*dim_sz*inner + k*inner + ii]
                  for o in range(outer) for k in idx_int for ii in range(inner)]
        return _NpShim._nd(result, new_sh)
    @staticmethod
    def pad(d, pad_width, constant_values=0):
        fl = _NpShim._fl(d)
        sh = d.shape if hasattr(d, 'shape') else (len(fl),)
        ndim = len(sh)
        # pad_width: list of (before, after) per dim
        result = fl; cur_sh = list(sh)
        for dim in range(ndim):
            before, after = pad_width[dim]
            new_sz = cur_sh[dim] + before + after
            inner = 1
            for i in range(dim+1, ndim): inner *= cur_sh[i]
            outer = len(result) // (cur_sh[dim] * inner)
            new_result = []
            for o in range(outer):
                for _ in range(before):
                    for _ in range(inner): new_result.append(float(constant_values))
                for k in range(cur_sh[dim]):
                    for ii in range(inner):
                        new_result.append(result[o*cur_sh[dim]*inner + k*inner + ii])
                for _ in range(after):
                    for _ in range(inner): new_result.append(float(constant_values))
            cur_sh[dim] = new_sz; result = new_result
        return _NpShim._nd(result, tuple(cur_sh))
    @staticmethod
    def random_binomial_mask(shape, p, keep_prob):
        import random as _rm
        total = 1
        for s in shape: total *= s
        return _NpShim._nd([1.0/keep_prob if _rm.random() < p else 0.0 for _ in range(total)], shape)
    @staticmethod
    def issubdtype(dtype, base): return False
    newaxis = None
    inf = float('inf')


_np = _NpShim()


# ── Activation layers ─────────────────────────────────────────────────────────

class PReLU(Module):
    """Parametric ReLU: max(0, x) + weight * min(0, x)."""
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        from vectorphile._core.ndarray import full as _nd_full
        self.weight = Tensor(_nd_full((num_parameters,), init, dtype='float32'), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        from vectorphile._core.ndarray import array as _nd_prelu
        xd = x._data
        if hasattr(xd, 'to_flat_list'): xfl = xd.to_flat_list()
        else: xfl = list(xd)
        xsh = xd.shape if hasattr(xd, 'shape') else (len(xfl),)
        wfl = self.weight._data.to_flat_list() if hasattr(self.weight._data, 'to_flat_list') else list(self.weight._data)
        C = len(wfl)
        # channel stride (xsh[1] == C for multi-channel, else 1)
        ch_stride = 1
        for i in range(2, len(xsh)): ch_stride *= xsh[i]
        out = [xfl[i] if xfl[i] >= 0 else wfl[(i // ch_stride) % C] * xfl[i] for i in range(len(xfl))]
        return Tensor(_nd_prelu(out).reshape(xsh), device=x.device, requires_grad=x.requires_grad)


class GLU(Module):
    """Gated Linear Unit: x[:, :d//2] * sigmoid(x[:, d//2:])"""
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        from vectorphile.nn import functional as _F
        return _F.glu(x, dim=self.dim)


class CELU(Module):
    """CELU activation: max(0,x) + min(0, alpha*(exp(x/alpha)-1))"""
    def __init__(self, alpha=1.0, inplace=False):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        from vectorphile.nn import functional as _F
        return _F.celu(x, alpha=self.alpha)


class LogSigmoid(Module):
    """LogSigmoid: log(sigmoid(x))"""
    def forward(self, x: Tensor) -> Tensor:
        from vectorphile.nn import functional as _F
        return _F.logsigmoid(x)


class Tanhshrink(Module):
    """Tanhshrink: x - tanh(x)"""
    def forward(self, x: Tensor) -> Tensor:
        from vectorphile.nn import functional as _F
        return _F.tanhshrink(x)


class Softshrink(Module):
    """Softshrink activation."""
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, x: Tensor) -> Tensor:
        from vectorphile.nn import functional as _F
        return _F.softshrink(x, lambd=self.lambd)


class Hardshrink(Module):
    """Hardshrink activation."""
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, x: Tensor) -> Tensor:
        from vectorphile.nn import functional as _F
        return _F.hardshrink(x, lambd=self.lambd)


class Hardtanh(Module):
    """Hardtanh activation: clamp(x, min_val, max_val)."""
    def __init__(self, min_val=-1.0, max_val=1.0, inplace=False):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: Tensor) -> Tensor:
        return x.clamp(self.min_val, self.max_val)


class Softplus(Module):
    """Softplus activation: (1/beta) * log(1 + exp(beta * x)), clamped at threshold."""
    def __init__(self, beta=1.0, threshold=20.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: Tensor) -> Tensor:
        import math as _m_sp
        from vectorphile._core.ndarray import array as _nd_sp
        flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
        sh = x._data.shape
        b, t = self.beta, self.threshold
        out = []
        for v in flat:
            bv = b * v
            if bv > t:
                out.append(v)  # linear beyond threshold
            else:
                out.append(_m_sp.log1p(_m_sp.exp(bv)) / b)
        nd = _nd_sp(out, dtype=str(x._data.dtype) if hasattr(x._data, 'dtype') else 'float32')
        nd = nd.reshape(sh)
        return Tensor(nd, device=x.device, requires_grad=x.requires_grad)


class Threshold(Module):
    """Threshold activation: x if x > threshold else value."""
    def __init__(self, threshold: float, value: float, inplace: bool = False):
        super().__init__()
        self.threshold = threshold
        self.value = value

    def forward(self, x: Tensor) -> Tensor:
        from vectorphile._core.ndarray import array as _nd_thr
        flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
        sh = x._data.shape
        t, v = self.threshold, self.value
        out = [val if val > t else v for val in flat]
        nd = _nd_thr(out, dtype=str(x._data.dtype) if hasattr(x._data, 'dtype') else 'float32')
        nd = nd.reshape(sh)
        return Tensor(nd, device=x.device, requires_grad=x.requires_grad)


class Swish(Module):
    """Swish activation: x * sigmoid(beta * x)."""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x: Tensor) -> Tensor:
        d = _np.asarray(x._data)
        return Tensor(d / (1 + _np.exp(-self.beta * d)))


class Snake(Module):
    """Snake activation: x + (1/a) * sin(a*x)^2 — periodic, good for periodic signals."""
    def __init__(self, a=1.0, trainable=True):
        super().__init__()
        from vectorphile._core.ndarray import array as _nd_arr
        if trainable:
            self.a = Tensor(_nd_arr([a], dtype='float32'), requires_grad=True)
        else:
            self._a = a

    def forward(self, x: Tensor) -> Tensor:
        a = float(_np.asarray(self.a._data)[0]) if hasattr(self, 'a') else self._a
        d = _np.asarray(x._data)
        return Tensor(d + (1.0 / a) * _np.sin(a * d) ** 2)


class Mamba(Module):
    """Simplified Mamba SSM block (State Space Model).

    A causal sequence model: y = SSM(x) with learnable A, B, C, D matrices.
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2,
                 dt_rank: int = None, bias: bool = True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        from vectorphile.nn.layers import Linear, Conv1d
        from vectorphile._core.ndarray import uniform as _nd_uniform, zeros as _nd_zeros

        # Input projection
        self.in_proj = Linear(d_model, self.d_inner * 2, bias=bias)
        # Depthwise conv
        self.conv1d = Conv1d(self.d_inner, self.d_inner, kernel_size=4,
                             padding=3, groups=self.d_inner, bias=True)
        # SSM parameters
        self.x_proj = Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = Linear(self.dt_rank, self.d_inner, bias=True)

        # A: d_inner x d_state (log init)
        k = _nd_uniform(1, d_state + 1, (self.d_inner, d_state))
        import math as _m2, array as _a
        from vectorphile._core.ndarray import array as _nd_arr, NDArray as _NDA, _c_strides as _cs, _DType as _DT
        k_flat = k.to_flat_list() if hasattr(k, 'to_flat_list') else list(k)
        log_a_flat = [_m2.log(max(v, 1e-8)) for v in k_flat]
        buf = _a.array('f', log_a_flat)
        log_a_shape = (self.d_inner, d_state)
        self.A_log = Tensor(_NDA(buf, log_a_shape, _cs(log_a_shape), _DT('float32')), requires_grad=True)
        self.D = Tensor(_nd_zeros(self.d_inner, dtype='float32'), requires_grad=True)
        self.out_proj = Linear(self.d_inner, d_model, bias=bias)

    def _ssm(self, u):
        """Run the SSM over sequence u: (B, L, D)."""
        from vectorphile._core.ndarray import array as _nd_arr
        import math as _m
        B, L, D = u.shape

        def _fl(t):
            if hasattr(t, 'to_flat_list'): return t.to_flat_list()
            if hasattr(t, '_data'):
                d = t._data
                return d.to_flat_list() if hasattr(d, 'to_flat_list') else list(d)
            return list(t)

        # project to dt, B_ssm, C_ssm
        x_dbl = self.x_proj(u.reshape(B * L, D))
        xd_fl = _fl(x_dbl)
        R = self.dt_rank; N = self.d_state; total_w = R + 2 * N
        # reshape xd_fl to (B, L, total_w)
        dt_fl   = [xd_fl[b*L*total_w + t*total_w + r]      for b in range(B) for t in range(L) for r in range(R)]
        b_fl    = [xd_fl[b*L*total_w + t*total_w + R + n]  for b in range(B) for t in range(L) for n in range(N)]
        c_fl    = [xd_fl[b*L*total_w + t*total_w + R+N+n]  for b in range(B) for t in range(L) for n in range(N)]

        dt_nd   = _nd_arr(dt_fl, 'float32').reshape((B * L, R))
        dt_full_fl = _fl(self.dt_proj(Tensor(dt_nd)))  # (B*L, D)
        # softplus
        dt_full_fl = [_m.log1p(_m.exp(v)) for v in dt_full_fl]

        # Flatten A_log to get A (negated, D x N)
        a_log_fl = _fl(self.A_log)
        a_fl     = [-_m.fabs(v) for v in a_log_fl]  # (D, N) flat
        d_fl     = _fl(self.D)                        # (D,)
        u_fl     = _fl(u)                             # (B, L, D)

        # Build NDArrays for the kernel
        x_nd     = _nd_arr(u_fl, 'float32').reshape((B, L, D))
        delta_nd = _nd_arr(dt_full_fl, 'float32').reshape((B, L, D))
        A_nd     = _nd_arr(a_fl, 'float32').reshape((D, N))
        B_nd     = _nd_arr(b_fl, 'float32').reshape((B, L, N))
        C_nd     = _nd_arr(c_fl, 'float32').reshape((B, L, N))
        D_nd     = _nd_arr(d_fl, 'float32')

        try:
            from vectorphile.cuda.triton_kernels import mamba_selective_scan
            return mamba_selective_scan(x_nd, delta_nd, A_nd, B_nd, C_nd, D_nd)
        except Exception:
            pass

        # Pure-VP CPU fallback
        result = [0.0] * (B * L * D)
        for b in range(B):
            h = [[0.0] * N for _ in range(D)]
            for t in range(L):
                base_dt = b * L * D + t * D
                for d in range(D):
                    dt_v = dt_full_fl[base_dt + d]
                    y = 0.0
                    for n in range(N):
                        a_bar = _m.exp(dt_v * a_fl[d * N + n])
                        b_bar = dt_v * b_fl[b * L * N + t * N + n]
                        h[d][n] = a_bar * h[d][n] + b_bar * u_fl[b * L * D + t * D + d]
                        y += c_fl[b * L * N + t * N + n] * h[d][n]
                    y += d_fl[d] * u_fl[b * L * D + t * D + d]
                    result[b * L * D + t * D + d] = y
        return _nd_arr(result, 'float32').reshape((B, L, D))

    def forward(self, x: Tensor) -> Tensor:
        import math as _m
        from vectorphile._core.ndarray import array as _nd_arr
        B, L, D = x.shape
        Di = self.d_inner

        def _fl(t):
            if hasattr(t, 'to_flat_list'): return t.to_flat_list()
            if hasattr(t, '_data'):
                d = t._data
                return d.to_flat_list() if hasattr(d, 'to_flat_list') else list(d)
            return list(t)

        xz = self.in_proj(x.reshape(B * L, D))
        xzfl = _fl(xz)  # (B*L, Di*2)
        u_fl = [xzfl[i * Di*2 + j] for i in range(B * L) for j in range(Di)]
        z_fl = [xzfl[i * Di*2 + Di + j] for i in range(B * L) for j in range(Di)]

        u_conv_in = _nd_arr(u_fl, 'float32').reshape((B, Di, L)).reshape((B, Di, L))
        # transpose to (B, Di, L) already; run conv and slice
        u_conv_out_fl = _fl(self.conv1d(Tensor(u_conv_in)))  # (B, Di, L+pad)
        # slice first L timesteps: shape (B, Di, L)
        u_conv_fl = u_conv_out_fl[:B * Di * L] if len(u_conv_out_fl) == B * Di * L else [
            u_conv_out_fl[b * Di * (L+3) + d * (L+3) + t]
            for b in range(B) for d in range(Di) for t in range(L)
        ]
        # silu + reshape to (B, L, Di)
        u_act_fl = [v / (1.0 + _m.exp(-v)) for v in u_conv_fl]
        # u_conv_fl is (B, Di, L); need (B, L, Di)
        u_act_bldi = [u_act_fl[b * Di * L + d * L + t] for b in range(B) for t in range(L) for d in range(Di)]
        u_nd = _nd_arr(u_act_bldi, 'float32').reshape((B, L, Di))

        y_nd = self._ssm(Tensor(u_nd))  # returns NDArray (B, L, Di)
        y_fl = y_nd.to_flat_list() if hasattr(y_nd, 'to_flat_list') else list(y_nd)

        # silu gate on z_fl (B*L, Di) → (B, L, Di)
        z_gate_fl = [z_fl[i] / (1.0 + _m.exp(-z_fl[i])) for i in range(len(z_fl))]
        # z_fl is (B*L, Di) order; reshape same as y → element-wise multiply
        yz_fl = [y_fl[i] * z_gate_fl[i] for i in range(B * L * Di)]

        output = self.out_proj(Tensor(_nd_arr(yz_fl, 'float32').reshape((B * L, Di))))
        return output.reshape(B, L, D)


# ── Convolutional layers ──────────────────────────────────────────────────────

class ConvTranspose1d(Module):
    """1-D Transposed Convolution (up-convolution / deconvolution)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, bias=True, dilation=1):
        super().__init__()
        self.in_ch = in_channels
        self.out_ch = out_channels
        self.ks = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        from vectorphile._core.ndarray import uniform as _nd_uniform, zeros as _nd_zeros
        k = math.sqrt(1.0 / (in_channels * kernel_size))
        self.weight = Tensor(_nd_uniform(-k, k, (in_channels, out_channels, kernel_size)), requires_grad=True)
        self.bias_param = None
        if bias:
            self.bias_param = Tensor(_nd_zeros(out_channels, dtype='float32'), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        N, Ci, L = x.shape
        Co, ks, s, p = self.out_ch, self.ks, self.stride, self.padding
        d = self.dilation
        L_out = (L - 1) * s - 2 * p + d * (ks - 1) + 1 + self.output_padding
        xd = _np.asarray(x._data)
        wd = _np.asarray(self.weight._data)  # (Ci, Co, ks)
        out = _np.zeros((N, Co, L_out), dtype='float32')
        for i in range(L):
            for ki in range(ks):
                j = i * s + ki * d - p
                if 0 <= j < L_out:
                    out[:, :, j] += _np.einsum('nc,cok->no', xd[:, :, i], wd[:, :, ki:ki+1]).squeeze(-1) \
                        if False else (_np.einsum('nc,co->no', xd[:, :, i], wd[:, :, ki]))
        result = Tensor(out)
        if self.bias_param is not None:
            result = result + Tensor(_np.asarray(self.bias_param._data)[None, :, None])
        return result


class DepthwiseSepConv2d(Module):
    """Depthwise-separable 2-D convolution (MobileNet building block)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        from vectorphile.nn.layers import Conv2d
        self.dw = Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                         padding=padding, groups=in_channels, bias=False)
        self.pw = Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.pw(self.dw(x))


class DepthwiseSepConv1d(Module):
    """Depthwise-separable 1-D convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        from vectorphile.nn.layers import Conv1d
        self.dw = Conv1d(in_channels, in_channels, kernel_size, stride=stride,
                         padding=padding, groups=in_channels, bias=False)
        self.pw = Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.pw(self.dw(x))


class PixelShuffle(Module):
    """Rearrange elements in (N, C*r^2, H, W) → (N, C, H*r, W*r)."""

    def __init__(self, upscale_factor: int):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x: Tensor) -> Tensor:
        r = self.r
        d = _np.asarray(x._data)
        N, C, H, W = d.shape
        Co = C // (r * r)
        d = d.reshape(N, Co, r, r, H, W)
        d = d.transpose(0, 1, 4, 2, 5, 3)
        return Tensor(d.reshape(N, Co, H * r, W * r))


class ChannelShuffle(Module):
    """Channel shuffle for group convolutions (ShuffleNet building block)."""

    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x: Tensor) -> Tensor:
        d = _np.asarray(x._data)
        N, C, H, W = d.shape
        g = self.groups
        d = d.reshape(N, g, C // g, H, W)
        d = d.transpose(0, 2, 1, 3, 4).reshape(N, C, H, W)
        return Tensor(d)


class Unfold(Module):
    """Extract sliding local patches from a batched image tensor.

    Output shape: (N, C*kh*kw, L) where L = number of patches.
    """

    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)

    def forward(self, x: Tensor) -> Tensor:
        d = _np.asarray(x._data)
        N, C, H, W = d.shape
        kh, kw = self.kernel_size
        dh, dw = self.dilation
        ph, pw = self.padding
        sh, sw = self.stride
        Ho = (H + 2*ph - dh*(kh-1) - 1) // sh + 1
        Wo = (W + 2*pw - dw*(kw-1) - 1) // sw + 1
        d_pad = _np.pad(d, ((0,0),(0,0),(ph,ph),(pw,pw)))
        patches = _np.zeros((N, C*kh*kw, Ho*Wo), dtype=d.dtype)
        for i in range(Ho):
            for j in range(Wo):
                for ki in range(kh):
                    for kj in range(kw):
                        hi = i * sh + ki * dh
                        wj = j * sw + kj * dw
                        col_idx = (ki * kw + kj)
                        for c in range(C):
                            patches[:, c * kh * kw + col_idx, i * Wo + j] = d_pad[:, c, hi, wj]
        return Tensor(patches)


class Fold(Module):
    """Combine an array of sliding local blocks into a large containing tensor."""

    def __init__(self, output_size, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)

    def forward(self, x: Tensor) -> Tensor:
        d = _np.asarray(x._data)
        N, CKK, L = d.shape
        H_out, W_out = self.output_size
        kh, kw = self.kernel_size
        C = CKK // (kh * kw)
        ph, pw = self.padding
        sh, sw = self.stride
        dh, dw = self.dilation
        Ho = (H_out + 2*ph - dh*(kh-1) - 1) // sh + 1
        Wo = (W_out + 2*pw - dw*(kw-1) - 1) // sw + 1
        H_pad = H_out + 2 * ph
        W_pad = W_out + 2 * pw
        output = _np.zeros((N, C, H_pad, W_pad), dtype=d.dtype)
        for i in range(Ho):
            for j in range(Wo):
                for ki in range(kh):
                    for kj in range(kw):
                        hi = i * sh + ki * dh
                        wj = j * sw + kj * dw
                        col_idx = ki * kw + kj
                        for c in range(C):
                            output[:, c, hi, wj] += d[:, c * kh * kw + col_idx, i * Wo + j]
        # Remove padding
        if ph > 0 or pw > 0:
            output = output[:, :, ph:ph+H_out, pw:pw+W_out]
        return Tensor(output)


# ── Transformer / attention extras ────────────────────────────────────────────

class MultiQueryAttention(Module):
    """Multi-Query Attention: all query heads share one key/value head (Shazeer 2019)."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 bias: bool = True, batch_first: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_p = dropout
        self.batch_first = batch_first

        from vectorphile._core.ndarray import uniform as _nd_uniform, zeros as _nd_zeros
        k = math.sqrt(1.0 / embed_dim)
        self.q_proj = Tensor(_nd_uniform(-k, k, (embed_dim, embed_dim)), requires_grad=True)
        self.k_proj = Tensor(_nd_uniform(-k, k, (self.head_dim, embed_dim)), requires_grad=True)
        self.v_proj = Tensor(_nd_uniform(-k, k, (self.head_dim, embed_dim)), requires_grad=True)
        self.out_proj = Tensor(_nd_uniform(-k, k, (embed_dim, embed_dim)), requires_grad=True)
        if bias:
            self.in_proj_bias = Tensor(_nd_zeros(embed_dim + 2 * self.head_dim, dtype='float32'), requires_grad=True)
            self.out_proj_bias = Tensor(_nd_zeros(embed_dim, dtype='float32'), requires_grad=True)
        else:
            object.__setattr__(self, 'in_proj_bias', None)
            object.__setattr__(self, 'out_proj_bias', None)

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                attn_mask=None, need_weights=True) -> tuple:
        qd = _np.asarray(q._data)
        kd = _np.asarray(k._data)
        vd = _np.asarray(v._data)
        if not self.batch_first:
            qd, kd, vd = qd.transpose(1,0,2), kd.transpose(1,0,2), vd.transpose(1,0,2)
        B, Tq, E = qd.shape
        _, Tk, _ = kd.shape
        H, dh = self.num_heads, self.head_dim

        Q = (qd @ _np.asarray(self.q_proj._data).T).reshape(B, Tq, H, dh).transpose(0,2,1,3)
        K = (kd @ _np.asarray(self.k_proj._data).T)[:, :, None, :].repeat(H, axis=2).transpose(0,2,1,3)
        V = (vd @ _np.asarray(self.v_proj._data).T)[:, :, None, :].repeat(H, axis=2).transpose(0,2,1,3)

        attn = (Q @ K.transpose(0,1,3,2)) * self.scale
        if attn_mask is not None:
            attn = attn + (attn_mask * -1e9 if _np.issubdtype(_np.asarray(attn_mask).dtype, _np.bool_)
                           else _np.asarray(attn_mask)[None, None])
        attn = _np.exp(attn - attn.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        out = (attn @ V).transpose(0,2,1,3).reshape(B, Tq, E)
        out = out @ _np.asarray(self.out_proj._data).T
        if self.out_proj_bias is not None:
            out = out + _np.asarray(self.out_proj_bias._data)
        if not self.batch_first:
            out = out.transpose(1,0,2)
        return Tensor(out), Tensor(attn.mean(axis=1)) if need_weights else None


class CrossAttention(Module):
    """Cross-Attention: query from one sequence, key/value from another."""

    def __init__(self, embed_dim: int, num_heads: int, kv_dim: int = None,
                 dropout: float = 0.0, bias: bool = True):
        super().__init__()
        from vectorphile.nn.attention import MultiheadAttention
        kv_dim = kv_dim or embed_dim
        self.q_proj = MultiheadAttention(embed_dim, num_heads, dropout=dropout,
                                         bias=bias, batch_first=True)
        self.kv_dim = kv_dim
        if kv_dim != embed_dim:
            from vectorphile.nn.layers import Linear
            self.k_proj = Linear(kv_dim, embed_dim, bias=False)
            self.v_proj = Linear(kv_dim, embed_dim, bias=False)
        else:
            object.__setattr__(self, 'k_proj', None)
            object.__setattr__(self, 'v_proj', None)

    def forward(self, query: Tensor, context: Tensor, attn_mask=None) -> Tensor:
        k = self.k_proj(context) if self.k_proj is not None else context
        v = self.v_proj(context) if self.v_proj is not None else context
        return self.q_proj(query, k, v, attn_mask=attn_mask)


class LinearAttention(Module):
    """Linear Attention O(N) complexity — kernel approximation of softmax attention."""

    def __init__(self, embed_dim: int, num_heads: int, feature_map: str = 'elu'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.feature_map_name = feature_map
        from vectorphile.nn.layers import Linear
        self.q_proj = Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = Linear(embed_dim, embed_dim)

    def _phi(self, x):
        """Feature map: ELU + 1 (keeps positive)."""
        return _np.where(x >= 0, x + 1, _np.exp(x))

    def forward(self, x: Tensor) -> Tensor:
        B, T, E = x.shape
        H, dh = self.num_heads, self.head_dim
        qd = _np.asarray(self.q_proj(x)._data).reshape(B, T, H, dh).transpose(0,2,1,3)
        kd = _np.asarray(self.k_proj(x)._data).reshape(B, T, H, dh).transpose(0,2,1,3)
        vd = _np.asarray(self.v_proj(x)._data).reshape(B, T, H, dh).transpose(0,2,1,3)
        q_phi = self._phi(qd)
        k_phi = self._phi(kd)
        # (B, H, dh, dh) = sum_t k_t^T v_t
        kv = k_phi.transpose(0,1,3,2) @ vd  # (B, H, dh, dh)
        # (B, H, T, dh) = q_phi @ kv
        out = q_phi @ kv
        # Normalize
        k_sum = k_phi.sum(axis=2, keepdims=True)  # (B, H, 1, dh)
        norm = (q_phi * k_sum).sum(axis=-1, keepdims=True).clip(1e-6)
        out = out / norm
        out = out.transpose(0,2,1,3).reshape(B, T, E)
        return self.out_proj(Tensor(out))


# ── Normalization extras ──────────────────────────────────────────────────────

class FilterResponseNorm(Module):
    """Filter Response Normalization (Kim & Park 2019)."""

    def __init__(self, num_features: int, eps: float = 1e-6, learnable_eps: bool = True):
        super().__init__()
        self.eps = eps
        from vectorphile._core.ndarray import ones as _nd_ones, zeros as _nd_zeros
        self.gamma = Tensor(_nd_ones((1, num_features, 1, 1), dtype='float32'), requires_grad=True)
        self.beta = Tensor(_nd_zeros((1, num_features, 1, 1), dtype='float32'), requires_grad=True)
        self.tau = Tensor(_nd_zeros((1, num_features, 1, 1), dtype='float32'), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        d = _np.asarray(x._data)
        # Normalize per-filter
        nu2 = (d * d).mean(axis=(2, 3), keepdims=True)
        normed = d / _np.sqrt(nu2 + self.eps)
        g = _np.asarray(self.gamma._data)
        b = _np.asarray(self.beta._data)
        t = _np.asarray(self.tau._data)
        y = g * normed + b
        return Tensor(_np.maximum(y, t))


class PowerNorm(Module):
    """Power Normalization — alternative to BatchNorm using quadratic mean."""

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        from vectorphile._core.ndarray import ones as _nd_ones, zeros as _nd_zeros
        self.gamma = Tensor(_nd_ones(num_features, dtype='float32'), requires_grad=True)
        self.beta = Tensor(_nd_zeros(num_features, dtype='float32'), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        d = _np.asarray(x._data)
        # Quadratic mean per sample per channel
        qm = _np.sqrt((d * d).mean(axis=-1, keepdims=True) + self.eps)
        normed = d / qm
        g = _np.asarray(self.gamma._data)
        b = _np.asarray(self.beta._data)
        return Tensor(g * normed + b)


# ── Regularization ────────────────────────────────────────────────────────────

class StochasticDepth(Module):
    """Stochastic Depth (DropPath): randomly drop entire residual branches during training."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        from vectorphile._core.ndarray import array as _nd_sd
        keep_prob = 1 - self.drop_prob
        xd = x._data; xfl = xd.to_flat_list() if hasattr(xd, 'to_flat_list') else list(xd)
        xsh = xd.shape if hasattr(xd, 'shape') else (len(xfl),)
        out_sd = [v / keep_prob if _random_el.random() < keep_prob else 0.0 for v in xfl]
        return Tensor(_nd_sd(out_sd).reshape(xsh), device=x.device, requires_grad=x.requires_grad)


class DropBlock2d(Module):
    """DropBlock: structured dropout for spatial feature maps (Ghiasi et al., 2018)."""

    def __init__(self, drop_prob: float = 0.1, block_size: int = 7):
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0:
            return x
        from vectorphile._core.ndarray import array as _nd_db
        xd = x._data; xfl = xd.to_flat_list() if hasattr(xd, 'to_flat_list') else list(xd)
        xsh = xd.shape if hasattr(xd, 'shape') else (len(xfl),)
        N, C, H, W = xsh
        p = self.drop_prob / (self.block_size ** 2)
        bsz = self.block_size
        seed_flat = [1.0 if _random_el.random() < p else 0.0 for _ in range(N*C*H*W)]
        # max-pool dilation (block_size x block_size)
        block_flat = [0.0]*(N*C*H*W)
        half = bsz // 2
        for n in range(N):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        v = 0.0
                        for di in range(-half, half+1):
                            for dj in range(-half, half+1):
                                nh = h+di; nw = w+dj
                                if 0 <= nh < H and 0 <= nw < W:
                                    if seed_flat[n*C*H*W + c*H*W + nh*W + nw] > 0: v = 1.0; break
                            if v: break
                        block_flat[n*C*H*W + c*H*W + h*W + w] = v
        keep_prob2 = max(1 - sum(block_flat)/len(block_flat), 1e-6)
        mask = [(1-b)/keep_prob2 for b in block_flat]
        out_db = [xfl[i]*mask[i] for i in range(len(xfl))]
        return Tensor(_nd_db(out_db).reshape(xsh), device=x.device, requires_grad=x.requires_grad)


# ── Positional encodings ──────────────────────────────────────────────────────

class RelativePositionEncoding(Module):
    """Relative position encoding (Shaw et al., 2018)."""

    def __init__(self, max_len: int, embed_dim: int, num_heads: int):
        super().__init__()
        self.max_len = max_len
        self.head_dim = embed_dim // num_heads
        from vectorphile._core.ndarray import uniform as _nd_uniform
        k = math.sqrt(1.0 / self.head_dim)
        # (2*max_len - 1, head_dim)
        self.rel_emb = Tensor(_nd_uniform(-k, k, (2 * max_len - 1, self.head_dim)), requires_grad=True)

    def forward(self, seq_len: int) -> Tensor:
        """Return (seq_len, seq_len, head_dim) relative bias table."""
        idx = _np.arange(seq_len)
        rel = idx[:, None] - idx[None, :] + self.max_len - 1
        rel = _np.clip(rel, 0, 2 * self.max_len - 2)
        embd = _np.asarray(self.rel_emb._data)
        return Tensor(embd[rel])


class ALiBiPositionBias(Module):
    """ALiBi: Attention with Linear Biases (Press et al., 2022)."""

    def __init__(self, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        # Precompute slopes
        n = 2 ** math.floor(math.log2(num_heads))
        m = 2 ** (-8 / n)
        slopes = _np.array([m ** (i + 1) for i in range(n)], dtype='float32')
        if num_heads > n:
            extra_m = 2 ** (-4 / n)
            extra_slopes = _np.array([extra_m ** (i + 1) for i in range(2 * (num_heads - n))][::2], dtype='float32')
            slopes = _np.concatenate([slopes, extra_slopes[:num_heads - n]])
        self._slopes = slopes
        self._max_seq_len = max_seq_len

    def forward(self, seq_len: int) -> Tensor:
        """Return (num_heads, seq_len, seq_len) bias."""
        pos = _np.arange(seq_len)
        dist = _np.abs(pos[:, None] - pos[None, :])  # (T, T)
        bias = -dist[None, :, :] * self._slopes[:, None, None]  # (H, T, T)
        return Tensor(bias.astype('float32'))


# ── Misc utility layers ───────────────────────────────────────────────────────

class Lambda(Module):
    """Wraps any callable as a Module."""
    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def forward(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class Residual(Module):
    """x + module(x) shortcut."""
    def __init__(self, module: Module):
        super().__init__()
        self.module = module

    def forward(self, x: Tensor) -> Tensor:
        return x + self.module(x)


class PreNorm(Module):
    """Pre-layer-norm wrapper: norm(x) fed to module, then added to skip."""
    def __init__(self, dim: int, module: Module, norm_cls=None):
        super().__init__()
        from vectorphile.nn.layers import LayerNorm
        self.norm = (norm_cls or LayerNorm)(dim)
        self.module = module

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.module(self.norm(x), **kwargs)


class Scale(Module):
    """Learnable scalar multiplier for a Tensor."""
    def __init__(self, init: float = 1.0, trainable: bool = True):
        super().__init__()
        from vectorphile._core.ndarray import array as _nd_arr
        self.scale = Tensor(_nd_arr([init], dtype='float32'), requires_grad=trainable)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale


class MixerMLP(Module):
    """MLP Mixer token/channel mixing MLP block (Tolstikhin et al., 2021)."""

    def __init__(self, num_patches: int, hidden_dim: int, channel_dim: int, dropout: float = 0.0):
        super().__init__()
        from vectorphile.nn.layers import Linear
        from vectorphile.nn.activations import GELU
        # Token mixing
        self.token_mix = Module.__new__(Module)
        Module.__init__(self.token_mix)
        from vectorphile.nn.sequential import Sequential
        from vectorphile.nn.layers import LayerNorm as LN
        self.token_norm = LN(channel_dim)
        self.token_mlp_fc1 = Linear(num_patches, hidden_dim)
        self.token_mlp_fc2 = Linear(hidden_dim, num_patches)
        # Channel mixing
        self.channel_norm = LN(channel_dim)
        self.channel_mlp_fc1 = Linear(channel_dim, channel_dim * 4)
        self.channel_mlp_fc2 = Linear(channel_dim * 4, channel_dim)
        self.dropout_p = dropout

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, num_patches, channel_dim)
        # Token mixing: apply MLP over patches dim
        normed = self.token_norm(x)
        d = _np.asarray(normed._data).transpose(0, 2, 1)  # (B, C, P)
        B, C, P = d.shape
        d_flat = d.reshape(B * C, P)
        t1 = Tensor(d_flat)
        t2 = self.token_mlp_fc1(t1).relu()
        t3 = self.token_mlp_fc2(t2)
        x = x + Tensor(_np.asarray(t3._data).reshape(B, C, P).transpose(0, 2, 1))
        # Channel mixing
        normed2 = self.channel_norm(x)
        B2, P2, C2 = normed2.shape
        c_flat = Tensor(_np.asarray(normed2._data).reshape(B2 * P2, C2))
        c2 = self.channel_mlp_fc1(c_flat).gelu()
        c3 = self.channel_mlp_fc2(c2)
        x = x + Tensor(_np.asarray(c3._data).reshape(B2, P2, C2))
        return x


class GatedMLP(Module):
    """gMLP — gated MLP block from Liu et al., 2021."""

    def __init__(self, dim: int, seq_len: int, ffn_dim: int = None):
        super().__init__()
        ffn_dim = ffn_dim or dim * 6
        from vectorphile.nn.layers import Linear, LayerNorm
        self.norm = LayerNorm(dim)
        self.proj_in = Linear(dim, ffn_dim)
        self.gate = Linear(seq_len, seq_len)
        self.proj_out = Linear(ffn_dim // 2, dim)

    def forward(self, x: Tensor) -> Tensor:
        d = _np.asarray(self.norm(x)._data)
        B, T, C = d.shape
        # Project + split gate
        proj = _np.asarray(self.proj_in(Tensor(d.reshape(B*T, C)))._data).reshape(B, T, -1)
        u, v = proj[:, :, :proj.shape[-1]//2], proj[:, :, proj.shape[-1]//2:]
        # Spatial gate: apply linear across sequence dim
        gate_v = _np.asarray(self.gate(Tensor(v.transpose(0,2,1)))._data).transpose(0,2,1)
        gated = Tensor(u * _np.tanh(gate_v))
        return x + self.proj_out(gated.reshape(B*T, -1)).reshape(B, T, -1)


class PatchEmbed(Module):
    """Image-to-patch embedding for Vision Transformers (ViT)."""

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        from vectorphile.nn.layers import Conv2d
        self.proj = Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W) → (B, num_patches, embed_dim)
        d = _np.asarray(self.proj(x)._data)  # (B, embed_dim, H', W')
        B, E, H, W = d.shape
        return Tensor(d.reshape(B, E, H * W).transpose(0, 2, 1))


class VisionTransformerBlock(Module):
    """Standard ViT transformer block with pre-LN, MHA, and FFN."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        from vectorphile.nn.layers import Linear, LayerNorm
        from vectorphile.nn.attention import MultiheadAttention
        self.norm1 = LayerNorm(embed_dim)
        self.attn = MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.fc1 = Linear(embed_dim, mlp_dim)
        self.fc2 = Linear(mlp_dim, embed_dim)
        self.dropout_p = dropout

    def forward(self, x: Tensor) -> Tensor:
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        # FFN with residual
        normed2 = self.norm2(x)
        d = _np.asarray(normed2._data)
        B, T, E = d.shape
        ff = Tensor(d.reshape(B*T, E))
        ff = self.fc1(ff).gelu()
        ff = self.fc2(ff)
        x = x + Tensor(_np.asarray(ff._data).reshape(B, T, E))
        return x


# ── Graph Neural Networks ─────────────────────────────────────────────────────

class GCNConv(Module):
    """Graph Convolutional Network layer (Kipf & Welling, 2017).

    h' = sigma(A_hat * H * W)  where A_hat = D^{-1/2} (A + I) D^{-1/2}
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 improved: bool = False, add_self_loops: bool = True):
        super().__init__()
        from vectorphile.nn.layers import Linear
        self.lin = Linear(in_features, out_features, bias=bias)
        self.improved = improved
        self.add_self_loops = add_self_loops

    def forward(self, x: Tensor, edge_index, edge_weight=None) -> Tensor:
        """
        Args:
            x: node features (N, in_features)
            edge_index: (2, E) int array [src, dst] edges
            edge_weight: optional (E,) float edge weights
        """
        xd = _np.asarray(x._data)
        N = xd.shape[0]
        ei = _np.asarray(edge_index._data) if isinstance(edge_index, Tensor) else _np.asarray(edge_index)
        # Build adjacency matrix
        A = _np.zeros((N, N), dtype='float32')
        A[ei[0], ei[1]] = 1.0
        if edge_weight is not None:
            ew = _np.asarray(edge_weight._data) if isinstance(edge_weight, Tensor) else _np.asarray(edge_weight)
            A[ei[0], ei[1]] = ew
        if self.add_self_loops:
            A += _np.eye(N, dtype='float32') * (2.0 if self.improved else 1.0)
        # Normalize: D^{-1/2} A D^{-1/2}
        deg = A.sum(axis=1)
        d_inv_sqrt = _np.where(deg > 0, 1.0 / _np.sqrt(deg), 0.0)
        A_norm = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
        # Aggregate
        agg = A_norm @ xd
        return self.lin(Tensor(agg))


class GATConv(Module):
    """Graph Attention Network layer (Veličković et al., 2018)."""

    def __init__(self, in_features: int, out_features: int, heads: int = 1,
                 dropout: float = 0.0, concat: bool = True, bias: bool = True,
                 negative_slope: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout_p = dropout

        from vectorphile._core.ndarray import uniform as _nd_uniform, zeros as _nd_zeros
        k = math.sqrt(1.0 / in_features)
        self.lin = Tensor(_nd_uniform(-k, k, (in_features, heads * out_features)), requires_grad=True)
        self.att = Tensor(_nd_uniform(-k, k, (1, heads, 2 * out_features)), requires_grad=True)
        if bias:
            if concat:
                self.bias_param = Tensor(_nd_zeros(heads * out_features, dtype='float32'), requires_grad=True)
            else:
                self.bias_param = Tensor(_nd_zeros(out_features, dtype='float32'), requires_grad=True)
        else:
            object.__setattr__(self, 'bias_param', None)

    def forward(self, x: Tensor, edge_index) -> Tensor:
        xd = _np.asarray(x._data)
        N = xd.shape[0]
        ei = _np.asarray(edge_index._data) if isinstance(edge_index, Tensor) else _np.asarray(edge_index)
        H, out_f = self.heads, self.out_features

        # Linear transform
        h = (xd @ _np.asarray(self.lin._data)).reshape(N, H, out_f)

        src, dst = ei[0], ei[1]
        att_data = _np.asarray(self.att._data).reshape(H, 2 * out_f)
        # Concat source and destination features
        alpha_raw = _np.concatenate([h[src], h[dst]], axis=-1)  # (E, H, 2*out_f)
        # Compute attention scores
        alpha = (alpha_raw * att_data[None]).sum(axis=-1)  # (E, H)
        # Leaky relu
        alpha = _np.where(alpha >= 0, alpha, self.negative_slope * alpha)
        # Softmax per destination node
        alpha_exp = _np.exp(alpha - alpha.max())
        alpha_sum = _np.zeros((N, H), dtype='float32')
        for e, d in enumerate(dst):
            alpha_sum[d] += alpha_exp[e]
        alpha_norm = alpha_exp / _np.maximum(alpha_sum[dst], 1e-6)

        # Aggregate
        out = _np.zeros((N, H, out_f), dtype='float32')
        for e, (s, d) in enumerate(zip(src, dst)):
            out[d] += alpha_norm[e, :, None] * h[s]

        if self.concat:
            out = out.reshape(N, H * out_f)
        else:
            out = out.mean(axis=1)

        if self.bias_param is not None:
            out = out + _np.asarray(self.bias_param._data)
        return Tensor(out)


class SAGEConv(Module):
    """GraphSAGE layer (Hamilton et al., 2017): mean aggregation."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        from vectorphile.nn.layers import Linear
        self.lin_self = Linear(in_features, out_features, bias=False)
        self.lin_neigh = Linear(in_features, out_features, bias=bias)

    def forward(self, x: Tensor, edge_index) -> Tensor:
        xd = _np.asarray(x._data)
        N = xd.shape[0]
        ei = _np.asarray(edge_index._data) if isinstance(edge_index, Tensor) else _np.asarray(edge_index)
        src, dst = ei[0], ei[1]
        # Mean-aggregate neighbors
        agg = _np.zeros_like(xd)
        cnt = _np.zeros(N, dtype='float32')
        for s, d in zip(src, dst):
            agg[d] += xd[s]
            cnt[d] += 1
        cnt = _np.maximum(cnt, 1)
        agg = agg / cnt[:, None]
        return self.lin_self(x) + self.lin_neigh(Tensor(agg))


class EdgeConv(Module):
    """EdgeConv (Wang et al., 2019): dynamic graph CNN for point clouds."""

    def __init__(self, in_features: int, out_features: int, k: int = 20):
        super().__init__()
        self.k = k
        from vectorphile.nn.layers import Linear
        from vectorphile.nn.activations import ReLU
        from vectorphile.nn.layers import BatchNorm1d
        self.mlp = Module.__new__(Module)
        Module.__init__(self.mlp)
        self.fc1 = Linear(in_features * 2, out_features)
        self.fc2 = Linear(out_features, out_features)

    def _knn(self, x, k):
        """Return indices of k nearest neighbors for each point."""
        d = _np.asarray(x._data)
        N = d.shape[0]
        dist = _np.sum((d[:, None] - d[None, :]) ** 2, axis=-1)
        idx = _np.argsort(dist, axis=1)[:, 1:k+1]
        return idx

    def forward(self, x: Tensor) -> Tensor:
        d = _np.asarray(x._data)
        N, C = d.shape
        idx = self._knn(x, self.k)  # (N, k)
        # Edge features: [x_i, x_j - x_i] for each neighbor
        x_j = d[idx.ravel()].reshape(N, self.k, C)
        x_i = d[:, None, :].repeat(self.k, axis=1)
        edge_feat = _np.concatenate([x_i, x_j - x_i], axis=-1).reshape(N * self.k, 2 * C)
        h = self.fc1(Tensor(edge_feat)).relu()
        h = self.fc2(h)
        # Max pool over neighbors
        h_pool = _np.asarray(h._data).reshape(N, self.k, -1).max(axis=1)
        return Tensor(h_pool)


# ── State Space Models ────────────────────────────────────────────────────────

class S4Block(Module):
    """Simplified S4 State Space Model block (Gu et al., 2021).

    Uses diagonal plus rank-1 (DPLR) parametrization approximation.
    """

    def __init__(self, d_model: int, d_state: int = 64, l_max: int = 1024, bidirectional: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.l_max = l_max
        self.bidirectional = bidirectional

        from vectorphile._core.ndarray import uniform as _nd_uniform, zeros as _nd_zeros
        from vectorphile.nn.layers import Linear, LayerNorm
        k = math.sqrt(1.0 / d_model)
        # State transition: A (diagonal), B (input), C (output)
        self.A_re = Tensor(_nd_uniform(-1, 0, (d_model, d_state)), requires_grad=True)
        self.A_im = Tensor(_nd_uniform(-math.pi, math.pi, (d_model, d_state)), requires_grad=True)
        self.B = Tensor(_nd_uniform(-k, k, (d_model, d_state)), requires_grad=True)
        self.C = Tensor(_nd_uniform(-k, k, (d_model, d_state)), requires_grad=True)
        self.D = Tensor(_nd_uniform(-k, k, (d_model,)), requires_grad=True)
        self.log_dt = Tensor(_nd_uniform(-5, -1, (d_model,)), requires_grad=True)
        self.norm = LayerNorm(d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        from vectorphile._core.ndarray import array as _nd_s4
        # x: (B, L, D)
        B_sz, L, D = x.shape
        xd = x._data
        xfl = xd.to_flat_list() if hasattr(xd, 'to_flat_list') else list(xd)
        def _fl(d):
            if hasattr(d, 'to_flat_list'): return d.to_flat_list()
            return list(d)
        dt_fl = [math.exp(max(-88.0, min(88.0, v))) for v in _fl(self.log_dt._data)]  # (D,)
        A_re_fl = _fl(self.A_re._data); A_im_fl = _fl(self.A_im._data)
        B_fl = _fl(self.B._data); C_fl = _fl(self.C._data); D_fl = _fl(self.D._data)
        N_st = self.d_state
        # A_d, B_d as complex lists: index [d*N_st+n]
        A_d = [complex(math.cos(A_im_fl[d*N_st+n])*math.exp(dt_fl[d]*A_re_fl[d*N_st+n]),
                       math.sin(A_im_fl[d*N_st+n])*math.exp(dt_fl[d]*A_re_fl[d*N_st+n]))
               for d in range(D) for n in range(N_st)]
        B_d_c = [complex(B_fl[d*N_st+n]*dt_fl[d], 0.0) for d in range(D) for n in range(N_st)]
        # SSM scan (slow pure Python)
        h = [complex(0, 0)]*(D*N_st)
        K_kern = []  # K_kern[t][d] = real part of sum_n h[d,n]*C[d,n]
        for t in range(L):
            k_t = [sum(h[d*N_st+n].real * C_fl[d*N_st+n] for n in range(N_st)) for d in range(D)]
            K_kern.append(k_t)
            h = [A_d[d*N_st+n]*h[d*N_st+n] + B_d_c[d*N_st+n] for d in range(D) for n in range(N_st)]
        # Direct time-domain convolution y[b,t,d] = sum_tau K[tau,d]*x[b,t-tau,d] (causal)
        y_fl = [0.0]*(B_sz*L*D)
        for b in range(B_sz):
            for t in range(L):
                for d in range(D):
                    s = 0.0
                    for tau in range(t+1):
                        s += K_kern[tau][d] * xfl[b*L*D + (t-tau)*D + d]
                    y_fl[b*L*D + t*D + d] = s + xfl[b*L*D + t*D + d] * D_fl[d]
        return self.output_proj(Tensor(_nd_s4(y_fl).reshape((B_sz, L, D))))


# ── Signal processing layers ──────────────────────────────────────────────────

class CausalConv1d(Module):
    """Causal 1-D convolution: output at t depends only on inputs at t' <= t."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 1, bias: bool = True):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        from vectorphile.nn.layers import Conv1d
        self.conv = Conv1d(in_channels, out_channels, kernel_size,
                           padding=self.padding, dilation=dilation, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        # Remove future context (causal shift)
        if self.padding > 0:
            d = _np.asarray(out._data)
            return Tensor(d[:, :, :-self.padding])
        return out


class WaveNetBlock(Module):
    """WaveNet residual block with gated activation and 1×1 skip connections."""

    def __init__(self, residual_channels: int, skip_channels: int,
                 dilation: int = 1, kernel_size: int = 2):
        super().__init__()
        self.dilated_conv = CausalConv1d(residual_channels, 2 * residual_channels,
                                         kernel_size=kernel_size, dilation=dilation)
        from vectorphile.nn.layers import Conv1d
        self.skip_conv = Conv1d(residual_channels, skip_channels, 1)
        self.res_conv = Conv1d(residual_channels, residual_channels, 1)

    def forward(self, x: Tensor, skip_acc=None) -> tuple:
        h = self.dilated_conv(x)
        hd = _np.asarray(h._data)
        # Gated activation
        tanh_part = _np.tanh(hd[:, :hd.shape[1]//2, :])
        sig_part = 1 / (1 + _np.exp(-hd[:, hd.shape[1]//2:, :]))
        z = Tensor(tanh_part * sig_part)
        skip = self.skip_conv(z)
        res = x + self.res_conv(z)
        if skip_acc is not None:
            skip_acc = skip_acc + skip
        return res, skip if skip_acc is None else skip_acc


__all__ = [
    'PReLU', 'GLU', 'Swish', 'Snake',
    'ConvTranspose1d', 'DepthwiseSepConv2d', 'DepthwiseSepConv1d',
    'PixelShuffle', 'ChannelShuffle', 'Unfold', 'Fold',
    'MultiQueryAttention', 'CrossAttention', 'LinearAttention',
    'FilterResponseNorm', 'PowerNorm',
    'StochasticDepth', 'DropBlock2d',
    'RelativePositionEncoding', 'ALiBiPositionBias',
    'Lambda', 'Residual', 'PreNorm', 'Scale',
    'MixerMLP', 'GatedMLP', 'PatchEmbed', 'VisionTransformerBlock',
    'GCNConv', 'GATConv', 'SAGEConv', 'EdgeConv',
    'S4Block', 'Mamba',
    'CausalConv1d', 'WaveNetBlock',
    'Softsign', 'Hardsigmoid', 'Softmin', 'Softmax2d',
]


class Softsign(Module):
    """Softsign activation: x / (1 + |x|)."""
    def forward(self, x: Tensor) -> Tensor:
        flat = x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data)
        out = [v / (1.0 + abs(v)) for v in flat]
        from vectorphile._core.ndarray import array as _nd
        return Tensor(_nd(out, x._data.shape if hasattr(x._data, 'shape') else (len(out),)))

    def __repr__(self): return 'Softsign()'


class Hardsigmoid(Module):
    """Hard sigmoid: clamp((x + 3) / 6, 0, 1)."""
    def forward(self, x: Tensor) -> Tensor:
        flat = x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data)
        out = [max(0.0, min(1.0, (v + 3.0) / 6.0)) for v in flat]
        from vectorphile._core.ndarray import array as _nd
        return Tensor(_nd(out, x._data.shape if hasattr(x._data, 'shape') else (len(out),)))

    def __repr__(self): return 'Hardsigmoid()'


class Softmin(Module):
    """Softmin: softmax of negated input."""
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        from vectorphile.nn.functional import softmax as _sm
        # Negate x then apply softmax
        flat = x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data)
        neg = [-v for v in flat]
        from vectorphile._core.ndarray import array as _nd
        neg_t = Tensor(_nd(neg, x._data.shape if hasattr(x._data, 'shape') else (len(neg),)))
        return _sm(neg_t, dim=self.dim)

    def __repr__(self): return f'Softmin(dim={self.dim})'


class Softmax2d(Module):
    """Apply softmax over channels (dim=1) for 4-D (N, C, H, W) inputs."""
    def forward(self, x: Tensor) -> Tensor:
        from vectorphile.nn.functional import softmax as _sm
        return _sm(x, dim=1)

    def __repr__(self): return 'Softmax2d()'
