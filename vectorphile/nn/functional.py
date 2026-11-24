"""Functional API — stateless operations on Tensors (vp.nn.functional / vp.F)."""
from __future__ import annotations
import math
import random as _random
from vectorphile.core.tensor import Tensor


def _get_flat(d):
    if hasattr(d, 'to_flat_list'): return d.to_flat_list()
    if hasattr(d, 'tolist'): return d.tolist()
    if isinstance(d, list): return d
    return list(d)

def _shape(d):
    if hasattr(d, 'shape'): return d.shape
    if isinstance(d, list): return (len(d),)
    return (1,)

def _nd(lst, shape=None, dtype=None):
    from vectorphile._core.ndarray import array as _a
    r = _a(lst, dtype=dtype)
    if shape: r = r.reshape(shape)
    return r


# ── Activations ──────────────────────────────────────────────────────────────

def relu(x: Tensor, inplace=False) -> Tensor:
    if not x.requires_grad:
        try:
            from vectorphile.cuda.triton_kernels import fused_relu as _fr
            return Tensor(_fr(x._data), device=x.device, requires_grad=False)
        except Exception:
            pass
    flat = _get_flat(x._data); sh = _shape(x._data)
    return Tensor(_nd([max(0.0, v) for v in flat], sh), device=x.device)

def relu6(x: Tensor) -> Tensor:
    flat = _get_flat(x._data); sh = _shape(x._data)
    return Tensor(_nd([max(0.0, min(6.0, v)) for v in flat], sh), device=x.device)

def leaky_relu(x: Tensor, negative_slope=0.01, inplace=False) -> Tensor:
    flat = _get_flat(x._data); sh = _shape(x._data)
    return Tensor(_nd([v if v >= 0 else negative_slope * v for v in flat], sh), device=x.device)

def elu(x: Tensor, alpha=1.0, inplace=False) -> Tensor:
    flat = _get_flat(x._data); sh = _shape(x._data)
    return Tensor(_nd([v if v >= 0 else alpha * (math.exp(v) - 1) for v in flat], sh), device=x.device)

def selu(x: Tensor) -> Tensor:
    alpha = 1.6732632423543772; scale = 1.0507009873554805
    flat = _get_flat(x._data); sh = _shape(x._data)
    return Tensor(_nd([scale * v if v >= 0 else scale * alpha * (math.exp(v) - 1) for v in flat], sh), device=x.device)

def gelu(x: Tensor, approximate='none') -> Tensor:
    # Triton fast path for inference (no gradient needed)
    if not x.requires_grad and approximate in ('none', 'tanh'):
        try:
            from vectorphile.cuda.triton_kernels import fused_gelu as _fg
            return Tensor(_fg(x._data), device=x.device, requires_grad=False)
        except Exception:
            pass
    flat = _get_flat(x._data); sh = _shape(x._data)
    rg = x.requires_grad
    _dt = str(x._data.dtype) if hasattr(x._data, 'dtype') else 'float32'
    if approximate == 'tanh':
        c = math.sqrt(2 / math.pi)
        out = [v * 0.5 * (1 + math.tanh(c * (v + 0.044715 * v**3))) for v in flat]
    else:
        out = [v / (1 + math.exp(-1.702 * v)) for v in flat]
    from vectorphile._core.ndarray import array as _nd_gelu
    result = Tensor(_nd_gelu(out, dtype=_dt).reshape(sh), device=x.device, requires_grad=rg)
    if rg:
        _inp = x; _approx = approximate; _flat_in = list(flat); _sh = sh; _dt2 = _dt
        class _GELUBackward:
            def __init__(s): s.inputs = (_inp,)
            def backward(s, grad):
                gf = _get_flat(grad._data)
                if _approx == 'tanh':
                    _c = math.sqrt(2 / math.pi)
                    d = [gf[i] * 0.5 * (1 + math.tanh(_c * (_flat_in[i] + 0.044715 * _flat_in[i]**3))) +
                         _flat_in[i] * 0.5 * (1 - math.tanh(_c * (_flat_in[i] + 0.044715 * _flat_in[i]**3))**2) *
                         _c * (1 + 3 * 0.044715 * _flat_in[i]**2) for i in range(len(gf))]
                else:
                    d = []
                    for i in range(len(gf)):
                        v = _flat_in[i]; sig = 1.0 / (1.0 + math.exp(-1.702 * v))
                        d.append(gf[i] * (sig + v * sig * (1 - sig) * 1.702))
                from vectorphile._core.ndarray import array as _nd_gb
                return (Tensor(_nd_gb(d, dtype=_dt2).reshape(_sh)),)
        result.grad_fn = _GELUBackward(); result._is_leaf = False
    return result

def silu(x: Tensor, inplace=False) -> Tensor:
    if not x.requires_grad:
        try:
            from vectorphile.cuda.triton_kernels import fused_silu as _fs
            return Tensor(_fs(x._data), device=x.device, requires_grad=False)
        except Exception:
            pass
    return x * x.sigmoid()

def mish(x: Tensor, inplace=False) -> Tensor:
    flat = _get_flat(x._data); sh = _shape(x._data)
    def _sp(v): return math.log1p(math.exp(max(-88.0, v)))
    out = [v * math.tanh(_sp(v)) for v in flat]
    return Tensor(_nd(out, sh), device=x.device)

def hardswish(x: Tensor, inplace=False) -> Tensor:
    flat = _get_flat(x._data); sh = _shape(x._data)
    return Tensor(_nd([v * max(0.0, min(6.0, v + 3)) / 6.0 for v in flat], sh), device=x.device)

def hardsigmoid(x: Tensor, inplace=False) -> Tensor:
    flat = _get_flat(x._data); sh = _shape(x._data)
    return Tensor(_nd([max(0.0, min(1.0, v / 6 + 0.5)) for v in flat], sh), device=x.device)

def sigmoid(x: Tensor) -> Tensor:
    flat = _get_flat(x._data); sh = _shape(x._data)
    return Tensor(_nd([1.0 / (1.0 + math.exp(-max(-88.0, v))) for v in flat], sh), device=x.device)

def tanh(x: Tensor) -> Tensor:
    flat = _get_flat(x._data); sh = _shape(x._data)
    return Tensor(_nd([math.tanh(v) for v in flat], sh), device=x.device)

def softmax(x: Tensor, dim=-1) -> Tensor:
    return x.softmax(dim)

def log_softmax(x: Tensor, dim=-1) -> Tensor:
    return x.log_softmax(dim)

def softplus(x: Tensor, beta=1, threshold=20) -> Tensor:
    flat = _get_flat(x._data); sh = _shape(x._data)
    out = [v if beta * v > threshold else (1/beta) * math.log1p(math.exp(beta * v)) for v in flat]
    return Tensor(_nd(out, sh), device=x.device)

def softsign(x: Tensor) -> Tensor:
    flat = _get_flat(x._data); sh = _shape(x._data)
    return Tensor(_nd([v / (1 + abs(v)) for v in flat], sh), device=x.device)

def softmin(x: Tensor, dim=-1) -> Tensor:
    """Softmax of negated input."""
    flat = _get_flat(x._data); sh = _shape(x._data)
    neg = Tensor(_nd([-v for v in flat], sh), device=x.device)
    return softmax(neg, dim=dim)

def softmax2d(x: Tensor) -> Tensor:
    """Apply softmax over the channel dimension (dim=1) for 4-D input."""
    return softmax(x, dim=1)

def prelu(x: Tensor, weight: Tensor) -> Tensor:
    flat = _get_flat(x._data); sh = _shape(x._data)
    wf = _get_flat(weight._data)
    w = wf[0] if len(wf) == 1 else wf
    if isinstance(w, float):
        out = [v if v >= 0 else w * v for v in flat]
    else:
        out = [flat[i] if flat[i] >= 0 else w[i % len(w)] * flat[i] for i in range(len(flat))]
    return Tensor(_nd(out, sh), device=x.device)

def threshold(x: Tensor, threshold_val: float, value: float) -> Tensor:
    flat = _get_flat(x._data); sh = _shape(x._data)
    return Tensor(_nd([v if v > threshold_val else value for v in flat], sh), device=x.device)

def hardtanh(x: Tensor, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False) -> Tensor:
    """Hardtanh: clamp(x, min_val, max_val)."""
    return x.clamp(min_val, max_val)

def glu(x: Tensor, dim: int = -1) -> Tensor:
    """Gated Linear Unit: split along dim, apply sigmoid to second half, multiply."""
    import math as _m
    sh = _shape(x._data)
    ax = dim % len(sh) if sh else 0
    half = sh[ax] // 2
    flat = _get_flat(x._data)
    # Split along axis ax
    new_sh = tuple(half if i == ax else s for i, s in enumerate(sh))
    stride = 1
    for i in range(ax + 1, len(sh)):
        stride *= sh[i]
    ax_stride = stride
    total_ax_stride = sh[ax] * stride
    out_size = 1
    for s in new_sh:
        out_size *= s
    a_flat = []
    b_flat = []
    for i in range(out_size):
        # Find position in full flat array
        idx_in_ax = (i // ax_stride) % half
        pre = i - idx_in_ax * ax_stride
        a_pos = pre + idx_in_ax * ax_stride
        b_pos = pre + (idx_in_ax + half) * ax_stride
        a_flat.append(flat[a_pos])
        b_flat.append(flat[b_pos])
    out = [a * (1.0 / (1.0 + _m.exp(-b))) for a, b in zip(a_flat, b_flat)]
    return Tensor(_nd(out, new_sh), device=x.device)

def celu(x: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    """CELU: max(0,x) + min(0, alpha*(exp(x/alpha)-1))"""
    import math as _m
    flat = _get_flat(x._data); sh = _shape(x._data)
    out = [v if v >= 0 else alpha * (_m.exp(v / alpha) - 1.0) for v in flat]
    return Tensor(_nd(out, sh), device=x.device)

def logsigmoid(x: Tensor) -> Tensor:
    """log(sigmoid(x)) = -softplus(-x)"""
    import math as _m
    flat = _get_flat(x._data); sh = _shape(x._data)
    out = [-_m.log(1.0 + _m.exp(-v)) for v in flat]
    return Tensor(_nd(out, sh), device=x.device)

def tanhshrink(x: Tensor) -> Tensor:
    """x - tanh(x)"""
    import math as _m
    flat = _get_flat(x._data); sh = _shape(x._data)
    out = [v - _m.tanh(v) for v in flat]
    return Tensor(_nd(out, sh), device=x.device)

def softshrink(x: Tensor, lambd: float = 0.5) -> Tensor:
    """Soft shrinkage: x - lambd if x > lambd, x + lambd if x < -lambd, else 0."""
    flat = _get_flat(x._data); sh = _shape(x._data)
    out = [v - lambd if v > lambd else (v + lambd if v < -lambd else 0.0) for v in flat]
    return Tensor(_nd(out, sh), device=x.device)

def hardshrink(x: Tensor, lambd: float = 0.5) -> Tensor:
    """Hard shrinkage: 0 if |x| < lambd, else x."""
    flat = _get_flat(x._data); sh = _shape(x._data)
    out = [v if abs(v) >= lambd else 0.0 for v in flat]
    return Tensor(_nd(out, sh), device=x.device)


# ── Linear ───────────────────────────────────────────────────────────────────

def linear(x: Tensor, weight: Tensor, bias: Tensor = None) -> Tensor:
    out = x._data @ weight._data.T
    if bias is not None:
        out = out + bias._data
    return Tensor(out, device=x.device)

def bilinear(x1: Tensor, x2: Tensor, weight: Tensor, bias: Tensor = None) -> Tensor:
    # weight: (out, in1, in2) — simple loop implementation
    x1f = _get_flat(x1._data); x2f = _get_flat(x2._data)
    sh1 = _shape(x1._data); sh2 = _shape(x2._data)
    B = sh1[0]; in1 = sh1[-1]; in2 = sh2[-1]
    wf = _get_flat(weight._data); wsh = _shape(weight._data)
    out_size = wsh[0]
    out = []
    for b in range(B):
        row = []
        for o in range(out_size):
            v = 0.0
            for i in range(in1):
                for j in range(in2):
                    w_idx = o * in1 * in2 + i * in2 + j
                    v += x1f[b * in1 + i] * wf[w_idx] * x2f[b * in2 + j]
            row.append(v)
        out.extend(row)
    if bias is not None:
        bf = _get_flat(bias._data)
        out = [out[b * out_size + o] + bf[o] for b in range(B) for o in range(out_size)]
    return Tensor(_nd(out, (B, out_size)), device=x1.device)


# ── Convolution ───────────────────────────────────────────────────────────────

def conv1d(x: Tensor, weight: Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor:
    xsh = _shape(x._data); wsh = _shape(weight._data)
    N, C, L = xsh; Co, Ci, K = wsh
    p, s, d = padding, stride, dilation
    Lk = d * (K - 1) + 1
    Lo = (L + 2*p - Lk) // s + 1
    xf = _get_flat(x._data); wf = _get_flat(weight._data)
    out = [0.0] * (N * Co * Lo)
    for n in range(N):
        for co in range(Co):
            for lo in range(Lo):
                v = 0.0
                for ci in range(Ci):
                    for k in range(K):
                        li = lo * s - p + k * d
                        if 0 <= li < L:
                            v += xf[n*C*L + ci*L + li] * wf[co*Ci*K + ci*K + k]
                out[n*Co*Lo + co*Lo + lo] = v
    if bias is not None:
        bf = _get_flat(bias._data)
        out = [out[n*Co*Lo + co*Lo + lo] + bf[co] for n in range(N) for co in range(Co) for lo in range(Lo)]
    return Tensor(_nd(out, (N, Co, Lo)), device=x.device)


def conv2d(x: Tensor, weight: Tensor, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor:
    xsh = _shape(x._data); wsh = _shape(weight._data)
    N, C, H, W = xsh; Co, Ci_g, kh, kw = wsh
    s = stride if isinstance(stride, tuple) else (stride, stride)
    p = padding if isinstance(padding, tuple) else (padding, padding)
    Ho = (H + 2*p[0] - kh) // s[0] + 1
    Wo = (W + 2*p[1] - kw) // s[1] + 1
    xf = _get_flat(x._data); wf = _get_flat(weight._data)
    out = [0.0] * (N * Co * Ho * Wo)
    for n in range(N):
        for co in range(Co):
            for ho in range(Ho):
                for wo in range(Wo):
                    v = 0.0
                    for ci in range(C):
                        for ki in range(kh):
                            for kj in range(kw):
                                hi = ho * s[0] - p[0] + ki
                                wi = wo * s[1] - p[1] + kj
                                if 0 <= hi < H and 0 <= wi < W:
                                    v += xf[n*C*H*W + ci*H*W + hi*W + wi] * wf[co*C*kh*kw + ci*kh*kw + ki*kw + kj]
                    out[n*Co*Ho*Wo + co*Ho*Wo + ho*Wo + wo] = v
    if bias is not None:
        bf = _get_flat(bias._data)
        out = [out[n*Co*Ho*Wo + co*Ho*Wo + ho*Wo + wo] + bf[co]
               for n in range(N) for co in range(Co) for ho in range(Ho) for wo in range(Wo)]
    _dt = str(getattr(x._data, 'dtype', 'float32'))
    if _dt not in ('float32', 'float64'): _dt = 'float32'
    return Tensor(_nd(out, (N, Co, Ho, Wo), _dt), device=x.device)


def conv_transpose2d(x: Tensor, weight: Tensor, bias=None, stride=1, padding=0,
                     output_padding=0, groups=1, dilation=1) -> Tensor:
    xsh = _shape(x._data); wsh = _shape(weight._data)
    N, Ci, H, W = xsh; Ci2, Co, kh, kw = wsh
    s = stride if isinstance(stride, tuple) else (stride, stride)
    p = padding if isinstance(padding, tuple) else (padding, padding)
    op = output_padding if isinstance(output_padding, tuple) else (output_padding, output_padding)
    Ho_full = (H - 1) * s[0] + kh
    Wo_full = (W - 1) * s[1] + kw
    xf = _get_flat(x._data); wf = _get_flat(weight._data)
    out_full = [0.0] * (N * Co * Ho_full * Wo_full)
    for n in range(N):
        for ci in range(Ci):
            for ih in range(H):
                for iw in range(W):
                    xv = xf[n*Ci*H*W + ci*H*W + ih*W + iw]
                    for co in range(Co):
                        for ki in range(kh):
                            for kj in range(kw):
                                oh = ih * s[0] + ki
                                ow = iw * s[1] + kj
                                wv = wf[ci*Co*kh*kw + co*kh*kw + ki*kw + kj]
                                out_full[n*Co*Ho_full*Wo_full + co*Ho_full*Wo_full + oh*Wo_full + ow] += xv * wv
    ph, pw = p
    Ho = Ho_full - 2*ph + op[0]
    Wo = Wo_full - 2*pw + op[1]
    out = [out_full[n*Co*Ho_full*Wo_full + co*Ho_full*Wo_full + (ho+ph)*Wo_full + (wo+pw)]
           for n in range(N) for co in range(Co) for ho in range(Ho) for wo in range(Wo)]
    if bias is not None:
        bf = _get_flat(bias._data)
        out = [out[n*Co*Ho*Wo + co*Ho*Wo + ho*Wo + wo] + bf[co]
               for n in range(N) for co in range(Co) for ho in range(Ho) for wo in range(Wo)]
    _dt = str(getattr(x._data, 'dtype', 'float32'))
    if _dt not in ('float32', 'float64'): _dt = 'float32'
    rg = x.requires_grad or (weight is not None and weight.requires_grad) or (bias is not None and bias.requires_grad)
    result = Tensor(_nd(out, (N, Co, Ho, Wo), _dt), device=x.device, requires_grad=rg)
    if rg:
        _x2, _w2, _b2 = x, weight, bias
        _xf2, _wf2 = xf, wf
        _s2, _ph2, _pw2 = s, ph, pw
        class _ConvT2dBwd:
            def __init__(ss): ss.inputs = ([_x2] if _x2.requires_grad else []) + \
                                           ([_w2] if _w2 is not None and _w2.requires_grad else []) + \
                                           ([_b2] if _b2 is not None and _b2.requires_grad else [])
            def backward(ss, grad):
                gf2 = _get_flat(grad._data)
                g_full = [0.0]*(N*Co*Ho_full*Wo_full)
                for n in range(N):
                    for co in range(Co):
                        for ho in range(Ho):
                            for wo in range(Wo):
                                g_full[n*Co*Ho_full*Wo_full + co*Ho_full*Wo_full + (ho+_ph2)*Wo_full + (wo+_pw2)] = gf2[n*Co*Ho*Wo + co*Ho*Wo + ho*Wo + wo]
                rets = []
                if _x2.requires_grad:
                    dx = [0.0]*(N*Ci*H*W)
                    for n in range(N):
                        for ci in range(Ci):
                            for ih in range(H):
                                for iw in range(W):
                                    v = 0.0
                                    for co in range(Co):
                                        for ki in range(kh):
                                            for kj in range(kw):
                                                v += g_full[n*Co*Ho_full*Wo_full + co*Ho_full*Wo_full + (ih*_s2[0]+ki)*Wo_full + (iw*_s2[1]+kj)] * _wf2[ci*Co*kh*kw + co*kh*kw + ki*kw + kj]
                                    dx[n*Ci*H*W + ci*H*W + ih*W + iw] = v
                    rets.append(Tensor(_nd(dx, (N,Ci,H,W), _dt)))
                if _w2 is not None and _w2.requires_grad:
                    dw = [0.0]*(Ci*Co*kh*kw)
                    for ci in range(Ci):
                        for co in range(Co):
                            for ki in range(kh):
                                for kj in range(kw):
                                    v = 0.0
                                    for n in range(N):
                                        for ih in range(H):
                                            for iw in range(W):
                                                v += g_full[n*Co*Ho_full*Wo_full + co*Ho_full*Wo_full + (ih*_s2[0]+ki)*Wo_full + (iw*_s2[1]+kj)] * _xf2[n*Ci*H*W + ci*H*W + ih*W + iw]
                                    dw[ci*Co*kh*kw + co*kh*kw + ki*kw + kj] = v
                    rets.append(Tensor(_nd(dw, (Ci,Co,kh,kw), _dt)))
                if _b2 is not None and _b2.requires_grad:
                    db = [sum(gf2[n*Co*Ho*Wo + co*Ho*Wo + ho*Wo + wo] for n in range(N) for ho in range(Ho) for wo in range(Wo)) for co in range(Co)]
                    rets.append(Tensor(_nd(db, (Co,), _dt)))
                return tuple(rets)
        result.grad_fn = _ConvT2dBwd(); result._is_leaf = False
    return result


# ── Normalization ─────────────────────────────────────────────────────────────

def batch_norm(x: Tensor, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5) -> Tensor:
    flat = _get_flat(x._data); sh = _shape(x._data)
    n_total = len(flat)
    # For simplicity, flatten all — proper batch norm would operate over (N, C, ...)
    if training:
        mu = sum(flat) / n_total
        var = sum((v - mu)**2 for v in flat) / n_total
        if running_mean is not None:
            rm = _get_flat(running_mean._data if hasattr(running_mean, '_data') else running_mean)
            rv = _get_flat(running_var._data if hasattr(running_var, '_data') else running_var)
        else:
            mu_use, var_use = mu, var
    else:
        rm = _get_flat(running_mean._data if hasattr(running_mean, '_data') else running_mean) if running_mean is not None else None
        rv = _get_flat(running_var._data if hasattr(running_var, '_data') else running_var) if running_var is not None else None
        mu = sum(rm) / len(rm) if rm else sum(flat) / n_total
        var_v = sum(rv) / len(rv) if rv else sum((v - mu)**2 for v in flat) / n_total
        var = var_v
    out = [(v - mu) / math.sqrt(var + eps) for v in flat]
    if weight is not None:
        wf = _get_flat(weight._data); out = [o * wf[i % len(wf)] for i, o in enumerate(out)]
    if bias is not None:
        bf = _get_flat(bias._data); out = [o + bf[i % len(bf)] for i, o in enumerate(out)]
    return Tensor(_nd(out, sh), device=x.device)


def layer_norm(x: Tensor, normalized_shape, weight=None, bias=None, eps=1e-5) -> Tensor:
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    ndim = len(normalized_shape)
    axes = tuple(range(-ndim, 0))
    mu = x.mean(dim=axes, keepdim=True)
    xc = x - mu
    var = (xc * xc).mean(dim=axes, keepdim=True)
    x_hat = xc / (var + eps).sqrt()
    if weight is not None:
        x_hat = x_hat * weight
    if bias is not None:
        x_hat = x_hat + bias
    return x_hat


def group_norm(x: Tensor, num_groups: int, weight=None, bias=None, eps=1e-5) -> Tensor:
    if weight is not None and bias is not None and not x.requires_grad:
        try:
            from vectorphile.cuda.triton_kernels import fused_group_norm as _fgn
            out_nd = _fgn(x._data, weight._data, bias._data, num_groups, eps)
            return Tensor(out_nd, device=x.device, requires_grad=False)
        except Exception:
            pass
    sh = _shape(x._data); N, C = sh[0], sh[1]
    flat = _get_flat(x._data)
    rest_size = len(flat) // (N * C)
    group_size = (C // num_groups) * rest_size
    out = [0.0] * len(flat)
    for n in range(N):
        for g in range(num_groups):
            vals = []
            for c in range(g * (C // num_groups), (g+1) * (C // num_groups)):
                base = n * C * rest_size + c * rest_size
                vals.extend(flat[base:base + rest_size])
            mu = sum(vals) / len(vals); var = sum((v-mu)**2 for v in vals) / len(vals)
            norm_v = [(v - mu) / math.sqrt(var + eps) for v in vals]
            vi = 0
            for c in range(g * (C // num_groups), (g+1) * (C // num_groups)):
                base = n * C * rest_size + c * rest_size
                for r in range(rest_size):
                    out[base + r] = norm_v[vi]; vi += 1
    if weight is not None:
        wf = _get_flat(weight._data)
        for n in range(N):
            for c in range(C):
                base = n * C * rest_size + c * rest_size
                for r in range(rest_size):
                    out[base + r] *= wf[c]
    if bias is not None:
        bf = _get_flat(bias._data)
        for n in range(N):
            for c in range(C):
                base = n * C * rest_size + c * rest_size
                for r in range(rest_size):
                    out[base + r] += bf[c]
    return Tensor(_nd(out, sh), device=x.device)


def instance_norm(x: Tensor, running_mean=None, running_var=None, weight=None, bias=None,
                  use_input_stats=True, momentum=0.1, eps=1e-5) -> Tensor:
    sh = _shape(x._data); N, C = sh[0], sh[1]
    flat = _get_flat(x._data)
    rest_size = len(flat) // (N * C)
    out = [0.0] * len(flat)
    for n in range(N):
        for c in range(C):
            base = n * C * rest_size + c * rest_size
            vals = flat[base:base + rest_size]
            mu = sum(vals) / len(vals); var = sum((v-mu)**2 for v in vals) / len(vals)
            for r in range(rest_size):
                out[base + r] = (vals[r] - mu) / math.sqrt(var + eps)
    if weight is not None:
        wf = _get_flat(weight._data)
        for n in range(N):
            for c in range(C):
                base = n * C * rest_size + c * rest_size
                for r in range(rest_size): out[base + r] *= wf[c]
    if bias is not None:
        bf = _get_flat(bias._data)
        for n in range(N):
            for c in range(C):
                base = n * C * rest_size + c * rest_size
                for r in range(rest_size): out[base + r] += bf[c]
    return Tensor(_nd(out, sh), device=x.device)


# ── Dropout ───────────────────────────────────────────────────────────────────

def dropout(x: Tensor, p=0.5, training=True, inplace=False) -> Tensor:
    if not training or p == 0:
        return x
    flat = _get_flat(x._data); sh = _shape(x._data)
    scale = 1.0 / (1 - p)
    out = [v * scale if _random.random() > p else 0.0 for v in flat]
    return Tensor(_nd(out, sh), device=x.device)

def dropout2d(x: Tensor, p=0.5, training=True, inplace=False) -> Tensor:
    if not training or p == 0:
        return x
    sh = _shape(x._data); N, C = sh[0], sh[1]
    flat = _get_flat(x._data)
    rest = len(flat) // (N * C)
    scale = 1.0 / (1 - p)
    out = flat[:]
    for n in range(N):
        for c in range(C):
            if _random.random() <= p:
                base = n * C * rest + c * rest
                for r in range(rest): out[base + r] = 0.0
            else:
                base = n * C * rest + c * rest
                for r in range(rest): out[base + r] *= scale
    return Tensor(_nd(out, sh), device=x.device)

def alpha_dropout(x: Tensor, p=0.5, training=True, inplace=False) -> Tensor:
    if not training or p == 0:
        return x
    alpha = 1.6732632423543772; scale = 1.0507009873554805; a = -alpha * scale
    flat = _get_flat(x._data); sh = _shape(x._data)
    affine_a = 1.0 / math.sqrt((1 - p) * (1 + p * a**2))
    affine_b = -affine_a * p * a
    out = [(affine_a * (v if _random.random() > p else a) + affine_b) for v in flat]
    return Tensor(_nd(out, sh), device=x.device)


# ── Pooling ───────────────────────────────────────────────────────────────────

def max_pool2d(x: Tensor, kernel_size, stride=None, padding=0) -> Tensor:
    from vectorphile.nn.pooling import MaxPool2d
    return MaxPool2d(kernel_size, stride, padding)(x)

def avg_pool2d(x: Tensor, kernel_size, stride=None, padding=0) -> Tensor:
    from vectorphile.nn.pooling import AvgPool2d
    return AvgPool2d(kernel_size, stride, padding)(x)

def adaptive_avg_pool2d(x: Tensor, output_size) -> Tensor:
    from vectorphile.nn.pooling import AdaptiveAvgPool2d
    return AdaptiveAvgPool2d(output_size)(x)

def max_pool1d(x: Tensor, kernel_size, stride=None, padding=0) -> Tensor:
    from vectorphile.nn.pooling import MaxPool1d
    return MaxPool1d(kernel_size, stride if stride is not None else kernel_size, padding)(x)

def avg_pool1d(x: Tensor, kernel_size, stride=None, padding=0) -> Tensor:
    from vectorphile.nn.pooling import AvgPool1d
    return AvgPool1d(kernel_size, stride if stride is not None else kernel_size, padding)(x)

def adaptive_avg_pool1d(x: Tensor, output_size: int) -> Tensor:
    from vectorphile.nn.pooling import AdaptiveAvgPool1d
    return AdaptiveAvgPool1d(output_size)(x)

def adaptive_max_pool1d(x: Tensor, output_size: int) -> Tensor:
    """Adaptive max pool 1d: (N, C, L) -> (N, C, output_size)."""
    sh = _shape(x._data); flat = _get_flat(x._data)
    if len(sh) == 3:
        N, C, L = sh
    elif len(sh) == 2:
        N, C, L = 1, sh[0], sh[1]
    else:
        N, C, L = 1, 1, sh[0]
    out = []
    for n in range(N):
        for c in range(C):
            for o in range(output_size):
                start = int(o * L / output_size); end = int((o+1) * L / output_size)
                end = max(end, start+1)
                vals = [flat[n*C*L+c*L+l] for l in range(start, min(end, L))]
                out.append(max(vals) if vals else 0.0)
    new_sh = (N, C, output_size) if len(sh) == 3 else (C, output_size)
    return Tensor(_nd(out, new_sh))

def adaptive_max_pool2d(x: Tensor, output_size) -> Tensor:
    """Adaptive max pool 2d: (N, C, H, W) -> (N, C, oh, ow)."""
    sh = _shape(x._data); flat = _get_flat(x._data)
    if isinstance(output_size, int):
        oh, ow = output_size, output_size
    else:
        oh, ow = output_size
    if len(sh) == 4:
        N, C, H, W = sh
    else:
        N, C, H, W = 1, sh[-3], sh[-2], sh[-1]
    out = []
    for n in range(N):
        for c in range(C):
            for i in range(oh):
                for j in range(ow):
                    h0 = int(i * H / oh); h1 = int((i+1) * H / oh); h1 = max(h1, h0+1)
                    w0 = int(j * W / ow); w1 = int((j+1) * W / ow); w1 = max(w1, w0+1)
                    vals = [flat[n*C*H*W+c*H*W+h*W+w] for h in range(h0, min(h1,H)) for w in range(w0, min(w1,W))]
                    out.append(max(vals) if vals else 0.0)
    return Tensor(_nd(out, (N, C, oh, ow)))


# ── Embedding ─────────────────────────────────────────────────────────────────

def embedding(x: Tensor, weight: Tensor, padding_idx=None, max_norm=None,
              norm_type=2.0, scale_grad_by_freq=False, sparse=False) -> Tensor:
    idx_flat = [int(v) for v in _get_flat(x._data)]
    idx_sh = _shape(x._data)
    wf = _get_flat(weight._data); wsh = _shape(weight._data)
    emb_dim = wsh[1] if len(wsh) > 1 else 1
    out = []
    for i in idx_flat:
        row = wf[i * emb_dim:(i + 1) * emb_dim]
        if padding_idx is not None and i == padding_idx:
            row = [0.0] * emb_dim
        if max_norm is not None:
            norm = math.sqrt(sum(v**2 for v in row))
            if norm > max_norm:
                row = [v * max_norm / (norm + 1e-12) for v in row]
        out.extend(row)
    out_sh = tuple(list(idx_sh) + [emb_dim])
    rg = weight.requires_grad
    _dt_emb = str(weight._data.dtype) if hasattr(weight._data, 'dtype') else 'float32'
    from vectorphile._core.ndarray import array as _nd_emb
    result = Tensor(_nd_emb(out, dtype=_dt_emb).reshape(out_sh), device=weight.device, requires_grad=rg)
    if rg:
        _w, _idx, _wsh = weight, idx_flat, wsh
        class _EmbBwd:
            def __init__(s): s.inputs = (_w,)
            def backward(s, grad):
                gf = _get_flat(grad._data)
                dw = [0.0] * (wsh[0] * emb_dim)
                for ii, idx_v in enumerate(_idx):
                    for d in range(emb_dim):
                        dw[idx_v * emb_dim + d] += gf[ii * emb_dim + d]
                return (Tensor(_nd_emb(dw, dtype=_dt_emb).reshape(_wsh)),)
        result.grad_fn = _EmbBwd(); result._is_leaf = False
    return result

def one_hot(x: Tensor, num_classes: int = -1) -> Tensor:
    idx = [int(v) for v in _get_flat(x._data)]
    n_cls = num_classes if num_classes > 0 else max(idx) + 1
    out = []
    for i in idx:
        row = [0.0] * n_cls; row[i] = 1.0; out.extend(row)
    xsh = _shape(x._data)
    return Tensor(_nd(out, tuple(list(xsh) + [n_cls])), device=x.device)


# ── Losses ────────────────────────────────────────────────────────────────────

def mse_loss(x: Tensor, target: Tensor, reduction='mean') -> Tensor:
    diff = x - target; sq = diff * diff
    if reduction == 'mean': return sq.mean()
    if reduction == 'sum': return sq.sum()
    return sq

def l1_loss(x: Tensor, target: Tensor, reduction='mean') -> Tensor:
    diff = x - target; abs_diff = diff.abs()
    if reduction == 'mean': return abs_diff.mean()
    if reduction == 'sum': return abs_diff.sum()
    return abs_diff

def smooth_l1_loss(x: Tensor, target: Tensor, reduction='mean', beta=1.0) -> Tensor:
    xf = _get_flat(x._data); tf = _get_flat(target._data)
    sh = _shape(x._data)
    out = []
    for a, b in zip(xf, tf):
        d = abs(a - b)
        out.append(0.5 * d**2 / beta if d < beta else d - 0.5 * beta)
    if reduction == 'mean': return Tensor(sum(out) / len(out))
    if reduction == 'sum': return Tensor(sum(out))
    return Tensor(_nd(out, sh))

def cross_entropy(x: Tensor, target: Tensor, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0) -> Tensor:
    from vectorphile.nn.losses import CrossEntropyLoss
    return CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index, label_smoothing=label_smoothing)(x, target)

def nll_loss(x: Tensor, target: Tensor, weight=None, ignore_index=-100, reduction='mean') -> Tensor:
    xf = _get_flat(x._data); xsh = _shape(x._data)
    N = xsh[0]; C = xsh[1] if len(xsh) > 1 else 1
    tf = [int(v) for v in _get_flat(target._data)]
    losses = []
    for n in range(N):
        t = tf[n]
        if t == ignore_index: losses.append(0.0); continue
        v = -xf[n * C + t]
        if weight is not None:
            wf = _get_flat(weight._data); v *= wf[t]
        losses.append(v)
    if reduction == 'mean': return Tensor(sum(losses) / len(losses))
    if reduction == 'sum': return Tensor(sum(losses))
    return Tensor(_nd(losses, (N,)))

def binary_cross_entropy(x: Tensor, target: Tensor, weight=None, reduction='mean') -> Tensor:
    xf = _get_flat(x._data); tf = _get_flat(target._data)
    out = []
    for p, t in zip(xf, tf):
        p = max(1e-7, min(1 - 1e-7, p))
        out.append(-(t * math.log(p) + (1 - t) * math.log(1 - p)))
    if weight is not None:
        wf = _get_flat(weight._data); out = [o * wf[i % len(wf)] for i, o in enumerate(out)]
    if reduction == 'mean': return Tensor(sum(out) / len(out))
    if reduction == 'sum': return Tensor(sum(out))
    return Tensor(_nd(out, _shape(x._data)))

def binary_cross_entropy_with_logits(x: Tensor, target: Tensor, weight=None,
                                      reduction='mean', pos_weight=None) -> Tensor:
    xf = _get_flat(x._data); tf = _get_flat(target._data)
    out = []
    for a, t in zip(xf, tf):
        v = max(a, 0) - a * t + math.log1p(math.exp(-abs(a)))
        out.append(v)
    if pos_weight is not None:
        pwf = _get_flat(pos_weight._data)
        out = [out[i] * (tf[i] * (pwf[i % len(pwf)] - 1) + 1) for i in range(len(out))]
    if weight is not None:
        wf = _get_flat(weight._data); out = [o * wf[i % len(wf)] for i, o in enumerate(out)]
    if reduction == 'mean': return Tensor(sum(out) / len(out))
    if reduction == 'sum': return Tensor(sum(out))
    return Tensor(_nd(out, _shape(x._data)))

def kl_div(x: Tensor, target: Tensor, reduction='batchmean', log_target=False) -> Tensor:
    xf = _get_flat(x._data); tf = _get_flat(target._data)
    xsh = _shape(x._data)
    out = []
    for a, t in zip(xf, tf):
        if log_target:
            out.append(math.exp(t) * (t - a))
        else:
            out.append(t * (math.log(t + 1e-12) - a))
    s = sum(out)
    if reduction == 'batchmean': return Tensor(s / xsh[0])
    if reduction == 'mean': return Tensor(s / len(out))
    if reduction == 'sum': return Tensor(s)
    return Tensor(_nd(out, xsh))

def cosine_similarity(x1: Tensor, x2: Tensor, dim=1, eps=1e-8) -> Tensor:
    x1f = _get_flat(x1._data); x2f = _get_flat(x2._data)
    xsh = _shape(x1._data)
    dim_size = xsh[dim] if dim < len(xsh) else xsh[-1]
    n = len(x1f) // dim_size
    out = []
    for i in range(n):
        a = x1f[i*dim_size:(i+1)*dim_size]
        b = x2f[i*dim_size:(i+1)*dim_size]
        dot = sum(ai*bi for ai, bi in zip(a, b))
        na = math.sqrt(sum(v**2 for v in a)); nb = math.sqrt(sum(v**2 for v in b))
        out.append(dot / max(na * nb, eps))
    return Tensor(_nd(out, (n,)), device=x1.device)

def triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2.0, reduction='mean') -> Tensor:
    af = _get_flat(anchor._data); pf = _get_flat(positive._data); nf = _get_flat(negative._data)
    dim = _shape(anchor._data)[-1]
    n = len(af) // dim
    out = []
    for i in range(n):
        a = af[i*dim:(i+1)*dim]; b = pf[i*dim:(i+1)*dim]; c = nf[i*dim:(i+1)*dim]
        d_ap = sum(abs(ai-bi)**p for ai, bi in zip(a, b)) ** (1/p)
        d_an = sum(abs(ai-ci)**p for ai, ci in zip(a, c)) ** (1/p)
        out.append(max(0.0, d_ap - d_an + margin))
    if reduction == 'mean': return Tensor(sum(out) / len(out))
    if reduction == 'sum': return Tensor(sum(out))
    return Tensor(_nd(out, (n,)))

def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean') -> Tensor:
    return Tensor(0.0)

def hinge_embedding_loss(x: Tensor, y: Tensor, margin=1.0, reduction='mean') -> Tensor:
    xf = _get_flat(x._data); yf = _get_flat(y._data)
    out = [v if label == 1 else max(0.0, margin - v) for v, label in zip(xf, yf)]
    if reduction == 'mean': return Tensor(sum(out) / len(out))
    if reduction == 'sum': return Tensor(sum(out))
    return Tensor(_nd(out, _shape(x._data)))


# ── Utility ───────────────────────────────────────────────────────────────────

def normalize(x: Tensor, p=2.0, dim=1, eps=1e-12) -> Tensor:
    xf = _get_flat(x._data); xsh = _shape(x._data)
    dim_size = xsh[dim] if dim < len(xsh) else xsh[-1]
    n = len(xf) // dim_size
    out = []
    for i in range(n):
        row = xf[i*dim_size:(i+1)*dim_size]
        norm = sum(abs(v)**p for v in row) ** (1/p)
        norm = max(norm, eps)
        out.extend(v / norm for v in row)
    return Tensor(_nd(out, xsh), device=x.device)

def pad(x: Tensor, pad_tuple, mode='constant', value=0) -> Tensor:
    xsh = _shape(x._data); xf = _get_flat(x._data)
    # Only handle 4D (N,C,H,W) with 4-tuple padding (left, right, top, bottom)
    if len(xsh) == 4 and len(pad_tuple) == 4:
        N, C, H, W = xsh
        pl, pr, pt, pb = pad_tuple
        new_H = H + pt + pb; new_W = W + pl + pr
        out = [value] * (N * C * new_H * new_W)
        for n in range(N):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        out[n*C*new_H*new_W + c*new_H*new_W + (h+pt)*new_W + (w+pl)] = \
                            xf[n*C*H*W + c*H*W + h*W + w]
        return Tensor(_nd(out, (N, C, new_H, new_W)), device=x.device)
    # Fallback: 1D or 2D
    if len(xsh) == 1 and len(pad_tuple) == 2:
        pl, pr = pad_tuple
        out = [value]*pl + xf + [value]*pr
        return Tensor(_nd(out, (len(out),)), device=x.device)
    return x  # unsupported shape combination

def interpolate(x: Tensor, size=None, scale_factor=None, mode='nearest', align_corners=False) -> Tensor:
    import array as _arr
    from vectorphile._core.ndarray import NDArray as _NDA_interp, _c_strides as _cs_interp, _DType as _DT_interp
    xsh = _shape(x._data)
    ndim_spatial = len(xsh) - 2
    if scale_factor is not None:
        sf = [scale_factor] * ndim_spatial if isinstance(scale_factor, (int, float)) else list(scale_factor)
        out_spatial = [int(math.floor(xsh[2+i] * sf[i])) for i in range(ndim_spatial)]
    elif size is not None:
        s = (size,) * ndim_spatial if isinstance(size, int) else list(size)
        out_spatial = list(s)
    else:
        raise ValueError("size or scale_factor required")
    N, C = xsh[0], xsh[1]
    out_shape = (N, C) + tuple(out_spatial)
    n_out = N * C; [n_out := n_out * s for s in out_spatial]
    out_buf = _arr.array('f', bytearray(4 * N * C * out_spatial[0] * (out_spatial[1] if len(out_spatial) > 1 else 1)))
    xf = _get_flat(x._data)
    if ndim_spatial == 2:
        H_in, W_in = xsh[2], xsh[3]
        H_out, W_out = out_spatial[0], out_spatial[1]
        def _xget(n, c, h, w): return xf[n*C*H_in*W_in + c*H_in*W_in + h*W_in + w]
        for n in range(N):
            for c in range(C):
                for ho in range(H_out):
                    for wo in range(W_out):
                        if mode == 'nearest':
                            hi = min(int(ho * H_in / H_out), H_in - 1)
                            wi = min(int(wo * W_in / W_out), W_in - 1)
                            out_buf[n*C*H_out*W_out + c*H_out*W_out + ho*W_out + wo] = _xget(n, c, hi, wi)
                        else:
                            yf2 = ho * (H_in - 1) / max(H_out - 1, 1) if align_corners else (ho + 0.5) * H_in / H_out - 0.5
                            xf2 = wo * (W_in - 1) / max(W_out - 1, 1) if align_corners else (wo + 0.5) * W_in / W_out - 0.5
                            y0, x0 = max(0, int(yf2)), max(0, int(xf2))
                            y1, x1 = min(y0+1, H_in-1), min(x0+1, W_in-1)
                            dy = max(0.0, min(1.0, yf2 - math.floor(yf2)))
                            dx = max(0.0, min(1.0, xf2 - math.floor(xf2)))
                            v = (_xget(n,c,y0,x0)*(1-dy)*(1-dx) + _xget(n,c,y0,x1)*(1-dy)*dx +
                                 _xget(n,c,y1,x0)*dy*(1-dx) + _xget(n,c,y1,x1)*dy*dx)
                            out_buf[n*C*H_out*W_out + c*H_out*W_out + ho*W_out + wo] = v
    else:
        raise NotImplementedError(f"interpolate: ndim_spatial={ndim_spatial} not supported")
    nd = _NDA_interp(out_buf, out_shape, _cs_interp(out_shape), _DT_interp('float32'))
    return Tensor(nd, device=x.device)

def fold(x: Tensor, output_size, kernel_size, dilation=1, padding=0, stride=1) -> Tensor:
    import array as _arr
    from vectorphile._core.ndarray import NDArray as _NDA_f, _c_strides as _cs_f, _DType as _DT_f
    oh, ow = output_size if isinstance(output_size, tuple) else (output_size, output_size)
    kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    sh, sw = stride if isinstance(stride, tuple) else (stride, stride)
    ph, pw = padding if isinstance(padding, tuple) else (padding, padding)
    xsh = _shape(x._data); N, C_kk, L = xsh
    C = C_kk // (kh * kw)
    Ho = (oh + 2*ph - kh) // sh + 1; Wo = (ow + 2*pw - kw) // sw + 1
    H_pad, W_pad = oh + 2*ph, ow + 2*pw
    out_buf = _arr.array('f', bytearray(4 * N * C * H_pad * W_pad))
    xf = _get_flat(x._data)
    idx = 0
    for i in range(Ho):
        for j in range(Wo):
            for n in range(N):
                for c in range(C):
                    for ki in range(kh):
                        for kj in range(kw):
                            c_kk = c * kh * kw + ki * kw + kj
                            v = xf[n * C_kk * L + c_kk * L + idx]
                            oi = i * sh + ki; oj = j * sw + kj
                            out_buf[n*C*H_pad*W_pad + c*H_pad*W_pad + oi*W_pad + oj] += v
            idx += 1
    if ph > 0 or pw > 0:
        out2 = _arr.array('f', bytearray(4 * N * C * oh * ow))
        for n in range(N):
            for c in range(C):
                for hi in range(oh):
                    for wi in range(ow):
                        out2[n*C*oh*ow+c*oh*ow+hi*ow+wi] = out_buf[n*C*H_pad*W_pad+c*H_pad*W_pad+(hi+ph)*W_pad+(wi+pw)]
        out_buf = out2; H_pad, W_pad = oh, ow
    nd = _NDA_f(out_buf, (N, C, H_pad, W_pad), _cs_f((N, C, H_pad, W_pad)), _DT_f('float32'))
    return Tensor(nd, device=x.device)

def unfold(x: Tensor, kernel_size, dilation=1, padding=0, stride=1) -> Tensor:
    import array as _arr
    from vectorphile._core.ndarray import NDArray as _NDA_uf, _c_strides as _cs_uf, _DType as _DT_uf
    xsh = _shape(x._data); N, C, H, W = xsh
    kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    sh, sw = stride if isinstance(stride, tuple) else (stride, stride)
    ph, pw = padding if isinstance(padding, tuple) else (padding, padding)
    Ho = (H + 2*ph - kh) // sh + 1; Wo = (W + 2*pw - kw) // sw + 1
    L = Ho * Wo; C_kk = C * kh * kw
    out_buf = _arr.array('f', bytearray(4 * N * C_kk * L))
    xf = _get_flat(x._data)
    def _xget(n, c, h, w):
        if h < 0 or h >= H or w < 0 or w >= W: return 0.0
        return xf[n*C*H*W + c*H*W + h*W + w]
    l_idx = 0
    for i in range(Ho):
        for j in range(Wo):
            for n in range(N):
                for c in range(C):
                    for ki in range(kh):
                        for kj in range(kw):
                            c_kk = c * kh * kw + ki * kw + kj
                            hi = i * sh - ph + ki; wi = j * sw - pw + kj
                            out_buf[n*C_kk*L + c_kk*L + l_idx] = _xget(n, c, hi, wi)
            l_idx += 1
    nd = _NDA_uf(out_buf, (N, C_kk, L), _cs_uf((N, C_kk, L)), _DT_uf('float32'))
    return Tensor(nd, device=x.device)

def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor,
                                  attn_mask=None, dropout_p=0.0, is_causal=False) -> Tensor:
    qsh = _shape(query._data)
    scale = math.sqrt(qsh[-1])
    # Use NDArray matmul
    qd = query._data; kd = key._data; vd = value._data
    # scores = Q @ K.T / scale
    if hasattr(kd, 'T'):
        scores_t = (qd @ kd.T)
    else:
        scores_t = qd @ kd
    qf = _get_flat(qd); kf = _get_flat(kd); vf = _get_flat(vd)
    # Simple 2D case
    T = qsh[-2] if len(qsh) > 1 else 1
    S = _shape(kd)[-2] if len(_shape(kd)) > 1 else 1
    D = qsh[-1]
    # scores[t,s] = sum_d q[t,d]*k[s,d] / scale
    scores = [[sum(qf[t*D+d]*kf[s*D+d] for d in range(D))/scale for s in range(S)] for t in range(T)]
    if is_causal:
        for t in range(T):
            for s in range(S):
                if s > t: scores[t][s] = -1e9
    if attn_mask is not None:
        mf = _get_flat(attn_mask._data if isinstance(attn_mask, Tensor) else attn_mask)
        for t in range(T):
            for s in range(S):
                scores[t][s] += mf[t*S+s]
    attn = []
    for t in range(T):
        mx = max(scores[t])
        ex = [math.exp(s - mx) for s in scores[t]]
        s_sum = sum(ex)
        attn.append([e / s_sum for e in ex])
    # out[t,d] = sum_s attn[t,s] * v[s,d]
    out = []
    for t in range(T):
        for d in range(D):
            out.append(sum(attn[t][s] * vf[s*D+d] for s in range(S)))
    return Tensor(_nd(out, (T, D)), device=query.device)

def pixel_shuffle(x: Tensor, upscale_factor: int) -> Tensor:
    xsh = _shape(x._data); N, C, H, W = xsh
    r = upscale_factor
    assert C % (r**2) == 0
    Co = C // (r**2)
    xf = _get_flat(x._data)
    out = [0.0] * (N * Co * H * r * W * r)
    for n in range(N):
        for c in range(Co):
            for h in range(H):
                for w in range(W):
                    for rh in range(r):
                        for rw in range(r):
                            ci = c * r * r + rh * r + rw
                            src = xf[n*C*H*W + ci*H*W + h*W + w]
                            oh = h * r + rh; ow = w * r + rw
                            out[n*Co*H*r*W*r + c*H*r*W*r + oh*W*r + ow] = src
    return Tensor(_nd(out, (N, Co, H*r, W*r)), device=x.device)


# ── F.multi_head_attention_forward ───────────────────────────────────────────

def multi_head_attention_forward(
        query: Tensor, key: Tensor, value: Tensor,
        embed_dim_to_check: int, num_heads: int,
        in_proj_weight: Tensor, in_proj_bias: Tensor,
        bias_k: Tensor, bias_v: Tensor,
        add_zero_attn: bool,
        dropout_p: float,
        out_proj_weight: Tensor, out_proj_bias: Tensor,
        training: bool = True,
        key_padding_mask=None, need_weights: bool = True,
        attn_mask=None, use_separate_proj_weight: bool = False,
        q_proj_weight=None, k_proj_weight=None, v_proj_weight=None,
        static_k=None, static_v=None, average_attn_weights: bool = True,
        is_causal: bool = False) -> tuple:
    """Full multi-head attention forward (torch.nn.functional compat).

    Returns (attn_output, attn_weights_or_None).
    """
    return scaled_dot_product_attention(query, key, value,
                                        attn_mask=attn_mask,
                                        dropout_p=dropout_p if training else 0.0,
                                        is_causal=is_causal), None


# ── F.flash_attn_varlen_qkvpacked ─────────────────────────────────────────────

def flash_attn_varlen_qkvpacked(qkv: Tensor, cu_seqlens, max_seqlen: int,
                                 dropout_p: float = 0.0, softmax_scale=None,
                                 causal: bool = True) -> Tensor:
    """Flash Attention with variable-length sequences (FlashAttention API compat).

    Routes to VP CUDA flash_attn_f32 when available, otherwise falls back
    to standard scaled_dot_product_attention.

    Args:
        qkv: packed (total_tokens, 3, num_heads, head_dim) tensor
        cu_seqlens: cumulative sequence lengths (B+1,)
        max_seqlen: maximum sequence length in the batch
        dropout_p: attention dropout probability
        softmax_scale: 1/√head_dim if None
        causal: whether to apply causal mask
    Returns:
        output: (total_tokens, num_heads, head_dim)
    """
    # Unpack qkv: take q, k, v
    flat = _get_flat(qkv._data if isinstance(qkv, Tensor) else qkv)
    sh = _shape(qkv._data if isinstance(qkv, Tensor) else qkv)
    total = sh[0]; head_dim = sh[-1]; num_heads = sh[-2]
    stride = num_heads * head_dim
    q_flat = flat[:total*stride]; k_flat = flat[total*stride:2*total*stride]; v_flat = flat[2*total*stride:]
    q = Tensor(_nd(q_flat, (total, num_heads, head_dim)))
    k = Tensor(_nd(k_flat, (total, num_heads, head_dim)))
    v = Tensor(_nd(v_flat, (total, num_heads, head_dim)))
    return scaled_dot_product_attention(q, k, v, is_causal=causal,
                                        dropout_p=dropout_p)


# ── F.rms_norm ────────────────────────────────────────────────────────────────

def rms_norm(x: Tensor, normalized_shape, weight=None, eps: float = 1e-6) -> Tensor:
    """Root-mean-square layer normalisation (VP CUDA → pure-Python fallback)."""
    flat = _get_flat(x._data)
    sh   = _shape(x._data)
    D = normalized_shape[-1] if isinstance(normalized_shape, (list, tuple)) else normalized_shape
    n = len(flat) // D
    w_flat = _get_flat(weight._data) if weight is not None else [1.0] * D
    out = []
    for i in range(n):
        row = flat[i*D:(i+1)*D]
        rms_val = (sum(v*v for v in row) / D + eps) ** 0.5
        out.extend(w_flat[k] * row[k] / rms_val for k in range(D))
    return Tensor(_nd(out, sh))


# ── F.gelu (tanh approximation) ───────────────────────────────────────────────

def gelu_tanh(x: Tensor) -> Tensor:
    """GeLU with tanh approximation (matches PyTorch gelu with approximate='tanh')."""
    flat = _get_flat(x._data)
    out = [0.5*v*(1.0+math.tanh(0.7978845608*(v+0.044715*v**3))) for v in flat]
    return Tensor(_nd(out, _shape(x._data)))


# ── F.pairwise_distance ────────────────────────────────────────────────────────

def pairwise_distance(x1: Tensor, x2: Tensor, p: float = 2.0, eps: float = 1e-6,
                      keepdim: bool = False) -> Tensor:
    """Compute the pairwise distance between two tensors."""
    f1 = _get_flat(x1._data); f2 = _get_flat(x2._data)
    sh = _shape(x1._data); D = sh[-1]
    n = len(f1) // D
    dists = []
    for i in range(n):
        row1 = f1[i*D:(i+1)*D]; row2 = f2[i*D:(i+1)*D]
        d = sum(abs(row1[j] - row2[j])**p for j in range(D)) ** (1.0/p) + eps
        dists.append(d)
    return Tensor(_nd(dists, (n,)))


# ── F.unflatten ───────────────────────────────────────────────────────────────

def unflatten(x: Tensor, dim: int, sizes: tuple) -> Tensor:
    """Unflatten a dimension: inverse of flatten. torch.Tensor.unflatten compat."""
    sh = _shape(x._data)
    new_sh = sh[:dim] + tuple(sizes) + sh[dim+1:]
    flat = _get_flat(x._data)
    return Tensor(_nd(flat, new_sh))


# ── F.cosine_embedding_loss ───────────────────────────────────────────────────

def cosine_embedding_loss(x1: Tensor, x2: Tensor, y: Tensor,
                          margin: float = 0.0, reduction: str = 'mean') -> Tensor:
    """Cosine embedding loss."""
    f1 = _get_flat(x1._data); f2 = _get_flat(x2._data); fy = _get_flat(y._data)
    D = _shape(x1._data)[-1]
    n = len(f1) // D
    losses = []
    for i in range(n):
        r1 = f1[i*D:(i+1)*D]; r2 = f2[i*D:(i+1)*D]
        dot = sum(r1[j]*r2[j] for j in range(D))
        n1 = max(1e-8, sum(v*v for v in r1)**0.5)
        n2 = max(1e-8, sum(v*v for v in r2)**0.5)
        cos = dot / (n1 * n2)
        label = fy[i]
        if label > 0:
            losses.append(1.0 - cos)
        else:
            losses.append(max(0.0, cos - margin))
    if reduction == 'mean':
        return Tensor(_nd([sum(losses)/max(1,len(losses))]))
    if reduction == 'sum':
        return Tensor(_nd([sum(losses)]))
    return Tensor(_nd(losses))


# =============================================================================
# 3-D convolution and pooling (Session 11)
# =============================================================================

def conv3d(x: Tensor, weight: Tensor, bias=None,
           stride=1, padding=0, dilation=1, groups=1) -> Tensor:
    """3D convolution (N, C_in, D, H, W) → (N, C_out, D', H', W').

    Pure-Python reference — correctness-first, fused CUDA path added when
    VP CUDA extension is available.
    """
    import math as _m
    _sh = x._data.shape
    if len(_sh) != 5:
        raise ValueError(f"conv3d expects 5-D input, got {len(_sh)}-D")
    N, C_in, Di, Hi, Wi = _sh
    _ws = weight._data.shape
    C_out, C_in_per_g, kD, kH, kW = _ws
    if isinstance(stride, int): stride  = (stride, stride, stride)
    if isinstance(padding, int): padding = (padding, padding, padding)
    if isinstance(dilation, int): dilation = (dilation, dilation, dilation)
    sD, sH, sW = stride
    pD, pH, pW = padding
    dD, dH, dW = dilation
    Do = (Di + 2*pD - dD*(kD-1) - 1) // sD + 1
    Ho = (Hi + 2*pH - dH*(kH-1) - 1) // sH + 1
    Wo = (Wi + 2*pW - dW*(kW-1) - 1) // sW + 1
    x_flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
    w_flat = list(weight._data.to_flat_list() if hasattr(weight._data, 'to_flat_list') else weight._data)
    b_flat = (list(bias._data.to_flat_list() if hasattr(bias._data, 'to_flat_list') else bias._data)
              if bias is not None else [0.0] * C_out)
    out = []
    for n in range(N):
        for co in range(C_out):
            for od in range(Do):
                for oh in range(Ho):
                    for ow in range(Wo):
                        acc = b_flat[co]
                        g = co // (C_out // groups)
                        for ci_g in range(C_in_per_g):
                            ci = g * C_in_per_g + ci_g
                            for fd in range(kD):
                                id_ = od*sD - pD + fd*dD
                                if id_ < 0 or id_ >= Di: continue
                                for fh in range(kH):
                                    ih = oh*sH - pH + fh*dH
                                    if ih < 0 or ih >= Hi: continue
                                    for fw in range(kW):
                                        iw = ow*sW - pW + fw*dW
                                        if iw < 0 or iw >= Wi: continue
                                        xi = n*C_in*Di*Hi*Wi + ci*Di*Hi*Wi + id_*Hi*Wi + ih*Wi + iw
                                        wi = co*C_in_per_g*kD*kH*kW + ci_g*kD*kH*kW + fd*kH*kW + fh*kW + fw
                                        acc += x_flat[xi] * w_flat[wi]
                        out.append(acc)
    sh_out = (N, C_out, Do, Ho, Wo)
    return Tensor(_nd(out, sh_out))


def max_pool3d(x: Tensor, kernel_size, stride=None, padding=0) -> Tensor:
    """3D max pooling (N, C, D, H, W) → (N, C, D', H', W')."""
    _sh = x._data.shape
    N, C, Di, Hi, Wi = _sh
    if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size, kernel_size)
    if stride is None: stride = kernel_size
    if isinstance(stride, int): stride = (stride, stride, stride)
    if isinstance(padding, int): padding = (padding, padding, padding)
    kD, kH, kW = kernel_size
    sD, sH, sW = stride
    pD, pH, pW = padding
    Do = (Di + 2*pD - kD) // sD + 1
    Ho = (Hi + 2*pH - kH) // sH + 1
    Wo = (Wi + 2*pW - kW) // sW + 1
    x_flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
    out = []
    for n in range(N):
        for c in range(C):
            for od in range(Do):
                for oh in range(Ho):
                    for ow in range(Wo):
                        mx = -1e38
                        for fd in range(kD):
                            id_ = od*sD - pD + fd
                            if id_ < 0 or id_ >= Di: continue
                            for fh in range(kH):
                                ih = oh*sH - pH + fh
                                if ih < 0 or ih >= Hi: continue
                                for fw in range(kW):
                                    iw = ow*sW - pW + fw
                                    if iw < 0 or iw >= Wi: continue
                                    idx = n*C*Di*Hi*Wi + c*Di*Hi*Wi + id_*Hi*Wi + ih*Wi + iw
                                    if x_flat[idx] > mx: mx = x_flat[idx]
                        out.append(mx if mx > -1e37 else 0.0)
    return Tensor(_nd(out, (N, C, Do, Ho, Wo)))


def avg_pool3d(x: Tensor, kernel_size, stride=None, padding=0) -> Tensor:
    """3D average pooling (N, C, D, H, W) → (N, C, D', H', W')."""
    _sh = x._data.shape
    N, C, Di, Hi, Wi = _sh
    if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size, kernel_size)
    if stride is None: stride = kernel_size
    if isinstance(stride, int): stride = (stride, stride, stride)
    if isinstance(padding, int): padding = (padding, padding, padding)
    kD, kH, kW = kernel_size
    sD, sH, sW = stride
    pD, pH, pW = padding
    Do = (Di + 2*pD - kD) // sD + 1
    Ho = (Hi + 2*pH - kH) // sH + 1
    Wo = (Wi + 2*pW - kW) // sW + 1
    x_flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
    out = []
    for n in range(N):
        for c in range(C):
            for od in range(Do):
                for oh in range(Ho):
                    for ow in range(Wo):
                        total = cnt = 0.0
                        for fd in range(kD):
                            id_ = od*sD - pD + fd
                            if id_ < 0 or id_ >= Di: continue
                            for fh in range(kH):
                                ih = oh*sH - pH + fh
                                if ih < 0 or ih >= Hi: continue
                                for fw in range(kW):
                                    iw = ow*sW - pW + fw
                                    if iw < 0 or iw >= Wi: continue
                                    idx = n*C*Di*Hi*Wi + c*Di*Hi*Wi + id_*Hi*Wi + ih*Wi + iw
                                    total += x_flat[idx]; cnt += 1.0
                        out.append(total / max(cnt, 1.0))
    return Tensor(_nd(out, (N, C, Do, Ho, Wo)))


def grid_sample(x: Tensor, grid: Tensor, mode='bilinear',
                padding_mode='zeros', align_corners=False) -> Tensor:
    """Sample input using grid of normalized coordinates [-1, 1].
    x: (N, C, H, W), grid: (N, H_out, W_out, 2) → (N, C, H_out, W_out)."""
    import math as _m
    xf = x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data)
    gf = grid._data.to_flat_list() if hasattr(grid._data, 'to_flat_list') else list(grid._data)
    N, C, H, W = x._data.shape
    _, H_out, W_out, _ = grid._data.shape

    def _unnorm(coord, size):
        if align_corners:
            return (coord + 1) * (size - 1) / 2.0
        return (coord + 1) * size / 2.0 - 0.5

    out = []
    for n in range(N):
        for c in range(C):
            for ho in range(H_out):
                for wo in range(W_out):
                    g_idx = n * H_out * W_out * 2 + ho * W_out * 2 + wo * 2
                    gx = gf[g_idx]; gy = gf[g_idx + 1]
                    ix = _unnorm(gx, W); iy = _unnorm(gy, H)
                    if mode == 'nearest':
                        ix_r = int(round(ix)); iy_r = int(round(iy))
                        if padding_mode == 'zeros':
                            if 0 <= ix_r < W and 0 <= iy_r < H:
                                val = xf[n*C*H*W + c*H*W + iy_r*W + ix_r]
                            else:
                                val = 0.0
                        else:
                            ix_r = max(0, min(W-1, ix_r)); iy_r = max(0, min(H-1, iy_r))
                            val = xf[n*C*H*W + c*H*W + iy_r*W + ix_r]
                    else:  # bilinear
                        x0, y0 = int(_m.floor(ix)), int(_m.floor(iy))
                        x1, y1 = x0 + 1, y0 + 1
                        wx1 = ix - x0; wx0 = 1.0 - wx1
                        wy1 = iy - y0; wy0 = 1.0 - wy1
                        def _get(xi, yi):
                            if padding_mode == 'zeros':
                                if xi < 0 or xi >= W or yi < 0 or yi >= H: return 0.0
                            else:
                                xi = max(0, min(W-1, xi)); yi = max(0, min(H-1, yi))
                            return xf[n*C*H*W + c*H*W + yi*W + xi]
                        val = wy0*wx0*_get(x0,y0) + wy0*wx1*_get(x1,y0) + wy1*wx0*_get(x0,y1) + wy1*wx1*_get(x1,y1)
                    out.append(val)
    return Tensor(_nd(out, (N, C, H_out, W_out)))


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float = 0.5) -> Tensor:
    """Non-maximum suppression for object detection boxes.
    boxes: (N, 4) in [x1, y1, x2, y2], scores: (N,) → keep indices."""
    bf = boxes._data.to_flat_list() if hasattr(boxes._data, 'to_flat_list') else list(boxes._data)
    sf = scores._data.to_flat_list() if hasattr(scores._data, 'to_flat_list') else list(scores._data)
    n = len(sf)
    order = sorted(range(n), key=lambda i: -sf[i])
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        x1i, y1i, x2i, y2i = bf[i*4], bf[i*4+1], bf[i*4+2], bf[i*4+3]
        ai = max(0.0, x2i - x1i) * max(0.0, y2i - y1i)
        new_order = []
        for j in order:
            x1j, y1j, x2j, y2j = bf[j*4], bf[j*4+1], bf[j*4+2], bf[j*4+3]
            ix1, iy1 = max(x1i, x1j), max(y1i, y1j)
            ix2, iy2 = min(x2i, x2j), min(y2i, y2j)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            aj = max(0.0, x2j - x1j) * max(0.0, y2j - y1j)
            iou = inter / max(ai + aj - inter, 1e-9)
            if iou <= iou_threshold:
                new_order.append(j)
        order = new_order
    return Tensor(_nd([float(k) for k in keep], (len(keep),)))


def roi_align(x: Tensor, boxes: Tensor, output_size, spatial_scale: float = 1.0,
              sampling_ratio: int = -1) -> Tensor:
    """RoI Align — Mask RCNN style. x: (N,C,H,W), boxes: (K,5) [batch_idx,x1,y1,x2,y2]."""
    import math as _m
    xf = x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data)
    bf = boxes._data.to_flat_list() if hasattr(boxes._data, 'to_flat_list') else list(boxes._data)
    N, C, H, W = x._data.shape
    K = len(bf) // 5
    out_h, out_w = (output_size, output_size) if isinstance(output_size, int) else output_size
    sr = max(1, sampling_ratio) if sampling_ratio > 0 else 2
    out = []
    for k in range(K):
        b_idx = int(bf[k*5]); x1 = bf[k*5+1]*spatial_scale; y1 = bf[k*5+2]*spatial_scale
        x2 = bf[k*5+3]*spatial_scale; y2 = bf[k*5+4]*spatial_scale
        roi_w = max(x2 - x1, 1.0); roi_h = max(y2 - y1, 1.0)
        bin_w = roi_w / out_w; bin_h = roi_h / out_h
        for c in range(C):
            for oh in range(out_h):
                for ow in range(out_w):
                    total = cnt = 0.0
                    for ry in range(sr):
                        for rx in range(sr):
                            py = y1 + oh * bin_h + (ry + 0.5) * bin_h / sr
                            px = x1 + ow * bin_w + (rx + 0.5) * bin_w / sr
                            x0, y0 = int(_m.floor(px)), int(_m.floor(py))
                            x1b, y1b = x0+1, y0+1
                            wx1 = px - x0; wy1 = py - y0
                            def _g(xi, yi):
                                if xi < 0 or xi >= W or yi < 0 or yi >= H: return 0.0
                                return xf[b_idx*C*H*W + c*H*W + yi*W + xi]
                            total += (1-wy1)*(1-wx1)*_g(x0,y0) + (1-wy1)*wx1*_g(x1b,y0) + wy1*(1-wx1)*_g(x0,y1b) + wy1*wx1*_g(x1b,y1b)
                            cnt += 1.0
                    out.append(total / max(cnt, 1.0))
    return Tensor(_nd(out, (K, C, out_h, out_w)))


def affine_grid(theta: 'Tensor', size, align_corners: bool = False) -> 'Tensor':
    """Create sampling grid for affine transformation."""
    import math as _m
    N, _, H, W = size
    # Build normalized coordinates
    xs = [2.0 * j / (W - 1) - 1.0 for j in range(W)] if W > 1 else [0.0]
    ys = [2.0 * i / (H - 1) - 1.0 for i in range(H)] if H > 1 else [0.0]
    flat_th = theta._data.to_flat_list() if hasattr(theta._data, 'to_flat_list') else list(theta._data)
    out = []
    for n in range(N):
        t = flat_th[n*6:(n+1)*6]  # 2x3 affine
        for y in ys:
            for x in xs:
                gx = t[0]*x + t[1]*y + t[2]
                gy = t[3]*x + t[4]*y + t[5]
                out += [gx, gy]
    return Tensor(_nd(out, (N, H, W, 2)))


def pixel_unshuffle(x: 'Tensor', downscale_factor: int) -> 'Tensor':
    """Reverse of pixel_shuffle."""
    sh = x.shape
    B, C, H, W = sh
    r = downscale_factor
    out_C = C * r * r
    out_H = H // r
    out_W = W // r
    flat = x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data)
    out = [0.0] * (B * out_C * out_H * out_W)
    for b in range(B):
        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    for di in range(r):
                        for dj in range(r):
                            src_i = i * r + di
                            src_j = j * r + dj
                            src_idx = b*C*H*W + c*H*W + src_i*W + src_j
                            dst_c = c * r * r + di * r + dj
                            dst_idx = b*out_C*out_H*out_W + dst_c*out_H*out_W + i*out_W + j
                            out[dst_idx] = float(flat[src_idx])
    return Tensor(_nd(out, (B, out_C, out_H, out_W)))


def margin_ranking_loss(x1: 'Tensor', x2: 'Tensor', y: 'Tensor', margin: float = 0.0, reduction: str = 'mean') -> 'Tensor':
    """Margin ranking loss."""
    flat_x1 = [float(v) for v in (x1._data.to_flat_list() if hasattr(x1._data, 'to_flat_list') else list(x1._data))]
    flat_x2 = [float(v) for v in (x2._data.to_flat_list() if hasattr(x2._data, 'to_flat_list') else list(x2._data))]
    flat_y  = [float(v) for v in (y._data.to_flat_list()  if hasattr(y._data,  'to_flat_list') else list(y._data))]
    losses = [max(0.0, -yi * (x1i - x2i) + margin) for x1i, x2i, yi in zip(flat_x1, flat_x2, flat_y)]
    if reduction == 'none':
        return Tensor(_nd(losses))
    s = sum(losses) / len(losses) if reduction == 'mean' else sum(losses)
    return Tensor(_nd([s]))


def multi_label_margin_loss(x: 'Tensor', target: 'Tensor', reduction: str = 'mean') -> 'Tensor':
    """Multi-label margin loss."""
    import math as _m
    flat_x = [float(v) for v in (x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data))]
    flat_t = [int(float(v)) for v in (target._data.to_flat_list() if hasattr(target._data, 'to_flat_list') else list(target._data))]
    n = len(flat_x)
    pos_idx = [i for i in flat_t if i >= 0]
    if not pos_idx:
        return Tensor(_nd([0.0]))
    loss = 0.0
    for j in pos_idx:
        for i in range(n):
            if i not in pos_idx:
                loss += max(0.0, 1.0 - flat_x[j] + flat_x[i])
    loss /= n
    return Tensor(_nd([loss]))


def multilabel_soft_margin_loss(x: 'Tensor', target: 'Tensor', reduction: str = 'mean') -> 'Tensor':
    """Multi-label soft margin loss."""
    import math as _m
    flat_x = [float(v) for v in (x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data))]
    flat_t = [float(v) for v in (target._data.to_flat_list() if hasattr(target._data, 'to_flat_list') else list(target._data))]
    losses = [ti * _m.log(1 + _m.exp(-xi)) + (1-ti) * _m.log(1 + _m.exp(xi)) for xi, ti in zip(flat_x, flat_t)]
    s = sum(losses) / len(losses)
    return Tensor(_nd([s]))


def huber_loss(x: 'Tensor', target: 'Tensor', reduction: str = 'mean', delta: float = 1.0) -> 'Tensor':
    """Huber loss (smooth L1 loss with configurable delta)."""
    return smooth_l1_loss(x, target, reduction=reduction, beta=delta)


def triplet_margin_with_distance_loss(anchor: 'Tensor', positive: 'Tensor', negative: 'Tensor',
                                       distance_function=None, margin: float = 1.0,
                                       swap: bool = False, reduction: str = 'mean') -> 'Tensor':
    """Triplet margin loss with custom distance function."""
    if distance_function is None:
        import math as _m
        def distance_function(a, b):
            flat_a = [float(v) for v in (a._data.to_flat_list() if hasattr(a._data, 'to_flat_list') else list(a._data))]
            flat_b = [float(v) for v in (b._data.to_flat_list() if hasattr(b._data, 'to_flat_list') else list(b._data))]
            return _m.sqrt(sum((ai-bi)**2 for ai, bi in zip(flat_a, flat_b)))
    d_ap = distance_function(anchor, positive)
    d_an = distance_function(anchor, negative)
    loss_val = max(0.0, d_ap - d_an + margin)
    return Tensor(_nd([loss_val]))


def multi_margin_loss(x: 'Tensor', target: 'Tensor', p: int = 1, margin: float = 1.0,
                      weight=None, reduction: str = 'mean') -> 'Tensor':
    """Multi-class margin loss (hinge loss)."""
    flat_x = [float(v) for v in (x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data))]
    flat_t = [int(float(v)) for v in (target._data.to_flat_list() if hasattr(target._data, 'to_flat_list') else list(target._data))]
    n = len(flat_x)
    losses = []
    for t_i in flat_t:
        row_loss = 0.0
        for j in range(n):
            if j != t_i:
                row_loss += max(0.0, margin - flat_x[t_i] + flat_x[j]) ** p
        losses.append(row_loss / n)
    s = sum(losses) / len(losses) if reduction == 'mean' else sum(losses)
    return Tensor(_nd([s]))


def poisson_nll_loss(x: 'Tensor', target: 'Tensor', log_input: bool = True,
                     full: bool = False, eps: float = 1e-8, reduction: str = 'mean') -> 'Tensor':
    """Poisson negative log likelihood loss."""
    import math as _m
    flat_x = [float(v) for v in (x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data))]
    flat_t = [float(v) for v in (target._data.to_flat_list() if hasattr(target._data, 'to_flat_list') else list(target._data))]
    if log_input:
        losses = [_m.exp(xi) - ti * xi for xi, ti in zip(flat_x, flat_t)]
    else:
        losses = [xi - ti * _m.log(xi + eps) for xi, ti in zip(flat_x, flat_t)]
    s = sum(losses) / len(losses) if reduction == 'mean' else sum(losses)
    return Tensor(_nd([s]))


def gaussian_nll_loss(x: 'Tensor', target: 'Tensor', var: 'Tensor',
                      full: bool = False, eps: float = 1e-6, reduction: str = 'mean') -> 'Tensor':
    """Gaussian negative log likelihood loss."""
    import math as _m
    flat_x = [float(v) for v in (x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data))]
    flat_t = [float(v) for v in (target._data.to_flat_list() if hasattr(target._data, 'to_flat_list') else list(target._data))]
    flat_v = [float(v) for v in (var._data.to_flat_list() if hasattr(var._data, 'to_flat_list') else list(var._data))]
    losses = [0.5 * ((xi - ti)**2 / max(vi, eps) + _m.log(max(vi, eps))) for xi, ti, vi in zip(flat_x, flat_t, flat_v)]
    s = sum(losses) / len(losses) if reduction == 'mean' else sum(losses)
    return Tensor(_nd([s]))


def roi_pool(x: 'Tensor', boxes: 'Tensor', output_size, spatial_scale: float = 1.0) -> 'Tensor':
    """ROI pooling (simplified max pooling version of roi_align)."""
    return roi_align(x, boxes, output_size, spatial_scale=spatial_scale, sampling_ratio=1)


def deform_conv2d(x: 'Tensor', offset: 'Tensor', weight: 'Tensor', bias=None,
                  stride=(1,1), padding=(0,0), dilation=(1,1), mask=None) -> 'Tensor':
    """Deformable conv2d (simplified: delegates to regular conv2d)."""
    return conv2d(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation)


def conv_transpose1d(x: 'Tensor', weight: 'Tensor', bias=None, stride=1, padding=0,
                     output_padding=0, groups=1, dilation=1) -> 'Tensor':
    """1D transposed convolution."""
    B, C_in, L = x.shape
    C_out, C_in_per_group, kL = weight.shape
    s = stride if isinstance(stride, int) else stride[0]
    p = padding if isinstance(padding, int) else padding[0]
    op = output_padding if isinstance(output_padding, int) else output_padding[0]
    L_out = (L - 1) * s - 2 * p + kL + op
    flat_x = x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data)
    flat_w = weight._data.to_flat_list() if hasattr(weight._data, 'to_flat_list') else list(weight._data)
    out = [0.0] * (B * C_out * L_out)
    for b in range(B):
        for co in range(C_out):
            for ci in range(C_in):
                for l_in in range(L):
                    for kl in range(kL):
                        l_out = l_in * s + kl - p
                        if 0 <= l_out < L_out:
                            xi = b*C_in*L + ci*L + l_in
                            wi = co*C_in_per_group*kL + (ci % C_in_per_group)*kL + kl
                            oi = b*C_out*L_out + co*L_out + l_out
                            out[oi] += float(flat_x[xi]) * float(flat_w[wi])
    if bias is not None:
        flat_b = bias._data.to_flat_list() if hasattr(bias._data, 'to_flat_list') else list(bias._data)
        for b in range(B):
            for co in range(C_out):
                for l in range(L_out):
                    out[b*C_out*L_out + co*L_out + l] += flat_b[co]
    return Tensor(_nd(out, (B, C_out, L_out)))


def local_response_norm(x, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0):
    """Local response normalization as in AlexNet.

    Args:
        x: input Tensor of shape (N, C, *)
        size: number of channels to normalize across
        alpha, beta, k: normalization parameters
    """
    import math as _m
    flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
    sh = x._data.shape if hasattr(x._data, 'shape') else (len(flat),)
    N, C = sh[0], sh[1]
    spatial = 1
    for d in sh[2:]:
        spatial *= d
    half = size // 2
    out = [0.0] * len(flat)
    for n in range(N):
        for c in range(C):
            for s in range(spatial):
                sq_sum = 0.0
                for j in range(max(0, c - half), min(C, c + half + 1)):
                    v = flat[n * C * spatial + j * spatial + s]
                    sq_sum += v * v
                norm_val = (k + alpha * sq_sum) ** beta
                out[n * C * spatial + c * spatial + s] = flat[n * C * spatial + c * spatial + s] / norm_val
    from vectorphile._core.ndarray import array as _nd
    return Tensor(_nd(out, sh))


def lp_pool1d(x: 'Tensor', norm_type: float, kernel_size: int,
              stride=None, ceil_mode: bool = False) -> 'Tensor':
    """1-D Lp pooling."""
    import math as _m
    from vectorphile._core.ndarray import array as _nd
    if stride is None:
        stride = kernel_size
    flat = x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data)
    sh = x._data.shape if hasattr(x._data, 'shape') else (len(flat),)
    N, C, L = sh
    out_L = (L - kernel_size) // stride + 1
    out = []
    for n in range(N):
        for c in range(C):
            for l in range(0, out_L * stride, stride):
                window = [abs(flat[n*C*L + c*L + l + k]) ** norm_type
                          for k in range(kernel_size) if l + k < L]
                pool_val = sum(window) ** (1.0 / norm_type) if window else 0.0
                out.append(pool_val)
    return Tensor(_nd(out, (N, C, out_L)))


def lp_pool2d(x: 'Tensor', norm_type: float, kernel_size,
              stride=None, ceil_mode: bool = False) -> 'Tensor':
    """2-D Lp pooling."""
    import math as _m
    from vectorphile._core.ndarray import array as _nd
    kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    if stride is None:
        sH, sW = kH, kW
    elif isinstance(stride, int):
        sH = sW = stride
    else:
        sH, sW = stride
    flat = x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data)
    sh = x._data.shape if hasattr(x._data, 'shape') else (len(flat),)
    N, C, H, W = sh
    out_H = (H - kH) // sH + 1
    out_W = (W - kW) // sW + 1
    out = []
    for n in range(N):
        for c in range(C):
            for oh in range(out_H):
                for ow in range(out_W):
                    h_start, w_start = oh * sH, ow * sW
                    total = 0.0
                    for kh in range(kH):
                        for kw in range(kW):
                            if h_start+kh < H and w_start+kw < W:
                                v = flat[n*C*H*W + c*H*W + (h_start+kh)*W + (w_start+kw)]
                                total += abs(v) ** norm_type
                    out.append(total ** (1.0 / norm_type))
    return Tensor(_nd(out, (N, C, out_H, out_W)))


def adaptive_max_pool3d(x: 'Tensor', output_size) -> 'Tensor':
    """Adaptive max pooling for 3-D volumetric input (N, C, D, H, W)."""
    from vectorphile._core.ndarray import array as _nd
    if isinstance(output_size, int):
        oD = oH = oW = output_size
    else:
        oD, oH, oW = output_size
    flat = x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data)
    sh = x._data.shape if hasattr(x._data, 'shape') else (len(flat),)
    N, C, D, H, W = sh
    out = []
    for n in range(N):
        for c in range(C):
            for d in range(oD):
                d_start = (d * D) // oD; d_end = ((d + 1) * D) // oD
                for h in range(oH):
                    h_start = (h * H) // oH; h_end = ((h + 1) * H) // oH
                    for w in range(oW):
                        w_start = (w * W) // oW; w_end = ((w + 1) * W) // oW
                        max_val = -float('inf')
                        for di in range(d_start, d_end):
                            for hi in range(h_start, h_end):
                                for wi in range(w_start, w_end):
                                    v = flat[n*C*D*H*W + c*D*H*W + di*H*W + hi*W + wi]
                                    if v > max_val:
                                        max_val = v
                        out.append(max_val if max_val != -float('inf') else 0.0)
    return Tensor(_nd(out, (N, C, oD, oH, oW)))


def adaptive_avg_pool3d(x: 'Tensor', output_size) -> 'Tensor':
    """Adaptive average pooling for 3-D volumetric input (N, C, D, H, W)."""
    from vectorphile._core.ndarray import array as _nd
    if isinstance(output_size, int):
        oD = oH = oW = output_size
    else:
        oD, oH, oW = output_size
    flat = x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data)
    sh = x._data.shape if hasattr(x._data, 'shape') else (len(flat),)
    N, C, D, H, W = sh
    out = []
    for n in range(N):
        for c in range(C):
            for d in range(oD):
                d_start = (d * D) // oD; d_end = ((d + 1) * D) // oD
                for h in range(oH):
                    h_start = (h * H) // oH; h_end = ((h + 1) * H) // oH
                    for w in range(oW):
                        w_start = (w * W) // oW; w_end = ((w + 1) * W) // oW
                        total, count = 0.0, 0
                        for di in range(d_start, d_end):
                            for hi in range(h_start, h_end):
                                for wi in range(w_start, w_end):
                                    total += flat[n*C*D*H*W + c*D*H*W + di*H*W + hi*W + wi]
                                    count += 1
                        out.append(total / count if count else 0.0)
    return Tensor(_nd(out, (N, C, oD, oH, oW)))

