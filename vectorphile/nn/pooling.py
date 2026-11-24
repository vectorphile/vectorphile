"""Pooling layers — MaxPool, AvgPool, AdaptiveAvgPool — with full autograd support."""
from __future__ import annotations
import math as _math_pool
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor


def _get_fl(d):
    if hasattr(d, 'to_flat_list'): return d.to_flat_list()
    if hasattr(d, '_buf') and d._buf is not None: return list(d._buf)
    return list(d)

def _nd(lst, shape=None, dtype=None):
    from vectorphile._core.ndarray import array as _a
    r = _a(lst, dtype=dtype)
    if shape: r = r.reshape(shape)
    return r


def _xdtype(xd):
    """Return dtype string from tensor data, defaulting to float32."""
    if hasattr(xd, 'dtype'):
        dt = xd.dtype
        return str(dt) if not hasattr(dt, 'name') else dt.name
    return 'float32'


class _MaxPool1dBackward:
    def __init__(self, inp, x_pad_fl, x_pad_sh, Lo, k, s, p, dt='float32'):
        self.inputs = (inp,)
        self.x_pad_fl = x_pad_fl; self.x_pad_sh = x_pad_sh
        self.Lo = Lo; self.k = k; self.s = s; self.p = p; self._dt = dt

    def backward(self, grad):
        x_inp = self.inputs[0]
        xsh = x_inp._data.shape if hasattr(x_inp._data, 'shape') else None
        N, C, L = xsh
        gf = _get_fl(grad._data)
        xpf = self.x_pad_fl; Lp = self.x_pad_sh[2]
        dx_pad = [0.0] * (N * C * Lp)
        k, s = self.k, self.s
        for n in range(N):
            for c in range(C):
                for i in range(self.Lo):
                    base = n*C*Lp + c*Lp
                    win = [xpf[base + i*s + kk] for kk in range(k)]
                    mx = max(win)
                    cnt = sum(1 for v in win if v == mx) or 1
                    g_val = gf[n*C*self.Lo + c*self.Lo + i] / cnt
                    for kk in range(k):
                        if win[kk] == mx:
                            dx_pad[base + i*s + kk] += g_val
        p = self.p
        dx = [dx_pad[n*C*Lp + c*Lp + p + ll] for n in range(N) for c in range(C) for ll in range(L)] if p > 0 else dx_pad
        return (Tensor(_nd(dx, (N, C, L), dtype=self._dt)),)


class MaxPool1d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        xd = x._data; xsh = xd.shape if hasattr(xd, 'shape') else None
        N, C, L = xsh; p, k, s = self.padding, self.kernel_size, self.stride
        _dt = _xdtype(xd)
        xf = _get_fl(xd)
        NEG_INF = float('-inf')
        Lp = L + 2*p
        x_pad = [NEG_INF]*(N*C*Lp)
        for n in range(N):
            for c in range(C):
                for ll in range(L):
                    x_pad[n*C*Lp + c*Lp + p + ll] = xf[n*C*L + c*L + ll]
        Lo = (L + 2*p - k) // s + 1
        out = [NEG_INF]*(N*C*Lo)
        for n in range(N):
            for c in range(C):
                base = n*C*Lp + c*Lp
                for i in range(Lo):
                    out[n*C*Lo + c*Lo + i] = max(x_pad[base + i*s + kk] for kk in range(k))
        rg = x.requires_grad
        result = Tensor(_nd(out, (N, C, Lo), dtype=_dt), device=x.device, requires_grad=rg)
        if rg:
            result.grad_fn = _MaxPool1dBackward(x, x_pad, (N, C, Lp), Lo, k, s, p, dt=_dt)
            result._is_leaf = False
        return result

    def __repr__(self):
        return f'MaxPool1d(kernel_size={self.kernel_size}, stride={self.stride})'


class _MaxPool2dBackward:
    def __init__(self, inp, x_pad_fl, x_pad_sh, Ho, Wo, kh, kw, sh, sw, ph, pw, dt='float32'):
        self.inputs = (inp,); self.x_pad_fl = x_pad_fl; self.x_pad_sh = x_pad_sh
        self.Ho = Ho; self.Wo = Wo; self.kh = kh; self.kw = kw
        self.sh = sh; self.sw = sw; self.ph = ph; self.pw = pw; self._dt = dt

    def backward(self, grad):
        x_inp = self.inputs[0]
        N, C, H, W = x_inp._data.shape if hasattr(x_inp._data, 'shape') else (0,0,0,0)
        gf = _get_fl(grad._data)
        xpf = self.x_pad_fl; _, _, Hp, Wp = self.x_pad_sh
        dx_pad = [0.0]*(N*C*Hp*Wp)
        kh, kw, sh, sw = self.kh, self.kw, self.sh, self.sw
        Ho, Wo = self.Ho, self.Wo
        for n in range(N):
            for c in range(C):
                for i in range(Ho):
                    for j in range(Wo):
                        base_p = n*C*Hp*Wp + c*Hp*Wp
                        win = [xpf[base_p + (i*sh+ki)*Wp + (j*sw+kj)] for ki in range(kh) for kj in range(kw)]
                        mx = max(win); cnt = sum(1 for v in win if v == mx) or 1
                        g_val = gf[n*C*Ho*Wo + c*Ho*Wo + i*Wo + j] / cnt
                        for ki in range(kh):
                            for kj in range(kw):
                                if win[ki*kw+kj] == mx:
                                    dx_pad[base_p + (i*sh+ki)*Wp + (j*sw+kj)] += g_val
        ph, pw = self.ph, self.pw
        if ph > 0 or pw > 0:
            dx = [dx_pad[n*C*Hp*Wp + c*Hp*Wp + (ph+h)*Wp + (pw+w)] for n in range(N) for c in range(C) for h in range(H) for w in range(W)]
        else:
            dx = dx_pad
        return (Tensor(_nd(dx, (N, C, H, W), dtype=self._dt)),)


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = (stride if isinstance(stride, tuple) else (stride, stride)) if stride else self.kernel_size
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def forward(self, x: Tensor) -> Tensor:
        xd = x._data; N, C, H, W = xd.shape if hasattr(xd, 'shape') else (0,0,0,0)
        kh, kw = self.kernel_size; sh, sw = self.stride; ph, pw = self.padding
        _dt = _xdtype(xd)
        xf = _get_fl(xd); NEG_INF = float('-inf')
        Hp, Wp = H+2*ph, W+2*pw
        x_pad = [NEG_INF]*(N*C*Hp*Wp)
        for n in range(N):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        x_pad[n*C*Hp*Wp + c*Hp*Wp + (ph+h)*Wp + (pw+w)] = xf[n*C*H*W + c*H*W + h*W + w]
        Ho = (H+2*ph-kh)//sh+1; Wo = (W+2*pw-kw)//sw+1
        out = [0.0]*(N*C*Ho*Wo)
        for n in range(N):
            for c in range(C):
                base_p = n*C*Hp*Wp + c*Hp*Wp
                for i in range(Ho):
                    for j in range(Wo):
                        out[n*C*Ho*Wo + c*Ho*Wo + i*Wo + j] = max(x_pad[base_p+(i*sh+ki)*Wp+(j*sw+kj)] for ki in range(kh) for kj in range(kw))
        rg = x.requires_grad
        result = Tensor(_nd(out, (N, C, Ho, Wo), dtype=_dt), device=x.device, requires_grad=rg)
        if rg:
            result.grad_fn = _MaxPool2dBackward(x, x_pad, (N,C,Hp,Wp), Ho, Wo, kh, kw, sh, sw, ph, pw, dt=_dt)
            result._is_leaf = False
        return result

    def __repr__(self):
        return f'MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride})'


class MaxPool3d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)*3
        self.stride = (stride if isinstance(stride, tuple) else (stride,)*3) if stride else self.kernel_size
        self.padding = padding if isinstance(padding, tuple) else (padding,)*3

    def forward(self, x: Tensor) -> Tensor:
        xd = x._data; N, C, D, H, W = xd.shape if hasattr(xd, 'shape') else (0,0,0,0,0)
        kd, kh, kw = self.kernel_size; sd, sh, sw = self.stride; pd, ph, pw = self.padding
        _dt = _xdtype(xd)
        xf = _get_fl(xd); NEG_INF = float('-inf')
        Dp, Hp, Wp = D+2*pd, H+2*ph, W+2*pw
        x_pad = [NEG_INF]*(N*C*Dp*Hp*Wp)
        for n in range(N):
            for c in range(C):
                for d in range(D):
                    for h in range(H):
                        for w in range(W):
                            x_pad[n*C*Dp*Hp*Wp + c*Dp*Hp*Wp + (pd+d)*Hp*Wp + (ph+h)*Wp + (pw+w)] = xf[n*C*D*H*W + c*D*H*W + d*H*W + h*W + w]
        Do = (D+2*pd-kd)//sd+1; Ho = (H+2*ph-kh)//sh+1; Wo = (W+2*pw-kw)//sw+1
        out = [0.0]*(N*C*Do*Ho*Wo)
        # Store argmax positions for backward
        argmax = [0]*(N*C*Do*Ho*Wo)
        for n in range(N):
            for c in range(C):
                base_p = n*C*Dp*Hp*Wp + c*Dp*Hp*Wp
                for dd in range(Do):
                    for i in range(Ho):
                        for j in range(Wo):
                            best_val = float('-inf'); best_idx = 0
                            for kdd in range(kd):
                                for ki in range(kh):
                                    for kj in range(kw):
                                        idx = base_p + (dd*sd+kdd)*Hp*Wp + (i*sh+ki)*Wp + (j*sw+kj)
                                        if x_pad[idx] > best_val:
                                            best_val = x_pad[idx]; best_idx = idx
                            oi = n*C*Do*Ho*Wo + c*Do*Ho*Wo + dd*Ho*Wo + i*Wo + j
                            out[oi] = best_val; argmax[oi] = best_idx
        rg = x.requires_grad
        result = Tensor(_nd(out, (N, C, Do, Ho, Wo), dtype=_dt), device=x.device, requires_grad=rg)
        if rg:
            _x = x; _xshape = (N,C,D,H,W)
            _argmax = argmax; _pd = pd; _ph = ph; _pw = pw
            class _MaxPool3dBackward:
                def __init__(s): s.inputs = (_x,)
                def backward(s, grad):
                    gf = _get_fl(grad._data)
                    dx_pad = [0.0]*(N*C*Dp*Hp*Wp)
                    for oi, gi in enumerate(gf):
                        dx_pad[_argmax[oi]] += gi
                    _N,_C,_D,_H,_W = _xshape
                    dx = [dx_pad[n*_C*Dp*Hp*Wp + c*Dp*Hp*Wp + (_pd+d)*Hp*Wp + (_ph+h)*Wp + (_pw+w)]
                          for n in range(_N) for c in range(_C) for d in range(_D) for h in range(_H) for w in range(_W)]
                    return (Tensor(_nd(dx, _xshape, dtype=_dt)),)
            result.grad_fn = _MaxPool3dBackward(); result._is_leaf = False
        return result


class AvgPool1d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        xd = x._data; N, C, L = xd.shape if hasattr(xd, 'shape') else (0,0,0)
        p, k, s = self.padding, self.kernel_size, self.stride
        _dt = _xdtype(xd)
        xf = _get_fl(xd); Lp = L+2*p
        x_pad = [0.0]*(N*C*Lp)
        for n in range(N):
            for c in range(C):
                for ll in range(L):
                    x_pad[n*C*Lp + c*Lp + p + ll] = xf[n*C*L + c*L + ll]
        Lo = (L+2*p-k)//s+1
        out = [sum(x_pad[n*C*Lp + c*Lp + i*s + kk] for kk in range(k)) / k
               for n in range(N) for c in range(C) for i in range(Lo)]
        return Tensor(_nd(out, (N, C, Lo), dtype=_dt), device=x.device)


class _AvgPool2dBackward:
    def __init__(self, inp, Ho, Wo, kh, kw, sh, sw, ph, pw, dt='float32'):
        self.inputs = (inp,)
        self.Ho = Ho; self.Wo = Wo; self.kh = kh; self.kw = kw
        self.sh = sh; self.sw = sw; self.ph = ph; self.pw = pw; self._dt = dt

    def backward(self, grad):
        x_inp = self.inputs[0]
        N, C, H, W = x_inp._data.shape if hasattr(x_inp._data, 'shape') else (0,0,0,0)
        gf = _get_fl(grad._data)
        kh, kw, sh, sw, ph, pw = self.kh, self.kw, self.sh, self.sw, self.ph, self.pw
        Ho, Wo = self.Ho, self.Wo
        Hp, Wp = H+2*ph, W+2*pw
        dx_pad = [0.0]*(N*C*Hp*Wp)
        scale = 1.0 / (kh * kw)
        for n in range(N):
            for c in range(C):
                for i in range(Ho):
                    for j in range(Wo):
                        g_val = gf[n*C*Ho*Wo + c*Ho*Wo + i*Wo + j] * scale
                        for ki in range(kh):
                            for kj in range(kw):
                                dx_pad[n*C*Hp*Wp + c*Hp*Wp + (i*sh+ki)*Wp + (j*sw+kj)] += g_val
        if ph > 0 or pw > 0:
            dx = [dx_pad[n*C*Hp*Wp + c*Hp*Wp + (ph+h)*Wp + (pw+w)] for n in range(N) for c in range(C) for h in range(H) for w in range(W)]
        else:
            dx = dx_pad
        _dt2 = _xdtype(x_inp._data)
        return (Tensor(_nd(dx, (N, C, H, W), dtype=_dt2)),)


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = (stride if isinstance(stride, tuple) else (stride, stride)) if stride else self.kernel_size
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

    def forward(self, x: Tensor) -> Tensor:
        xd = x._data; N, C, H, W = xd.shape if hasattr(xd, 'shape') else (0,0,0,0)
        kh, kw = self.kernel_size; sh, sw = self.stride; ph, pw = self.padding
        _dt = _xdtype(xd)
        xf = _get_fl(xd); Hp, Wp = H+2*ph, W+2*pw
        x_pad = [0.0]*(N*C*Hp*Wp)
        for n in range(N):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        x_pad[n*C*Hp*Wp + c*Hp*Wp + (ph+h)*Wp + (pw+w)] = xf[n*C*H*W + c*H*W + h*W + w]
        Ho = (H+2*ph-kh)//sh+1; Wo = (W+2*pw-kw)//sw+1; scale = 1.0/(kh*kw)
        out = [sum(x_pad[n*C*Hp*Wp + c*Hp*Wp + (i*sh+ki)*Wp + (j*sw+kj)] for ki in range(kh) for kj in range(kw)) * scale
               for n in range(N) for c in range(C) for i in range(Ho) for j in range(Wo)]
        rg = x.requires_grad
        result = Tensor(_nd(out, (N, C, Ho, Wo), dtype=_dt), device=x.device, requires_grad=rg)
        if rg:
            result.grad_fn = _AvgPool2dBackward(x, Ho, Wo, kh, kw, sh, sw, ph, pw, dt=_dt)
            result._is_leaf = False
        return result

    def __repr__(self):
        return f'AvgPool2d(kernel_size={self.kernel_size}, stride={self.stride})'


class AvgPool3d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)*3
        self.stride = (stride if isinstance(stride, tuple) else (stride,)*3) if stride else self.kernel_size
        self.padding = padding if isinstance(padding, tuple) else (padding,)*3

    def forward(self, x: Tensor) -> Tensor:
        xd = x._data; N, C, D, H, W = xd.shape if hasattr(xd, 'shape') else (0,0,0,0,0)
        kd, kh, kw = self.kernel_size; sd, sh, sw = self.stride; pd, ph, pw = self.padding
        _dt = _xdtype(xd)
        xf = _get_fl(xd); Dp, Hp, Wp = D+2*pd, H+2*ph, W+2*pw
        x_pad = [0.0]*(N*C*Dp*Hp*Wp)
        for n in range(N):
            for c in range(C):
                for d in range(D):
                    for h in range(H):
                        for w in range(W):
                            x_pad[n*C*Dp*Hp*Wp + c*Dp*Hp*Wp + (pd+d)*Hp*Wp + (ph+h)*Wp + (pw+w)] = xf[n*C*D*H*W + c*D*H*W + d*H*W + h*W + w]
        Do = (D+2*pd-kd)//sd+1; Ho = (H+2*ph-kh)//sh+1; Wo = (W+2*pw-kw)//sw+1; scale = 1.0/(kd*kh*kw)
        out = [sum(x_pad[n*C*Dp*Hp*Wp + c*Dp*Hp*Wp + (dd*sd+kdd)*Hp*Wp + (i*sh+ki)*Wp + (j*sw+kj)]
                   for kdd in range(kd) for ki in range(kh) for kj in range(kw)) * scale
               for n in range(N) for c in range(C) for dd in range(Do) for i in range(Ho) for j in range(Wo)]
        rg = x.requires_grad
        result = Tensor(_nd(out, (N, C, Do, Ho, Wo), dtype=_dt), device=x.device, requires_grad=rg)
        if rg:
            _x = x; _xshape = (N,C,D,H,W)
            class _AvgPool3dBackward:
                def __init__(s): s.inputs = (_x,)
                def backward(s, grad):
                    gf = _get_fl(grad._data)
                    dx_pad = [0.0]*(N*C*Dp*Hp*Wp)
                    gi = 0
                    for n in range(N):
                        for c in range(C):
                            for dd in range(Do):
                                for i in range(Ho):
                                    for j in range(Wo):
                                        g_val = gf[gi] * scale; gi += 1
                                        for kdd in range(kd):
                                            for ki in range(kh):
                                                for kj in range(kw):
                                                    dx_pad[n*C*Dp*Hp*Wp + c*Dp*Hp*Wp + (dd*sd+kdd)*Hp*Wp + (i*sh+ki)*Wp + (j*sw+kj)] += g_val
                    _N,_C,_D,_H,_W = _xshape
                    dx = [dx_pad[n*_C*Dp*Hp*Wp + c*Dp*Hp*Wp + (pd+d)*Hp*Wp + (ph+h)*Wp + (pw+w)]
                          for n in range(_N) for c in range(_C) for d in range(_D) for h in range(_H) for w in range(_W)]
                    return (Tensor(_nd(dx, _xshape, dtype=_dt)),)
            result.grad_fn = _AvgPool3dBackward(); result._is_leaf = False
        return result


class AdaptiveAvgPool1d(Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        xd = x._data; N, C, L = xd.shape if hasattr(xd, 'shape') else (0,0,0)
        _dt = _xdtype(xd)
        xf = _get_fl(xd); os = self.output_size
        out = []
        for n in range(N):
            for c in range(C):
                for i in range(os):
                    s = int(i * L // os); e = int((i+1) * L // os) if i < os-1 else L
                    chunk = [xf[n*C*L + c*L + ll] for ll in range(s, e)]
                    out.append(sum(chunk) / len(chunk) if chunk else 0.0)
        return Tensor(_nd(out, (N, C, os), dtype=_dt), device=x.device)


class _AdaptiveAvgPool2dBackward:
    def __init__(self, inp, windows, oh, ow, dt='float32'):
        self.inputs = (inp,); self.windows = windows; self.oh = oh; self.ow = ow; self._dt = dt

    def backward(self, grad):
        x_inp = self.inputs[0]
        N, C, H, W = x_inp._data.shape if hasattr(x_inp._data, 'shape') else (0,0,0,0)
        gf = _get_fl(grad._data)
        dx = [0.0]*(N*C*H*W)
        for idx, (hs, he, ws, we) in enumerate(self.windows):
            i, j = idx // self.ow, idx % self.ow
            kh2 = he-hs; kw2 = we-ws; scale2 = 1.0/(kh2*kw2) if kh2*kw2 > 0 else 0.0
            for n in range(N):
                for c in range(C):
                    g_val = gf[n*C*self.oh*self.ow + c*self.oh*self.ow + i*self.ow + j] * scale2
                    for h in range(hs, he):
                        for w in range(ws, we):
                            dx[n*C*H*W + c*H*W + h*W + w] += g_val
        return (Tensor(_nd(dx, (N, C, H, W), dtype=self._dt)),)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        xd = x._data; N, C, H, W = xd.shape if hasattr(xd, 'shape') else (0,0,0,0)
        _dt = _xdtype(xd)
        xf = _get_fl(xd); oh, ow = self.output_size
        out = []; windows = []
        for i in range(oh):
            hs = int(i * H // oh); he = int((i+1) * H // oh) if i < oh-1 else H
            for j in range(ow):
                ws = int(j * W // ow); we = int((j+1) * W // ow) if j < ow-1 else W
                windows.append((hs, he, ws, we))
                kh2 = he-hs; kw2 = we-ws; cnt = kh2*kw2
                for n in range(N):
                    for c in range(C):
                        s2 = sum(xf[n*C*H*W + c*H*W + h*W + w] for h in range(hs, he) for w in range(ws, we))
                        out.append(s2 / cnt if cnt > 0 else 0.0)
        # out built in (oh*ow, N, C) order — need (N, C, oh, ow)
        out_nc = [0.0]*(N*C*oh*ow)
        idx2 = 0
        for i in range(oh):
            for j in range(ow):
                for n in range(N):
                    for c in range(C):
                        out_nc[n*C*oh*ow + c*oh*ow + i*ow + j] = out[idx2]; idx2 += 1
        rg = x.requires_grad
        result = Tensor(_nd(out_nc, (N, C, oh, ow), dtype=_dt), device=x.device, requires_grad=rg)
        if rg:
            result.grad_fn = _AdaptiveAvgPool2dBackward(x, windows, oh, ow, dt=_dt)
            result._is_leaf = False
        return result

    def __repr__(self):
        return f'AdaptiveAvgPool2d(output_size={self.output_size})'


class AdaptiveMaxPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        xd = x._data; N, C, H, W = xd.shape if hasattr(xd, 'shape') else (0,0,0,0)
        _dt = _xdtype(xd)
        xf = _get_fl(xd); oh, ow = self.output_size
        out_nc = [0.0]*(N*C*oh*ow); argmax = [0]*(N*C*oh*ow)
        for i in range(oh):
            hs = int(_math_pool.floor(i * H / oh)); he = int(_math_pool.ceil((i+1) * H / oh))
            for j in range(ow):
                ws = int(_math_pool.floor(j * W / ow)); we = int(_math_pool.ceil((j+1) * W / ow))
                for n in range(N):
                    for c in range(C):
                        best_val = float('-inf'); best_flat = 0
                        for h in range(hs, he):
                            for w in range(ws, we):
                                v = xf[n*C*H*W + c*H*W + h*W + w]
                                if v > best_val:
                                    best_val = v; best_flat = n*C*H*W + c*H*W + h*W + w
                        oi = n*C*oh*ow + c*oh*ow + i*ow + j
                        out_nc[oi] = best_val; argmax[oi] = best_flat
        rg = x.requires_grad
        result = Tensor(_nd(out_nc, (N, C, oh, ow), dtype=_dt), device=x.device, requires_grad=rg)
        if rg:
            _x = x; _xshape = (N, C, H, W); _argmax = argmax
            class _AdaptiveMaxPool2dBwd:
                def __init__(s): s.inputs = (_x,)
                def backward(s, grad):
                    gf = _get_fl(grad._data)
                    dx = [0.0]*(N*C*H*W)
                    for oi, gi in enumerate(gf):
                        dx[_argmax[oi]] += gi
                    return (Tensor(_nd(dx, _xshape, dtype=_dt)),)
            result.grad_fn = _AdaptiveMaxPool2dBwd(); result._is_leaf = False
        return result


GlobalAvgPool2d = lambda: AdaptiveAvgPool2d(1)
