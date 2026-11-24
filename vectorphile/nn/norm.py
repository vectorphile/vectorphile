"""Additional normalization layers."""
from __future__ import annotations
import math as _math_norm
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor


class GroupNorm(Module):
    def __init__(self, num_groups: int, num_channels: int, eps=1e-5, affine=True):
        super().__init__()
        assert num_channels % num_groups == 0
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            from vectorphile._core.ndarray import ones as _nd_ones, zeros as _nd_zeros
            self.weight = Tensor(_nd_ones(num_channels, dtype='float32'), requires_grad=True)
            self.bias_param = Tensor(_nd_zeros(num_channels, dtype='float32'), requires_grad=True)
            self._parameters['weight'] = self.weight
            self._parameters['bias'] = self._parameters.pop('bias_param', self.bias_param)

    def forward(self, x: Tensor) -> Tensor:
        N, C = x.shape[:2]
        rest = x.shape[2:]
        G = self.num_groups
        # Reshape to (N, G, C//G, *rest) using Tensor ops with grad
        gshape = (N, G, C // G) + rest
        data = x.reshape(gshape)
        # Compute mean and var over dims (2..) keeping dims
        reduce_axes = tuple(range(2, data.ndim))
        mean = data.mean(dim=reduce_axes, keepdim=True)
        # var = mean((data - mean)^2)
        diff = data - mean
        var = (diff * diff).mean(dim=reduce_axes, keepdim=True)
        # Normalize
        inv_std = (var + self.eps).sqrt()
        # We need 1/inv_std — use reciprocal via power or division
        normed = diff / inv_std
        # Reshape back to (N, C, *rest)
        normed = normed.reshape((N, C) + rest)
        if self.affine:
            w = self.weight.reshape((1, C) + (1,) * len(rest))
            b = self.bias_param.reshape((1, C) + (1,) * len(rest))
            normed = normed * w + b
        return normed

    def __repr__(self):
        return f'GroupNorm(num_groups={self.num_groups}, num_channels={self.num_channels})'


class InstanceNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            from vectorphile._core.ndarray import ones as _nd_ones, zeros as _nd_zeros
            self.weight = Tensor(_nd_ones(num_features, dtype='float32'), requires_grad=True)
            self.bias_param = Tensor(_nd_zeros(num_features, dtype='float32'), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        N, C, L = x.shape[0], x.shape[1], x.shape[2]
        mean = x.mean(dim=2, keepdim=True)
        diff = x - mean
        var  = (diff * diff).mean(dim=2, keepdim=True)
        normed = diff / (var + self.eps).sqrt()
        if self.affine:
            w = self.weight.reshape((1, C, 1))
            b = self.bias_param.reshape((1, C, 1))
            normed = normed * w + b
        return normed


class InstanceNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            from vectorphile._core.ndarray import ones as _nd_ones, zeros as _nd_zeros
            self.weight = Tensor(_nd_ones(num_features, dtype='float32'), requires_grad=True)
            self.bias_param = Tensor(_nd_zeros(num_features, dtype='float32'), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        mean = x.mean(dim=(2, 3), keepdim=True)
        diff = x - mean
        var  = (diff * diff).mean(dim=(2, 3), keepdim=True)
        normed = diff / (var + self.eps).sqrt()
        if self.affine:
            w = self.weight.reshape((1, C, 1, 1))
            b = self.bias_param.reshape((1, C, 1, 1))
            normed = normed * w + b
        return normed

    def __repr__(self):
        return f'InstanceNorm2d({self.num_features})'


class InstanceNorm3d(Module):
    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            from vectorphile._core.ndarray import ones as _nd_ones3, zeros as _nd_zeros3
            self.weight = Tensor(_nd_ones3(num_features, dtype='float32'), requires_grad=True)
            self.bias_param = Tensor(_nd_zeros3(num_features, dtype='float32'), requires_grad=True)
            self._parameters['weight'] = self.weight
            self._parameters['bias'] = self._parameters.pop('bias_param', self.bias_param)

    def forward(self, x: Tensor) -> Tensor:
        from vectorphile._core.ndarray import array as _nd_in3
        xd = x._data
        if hasattr(xd, 'to_flat_list'): xfl = xd.to_flat_list()
        else: xfl = list(xd)
        xsh = xd.shape if hasattr(xd, 'shape') else (len(xfl),)
        N3, C3, D3, H3, W3 = xsh; spatial = D3*H3*W3
        out3 = [0.0]*len(xfl)
        for n3 in range(N3):
            for c3 in range(C3):
                base3 = n3*C3*spatial + c3*spatial
                chunk3 = xfl[base3:base3+spatial]
                mu3 = sum(chunk3)/spatial
                va3 = sum((v-mu3)**2 for v in chunk3)/spatial
                inv3 = 1.0/_math_norm.sqrt(va3+self.eps)
                for i3, v3 in enumerate(chunk3): out3[base3+i3] = (v3-mu3)*inv3
        if self.affine:
            if hasattr(self.weight._data, 'to_flat_list'): wf3 = self.weight._data.to_flat_list()
            else: wf3 = list(self.weight._data)
            if hasattr(self.bias_param._data, 'to_flat_list'): bf3n = self.bias_param._data.to_flat_list()
            else: bf3n = list(self.bias_param._data)
            for n3 in range(N3):
                for c3 in range(C3):
                    base3 = n3*C3*spatial + c3*spatial
                    for i3 in range(spatial): out3[base3+i3] = out3[base3+i3]*wf3[c3]+bf3n[c3]
        return Tensor(_nd_in3(out3).reshape(xsh), device=x.device)


class RMSNorm(Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        from vectorphile._core.ndarray import ones as _nd_ones
        self.weight = Tensor(_nd_ones(normalized_shape, dtype='float32'), requires_grad=True)
        self._parameters['weight'] = self.weight

    def forward(self, x: Tensor) -> Tensor:
        ndim = len(self.normalized_shape)
        # reduce_axes: last ndim axes
        reduce_axes = tuple(range(x.ndim - ndim, x.ndim))
        # rms² = mean(x²) + eps
        x2 = x * x
        rms2 = x2.mean(dim=reduce_axes, keepdim=True) + Tensor.full((1,) * x.ndim, fill_value=self.eps)
        rms = rms2.sqrt()
        # broadcast weight to match x shape
        w = self.weight
        for _ in range(x.ndim - ndim):
            w = w.unsqueeze(0)
        return (x / rms) * w

    def __repr__(self):
        return f'RMSNorm({self.normalized_shape})'


class LocalResponseNorm(Module):
    def __init__(self, size: int, alpha=1e-4, beta=0.75, k=1.0):
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x: Tensor) -> Tensor:
        from vectorphile._core.ndarray import array as _nd_lrn
        xd = x._data
        if hasattr(xd, 'to_flat_list'): xfl = xd.to_flat_list()
        else: xfl = list(xd)
        xsh = xd.shape if hasattr(xd, 'shape') else (len(xfl),)
        N, C, H, W = xsh; half = self.size // 2
        sq = [v*v for v in xfl]
        out_lrn = [0.0]*len(xfl)
        for n in range(N):
            for c in range(C):
                lo = max(0, c-half); hi = min(C, c+half+1)
                for h in range(H):
                    for w in range(W):
                        s = sum(sq[n*C*H*W + cc*H*W + h*W + w] for cc in range(lo, hi))
                        denom = (self.k + self.alpha * s) ** self.beta
                        out_lrn[n*C*H*W + c*H*W + h*W + w] = xfl[n*C*H*W + c*H*W + h*W + w] / denom
        return Tensor(_nd_lrn(out_lrn).reshape(xsh), device=x.device)
