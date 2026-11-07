"""Additional normalization layers."""
from __future__ import annotations
import numpy as np
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
            self.weight = Tensor(np.ones(num_channels, dtype=np.float32), requires_grad=True)
            self.bias_param = Tensor(np.zeros(num_channels, dtype=np.float32), requires_grad=True)
            self._parameters['weight'] = self.weight
            self._parameters['bias_param'] = self.bias_param

    def forward(self, x: Tensor) -> Tensor:
        N, C = x._data.shape[:2]
        rest = x._data.shape[2:]
        data = x._data.reshape(N, self.num_groups, C // self.num_groups, *rest)
        axes = tuple(range(2, data.ndim))
        mean = data.mean(axis=axes, keepdims=True)
        var = data.var(axis=axes, keepdims=True)
        data = (data - mean) / np.sqrt(var + self.eps)
        data = data.reshape(N, C, *rest)
        if self.affine:
            data = data * self.weight._data.reshape(1, C, *[1]*len(rest))
            data = data + self.bias_param._data.reshape(1, C, *[1]*len(rest))
        return Tensor(data.astype(np.float32), device=x.device)

    def __repr__(self):
        return f'GroupNorm(num_groups={self.num_groups}, num_channels={self.num_channels})'


class InstanceNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
            self.bias_param = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
            self._parameters['weight'] = self.weight
            self._parameters['bias_param'] = self.bias_param

    def forward(self, x: Tensor) -> Tensor:
        N, C, L = x._data.shape
        mean = x._data.mean(axis=2, keepdims=True)
        var = x._data.var(axis=2, keepdims=True)
        out = (x._data - mean) / np.sqrt(var + self.eps)
        if self.affine:
            out = out * self.weight._data[None, :, None] + self.bias_param._data[None, :, None]
        return Tensor(out.astype(np.float32), device=x.device)


class InstanceNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
            self.bias_param = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
            self._parameters['weight'] = self.weight
            self._parameters['bias_param'] = self.bias_param

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x._data.shape
        mean = x._data.mean(axis=(2, 3), keepdims=True)
        var = x._data.var(axis=(2, 3), keepdims=True)
        out = (x._data - mean) / np.sqrt(var + self.eps)
        if self.affine:
            out = out * self.weight._data[None, :, None, None] + self.bias_param._data[None, :, None, None]
        return Tensor(out.astype(np.float32), device=x.device)

    def __repr__(self):
        return f'InstanceNorm2d({self.num_features})'


class InstanceNorm3d(Module):
    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
            self.bias_param = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
            self._parameters['weight'] = self.weight
            self._parameters['bias_param'] = self.bias_param

    def forward(self, x: Tensor) -> Tensor:
        mean = x._data.mean(axis=(2, 3, 4), keepdims=True)
        var = x._data.var(axis=(2, 3, 4), keepdims=True)
        out = (x._data - mean) / np.sqrt(var + self.eps)
        if self.affine:
            out = out * self.weight._data[None, :, None, None, None] + self.bias_param._data[None, :, None, None, None]
        return Tensor(out.astype(np.float32), device=x.device)


class RMSNorm(Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = Tensor(np.ones(normalized_shape, dtype=np.float32), requires_grad=True)
        self._parameters['weight'] = self.weight

    def forward(self, x: Tensor) -> Tensor:
        ndim = len(self.normalized_shape)
        axes = tuple(range(-ndim, 0))
        rms = np.sqrt((x._data ** 2).mean(axis=axes, keepdims=True) + self.eps)
        return Tensor((x._data / rms * self.weight._data).astype(np.float32), device=x.device)

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
        data = x._data
        N, C, H, W = data.shape
        half = self.size // 2
        sq = data ** 2
        out = np.zeros_like(data)
        for c in range(C):
            lo = max(0, c - half)
            hi = min(C, c + half + 1)
            denom = (self.k + self.alpha * sq[:, lo:hi, :, :].sum(axis=1, keepdims=True)) ** self.beta
            out[:, c:c+1, :, :] = data[:, c:c+1, :, :] / denom
        return Tensor(out.astype(np.float32), device=x.device)
