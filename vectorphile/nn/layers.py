"""Neural network layers — all heavy ops dispatched through vectorphile.backend."""
from __future__ import annotations
import numpy as np
import math
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor
from vectorphile.autograd.engine import Function
import vectorphile.backend as _backend


class _LinearBackward(Function):
    def __init__(self, x, weight):
        super().__init__(x, weight)
        self.x = x
        self.weight = weight

    def backward(self, grad):
        dx = Tensor(grad._data @ self.weight._data)
        dw = Tensor(grad._data.T @ self.x._data)
        return dx, dw


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        k = math.sqrt(1.0 / in_features)
        self.weight = Tensor(
            np.random.uniform(-k, k, (out_features, in_features)).astype(np.float32),
            requires_grad=True
        )
        self.bias_param = None
        if bias:
            self.bias_param = Tensor(np.zeros(out_features, dtype=np.float32), requires_grad=True)
            self._parameters['bias_param'] = self.bias_param
        self._parameters['weight'] = self.weight

    def forward(self, x: Tensor) -> Tensor:
        # Dispatch matmul through backend (torch/CUDA/numpy)
        result_data = _backend.matmul(x._data, self.weight._data.T)
        req_grad = x.requires_grad or self.weight.requires_grad
        result = Tensor(result_data, device=x.device, requires_grad=req_grad)
        if req_grad:
            result.grad_fn = _LinearBackward(x, self.weight)
            result._is_leaf = False
        if self.bias_param is not None:
            result = result + self.bias_param
        return result

    @property
    def bias(self):
        return self.bias_param

    def __repr__(self):
        return f'Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias_param is not None})'


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
        self.weight = Tensor(
            np.random.uniform(-k, k, (out_channels, in_channels, kernel_size)).astype(np.float32),
            requires_grad=True
        )
        self._parameters['weight'] = self.weight
        self.bias_param = None
        if bias:
            self.bias_param = Tensor(np.zeros(out_channels, dtype=np.float32), requires_grad=True)
            self._parameters['bias_param'] = self.bias_param

    @property
    def bias(self):
        return self.bias_param

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, L]
        from scipy.signal import fftconvolve
        xd = x._data
        wd = self.weight._data
        B, C_in, L = xd.shape
        C_out, _, K = wd.shape
        p = self.padding
        s = self.stride
        if p > 0:
            xd = np.pad(xd, ((0,0),(0,0),(p,p)))
        L_out = (L + 2*p - K) // s + 1
        out = np.zeros((B, C_out, L_out), dtype=np.float32)
        for oc in range(C_out):
            for ic in range(C_in):
                for b in range(B):
                    conv = np.convolve(xd[b, ic], wd[oc, ic, ::-1], mode='valid')
                    out[b, oc] += conv[::s][:L_out]
        if self.bias_param is not None:
            out += self.bias_param._data[None, :, None]
        return Tensor(out, device=x.device)


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
        self.weight = Tensor(
            np.random.uniform(-k, k, (out_channels, in_channels, kh, kw)).astype(np.float32),
            requires_grad=True
        )
        self._parameters['weight'] = self.weight
        self.bias_param = None
        if bias:
            self.bias_param = Tensor(np.zeros(out_channels, dtype=np.float32), requires_grad=True)
            self._parameters['bias_param'] = self.bias_param

    def forward(self, x: Tensor) -> Tensor:
        # Use backend.conv2d (torch CUDA or im2col+BLAS — always faster than loops)
        b = self.bias_param._data if self.bias_param is not None else None
        sh, sw = self.stride
        ph, pw = self.padding
        # Backend conv2d expects single int stride/padding; use first value
        out = _backend.conv2d(x._data, self.weight._data, b, stride=sh, padding=ph)
        return Tensor(out, device=x.device)

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
            self.weight     = Tensor(np.ones(num_features,  dtype=np.float32), requires_grad=True)
            self.bias_param = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
            self._parameters['weight']     = self.weight
            self._parameters['bias_param'] = self.bias_param
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var  = np.ones(num_features,  dtype=np.float32)

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight._data     if self.affine else np.ones(self.num_features,  np.float32)
        b = self.bias_param._data if self.affine else np.zeros(self.num_features, np.float32)
        out = _backend.batch_norm(x._data, w, b, self.running_mean, self.running_var,
                                   self.training, self.momentum, self.eps)
        return Tensor(out.astype(np.float32), device=x.device)


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.weight     = Tensor(np.ones(num_features,  dtype=np.float32), requires_grad=True)
            self.bias_param = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
            self._parameters['weight']     = self.weight
            self._parameters['bias_param'] = self.bias_param
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var  = np.ones(num_features,  dtype=np.float32)

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight._data     if self.affine else np.ones(self.num_features,  np.float32)
        b = self.bias_param._data if self.affine else np.zeros(self.num_features, np.float32)
        out = _backend.batch_norm(x._data, w, b, self.running_mean, self.running_var,
                                   self.training, self.momentum, self.eps)
        return Tensor(out.astype(np.float32), device=x.device)


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight     = Tensor(np.ones(normalized_shape,  dtype=np.float32), requires_grad=True)
            self.bias_param = Tensor(np.zeros(normalized_shape, dtype=np.float32), requires_grad=True)
            self._parameters['weight']     = self.weight
            self._parameters['bias_param'] = self.bias_param

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight._data     if self.elementwise_affine else np.ones(self.normalized_shape,  np.float32)
        b = self.bias_param._data if self.elementwise_affine else np.zeros(self.normalized_shape, np.float32)
        out = _backend.layer_norm(x._data, w, b, self.eps)
        return Tensor(out.astype(np.float32), device=x.device)


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        mask = (np.random.rand(*x._data.shape) > self.p).astype(np.float32)
        return Tensor(x._data * mask / (1.0 - self.p), device=x.device, requires_grad=x.requires_grad)

    def __repr__(self):
        return f'Dropout(p={self.p})'


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.padding_idx    = padding_idx
        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02,
            requires_grad=True
        )
        self._parameters['weight'] = self.weight

    def forward(self, indices: Tensor) -> Tensor:
        # Dispatch through backend (torch embedding handles padding_idx natively)
        out = _backend.embedding(self.weight._data, indices._data)
        if self.padding_idx is not None:
            idx = indices._data.astype(int)
            mask = (idx == self.padding_idx)[..., None]
            out  = np.where(mask, 0.0, out)
        return Tensor(out.astype(np.float32), device=indices.device)

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
