"""Neural network layers."""
from __future__ import annotations
import numpy as np
import math
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor
from vectorphile.autograd.engine import Function


class _LinearBackward(Function):
    """Custom backward for y = x @ W.T  (W is weight matrix, shape out×in)."""
    def __init__(self, x, weight):
        super().__init__(x, weight)
        self.x = x
        self.weight = weight

    def backward(self, grad):
        # grad shape: (N, out)
        # dL/dx = grad @ W        (N, in)
        # dL/dW = grad.T @ x      (out, in)
        dx = Tensor(grad._data @ self.weight._data)
        dw = Tensor(grad._data.T @ self.x._data)
        return dx, dw


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        k = math.sqrt(1.0 / in_features)
        self.weight = Tensor(
            np.random.uniform(-k, k, (out_features, in_features)).astype(np.float32),
            requires_grad=True
        )
        self.bias_param = None
        if bias:
            self.bias_param = Tensor(
                np.zeros(out_features, dtype=np.float32), requires_grad=True
            )
            self._parameters['bias_param'] = self.bias_param
        self._parameters['weight'] = self.weight

    def forward(self, x: Tensor) -> Tensor:
        # Manually compute matmul to keep weight in autograd graph
        # x: (N, in), weight: (out, in) → out: (N, out)
        from vectorphile.autograd.engine import MatMulBackward
        import numpy as np
        result_data = x._data @ self.weight._data.T
        req_grad = x.requires_grad or self.weight.requires_grad
        # Create a transposed view for the backward pass
        weight_t = Tensor(self.weight._data.T, device=self.weight.device, requires_grad=False)
        weight_t._data = self.weight._data.T  # view, not copy
        result = Tensor(result_data, device=x.device, requires_grad=req_grad)
        if req_grad:
            # Custom backward: dL/dx = grad @ W,  dL/dW = grad.T @ x
            result.grad_fn = _LinearBackward(x, self.weight)
            result._is_leaf = False
        if self.bias_param is not None:
            result = result + self.bias_param
        return result

    def __repr__(self):
        return f'Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias_param is not None})'


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
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
        N, C, H, W = x._data.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        x_pad = np.pad(x._data, ((0,0),(0,0),(ph,ph),(pw,pw)), mode='constant')
        Ho = (H + 2*ph - kh) // sh + 1
        Wo = (W + 2*pw - kw) // sw + 1
        out = np.zeros((N, self.out_channels, Ho, Wo), dtype=np.float32)
        w = self.weight._data
        for i in range(Ho):
            for j in range(Wo):
                patch = x_pad[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw]
                out[:, :, i, j] = np.einsum('nckl,ockl->no', patch, w)
        if self.bias_param is not None:
            out += self.bias_param._data[None, :, None, None]
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
            self.weight = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
            self.bias_param = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
            self._parameters['weight'] = self.weight
            self._parameters['bias_param'] = self.bias_param
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

    def forward(self, x: Tensor) -> Tensor:
        data = x._data
        if self.training:
            mean = data.mean(axis=0)
            var = data.var(axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean, var = self.running_mean, self.running_var
        x_norm = (data - mean) / np.sqrt(var + self.eps)
        if self.affine:
            x_norm = x_norm * self.weight._data + self.bias_param._data
        return Tensor(x_norm.astype(np.float32), device=x.device)


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.weight = Tensor(np.ones(num_features, dtype=np.float32), requires_grad=True)
            self.bias_param = Tensor(np.zeros(num_features, dtype=np.float32), requires_grad=True)
            self._parameters['weight'] = self.weight
            self._parameters['bias_param'] = self.bias_param
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)

    def forward(self, x: Tensor) -> Tensor:
        data = x._data  # (N, C, H, W)
        if self.training:
            mean = data.mean(axis=(0, 2, 3))
            var = data.var(axis=(0, 2, 3))
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean, var = self.running_mean, self.running_var
        x_norm = (data - mean[None,:,None,None]) / np.sqrt(var[None,:,None,None] + self.eps)
        if self.affine:
            x_norm = x_norm * self.weight._data[None,:,None,None] + self.bias_param._data[None,:,None,None]
        return Tensor(x_norm.astype(np.float32), device=x.device)


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Tensor(np.ones(normalized_shape, dtype=np.float32), requires_grad=True)
            self.bias_param = Tensor(np.zeros(normalized_shape, dtype=np.float32), requires_grad=True)
            self._parameters['weight'] = self.weight
            self._parameters['bias_param'] = self.bias_param

    def forward(self, x: Tensor) -> Tensor:
        ndim = len(self.normalized_shape)
        axes = tuple(range(-ndim, 0))
        mean = x._data.mean(axis=axes, keepdims=True)
        var = x._data.var(axis=axes, keepdims=True)
        x_norm = (x._data - mean) / np.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_norm = x_norm * self.weight._data + self.bias_param._data
        return Tensor(x_norm.astype(np.float32), device=x.device)


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
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02,
            requires_grad=True
        )
        self._parameters['weight'] = self.weight

    def forward(self, indices: Tensor) -> Tensor:
        idx = indices._data.astype(int)
        out = self.weight._data[idx]
        if self.padding_idx is not None:
            mask = (idx == self.padding_idx)[..., None]
            out = np.where(mask, 0.0, out)
        return Tensor(out.astype(np.float32), device=indices.device)

    def __repr__(self):
        return f'Embedding({self.num_embeddings}, {self.embedding_dim})'
