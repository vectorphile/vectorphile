"""Activation function modules."""
import numpy as np
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor


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
        return Tensor(np.where(x._data >= 0, x._data, self.negative_slope * x._data), device=x.device)
    def __repr__(self): return f'LeakyReLU(negative_slope={self.negative_slope})'


class ELU(Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.where(x._data >= 0, x._data, self.alpha * (np.exp(x._data) - 1)), device=x.device)
    def __repr__(self): return f'ELU(alpha={self.alpha})'


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        d = x._data
        return Tensor(0.5 * d * (1 + np.tanh(np.sqrt(2/np.pi) * (d + 0.044715 * d**3))), device=x.device)
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
        from vectorphile.core.tensor import softmax
        return softmax(x, self.dim)
    def __repr__(self): return f'Softmax(dim={self.dim})'


class LogSoftmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, x: Tensor) -> Tensor:
        from vectorphile.core.tensor import log_softmax
        return log_softmax(x, self.dim)
    def __repr__(self): return f'LogSoftmax(dim={self.dim})'


class SELU(Module):
    _alpha = 1.6732632423543772
    _scale = 1.0507009873554805
    def forward(self, x: Tensor) -> Tensor:
        d = x._data
        return Tensor(self._scale * np.where(d >= 0, d, self._alpha * (np.exp(d) - 1)), device=x.device)
    def __repr__(self): return 'SELU()'


class SiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(x._data * (1 / (1 + np.exp(-x._data))), device=x.device)
    def __repr__(self): return 'SiLU()'


class Mish(Module):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(x._data * np.tanh(np.log1p(np.exp(x._data))), device=x.device)
    def __repr__(self): return 'Mish()'


class Hardswish(Module):
    def forward(self, x: Tensor) -> Tensor:
        d = x._data
        return Tensor(d * np.clip(d + 3, 0, 6) / 6, device=x.device)
    def __repr__(self): return 'Hardswish()'
