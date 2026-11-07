"""Weight initialization functions."""
from __future__ import annotations
import numpy as np
import math
from vectorphile.core.tensor import Tensor


def _calculate_fan(tensor: Tensor):
    ndim = tensor.ndim
    if ndim < 2:
        raise ValueError("Fan calculation requires tensor with >= 2 dimensions")
    fan_in = tensor.shape[1]
    fan_out = tensor.shape[0]
    if ndim > 2:
        receptive = int(np.prod(tensor.shape[2:]))
        fan_in *= receptive
        fan_out *= receptive
    return fan_in, fan_out


def uniform_(tensor: Tensor, a=0.0, b=1.0) -> Tensor:
    tensor._data[:] = np.random.uniform(a, b, tensor.shape).astype(tensor._data.dtype)
    return tensor


def normal_(tensor: Tensor, mean=0.0, std=1.0) -> Tensor:
    tensor._data[:] = np.random.normal(mean, std, tensor.shape).astype(tensor._data.dtype)
    return tensor


def constant_(tensor: Tensor, val: float) -> Tensor:
    tensor._data[:] = val
    return tensor


def zeros_(tensor: Tensor) -> Tensor:
    tensor._data[:] = 0
    return tensor


def ones_(tensor: Tensor) -> Tensor:
    tensor._data[:] = 1
    return tensor


def eye_(tensor: Tensor) -> Tensor:
    assert tensor.ndim == 2, "eye_ requires 2D tensor"
    tensor._data[:] = np.eye(*tensor.shape, dtype=tensor._data.dtype)
    return tensor


def trunc_normal_(tensor: Tensor, mean=0.0, std=1.0, a=-2.0, b=2.0) -> Tensor:
    vals = np.random.normal(mean, std, tensor.shape)
    vals = np.clip(vals, a, b)
    tensor._data[:] = vals.astype(tensor._data.dtype)
    return tensor


def xavier_uniform_(tensor: Tensor, gain=1.0) -> Tensor:
    fan_in, fan_out = _calculate_fan(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    bound = math.sqrt(3.0) * std
    tensor._data[:] = np.random.uniform(-bound, bound, tensor.shape).astype(tensor._data.dtype)
    return tensor


def xavier_normal_(tensor: Tensor, gain=1.0) -> Tensor:
    fan_in, fan_out = _calculate_fan(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    tensor._data[:] = np.random.normal(0, std, tensor.shape).astype(tensor._data.dtype)
    return tensor


def kaiming_uniform_(tensor: Tensor, a=0, mode='fan_in', nonlinearity='leaky_relu') -> Tensor:
    fan_in, fan_out = _calculate_fan(tensor)
    fan = fan_in if mode == 'fan_in' else fan_out
    neg_slope = a
    gain = math.sqrt(2.0 / (1 + neg_slope ** 2))
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    tensor._data[:] = np.random.uniform(-bound, bound, tensor.shape).astype(tensor._data.dtype)
    return tensor


def kaiming_normal_(tensor: Tensor, a=0, mode='fan_in', nonlinearity='leaky_relu') -> Tensor:
    fan_in, fan_out = _calculate_fan(tensor)
    fan = fan_in if mode == 'fan_in' else fan_out
    neg_slope = a
    gain = math.sqrt(2.0 / (1 + neg_slope ** 2))
    std = gain / math.sqrt(fan)
    tensor._data[:] = np.random.normal(0, std, tensor.shape).astype(tensor._data.dtype)
    return tensor


def orthogonal_(tensor: Tensor, gain=1.0) -> Tensor:
    rows = tensor.shape[0]
    cols = int(np.prod(tensor.shape[1:]))
    flat = np.random.normal(0, 1, (rows, cols))
    if rows < cols:
        flat = flat.T
    q, r = np.linalg.qr(flat)
    d = np.diagonal(r)
    q *= np.sign(d)
    if rows < cols:
        q = q.T
    tensor._data[:] = (gain * q.reshape(tensor.shape)).astype(tensor._data.dtype)
    return tensor


def sparse_(tensor: Tensor, sparsity: float, std=0.01) -> Tensor:
    assert tensor.ndim == 2, "sparse_ requires 2D tensor"
    rows, cols = tensor.shape
    num_zeros = int(math.ceil(sparsity * rows))
    tensor._data[:] = np.random.normal(0, std, tensor.shape).astype(tensor._data.dtype)
    for col in range(cols):
        row_indices = np.random.choice(rows, size=num_zeros, replace=False)
        tensor._data[row_indices, col] = 0
    return tensor


def dirac_(tensor: Tensor, groups=1) -> Tensor:
    assert tensor.ndim >= 3, "dirac_ requires >= 3D tensor"
    sizes = tensor.shape
    min_dim = min(sizes[0] // groups, sizes[1])
    tensor._data[:] = 0
    for g in range(groups):
        for d in range(min_dim):
            center = tuple(s // 2 for s in sizes[2:])
            tensor._data[g * (sizes[0] // groups) + d, d][center] = 1
    return tensor


# Convenience gain lookup
def calculate_gain(nonlinearity: str, param=None) -> float:
    table = {
        'linear': 1.0, 'conv1d': 1.0, 'conv2d': 1.0, 'conv3d': 1.0,
        'sigmoid': 1.0, 'tanh': 5.0/3.0, 'relu': math.sqrt(2.0),
        'selu': 3.0/4.0, 'leaky_relu': math.sqrt(2.0 / (1 + (param or 0.01)**2)),
    }
    return table.get(nonlinearity, 1.0)
