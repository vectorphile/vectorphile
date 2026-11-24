"""Weight initialization functions — pure Python."""
from __future__ import annotations
import math
import random as _random
from vectorphile.core.tensor import Tensor


def _randn_box_muller():
    u1 = max(_random.random(), 1e-10); u2 = _random.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def _get_flat(d):
    if hasattr(d, 'to_flat_list'): return d.to_flat_list()
    if hasattr(d, 'tolist'): return d.tolist()
    if isinstance(d, list): return d
    return list(d)


def _set_flat(d, vals):
    """Set all elements of an NDArray-like from flat list."""
    if hasattr(d, '_buf') and d._buf is not None:
        for i, v in enumerate(vals): d._buf[i] = float(v)
    elif hasattr(d, '__setitem__'):
        flat = _get_flat(d)
        for i, v in enumerate(vals): flat[i] = float(v)


def _prod(shape):
    r = 1
    for s in shape: r *= s
    return r


def _calculate_fan(tensor: Tensor):
    if tensor.ndim < 2:
        raise ValueError("Fan calculation requires tensor with >= 2 dimensions")
    fan_in = tensor.shape[1]; fan_out = tensor.shape[0]
    if tensor.ndim > 2:
        receptive = _prod(tensor.shape[2:])
        fan_in *= receptive; fan_out *= receptive
    return fan_in, fan_out


def uniform_(tensor: Tensor, a=0.0, b=1.0) -> Tensor:
    n = _prod(tensor.shape)
    vals = [_random.uniform(a, b) for _ in range(n)]
    _set_flat(tensor._data, vals)
    return tensor


def normal_(tensor: Tensor, mean=0.0, std=1.0) -> Tensor:
    n = _prod(tensor.shape)
    vals = [mean + std * _randn_box_muller() for _ in range(n)]
    _set_flat(tensor._data, vals)
    return tensor


def constant_(tensor: Tensor, val: float) -> Tensor:
    n = _prod(tensor.shape)
    _set_flat(tensor._data, [float(val)] * n)
    return tensor


def zeros_(tensor: Tensor) -> Tensor:
    return constant_(tensor, 0.0)


def ones_(tensor: Tensor) -> Tensor:
    return constant_(tensor, 1.0)


def eye_(tensor: Tensor) -> Tensor:
    assert tensor.ndim == 2, "eye_ requires 2D tensor"
    rows, cols = tensor.shape
    vals = [1.0 if i == j else 0.0 for i in range(rows) for j in range(cols)]
    _set_flat(tensor._data, vals)
    return tensor


def trunc_normal_(tensor: Tensor, mean=0.0, std=1.0, a=-2.0, b=2.0) -> Tensor:
    n = _prod(tensor.shape)
    vals = []
    for _ in range(n):
        v = mean + std * _randn_box_muller()
        vals.append(max(a, min(b, v)))
    _set_flat(tensor._data, vals)
    return tensor


def xavier_uniform_(tensor: Tensor, gain=1.0) -> Tensor:
    fan_in, fan_out = _calculate_fan(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound)


def xavier_normal_(tensor: Tensor, gain=1.0) -> Tensor:
    fan_in, fan_out = _calculate_fan(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return normal_(tensor, 0.0, std)


def kaiming_uniform_(tensor: Tensor, a=0, mode='fan_in', nonlinearity='leaky_relu') -> Tensor:
    fan_in, fan_out = _calculate_fan(tensor)
    fan = fan_in if mode == 'fan_in' else fan_out
    gain = math.sqrt(2.0 / (1 + a ** 2))
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound)


def kaiming_normal_(tensor: Tensor, a=0, mode='fan_in', nonlinearity='leaky_relu') -> Tensor:
    fan_in, fan_out = _calculate_fan(tensor)
    fan = fan_in if mode == 'fan_in' else fan_out
    gain = math.sqrt(2.0 / (1 + a ** 2))
    std = gain / math.sqrt(fan)
    return normal_(tensor, 0.0, std)


def orthogonal_(tensor: Tensor, gain=1.0) -> Tensor:
    rows = tensor.shape[0]
    cols = _prod(tensor.shape[1:])
    # Generate random matrix and QR decompose via Gram-Schmidt
    flat = [[_randn_box_muller() for _ in range(cols)] for _ in range(rows)]
    if rows < cols:
        flat = [[flat[r][c] for r in range(rows)] for c in range(cols)]
        rows, cols = cols, rows
    # Gram-Schmidt orthogonalization
    q = []
    for v in flat:
        u = v[:]
        for b in q:
            proj = sum(u[i]*b[i] for i in range(len(u)))
            u = [u[i] - proj * b[i] for i in range(len(u))]
        norm = math.sqrt(sum(x**2 for x in u))
        if norm < 1e-10: q.append([0.0]*len(u)); continue
        q.append([x / norm for x in u])
    # Flatten and set
    if tensor.shape[0] < _prod(tensor.shape[1:]):
        q_use = [[q[c][r] for c in range(cols)] for r in range(rows)][:tensor.shape[0]]
    else:
        q_use = q[:tensor.shape[0]]
    vals = [gain * q_use[r][c] for r in range(len(q_use)) for c in range(len(q_use[0]))]
    _set_flat(tensor._data, vals)
    return tensor


def sparse_(tensor: Tensor, sparsity: float, std=0.01) -> Tensor:
    assert tensor.ndim == 2, "sparse_ requires 2D tensor"
    rows, cols = tensor.shape
    num_zeros = int(math.ceil(sparsity * rows))
    normal_(tensor, 0.0, std)
    flat = _get_flat(tensor._data)
    for col in range(cols):
        row_indices = _random.sample(range(rows), min(num_zeros, rows))
        for r in row_indices: flat[r * cols + col] = 0.0
    _set_flat(tensor._data, flat)
    return tensor


def dirac_(tensor: Tensor, groups=1) -> Tensor:
    assert tensor.ndim >= 3, "dirac_ requires >= 3D tensor"
    zeros_(tensor)
    sizes = tensor.shape; min_dim = min(sizes[0] // groups, sizes[1])
    flat = _get_flat(tensor._data)
    for g in range(groups):
        for d in range(min_dim):
            center = tuple(s // 2 for s in sizes[2:])
            # Compute flat index
            idx = (g * (sizes[0] // groups) + d) * _prod(sizes[1:]) + d * _prod(sizes[2:])
            for ci in center: idx = idx * sizes[2 + center.index(ci)] + ci if False else idx
            # Simple center index
            spatial = _prod(sizes[2:])
            center_idx = spatial // 2 if spatial > 0 else 0
            flat_i = (g * (sizes[0]//groups) + d) * _prod(sizes[1:]) + d * spatial + center_idx
            if flat_i < len(flat): flat[flat_i] = 1.0
    _set_flat(tensor._data, flat)
    return tensor


def calculate_gain(nonlinearity: str, param=None) -> float:
    table = {
        'linear': 1.0, 'conv1d': 1.0, 'conv2d': 1.0, 'conv3d': 1.0,
        'sigmoid': 1.0, 'tanh': 5.0/3.0, 'relu': math.sqrt(2.0),
        'selu': 3.0/4.0, 'leaky_relu': math.sqrt(2.0 / (1 + (param or 0.01)**2)),
    }
    return table.get(nonlinearity, 1.0)


def lecun_normal_(tensor: Tensor) -> Tensor:
    """LeCun normal init: N(0, 1/fan_in) — optimal for SELU networks."""
    fan_in, _ = _calculate_fan(tensor)
    std = math.sqrt(1.0 / fan_in) if fan_in > 0 else 1.0
    return normal_(tensor, 0.0, std)


def lecun_uniform_(tensor: Tensor) -> Tensor:
    """LeCun uniform init: U(-sqrt(3/fan_in), sqrt(3/fan_in))."""
    fan_in, _ = _calculate_fan(tensor)
    limit = math.sqrt(3.0 / fan_in) if fan_in > 0 else 1.0
    return uniform_(tensor, -limit, limit)


def variance_scaling_(tensor: Tensor, scale=1.0, mode='fan_in',
                      distribution='truncated_normal') -> Tensor:
    """Generalised variance-scaling init (Keras-compatible).

    Args:
        scale: scaling factor (default 1.0)
        mode: 'fan_in', 'fan_out', or 'fan_avg'
        distribution: 'truncated_normal', 'normal', or 'uniform'
    """
    fan_in, fan_out = _calculate_fan(tensor)
    if mode == 'fan_in':
        n = fan_in
    elif mode == 'fan_out':
        n = fan_out
    else:
        n = (fan_in + fan_out) / 2.0
    std = math.sqrt(scale / max(n, 1e-8))
    if distribution == 'uniform':
        limit = math.sqrt(3.0) * std
        return uniform_(tensor, -limit, limit)
    return trunc_normal_(tensor, 0.0, std)


def glorot_uniform_(tensor: Tensor) -> Tensor:
    """Alias for xavier_uniform_ (Glorot & Bengio 2010)."""
    return xavier_uniform_(tensor)


def glorot_normal_(tensor: Tensor) -> Tensor:
    """Alias for xavier_normal_ (Glorot & Bengio 2010)."""
    return xavier_normal_(tensor)


def he_uniform_(tensor: Tensor, a: float = 0.0) -> Tensor:
    """Alias for kaiming_uniform_ (He et al. 2015)."""
    return kaiming_uniform_(tensor, a=a, nonlinearity='relu')


def he_normal_(tensor: Tensor, a: float = 0.0) -> Tensor:
    """Alias for kaiming_normal_ (He et al. 2015)."""
    return kaiming_normal_(tensor, a=a, nonlinearity='relu')
