"""Random number generation."""
import numpy as np
from vectorphile.core.tensor import Tensor


def seed(s: int):
    np.random.seed(s)

def rand(*shape, device='cpu') -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Tensor(np.random.rand(*shape).astype(np.float32), device=device)

def randn(*shape, device='cpu') -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Tensor(np.random.randn(*shape).astype(np.float32), device=device)

def randint(low, high, shape, device='cpu') -> Tensor:
    return Tensor(np.random.randint(low, high, shape).astype(np.int64), device=device)

def choice(a, size=None, replace=True, p=None) -> Tensor:
    if isinstance(a, Tensor):
        a = a._data
    return Tensor(np.random.choice(a, size=size, replace=replace, p=p))

def shuffle(x: Tensor) -> Tensor:
    data = x._data.copy()
    np.random.shuffle(data)
    return Tensor(data)

def normal(loc=0.0, scale=1.0, size=None, device='cpu') -> Tensor:
    return Tensor(np.random.normal(loc, scale, size).astype(np.float32), device=device)

def uniform(low=0.0, high=1.0, size=None, device='cpu') -> Tensor:
    return Tensor(np.random.uniform(low, high, size).astype(np.float32), device=device)

def binomial(n, p, size=None) -> Tensor:
    return Tensor(np.random.binomial(n, p, size).astype(np.float32))

def poisson(lam=1.0, size=None) -> Tensor:
    return Tensor(np.random.poisson(lam, size).astype(np.float32))

def exponential(scale=1.0, size=None) -> Tensor:
    return Tensor(np.random.exponential(scale, size).astype(np.float32))

def multinomial(input: Tensor, num_samples: int, replacement=False) -> Tensor:
    probs = input._data
    probs = probs / probs.sum(axis=-1, keepdims=True)
    if probs.ndim == 1:
        return Tensor(np.random.choice(len(probs), size=num_samples, replace=replacement, p=probs).astype(np.int64))
    result = np.stack([
        np.random.choice(probs.shape[-1], size=num_samples, replace=replacement, p=row)
        for row in probs.reshape(-1, probs.shape[-1])
    ])
    return Tensor(result.reshape(*probs.shape[:-1], num_samples).astype(np.int64))

def permutation(x) -> Tensor:
    if isinstance(x, int):
        return Tensor(np.random.permutation(x).astype(np.int64))
    return Tensor(np.random.permutation(x._data if isinstance(x, Tensor) else x))

def bernoulli(p: Tensor) -> Tensor:
    return Tensor((np.random.rand(*p._data.shape) < p._data).astype(np.float32))
