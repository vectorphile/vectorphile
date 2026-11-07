"""nn utility functions — gradient clipping, model serialization, etc."""
from __future__ import annotations
import numpy as np
from typing import Iterable, Union
from vectorphile.core.tensor import Tensor
from vectorphile.nn.module import Module


def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0) -> float:
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0
    if norm_type == float('inf'):
        total_norm = max(p.grad._data.max() for p in params)
    else:
        total_norm = sum(np.linalg.norm(p.grad._data.flatten(), ord=norm_type) ** norm_type for p in params) ** (1.0 / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in params:
            p.grad._data *= clip_coef
    return float(total_norm)


def clip_grad_value_(parameters, clip_value: float) -> None:
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    for p in parameters:
        if p.grad is not None:
            p.grad._data = np.clip(p.grad._data, -clip_value, clip_value)


def parameters_to_vector(parameters) -> Tensor:
    parts = [p._data.flatten() for p in parameters]
    return Tensor(np.concatenate(parts))


def vector_to_parameters(vec: Tensor, parameters) -> None:
    pointer = 0
    for p in parameters:
        num = p._data.size
        p._data[:] = vec._data[pointer:pointer+num].reshape(p._data.shape)
        pointer += num


def model_summary(model: Module) -> str:
    lines = [f"{'Layer':<30} {'Output Shape':<20} {'Params':>10}"]
    lines.append('-' * 62)
    total_params = 0
    for name, param in model._parameters.items():
        n = param._data.size
        total_params += n
        lines.append(f"{name:<30} {str(param.shape):<20} {n:>10,}")
    lines.append('=' * 62)
    lines.append(f"{'Total Parameters':<30} {'':20} {total_params:>10,}")
    return '\n'.join(lines)


def weight_norm(module: Module, name='weight', dim=0):
    weight = getattr(module, name)
    norms = np.linalg.norm(weight._data.reshape(weight.shape[0], -1), axis=1, keepdims=True)
    norms = norms.reshape(-1, *[1]*(weight.ndim-1))
    weight._data = weight._data / (norms + 1e-12)
    return module


def remove_weight_norm(module: Module, name='weight'):
    return module


def spectral_norm(module: Module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    weight = getattr(module, name)
    W = weight._data.reshape(weight.shape[0], -1)
    u = np.random.randn(W.shape[0])
    u /= np.linalg.norm(u)
    for _ in range(n_power_iterations):
        v = W.T @ u
        v /= np.linalg.norm(v) + eps
        u = W @ v
        u /= np.linalg.norm(u) + eps
    sigma = u @ W @ v
    weight._data = weight._data / (sigma + eps)
    return module


def get_total_params(model: Module, trainable_only=False) -> int:
    total = 0
    for p in model._parameters.values():
        if not trainable_only or p.requires_grad:
            total += p._data.size
    return total


def freeze_params(model: Module) -> None:
    for p in model._parameters.values():
        p.requires_grad = False


def unfreeze_params(model: Module) -> None:
    for p in model._parameters.values():
        p.requires_grad = True
