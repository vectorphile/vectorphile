"""Gradient utility functions."""
from __future__ import annotations
import math


def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0) -> float:
    """Clip gradient norms of parameters in-place. Returns total norm before clipping."""
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if norm_type == math.inf:
        total_norm = max(
            abs(g) for p in params
            for g in p.grad._data._buf
        )
    else:
        total_norm = 0.0
        for p in params:
            grad_data = p.grad._data
            for v in grad_data._buf:
                total_norm += abs(v) ** norm_type
        total_norm = total_norm ** (1.0 / norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            grad_buf = p.grad._data._buf
            for i in range(len(grad_buf)):
                grad_buf[i] *= clip_coef

    return float(total_norm)


def clip_grad_value_(parameters, clip_value: float) -> None:
    """Clip gradient values of parameters in-place to [-clip_value, clip_value]."""
    clip_value = float(clip_value)
    for p in parameters:
        if p.grad is None:
            continue
        grad_buf = p.grad._data._buf
        for i in range(len(grad_buf)):
            v = grad_buf[i]
            if v > clip_value:
                grad_buf[i] = clip_value
            elif v < -clip_value:
                grad_buf[i] = -clip_value
