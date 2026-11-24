"""
Muon — Momentum + Orthogonalization Using Newton's method.

Developed by Keller Jordan (2024). Empirically the fastest-converging optimizer
for training neural networks, beating AdamW on GPT-class models.
"""
from __future__ import annotations
import math as _math
from vectorphile.optim.optimizer import Optimizer
from vectorphile._vp_helpers import *


def _ns_step(X):
    """One Newton-Schulz iteration: X ← 1.5·X − 0.5·X·Xᵀ·X"""
    _is_gpu = type(X).__name__ == 'GArray'
    if _is_gpu:
        t = X.t
        return type(X)(1.5 * t - 0.5 * t @ (t.T @ t))
    # Pure NDArray path
    return X * 1.5 - (X @ (X.T @ X)) * 0.5


def _nd_norm_frob(X):
    """Frobenius norm of an NDArray."""
    flat = X.to_flat_list() if hasattr(X, 'to_flat_list') else list(X._buf)
    return _math.sqrt(sum(v * v for v in flat))


def _orthogonalize(G, n_steps: int = 5):
    """Map G to its orthogonal factor via Newton-Schulz iteration."""
    _is_gpu = type(G).__name__ == 'GArray'
    if _is_gpu:
        t = G.t.float()
        norm = t.norm()
        if norm < 1e-30:
            return G
        X = t / norm
        for _ in range(n_steps):
            A = X @ X.T
            X = (1.5 * X) - (0.5 * A @ X)
        return type(G)(X * _math.sqrt(max(G.t.shape[-2], G.t.shape[-1])))

    # Pure NDArray path
    try:
        norm = _nd_norm_frob(G)
        if norm < 1e-30:
            return G
        X = G * (1.0 / norm)
        for _ in range(n_steps):
            A = X @ X.T
            X = X * 1.5 - (A @ X) * 0.5
        scale = _math.sqrt(max(G.shape[-2], G.shape[-1]))
        return X * scale
    except Exception:
        return G


class Muon(Optimizer):
    """Muon — fastest-converging optimizer for neural networks."""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 ns_steps: int = 5, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.state = {}

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta = group['momentum']
            ns  = group['ns_steps']
            wd  = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad._data
                pid = id(p)
                if pid not in self.state:
                    self.state[pid] = {'buf': g * 0.0}
                s = self.state[pid]
                if wd != 0.0:
                    p._data = p._data * (1.0 - lr * wd)
                s['buf'] = beta * s['buf'] + g
                g_ns = g + beta * s['buf']
                if g_ns.ndim == 2:
                    update = _orthogonalize(g_ns, n_steps=ns)
                else:
                    update = g_ns
                p._data = p._data - lr * update


class ScheduleFreeAdam(Optimizer):
    """Schedule-Free Adam — optimal without any learning rate schedule."""

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0,
                 warmup_steps: int = 0, r: float = 0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        warmup_steps=warmup_steps, r=r)
        super().__init__(params, defaults)
        self.state = {}

    def _lr_scale(self, step: int, warmup: int) -> float:
        if warmup <= 0 or step >= warmup:
            return 1.0
        return step / warmup

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            wd  = group['weight_decay']
            warmup = group['warmup_steps']
            r  = group['r']
            for p in group['params']:
                if p.grad is None:
                    continue
                pid = id(p)
                if pid not in self.state:
                    self.state[pid] = {
                        'step': 0,
                        'x': p._data * 1.0,
                        'z': p._data * 1.0,
                        'm': p._data * 0.0,
                        'v': p._data * 0.0,
                    }
                s = self.state[pid]
                s['step'] += 1
                t = s['step']
                alpha = lr * self._lr_scale(t, warmup)
                g = p.grad._data
                s['m'] = b1 * s['m'] + (1.0 - b1) * g
                s['v'] = b2 * s['v'] + (1.0 - b2) * g ** 2
                m_hat = s['m'] / (1.0 - b1 ** t)
                v_hat = s['v'] / (1.0 - b2 ** t)
                # NDArray sqrt + eps
                from vectorphile._core.ndarray import sqrt as _nd_sqrt
                step_dir = m_hat / (_nd_sqrt(v_hat) + eps)
                if wd != 0.0:
                    s['x'] = s['x'] * (1.0 - alpha * wd)
                s['x'] = s['x'] - alpha * step_dir
                c = 1.0 / (1.0 + r * t) if r > 0 else 1.0 / t
                s['z'] = (1.0 - c) * s['z'] + c * s['x']
                p._data = (1.0 - b1) * s['z'] + b1 * s['x']

    def eval(self):
        for group in self.param_groups:
            for p in group['params']:
                pid = id(p)
                if pid in self.state:
                    p._data = self.state[pid]['z']

    def train(self):
        for group in self.param_groups:
            for p in group['params']:
                pid = id(p)
                if pid in self.state:
                    b1 = group['betas'][0]
                    s = self.state[pid]
                    p._data = (1.0 - b1) * s['z'] + b1 * s['x']
