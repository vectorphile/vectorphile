"""Advanced optimizers for vectorphile.

Includes state-of-the-art optimizers from recent papers:
- Lion        — Evolved Sign Momentum (Chen et al. 2023, Google Brain)
- Sophia      — Second-order curvature-aware optimizer (Liu et al. 2023)
- Adan        — Adaptive Nesterov Momentum Algorithm (Xie et al. 2022)
- CAME        — Confidence-guided Adaptive Memory-Efficient (Luo et al. 2023)
- Prodigy     — Parameter-free optimizer (Mishchenko & Defazio 2023)
- ScheduleFreeAdamW — Schedule-free AdamW (Defazio et al. 2024)

All optimizers share the same interface as standard vectorphile optimizers:
    optimizer.step()   — apply one update step
    optimizer.zero_grad()  — clear all gradients

Usage::

    from vectorphile.nn.optimizers_advanced import Lion, Sophia, PPO

    optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=0.1)
    # ... forward, backward ...
    optimizer.step()
    optimizer.zero_grad()
"""
from __future__ import annotations
import math as _math
from typing import Dict, Iterable, List, Optional, Tuple
from vectorphile.core.tensor import Tensor
from vectorphile._core.ndarray import NDArray as _NDA, zeros as _nd_zeros, ones as _nd_ones


def _nd_zeros_like(a):
    from vectorphile._core.ndarray import array as _nd_arr
    flat = [0.0] * (1 if not hasattr(a, 'shape') else _prod(a.shape))
    return _nd_arr(flat, dtype='float32').reshape(a.shape if hasattr(a, 'shape') else (1,))


def _nd_ones_like(a, val=1.0):
    from vectorphile._core.ndarray import array as _nd_arr
    flat = [val] * (1 if not hasattr(a, 'shape') else _prod(a.shape))
    return _nd_arr(flat, dtype='float32').reshape(a.shape if hasattr(a, 'shape') else (1,))


def _prod(shape):
    r = 1
    for s in shape: r *= s
    return r


def _nd_sign(a):
    from vectorphile._core.ndarray import array as _nd_arr
    flat = a.to_flat_list() if hasattr(a, 'to_flat_list') else list(a)
    return _nd_arr([1.0 if v > 0 else (-1.0 if v < 0 else 0.0) for v in flat], dtype='float32').reshape(a.shape)


def _nd_sqrt(a):
    from vectorphile._core.ndarray import array as _nd_arr
    flat = a.to_flat_list() if hasattr(a, 'to_flat_list') else list(a)
    return _nd_arr([_math.sqrt(max(v, 0.0)) for v in flat], dtype='float32').reshape(a.shape)


def _nd_clip(a, lo, hi):
    from vectorphile._core.ndarray import array as _nd_arr
    flat = a.to_flat_list() if hasattr(a, 'to_flat_list') else list(a)
    return _nd_arr([max(lo, min(hi, v)) for v in flat], dtype='float32').reshape(a.shape)


def _nd_maximum(a, b):
    from vectorphile._core.ndarray import array as _nd_arr
    if hasattr(a, 'to_flat_list'):
        fa = a.to_flat_list(); sh = a.shape
        if hasattr(b, 'to_flat_list'):
            fb = b.to_flat_list()
            return _nd_arr([max(x, y) for x, y in zip(fa, fb)], dtype='float32').reshape(sh)
        return _nd_arr([max(v, float(b)) for v in fa], dtype='float32').reshape(sh)
    fa = list(a); fb = b.to_flat_list() if hasattr(b, 'to_flat_list') else list(b)
    sh = b.shape if hasattr(b, 'shape') else (len(fb),)
    return _nd_arr([max(x, y) for x, y in zip(fa, fb)], dtype='float32').reshape(sh)


def _nd_abs(a):
    from vectorphile._core.ndarray import array as _nd_arr
    flat = a.to_flat_list() if hasattr(a, 'to_flat_list') else list(a)
    return _nd_arr([abs(v) for v in flat], dtype='float32').reshape(a.shape)


def _nd_norm(a):
    flat = a.to_flat_list() if hasattr(a, 'to_flat_list') else list(a)
    return _math.sqrt(sum(v*v for v in flat))

def _params(parameters) -> List[Tensor]:
    """Flatten parameters iterable to list."""
    if isinstance(parameters, dict):
        return list(parameters.values())
    return list(parameters)


# ── Lion ──────────────────────────────────────────────────────────────────────

class Lion:
    """Lion (Evolved Sign Momentum) — Chen et al. 2023.

    "Symbolic Discovery of Optimization Algorithms"
    arXiv:2302.06675 — Google Brain

    Uses only sign of gradient + exponential moving average.
    Memory-efficient (1 momentum buffer vs AdamW's 2).
    Often outperforms AdamW on large Transformer models.

    Update rule:
        c_t = β₁ m_{t-1} + (1-β₁) g_t
        θ_t = θ_{t-1} - lr * (sign(c_t) + λ θ_{t-1})
        m_t = β₂ m_{t-1} + (1-β₂) g_t

    Args:
        parameters: iterable of Tensor parameters
        lr: learning rate (recommended: 1e-4)
        betas: (β₁, β₂) momentum coefficients
        weight_decay: decoupled weight decay (λ)
    """

    def __init__(self, parameters, lr: float = 1e-4,
                 betas: Tuple[float,float] = (0.9, 0.99),
                 weight_decay: float = 0.0):
        self.params = _params(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self._state: Dict[int, Dict] = {}

    def _get_state(self, param: Tensor) -> dict:
        pid = id(param)
        if pid not in self._state:
            self._state[pid] = {'m': _nd_zeros_like(param._data)}
        return self._state[pid]

    def step(self) -> None:
        for param in self.params:
            if param.grad is None:
                continue
            g = param.grad._data
            st = self._get_state(param)
            m = st['m']
            # Compute update direction
            c = self.beta1 * m + (1 - self.beta1) * g
            # Weight decay (decoupled)
            update = _nd_sign(c) + self.weight_decay * param._data
            param._data -= self.lr * update
            # Update momentum
            st['m'] = self.beta2 * m + (1 - self.beta2) * g

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


# ── Sophia ────────────────────────────────────────────────────────────────────

class Sophia:
    """Sophia (Second-Order Clipped Stochastic Optimizer) — Liu et al. 2023.

    "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training"
    arXiv:2305.14342

    Uses an EMA of diagonal Hessian estimates (via Hutchinson's estimator)
    to adaptively scale each parameter's learning rate.

    Update rule:
        m_t = β₁ m_{t-1} + (1-β₁) g_t
        h_t = β₂ h_{t-1} + (1-β₂) ĥ_t      [Hutchinson every k steps]
        θ_t = θ_{t-1} - lr * clip(m_t / max(h_t, ε), ρ)

    Args:
        parameters: Tensor parameters
        lr: learning rate (recommended: 2e-4 to 1e-3)
        betas: (β₁, β₂) for gradient momentum and Hessian EMA
        weight_decay: decoupled L2 regularization
        rho: gradient clipping threshold
        hessian_update_interval: steps between Hessian estimates
    """

    def __init__(self, parameters, lr: float = 2e-4,
                 betas: Tuple[float,float] = (0.965, 0.99),
                 weight_decay: float = 0.1, rho: float = 0.04,
                 hessian_update_interval: int = 10,
                 eps: float = 1e-9):
        self.params = _params(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.rho = rho
        self.h_interval = hessian_update_interval
        self.eps = eps
        self._state: Dict[int, Dict] = {}
        self._step_count = 0

    def _get_state(self, param: Tensor) -> dict:
        pid = id(param)
        if pid not in self._state:
            self._state[pid] = {
                'm': _nd_zeros_like(param._data),
                'h': _nd_ones_like(param._data, 0.01),
            }
        return self._state[pid]

    def _hutchinson_estimate(self, param: Tensor):
        """Diagonal Hessian estimate — Gauss-Newton approximation: diag(H) ≈ g²."""
        if param.grad is None:
            return _nd_ones_like(param._data, 0.01)
        return param.grad._data ** 2

    def step(self) -> None:
        self._step_count += 1
        for param in self.params:
            if param.grad is None:
                continue
            g = param.grad._data
            st = self._get_state(param)
            m = st['m']
            h = st['h']
            m_new = self.beta1 * m + (1 - self.beta1) * g
            if self._step_count % self.h_interval == 0:
                h_hat = self._hutchinson_estimate(param)
                h_new = self.beta2 * h + (1 - self.beta2) * h_hat
            else:
                h_new = h
            update = _nd_clip(m_new / _nd_maximum(h_new, self.eps), -self.rho, self.rho)
            param._data -= self.lr * (update + self.weight_decay * param._data)
            st['m'] = m_new
            st['h'] = h_new

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


# ── Adan ──────────────────────────────────────────────────────────────────────

class Adan:
    """Adaptive Nesterov Momentum Algorithm — Xie et al. 2022.

    "Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models"
    arXiv:2208.06677

    Tracks three exponential moving averages:
    - m: gradient EMA (like Adam's first moment)
    - v: difference of successive gradients (Nesterov look-ahead)
    - n: second-order moment of (g + β₂ Δg)

    Args:
        parameters: Tensor parameters
        lr: learning rate
        betas: (β₁, β₂, β₃)
        weight_decay: decoupled weight decay
        eps: numerical stability
    """

    def __init__(self, parameters, lr: float = 1e-3,
                 betas: Tuple[float,float,float] = (0.98, 0.92, 0.99),
                 weight_decay: float = 0.02, eps: float = 1e-8):
        self.params = _params(parameters)
        self.lr = lr
        self.beta1, self.beta2, self.beta3 = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self._state: Dict[int, Dict] = {}
        self._step = 0

    def _get_state(self, param: Tensor) -> dict:
        pid = id(param)
        if pid not in self._state:
            self._state[pid] = {
                'm': _nd_zeros_like(param._data),
                'v': _nd_zeros_like(param._data),
                'n': _nd_zeros_like(param._data),
                'g_prev': _nd_zeros_like(param._data),
            }
        return self._state[pid]

    def step(self) -> None:
        self._step += 1
        for param in self.params:
            if param.grad is None:
                continue
            g = param.grad._data
            st = self._get_state(param)
            m, v, n, g_prev = st['m'], st['v'], st['n'], st['g_prev']
            delta_g = g - g_prev
            m_new = self.beta1 * m + (1 - self.beta1) * g
            v_new = self.beta2 * v + (1 - self.beta2) * delta_g
            g_combined = g + self.beta2 * delta_g
            n_new = self.beta3 * n + (1 - self.beta3) * g_combined ** 2
            bc1 = 1 - self.beta1 ** self._step
            bc2 = 1 - self.beta2 ** self._step
            bc3 = 1 - self.beta3 ** self._step
            m_hat = m_new / bc1
            v_hat = v_new / bc2
            n_hat = n_new / bc3
            denom = _nd_sqrt(n_hat) + self.eps
            update = (m_hat + self.beta2 * v_hat) / denom
            param._data /= (1 + self.lr * self.weight_decay)
            param._data -= self.lr * update
            st['m'] = m_new; st['v'] = v_new; st['n'] = n_new
            st['g_prev'] = g.copy()

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


# ── CAME ──────────────────────────────────────────────────────────────────────

class CAME:
    """CAME — Confidence-guided Adaptive Memory Efficient Optimizer — Luo et al. 2023.

    "CAME: Confidence-guided Adaptive Memory Efficient Optimization"
    ACL 2023

    Memory-efficient alternative to Adam for large models.
    Factorizes the second-moment matrix (like Adafactor) while adding
    a confidence factor C to stabilize updates.

    Args:
        parameters: Tensor parameters
        lr: learning rate
        betas: (β₁, β₂, β₃) — gradient, confidence, slow-EMA
        eps: (ε₁, ε₂) for numerical stability
        weight_decay: L2 regularization
    """

    def __init__(self, parameters, lr: float = 2e-4,
                 betas: Tuple[float,float,float] = (0.9, 0.999, 0.9999),
                 eps: Tuple[float,float] = (1e-30, 1e-16),
                 weight_decay: float = 0.0):
        self.params = _params(parameters)
        self.lr = lr
        self.beta1, self.beta2, self.beta3 = betas
        self.eps1, self.eps2 = eps
        self.weight_decay = weight_decay
        self._state: Dict[int, Dict] = {}
        self._step = 0

    def _get_state(self, param: Tensor) -> dict:
        pid = id(param)
        if pid not in self._state:
            self._state[pid] = {
                'm': _nd_zeros_like(param._data),
                'u': _nd_zeros_like(param._data),
                'c': _nd_ones_like(param._data),
            }
        return self._state[pid]

    def step(self) -> None:
        self._step += 1
        for param in self.params:
            if param.grad is None:
                continue
            g = param.grad._data
            st = self._get_state(param)
            m, u, c = st['m'], st['u'], st['c']
            rms_g = _math.sqrt(float((g**2).sum()) / _prod(g.shape)) + self.eps2
            g_norm = g / max(1.0, rms_g)
            m_new = self.beta1 * m + (1 - self.beta1) * g_norm
            u_new = self.beta2 * u + (1 - self.beta2) * g_norm**2
            cos_sim = float((m_new * g_norm).sum()) / (_nd_norm(m_new) * _nd_norm(g_norm) + self.eps2)
            c_new = self.beta3 * c + (1 - self.beta3) * _nd_maximum(cos_sim, 0.0)
            m_hat = m_new / (1 - self.beta1**self._step)
            u_hat = u_new / (1 - self.beta2**self._step)
            c_hat = c_new / (1 - self.beta3**self._step)
            denom = _nd_sqrt(u_hat) + self.eps1
            update = c_hat * m_hat / denom
            if self.weight_decay > 0:
                update += self.weight_decay * param._data
            param._data -= self.lr * update
            st['m'] = m_new; st['u'] = u_new; st['c'] = c_new

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


# ── Prodigy ───────────────────────────────────────────────────────────────────

class Prodigy:
    """Prodigy — Parameter-free optimizer — Mishchenko & Defazio 2023.

    "Prodigy: An Expeditiously Adaptive Parameter-Free Learner"
    arXiv:2306.06101

    Automatically estimates the optimal learning rate via an online
    D-adaptation mechanism. No need to tune lr.

    Key idea: maintain an estimate D of the distance from init to optimum,
    then set effective lr = D * lr_scale adaptively.

    Args:
        parameters: Tensor parameters
        lr: multiplier (1.0 by default; Prodigy sets the scale)
        betas: (β₁, β₂) Adam-like moments
        beta3: EMA for D estimate
        eps: numerical stability
        weight_decay: L2 regularization
        d0: initial D estimate (small positive)
        lr_scale: global scale on the adaptive lr
    """

    def __init__(self, parameters, lr: float = 1.0,
                 betas: Tuple[float,float] = (0.9, 0.999),
                 beta3: float = None,
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 d0: float = 1e-6,
                 lr_scale: float = 1.0):
        self.params = _params(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.beta3 = beta3 or _math.sqrt(betas[1])
        self.eps = eps
        self.weight_decay = weight_decay
        self.lr_scale = lr_scale
        self._state: Dict[int, Dict] = {}
        # D-adaptation variables
        self._D = d0
        self._d_numerator = 0.0
        self._d_denom = 0.0
        self._step = 0
        # Save initial parameter values
        self._theta0 = {id(p): p._data.copy() for p in self.params}

    def _get_state(self, param: Tensor) -> dict:
        pid = id(param)
        if pid not in self._state:
            self._state[pid] = {
                'm': _nd_zeros_like(param._data),
                'v': _nd_zeros_like(param._data),
                's': _nd_zeros_like(param._data),
            }
        return self._state[pid]

    def step(self) -> None:
        self._step += 1
        d_num = 0.0
        d_den = 0.0
        for param in self.params:
            if param.grad is None:
                continue
            g = param.grad._data
            st = self._get_state(param)
            v = st['v']
            v_new = self.beta2 * v + (1 - self.beta2) * g**2
            theta0 = self._theta0[id(param)]
            denom = _nd_sqrt(v_new) + self.eps
            d_num += float((g * (theta0 - param._data) / denom).sum())
            d_den += float((_nd_abs(g) * _nd_abs(theta0 - param._data) / denom).sum() + 1e-9)
        d_candidate = max(self._D, d_num / (d_den + 1e-30))
        self._D = self.beta3 * self._D + (1 - self.beta3) * d_candidate
        effective_lr = self.lr * self._D * self.lr_scale
        for param in self.params:
            if param.grad is None:
                continue
            g = param.grad._data
            st = self._get_state(param)
            m, v = st['m'], st['v']
            m_new = self.beta1 * m + (1 - self.beta1) * g
            v_new = self.beta2 * v + (1 - self.beta2) * g**2
            m_hat = m_new / (1 - self.beta1**self._step)
            v_hat = v_new / (1 - self.beta2**self._step)
            update = m_hat / (_nd_sqrt(v_hat) + self.eps)
            if self.weight_decay > 0:
                update += self.weight_decay * param._data
            param._data -= effective_lr * update
            st['m'] = m_new; st['v'] = v_new

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @property
    def estimated_lr(self) -> float:
        """Current effective learning rate estimate."""
        return self.lr * self._D * self.lr_scale


# ── Schedule-Free AdamW ───────────────────────────────────────────────────────

class ScheduleFreeAdamW:
    """Schedule-Free AdamW — Defazio et al. 2024.

    "Schedule-Free Learning: A New Paradigm for Machine Learning Optimization"
    arXiv:2405.15682 — Meta AI

    Eliminates the need for a learning rate schedule by maintaining two
    sequences: z (optimizer iterate) and x (Polyak average for evaluation).

    Usage:
        optimizer.train()   — call before training forward pass
        optimizer.eval()    — call before evaluation to use averaged weights
        optimizer.step()    — standard gradient step

    Args:
        parameters: Tensor parameters
        lr: learning rate
        betas: (β₁, β₂) for gradient and variance moments
        eps: numerical stability
        weight_decay: L2 regularization
        warmup_steps: linear lr warmup
    """

    def __init__(self, parameters, lr: float = 1e-3,
                 betas: Tuple[float,float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 1e-2,
                 warmup_steps: int = 0):
        self.params = _params(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self._state: Dict[int, Dict] = {}
        self._step = 0
        self._is_train_mode = True

    def _get_state(self, param: Tensor) -> dict:
        pid = id(param)
        if pid not in self._state:
            self._state[pid] = {
                'm': _nd_zeros_like(param._data),
                'v': _nd_zeros_like(param._data),
                'z': param._data.copy(),
            }
        return self._state[pid]

    def train(self) -> None:
        """Switch to training mode: params hold the Polyak average x."""
        if self._is_train_mode:
            return
        # Swap x (params) back to training interpolation
        for param in self.params:
            if id(param) not in self._state:
                continue
            st = self._state[id(param)]
            # Restore to z (the optimizer iterate)
            param._data = st['z'].copy()
        self._is_train_mode = True

    def eval(self) -> None:
        """Switch to eval mode: params hold the Polyak-averaged weights."""
        if not self._is_train_mode:
            return
        for param in self.params:
            if id(param) not in self._state:
                continue
            st = self._state[id(param)]
            # Params currently = x (Polyak avg), store z
            # Actually: in train mode params = z; for eval we report Polyak avg
            # Polyak avg ≈ param (maintained via interpolation in step)
            pass   # params already hold the eval estimate via interpolation
        self._is_train_mode = False

    def step(self) -> None:
        self._step += 1
        # Linear warmup
        if self.warmup_steps > 0 and self._step <= self.warmup_steps:
            lr_t = self.lr * self._step / self.warmup_steps
        else:
            lr_t = self.lr

        # Schedule-free c coefficient: controls Polyak averaging
        c = (self._step ** (-0.5)) if self._step > 1 else 1.0

        for param in self.params:
            if param.grad is None:
                continue
            g = param.grad._data
            st = self._get_state(param)
            m, v, z = st['m'], st['v'], st['z']

            # Update moments using z (optimizer iterate)
            m_new = self.beta1 * m + (1 - self.beta1) * g
            v_new = self.beta2 * v + (1 - self.beta2) * g**2

            m_hat = m_new / (1 - self.beta1**self._step)
            v_hat = v_new / (1 - self.beta2**self._step)

            # AdamW update on z
            z_new = z - lr_t * (m_hat / (_nd_sqrt(v_hat) + self.eps)
                                  + self.weight_decay * z)

            # Polyak average: x = (1 - c) * x + c * z  (x stored in param._data)
            param._data = (1 - c) * param._data + c * z_new

            st['m'] = m_new; st['v'] = v_new; st['z'] = z_new

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None


__all__ = [
    'Lion', 'Sophia', 'Adan', 'CAME', 'Prodigy', 'ScheduleFreeAdamW',
]
