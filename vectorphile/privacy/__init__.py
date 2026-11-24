"""vectorphile.privacy — Differential Privacy and Secure ML — pure Python."""
from __future__ import annotations
import math
import random as _rand
from typing import Any, Callable, Dict, List, Optional, Tuple


def _flat(x) -> List[float]:
    if hasattr(x, 'to_flat_list'):
        return [float(v) for v in x.to_flat_list()]
    if hasattr(x, 'tolist'):
        v = x.tolist()
        if isinstance(v, list) and v and isinstance(v[0], list):
            return [float(item) for row in v for item in row]
        return [float(item) for item in v]
    if isinstance(x, list):
        return [float(item) for item in x]
    return [float(x)]

def _norm(v: List[float]) -> float:
    return math.sqrt(sum(x*x for x in v))

def _box_muller(mu: float = 0.0, sigma: float = 1.0) -> float:
    u1 = max(_rand.random(), 1e-15)
    u2 = _rand.random()
    return mu + sigma * math.sqrt(-2*math.log(u1)) * math.cos(2*math.pi*u2)

def _laplace_sample(mu: float = 0.0, b: float = 1.0) -> float:
    u = _rand.random() - 0.5
    return mu - b * (1 if u >= 0 else -1) * math.log(1 - 2*abs(u))


# ---------------------------------------------------------------------------
# Basic mechanisms
# ---------------------------------------------------------------------------

class GaussianMechanism:
    """Add calibrated Gaussian noise for (ε, δ)-DP."""

    def __init__(self, epsilon: float, delta: float, sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.sigma = self._calibrate_sigma()

    def _calibrate_sigma(self) -> float:
        return (math.sqrt(2 * math.log(1.25 / self.delta)) *
                self.sensitivity / self.epsilon)

    def __call__(self, x) -> List[float]:
        arr = _flat(x)
        return [v + _box_muller(0, self.sigma) for v in arr]

    def privatize(self, x) -> List[float]:
        return self(x)


class LaplaceMechanism:
    """Add calibrated Laplace noise for ε-DP."""

    def __init__(self, epsilon: float, sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.scale = sensitivity / epsilon

    def __call__(self, x) -> List[float]:
        arr = _flat(x)
        return [v + _laplace_sample(0, self.scale) for v in arr]


class ExponentialMechanism:
    """Exponential mechanism for categorical outputs."""

    def __init__(self, epsilon: float, sensitivity: float = 1.0):
        self.epsilon = epsilon
        self.sensitivity = sensitivity

    def __call__(self, scores) -> int:
        sc = _flat(scores)
        weights = [math.exp(self.epsilon * s / (2 * self.sensitivity)) for s in sc]
        total = sum(weights)
        probs = [w/total for w in weights]
        r = _rand.random(); cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                return i
        return len(probs) - 1


# ---------------------------------------------------------------------------
# Privacy accounting
# ---------------------------------------------------------------------------

class PrivacyAccountant:
    """Track privacy budget across training steps (RDP-based)."""

    def __init__(self, noise_multiplier: float = 1.0,
                 sample_rate: float = 0.01,
                 delta: float = 1e-5):
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.delta = delta
        self._steps = 0
        self._orders = list(range(2, 64)) + [128, 256]

    def step(self, n_steps: int = 1) -> None:
        self._steps += n_steps

    def _rdp_to_dp(self, rdp: float, alpha: float, delta: float) -> float:
        if rdp == float('inf'):
            return float('inf')
        return rdp + math.log((alpha - 1) / alpha) - (math.log(delta) + math.log(alpha)) / (alpha - 1)

    def get_privacy_spent(self, delta: Optional[float] = None) -> Tuple[float, float]:
        delta = delta or self.delta
        q = self.sample_rate; sigma = self.noise_multiplier; T = self._steps
        eps_list = []
        for alpha in self._orders:
            rdp = T * q ** 2 * alpha / (2 * sigma ** 2)
            eps = self._rdp_to_dp(rdp, alpha, delta)
            eps_list.append(eps)
        return float(min(eps_list)), delta

    @property
    def steps(self) -> int:
        return self._steps


# ---------------------------------------------------------------------------
# DP-SGD Optimizer
# ---------------------------------------------------------------------------

class DPSGDOptimizer:
    """Differentially Private SGD (Abadi et al., 2016)."""

    def __init__(self, params, lr: float = 0.01,
                 noise_multiplier: float = 1.1,
                 max_grad_norm: float = 1.0,
                 sample_rate: float = 0.01,
                 accountant: Optional[PrivacyAccountant] = None):
        self.params = list(params)
        self.lr = lr
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.accountant = accountant or PrivacyAccountant(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        from vectorphile._core.ndarray import array as _nd
        for p in self.params:
            if p.grad is None:
                continue
            g = _flat(p.grad._data if hasattr(p.grad, '_data') else p.grad)
            norm = _norm(g)
            if norm > self.max_grad_norm:
                g = [v * self.max_grad_norm / norm for v in g]
            sigma = self.noise_multiplier * self.max_grad_norm
            noisy_g = [v + _box_muller(0, sigma) for v in g]
            from vectorphile.core.tensor import Tensor
            nd = _nd(noisy_g)
            if hasattr(nd, 'reshape') and hasattr(p._data, 'shape'):
                nd = nd.reshape(p._data.shape)
            p.grad._data = nd
            # Update: p._data -= lr * noisy_g
            pd = _flat(p._data)
            new_pd = [pd[i] - self.lr * noisy_g[i] for i in range(len(pd))]
            new_nd = _nd(new_pd)
            if hasattr(new_nd, 'reshape') and hasattr(p._data, 'shape'):
                new_nd = new_nd.reshape(p._data.shape)
            p._data = new_nd
        self.accountant.step()


# ---------------------------------------------------------------------------
# Local Differential Privacy
# ---------------------------------------------------------------------------

class RandomizedResponse:
    """Local DP randomized response for binary data."""

    def __init__(self, epsilon: float):
        self.epsilon = epsilon
        self.p = math.exp(epsilon) / (math.exp(epsilon) + 1)

    def __call__(self, value: bool) -> bool:
        return bool(value) if _rand.random() < self.p else not bool(value)

    def privatize_array(self, values) -> List[bool]:
        return [self(bool(v)) for v in values]


class RAPPOR:
    """Simplified RAPPOR."""

    def __init__(self, epsilon: float, k: int = 256):
        self.epsilon = epsilon
        self.k = k
        self.p = 0.5
        self.q = 1 / (math.exp(epsilon / 2) + 1)
        self.f = 0.5

    def encode(self, value: int) -> List[int]:
        bloom = [False] * self.k
        bloom[value % self.k] = True
        prr = [(_rand.random() < (1 - self.p / 2) if b else _rand.random() < self.p / 2) for b in bloom]
        irr = [(_rand.random() < self.q if b else _rand.random() < (1 - self.q)) for b in prr]
        return [int(v) for v in irr]


__all__ = [
    'GaussianMechanism', 'LaplaceMechanism', 'ExponentialMechanism',
    'PrivacyAccountant', 'DPSGDOptimizer',
    'RandomizedResponse', 'RAPPOR',
]
