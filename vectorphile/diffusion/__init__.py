"""Diffusion model schedulers — DDPM, DDIM, Flow Matching.

Accepts and returns the same array type as input (numpy, NDArray, or list).
Scheduler state (betas, alphas) stored as Python lists.
"""
import math
import random
from typing import List, Union


def _cumprod(vals: List[float]) -> List[float]:
    result = []; acc = 1.0
    for v in vals:
        acc *= v
        result.append(acc)
    return result


def _linspace(start: float, stop: float, n: int) -> List[float]:
    if n == 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _randn_like_flat(n: int) -> List[float]:
    result = []
    while len(result) < n:
        u1 = random.random() + 1e-15; u2 = random.random()
        result.extend([math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2),
                       math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)])
    return result[:n]


def _is_numpy(x):
    t = type(x); return t.__module__ == 'numpy' or (hasattr(x, 'flatten') and hasattr(x, 'dtype') and not hasattr(x, 'to_flat_list'))


def _apply_elementwise_numpy(x, noise, abar_arr):
    """DDPM add_noise for numpy input with possibly batched t."""
    import numpy as _np
    abar = _np.asarray(abar_arr).ravel().astype('float32')
    # Reshape abar to broadcast: (B,) → (B, 1, 1, ...) matching x.shape
    view = abar
    for _ in range(x.ndim - 1):
        view = view[:, _np.newaxis] if view.ndim < x.ndim else view
    return (_np.sqrt(view) * x + _np.sqrt(1.0 - view) * noise).astype(_np.float32)


class DDPMScheduler:
    """DDPM (Ho et al. 2020) noise scheduler."""

    def __init__(self, num_train_timesteps: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02,
                 beta_schedule: str = 'linear'):
        self.T = num_train_timesteps
        if beta_schedule == 'cosine':
            steps = num_train_timesteps + 1
            x = _linspace(0, num_train_timesteps, steps)
            acp_raw = [math.cos(((xi / num_train_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2 for xi in x]
            acp = [v / acp_raw[0] for v in acp_raw]
            betas = [_clip(1.0 - acp[i + 1] / acp[i], 0.0, 0.999) for i in range(num_train_timesteps)]
        else:
            betas = _linspace(beta_start, beta_end, num_train_timesteps)
        self.betas = betas
        self.alphas = [1.0 - b for b in betas]
        self.alphas_cumprod = _cumprod(self.alphas)
        self._timesteps = list(range(num_train_timesteps - 1, -1, -1))

    def set_timesteps(self, num_inference_steps: int):
        step = max(1, self.T // num_inference_steps)
        self._timesteps = list(range(self.T - 1, -1, -step))[:num_inference_steps]

    def add_noise(self, x0, noise, t):
        if _is_numpy(x0):
            import numpy as _np
            t_arr = _np.asarray(t).ravel()
            abar = _np.array([self.alphas_cumprod[int(ti)] for ti in t_arr], dtype='float32')
            # Build broadcast shape
            view = abar
            for _ in range(x0.ndim - 1):
                view = view[..., _np.newaxis]
            return (_np.sqrt(view) * x0 + _np.sqrt(1.0 - view) * noise).astype(_np.float32)
        # Pure Python list path
        t_int = int(t) if not hasattr(t, '__iter__') else int(list(t)[0])
        abar = self.alphas_cumprod[t_int]
        sq_a = math.sqrt(max(0.0, abar)); sq_1a = math.sqrt(max(0.0, 1.0 - abar))
        x_flat = list(x0); n_flat = list(noise)
        return [sq_a * a + sq_1a * n for a, n in zip(x_flat, n_flat)]

    def step(self, model_output, timestep: int, sample):
        t = int(timestep)
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_prev = self.alphas_cumprod[t - 1] if t > 0 else 1.0
        beta_t = self.betas[t]
        if _is_numpy(sample):
            import numpy as _np
            sq_1at = math.sqrt(max(0.0, 1.0 - alpha_bar_t))
            sq_at = math.sqrt(max(1e-15, alpha_bar_t))
            pred_x0 = _np.clip((sample - sq_1at * model_output) / sq_at, -1.0, 1.0)
            c1 = math.sqrt(max(0.0, alpha_bar_prev)) * beta_t / max(1e-15, 1.0 - alpha_bar_t)
            c2 = math.sqrt(max(0.0, alpha_t)) * (1.0 - alpha_bar_prev) / max(1e-15, 1.0 - alpha_bar_t)
            mean = c1 * pred_x0 + c2 * sample
            if t > 0:
                variance = beta_t * (1.0 - alpha_bar_prev) / max(1e-15, 1.0 - alpha_bar_t)
                noise = _np.random.randn(*sample.shape).astype('float32')
                return (mean + math.sqrt(max(0.0, variance)) * noise).astype('float32')
            return mean.astype('float32')
        # List path
        sq_1at = math.sqrt(max(0.0, 1.0 - alpha_bar_t)); sq_at = math.sqrt(max(1e-15, alpha_bar_t))
        pred_x0 = [_clip((s - sq_1at * m) / sq_at, -1.0, 1.0) for s, m in zip(sample, model_output)]
        c1 = math.sqrt(max(0.0, alpha_bar_prev)) * beta_t / max(1e-15, 1.0 - alpha_bar_t)
        c2 = math.sqrt(max(0.0, alpha_t)) * (1.0 - alpha_bar_prev) / max(1e-15, 1.0 - alpha_bar_t)
        mean = [c1 * p + c2 * s for p, s in zip(pred_x0, sample)]
        if t > 0:
            variance = beta_t * (1.0 - alpha_bar_prev) / max(1e-15, 1.0 - alpha_bar_t)
            noise = _randn_like_flat(len(sample))
            return [m + math.sqrt(max(0.0, variance)) * n for m, n in zip(mean, noise)]
        return mean


class DDIMScheduler:
    """DDIM (Song et al. 2020) deterministic sampler."""

    def __init__(self, num_train_timesteps: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02,
                 eta: float = 0.0):
        self.T = num_train_timesteps
        self.eta = eta
        betas = _linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas_cumprod = _cumprod([1.0 - b for b in betas])
        self._timesteps = list(range(num_train_timesteps - 1, -1, -1))

    def set_timesteps(self, num_inference_steps: int):
        step = max(1, self.T // num_inference_steps)
        self._timesteps = list(range(self.T - 1, -1, -step))[:num_inference_steps]

    def step(self, model_output, timestep: int, sample):
        t = int(timestep)
        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_prev = self.alphas_cumprod[t - 1] if t > 0 else 1.0
        sigma = self.eta * math.sqrt(max(0.0, (1.0 - alpha_bar_prev) / max(1e-15, 1.0 - alpha_bar_t) *
                                         max(0.0, 1.0 - alpha_bar_t / max(1e-15, alpha_bar_prev))))
        sq_1at = math.sqrt(max(0.0, 1.0 - alpha_bar_t)); sq_at = math.sqrt(max(1e-15, alpha_bar_t))
        dir_coef = math.sqrt(max(0.0, 1.0 - alpha_bar_prev - sigma ** 2))
        sq_ap = math.sqrt(max(0.0, alpha_bar_prev))
        if _is_numpy(sample):
            import numpy as _np
            pred_x0 = _np.clip((sample - sq_1at * model_output) / sq_at, -1.0, 1.0)
            noise = _np.random.randn(*sample.shape).astype('float32') if self.eta > 0 else _np.zeros_like(sample)
            return (sq_ap * pred_x0 + dir_coef * model_output + sigma * noise).astype('float32')
        pred_x0 = [_clip((s - sq_1at * m) / sq_at, -1.0, 1.0) for s, m in zip(sample, model_output)]
        noise = _randn_like_flat(len(sample)) if self.eta > 0 else [0.0] * len(sample)
        return [sq_ap * p + dir_coef * m + sigma * n for p, m, n in zip(pred_x0, model_output, noise)]


class FlowMatchingScheduler:
    """Conditional Flow Matching — linear interpolation path."""

    def __init__(self, num_train_timesteps: int = 1000):
        self.T = num_train_timesteps

    def add_noise(self, x0, noise, t):
        if _is_numpy(x0):
            import numpy as _np
            t_arr = _np.asarray(t, dtype='float32').ravel()
            view = t_arr
            for _ in range(x0.ndim - 1):
                view = view[..., _np.newaxis]
            return (view * x0 + (1.0 - view) * noise).astype('float32')
        tv = float(t) if not hasattr(t, '__iter__') else float(list(t)[0])
        return [tv * a + (1.0 - tv) * n for a, n in zip(x0, noise)]

    def step(self, velocity, t: float, x_t):
        dt = 1.0 / self.T
        if _is_numpy(x_t):
            return (x_t + dt * velocity).astype('float32')
        return [x + dt * v for x, v in zip(x_t, velocity)]


class DPMSolverScheduler:
    """DPM-Solver++ (Lu et al., 2022): fast ODE solver for diffusion models."""

    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001,
                 beta_end: float = 0.02, solver_order: int = 2):
        self.num_train_timesteps = num_train_timesteps
        self.solver_order = solver_order
        betas = _linspace(beta_start, beta_end, num_train_timesteps)
        alphas = [1.0 - b for b in betas]
        self.alphas_cumprod = _cumprod(alphas)
        self.timesteps: List[int] = list(range(num_train_timesteps - 1, -1, -1))
        self._prev_sample = None
        self._model_outputs: List = []

    def set_timesteps(self, num_inference_steps: int):
        step = self.num_train_timesteps // num_inference_steps
        self.timesteps = list(range(self.num_train_timesteps - 1, -1, -step))[:num_inference_steps]
        self._model_outputs = []

    def _lambda(self, t: int) -> float:
        alpha_t = math.sqrt(self.alphas_cumprod[t])
        sigma_t = math.sqrt(1 - self.alphas_cumprod[t])
        return math.log(alpha_t / (sigma_t + 1e-10))

    def step(self, model_output, timestep: int, sample):
        t = timestep
        self._model_outputs.append(model_output)
        if len(self._model_outputs) > self.solver_order:
            self._model_outputs.pop(0)
        # First-order DPM-Solver (Euler-equivalent)
        if len(self._model_outputs) == 1 or self.solver_order == 1:
            prev_t = self.timesteps[self.timesteps.index(t) + 1] if t in self.timesteps[:-1] else 0
            alpha_t = math.sqrt(self.alphas_cumprod[t])
            sigma_t = math.sqrt(1 - self.alphas_cumprod[t])
            alpha_s = math.sqrt(self.alphas_cumprod[prev_t])
            sigma_s = math.sqrt(1 - self.alphas_cumprod[prev_t])
            lam_t = self._lambda(t)
            lam_s = self._lambda(prev_t)
            h = lam_s - lam_t
            if _is_numpy(sample):
                return ((sigma_s / sigma_t) * sample -
                        (alpha_s * (math.exp(-h) - 1)) * model_output).astype('float32')
            return [(sigma_s / sigma_t) * xi - alpha_s * (math.exp(-h) - 1) * mi
                    for xi, mi in zip(sample, model_output)]
        # Second-order: use last two model outputs
        prev_t = self.timesteps[self.timesteps.index(t) + 1] if t in self.timesteps[:-1] else 0
        alpha_s = math.sqrt(self.alphas_cumprod[prev_t])
        sigma_s = math.sqrt(1 - self.alphas_cumprod[prev_t])
        sigma_t = math.sqrt(1 - self.alphas_cumprod[t])
        h = self._lambda(prev_t) - self._lambda(t)
        m0 = self._model_outputs[-1]
        m1 = self._model_outputs[-2]
        if _is_numpy(sample):
            return ((sigma_s / sigma_t) * sample -
                    alpha_s * ((math.exp(-h) - 1) * m0 + 0.5 * (math.exp(-h) - 1) * (m0 - m1))).astype('float32')
        return [(sigma_s / sigma_t) * xi - alpha_s * ((math.exp(-h) - 1) * m0i +
                0.5 * (math.exp(-h) - 1) * (m0i - m1i))
                for xi, m0i, m1i in zip(sample, m0, m1)]

    def add_noise(self, x0, noise, timestep: int):
        alpha_t = math.sqrt(self.alphas_cumprod[timestep])
        sigma_t = math.sqrt(1 - self.alphas_cumprod[timestep])
        if _is_numpy(x0):
            return (alpha_t * x0 + sigma_t * noise).astype('float32')
        return [alpha_t * a + sigma_t * n for a, n in zip(x0, noise)]


__all__ = ['DDPMScheduler', 'DDIMScheduler', 'FlowMatchingScheduler', 'DPMSolverScheduler']
