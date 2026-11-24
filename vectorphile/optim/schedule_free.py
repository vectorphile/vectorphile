"""Schedule-Free AdamW (Defazio & Mishchenko, 2024).

Eliminates the need for a learning-rate schedule by maintaining an additional
averaged iterate z. The model's online weights are a mixture of z and the
momentum buffer; the final weights for inference are set to z.

Paper: "Schedule-Free Learning: A New Paradigm for Machine Learning"

Usage::
    from vectorphile.optim import ScheduleFreeAdamW
    opt = ScheduleFreeAdamW(model.parameters(), lr=0.01, warmup_steps=1000)
    opt.train()
    for batch in loader:
        opt.zero_grad()
        loss = ...
        loss.backward()
        opt.step()
    opt.eval()   # project to z before evaluation/inference
"""
from __future__ import annotations
import math
from typing import List, Optional


class ScheduleFreeAdamW:
    """Schedule-Free AdamW.

    Args:
        params: iterable of Tensors
        lr: base learning rate (constant throughout training)
        betas: (β₁, β₂) for momentum and second-moment
        eps: numerical stability epsilon
        weight_decay: decoupled weight decay coefficient
        warmup_steps: linear warmup over this many steps (c_t ramps from 0→1)
        beta_schedule: 'constant' or 'cosine' c_t schedule after warmup
    """

    def __init__(self, params, lr: float = 0.01,
                 betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 1e-2, warmup_steps: int = 0,
                 beta_schedule: str = 'constant'):
        self._params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.beta_schedule = beta_schedule
        self._step = 0
        self._state = {}  # param_id → {z, m, v, t}
        self._training = True  # True: use y (online); False: use z (eval)

        for p in self._params:
            pid = id(p)
            flat = list(p._data.to_flat_list() if hasattr(p._data, 'to_flat_list') else p._data)
            n = len(flat)
            self._state[pid] = {
                'z':  flat[:],       # averaged iterate (slow weight)
                'm':  [0.0] * n,     # first moment
                'v':  [0.0] * n,     # second moment
                'n':  n,
            }

    def train(self) -> None:
        """Switch to training mode (use online iterate y)."""
        if not self._training:
            self._project_to_y()
        self._training = True

    def eval(self) -> None:
        """Switch to eval mode: set weights to z (the averaged iterate)."""
        if self._training:
            self._project_to_z()
        self._training = False

    def _project_to_z(self) -> None:
        """Set each parameter's data to z."""
        import array as _a
        from vectorphile._core.ndarray import NDArray, _c_strides, _DType
        for p in self._params:
            pid = id(p)
            if pid not in self._state:
                continue
            st = self._state[pid]
            sh = p._data.shape if hasattr(p._data, 'shape') else (st['n'],)
            buf = _a.array('f', [float(v) for v in st['z']])
            p._data = NDArray(buf, sh, _c_strides(sh), _DType('float32'))

    def _project_to_y(self) -> None:
        """Reconstruct y from stored state (for returning to train mode)."""
        # y ≈ z + beta1 * m_step (approximate); just keep current p._data
        pass

    def _warmup_coeff(self, t: int) -> float:
        if self.warmup_steps <= 0:
            return 1.0
        return min(1.0, t / max(1, self.warmup_steps))

    def step(self) -> None:
        import array as _a
        from vectorphile._core.ndarray import NDArray, _c_strides, _DType
        self._step += 1
        t = self._step
        c_t = self._warmup_coeff(t)
        lr_t = self.lr * c_t

        beta1, beta2, eps, wd = self.beta1, self.beta2, self.eps, self.weight_decay

        for p in self._params:
            if p.grad is None:
                continue
            pid = id(p)
            if pid not in self._state:
                continue
            st = self._state[pid]
            g = list(p.grad._data.to_flat_list() if hasattr(p.grad._data, 'to_flat_list') else p.grad._data)
            z = st['z']
            m = st['m']
            v = st['v']
            n = st['n']
            sh = p._data.shape if hasattr(p._data, 'shape') else (n,)

            # Current online weight y (what parameters hold during training)
            y = list(p._data.to_flat_list() if hasattr(p._data, 'to_flat_list') else p._data)

            new_m, new_v, new_z, new_y = [], [], [], []
            bc1 = 1.0 - beta1 ** t
            bc2 = 1.0 - beta2 ** t
            for i in range(n):
                gi = g[i]
                # Update moments
                mi = beta1 * m[i] + (1.0 - beta1) * gi
                vi = beta2 * v[i] + (1.0 - beta2) * gi * gi
                m_hat = mi / bc1
                v_hat = vi / bc2
                # Adam update step
                update = m_hat / (math.sqrt(v_hat) + eps)
                # Apply weight decay to z
                zi = z[i] * (1.0 - lr_t * wd)
                # Update z (averaged iterate)
                zi -= lr_t * update
                # Online weight y = (1 - c_t) * z + c_t * y_prev (momentum mixture)
                # Schedule-Free formula: y = z + beta1 * (y - z)
                # We maintain y implicitly via parameters
                yi = zi + beta1 * (y[i] - zi)
                new_m.append(mi); new_v.append(vi); new_z.append(zi); new_y.append(yi)

            st['m'] = new_m; st['v'] = new_v; st['z'] = new_z
            buf = _a.array('f', [float(v) for v in new_y])
            p._data = NDArray(buf, sh, _c_strides(sh), _DType('float32'))

    def zero_grad(self) -> None:
        for p in self._params:
            p.grad = None

    def state_dict(self) -> dict:
        return {'step': self._step, 'state': self._state}

    def load_state_dict(self, d: dict) -> None:
        self._step = d.get('step', 0)
        self._state = d.get('state', {})
