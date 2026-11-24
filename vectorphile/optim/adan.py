"""Adan — Adaptive Nesterov Momentum Algorithm (Xie et al., 2022).

Uses three moments (gradient, gradient difference, second moment of gradient
difference) to achieve superior convergence especially for Transformers.

Paper: "Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models"
"""
from __future__ import annotations
import math


class Adan:
    """Adan optimizer.

    Args:
        params: iterable of Tensors
        lr: learning rate
        betas: (β₁, β₂, β₃) for the three EMA moments
        eps: numerical stability
        weight_decay: decoupled weight decay
        no_prox: if True, use simplified weight decay (AdamW-style)
    """

    def __init__(self, params, lr: float = 1e-3,
                 betas=(0.98, 0.92, 0.99), eps: float = 1e-8,
                 weight_decay: float = 0.0, no_prox: bool = False):
        self._params = list(params)
        self.lr = lr
        self.beta1, self.beta2, self.beta3 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.no_prox = no_prox
        self._step = 0
        self._state = {}

        for p in self._params:
            pid = id(p)
            flat = list(p._data.to_flat_list() if hasattr(p._data, 'to_flat_list') else p._data)
            n = len(flat)
            self._state[pid] = {
                'm':    [0.0] * n,  # first moment (EMA of g)
                'v':    [0.0] * n,  # second moment (EMA of Δg)
                'n':    [0.0] * n,  # third moment (EMA of (Δg)²)
                'g_prev': flat[:],  # previous gradient
            }

    def step(self) -> None:
        import array as _a
        from vectorphile._core.ndarray import NDArray, _c_strides, _DType
        self._step += 1
        t = self._step
        b1, b2, b3 = self.beta1, self.beta2, self.beta3
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        bc3 = 1.0 - b3 ** t

        for p in self._params:
            if p.grad is None:
                continue
            pid = id(p)
            if pid not in self._state:
                continue
            st = self._state[pid]
            g = list(p.grad._data.to_flat_list() if hasattr(p.grad._data, 'to_flat_list') else p.grad._data)
            m = st['m']; v = st['v']; n_st = st['n']; g_prev = st['g_prev']
            sh = p._data.shape if hasattr(p._data, 'shape') else (len(g),)
            w = list(p._data.to_flat_list() if hasattr(p._data, 'to_flat_list') else p._data)

            new_m, new_v, new_n, new_w = [], [], [], []
            for i in range(len(g)):
                gi = g[i]
                dg = gi - g_prev[i]  # gradient difference
                # Update three moments
                mi = b1 * m[i] + (1.0 - b1) * gi
                vi = b2 * v[i] + (1.0 - b2) * dg
                ni = b3 * n_st[i] + (1.0 - b3) * dg * dg
                # Bias correction
                m_hat = mi / bc1
                v_hat = vi / bc2
                n_hat = ni / bc3
                # Adan update
                step_size = m_hat + b2 * v_hat
                denom = math.sqrt(n_hat) + self.eps
                update = step_size / denom
                # Weight update
                wi = w[i]
                if self.no_prox:
                    wi = wi * (1.0 - self.lr * self.weight_decay) - self.lr * update
                else:
                    wi = (wi - self.lr * update) / (1.0 + self.lr * self.weight_decay)
                new_m.append(mi); new_v.append(vi); new_n.append(ni); new_w.append(wi)

            st['m'] = new_m; st['v'] = new_v; st['n'] = new_n; st['g_prev'] = g[:]
            buf = _a.array('f', [float(v) for v in new_w])
            p._data = NDArray(buf, sh, _c_strides(sh), _DType('float32'))

    def zero_grad(self) -> None:
        for p in self._params:
            p.grad = None

    def state_dict(self) -> dict:
        return {'step': self._step, 'state': self._state}

    def load_state_dict(self, d: dict) -> None:
        self._step = d.get('step', 0)
        self._state = d.get('state', {})
