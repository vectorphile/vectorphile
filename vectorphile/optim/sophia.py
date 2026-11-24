"""Sophia optimizer (Liu et al., 2023)."""
import math as _math
from vectorphile.optim.optimizer import Optimizer
from vectorphile.optim.adam import _to_np, _write_back


class Sophia(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), eps=1e-8,
                 weight_decay=0.1, rho=0.04, k=10):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, rho=rho, k=k)
        super().__init__(params, defaults)
        self._step_count = 0

    def step(self, closure=None):
        self._step_count += 1
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            rho = group['rho']
            clip_val = 1.0 / (rho + eps)
            for p in group['params']:
                if p.grad is None:
                    continue
                g, sh = _to_np(p.grad._data)
                g = [float(v) for v in g]
                n = len(g)
                state = self.state[id(p)]
                if 'step' not in state:
                    state['step'] = 0
                    state['m'] = [0.0] * n
                    state['h'] = [1.0] * n
                state['step'] += 1
                m, h = state['m'], state['h']
                m = [beta1 * m[i] + (1 - beta1) * g[i] for i in range(n)]
                if state['step'] % group['k'] == 0:
                    h = [beta2 * h[i] + (1 - beta2) * g[i] * g[i] for i in range(n)]
                state['m'], state['h'] = m, h
                p_flat, p_sh = _to_np(p._data)
                p_flat = [float(v) for v in p_flat]
                if wd != 0:
                    p_flat = [v * (1 - lr * wd) for v in p_flat]
                update = [m[i] / max(rho * h[i], eps) for i in range(n)]
                update = [max(-clip_val, min(clip_val, u)) for u in update]
                p_flat = [p_flat[i] - lr * update[i] for i in range(n)]
                _write_back(p, p_flat, p_sh)
