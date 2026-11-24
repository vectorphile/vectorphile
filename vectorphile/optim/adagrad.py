"""Adagrad optimizer."""
import math as _math
from vectorphile.optim.optimizer import Optimizer
from vectorphile.optim.adam import _to_np, _write_back, _vec_scale, _vec_add, _vec_sub


class Adagrad(Optimizer):
    def __init__(self, params, lr=0.01, lr_decay=0.0, weight_decay=0.0, eps=1e-10):
        defaults = dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)
        self.state = {}

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            lr_decay = group['lr_decay']
            wd = group['weight_decay']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                g, sh = _to_np(p.grad._data)
                g = g[:]
                p_flat, p_sh = _to_np(p._data)
                if wd != 0:
                    g = _vec_add(g, _vec_scale(p_flat, wd))
                pid = id(p)
                n = len(g)
                if pid not in self.state:
                    self.state[pid] = {'step': 0, 'sum': [0.0] * n}
                s = self.state[pid]
                s['step'] += 1
                clr = lr / (1 + (s['step'] - 1) * lr_decay)
                s['sum'] = _vec_add(s['sum'], [v * v for v in g])
                denom = [_math.sqrt(max(v, 0.0)) + eps for v in s['sum']]
                update = [g[i] / denom[i] for i in range(n)]
                _write_back(p, _vec_sub(p_flat, _vec_scale(update, clr)), p_sh)
