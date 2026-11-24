"""Adadelta optimizer (Zeiler 2012)."""
import math as _math
from vectorphile.optim.optimizer import Optimizer
from vectorphile.optim.adam import _to_np, _write_back


class Adadelta(Optimizer):
    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0):
        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            rho = group['rho']
            eps = group['eps']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g, sh = _to_np(p.grad._data)
                g = [float(v) for v in g]
                p_flat, p_sh = _to_np(p._data)
                p_flat = [float(v) for v in p_flat]
                if wd != 0:
                    g = [g[i] + wd * p_flat[i] for i in range(len(g))]
                state = self.state[id(p)]
                n = len(g)
                if 'sq_avg' not in state:
                    state['sq_avg'] = [0.0] * n
                    state['acc_delta'] = [0.0] * n
                sq_avg = state['sq_avg']
                acc_delta = state['acc_delta']
                sq_avg = [rho * sq_avg[i] + (1 - rho) * g[i] * g[i] for i in range(n)]
                delta = [_math.sqrt(max(acc_delta[i] + eps, 0.0)) / _math.sqrt(max(sq_avg[i] + eps, 0.0)) * (-g[i])
                         for i in range(n)]
                acc_delta = [rho * acc_delta[i] + (1 - rho) * delta[i] * delta[i] for i in range(n)]
                state['sq_avg'] = sq_avg
                state['acc_delta'] = acc_delta
                new_p = [p_flat[i] + lr * delta[i] for i in range(n)]
                _write_back(p, new_p, p_sh)
