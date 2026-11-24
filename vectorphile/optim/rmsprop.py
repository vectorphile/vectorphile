"""RMSprop optimizer."""
import math as _math
from vectorphile.optim.optimizer import Optimizer
from vectorphile.optim.adam import _to_np, _write_back, _vec_scale, _vec_add, _vec_sub


class RMSprop(Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)
        self.state = {}

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            eps = group['eps']
            wd = group['weight_decay']
            mu = group['momentum']
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
                    self.state[pid] = {'sq': [0.0] * n, 'buf': [0.0] * n}
                s = self.state[pid]
                s['sq'] = [alpha * s['sq'][i] + (1 - alpha) * g[i] * g[i] for i in range(n)]
                step_v = [g[i] / (_math.sqrt(max(s['sq'][i], 0.0)) + eps) for i in range(n)]
                if mu != 0:
                    s['buf'] = _vec_add(_vec_scale(s['buf'], mu), step_v)
                    step_v = s['buf']
                _write_back(p, _vec_sub(p_flat, _vec_scale(step_v, lr)), p_sh)
