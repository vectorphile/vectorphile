"""SGD optimizer with momentum and weight decay."""
from vectorphile.optim.optimizer import Optimizer
from vectorphile.optim.adam import _to_np, _write_back, _vec_scale, _vec_add, _vec_sub


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False, dampening=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, dampening=dampening)
        super().__init__(params, defaults)
        self.state = {}

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for p in group['params']:
                if p.grad is None:
                    continue
                g, sh = _to_np(p.grad._data)
                g = g[:]
                p_flat, p_sh = _to_np(p._data)
                if weight_decay != 0:
                    g = _vec_add(g, _vec_scale(p_flat, weight_decay))
                pid = id(p)
                if momentum != 0:
                    n = len(g)
                    if pid not in self.state:
                        self.state[pid] = {'momentum_buffer': [0.0] * n}
                    buf = self.state[pid]['momentum_buffer']
                    buf = _vec_add(_vec_scale(buf, momentum), _vec_scale(g, 1 - dampening))
                    self.state[pid]['momentum_buffer'] = buf
                    if nesterov:
                        g = _vec_add(g, _vec_scale(buf, momentum))
                    else:
                        g = buf
                _write_back(p, _vec_sub(p_flat, _vec_scale(g, lr)), p_sh)
