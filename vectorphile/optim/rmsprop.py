"""RMSprop and Adagrad optimizers."""
import numpy as np
from vectorphile.optim.optimizer import Optimizer


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
                g = p.grad._data.copy()
                if wd != 0:
                    g += wd * p._data
                pid = id(p)
                if pid not in self.state:
                    self.state[pid] = {'sq': np.zeros_like(p._data), 'buf': np.zeros_like(p._data)}
                s = self.state[pid]
                s['sq'] = alpha * s['sq'] + (1 - alpha) * g**2
                step = g / (np.sqrt(s['sq']) + eps)
                if mu != 0:
                    s['buf'] = mu * s['buf'] + step
                    step = s['buf']
                p._data -= lr * step
