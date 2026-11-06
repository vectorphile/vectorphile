"""Adagrad optimizer."""
import numpy as np
from vectorphile.optim.optimizer import Optimizer


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
                g = p.grad._data.copy()
                if wd != 0:
                    g += wd * p._data
                pid = id(p)
                if pid not in self.state:
                    self.state[pid] = {'step': 0, 'sum': np.zeros_like(p._data)}
                s = self.state[pid]
                s['step'] += 1
                clr = lr / (1 + (s['step'] - 1) * lr_decay)
                s['sum'] += g**2
                p._data -= clr * g / (np.sqrt(s['sum']) + eps)
