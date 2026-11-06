"""Adam and AdamW optimizers."""
import numpy as np
from vectorphile.optim.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)
        self.state = {}

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            amsgrad = group['amsgrad']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad._data
                pid = id(p)
                if pid not in self.state:
                    self.state[pid] = {'step': 0, 'm': np.zeros_like(p._data), 'v': np.zeros_like(p._data)}
                    if amsgrad:
                        self.state[pid]['v_max'] = np.zeros_like(p._data)
                s = self.state[pid]
                s['step'] += 1
                s['m'] = b1 * s['m'] + (1 - b1) * g
                s['v'] = b2 * s['v'] + (1 - b2) * g**2
                m_hat = s['m'] / (1 - b1**s['step'])
                v_hat = s['v'] / (1 - b2**s['step'])
                if amsgrad:
                    s['v_max'] = np.maximum(s['v_max'], v_hat)
                    denom = np.sqrt(s['v_max']) + eps
                else:
                    denom = np.sqrt(v_hat) + eps
                if wd != 0:
                    p._data -= lr * wd * p._data
                p._data -= lr * m_hat / denom


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.state = {}

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                # Decoupled weight decay
                p._data *= (1 - lr * wd)
                g = p.grad._data
                pid = id(p)
                if pid not in self.state:
                    self.state[pid] = {'step': 0, 'm': np.zeros_like(p._data), 'v': np.zeros_like(p._data)}
                s = self.state[pid]
                s['step'] += 1
                s['m'] = b1 * s['m'] + (1 - b1) * g
                s['v'] = b2 * s['v'] + (1 - b2) * g**2
                m_hat = s['m'] / (1 - b1**s['step'])
                v_hat = s['v'] / (1 - b2**s['step'])
                p._data -= lr * m_hat / (np.sqrt(v_hat) + eps)
