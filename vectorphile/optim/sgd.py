"""SGD optimizer with momentum and weight decay."""
import numpy as np
from vectorphile.optim.optimizer import Optimizer
from vectorphile.core.tensor import Tensor


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
                g = p.grad._data.copy()
                if weight_decay != 0:
                    g += weight_decay * p._data
                pid = id(p)
                if momentum != 0:
                    if pid not in self.state:
                        self.state[pid] = {'momentum_buffer': np.zeros_like(p._data)}
                    buf = self.state[pid]['momentum_buffer']
                    buf *= momentum
                    buf += (1 - dampening) * g
                    if nesterov:
                        g = g + momentum * buf
                    else:
                        g = buf
                p._data -= lr * g
