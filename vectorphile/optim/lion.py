"""Lion optimizer — Chen et al. 2023 (Symbolic Discovery of Optimization Algorithms).

Memory-efficient: stores only momentum (not second moment like Adam).
Performs a sign update: p -= lr * (sign(beta1*m + (1-beta1)*g) + wd*p)
"""
import math
from vectorphile.optim.optimizer import Optimizer
from vectorphile._core.ndarray import zeros as _nd_zeros


class Lion(Optimizer):
    """Lion optimizer.

    Args:
        params     : model parameters
        lr         : learning rate (default 1e-4; use ~3-10x smaller than Adam)
        betas      : (beta1, beta2) momentum coefficients  (default (0.9, 0.99))
        weight_decay: L2 regularisation coefficient
    """

    def __init__(self, params, lr: float = 1e-4,
                 betas=(0.9, 0.99), weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.state: dict = {}

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad._data
                state = self.state.setdefault(id(p), {})
                if 'm' not in state:
                    state['m'] = _nd_zeros(p._data.shape, dtype='float32')
                m = state['m']
                # update direction: sign(beta1 * m + (1 - beta1) * g)
                c = beta1 * m + (1.0 - beta1) * g
                # sign
                import array as _a
                buf = _a.array('f', [
                    1.0 if v > 0 else (-1.0 if v < 0 else 0.0)
                    for v in c._buf
                ])
                from vectorphile._core.ndarray import NDArray
                sign_c = NDArray(buf, c.shape, c.strides, 'float32')
                # weight decay + update
                update = sign_c + wd * p._data
                p._data = p._data - lr * update
                # update momentum
                state['m'] = beta2 * m + (1.0 - beta2) * g
