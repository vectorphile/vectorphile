"""RAdam — Rectified Adam (Liu et al. 2019).

Fixes the variance issue in early training steps by detecting when the
exponential moving average variance estimate is reliable and switching
between SGD-with-momentum and adaptive updates accordingly.
"""
import math
from vectorphile.optim.optimizer import Optimizer
from vectorphile._core.ndarray import zeros as _nd_zeros


class RAdam(Optimizer):
    """Rectified Adam optimizer.

    Args:
        params      : model parameters
        lr          : learning rate (default 1e-3)
        betas       : (beta1, beta2)  (default (0.9, 0.999))
        eps         : numerical stability (default 1e-8)
        weight_decay: L2 regularisation
    """

    def __init__(self, params, lr: float = 1e-3,
                 betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.state: dict = {}

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad._data
                if wd != 0.0:
                    g = g + wd * p._data
                state = self.state.setdefault(id(p), {})
                if 't' not in state:
                    state['t'] = 0
                    state['m'] = _nd_zeros(g.shape, dtype='float32')
                    state['v'] = _nd_zeros(g.shape, dtype='float32')
                state['t'] += 1
                t = state['t']
                state['m'] = beta1 * state['m'] + (1.0 - beta1) * g
                state['v'] = beta2 * state['v'] + (1.0 - beta2) * g * g

                m_hat = state['m'] / (1.0 - beta1 ** t)
                # maximum length of the approximated SMA
                rho_inf = 2.0 / (1.0 - beta2) - 1.0
                rho_t   = rho_inf - 2.0 * t * beta2 ** t / (1.0 - beta2 ** t)

                if rho_t > 4.0:
                    # variance is tractable — use adaptive update
                    rect = math.sqrt(
                        (rho_t - 4.0) * (rho_t - 2.0) * rho_inf /
                        ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
                    )
                    v_hat = (state['v'] / (1.0 - beta2 ** t)) ** 0.5
                    p._data = p._data - lr * rect * m_hat / (v_hat + eps)
                else:
                    # variance is intractable — fall back to SGD step
                    p._data = p._data - lr * m_hat
