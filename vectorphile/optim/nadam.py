"""NAdam — Nesterov-accelerated Adaptive Moment Estimation (Dozat 2016).

Combines Adam's adaptive learning rates with Nesterov momentum:
    m_hat = beta1 * m_t / (1 - beta1^(t+1)) + (1 - beta1) * g / (1 - beta1^t)
    p -= lr * m_hat / (sqrt(v_hat) + eps)
"""
import math
from vectorphile.optim.optimizer import Optimizer
from vectorphile._core.ndarray import zeros as _nd_zeros


class NAdam(Optimizer):
    """Nesterov Adam optimizer.

    Args:
        params         : model parameters
        lr             : learning rate (default 2e-3)
        betas          : (beta1, beta2) coefficients  (default (0.9, 0.999))
        eps            : numerical stability (default 1e-8)
        weight_decay   : L2 regularisation
        momentum_decay : momentum schedule decay (default 4e-3)
    """

    def __init__(self, params, lr: float = 2e-3,
                 betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, momentum_decay: float = 4e-3):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, momentum_decay=momentum_decay)
        super().__init__(params, defaults)
        self.state: dict = {}

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            mu_product = getattr(self, '_mu_product', 1.0)

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
                m, v = state['m'], state['v']
                mu_t  = beta1 * (1.0 - 0.5 * 0.96 ** (t * group['momentum_decay']))
                mu_t1 = beta1 * (1.0 - 0.5 * 0.96 ** ((t + 1) * group['momentum_decay']))
                mu_product_now = mu_product * mu_t
                mu_product_next = mu_product_now * mu_t1
                state['m'] = beta1 * m + (1.0 - beta1) * g
                state['v'] = beta2 * v + (1.0 - beta2) * g * g
                m_hat = (mu_t1 * state['m'] / (1.0 - mu_product_next) +
                         (1.0 - mu_t) * g / (1.0 - mu_product_now))
                v_hat = state['v'] / (1.0 - beta2 ** t)
                p._data = p._data - lr * m_hat / (v_hat ** 0.5 + eps)
            self._mu_product = mu_product * beta1
