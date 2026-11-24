"""LARS and LAMB optimizers.

LARS  (You et al. 2017) — Layer-wise Adaptive Rate Scaling.
      Scales each layer's LR by the ratio of weight norm to gradient norm.
      Used for large-batch SGD (ImageNet, BERT pre-training).

LAMB  (You et al. 2019) — Layer-wise Adaptive Moments for Batch training.
      Combines Adam's adaptive moments with LARS-style trust ratio.
      Used for BERT large-batch pre-training.
"""
import math
from vectorphile.optim.optimizer import Optimizer
from vectorphile._core.ndarray import zeros as _nd_zeros


def _l2_norm(x):
    s = sum(float(v) ** 2 for v in x._buf)
    return math.sqrt(s)


class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling.

    Args:
        params      : model parameters
        lr          : global learning rate
        momentum    : momentum factor
        weight_decay: L2 regularisation
        eta         : LARS coefficient (trust ratio multiplier, default 0.001)
        eps         : numerical stability for norm ratio
    """

    def __init__(self, params, lr: float = 0.1, momentum: float = 0.9,
                 weight_decay: float = 1e-4, eta: float = 0.001, eps: float = 1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        eta=eta, eps=eps)
        super().__init__(params, defaults)
        self.state: dict = {}

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            mu = group['momentum']
            wd = group['weight_decay']
            eta = group['eta']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad._data
                p_norm = _l2_norm(p._data)
                g_norm = _l2_norm(g)

                if p_norm > 0 and g_norm > 0:
                    local_lr = eta * p_norm / (g_norm + wd * p_norm + eps)
                else:
                    local_lr = 1.0

                g_eff = local_lr * (g + wd * p._data)

                state = self.state.setdefault(id(p), {})
                if 'v' not in state:
                    state['v'] = _nd_zeros(g.shape, dtype='float32')
                state['v'] = mu * state['v'] + g_eff
                p._data = p._data - lr * state['v']


class LAMB(Optimizer):
    """Layer-wise Adaptive Moments for Batch training.

    Args:
        params      : model parameters
        lr          : learning rate (default 1e-3)
        betas       : (beta1, beta2)  (default (0.9, 0.999))
        eps         : numerical stability
        weight_decay: L2 regularisation
        clamp_value : trust ratio clamp  (None = no clamp)
    """

    def __init__(self, params, lr: float = 1e-3,
                 betas=(0.9, 0.999), eps: float = 1e-6,
                 weight_decay: float = 0.01, clamp_value: float = 10.0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, clamp_value=clamp_value)
        super().__init__(params, defaults)
        self.state: dict = {}

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            clamp = group['clamp_value']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad._data
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
                v_hat = state['v'] / (1.0 - beta2 ** t)

                # Adam update direction + weight decay
                adam_step = m_hat / (v_hat ** 0.5 + eps) + wd * p._data

                # trust ratio
                p_norm    = _l2_norm(p._data)
                step_norm = _l2_norm(adam_step)
                if p_norm > 0 and step_norm > 0:
                    trust = p_norm / step_norm
                    if clamp is not None:
                        trust = min(trust, clamp)
                else:
                    trust = 1.0

                p._data = p._data - lr * trust * adam_step
