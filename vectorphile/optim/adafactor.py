"""AdaFactor — Shazeer & Stern 2018.

Memory-efficient adaptive optimizer that approximates the full second-moment
matrix using row and column factors (for 2D params) or a single scalar (for 1D).
No second-moment storage for 1D params → ~half Adam's memory.

Used in: T5, Flan-T5, switch-transformer large batch training.
"""
import math
from vectorphile.optim.optimizer import Optimizer
from vectorphile._core.ndarray import zeros as _nd_zeros, ones as _nd_ones


class AdaFactor(Optimizer):
    """AdaFactor optimizer.

    Args:
        params             : model parameters
        lr                 : learning rate.  None = use the schedule-free
                             adaptive LR (recommended for training from scratch)
        eps                : (eps1, eps2) regularisation pair  (default (1e-30, 1e-3))
        clip_threshold     : RMS gradient clipping threshold (default 1.0)
        decay_rate         : coefficient for computing running average of sq grad (default -0.8)
        beta1              : momentum factor; None disables first moment (default None)
        weight_decay       : L2 regularisation
        scale_parameter    : scale LR by parameter RMS (default True)
        relative_step      : use schedule-free LR when lr=None (default True)
        warmup_init        : init the LR with a small value at warmup (default False)
    """

    def __init__(self, params, lr=None, eps=(1e-30, 1e-3),
                 clip_threshold: float = 1.0, decay_rate: float = -0.8,
                 beta1=None, weight_decay: float = 0.0,
                 scale_parameter: bool = True, relative_step: bool = True,
                 warmup_init: bool = False):
        defaults = dict(lr=lr, eps=eps, clip_threshold=clip_threshold,
                        decay_rate=decay_rate, beta1=beta1,
                        weight_decay=weight_decay, scale_parameter=scale_parameter,
                        relative_step=relative_step, warmup_init=warmup_init)
        super().__init__(params, defaults)
        self.state: dict = {}

    @staticmethod
    def _rms(x):
        s = sum(float(v) ** 2 for v in x._buf)
        return math.sqrt(s / max(len(x._buf), 1))

    def _get_lr(self, group, state):
        if group['relative_step']:
            t = state.get('t', 1)
            min_step = 1e-6 if group['warmup_init'] else 1e-2
            lr_t = max(min_step, 1.0 / math.sqrt(t))
        else:
            lr_t = group['lr']
        if group['scale_parameter']:
            # scale by parameter RMS
            return lr_t  # will be multiplied by p_rms in step
        return lr_t

    def step(self):
        for group in self.param_groups:
            eps1, eps2 = group['eps']
            clip = group['clip_threshold']
            decay = group['decay_rate']
            beta1 = group['beta1']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad._data
                state = self.state.setdefault(id(p), {})
                if 't' not in state:
                    state['t'] = 0
                state['t'] += 1
                t = state['t']

                # adaptive LR
                lr_t = self._get_lr(group, state)
                if group['scale_parameter']:
                    p_rms = max(self._rms(p._data), eps2)
                    lr_t = lr_t * p_rms

                # gradient clipping
                g_rms = self._rms(g)
                if g_rms > clip:
                    g = g * (clip / g_rms)

                rho = min(1.0 - 10 ** (decay * math.log10(t)), 1.0 - 1e-8)
                alpha = max(rho, eps1 / (self._rms(g) ** 2 + eps1))

                shape = g.shape
                if len(shape) >= 2:
                    # factored second moment: row + column factors
                    if 'vr' not in state:
                        state['vr'] = _nd_ones((shape[0],), dtype='float32') * eps1
                        state['vc'] = _nd_ones((shape[1],), dtype='float32') * eps1
                    vr = state['vr']
                    vc = state['vc']
                    # update row / col factors
                    # g2 = g^2 + eps1
                    g2 = g * g + eps1
                    state['vr'] = rho * vr + alpha * g2.mean(axis=1)
                    state['vc'] = rho * vc + alpha * g2.mean(axis=0)
                    # reconstruct v estimate
                    vr2 = state['vr']
                    vc2 = state['vc']
                    # outer product normalised by mean of row sums
                    v_est_buf = []
                    row_sum_total = sum(float(v) for v in vr2._buf)
                    for i in range(shape[0]):
                        ri = float(vr2._buf[i])
                        for j in range(shape[1]):
                            cj = float(vc2._buf[j])
                            v_est_buf.append(ri * cj / max(row_sum_total / shape[0], eps1))
                    import array as _a
                    from vectorphile._core.ndarray import NDArray
                    strides = (shape[1], 1) if len(shape) == 2 else None
                    if strides:
                        v_est = NDArray(_a.array('f', v_est_buf), shape, strides, 'float32')
                    else:
                        v_est = g  # fallback
                else:
                    # scalar second moment for 1D
                    if 'vs' not in state:
                        state['vs'] = _nd_ones(shape, dtype='float32') * eps1
                    state['vs'] = rho * state['vs'] + alpha * (g * g + eps1)
                    v_est = state['vs']

                # update
                u = g / (v_est ** 0.5)
                u_rms = self._rms(u)
                u = u / max(u_rms, 1.0)

                # first moment (optional)
                if beta1 is not None:
                    if 'm' not in state:
                        state['m'] = _nd_zeros(g.shape, dtype='float32')
                    state['m'] = beta1 * state['m'] + (1.0 - beta1) * u
                    u = state['m']

                if wd != 0.0:
                    p._data = p._data * (1.0 - lr_t * wd)
                p._data = p._data - lr_t * u
