"""Sharpness-Aware Minimization (SAM) optimizer."""
from vectorphile.optim.optimizer import Optimizer
from vectorphile.optim.adam import _to_np, _write_back, _vec_scale, _vec_add, _vec_sub
import math


def _norm(flat):
    return math.sqrt(sum(v * v for v in flat))


class SAM(Optimizer):
    """Sharpness-Aware Minimization optimizer.

    Wraps a base optimizer and perturbs parameters toward the sharpest
    ascent direction before the actual update step.

    Usage::
        optimizer = SAM(model.parameters(), SGD, lr=0.1, momentum=0.9)
        # first forward+backward
        loss.backward()
        optimizer.first_step(zero_grad=True)
        # second forward+backward (same batch)
        loss2.backward()
        optimizer.second_step(zero_grad=True)
    """
    def __init__(self, params, base_optimizer_cls, rho=0.05, adaptive=False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self._saved = {}

    def _grad_norm(self):
        norm_sq = 0.0
        for group in self.param_groups:
            adaptive = group.get('adaptive', False)
            for p in group['params']:
                if p.grad is None:
                    continue
                g_flat, _ = _to_np(p.grad._data)
                if adaptive:
                    p_flat, _ = _to_np(p._data)
                    scale = [abs(pv) for pv in p_flat]
                    g_scaled = [gv * sv for gv, sv in zip(g_flat, scale)]
                    norm_sq += sum(v * v for v in g_scaled)
                else:
                    norm_sq += sum(v * v for v in g_flat)
        return math.sqrt(norm_sq)

    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm() + 1e-12
        for group in self.param_groups:
            rho = group['rho']
            adaptive = group.get('adaptive', False)
            scale = rho / grad_norm
            for p in group['params']:
                if p.grad is None:
                    continue
                g_flat, _ = _to_np(p.grad._data)
                p_flat, p_sh = _to_np(p._data)
                # save original param
                self._saved[id(p)] = (list(p_flat), p_sh)
                if adaptive:
                    abs_p = [abs(pv) for pv in p_flat]
                    e = [gv * av * scale for gv, av in zip(g_flat, abs_p)]
                else:
                    e = [gv * scale for gv in g_flat]
                _write_back(p, _vec_add(p_flat, e), p_sh)
        if zero_grad:
            self.zero_grad()

    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if id(p) in self._saved:
                    orig, sh = self._saved.pop(id(p))
                    _write_back(p, orig, sh)
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("SAM requires explicit first_step/second_step calls.")

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                p.grad = None
