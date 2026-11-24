"""Lookahead optimizer wrapper (Zhang et al., 2019)."""
from vectorphile.optim.optimizer import Optimizer
from vectorphile.optim.adam import _to_np, _write_back, _vec_add, _vec_sub, _vec_scale


class Lookahead(Optimizer):
    """Lookahead wraps any base optimizer.

    Maintains a set of slow weights updated every k steps by interpolating
    between slow and fast weights::

        slow = slow + alpha * (fast - slow)

    Usage::
        base = Adam(model.parameters(), lr=1e-3)
        optimizer = Lookahead(base, k=5, alpha=0.5)
        optimizer.step()
    """
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self._step_count = 0
        self._slow = {}
        # copy current fast weights as slow weights
        for group in base_optimizer.param_groups:
            for p in group['params']:
                p_flat, sh = _to_np(p._data)
                self._slow[id(p)] = (list(p_flat), sh)

        # Expose param_groups for compatibility
        self.param_groups = base_optimizer.param_groups
        self.defaults = getattr(base_optimizer, 'defaults', {})

    def step(self):
        self.base_optimizer.step()
        self._step_count += 1
        if self._step_count % self.k == 0:
            self._sync_slow_weights()

    def _sync_slow_weights(self):
        alpha = self.alpha
        for group in self.param_groups:
            for p in group['params']:
                p_flat, sh = _to_np(p._data)
                if id(p) not in self._slow:
                    self._slow[id(p)] = (list(p_flat), sh)
                    continue
                slow_flat, _ = self._slow[id(p)]
                # slow = slow + alpha * (fast - slow)
                diff = _vec_sub(p_flat, slow_flat)
                new_slow = _vec_add(slow_flat, _vec_scale(diff, alpha))
                self._slow[id(p)] = (new_slow, sh)
                # set fast = slow
                _write_back(p, new_slow, sh)

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def state_dict(self):
        return {
            'base': getattr(self.base_optimizer, 'state_dict', lambda: {})(),
            'slow': {k: v[0] for k, v in self._slow.items()},
            'step_count': self._step_count,
        }

    def load_state_dict(self, state):
        self._step_count = state.get('step_count', 0)
        for k, v in state.get('slow', {}).items():
            if k in self._slow:
                _, sh = self._slow[k]
                self._slow[k] = (v, sh)
        if hasattr(self.base_optimizer, 'load_state_dict') and 'base' in state:
            self.base_optimizer.load_state_dict(state['base'])
