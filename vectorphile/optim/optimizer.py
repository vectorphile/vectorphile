"""Base Optimizer class."""
from typing import List, Dict, Any


class Optimizer:
    def __init__(self, params, defaults: Dict[str, Any]):
        if isinstance(params, dict):
            params = list(params.values())
        self.param_groups: List[Dict] = []
        self.defaults = defaults
        self.add_param_group({'params': list(params)})

    def add_param_group(self, param_group: Dict):
        group = {**self.defaults, **param_group}
        self.param_groups.append(group)

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad = None

    def step(self):
        raise NotImplementedError

    def state_dict(self) -> Dict:
        return {'state': self.state, 'param_groups': self.param_groups}

    def load_state_dict(self, state_dict: Dict):
        self.state = state_dict['state']
