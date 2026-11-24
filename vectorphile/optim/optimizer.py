"""Base Optimizer class."""
from typing import List, Dict, Any


class Optimizer:
    def __init__(self, params, defaults: Dict[str, Any]):
        self.param_groups: List[Dict] = []
        self.defaults = defaults
        if isinstance(params, dict):
            params = list(params.values())
        # Support list of param-group dicts (e.g. from lora_plus_param_groups)
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for group in params:
                self.add_param_group(group)
        else:
            self.add_param_group({'params': list(params)})

    def add_param_group(self, param_group: Dict):
        group = {**self.defaults, **param_group}
        self.param_groups.append(group)

    @property
    def lr(self) -> float:
        return self.param_groups[0]['lr']

    @lr.setter
    def lr(self, value: float):
        for g in self.param_groups:
            g['lr'] = value

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
