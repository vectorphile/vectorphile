"""Container modules — ModuleList, ModuleDict, ParameterList, ParameterDict."""
from __future__ import annotations
from collections import OrderedDict
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor


class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        self._modules_list = []
        if modules:
            for i, m in enumerate(modules):
                self.append(m)

    def append(self, module: Module) -> 'ModuleList':
        self._modules_list.append(module)
        self._parameters.update({f'_m{len(self._modules_list)-1}_{k}': v
                                   for k, v in module._parameters.items()})
        return self

    def extend(self, modules) -> 'ModuleList':
        for m in modules:
            self.append(m)
        return self

    def insert(self, index: int, module: Module) -> None:
        self._modules_list.insert(index, module)
        # Rebuild param keys
        self._parameters = {}
        for i, m in enumerate(self._modules_list):
            self._parameters.update({f'_m{i}_{k}': v for k, v in m._parameters.items()})

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return ModuleList(self._modules_list[idx])
        return self._modules_list[idx]

    def __setitem__(self, idx, module):
        self._modules_list[idx] = module

    def __len__(self):
        return len(self._modules_list)

    def __iter__(self):
        return iter(self._modules_list)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("ModuleList does not define forward — iterate and call each module manually.")

    def __repr__(self):
        lines = [f'  ({i}): {m}' for i, m in enumerate(self._modules_list)]
        return 'ModuleList(\n' + '\n'.join(lines) + '\n)'


class ModuleDict(Module):
    def __init__(self, modules=None):
        super().__init__()
        self._modules_dict: OrderedDict = OrderedDict()
        if modules:
            for k, v in (modules.items() if isinstance(modules, dict) else modules):
                self[k] = v

    def __setitem__(self, key: str, module: Module):
        self._modules_dict[key] = module
        self._parameters.update({f'_{key}_{k}': v for k, v in module._parameters.items()})

    def __getitem__(self, key: str) -> Module:
        return self._modules_dict[key]

    def __delitem__(self, key: str):
        del self._modules_dict[key]

    def __len__(self):
        return len(self._modules_dict)

    def __iter__(self):
        return iter(self._modules_dict)

    def __contains__(self, key):
        return key in self._modules_dict

    def keys(self):
        return self._modules_dict.keys()

    def values(self):
        return self._modules_dict.values()

    def items(self):
        return self._modules_dict.items()

    def update(self, modules):
        for k, v in (modules.items() if isinstance(modules, dict) else modules):
            self[k] = v

    def forward(self, *args, **kwargs):
        raise NotImplementedError("ModuleDict does not define forward.")

    def __repr__(self):
        lines = [f'  ({k}): {v}' for k, v in self._modules_dict.items()]
        return 'ModuleDict(\n' + '\n'.join(lines) + '\n)'


class ParameterList(Module):
    def __init__(self, parameters=None):
        super().__init__()
        self._params_list = []
        if parameters:
            for p in parameters:
                self.append(p)

    def append(self, param: Tensor) -> 'ParameterList':
        idx = len(self._params_list)
        self._params_list.append(param)
        self._parameters[f'_p{idx}'] = param
        return self

    def __getitem__(self, idx):
        return self._params_list[idx]

    def __setitem__(self, idx, param):
        self._params_list[idx] = param
        self._parameters[f'_p{idx}'] = param

    def __len__(self):
        return len(self._params_list)

    def __iter__(self):
        return iter(self._params_list)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class ParameterDict(Module):
    def __init__(self, parameters=None):
        super().__init__()
        if parameters:
            for k, v in (parameters.items() if isinstance(parameters, dict) else parameters):
                self[k] = v

    def __setitem__(self, key: str, param: Tensor):
        self._parameters[key] = param

    def __getitem__(self, key: str) -> Tensor:
        return self._parameters[key]

    def __delitem__(self, key: str):
        del self._parameters[key]

    def __contains__(self, key):
        return key in self._parameters

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        return iter(self._parameters)

    def keys(self):
        return self._parameters.keys()

    def values(self):
        return self._parameters.values()

    def items(self):
        return self._parameters.items()

    def update(self, params):
        for k, v in (params.items() if isinstance(params, dict) else params):
            self[k] = v

    def forward(self, *args, **kwargs):
        raise NotImplementedError
