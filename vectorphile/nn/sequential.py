"""Sequential container."""
from collections import OrderedDict
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor


class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self._modules[key] = module
                setattr(self, key, module)
        else:
            for i, module in enumerate(args):
                self._modules[str(i)] = module
                object.__setattr__(self, str(i), module)

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x

    def append(self, module: Module) -> 'Sequential':
        idx = str(len(self._modules))
        self._modules[idx] = module
        object.__setattr__(self, idx, module)
        return self

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            keys = list(self._modules.keys())[idx]
            return Sequential(OrderedDict((k, self._modules[k]) for k in keys))
        return list(self._modules.values())[idx]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __repr__(self):
        lines = ['Sequential(']
        for i, (name, m) in enumerate(self._modules.items()):
            lines.append(f'  ({name}): {repr(m)}')
        lines.append(')')
        return '\n'.join(lines)
