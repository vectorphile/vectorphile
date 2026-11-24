"""Sequential container."""
from collections import OrderedDict
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor

_ACT_MODULE_NAMES = {'ReLU': 'relu', 'GELU': 'gelu', 'SiLU': 'silu', 'Sigmoid': 'sigmoid', 'Tanh': 'tanh'}


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
        self._fuse_linear_acts()

    def _fuse_linear_acts(self):
        # Activation fusion disabled — matmul_act_f32 C kernel epilogue is broken
        pass

    def forward(self, x: Tensor) -> Tensor:
        from vectorphile.nn.layers import Linear
        mods = list(self._modules.values())
        skip_next = False
        for i, module in enumerate(mods):
            if skip_next:
                skip_next = False
                continue
            x = module(x)
            # If the linear had a fuse_act set and the next module is the matching activation, skip it
            if (isinstance(module, Linear)
                    and getattr(module, '_fuse_act', None) is not None
                    and i + 1 < len(mods)
                    and _ACT_MODULE_NAMES.get(type(mods[i + 1]).__name__) == module._fuse_act):
                skip_next = True
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
