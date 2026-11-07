"""Base Module class — mirrors PyTorch's nn.Module API."""
from __future__ import annotations
import numpy as np
from typing import Dict, Iterator, List, Optional, Tuple, Any
from collections import OrderedDict


class Module:
    def __init__(self):
        self._parameters: OrderedDict = OrderedDict()
        self._modules: OrderedDict = OrderedDict()
        self._buffers: OrderedDict = OrderedDict()
        self.training: bool = True

    def __setattr__(self, name: str, value):
        from vectorphile.core.tensor import Tensor
        if isinstance(value, Tensor) and value.requires_grad:
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self) -> Iterator:
        for p in self._parameters.values():
            yield p
        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self, prefix='') -> Iterator[Tuple[str, Any]]:
        for name, p in self._parameters.items():
            yield (f'{prefix}.{name}' if prefix else name), p
        for mname, module in self._modules.items():
            full = f'{prefix}.{mname}' if prefix else mname
            yield from module.named_parameters(prefix=full)

    def buffers(self) -> Iterator:
        yield from []

    def named_buffers(self, prefix='') -> Iterator[Tuple[str, Any]]:
        yield from []

    def children(self) -> Iterator['Module']:
        yield from self._modules.values()

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        yield from self._modules.items()

    def modules(self) -> Iterator['Module']:
        yield self
        for m in self._modules.values():
            yield from m.modules()

    def named_modules(self, prefix='') -> Iterator[Tuple[str, 'Module']]:
        yield prefix or '', self
        for name, m in self._modules.items():
            full = f'{prefix}.{name}' if prefix else name
            yield from m.named_modules(prefix=full)

    def state_dict(self) -> OrderedDict:
        sd = OrderedDict()
        for name, p in self.named_parameters():
            sd[name] = p._data.copy()
        return sd

    def load_state_dict(self, state_dict: dict, strict=True):
        from vectorphile.core.tensor import Tensor
        own = dict(self.named_parameters())
        for key, val in state_dict.items():
            if key in own:
                own[key]._data[:] = val
            elif strict:
                raise KeyError(f'Unexpected key "{key}" in state_dict')

    def train(self, mode=True) -> 'Module':
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> 'Module':
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self):
        lines = [self.__class__.__name__ + '(']
        for name, m in self._modules.items():
            mod_str = repr(m)
            mod_str = '  ' + mod_str.replace('\n', '\n  ')
            lines.append(f'  ({name}): {mod_str}')
        lines.append(')')
        return '\n'.join(lines)

    def register_buffer(self, name: str, tensor):
        self._buffers[name] = tensor
        object.__setattr__(self, name, tensor)

    def apply(self, fn) -> 'Module':
        for m in self.modules():
            fn(m)
        return self
