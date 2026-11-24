"""Base Module class — mirrors PyTorch's nn.Module API."""
from __future__ import annotations
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
            v = p._data.copy()
            try:
                import numpy as _np_sd
                flat = v.to_flat_list() if hasattr(v, 'to_flat_list') else list(v)
                sh = v.shape if hasattr(v, 'shape') else (len(flat),)
                v = _np_sd.array(flat, dtype='float32').reshape(sh)
            except Exception:
                pass
            sd[name] = v
        return sd

    def load_state_dict(self, state_dict: dict, strict=True):
        from vectorphile.core.tensor import Tensor
        from vectorphile._core.ndarray import array as _nd_lsd
        own = dict(self.named_parameters())
        missing_keys = []
        unexpected_keys = []
        for key, val in state_dict.items():
            if key in own:
                p = own[key]
                # Rebuild _data from val (which may be list, NDArray, or Tensor)
                if isinstance(val, Tensor):
                    flat = val._data.to_flat_list() if hasattr(val._data, 'to_flat_list') else list(val._data)
                    sh = val.shape
                elif hasattr(val, 'to_flat_list'):
                    flat = val.to_flat_list(); sh = val.shape
                elif hasattr(val, 'tolist'):
                    flat = val.flatten().tolist(); sh = val.shape
                else:
                    flat = list(val); sh = p.shape
                new_nd = _nd_lsd(flat, 'float32')
                if len(sh) > 1:
                    new_nd = new_nd.reshape(sh)
                p._data = new_nd
            else:
                unexpected_keys.append(key)
                if strict:
                    raise KeyError(f'Unexpected key "{key}" in state_dict')
        for key in own:
            if key not in state_dict:
                missing_keys.append(key)
                if strict:
                    raise KeyError(f'Missing key "{key}" in state_dict')
        return {'missing_keys': missing_keys, 'unexpected_keys': unexpected_keys}

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

    def register_forward_hook(self, hook):
        """Register a hook called after forward(). Returns a handle with .remove()."""
        if not hasattr(self, '_forward_hooks'):
            self._forward_hooks = {}
        h_id = id(hook)
        self._forward_hooks[h_id] = hook
        class _Handle:
            def __init__(self, hooks, key):
                self._hooks = hooks; self._key = key
            def remove(self):
                self._hooks.pop(self._key, None)
        return _Handle(self._forward_hooks, h_id)

    def register_backward_hook(self, hook):
        """Register a hook called after backward(). Returns a handle with .remove()."""
        if not hasattr(self, '_backward_hooks'):
            self._backward_hooks = {}
        h_id = id(hook)
        self._backward_hooks[h_id] = hook
        class _Handle:
            def __init__(self, hooks, key):
                self._hooks = hooks; self._key = key
            def remove(self):
                self._hooks.pop(self._key, None)
        return _Handle(self._backward_hooks, h_id)

    def __call__(self, *args, **kwargs):
        result = self.forward(*args, **kwargs)
        if hasattr(self, '_forward_hooks') and self._forward_hooks:
            for hook in list(self._forward_hooks.values()):
                try:
                    hook_out = hook(self, args, result)
                    if hook_out is not None:
                        result = hook_out
                except Exception:
                    pass
        return result

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

    def _cast_params(self, dtype: str) -> 'Module':
        """Cast all parameters and buffers to `dtype` in-place."""
        for p in self.parameters():
            try:
                p._data = p._data.astype(dtype)
            except Exception:
                pass
        return self

    def half(self) -> 'Module':
        """Cast all parameters to float16 (FP16)."""
        return self._cast_params('float16')

    def float(self) -> 'Module':
        """Cast all parameters to float32."""
        return self._cast_params('float32')

    def double(self) -> 'Module':
        """Cast all parameters to float64."""
        return self._cast_params('float64')

    def cpu(self) -> 'Module':
        """Move all parameters to CPU."""
        return self.to(device='cpu')

    def cuda(self, device_id: int = 0) -> 'Module':
        """Move all parameters to CUDA device."""
        return self.to(device='cuda')

    def to(self, device=None, dtype=None) -> 'Module':
        """Move and/or cast all parameters to device/dtype.

        Args:
            device: 'cpu' or 'cuda' (string). Device moves are recorded but not
                    physically performed in pure-VP mode.
            dtype: target dtype string, e.g. 'float16', 'bfloat16', 'float32'.
        """
        if dtype is not None:
            dtype_str = str(dtype) if not isinstance(dtype, str) else dtype
            self._cast_params(dtype_str)
        if device is not None:
            dev_str = str(device) if not isinstance(device, str) else device
            self._device = dev_str
            for p in self.parameters():
                try:
                    p.device = dev_str
                except Exception:
                    pass
        return self

    def requires_grad_(self, requires_grad: bool = True) -> 'Module':
        for p in self.parameters():
            p.requires_grad = requires_grad
        return self

    def freeze(self) -> 'Module':
        return self.requires_grad_(False)

    def unfreeze(self) -> 'Module':
        return self.requires_grad_(True)
