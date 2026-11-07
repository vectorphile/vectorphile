"""vectorphile.nn.hooks — Forward/backward hooks for Module introspection."""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
import weakref


class RemovableHook:
    """Handle returned by register_*_hook; call .remove() to detach."""

    def __init__(self, hooks_dict: dict, hook_id: int):
        self._hooks_dict = weakref.ref(hooks_dict)
        self._id = hook_id

    def remove(self):
        d = self._hooks_dict()
        if d is not None and self._id in d:
            del d[self._id]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()


# ── Module mixin ──────────────────────────────────────────────────────────────

class HooksMixin:
    """Mixin that adds hook registration to any Module subclass.

    Usage — add to your Module::

        class Linear(HooksMixin, Module):
            ...
    """

    def _ensure_hooks(self):
        if not hasattr(self, '_forward_hooks'):
            self._forward_hooks: Dict[int, Callable] = {}
            self._forward_pre_hooks: Dict[int, Callable] = {}
            self._backward_hooks: Dict[int, Callable] = {}
            self._hook_counter = 0

    def register_forward_hook(
        self,
        hook: Callable[['HooksMixin', Tuple, Any], Optional[Any]],
    ) -> RemovableHook:
        """Register ``hook(module, input, output) -> None | modified_output``.

        Called after every forward pass. If the hook returns a value it
        replaces the original output.
        """
        self._ensure_hooks()
        hid = self._hook_counter
        self._hook_counter += 1
        self._forward_hooks[hid] = hook
        return RemovableHook(self._forward_hooks, hid)

    def register_forward_pre_hook(
        self,
        hook: Callable[['HooksMixin', Tuple], Optional[Tuple]],
    ) -> RemovableHook:
        """Register ``hook(module, input) -> None | modified_input``.

        Called before the forward pass. If the hook returns a value it
        replaces the input tuple passed to forward.
        """
        self._ensure_hooks()
        hid = self._hook_counter
        self._hook_counter += 1
        self._forward_pre_hooks[hid] = hook
        return RemovableHook(self._forward_pre_hooks, hid)

    def register_backward_hook(
        self,
        hook: Callable[['HooksMixin', Tuple, Tuple], Optional[Tuple]],
    ) -> RemovableHook:
        """Register ``hook(module, grad_input, grad_output) -> None | modified_grad_input``.

        Note: backward hooks require the autograd system to call
        ``_call_backward_hooks`` explicitly.
        """
        self._ensure_hooks()
        hid = self._hook_counter
        self._hook_counter += 1
        self._backward_hooks[hid] = hook
        return RemovableHook(self._backward_hooks, hid)

    # ── Internal call helpers (called by Module.__call__ / autograd) ─────────

    def _call_forward_pre_hooks(self, input_args: tuple) -> tuple:
        self._ensure_hooks()
        for hook in self._forward_pre_hooks.values():
            result = hook(self, input_args)
            if result is not None:
                input_args = result if isinstance(result, tuple) else (result,)
        return input_args

    def _call_forward_hooks(self, input_args: tuple, output: Any) -> Any:
        self._ensure_hooks()
        for hook in self._forward_hooks.values():
            result = hook(self, input_args, output)
            if result is not None:
                output = result
        return output

    def _call_backward_hooks(self, grad_input: tuple, grad_output: tuple) -> Optional[tuple]:
        self._ensure_hooks()
        for hook in self._backward_hooks.values():
            result = hook(self, grad_input, grad_output)
            if result is not None:
                grad_input = result
        return grad_input


# ── Standalone functional registration ───────────────────────────────────────

def register_forward_hook(module, hook: Callable) -> RemovableHook:
    """Functional version — equivalent to ``module.register_forward_hook(hook)``."""
    if hasattr(module, 'register_forward_hook'):
        return module.register_forward_hook(hook)
    # Fallback: monkey-patch a minimal hooks dict
    if not hasattr(module, '_forward_hooks'):
        module._forward_hooks = {}
        module._hook_counter = 0
        _patch_call(module)
    hid = module._hook_counter
    module._hook_counter += 1
    module._forward_hooks[hid] = hook
    return RemovableHook(module._forward_hooks, hid)


def register_backward_hook(module, hook: Callable) -> RemovableHook:
    if hasattr(module, 'register_backward_hook'):
        return module.register_backward_hook(hook)
    if not hasattr(module, '_backward_hooks'):
        module._backward_hooks = {}
        module._hook_counter = getattr(module, '_hook_counter', 0)
    hid = module._hook_counter
    module._hook_counter += 1
    module._backward_hooks[hid] = hook
    return RemovableHook(module._backward_hooks, hid)


def _patch_call(module):
    """Wrap module.__call__ so forward hooks fire."""
    original_call = module.__class__.__call__

    def new_call(self, *args, **kwargs):
        output = original_call(self, *args, **kwargs)
        for hook in getattr(self, '_forward_hooks', {}).values():
            result = hook(self, args, output)
            if result is not None:
                output = result
        return output

    module.__class__ = type(
        module.__class__.__name__,
        (module.__class__,),
        {'__call__': new_call},
    )


# ── Utility hooks ─────────────────────────────────────────────────────────────

class FeatureExtractor:
    """Extract intermediate activations from named layers.

    Example::

        extractor = FeatureExtractor(model, layers=['layer1', 'layer2'])
        output = model(x)
        features = extractor.features  # {'layer1': ..., 'layer2': ...}
        extractor.clear()
    """

    def __init__(self, model, layers: List[str]):
        self.model = model
        self.features: Dict[str, Any] = {}
        self._handles: List[RemovableHook] = []
        for name in layers:
            module = self._get_module(name)
            handle = register_forward_hook(module, self._make_hook(name))
            self._handles.append(handle)

    def _get_module(self, name: str):
        parts = name.split('.')
        m = self.model
        for p in parts:
            m = getattr(m, p)
        return m

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self.features[name] = output
        return hook

    def clear(self):
        self.features.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove()


class GradientRecorder:
    """Record gradients flowing through named parameters.

    Example::

        recorder = GradientRecorder(model, ['weight', 'bias'])
        loss.backward()
        grads = recorder.gradients  # {'weight': ndarray, ...}
    """

    def __init__(self, model, param_names: List[str]):
        self.gradients: Dict[str, Any] = {}
        self._handles: List[RemovableHook] = []
        for name, param in model.named_parameters():
            if name in param_names or any(name.endswith(n) for n in param_names):
                self._register(name, param)

    def _register(self, name: str, param):
        import numpy as np

        def hook(grad):
            self.gradients[name] = np.copy(grad) if hasattr(grad, 'copy') else grad
            return grad

        # Attach as a grad_fn hook if tensor has grad attribute
        if hasattr(param, '_grad_hooks'):
            hid = len(param._grad_hooks)
            param._grad_hooks[hid] = hook
            self._handles.append(RemovableHook(param._grad_hooks, hid))

    def remove(self):
        for h in self._handles:
            h.remove()
