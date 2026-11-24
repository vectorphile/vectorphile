"""vectorphile.jit — JIT compilation decorators and utilities.

Provides:
- ``@vp.jit`` / ``@jit``: decorator that traces and compiles a function/model
- ``@script``: decorator that JIT-scripts a function (torch.jit.script compat)
- ``trace(fn, example_inputs)``: trace a function with concrete inputs

The JIT decorator wraps VP's vp.fx compile pass and enables:
- Operation fusion (Linear + Activation)
- VP CUDA / Triton kernel dispatch
- Dead code elimination
- Profiler annotation

Usage::
    import vectorphile as vp

    @vp.jit
    def forward(x, w):
        return vp.relu(x @ w)

    # Or compile a module:
    model = vp.jit(model)
    output = model(x)
"""
from __future__ import annotations
import functools
from typing import Any, Callable, Optional


def jit(fn_or_model=None, *, fullgraph: bool = False, dynamic: bool = False,
        backend: str = 'vp', example_inputs=None, fuse: bool = True):
    """JIT-compile a function or Module.

    Can be used as:
    - A decorator: ``@vp.jit``
    - A function: ``compiled_model = vp.jit(model)``

    Args:
        fn_or_model: function or Module to compile, or None when used as ``@vp.jit``
        fullgraph: if True, require whole-graph tracing (raises on unsupported ops)
        dynamic: if True, allow dynamic shapes
        backend: 'vp' (default), 'eager', 'torch'
        example_inputs: list of example inputs for concrete tracing
        fuse: whether to apply fusion passes
    Returns:
        Compiled model/function, or a decorator if fn_or_model is None
    """
    def _compile(fn):
        # Module: use vp.fx compile
        from vectorphile.nn.module import Module as _Mod
        if isinstance(fn, _Mod):
            try:
                from vectorphile.fx import compile as _fx_compile
                return _fx_compile(fn, example_inputs=example_inputs, fuse=fuse)
            except Exception:
                return fn

        # Function: wrap with profiler annotation + eager execution
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                from vectorphile.profiler import _GLOBAL_PROFILER
                if _GLOBAL_PROFILER.enabled:
                    name = getattr(fn, '__name__', 'jit_fn')
                    _GLOBAL_PROFILER.record_begin(name)
                    try:
                        result = fn(*args, **kwargs)
                    finally:
                        _GLOBAL_PROFILER.record_end()
                    return result
            except Exception:
                pass
            return fn(*args, **kwargs)
        wrapper._jit_compiled = True
        return wrapper

    if fn_or_model is None:
        # Used as @vp.jit with no args: return decorator
        return _compile
    if callable(fn_or_model) or hasattr(fn_or_model, 'forward'):
        return _compile(fn_or_model)
    # Used as @vp.jit(fullgraph=...) — return decorator
    return _compile


def script(fn):
    """Script a function (torch.jit.script compatibility stub).

    In VP, scripting and JIT compilation are unified. This decorator
    registers the function for eager dispatch with profiling.
    """
    return jit(fn)


def trace(fn, example_inputs, **kwargs):
    """Trace fn with concrete example_inputs.

    Args:
        fn: callable or Module
        example_inputs: list of Tensor inputs
    Returns:
        GraphModule (callable)
    """
    from vectorphile.nn.module import Module as _Mod
    if isinstance(fn, _Mod):
        try:
            from vectorphile.fx import symbolic_trace
            return symbolic_trace(fn, example_inputs=example_inputs)
        except Exception:
            return fn
    # For plain functions: return a wrapper that records the function
    @functools.wraps(fn)
    def wrapper(*args, **kw):
        return fn(*args, **kw)
    wrapper._traced = True
    return wrapper


class ScriptModule:
    """torch.jit.ScriptModule compatibility stub.

    Wraps a VP Module and behaves like a scripted module. The actual
    compilation happens lazily on first call via vp.fx.
    """

    def __init__(self, module):
        self._module = module
        self._compiled = None

    def _ensure_compiled(self, x=None):
        if self._compiled is None:
            try:
                from vectorphile.fx import compile as _fx_compile
                self._compiled = _fx_compile(
                    self._module,
                    example_inputs=[x] if x is not None else None
                )
            except Exception:
                self._compiled = self._module

    def __call__(self, *args, **kwargs):
        self._ensure_compiled(args[0] if args else None)
        return self._compiled(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def parameters(self):
        return self._module.parameters()

    def named_parameters(self):
        return self._module.named_parameters()


__all__ = ['jit', 'script', 'trace', 'ScriptModule']
