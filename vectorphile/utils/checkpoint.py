"""vectorphile.utils.checkpoint — Gradient checkpointing utilities.

Provides torch.utils.checkpoint-compatible API:
- ``checkpoint(fn, *args)``: recompute activations on backward
- ``checkpoint_sequential(modules, segments, x)``: segment-level checkpointing

These reduce peak memory by O(sqrt(N)) at the cost of one extra forward pass
per checkpoint segment, allowing training of much larger models.

Usage::
    from vectorphile.utils.checkpoint import checkpoint, checkpoint_sequential

    # Single function checkpoint
    y = checkpoint(my_layer, x)

    # Sequential module checkpointing (GPT-style)
    layers = [TransformerBlock(d_model=512) for _ in range(24)]
    y = checkpoint_sequential(layers, segments=6, x=x)
"""
from __future__ import annotations
from typing import Callable, List, Any


def checkpoint(function: Callable, *args, use_reentrant: bool = True, **kwargs):
    """Run function with gradient checkpointing.

    Forward pass stores no intermediate activations. During backward, the
    forward pass is recomputed to regenerate needed tensors.

    Args:
        function: callable to checkpoint
        *args: inputs to function
        use_reentrant: ignored (for torch compat); VP always uses non-reentrant path
        **kwargs: keyword args passed through to function

    Returns:
        Output of function(*args, **kwargs)
    """
    from vectorphile.autograd.functional import checkpoint as _cp
    if kwargs:
        def _fn_with_kwargs(*a):
            return function(*a, **kwargs)
        return _cp(_fn_with_kwargs, *args)
    return _cp(function, *args)


def checkpoint_sequential(modules: List, segments: int, input, **kwargs):
    """Apply gradient checkpointing across a list of modules.

    Splits the module list into `segments` groups. Within each group,
    activations are discarded and recomputed during backward.

    Args:
        modules: list of Module objects
        segments: number of checkpoint segments (more = less memory, more compute)
        input: input tensor to pass through modules
        **kwargs: unused keyword args (for torch compat)

    Returns:
        Output tensor after all modules
    """
    from vectorphile.autograd.functional import checkpoint as _cp
    if not modules:
        return input
    n = len(modules)
    seg_size = max(1, n // max(1, segments))
    segs = []
    for i in range(0, n, seg_size):
        segs.append(list(modules[i:i+seg_size]))

    x = input
    for seg in segs:
        def _make_fn(mods):
            def _fn(inp):
                for m in mods:
                    inp = m(inp)
                return inp
            return _fn
        x = _cp(_make_fn(seg), x)
    return x


__all__ = ['checkpoint', 'checkpoint_sequential']
