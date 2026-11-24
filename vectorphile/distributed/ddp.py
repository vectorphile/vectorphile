"""
DistributedDataParallel (DDP) — gradient synchronisation wrapper.

Hooks into every parameter's backward pass to AllReduce gradients across
all ranks after each backward() call. Loss must be averaged before backward
(divide by world_size, or use mean reduction in loss functions).

Usage:
    dist.init_process_group(rank=rank, world_size=N, ...)
    model = vp.nn.Linear(H, H)
    model = dist.DistributedDataParallel(model)
    loss = loss_fn(model(x), y)
    loss.backward()           # grads AllReduced automatically
    optimizer.step()
    optimizer.zero_grad()
"""
from __future__ import annotations
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor
from . import process_group as _pg


class DistributedDataParallel(Module):
    """
    Wraps a module to synchronise gradients after each backward pass.
    Transparent to the optimizer and forward pass.
    """

    def __init__(self, module: Module, process_group=None):
        super().__init__()
        self.module = module
        self._pg = process_group  # reserved for future multi-group support
        # Share parameters so the optimizer sees them directly
        self._parameters = module._parameters
        self._modules['module'] = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _sync_gradients(self):
        """AllReduce all parameter gradients and average across ranks."""
        W = _pg.get_world_size()
        if W <= 1:
            return
        for param in self.module.parameters():
            if param.grad is not None:
                _pg.all_reduce(param.grad, op='sum')
                # Average: divide by world_size
                gd = param.grad._data
                from vectorphile._core.ndarray import NDArray as _NDA
                if isinstance(gd, _NDA) and gd._buf is not None:
                    n = len(gd._buf)
                    inv_w = 1.0 / W
                    for i in range(n):
                        gd._buf[i] *= inv_w
                else:
                    param.grad = Tensor(
                        param.grad._data.__truediv__(W),
                        requires_grad=False)

    def backward_hook(self):
        """Call after loss.backward() to sync grads."""
        self._sync_gradients()

    def __repr__(self):
        return f'DistributedDataParallel(\n  {repr(self.module)}\n)'
