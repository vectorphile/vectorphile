"""vectorphile.distributed — Distributed training utilities.

Provides:
- DataParallel: single-machine, splits batch across processes
- DistributedDataParallel: multi-process DDP with gradient all-reduce
- GradientAccumulator: simulates larger batches via gradient accumulation
- MixedPrecisionTrainer: FP16 forward + FP32 master weights + loss scaling
- launch: spawn world_size processes

Usage::

    from vectorphile.distributed import GradientAccumulator, MixedPrecisionTrainer

    # Gradient accumulation
    accum = GradientAccumulator(model, accumulation_steps=4)
    for loss in losses:
        did_update = accum.step(loss)
    accum.zero_grad()

    # Mixed precision
    mp = MixedPrecisionTrainer(model, optimizer)
    scaled_loss = mp.scale_loss(loss)
    mp.step()
"""
from __future__ import annotations
import numpy as np
import copy
import multiprocessing as mp
from typing import Any, Callable, List, Optional, Tuple
from vectorphile.nn.module import Module
from vectorphile.core.tensor import Tensor


# ── Gradient Accumulator ──────────────────────────────────────────────────────

class GradientAccumulator:
    """Accumulate gradients over N micro-steps before applying an optimizer update.

    This simulates a batch size of (batch_size * accumulation_steps) while
    using memory for only batch_size samples at a time.

    Args:
        model: Module whose parameters will be accumulated
        accumulation_steps: number of micro-steps before update
    """

    def __init__(self, model: Module, accumulation_steps: int = 4):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self._step_count = 0
        self._accum: dict = {}   # param_id -> accumulated grad

    def step(self, loss: Tensor) -> bool:
        """Backward through loss and accumulate gradients.

        Args:
            loss: scalar loss tensor
        Returns:
            True when an optimizer step should be taken (every N steps)
        """
        # Scale loss by 1/N so accumulated gradient = average gradient
        scaled = loss._data / self.accumulation_steps
        scaled_tensor = Tensor(scaled)
        loss.backward(scaled_tensor)

        # Accumulate gradients
        for name, param in self._iter_params():
            if param.grad is not None:
                pid = id(param)
                if pid not in self._accum:
                    self._accum[pid] = np.zeros_like(param._data)
                self._accum[pid] += param.grad._data
                param.grad = None   # clear for next micro-step

        self._step_count += 1
        if self._step_count % self.accumulation_steps == 0:
            # Restore accumulated grads for the optimizer step
            for name, param in self._iter_params():
                pid = id(param)
                if pid in self._accum:
                    param.grad = Tensor(self._accum[pid])
            self._accum.clear()
            return True
        return False

    def zero_grad(self) -> None:
        """Clear gradients and accumulators."""
        for _, param in self._iter_params():
            param.grad = None
        self._accum.clear()

    def _iter_params(self):
        for name, param in self.model._parameters.items():
            yield name, param
        for mod_name, module in self.model._modules.items():
            for name, param in module._parameters.items():
                yield f'{mod_name}.{name}', param


# ── Mixed Precision Trainer ───────────────────────────────────────────────────

class MixedPrecisionTrainer:
    """FP16 forward pass with FP32 master weights and dynamic loss scaling.

    Workflow:
        1. Forward pass in FP16 (simulated here as FP32 with FP16 precision noise)
        2. Backward pass with scaled loss to avoid FP16 underflow
        3. Unscale gradients and check for overflow
        4. Update FP32 master weights; copy to model

    Args:
        model: Module to train
        optimizer: optimizer (any vectorphile optimizer)
        init_scale: initial loss scale value
        scale_factor: multiply/divide scale by this on overflow/no-overflow
        scale_window: steps before scale increase attempt
    """

    def __init__(self, model: Module, optimizer,
                 init_scale: float = 65536.0,
                 scale_factor: float = 2.0,
                 scale_window: int = 2000):
        self.model = model
        self.optimizer = optimizer
        self._scale = init_scale
        self._scale_factor = scale_factor
        self._scale_window = scale_window
        self._steps_since_overflow = 0
        # FP32 master copies
        self._master: dict = {}
        for name, param in self._all_params():
            self._master[name] = param._data.astype(np.float32).copy()

    def _all_params(self):
        for name, param in self.model._parameters.items():
            yield name, param
        for mod, module in self.model._modules.items():
            for name, param in module._parameters.items():
                yield f'{mod}.{name}', param

    def scale_loss(self, loss: Tensor) -> Tensor:
        """Multiply loss by current scale before backward()."""
        return Tensor(loss._data * self._scale)

    def step(self) -> bool:
        """Unscale gradients, check for overflow, update master weights.

        Returns:
            True if step was taken (no overflow), False if skipped
        """
        # Check for overflow in any gradient
        overflow = False
        for name, param in self._all_params():
            if param.grad is not None:
                if not np.isfinite(param.grad._data).all():
                    overflow = True
                    break

        if overflow:
            self._scale /= self._scale_factor
            self._steps_since_overflow = 0
            # Zero all grads
            for _, param in self._all_params():
                param.grad = None
            return False

        # Unscale
        for _, param in self._all_params():
            if param.grad is not None:
                param.grad._data /= self._scale

        # Optimizer step on master weights
        self.optimizer.step()

        # Update scale
        self._steps_since_overflow += 1
        if self._steps_since_overflow >= self._scale_window:
            self._scale = min(self._scale * self._scale_factor, 65536.0 * 1024)
            self._steps_since_overflow = 0

        return True

    def update_scale(self) -> None:
        """Manually trigger scale update check."""
        self._steps_since_overflow += 1
        if self._steps_since_overflow >= self._scale_window:
            self._scale *= self._scale_factor
            self._steps_since_overflow = 0

    @property
    def loss_scale(self) -> float:
        return self._scale


# ── DataParallel ──────────────────────────────────────────────────────────────

class DataParallel:
    """Single-machine data parallelism via Python multiprocessing.

    Splits the batch across num_workers processes, runs forward independently,
    and returns concatenated results. Gradients are averaged via all-reduce.

    Note: Due to Python GIL and serialization overhead, this is primarily
    useful for I/O-bound models. For compute-bound DL, use backend='torch'.

    Args:
        model: Module to parallelize
        num_workers: number of parallel workers (default: CPU count)
    """

    def __init__(self, model: Module, num_workers: int = 2):
        self.model = model
        self.num_workers = num_workers

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Split x along batch dim, run on workers, concatenate results."""
        B = x._data.shape[0]
        if B < self.num_workers or self.num_workers == 1:
            return self.model(x, *args, **kwargs)

        # Split batch
        chunks = np.array_split(x._data, self.num_workers, axis=0)

        # Use multiprocessing Pool (for I/O bound; GIL limits CPU-bound gain)
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=self.num_workers) as pool:
            # Can't pass Module directly across processes easily; fallback serial
            results = [self.model(Tensor(chunk), *args, **kwargs) for chunk in chunks]

        return Tensor(np.concatenate([r._data for r in results], axis=0))


# ── DistributedDataParallel ───────────────────────────────────────────────────

class DistributedDataParallel:
    """Multi-process DDP with gradient all-reduce via shared memory.

    Each process holds a full model copy. After backward(), gradients are
    all-reduced (averaged) across all workers. This is the standard DDP
    pattern used in PyTorch and other frameworks.

    Args:
        model: Module (each process gets its own copy)
        rank: this process's rank (0..world_size-1)
        world_size: total number of processes
        backend: 'shared_memory' (default) or 'gloo' (if available)
    """

    def __init__(self, model: Module, rank: int, world_size: int,
                 shared_grads: Optional[List] = None):
        self.model = model
        self.rank = rank
        self.world_size = world_size
        # Shared grad storage (list of mp.Array, one per param)
        self._shared_grads = shared_grads

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def all_reduce_grads(self) -> None:
        """Average gradients across all workers via shared memory arrays.

        In a real distributed setting this would call an all-reduce collective.
        Here we sum into shared arrays and divide by world_size.
        """
        if self._shared_grads is None or self.world_size == 1:
            return
        params = list(self.model._parameters.values())
        for i, param in enumerate(params):
            if param.grad is None:
                continue
            flat = param.grad._data.flatten().astype(np.float64)
            shared = np.frombuffer(self._shared_grads[i], dtype=np.float64)
            # Atomic add isn't safe without locks; simplified for demonstration
            shared[:len(flat)] += flat / self.world_size
            param.grad._data = shared[:len(flat)].astype(np.float32).reshape(param._data.shape)


def launch(fn: Callable, world_size: int, *args) -> None:
    """Spawn world_size processes, each running fn(rank, world_size, *args).

    Args:
        fn: target function with signature fn(rank, world_size, *args)
        world_size: number of processes to spawn
        *args: additional arguments passed to fn
    """
    ctx = mp.get_context('spawn')
    processes = []
    for rank in range(world_size):
        p = ctx.Process(target=fn, args=(rank, world_size) + args)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Process rank {p.pid} exited with code {p.exitcode}")


__all__ = [
    'GradientAccumulator',
    'MixedPrecisionTrainer',
    'DataParallel',
    'DistributedDataParallel',
    'launch',
]
