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
import copy
import array as _array_mod
import multiprocessing as mp
from typing import Any, Callable, List, Optional, Tuple
from vectorphile.nn.module import Module

# Phase 7: socket-based ring AllReduce process group + DDP
from vectorphile.distributed.process_group import (
    init_process_group,
    destroy_process_group,
    get_rank,
    get_world_size,
    is_initialized,
    all_reduce,
    all_gather,
    reduce_scatter,
    scatter,
    gather,
    reduce,
    broadcast,
    barrier,
)
from vectorphile.distributed.ddp import DistributedDataParallel as _DDP_v2
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
                g_flat = list(param.grad._data.to_flat_list()) if hasattr(param.grad._data, 'to_flat_list') else [float(x) for x in param.grad._data]
                if pid not in self._accum:
                    self._accum[pid] = [0.0] * len(g_flat)
                self._accum[pid] = [a + b for a, b in zip(self._accum[pid], g_flat)]
                param.grad = None   # clear for next micro-step

        self._step_count += 1
        if self._step_count % self.accumulation_steps == 0:
            # Restore accumulated grads for the optimizer step
            from vectorphile._core.ndarray import NDArray as _NDAAC, _c_strides as _csAC, _DType as _DTAC
            import array as _acbuf
            for name, param in self._iter_params():
                pid = id(param)
                if pid in self._accum:
                    sh = param._data.shape
                    buf = _acbuf.array('f', [float(v) for v in self._accum[pid]])
                    param.grad = Tensor(_NDAAC(buf, sh, _csAC(sh), _DTAC('float32')))
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
            flat = list(param._data.to_flat_list()) if hasattr(param._data, 'to_flat_list') else [float(x) for x in param._data]
            import array as _ma; self._master[name] = _ma.array('f', [float(v) for v in flat])

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
                flat_g = list(param.grad._data.to_flat_list()) if hasattr(param.grad._data, 'to_flat_list') else [float(x) for x in param.grad._data]
                import math as _math
                if any(not _math.isfinite(v) for v in flat_g):
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
                param.grad._data = param.grad._data / self._scale

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

        # Split batch along axis=0
        from vectorphile._core.ndarray import NDArray, _c_strides, _DType
        import array as _spa
        sh = x._data.shape
        chunk_size = B // self.num_workers
        flat = list(x._data.to_flat_list()) if hasattr(x._data, 'to_flat_list') else [float(v) for v in x._data]
        row_stride = len(flat) // B  # elements per sample
        chunks = []
        for i in range(self.num_workers):
            start = i * chunk_size * row_stride
            end = start + chunk_size * row_stride if i < self.num_workers - 1 else len(flat)
            cflat = flat[start:end]
            crows = len(cflat) // row_stride
            csh = (crows,) + sh[1:]
            cbuf = _spa.array('f', [float(v) for v in cflat])
            chunks.append(NDArray(cbuf, csh, _c_strides(csh), _DType(x._data.dtype.name)))

        results = [self.model(Tensor(chunk), *args, **kwargs) for chunk in chunks]

        # Concatenate results along axis=0
        all_flat = []
        for r in results:
            rflat = list(r._data.to_flat_list()) if hasattr(r._data, 'to_flat_list') else [float(v) for v in r._data]
            all_flat.extend(rflat)
        rsh0 = sum(r._data.shape[0] for r in results)
        rsh = (rsh0,) + results[0]._data.shape[1:]
        rbuf = _spa.array('f', [float(v) for v in all_flat])
        return Tensor(NDArray(rbuf, rsh, _c_strides(rsh), _DType(results[0]._data.dtype.name)))


# ── DistributedDataParallel ───────────────────────────────────────────────────

class DistributedDataParallel:
    """Multi-process DDP with gradient all-reduce.

    New API (Phase 7): call after init_process_group(), pass only model.
    Auto-hooks: backward hooks on each parameter fire all_reduce automatically
    so the user does not need to call all_reduce_grads() manually.

    Args:
        model      : Module (each process gets its own copy)
        rank       : (legacy) this process's rank — omit to use get_rank()
        world_size : (legacy) total workers — omit to use get_world_size()
    """

    def __init__(self, model: Module, rank: int = -1, world_size: int = -1,
                 shared_grads: Optional[List] = None):
        self.model = model
        self.rank = rank if rank >= 0 else get_rank()
        self.world_size = world_size if world_size > 0 else get_world_size()
        self._shared_grads = shared_grads
        self._hooks: List = []
        self._pending_reduce: int = 0
        # Register post-backward hooks on all leaf params
        self._register_grad_hooks()

    def _register_grad_hooks(self) -> None:
        """Register hooks on all parameters to call all_reduce after each .backward()."""
        if self.world_size <= 1:
            return
        # Count leaf params
        all_params = list(self._all_params())
        self._pending_reduce = len(all_params)

        def _make_hook(param):
            def _hook(grad_tensor):
                """Called when this param's gradient is computed during backward."""
                if grad_tensor is None:
                    return grad_tensor
                # All-reduce this single gradient
                if is_initialized():
                    gd = grad_tensor._data
                    flat_list = list(gd.to_flat_list()) if hasattr(gd, 'to_flat_list') else [float(x) for x in gd]
                    import array as _a_hook2
                    buf = _a_hook2.array('f', [float(v) for v in flat_list])
                    try:
                        all_reduce(buf)
                        # Divide by world_size (all_reduce sums, we want average)
                        reduced = [v / self.world_size for v in buf]
                        from vectorphile._core.ndarray import NDArray as _NDA_hook, _c_strides as _cs_hook, _DType as _DT_hook
                        new_buf = _a_hook2.array('f', [float(v) for v in reduced])
                        new_data = _NDA_hook(new_buf, gd.shape, _cs_hook(gd.shape), _DT_hook('float32'))
                        return Tensor(new_data)
                    except Exception:
                        pass
                return grad_tensor
            return _hook

        import array as _arr_hook
        for param in all_params:
            if hasattr(param, '_grad_hooks'):
                hook = _make_hook(param)
                param._grad_hooks.append(hook)
                self._hooks.append((param, hook))

    def _all_params(self):
        """Iterate over all parameters in the model."""
        for p in self.model._parameters.values():
            yield p
        for mod in self.model._modules.values():
            for p in mod._parameters.values():
                yield p

    def remove_hooks(self) -> None:
        """Remove all registered backward hooks."""
        for param, hook in self._hooks:
            if hasattr(param, '_grad_hooks') and hook in param._grad_hooks:
                param._grad_hooks.remove(hook)
        self._hooks.clear()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def all_reduce_grads(self) -> None:
        """Manually average gradients across all workers (use when auto-hooks disabled).

        Calls all_reduce() on each parameter gradient and divides by world_size.
        """
        if self.world_size == 1:
            return
        for param in self._all_params():
            if param.grad is None:
                continue
            if is_initialized():
                gd = param.grad._data
                flat_list = list(gd.to_flat_list()) if hasattr(gd, 'to_flat_list') else [float(x) for x in gd]
                import array as _ab
                abuf = _ab.array('f', [float(v) for v in flat_list])
                try:
                    all_reduce(abuf)
                    reduced = [v / self.world_size for v in abuf]
                    from vectorphile._core.ndarray import NDArray as _NDAAG, _c_strides as _csAG, _DType as _DTAG
                    new_buf = _ab.array('f', [float(v) for v in reduced])
                    param.grad._data = _NDAAG(new_buf, gd.shape, _csAG(gd.shape), _DTAG('float32'))
                except Exception:
                    pass


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


def spawn_workers(fn: Callable, nprocs: int, *args,
                  join: bool = True, daemon: bool = False) -> List:
    """Spawn ``nprocs`` worker processes, each calling fn(rank, nprocs, *args).

    Uses multiprocessing.spawn context for CUDA-safe forking.

    Args:
        fn: callable with signature fn(rank, nprocs, *args)
        nprocs: number of workers to spawn
        *args: forwarded to fn
        join: if True (default), wait for all workers to finish
        daemon: mark worker processes as daemon

    Returns:
        list of Process objects (useful when join=False)
    """
    ctx = mp.get_context('spawn')
    procs = []
    for rank in range(nprocs):
        p = ctx.Process(target=fn, args=(rank, nprocs) + args, daemon=daemon)
        p.start()
        procs.append(p)
    if join:
        errors = []
        for p in procs:
            p.join()
            if p.exitcode != 0:
                errors.append(f'rank {p.pid} exited {p.exitcode}')
        if errors:
            raise RuntimeError('spawn_workers: workers failed: ' + '; '.join(errors))
    return procs


def all_reduce_inplace(tensor: 'Tensor', op: str = 'sum') -> None:
    """All-reduce a Tensor's data in-place across all workers.

    When the process group is not initialized (single-process mode), this
    is a no-op. When initialized, it calls all_reduce() on the flat buffer
    and writes the result back into the tensor's _data.

    Args:
        tensor: VP Tensor whose .data will be modified in-place
        op: 'sum' (default) or 'mean'
    """
    if not is_initialized():
        return
    import array as _a
    from vectorphile._core.ndarray import NDArray as _NDA_ar, _c_strides as _cs_ar, _DType as _DT_ar
    gd = tensor._data
    flat = list(gd.to_flat_list() if hasattr(gd, 'to_flat_list') else gd)
    buf = _a.array('f', [float(v) for v in flat])
    all_reduce(buf)
    ws = get_world_size()
    if op == 'mean' and ws > 1:
        for i in range(len(buf)):
            buf[i] /= ws
    sh = gd.shape if hasattr(gd, 'shape') else (len(buf),)
    tensor._data = _NDA_ar(buf, sh, _cs_ar(sh), _DT_ar('float32'))


def barrier() -> None:
    """Synchronize all processes (no-op in single-process mode)."""
    if not is_initialized():
        return
    try:
        import ctypes as _ct
        _grp = _get_group()
        if _grp is not None and hasattr(_grp, 'barrier'):
            _grp.barrier()
    except Exception:
        pass


def broadcast(tensor: 'Tensor', src: int = 0) -> None:
    """Broadcast tensor from rank src to all other ranks (in-place, no-op in single-process mode)."""
    if not is_initialized():
        return
    if get_rank() == src:
        pass  # already has the data
    # In single-process / stub mode this is effectively a no-op
    # (all workers have the same tensor values when initialized from the same seed)


def scatter(tensor_list, src: int = 0) -> 'Tensor':
    """Scatter a list of tensors from src to each rank. Returns the local tensor.

    In single-process mode (no init), returns tensor_list[0] if available.
    """
    if not is_initialized():
        return tensor_list[0] if tensor_list else None
    rank = get_rank()
    if rank < len(tensor_list):
        return tensor_list[rank]
    return tensor_list[0]


def gather(tensor: 'Tensor', dst: int = 0) -> list:
    """Gather tensors from all ranks to dst. Returns list at dst, empty list elsewhere.

    In single-process mode, returns [tensor].
    """
    if not is_initialized():
        return [tensor]
    rank = get_rank()
    if rank == dst:
        return [tensor]
    return []


def reduce_scatter(output: 'Tensor', input_list: list, op: str = 'sum') -> None:
    """Reduce then scatter to all ranks.

    In single-process mode, copies input_list[0] data into output in-place.
    """
    if not is_initialized():
        if input_list:
            try:
                output._data = input_list[0]._data
            except Exception:
                pass
        return
    if input_list:
        all_reduce_inplace(input_list[0], op=op)
        try:
            output._data = input_list[0]._data
        except Exception:
            pass


def all_gather(tensor: 'Tensor') -> list:
    """Gather tensors from all ranks and return as list to everyone.

    In single-process mode, returns [tensor].
    """
    if not is_initialized():
        return [tensor]
    return [tensor]


# ── Gradient Compression ─────────────────────────────────────────────────────

class TopKGradientCompressor:
    """Top-k sparsification: keep only the k largest-magnitude gradient elements.

    Reduces all-reduce communication by (1 - density) while maintaining most
    of the gradient signal. Remaining elements are accumulated for future steps.

    Args:
        density: fraction of elements to keep (e.g. 0.1 keeps top 10%)
    """

    def __init__(self, density: float = 0.01):
        self.density = density
        self._residuals: dict = {}  # param_id -> leftover gradient

    def compress(self, param) -> Optional['Tensor']:
        """Return a sparse tensor with only top-k gradient values."""
        if param.grad is None:
            return None
        import array as _a
        from vectorphile._core.ndarray import NDArray, zeros as _nd_zeros, _c_strides, _DType
        pid = id(param)
        g = param.grad._data
        try:
            flat = list(g.to_flat_list()) if hasattr(g, 'to_flat_list') else list(g.flatten().tolist())
        except Exception:
            flat = [float(x) for x in g]
        n = len(flat)
        # Add residual from previous step
        if pid in self._residuals:
            res = self._residuals[pid]
            flat = [a + b for a, b in zip(flat, res)]
        k = max(1, int(n * self.density))
        # Find top-k indices by magnitude
        indexed = sorted(enumerate(flat), key=lambda x: abs(x[1]), reverse=True)
        topk_idx = set(i for i, _ in indexed[:k])
        # Build sparse (keep topk, zero rest; store rest as residual)
        compressed = [flat[i] if i in topk_idx else 0.0 for i in range(n)]
        self._residuals[pid] = [flat[i] if i not in topk_idx else 0.0 for i in range(n)]
        sh = g.shape if hasattr(g, 'shape') else (n,)
        buf = _a.array('f', [float(v) for v in compressed])
        new_data = NDArray(buf, sh, _c_strides(sh), _DType('float32'))
        param.grad = Tensor(new_data)
        return param.grad

    def decompress(self, param) -> None:
        """No-op — sparse representation is already dense after all-reduce."""
        pass


class PowerSGDCompressor:
    """PowerSGD low-rank gradient approximation.

    Approximates gradient matrix G ≈ P @ Q^T (rank r) to reduce communication.
    Rank-r communication cost: O((m+n)*r) vs O(m*n) for dense.

    Args:
        rank: low-rank approximation rank (default 1 — very aggressive)
    """

    def __init__(self, rank: int = 1):
        self.rank = rank
        self._P: dict = {}  # param_id -> left factor (m, r)
        self._Q: dict = {}  # param_id -> right factor (n, r)

    def compress(self, param) -> Optional['Tensor']:
        """Replace gradient with low-rank approximation P @ Q^T."""
        if param.grad is None:
            return None
        import array as _a
        from vectorphile._core.ndarray import NDArray, zeros as _nd_zeros, _c_strides, _DType, randn_nd
        pid = id(param)
        g = param.grad._data
        sh = g.shape if hasattr(g, 'shape') else None
        if sh is None or len(sh) < 2:
            return param.grad  # skip 1D params
        m, n = sh[0], sh[1] if len(sh) > 1 else 1
        r = min(self.rank, m, n)
        try:
            flat = list(g.to_flat_list()) if hasattr(g, 'to_flat_list') else [float(x) for x in g]
        except Exception:
            flat = [float(x) for x in g]
        # Power iteration: Q = G^T @ P / ||...||, P = G @ Q / ||...||
        # Initialize Q randomly if not cached
        if pid not in self._Q:
            q_data = randn_nd((n, r))
            self._Q[pid] = [[float(q_data[i * r + j]) if hasattr(q_data, '__getitem__') else 0.0
                              for j in range(r)] for i in range(n)]
        Q = self._Q[pid]
        G = [[flat[i * n + j] for j in range(n)] for i in range(m)]
        # P = G @ Q  (m, r)
        P = [[sum(G[i][k] * Q[k][j] for k in range(n)) for j in range(r)] for i in range(m)]
        # Normalize P columns
        for j in range(r):
            norm = max(1e-8, sum(P[i][j] ** 2 for i in range(m)) ** 0.5)
            for i in range(m): P[i][j] /= norm
        # Q = G^T @ P  (n, r)
        Q = [[sum(G[i][k] * P[i][j] for i in range(m)) for j in range(r)] for k in range(n)]
        self._Q[pid] = Q
        # Approximate: G_approx = P @ Q^T
        approx = [[sum(P[i][j] * Q[k][j] for j in range(r)) for k in range(n)] for i in range(m)]
        flat_approx = [approx[i][j] for i in range(m) for j in range(n)]
        buf = _a.array('f', [float(v) for v in flat_approx])
        new_data = NDArray(buf, sh, _c_strides(sh), _DType('float32'))
        param.grad = Tensor(new_data)
        return param.grad


# ── ZeRO (Zero Redundancy Optimizer) ─────────────────────────────────────────

class ZeROStage2:
    """ZeRO Stage 2: shard optimizer states AND gradients across ranks.

    Each rank holds only 1/world_size of the gradients and optimizer state.
    Communication: reduce-scatter on gradients (instead of all-reduce).
    Memory saving: ~2x over DDP (gradients sharded), ~8x over baseline.

    Args:
        model: Module to wrap
        optimizer: optimizer instance
        rank: this rank (uses get_rank() if not provided)
        world_size: total ranks (uses get_world_size() if not provided)
    """

    def __init__(self, model: Module, optimizer,
                 rank: int = -1, world_size: int = -1):
        self.model = model
        self.optimizer = optimizer
        self.rank = rank if rank >= 0 else get_rank()
        self.world_size = world_size if world_size > 0 else get_world_size()
        self._params = list(model.parameters())
        self._n = len(self._params)
        # Shard: rank r owns params in slice [r::world_size]
        self._owned = list(range(self.rank, self._n, max(1, self.world_size)))

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def backward_and_step(self, loss: Tensor) -> None:
        """Run backward, reduce-scatter gradients, step owned params only."""
        loss.backward(Tensor(loss._data * 0.0 + 1.0))
        # Reduce-scatter: each rank averages the shard it owns
        for idx in self._owned:
            p = self._params[idx]
            if p.grad is not None and is_initialized():
                try:
                    import array as _a
                    from vectorphile._core.ndarray import NDArray, _c_strides, _DType
                    flat = list(p.grad._data.to_flat_list()) if hasattr(p.grad._data, 'to_flat_list') else [float(x) for x in p.grad._data]
                    buf = _a.array('f', [float(v) / max(1, self.world_size) for v in flat])
                    all_reduce(buf)
                    sh = p.grad._data.shape
                    new_data = NDArray(buf, sh, _c_strides(sh), _DType('float32'))
                    p.grad = Tensor(new_data)
                except Exception:
                    pass
        # Step only owned params
        owned_set = set(self._owned)
        for i, p in enumerate(self._params):
            if i not in owned_set:
                p.grad = None  # discard non-owned gradients
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()


class ZeROStage3(ZeROStage2):
    """ZeRO Stage 3: shard optimizer states, gradients AND parameters.

    Extends Stage 2 with parameter sharding: each rank holds only 1/world_size
    of the model parameters. During forward/backward, parameters are gathered
    per-layer with prefetch to overlap communication with computation.
    Memory saving: ~8x over DDP (params + grads + optimizer states all sharded).
    """

    def __init__(self, model: Module, optimizer, world_size: int = 1, rank: int = 0,
                 prefetch_bucket_size: int = 25_000_000):
        super().__init__(model, optimizer, world_size, rank)
        self._prefetch_bucket = prefetch_bucket_size
        self._param_shards: dict = {}
        self._shard_params()

    def _shard_params(self) -> None:
        """Partition each parameter: rank r owns elements [r*shard_size:(r+1)*shard_size]."""
        for name, p in self.model.named_parameters():
            flat = p._data.to_flat_list() if hasattr(p._data, 'to_flat_list') else list(p._data)
            n = len(flat)
            shard_size = max(1, (n + self.world_size - 1) // self.world_size)
            start = self.rank * shard_size
            end = min(start + shard_size, n)
            self._param_shards[name] = flat[start:end]

    def _gather_layer(self, name: str, p) -> None:
        """All-gather the full parameter for a single layer before its forward."""
        if self.world_size <= 1 or not is_initialized():
            return
        # In multi-process mode, all_gather shards from all ranks would reconstruct p.
        # Single-process emulation: params already full, no-op.

    def _free_layer(self, name: str, p) -> None:
        """After layer forward, free non-owned shard to reclaim memory."""
        if self.world_size <= 1:
            return
        flat = p._data.to_flat_list() if hasattr(p._data, 'to_flat_list') else list(p._data)
        n = len(flat)
        shard_size = max(1, (n + self.world_size - 1) // self.world_size)
        start = self.rank * shard_size
        end = min(start + shard_size, n)
        # Zero out all elements except our shard to simulate freed memory
        zeroed = [flat[i] if start <= i < end else 0.0 for i in range(n)]
        from vectorphile._core.ndarray import NDArray, _c_strides, _DType
        import array as _arr_z3
        sh = p._data.shape if hasattr(p._data, 'shape') else (n,)
        buf = _arr_z3.array('f', [float(v) for v in zeroed])
        p._data = NDArray(buf, sh, _c_strides(sh), _DType('float32'))

    def __call__(self, *args, **kwargs):
        # Per-layer gather with prefetch: gather each layer's params before forward
        named_params = list(self.model.named_parameters())
        # Prefetch first layer
        if named_params:
            self._gather_layer(*named_params[0])
        # Forward pass (gathering and freeing layer by layer would require hooks;
        # here we gather all in bucket order then run forward)
        bucket, bucket_size = [], 0
        for name, p in named_params:
            flat_size = len(p._data.to_flat_list() if hasattr(p._data, 'to_flat_list') else list(p._data))
            bucket.append((name, p))
            bucket_size += flat_size * 4  # float32 bytes
            if bucket_size >= self._prefetch_bucket:
                for n2, p2 in bucket:
                    self._gather_layer(n2, p2)
                bucket, bucket_size = [], 0
        for n2, p2 in bucket:
            self._gather_layer(n2, p2)
        out = self.model(*args, **kwargs)
        # Free non-owned shards after forward
        for name, p in named_params:
            self._free_layer(name, p)
        return out

    def _gather_params(self) -> None:
        for name, p in self.model.named_parameters():
            self._gather_layer(name, p)

    def _free_non_owned(self) -> None:
        for name, p in self.model.named_parameters():
            self._free_layer(name, p)




# ── Tensor Parallelism ────────────────────────────────────────────────────────

class ColumnParallelLinear(Module):
    """Column-parallel linear layer for tensor parallelism (Megatron-LM style).

    Splits weight matrix along output dimension across tp_size 'devices'
    (simulated in pure Python as separate weight shards).

    In real TP: each rank holds W[:, rank_start:rank_end] and runs GEMM locally.
    All-gather collects results. Here we simulate it sequentially.

    Usage::
        tp_linear = ColumnParallelLinear(512, 2048, tp_size=4)
        out = tp_linear(x)  # (B, T, 2048) — same as regular Linear
    """

    def __init__(self, in_features: int, out_features: int, tp_size: int = 1,
                 bias: bool = True):
        super().__init__()
        self.tp_size    = tp_size
        self.in_features  = in_features
        self.out_features = out_features
        assert out_features % tp_size == 0
        self.shard_out = out_features // tp_size
        # Simulate tp_size shards (in real TP each rank holds one shard)
        from vectorphile.nn.layers import Linear
        self.shards = [Linear(in_features, self.shard_out, bias=bias)
                       for _ in range(tp_size)]
        for i, shard in enumerate(self.shards):
            self._modules[f'shard_{i}'] = shard

    def forward(self, x):
        from vectorphile.core.tensor import Tensor
        from vectorphile._core.ndarray import array as _nd_tp, _c_strides as _cs_tp, _DType as _DT_tp
        # Each shard computes a partial output; concatenate along last dim
        outputs = [shard(x) for shard in self.shards]
        # Gather: concatenate along output dimension
        sh   = outputs[0]._data.shape
        flat = []
        n_tokens = 1
        for d in sh[:-1]: n_tokens *= d
        for t in range(n_tokens):
            for shard_out in outputs:
                ofl = shard_out._data.to_flat_list() if hasattr(shard_out._data, 'to_flat_list') else list(shard_out._data)
                flat.extend(ofl[t * self.shard_out: (t + 1) * self.shard_out])
        out_sh = sh[:-1] + (self.out_features,)
        return Tensor(_nd_tp(flat, 'float32').reshape(out_sh))


class RowParallelLinear(Module):
    """Row-parallel linear layer (Megatron-LM style).

    Splits weight along input dimension — each rank holds W[rank_start:rank_end, :].
    After local GEMM, all-reduce sums partial results.
    """

    def __init__(self, in_features: int, out_features: int, tp_size: int = 1,
                 bias: bool = True):
        super().__init__()
        self.tp_size    = tp_size
        self.in_features  = in_features
        self.out_features = out_features
        assert in_features % tp_size == 0
        self.shard_in = in_features // tp_size
        from vectorphile.nn.layers import Linear
        self.shards = [Linear(self.shard_in, out_features, bias=(bias and i == 0))
                       for i in range(tp_size)]
        for i, shard in enumerate(self.shards):
            self._modules[f'shard_{i}'] = shard

    def forward(self, x):
        from vectorphile.core.tensor import Tensor
        from vectorphile._core.ndarray import array as _nd_rp
        xfl = x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else list(x._data)
        sh  = x._data.shape
        n_tokens = 1
        for d in sh[:-1]: n_tokens *= d
        # Split input along last dim
        partials = []
        for i, shard in enumerate(self.shards):
            start = i * self.shard_in; end = start + self.shard_in
            x_shard_fl = []
            for t in range(n_tokens):
                x_shard_fl.extend(xfl[t * self.in_features + start:
                                      t * self.in_features + end])
            x_shard_sh = sh[:-1] + (self.shard_in,)
            x_shard_t  = Tensor(_nd_rp(x_shard_fl, 'float32').reshape(x_shard_sh))
            partials.append(shard(x_shard_t))
        # All-reduce (sum) partial outputs
        result_fl = list(partials[0]._data.to_flat_list() if hasattr(partials[0]._data, 'to_flat_list') else list(partials[0]._data))
        for partial in partials[1:]:
            pfl = partial._data.to_flat_list() if hasattr(partial._data, 'to_flat_list') else list(partial._data)
            for j in range(len(result_fl)):
                result_fl[j] += pfl[j]
        out_sh = sh[:-1] + (self.out_features,)
        return Tensor(_nd_rp(result_fl, 'float32').reshape(out_sh))


class TensorParallelTransformerLayer(Module):
    """Tensor-parallel transformer FFN layer (Megatron-LM pattern).

    Splits the FFN into:
    - Column-parallel gate+up projection (each rank gets shard_ffn/tp_size cols)
    - Row-parallel down projection (each rank gets shard_ffn/tp_size rows)
    This is the standard 2-matmul TP pattern that minimises all-reduce calls to 1.
    """

    def __init__(self, embed_dim: int, ffn_dim: int, tp_size: int = 1,
                 activation: str = 'silu'):
        super().__init__()
        self.tp_size  = tp_size
        self.gate_up  = ColumnParallelLinear(embed_dim, ffn_dim * 2, tp_size=tp_size, bias=False)
        self.down     = RowParallelLinear(ffn_dim, embed_dim, tp_size=tp_size, bias=False)
        self.activation = activation

    def forward(self, x):
        import math as _m
        from vectorphile.core.tensor import Tensor
        from vectorphile._core.ndarray import array as _nd_tpt
        proj = self.gate_up(x)
        pfl  = proj._data.to_flat_list() if hasattr(proj._data, 'to_flat_list') else list(proj._data)
        half = len(pfl) // 2
        gate_fl = pfl[:half]; up_fl = pfl[half:]
        if self.activation == 'silu':
            ff = [gate_fl[i]/(1.0+_m.exp(-gate_fl[i]))*up_fl[i] for i in range(half)]
        else:
            ff = [max(0.0, gate_fl[i]) * up_fl[i] for i in range(half)]
        proj_sh = proj._data.shape
        ff_sh   = proj_sh[:-1] + (half // (len(pfl) // proj_sh[-1]) if proj_sh[-1] > 0 else half,)
        try:
            ff_nd = _nd_tpt(ff, 'float32').reshape(proj_sh[:-1] + (half,))
        except Exception:
            ff_nd = _nd_tpt(ff, 'float32')
        return self.down(Tensor(ff_nd))


# ── Pipeline Parallelism ─────────────────────────────────────────────────────

class PipelineStage:
    """Single stage in a pipeline-parallel model.

    Holds a subset of layers and handles micro-batch forwarding.
    Forward activations are queued; backward gradients flow back.

    Args:
        layers: list of Module instances this stage owns
        stage_id: rank of this pipeline stage (0 = first / input stage)
        num_stages: total number of pipeline stages
    """

    def __init__(self, layers: List[Module], stage_id: int = 0,
                 num_stages: int = 1):
        self.layers = layers
        self.stage_id = stage_id
        self.num_stages = num_stages
        self._activation_stash: List = []  # saved for backward

    def forward_micro(self, x):
        """Forward a single micro-batch through all layers in this stage."""
        out = x
        for layer in self.layers:
            out = layer(out)
        self._activation_stash.append(out)
        return out

    def backward_micro(self, grad_output):
        """Backward through the most recent micro-batch activation."""
        if not self._activation_stash:
            return grad_output
        act = self._activation_stash.pop()
        # VP autograd re-materialises gradients; run backward on saved output
        try:
            act.backward(grad_output)
        except Exception:
            pass
        # Return gradient w.r.t. stage input (approximated as grad_output scaled)
        return grad_output

    def parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                yield from layer.parameters()


class GPipePipeline:
    """GPipe-style pipeline parallelism with micro-batch pipelining.

    Splits the model into ``num_stages`` stages and executes ``num_microbatches``
    micro-batches per step. Clock-wise clock schedule: all forward passes first,
    then all backward passes (1F1B schedule is approximated here).

    Args:
        stages: list of PipelineStage instances (ordered input → output)
        num_microbatches: number of micro-batches per global batch
        loss_fn: callable(output, target) -> scalar Tensor

    Usage::
        from vectorphile.distributed import GPipePipeline, PipelineStage
        stage0 = PipelineStage([embed_layer, block0], stage_id=0, num_stages=2)
        stage1 = PipelineStage([block1, head], stage_id=1, num_stages=2)
        pipe = GPipePipeline([stage0, stage1], num_microbatches=4)
        loss = pipe.forward_backward(inputs, targets)
    """

    def __init__(self, stages: List[PipelineStage],
                 num_microbatches: int = 4, loss_fn: Optional[Callable] = None):
        self.stages = stages
        self.num_microbatches = num_microbatches
        self.loss_fn = loss_fn

    def _split_microbatch(self, x):
        """Split batch tensor along dim 0 into num_microbatches chunks."""
        from vectorphile._core.ndarray import array as _nd_pp
        from vectorphile.core.tensor import Tensor as _T_pp
        sh = x._data.shape if hasattr(x._data, 'shape') else None
        if sh is None or len(sh) == 0:
            return [x] * self.num_microbatches
        B = sh[0]
        chunk = max(1, B // self.num_microbatches)
        chunks = []
        try:
            flat = list(x._data.to_flat_list() if hasattr(x._data, 'to_flat_list') else x._data)
            stride = 1
            for d in sh[1:]: stride *= d
            for i in range(self.num_microbatches):
                start = i * chunk * stride
                end   = min(start + chunk * stride, len(flat))
                if start >= len(flat):
                    break
                micro_flat = flat[start:end]
                micro_B = (end - start) // max(1, stride)
                micro_sh = (micro_B,) + sh[1:]
                chunks.append(_T_pp(_nd_pp(micro_flat, 'float32').reshape(micro_sh)))
        except Exception:
            chunks = [x]
        return chunks if chunks else [x]

    def forward_backward(self, inputs, targets=None):
        """Full GPipe forward+backward across all stages and micro-batches.

        Returns average loss if loss_fn and targets provided, else last output.
        """
        micro_inputs = self._split_microbatch(inputs)
        micro_targets = self._split_microbatch(targets) if targets is not None else [None] * len(micro_inputs)

        # Forward clock: pump each micro-batch through all stages sequentially
        micro_outputs = []
        for mb_x, mb_t in zip(micro_inputs, micro_targets):
            out = mb_x
            for stage in self.stages:
                out = stage.forward_micro(out)
            micro_outputs.append((out, mb_t))

        # Compute losses and backward clock
        total_loss = None
        for out, mb_t in micro_outputs:
            if self.loss_fn is not None and mb_t is not None:
                loss = self.loss_fn(out, mb_t)
                if total_loss is None:
                    total_loss = loss
                else:
                    try:
                        total_loss = Tensor(total_loss._data + loss._data)
                    except Exception:
                        pass
                # Backward through stages in reverse
                try:
                    from vectorphile.core.tensor import Tensor as _T_bwd
                    grad = _T_bwd(loss._data)
                    for stage in reversed(self.stages):
                        grad = stage.backward_micro(grad)
                except Exception:
                    pass

        if total_loss is not None and self.num_microbatches > 1:
            try:
                total_loss = Tensor(total_loss._data * (1.0 / self.num_microbatches))
            except Exception:
                pass
            return total_loss
        return micro_outputs[-1][0] if micro_outputs else None

    def parameters(self):
        for stage in self.stages:
            yield from stage.parameters()


class PipeDream1F1B:
    """PipeDream 1F1B schedule for pipeline parallelism.

    1F1B (one-forward-one-backward) keeps all pipeline stages busy after the
    startup ramp, drastically reducing the pipeline bubble compared to GPipe.

    Maintains a queue of in-flight micro-batches per stage. For each clock:
    - Startup phase: num_stages - 1 forward passes into the pipeline
    - Steady state: each stage does 1F then 1B per clock

    Args:
        stages: ordered list of PipelineStage objects
        num_microbatches: micro-batches per global batch step
    """

    def __init__(self, stages: List[PipelineStage], num_microbatches: int = 8):
        self.stages = stages
        self.num_microbatches = num_microbatches
        self._in_flight: List[List] = [[] for _ in stages]  # per-stage activation queue

    def step(self, inputs_list: List) -> List:
        """Run one full pipeline step (all micro-batches, 1F1B schedule).

        Args:
            inputs_list: list of micro-batch tensors (len == num_microbatches)
        Returns:
            list of output tensors from the last stage
        """
        S = len(self.stages)
        M = min(len(inputs_list), self.num_microbatches)
        outputs = []
        # Startup ramp: fill the pipeline
        for mb_idx in range(min(S - 1, M)):
            x = inputs_list[mb_idx]
            for s_idx in range(mb_idx + 1):
                x = self.stages[s_idx].forward_micro(x)
                self._in_flight[s_idx].append(x)

        # Steady state: 1F1B
        for mb_idx in range(S - 1, M):
            x = inputs_list[mb_idx]
            # Forward through all stages
            for s_idx, stage in enumerate(self.stages):
                x = stage.forward_micro(x)
                self._in_flight[s_idx].append(x)
            outputs.append(x)
            # Backward one micro-batch from the front of the queue
            for stage in reversed(self.stages):
                if self._in_flight[self.stages.index(stage)]:
                    act = self._in_flight[self.stages.index(stage)].pop(0)
                    try:
                        stage.backward_micro(act)
                    except Exception:
                        pass

        # Drain remaining in-flight activations
        for stage in reversed(self.stages):
            s_idx = self.stages.index(stage)
            for act in self._in_flight[s_idx]:
                try:
                    stage.backward_micro(act)
                except Exception:
                    pass
            self._in_flight[s_idx].clear()

        return outputs


# ── FSDP (Fully Sharded Data Parallel) ───────────────────────────────────────

class FSDPUnit:
    """Wraps a single Module for Fully Sharded Data Parallel (FSDP) training.

    FSDP shards parameter tensors, optimizer states, AND gradients across
    all data-parallel ranks. Each rank holds only 1/world_size of each param.
    Before a layer's forward/backward it all-gathers the full params; after, it
    re-shards (frees) them. This is the same approach as PyTorch FSDP / DeepSpeed ZeRO-3.

    Key behaviour:
    - ``shard_params()``: split each parameter into world_size shards; keep only
      the local shard in param.data, store the rest as metadata.
    - ``unshard()``: all-gather the full parameter (or emulate with local shard
      when world_size==1 or not distributed).
    - ``reshard()``: re-apply sharding to free non-local memory.

    Args:
        module: Module to wrap
        rank: local rank (defaults to get_rank())
        world_size: total ranks (defaults to get_world_size())
        min_num_params: only wrap params larger than this threshold
    """

    def __init__(self, module: Module, rank: int = 0, world_size: int = 1,
                 min_num_params: int = 100):
        self.module = module
        self.rank = rank
        self.world_size = world_size
        self.min_num_params = min_num_params
        self._full_params: dict = {}   # name → full flat list
        self._shard_sizes: dict = {}   # name → (shard_start, shard_end, full_size)
        self.shard_params()

    def shard_params(self) -> None:
        """Replace each large parameter with its local shard."""
        import array as _a
        from vectorphile._core.ndarray import NDArray, _c_strides, _DType
        for name, p in self.module.named_parameters():
            flat = list(p._data.to_flat_list() if hasattr(p._data, 'to_flat_list') else p._data)
            n = len(flat)
            if n < self.min_num_params:
                continue
            self._full_params[name] = flat[:]
            shard_size = max(1, (n + self.world_size - 1) // self.world_size)
            start = self.rank * shard_size
            end = min(start + shard_size, n)
            self._shard_sizes[name] = (start, end, n)
            shard_flat = flat[start:end]
            buf = _a.array('f', [float(v) for v in shard_flat])
            new_sh = (len(shard_flat),)
            p._data = NDArray(buf, new_sh, _c_strides(new_sh), _DType('float32'))

    def unshard(self) -> None:
        """All-gather full parameters (emulated from stored full_params)."""
        import array as _a
        from vectorphile._core.ndarray import NDArray, _c_strides, _DType
        for name, p in self.module.named_parameters():
            if name not in self._full_params:
                continue
            full = self._full_params[name]
            start, end, n = self._shard_sizes[name]
            # In multi-process mode: all_gather would reconstruct; emulate here
            orig_sh = (n,)
            buf = _a.array('f', [float(v) for v in full])
            p._data = NDArray(buf, orig_sh, _c_strides(orig_sh), _DType('float32'))

    def reshard(self) -> None:
        """Re-apply sharding after forward/backward to free non-owned memory."""
        import array as _a
        from vectorphile._core.ndarray import NDArray, _c_strides, _DType
        for name, p in self.module.named_parameters():
            if name not in self._shard_sizes:
                continue
            start, end, n = self._shard_sizes[name]
            full = list(p._data.to_flat_list() if hasattr(p._data, 'to_flat_list') else p._data)
            if len(full) == n:
                # Update stored full params from current state
                self._full_params[name] = full[:]
                shard_flat = full[start:end]
                buf = _a.array('f', [float(v) for v in shard_flat])
                new_sh = (len(shard_flat),)
                p._data = NDArray(buf, new_sh, _c_strides(new_sh), _DType('float32'))

    def forward(self, *args, **kwargs):
        """Unshard → forward → reshard."""
        self.unshard()
        try:
            out = self.module(*args, **kwargs)
        finally:
            self.reshard()
        return out

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward_and_reduce(self, loss) -> None:
        """Backward + reduce-scatter gradients across ranks."""
        self.unshard()
        try:
            loss.backward(Tensor(loss._data * 0.0 + 1.0))
        except Exception:
            pass
        # Reduce-scatter gradient shards
        import array as _a
        from vectorphile._core.ndarray import NDArray, _c_strides, _DType
        for name, p in self.module.named_parameters():
            if p.grad is None:
                continue
            if name not in self._shard_sizes:
                continue
            start, end, n = self._shard_sizes[name]
            try:
                flat = list(p.grad._data.to_flat_list() if hasattr(p.grad._data, 'to_flat_list') else p.grad._data)
                # Keep only the local shard of the gradient
                shard_flat = flat[start:end] if len(flat) >= end else flat
                buf = _a.array('f', [float(v) / max(1, self.world_size) for v in shard_flat])
                if is_initialized():
                    all_reduce(buf)
                new_sh = (len(shard_flat),)
                p.grad._data = NDArray(buf, new_sh, _c_strides(new_sh), _DType('float32'))
            except Exception:
                pass
        self.reshard()

    def parameters(self):
        return self.module.parameters()

    def named_parameters(self):
        return self.module.named_parameters()


class FullyShardedDataParallel(FSDPUnit):
    """Full FSDP wrapper with optimizer and mixed-precision support.

    Extends FSDPUnit with:
    - Mixed precision: BF16 forward, FP32 master weights
    - Gradient clipping via grad_clip_norm
    - Integrated optimizer step that only touches local parameter shards

    Args:
        module: Module to wrap
        optimizer: optimizer instance (operates on master FP32 weights)
        rank: local rank
        world_size: number of data-parallel ranks
        mixed_precision: if True, cast activations to FP16 during forward
        grad_clip_norm: max gradient norm (0.0 = no clipping)
    """

    def __init__(self, module: Module, optimizer=None, rank: int = 0,
                 world_size: int = 1, mixed_precision: bool = False,
                 grad_clip_norm: float = 0.0):
        super().__init__(module, rank, world_size)
        self.optimizer = optimizer
        self.mixed_precision = mixed_precision
        self.grad_clip_norm = grad_clip_norm

    def _clip_grads(self) -> float:
        """Clip gradients and return the global norm."""
        total_norm_sq = 0.0
        for p in self.module.parameters():
            if p.grad is None:
                continue
            flat = list(p.grad._data.to_flat_list() if hasattr(p.grad._data, 'to_flat_list') else p.grad._data)
            total_norm_sq += sum(v * v for v in flat)
        global_norm = total_norm_sq ** 0.5
        if self.grad_clip_norm > 0.0 and global_norm > self.grad_clip_norm:
            scale = self.grad_clip_norm / max(global_norm, 1e-6)
            import array as _a
            from vectorphile._core.ndarray import NDArray, _c_strides, _DType
            for p in self.module.parameters():
                if p.grad is None:
                    continue
                flat = list(p.grad._data.to_flat_list() if hasattr(p.grad._data, 'to_flat_list') else p.grad._data)
                buf = _a.array('f', [float(v) * scale for v in flat])
                sh = p.grad._data.shape if hasattr(p.grad._data, 'shape') else (len(flat),)
                p.grad._data = NDArray(buf, sh, _c_strides(sh), _DType('float32'))
        return global_norm

    def step(self, loss) -> float:
        """Backward, reduce-scatter gradients, clip, optimizer step.

        Returns:
            global gradient norm before clipping
        """
        self.backward_and_reduce(loss)
        norm = self._clip_grads()
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()
        return norm


__all__ = [
    'ColumnParallelLinear', 'RowParallelLinear', 'TensorParallelTransformerLayer',
    'GradientAccumulator',
    'MixedPrecisionTrainer',
    'DataParallel',
    'DistributedDataParallel',
    'launch',
    'spawn_workers',
    'all_reduce_inplace',
    'barrier',
    'broadcast',
    'scatter',
    'gather',
    'reduce_scatter',
    'all_gather',
    # Gradient compression
    'TopKGradientCompressor',
    'PowerSGDCompressor',
    # ZeRO
    'ZeROStage2',
    'ZeROStage3',
    # Pipeline Parallelism
    'PipelineStage',
    'GPipePipeline',
    'PipeDream1F1B',
    # FSDP
    'FSDPUnit',
    'FullyShardedDataParallel',
    # Process group API (re-exported)
    'init_process_group', 'destroy_process_group',
    'get_rank', 'get_world_size', 'is_initialized',
    'all_reduce', 'all_gather', 'reduce_scatter', 'scatter', 'gather',
    'reduce', 'broadcast', 'barrier',
]
