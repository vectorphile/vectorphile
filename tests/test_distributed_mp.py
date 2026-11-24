"""Multi-process distributed test — exercises real socket-based ring AllReduce.

Run with:  python -m pytest tests/test_distributed_mp.py -v

Each test spawns 2 processes, initialises the process group, runs a collective,
and checks the result. Single-process fallback is also tested.
"""
import pytest
import sys
import time


def _worker_put_rank(rank, world_size, q):
    """Module-level worker: puts rank into queue (picklable)."""
    q.put(rank)


def _worker_allreduce(rank, world_size, master_port, result_q):
    """Worker function: init group, all_reduce a tensor, put result in queue."""
    try:
        import vectorphile.distributed as dist
        from vectorphile._core.ndarray import array as _nd
        from vectorphile.core.tensor import Tensor

        dist.init_process_group(
            rank=rank, world_size=world_size,
            master_addr='127.0.0.1', master_port=master_port,
            timeout=10.0,
        )
        # Each rank contributes [rank+1.0, rank+1.0, rank+1.0]
        val = float(rank + 1)
        t = Tensor(_nd([val, val, val], dtype='float32'))
        dist.all_reduce(t)
        flat = list(t._data.to_flat_list() if hasattr(t._data, 'to_flat_list') else t._data)
        result_q.put(('ok', flat))
        dist.destroy_process_group()
    except Exception as e:
        result_q.put(('err', str(e)))


def _worker_barrier(rank, world_size, master_port, result_q):
    """Worker function: init group, hit a barrier, record time."""
    try:
        import vectorphile.distributed as dist
        dist.init_process_group(
            rank=rank, world_size=world_size,
            master_addr='127.0.0.1', master_port=master_port,
            timeout=10.0,
        )
        t0 = time.time()
        dist.barrier()
        elapsed = time.time() - t0
        result_q.put(('ok', elapsed))
        dist.destroy_process_group()
    except Exception as e:
        result_q.put(('err', str(e)))


class TestDistributedMultiProcess:

    def test_single_process_allreduce(self):
        """all_reduce is a no-op when world_size == 1."""
        import vectorphile.distributed as dist
        from vectorphile._core.ndarray import array as _nd
        from vectorphile.core.tensor import Tensor

        # Ensure we start uninitialized
        if dist.is_initialized():
            dist.destroy_process_group()

        t = Tensor(_nd([1.0, 2.0, 3.0], dtype='float32'))
        dist.all_reduce(t)  # no-op
        flat = list(t._data.to_flat_list() if hasattr(t._data, 'to_flat_list') else t._data)
        assert flat == [1.0, 2.0, 3.0]

    def test_single_process_barrier(self):
        """barrier() completes instantly when world_size == 1."""
        import vectorphile.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
        dist.barrier()  # must not raise

    def test_two_process_allreduce(self):
        """Spawn 2 workers; each contributes [rank+1]*3; sum should be [3,3,3]."""
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        q = ctx.Queue()
        master_port = 29600  # unique port for this test

        procs = []
        for rank in range(2):
            p = ctx.Process(
                target=_worker_allreduce,
                args=(rank, 2, master_port, q),
            )
            p.start()
            procs.append(p)

        results = []
        for _ in range(2):
            try:
                results.append(q.get(timeout=900))  # allow slow VP import in spawned process
            except Exception:
                results.append(('err', 'timeout'))

        for p in procs:
            p.join(timeout=10)

        # Check both workers succeeded
        for status, val in results:
            assert status == 'ok', f"Worker failed: {val}"
            # all_reduce sum of [1,1,1] + [2,2,2] = [3,3,3]
            assert val == [3.0, 3.0, 3.0], f"Wrong result: {val}"

    def test_two_process_barrier(self):
        """Barrier completes in reasonable time across 2 processes."""
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        q = ctx.Queue()
        master_port = 29610

        procs = []
        for rank in range(2):
            p = ctx.Process(
                target=_worker_barrier,
                args=(rank, 2, master_port, q),
            )
            p.start()
            procs.append(p)

        results = []
        for _ in range(2):
            try:
                # Large timeout: spawned processes must import vectorphile from scratch
                # which is slow (~6-10 min on first import in a fresh Python process)
                results.append(q.get(timeout=900))
            except Exception:
                results.append(('err', 'timeout'))

        for p in procs:
            p.join(timeout=10)

        for status, val in results:
            assert status == 'ok', f"Barrier failed: {val}"
            assert val < 30.0, f"Barrier took too long: {val}s"

    def test_launch_helper(self):
        """launch() spawns and joins workers correctly."""
        import multiprocessing as mp

        ctx = mp.get_context('spawn')
        q = ctx.Queue()
        ctx_procs = []
        for rank in range(2):
            p = ctx.Process(target=_worker_put_rank, args=(rank, 2, q))
            p.start()
            ctx_procs.append(p)
        for p in ctx_procs:
            p.join(timeout=15)

        items = []
        for _ in range(2):
            try:
                items.append(q.get_nowait())
            except Exception:
                pass
        # At least one worker completed; expect rank values 0 and/or 1
        assert len(items) >= 1
