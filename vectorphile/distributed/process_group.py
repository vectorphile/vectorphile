"""
vectorphile distributed process group — socket-based ring AllReduce.

No MPI, no NCCL, no third-party deps. Pure Python sockets + multiprocessing.

Usage:
    # In each worker process:
    import vectorphile.distributed as dist
    dist.init_process_group(rank=rank, world_size=world_size,
                            master_addr='127.0.0.1', master_port=29500)
    dist.all_reduce(tensor)   # in-place sum across all ranks
    dist.barrier()
    dist.destroy_process_group()
"""
from __future__ import annotations
import socket
import struct
import array as _arr
import os
from typing import Optional

# Global state
_rank: int = -1
_world_size: int = 1
_master_addr: str = ''
_master_port: int = 0
_socks: list = []        # ring: socks[0] = recv from left, socks[1] = send to right
_initialized: bool = False


def is_initialized() -> bool:
    return _initialized


def get_rank() -> int:
    return _rank


def get_world_size() -> int:
    return _world_size


def init_process_group(rank: int, world_size: int,
                       master_addr: str = '127.0.0.1',
                       master_port: int = 29500,
                       timeout: float = 60.0):
    """
    Initialise the process group for ring-AllReduce.

    Call this once per process, with each process having a unique rank in
    [0, world_size). Rank 0 acts as the rendezvous coordinator.
    """
    global _rank, _world_size, _master_addr, _master_port, _socks, _initialized
    _rank = rank
    _world_size = world_size
    _master_addr = master_addr
    _master_port = master_port

    if world_size == 1:
        _initialized = True
        return

    # -- Ring topology: each rank connects to the next rank -----------------
    # Rank r listens on port (master_port + r + 1) for connection from rank (r-1).
    # Rank r connects to port (master_port + ((r+1) % world_size) + 1).
    # Port assignment: rank r listens on master_port + r + 1
    my_listen_port  = master_port + rank + 1
    next_rank       = (rank + 1) % world_size
    next_listen_port = master_port + next_rank + 1

    # Create listener socket
    listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_sock.bind(('0.0.0.0', my_listen_port))
    listen_sock.listen(1)
    listen_sock.settimeout(timeout)

    # Connect to next rank (may need to retry until it's up)
    import time
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    send_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    deadline = time.time() + timeout
    while True:
        try:
            send_sock.connect((master_addr, next_listen_port))
            break
        except (ConnectionRefusedError, OSError):
            if time.time() > deadline:
                raise RuntimeError(f'rank {rank}: timeout connecting to rank {next_rank}')
            time.sleep(0.05)

    # Accept connection from previous rank
    recv_conn, _ = listen_sock.accept()
    recv_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    listen_sock.close()

    _socks = [recv_conn, send_sock]   # [recv_from_prev, send_to_next]
    _initialized = True


def destroy_process_group():
    global _initialized, _socks
    for s in _socks:
        try: s.close()
        except Exception: pass
    _socks = []
    _initialized = False


# -- Low-level send/recv of raw float32 buffers ----------------------------

def _send_buf(buf):
    """Send array.array('f', ...) over send socket."""
    data = bytes(buf)
    total = len(data)
    sent = 0
    while sent < total:
        n = _socks[1].send(data[sent:])
        if n == 0:
            raise RuntimeError('send socket closed')
        sent += n


def _recv_buf(nbytes: int):
    """Receive exactly nbytes into new array.array('f', ...)."""
    data = bytearray(nbytes)
    view = memoryview(data)
    received = 0
    while received < nbytes:
        n = _socks[0].recv_into(view[received:], nbytes - received)
        if n == 0:
            raise RuntimeError('recv socket closed')
        received += n
    return _arr.array('f', bytes(data))


def _add_bufs(a, b):
    """In-place a += b for array.array('f', ...) of same length."""
    for i in range(len(a)):
        a[i] += b[i]


# -- Ring AllReduce (sum) ---------------------------------------------------
#
# Two phases (Rabenseifner algorithm):
#   Phase 1 — Reduce-scatter: each rank accumulates its chunk
#   Phase 2 — AllGather: each rank broadcasts its chunk
#
# Each phase requires (world_size - 1) steps.

def all_reduce(tensor, op: str = 'sum'):
    """
    In-place AllReduce across all ranks.

    tensor: vectorphile Tensor or array.array('f', ...) — must be float32.
    op    : 'sum' (only op currently supported).
    """
    if _world_size == 1:
        return

    from vectorphile.core.tensor import Tensor as _T
    from vectorphile._core.ndarray import NDArray as _NDA

    # Extract flat float32 buffer
    if isinstance(tensor, _T):
        td = tensor._data
        if isinstance(td, _NDA) and td._buf is not None:
            buf = td._buf
        else:
            flat_list = td.to_flat_list() if hasattr(td, 'to_flat_list') else list(td) if hasattr(td, '__iter__') else [float(td)]
            buf = _arr.array('f', [float(v) for v in flat_list])
    elif isinstance(tensor, _arr.array) and tensor.typecode == 'f':
        buf = tensor
    else:
        raise TypeError('all_reduce: tensor must be a Tensor or array.array("f")')

    n = len(buf)
    W = _world_size

    # Chunk boundaries (evenly split, last chunk gets remainder)
    chunk_size = (n + W - 1) // W
    chunks = []
    for w in range(W):
        s = w * chunk_size
        e = min(s + chunk_size, n)
        chunks.append((s, e))

    # -- Phase 1: Reduce-scatter ----------------------------------------
    # Ring: in step s, rank r:
    #   sends chunk (r - s) % W  to right
    #   receives chunk (r - s - 1) % W from left, adds to local chunk
    for step in range(W - 1):
        send_chunk_idx = (_rank - step) % W
        recv_chunk_idx = (_rank - step - 1) % W
        s_send, e_send = chunks[send_chunk_idx]
        s_recv, e_recv = chunks[recv_chunk_idx]

        send_data = _arr.array('f', buf[s_send:e_send])
        _send_buf(send_data)
        recv_data = _recv_buf((e_recv - s_recv) * 4)
        _add_bufs_into(buf, s_recv, recv_data)

    # -- Phase 2: AllGather -----------------------------------------------
    # Ring: in step s, rank r sends chunk (r - s + 1) % W to right
    for step in range(W - 1):
        send_chunk_idx = (_rank - step + 1) % W
        recv_chunk_idx = (_rank - step) % W
        s_send, e_send = chunks[send_chunk_idx]
        s_recv, e_recv = chunks[recv_chunk_idx]

        send_data = _arr.array('f', buf[s_send:e_send])
        _send_buf(send_data)
        recv_data = _recv_buf((e_recv - s_recv) * 4)
        for i, v in enumerate(recv_data):
            buf[s_recv + i] = v

    # Write back if needed
    if isinstance(tensor, _T):
        td = tensor._data
        if isinstance(td, _NDA) and td._buf is not None and td._buf is buf:
            pass  # already mutated in-place
        else:
            if isinstance(td, _NDA):
                for i, v in enumerate(buf):
                    td._buf[i] = v


def _add_bufs_into(buf, offset, incoming):
    for i, v in enumerate(incoming):
        buf[offset + i] += v


def broadcast(tensor, src: int = 0):
    """Broadcast tensor from rank src to all other ranks."""
    if _world_size == 1:
        return

    from vectorphile.core.tensor import Tensor as _T
    from vectorphile._core.ndarray import NDArray as _NDA

    if isinstance(tensor, _T):
        td = tensor._data
        buf = td._buf if (isinstance(td, _NDA) and td._buf is not None) else None
    else:
        buf = tensor

    if buf is None:
        return  # can't broadcast without buffer

    n = len(buf)
    # Tree broadcast: source sends to all ranks via a ring-doubling pattern
    # Phase 1: src sends to next rank; each receiver forwards to its next
    # Simple ring: rotate buf starting from src rank around the ring
    # At step s: rank (src + s) % W sends to (src + s + 1) % W
    W = _world_size
    for step in range(W - 1):
        sender = (src + step) % W
        receiver = (src + step + 1) % W
        if _rank == sender:
            _send_buf(_arr.array('f', buf))
        elif _rank == receiver:
            recv = _recv_buf(n * 4)
            for i, v in enumerate(recv):
                buf[i] = v


def all_gather(tensors_out, tensor_in):
    """Gather tensor_in from all ranks into tensors_out list.

    tensors_out: list of Tensors of same shape, one per rank (pre-allocated)
    tensor_in:   local Tensor to contribute
    """
    if _world_size == 1:
        tensors_out[0] = tensor_in
        return

    from vectorphile.core.tensor import Tensor as _T
    from vectorphile._core.ndarray import NDArray as _NDA

    def _get_buf(t):
        td = t._data
        if isinstance(td, _NDA) and td._buf is not None:
            return td._buf
        flat_list = td.to_flat_list() if hasattr(td, 'to_flat_list') else list(td) if hasattr(td, '__iter__') else [float(td)]
        return _arr.array('f', [float(v) for v in flat_list])

    local_buf = _get_buf(tensor_in)
    n = len(local_buf)
    W = _world_size

    # Collect all buffers: ring rotate W-1 times
    gathered = [None] * W
    gathered[_rank] = _arr.array('f', local_buf)
    send_buf = _arr.array('f', local_buf)
    for step in range(W - 1):
        _send_buf(send_buf)
        recv_buf = _recv_buf(n * 4)
        src_rank = (_rank - step - 1) % W
        gathered[src_rank] = recv_buf
        send_buf = recv_buf

    # Write into tensors_out
    for r in range(W):
        tensors_out[r] = _T(gathered[r])


def reduce_scatter(output, input_list, op: str = 'sum'):
    """Scatter-reduce: each rank gets sum of input_list[rank] from all ranks.

    input_list: list of W Tensors, one per rank destination
    output:     Tensor to receive the reduced chunk for this rank
    """
    if _world_size == 1:
        output._data = input_list[0]._data
        return

    # Simple implementation: all_reduce then scatter
    from vectorphile.core.tensor import Tensor as _T
    from vectorphile._core.ndarray import array as _nd_array

    # Stack all inputs by concatenating flat lists
    flat_parts = []
    n_per = None
    for t in input_list:
        d = t._data
        fl = d.to_flat_list() if hasattr(d, 'to_flat_list') else list(d) if hasattr(d, '__iter__') else [float(d)]
        if n_per is None:
            n_per = len(fl)
        flat_parts.extend(fl)
    flat = _arr.array('f', [float(v) for v in flat_parts])
    # AllReduce
    t_flat = _T(flat)
    all_reduce(t_flat, op=op)
    # Extract chunk for this rank
    chunk_list = list(flat)[_rank * n_per: (_rank + 1) * n_per]
    sh = output._data.shape if hasattr(output._data, 'shape') else (len(chunk_list),)
    output._data = _nd_array(chunk_list, dtype='float32').reshape(sh)


def scatter(output, input_list=None, src: int = 0):
    """Scatter input_list from src rank to all ranks.

    output:     Tensor to receive the chunk for this rank
    input_list: list of W Tensors on rank src; None on other ranks
    src:        rank that scatters data
    """
    if _world_size == 1:
        if input_list is not None:
            output._data = input_list[0]._data
        return

    from vectorphile.core.tensor import Tensor as _T
    from vectorphile._core.ndarray import NDArray as _NDA, array as _nd_array

    def _nd_to_buf(d):
        fl = d.to_flat_list() if hasattr(d, 'to_flat_list') else list(d) if hasattr(d, '__iter__') else [float(d)]
        return _arr.array('f', [float(v) for v in fl])

    if _rank == src:
        for r in range(_world_size):
            if r == src:
                output._data = input_list[r]._data
            else:
                _send_buf(_nd_to_buf(input_list[r]._data))
    else:
        # Receive size from src — use same size as local output
        d_out = output._data
        fl = d_out.to_flat_list() if hasattr(d_out, 'to_flat_list') else list(d_out) if hasattr(d_out, '__iter__') else [float(d_out)]
        n = len(fl)
        recv = _recv_buf(n * 4)
        sh = d_out.shape if hasattr(d_out, 'shape') else (n,)
        output._data = _nd_array(list(recv), dtype='float32').reshape(sh)


def gather(input, dst: int = 0):
    """Gather tensors from all ranks to rank dst.

    Returns list of W Tensors on rank dst, None on others.
    """
    if _world_size == 1:
        return [input]

    from vectorphile.core.tensor import Tensor as _T
    from vectorphile._core.ndarray import array as _nd_array

    d_in = input._data
    fl = d_in.to_flat_list() if hasattr(d_in, 'to_flat_list') else list(d_in) if hasattr(d_in, '__iter__') else [float(d_in)]
    buf = _arr.array('f', [float(v) for v in fl])
    n = len(buf)
    sh = d_in.shape if hasattr(d_in, 'shape') else (n,)

    if _rank == dst:
        gathered = [None] * _world_size
        gathered[dst] = _T(buf)
        for r in range(_world_size):
            if r != dst:
                recv = _recv_buf(n * 4)
                gathered[r] = _T(_nd_array(list(recv), dtype='float32').reshape(sh))
        return gathered
    else:
        _send_buf(buf)
        return None


def reduce(tensor, dst: int = 0, op: str = 'sum'):
    """Reduce tensor to rank dst.

    After this call, only rank dst has the reduced result; others have unchanged data.
    """
    if _world_size == 1:
        return

    from vectorphile.core.tensor import Tensor as _T
    from vectorphile._core.ndarray import NDArray as _NDA

    td = tensor._data
    if isinstance(td, _NDA) and td._buf is not None:
        buf = td._buf
    else:
        fl = td.to_flat_list() if hasattr(td, 'to_flat_list') else list(td) if hasattr(td, '__iter__') else [float(td)]
        buf = _arr.array('f', [float(v) for v in fl])
    n = len(buf)

    # Simple tree reduce via ring (send to dst in stages)
    # First do all_reduce then dst keeps; non-dst ranks unchanged
    buf_copy = _arr.array('f', buf)
    temp_t = _T(buf_copy)
    all_reduce(temp_t, op=op)
    if _rank == dst:
        # Write reduced result back
        for i in range(n):
            buf[i] = buf_copy[i]


def barrier():
    """Synchronise all ranks — blocks until all have called barrier()."""
    if _world_size == 1:
        return
    # Exchange a single byte around the ring W-1 times
    token = _arr.array('B', [1])
    for _ in range(_world_size - 1):
        data = bytes(token)
        _socks[1].sendall(data)
        _socks[0].recv(1)
